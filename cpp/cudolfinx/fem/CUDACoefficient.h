// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/interpolate.h>
#include <memory>
#include <span>
#include <vector>
#include <cudolfinx/fem/CUDAInterpolate.h>

namespace dolfinx::fem
{
/// @brief a wrapper around a Function
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDACoefficient
{
public:
  
  /// @brief Construct a new CUDACoefficient
  CUDACoefficient(std::shared_ptr<Function<T, U>> f) {
    _f = f;
    _x = f->x();
    _dvalues_size = _x->bs() * (_x->index_map()->size_local()+_x->index_map()->num_ghosts()) * sizeof(T);
    CUDA::safeMemAlloc(&_dvalues, _dvalues_size);
    copy_host_values_to_device();
    init_interpolation();

    _basis_values = {};

    auto dofmap = _f->function_space()->dofmap();
    _ddofmap = dolfinx::fem::CUDADofMap(*dofmap);
  }

  /// Copy to device, allocating GPU memory if required
  void copy_host_values_to_device()
  {
    CUDA::safeMemcpyHtoD(_dvalues, (void*)(_x->array().data()), _dvalues_size);
  }

  /// Compute physical interpolation points on host and copy to device
  void init_interpolation()
  {
    auto fn_space = _f->function_space();
    int tdim = fn_space->mesh()->topology()->dim();
    auto cmap = fn_space->mesh()->topology()->index_map(tdim);
    assert(cmap);

    std::vector<std::int32_t> cells(cmap->size_local() + cmap->num_ghosts(), 0);
    std::iota(cells.begin(), cells.end(), 0);

    assert(fn_space->element());
    assert(fn_space->mesh());

    // Interpolation coordinates
    _interp_pts = dolfinx::fem::interpolation_coords<T>(
        *fn_space->element(), fn_space->mesh()->geometry(), cells);
    _num_interp_pts = _interp_pts.size() / 3;
    _dinterp_size = _interp_pts.size() * sizeof(T); // bytes

    CUDA::safeMemAlloc(&_dinterp_pts, _dinterp_size);
    CUDA::safeMemcpyHtoD(_dinterp_pts, (void *)(_interp_pts.data()),
                         _dinterp_size);

    // Set view of x,y,z coordinates
    T* tmp = (T*) _dinterp_pts;
    _dxs = &tmp[0];
    _dys = &tmp[_num_interp_pts];
    _dzs = &tmp[2*_num_interp_pts];

    // Interpolation mask
    _interp_mask = CUDA::get_interpolate_mask(
        *_f, {(std::size_t)fn_space->value_size(), _num_interp_pts}, cells);
    CUDA::safeMemAlloc(&_dinterp_mask, _interp_mask.size() * sizeof(int));
    CUDA::safeMemcpyHtoD(_dinterp_mask, (void *)(_interp_mask.data()),
                         _interp_mask.size() * sizeof(int));
  }

  /// Interpolate a scalar function which accepts a vector of coordinates
  /// with shape (3, num_points)
  void interpolate(std::function<std::vector<T>(std::vector<T>&)> g)
  {
    std::vector<T> g_eval = g(_interp_pts);
    //assert(g_eval.size() == _interp_pts.size() / 3);

    CUDA::interpolate(*_f, _interp_mask, g_eval);
    copy_host_values_to_device();
  }

  /// Test interpolating 1 + x^2 + y^2 + z^2
  void cuda_interpolate_test() {
    CUDA::cuda_wrapper_interpolate(_dvalues_size/sizeof(T), _num_interp_pts, _dxs, _dys, _dzs,
                                   (int*)_dinterp_mask, (T*)_dvalues);
  }

  /// Evaluate the function at given coordinates x
  std::vector<T> eval(std::span<const T> x, std::array<std::size_t, 2> xshape,
            const std::vector<int>& cells) {
    if (_basis_values.empty()) {
      _basis_values = CUDA::eval_reference_basis(*_f, x, xshape, cells);
      CUDA::safeMemAlloc(&_dbasis_values, _basis_values.size()*sizeof(T));
      CUDA::safeMemcpyHtoD(_dbasis_values, (void*)(_basis_values.data()),
                           _basis_values.size() * sizeof(T));
    }
    //return CUDA::basis_expand(*_f, _basis_values, cells);
    auto space_dimension = _f->function_space()->element()->space_dimension(); // no. of DOF
    //std::cout << "dofmap num_dofs: " << _ddofmap.num_dofs() << std::endl;
    //std::cout << "actual dofs" << cells.size()*space_dimension << std::endl;
    //assert(_ddofmap.num_dofs() == space_dimension*cells.size());
    return CUDA::cuda_basis_expand(*_f, _ddofmap.dofs_per_cell(), _dvalues, _dbasis_values, cells);
  }

  /// Get pointer to vector data on device
  CUdeviceptr device_values() const { return _dvalues; }

  /// Copy device coefficient array to host, then return.
  std::shared_ptr<const dolfinx::la::Vector<T>> x() const {
    std::vector<T> coeffs(_dvalues_size / sizeof(T));
    CUDA::safeMemcpyDtoH(coeffs.data(), _dvalues, _dvalues_size);
    _x->array() = coeffs;
    return _x;
  }

  std::vector<int> interp_mask() const { return _interp_mask; }

  ~CUDACoefficient() {
    if (_dvalues)
      cuMemFree(_dvalues);

    if (_dinterp_pts)
      cuMemFree(_dinterp_pts);

    if (_dinterp_mask)
      cuMemFree(_dinterp_mask);
  }

private:
  // Device-side coefficient array
  CUdeviceptr _dvalues;
  // Size of coefficient array
  size_t _dvalues_size;
  // Pointer to host-side Function
  std::shared_ptr<dolfinx::fem::Function<T, U>> _f;
  // Pointer to host-side coefficient vector
  std::shared_ptr<dolfinx::la::Vector<T>> _x;

  // Number of interpolation points
  size_t _num_interp_pts;
  // Host vector of interpolation coordinates with shape (3, num_points)
  std::vector<T> _interp_pts;
  // Device-side interpolation coordinates
  CUdeviceptr _dinterp_pts;
  // Size of interpolation coordinate vector in bytes
  size_t _dinterp_size;
  // Device pointers to x, y, z slices of _dinterp_pts
  T *_dxs, *_dys, *_dzs;

  // Interpolation DOF map.
  std::vector<int> _interp_mask;
  CUdeviceptr _dinterp_mask;

  // Reference basis evaluations at previously given coordinate points
  std::vector<T> _basis_values;
  CUdeviceptr _dbasis_values;

  dolfinx::fem::CUDADofMap _ddofmap;
  CUdeviceptr _dunrolled_dofs;
};

template class dolfinx::fem::CUDACoefficient<double>;
}
