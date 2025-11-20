// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/interpolate.h>
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

    _interp_pts = dolfinx::fem::interpolation_coords<T>(
        *fn_space->element(), fn_space->mesh()->geometry(), cells);
    _dinterp_size = _interp_pts.size();

    CUDA::safeMemAlloc(&_dinterp_pts, _dinterp_size);
    CUDA::safeMemcpyHtoD(_dinterp_pts, (void *)(_interp_pts.data()),
                         _dinterp_size);

    // Interpolation mask
    _interp_mask = CUDA::get_interpolate_mask(*_f,  {1, _dinterp_size}, cells);
    CUDA::safeMemAlloc(&_dinterp_mask, _interp_mask.size());
    CUDA::safeMemcpyHtoD(_dinterp_mask, (void *)(_interp_mask.data()), _interp_mask.size());
  }

  /// Interpolate a scalar function which accepts a vector of coordinates
  /// with shape (3, num_points)
  void interpolate(std::function<std::vector<T>(std::vector<T>&)> g)
  {
    std::vector<T> g_eval = g(_interp_pts);
    assert(g_eval.size() == _interp_pts.size() / 3);

    CUDA::interpolate(*_f, _interp_mask, g_eval);
  }

  /// Get pointer to vector data on device
  CUdeviceptr device_values() const
  {
    return _dvalues;
  }

  /// Get pointer to device interpolation coordinates
  CUdeviceptr interpolation_coords() const
  {
    return _dinterp_pts;
  }

  std::vector<int> interp_mask() const
  {
    return _interp_mask;
  }

  CUdeviceptr device_interp_mask() const
  {
    return _dinterp_mask;
  }


  ~CUDACoefficient()
  {
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
  std::shared_ptr<dolfinx::fem::Function<T,U>> _f;
  // Pointer to host-side coefficient vector
  std::shared_ptr<const dolfinx::la::Vector<T>> _x;

  // Host vector of interpolation coordinates
  std::vector<T> _interp_pts;
  // Device-side interpolation coordinates
  CUdeviceptr _dinterp_pts;
  size_t _dinterp_size;

  // Interpolation DOF map
  std::vector<int> _interp_mask;
  CUdeviceptr _dinterp_mask;
};

template class dolfinx::fem::CUDACoefficient<double>;
}
