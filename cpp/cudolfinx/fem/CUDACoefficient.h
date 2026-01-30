// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
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
          std::floating_point U = dolfinx::scalar_value_t<T>>
class CUDACoefficient
{
public:
  
  /// @brief Construct a new CUDACoefficient
  CUDACoefficient(std::shared_ptr<const Function<T, U>> f) {
    _f = f;
    _x = f->x();
    _g = nullptr;
    _values.assign(_x->array().begin(), _x->array().end());
    _dvalues_size = _x->bs() * (_x->index_map()->size_local()+_x->index_map()->num_ghosts()) * sizeof(T);
    CUDA::safeMemAlloc(&_dvalues, _dvalues_size);
    copy_host_values_to_device();

    // Count total no. of cells
    auto mesh = f->function_space()->mesh();
    auto map = mesh->topology()->index_map(mesh->topology()->dim());
    _num_cells = map->size_local() + map->num_ghosts();

  }

  /// Copy to device, allocating GPU memory if required
  void copy_host_values_to_device()
  {
    CUDA::safeMemcpyHtoD(_dvalues, (void*)(_x->array().data()), _dvalues_size);
  }

  void copy_device_values_to_host()
  {
    CUDA::safeMemcpyDtoH((void*)_values.data(), _dvalues, _dvalues_size);
  }

  /// Interpolate a Function associated with the same mesh over all cells.
  /// Update host-side coefficient vector.
  void interpolate(std::shared_ptr<dolfinx::fem::Function<T, U>> g) {
    // If we haven't used this function before, need to initialize the necessary operators
    if (_g != g) {
      _g = g;
      auto element0 = _g->function_space()->element();
      assert(element0);
      auto element1 = _f->function_space()->element();

      // Copy DOF vector of g to device
      auto _y = _g->x();
      int _dvalues_g_size = _y->bs() * (_y->index_map()->size_local()+_y->index_map()->num_ghosts()) * sizeof(T);
      CUDA::safeMemAlloc(&_dvalues_g, _dvalues_g_size);
      CUDA::safeMemcpyHtoD(_dvalues_g, (void*)(_y->array().data()), _dvalues_g_size);

      // Create interpolation operator
      auto [x, y] = element1->create_interpolation_operator(*element0);
      _i_m = x;
      _im_shape = y;
      //std::cout << "im_shape[0] = " << _im_shape[0] << std::endl;
      //std::cout << "im_shape[1] = " << _im_shape[1] << std::endl;
      CUDA::safeMemAlloc(&_d_i_m, _i_m.size()*sizeof(T));
      CUDA::safeMemcpyHtoD(_d_i_m, (void*)(_i_m.data()), _i_m.size()*sizeof(T));

      // Create interpolation mappings once
      _M0 = CUDA::create_interpolation_map(*_g);
      _M1 = CUDA::create_interpolation_map(*_f);

      // Copy interpolation maps to device
      CUDA::safeMemAlloc(&_dM0, _M0.size()*sizeof(int));
      CUDA::safeMemAlloc(&_dM1, _M1.size()*sizeof(int));
      CUDA::safeMemcpyHtoD(_dM0, (void*)(_M0.data()), _M0.size()*sizeof(int));
      CUDA::safeMemcpyHtoD(_dM1, (void*)(_M1.data()), _M1.size()*sizeof(int));
      
    }

    CUDA::interpolate_same_map<T>(_dvalues, _dvalues_g, _im_shape, _num_cells, _d_i_m, _dM0, _dM1);
    copy_device_values_to_host();
  }


  /// Return a copy of host-side coefficient vector
  std::vector<T> values() const {
    return _values;
  }

  /// Get pointer to vector data on device
  CUdeviceptr device_values() const { return _dvalues; }


  ~CUDACoefficient() {
    if (_dvalues)
      cuMemFree(_dvalues);

  }

private:
  // Host-side coefficient array. Any time _dvalues is updated, this is also updated.
  std::vector<T> _values;
  // Device-side coefficient array
  CUdeviceptr _dvalues;
  // Size of coefficient array
  size_t _dvalues_size;
  // Pointer to host-side Function
  std::shared_ptr<const dolfinx::fem::Function<T, U>> _f;
  // Pointer to host-side coefficient vector
  std::shared_ptr<const dolfinx::la::Vector<T>> _x;

  // Total number of cells
  size_t _num_cells;

  std::shared_ptr<dolfinx::fem::Function<T, U>> _g;
  CUdeviceptr _dvalues_g;
  // Interpolation operator for _g
  std::vector<T> _i_m;
  std::array<std::size_t, 2> _im_shape;
  // Device-side
  CUdeviceptr _d_i_m;

  // Interpolation maps
  std::vector<std::int32_t> _M0;
  std::vector<std::int32_t> _M1;
  CUdeviceptr _dM0;
  CUdeviceptr _dM1;
};

template class dolfinx::fem::CUDACoefficient<double>;
}
