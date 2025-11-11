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

  }

  /// Copy to device, allocating GPU memory if required
  void copy_host_values_to_device()
  {
    CUDA::safeMemcpyHtoD(_dvalues, (void*)(_x->array().data()), _dvalues_size);
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
      assert(element1);

      // Copy DOF vector of g to device
      auto _y = _g->x();
      int _dvalues_g_size = _y->bs() * (_y->index_map()->size_local()+_y->index_map()->num_ghosts()) * sizeof(T);
      CUDA::safeMemAlloc(&_dvalues_g, _dvalues_g_size);
      CUDA::safeMemcpyHtoD(_dvalues_g, (void*)(_y->array().data()), _dvalues_g_size);

      // Create interpolation operator
      auto [x, y] = element1->create_interpolation_operator(*element0);
      _i_m = x;
      _im_shape = y;
      std::cout << "im_shape[0] = " << _im_shape[0] << std::endl;
      std::cout << "im_shape[1] = " << _im_shape[1] << std::endl;
      CUDA::safeMemAlloc(&_d_i_m, _i_m.size()*sizeof(T));
      CUDA::safeMemcpyHtoD(_d_i_m, (void*)(_i_m.data()), _i_m.size()*sizeof(T));

      // Create interpolation mappings once
      CUDA::create_interpolation_maps(*_f, *_g, _i_m, _im_shape, _A_star, _B_star);

      CUDA::safeMemAlloc(&_dof0_mask, _A_star.size()*sizeof(int));
      CUDA::safeMemAlloc(&_dof1_mask, _B_star.size()*sizeof(int));
      CUDA::safeMemcpyHtoD(_dof0_mask, (void*)(_A_star.data()), _A_star.size()*sizeof(int));
      CUDA::safeMemcpyHtoD(_dof1_mask, (void*)(_B_star.data()), _B_star.size()*sizeof(int));
      
    }

    //CUDA::interpolate_same_map(*_f, *_g, _i_m, _im_shape, _A_star, _B_star);
    std::vector<T> output;
    CUDA::cuda_interpolate_same_map(*_f, *_g, _dvalues, _dvalues_size, _dvalues_g, _d_i_m, _im_shape, _dof0_mask,
                                    _dof1_mask, output);
    _values = output;
  }


  /// Return a copy of coefficient vector
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



  std::shared_ptr<dolfinx::fem::Function<T, U>> _g;
  CUdeviceptr _dvalues_g;
  // Interpolation operator for _g
  std::vector<T> _i_m;
  std::array<std::size_t, 2> _im_shape;
  // Device-side
  CUdeviceptr _d_i_m;

  // Interpolation maps
  std::vector<std::int32_t> _A_star;
  std::vector<std::int32_t> _B_star;
  CUdeviceptr _dof0_mask;
  CUdeviceptr _dof1_mask;
};

template class dolfinx::fem::CUDACoefficient<double>;
}
