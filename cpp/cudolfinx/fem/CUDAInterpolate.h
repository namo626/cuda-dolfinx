#ifndef CUDAINTERPOLATE_H_
#define CUDAINTERPOLATE_H_

#include <functional>
#include <vector>
#include <utility>
#include <memory>
#include <dolfinx/fem/Function.h>
#include <cudolfinx/fem/CUDADofMap.h>

namespace dolfinx::CUDA {

template <dolfinx::scalar T, std::floating_point U>
std::vector<int> get_interpolate_mask(dolfinx::fem::Function<T, U>&, 
                 std::array<std::size_t, 2>,
                 std::span<std::int32_t>);

template <dolfinx::scalar T, std::floating_point U>
void interpolate(dolfinx::fem::Function<T, U> &, const std::vector<int> &,
                 const std::vector<T>&);

void cuda_wrapper_interpolate(int dof_count, int num_points, const double *xs,
                              const double *ys, const double *zs, 
                              const int *mask, double *x);

template <dolfinx::scalar T, std::floating_point U>
std::vector<T> eval_reference_basis(const dolfinx::fem::Function<T, U> &f,
          std::span<const T> x, std::array<std::size_t, 2> xshape,
          std::span<const std::int32_t> cells);
template <dolfinx::scalar T, std::floating_point U>
std::vector<T> basis_expand(const dolfinx::fem::Function<T, U> &f,
                            const std::vector<T> &basis_values,
                            const std::vector<int>& cells);

std::vector<double> cuda_basis_expand(const dolfinx::fem::Function<double,double> &f,
                                 CUdeviceptr dofmap, CUdeviceptr coeffs,
                                 CUdeviceptr dbasis_values, const std::vector<int>& cells); 

template <dolfinx::scalar T, std::floating_point U>
void create_interpolation_maps(const dolfinx::fem::Function<T, U>& u1,
                          const dolfinx::fem::Function<T, U>& u0,
                          std::vector<T>& i_m, std::array<std::size_t, 2> im_shape,
                          std::vector<std::int32_t>& dofs0_map,
                          std::vector<std::int32_t>& dofs1_map);
void cuda_interpolate_same_map(dolfinx::fem::Function<double, double> &u1,
                               dolfinx::fem::Function<double, double> &u0,
                               CUdeviceptr _x,
                               CUdeviceptr _y,
                               CUdeviceptr i_m,
                               std::array<std::size_t, 2> im_shape,
                               CUdeviceptr dofs0_map, CUdeviceptr dofs1_map);

template <dolfinx::scalar T, std::floating_point U>
void interpolate_same_map(dolfinx::fem::Function<T, U>& u1,
                          dolfinx::fem::Function<T, U>& u0,
                          std::vector<T>& i_m, std::array<std::size_t, 2> im_shape,
                          const std::vector<std::int32_t>& dofs0_map,
                          const std::vector<std::int32_t>& dofs1_map);

}

#endif // CUDAINTERPOLATE_H_
