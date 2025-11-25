#ifndef CUDAINTERPOLATE_H_
#define CUDAINTERPOLATE_H_

#include <functional>
#include <vector>
#include <utility>
#include <memory>
#include <dolfinx/fem/Function.h>

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
}

#endif // CUDAINTERPOLATE_H_
