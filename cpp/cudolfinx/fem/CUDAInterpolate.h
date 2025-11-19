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
}

#endif // CUDAINTERPOLATE_H_
