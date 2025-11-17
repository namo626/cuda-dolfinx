#ifndef CUDAINTERPOLATE_H_
#define CUDAINTERPOLATE_H_

#include <functional>
#include <vector>
#include <utility>
#include <memory>
#include <dolfinx/fem/Function.h>

namespace dolfinx::CUDA {

void interpolate(std::shared_ptr<const dolfinx::fem::Function<double, double>> F,
                 const std::function <
                     std::pair<std::vector<double>, std::vector<std::size_t>>(
                         double*,  size_t) &f );
}

#endif // CUDAINTERPOLATE_H_
