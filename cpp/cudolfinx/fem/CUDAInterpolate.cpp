#include "CUDAInterpolate.h"
#include <cstddef>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/fem/interpolate.h>
#include <vector>

namespace dolfinx::CUDA {
void interpolate(std::shared_ptr<const dolfinx::fem::Function<double, double>> F,
                 const std::function <
                     std::pair<std::vector<double>, std::vector<std::size_t>>(
                         double*, size_t) &f,
                 std::vector<double> interp_pts)
{

  auto fn_space = F->function_space();
  const auto [fs, fshape] = f(interp_pts);

  std::vector<std::int32_t> cells(cmap->size_local() + cmap->num_ghosts(), 0);
  std::iota(cells.begin(), cells.end(), 0);

  std::span<T> coeffs = F.x()->mutable_array();

  for (std::size_t c = 0; c < cells.size(); ++c) {
    const std::int32_t cell = cells[c];
    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
    for (int k = 0; k < element_bs; ++k) {
      // num_scalar_dofs is the number of interpolation points per
      // cell in this case (interpolation matrix is identity)
      std::copy_n(std::next(f.begin(), k * f_shape1 + c * num_scalar_dofs),
                  num_scalar_dofs, _coeffs.begin());
      apply_inv_transpose_dof_transformation(_coeffs, cell_info, cell, 1);
      for (int i = 0; i < num_scalar_dofs; ++i) {
        const int dof = i * element_bs + k;
        std::div_t pos = std::div(dof, dofmap_bs);
        coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
      }
    }
  }
}


}
