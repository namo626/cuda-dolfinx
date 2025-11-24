#include "CUDAInterpolate.h"
#include <cstddef>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/fem/interpolate.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dolfinx::CUDA {

template <dolfinx::scalar T, std::floating_point U>
std::vector<T> basis_expand(const dolfinx::fem::Function<T, U> &u,
                            const std::vector<T> &basis_values,
                            const std::vector<int>& cells) {

  auto _function_space = u.function_space();
  auto element = _function_space->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size =
      element->reference_value_size() / bs_element;
  const std::size_t value_size = _function_space->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element; // no. of DOF

  assert(basis_values.size() == space_dimension*value_size*cells.size());

  std::vector<T> u(cells.size() * value_size);

  for (std::size_t p = 0; p < cells.size(); ++p) {
    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
    for (std::size_t i = 0; i < space_dimension; ++i) {
      auto coeff = u.x()[dofs[i]];
      for (std::size_t j = 0; j < value_size; ++j) {
        u[p * ushape[1] + (j * bs_element + k)] +=
          coeff * basis_values((i * value_size + j) * space_dimension + p);
      }
    }
  }

  return u;
}

template <dolfinx::scalar T, std::floating_point U>
void interpolate(dolfinx::fem::Function<T, U> &u, const std::vector<int> &mask,
                 const std::vector<T> &f) {
  std::span<T> coeffs = u.x()->mutable_array();

  for (std::size_t i = 0; i < coeffs.size(); i++) {
    coeffs[i] = f[mask[i]];
  }
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<int> get_interpolate_mask(dolfinx::fem::Function<T, U>& u,
                 std::array<std::size_t, 2> fshape,
                 std::span<std::int32_t> cells)
{
  auto element = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();
  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology()->dim();
  const bool symmetric = u.function_space()->symmetric();

  if (fshape[0] != (std::size_t)u.function_space()->value_size())
    throw std::runtime_error("Interpolation data has the wrong shape/size.");

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  const std::size_t f_shape1 = fshape[1];

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = u.function_space()->value_size() / element_bs;

  std::span<T> coeffs = u.x()->mutable_array();
  std::vector<int> dof_mask(coeffs.size());
  std::vector<T> _coeffs(num_scalar_dofs);
  std::vector<int> cell_perm(num_scalar_dofs);

  // This assumes that any element with an identity interpolation matrix
  // is a point evaluation
  if (element->map_ident() && element->interpolation_ident()) {
    // Point evaluation element *and* the geometric map is the identity,
    // e.g. not Piola mapped
    auto apply_inv_transpose_dof_transformation_int =
        element->template dof_transformation_fn<int>(
            dolfinx::fem::doftransform::inverse_transpose, true);

    // Loop over cells
    for (std::size_t c = 0; c < cells.size(); ++c) {
      const std::int32_t cell = cells[c];
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k) {
        // num_scalar_dofs is the number of interpolation points per
        // cell in this case (interpolation matrix is identity)
        // std::copy_n(std::next(f.begin(), k * f_shape1 + c * num_scalar_dofs),
        //             num_scalar_dofs, _coeffs.begin());
        std::iota(cell_perm.begin(), cell_perm.end(), k * f_shape1 + c*num_scalar_dofs);

        //apply_inv_transpose_dof_transformation(_coeffs, cell_info, cell, 1);
        apply_inv_transpose_dof_transformation_int(cell_perm, cell_info, cell, 1);

        for (int i = 0; i < num_scalar_dofs; ++i) {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          //coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
          dof_mask[dofmap_bs * dofs[pos.quot] + pos.rem] = (int) cell_perm[i];
        }
      }
    }
  } else {
    throw std::runtime_error("Interpolation is currently only supported for non Piola-mapped elements.");
  }

  return dof_mask;
}

template std::vector<int>
get_interpolate_mask<double, double>(dolfinx::fem::Function<double, double> &,
                                     std::array<std::size_t, 2>,
                                     std::span<std::int32_t>);

template void
interpolate<double, double>(dolfinx::fem::Function<double, double> &,
                            const std::vector<int> &, const std::vector<double>&);
} // namespace dolfinx::CUDA
