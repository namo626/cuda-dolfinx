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
void interpolate_same_map(dolfinx::fem::Function<T, U>& u1,
                          dolfinx::fem::Function<T, U>& u0,
                          std::vector<T>& i_m, std::array<std::size_t, 2> im_shape) {

  auto V0 = u0.function_space();
  assert(V0);
  auto V1 = u1.function_space();
  assert(V1);
  auto mesh0 = V0->mesh();
  assert(mesh0);

  auto mesh1 = V1->mesh();
  assert(mesh1);

  auto element0 = V0->element();
  assert(element0);
  auto element1 = V1->element();
  assert(element1);

  assert(mesh0->topology()->dim());
  const int tdim = mesh0->topology()->dim();
  auto map = mesh0->topology()->index_map(tdim);
  assert(map);
  std::span<T> u1_array = u1.x()->mutable_array();
  std::span<const T> u0_array = u0.x()->array();

  // Get all cells
  std::vector<std::int32_t> cells(map->size_local() + map->num_ghosts(), 0);
  std::iota(cells.begin(), cells.end(), 0);
  auto cells0 = cells;
  auto cells1 = cells;

  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh0->topology_mutable()->create_entity_permutations();
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    mesh1->topology_mutable()->create_entity_permutations();
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap1 = V1->dofmap();
  auto dofmap0 = V0->dofmap();

  // Get block sizes and dof transformation operators
  const int bs1 = dofmap1->bs();
  const int bs0 = dofmap0->bs();
  auto apply_dof_transformation = element0->template dof_transformation_fn<T>(
      dolfinx::fem::doftransform::transpose, false);
  auto apply_inverse_dof_transform
      = element1->template dof_transformation_fn<T>(
          dolfinx::fem::doftransform::inverse_transpose, false);

  // Create working array
  std::vector<T> local0(element0->space_dimension());
  std::vector<T> local1(element1->space_dimension());


  // Iterate over mesh and interpolate on each cell
  using X = typename dolfinx::scalar_value_type_t<T>;
  for (std::size_t c = 0; c < cells0.size(); c++)
  {
    // Pack and transform cell dofs to reference ordering
    std::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(cells0[c]);
    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        local0[bs0 * i + k] = u0_array[bs0 * dofs0[i] + k];

    apply_dof_transformation(local0, cell_info0, cells0[c], 1);

    // FIXME: Get compile-time ranges from Basix
    // Apply interpolation operator
    std::ranges::fill(local1, 0);
    for (std::size_t i = 0; i < im_shape[0]; ++i)
      for (std::size_t j = 0; j < im_shape[1]; ++j)
        local1[i] += static_cast<X>(i_m[im_shape[1] * i + j]) * local0[j];

    apply_inverse_dof_transform(local1, cell_info1, cells1[c], 1);
    std::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(cells1[c]);
    for (std::size_t i = 0; i < dofs1.size(); ++i)
      for (int k = 0; k < bs1; ++k)
        u1_array[bs1 * dofs1[i] + k] = local1[bs1 * i + k];
  }
}

namespace impl
{
/// @brief Convenience typdef
template <typename T, std::size_t D>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;
}
  /// @brief Evaluate the reference basis functions at points.
  ///
  /// @param[in] x The coordinates of the points. It has shape
  /// (num_points, 3) and storage is row-major.
  /// @param[in] xshape Shape of `x`.
  /// @param[in] cells Cell indices such that `cells[i]` is the index of
  /// the cell that contains the point x(i). Negative cell indices can
  /// be passed, in which case the corresponding point is ignored.
  /// @param[out] Reference basis values at the points. Values are not computed for
  /// points with a negative cell index. This argument must be passed
  /// with the correct size. Storage is row-major.
  /// @param[in] ushape Shape of `u`.
template <dolfinx::scalar T, std::floating_point U>
std::vector<T> eval_reference_basis(const dolfinx::fem::Function<T, U> &f,
                                    std::span<const T> x,
                                    std::array<std::size_t, 2> xshape,
                                    std::span<const std::int32_t> cells) {
  using geometry_type = T;
  using value_type = U;

  if (cells.empty())
    return {};

  assert(x.size() == xshape[0] * xshape[1]);

  // TODO: This could be easily made more efficient by exploiting
  // points being ordered by the cell to which they belong.

  if (xshape[0] != cells.size()) {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }


  auto _function_space = f.function_space();
  // Get mesh
  assert(_function_space);
  auto mesh = _function_space->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get coordinate map
  const dolfinx::fem::CoordinateElement<geometry_type> &cmap = mesh->geometry().cmap();

  // Get geometry data
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  auto x_g = mesh->geometry().x();

  // Get element
  auto element = _function_space->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size =
      element->reference_value_size() / bs_element;
  const std::size_t value_size = _function_space->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element) {
    throw std::runtime_error("Function::eval is not supported for mixed "
                             "elements. Extract subspaces.");
  }

  // Create work vector for expansion coefficients
  std::vector<value_type> coefficients(space_dimension * bs_element);

  // Get dofmap
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _function_space->dofmap();
  assert(dofmap);
  const int bs_dof = dofmap->bs();

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations()) {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  std::vector<geometry_type> coord_dofs_b(num_dofs_g * gdim);
  impl::mdspan_t<geometry_type, 2> coord_dofs(coord_dofs_b.data(), num_dofs_g,
                                              gdim);
  std::vector<geometry_type> xp_b(1 * gdim);
  impl::mdspan_t<geometry_type, 2> xp(xp_b.data(), 1, gdim);

  // Loop over points
  std::span<const value_type> _v = f.x()->array();

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<geometry_type> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  impl::mdspan_t<const geometry_type, 4> phi0(phi0_b.data(), phi0_shape);
  cmap.tabulate(1, std::vector<geometry_type>(tdim), {1, tdim}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Data structure for evaluating geometry basis at specific points.
  // Used in non-affine case.
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
  std::vector<geometry_type> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  impl::mdspan_t<const geometry_type, 4> phi(phi_b.data(), phi_shape);
  auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Reference coordinates for each point
  std::vector<geometry_type> Xb(xshape[0] * tdim);
  impl::mdspan_t<geometry_type, 2> X(Xb.data(), xshape[0], tdim);

  // Geometry data at each point
  std::vector<geometry_type> J_b(xshape[0] * gdim * tdim);
  impl::mdspan_t<geometry_type, 3> J(J_b.data(), xshape[0], gdim, tdim);
  std::vector<geometry_type> K_b(xshape[0] * tdim * gdim);
  impl::mdspan_t<geometry_type, 3> K(K_b.data(), xshape[0], tdim, gdim);
  std::vector<geometry_type> detJ(xshape[0]);
  std::vector<geometry_type> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p) {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t i = 0; i < num_dofs_g; ++i) {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[p * xshape[1] + j];

    auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    std::array<geometry_type, 3> Xpb = {0, 0, 0};
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        geometry_type,
        MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
            std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine()) {
      dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian(dphi0, coord_dofs, _J);
      dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian_inverse(_J, _K);
      std::array<geometry_type, 3> x0 = {0, 0, 0};
      for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
        x0[i] += coord_dofs(0, i);
      dolfinx::fem::CoordinateElement<geometry_type>::pull_back_affine(Xp, _K, x0, xp);
      detJ[p] = dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian_determinant(
          _J, det_scratch);
    } else {
      // Pull-back physical point xp to reference coordinate Xp
      cmap.pull_back_nonaffine(Xp, xp, coord_dofs);
      cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
      dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian(dphi, coord_dofs, _J);
      dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian_inverse(_J, _K);
      detJ[p] = dolfinx::fem::CoordinateElement<geometry_type>::compute_jacobian_determinant(
          _J, det_scratch);
    }

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Prepare basis function data structures
  std::vector<geometry_type> basis_derivatives_reference_values_b(
      1 * xshape[0] * space_dimension * reference_value_size);
  impl::mdspan_t<const geometry_type, 4> basis_derivatives_reference_values(
      basis_derivatives_reference_values_b.data(), 1, xshape[0],
      space_dimension, reference_value_size);
  std::vector<geometry_type> basis_values_b(space_dimension * value_size);
  impl::mdspan_t<geometry_type, 2> basis_values(basis_values_b.data(),
                                                space_dimension, value_size);
  std::vector<geometry_type> basis_values_p(space_dimension * value_size *
                                            cells.size());
  impl::mdspan_t<geometry_type, 3> basis_values_span(
      basis_values_p.data(), space_dimension, value_size, cells.size());

  // Compute basis on reference element
  element->tabulate(basis_derivatives_reference_values_b, Xb,
                    {X.extent(0), X.extent(1)}, 0);

  using xu_t = impl::mdspan_t<geometry_type, 2>;
  using xU_t = impl::mdspan_t<const geometry_type, 2>;
  using xJ_t = impl::mdspan_t<const geometry_type, 2>;
  using xK_t = impl::mdspan_t<const geometry_type, 2>;
  auto push_forward_fn =
      element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  // Transformation function for basis function values
  auto apply_dof_transformation =
      element->template dof_transformation_fn<geometry_type>(
          dolfinx::fem::doftransform::standard);

  // Size of tensor for symmetric elements, unused in non-symmetric case, but
  // placed outside the loop for pre-computation.
  int matrix_size;
  if (element->symmetric()) {
    matrix_size = 0;
    while (matrix_size * matrix_size < value_size)
      ++matrix_size;
  }

  const std::size_t num_basis_values = space_dimension * reference_value_size;
  for (std::size_t p = 0; p < cells.size(); ++p) {
    const int cell_index = cells[p];
    if (cell_index < 0) // Skip negative cell indices
      continue;

    // Permute the reference basis function values to account for the
    // cell's orientation
    apply_dof_transformation(
        std::span(basis_derivatives_reference_values_b.data() +
                      p * num_basis_values,
                  num_basis_values),
        cell_info, cell_index, reference_value_size);

    {
      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_derivatives_reference_values, 0, p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      push_forward_fn(basis_values, _U, _J, detJ[p], _K);
    }

    for (int i = 0; i < space_dimension; i++) {
      for (int j = 0; j < value_size; j++) {
        basis_values_span(i, j, p) = basis_values(i, j);
      }
    }
  }
  return basis_values_p;
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<T> basis_expand(const dolfinx::fem::Function<T, U> &f,
                            const std::vector<T> &basis_values,
                            const std::vector<int>& cells) {

  auto _function_space = f.function_space();
  auto element = _function_space->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size =
      element->reference_value_size() / bs_element;
  const std::size_t value_size = _function_space->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element; // no. of DOF

  assert(basis_values.size() == space_dimension*value_size*cells.size());

  std::vector<T> u(cells.size() * value_size);
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _function_space->dofmap();

  for (std::size_t p = 0; p < cells.size(); ++p) {
    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cells[p]);
    for (std::size_t i = 0; i < space_dimension; ++i) {
      auto coeff = f.x()->array()[dofs[i]];
      for (std::size_t j = 0; j < value_size; ++j) {
        u[p * value_size + (j)] +=
          coeff * basis_values[(i * value_size + j) * cells.size() + p];
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

template std::vector<double> basis_expand(const dolfinx::fem::Function<double,double> &u,
                                     const std::vector<double> &basis_values,
                                     const std::vector<int> &cells);

template std::vector<double> eval_reference_basis(
    const dolfinx::fem::Function<double, double> &f, std::span<const double> x,
    std::array<std::size_t, 2> xshape, std::span<const std::int32_t> cells);

template void
interpolate_same_map<double,double>(dolfinx::fem::Function<double,double>& u1,
                          dolfinx::fem::Function<double,double>& u0,
                          std::vector<double>& i_m, std::array<std::size_t, 2> im_shape); 
} // namespace dolfinx::CUDA
// // namespace dolfinx::CUDA
