#pragma once

#include <cstddef>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/fem/interpolate.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace dolfinx::CUDA {

template<std::floating_point T>
void wrapper_cuda_interpolate_same_map(int P, int K, int C, CUdeviceptr _x,
                                       CUdeviceptr _y, CUdeviceptr i_m,
                                       CUdeviceptr dofs0_map,
                                       CUdeviceptr dofs1_map, int dvalues_size,
                                       std::array<std::size_t, 2> im_shape,
                                       std::vector<T> &output);

template<std::floating_point T>
void interpolate_same_map(const dolfinx::fem::Function<T, T> &u1,
                               const dolfinx::fem::Function<T, T> &u0,
                               CUdeviceptr _x,
                               int dvalues_size,
                               CUdeviceptr _y,
                               CUdeviceptr i_m,
                               std::array<std::size_t, 2> im_shape,
                               CUdeviceptr dofs0_map, CUdeviceptr dofs1_map,
                               std::vector<T>& output) {

  auto V0 = u0.function_space();
  auto V1 = u1.function_space();
  auto mesh0 = V0->mesh();
  const int tdim = mesh0->topology()->dim();
  auto map = mesh0->topology()->index_map(tdim);

  // Get all cells
  std::vector<std::int32_t> cells(map->size_local() + map->num_ghosts(), 0);
  const std::size_t P = im_shape[0];
  const std::size_t K = im_shape[1];
  const std::size_t C = cells.size();

  wrapper_cuda_interpolate_same_map(P,K,C, _x,_y,i_m,dofs0_map, dofs1_map, dvalues_size, im_shape, output);
}

template <dolfinx::scalar T, std::floating_point U>
void create_interpolation_maps(const dolfinx::fem::Function<T, U>& u1,
                          const dolfinx::fem::Function<T, U>& u0,
                          std::vector<T>& i_m, std::array<std::size_t, 2> im_shape,
                          std::vector<std::int32_t>& dofs0_map,
                          std::vector<std::int32_t>& dofs1_map) {

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
  std::span<const T> u1_array = u1.x()->array();
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
  auto apply_dof_transformation_0 = element0->template dof_transformation_fn<std::int32_t>(
      dolfinx::fem::doftransform::transpose, false);
  auto apply_dof_transformation_1 = element1->template dof_transformation_fn<std::int32_t>(
      dolfinx::fem::doftransform::transpose, false);


  assert(im_shape[0] == element1->space_dimension());
  assert(im_shape[1] == element0->space_dimension());


  std::vector<std::int32_t> local_dofs0(im_shape[1]);
  std::vector<std::int32_t> local_dofs1(im_shape[0]);
  dofs0_map.resize(cells0.size() * im_shape[1]);
  dofs1_map.resize(cells0.size() * im_shape[0]);

  for (std::size_t c = 0; c < cells0.size(); c++) {
    // Pack and transform cell dofs to reference ordering
    std::span<const std::int32_t> D_c = dofmap0->cell_dofs(cells0[c]);
    // local_dofs0 = [ 0, ..., k-1 ]
    std::iota(local_dofs0.begin(), local_dofs0.end(), 0);

    // Permute the vector [0, ..., k-1]
    apply_dof_transformation_0(local_dofs0, cell_info0, cells0[c], 1);

    for (std::size_t i = 0; i < im_shape[1]; i++) {
      dofs0_map[i * cells0.size() + c] = D_c[local_dofs0[i]];
    }

    // local_dofs1 = [ 0, ..., p-1 ]
    std::iota(local_dofs1.begin(), local_dofs1.end(), 0);
    apply_dof_transformation_1(local_dofs1, cell_info1, cells0[c], 1);

    std::span<const std::int32_t> E_c = dofmap1->cell_dofs(cells1[c]);

    for (std::size_t i = 0; i < im_shape[0]; ++i) {
      dofs1_map[i * cells1.size() + c] = E_c[local_dofs1[i]];
    }
  }
}


}
