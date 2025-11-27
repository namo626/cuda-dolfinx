#include <basix/finite-element.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dolfinx.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/dolfinx_geometry.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <cudolfinx/fem/CUDACoefficient.h>
#include <memory>
#include <span>
#include <random>
#include <typeinfo>
#include <chrono>
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

template<typename T, typename U>
void allClose(const T& v1, const U& v2) {
  assert(v1.size() == v2.size());
  for (std::size_t i = 0; i < v1.size(); i++) {
    assert(std::abs((v1[i] - v2[i])/(v1[i])) < 1e-8);
  }
}

template<typename T>
void printVec(const std::vector<T>& vec) {
  for (auto v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}
template<typename T>
void printSpan(const std::span<T>& sp) {
  for (auto v : sp) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}


using namespace dolfinx;
using T = double;

std::vector<T> random_3d(T x0, T x1, int num_pts) {
  std::vector<T> coords;
  for (int i = 0; i < num_pts*3; i++) {
    T scale = std::rand() / (T) RAND_MAX;
    coords.push_back(x0 + scale*(x1-x0));
  }
  return coords;
}

std::vector<int> get_cells(int elem_size, std::vector<T> coords) {
  std::vector<int> cells;
  for ( auto x : coords ) {
    cells.push_back((int)std::floor(x / (T)elem_size));
  }
  return cells;
}

int main(int argc, char* argv[]) {
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  
  // auto element = basix::FiniteElement<T>(
  //       basix::element::family::P, basix::cell::type::triangle, 2,
  //       basix::element::lagrange_variant::unset,
  //       basix::element::dpc_variant::unset, false);

  const int num_cells = 20;
  const T lower = 0.;
  const T upper = 100.;
  const int num_evals = 100000;

  auto element = basix::create_element<T>(
                                          basix::element::family::P,
                                          basix::cell::type::tetrahedron,
                                          8,
                                          basix::element::lagrange_variant::equispaced,
                                          basix::element::dpc_variant::unset, false);
  const auto mesh = std::make_shared<mesh::Mesh<T>>(
                                                    mesh::create_box(
                                                                     MPI_COMM_WORLD,
                                                                     {{{lower,lower,lower},{upper,upper,upper}}},
                                                                     {num_cells,num_cells,num_cells},
                                                                     mesh::CellType::tetrahedron
                                                                     ));
  
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element, {}));

  auto f = std::make_shared<fem::Function<T>>(V);
  auto f_d = std::make_shared<fem::Function<T>>(V);

  auto dof = f->x()->array();


  /* Create coordinates */
  auto v_coords = random_3d(lower, upper, num_evals);
  std::span<const T> coords(v_coords);

  const int tdim = mesh->topology()->dim();
  auto cell_map = mesh->topology()->index_map(tdim);
  std::vector<std::int32_t> entities(cell_map->size_local(), 0);
  std::cout << "Entities size = " << entities.size() << std::endl;

  for (int i = 0; i < entities.size(); i++) {
    entities[i] = i;
  }

  const auto bb_tree = geometry::BoundingBoxTree<T>(*mesh, tdim, entities ,1e-8);
  std::cout << "Num_bboxes = " << bb_tree.num_bboxes() << std::endl;
  auto potential_cells = geometry::compute_collisions(bb_tree, coords);
  auto colliding_cells = geometry::compute_colliding_cells(*mesh, potential_cells, coords);

  std::vector<std::int32_t> cells;
  for (int i = 0; i < num_evals; i++) {
    if (colliding_cells.links(i).size() > 0)
      cells.push_back(colliding_cells.links(i)[0]);
  }

  std::cout << "Colliding cells size = " << cells.size() << std::endl;


  //printVec(cells);
  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f;
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          f.push_back(1 + 0.10*x(0,p)*x(0,p) + 0.2*x(1,p)*x(1,p) + 0.3*x(2,p)*x(2,p));
        }

        return {f, {f.size()}};
      });
  const int value_size = 1;
  std::vector<T> u(num_evals );

  const int ITER = 5;
  auto t1 = high_resolution_clock::now();
  for (int i = 0; i < ITER; i++) {
    f->eval(coords, {num_evals, 3}, cells, u, {num_evals, value_size});
  }
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms = t2 - t1;

  std::cout << "Average CPU eval time: " << ms.count() / (double)ITER << "ms\n";

  /// GPU version
  CUdevice cuDevice = 0;
  CUcontext cuContext;
  cuInit(0);
  cuCtxCreate(&cuContext, 0, cuDevice);

  auto coeffs = dolfinx::fem::CUDACoefficient<double>(f_d) ;
  coeffs.interpolate([](std::vector<double> x) -> std::vector<double> {
    int num_pts = x.size() / 3;
    std::vector<double> fs(num_pts);
    for (std::size_t p = 0; p < num_pts; ++p) {
      fs[p] = 1 + 0.10 * x[p] * x[p] + 0.20 * x[num_pts + p] * x[num_pts + p] +
              0.30 * x[2 * num_pts + p] * x[2 * num_pts + p];
    }
    return fs;
  });
  std::vector<T> v;
  // Initialize the basis values at coords
  v = coeffs.eval(coords, {num_evals,3}, cells);
  allClose(u, v);

  t1 = high_resolution_clock::now();
  for (int i = 0; i < ITER; i++) {
    coeffs.eval(coords, {num_evals,3}, cells);
  }
  t2 = high_resolution_clock::now();
  ms = t2 - t1;
  std::cout << "Avg eval time: " << ms.count()/(double)ITER << "ms" << std::endl;

  return 0;
}

