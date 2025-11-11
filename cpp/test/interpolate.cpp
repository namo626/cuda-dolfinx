#include <cstddef>
#include <cudolfinx/fem/CUDACoefficient.h>
#include <dolfinx.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/generation.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>

template<typename T>
void printVec(const T& vec) {
  for (auto v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

template<typename T, typename U>
void allClose(const T& v1, const U& v2) {
  assert(v1.size() == v2.size());
  for (std::size_t i = 0; i < v1.size(); i++) {
    assert(std::abs((v1[i] - v2[i])/(v1[i])) < 1e-8);
  }
}

using T = double;
using namespace dolfinx;

int main(int argc, char* argv[]) {
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  
  CUdevice cuDevice = 0;
  CUcontext cuContext;
  const char * cuda_err_description;

  cuInit(0);
  int device_count;
  CUresult cuda_err = cuDeviceGetCount(&device_count);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error("cuDeviceGetCount failed with " +
                             std::string(cuda_err_description) + " at " +
                             std::string(__FILE__) + ":" +
                             std::to_string(__LINE__));
  }
  std::cout << "No. of devices: " << device_count << std::endl;

  cuCtxCreate(&cuContext, 0, cuDevice);
  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, 2,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  const auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_interval(MPI_COMM_WORLD, 10, {0,10}));
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element, {}));
  auto f = std::make_shared<fem::Function<T>>(V);
  auto f_true = std::make_shared<fem::Function<T>>(V);
  auto coeffs = dolfinx::fem::CUDACoefficient<double>(f);

  /* Reference version */
  f_true->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f;
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          f.push_back(1 + 0.10*x(0,p)*x(0,p) + 0.2*x(1,p)*x(1,p) + 0.3*x(2,p)*x(2,p));
        }

        return {f, {f.size()}};
      });
  //printVec(f_true->x()->array());


  /* Our version */

  // coeffs.interpolate(g);
  // //printVec(f->x()->array());
  // allClose(f_true->x()->array(), f->x()->array());

  coeffs.cuda_interpolate_test();
  auto d_coeffs = coeffs.x()->array();

  allClose(f_true->x()->array(), d_coeffs);

  std::cout << "PASSED" << std::endl;
  return 0;
}
