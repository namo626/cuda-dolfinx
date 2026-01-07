#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <cudolfinx/fem/CUDACoefficient.h>


namespace nb = nanobind;

template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void declare_cuda_coefficient(nb::module_& m)
{
  nb::class_<dolfinx::fem::CUDACoefficient<T,U>>(m, "CUDACoefficient", "Device side function")
  .def(nb::init<std::shared_ptr<const dolfinx::fem::Function<T,U>>>(),
  "Create device side function from a given Function")
  .def("interpolate",
       [](dolfinx::fem::CUDACoefficient<T,U>& self,
          std::shared_ptr<dolfinx::fem::Function<T,U>> g) {
         self.interpolate(g);
       },
       "Interpolate from another Function defined on the same mesh")
  .def("values",
       [](dolfinx::fem::CUDACoefficient<T,U>& self) {
         auto v = self.values();
         return nb::ndarray<T, nb::numpy, nb::c_contig>(v.data(), {v.size()}).cast();
       });
}

namespace cudolfinx_wrappers
{
void coefficient(nb::module_& m) {
    declare_cuda_coefficient<double,double>(m);
}
}
