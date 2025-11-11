from cudolfinx import cpp as _cucpp
from dolfinx.fem.function import Function
from cudolfinx.context import get_cuda_context

class Coefficient:
    def __init__(self,
                 f: Function):
        self._ctx = get_cuda_context()
        self._cpp_object = _cucpp.fem.CUDACoefficient(f._cpp_object)

    def interpolate(self,
                    g: Function):
        return self._cpp_object.interpolate(g._cpp_object)

    def values(self):
        return self._cpp_object.values()
