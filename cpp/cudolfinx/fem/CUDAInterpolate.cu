#include "CUDAInterpolate.h"

__device__ double func(double x, double y, double z) {
    return 1. + 0.1*x*x + 0.2*y*y + 0.3*z*z;
}

__global__ void cuda_eval_coordinates(int num_points, const double *xs,
                                      const double *ys, const double *zs,
                                      double *evals) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < num_points) {
    evals[ind] = func(xs[ind], ys[ind], zs[ind]);
  }
}
__global__ void cuda_copy_interpolate(int dof_count, double *x, const int *mask,
                                      const double *evals) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < dof_count) {
    x[ind] = evals[mask[ind]];
  }
}

__global__ void _basis_expand(int num_cells, int bs_element, int space_dimension,
                            int value_size, double* u,
                            double* coeffs, double* basis_values_p, int* dofmap)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < num_cells) {
    for (int k = 0; k < bs_element; ++k) {
      for (std::size_t i = 0; i < space_dimension; ++i) {
        int coeff_ind = dofmap[p * space_dimension + i];
        for (std::size_t j = 0; j < value_size; ++j) {
          u[p * value_size + (j * bs_element + k)] +=
              coeffs[coeff_ind] *
              basis_values_p[(i * value_size + j) * num_cells + p];
        }
      }
    }
  }
}

namespace dolfinx::CUDA {
    
void cuda_wrapper_interpolate(int dof_count, int num_points, const double *xs,
                              const double *ys, const double *zs,
                              const int *mask, double *x) {

    double* d_evals;
    cudaMalloc((void **)&d_evals, num_points * sizeof(double));

    cuda_eval_coordinates<<<num_points / 128 + 1, 128>>>(
        num_points, xs, ys, zs,
        d_evals);
    cuda_copy_interpolate<<<dof_count / 128 + 1, 128>>>(dof_count, x, mask,
                                                        d_evals);
    cudaFree(d_evals);
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<T> cuda_basis_expand(const dolfinx::fem::Function<T, U> &f,
                                 CUdeviceptr dofmap, CUdeviceptr coeffs,
                                 CUdeviceptr dbasis_values, int num_cells) {

  auto _function_space = f.function_space();
  auto element = _function_space->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size =
      element->reference_value_size() / bs_element;
  const std::size_t value_size = _function_space->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element; // no. of DOF


  std::vector<T> u(num_cells * value_size);
  double* d_u;
  cudaMalloc((void**)&d_u, u.size()*sizeof(double));

  _basis_expand<<<1,1>>>(num_cells, 1, space_dimension, value_size,
                         d_u , (double*)coeffs, (double*)dbasis_values, (int*)dofmap);

  cudaMemcpy(u, d_u, u.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_u);

  return u;
}
}
