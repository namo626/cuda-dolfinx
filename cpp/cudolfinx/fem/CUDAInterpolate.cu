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
}
}
