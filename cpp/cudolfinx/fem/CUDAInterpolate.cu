#include "CUDAInterpolate.h"

__device__ double func(double x, double y, double z) { return x * x + 1; }

template<typename Func>
__global__ void cuda_eval_coordinates(int num_points, const double *xs,
                                      const double *ys, const double *zs,
                                      Func f, double *evals) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < num_points) {
    evals[ind] = f(xs[ind], ys[ind], zs[ind]);
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
        [] __device__(double x, double y, double z) { return  x * x; }, d_evals);
    cuda_copy_interpolate<<<dof_count / 128 + 1, 128>>>(dof_count, x, mask,
                                                        d_evals);
}
}
