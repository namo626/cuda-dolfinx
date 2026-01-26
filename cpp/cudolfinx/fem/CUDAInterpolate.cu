#include <cuda.h>
#include <vector>
#include <array>

__device__ double func(double x, double y, double z) {
    return 1. + 0.1*x*x + 0.2*y*y + 0.3*z*z;
}


__global__ void _gather_dofs(double* A, const double* _y, const int* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] = _y[mask[i]];
    }
}

__global__ void _scatter_dofs(const double* B, double* _x, const int* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        _x[mask[i]] = B[i];
        //_x[i] = 1.0;
    }
}

__global__ void _matmul(double* B, double* i_m, double* A, int P, int K, int C) {

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < P) && (col < C)) {
        double s = 0;
        for (int k = 0; k < K; k++) {
            s += i_m[row*K + k] * A[k*C + col];
        }
        B[row*C + col] = s;
    }
}
namespace dolfinx::CUDA {

void wrapper_cuda_interpolate_same_map(int P, int K, int C,
                                       CUdeviceptr _x,
                                       CUdeviceptr _y,
                                       CUdeviceptr i_m,
                                       CUdeviceptr dofs0_map,
                                       CUdeviceptr dofs1_map, int dvalues_size,
                                       std::array<std::size_t, 2> im_shape,
                                       std::vector<double> &output) {

  double *A, *B;
  cudaMalloc((void **)&A, C * K * sizeof(double));
  cudaMalloc((void **)&B, C * P * sizeof(double));

  _gather_dofs<<<C * K / 128 + 1, 128>>>(A, (double *)_y, (int *)dofs0_map,
                                         C * K);

  dim3 dimGrid(C / 16 + 1, P / 16 + 1, 1);
  dim3 dimBlock(16, 16, 1);
  _matmul<<<dimGrid, dimBlock>>>(B, (double *)i_m, A, P, K, C);

  double *tmp;
  cudaMalloc((void **)&tmp, dvalues_size);
  output.resize(dvalues_size / sizeof(double));

  _scatter_dofs<<<C * P / 128 + 1, 128>>>(B, tmp, (int *)dofs1_map, C * P);

  cudaMemcpy(output.data(), tmp, output.size() * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(tmp);
  cudaFree(A);
  cudaFree(B);
}
}
