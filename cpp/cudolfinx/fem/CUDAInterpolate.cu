#include <cuda.h>
#include <vector>
#include <array>
#include <concepts>

__device__ double func(double x, double y, double z) {
    return 1. + 0.1*x*x + 0.2*y*y + 0.3*z*z;
}

// Assign A with entries of B as specified in M.
// A and M are of size n.
template<std::floating_point T>
__global__ void _mask_right(T* A, const T* B, const int* M, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] = B[M[i]];
    }
}

// Write each entry B[i] to the location M[i] in A.
// B and M are of size n.
template<std::floating_point T>
__global__ void _mask_left(T* A, const T* B, const int* M, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[M[i]] = B[i];
    }
}

// Compute C = AB
// A is m x k, B is k x n, C is m x n
template<std::floating_point T>
__global__ void _matmul(T* C, const T* A, const T* B, int m, int k, int n) {

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < m) && (col < n)) {
        T s = 0;
        for (int kk = 0; kk < k; kk++) {
            s += A[row*k + kk] * B[kk*n + col];
        }
        C[row*n + col] = s;
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

  const int numThreads = 128;
  _mask_right<<<n*k/numThreads+1, numThreads>>>(X0, (double *)x0, (int *)M0,
                                         n*k);

  const int matSize = 16;
  dim3 dimGrid(n / matSize + 1, m / matSize + 1, 1);
  dim3 dimBlock(matSize, matSize, 1);
  _matmul<<<dimGrid, dimBlock>>>(X1, (double *)i_m, X0, m, k, n);

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
