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

__global__ void _basis_expand(int num_cells, const int* cells, int bs_element, int space_dimension,
                            int value_size, double* u,
                            double* coeffs, double* basis_values_p, int* dofmap)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < num_cells) {
      u[p] = 0.;
      int cell_ind = cells[p];
    for (int k = 0; k < bs_element; ++k) {
      for (std::size_t i = 0; i < space_dimension; ++i) {
        int coeff_ind = dofmap[cell_ind * space_dimension + i];
        for (std::size_t j = 0; j < value_size; ++j) {
          u[p * value_size + (j * bs_element + k)] +=
              coeffs[coeff_ind] *
              basis_values_p[(i * value_size + j) * num_cells + p];
        }
      }
    }
  }
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

std::vector<double> cuda_basis_expand(const dolfinx::fem::Function<double, double> &f,
                                 CUdeviceptr dofmap, CUdeviceptr coeffs,
                                 CUdeviceptr dbasis_values, const std::vector<int>& cells) {

    const int num_cells = cells.size();
  auto _function_space = f.function_space();
  auto element = _function_space->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size =
      element->reference_value_size() / bs_element;
  const std::size_t value_size = _function_space->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element; // no. of DOF


  std::vector<double> u(num_cells * value_size);
  double* d_u;
  int* d_cells;
  cudaMalloc((void**)&d_u, u.size()*sizeof(double));
  cudaMalloc((void**)&d_cells, cells.size()*sizeof(int));
  cudaMemcpy(d_cells, cells.data(), cells.size()*sizeof(int), cudaMemcpyHostToDevice);

  _basis_expand<<<num_cells/128+1, 128>>>(num_cells, d_cells, 1, space_dimension, value_size,
                         d_u , (double*)coeffs, (double*)dbasis_values, (int*)dofmap);

  cudaMemcpy(u.data(), d_u, u.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  cudaFree(d_cells);

  return u;
}

void cuda_interpolate_same_map(const dolfinx::fem::Function<double, double> &u1,
                               const dolfinx::fem::Function<double, double> &u0,
                               CUdeviceptr _x,
                               CUdeviceptr _y,
                               CUdeviceptr i_m,
                               std::array<std::size_t, 2> im_shape,
                               CUdeviceptr dofs0_map, CUdeviceptr dofs1_map) {

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


  double *A, *B;
  cudaMalloc((void**)&A, C*K*sizeof(double));
  cudaMalloc((void**)&B, C*P*sizeof(double));

  _gather_dofs<<<C*K/128+1, 128>>>(A, (double*)_y, (int*)dofs0_map, C*K);

  dim3 dimGrid(C/16+1, P/16+1, 1);
  dim3 dimBlock(16,16, 1);
  _matmul<<<dimGrid,dimBlock>>>(B, (double*)i_m, A, P, K, C);

  _scatter_dofs<<<C*P/128+1, 128>>>(B, (double*)_x, (int*)dofs1_map, C*P);



  cudaFree(A);
  cudaFree(B);

}
}
