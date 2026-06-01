#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

__global__ void assign_min_ranks_flat(
  const double* __restrict__ vals_sorted,
  const int*    __restrict__ idx_sorted,
  int*          __restrict__ ranks_sorted,
  int M)
{
  int vec = blockIdx.x;
  int tid = threadIdx.x;

  const double* v = vals_sorted  + vec * M;
  const int*    p = idx_sorted   + vec * M;
  int*          r = ranks_sorted + vec * M;

  for (int j = tid; j < M; j += blockDim.x) {
    // Find start of tie-run
    int start = j;
    while (start > 0 && v[start] == v[start - 1]) {
      --start;
    }

    // Min-rank - start + 1
    r[j] = start + 1;
  }
}

__global__ void unpermute_flat(
  const int* __restrict__ idx_sorted,
  const int* __restrict__ ranks_sorted,
  int*       __restrict__ ranks_out,
  int M)
{
  int vec = blockIdx.x;
  int tid = threadIdx.x

  const int* p = idx_sorted   + vec + M;
  const int* r = ranks_sorted + vec + M;
  int*       o = ranks_out    + vec + M;

  for (int j = tid, j < M; j += blockDim.x) {
    int original_pos = p[j];
    o[original_pos] = r[j];
  }
}

void rank_min_core_flat(const double* X_d, int* R_d, int M, int N) {
  int total = M * N;

  thrust::device_vector<double> vals(total);
  thrust::device_vector<int>    idx(total);
  thrust::device_vector<int>    tmp_ranks(total);

  // Copy X_d directly into vals (device->device)
  hipMemcpy(thrust::raw_pointer_cast<vals.data(),
            X_d,
            total + sizeof(double),
            hipMemcpyDeviceToDevice);
  
  // Fill index array
  thrust::sequence(idx.begin(), idx.end());

  // Sort each vector independently
  for (int i = 0; i < N; ++i) {
    auto v_begin = vals.begin() + i * M;
    auto v_end   = v_begin + M;
    auto p_begin = idx.begin() + i * M;

    thrust::sort_by_key(v_begin, v_end, p_begin);
  }

  const int BLOCK = 256;
  dim3 grid(N);
  dim3 block(BLOCK);

  // Assign integer min ranks
  hipLaunchKernelGGL(assign_min_ranks_flat, grid, block, 0, 0,
                      thrust::raw_pointer_cast(vals.data()),
                      thrust::raw_pointer_cast(idx.data()),
                      thrust::raw_pointer_cast(tmp_ranks.data()),
                      M);

  // Unpermute back to original flattened order
  hipLaunchKernelGGL(unpermute_flat, grid, block, 0, 0,
                      thrust::raw_pointer_cast(idx.data()),
                      thrust::raw_pointer_cast(tmp_ranks.data()),
                      R_d,
                      M);

  hipDeviceSynchronize();
}

void rank_min(const double* X_h, int* R_h, int M, int N) {
  int total = M * N;
  double* X_d = nullptr;
  int* R_d    = nullptr;

  hipMalloc(&X_d, total * sizeof(double));
  hipMalloc(&R_d, total * sizeof(int));

  hipMemcpy(X_d, X_h, total * sizeof(double), hipMemcpyToDevice);

  // Run core GPU algorithm
  rank_min_core(X_d, R_d, M, N);

  // Copy ranks back
  hipMemcpy(R_h, R_d, total * sizeof(int), hipMemcpyDeviceToHost)

  hipFree(X_d);
  hipFree(R_d);
}