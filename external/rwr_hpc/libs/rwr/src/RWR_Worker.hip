#include "rwr/RWR_Worker.hpp"
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <batch_generator/BatchGen.hpp>
#include "rwr/utils.hpp"

// Small epsilon to avoid division by ~0 in relative check
static inline double _safe_eps() { return 1e-16; }

__global__ void scale_vector_kernel (
  double* __restrict__ P_restart,     // Output: α · P_init
  const double* __restrict__ P_init,  // Input seed vector
  const double alpha,                 // scale factor
  const std::size_t N                 // Total elements in vetors
) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    P_restart[idx] = alpha * P_init[idx];
  }
}

__global__ void fused_add_interlayer_restart (
  double* __restrict__ P_out,
  const double* __restrict__ P_in,
  const double* __restrict__ P_restart,
  const double* __restrict__ D_scaled,
  std::size_t N, size_t L, size_t K) {

  std::size_t node = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t seed = blockIdx.y * blockDim.y + threadIdx.y;

  if (node >= N || seed >= K) return;

  double sum = 0.0;
  std::size_t seed_offest = seed * N * L;

  // Unroll over L with wavefront size 64
  // const std::size_t unroll_factor = 64;
  // for (std::size_t l = threadIdx.x % unroll_factor; l < L; l += unroll_factor) {
  for (std::size_t l = 0; l < L; ++l) {
    std::size_t i = l * N + node;
    std::size_t idx = i + seed_offest;

    double weight = D_scaled[i];
    double val = P_in[idx];
    if (weight != 0.0 && val != 0.0) {
      sum += weight * val;
    }
  }

  // Write result for each layer with subtractive inter-layer update + restart
  for (std::size_t l = 0; l < L; ++l) {
    std::size_t i = l * N + node;
    std::size_t idx = i + seed_offest;
    double own = D_scaled[i] * P_in[idx];
    P_out[idx] = (sum - own) + P_restart[idx];
  }
}

__global__ void zero_if_restart_nonzero(
  const double* __restrict__ d_P_restart,  // [NL * K]
  double* __restrict__ d_P_out,            // [NL * K]
  std::size_t NLK                          // total size = N * L * K
) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NLK) return;

  if (d_P_restart[idx] != 0.0) {
    d_P_out[idx] = 0.0;
  }
}

__global__ void reduce_geometric_mean_batched_inplace(
  double* __restrict__ g_odata,  // [N * L * K], will store result at beginning
  const double* __restrict__ g_idata,  // [N * L * K]
  std::size_t N,
  std::size_t L,
  std::size_t K)
{
  std::size_t node = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t seed = blockIdx.y * blockDim.y + threadIdx.y;

  if (node >= N || seed >= K) return;

  double log_sum = 0.0;
  std::size_t seed_offset = seed * N * L;
  // Input is interpreted as node + (l * N) + (seed * N * L)
  int has_zero = 0;
  for (std::size_t layer = 0; layer < L; ++layer) {
    std::size_t idx = node + (layer * N) + seed_offset;
    double val = g_idata[idx];
    if (val == 0.0) {
      has_zero = 1;
    } else {
      log_sum += log(val);
    }
  }

  if (has_zero) {
    g_odata[node + seed * N] = 0.0;
  } else {
    g_odata[node + seed * N] = exp(log_sum / static_cast<double>(L));
  }
  
}

__global__ void reduce_arithmetic_mean_batched_inplace(
  double* __restrict__ g_odata,  // [N * L * K], will store result at beginning
  const double* __restrict__ g_idata,  // [N * L * K]
  std::size_t N,
  std::size_t L,
  std::size_t K)
{
  std::size_t node = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t seed = blockIdx.y * blockDim.y + threadIdx.y;

  if (node >= N || seed >= K) return;

  double sum = 0.0;
  std::size_t seed_offset = seed * N * L;
  // Input is interpreted as node + (l * N) + (seed * N * L)
    for (std::size_t layer = 0; layer < L; ++layer) {
    std::size_t idx = node + (layer * N) + seed_offset;
    sum += g_idata[idx];
  }

  g_odata[node + seed * N] = sum / static_cast<double>(L);
}


namespace rwr {
RWR_Worker::RWR_Worker(const unsigned long N,
                       const unsigned long L,
                       const unsigned long intra_nnz,
                       const unsigned long max_K,
                       const double alpha,
                       const unsigned long _max_gpu_bytes) : MAX_GPU_BYTES(_max_gpu_bytes),
                                                      N_(N),
                                                      L_(L),
                                                      NL_(N_ * L_),
                                                      nnz_(intra_nnz),
                                                      alpha_(alpha),
                                                      max_micro_batch_size(calc_max_micro_batch_size(max_K)),
                                                      allocated(false),
                                                      K_(0) {
  CHECK_HIPBLAS( hipblasCreate(&hipblas_handle) )
  CHECK_HIPSPARSE( hipsparseCreate(&hipsparse_handle) )

  // Allocate memory. Allow upto max_micro_batch_size
  allocate_memory();
  setup_sparse_descriptors();
}

RWR_Worker::~RWR_Worker() {
  if (allocated) {
    CHECK_HIP( hipFree(d_intra_values) )
    CHECK_HIP( hipFree(d_col_idx) )
    CHECK_HIP( hipFree(d_row_ptr) )

    CHECK_HIP( hipFree(d_inter_values) )

    CHECK_HIP( hipFree(d_P_in) )
    CHECK_HIP( hipFree(d_P_restart) )
    CHECK_HIP( hipFree(d_P_out) )

    CHECK_HIP( hipFree(d_norm) )
    CHECK_HIP( hipFree(d_norm_ref) )

    CHECK_HIP( hipFree(d_spmm_buffer) )

    CHECK_HIPSPARSE( hipsparseDestroySpMat(spmat_descr) )
    CHECK_HIPSPARSE( hipsparseDestroyDnMat(dmat_input_descr) )
    CHECK_HIPSPARSE( hipsparseDestroyDnMat(dmat_output_descr) )

    CHECK_HIPBLAS( hipblasDestroy(hipblas_handle) )
    CHECK_HIPSPARSE( hipsparseDestroy(hipsparse_handle) )
  }
}

void RWR_Worker::transfer_intra_transition_matrix(const std::vector<double> &values,
                                            const std::vector<int32_t> &col_idx,
                                            const std::vector<int32_t> &row_ptr) {
  // Veryify vector size matches allocated memory
  if (values.size() != nnz_) {
    throw std::invalid_argument("RWR_Worker::transfer_intra_transition_matrix - values.size must equal allocated nnz");
  }
  if (col_idx.size() != nnz_) {
    throw std::invalid_argument("RWR_Worker::transfer_intra_transition_matrix - col_idx.size must equal allocated nnz");
  }
  if (row_ptr.size() != N_*L_ + 1) {
    throw std::invalid_argument("RWR_Worker::transfer_intra_transition_matrix - row_ptr.size must equal allocated (NL+1)");
  }

  // Make sure memory has been allocated
  if (!allocated) {
    allocate_memory();
  }

  CHECK_HIP( hipMemcpy(d_intra_values, values.data(), nnz_ * sizeof(double), hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(d_col_idx, col_idx.data(), nnz_ * sizeof(int32_t), hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(d_row_ptr, row_ptr.data(), (NL_ + 1) * sizeof(int32_t), hipMemcpyHostToDevice) )

  double one_minus_alpha = 1.0 - alpha_;
  CHECK_HIPBLAS( hipblasDscal(
    hipblas_handle,
    nnz_,                // number of elements in x
    &one_minus_alpha,    // pointer to scalar
    d_intra_values,      // pointer to vector x (in-place)
    1                    // stride (usually 1)
  ) )

}

void RWR_Worker::rwr_batch(std::vector<double>& encodings,
                    const std::vector<std::vector<std::string>>& seed_vec_set,
                    const std::vector<std::string>& node_labels,
                    const std::string& reduction_method,
                    const std::vector<double>& tau,
                    const double threshold) {
  // Allocate space for encodings
  const std::size_t N = node_labels.size();
  const std::size_t encoding_size = get_encoding_size(reduction_method, N_, L_);
  const std::size_t n_encodings = seed_vec_set.size();
  encodings.resize(encoding_size * n_encodings);

  // Determine maximum micro batch size
  const auto max_micro_batch_size = get_max_batch_size();

  // Create batch generator and loop until all micro batches encoded
  unsigned long offset = 0;
  BatchGen<unsigned long> gen(0, n_encodings-1, max_micro_batch_size);
  while (!gen.done()) {

    // Get the local index (batch_start and batch_stop) for current micro batch
    auto [batch_start, batch_stop] = gen.next();
    
    // Create slice of seed_vec_set for current micro batch
    std::vector<std::vector<std::string>> batch_seeds(seed_vec_set.begin() + batch_start,
                                                      seed_vec_set.begin() + batch_stop + 1);
    
    // Create initial probability vectors from seed_vec_set. Save in flatened vector
    std::vector<double> init_prob = get_init_probs(batch_seeds, node_labels, L_, tau);

    // Encode batch and store in flatened array. Use offset to determine starting location
    // of current batch
    rwr_micro_batch(encodings, offset, init_prob, reduction_method, threshold);
  }
}

void RWR_Worker::rwr_micro_batch(std::vector<double> &encodings,
                                      unsigned long &offset,
                                      const std::vector<double> &init_prob,
                                      const std::string &reduction_method,
                                      const double threshold) {
  // Verify that init_prob is multiple of NL
  if (init_prob.size() % NL_ != 0) {
    throw std::runtime_error("RWR_Worker::run_rwr_batch - init_prob must be mulitple of NL");
  }

  K_ = init_prob.size() / NL_;
  if (K_ > max_micro_batch_size) {
    fprintf(stderr, "batch size %lu vs. %lu\n", K_, max_micro_batch_size);
    throw std::runtime_error("RWR_Worker::run_rwr_batch - K must not exceed max_micro_batch_size");
  }

  // Determine size of reduced vectors
  const unsigned long reduced_vector_size = N_;
  const unsigned long reduced_batch_size = K_ * reduced_vector_size;

  // Copy initial probabilites to GPU
  CHECK_HIP( hipMemcpy(d_P_in, init_prob.data(), init_prob.size() * sizeof(double), hipMemcpyHostToDevice) )

  // Set P restart as alpha * P initial
  std::size_t total_elements = init_prob.size();
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  scale_vector_kernel<<<grid_size, block_size>>>(
    d_P_restart, d_P_in, alpha_, total_elements
  );
  CHECK_HIP( hipGetLastError() )

  // Perform RWR
  // Perform several Markov steps before checking norm
  while (true) {
    std::size_t checks_every = 3;
    std::size_t max_iters    = 1000;
    std::size_t iter = 0;
    while (iter++ < checks_every) {
      add_interlayer_and_restart(); // computes inter-layer + restart into P_out
      run_spmm();                   // adds T_intra * P_in into P_out

      std::swap(d_P_in, d_P_out);
      CHECK_HIPSPARSE( hipsparseDnMatSetValues(dmat_input_descr, d_P_in) )
      CHECK_HIPSPARSE( hipsparseDnMatSetValues(dmat_output_descr, d_P_out) )
    }

    // Check if all vectors have converged
    if (converged(threshold, threshold) || iter == max_iters) { break; }
  }

  // Zero encodings at non-zero seeds values
  // zero_seeds();

  // Reduce rwr encodings
  reduce(reduction_method);

  // Copy results into the flattened buffer
  CHECK_HIP( hipMemcpy(encodings.data() + offset,
            d_P_out,
            reduced_batch_size * sizeof(double),
            hipMemcpyDeviceToHost) )
  
  // Update offest
  offset +=  reduced_batch_size;
}

unsigned long rwr::RWR_Worker::get_max_batch_size() const {
  return max_micro_batch_size;
}

void rwr::RWR_Worker::transfer_inter_transition_matrix(const std::vector<double> &values) {
  // Veryify vector size matches allocated memory
  if (values.size() != NL_) {
    throw std::invalid_argument("RWR_Worker::transfer_inter_transition_matrix - values.size() must equal allocated NL");
  }
  
  // Make sure memory has been allocated
  if (!allocated) {
    allocate_memory();
  }

  CHECK_HIP( hipMemcpy(d_inter_values, values.data(), NL_ * sizeof(double), hipMemcpyHostToDevice) )

  double one_minus_alpha = 1.0 - alpha_;
  CHECK_HIPBLAS( hipblasDscal(
    hipblas_handle,
    NL_,                 // number of elements in x
    &one_minus_alpha,    // pointer to scalar
    d_inter_values,      // pointer to vector x (in-place)
    1                    // stride (usually 1)
  ) )
}

unsigned long RWR_Worker::calc_max_micro_batch_size(const unsigned long max_K) {
  unsigned long fixed_mem = nnz_ * sizeof(double) +        // intra_values
                            nnz_ * sizeof(int32_t) +       // col_idx
                            (NL_ + 1) * sizeof(int32_t) +  // row_ptr
                            NL_ * sizeof(double);          // inter_values
  
  unsigned long per_B = (3 * NL_ + 2) * sizeof(double); // d_P_in, d_P_out, d_P_restart, d_norm, d_norm_ref

  // Can use this call to get available memory at run time
  // std::size_t free_mem = 0, total_mem = 0;
  // hipMemGetInfo(&free_mem, &total_mem);

  if (fixed_mem >= MAX_GPU_BYTES) {
    throw std::runtime_error("Fixed memory for RWR to too large for GPU");
    return 0;
    // TODO: add error or exception?
  }

  unsigned long calc_max_K = (MAX_GPU_BYTES - fixed_mem) / per_B;

  calc_max_K =  calc_max_K < max_K ? calc_max_K : max_K;
  // fprintf(stderr, "RWR mem based on max_K: %lu\n", fixed_mem + calc_max_K * per_B);

  return calc_max_K;
}

void RWR_Worker::allocate_memory() {
  // Allocate memory for intra-layer transition matrix 
  CHECK_HIP( hipMalloc(&d_intra_values, nnz_ * sizeof(double)) )
  CHECK_HIP( hipMalloc(&d_col_idx, nnz_ * sizeof(int32_t)) )
  CHECK_HIP( hipMalloc(&d_row_ptr, (NL_ + 1) * sizeof(int32_t)) )

  // Allocate memory for inter-layer transition matrix
  CHECK_HIP( hipMalloc(&d_inter_values, NL_ * sizeof(double)) )

  // Allocate memory for RWR vectors (init, restat, out )
  const unsigned long vec_bytes = NL_ * max_micro_batch_size * sizeof(double);
  CHECK_HIP( hipMalloc(&d_P_in, vec_bytes) )
  CHECK_HIP( hipMalloc(&d_P_restart, vec_bytes) )
  CHECK_HIP( hipMalloc(&d_P_out, vec_bytes) )
  CHECK_HIP( hipMalloc(&d_norm, max_micro_batch_size * sizeof(double)))
  CHECK_HIP( hipMalloc(&d_norm_ref, max_micro_batch_size * sizeof(double)))

  // Set allocated flag
  allocated = true;
}

void RWR_Worker::setup_sparse_descriptors() {
  CHECK_HIPSPARSE( hipsparseCreateCsr(&spmat_descr, NL_, NL_, nnz_,
                      d_row_ptr, d_col_idx, d_intra_values,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F) )

  CHECK_HIPSPARSE( hipsparseCreateDnMat(&dmat_input_descr, NL_, max_micro_batch_size, NL_, d_P_in,
                        HIP_R_64F, HIPSPARSE_ORDER_COL) )

  CHECK_HIPSPARSE( hipsparseCreateDnMat(&dmat_output_descr, NL_, max_micro_batch_size, NL_, d_P_out,
                        HIP_R_64F, HIPSPARSE_ORDER_COL) )
  
  double alpha = 1.0, beta = 1.0;
  CHECK_HIPSPARSE( hipsparseSpMM_bufferSize(
    hipsparse_handle,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    spmat_descr,
    dmat_input_descr,
    &beta,
    dmat_output_descr,
    HIP_R_64F,
    HIPSPARSE_SPMM_CSR_ALG2,
    &spmm_buffer_size) )

  CHECK_HIP( hipMalloc(&d_spmm_buffer, spmm_buffer_size) )

  // fprintf(stderr, "Reserved %lu bytes for SpMM buffer\n", spmm_buffer_size);
}

void RWR_Worker::add_interlayer_and_restart() {
  hipDeviceProp_t prop;
  int device;
  CHECK_HIP( hipGetDevice(&device) )
  CHECK_HIP( hipGetDeviceProperties(&prop, device) )

  std::size_t max_threads_per_block = prop.maxThreadsPerBlock;

  // Use 64 for AMD wavefront size
  std::size_t block_x = 64;
  std::size_t block_y = std::min(max_threads_per_block / block_x, K_);
  if (block_y < 1) block_y = 1;

  dim3 block(block_x, block_y);
  dim3 grid((N_ + block_x - 1) / block_x, (K_ + block_y - 1) / block_y);

  fused_add_interlayer_restart<<<grid, block>>>(
    d_P_out, d_P_in, d_P_restart, d_inter_values,
    N_, L_, K_
  );
  CHECK_HIP( hipGetLastError() )
}

void RWR_Worker::run_spmm() {
  double one = 1.0;
  CHECK_HIPSPARSE( hipsparseSpMM(
    hipsparse_handle,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    &one,
    spmat_descr,
    dmat_input_descr,
    &one,
    dmat_output_descr,
    HIP_R_64F,
    HIPSPARSE_SPMM_CSR_ALG2,
    d_spmm_buffer) )
}

void RWR_Worker::zero_seeds() {
  std::size_t NLK = N_ * L_ * K_;
  dim3 block(256);
  dim3 grid((NLK + block.x - 1) / block.x);

  zero_if_restart_nonzero<<<grid, block>>>(
    d_P_restart, d_P_in, NLK
  );
  CHECK_HIP( hipGetLastError() )
}

void RWR_Worker::reduce(const std::string &encoding_method) {
  hipDeviceProp_t prop;
  int device;
  CHECK_HIP( hipGetDevice(&device) )
  CHECK_HIP( hipGetDeviceProperties(&prop, device) )

  std::size_t max_threads_per_block = prop.maxThreadsPerBlock;

  // Use 64 for AMD wavefront size
  std::size_t block_x = 64;
  std::size_t block_y = std::min(max_threads_per_block / block_x, K_);
  if (block_y < 1) block_y = 1;

  dim3 block(block_x, block_y);
  dim3 grid((N_ + block_x - 1) / block_x, (K_ + block_y - 1) / block_y);

  if (encoding_method.compare("geometric") == 0) {
    reduce_geometric_mean_batched_inplace<<<grid, block>>>(
      d_P_out, d_P_in, N_, L_, K_
    );
    CHECK_HIP( hipGetLastError() )
  } else if (encoding_method.compare("arithmetic") == 0) {
    reduce_arithmetic_mean_batched_inplace<<<grid, block>>>(
      d_P_out, d_P_in, N_, L_, K_
    );
    CHECK_HIP( hipGetLastError() )
  } else {
    throw std::invalid_argument("RWR_Worker::reduce - unknown encoding method");
  }
}

// Returns true iff for every k:  ||Δ_k||2 < abs_tol  OR  ||Δ_k||2 / max(||r_k||2, eps) < rel_tol
bool RWR_Worker::converged(const double abs_tol, const double rel_tol) {
  // We assume at this point:
  // - d_P_in  holds the current vectors r_k (shape: NL_ × K_)
  // - d_P_out holds the previous vectors r_k(prev) (same shape)
  //
  // We will:
  //   1) Compute per-batch ||r_k||2 into d_norm_ref
  //   2) Overwrite d_P_out with Δ_k = r_k(prev) - r_k (via DAXPY with alpha=-1)
  //   3) Compute per-batch ||Δ_k||2 into d_norm
  //   4) Copy both norms to host and check absolute OR relative tolerances
  
  // Stride info for hipBLAS batched norms
  const int    n         = static_cast<int>(NL_);         // elements per vector
  const int    incx      = 1;                             // contiguous
  const long   stride    = static_cast<long>(NL_);        // stride between vectors (in elements)
  const int    batch     = static_cast<int>(K_);
  const double alpha_neg = -1.0;

  // (1) Compute ||r_k||2 into d_norm_ref
  CHECK_HIPBLAS(
    hipblasDnrm2StridedBatched(
      hipblas_handle,
      n,                 // number of elements per vector
      d_P_in,            // pointer to first vector r_0
      incx,              // stride within a vector
      stride,            // stride between vectors
      batch,             // number of vectors
      d_norm_ref         // output array [K_], device
    )
  )

  // (2) Calculate difference between d_P_in and d_P_out. Store in d_P_out
  const long total_elements = static_cast<long>(NL_) * static_cast<long>(K_);
  CHECK_HIPBLAS( hipblasDaxpy(hipblas_handle, total_elements,
                              &alpha_neg, d_P_in, 1, d_P_out, 1))

  // (3) Compute ||Δ_k||2 into d_norm
  CHECK_HIPBLAS(
    hipblasDnrm2StridedBatched(
      hipblas_handle,
      n,            // elements in each vector
      d_P_out,      // pointer to first vector
      incx,         // stride between elements
      stride,       // stride between vectors
      batch,        // number of vectors
      d_norm        // output buffer: one norm per encoding
    )
  )

  // (4) Copy norms to host and test tolerances
  std::vector<double> h_norm_ref(K_);
  std::vector<double> h_norm_diff(K_);
  CHECK_HIP( hipMemcpy(h_norm_ref.data(), d_norm_ref, K_ * sizeof(double), hipMemcpyDeviceToHost) )
  CHECK_HIP( hipMemcpy(h_norm_diff.data(), d_norm,    K_ * sizeof(double), hipMemcpyDeviceToHost) )

  const double eps = _safe_eps();

  for (std::size_t k = 0; k < K_; ++k) {
    const double ref = h_norm_ref[k];
    const double diff = h_norm_diff[k];

    const bool abs_ok = (diff < abs_tol);
    const bool rel_ok = (rel_tol > 0.0) ? (diff / std::max(ref, eps) < rel_tol) : false;
    if (!(abs_ok || rel_ok)) {
      return false; // at least one vector hasn't converged
    }
  }
  return true; // all vectors passed either abs or rel test
}

}
