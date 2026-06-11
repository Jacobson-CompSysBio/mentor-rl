#pragma once
#include <vector>
#include <mpi.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

#define CHECK_HIP(func)                                                   \
{                                                                         \
  hipError_t status = (func);                                             \
  if (status != hipSuccess) {                                             \
    int rank = 0;                                                         \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
    fprintf(stderr, "[Rank %d] HIP API failed at %s:%d: %s (%d)\n",       \
            rank, __FILE__, __LINE__, hipGetErrorString(status), status); \
    MPI_Abort(MPI_COMM_WORLD, 1);                                         \
  }                                                                       \
}

#define CHECK_HIPBLAS(func)                                                   \
{                                                                             \
  hipblasStatus_t status = (func);                                            \
  if (status != HIPBLAS_STATUS_SUCCESS) {                                     \
    int rank = 0;                                                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                     \
    fprintf(stderr, "[Rank %d] HIPBLAS API failed at %s:%d with error: %d\n", \
            rank, __FILE__, __LINE__, status);                                \
    MPI_Abort(MPI_COMM_WORLD, 1);                                             \
  }                                                                           \
}

#define CHECK_HIPSPARSE(func)                                                   \
{                                                                               \
  hipsparseStatus_t status = (func);                                            \
  if (status != HIPSPARSE_STATUS_SUCCESS) {                                     \
    int rank = 0;                                                               \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                       \
    fprintf(stderr, "[Rank %d] HIPSPARSE API failed at %s:%d with error: %d\n", \
            rank, __FILE__, __LINE__, status);                                  \
    MPI_Abort(MPI_COMM_WORLD, 1);                                               \
  }                                                                             \
}

namespace rwr {
class RWR_Worker {
public:
  RWR_Worker(const unsigned long N,
             const unsigned long L,
             const unsigned long intra_nnz,
             const unsigned long max_K,
             const double alpha = 0.7,
             const unsigned long _max_gpu_bytes = 40ULL << 30);
  ~RWR_Worker();

  void transfer_intra_transition_matrix(const std::vector<double> &values,
                                  const std::vector<int32_t> &col_idx,
                                  const std::vector<int32_t> &row_ptr);
  void transfer_inter_transition_matrix(const std::vector<double> &values);
  // void run_rwr_batch(std::vector<double> &encodings,
  //                    unsigned long offset,
  //                    const std::vector<double> &init_prob,
  //                    const std::string &reduction_method = "geometric",
  //                    const double threshold = 1e-8);
  void rwr_batch(std::vector<double>& encodings,
                 const std::vector<std::vector<std::string>>& seed_vec_set,
                 const std::vector<std::string>& node_labels,
                 const std::string& reduction_method = "geometric",
                 const std::vector<double>& tau = {},
                 const double threshold = 1e-8);

  unsigned long get_max_batch_size() const;

protected:
  unsigned long calc_max_micro_batch_size(const unsigned long max_K);
  void allocate_memory();
  void setup_sparse_descriptors(); 
  void add_interlayer_and_restart();
  void run_spmm();
  void zero_seeds();
  void reduce(const std::string &ruduction_method = "geometric");
  bool converged(const double abs_tol, const double rel_tol);
  void rwr_micro_batch(std::vector<double> &encodings,
                       unsigned long &offset,
                       const std::vector<double> &init_prob,
                       const std::string &reduction_method = "geometric",
                       const double threshold = 1e-8);

private:
  const unsigned long MAX_GPU_BYTES;  // maximum size to allocate on GPU
  const unsigned long N_;             // Number of unique node in mp
  const unsigned long L_;             // Number of layers in mp
  const unsigned long NL_;
  const unsigned long nnz_;           // Number of non-zero elements in intra-layer transition matrix
  const double alpha_;                // Probabilty of restart
  const unsigned long max_micro_batch_size;
  bool allocated;

  unsigned long K_; // current batch size
  
  // Device intra-layer transition matrix
  double *d_intra_values = nullptr; // [nnz]
  int32_t *d_col_idx = nullptr;     // [nnz]
  int32_t *d_row_ptr = nullptr;     // [NL + 1]

  // Device inter-layer transition matrix values
  double *d_inter_values = nullptr; // [NL]

  // RWR vectors
  double *d_P_in = nullptr;         // [NL * B]
  double *d_P_restart = nullptr;    // [NL * B]
  double *d_P_out = nullptr;        // [NL * B]
  double *d_norm = nullptr;         // [B]
  double* d_norm_ref = nullptr;

  // Buffer for SpMM
  void* d_spmm_buffer = nullptr;
  std::size_t spmm_buffer_size = 0;

  // hipBLAS handle
  hipblasHandle_t hipblas_handle;

  // hipSPARSE hanld and descriptors
  hipsparseHandle_t hipsparse_handle;
  hipsparseSpMatDescr_t spmat_descr;
  hipsparseDnMatDescr_t dmat_input_descr, dmat_output_descr;
};

} // namespace rwr