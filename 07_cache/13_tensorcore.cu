// 13_tensorcore.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,        \
                   __LINE__, static_cast<int>(st__));                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void float_to_half_kernel(const float *__restrict__ src,
                                     half *__restrict__ dst,
                                     std::size_t count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < count; i += stride) {
    dst[i] = __float2half_rn(src[i]);
  }
}

// A: column-major, shape m x k, index A[col * m + row]
// B: column-major, shape k x n, index B[col * k + row]
// C: column-major, shape m x n, index C[col * m + row]
//
// This version:
// - BM=128, BN=128, BK=32
// - 8 warps/block
// - Each warp computes a 32 x 64 C tile = 2 x 4 WMMA fragments
// - Compared with BK=16, synchronization frequency is reduced.
// - Compared with BK=64, register pressure is lower.
template <int BM, int BN, int BK, int SKEW, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void wmma_gemm_kernel(int dim_m, int dim_n, int dim_k,
                      const half *__restrict__ A,
                      const half *__restrict__ B,
                      float *__restrict__ C) {
  constexpr int THREADS = WARPS_PER_BLOCK * 32;

  static_assert(BM == 128 && BN == 128 && BK == 32,
                "This kernel is tuned for 128x128x32 tiles.");
  static_assert(WARPS_PER_BLOCK == 8,
                "This kernel expects 8 warps/block.");

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  __shared__ half As[BK][BM + SKEW];
  __shared__ half Bs[BK][BN + SKEW];

  __shared__ float Cscratch[WARPS_PER_BLOCK][16 * 16];

  // 8 warps/block:
  // warp_row = 0..3, warp_col = 0..1
  // each warp computes 32 x 64.
  const int warp_row = warp_id & 3;
  const int warp_col = warp_id >> 2;

  const int warp_m0 = warp_row * 32;
  const int warp_n0 = warp_col * 64;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];

#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int k0 = 0; k0 < dim_k; k0 += BK) {
    // Load A tile: BM x BK into shared memory.
    for (int idx = tid; idx < BM * BK; idx += THREADS) {
      const int kk = idx / BM;
      const int mm = idx - kk * BM;

      const int gm = block_m + mm;
      const int gk = k0 + kk;

      As[kk][mm] = (gm < dim_m && gk < dim_k)
                       ? A[static_cast<std::size_t>(gk) * dim_m + gm]
                       : __float2half(0.0f);
    }

    // Load B tile: BK x BN into shared memory.
    // B is column-major: B[gn * dim_k + gk].
    // Therefore, consecutive gk values are contiguous in memory.
    // This mapping makes consecutive threads read consecutive B elements.
    for (int idx = tid; idx < BN * BK; idx += THREADS) {
      const int nn = idx / BK;
      const int kk = idx - nn * BK;

      const int gn = block_n + nn;
      const int gk = k0 + kk;

      Bs[kk][nn] = (gn < dim_n && gk < dim_k)
                      ? B[static_cast<std::size_t>(gn) * dim_k + gk]
                      : __float2half(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;

        const int a_m = warp_m0 + r * 16;

        wmma::load_matrix_sync(a_frag,
                               &As[kk][a_m],
                               BM + SKEW);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

          const int b_n = warp_n0 + c * 16;

          wmma::load_matrix_sync(b_frag,
                                 &Bs[kk][b_n],
                                 BN + SKEW);

          wmma::mma_sync(acc[r][c],
                         a_frag,
                         b_frag,
                         acc[r][c]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int c_m = block_m + warp_m0 + r * 16;
      const int c_n = block_n + warp_n0 + c * 16;

      if (c_m + 15 < dim_m && c_n + 15 < dim_n) {
        wmma::store_matrix_sync(&C[static_cast<std::size_t>(c_n) * dim_m + c_m],
                                acc[r][c],
                                dim_m,
                                wmma::mem_col_major);
      } else if (c_m < dim_m && c_n < dim_n) {
        wmma::store_matrix_sync(Cscratch[warp_id],
                                acc[r][c],
                                16,
                                wmma::mem_col_major);

        __syncwarp();

        for (int idx = lane_id; idx < 16 * 16; idx += 32) {
          const int mm = idx & 15;
          const int nn = idx >> 4;

          if (c_m + mm < dim_m && c_n + nn < dim_n) {
            C[static_cast<std::size_t>(c_n + nn) * dim_m + (c_m + mm)] =
                Cscratch[warp_id][nn * 16 + mm];
          }
        }

        __syncwarp();
      }
    }
  }
}

static float time_cublas_gemm(cublasHandle_t handle, int Nt,
                              int m, int n, int k,
                              const half *Ah, const half *Bh, float *C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < 2; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              Ah, CUDA_R_16F, m,
                              Bh, CUDA_R_16F, k,
                              &beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < Nt; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              Ah, CUDA_R_16F, m,
                              Bh, CUDA_R_16F, k,
                              &beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return (ms * 1.0e-3f) / Nt;
}

static float time_my_gemm(int Nt, int m, int n, int k,
                          const half *Ah, const half *Bh, float *C2) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int SKEW = 8;
  constexpr int WARPS = 8;
  constexpr int THREADS = WARPS * 32;

  dim3 block(THREADS);
  dim3 grid((m + BM - 1) / BM,
            (n + BN - 1) / BN);

  for (int i = 0; i < 2; ++i) {
    wmma_gemm_kernel<BM, BN, BK, SKEW, WARPS>
        <<<grid, block>>>(m, n, k, Ah, Bh, C2);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < Nt; ++i) {
    wmma_gemm_kernel<BM, BN, BK, SKEW, WARPS>
        <<<grid, block>>>(m, n, k, Ah, Bh, C2);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return (ms * 1.0e-3f) / Nt;
}

int main(int argc, char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  int Nt = 10;

  if (argc >= 4) {
    m = std::atoi(argv[1]);
    k = std::atoi(argv[2]);
    n = std::atoi(argv[3]);
  }

  if (argc >= 5) {
    Nt = std::atoi(argv[4]);
  }

  const std::size_t sizeA = static_cast<std::size_t>(m) * k;
  const std::size_t sizeB = static_cast<std::size_t>(k) * n;
  const std::size_t sizeC = static_cast<std::size_t>(m) * n;

  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *C2 = nullptr;

  half *Ah = nullptr;
  half *Bh = nullptr;

  CUDA_CHECK(cudaMallocManaged(&A, sizeA * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, sizeB * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C, sizeC * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C2, sizeC * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&Ah, sizeA * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&Bh, sizeB * sizeof(half)));

  srand48(0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      A[static_cast<std::size_t>(k) * i + j] =
          static_cast<float>(drand48());
    }
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      B[static_cast<std::size_t>(n) * i + j] =
          static_cast<float>(drand48());
    }
  }

  constexpr int CONV_THREADS = 256;

  const int convBlocksA =
      static_cast<int>((sizeA + CONV_THREADS - 1) / CONV_THREADS);

  const int convBlocksB =
      static_cast<int>((sizeB + CONV_THREADS - 1) / CONV_THREADS);

  float_to_half_kernel<<<convBlocksA, CONV_THREADS>>>(A, Ah, sizeA);
  float_to_half_kernel<<<convBlocksB, CONV_THREADS>>>(B, Bh, sizeB);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

  const float tcublas =
      time_cublas_gemm(cublas_handle, Nt, m, n, k, Ah, Bh, C);

  const float tmy =
      time_my_gemm(Nt, m, n, k, Ah, Bh, C2);

  const double num_flops =
      2.0 * static_cast<double>(m) *
      static_cast<double>(n) *
      static_cast<double>(k);

  const double cublas_gflops =
      num_flops / tcublas / 1.0e9;

  const double my_gflops =
      num_flops / tmy / 1.0e9;

  CUDA_CHECK(cudaDeviceSynchronize());

  double err = 0.0;

  for (std::size_t i = 0; i < sizeC; ++i) {
    const double diff =
        static_cast<double>(C[i]) - static_cast<double>(C2[i]);
    err += std::fabs(diff);
  }

  const double mean_abs_err =
      err / static_cast<double>(sizeC);

  std::printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n",
              cublas_gflops, my_gflops);

  std::printf("error: %.6f\n", mean_abs_err);

  CUBLAS_CHECK(cublasDestroy(cublas_handle));

  CUDA_CHECK(cudaFree(Ah));
  CUDA_CHECK(cudaFree(Bh));

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(C2));

  return 0;
}

