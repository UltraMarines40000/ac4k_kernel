// Copyright 2024-2026 AC4K Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "ac4k_kernel/ops.h"
#include "utils.cuh"

#define CHECK_TYPE(x, st, m)                                                   \
  TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m)                                                    \
  TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m)                                                 \
  TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m)                                                  \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_CONTIGUOUS(x, m);                                                      \
  CHECK_TYPE(x, st, m)

namespace ac4k {

//===----------------------------------------------------------------------===//
// FP32 GEMM Kernel
//
// Computes C = alpha * A @ B + beta * C
// A: [M, K] row-major, B: [K, N] row-major, C: [M, N] row-major
// All tensors are FP32.
//
// Tiled approach with shared memory:
//   - Tile size: BM x BN for output, BK for reduction
//   - Each thread computes TM x TN elements of the output tile
//===----------------------------------------------------------------------===//

// Tile dimensions
constexpr int GEMM_FP32_BM = 128;
constexpr int GEMM_FP32_BN = 128;
constexpr int GEMM_FP32_BK = 8;
constexpr int GEMM_FP32_TM = 8;
constexpr int GEMM_FP32_TN = 8;

// Number of threads per block
constexpr int GEMM_FP32_THREADS =
    (GEMM_FP32_BM / GEMM_FP32_TM) * (GEMM_FP32_BN / GEMM_FP32_TN);
// = 16 * 16 = 256

// Shared memory size per stage
constexpr int GEMM_FP32_SMEM_A_SIZE = GEMM_FP32_BM * GEMM_FP32_BK * sizeof(float);
constexpr int GEMM_FP32_SMEM_B_SIZE = GEMM_FP32_BN * GEMM_FP32_BK * sizeof(float);
constexpr int GEMM_FP32_SMEM_SIZE = GEMM_FP32_SMEM_A_SIZE + GEMM_FP32_SMEM_B_SIZE;

__global__ void __launch_bounds__(GEMM_FP32_THREADS)
    gemm_fp32_kernel(const float *__restrict__ A, const float *__restrict__ B,
                     float *__restrict__ C, const float alpha, const float beta,
                     const int M, const int N, const int K) {
  // Thread indices
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;
  const int tid = tz * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;

  // Block indices
  const int bx = blockIdx.x; // tile column
  const int by = blockIdx.y; // tile row

  // Output tile origin
  const int row_base = by * GEMM_FP32_BM;
  const int col_base = bx * GEMM_FP32_BN;

  // Each thread computes TM x TN output elements
  const int thread_row = tid / (GEMM_FP32_BN / GEMM_FP32_TN);
  const int thread_col = tid % (GEMM_FP32_BN / GEMM_FP32_TN);

  // Shared memory
  extern __shared__ __align__(16) char smem_buf[];
  float *smem_A = reinterpret_cast<float *>(smem_buf);
  float *smem_B = reinterpret_cast<float *>(smem_buf + GEMM_FP32_SMEM_A_SIZE);

  // Accumulator registers
  float accum[GEMM_FP32_TM][GEMM_FP32_TN] = {0.0f};

  // Pointers to global memory for this thread's output
  const int global_row = row_base + thread_row * GEMM_FP32_TM;
  const int global_col = col_base + thread_col * GEMM_FP32_TN;

  // Number of threads for cooperative loading
  const int load_stride = GEMM_FP32_THREADS;

  // Loop over K dimension in tiles of BK
  for (int k_pos = 0; k_pos < K; k_pos += GEMM_FP32_BK) {
    // Load tile of A [BM, BK] into shared memory
    for (int idx = tid; idx < GEMM_FP32_BM * GEMM_FP32_BK; idx += load_stride) {
      int r = idx / GEMM_FP32_BK;
      int c = idx % GEMM_FP32_BK;
      int g_row = row_base + r;
      int g_col = k_pos + c;
      if (g_row < M && g_col < K) {
        smem_A[r * GEMM_FP32_BK + c] = A[g_row * K + g_col];
      } else {
        smem_A[r * GEMM_FP32_BK + c] = 0.0f;
      }
    }

    // Load tile of B [BK, BN] into shared memory
    for (int idx = tid; idx < GEMM_FP32_BN * GEMM_FP32_BK; idx += load_stride) {
      int r = idx / GEMM_FP32_BN;
      int c = idx % GEMM_FP32_BN;
      int g_row = k_pos + r;
      int g_col = col_base + c;
      if (g_row < K && g_col < N) {
        smem_B[r * GEMM_FP32_BN + c] = B[g_row * N + g_col];
      } else {
        smem_B[r * GEMM_FP32_BN + c] = 0.0f;
      }
    }

    __syncthreads();

    // Compute partial dot product
    for (int kk = 0; kk < GEMM_FP32_BK; ++kk) {
      for (int ii = 0; ii < GEMM_FP32_TM; ++ii) {
        float a_val = smem_A[(thread_row * GEMM_FP32_TM + ii) * GEMM_FP32_BK + kk];
        for (int jj = 0; jj < GEMM_FP32_TN; ++jj) {
          float b_val = smem_B[kk * GEMM_FP32_BN + thread_col * GEMM_FP32_TN + jj];
          accum[ii][jj] += a_val * b_val;
        }
      }
    }

    __syncthreads();
  }

  // Write results to global memory: C = alpha * A @ B + beta * C
  for (int ii = 0; ii < GEMM_FP32_TM; ++ii) {
    for (int jj = 0; jj < GEMM_FP32_TN; ++jj) {
      int g_row = global_row + ii;
      int g_col = global_col + jj;
      if (g_row < M && g_col < N) {
        float val = alpha * accum[ii][jj];
        if (beta != 0.0f) {
          val += beta * C[g_row * N + g_col];
        }
        C[g_row * N + g_col] = val;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Host launch function
//===----------------------------------------------------------------------===//

void gemm_fp32(torch::Tensor &C, const torch::Tensor &A,
               const torch::Tensor &B, float alpha, float beta) {
  // Check inputs
  TORCH_CHECK(A.dim() == 2, "A must be a 2D matrix");
  TORCH_CHECK(B.dim() == 2, "B must be a 2D matrix");
  TORCH_CHECK(C.dim() == 2, "C must be a 2D matrix");
  CHECK_INPUT(A, at::ScalarType::Float, "A");
  CHECK_INPUT(B, at::ScalarType::Float, "B");
  CHECK_INPUT(C, at::ScalarType::Float, "C");

  const int M = A.size(0);
  const int K = A.size(1);
  const int K2 = B.size(0);
  const int N = B.size(1);

  TORCH_CHECK(K == K2, "A and B inner dimensions must match: ", K, " vs ", K2);
  TORCH_CHECK(C.size(0) == M, "C size(0) must be ", M);
  TORCH_CHECK(C.size(1) == N, "C size(1) must be ", N);

  // Grid dimensions
  dim3 grid(ceil_div(N, GEMM_FP32_BN), ceil_div(M, GEMM_FP32_BM));
  dim3 block(GEMM_FP32_THREADS);

  // Get CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Launch kernel
  gemm_fp32_kernel<<<grid, block, GEMM_FP32_SMEM_SIZE, stream>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), alpha,
      beta, M, N, K);
}

} // namespace ac4k