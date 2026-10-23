// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include "asr_kernels.h"
#include "common.cuh"

constexpr int TILE = 16;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + 1];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ======================== BLANK 3.1 =========================
    // Load this thread's element of the x tile into xs. Use 0 outside x.
    xs[ty][tx] = (row < M && k0 + tx < K) ? x[row * K + k0 + tx] : 0.0f;
    // ============================================================

    // ======================== BLANK 3.2 =========================
    // Load this thread's element of the w tile, so that ws[n][k] holds
    // w[blockIdx.x * TILE + n][k0 + k]. Use 0 outside w. Threads with consecutive
    // tx must read consecutive addresses of w.
    int wn = blockIdx.x * TILE + ty;
    ws[ty][tx] = (wn < N && k0 + tx < K) ? w[wn * K + k0 + tx] : 0.0f;
    __syncthreads();
    // ============================================================

    // ======================== BLANK 3.3 =========================
    // Add this K step's contribution to acc.
    for (int k = 0; k < TILE; k++) {
      acc += xs[ty][k] * ws[tx][k];
    }
    __syncthreads();
    // ============================================================
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: for outputs inside y, add bias[col] (bias may be nullptr), apply
  // gelu() when act == ACT_GELU, and store.
  if (row < M && col < N) {
    float b = bias != nullptr ? bias[col] : 0.0f;
    float val = acc + b;
    y[row * N + col] = act == ACT_GELU ? gelu(val) : val;
  }
  // ============================================================
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
  RETURN_LAUNCH_ERROR();
}
