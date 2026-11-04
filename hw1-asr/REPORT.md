# hw1-asr hand-in results

- Date: 2026-11-04T14:16:15+00:00
- Job: 3699333 on saxa.inf.ed.ac.uk
- Commit: 4430fb3705ecc11265400f331764328a91d879be (branch handin-results)
- Mode: full

Raw outputs for reference while writing the report. Before/after pairs: each
experiment below is compared with the Baseline sections, from the same node.

## GPU

`$ nvidia-smi -L`

```text
GPU 0: NVIDIA H200 (UUID: GPU-d222a9bf-dd87-250e-963f-f9ea6a0647dd)
  MIG 1g.18gb     Device  0: (UUID: MIG-629b3814-12a3-524c-8b70-11a8073906fc)
```

exit status: 0

## Uncommitted changes in csrc/ (empty = results match the commit)

`$ git status --short csrc/`

```text
```

exit status: 0

## Device facts

`$ build/experiments/devinfo`

```text
name                         NVIDIA H200 MIG 1g.18gb
compute capability           9.0
SMs                          16
global memory                16.0 GiB
memory clock                 3201 MHz
memory bus width             704 bit
peak memory bandwidth        563 GB/s (2 x clock x bus width)
L2 cache                     6.9 MiB
shared memory per block      49152 B (default), 232448 B (opt-in)
shared memory per SM         233472 B
registers per block          65536
max threads per block        1024
```

exit status: 0

On a MIG slice the bandwidth above is the whole H200's; the slice gets a share of it.

## nvidia-smi query

`$ nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.memory --format=csv`

```text
name, compute_cap, memory.total [MiB], clocks.max.memory [MHz]
NVIDIA H200, 9.0, 143771 MiB, 3201 MHz
```

exit status: 0

## Baseline: make test

`$ make test`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.002 ms      7.6 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.063 ms    243.1 GB/s (43% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.007 ms    186.4 GB/s (33% of peak)

Task 1 (RoPE): ALL PASS
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.68e-07     0.003 ms     12.9 GB/s ( 2% of peak)
rmsnorm, text decoder                PASS  max_err=4.35e-07     0.005 ms    214.7 GB/s (38% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.004 ms     11.1 GB/s ( 2% of peak)
layernorm, audio encoder             PASS  max_err=5.42e-07     0.060 ms    257.6 GB/s (46% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.04e-09     0.003 ms      7.6 GB/s ( 1% of peak)
softmax, large inputs (+-100)        PASS  max_err=2.67e-08     0.004 ms      4.4 GB/s ( 1% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.002 ms      1.9 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=8.13e-11     0.099 ms      4.8 GB/s ( 1% of peak)
softmax, 64 x 4096                   PASS  max_err=7.95e-10     0.009 ms    232.9 GB/s (41% of peak)

Task 2 (reductions): ALL PASS
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny 3 x 5, K = 7                    PASS  max_err=9.76e-09     0.002 ms        0 GFLOP/s
    for comparison: naive      0 GFLOP/s, cuBLAS      0 GFLOP/s (CPU reference)
odd sizes 37 x 100, K = 33, bias     PASS  max_err=1.55e-07     0.003 ms       79 GFLOP/s
    for comparison: naive     58 GFLOP/s, cuBLAS     42 GFLOP/s (CPU reference)
odd sizes, bias + GELU               PASS  max_err=7.70e-07     0.005 ms      189 GFLOP/s
    for comparison: naive    131 GFLOP/s, cuBLAS    155 GFLOP/s (CPU reference)
decoder q_proj (64x2048x2048)        PASS  max_err=1.14e-05     0.611 ms      879 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4232 GFLOP/s (cuBLAS reference)
decoder gate_proj (64x6144x2048)     PASS  max_err=0.00e+00     1.766 ms      912 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4136 GFLOP/s (cuBLAS reference)
decoder down_proj (64x2048x6144)     PASS  max_err=3.72e-05     1.828 ms      881 GFLOP/s
    for comparison: naive    109 GFLOP/s, cuBLAS   5167 GFLOP/s (cuBLAS reference)
lm_head, last token (1x59264x2048)   PASS  max_err=1.24e-05     4.059 ms       60 GFLOP/s
    for comparison: naive     58 GFLOP/s, cuBLAS    259 GFLOP/s (cuBLAS reference)
encoder q_proj (1500x1280x1280)      PASS  max_err=1.01e-05     5.250 ms      936 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   5008 GFLOP/s (cuBLAS reference)
encoder fc1 + GELU (1500x5120x1280)  PASS  max_err=6.92e-07    21.106 ms      932 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   6184 GFLOP/s (cuBLAS reference)
conv2 via im2col (1500x1280x3840)    PASS  max_err=2.57e-05    16.020 ms      920 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   5862 GFLOP/s (cuBLAS reference)

Task 3 (tiled GEMM): ALL PASS
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny, one tile (4 queries, 5 keys)   PASS  max_err=8.15e-08     0.006 ms        0 GFLOP/s
several tiles, peaky scores          PASS  max_err=4.97e-07     0.021 ms       11 GFLOP/s
GQA 4 heads / 2 kv heads             PASS  max_err=1.30e-07     0.010 ms       58 GFLOP/s
causal, seq_q = seq_k                PASS  max_err=5.16e-07     0.039 ms       64 GFLOP/s
causal, 5 new queries, 37 keys       PASS  max_err=1.78e-07     0.011 ms       17 GFLOP/s
causal, one new query                PASS  max_err=3.39e-07     0.038 ms       15 GFLOP/s
text decoder (GQA 16/4, causal)      PASS  max_err=5.21e-07     0.084 ms      201 GFLOP/s
audio encoder (20 heads, 1500)       PASS  max_err=2.68e-07    27.907 ms      413 GFLOP/s

Task 4 (attention): ALL PASS
```

exit status: 0

## Baseline: ./run.sh --profile

`$ ./run.sh --profile`

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
GPU: NVIDIA H200 MIG 1g.18gb (sm_90)
Kernels: rope=cuda  reduce=cuda  gemm=cuda  attention=cuda  provided=cuda
Library: /home/s2180472/edin-mls-26-autumn/hw1-asr/build/sm_90/libasr_kernels.so
Loading the model (about 9 GB of float32 weights)...
Loaded in 42.0 s
Audio: 3.50 s, mel features (1, 128, 3000), prompt 59 tokens
  run 1:   5909.5 ms  (13 tokens)
  run 2:   5909.6 ms  (13 tokens)
  run 3:   5909.6 ms  (13 tokens)

GPU time per operation, one transcription (5919.7 ms in kernels):
  operation    runs on    calls   total ms   share
  linear       cuda        2757     4963.7   83.9%
  attention    cuda         396      929.0   15.7%
  rmsnorm      cuda         741        7.6    0.1%
  silu_mul     cuda         364        6.7    0.1%
  rope         cuda         396        6.2    0.1%
  layernorm    cuda          65        4.1    0.1%
  softmax      cuda          13        1.8    0.0%
  embedding    cuda          13        0.4    0.0%
  im2col       cuda           2        0.3    0.0%

Time: 5909.6 ms per transcription (+/- 0.1 ms), 454.6 ms per token
Peak GPU memory: 8.56 GiB
Transcription: Concord returned to its place amidst the tents.
Expected:      Concord returned to its place amidst the tents
Accuracy: 100.0%
Status: PASS
```

exit status: 0

# Try this experiments

## Task 1, Try this 1: ROPE_THREADS sweep (watch the audio encoder GB/s)

Baseline value ROPE_THREADS = 256: see the Baseline sections.

### ROPE_THREADS = 32

```diff
--- csrc/task1_rope.cu	2026-10-23 11:26:33.000000000 +0100
+++ build/experiments/t1_rope_32/csrc/task1_rope.cu	2026-11-04 14:17:54.000000000 +0000
@@ -21,7 +21,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int ROPE_THREADS = 256;
+constexpr int ROPE_THREADS = 32;
 
 __global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                             int num_q_heads, int num_k_heads, int q_rows, int k_rows,
```

## t1_rope_32: make test1

`$ make -s test1 SRC=build/experiments/t1_rope_32/csrc BUILD=build/experiments/t1_rope_32/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.002 ms      7.6 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.069 ms    223.5 GB/s (40% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.008 ms    165.7 GB/s (29% of peak)

Task 1 (RoPE): ALL PASS
```

exit status: 0

### ROPE_THREADS = 64

```diff
--- csrc/task1_rope.cu	2026-10-23 11:26:33.000000000 +0100
+++ build/experiments/t1_rope_64/csrc/task1_rope.cu	2026-11-04 14:18:00.000000000 +0000
@@ -21,7 +21,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int ROPE_THREADS = 256;
+constexpr int ROPE_THREADS = 64;
 
 __global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                             int num_q_heads, int num_k_heads, int q_rows, int k_rows,
```

## t1_rope_64: make test1

`$ make -s test1 SRC=build/experiments/t1_rope_64/csrc BUILD=build/experiments/t1_rope_64/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.002 ms      7.6 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.063 ms    242.7 GB/s (43% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.007 ms    185.6 GB/s (33% of peak)

Task 1 (RoPE): ALL PASS
```

exit status: 0

### ROPE_THREADS = 128

```diff
--- csrc/task1_rope.cu	2026-10-23 11:26:33.000000000 +0100
+++ build/experiments/t1_rope_128/csrc/task1_rope.cu	2026-11-04 14:18:05.000000000 +0000
@@ -21,7 +21,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int ROPE_THREADS = 256;
+constexpr int ROPE_THREADS = 128;
 
 __global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                             int num_q_heads, int num_k_heads, int q_rows, int k_rows,
```

## t1_rope_128: make test1

`$ make -s test1 SRC=build/experiments/t1_rope_128/csrc BUILD=build/experiments/t1_rope_128/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.002 ms      7.7 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.063 ms    243.2 GB/s (43% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.007 ms    186.4 GB/s (33% of peak)

Task 1 (RoPE): ALL PASS
```

exit status: 0

### ROPE_THREADS = 512

```diff
--- csrc/task1_rope.cu	2026-10-23 11:26:33.000000000 +0100
+++ build/experiments/t1_rope_512/csrc/task1_rope.cu	2026-11-04 14:18:10.000000000 +0000
@@ -21,7 +21,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int ROPE_THREADS = 256;
+constexpr int ROPE_THREADS = 512;
 
 __global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                             int num_q_heads, int num_k_heads, int q_rows, int k_rows,
```

## t1_rope_512: make test1

`$ make -s test1 SRC=build/experiments/t1_rope_512/csrc BUILD=build/experiments/t1_rope_512/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.002 ms      7.1 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.064 ms    240.3 GB/s (43% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.007 ms    184.7 GB/s (33% of peak)

Task 1 (RoPE): ALL PASS
```

exit status: 0

### ROPE_THREADS = 1024

```diff
--- csrc/task1_rope.cu	2026-10-23 11:26:33.000000000 +0100
+++ build/experiments/t1_rope_1024/csrc/task1_rope.cu	2026-11-04 14:18:15.000000000 +0000
@@ -21,7 +21,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int ROPE_THREADS = 256;
+constexpr int ROPE_THREADS = 1024;
 
 __global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                             int num_q_heads, int num_k_heads, int q_rows, int k_rows,
```

## t1_rope_1024: make test1

`$ make -s test1 SRC=build/experiments/t1_rope_1024/csrc BUILD=build/experiments/t1_rope_1024/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny (2 x 3 tokens, Hq=3, Hk=1)      PASS  max_err=1.08e-07     0.002 ms      0.7 GB/s ( 0% of peak)
odd sizes, partial rotation          PASS  max_err=1.08e-07     0.003 ms      6.0 GB/s ( 1% of peak)
audio encoder (partial, 32/64)       PASS  max_err=1.93e-07     0.065 ms    236.6 GB/s (42% of peak)
text decoder (GQA 16/4, full)        PASS  max_err=1.64e-07     0.007 ms    186.9 GB/s (33% of peak)

Task 1 (RoPE): ALL PASS
```

exit status: 0

## Task 2, Try this 1: REDUCE_THREADS sweep (watch layernorm audio encoder and softmax vocabulary row)

Baseline value REDUCE_THREADS = 256: see the Baseline sections.

### REDUCE_THREADS = 32

```diff
--- csrc/task2_reduce.cu	2026-10-23 14:38:16.000000000 +0100
+++ build/experiments/t2_reduce_32/csrc/task2_reduce.cu	2026-11-04 14:18:21.000000000 +0000
@@ -11,7 +11,7 @@
 #include "asr_kernels.h"
 #include "task2_reduce.cuh"
 
-constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024
+constexpr int REDUCE_THREADS = 32;  // threads per block: a multiple of 32, at most 1024
 
 // Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 __global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
```

## t2_reduce_32: make test2

`$ make -s test2 SRC=build/experiments/t2_reduce_32/csrc BUILD=build/experiments/t2_reduce_32/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.68e-07     0.012 ms      3.3 GB/s ( 1% of peak)
rmsnorm, text decoder                PASS  max_err=3.83e-07     0.023 ms     45.2 GB/s ( 8% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.014 ms      2.9 GB/s ( 1% of peak)
layernorm, audio encoder             PASS  max_err=5.52e-07     0.094 ms    162.6 GB/s (29% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.04e-09     0.012 ms      2.1 GB/s ( 0% of peak)
softmax, large inputs (+-100)        PASS  max_err=2.67e-08     0.013 ms      1.3 GB/s ( 0% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.005 ms      1.0 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=5.23e-10     0.788 ms      0.6 GB/s ( 0% of peak)
softmax, 64 x 4096                   PASS  max_err=1.13e-09     0.041 ms     51.0 GB/s ( 9% of peak)

Task 2 (reductions): ALL PASS
```

exit status: 0

### REDUCE_THREADS = 64

```diff
--- csrc/task2_reduce.cu	2026-10-23 14:38:16.000000000 +0100
+++ build/experiments/t2_reduce_64/csrc/task2_reduce.cu	2026-11-04 14:18:26.000000000 +0000
@@ -11,7 +11,7 @@
 #include "asr_kernels.h"
 #include "task2_reduce.cuh"
 
-constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024
+constexpr int REDUCE_THREADS = 64;  // threads per block: a multiple of 32, at most 1024
 
 // Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 __global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
```

## t2_reduce_64: make test2

`$ make -s test2 SRC=build/experiments/t2_reduce_64/csrc BUILD=build/experiments/t2_reduce_64/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.68e-07     0.007 ms      5.7 GB/s ( 1% of peak)
rmsnorm, text decoder                PASS  max_err=4.18e-07     0.013 ms     81.3 GB/s (14% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.008 ms      5.0 GB/s ( 1% of peak)
layernorm, audio encoder             PASS  max_err=5.42e-07     0.057 ms    268.4 GB/s (48% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.55e-09     0.007 ms      3.5 GB/s ( 1% of peak)
softmax, large inputs (+-100)        PASS  max_err=2.67e-08     0.008 ms      2.2 GB/s ( 0% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.004 ms      1.3 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=2.80e-10     0.388 ms      1.2 GB/s ( 0% of peak)
softmax, 64 x 4096                   PASS  max_err=1.12e-09     0.022 ms     96.1 GB/s (17% of peak)

Task 2 (reductions): ALL PASS
```

exit status: 0

### REDUCE_THREADS = 128

```diff
--- csrc/task2_reduce.cu	2026-10-23 14:38:16.000000000 +0100
+++ build/experiments/t2_reduce_128/csrc/task2_reduce.cu	2026-11-04 14:18:31.000000000 +0000
@@ -11,7 +11,7 @@
 #include "asr_kernels.h"
 #include "task2_reduce.cuh"
 
-constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024
+constexpr int REDUCE_THREADS = 128;  // threads per block: a multiple of 32, at most 1024
 
 // Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 __global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
```

## t2_reduce_128: make test2

`$ make -s test2 SRC=build/experiments/t2_reduce_128/csrc BUILD=build/experiments/t2_reduce_128/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.51e-07     0.004 ms      9.0 GB/s ( 2% of peak)
rmsnorm, text decoder                PASS  max_err=4.18e-07     0.008 ms    137.9 GB/s (24% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.005 ms      7.7 GB/s ( 1% of peak)
layernorm, audio encoder             PASS  max_err=5.17e-07     0.056 ms    273.4 GB/s (49% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.04e-09     0.004 ms      5.4 GB/s ( 1% of peak)
softmax, large inputs (+-100)        PASS  max_err=1.89e-08     0.005 ms      3.3 GB/s ( 1% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.003 ms      1.6 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=1.35e-10     0.196 ms      2.4 GB/s ( 0% of peak)
softmax, 64 x 4096                   PASS  max_err=8.85e-10     0.013 ms    166.2 GB/s (29% of peak)

Task 2 (reductions): ALL PASS
```

exit status: 0

### REDUCE_THREADS = 512

```diff
--- csrc/task2_reduce.cu	2026-10-23 14:38:16.000000000 +0100
+++ build/experiments/t2_reduce_512/csrc/task2_reduce.cu	2026-11-04 14:18:36.000000000 +0000
@@ -11,7 +11,7 @@
 #include "asr_kernels.h"
 #include "task2_reduce.cuh"
 
-constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024
+constexpr int REDUCE_THREADS = 512;  // threads per block: a multiple of 32, at most 1024
 
 // Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 __global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
```

## t2_reduce_512: make test2

`$ make -s test2 SRC=build/experiments/t2_reduce_512/csrc BUILD=build/experiments/t2_reduce_512/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.68e-07     0.003 ms     14.2 GB/s ( 3% of peak)
rmsnorm, text decoder                PASS  max_err=4.18e-07     0.004 ms    277.1 GB/s (49% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.003 ms     13.3 GB/s ( 2% of peak)
layernorm, audio encoder             PASS  max_err=5.17e-07     0.075 ms    206.0 GB/s (37% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.14e-09     0.003 ms      8.6 GB/s ( 2% of peak)
softmax, large inputs (+-100)        PASS  max_err=1.95e-08     0.003 ms      4.8 GB/s ( 1% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.002 ms      2.2 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=8.13e-11     0.051 ms      9.3 GB/s ( 2% of peak)
softmax, 64 x 4096                   PASS  max_err=7.95e-10     0.009 ms    245.0 GB/s (43% of peak)

Task 2 (reductions): ALL PASS
```

exit status: 0

### REDUCE_THREADS = 1024

```diff
--- csrc/task2_reduce.cu	2026-10-23 14:38:16.000000000 +0100
+++ build/experiments/t2_reduce_1024/csrc/task2_reduce.cu	2026-11-04 14:18:42.000000000 +0000
@@ -11,7 +11,7 @@
 #include "asr_kernels.h"
 #include "task2_reduce.cuh"
 
-constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024
+constexpr int REDUCE_THREADS = 1024;  // threads per block: a multiple of 32, at most 1024
 
 // Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 __global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
```

## t2_reduce_1024: make test2

`$ make -s test2 SRC=build/experiments/t2_reduce_1024/csrc BUILD=build/experiments/t2_reduce_1024/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

rmsnorm, one short row               PASS  max_err=8.27e-08     0.002 ms      0.1 GB/s ( 0% of peak)
rmsnorm, odd cols                    PASS  max_err=3.68e-07     0.002 ms     16.0 GB/s ( 3% of peak)
rmsnorm, text decoder                PASS  max_err=3.95e-07     0.005 ms    217.9 GB/s (39% of peak)
layernorm, odd cols (example)        PASS  max_err=3.91e-07     0.003 ms     14.3 GB/s ( 3% of peak)
layernorm, audio encoder             PASS  max_err=5.62e-07     0.114 ms    134.7 GB/s (24% of peak)
softmax, one short row               PASS  max_err=1.11e-08     0.002 ms      0.0 GB/s ( 0% of peak)
softmax, odd cols                    PASS  max_err=1.04e-09     0.003 ms      8.8 GB/s ( 2% of peak)
softmax, large inputs (+-100)        PASS  max_err=1.95e-08     0.003 ms      5.4 GB/s ( 1% of peak)
softmax, all negative (-200..-150)   PASS  max_err=6.11e-09     0.003 ms      1.9 GB/s ( 0% of peak)
softmax, vocabulary row (59264)      PASS  max_err=4.02e-11     0.030 ms     15.8 GB/s ( 3% of peak)
softmax, 64 x 4096                   PASS  max_err=7.95e-10     0.010 ms    220.1 GB/s (39% of peak)

Task 2 (reductions): ALL PASS
```

exit status: 0

## Task 3, Try this 1: TILE sweep (encoder cases, and lm_head at TILE = 32)

Baseline value TILE = 16: see the Baseline sections.

### TILE = 8

```diff
--- csrc/task3_gemm.cu	2026-10-23 15:19:07.000000000 +0100
+++ build/experiments/t3_tile_8/csrc/task3_gemm.cu	2026-11-04 14:18:47.000000000 +0000
@@ -16,7 +16,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int TILE = 16;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)
+constexpr int TILE = 8;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)
 
 __global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                     float* y, int M, int N, int K, int act) {
```

## t3_tile_8: make test3

`$ make -s test3 SRC=build/experiments/t3_tile_8/csrc BUILD=build/experiments/t3_tile_8/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny 3 x 5, K = 7                    PASS  max_err=9.76e-09     0.002 ms        0 GFLOP/s
    for comparison: naive      0 GFLOP/s, cuBLAS      0 GFLOP/s (CPU reference)
odd sizes 37 x 100, K = 33, bias     PASS  max_err=1.55e-07     0.003 ms       79 GFLOP/s
    for comparison: naive     65 GFLOP/s, cuBLAS     42 GFLOP/s (CPU reference)
odd sizes, bias + GELU               PASS  max_err=7.70e-07     0.006 ms      153 GFLOP/s
    for comparison: naive    132 GFLOP/s, cuBLAS    151 GFLOP/s (CPU reference)
decoder q_proj (64x2048x2048)        PASS  max_err=1.14e-05     0.915 ms      587 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4243 GFLOP/s (cuBLAS reference)
decoder gate_proj (64x6144x2048)     PASS  max_err=0.00e+00     2.688 ms      599 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4137 GFLOP/s (cuBLAS reference)
decoder down_proj (64x2048x6144)     PASS  max_err=3.72e-05     2.730 ms      590 GFLOP/s
    for comparison: naive    109 GFLOP/s, cuBLAS   5179 GFLOP/s (cuBLAS reference)
lm_head, last token (1x59264x2048)   PASS  max_err=1.24e-05     2.911 ms       83 GFLOP/s
    for comparison: naive     58 GFLOP/s, cuBLAS    261 GFLOP/s (cuBLAS reference)
encoder q_proj (1500x1280x1280)      PASS  max_err=1.01e-05     8.114 ms      606 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   5007 GFLOP/s (cuBLAS reference)
encoder fc1 + GELU (1500x5120x1280)  PASS  max_err=6.92e-07    32.463 ms      606 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   6181 GFLOP/s (cuBLAS reference)
conv2 via im2col (1500x1280x3840)    PASS  max_err=2.57e-05    24.420 ms      604 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   5872 GFLOP/s (cuBLAS reference)

Task 3 (tiled GEMM): ALL PASS
```

exit status: 0

### TILE = 32

```diff
--- csrc/task3_gemm.cu	2026-10-23 15:19:07.000000000 +0100
+++ build/experiments/t3_tile_32/csrc/task3_gemm.cu	2026-11-04 14:18:59.000000000 +0000
@@ -16,7 +16,7 @@
 #include "asr_kernels.h"
 #include "common.cuh"
 
-constexpr int TILE = 16;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)
+constexpr int TILE = 32;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)
 
 __global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                     float* y, int M, int N, int K, int act) {
```

## t3_tile_32: make test3

`$ make -s test3 SRC=build/experiments/t3_tile_32/csrc BUILD=build/experiments/t3_tile_32/build SM=90`

```text
GPU: NVIDIA H200 MIG 1g.18gb (sm_90), 16 SMs, peak memory bandwidth about 563 GB/s

tiny 3 x 5, K = 7                    PASS  max_err=9.76e-09     0.003 ms        0 GFLOP/s
    for comparison: naive      0 GFLOP/s, cuBLAS      0 GFLOP/s (CPU reference)
odd sizes 37 x 100, K = 33, bias     PASS  max_err=1.55e-07     0.004 ms       57 GFLOP/s
    for comparison: naive     61 GFLOP/s, cuBLAS     42 GFLOP/s (CPU reference)
odd sizes, bias + GELU               PASS  max_err=7.70e-07     0.008 ms      119 GFLOP/s
    for comparison: naive    124 GFLOP/s, cuBLAS    154 GFLOP/s (CPU reference)
decoder q_proj (64x2048x2048)        PASS  max_err=1.14e-05     0.517 ms     1039 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4243 GFLOP/s (cuBLAS reference)
decoder gate_proj (64x6144x2048)     PASS  max_err=0.00e+00     1.509 ms     1068 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   4135 GFLOP/s (cuBLAS reference)
decoder down_proj (64x2048x6144)     PASS  max_err=3.72e-05     1.541 ms     1045 GFLOP/s
    for comparison: naive    109 GFLOP/s, cuBLAS   5186 GFLOP/s (cuBLAS reference)
lm_head, last token (1x59264x2048)   PASS  max_err=1.24e-05     7.121 ms       34 GFLOP/s
    for comparison: naive     58 GFLOP/s, cuBLAS    260 GFLOP/s (cuBLAS reference)
encoder q_proj (1500x1280x1280)      PASS  max_err=1.01e-05     4.445 ms     1106 GFLOP/s
    for comparison: naive    111 GFLOP/s, cuBLAS   5024 GFLOP/s (cuBLAS reference)
encoder fc1 + GELU (1500x5120x1280)  PASS  max_err=6.92e-07    18.353 ms     1071 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   6182 GFLOP/s (cuBLAS reference)
conv2 via im2col (1500x1280x3840)    PASS  max_err=2.57e-05    13.737 ms     1073 GFLOP/s
    for comparison: naive    110 GFLOP/s, cuBLAS   5867 GFLOP/s (cuBLAS reference)

Task 3 (tiled GEMM): ALL PASS
```

exit status: 0

## Task 4, Try this 3: attention on PyTorch, everything else on your kernels

No code change. Compare the attention line with the Baseline profile.

## ./run.sh --cuda task1,task2,task3 --profile

`$ ./run.sh --cuda task1,task2,task3 --profile`

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
GPU: NVIDIA H200 MIG 1g.18gb (sm_90)
Kernels: rope=cuda  reduce=cuda  gemm=cuda  attention=torch  provided=cuda
Library: /home/s2180472/edin-mls-26-autumn/hw1-asr/build/sm_90/libasr_kernels.so
Loading the model (about 9 GB of float32 weights)...
Loaded in 28.7 s
Audio: 3.50 s, mel features (1, 128, 3000), prompt 59 tokens
  run 1:   5143.1 ms  (13 tokens)
  run 2:   5143.4 ms  (13 tokens)
  run 3:   5143.0 ms  (13 tokens)

GPU time per operation, one transcription (5152.0 ms in kernels):
  operation    runs on    calls   total ms   share
  linear       cuda        2757     4963.6   96.3%
  attention    torch        396      161.3    3.1%
  rmsnorm      cuda         741        7.7    0.1%
  silu_mul     cuda         364        6.7    0.1%
  rope         cuda         396        6.1    0.1%
  layernorm    cuda          65        4.1    0.1%
  softmax      cuda          13        1.8    0.0%
  embedding    cuda          13        0.4    0.0%
  im2col       cuda           2        0.3    0.0%

Time: 5143.2 ms per transcription (+/- 0.2 ms), 395.6 ms per token
Peak GPU memory: 8.90 GiB
Transcription: Concord returned to its place amidst the tents.
Expected:      Concord returned to its place amidst the tents
Accuracy: 100.0%
Status: PASS
```

exit status: 0

