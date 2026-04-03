/*
 * crystal_kernel.cu — {0,1,3} ternary GEMM kernel
 *
 * Crystal arithmetic for the three weight values:
 *   0 (void)   → skip entirely (zero multiplications)
 *   1 (unit)   → Y += X       (no multiply, just add)
 *   3 (prime)  → Y += X + X + X  (shift-add: (X<<1)+X)
 *
 * 2-bit encoding in packed uint32:
 *   00 → 0   void
 *   01 → 1   unit
 *   11 → 3   prime
 *   10 → 1   defensive (should not occur after packing)
 *
 * Layout:
 *   X      : (M, K)     fp16 activations
 *   W      : (N, K/16)  packed uint32 weights (row-major, 16 weights per word)
 *   Y      : (M, N)     fp16 output
 *   bias   : (N,)       fp16 bias or nullptr
 *
 * Simple implementation — correctness first.
 * Each thread computes ONE output element Y[m, n].
 * Thread (tx, ty) in block (bx, by) handles:
 *   m = by * blockDim.y + ty
 *   n = bx * blockDim.x + tx
 *
 * No tiling/shared memory in v1. Straightforward for verification.
 */

#include <cuda_fp16.h>
#include <stdint.h>

// ── kernel ────────────────────────────────────────────────────────────────

extern "C" __global__ void crystal_gemm(
    const __half*        X,      // (M, K)
    const unsigned int*  W,      // (N, K/16) packed 2-bit
    __half*              Y,      // (M, N)
    const __half*        bias,   // (N,) or nullptr
    int M, int N, int K
)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;

    // Number of uint32 words per row of W
    int words_per_row = K / 16;

    // Pointer to this output neuron's weight row
    const unsigned int* w_row = W + (long)n * words_per_row;

    // Pointer to this input sample's activation row
    const __half* x_row = X + (long)m * K;

    for (int word_idx = 0; word_idx < words_per_row; word_idx++) {
        unsigned int word = w_row[word_idx];

        // 16 weights packed in this word (LSB = weight 0)
        int k_base = word_idx * 16;

        #pragma unroll 16
        for (int bit_pos = 0; bit_pos < 16; bit_pos++) {
            int k = k_base + bit_pos;
            unsigned int code = (word >> (2 * bit_pos)) & 0x3u;

            float x = __half2float(x_row[k]);

            switch (code) {
                case 0u:              // void: skip
                    break;
                case 1u:              // unit: add once
                    acc += x;
                    break;
                case 2u:              // defensive: treat as 1
                    acc += x;
                    break;
                case 3u:              // prime: triple (shift-add: 2x + x)
                    acc += x + x + x;
                    break;
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        acc += __half2float(bias[n]);
    }

    Y[(long)m * N + n] = __float2half(acc);
}


// ── launch wrapper (called from C++ extension) ────────────────────────────

extern "C" void launch_crystal_gemm(
    const __half*        X,
    const unsigned int*  W,
    __half*              Y,
    const __half*        bias,
    int M, int N, int K,
    cudaStream_t         stream
)
{
    // 2D grid: each thread handles one (m, n) output element
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    crystal_gemm<<<grid, block, 0, stream>>>(X, W, Y, bias, M, N, K);
}
