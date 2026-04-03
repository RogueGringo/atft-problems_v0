# Crystal CUDA Kernel — Forward-Pass Benchmark

**Date:** 2026-04-02
**Status:** Approved for implementation
**Author:** Aaron Jones + Claude

---

## Purpose

Drop-in CUDA kernel replacing `torch.nn.functional.linear` for {0,1,3} packed weights. Zero floating-point multiplications. Three-way branch: skip (void), copy (identity), shift-add (prime). Head-to-head benchmark against PyTorch fp16 `torch.mm` on RTX 5070.

**Deliverable:** One speedup number at 4096×4096.

---

## Files

```
products/ternary-architect/cuda/
├── crystal_kernel.cu     — CUDA kernel (crystal_gemm)
├── crystal_ops.py        — Python bindings (JIT via load_inline)
├── pack_weights.py       — pack/unpack {0,1,3} ↔ 2-bit uint32
└── benchmark.py          — head-to-head vs torch.mm fp16
```

---

## Weight Packing

2-bit encoding: 00=void(0), 01=identity(1), 11=prime(3), 10=reserved→identity.
16 weights per uint32. Matrix (4096×4096) = 4MB packed vs 32MB fp16 (8× compression).

K must be multiple of 16. Pad with identity if not.

---

## Kernel

Tiled GEMM. Shared memory for input tile. Inner loop: extract 2-bit weight from packed uint32, switch on value, accumulate in fp32, write output in fp16. No multiply instruction.

Block: TILE_M=64, TILE_N=64, TILE_K=64.
Switch: 00→skip, 01→acc+=val, 11→acc+=val+val+val, 10→acc+=val (defensive).

---

## Python Interface

`CrystalLinear(nn.Module)` — drop-in for nn.Linear. Stores packed_w as non-grad parameter. Forward calls crystal_cuda.crystal_forward. Bias supported (fp16).

`pack_ternary(weights)` — (N,K) {0,1,3} tensor → (N, K//16) uint32.
`unpack_ternary(packed, K)` — reverse. Reserved pattern 10 → identity 1.

---

## Benchmark

Warmup 100 + measure 1000 iterations at (M=4096, K=4096, N=4096).
Report: fp16 ms, crystal ms, speedup×, max error, memory compression.
Correctness check: max |ref - out| < 0.01.

---

## Error Handling

1. K%16 assertion with clear message
2. Reserved bit pattern 10 → identity (defensive, logged)
3. NaN passthrough (IEEE 754 standard)
4. No backward support (clear error if attempted)
5. Thread-safe, re-entrant, stateless kernel

---

## Constraints

- RTX 5070, SM 12.0, CUDA 12.x
- PyTorch 2.x with torch.utils.cpp_extension
- fp16 input/output, fp32 accumulation
- No external dependencies beyond CUDA toolkit
