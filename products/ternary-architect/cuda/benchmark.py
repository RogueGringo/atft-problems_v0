#!/usr/bin/env python3
"""benchmark.py — Crystal {0,1,3} kernel vs PyTorch fp16 torch.mm.

Measures:
  - Latency (ms) for M×K @ K×N matrix multiply
  - Speedup of crystal kernel vs fp16 torch.mm
  - Memory usage (packed 2-bit vs fp16 weights)
  - Correctness: max absolute error vs reference

Run:
    cd products/ternary-architect/cuda && python3 benchmark.py

The crystal kernel uses no floating-point multiplications:
  w=0 → skip
  w=1 → add
  w=3 → add + add + add   (shift-add)
"""
from __future__ import annotations

import sys
import os
import time
import pathlib

# Ensure this directory is in the path (for pack_weights, crystal_ops)
_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

import torch


def benchmark_crystal_vs_float(
    M: int = 4096,
    K: int = 4096,
    N: int = 4096,
    warmup: int = 100,
    iters: int = 1000,
    verbose: bool = True,
) -> dict:
    """Head-to-head benchmark: crystal kernel vs fp16 torch.mm.

    Parameters
    ----------
    M, K, N : int
        Matrix dimensions: (M×K) @ (K×N) → (M×N).
    warmup : int
        Warmup iterations (not timed).
    iters : int
        Timed iterations.
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict with keys: mm_ms, crystal_ms, speedup, max_error, memory_ratio, backend
    """
    # ── imports ──────────────────────────────────────────────────────────

    try:
        from pack_weights import pack_ternary, unpack_ternary
    except ImportError as e:
        print(f"ERROR: Could not import pack_weights: {e}")
        sys.exit(1)

    try:
        from crystal_ops import crystal_forward, backend_info
    except ImportError as e:
        print(f"ERROR: Could not import crystal_ops: {e}")
        sys.exit(1)

    # ── check GPU ─────────────────────────────────────────────────────────

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    info = backend_info()
    backend = info["backend"]

    if backend == "unavailable":
        print("ERROR: No crystal backend available (no nvcc, no triton).")
        sys.exit(1)

    if verbose:
        print("=" * 60)
        print("Crystal GEMM Benchmark")
        print("=" * 60)
        print(f"GPU:         {gpu_name}")
        print(f"Backend:     {backend}")
        print(f"Shape:       M={M}, K={K}, N={N}")
        print(f"Warmup:      {warmup} iters")
        print(f"Timed:       {iters} iters")
        print()

    # ── 1. Create inputs ──────────────────────────────────────────────────

    torch.manual_seed(42)

    # fp16 activation
    x = torch.randn(M, K, device=device, dtype=torch.float16)

    # fp16 weight matrix (for torch.mm baseline)
    w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)

    # Ternary weight matrix — random {0, 1, 3}
    idx = torch.randint(0, 3, (N, K), device="cpu")
    values = torch.tensor([0.0, 1.0, 3.0])
    w_ternary = values[idx]   # (N, K) cpu float32

    # ── 2. Pack ternary weights ───────────────────────────────────────────

    packed = pack_ternary(w_ternary).to(device=device, dtype=torch.int32)
    # (N, K//16) int32 on GPU

    # ── 3. Warmup ─────────────────────────────────────────────────────────

    if verbose:
        print("Warming up torch.mm (fp16)...")
    for _ in range(warmup):
        _ = torch.mm(x, w_fp16.T)
    torch.cuda.synchronize()

    if verbose:
        print(f"Warming up crystal forward (backend={backend})...")
    for _ in range(warmup):
        try:
            _ = crystal_forward(x, packed, None, N, K)
        except Exception as e:
            print(f"\nERROR during crystal warmup: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    torch.cuda.synchronize()

    # ── 4. Time torch.mm ──────────────────────────────────────────────────

    if verbose:
        print(f"\nTiming torch.mm ({iters} iters)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.mm(x, w_fp16.T)
    torch.cuda.synchronize()
    mm_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # ── 5. Time crystal forward ───────────────────────────────────────────

    if verbose:
        print(f"Timing crystal forward ({iters} iters)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = crystal_forward(x, packed, None, N, K)
    torch.cuda.synchronize()
    crystal_ms = (time.perf_counter() - t0) * 1000.0 / iters

    speedup = mm_ms / crystal_ms if crystal_ms > 0 else float("inf")

    # ── 6. Correctness check ──────────────────────────────────────────────

    # Unpack weights and compute reference via torch.mm
    w_unpacked = unpack_ternary(packed.cpu(), K)  # (N, K) float32
    w_unpacked_gpu = w_unpacked.to(device=device, dtype=torch.float16)

    # Crystal result
    y_crystal = crystal_forward(x, packed, None, N, K).float()

    # Reference: fp16 matmul with the same ternary weights
    y_ref = torch.mm(x.float(), w_unpacked_gpu.T.float())

    max_error = (y_crystal - y_ref).abs().max().item()
    mean_error = (y_crystal - y_ref).abs().mean().item()

    # ── 7. Memory ─────────────────────────────────────────────────────────

    fp16_bytes = N * K * 2          # fp16 = 2 bytes per element
    packed_bytes = packed.numel() * 4  # int32 = 4 bytes per word, 16 weights per word
    memory_ratio = fp16_bytes / packed_bytes  # how many times smaller

    # ── 8. Print results ──────────────────────────────────────────────────

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  torch.mm (fp16):     {mm_ms:.3f} ms/iter")
        print(f"  crystal forward:     {crystal_ms:.3f} ms/iter")
        print(f"  Speedup:             {speedup:.2f}x  ", end="")
        if speedup >= 1.0:
            print(f"(crystal is {speedup:.2f}x faster)")
        else:
            print(f"(crystal is {1/speedup:.2f}x SLOWER — expected for simple kernel v1)")
        print()
        print(f"  Max absolute error:  {max_error:.6f}")
        print(f"  Mean absolute error: {mean_error:.6f}")
        ok = max_error < 2.0  # fp16 has ~1e-3 relative error, ternary outputs scale large
        print(f"  Correctness:         {'PASS' if ok else 'FAIL'}")
        print()
        print(f"  Weight memory:")
        print(f"    fp16 weights:      {fp16_bytes / 1024**2:.1f} MB")
        print(f"    packed (2-bit):    {packed_bytes / 1024**2:.1f} MB")
        print(f"    Compression:       {memory_ratio:.1f}x  (vs fp16)")
        print()
        print(f"  Note: Crystal v1 is NOT expected to beat cuBLAS.")
        print(f"  This kernel validates correctness and the 2-bit packing.")
        print(f"  Tiled shared-memory version is future work.")
        print("=" * 60)

    return {
        "mm_ms": mm_ms,
        "crystal_ms": crystal_ms,
        "speedup": speedup,
        "max_error": max_error,
        "mean_error": mean_error,
        "memory_ratio": memory_ratio,
        "fp16_bytes": fp16_bytes,
        "packed_bytes": packed_bytes,
        "backend": backend,
        "gpu": gpu_name,
    }


# ── also run a small sanity-check at lower sizes ─────────────────────────

def correctness_sweep(verbose: bool = True):
    """Run crystal kernel at multiple sizes and check correctness."""
    from pack_weights import pack_ternary, unpack_ternary
    from crystal_ops import crystal_forward

    device = torch.device("cuda")

    if verbose:
        print("\nCorrectness sweep:")

    all_ok = True
    for M, K, N in [(1, 16, 8), (4, 32, 16), (16, 64, 32), (64, 128, 64), (256, 256, 128)]:
        torch.manual_seed(0)
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        idx = torch.randint(0, 3, (N, K))
        w_ternary = torch.tensor([0.0, 1.0, 3.0])[idx]

        packed = pack_ternary(w_ternary).to(device=device, dtype=torch.int32)
        w_gpu = w_ternary.to(device=device, dtype=torch.float16)

        y = crystal_forward(x, packed, None, N, K).float()
        ref = torch.mm(x.float(), w_gpu.T.float())

        err = (y - ref).abs().max().item()
        ok = err < 1.0  # loose threshold for fp16 accumulation differences
        all_ok = all_ok and ok

        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  M={M:4d}, K={K:4d}, N={N:4d} — max_err={err:.4f} — {status}")

    if verbose:
        print(f"  Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")

    return all_ok


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crystal GEMM benchmark")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--no-correctness-sweep", action="store_true")
    args = parser.parse_args()

    # Run correctness sweep first (fast)
    if not args.no_correctness_sweep:
        ok = correctness_sweep(verbose=True)
        if not ok:
            print("\nERROR: correctness failures — aborting benchmark")
            sys.exit(1)

    # Main benchmark
    results = benchmark_crystal_vs_float(
        M=args.M,
        K=args.K,
        N=args.N,
        warmup=args.warmup,
        iters=args.iters,
        verbose=True,
    )
