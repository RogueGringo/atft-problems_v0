#!/usr/bin/env python3
"""crystal_ops.py — Crystal {0,1,3} forward pass, two backends.

Backend selection (automatic):
  1. load_inline (nvcc)  — requires CUDA_HOME / nvcc in PATH
  2. Triton               — pure Python JIT, no nvcc needed, ships with PyTorch 2.x

The same crystal arithmetic applies to both:
  w = 0 (void)  → skip
  w = 1 (unit)  → acc += x
  w = 3 (prime) → acc += x + x + x   (shift-add, no fp multiply)

2-bit packed weight layout (matches crystal_kernel.cu):
  00 → 0, 01 → 1, 11 → 3, 10 → 1 (defensive)
  16 weights per uint32, LSB = weight 0.

Usage:
    from crystal_ops import crystal_forward, CrystalLinear
"""
from __future__ import annotations

import os
import sys
import pathlib
import torch
import torch.nn as nn
from typing import Optional

# ── resolve paths ─────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).parent.resolve()
_CU_SOURCE = _HERE / "crystal_kernel.cu"

# ── backend registry ──────────────────────────────────────────────────────

_backend: str = "none"          # will be set after init
_crystal_ext = None             # load_inline module (if available)


# ── Backend 1: load_inline (nvcc path) ───────────────────────────────────

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Declare the launch wrapper defined in crystal_kernel.cu
extern "C" void launch_crystal_gemm(
    const __half* X, const unsigned int* W, __half* Y, const __half* bias,
    int M, int N, int K, cudaStream_t stream);

torch::Tensor crystal_forward_cpp(
    torch::Tensor x,
    torch::Tensor packed_w,
    c10::optional<torch::Tensor> bias,
    int64_t N,
    int64_t K)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(packed_w.is_cuda(), "packed_w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be float16");
    TORCH_CHECK(packed_w.dtype() == torch::kInt32, "packed_w must be int32");

    int M = x.size(0);
    auto y = torch::empty({M, N}, x.options());

    const __half* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        bias_ptr = reinterpret_cast<const __half*>(bias->data_ptr<at::Half>());
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    launch_crystal_gemm(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const unsigned int*>(packed_w.data_ptr<int32_t>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        bias_ptr,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        stream
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crystal_forward", &crystal_forward_cpp, "Crystal {0,1,3} forward pass",
          py::arg("x"), py::arg("packed_w"), py::arg("bias"), py::arg("N"), py::arg("K"));
}
"""


def _try_load_inline() -> bool:
    """Attempt JIT compilation via torch.utils.cpp_extension.load_inline.
    Returns True if successful."""
    global _crystal_ext, _backend

    try:
        from torch.utils.cpp_extension import load_inline, CUDA_HOME
        if CUDA_HOME is None:
            return False

        cu_source = _CU_SOURCE.read_text()

        _crystal_ext = load_inline(
            name="crystal_ext",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=cu_source,
            functions=["crystal_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        _backend = "load_inline"
        return True

    except Exception as e:
        return False


# ── Backend 2: Triton ─────────────────────────────────────────────────────

def _setup_triton() -> bool:
    """Set up Triton-based crystal kernel. Returns True if available."""
    global _backend

    try:
        import triton  # noqa: F401
        _backend = "triton"
        return True
    except ImportError:
        return False


# ── initialize backend ────────────────────────────────────────────────────

def _init_backend():
    """Select backend: try load_inline first, then Triton, then error."""
    global _backend
    if _backend != "none":
        return  # already initialized

    if _try_load_inline():
        return
    if _setup_triton():
        return

    _backend = "unavailable"


# ── Triton kernel implementation ──────────────────────────────────────────
# Defined at module level (required by triton.jit — must be in a .py file).

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _crystal_gemm_kernel(
        X_ptr,           # (M, K) fp16
        W_ptr,           # (N, K//16) int32 (packed 2-bit)
        Y_ptr,           # (M, N) fp16
        bias_ptr,        # (N,) fp16 or 0 (null)
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,   # rows of X per program
        BLOCK_N: tl.constexpr,   # rows of W per program (= output cols)
    ):
        """Triton crystal GEMM — each program computes a (BLOCK_M, BLOCK_N) tile.

        Inner loop iterates over K in steps of 16 (one packed uint32).
        For each word we extract 16 × 2-bit codes and apply crystal arithmetic.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Row/col ranges for this program
        m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
        n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

        m_mask = m_range < M
        n_mask = n_range < N

        # Accumulator in fp32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Words per row of W
        words = K // 16

        for word_idx in range(words):
            k_base = word_idx * 16

            # Load packed word for each neuron in n_range: (BLOCK_N,)
            w_ptrs = W_ptr + n_range * stride_wn + word_idx * stride_wk
            word = tl.load(w_ptrs, mask=n_mask, other=0)  # (BLOCK_N,) int32

            # Unroll 16 weights from this word
            for bit_pos in tl.static_range(16):
                k = k_base + bit_pos

                # Extract 2-bit code for each neuron: (BLOCK_N,)
                shift = 2 * bit_pos
                code = (word >> shift) & 3  # (BLOCK_N,)

                # Load activation for each sample: (BLOCK_M,)
                x_ptrs = X_ptr + m_range * stride_xm + k * stride_xk
                x_val = tl.load(x_ptrs, mask=m_mask, other=0.0)  # (BLOCK_M,)
                x_fp32 = x_val.to(tl.float32)

                # Outer product masks for each crystal value
                # code is (BLOCK_N,), x is (BLOCK_M,)
                code_bc = code[None, :]              # (1, BLOCK_N)
                x_bc   = x_fp32[:, None]            # (BLOCK_M, 1)

                # void (0): contributes 0 — no op
                # unit (1): contributes x
                # defensive (2): contributes x (same as 1)
                # prime (3): contributes 3*x

                contrib = tl.where(code_bc == 0,
                                   tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                                   tl.where(code_bc == 3,
                                            x_bc + x_bc + x_bc,
                                            x_bc))            # covers 1 and 2
                acc += contrib

        # Add bias
        if HAS_BIAS:
            b_ptrs = bias_ptr + n_range
            bias_vec = tl.load(b_ptrs, mask=n_mask, other=0.0)
            bias_fp32 = bias_vec.to(tl.float32)
            acc += bias_fp32[None, :]   # broadcast over m

        # Write output — convert fp32 accumulator back to fp16
        acc_h = acc.to(tl.float16)
        y_ptrs = (Y_ptr
                  + m_range[:, None] * stride_ym
                  + n_range[None, :] * stride_yn)
        tl.store(y_ptrs, acc_h, mask=m_mask[:, None] & n_mask[None, :])


def _crystal_forward_triton(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    bias: Optional[torch.Tensor],
    N: int,
    K: int,
) -> torch.Tensor:
    """Run crystal forward pass via Triton kernel."""
    M = x.size(0)
    y = torch.empty((M, N), dtype=torch.float16, device=x.device)

    has_bias = bias is not None and bias.numel() > 0
    bias_ptr = bias if has_bias else x  # dummy — not used when HAS_BIAS=False

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _crystal_gemm_kernel[grid](
        x, packed_w, y,
        bias_ptr if has_bias else x,   # bias_ptr (ignored when HAS_BIAS=False)
        M, N, K,
        x.stride(0), x.stride(1),
        packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y


# ── public API ────────────────────────────────────────────────────────────

def crystal_forward(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    bias: Optional[torch.Tensor],
    N: int,
    K: int,
) -> torch.Tensor:
    """Forward pass using the crystal kernel.

    Parameters
    ----------
    x : torch.Tensor
        Shape (M, K), dtype float16, on CUDA.
    packed_w : torch.Tensor
        Shape (N, K//16), dtype int32, on CUDA. Packed 2-bit {0,1,3} weights.
    bias : torch.Tensor or None
        Shape (N,), dtype float16, on CUDA. Optional.
    N : int
        Number of output features.
    K : int
        Number of input features (must equal x.shape[1]).

    Returns
    -------
    torch.Tensor
        Shape (M, N), dtype float16.
    """
    _init_backend()

    assert x.is_cuda, "x must be on CUDA"
    assert packed_w.is_cuda, "packed_w must be on CUDA"
    assert x.dtype == torch.float16, f"x must be float16, got {x.dtype}"
    assert packed_w.dtype == torch.int32, f"packed_w must be int32, got {packed_w.dtype}"

    if _backend == "load_inline":
        opt_bias = bias if bias is not None else torch.Tensor()
        return _crystal_ext.crystal_forward(x, packed_w, opt_bias, N, K)

    if _backend == "triton":
        return _crystal_forward_triton(x, packed_w, bias, N, K)

    raise RuntimeError(
        "No crystal backend available. "
        "Install nvcc (for load_inline) or triton (pip install triton)."
    )


# ── CrystalLinear ─────────────────────────────────────────────────────────

class CrystalLinear(nn.Module):
    """Drop-in replacement for nn.Linear with packed {0,1,3} ternary weights.

    Weights are stored as 2-bit packed uint32 tensors (8× vs fp32).
    Forward pass uses the crystal kernel (Triton or nvcc).

    This module does NOT train — it is an inference module.
    Use TernaryLinear for training, then convert with CrystalLinear.from_ternary().

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % 16 == 0, (
            f"CrystalLinear requires in_features divisible by 16, got {in_features}"
        )
        self.in_features = in_features
        self.out_features = out_features

        # Packed weights: (out_features, in_features // 16) int32
        self.register_buffer(
            "packed_weight",
            torch.zeros(out_features, in_features // 16, dtype=torch.int32)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure fp16
        if x.dtype != torch.float16:
            x = x.half()
        return crystal_forward(
            x,
            self.packed_weight,
            self.bias,
            self.out_features,
            self.in_features,
        )

    @classmethod
    def from_ternary(cls, linear: nn.Module) -> "CrystalLinear":
        """Convert a TernaryLinear (or nn.Linear) to CrystalLinear.

        Quantizes weights to {0,1,3} and packs to 2-bit.
        """
        from pack_weights import pack_ternary

        w = linear.weight.data.detach().cpu()

        # Quantize: round to nearest of {0,1,3}
        values = torch.tensor([0.0, 1.0, 3.0])
        dists = (w.unsqueeze(-1) - values).abs()
        w_q = values[dists.argmin(-1)]

        has_bias = (hasattr(linear, "bias") and linear.bias is not None)
        crystal = cls(linear.in_features, linear.out_features, bias=has_bias)

        packed = pack_ternary(w_q)
        crystal.packed_weight.copy_(packed)

        if has_bias:
            crystal.bias.copy_(linear.bias.data.detach().float().half())

        return crystal.cuda()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, backend={_backend}, "
            f"packed_bytes={self.packed_weight.numel() * 4}"
        )


# ── module info ───────────────────────────────────────────────────────────

def backend_info() -> dict:
    """Return information about the active backend."""
    _init_backend()
    return {
        "backend": _backend,
        "cuda_home": os.environ.get("CUDA_HOME", "not set"),
        "triton_available": _TRITON_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    _init_backend()
    info = backend_info()
    print("Crystal backend info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if _backend in ("triton", "load_inline"):
        # Quick sanity check
        M, K, N = 4, 32, 8
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)

        # Pack a simple weight matrix (all ones)
        sys.path.insert(0, str(_HERE))
        from pack_weights import pack_ternary
        w = torch.ones(N, K)
        packed = pack_ternary(w).cuda()

        y = crystal_forward(x, packed, None, N, K)
        print(f"\nSanity check: x={x.shape} × W={w.shape} → y={y.shape}")

        # Reference: plain matmul
        ref = x.float() @ w.T.cuda().float()
        err = (y.float() - ref).abs().max().item()
        print(f"Max error vs reference: {err:.4f}")
        print("Sanity check PASS" if err < 0.5 else "Sanity check FAIL")
