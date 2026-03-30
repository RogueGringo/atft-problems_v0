#!/usr/bin/env python3
"""TernaryLinear — The {0, 1, 3} linear layer.

Each weight is one of three values:
  0 (void)   — no connection, structured absence
  1 (unit)   — identity transport, connection exists
  3 (prime)  — irreducible amplification, generates structure

Training: latent fp32 weights + straight-through estimator (STE).
Inference: no floating-point multiplier needed.
  x * 0 = skip (FREE)
  x * 1 = pass (FREE)
  x * 3 = (x << 1) + x (CHEAP)

2 bits per weight. Same as BitNet b1.58.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Quantization functions ────────────────────────────────────────────────

import math

SQRT2 = math.sqrt(2)

WEIGHT_SETS = {
    "013": torch.tensor([0.0, 1.0, 3.0]),
    "n101": torch.tensor([-1.0, 0.0, 1.0]),  # BitNet
    "012": torch.tensor([0.0, 1.0, 2.0]),     # ablation
    "015": torch.tensor([0.0, 1.0, 5.0]),     # ablation
    "017": torch.tensor([0.0, 1.0, 7.0]),     # ablation
    "0123": torch.tensor([0.0, 1.0, 2.0, 3.0]),  # 4-level ablation
    # The prime sequence — transport maps ARE the primes
    # 0=void, 1=unit, √2=diagonal(2 geometricized), 3,5,7,11=irreducible amplifiers
    "primes": torch.tensor([0.0, 1.0, SQRT2, 3.0, 5.0, 7.0, 11.0]),
}


def quantize_to_set(w: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Snap each weight to the nearest value in the set.

    This is the forward quantization — deterministic, differentiable via STE.
    """
    values = values.to(w.device, w.dtype)
    # Expand for broadcasting: w is (...,), values is (k,)
    # distances: (..., k)
    dists = (w.unsqueeze(-1) - values).abs()
    indices = dists.argmin(dim=-1)
    return values[indices]


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator for discrete quantization.

    Forward: snap to nearest value in set.
    Backward: gradient passes through as if quantization didn't happen.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return quantize_to_set(w, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradient through unchanged
        return grad_output, None


# ── TernaryLinear layer ───────────────────────────────────────────────────

class TernaryLinear(nn.Module):
    """Linear layer with ternary weights from a configurable value set.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
        Whether to include a learnable bias (fp32, not quantized).
    weight_set : str
        Key into WEIGHT_SETS. Default "013" = {0, 1, 3}.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, weight_set: str = "013"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_set = weight_set

        # Register the value set as a buffer (not a parameter)
        self.register_buffer("values", WEIGHT_SETS[weight_set].clone())

        # Latent weights — what the optimizer sees (fp32, continuous)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize latent weights so they spread across the value set."""
        v = self.values
        lo, hi = v.min().item(), v.max().item()
        # Uniform over the range of the value set
        nn.init.uniform_(self.weight, lo - 0.3, hi + 0.3)

    def get_quantized_weight(self) -> torch.Tensor:
        """Return the quantized weight matrix (for inference / diagnostics)."""
        return STEQuantize.apply(self.weight, self.values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = STEQuantize.apply(self.weight, self.values)
        return F.linear(x, w_q, self.bias)

    def weight_distribution(self) -> dict[str, float]:
        """Return fraction of weights at each value (diagnostic)."""
        with torch.no_grad():
            w_q = quantize_to_set(self.weight, self.values)
            total = w_q.numel()
            dist = {}
            for v in self.values:
                count = (w_q == v.item()).sum().item()
                dist[f"w={v.item():.0f}"] = count / total
            return dist

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, set={self.weight_set} "
                f"values={self.values.tolist()}")


class FP16Linear(nn.Module):
    """Standard linear layer (fp16/fp32) as baseline control."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_set: str = "fp16"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_set = weight_set

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def weight_distribution(self) -> dict[str, str]:
        return {"type": "continuous", "std": f"{self.linear.weight.std().item():.4f}"}

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def get_quantized_weight(self) -> torch.Tensor:
        return self.linear.weight.data

    def extra_repr(self) -> str:
        return self.linear.extra_repr()


def make_linear(in_features: int, out_features: int,
                bias: bool = True, weight_set: str = "013") -> nn.Module:
    """Factory: create the right linear layer for a given weight set."""
    if weight_set == "fp16":
        return FP16Linear(in_features, out_features, bias=bias)
    return TernaryLinear(in_features, out_features, bias=bias,
                         weight_set=weight_set)
