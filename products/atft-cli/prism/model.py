"""
prism/model.py — Model loading and ternary weight extraction.

Works with ANY saved model.pt that contains TernaryLinear or BitFlipLinear
weights — does not require the model class definition at load time.

Crystal vocabulary:
  void     — weight == 0  (structured absence)
  identity — weight == 1  (identity transport)
  prime    — weight == 3  (irreducible amplification)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ── Import quantize_to_set from ternary_linear.py ────────────────────────
_TERNARY_DIR = str(
    Path(__file__).resolve().parents[2] / "ternary-architect"
)
if _TERNARY_DIR not in sys.path:
    sys.path.insert(0, _TERNARY_DIR)

from ternary_linear import quantize_to_set  # noqa: E402

# The canonical {0, 1, 3} value tensor (CPU fp32)
_CRYSTAL_VALUES = torch.tensor([0.0, 1.0, 3.0])

# Tolerance for "already quantized" check
_EPS = 1e-4


def _is_quantized(t: torch.Tensor) -> bool:
    """Return True if every value in *t* is within _EPS of {0, 1, 3}."""
    residuals = (t.unsqueeze(-1) - _CRYSTAL_VALUES).abs().min(dim=-1).values
    return bool((residuals < _EPS).all())


def extract_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load a model checkpoint and return quantized {0,1,3} weight tensors.

    Parameters
    ----------
    model_path : str
        Path to the ``.pt`` checkpoint file.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping of parameter name → quantized weight tensor (cpu, float32).
        Only parameters whose name ends with ``.weight`` are included.
    """
    raw = torch.load(model_path, map_location="cpu", weights_only=True)

    # The checkpoint might be a raw state dict or wrapped under "model_state"
    if isinstance(raw, dict) and "model_state" in raw:
        state_dict = raw["model_state"]
    elif isinstance(raw, dict) and not any(
        isinstance(v, dict) for v in list(raw.values())[:3]
    ):
        # Looks like a flat state dict already
        state_dict = raw
    elif isinstance(raw, dict):
        # Try to find the most likely state-dict key
        for key in ("state_dict", "model", "weights"):
            if key in raw:
                state_dict = raw[key]
                break
        else:
            state_dict = raw
    else:
        raise ValueError(f"Cannot interpret checkpoint format at {model_path}")

    result: dict[str, torch.Tensor] = {}
    crystal_values = _CRYSTAL_VALUES

    for name, param in state_dict.items():
        if not name.endswith(".weight"):
            continue
        if not isinstance(param, torch.Tensor):
            continue

        t = param.detach().float().cpu()

        if _is_quantized(t):
            result[name] = t
        else:
            result[name] = quantize_to_set(t, crystal_values)

    return result


def weight_stats(weights: dict[str, torch.Tensor]) -> dict:
    """Compute global crystal ratios across all extracted weight tensors.

    Returns
    -------
    dict with keys:
        void      — fraction of weights == 0
        identity  — fraction of weights == 1
        prime     — fraction of weights == 3
        n_weights — total number of weights examined
    """
    total = 0
    n_void = 0
    n_identity = 0
    n_prime = 0

    for t in weights.values():
        total += t.numel()
        n_void += int((t == 0.0).sum().item())
        n_identity += int((t == 1.0).sum().item())
        n_prime += int((t == 3.0).sum().item())

    if total == 0:
        return {"void": 0.0, "identity": 0.0, "prime": 0.0, "n_weights": 0}

    return {
        "void": n_void / total,
        "identity": n_identity / total,
        "prime": n_prime / total,
        "n_weights": total,
    }


def per_layer_stats(weights: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Compute crystal ratios per layer.

    Returns
    -------
    dict[str, dict]
        Maps each weight name to a dict with keys void/identity/prime/n_weights.
    """
    result: dict[str, dict] = {}
    for name, t in weights.items():
        total = t.numel()
        n_void = int((t == 0.0).sum().item())
        n_identity = int((t == 1.0).sum().item())
        n_prime = int((t == 3.0).sum().item())
        result[name] = {
            "void": n_void / total if total else 0.0,
            "identity": n_identity / total if total else 0.0,
            "prime": n_prime / total if total else 0.0,
            "n_weights": total,
        }
    return result
