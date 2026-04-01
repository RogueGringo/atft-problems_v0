"""The irreducible computation — {0,1,3} ternary forward pass.

Phase A: ternary_matmul_dense kernel operational, full forward pass stubbed.
Phase B: ONNX-QNN export for Hexagon NPU acceleration.
"""
from __future__ import annotations
import numpy as np

def ternary_matmul_dense(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Ternary matrix multiply using only skip/pass/shift-add.

    x:       (B, D_in) INT16 activations
    weights: (D_in, D_out) UINT8 dense ternary values {0, 1, 3}

    Accumulation in INT32 to prevent overflow.
    """
    x = x.astype(np.int32)
    w = weights.astype(np.int32)
    mask_1 = (w == 1).astype(np.int32)
    mask_3 = (w == 3).astype(np.int32)
    effective_w = mask_1 + mask_3 * 3
    result = x @ effective_w
    return result.astype(np.int32)

def forward_pass(x: np.ndarray, weights_path: str) -> dict:
    """Full ternary forward pass — Phase A stub."""
    raise NotImplementedError(
        "Full forward pass requires pre-trained weights from desktop. "
        "Use export_for_arm() on the desktop to create .npz weights, "
        "then transfer to ARM laptop. See spec: ARM-004."
    )
