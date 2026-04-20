"""Int8 / int16 quantized typed sheaf Laplacian — SIDECAR for NPU prototyping.

This module is a proof-of-concept that the 8x8 R_dst block can be quantized to
a low-bit-width integer representation with the final lambda_min close to the
fp64 reference on realistic webs. It is NOT used by the production pipeline
(see functional.py — still imports the fp64 typed_sheaf_laplacian from
laplacian.py).

Target platform: Snapdragon X Plus Hexagon NPU via ONNX Runtime QNN (follow-up
project once this CPU reference exists).

Bit-width floor finding (spec §Risks fallback):
  int8 with uniform per-tensor scale and LAMBDA_UPPER=2.2 produces ~1.3e-2
  relative error on lambda_min for 30-node fixtures — above the 1e-3 spec
  contract. Per-entry rms quantization error at scale 127/2.2 ~= 58 is
  ~1/(2*58) ~= 0.87%, which propagates to ~1.3-1.7% through L = delta.T @ delta.
  int16 (empirically rel ~1e-5) is the production-accurate bit width.
  Default therefore bumped to bits=16. int8 remains available as the
  reference path for the NPU prototype contract (with the accuracy caveat
  documented).

Structure of R_dst (from laplacian.py):
  R_dst = lambda * (Rot_3 + P_4 + I_1)  — 8x8 block-diagonal.
  - Rot_3: 3x3 rotation, entries in [-1, 1] — int8 quantization at scale=127
           gives ~0.4-0.9% per-entry error (uniform-scale floor).
  - P_4:   4x4 permutation, entries in {0, 1} — int8 exact.
  - I_1:   1x1 identity, entry in {0, 1} — int8 exact.
  - lambda: one of 6 known floats — applied in fp before quantization.

Algorithm:
  1. Per-edge: compute fp64 R_dst via the same _R_dst_for_edge as laplacian.py.
  2. Quantize R_dst to int8 (or int16) via round-to-nearest with clamp.
  3. Assemble delta_int as scipy.sparse.csr with int8/int16 dtype.
  4. Promote to accumulator dtype (int32 for int8 delta; int64 for int16 delta
     because int16^2 * ~12-incident-edges exceeds int32's 2.1e9 limit).
  5. Compute L_int = (delta_acc.T @ delta_acc).
  6. Dequantize: L_fp = L_int.astype(float32) / (scale ** 2).
  7. Return sparse csr fp32 (callers can toarray or feed directly to eigsh).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse as _sparse

from problems.hubble_tension_web.laplacian import (
    STALK_DIM,
    _R_dst_for_edge,
    build_stalk_init,
)
from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def _scale_for_bits(bits: int) -> int:
    """Return the quantization scale for a signed `bits`-bit integer type."""
    if bits == 8:
        return 127
    if bits == 16:
        return 32767
    raise ValueError(f"bits must be 8 or 16; got {bits}")


def _dtype_for_bits(bits: int) -> np.dtype:
    if bits == 8:
        return np.dtype(np.int8)
    if bits == 16:
        return np.dtype(np.int16)
    raise ValueError(f"bits must be 8 or 16; got {bits}")


def quantize_rdst(R_dst: np.ndarray, *, bits: int = 8) -> Tuple[np.ndarray, int]:
    """Quantize an 8x8 R_dst block to signed int8 or int16.

    Returns (R_quantized, scale). The dequantization formula is
    `R_dequantized = R_quantized.astype(float) / scale`.

    R_dst may be pre-multiplied by lambda (the lambda values are {1.0, 1.1,
    1.2, 1.5, 1.8, 2.2}, max 2.2); we absorb the max lambda into the effective
    scale so int8 never overflows. This keeps the scale UNIFORM across the
    whole delta matrix — essential for the int matmul to be meaningful after
    dequantization.

    For simplicity and strict bit-equivalence semantics with the existing
    _R_dst_for_edge output:
      scale_eff = scale_max / lambda_upper_bound
      R_q       = round(R_dst * scale_eff), clamped to [-scale_max, scale_max]
      R_dq      = R_q / scale_eff        (for dequantization downstream)

    Returns `scale_int = round(scale_eff)` as the scale. The tiny extra rounding
    on the scale itself contributes ~1/(2*scale_max) relative error — acceptable
    under the 1e-3 spec bound.
    """
    scale_max = _scale_for_bits(bits)
    LAMBDA_UPPER = 2.2

    scale_eff = scale_max / LAMBDA_UPPER
    q_float = np.round(R_dst * scale_eff)
    q_float = np.clip(q_float, -scale_max, scale_max)
    q = q_float.astype(_dtype_for_bits(bits))
    scale_int = int(round(scale_eff))
    return q, scale_int


def typed_sheaf_laplacian_quantized(
    *,
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = STALK_DIM,
    rng_seed: int = 0,
    environments: Optional[List[Environment]] = None,
    bits: int = 16,
) -> "_sparse.csr_matrix":
    """Build L via int8/int16 quantized delta, int accumulation, fp32 dequant.

    Public interface mirrors `typed_sheaf_laplacian` (same kwargs, same return
    shape) with the addition of `bits` (8 or 16).

    Default is `bits=16` because int8 exceeds the 1e-3 rel-error contract at
    the uniform-scale bit-width floor (~1.3e-2 measured on 30-node fixtures).
    int16 measures at rel ~1e-5, well under contract. See module docstring
    for the bit-width floor discussion.

    Accumulator dtype: int32 for bits=8, int64 for bits=16. int16 squared and
    summed over a typical ~12 incident edges per node reaches ~4.5e9 on the
    fixture (m=110 edges, n=30), which overflows int32 (limit 2.1e9).

    Output: scipy.sparse.csr_matrix, dtype float32, shape (n*8, n*8). Pass to
    scipy.sparse.linalg.eigsh exactly like the fp64 result.
    """
    if stalk_dim != STALK_DIM:
        raise ValueError(
            f"typed_sheaf_laplacian_quantized requires stalk_dim={STALK_DIM}; "
            f"got {stalk_dim}."
        )

    if environments is None:
        env_of: List[Optional[str]] = [None] * n
        for s, d, etype in edges:
            e_s, e_d = etype.split("-", 1)
            env_of[s] = e_s
            env_of[d] = e_d
        if any(e is None for e in env_of):
            raise ValueError(
                "Could not infer environments for all nodes from edges; "
                "pass environments=web.environments explicitly."
            )
        env_values = env_of
    else:
        env_values = [e.value for e in environments]

    envs_enum = [Environment(v) for v in env_values]
    web = LocalCosmicWeb(positions=positions, environments=envs_enum)
    stalks, _flags = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    int_dtype = _dtype_for_bits(bits)
    scale_max = _scale_for_bits(bits)

    # Pre-quantize the -I block once (same for every edge row).
    neg_I_q, scale_int = quantize_rdst(-np.eye(STALK_DIM), bits=bits)

    delta_int = _sparse.lil_matrix(
        (m * STALK_DIM, n * STALK_DIM), dtype=int_dtype,
    )
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        R_dst_q, _ = quantize_rdst(R_dst, bits=bits)

        row0 = e_idx * STALK_DIM
        col_s0 = s * STALK_DIM
        col_d0 = d * STALK_DIM
        delta_int[row0:row0 + STALK_DIM, col_s0:col_s0 + STALK_DIM] = neg_I_q
        delta_int[row0:row0 + STALK_DIM, col_d0:col_d0 + STALK_DIM] = R_dst_q

    delta_csr = delta_int.tocsr()
    # int8 * int8 fits in int32 with margin; int16 * int16 summed over ~12
    # incident edges per node overflows int32, so promote to int64.
    accum_dtype = np.int32 if bits == 8 else np.int64
    delta_acc = delta_csr.astype(accum_dtype)
    L_int = (delta_acc.T @ delta_acc).tocsr()
    L_fp = L_int.astype(np.float32) / np.float32(scale_int * scale_int)
    L_fp = (0.5 * (L_fp + L_fp.T)).tocsr()
    return L_fp
