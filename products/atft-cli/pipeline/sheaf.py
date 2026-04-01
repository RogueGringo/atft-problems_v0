"""
pipeline/sheaf.py — Sheaf Laplacian spectral analysis.

For each layer's weight matrix, constructs the sheaf Laplacian L_F and
analyzes its spectrum: kernel dimension, spectral gap, and Gini of
eigenvalues.

Large matrices (>256 rows or cols) are projected to 256×256 via random
subsampling before Laplacian construction.

Usage (standalone):
    python -m pipeline.sheaf --model path/to/model.pt

Usage (piped):
    <previous_stage_output> | python -m pipeline.sheaf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Add artifact-analysis to sys.path ─────────────────────────────────────
_SHEAF_DIR = str(Path(__file__).resolve().parents[2] / "artifact-analysis")
if _SHEAF_DIR not in sys.path:
    sys.path.insert(0, _SHEAF_DIR)

from sheaf_laplacian import (  # noqa: E402
    analyze_sheaf_laplacian,
    sheaf_laplacian_from_weights,
)

try:
    from ..prism.model import extract_weights
    from ..utils.io import emit, make_result, merge_results, read_stdin_json, summary
except ImportError:
    from prism.model import extract_weights  # type: ignore[no-redef]
    from utils.io import emit, make_result, merge_results, read_stdin_json, summary  # type: ignore[no-redef]

_PROJ_SIZE = 256


def _project_weight(W: torch.Tensor) -> np.ndarray:
    """Project weight tensor to at most _PROJ_SIZE × _PROJ_SIZE."""
    rng = np.random.RandomState(42)
    out_dim, in_dim = W.shape
    row_idx = rng.choice(out_dim, min(_PROJ_SIZE, out_dim), replace=False)
    col_idx = rng.choice(in_dim, min(_PROJ_SIZE, in_dim), replace=False)
    return W[row_idx][:, col_idx].numpy()


def _layer_sheaf(tensor: torch.Tensor, name: str = "") -> dict | None:
    """Build and analyze sheaf Laplacian for a single weight tensor.

    Returns None for 1D tensors (e.g. LayerNorm weight/bias).
    """
    if tensor.dim() < 2:
        return None
    W = tensor
    if W.shape[0] > _PROJ_SIZE or W.shape[1] > _PROJ_SIZE:
        W_np = _project_weight(W)
    else:
        W_np = W.numpy()

    L = sheaf_laplacian_from_weights(W_np)
    analysis = analyze_sheaf_laplacian(L, name=name)

    return {
        "kernel_dim": analysis["kernel_dim"],
        "spectral_gap": analysis["spectral_gap"],
        "gini_eigenvalues": analysis["gini_eigenvalues"],
        "eff_rank_eigenvalues": analysis["eff_rank_eigenvalues"],
        "matrix_size": analysis["matrix_size"],
        "sampled": analysis["sampled"],
    }


def run(
    model_path: str | None = None,
    weights: dict | None = None,
    top_n: int | None = None,
    **kwargs,
) -> None:
    """Run the sheaf Laplacian analysis stage.

    Parameters
    ----------
    model_path:
        Path to the model checkpoint.  If *None*, extracted from piped stdin.
    weights:
        Pre-loaded weight dict (e.g. from ``extract_weights``).  If provided,
        the checkpoint is not re-read.
    top_n:
        If set, only analyze the first *top_n* layers (for speed).
    """
    # ── 1. Resolve model path ──────────────────────────────────────────────
    piped = read_stdin_json()

    if model_path is None and piped is not None:
        model_path = piped.get("meta", {}).get("model")

    if model_path is None and weights is None:
        summary("sheaf: ERROR — no model path (use --model or pipe from previous stage)")
        sys.exit(1)

    # ── 2. Load weights ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if weights is None:
        weights = extract_weights(model_path)

    # ── 3. Compute per-layer sheaf analysis ────────────────────────────────
    per_layer: dict[str, dict] = {}
    layer_items = list(weights.items())

    # Filter to 2D layers first, then apply top_n
    layer_items_2d = [(n, t) for n, t in layer_items if t.dim() >= 2]
    if top_n is not None:
        layer_items_2d = layer_items_2d[:top_n]

    for name, tensor in layer_items_2d:
        result_layer = _layer_sheaf(tensor, name=name)
        if result_layer is not None:
            per_layer[name] = result_layer

    elapsed = time.perf_counter() - t0

    # ── 4. Aggregate global stats ──────────────────────────────────────────
    total_kernel_dim = sum(v["kernel_dim"] for v in per_layer.values())
    n_layers = len(per_layer)
    mean_spectral_gap = (
        float(np.mean([v["spectral_gap"] for v in per_layer.values()]))
        if n_layers > 0
        else 0.0
    )
    mean_gini = (
        float(np.mean([v["gini_eigenvalues"] for v in per_layer.values()]))
        if n_layers > 0
        else 0.0
    )

    # ── 5. Build result dict ───────────────────────────────────────────────
    result = {
        "total_kernel_dim": total_kernel_dim,
        "mean_spectral_gap": mean_spectral_gap,
        "mean_gini_eigenvalues": mean_gini,
        "per_layer": per_layer,
    }

    meta = {"model": model_path} if model_path else {}

    # ── 6. Build envelope ──────────────────────────────────────────────────
    if piped is not None:
        envelope = merge_results(piped, "sheaf", result, meta=meta)
    else:
        envelope = make_result("sheaf", result, meta=meta)

    # ── 7. Emit JSON to stdout ─────────────────────────────────────────────
    emit(envelope)

    # ── 8. Human summary to stderr ─────────────────────────────────────────
    summary(
        f"sheaf: kernel_dim={total_kernel_dim} spectral_gap={mean_spectral_gap:.4f}"
        f" gini_eig={mean_gini:.4f} layers={n_layers} ({elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sheaf Laplacian spectral analysis")
    parser.add_argument("--model", dest="model_path", help="Path to model checkpoint")
    parser.add_argument(
        "--top-n", type=int, default=None, help="Only analyze first N layers"
    )
    args = parser.parse_args()
    run(model_path=args.model_path, top_n=args.top_n)


if __name__ == "__main__":
    main()
