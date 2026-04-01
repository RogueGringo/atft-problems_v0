"""
pipeline/full.py — Full measurement pass: crystal + persistence + sheaf.

Loads weights ONCE and runs all three measurements in sequence.  In full
mode, sheaf analysis is restricted to the top 5 layers for speed.

Usage (standalone):
    python -m pipeline.full --model path/to/model.pt

Usage (piped):
    <previous_stage_output> | python -m pipeline.full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Add dependency paths ───────────────────────────────────────────────────
_TOPO_DIR = str(Path(__file__).resolve().parents[2] / "topological-router")
_SHEAF_DIR = str(Path(__file__).resolve().parents[2] / "artifact-analysis")
if _TOPO_DIR not in sys.path:
    sys.path.insert(0, _TOPO_DIR)
if _SHEAF_DIR not in sys.path:
    sys.path.insert(0, _SHEAF_DIR)

from topo_measures import gini_fast, h0_persistence  # noqa: E402
from sheaf_laplacian import (  # noqa: E402
    analyze_sheaf_laplacian,
    sheaf_laplacian_from_weights,
)

try:
    from ..prism.model import extract_weights, per_layer_stats, weight_stats
    from ..utils.io import emit, make_result, merge_results, read_stdin_json, summary
except ImportError:
    from prism.model import extract_weights, per_layer_stats, weight_stats  # type: ignore[no-redef]
    from utils.io import emit, make_result, merge_results, read_stdin_json, summary  # type: ignore[no-redef]

_MAX_ROWS = 200
_PROJ_SIZE = 256
_SHEAF_TOP_N = 5


# ── Crystal helpers ────────────────────────────────────────────────────────

def _run_crystal(weights: dict) -> dict:
    stats = weight_stats(weights)
    layers = per_layer_stats(weights)
    return {
        "crystal": {
            "void": stats["void"],
            "identity": stats["identity"],
            "prime": stats["prime"],
        },
        "n_weights": stats["n_weights"],
        "per_layer": layers,
    }


# ── Persistence helpers ────────────────────────────────────────────────────

def _layer_persistence(tensor) -> dict | None:
    """Returns None for 1D tensors."""
    if tensor.dim() < 2:
        return None
    W = tensor.numpy()
    rng = np.random.RandomState(42)
    if W.shape[0] > _MAX_ROWS:
        idx = rng.choice(W.shape[0], _MAX_ROWS, replace=False)
        W = W[idx]
    bars = h0_persistence(W)
    n_bars = len(bars)
    h0_gini = float(gini_fast(bars)) if n_bars > 0 else 0.0
    return {
        "n_bars": n_bars,
        "h0_gini": h0_gini,
        "max_bar": float(bars.max()) if n_bars > 0 else 0.0,
        "mean_bar": float(bars.mean()) if n_bars > 0 else 0.0,
    }


def _run_persistence(weights: dict) -> dict:
    per_layer: dict[str, dict] = {}
    for name, tensor in weights.items():
        layer_stats = _layer_persistence(tensor)
        if layer_stats is None:
            continue
        per_layer[name] = layer_stats
    total_bars = sum(v["n_bars"] for v in per_layer.values())
    if total_bars > 0:
        global_gini = float(
            sum(v["h0_gini"] * v["n_bars"] for v in per_layer.values()) / total_bars
        )
    else:
        global_gini = 0.0
    return {
        "h0_gini": global_gini,
        "n_bars": total_bars,
        "per_layer": per_layer,
    }


# ── Sheaf helpers ──────────────────────────────────────────────────────────

def _project_weight(W_tensor, rng=None):
    """Project a weight tensor to at most _PROJ_SIZE x _PROJ_SIZE, returns np array."""
    if rng is None:
        rng = np.random.RandomState(42)
    out_dim, in_dim = W_tensor.shape
    row_idx = rng.choice(out_dim, min(_PROJ_SIZE, out_dim), replace=False)
    col_idx = rng.choice(in_dim, min(_PROJ_SIZE, in_dim), replace=False)
    # Use torch indexing, then convert to numpy
    return W_tensor[row_idx][:, col_idx].numpy()


def _layer_sheaf(tensor, name: str = "") -> dict | None:
    """Returns None for 1D tensors."""
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


def _run_sheaf(weights: dict, top_n: int = _SHEAF_TOP_N) -> dict:
    per_layer: dict[str, dict] = {}
    # Filter 2D tensors first, then take top_n
    layer_items_2d = [(n, t) for n, t in weights.items() if t.dim() >= 2][:top_n]
    for name, tensor in layer_items_2d:
        result_layer = _layer_sheaf(tensor, name=name)
        if result_layer is not None:
            per_layer[name] = result_layer
    n_layers = len(per_layer)
    total_kernel_dim = sum(v["kernel_dim"] for v in per_layer.values())
    mean_spectral_gap = (
        float(np.mean([v["spectral_gap"] for v in per_layer.values()])) if n_layers > 0 else 0.0
    )
    mean_gini = (
        float(np.mean([v["gini_eigenvalues"] for v in per_layer.values()])) if n_layers > 0 else 0.0
    )
    return {
        "total_kernel_dim": total_kernel_dim,
        "mean_spectral_gap": mean_spectral_gap,
        "mean_gini_eigenvalues": mean_gini,
        "per_layer": per_layer,
    }


# ── Main entry ─────────────────────────────────────────────────────────────

def run(model_path: str | None = None, **kwargs) -> None:
    """Run the full pipeline: crystal + persistence + sheaf on one weight load.

    Parameters
    ----------
    model_path:
        Path to the model checkpoint.  If *None*, extracted from piped stdin.
    """
    # ── 1. Resolve model path ──────────────────────────────────────────────
    piped = read_stdin_json()

    if model_path is None and piped is not None:
        model_path = piped.get("meta", {}).get("model")

    if model_path is None:
        summary("full: ERROR — no model path (use --model or pipe from previous stage)")
        sys.exit(1)

    # ── 2. Load weights ONCE ───────────────────────────────────────────────
    t0 = time.perf_counter()
    weights = extract_weights(model_path)

    # ── 3. Run all three measurements ─────────────────────────────────────
    crystal_result = _run_crystal(weights)
    persistence_result = _run_persistence(weights)
    sheaf_result = _run_sheaf(weights, top_n=_SHEAF_TOP_N)

    elapsed = time.perf_counter() - t0

    # ── 4. Build combined result ───────────────────────────────────────────
    result = {
        "crystal": crystal_result,
        "persistence": persistence_result,
        "sheaf": sheaf_result,
    }

    meta = {"model": model_path}

    # ── 5. Build envelope ──────────────────────────────────────────────────
    if piped is not None:
        envelope = merge_results(piped, "full", result, meta=meta)
    else:
        envelope = make_result("full", result, meta=meta)

    # ── 6. Emit JSON to stdout ─────────────────────────────────────────────
    emit(envelope)

    # ── 7. Human summary to stderr ─────────────────────────────────────────
    c = crystal_result["crystal"]
    n = crystal_result["n_weights"]
    n_str = f"{n / 1e6:.1f}M" if n >= 1e6 else str(n)
    p = persistence_result
    s = sheaf_result
    summary(
        f"full: crystal=({c['void']:.3f}/{c['identity']:.3f}/{c['prime']:.3f}) "
        f"n={n_str} | "
        f"persist: gini={p['h0_gini']:.4f} bars={p['n_bars']} | "
        f"sheaf: kernel={s['total_kernel_dim']} gap={s['mean_spectral_gap']:.4f} "
        f"({elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: crystal + persistence + sheaf")
    parser.add_argument("--model", dest="model_path", help="Path to model checkpoint")
    args = parser.parse_args()
    run(model_path=args.model_path)


if __name__ == "__main__":
    main()
