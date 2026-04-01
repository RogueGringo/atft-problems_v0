"""
pipeline/persistence.py — H₀ persistence + Gini measurement.

For each layer's weight matrix, computes H₀ persistence bars (birth-death
pairs via union-find on pairwise distances).  Samples rows when a matrix
exceeds 200 rows to keep computation tractable.  Reports the Gini
coefficient of all bar lengths — a measure of topological heterogeneity.

Usage (standalone):
    python -m pipeline.persistence --model path/to/model.pt

Usage (piped):
    <previous_stage_output> | python -m pipeline.persistence
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Add topological-router to sys.path ────────────────────────────────────
_TOPO_DIR = str(Path(__file__).resolve().parents[2] / "topological-router")
if _TOPO_DIR not in sys.path:
    sys.path.insert(0, _TOPO_DIR)

from topo_measures import gini_fast, h0_persistence  # noqa: E402

try:
    from ..prism.model import extract_weights
    from ..utils.io import emit, make_result, merge_results, read_stdin_json, summary
except ImportError:
    from prism.model import extract_weights  # type: ignore[no-redef]
    from utils.io import emit, make_result, merge_results, read_stdin_json, summary  # type: ignore[no-redef]

# Row-sample cap for H₀ computation
_MAX_ROWS = 200


def _layer_persistence(t) -> dict | None:
    """Compute H₀ persistence stats for a single weight tensor.

    Returns None for 1D tensors (e.g. LayerNorm weight/bias).
    """
    if t.dim() < 2:
        return None

    W = t.numpy()  # shape (out, in)

    # Sample rows if needed
    rng = np.random.RandomState(42)
    if W.shape[0] > _MAX_ROWS:
        idx = rng.choice(W.shape[0], _MAX_ROWS, replace=False)
        W = W[idx]

    bars = h0_persistence(W)  # bar lengths array

    n_bars = len(bars)
    h0_gini = float(gini_fast(bars)) if n_bars > 0 else 0.0
    max_bar = float(bars.max()) if n_bars > 0 else 0.0
    mean_bar = float(bars.mean()) if n_bars > 0 else 0.0

    return {
        "n_bars": n_bars,
        "h0_gini": h0_gini,
        "max_bar": max_bar,
        "mean_bar": mean_bar,
    }


def run(model_path: str | None = None, weights: dict | None = None, **kwargs) -> None:
    """Run the persistence measurement stage.

    Parameters
    ----------
    model_path:
        Path to the model checkpoint.  If *None*, extracted from piped stdin.
    weights:
        Pre-loaded weight dict (e.g. from ``extract_weights``).  If provided,
        ``model_path`` is still recorded in metadata but the checkpoint is not
        re-read.
    """
    # ── 1. Resolve model path ──────────────────────────────────────────────
    piped = read_stdin_json()

    if model_path is None and piped is not None:
        model_path = piped.get("meta", {}).get("model")

    if model_path is None and weights is None:
        summary("persistence: ERROR — no model path (use --model or pipe from previous stage)")
        sys.exit(1)

    # ── 2. Load weights ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if weights is None:
        weights = extract_weights(model_path)

    # ── 3. Compute per-layer persistence ──────────────────────────────────
    per_layer: dict[str, dict] = {}

    for name, tensor in weights.items():
        layer_stats = _layer_persistence(tensor)
        if layer_stats is None:
            continue  # skip 1D tensors
        per_layer[name] = layer_stats

    elapsed = time.perf_counter() - t0

    # Aggregate global stats: collect n_bars and gini across layers
    total_bars = sum(v["n_bars"] for v in per_layer.values())
    # Weighted mean gini (weighted by n_bars)
    if total_bars > 0:
        global_gini = float(
            sum(v["h0_gini"] * v["n_bars"] for v in per_layer.values()) / total_bars
        )
    else:
        global_gini = 0.0

    # ── 4. Build result dict ───────────────────────────────────────────────
    result = {
        "h0_gini": global_gini,
        "n_bars": total_bars,
        "per_layer": per_layer,
    }

    meta = {"model": model_path} if model_path else {}

    # ── 5. Build envelope ──────────────────────────────────────────────────
    if piped is not None:
        envelope = merge_results(piped, "persistence", result, meta=meta)
    else:
        envelope = make_result("persistence", result, meta=meta)

    # ── 6. Emit JSON to stdout ─────────────────────────────────────────────
    emit(envelope)

    # ── 7. Human summary to stderr ─────────────────────────────────────────
    summary(
        f"persistence: h0_gini={global_gini:.4f} n_bars={total_bars}"
        f" layers={len(per_layer)} ({elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure H₀ persistence + Gini")
    parser.add_argument("--model", dest="model_path", help="Path to model checkpoint")
    args = parser.parse_args()
    run(model_path=args.model_path)


if __name__ == "__main__":
    main()
