"""
pipeline/crystal.py — Crystal weight distribution measurement.

Reads a model checkpoint and computes the global {0,1,3} crystal ratios
(void / identity / prime) plus per-layer breakdown.

Usage (standalone):
    python -m pipeline.crystal --model path/to/model.pt

Usage (piped):
    <previous_stage_output> | python -m pipeline.crystal
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    from ..prism.model import extract_weights, per_layer_stats, weight_stats
    from ..utils.io import emit, make_result, merge_results, read_stdin_json, summary
except ImportError:
    from prism.model import extract_weights, per_layer_stats, weight_stats  # type: ignore[no-redef]
    from utils.io import emit, make_result, merge_results, read_stdin_json, summary  # type: ignore[no-redef]


def run(model_path: str | None = None, **kwargs) -> None:
    """Run the crystal measurement stage.

    Parameters
    ----------
    model_path:
        Path to the model checkpoint.  If *None*, the path is extracted from
        piped stdin JSON (``previous.meta.model``).
    """
    # ── 1. Resolve model path ──────────────────────────────────────────────
    piped = read_stdin_json()

    if model_path is None and piped is not None:
        model_path = piped.get("meta", {}).get("model")

    if model_path is None:
        summary("crystal: ERROR — no model path (use --model or pipe from previous stage)")
        sys.exit(1)

    # ── 2. Load weights ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    weights = extract_weights(model_path)
    elapsed = time.perf_counter() - t0

    # ── 3. Compute measurements ────────────────────────────────────────────
    stats = weight_stats(weights)
    layers = per_layer_stats(weights)

    # ── 4. Build result dict ───────────────────────────────────────────────
    result = {
        "crystal": {
            "void": stats["void"],
            "identity": stats["identity"],
            "prime": stats["prime"],
        },
        "n_weights": stats["n_weights"],
        "per_layer": layers,
    }

    meta = {"model": model_path}

    # ── 5. Build envelope ──────────────────────────────────────────────────
    if piped is not None:
        envelope = merge_results(piped, "crystal", result, meta=meta)
    else:
        envelope = make_result("crystal", result, meta=meta)

    # ── 6. Emit JSON to stdout ─────────────────────────────────────────────
    emit(envelope)

    # ── 7. Human summary to stderr ─────────────────────────────────────────
    n = stats["n_weights"]
    n_str = f"{n / 1e6:.1f}M" if n >= 1e6 else str(n)
    summary(
        f"crystal: void={stats['void']:.3f} identity={stats['identity']:.3f}"
        f" prime={stats['prime']:.3f} ({n_str} weights, {elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure crystal weight distribution")
    parser.add_argument("--model", dest="model_path", help="Path to model checkpoint")
    args = parser.parse_args()
    run(model_path=args.model_path)


if __name__ == "__main__":
    main()
