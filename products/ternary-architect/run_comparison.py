#!/usr/bin/env python3
"""Run the full {0,1,3} comparison experiment.

Trains 4 models (same architecture, different weight sets) and
runs the complete measurement battery on each.

Weight sets:
  013  — {0, 1, 3} (the thesis)
  n101 — {-1, 0, 1} (BitNet baseline)
  012  — {0, 1, 2} (ablation: decomposable, should underperform 013)
  fp16 — standard fp16 (hierarchical baseline)

Usage:
    python run_comparison.py                    # full comparison
    python run_comparison.py --quick            # 1000 steps each (fast test)
    python run_comparison.py --weight_sets 013 n101  # subset
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "results"
TRAIN_SCRIPT = str(Path(__file__).parent / "train.py")
MEASURE_SCRIPT = str(Path(__file__).parent / "measure.py")


def find_latest_run(weight_set: str) -> Path | None:
    """Find the most recent run for a weight set."""
    runs = sorted(OUTPUT_DIR.glob(f"{weight_set}_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def run_experiment(weight_set: str, args) -> Path | None:
    """Train one model and return the run directory."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {weight_set}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--weight_set", weight_set,
        "--size", args.size,
        "--max_steps", str(args.max_steps),
        "--n_samples", str(args.n_samples),
        "--batch_size", str(args.batch_size),
        "--ckpt_every", str(args.ckpt_every),
        "--log_every", str(args.log_every),
    ]

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        print(f"  FAILED: {weight_set} (exit code {result.returncode})")
        return None

    return find_latest_run(weight_set)


def run_analysis(run_dir: Path) -> dict | None:
    """Run full measurement battery on a trained model."""
    print(f"\n  Analyzing: {run_dir.name}")

    cmd = [sys.executable, MEASURE_SCRIPT, str(run_dir)]
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        print(f"  Analysis FAILED for {run_dir.name}")
        return None

    analysis_path = run_dir / "full_analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            return json.load(f)
    return None


def print_comparison(results: dict):
    """Print a summary comparison table."""
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*80}\n")

    # Header
    sets = list(results.keys())
    print(f"{'Metric':<30s}", end="")
    for ws in sets:
        print(f"  {ws:>12s}", end="")
    print()
    print("-" * (30 + 14 * len(sets)))

    # Training metrics
    metrics = [
        ("Final loss", lambda r: r.get("final_loss", "?")),
        ("Final ppl", lambda r: r.get("final_ppl", "?")),
    ]

    # Topology metrics
    topo_metrics = [
        ("Eff rank (mean)", lambda r: r.get("mean_eff_rank", "?")),
        ("Spectral gap (mean)", lambda r: r.get("mean_spectral_gap", "?")),
        ("Gini SV (mean)", lambda r: r.get("mean_gini_sv", "?")),
    ]

    # Zero topology
    zero_metrics = [
        ("Zero gini (trained)", lambda r: r.get("zero_gini_trained", "?")),
        ("Zero gini (random)", lambda r: r.get("zero_gini_random", "?")),
        ("Zero gini ratio", lambda r: r.get("zero_gini_ratio", "?")),
        ("Zero ablation deg%", lambda r: r.get("zero_ablation_deg", "?")),
    ]

    # Iterative inference
    iter_metrics = [
        ("Iter converged?", lambda r: r.get("iter_converged", "?")),
        ("Iter count", lambda r: r.get("iter_count", "?")),
    ]

    all_metrics = metrics + topo_metrics + zero_metrics + iter_metrics

    for label, fn in all_metrics:
        print(f"{label:<30s}", end="")
        for ws in sets:
            val = fn(results[ws])
            if isinstance(val, float):
                print(f"  {val:>12.3f}", end="")
            else:
                print(f"  {str(val):>12s}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_sets", nargs="+",
                        default=["013", "n101", "012", "fp16"])
    parser.add_argument("--size", default="small")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1000 steps, 10000 samples")

    args = parser.parse_args()

    if args.quick:
        args.max_steps = 1000
        args.n_samples = 10000
        args.ckpt_every = 200
        args.log_every = 25

    start = time.time()
    all_results = {}

    # Phase 1: Train all models
    run_dirs = {}
    for ws in args.weight_sets:
        run_dir = run_experiment(ws, args)
        if run_dir:
            run_dirs[ws] = run_dir

    # Phase 2: Full analysis on each
    print(f"\n\n{'='*60}")
    print(f"  RUNNING FULL MEASUREMENT BATTERY")
    print(f"{'='*60}")

    for ws, run_dir in run_dirs.items():
        analysis = run_analysis(run_dir)
        if analysis:
            # Extract summary metrics
            summary = {}

            # Training log
            log_path = run_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                if log.get("steps"):
                    summary["final_loss"] = log["steps"][-1].get("loss", "?")
                    summary["final_ppl"] = log["steps"][-1].get("ppl", "?")
                if log.get("final", {}).get("hidden_topology"):
                    ht = log["final"]["hidden_topology"]
                    summary["mean_eff_rank"] = ht.get("mean_eff_rank", "?")
                    summary["mean_spectral_gap"] = ht.get("mean_spectral_gap", "?")
                    summary["mean_gini_sv"] = ht.get("mean_gini_sv", "?")

            # Zero topology
            zt = analysis.get("zero_topology_summary", {})
            summary["zero_gini_trained"] = zt.get("mean_trained_gini", "N/A")
            summary["zero_gini_random"] = zt.get("mean_random_gini", "N/A")
            summary["zero_gini_ratio"] = zt.get("gini_ratio", "N/A")

            # Zero ablation
            za = analysis.get("zero_ablation", {})
            summary["zero_ablation_deg"] = za.get("degradation_pct", "N/A")

            # Iterative inference
            ii = analysis.get("iterative_inference", {})
            summary["iter_converged"] = ii.get("converged", "N/A")
            summary["iter_count"] = ii.get("n_iterations", "N/A")

            all_results[ws] = summary

    # Print comparison
    if all_results:
        print_comparison(all_results)

    # Save comparison
    elapsed = time.time() - start
    comparison = {
        "weight_sets": args.weight_sets,
        "size": args.size,
        "max_steps": args.max_steps,
        "elapsed_total": elapsed,
        "results": all_results,
        "run_dirs": {ws: str(d) for ws, d in run_dirs.items()},
    }

    out_path = OUTPUT_DIR / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
