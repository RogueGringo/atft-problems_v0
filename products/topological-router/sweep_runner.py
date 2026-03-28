"""Module 4: Cross-Architecture Sweep Runner

Runs Modules 1-3 on all 6 models in the evaluation suite and produces a
cross-architecture comparison summary.  Models are loaded/unloaded one at a
time; exceptions are caught per-model so a single failure does not abort the
full sweep.
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prime_basis import run_module1          # noqa: E402
from adaptive_explorer import run_module2    # noqa: E402
from topology_comparison import run_module3  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "HuggingFaceTB/SmolLM2-360M-Instruct", "family": "SmolLM2", "size": 0.36},
    {"name": "Qwen/Qwen2.5-0.5B",                   "family": "Qwen2.5", "size": 0.5},
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "family": "TinyLlama", "size": 1.1},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct",          "family": "Qwen2.5", "size": 1.5},
    {"name": "Qwen/Qwen2.5-3B-Instruct",            "family": "Qwen2.5", "size": 3.0},
    {"name": "Qwen/Qwen2.5-7B-Instruct-AWQ",        "family": "Qwen2.5", "size": 7.0},
]

RESULTS_DIR = Path(__file__).parent / "results"
BASIS_DIR = RESULTS_DIR / "basis_discovery"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_short(model_name: str) -> str:
    """Canonical short name used by Modules 1-3 for file naming."""
    return model_name.replace("/", "_").replace("\\", "_")


def _find_basis_file(model_name: str) -> Path | None:
    """Locate the prime_basis .pt file saved by Module 1.

    Tries the canonical name first, then falls back to a glob.
    """
    short = _model_short(model_name)
    canonical = BASIS_DIR / f"prime_basis_{short}.pt"
    if canonical.exists():
        return canonical

    # Fallback: glob for anything matching
    tail = model_name.split("/")[-1]
    candidates = list(BASIS_DIR.glob(f"prime_basis_*{tail}*.pt"))
    if candidates:
        return candidates[0]
    return None


def _safe_get(d: dict, *keys, default=None):
    """Nested dict access that never throws."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def _force_cleanup():
    """Aggressive memory cleanup between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------

def _extract_summary(
    model_entry: dict,
    m1_result: dict | None,
    m2_result: dict | None,
    m3_result: dict | None,
) -> dict:
    """Pull the key stats from each module's output into a flat summary row."""
    summary = {
        "model": model_entry["name"],
        "family": model_entry["family"],
        "size_B": model_entry["size"],
    }

    # Module 1 stats
    if m1_result is not None:
        summary["prime_ratio_mean"] = _safe_get(m1_result, "mean", default=None)
        summary["prime_ratio_std"] = _safe_get(m1_result, "std", default=None)
        summary["prime_ratio_median"] = _safe_get(m1_result, "median", default=None)
        summary["hidden_dim"] = _safe_get(m1_result, "hidden_dim", default=None)
        summary["target_layer"] = _safe_get(m1_result, "target_layer", default=None)
    else:
        summary.update({
            "prime_ratio_mean": None, "prime_ratio_std": None,
            "prime_ratio_median": None, "hidden_dim": None,
            "target_layer": None,
        })

    # Module 2 stats
    if m2_result is not None:
        summary["adaptive_basis_size"] = _safe_get(m2_result, "final_basis_size", default=None)
        summary["converged"] = _safe_get(m2_result, "converged", default=None)
        summary["convergence_gini"] = _safe_get(m2_result, "final_gini", default=None)
        summary["n_iterations"] = len(m2_result.get("basis_growth_log", []))
    else:
        summary.update({
            "adaptive_basis_size": None, "converged": None,
            "convergence_gini": None, "n_iterations": None,
        })

    # Module 3 stats
    if m3_result is not None:
        classification = _safe_get(m3_result, "classification", default={})
        m1_topo = _safe_get(m3_result, "metric1_topological_isomorphism", default={})
        m2_coh = _safe_get(m3_result, "metric2_cross_subspace_coherence", default={})
        summary["outcome_cell"] = _safe_get(classification, "cell", default=None)
        summary["gini_prime"] = _safe_get(m1_topo, "gini_prime", default=None)
        summary["gini_residual"] = _safe_get(m1_topo, "gini_residual", default=None)
        summary["isomorphic"] = _safe_get(m1_topo, "isomorphic", default=None)
        summary["coherence"] = _safe_get(m2_coh, "normalized_spectral_sum", default=None)
        summary["high_coherence"] = _safe_get(m2_coh, "high_coherence", default=None)
    else:
        summary.update({
            "outcome_cell": None, "gini_prime": None, "gini_residual": None,
            "isomorphic": None, "coherence": None, "high_coherence": None,
        })

    return summary


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_summary_table(summaries: list[dict]) -> None:
    """Print a formatted text table of results."""
    # Header
    hdr = (
        f"{'Model':<45s} {'Size':>5s} {'PrimeR':>7s} {'BasisSz':>8s} "
        f"{'Conv?':>5s} {'Gini':>6s} {'Cell':>20s} {'GiniP':>6s} "
        f"{'GiniR':>6s} {'Coh':>7s}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print("  CROSS-ARCHITECTURE SWEEP SUMMARY")
    print(sep)
    print(hdr)
    print(sep)

    for s in summaries:
        short_name = s["model"].split("/")[-1][:44]
        size = f"{s['size_B']:.1f}B"
        pr = f"{s['prime_ratio_mean']:.4f}" if s.get("prime_ratio_mean") is not None else "  ---"
        bs = f"{s['adaptive_basis_size']}" if s.get("adaptive_basis_size") is not None else " ---"
        cv = "Y" if s.get("converged") else ("N" if s.get("converged") is not None else "-")
        gi = f"{s['convergence_gini']:.4f}" if s.get("convergence_gini") is not None else "  ---"
        cell = s.get("outcome_cell") or "---"
        gp = f"{s['gini_prime']:.4f}" if s.get("gini_prime") is not None else "  ---"
        gr = f"{s['gini_residual']:.4f}" if s.get("gini_residual") is not None else "  ---"
        co = f"{s['coherence']:.4f}" if s.get("coherence") is not None else "  ---"

        print(
            f"{short_name:<45s} {size:>5s} {pr:>7s} {bs:>8s} "
            f"{cv:>5s} {gi:>6s} {cell:>20s} {gp:>6s} {gr:>6s} {co:>7s}"
        )

    print(sep)


def _print_family_averages(summaries: list[dict]) -> None:
    """Print per-family average statistics."""
    families: dict[str, list[dict]] = {}
    for s in summaries:
        fam = s["family"]
        families.setdefault(fam, []).append(s)

    print("\n  BY-FAMILY AVERAGES")
    print(f"  {'Family':<15s} {'#Models':>7s} {'AvgPrimeR':>10s} {'AvgBasis':>9s} "
          f"{'AvgGini':>8s} {'AvgCoh':>8s}")
    print("  " + "-" * 60)

    for fam, rows in sorted(families.items()):
        n = len(rows)
        avg_pr = _avg([r.get("prime_ratio_mean") for r in rows])
        avg_bs = _avg([r.get("adaptive_basis_size") for r in rows])
        avg_gi = _avg([r.get("convergence_gini") for r in rows])
        avg_co = _avg([r.get("coherence") for r in rows])
        print(
            f"  {fam:<15s} {n:>7d} "
            f"{_fmt(avg_pr, '.4f'):>10s} {_fmt(avg_bs, '.0f'):>9s} "
            f"{_fmt(avg_gi, '.4f'):>8s} {_fmt(avg_co, '.4f'):>8s}"
        )
    print()


def _avg(vals: list) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def _fmt(val, spec: str) -> str:
    if val is None:
        return "---"
    return f"{val:{spec}}"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(device: str = "cuda") -> list[dict]:
    """Run Modules 1-3 on every model and produce a cross-architecture summary.

    Returns
    -------
    list[dict] — one summary dict per model
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("[sweep] CUDA not available, falling back to CPU")
        device = "cpu"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASIS_DIR.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    summaries: list[dict] = []
    errors: list[dict] = []

    sweep_t0 = time.time()

    for idx, model_entry in enumerate(MODELS, 1):
        model_name = model_entry["name"]
        banner = f"[{idx}/{n_models}] {model_name} ({model_entry['size']}B)"
        print(f"\n{'=' * 70}")
        print(f"  {banner}")
        print(f"{'=' * 70}")

        m1_result = None
        m2_result = None
        m3_result = None
        model_t0 = time.time()

        # ---- Module 1: Prime Basis Seed ----
        try:
            print(f"\n  --- Module 1: Prime Basis ---")
            m1_result = run_module1(model_name, device=device)
            print(f"  Module 1 OK: prime_ratio_mean={m1_result.get('mean', '?')}")
        except Exception as e:
            msg = f"Module 1 FAILED for {model_name}: {e}"
            print(f"  *** {msg}")
            traceback.print_exc()
            errors.append({"model": model_name, "module": 1, "error": str(e)})
        finally:
            _force_cleanup()

        # ---- Find the prime basis .pt file ----
        basis_path = _find_basis_file(model_name)
        if basis_path is None:
            msg = f"No prime_basis .pt found for {model_name} — skipping Modules 2-3"
            print(f"  *** {msg}")
            errors.append({"model": model_name, "module": 2, "error": msg})
            summaries.append(_extract_summary(model_entry, m1_result, None, None))
            continue

        print(f"  Prime basis file: {basis_path.name}")

        # ---- Module 2: Adaptive Explorer ----
        try:
            print(f"\n  --- Module 2: Adaptive Explorer ---")
            m2_result = run_module2(model_name, basis_path, device=device)
            print(f"  Module 2 OK: basis_size={m2_result.get('final_basis_size', '?')}, "
                  f"converged={m2_result.get('converged', '?')}")
        except Exception as e:
            msg = f"Module 2 FAILED for {model_name}: {e}"
            print(f"  *** {msg}")
            traceback.print_exc()
            errors.append({"model": model_name, "module": 2, "error": str(e)})
        finally:
            _force_cleanup()

        # ---- Module 3: Topology Comparison ----
        try:
            print(f"\n  --- Module 3: Topology Comparison ---")
            m3_result = run_module3(model_name, device=device)
            cell = _safe_get(m3_result, "classification", "cell", default="?")
            print(f"  Module 3 OK: outcome_cell={cell}")
        except Exception as e:
            msg = f"Module 3 FAILED for {model_name}: {e}"
            print(f"  *** {msg}")
            traceback.print_exc()
            errors.append({"model": model_name, "module": 3, "error": str(e)})
        finally:
            _force_cleanup()

        # ---- Collect summary ----
        summary = _extract_summary(model_entry, m1_result, m2_result, m3_result)
        summary["elapsed_s"] = round(time.time() - model_t0, 1)
        summaries.append(summary)
        print(f"\n  {banner} completed in {summary['elapsed_s']:.0f}s")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    sweep_elapsed = time.time() - sweep_t0

    _print_summary_table(summaries)
    _print_family_averages(summaries)

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e['model']} module {e['module']}: {e['error'][:80]}")
        print()

    print(f"  Total sweep time: {sweep_elapsed / 60:.1f} minutes")
    print(f"  Models completed: {len(summaries)}/{n_models}")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    out_path = RESULTS_DIR / "sweep_summary.json"
    save_data = {
        "sweep_time_s": round(sweep_elapsed, 1),
        "n_models": n_models,
        "n_errors": len(errors),
        "device": device,
        "summaries": summaries,
        "errors": errors,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {out_path}")

    return summaries


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_sweep()
