#!/usr/bin/env python3
"""Experiment 1: The Topology of Truth vs. Falsehood

Does truthful reasoning have a different topological signature than
false/hallucinated reasoning?

Pipeline:
  1. Load GSM8K (math reasoning) from HuggingFace
  2. Take correct solutions as "truth" set
  3. Generate corrupted versions as "falsehood" set (shuffle reasoning steps)
  4. Embed both sets via frozen sentence-transformer → point clouds in R^d
  5. Compute H0 and H1 persistence on both point clouds
  6. Compute Gini curves on persistence bars
  7. Compare: do truth and falsehood have distinguishable topological signatures?

If yes: the foundation holds — topology differentiates semantic coherence.
If no: the theory needs revision before building any architecture on it.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

OUTPUT_DIR = Path(__file__).parent / "results" / "experiment_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load and prepare data ────────────────────────────────────────────────

def load_gsm8k(n_samples: int = 200):
    """Load GSM8K math reasoning dataset.

    Returns list of dicts with 'question', 'answer' (full chain), 'final' (number).
    """
    from datasets import load_dataset
    print(f"Loading GSM8K ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for item in ds:
        question = item["question"]
        answer = item["answer"]  # multi-step reasoning chain
        # Extract final numerical answer (after ####)
        final = answer.split("####")[-1].strip() if "####" in answer else ""
        samples.append({
            "question": question,
            "answer": answer,
            "final": final,
        })

    print(f"  Loaded {len(samples)} reasoning chains")
    return samples


def corrupt_reasoning(samples: list[dict], seed: int = 42) -> list[dict]:
    """Create corrupted versions of reasoning chains.

    Corruption strategies (applied randomly):
      1. Shuffle reasoning steps (logical order broken)
      2. Swap numbers between steps (arithmetic corrupted)
      3. Replace a step with one from a different problem (context break)

    The corruption preserves surface-level plausibility (looks like math)
    but breaks the logical/semantic coherence.
    """
    rng = random.Random(seed)
    corrupted = []

    for s in samples:
        answer = s["answer"]
        # Split into reasoning steps (lines or sentences)
        steps = [line.strip() for line in answer.split("\n") if line.strip()]

        if len(steps) < 3:
            # Too short to corrupt meaningfully — shuffle what we have
            rng.shuffle(steps)
            corrupted.append({
                "question": s["question"],
                "answer": "\n".join(steps),
                "final": s["final"],
                "corruption": "shuffle",
            })
            continue

        strategy = rng.choice(["shuffle", "swap_numbers", "cross_pollinate"])

        if strategy == "shuffle":
            # Keep first and last step, shuffle middle
            middle = steps[1:-1]
            rng.shuffle(middle)
            new_steps = [steps[0]] + middle + [steps[-1]]

        elif strategy == "swap_numbers":
            # Find steps with numbers and swap them
            import re
            new_steps = list(steps)
            num_steps = [(i, re.findall(r'\d+', s)) for i, s in enumerate(new_steps)]
            num_steps = [(i, nums) for i, nums in num_steps if nums]
            if len(num_steps) >= 2:
                i1, i2 = rng.sample(range(len(num_steps)), 2)
                idx1, nums1 = num_steps[i1]
                idx2, nums2 = num_steps[i2]
                # Swap first number in each
                if nums1 and nums2:
                    new_steps[idx1] = new_steps[idx1].replace(nums1[0], "TEMP", 1)
                    new_steps[idx1] = new_steps[idx1].replace("TEMP", nums2[0], 1)
                    new_steps[idx2] = new_steps[idx2].replace(nums2[0], nums1[0], 1)

        elif strategy == "cross_pollinate":
            # Replace a middle step with one from a different problem
            other = rng.choice(samples)
            other_steps = [l.strip() for l in other["answer"].split("\n") if l.strip()]
            if len(other_steps) > 2:
                inject_idx = rng.randint(1, len(steps) - 2)
                donor_idx = rng.randint(1, len(other_steps) - 2)
                new_steps = list(steps)
                new_steps[inject_idx] = other_steps[donor_idx]
            else:
                new_steps = list(steps)
                rng.shuffle(new_steps)

        corrupted.append({
            "question": s["question"],
            "answer": "\n".join(new_steps),
            "final": s["final"],
            "corruption": strategy,
        })

    print(f"  Corrupted {len(corrupted)} reasoning chains")
    return corrupted


# ── 2. Embed to point clouds ────────────────────────────────────────────────

def embed_reasoning_chains(samples: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    """Embed each reasoning chain as a point cloud.

    Each step in the chain → one point in R^d.
    Each chain → one point cloud.

    Returns list of (n_steps, d) numpy arrays.
    """
    from sentence_transformers import SentenceTransformer
    print(f"  Loading encoder: {model_name}...")
    encoder = SentenceTransformer(model_name)

    point_clouds = []
    for s in samples:
        # Split chain into steps
        steps = [line.strip() for line in s["answer"].split("\n") if line.strip()]
        if len(steps) < 2:
            steps = [s["question"]] + steps  # prepend question if too short

        # Embed each step → point in R^d
        embeddings = encoder.encode(steps, convert_to_numpy=True)
        point_clouds.append(embeddings)

    print(f"  Embedded {len(point_clouds)} chains, dim={point_clouds[0].shape[1]}")
    return point_clouds


# ── 3. Compute persistence ──────────────────────────────────────────────────

def compute_persistence(point_cloud: np.ndarray, max_dim: int = 1) -> dict:
    """Compute persistent homology on a point cloud.

    Returns dict with H0 and H1 bars, birth/death pairs.
    """
    from ripser import ripser

    result = ripser(point_cloud, maxdim=max_dim)
    diagrams = result["dgms"]

    out = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        # Filter out infinite bars for statistics
        finite = dgm[dgm[:, 1] < np.inf] if len(dgm) > 0 else np.array([]).reshape(0, 2)
        bars = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        out[f"H{dim}"] = {
            "n_bars": len(bars),
            "bars": bars,
            "mean_bar": float(bars.mean()) if len(bars) > 0 else 0.0,
            "max_bar": float(bars.max()) if len(bars) > 0 else 0.0,
            "total_persistence": float(bars.sum()) if len(bars) > 0 else 0.0,
        }

    return out


def gini(values: np.ndarray) -> float:
    """Gini coefficient of an array of values."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * sorted_v).sum() / (n * sorted_v.sum()))


# ── 4. Run experiment ────────────────────────────────────────────────────────

def run_experiment(n_samples: int = 200):
    """Full Experiment 1 pipeline."""
    start = time.time()

    # Load data
    truth_samples = load_gsm8k(n_samples)
    false_samples = corrupt_reasoning(truth_samples)

    # Embed
    print("\nEmbedding truth chains...")
    truth_clouds = embed_reasoning_chains(truth_samples)
    print("Embedding false chains...")
    false_clouds = embed_reasoning_chains(false_samples)

    # Compute persistence for each chain
    print("\nComputing persistence...")
    truth_stats = []
    false_stats = []

    for i, (tc, fc) in enumerate(zip(truth_clouds, false_clouds)):
        if i % 50 == 0:
            print(f"  {i}/{len(truth_clouds)}...")

        t_pers = compute_persistence(tc)
        f_pers = compute_persistence(fc)

        # Gini of H0 and H1 bars
        t_gini_h0 = gini(t_pers["H0"]["bars"])
        t_gini_h1 = gini(t_pers["H1"]["bars"])
        f_gini_h0 = gini(f_pers["H0"]["bars"])
        f_gini_h1 = gini(f_pers["H1"]["bars"])

        truth_stats.append({
            "idx": i,
            "n_steps": len(tc),
            "H0_n_bars": t_pers["H0"]["n_bars"],
            "H0_mean": t_pers["H0"]["mean_bar"],
            "H0_max": t_pers["H0"]["max_bar"],
            "H0_total": t_pers["H0"]["total_persistence"],
            "H0_gini": t_gini_h0,
            "H1_n_bars": t_pers["H1"]["n_bars"],
            "H1_mean": t_pers["H1"]["mean_bar"],
            "H1_max": t_pers["H1"]["max_bar"],
            "H1_total": t_pers["H1"]["total_persistence"],
            "H1_gini": t_gini_h1,
        })
        false_stats.append({
            "idx": i,
            "n_steps": len(fc),
            "corruption": false_samples[i]["corruption"],
            "H0_n_bars": f_pers["H0"]["n_bars"],
            "H0_mean": f_pers["H0"]["mean_bar"],
            "H0_max": f_pers["H0"]["max_bar"],
            "H0_total": f_pers["H0"]["total_persistence"],
            "H0_gini": f_gini_h0,
            "H1_n_bars": f_pers["H1"]["n_bars"],
            "H1_mean": f_pers["H1"]["mean_bar"],
            "H1_max": f_pers["H1"]["max_bar"],
            "H1_total": f_pers["H1"]["total_persistence"],
            "H1_gini": f_gini_h1,
        })

    elapsed = time.time() - start

    # ── Aggregate comparison ──────────────────────────────────────────
    def agg(stats, key):
        vals = [s[key] for s in stats]
        return {"mean": np.mean(vals), "std": np.std(vals), "median": np.median(vals)}

    metrics = [
        "H0_n_bars", "H0_mean", "H0_max", "H0_total", "H0_gini",
        "H1_n_bars", "H1_mean", "H1_max", "H1_total", "H1_gini",
    ]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: Topology of Truth vs. Falsehood")
    print(f"  {n_samples} GSM8K reasoning chains, {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'Truth (mean±std)':<25} {'False (mean±std)':<25} {'Δ%':>8}")
    print("-" * 78)

    results = {}
    for m in metrics:
        t = agg(truth_stats, m)
        f = agg(false_stats, m)
        delta_pct = ((f["mean"] - t["mean"]) / t["mean"] * 100) if t["mean"] != 0 else 0
        print(f"{m:<20} {t['mean']:>8.4f} ± {t['std']:<8.4f}   {f['mean']:>8.4f} ± {f['std']:<8.4f}   {delta_pct:>+7.1f}%")
        results[m] = {"truth": t, "false": f, "delta_pct": delta_pct}

    # Statistical significance (Welch's t-test)
    from scipy import stats as scipy_stats
    print(f"\nStatistical significance (Welch's t-test, p < 0.05):")
    print(f"{'Metric':<20} {'t-stat':>10} {'p-value':>12} {'Significant?':>14}")
    print("-" * 58)
    for m in metrics:
        t_vals = [s[m] for s in truth_stats]
        f_vals = [s[m] for s in false_stats]
        t_stat, p_val = scipy_stats.ttest_ind(t_vals, f_vals, equal_var=False)
        sig = "YES ***" if p_val < 0.001 else "YES *" if p_val < 0.05 else "no"
        print(f"{m:<20} {t_stat:>10.3f} {p_val:>12.6f} {sig:>14}")
        results[m]["t_stat"] = float(t_stat)
        results[m]["p_value"] = float(p_val)

    print(f"\n{'='*70}")

    # Save
    log = {
        "experiment": "topology_of_truth",
        "n_samples": n_samples,
        "elapsed": elapsed,
        "results": results,
        "truth_stats": truth_stats,
        "false_stats": false_stats,
    }
    with open(OUTPUT_DIR / "experiment_1_results.json", "w") as f:
        json.dump(log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"Saved → {OUTPUT_DIR / 'experiment_1_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    run_experiment(args.n_samples)
