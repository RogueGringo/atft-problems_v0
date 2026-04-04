#!/usr/bin/env python3
"""Experiment 1b: Fine-grained topology of truth vs falsehood.

Same as Experiment 1 but with MORE points per cloud:
  - Split each reasoning step into overlapping windows of ~5 words
  - Each window → one embedding → one point in R^d
  - Gives 30-100+ points per chain instead of 3-7
  - H1 (loops/cycles) should emerge at this resolution

The question: does truthful reasoning have different H1 structure
than corrupted reasoning when we look at finer granularity?
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "results" / "experiment_1b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gsm8k(n_samples: int = 200):
    from datasets import load_dataset
    print(f"Loading GSM8K ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(min(n_samples, len(ds))))
    samples = []
    for item in ds:
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "final": item["answer"].split("####")[-1].strip() if "####" in item["answer"] else "",
        })
    print(f"  Loaded {len(samples)} reasoning chains")
    return samples


def corrupt_reasoning(samples, seed=42):
    rng = random.Random(seed)
    corrupted = []
    for s in samples:
        steps = [l.strip() for l in s["answer"].split("\n") if l.strip()]
        if len(steps) < 3:
            rng.shuffle(steps)
            corrupted.append({"question": s["question"], "answer": "\n".join(steps),
                              "final": s["final"], "corruption": "shuffle"})
            continue

        strategy = rng.choice(["shuffle", "swap_numbers", "cross_pollinate"])
        new_steps = list(steps)

        if strategy == "shuffle":
            middle = steps[1:-1]
            rng.shuffle(middle)
            new_steps = [steps[0]] + middle + [steps[-1]]
        elif strategy == "swap_numbers":
            import re
            num_steps = [(i, re.findall(r'\d+', s)) for i, s in enumerate(new_steps)]
            num_steps = [(i, n) for i, n in num_steps if n]
            if len(num_steps) >= 2:
                i1, i2 = rng.sample(range(len(num_steps)), 2)
                idx1, n1 = num_steps[i1]
                idx2, n2 = num_steps[i2]
                if n1 and n2:
                    new_steps[idx1] = new_steps[idx1].replace(n1[0], "TEMP", 1).replace("TEMP", n2[0], 1)
                    new_steps[idx2] = new_steps[idx2].replace(n2[0], n1[0], 1)
        elif strategy == "cross_pollinate":
            other = rng.choice(samples)
            other_steps = [l.strip() for l in other["answer"].split("\n") if l.strip()]
            if len(other_steps) > 2:
                inject_idx = rng.randint(1, len(steps) - 2)
                donor_idx = rng.randint(1, len(other_steps) - 2)
                new_steps[inject_idx] = other_steps[donor_idx]
            else:
                rng.shuffle(new_steps)

        corrupted.append({"question": s["question"], "answer": "\n".join(new_steps),
                          "final": s["final"], "corruption": strategy})

    print(f"  Corrupted {len(corrupted)} reasoning chains")
    return corrupted


def sliding_window_chunks(text: str, window_words: int = 5, stride_words: int = 3) -> list[str]:
    """Split text into overlapping word windows for fine-grained embedding."""
    words = text.split()
    if len(words) <= window_words:
        return [text]
    chunks = []
    for i in range(0, len(words) - window_words + 1, stride_words):
        chunk = " ".join(words[i:i + window_words])
        chunks.append(chunk)
    return chunks


def embed_fine_grained(samples, window_words=5, stride_words=3,
                       model_name="all-MiniLM-L6-v2"):
    """Embed each reasoning chain as a dense point cloud.

    Sliding window over the full text (question + answer) → many points per chain.
    """
    from sentence_transformers import SentenceTransformer
    print(f"  Loading encoder: {model_name}...")
    encoder = SentenceTransformer(model_name)

    point_clouds = []
    n_points_list = []

    for s in samples:
        full_text = s["question"] + " " + s["answer"]
        chunks = sliding_window_chunks(full_text, window_words, stride_words)
        embeddings = encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        point_clouds.append(embeddings)
        n_points_list.append(len(chunks))

    mean_pts = np.mean(n_points_list)
    print(f"  Embedded {len(point_clouds)} chains, "
          f"avg {mean_pts:.0f} points/chain (range {min(n_points_list)}-{max(n_points_list)}), "
          f"dim={point_clouds[0].shape[1]}")
    return point_clouds


def compute_persistence(point_cloud, max_dim=1):
    from ripser import ripser
    # Subsample if too many points (ripser is O(n^3))
    if len(point_cloud) > 150:
        idx = np.random.choice(len(point_cloud), 150, replace=False)
        point_cloud = point_cloud[idx]

    result = ripser(point_cloud, maxdim=max_dim)
    diagrams = result["dgms"]

    out = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
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


def gini(values):
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * sorted_v).sum() / (n * sorted_v.sum()))


def run_experiment(n_samples=200, window_words=5, stride_words=3):
    start = time.time()

    truth_samples = load_gsm8k(n_samples)
    false_samples = corrupt_reasoning(truth_samples)

    print("\nEmbedding truth chains (fine-grained)...")
    truth_clouds = embed_fine_grained(truth_samples, window_words, stride_words)
    print("Embedding false chains (fine-grained)...")
    false_clouds = embed_fine_grained(false_samples, window_words, stride_words)

    print("\nComputing persistence (H0 + H1)...")
    truth_stats = []
    false_stats = []

    for i, (tc, fc) in enumerate(zip(truth_clouds, false_clouds)):
        if i % 50 == 0:
            print(f"  {i}/{len(truth_clouds)}...")

        t_pers = compute_persistence(tc)
        f_pers = compute_persistence(fc)

        for label, pers, stats_list, extra in [
            ("truth", t_pers, truth_stats, {}),
            ("false", f_pers, false_stats, {"corruption": false_samples[i]["corruption"]}),
        ]:
            entry = {"idx": i, "n_points": len(tc if label == "truth" else fc), **extra}
            for dim in ["H0", "H1"]:
                p = pers[dim]
                entry[f"{dim}_n_bars"] = p["n_bars"]
                entry[f"{dim}_mean"] = p["mean_bar"]
                entry[f"{dim}_max"] = p["max_bar"]
                entry[f"{dim}_total"] = p["total_persistence"]
                entry[f"{dim}_gini"] = gini(p["bars"])
            stats_list.append(entry)

    elapsed = time.time() - start

    # Aggregate
    metrics = [
        "H0_n_bars", "H0_mean", "H0_max", "H0_total", "H0_gini",
        "H1_n_bars", "H1_mean", "H1_max", "H1_total", "H1_gini",
    ]

    print(f"\n{'='*78}")
    print(f"  EXPERIMENT 1b: Fine-Grained Topology of Truth vs. Falsehood")
    print(f"  {n_samples} GSM8K chains, window={window_words} words, stride={stride_words}")
    print(f"  avg points/cloud: truth={np.mean([s['n_points'] for s in truth_stats]):.0f}, "
          f"false={np.mean([s['n_points'] for s in false_stats]):.0f}")
    print(f"  {elapsed:.0f}s")
    print(f"{'='*78}")
    print(f"\n{'Metric':<20} {'Truth (mean±std)':<25} {'False (mean±std)':<25} {'Δ%':>8}")
    print("-" * 78)

    results = {}
    for m in metrics:
        t_vals = [s[m] for s in truth_stats]
        f_vals = [s[m] for s in false_stats]
        t_mean, t_std = np.mean(t_vals), np.std(t_vals)
        f_mean, f_std = np.mean(f_vals), np.std(f_vals)
        delta = ((f_mean - t_mean) / t_mean * 100) if t_mean != 0 else 0
        print(f"{m:<20} {t_mean:>8.4f} ± {t_std:<8.4f}   {f_mean:>8.4f} ± {f_std:<8.4f}   {delta:>+7.1f}%")
        results[m] = {"truth_mean": t_mean, "truth_std": t_std,
                       "false_mean": f_mean, "false_std": f_std, "delta_pct": delta}

    from scipy import stats as scipy_stats
    print(f"\nStatistical significance (Welch's t-test):")
    print(f"{'Metric':<20} {'t-stat':>10} {'p-value':>12} {'Significant?':>14}")
    print("-" * 58)

    sig_count = 0
    for m in metrics:
        t_vals = [s[m] for s in truth_stats]
        f_vals = [s[m] for s in false_stats]
        t_stat, p_val = scipy_stats.ttest_ind(t_vals, f_vals, equal_var=False)
        sig = "YES ***" if p_val < 0.001 else "YES **" if p_val < 0.01 else "YES *" if p_val < 0.05 else "no"
        if p_val < 0.05:
            sig_count += 1
        print(f"{m:<20} {t_stat:>10.3f} {p_val:>12.6f} {sig:>14}")
        results[m]["t_stat"] = float(t_stat)
        results[m]["p_value"] = float(p_val)

    print(f"\n  {sig_count}/{len(metrics)} metrics significantly different (p < 0.05)")
    print(f"{'='*78}")

    log = {
        "experiment": "topology_of_truth_fine_grained",
        "n_samples": n_samples,
        "window_words": window_words,
        "stride_words": stride_words,
        "elapsed": elapsed,
        "results": {k: {kk: float(vv) if isinstance(vv, np.floating) else vv
                        for kk, vv in v.items()} for k, v in results.items()},
    }
    with open(OUTPUT_DIR / "experiment_1b_results.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Saved → {OUTPUT_DIR / 'experiment_1b_results.json'}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    run_experiment(args.n_samples, args.window, args.stride)
