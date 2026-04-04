#!/usr/bin/env python3
"""Experiment 1c: Decoherence Spectrum — isolating what breaks topology.

Instead of one corruption, test a SPECTRUM of decoherence:
  Level 0: Original (truth) — baseline
  Level 1: Shuffle middle steps (mild — logic order broken)
  Level 2: Replace numbers with random numbers (arithmetic broken)
  Level 3: Replace steps with random steps from other problems (context broken)
  Level 4: Replace steps with grammatically valid nonsense (semantic broken)
  Level 5: Full random word salad (complete decoherence)

Each level isolates a different structural property:
  L1 tests: does ORDER matter topologically?
  L2 tests: does NUMERICAL CONSISTENCY matter?
  L3 tests: does CONTEXTUAL COHERENCE matter?
  L4 tests: does SEMANTIC MEANING matter?
  L5 tests: does GRAMMATICAL STRUCTURE matter?

If topology tracks decoherence monotonically (more corruption → bigger
topological shift), we have a GRADED instrument, not just a binary classifier.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "results" / "experiment_1c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gsm8k(n_samples=200):
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
    print(f"  Loaded {len(samples)} chains")
    return samples


# ── Corruption levels ────────────────────────────────────────────────────────

def corrupt_L1_shuffle(sample, rng):
    """L1: Shuffle middle reasoning steps. Logic order broken."""
    steps = [l.strip() for l in sample["answer"].split("\n") if l.strip()]
    if len(steps) > 2:
        middle = steps[1:-1]
        rng.shuffle(middle)
        steps = [steps[0]] + middle + [steps[-1]]
    return "\n".join(steps)


def corrupt_L2_numbers(sample, rng):
    """L2: Replace all numbers with random ones. Arithmetic broken."""
    text = sample["answer"]
    def replace_num(match):
        orig = match.group(0)
        magnitude = len(orig)
        return str(rng.randint(1, 10**magnitude - 1))
    return re.sub(r'\d+', replace_num, text)


def corrupt_L3_cross(sample, all_samples, rng):
    """L3: Replace each step with a step from a random other problem. Context broken."""
    steps = [l.strip() for l in sample["answer"].split("\n") if l.strip()]
    new_steps = []
    for i, step in enumerate(steps):
        if i == 0:
            new_steps.append(step)  # keep first step
        else:
            donor = rng.choice(all_samples)
            donor_steps = [l.strip() for l in donor["answer"].split("\n") if l.strip()]
            if donor_steps:
                new_steps.append(rng.choice(donor_steps))
            else:
                new_steps.append(step)
    return "\n".join(new_steps)


def corrupt_L4_semantic(sample, rng):
    """L4: Replace steps with grammatically valid but semantically nonsensical text."""
    templates = [
        "The {noun} {verb} approximately {num} {unit} per {time}.",
        "Therefore, the total {noun} is {num} {unit}.",
        "Since each {noun} has {num} {unit}, we can calculate the {noun2}.",
        "We know that {num} {unit} divided by {num2} gives us {num3}.",
        "Adding the {noun} from both sides, we get {num} {unit}.",
        "The remaining {noun} after {verb}ing is {num} {unit}.",
    ]
    nouns = ["elephant", "trajectory", "sandwich", "molecule", "planet", "frequency"]
    verbs = ["oscillates", "consumes", "rotates", "dissolves", "amplifies"]
    units = ["meters", "gallons", "photons", "dollars", "radians", "hertz"]
    times = ["second", "century", "orbit", "heartbeat"]

    steps = [l.strip() for l in sample["answer"].split("\n") if l.strip()]
    new_steps = []
    for step in steps:
        t = rng.choice(templates)
        t = t.replace("{noun}", rng.choice(nouns))
        t = t.replace("{noun2}", rng.choice(nouns))
        t = t.replace("{verb}", rng.choice(verbs))
        t = t.replace("{unit}", rng.choice(units))
        t = t.replace("{time}", rng.choice(times))
        t = t.replace("{num}", str(rng.randint(1, 999)))
        t = t.replace("{num2}", str(rng.randint(1, 99)))
        t = t.replace("{num3}", str(rng.randint(1, 9999)))
        new_steps.append(t)
    return "\n".join(new_steps)


def corrupt_L5_random(sample, rng):
    """L5: Pure random word salad. Complete decoherence."""
    words = [
        "the", "of", "and", "to", "in", "that", "is", "for", "it", "with",
        "as", "was", "on", "are", "be", "has", "from", "or", "an", "but",
        "not", "by", "this", "they", "which", "had", "at", "one", "have",
        "banana", "quantum", "purple", "seventeen", "giraffe", "calculus",
        "therefore", "because", "multiply", "subtract", "total", "each",
        "hamburger", "neutron", "velocity", "triangle", "running", "above",
    ]
    steps = [l.strip() for l in sample["answer"].split("\n") if l.strip()]
    new_steps = []
    for step in steps:
        n_words = len(step.split())
        salad = " ".join(rng.choice(words) for _ in range(n_words))
        new_steps.append(salad)
    return "\n".join(new_steps)


def apply_corruption(samples, level, seed=42):
    rng = random.Random(seed)
    corrupted = []
    for s in samples:
        if level == 0:
            text = s["answer"]
        elif level == 1:
            text = corrupt_L1_shuffle(s, rng)
        elif level == 2:
            text = corrupt_L2_numbers(s, rng)
        elif level == 3:
            text = corrupt_L3_cross(s, samples, rng)
        elif level == 4:
            text = corrupt_L4_semantic(s, rng)
        elif level == 5:
            text = corrupt_L5_random(s, rng)
        else:
            text = s["answer"]

        corrupted.append({
            "question": s["question"],
            "answer": text,
            "level": level,
        })
    return corrupted


# ── Embedding and persistence ────────────────────────────────────────────────

def sliding_window_chunks(text, window_words=5, stride_words=3):
    words = text.split()
    if len(words) <= window_words:
        return [text]
    return [" ".join(words[i:i+window_words])
            for i in range(0, len(words) - window_words + 1, stride_words)]


def embed_chains(samples, encoder, window_words=5, stride_words=3):
    clouds = []
    for s in samples:
        full_text = s["question"] + " " + s["answer"]
        chunks = sliding_window_chunks(full_text, window_words, stride_words)
        emb = encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        clouds.append(emb)
    return clouds


def compute_persistence(cloud, max_dim=1):
    from ripser import ripser
    if len(cloud) > 100:
        idx = np.random.choice(len(cloud), 100, replace=False)
        cloud = cloud[idx]
    result = ripser(cloud, maxdim=max_dim)
    out = {}
    for dim in range(max_dim + 1):
        dgm = result["dgms"][dim]
        finite = dgm[dgm[:, 1] < np.inf] if len(dgm) > 0 else np.array([]).reshape(0, 2)
        bars = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
        out[f"H{dim}"] = {
            "n_bars": len(bars), "bars": bars,
            "mean": float(bars.mean()) if len(bars) > 0 else 0.0,
            "max": float(bars.max()) if len(bars) > 0 else 0.0,
            "total": float(bars.sum()) if len(bars) > 0 else 0.0,
        }
    return out


def gini(values):
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    s = np.sort(values)
    n = len(s)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * s).sum() / (n * s.sum()))


# ── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(n_samples=200):
    start = time.time()

    samples = load_gsm8k(n_samples)

    from sentence_transformers import SentenceTransformer
    print("Loading encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    levels = {
        0: "L0: Truth (original)",
        1: "L1: Shuffle steps (order broken)",
        2: "L2: Random numbers (arithmetic broken)",
        3: "L3: Cross-problem steps (context broken)",
        4: "L4: Nonsense grammar (semantics broken)",
        5: "L5: Word salad (complete decoherence)",
    }

    all_stats = {}

    for level, desc in levels.items():
        print(f"\n{'─'*60}")
        print(f"  {desc}")
        print(f"{'─'*60}")

        corrupted = apply_corruption(samples, level)
        clouds = embed_chains(corrupted, encoder)

        stats = []
        for i, cloud in enumerate(clouds):
            if i % 50 == 0:
                print(f"    persistence {i}/{len(clouds)}...")
            pers = compute_persistence(cloud)
            entry = {
                "H0_mean": pers["H0"]["mean"], "H0_max": pers["H0"]["max"],
                "H0_total": pers["H0"]["total"], "H0_n": pers["H0"]["n_bars"],
                "H0_gini": gini(pers["H0"]["bars"]),
                "H1_mean": pers["H1"]["mean"], "H1_max": pers["H1"]["max"],
                "H1_total": pers["H1"]["total"], "H1_n": pers["H1"]["n_bars"],
                "H1_gini": gini(pers["H1"]["bars"]),
            }
            stats.append(entry)
        all_stats[level] = stats

    elapsed = time.time() - start

    # ── Report ────────────────────────────────────────────────────────
    metrics = ["H0_mean", "H0_max", "H0_gini", "H0_total",
               "H1_mean", "H1_max", "H1_gini", "H1_total", "H1_n"]

    print(f"\n\n{'='*90}")
    print(f"  DECOHERENCE SPECTRUM — {n_samples} GSM8K chains, {elapsed:.0f}s")
    print(f"{'='*90}\n")

    # Header
    header = f"{'Metric':<12}"
    for level, desc in levels.items():
        header += f" {'L'+str(level):>10}"
    print(header)
    print("-" * (12 + 11 * len(levels)))

    results = {}
    for m in metrics:
        row = f"{m:<12}"
        means = []
        for level in levels:
            vals = [s[m] for s in all_stats[level]]
            mean = np.mean(vals)
            means.append(mean)
            row += f" {mean:>10.4f}"
        print(row)
        results[m] = {f"L{l}": float(means[l]) for l in range(len(levels))}

    # Monotonicity check: does topology track decoherence?
    print(f"\nMonotonicity check (does metric change consistently with decoherence?):")
    from scipy import stats as scipy_stats
    print(f"{'Metric':<12} {'Spearman r':>12} {'p-value':>12} {'Monotone?':>12}")
    print("-" * 50)
    for m in metrics:
        means = [results[m][f"L{l}"] for l in range(len(levels))]
        r, p = scipy_stats.spearmanr(list(range(len(levels))), means)
        mono = "YES ***" if p < 0.001 else "YES *" if p < 0.05 else "no"
        print(f"{m:<12} {r:>12.4f} {p:>12.6f} {mono:>12}")
        results[m]["spearman_r"] = float(r)
        results[m]["spearman_p"] = float(p)

    # Pairwise: L0 (truth) vs each corruption level
    print(f"\nL0 (truth) vs each corruption level (Welch's t-test on H0_mean):")
    print(f"{'Level':<40} {'t-stat':>10} {'p-value':>12} {'Sig?':>8}")
    print("-" * 72)
    baseline = [s["H0_mean"] for s in all_stats[0]]
    for level in range(1, len(levels)):
        corrupted_vals = [s["H0_mean"] for s in all_stats[level]]
        t, p = scipy_stats.ttest_ind(baseline, corrupted_vals, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {levels[level]:<38} {t:>10.3f} {p:>12.6f} {sig:>8}")

    # Same for H1_total
    print(f"\nL0 vs each level (H1_total):")
    print(f"{'Level':<40} {'t-stat':>10} {'p-value':>12} {'Sig?':>8}")
    print("-" * 72)
    baseline_h1 = [s["H1_total"] for s in all_stats[0]]
    for level in range(1, len(levels)):
        corrupted_vals = [s["H1_total"] for s in all_stats[level]]
        t, p = scipy_stats.ttest_ind(baseline_h1, corrupted_vals, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {levels[level]:<38} {t:>10.3f} {p:>12.6f} {sig:>8}")

    print(f"\n{'='*90}")

    log = {
        "experiment": "decoherence_spectrum",
        "n_samples": n_samples,
        "elapsed": elapsed,
        "levels": {str(k): v for k, v in levels.items()},
        "results": results,
    }
    with open(OUTPUT_DIR / "experiment_1c_results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved → {OUTPUT_DIR / 'experiment_1c_results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    run_experiment(args.n_samples)
