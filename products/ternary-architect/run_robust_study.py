#!/usr/bin/env python3
"""
100K ROBUST STUDY — BitFlip from identity init across 7 domains.

The experiment that decides: does the data carve the crystal,
or is the crystal a method artifact?

BitFlip is the only method that crosses quantization boundaries.
Identity init means every departure from 0/100/0 was data-driven.
100K steps allows full convergence.

Sequential runs. One GPU. ~24.5 hours total.
"""
import subprocess
import sys
import json
import shutil
import time
from pathlib import Path

DATASETS = [
    ("tinystories", "200000", "Simple children's fiction (CONTROL)"),
    ("wikitext", "200000", "English Wikipedia encyclopedia"),
    ("korean", "100000", "Korean Wikipedia"),
    ("chinese", "100000", "Chinese Wikipedia"),
    ("arabic", "100000", "Arabic Wikipedia"),
    ("kant", "200000", "Kant's Critique of Pure Reason"),
    ("animalfarm", "200000", "Animal Farm (allegory)"),
]

BASE_CMD = [
    sys.executable, "run_long.py",
    "--size", "small",
    "--bitflip",
    "--init_mode", "identity",
    "--flip_pct", "0.001",
    "--flip_cycle", "100",
    "--flip_warmup", "500",
    "--flip_cooldown", "10000",
    "--max_steps", "100000",
    "--batch_size", "8",
    "--effective_batch", "32",
]

results = {}
start_all = time.time()

for i, (dataset, n_samples, desc) in enumerate(DATASETS):
    print(f"\n{'='*70}")
    print(f"  RUN {i+1}/{len(DATASETS)}: {dataset} — {desc}")
    print(f"  BitFlip from identity init, 100K steps")
    print(f"  Elapsed: {(time.time()-start_all)/3600:.1f} hrs")
    print(f"{'='*70}\n", flush=True)

    cmd = BASE_CMD + ["--dataset", dataset, "--n_samples", n_samples]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    # Read results from the default output dir
    # BitFlip identity init writes to long_run_small_identity_bitflip/
    log_path = Path("results/long_run_small_identity_bitflip/training_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        final = log.get("final", {})
        w = final.get("weight_dist", {})
        test_ppl = final.get("test_ppl", {})
        ppl = test_ppl.get("ppl", 0) if isinstance(test_ppl, dict) else 0

        crystal = f"{w.get('zero',0):.4f}/{w.get('one',0):.4f}/{w.get('three',0):.4f}"
        asymmetry = abs(w.get('zero', 0) - w.get('three', 0))

        results[dataset] = {
            "crystal": crystal,
            "void": w.get("zero", 0),
            "identity": w.get("one", 0),
            "prime": w.get("three", 0),
            "ppl": ppl,
            "asymmetry": asymmetry,
            "elapsed_min": elapsed / 60,
        }

        print(f"\n  RESULT {dataset}:")
        print(f"    Crystal: {crystal}")
        print(f"    PPL: {ppl:.1f}")
        print(f"    Asymmetry |void-prime|: {asymmetry:.4f}")
        print(f"    Time: {elapsed/60:.1f} min")

        # Copy to named dir
        out = Path(f"results/robust_study_{dataset}")
        out.mkdir(exist_ok=True)
        shutil.copy(log_path, out / "training_log.json")
        model_path = log_path.parent / "model.pt"
        if model_path.exists():
            shutil.copy(model_path, out / "model.pt")
    else:
        print(f"  WARNING: No results found for {dataset}")
        results[dataset] = {"error": "no results file"}

# ── FINAL COMPARISON TABLE ──
print(f"\n\n{'='*70}")
print(f"  100K ROBUST STUDY — COMPLETE")
print(f"  Total time: {(time.time()-start_all)/3600:.1f} hours")
print(f"{'='*70}\n")

print(f"{'Dataset':>15s} | {'Void':>8s} {'Ident':>8s} {'Prime':>8s} | {'|V-P|':>8s} | {'PPL':>8s}")
print("-" * 70)
for dataset, r in results.items():
    if "error" in r:
        print(f"{dataset:>15s} | {'ERROR':>8s}")
        continue
    sym = "SYMMETRIC" if r["asymmetry"] < 0.005 else f"{r['asymmetry']:.4f}"
    print(f"{dataset:>15s} | {r['void']:8.4f} {r['identity']:8.4f} {r['prime']:8.4f} | {sym:>8s} | {r['ppl']:8.1f}")

# Key question
all_primes = [r["prime"] for r in results.values() if "prime" in r]
if all_primes:
    prime_std = (max(all_primes) - min(all_primes))
    print(f"\nPrime range across datasets: {min(all_primes):.4f} to {max(all_primes):.4f} (span={prime_std:.4f})")
    if prime_std > 0.02:
        print("VERDICT: DATA DETERMINES CRYSTAL — different datasets produce different ratios")
    else:
        print("VERDICT: METHOD ARTIFACT — all datasets produce the same ratio")

# Save summary
summary = {
    "experiment": "100K_robust_study",
    "method": "BitFlip from identity init",
    "steps": 100000,
    "model": "small (6L × 512, 45M params)",
    "results": results,
    "total_hours": (time.time() - start_all) / 3600,
}
with open("results/robust_study_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSummary saved to results/robust_study_summary.json")
