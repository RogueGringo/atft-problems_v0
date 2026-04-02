#!/usr/bin/env python3
"""Full language + modality sweep — run all datasets sequentially.

Fires each experiment one after the other on the same GPU.
Builds the complete crystal comparison table.
"""
import subprocess
import sys
import json
from pathlib import Path

DATASETS = [
    # ("name", "n_samples", "description")
    ("code", "50000", "Python source code"),
    ("arabic", "100000", "Arabic Wikipedia"),
    ("japanese", "100000", "Japanese Wikipedia"),
]

BASE_CMD = [
    sys.executable, "run_harmonic.py",
    "--stage", "prism",
    "--width", "2048",
    "--n_heads", "16",
    "--prism_layers", "1",
    "--max_steps", "20000",
    "--batch_size", "8",
    "--effective_batch", "32",
    "--ternary_decay", "0.0",
]

results = {}

for dataset, n_samples, desc in DATASETS:
    print(f"\n{'='*60}")
    print(f"  SWEEP: {dataset} ({desc})")
    print(f"{'='*60}\n", flush=True)

    cmd = BASE_CMD + ["--dataset", dataset, "--n_samples", n_samples]
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Read results
    log_path = Path(f"results/harmonic_prism_1L_2048/training_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        final = log.get("final", {})
        w = final.get("weight_dist", {})
        ppl = final.get("test_ppl", {}).get("ppl", 0) if isinstance(final.get("test_ppl"), dict) else 0

        results[dataset] = {
            "crystal": f"{w.get('zero',0):.3f}/{w.get('one',0):.3f}/{w.get('three',0):.3f}",
            "ppl": ppl,
        }
        print(f"\n  RESULT {dataset}: {results[dataset]['crystal']} PPL={ppl:.1f}")

        # Copy to named dir
        import shutil
        out = Path(f"results/sweep_{dataset}")
        out.mkdir(exist_ok=True)
        shutil.copy(log_path, out / "training_log.json")

print(f"\n\n{'='*60}")
print("FULL SWEEP RESULTS")
print(f"{'='*60}")
print(f"{'Dataset':>15s} | {'Crystal':>15s} | {'PPL':>8s}")
print("-" * 45)

# Include prior results
prior = {
    "english": "0.222/0.417/0.361 PPL=55.7",
    "korean": "0.222/0.417/0.361 PPL=17.0",
    "chinese": "(running)",
    "kant": "0.222/0.417/0.361 PPL=179.4",
    "sep": "0.222/0.417/0.361 PPL=95.2",
    "animalfarm": "0.222/0.417/0.361 PPL=381.4",
    "tinystories": "0.050/0.950/0.000 PPL=14.5",
    "drilling": "0.154/0.696/0.150 PPL=~1.0",
}

for name, val in prior.items():
    print(f"  {name:>13s} | {val}")

for name, r in results.items():
    print(f"  {name:>13s} | {r['crystal']:>15s} | {r['ppl']:>8.1f}")

print(f"\n{'='*60}")
print("DONE — ALL DATASETS SWEPT")
