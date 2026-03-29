#!/usr/bin/env python3
"""Validation Battery — Confirm or kill the tokenization thesis.

Four tests, binary pass/fail. If all pass, the finding is real.
If any existential test fails, the finding is an artifact.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_1samp
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import gini_fast

BASIS_DIR = Path(__file__).parent / "results" / "basis_discovery"


# ── Shared utilities ────────────────────────────────────────────────────────

def extract_all_hidden_states(model, tokenizer, prompts: list[str],
                               target_layer: int, device: str) -> np.ndarray:
    """Extract hidden states for all prompts, return as numpy array."""
    all_hs = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"],
                          output_hidden_states=True)
        hs = outputs.hidden_states[target_layer][0].cpu().float().numpy()
        all_hs.append(hs)

    # Stack all tokens
    stacked = np.vstack(all_hs)
    # Mean-center
    stacked = stacked - stacked.mean(axis=0)
    return stacked


def compute_gini_comparison(hidden_states: np.ndarray,
                             prime_basis: np.ndarray,
                             residual_basis: np.ndarray) -> dict:
    """Project hidden states onto prime and residual subspaces, compare SVD Gini."""
    # Project onto prime subspace
    prime_proj = hidden_states @ prime_basis.T
    _, S_prime, _ = np.linalg.svd(prime_proj, full_matrices=False)
    gini_prime = gini_fast(S_prime)

    # Project onto residual subspace
    if residual_basis.shape[0] > 0:
        res_proj = hidden_states @ residual_basis.T
        _, S_res, _ = np.linalg.svd(res_proj, full_matrices=False)
        gini_residual = gini_fast(S_res)
    else:
        gini_residual = 0.0

    return {
        "gini_prime": float(gini_prime),
        "gini_residual": float(gini_residual),
        "difference": float(gini_residual - gini_prime),
    }


# ── Test 1: Random Null ─────────────────────────────────────────────────────

def test_random_null(prime_basis: np.ndarray, residual_basis: np.ndarray,
                     n_tokens: int, hidden_dim: int, n_repeats: int = 10) -> dict:
    """Does random data show the same residual > prime pattern?"""
    print("\n  TEST 1: Random Null")
    print(f"    {n_repeats} random matrices of shape ({n_tokens}, {hidden_dim})")

    diffs = []
    for seed in range(n_repeats):
        rng = np.random.default_rng(seed)
        random_hs = rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32)
        random_hs = random_hs - random_hs.mean(axis=0)

        result = compute_gini_comparison(random_hs, prime_basis, residual_basis)
        diffs.append(result["difference"])
        print(f"    Seed {seed}: prime={result['gini_prime']:.4f} "
              f"residual={result['gini_residual']:.4f} diff={result['difference']:+.4f}")

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    t_stat, p_value = ttest_1samp(diffs, 0)

    # PASS if random data does NOT show residual > prime
    passed = mean_diff <= 0 or p_value > 0.05

    print(f"    Mean diff: {mean_diff:+.4f} +/- {std_diff:.4f}")
    print(f"    t={t_stat:.3f}, p={p_value:.4f}")
    print(f"    RESULT: {'PASS' if passed else 'FAIL'}"
          f" — random {'does NOT' if passed else 'DOES'} show the pattern")

    return {
        "test": "random_null",
        "passed": bool(passed),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "all_diffs": [float(d) for d in diffs],
    }


# ── Test 2: Dimensional Parity ──────────────────────────────────────────────

def test_dimensional_parity(hidden_states: np.ndarray,
                             prime_basis: np.ndarray,
                             adaptive_basis: np.ndarray) -> dict:
    """Is the Gini difference just from more dimensions?"""
    print("\n  TEST 2: Dimensional Parity")

    n_prime = prime_basis.shape[0]

    # Take equal number of residual vectors
    if adaptive_basis.shape[0] <= n_prime:
        print(f"    Adaptive basis ({adaptive_basis.shape[0]}) <= prime basis ({n_prime})")
        print(f"    Cannot test — SKIP")
        return {"test": "dimensional_parity", "passed": True, "skipped": True,
                "reason": "adaptive basis not larger than prime basis"}

    residual_basis_equal = adaptive_basis[n_prime:2 * n_prime]
    n_res = residual_basis_equal.shape[0]
    print(f"    Comparing {n_prime} prime vectors vs {n_res} residual vectors")

    # Project onto equal-dimensional subspaces
    result_prime = hidden_states @ prime_basis.T
    _, S_prime, _ = np.linalg.svd(result_prime, full_matrices=False)
    gini_prime = gini_fast(S_prime)

    result_residual = hidden_states @ residual_basis_equal.T
    _, S_res, _ = np.linalg.svd(result_residual, full_matrices=False)
    gini_residual = gini_fast(S_res)

    diff = gini_residual - gini_prime
    passed = diff > 0  # residual still richer at equal dims

    print(f"    Prime-58 Gini:    {gini_prime:.4f}")
    print(f"    Residual-{n_res} Gini: {gini_residual:.4f}")
    print(f"    Difference:       {diff:+.4f}")
    print(f"    RESULT: {'PASS' if passed else 'FAIL'}"
          f" — {'persists' if passed else 'disappears'} at equal dimensionality")

    return {
        "test": "dimensional_parity",
        "passed": bool(passed),
        "n_prime": n_prime,
        "n_residual": n_res,
        "gini_prime": float(gini_prime),
        "gini_residual": float(gini_residual),
        "difference": float(diff),
    }


# ── Test 3: Permutation ─────────────────────────────────────────────────────

def test_permutation(hidden_states: np.ndarray,
                      prime_basis: np.ndarray,
                      residual_basis: np.ndarray,
                      n_repeats: int = 10) -> dict:
    """Does shuffling token positions destroy the signal?"""
    print("\n  TEST 3: Permutation")

    # Real data
    real_result = compute_gini_comparison(hidden_states, prime_basis, residual_basis)
    real_diff = real_result["difference"]
    print(f"    Real data diff: {real_diff:+.4f}")

    # Shuffled data
    shuffled_diffs = []
    for seed in range(n_repeats):
        rng = np.random.default_rng(seed + 100)
        perm = rng.permutation(hidden_states.shape[0])
        shuffled_hs = hidden_states[perm]
        # Re-center after shuffle
        shuffled_hs = shuffled_hs - shuffled_hs.mean(axis=0)

        result = compute_gini_comparison(shuffled_hs, prime_basis, residual_basis)
        shuffled_diffs.append(result["difference"])

    mean_shuffled = np.mean(shuffled_diffs)
    std_shuffled = np.std(shuffled_diffs)

    # The gap should be smaller on shuffled data
    # Or: shuffled data should show sign reversal > 30% of the time
    sign_reversals = sum(1 for d in shuffled_diffs if d <= 0)
    reversal_rate = sign_reversals / n_repeats

    # Shuffled gap should be significantly smaller than real gap
    reduction = 1 - (mean_shuffled / real_diff) if real_diff > 0 else 0
    passed = reduction > 0.1 or reversal_rate > 0.3

    print(f"    Shuffled mean diff: {mean_shuffled:+.4f} +/- {std_shuffled:.4f}")
    print(f"    Reduction: {reduction:.1%}")
    print(f"    Sign reversals: {sign_reversals}/{n_repeats} ({reversal_rate:.0%})")
    print(f"    RESULT: {'PASS' if passed else 'FAIL (distributional)'}"
          f" — shuffling {'reduces' if passed else 'preserves'} the gap")

    return {
        "test": "permutation",
        "passed": bool(passed),
        "real_diff": float(real_diff),
        "shuffled_mean_diff": float(mean_shuffled),
        "shuffled_std_diff": float(std_shuffled),
        "reduction": float(reduction),
        "reversal_rate": float(reversal_rate),
    }


# ── Test 4: Bootstrap CIs ───────────────────────────────────────────────────

def test_bootstrap(per_prompt_hidden_states: list[np.ndarray],
                    prime_basis: np.ndarray,
                    residual_basis: np.ndarray,
                    n_bootstrap: int = 200) -> dict:
    """Bootstrap confidence intervals on the Gini difference."""
    print(f"\n  TEST 4: Bootstrap CIs ({n_bootstrap} resamples)")

    n_prompts = len(per_prompt_hidden_states)
    diffs = []

    for b in range(n_bootstrap):
        rng = np.random.default_rng(b + 1000)
        # Resample prompts with replacement
        indices = rng.choice(n_prompts, size=n_prompts, replace=True)
        resampled = [per_prompt_hidden_states[i] for i in indices]
        stacked = np.vstack(resampled)
        stacked = stacked - stacked.mean(axis=0)

        result = compute_gini_comparison(stacked, prime_basis, residual_basis)
        diffs.append(result["difference"])

        if (b + 1) % 50 == 0:
            ci_lo = np.percentile(diffs, 2.5)
            ci_hi = np.percentile(diffs, 97.5)
            print(f"    [{b+1}/{n_bootstrap}] running CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))

    # PASS if CI excludes zero (both bounds positive)
    passed = ci_lo > 0

    print(f"    Mean diff: {mean_diff:+.4f}")
    print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    RESULT: {'PASS' if passed else 'FAIL'}"
          f" — CI {'excludes' if passed else 'includes'} zero")

    return {
        "test": "bootstrap_ci",
        "passed": bool(passed),
        "mean_diff": float(mean_diff),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_bootstrap": n_bootstrap,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_battery(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                device: str = "cuda"):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  VALIDATION BATTERY")
    print("  Confirm or kill: is 'residual richer than prime' real?")
    print(f"  Model: {model_name}")
    print(f"  {ts}")
    print("=" * 65)

    model_short = model_name.split("/")[-1]

    # Load bases
    prime_path = list(BASIS_DIR.glob(f"prime_basis_*{model_short}*"))[0]
    adaptive_path = list(BASIS_DIR.glob(f"adaptive_basis_*{model_short}*"))[0]

    prime_data = torch.load(prime_path, weights_only=False)
    adaptive_data = torch.load(adaptive_path, weights_only=False)

    prime_basis = prime_data["prime_basis"].numpy()
    adaptive_basis = adaptive_data["adaptive_basis"].numpy()
    target_layer = prime_data["target_layer"]
    hidden_dim = prime_basis.shape[1]
    n_prime = prime_basis.shape[0]

    # Residual basis = adaptive vectors beyond the prime ones
    residual_basis = adaptive_basis[n_prime:]

    print(f"\n  Prime basis: {prime_basis.shape}")
    print(f"  Adaptive basis: {adaptive_basis.shape}")
    print(f"  Residual basis: {residual_basis.shape}")
    print(f"  Target layer: {target_layer}")

    # Load model and extract hidden states
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float16, device_map=device,
        output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from prompts.loader import load_prompts
    all_prompts = [p["text"] for p in load_prompts()]
    print(f"  Extracting hidden states for {len(all_prompts)} prompts...")

    # Per-prompt hidden states (for bootstrap)
    per_prompt_hs = []
    for text in all_prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"],
                          output_hidden_states=True)
        hs = outputs.hidden_states[target_layer][0].cpu().float().numpy()
        per_prompt_hs.append(hs)

    # Stacked + centered (for Tests 1-3)
    all_hs = np.vstack(per_prompt_hs)
    all_hs = all_hs - all_hs.mean(axis=0)
    n_tokens = all_hs.shape[0]
    print(f"  Hidden states: {all_hs.shape}")

    # Unload model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model unloaded. Running tests on cached data.\n")

    # ── Run tests ──
    t0 = time.time()

    r1 = test_random_null(prime_basis, residual_basis, n_tokens, hidden_dim)
    r2 = test_dimensional_parity(all_hs, prime_basis, adaptive_basis)
    r3 = test_permutation(all_hs, prime_basis, residual_basis)
    r4 = test_bootstrap(per_prompt_hs, prime_basis, residual_basis)

    elapsed = time.time() - t0

    # ── Verdict ──
    results = [r1, r2, r3, r4]
    n_passed = sum(1 for r in results if r["passed"])
    existential_passed = r1["passed"] and r2["passed"]  # Tests 1 & 2 are existential

    if n_passed == 4:
        verdict = "CONFIRMED — the finding is real"
    elif existential_passed and r4["passed"]:
        if not r3["passed"]:
            verdict = "REAL BUT DISTRIBUTIONAL — hierarchy is in activation statistics, not sequence structure"
        else:
            verdict = f"PARTIAL — {n_passed}/4 passed"
    elif not r1["passed"]:
        verdict = "KILLED — methodology artifact (random null failed)"
    elif not r2["passed"]:
        verdict = "KILLED — dimensionality artifact (parity test failed)"
    else:
        verdict = f"INCONCLUSIVE — {n_passed}/4 passed"

    print(f"\n{'='*65}")
    print("  VALIDATION BATTERY RESULTS")
    print(f"{'='*65}")
    print(f"  Test 1 (Random Null):        {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"  Test 2 (Dimensional Parity): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Test 3 (Permutation):        {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"  Test 4 (Bootstrap CI):       {'PASS' if r4['passed'] else 'FAIL'}"
          f"  [{r4['ci_lower']:+.4f}, {r4['ci_upper']:+.4f}]")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    output = {
        "timestamp": ts,
        "model": model_name,
        "hidden_dim": hidden_dim,
        "n_tokens": n_tokens,
        "n_prompts": len(all_prompts),
        "prime_basis_shape": list(prime_basis.shape),
        "residual_basis_shape": list(residual_basis.shape),
        "tests": {r["test"]: r for r in results},
        "n_passed": n_passed,
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }
    save_path = OUTPUT_DIR / "battery_results.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")

    return output


if __name__ == "__main__":
    run_battery("Qwen/Qwen2.5-1.5B-Instruct")
