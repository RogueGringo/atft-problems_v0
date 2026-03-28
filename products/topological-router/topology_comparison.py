"""Module 3: Topology Comparison

Computes four topology metrics comparing the prime subspace and residual
subspace of a model's hidden states. Uses Modules 1+2 output and produces
a 2x2 outcome classification.

Metrics
-------
1. Topological Isomorphism — H0 Gini similarity between prime and residual projections
2. Cross-Subspace Coherence — fiber transport cost across kNN edges
3. Persistence Under Projection — spectral gap / Gini signal survival per subspace
4. Convergence — stability of the adaptive basis Gini trajectory
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import h0_gini, gini_fast, spectral_gap, effective_rank  # noqa: E402
from prompts.loader import load_prompts  # noqa: E402
from adaptive_explorer import extract_hidden_states  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
BASIS_DIR = RESULTS_DIR / "basis_discovery"
OUTPUT_DIR = RESULTS_DIR / "topology_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric 1 — Topological Isomorphism
# ---------------------------------------------------------------------------

def metric1_topological_isomorphism(
    prime_basis: torch.Tensor,
    adaptive_basis: torch.Tensor,
    hidden_states: torch.Tensor,
) -> dict:
    """Compare H0 Gini of prime-projected vs residual-projected hidden states.

    Parameters
    ----------
    prime_basis    : (n_prime, hidden_dim)
    adaptive_basis : (n_full, hidden_dim)
    hidden_states  : (n_tokens, hidden_dim)

    Returns
    -------
    dict with gini_prime, gini_residual, relative_diff, isomorphic
    """
    n_prime = prime_basis.shape[0]

    # Project onto prime subspace
    prime_proj = (hidden_states @ prime_basis.T).numpy()  # (n_tokens, n_prime)
    gini_prime = h0_gini(prime_proj)

    # Residual basis = adaptive vectors beyond the primes
    if adaptive_basis.shape[0] > n_prime:
        residual_basis = adaptive_basis[n_prime:]  # (n_residual, hidden_dim)
        residual_proj = (hidden_states @ residual_basis.T).numpy()
        gini_residual = h0_gini(residual_proj)
    else:
        gini_residual = 0.0

    # Relative difference
    denom = max(abs(gini_prime), abs(gini_residual), 1e-10)
    relative_diff = abs(gini_prime - gini_residual) / denom

    return {
        "gini_prime": float(gini_prime),
        "gini_residual": float(gini_residual),
        "relative_diff": float(relative_diff),
        "isomorphic": bool(relative_diff < 0.20),
    }


# ---------------------------------------------------------------------------
# Metric 2 — Cross-Subspace Coherence
# ---------------------------------------------------------------------------

def metric2_cross_subspace_coherence(
    prime_basis: torch.Tensor,
    adaptive_basis: torch.Tensor,
    hidden_states: torch.Tensor,
    n_positions: int = 200,
    k_neighbors: int = 10,
) -> dict:
    """Fiber transport cost: how well prime-space fibers survive kNN transport
    when scaled by residual-space correlation.

    Parameters
    ----------
    prime_basis    : (n_prime, hidden_dim)
    adaptive_basis : (n_full, hidden_dim)
    hidden_states  : (n_tokens, hidden_dim)
    n_positions    : subsample cap
    k_neighbors    : kNN parameter

    Returns
    -------
    dict with spectral_sum, normalized_spectral_sum, high_coherence, n_positions, n_edges
    """
    n_prime = prime_basis.shape[0]
    hs = hidden_states.float()

    # Subsample
    if hs.shape[0] > n_positions:
        idx = torch.randperm(hs.shape[0])[:n_positions]
        hs = hs[idx]
    actual_n = hs.shape[0]

    # Prime-space projections (fibers)
    fibers = hs @ prime_basis.T  # (n, n_prime)

    # Check for residual basis
    if adaptive_basis.shape[0] <= n_prime:
        return {
            "spectral_sum": 0.0,
            "normalized_spectral_sum": 0.0,
            "high_coherence": True,
            "n_positions": actual_n,
            "n_edges": 0,
        }

    residual_basis = adaptive_basis[n_prime:]
    res_proj = hs @ residual_basis.T  # (n, n_residual)

    # kNN graph from full hidden-state distances
    dists = torch.cdist(hs, hs)  # (n, n)
    # Zero out self-distance to avoid self-loops
    dists.fill_diagonal_(float("inf"))
    _, knn_idx = dists.topk(k_neighbors, largest=False, dim=1)  # (n, k)

    # Collect edges as unique pairs
    edges = set()
    for i in range(actual_n):
        for j in knn_idx[i].tolist():
            edge = (min(i, j), max(i, j))
            edges.add(edge)
    edges = list(edges)
    n_edges = len(edges)

    if n_edges == 0:
        return {
            "spectral_sum": 0.0,
            "normalized_spectral_sum": 0.0,
            "high_coherence": True,
            "n_positions": actual_n,
            "n_edges": 0,
        }

    # Normalize residual projections for cosine similarity
    res_norms = res_proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    res_normed = res_proj / res_norms

    # Compute transport cost across edges
    total_diff_sq = 0.0
    for i, j in edges:
        # Residual correlation (cosine similarity)
        res_corr = float((res_normed[i] * res_normed[j]).sum())
        # Transport: scale fiber_j by residual correlation
        transported = fibers[j] * res_corr
        diff = fibers[i] - transported
        total_diff_sq += float((diff ** 2).sum())

    spectral_sum = total_diff_sq / n_edges

    # Normalize by fiber variance
    fiber_var = float(fibers.var().item())
    fiber_var = max(fiber_var, 1e-10)
    normalized = spectral_sum / fiber_var

    return {
        "spectral_sum": float(spectral_sum),
        "normalized_spectral_sum": float(normalized),
        "high_coherence": bool(normalized < 2.0),
        "n_positions": actual_n,
        "n_edges": n_edges,
    }


# ---------------------------------------------------------------------------
# Metric 3 — Persistence Under Projection
# ---------------------------------------------------------------------------

def metric3_persistence_under_projection(
    prime_basis: torch.Tensor,
    adaptive_basis: torch.Tensor,
    correct_hs: torch.Tensor,
    wrong_hs: torch.Tensor,
) -> dict:
    """Check if topology signal (correct vs wrong answers) survives projection
    onto each subspace.

    Parameters
    ----------
    prime_basis    : (n_prime, hidden_dim)
    adaptive_basis : (n_full, hidden_dim)
    correct_hs     : (n_correct_tokens, hidden_dim)
    wrong_hs       : (n_wrong_tokens, hidden_dim)

    Returns
    -------
    Nested dict with results per subspace (prime, full, residual).
    """
    n_prime = prime_basis.shape[0]

    # Build residual basis
    if adaptive_basis.shape[0] > n_prime:
        residual_basis = adaptive_basis[n_prime:]
    else:
        residual_basis = None

    bases = {
        "prime": prime_basis,
        "full": adaptive_basis,
    }
    if residual_basis is not None:
        bases["residual"] = residual_basis

    results = {}
    for name, basis in bases.items():
        # Project both sets
        correct_proj = correct_hs @ basis.T  # (n, k)
        wrong_proj = wrong_hs @ basis.T

        # Spectral gap
        sg_correct = spectral_gap(correct_proj)
        sg_wrong = spectral_gap(wrong_proj)
        sg_diff = sg_correct - sg_wrong

        # H0 Gini
        gini_correct = h0_gini(correct_proj.numpy())
        gini_wrong = h0_gini(wrong_proj.numpy())
        gini_diff = gini_correct - gini_wrong

        signal_survives = abs(sg_diff) > 0.1

        results[name] = {
            "sg_correct": float(sg_correct),
            "sg_wrong": float(sg_wrong),
            "sg_diff": float(sg_diff),
            "gini_correct": float(gini_correct),
            "gini_wrong": float(gini_wrong),
            "gini_diff": float(gini_diff),
            "signal_survives": bool(signal_survives),
        }

    return results


# ---------------------------------------------------------------------------
# Metric 4 — Convergence
# ---------------------------------------------------------------------------

def metric4_convergence(convergence_trajectory: list[float]) -> dict:
    """Check if the adaptive basis expansion converged.

    Parameters
    ----------
    convergence_trajectory : list of Gini values per iteration

    Returns
    -------
    dict with converged, final_gini, n_iterations, trajectory, mean_change_last_3
    """
    n = len(convergence_trajectory)
    if n < 4:
        return {
            "converged": False,
            "final_gini": convergence_trajectory[-1] if n > 0 else 0.0,
            "n_iterations": n,
            "trajectory": convergence_trajectory,
            "mean_change_last_3": float("inf"),
        }

    # Compute absolute differences between consecutive iterations
    diffs = [abs(convergence_trajectory[i] - convergence_trajectory[i - 1])
             for i in range(1, n)]

    last_3_diffs = diffs[-3:]
    mean_change = sum(last_3_diffs) / len(last_3_diffs)
    converged = all(d < 0.005 for d in last_3_diffs)

    return {
        "converged": bool(converged),
        "final_gini": float(convergence_trajectory[-1]),
        "n_iterations": n,
        "trajectory": convergence_trajectory,
        "mean_change_last_3": float(mean_change),
    }


# ---------------------------------------------------------------------------
# Outcome Classification (2x2)
# ---------------------------------------------------------------------------

def classify_outcome(m1_result: dict, m2_result: dict) -> dict:
    """Map Metric 1 + Metric 2 results to a 2x2 outcome grid.

    Rows: Gini similar (isomorphic) vs dissimilar
    Cols: High coherence vs low coherence

    Cells
    -----
    (similar, high)     → latent_truth
    (similar, low)      → independent_manifolds
    (dissimilar, high)  → transformation
    (dissimilar, low)   → noise
    """
    gini_similar = m1_result["isomorphic"]
    high_coherence = m2_result["high_coherence"]

    if gini_similar and high_coherence:
        cell = "latent_truth"
        interpretation = (
            "Prime and residual subspaces share similar topological structure "
            "with high cross-subspace coherence — the model likely encodes a "
            "unified latent representation that spans both subspaces."
        )
        action = "Proceed with unified topology analysis."
    elif gini_similar and not high_coherence:
        cell = "independent_manifolds"
        interpretation = (
            "Prime and residual subspaces have similar Gini structure but low "
            "coherence — they encode similar complexity independently, suggesting "
            "redundant or parallel representations."
        )
        action = "Investigate whether subspaces capture different features with similar topology."
    elif not gini_similar and high_coherence:
        cell = "transformation"
        interpretation = (
            "Prime and residual subspaces differ in topological structure but "
            "maintain high coherence — the residual transforms the prime representation "
            "while preserving relational structure."
        )
        action = "Characterize the transformation mapping between subspaces."
    else:
        cell = "noise"
        interpretation = (
            "Prime and residual subspaces differ in structure with low coherence — "
            "the residual subspace may be capturing noise rather than structured "
            "representations."
        )
        action = "Consider discarding residual subspace or refining basis discovery."

    return {
        "cell": cell,
        "gini_similar": gini_similar,
        "high_coherence": high_coherence,
        "interpretation": interpretation,
        "action": action,
    }


# ---------------------------------------------------------------------------
# Run Module 3
# ---------------------------------------------------------------------------

def run_module3(model_name: str, device: str = "cuda") -> dict:
    """Run all four topology comparison metrics and classify outcome.

    Parameters
    ----------
    model_name : HuggingFace model name (e.g. "Qwen/Qwen2.5-0.5B")
    device     : "cuda" or "cpu"

    Returns
    -------
    dict with all metric results and classification
    """
    model_short = model_name.replace("/", "_").replace("\\", "_")

    print(f"\n{'=' * 60}")
    print(f"  Module 3 — Topology Comparison")
    print(f"  model: {model_name}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 1. Load Module 1 + Module 2 .pt files
    # ------------------------------------------------------------------
    # Find prime basis file via glob
    prime_files = list(BASIS_DIR.glob(f"prime_basis_*{model_short}*.pt"))
    if not prime_files:
        # Try with just the model short name (e.g., Qwen2.5-0.5B)
        short = model_name.split("/")[-1]
        prime_files = list(BASIS_DIR.glob(f"prime_basis_*{short}*.pt"))
    if not prime_files:
        raise FileNotFoundError(
            f"No prime_basis .pt file found in {BASIS_DIR} for {model_name}"
        )
    prime_path = prime_files[0]
    print(f"  Loading Module 1: {prime_path.name}")
    m1_data = torch.load(prime_path, map_location="cpu", weights_only=True)
    prime_basis = m1_data["prime_basis"].float()
    target_layer = int(m1_data["target_layer"])
    n_prime = prime_basis.shape[0]
    print(f"    prime_basis: {prime_basis.shape}, target_layer: {target_layer}")

    # Find adaptive basis file via glob
    adaptive_files = list(BASIS_DIR.glob(f"adaptive_basis_*{model_short}*.pt"))
    if not adaptive_files:
        short = model_name.split("/")[-1]
        adaptive_files = list(BASIS_DIR.glob(f"adaptive_basis_*{short}*.pt"))
    if not adaptive_files:
        raise FileNotFoundError(
            f"No adaptive_basis .pt file found in {BASIS_DIR} for {model_name}"
        )
    adaptive_path = adaptive_files[0]
    print(f"  Loading Module 2: {adaptive_path.name}")
    m2_data = torch.load(adaptive_path, map_location="cpu", weights_only=False)
    adaptive_basis = m2_data["adaptive_basis"].float()
    convergence_trajectory = m2_data["convergence_trajectory"]
    n_full = adaptive_basis.shape[0]
    n_residual = n_full - n_prime
    print(f"    adaptive_basis: {adaptive_basis.shape} "
          f"(prime={n_prime}, residual={n_residual})")

    # ------------------------------------------------------------------
    # 2. Load model + extract hidden states from diverse prompts
    # ------------------------------------------------------------------
    print(f"\n  Loading model...")
    if "AWQ" in model_name:
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
        model = awq.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            output_hidden_states=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_prompts = load_prompts()
    # Select 50 diverse prompts
    import random
    random.seed(42)
    if len(all_prompts) > 50:
        selected = random.sample(all_prompts, 50)
    else:
        selected = all_prompts
    prompt_texts = [p["text"] for p in selected]

    print(f"  Extracting hidden states from {len(prompt_texts)} prompts "
          f"(layer {target_layer})...")
    t0 = time.time()
    hidden_states = extract_hidden_states(
        model, tokenizer, prompt_texts, target_layer, device
    )
    print(f"    hidden_states: {hidden_states.shape} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Run Metric 1 — Topological Isomorphism
    # ------------------------------------------------------------------
    print(f"\n  Metric 1: Topological Isomorphism...")
    t0 = time.time()
    m1_result = metric1_topological_isomorphism(
        prime_basis, adaptive_basis, hidden_states
    )
    print(f"    gini_prime={m1_result['gini_prime']:.4f}, "
          f"gini_residual={m1_result['gini_residual']:.4f}, "
          f"relative_diff={m1_result['relative_diff']:.4f}, "
          f"isomorphic={m1_result['isomorphic']} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Run Metric 2 — Cross-Subspace Coherence
    # ------------------------------------------------------------------
    print(f"\n  Metric 2: Cross-Subspace Coherence...")
    t0 = time.time()
    m2_result = metric2_cross_subspace_coherence(
        prime_basis, adaptive_basis, hidden_states
    )
    print(f"    spectral_sum={m2_result['spectral_sum']:.4f}, "
          f"normalized={m2_result['normalized_spectral_sum']:.4f}, "
          f"high_coherence={m2_result['high_coherence']}, "
          f"n_edges={m2_result['n_edges']} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Run Metric 3 — Persistence Under Projection (if data available)
    # ------------------------------------------------------------------
    m3_result = None
    # Look for clean bench results
    short = model_name.split("/")[-1]
    clean_bench_files = list(RESULTS_DIR.glob(f"clean_bench_mmlu_physics_{short}*.json"))
    if not clean_bench_files:
        clean_bench_files = list(RESULTS_DIR.glob(f"clean_bench_*{short}*.json"))

    if clean_bench_files:
        bench_path = clean_bench_files[0]
        print(f"\n  Metric 3: Persistence Under Projection...")
        print(f"    Loading clean bench: {bench_path.name}")
        with open(bench_path) as f:
            bench_data = json.load(f)

        # Extract correct/incorrect question texts
        correct_texts = []
        wrong_texts = []
        for r in bench_data.get("results", []):
            q = r.get("question", "")
            if not q:
                continue
            if r.get("lk_correct", False):
                correct_texts.append(q)
            else:
                wrong_texts.append(q)

        if correct_texts and wrong_texts:
            print(f"    Correct: {len(correct_texts)} questions, "
                  f"Wrong: {len(wrong_texts)} questions")
            t0 = time.time()
            correct_hs = extract_hidden_states(
                model, tokenizer, correct_texts, target_layer, device
            )
            wrong_hs = extract_hidden_states(
                model, tokenizer, wrong_texts, target_layer, device
            )
            m3_result = metric3_persistence_under_projection(
                prime_basis, adaptive_basis, correct_hs, wrong_hs
            )
            for name, sub in m3_result.items():
                print(f"    [{name}] sg_diff={sub['sg_diff']:.4f}, "
                      f"gini_diff={sub['gini_diff']:.4f}, "
                      f"signal_survives={sub['signal_survives']}")
            print(f"    ({time.time() - t0:.1f}s)")
        else:
            print(f"    Skipping Metric 3: insufficient correct/wrong splits")
    else:
        print(f"\n  Metric 3: SKIPPED (no clean bench data for {short})")

    # ------------------------------------------------------------------
    # 6. Run Metric 4 — Convergence
    # ------------------------------------------------------------------
    print(f"\n  Metric 4: Convergence...")
    m4_result = metric4_convergence(convergence_trajectory)
    print(f"    converged={m4_result['converged']}, "
          f"final_gini={m4_result['final_gini']:.4f}, "
          f"n_iterations={m4_result['n_iterations']}, "
          f"mean_change_last_3={m4_result['mean_change_last_3']:.6f}")

    # ------------------------------------------------------------------
    # 7. Classify outcome
    # ------------------------------------------------------------------
    classification = classify_outcome(m1_result, m2_result)
    print(f"\n  Classification: {classification['cell']}")
    print(f"    {classification['interpretation']}")
    print(f"    Action: {classification['action']}")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output = {
        "model": model_name,
        "target_layer": target_layer,
        "n_prime": n_prime,
        "n_full": n_full,
        "n_residual": n_residual,
        "n_tokens": int(hidden_states.shape[0]),
        "metric1_topological_isomorphism": m1_result,
        "metric2_cross_subspace_coherence": m2_result,
        "metric3_persistence_under_projection": m3_result,
        "metric4_convergence": {
            k: v for k, v in m4_result.items() if k != "trajectory"
        },
        "convergence_trajectory": m4_result["trajectory"],
        "classification": classification,
    }

    out_path = OUTPUT_DIR / f"metrics_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  Module 3 Complete")
    print(f"{'=' * 60}\n")

    return output


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_module3("Qwen/Qwen2.5-0.5B")
