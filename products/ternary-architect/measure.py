#!/usr/bin/env python3
"""Measurement battery for {0,1,3} experiments.

Includes all standard topology measures PLUS the novel ones:
  - Zero-mask topology (H₀/H₁ of where the zeros are)
  - Training persistence (birth/death of weight commitments)
  - Weight transition dynamics
  - Iterative inference convergence

Usage:
    python measure.py results/013_small_2026-03-29T12-00-00/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Import shared measures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))
from topo_measures import effective_rank, spectral_gap, gini_fast, h0_persistence, h0_gini


# ── Zero-pattern topology ─────────────────────────────────────────────────

def zero_mask_topology(weight_matrix: torch.Tensor, max_n: int = 500) -> dict:
    """Compute topology of WHERE the zeros are in a weight matrix.

    Treats zero-positions as points in (row, col) coordinate space.
    Computes H₀ persistence of this point cloud.

    If the zeros are structured (not random), their persistence diagram
    will differ from random sparsity at the same density.
    """
    w = weight_matrix.detach().cpu()
    zero_mask = (w == 0)
    zero_positions = torch.nonzero(zero_mask).float().numpy()  # (n_zeros, 2)

    if len(zero_positions) < 3:
        return {"n_zeros": len(zero_positions), "sparsity": 0.0,
                "h0_bars": [], "h0_gini": 0.0}

    sparsity = len(zero_positions) / w.numel()

    # Normalize coordinates to [0, 1] for comparable distances
    if zero_positions.shape[0] > 0:
        for d in range(zero_positions.shape[1]):
            rng = zero_positions[:, d].max() - zero_positions[:, d].min()
            if rng > 0:
                zero_positions[:, d] = (zero_positions[:, d] - zero_positions[:, d].min()) / rng

    # H₀ persistence of zero positions
    bars = h0_persistence(zero_positions, max_n=max_n)
    gini = gini_fast(bars) if len(bars) > 0 else 0.0

    # Compare against random baseline
    rng = np.random.RandomState(42)
    n_zeros = len(zero_positions)
    random_positions = rng.rand(n_zeros, 2)
    random_bars = h0_persistence(random_positions, max_n=max_n)
    random_gini = gini_fast(random_bars) if len(random_bars) > 0 else 0.0

    return {
        "n_zeros": n_zeros,
        "sparsity": sparsity,
        "h0_gini_trained": gini,
        "h0_gini_random": random_gini,
        "gini_ratio": gini / random_gini if random_gini > 0 else float("inf"),
        "h0_mean_bar_trained": float(bars.mean()) if len(bars) > 0 else 0.0,
        "h0_mean_bar_random": float(random_bars.mean()) if len(random_bars) > 0 else 0.0,
        "h0_max_bar_trained": float(bars.max()) if len(bars) > 0 else 0.0,
        "h0_max_bar_random": float(random_bars.max()) if len(random_bars) > 0 else 0.0,
    }


def zero_mask_ablation(model, tokenizer, sample_text: str,
                       device: torch.device) -> dict:
    """Test whether zero positions carry information.

    1. Get baseline perplexity
    2. Permute ONLY the zero positions (keep 1s and 3s in place)
    3. Measure perplexity degradation

    If perplexity degrades → zero arrangement carries task-relevant info.
    """
    from ternary_linear import TernaryLinear

    model.eval()
    toks = tokenizer.encode(sample_text, return_tensors="pt").to(device)
    if toks.shape[1] < 2:
        return {"error": "text too short"}

    targets = toks[:, 1:]
    inputs = toks[:, :-1]

    # Baseline perplexity
    with torch.no_grad():
        logits, loss = model(inputs, targets=targets)
        baseline_ppl = torch.exp(loss).item()

    # Permute zeros within each ternary layer
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w_q = module.get_quantized_weight()
            original_weights[name] = module.weight.data.clone()

            # Find zero positions
            zero_mask = (w_q == 0)
            nonzero_mask = ~zero_mask

            # Randomly permute which positions are zero
            flat_mask = zero_mask.flatten()
            perm = torch.randperm(flat_mask.numel(), device=flat_mask.device)
            permuted_mask = flat_mask[perm].view(zero_mask.shape)

            # Reconstruct: zeros go to permuted positions,
            # nonzero values stay in original positions
            # (This is approximate — we swap zero/nonzero assignments)
            new_weight = module.weight.data.clone()
            # Set positions that were nonzero but are now zero to 0-region
            became_zero = permuted_mask & ~zero_mask
            new_weight[became_zero] = 0.0  # Will quantize to 0
            # Set positions that were zero but are now nonzero to 1-region
            became_nonzero = ~permuted_mask & zero_mask
            new_weight[became_nonzero] = 1.0  # Will quantize to 1

            module.weight.data = new_weight

    # Permuted perplexity
    with torch.no_grad():
        _, loss_permuted = model(inputs, targets=targets)
        permuted_ppl = torch.exp(loss_permuted).item()

    # Restore original weights
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear) and name in original_weights:
            module.weight.data = original_weights[name]

    return {
        "baseline_ppl": baseline_ppl,
        "permuted_ppl": permuted_ppl,
        "ppl_ratio": permuted_ppl / baseline_ppl if baseline_ppl > 0 else float("inf"),
        "degradation_pct": (permuted_ppl - baseline_ppl) / baseline_ppl * 100,
    }


# ── Training persistence ──────────────────────────────────────────────────

def training_persistence(snapshots: list[dict[str, torch.Tensor]]) -> dict:
    """Compute persistence diagram of the training trajectory.

    For each weight position:
      birth = first snapshot where it takes its final value
      death = last snapshot where it changes away from final value
      bar_length = death - birth (in snapshot units)

    Long bars = early-committed structure (load-bearing)
    Short bars = late oscillation (noise being filtered)
    """
    if len(snapshots) < 2:
        return {"error": "need >= 2 snapshots"}

    final = snapshots[-1]
    all_bars = []

    for name in final:
        final_w = final[name]
        n_snaps = len(snapshots)

        # For each position, find first and last match with final value
        for i, snap in enumerate(snapshots):
            if name not in snap:
                continue
            matches = (snap[name] == final_w)
            # Track per-position: when did this first match?
            if i == 0:
                first_match = torch.zeros_like(final_w, dtype=torch.long)
                last_change = torch.zeros_like(final_w, dtype=torch.long)
                first_match[:] = n_snaps  # default: never matched
                prev = snap[name]
            else:
                changed = (snap[name] != prev)
                last_change[changed] = i
                prev = snap[name]

            # Update first match
            newly_matched = matches & (first_match == n_snaps)
            first_match[newly_matched] = i

        # Bar length = last_change - first_match (can be negative = always stable)
        bars = (last_change.float() - first_match.float()).flatten().numpy()
        all_bars.extend(bars.tolist())

    all_bars = np.array(all_bars)

    # Split into stable (bar <= 0) and oscillating (bar > 0)
    stable = all_bars[all_bars <= 0]
    oscillating = all_bars[all_bars > 0]

    return {
        "n_total": len(all_bars),
        "n_stable": len(stable),
        "n_oscillating": len(oscillating),
        "pct_stable": len(stable) / len(all_bars) * 100 if len(all_bars) > 0 else 0,
        "mean_bar": float(all_bars.mean()) if len(all_bars) > 0 else 0,
        "gini_bars": gini_fast(np.abs(all_bars)) if len(all_bars) > 0 else 0,
        "longest_bar": float(all_bars.max()) if len(all_bars) > 0 else 0,
    }


# ── Iterative inference ───────────────────────────────────────────────────

def iterative_inference(model, input_ids: torch.Tensor,
                        max_iter: int = 20, epsilon: float = 1e-3) -> dict:
    """Loop hidden states back through the model and measure convergence.

    Instead of single forward pass, iterate until topology stabilizes.
    This tests whether the {0,1,3} circuit has fixed-point behavior.
    """
    model.eval()
    device = input_ids.device

    with torch.no_grad():
        # First pass — get initial hidden states
        _, _, hidden_states = model(input_ids, return_hidden=True)
        prev_h = hidden_states[-1][0]  # last layer, first batch (T, d)

        trajectory = [{
            "iter": 0,
            "eff_rank": effective_rank(prev_h),
            "spectral_gap": spectral_gap(prev_h),
        }]

        # Subsequent passes — feed hidden state back as input
        # (project back to token space, then re-embed)
        for it in range(1, max_iter + 1):
            # Project to logits, take argmax, re-embed
            logits = model.lm_head(model.ln_f(prev_h.unsqueeze(0)))
            next_tokens = logits.argmax(dim=-1)

            # Forward pass with new tokens
            _, _, hidden_states = model(next_tokens, return_hidden=True)
            curr_h = hidden_states[-1][0]

            # Convergence metric
            delta = (curr_h - prev_h).norm() / (prev_h.norm() + 1e-10)

            trajectory.append({
                "iter": it,
                "eff_rank": effective_rank(curr_h),
                "spectral_gap": spectral_gap(curr_h),
                "delta": delta.item(),
            })

            if delta.item() < epsilon:
                break

            prev_h = curr_h

    converged = trajectory[-1].get("delta", 1.0) < epsilon
    n_iter = len(trajectory)

    return {
        "converged": converged,
        "n_iterations": n_iter,
        "final_delta": trajectory[-1].get("delta", None),
        "trajectory": trajectory,
        "eff_rank_start": trajectory[0]["eff_rank"],
        "eff_rank_end": trajectory[-1]["eff_rank"],
        "spectral_gap_start": trajectory[0]["spectral_gap"],
        "spectral_gap_end": trajectory[-1]["spectral_gap"],
    }


# ── Full analysis of a trained run ────────────────────────────────────────

def analyze_run(run_dir: str | Path):
    """Run the complete measurement battery on a trained model."""
    run_dir = Path(run_dir)
    print(f"Analyzing: {run_dir.name}")

    # Load training log
    with open(run_dir / "training_log.json") as f:
        log = json.load(f)

    config = log["config"]
    weight_set = config["weight_set"]

    # Rebuild model
    from ternary_transformer import build_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(size=config["size"], weight_set=weight_set,
                        vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device,
                                     weights_only=True))
    model = model.to(device)
    model.eval()

    results = {
        "run": run_dir.name,
        "config": config,
    }

    # ── Zero-mask topology ────────────────────────────────────────────
    if weight_set != "fp16":
        print("  Computing zero-mask topology...")
        from ternary_linear import TernaryLinear
        zero_results = {}
        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                w_q = module.get_quantized_weight()
                zero_results[name] = zero_mask_topology(w_q)
        results["zero_topology"] = zero_results

        # Aggregate
        trained_ginis = [v["h0_gini_trained"] for v in zero_results.values()]
        random_ginis = [v["h0_gini_random"] for v in zero_results.values()]
        results["zero_topology_summary"] = {
            "mean_trained_gini": float(np.mean(trained_ginis)),
            "mean_random_gini": float(np.mean(random_ginis)),
            "gini_ratio": float(np.mean(trained_ginis)) / float(np.mean(random_ginis))
                if np.mean(random_ginis) > 0 else float("inf"),
        }
        print(f"    Zero gini (trained): {np.mean(trained_ginis):.3f}")
        print(f"    Zero gini (random):  {np.mean(random_ginis):.3f}")

    # ── Zero-mask ablation ────────────────────────────────────────────
    if weight_set != "fp16":
        print("  Running zero-mask ablation...")
        test_text = "Once upon a time there was a little girl who loved to play in the garden"
        ablation = zero_mask_ablation(model, tokenizer, test_text, device)
        results["zero_ablation"] = ablation
        print(f"    Baseline ppl: {ablation.get('baseline_ppl', 0):.1f}")
        print(f"    Permuted ppl: {ablation.get('permuted_ppl', 0):.1f}")
        print(f"    Degradation:  {ablation.get('degradation_pct', 0):.1f}%")

    # ── Training persistence ──────────────────────────────────────────
    snap_path = run_dir / "weight_snapshots.pt"
    if snap_path.exists():
        print("  Computing training persistence...")
        snapshots = torch.load(snap_path, weights_only=False)
        persistence = training_persistence(snapshots)
        results["training_persistence"] = persistence
        print(f"    Stable weights: {persistence.get('pct_stable', 0):.1f}%")
        print(f"    Gini of bars:   {persistence.get('gini_bars', 0):.3f}")

    # ── Iterative inference ───────────────────────────────────────────
    print("  Testing iterative inference convergence...")
    test_prompt = "Once upon a time"
    toks = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    iter_results = iterative_inference(model, toks, max_iter=20)
    results["iterative_inference"] = iter_results
    print(f"    Converged: {iter_results['converged']} ({iter_results['n_iterations']} iters)")
    print(f"    Eff rank: {iter_results['eff_rank_start']:.1f} → {iter_results['eff_rank_end']:.1f}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(run_dir / "full_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Analysis saved to {run_dir / 'full_analysis.json'}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure.py <run_dir>")
        sys.exit(1)
    analyze_run(sys.argv[1])
