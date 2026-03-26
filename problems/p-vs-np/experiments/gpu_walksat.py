"""GPU-accelerated WalkSAT — all walks run in parallel on tensor cores.

The bottleneck in v6: Python loop over clauses × variables × steps.
Fix: represent everything as tensors, evaluate all clauses for all walks
simultaneously via batched indexing.

Architecture:
  assignments: (K, N) tensor — K parallel walks, N variables each
  var_indices: (M, 3) tensor — clause variable indices
  negations: (M, 3) tensor — clause negation mask

  Each step:
    1. Evaluate ALL clauses for ALL walks: (K, M) satisfaction matrix
    2. Find unsatisfied clauses per walk: (K, M) mask
    3. Pick random unsatisfied clause per walk: (K,) indices
    4. With probability p: flip random var in chosen clause
       With probability 1-p: flip var that minimizes breaks
    5. Update assignments

  Steps 1-3 are pure tensor ops (matmul, indexing, randint).
  Step 4 is the only branching — but it's batched across walks.
"""
import numpy as np
import torch


def gpu_walksat(var_indices_np, negations_np, n_vars, n_clauses,
                n_walks=500, max_steps=5000, noise_p=0.57, seed=42,
                device="cuda"):
    """GPU-parallel WalkSAT.

    Returns: (n_walks, n_vars) numpy array of best assignments,
             (n_walks,) numpy array of best unsat counts
    """
    rng = np.random.default_rng(seed)
    K = n_walks
    N = n_vars
    M = n_clauses

    vi = torch.tensor(var_indices_np, dtype=torch.long, device=device)  # (M, 3)
    neg = torch.tensor(negations_np, dtype=torch.float32, device=device)  # (M, 3)

    # Random starting assignments: (K, N)
    assignments = torch.tensor(
        rng.integers(0, 2, size=(K, N)), dtype=torch.float32, device=device
    )

    best_assignments = assignments.clone()
    best_unsat = torch.full((K,), M, dtype=torch.int32, device=device)

    for step in range(max_steps):
        # 1. Evaluate all clauses for all walks: (K, M, 3)
        lit_vals = assignments[:, vi]  # (K, M, 3) — variable values at clause positions
        lit_vals = torch.abs(lit_vals - neg.unsqueeze(0))  # apply negation
        clause_sat = lit_vals.max(dim=2).values  # (K, M) — 1 if sat, 0 if not

        # 2. Count unsatisfied per walk
        unsat_mask = (clause_sat < 0.5)  # (K, M) bool
        n_unsat = unsat_mask.sum(dim=1)  # (K,)

        # Update best
        improved = n_unsat < best_unsat
        best_unsat = torch.where(improved, n_unsat, best_unsat)
        best_assignments[improved] = assignments[improved]

        # Check if all walks satisfied
        if (n_unsat == 0).all():
            break

        # 3. Pick random unsatisfied clause per walk
        # For walks with no unsat clauses, pick clause 0 (won't flip)
        active = n_unsat > 0  # (K,) which walks still have unsat

        if not active.any():
            break

        # Sample one unsatisfied clause per walk using Gumbel-max trick
        # Set satisfied clauses to -inf so they're never picked
        logits = torch.where(unsat_mask, torch.zeros_like(clause_sat), torch.tensor(-1e9, device=device))
        chosen_clause = torch.argmax(logits + torch.rand_like(logits.float()) * 0.01, dim=1)  # (K,)

        # 4. Decide: noise flip or greedy flip
        noise_mask = torch.tensor(rng.random(K) < noise_p, dtype=torch.bool, device=device)

        # Noise: pick random literal in chosen clause
        random_lit_idx = torch.tensor(rng.integers(0, 3, size=K), dtype=torch.long, device=device)
        noise_var = vi[chosen_clause, random_lit_idx]  # (K,) — variable to flip for noise

        # Greedy: pick variable in chosen clause that appears in fewest other unsat clauses
        # Simplified: pick the variable in the clause with lowest current unsat count
        clause_vars = vi[chosen_clause]  # (K, 3) — 3 variables per chosen clause
        # For each variable, count how many unsat clauses it appears in
        # Simplified greedy: flip each of 3 vars tentatively, count resulting unsat, pick best
        greedy_var = clause_vars[:, 0]  # default: first var
        best_score = torch.full((K,), M, dtype=torch.float32, device=device)

        for lit_idx in range(3):
            v = clause_vars[:, lit_idx]  # (K,)
            # Tentative flip
            temp = assignments.clone()
            temp[torch.arange(K, device=device), v] = 1.0 - temp[torch.arange(K, device=device), v]
            # Evaluate
            temp_lit = temp[:, vi]
            temp_lit = torch.abs(temp_lit - neg.unsqueeze(0))
            temp_sat = temp_lit.max(dim=2).values
            temp_unsat = (temp_sat < 0.5).sum(dim=1).float()
            # Update best
            better = temp_unsat < best_score
            best_score = torch.where(better, temp_unsat, best_score)
            greedy_var = torch.where(better, v, greedy_var)

        # Choose noise or greedy variable
        flip_var = torch.where(noise_mask & active, noise_var, greedy_var)
        flip_var = torch.where(active, flip_var, torch.zeros_like(flip_var))  # inactive: no flip

        # Flip
        assignments[torch.arange(K, device=device), flip_var] = \
            1.0 - assignments[torch.arange(K, device=device), flip_var]

        # Don't flip inactive walks (set back)
        assignments[~active] = best_assignments[~active]

    return best_assignments.cpu().numpy().astype(np.int8), best_unsat.cpu().numpy()
