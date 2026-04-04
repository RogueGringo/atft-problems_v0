#!/usr/bin/env python3
"""v11 Isomorphism Analyzer — Does the Universal Machine Code exist?

Loads the trained v11 restriction maps and measures whether conceptually
equivalent edge types from different domains converge to similar geometric
rotations.

If R_advcl (English "if") ≈ R_code_If_test (Python if) ≈ R_op_compare (math),
then the network has decoupled LOGIC from NOTATION.

Metrics computed for each matrix pair:
  - Cosine similarity of flattened matrices
  - Frobenius norm of difference (normalized)
  - Principal angle (subspace alignment)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path(__file__).parent / "results" / "v11_checkpoints" / "v11_universal_manifold.pt"
REGISTRY_PATH = Path(__file__).parent / "results" / "v9_compiled_graphs" / "v9_edge_registry.json"
OUTPUT_PATH = Path(__file__).parent / "results" / "v11_checkpoints" / "isomorphism_report.json"


def load_restriction_maps() -> tuple[dict[str, np.ndarray], dict]:
    """Load all trained restriction matrices, indexed by edge type name."""
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    id_to_name = registry["id_to_name"]

    maps = {}
    for key, tensor in state.items():
        if key.startswith("sheaf.restriction_maps.") and key.endswith(".weight"):
            # key format: "sheaf.restriction_maps.<id_str>.weight"
            parts = key.split(".")
            map_id = parts[2]
            # Resolve ID to human-readable name
            if map_id in id_to_name:
                name = id_to_name[map_id]
            elif map_id == "bridge":
                name = "bridge"
            else:
                name = f"unknown_{map_id}"
            maps[name] = tensor.numpy()

    print(f"Loaded {len(maps)} restriction matrices")
    return maps, registry


def matrix_similarity(A: np.ndarray, B: np.ndarray) -> dict:
    """Compute multiple similarity metrics between two matrices."""
    # 1. Cosine of flattened
    a_flat = A.flatten()
    b_flat = B.flatten()
    cos_sim = float(np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))

    # 2. Normalized Frobenius distance
    # Normalize each matrix by its Frobenius norm, then compute distance
    A_norm = A / (np.linalg.norm(A, 'fro') + 1e-12)
    B_norm = B / (np.linalg.norm(B, 'fro') + 1e-12)
    frob_dist = float(np.linalg.norm(A_norm - B_norm, 'fro'))

    # 3. Principal angle (subspace alignment)
    # Use SVD to find orthonormal bases
    try:
        Ua, _, _ = np.linalg.svd(A, full_matrices=False)
        Ub, _, _ = np.linalg.svd(B, full_matrices=False)
        # Canonical correlations
        M = Ua.T @ Ub
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        # Principal angle (smallest) in degrees
        principal_angle = float(np.degrees(np.arccos(s[0]))) if len(s) > 0 else 180.0
    except Exception:
        principal_angle = -1.0

    return {
        "cosine": cos_sim,
        "frobenius_dist": frob_dist,
        "principal_angle_deg": principal_angle,
    }


def analyze_cluster(name: str, concept_names: list[str], maps: dict[str, np.ndarray]):
    """Analyze geometric convergence within a conceptual cluster."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Filter to existing concepts
    available = [(n, maps[n]) for n in concept_names if n in maps]
    missing = [n for n in concept_names if n not in maps]

    if missing:
        print(f"  Missing from registry: {missing}")
    if len(available) < 2:
        print(f"  Not enough concepts to compare ({len(available)})")
        return {"cluster": name, "pairs": [], "missing": missing}

    print(f"  Comparing {len(available)} concepts:")
    for n, _ in available:
        print(f"    - {n}")

    results = []
    print(f"\n  {'Pair':<50s}  {'Cosine':>7s}  {'FrobD':>6s}  {'Angle':>7s}")
    print(f"  {'─'*76}")

    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            name_a, mat_a = available[i]
            name_b, mat_b = available[j]
            sim = matrix_similarity(mat_a, mat_b)
            pair = f"{name_a[:24]} ↔ {name_b[:24]}"
            print(f"  {pair:<50s}  {sim['cosine']:>+7.4f}  "
                  f"{sim['frobenius_dist']:>6.3f}  "
                  f"{sim['principal_angle_deg']:>6.1f}°")
            results.append({
                "a": name_a, "b": name_b,
                **sim,
            })

    return {"cluster": name, "concepts": [n for n, _ in available],
            "pairs": results, "missing": missing}


def compare_to_random_baseline(maps: dict[str, np.ndarray], n_pairs: int = 100):
    """Baseline: what does 'random' similarity look like?"""
    names = list(maps.keys())
    rng = np.random.default_rng(42)

    cosines = []
    frob_dists = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(names), 2, replace=False)
        sim = matrix_similarity(maps[names[a]], maps[names[b]])
        cosines.append(sim["cosine"])
        frob_dists.append(sim["frobenius_dist"])

    return {
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "mean_frob": float(np.mean(frob_dists)),
        "std_frob": float(np.std(frob_dists)),
    }


def main():
    print(f"\n{'='*70}")
    print(f"  v11 ISOMORPHISM ANALYSIS — Universal Machine Code Test")
    print(f"{'='*70}\n")

    maps, registry = load_restriction_maps()

    # Show all available edge type names
    available_names = sorted(maps.keys())
    print(f"\nAll edge types in trained model (first 30):")
    for n in available_names[:30]:
        print(f"  {n}")
    print(f"  ... ({len(available_names) - 30} more)" if len(available_names) > 30 else "")

    # Random baseline
    print(f"\n\nRandom baseline (what 'unrelated' looks like):")
    baseline = compare_to_random_baseline(maps)
    print(f"  Mean cosine: {baseline['mean_cosine']:+.4f} ± {baseline['std_cosine']:.4f}")
    print(f"  Mean Frob distance: {baseline['mean_frob']:.3f} ± {baseline['std_frob']:.3f}")
    print(f"  (Values far from baseline mean = significant convergence)")

    # ── Conceptual clusters ───────────────────────────────────────────
    clusters = {}

    # Conditional Logic: if/else, conditionals, comparisons
    clusters["conditional"] = analyze_cluster(
        "CONDITIONAL LOGIC (if/else/compare across notations)",
        ["advcl", "mark", "code_If_test", "code_If_body", "code_Compare_ops"],
        maps
    )

    # Sequential/Temporal: next-step, temporal progression
    clusters["sequential"] = analyze_cluster(
        "SEQUENTIAL/TEMPORAL (progression across notations)",
        ["conj", "step_next", "t_next", "code_Module_body"],
        maps
    )

    # Containment/Body: what's inside what
    clusters["containment"] = analyze_cluster(
        "CONTAINMENT/BODY (nested structures)",
        ["code_FunctionDef_body", "code_For_body", "code_If_body",
         "code_While_body", "acl"],
        maps
    )

    # Arguments/Operands: inputs to operations
    clusters["operands"] = analyze_cluster(
        "OPERANDS (inputs to operations)",
        ["dobj", "operand_left", "operand_right",
         "code_Call_args", "code_BinOp_left", "code_BinOp_right"],
        maps
    )

    # Causal/Relational: cause-effect, correlation
    clusters["causal"] = analyze_cluster(
        "CAUSAL/RELATIONAL (cause and effect)",
        ["prep", "computes", "causal_correlation", "state_transition"],
        maps
    )

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "baseline": baseline,
        "clusters": clusters,
        "n_restriction_maps": len(maps),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n\n{'='*70}")
    print(f"  Report saved: {OUTPUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
