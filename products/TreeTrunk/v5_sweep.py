#!/usr/bin/env python3
"""v5 α/β Tuning Sweep — Find the balance between relevance and coherence.

For each test query, pre-computes ALL candidate scores (cosine, archetype,
H0, sheaf), then sweeps across α/β ratios to find the optimal balance.

The key insight: cosine knows WHAT to find, topology knows HOW the answer
connects. This sweep finds where those forces equilibrate.

Sweep dimensions:
  α (cosine weight): how strongly to anchor to semantic relevance
  β (topology weight): how strongly to rank by structural coherence

Output: results/v5_nq_navigation/sweep_results.json
"""
from __future__ import annotations

import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import faiss

warnings.filterwarnings("ignore", message=".*more columns than rows.*")

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
STENCIL_DIR = Path(__file__).parent / "results" / "v5_nq_stencils"
OUTPUT_DIR = Path(__file__).parent / "results" / "v5_nq_navigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import navigator components
from v5_nq_navigator import (
    UniversalSliceNavigator,
    embed_sliding_window,
    compute_signature,
    compute_h0_tightness,
    sheaf_truth_filter,
)


def run_sweep(n_test=30, k_faiss=50, k_sheaf=5):
    """Pre-compute all signals, then sweep α/β ratios."""
    start = time.time()

    nav = UniversalSliceNavigator()

    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    answerable = [q for q in qa_pairs if q["has_answer"]]
    random.seed(42)
    test_qa = random.sample(answerable, min(n_test, len(answerable)))

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Pre-compute ALL candidate scores for every query
    # ══════════════════════════════════════════════════════════════════
    print(f"\nPhase 1: Pre-computing scores for {n_test} queries × {k_faiss} candidates...")

    query_data = []

    for i, qa in enumerate(test_qa):
        q = qa["question"]
        answer = qa["answer"]
        expected_title = qa["title"]

        if i % 10 == 0:
            print(f"  Query {i}/{n_test}...")

        # Embed query
        q_emb = nav.encoder.encode([q], convert_to_numpy=True)
        q_norm = q_emb.copy()
        faiss.normalize_L2(q_norm)
        q_cloud_fine = embed_sliding_window(q, nav.encoder, window=5, stride=1)
        q_cloud_coarse = embed_sliding_window(q, nav.encoder, window=5, stride=3)

        # FAISS retrieval
        scores_faiss, indices_faiss = nav.index.search(q_norm, k_faiss)
        scores_faiss = scores_faiss[0]
        indices_faiss = indices_faiss[0]

        # Compute all candidate features
        art_sig_cache = {}
        candidates = []

        for rank in range(min(k_faiss, len(indices_faiss))):
            idx = int(indices_faiss[rank])
            if idx < 0:
                continue

            chunk_text = nav.chunks[idx]
            meta = nav.chunk_meta[idx]
            art_idx = meta["article_idx"]
            title = nav.article_titles[art_idx]

            # Article-level signature (cached)
            if art_idx not in art_sig_cache:
                chunk_ids = nav.art_to_chunks.get(art_idx, [idx])
                a_cloud = nav.embeddings[chunk_ids]
                sig = compute_signature(q_cloud_coarse, a_cloud)
                arch_id, arch_dist = nav.predict_archetype(sig)
                combined = np.vstack([q_cloud_coarse, a_cloud])
                if len(combined) > 80:
                    sub = np.random.choice(len(combined), 80, replace=False)
                    combined = combined[sub]
                h0 = compute_h0_tightness(combined)
                art_sig_cache[art_idx] = {
                    "archetype_id": arch_id,
                    "archetype_dist": float(arch_dist),
                    "h0_tightness": float(h0),
                }

            cached = art_sig_cache[art_idx]

            # Check ground truth
            title_hit = title == expected_title
            answer_hit = answer.lower() in chunk_text.lower() if answer else False

            candidates.append({
                "idx": idx,
                "text": chunk_text[:150],
                "title": title,
                "cosine_rank": rank,
                "cosine_sim": float(scores_faiss[rank]),
                "archetype_id": cached["archetype_id"],
                "archetype_dist": cached["archetype_dist"],
                "h0_tightness": cached["h0_tightness"],
                "title_hit": title_hit,
                "answer_hit": answer_hit,
            })

        # Compute sheaf for top candidates by cosine (the pool we'll re-rank)
        # Pre-compute sheaf for top-10 by cosine so sweep can use them
        for cand in candidates[:10]:
            a_cloud = embed_sliding_window(
                cand["text"], nav.encoder, window=5, stride=1
            )
            full_cloud = np.vstack([q_cloud_fine, a_cloud])
            sheaf = sheaf_truth_filter(full_cloud, k_neighbors=5, stalk_dim=8)
            cand["lambda1"] = sheaf["spectral_gap"]
            cand["kernel_dim"] = sheaf["kernel_dim"]

        # Rest get inf (not evaluated by sheaf)
        for cand in candidates[10:]:
            cand["lambda1"] = float('inf')
            cand["kernel_dim"] = -1

        query_data.append({
            "question": q[:100],
            "answer": answer[:60],
            "expected_title": expected_title,
            "candidates": candidates,
        })

    precompute_time = time.time() - start
    print(f"  Pre-computation done: {precompute_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Sweep α/β ratios
    # ══════════════════════════════════════════════════════════════════

    # Scoring function: combined = α * (1 - cosine_sim) + β * archetype_dist + γ * h0
    # We sweep different balances
    configs = [
        {"name": "cosine_only",    "w_cos": 1.0, "w_arch": 0.0, "w_h0": 0.0},
        {"name": "90cos_10topo",   "w_cos": 0.9, "w_arch": 0.05, "w_h0": 0.05},
        {"name": "80cos_20topo",   "w_cos": 0.8, "w_arch": 0.10, "w_h0": 0.10},
        {"name": "70cos_30topo",   "w_cos": 0.7, "w_arch": 0.15, "w_h0": 0.15},
        {"name": "60cos_40topo",   "w_cos": 0.6, "w_arch": 0.20, "w_h0": 0.20},
        {"name": "50cos_50topo",   "w_cos": 0.5, "w_arch": 0.25, "w_h0": 0.25},
        {"name": "40cos_60topo",   "w_cos": 0.4, "w_arch": 0.30, "w_h0": 0.30},
        {"name": "30cos_70topo",   "w_cos": 0.3, "w_arch": 0.35, "w_h0": 0.35},
        {"name": "20cos_80topo",   "w_cos": 0.2, "w_arch": 0.40, "w_h0": 0.40},
        {"name": "topo_only",      "w_cos": 0.0, "w_arch": 0.50, "w_h0": 0.50},
    ]

    print(f"\nPhase 2: Sweeping {len(configs)} α/β configurations...")
    print(f"{'Config':<20s} {'Title@1':>8s} {'Title@5':>8s} {'Ans@1':>8s} {'Ans@5':>8s} {'AvgCosRk':>9s}")
    print("─" * 65)

    sweep_results = []

    for cfg in configs:
        w_cos = cfg["w_cos"]
        w_arch = cfg["w_arch"]
        w_h0 = cfg["w_h0"]

        title_at_1 = 0
        title_at_5 = 0
        ans_at_1 = 0
        ans_at_5 = 0
        cos_ranks = []

        for qd in query_data:
            cands = qd["candidates"]
            if not cands:
                continue

            # Normalize scores to [0,1] for fair weighting
            cos_sims = [c["cosine_sim"] for c in cands]
            arch_dists = [c["archetype_dist"] for c in cands]
            h0s = [c["h0_tightness"] for c in cands]

            max_cos = max(cos_sims) if cos_sims else 1.0
            min_cos = min(cos_sims) if cos_sims else 0.0
            cos_range = max_cos - min_cos if max_cos > min_cos else 1.0

            max_arch = max(arch_dists) if arch_dists else 1.0
            min_arch = min(arch_dists) if arch_dists else 0.0
            arch_range = max_arch - min_arch if max_arch > min_arch else 1.0

            max_h0 = max(h0s) if h0s else 1.0
            min_h0 = min(h0s) if h0s else 0.0
            h0_range = max_h0 - min_h0 if max_h0 > min_h0 else 1.0

            # Score each candidate (lower is better)
            scored = []
            for c in cands:
                # Cosine: invert (higher sim = lower score)
                cos_score = 1.0 - (c["cosine_sim"] - min_cos) / cos_range
                # Archetype: normalize
                arch_score = (c["archetype_dist"] - min_arch) / arch_range
                # H0: normalize
                h0_score = (c["h0_tightness"] - min_h0) / h0_range

                combined = w_cos * cos_score + w_arch * arch_score + w_h0 * h0_score
                scored.append((combined, c))

            scored.sort(key=lambda x: x[0])
            ranked = [s[1] for s in scored]

            # Evaluate
            if ranked[0]["title_hit"]: title_at_1 += 1
            if ranked[0]["answer_hit"]: ans_at_1 += 1
            if any(r["title_hit"] for r in ranked[:5]): title_at_5 += 1
            if any(r["answer_hit"] for r in ranked[:5]): ans_at_5 += 1
            cos_ranks.append(ranked[0]["cosine_rank"])

        n = len(query_data)
        avg_cos_rank = np.mean(cos_ranks)

        result = {
            "config": cfg["name"],
            "w_cos": w_cos, "w_arch": w_arch, "w_h0": w_h0,
            "title_at_1": title_at_1, "title_at_1_pct": round(title_at_1/n*100, 1),
            "title_at_5": title_at_5, "title_at_5_pct": round(title_at_5/n*100, 1),
            "ans_at_1": ans_at_1, "ans_at_1_pct": round(ans_at_1/n*100, 1),
            "ans_at_5": ans_at_5, "ans_at_5_pct": round(ans_at_5/n*100, 1),
            "avg_cosine_rank": round(avg_cos_rank, 1),
        }
        sweep_results.append(result)

        print(f"{cfg['name']:<20s} "
              f"{title_at_1:>3d}/{n} "
              f"{title_at_5:>3d}/{n} "
              f"{ans_at_1:>3d}/{n} "
              f"{ans_at_5:>3d}/{n} "
              f"{avg_cos_rank:>8.1f}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Sheaf verification on best config's top-5
    # ══════════════════════════════════════════════════════════════════
    # Find best config by title@1 + ans@1
    best = max(sweep_results, key=lambda r: r["title_at_1"] + r["ans_at_1"])

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  SWEEP COMPLETE — {elapsed:.0f}s")
    print(f"{'='*65}")
    print(f"  Best config: {best['config']}")
    print(f"    Title@1: {best['title_at_1_pct']}%")
    print(f"    Ans@1:   {best['ans_at_1_pct']}%")
    print(f"    Title@5: {best['title_at_5_pct']}%")
    print(f"    Ans@5:   {best['ans_at_5_pct']}%")
    print(f"{'='*65}")

    log = {
        "experiment": "v5_alpha_beta_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_test": n_test,
        "k_faiss": k_faiss,
        "configs": sweep_results,
        "best_config": best,
    }
    with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved → {OUTPUT_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=30)
    args = parser.parse_args()
    run_sweep(args.n_test)
