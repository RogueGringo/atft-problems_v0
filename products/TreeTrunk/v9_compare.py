#!/usr/bin/env python3
"""v9 vs v8 Head-to-Head — Same queries, same FAISS, different sheaf.

Compares:
  v8: k-NN sheaf (v7 trained, 128-dim, proximity-based edges)
  v9: typed dependency sheaf (52 typed restriction maps, grammar-based edges)

Both use the same FAISS retrieval (384-dim cosine anchor).
The comparison isolates the sheaf's contribution.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import faiss

warnings.filterwarnings("ignore")

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
V8_DIR = Path(__file__).parent / "results" / "v8_faiss_prep"
V7_CKPT = Path(__file__).parent / "results" / "v7_checkpoints" / "v7_trainer_final.pt"
V9_CKPT = Path(__file__).parent / "results" / "v9_checkpoints" / "v9_final.pt"

from v6_topological_trainer import TextFeatureMap, DifferentiableSheafLaplacian
from v9_causal_graph import TextCausalGrapher, DependencyGraphSheafLaplacian, TypedSheafTopologyLoss


def run_comparison(n_test=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── FAISS (shared) ────────────────────────────────────────────────
    print("Loading shared FAISS index (384-dim)...")
    index = faiss.read_index(str(BASE_MAP_DIR / "faiss_index.bin"))
    if hasattr(index, 'nprobe'):
        index.nprobe = 64

    with open(BASE_MAP_DIR / "chunk_texts.json") as f:
        chunks = json.load(f)
    with open(BASE_MAP_DIR / "chunk_meta.json") as f:
        chunk_meta = json.load(f)
    with open(BASE_MAP_DIR / "articles.json") as f:
        article_titles = json.load(f)
    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        all_qa = json.load(f)

    title_to_idx = {t: i for i, t in enumerate(article_titles)}

    # ── v8 sheaf (k-NN, trained projection) ───────────────────────────
    print("Loading v8 sheaf (k-NN, v7 trained)...")
    v8_feat = TextFeatureMap(out_dim=128, vocab_size=1, backbone_dim=384).to(device)
    v8_sheaf = DifferentiableSheafLaplacian(d_in=128, stalk_dim=8, k=4).to(device)

    ckpt = torch.load(V7_CKPT, map_location=device, weights_only=False)
    v8_feat.load_state_dict(
        {k.replace("feature_map.", ""): v for k, v in ckpt.items()
         if k.startswith("feature_map.")}, strict=False)
    v8_sheaf.load_state_dict(
        {k.replace("sheaf_loss_fn.sheaf.", ""): v for k, v in ckpt.items()
         if k.startswith("sheaf_loss_fn.sheaf.")}, strict=False)
    v8_feat.eval(); v8_sheaf.eval()

    # ── v9 sheaf (typed dependency graph) ─────────────────────────────
    print("Loading v9 sheaf (typed, 52 maps)...")
    v9_loss = TypedSheafTopologyLoss(d_in=384, stalk_dim=8, margin=0.5).to(device)
    v9_state = torch.load(V9_CKPT, map_location=device, weights_only=False)
    # Pre-create restriction maps that exist in the checkpoint
    for key in v9_state:
        if key.startswith("sheaf.restriction_maps."):
            map_id = key.split(".")[2]
            v9_loss.sheaf._get_restriction(map_id)
    v9_loss.load_state_dict(v9_state)
    v9_loss.eval()

    grapher = TextCausalGrapher("en_core_web_sm")

    # ── Test queries ──────────────────────────────────────────────────
    answerable = [q for q in all_qa if q["has_answer"]]
    unseen = answerable[5000:]
    rng = np.random.default_rng(99)
    test_idx = rng.choice(len(unseen), min(n_test, len(unseen)), replace=False)
    test_qa = [unseen[i] for i in test_idx]

    print(f"\nHead-to-head: {len(test_qa)} queries")
    print(f"{'='*75}")

    k_faiss = 50
    k_eval = 5

    v8_title_at_1 = 0
    v9_title_at_1 = 0
    v8_answer_at_5 = 0
    v9_answer_at_5 = 0
    v8_deltas = []
    v9_deltas = []
    v8_times = []
    v9_times = []

    for i, qa in enumerate(test_qa):
        question = qa["question"]
        answer = qa["answer"]
        expected_art = title_to_idx.get(qa["title"], -1)

        # FAISS search (shared)
        q_emb = encoder.encode([question], convert_to_numpy=True)
        q_norm = q_emb.copy()
        faiss.normalize_L2(q_norm)
        scores, indices = index.search(q_norm, k_faiss)

        # ── v8 evaluation ─────────────────────────────────────────────
        t0 = time.time()
        q_cloud_384 = encoder.encode(
            [" ".join(question.split()[j:j+5])
             for j in range(0, max(1, len(question.split())-4), 3)]
            or [question],
            convert_to_numpy=True, show_progress_bar=False
        )

        v8_candidates = []
        for rank in range(min(k_eval, len(indices[0]))):
            idx = int(indices[0][rank])
            if idx < 0: continue
            chunk = chunks[idx]
            art = chunk_meta[idx]["article_idx"]

            a_cloud = encoder.encode(
                [" ".join(chunk.split()[j:j+5])
                 for j in range(0, max(1, len(chunk.split())-4), 3)]
                or [chunk],
                convert_to_numpy=True, show_progress_bar=False
            )
            full = np.vstack([q_cloud_384, a_cloud])
            with torch.no_grad():
                ft = torch.tensor(full, dtype=torch.float32, device=device)
                proj = v8_feat(ft)
                eigs = v8_sheaf(proj)
                nz = eigs[eigs > 1e-6]
                gap = float(nz[0].item()) if len(nz) > 0 else 0.0

            v8_candidates.append({
                "art": art, "gap": gap, "rank": rank,
                "has_answer": answer.lower() in chunk.lower(),
            })

        v8_candidates.sort(key=lambda c: c["gap"])
        v8_time = time.time() - t0
        v8_times.append(v8_time)

        if v8_candidates and v8_candidates[0]["art"] == expected_art:
            v8_title_at_1 += 1
        if any(c["has_answer"] for c in v8_candidates[:5]):
            v8_answer_at_5 += 1

        v8_correct_gap = next((c["gap"] for c in v8_candidates if c["art"] == expected_art), None)
        v8_wrong_gap = next((c["gap"] for c in v8_candidates if c["art"] != expected_art), None)
        if v8_correct_gap is not None and v8_wrong_gap is not None:
            v8_deltas.append(v8_wrong_gap - v8_correct_gap)

        # ── v9 evaluation ─────────────────────────────────────────────
        t0 = time.time()
        q_graph = grapher.parse(question)

        v9_candidates = []
        for rank in range(min(k_eval, len(indices[0]))):
            idx = int(indices[0][rank])
            if idx < 0: continue
            chunk = chunks[idx]
            art = chunk_meta[idx]["article_idx"]

            a_graph = grapher.parse(chunk)

            # Merge graphs
            all_nodes = q_graph.node_texts + a_graph.node_texts
            n_q = len(q_graph.node_texts)
            merged_edges = []
            for s, d, t in q_graph.edges:
                merged_edges.append((s, d, t))
            for s, d, t in a_graph.edges:
                merged_edges.append((s + n_q, d + n_q, t))
            if n_q > 0 and len(a_graph.node_texts) > 0:
                merged_edges.append((n_q - 1, n_q, "bridge"))

            # Encode and evaluate
            node_emb = torch.tensor(
                encoder.encode(all_nodes, convert_to_numpy=True, show_progress_bar=False),
                dtype=torch.float32, device=device
            )
            with torch.no_grad():
                eigs = v9_loss.sheaf(node_emb, merged_edges)
                nz = eigs[eigs > 1e-6]
                gap = float(nz[0].item()) if len(nz) > 0 else 0.0

            v9_candidates.append({
                "art": art, "gap": gap, "rank": rank,
                "has_answer": answer.lower() in chunk.lower(),
            })

        v9_candidates.sort(key=lambda c: c["gap"])
        v9_time = time.time() - t0
        v9_times.append(v9_time)

        if v9_candidates and v9_candidates[0]["art"] == expected_art:
            v9_title_at_1 += 1
        if any(c["has_answer"] for c in v9_candidates[:5]):
            v9_answer_at_5 += 1

        v9_correct_gap = next((c["gap"] for c in v9_candidates if c["art"] == expected_art), None)
        v9_wrong_gap = next((c["gap"] for c in v9_candidates if c["art"] != expected_art), None)
        if v9_correct_gap is not None and v9_wrong_gap is not None:
            v9_deltas.append(v9_wrong_gap - v9_correct_gap)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_test}  "
                  f"v8: T@1={v8_title_at_1/(i+1)*100:.0f}%  "
                  f"v9: T@1={v9_title_at_1/(i+1)*100:.0f}%")

    n = len(test_qa)
    print(f"\n{'='*75}")
    print(f"  HEAD-TO-HEAD RESULTS — {n} queries")
    print(f"{'='*75}")
    print(f"  {'Metric':<30s}  {'v8 (k-NN)':>12s}  {'v9 (typed)':>12s}")
    print(f"  {'─'*60}")
    print(f"  {'Title@1':<30s}  {v8_title_at_1/n*100:>11.1f}%  {v9_title_at_1/n*100:>11.1f}%")
    print(f"  {'Answer@5':<30s}  {v8_answer_at_5/n*100:>11.1f}%  {v9_answer_at_5/n*100:>11.1f}%")
    print(f"  {'Mean Δ(N-P)':<30s}  {np.mean(v8_deltas) if v8_deltas else 0:>+11.5f}  {np.mean(v9_deltas) if v9_deltas else 0:>+11.5f}")
    print(f"  {'Δ positive rate':<30s}  {sum(1 for d in v8_deltas if d>0)/max(len(v8_deltas),1)*100:>11.1f}%  {sum(1 for d in v9_deltas if d>0)/max(len(v9_deltas),1)*100:>11.1f}%")
    print(f"  {'Avg latency':<30s}  {np.mean(v8_times)*1000:>10.0f}ms  {np.mean(v9_times)*1000:>10.0f}ms")
    print(f"  {'Typed restriction maps':<30s}  {'N/A':>12s}  {len(v9_loss.sheaf.restriction_maps):>12d}")
    print(f"{'='*75}")

    results = {
        "n_test": n,
        "v8": {"title_at_1": v8_title_at_1, "answer_at_5": v8_answer_at_5,
               "mean_delta": float(np.mean(v8_deltas)) if v8_deltas else 0,
               "delta_pos_pct": sum(1 for d in v8_deltas if d>0)/max(len(v8_deltas),1)*100,
               "avg_ms": float(np.mean(v8_times))*1000},
        "v9": {"title_at_1": v9_title_at_1, "answer_at_5": v9_answer_at_5,
               "mean_delta": float(np.mean(v9_deltas)) if v9_deltas else 0,
               "delta_pos_pct": sum(1 for d in v9_deltas if d>0)/max(len(v9_deltas),1)*100,
               "avg_ms": float(np.mean(v9_times))*1000},
    }
    out = Path(__file__).parent / "results" / "v9_checkpoints" / "v8_vs_v9_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=200)
    args = parser.parse_args()
    run_comparison(args.n_test)
