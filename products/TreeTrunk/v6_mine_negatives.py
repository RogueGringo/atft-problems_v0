#!/usr/bin/env python3
"""v6 Hard Negative Mining — Pre-compute FAISS cosine collisions.

For each answerable NQ question:
  1. Encode question with sentence-transformer
  2. FAISS search → top-50 cosine-similar chunks
  3. Filter to chunks from WRONG articles only
  4. Save the chunk indices as a static cache

The training loop reads from this cache instead of querying FAISS live.
This keeps the 55ms/step GPU throughput intact.

Output: results/v6_checkpoints/hard_negatives.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import faiss

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
OUTPUT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mine_hard_negatives(n_queries: int = 5000, k_faiss: int = 50):
    """Pre-mine hard negatives for training."""
    start = time.time()

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading base map...")
    index = faiss.read_index(str(BASE_MAP_DIR / "faiss_index.bin"))
    if hasattr(index, 'nprobe'):
        index.nprobe = 64

    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        all_qa = json.load(f)
    with open(BASE_MAP_DIR / "chunk_meta.json") as f:
        chunk_meta = json.load(f)
    with open(BASE_MAP_DIR / "articles.json") as f:
        article_titles = json.load(f)

    answerable = [q for q in all_qa if q["has_answer"]]
    title_to_idx = {t: i for i, t in enumerate(article_titles)}

    # Build article → chunk lookup
    art_to_chunks = {}
    for i, m in enumerate(chunk_meta):
        art_to_chunks.setdefault(m["article_idx"], []).append(i)

    # ── Encode questions ──────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print(f"Encoding {min(n_queries, len(answerable))} questions...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    sample = answerable[:n_queries]
    questions = [q["question"] for q in sample]
    q_embeddings = encoder.encode(questions, convert_to_numpy=True,
                                   show_progress_bar=True, batch_size=256)

    # Normalize for FAISS inner product
    q_norm = q_embeddings.copy()
    faiss.normalize_L2(q_norm)

    # ── FAISS batch search ────────────────────────────────────────────
    print(f"FAISS search: {len(questions)} queries × top-{k_faiss}...")
    scores, indices = index.search(q_norm, k_faiss)

    # ── Filter to wrong-article chunks ────────────────────────────────
    print("Filtering to hard negatives (wrong article, high cosine)...")
    cache = []
    n_with_negatives = 0

    for i, qa in enumerate(sample):
        correct_art = title_to_idx.get(qa["title"])
        if correct_art is None:
            continue

        hard_negs = []
        for rank in range(k_faiss):
            idx = int(indices[i][rank])
            if idx < 0:
                continue
            chunk_art = chunk_meta[idx]["article_idx"]
            if chunk_art != correct_art:
                hard_negs.append({
                    "chunk_idx": idx,
                    "cosine_sim": float(scores[i][rank]),
                    "article_idx": chunk_art,
                    "cosine_rank": rank,
                })

        if hard_negs:
            n_with_negatives += 1

        # Also store positive chunk indices (from correct article)
        pos_chunks = art_to_chunks.get(correct_art, [])

        cache.append({
            "qa_idx": i,
            "question": qa["question"][:100],
            "answer": qa["answer"][:60],
            "title": qa["title"],
            "correct_art": correct_art,
            "pos_chunk_ids": pos_chunks[:20],  # cap for JSON size
            "hard_neg_chunks": hard_negs[:20],  # top-20 hardest
        })

    elapsed = time.time() - start

    # ── Save ──────────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "hard_negatives.json"
    with open(output_path, "w") as f:
        json.dump(cache, f)

    # Stats
    avg_negs = np.mean([len(c["hard_neg_chunks"]) for c in cache])
    avg_pos = np.mean([len(c["pos_chunk_ids"]) for c in cache])
    avg_cos = np.mean([c["hard_neg_chunks"][0]["cosine_sim"]
                        for c in cache if c["hard_neg_chunks"]])

    print(f"\n{'='*60}")
    print(f"  HARD NEGATIVE MINING COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  {len(cache):,} QA pairs mined")
    print(f"  {n_with_negatives:,} have hard negatives ({n_with_negatives/len(cache)*100:.0f}%)")
    print(f"  Avg hard negatives per query: {avg_negs:.1f}")
    print(f"  Avg positive chunks per query: {avg_pos:.1f}")
    print(f"  Avg top-1 hard neg cosine sim: {avg_cos:.3f}")
    print(f"  Saved → {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=5000)
    parser.add_argument("--k_faiss", type=int, default=50)
    args = parser.parse_args()
    mine_hard_negatives(args.n_queries, args.k_faiss)
