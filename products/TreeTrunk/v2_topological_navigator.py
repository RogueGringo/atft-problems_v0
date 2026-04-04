#!/usr/bin/env python3
"""v2 Topological Navigator — H0-ranked navigation, not vector arithmetic.

Upgrade from v1:
  v1: mean displacement → nearest neighbor (semantic retrieval)
  v2: FAISS neighborhood → H0 tightness ranking (topological navigation)

The engine doesn't pick the closest chunk. It picks the chunk that
forms the TIGHTEST topological path from input to output — the one
with the shortest H0 persistence bars, meaning maximum structural
coherence between question and answer.

Pipeline:
  1. FAISS index over base map (fast neighborhood lookup)
  2. For each query: find top-K candidates via FAISS
  3. For each candidate: build mini point cloud [query_path + candidate_path]
  4. Compute H0 persistence on each mini cloud
  5. Rank by H0 tightness (shortest bars = most coherent)
  6. Return the topologically tightest answer

This is where retrieval becomes navigation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import faiss

OUTPUT_DIR = Path(__file__).parent / "results" / "v2_navigator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Base Map with FAISS index
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexedBaseMap:
    """Knowledge substrate with FAISS index for fast neighborhood lookup."""
    embeddings: np.ndarray      # (N, d)
    texts: list[str]            # (N,)
    index: object               # FAISS index
    chunk_sources: list[int]    # which source problem each chunk came from
    metadata: dict


def build_indexed_base_map(encoder, n_chunks=10000, chunk_words=30, stride_words=15):
    """Build base map from GSM8K with FAISS index."""
    from datasets import load_dataset

    print(f"Building indexed base map ({n_chunks} chunks)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    chunks = []
    sources = []
    for idx, item in enumerate(ds):
        full = item["question"] + "\n" + item["answer"]
        words = full.split()
        for i in range(0, max(1, len(words) - chunk_words + 1), stride_words):
            chunk = " ".join(words[i:i + chunk_words])
            if len(chunk.split()) >= 8:
                chunks.append(chunk)
                sources.append(idx)
            if len(chunks) >= n_chunks:
                break
        if len(chunks) >= n_chunks:
            break

    chunks = chunks[:n_chunks]
    sources = sources[:n_chunks]
    print(f"  {len(chunks)} chunks from {len(set(sources))} problems")

    print(f"  Encoding...")
    embeddings = encoder.encode(chunks, convert_to_numpy=True,
                                show_progress_bar=True, batch_size=128)

    # Build FAISS index (L2 on normalized vectors = cosine)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)  # inner product on normalized = cosine
    index.add(embeddings)

    print(f"  FAISS index: {index.ntotal} vectors in R^{d}")

    return IndexedBaseMap(
        embeddings=embeddings,
        texts=chunks,
        index=index,
        chunk_sources=sources,
        metadata={"n_chunks": len(chunks), "dim": d},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Topological Navigator
# ══════════════════════════════════════════════════════════════════════════════

def compute_h0_tightness(cloud: np.ndarray) -> float:
    """Compute H0 tightness = mean bar length of H0 persistence.

    Lower = tighter = more coherent path.
    """
    from ripser import ripser
    if len(cloud) < 2:
        return float('inf')
    if len(cloud) > 80:
        idx = np.random.choice(len(cloud), 80, replace=False)
        cloud = cloud[idx]

    result = ripser(cloud, maxdim=0)
    dgm = result["dgms"][0]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) == 0:
        return float('inf')
    bars = finite[:, 1] - finite[:, 0]
    return float(bars.mean())


def embed_fine(text: str, encoder, window=5, stride=3) -> np.ndarray:
    """Embed text as fine-grained point cloud (sliding window)."""
    words = text.split()
    if len(words) <= window:
        chunks = [text]
    else:
        chunks = [" ".join(words[i:i+window])
                  for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


@dataclass
class NavResult:
    """Result of topological navigation."""
    query: str
    candidates_checked: int
    best_text: str
    best_h0: float
    best_idx: int
    best_cosine: float
    # Comparison: cosine-best vs topo-best
    cosine_best_text: str
    cosine_best_h0: float
    cosine_best_idx: int
    # Full ranking
    rankings: list[dict] = field(default_factory=list)


def navigate_topological(
    query: str,
    base_map: IndexedBaseMap,
    encoder,
    k_candidates: int = 50,
    k_topo_eval: int = 20,
) -> NavResult:
    """Navigate from query to answer using H0 tightness.

    1. FAISS: find top-K candidates by cosine similarity (fast)
    2. For each of top-K_eval candidates:
       - Build mini cloud: [query fine-grained] + [candidate fine-grained]
       - Compute H0 tightness
    3. Rank by H0 (lower = more coherent)
    4. Return the topologically tightest candidate

    Compares cosine-best vs topo-best to show when topology disagrees
    with pure vector similarity.
    """
    # 1. Embed query
    q_emb = encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # 2. FAISS lookup
    scores, indices = base_map.index.search(q_emb, k_candidates)
    scores = scores[0]   # (k_candidates,)
    indices = indices[0]  # (k_candidates,)

    # Cosine best (what standard retrieval would pick)
    cosine_best_idx = int(indices[0])
    cosine_best_text = base_map.texts[cosine_best_idx]

    # 3. Fine-grained embedding of query
    q_cloud = embed_fine(query, encoder)

    # 4. Evaluate top-K_eval candidates topologically
    rankings = []
    for rank in range(min(k_topo_eval, len(indices))):
        c_idx = int(indices[rank])
        c_text = base_map.texts[c_idx]
        c_cosine = float(scores[rank])

        # Build mini cloud: query path + candidate path
        c_cloud = embed_fine(c_text, encoder)
        combined = np.vstack([q_cloud, c_cloud])

        h0 = compute_h0_tightness(combined)

        rankings.append({
            "idx": c_idx,
            "text": c_text[:200],
            "cosine": c_cosine,
            "h0": h0,
            "cosine_rank": rank,
        })

    # 5. Rank by H0 tightness (ascending — lower = tighter)
    rankings.sort(key=lambda r: r["h0"])

    # Topo best
    best = rankings[0]

    # H0 of cosine best
    cosine_best_entry = next(r for r in rankings if r["idx"] == cosine_best_idx)

    return NavResult(
        query=query,
        candidates_checked=len(rankings),
        best_text=best["text"],
        best_h0=best["h0"],
        best_idx=best["idx"],
        best_cosine=best["cosine"],
        cosine_best_text=cosine_best_text[:200],
        cosine_best_h0=cosine_best_entry["h0"],
        cosine_best_idx=cosine_best_idx,
        rankings=rankings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_v2(n_base=10000, n_test=20, k_candidates=50, k_topo=20):
    start = time.time()

    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build indexed base map
    base_map = build_indexed_base_map(encoder, n_chunks=n_base)

    # Load test set
    print(f"\nNavigating {n_test} test questions (K={k_candidates}, topo_eval={k_topo})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(ds.select(range(min(n_test, len(ds)))))

    results = []
    topo_disagrees = 0
    topo_wins_h0 = 0

    for i, item in enumerate(test_items):
        q = item["question"]
        true_answer = item["answer"]
        true_final = true_answer.split("####")[-1].strip() if "####" in true_answer else ""

        nav = navigate_topological(q, base_map, encoder, k_candidates, k_topo)

        # Did topology pick a different answer than cosine?
        disagree = nav.best_idx != nav.cosine_best_idx
        if disagree:
            topo_disagrees += 1
            if nav.best_h0 < nav.cosine_best_h0:
                topo_wins_h0 += 1

        print(f"\n{'─'*70}")
        print(f"  Q{i}: {q[:90]}...")
        print(f"  True: #### {true_final}")
        print(f"  Cosine best (H0={nav.cosine_best_h0:.4f}): {nav.cosine_best_text[:100]}...")
        print(f"  Topo  best  (H0={nav.best_h0:.4f}): {nav.best_text[:100]}...")
        if disagree:
            print(f"  >>> TOPOLOGY DISAGREES — picked different chunk (ΔH0={nav.cosine_best_h0 - nav.best_h0:.4f})")
        else:
            print(f"  === Topology agrees with cosine")

        results.append({
            "question": q[:200],
            "true_final": true_final,
            "cosine_best": nav.cosine_best_text[:200],
            "cosine_h0": nav.cosine_best_h0,
            "topo_best": nav.best_text[:200],
            "topo_h0": nav.best_h0,
            "disagree": disagree,
            "topo_cosine_rank": next(
                r["cosine_rank"] for r in nav.rankings if r["idx"] == nav.best_idx
            ),
        })

    elapsed = time.time() - start

    # Summary
    n_disagree = sum(1 for r in results if r["disagree"])
    avg_topo_h0 = np.mean([r["topo_h0"] for r in results])
    avg_cosine_h0 = np.mean([r["cosine_h0"] for r in results])

    # When topo disagrees, what cosine rank did topo pick?
    disagree_ranks = [r["topo_cosine_rank"] for r in results if r["disagree"]]

    print(f"\n{'='*70}")
    print(f"  V2 TOPOLOGICAL NAVIGATOR — RESULTS")
    print(f"{'='*70}")
    print(f"  {n_test} queries | {elapsed:.0f}s | base={n_base} chunks")
    print(f"  Candidates per query: {k_candidates} (FAISS) → {k_topo} (topo eval)")
    print(f"")
    print(f"  Topology disagrees with cosine: {n_disagree}/{n_test} ({n_disagree/n_test*100:.0f}%)")
    print(f"  Avg H0 (cosine pick):  {avg_cosine_h0:.4f}")
    print(f"  Avg H0 (topo pick):    {avg_topo_h0:.4f}")
    print(f"  H0 improvement:        {(avg_cosine_h0 - avg_topo_h0)/avg_cosine_h0*100:.1f}%")
    if disagree_ranks:
        print(f"  When topo disagrees, it picks cosine rank: "
              f"mean={np.mean(disagree_ranks):.1f}, range={min(disagree_ranks)}-{max(disagree_ranks)}")
    print(f"{'='*70}")

    log = {
        "experiment": "v2_topological_navigator",
        "elapsed": elapsed,
        "n_test": n_test,
        "k_candidates": k_candidates,
        "k_topo": k_topo,
        "n_disagree": n_disagree,
        "avg_h0_cosine": float(avg_cosine_h0),
        "avg_h0_topo": float(avg_topo_h0),
        "results": results,
    }
    with open(OUTPUT_DIR / "v2_results.json", "w") as f:
        json.dump(log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Saved → {OUTPUT_DIR / 'v2_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_base", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=30)
    parser.add_argument("--k_candidates", type=int, default=50)
    parser.add_argument("--k_topo", type=int, default=20)
    args = parser.parse_args()
    run_v2(args.n_base, args.n_test, args.k_candidates, args.k_topo)
