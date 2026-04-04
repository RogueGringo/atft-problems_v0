#!/usr/bin/env python3
"""v3 Sheaf Navigator — H0 tightness + Sheaf Laplacian truth filter.

Upgrade from v2:
  v2: FAISS → H0 ranking (topological tightness)
  v3: FAISS → H0 ranking → Sheaf Laplacian filter (logical consistency)

The Sheaf Laplacian L_F encodes whether a path through embedding space
is GLOBALLY CONSISTENT. A low spectral gap λ₁ means the path forms
a coherent global section — no logical contradictions or hallucinated leaps.

Pipeline:
  1. FAISS: top-50 cosine candidates
  2. H0 filter: rank by persistence tightness, take top-5
  3. Sheaf Laplacian: for each top-5, construct the cellular sheaf over the
     Q→A point cloud, compute L_F, measure spectral gap λ₁
  4. Select the candidate with smallest λ₁ (most globally consistent)

Three layers of filtration: semantic → topological → logical.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import faiss

OUTPUT_DIR = Path(__file__).parent / "results" / "v3_sheaf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sheaf Laplacian on Point Clouds
# ══════════════════════════════════════════════════════════════════════════════

def sheaf_laplacian_from_cloud(
    points: np.ndarray,
    k_neighbors: int = 5,
    stalk_dim: int = 8,
) -> tuple[np.ndarray, dict]:
    """Construct the BLOCK Sheaf Laplacian L_F for a point cloud.

    The cellular sheaf on the k-NN graph:
      - Vertex stalks: R^s (PCA-reduced, s = stalk_dim)
      - Edge stalks: R^s
      - Restriction maps: orthogonal projection of v_i onto v_j direction
        If concepts flow logically, projection is near-identity.
        If there's a hallucinated leap, projection rotates violently.

    Construction:
      1. PCA reduce to stalk_dim dimensions
      2. Build k-NN graph
      3. For each edge (i,j): compute restriction map as the rotation
         that aligns the local frame at i with the local frame at j
      4. Build coboundary δ⁰ as block matrix
      5. L_F = (δ⁰)^T @ δ⁰

    Returns (L_F eigenvalues, diagnostics dict).
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist

    n = len(points)
    if n < 3:
        return np.zeros((1, 1)), {"spectral_gap": float('inf'), "kernel_dim": 0}

    # 1. PCA reduction to stalk_dim
    s = min(stalk_dim, points.shape[1], n - 1)
    pca = PCA(n_components=s)
    X = pca.fit_transform(points)  # (n, s)

    # 2. k-NN graph
    k = min(k_neighbors, n - 1)
    dists = cdist(X, X)
    edges = []
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:k+1]
        for j in neighbors:
            if i < j:
                edges.append((i, j))
    # Deduplicate
    edges = list(set(edges))
    m = len(edges)

    if m == 0:
        return np.zeros((n * s, n * s)), {"spectral_gap": float('inf'), "kernel_dim": 0}

    # 3. Restriction maps: for edge (i,j), compute the rotation
    #    that best aligns the neighborhood of i with neighborhood of j.
    #    Simplified: use the outer product of normalized vectors as
    #    a rank-1 projection that captures directional alignment.
    #
    #    F_{i→e} = I (identity — project i's stalk directly)
    #    F_{j→e} = R_{ij} (rotation aligning j's local frame to i's)
    #
    #    For efficiency: R_{ij} = I - 2 * (d_hat @ d_hat^T) where
    #    d_hat = normalized displacement. This is a Householder reflection
    #    that captures how much the semantic direction changes.

    # 4. Build coboundary δ⁰: (m*s) × (n*s) block matrix
    #    For edge e_k = (i, j): row block k has F_{i→e} at column block i
    #    and -F_{j→e} at column block j
    delta = np.zeros((m * s, n * s))

    for k_edge, (i, j) in enumerate(edges):
        d = X[j] - X[i]
        d_norm = np.linalg.norm(d)

        if d_norm < 1e-10:
            # Coincident points — identity restriction
            R_ij = np.eye(s)
        else:
            d_hat = d / d_norm
            # Householder-like restriction: captures directional change
            # R = I - alpha * d_hat @ d_hat^T
            # alpha scales with distance (closer = more identity-like)
            alpha = min(1.0, d_norm)  # cap at 1 to keep bounded
            R_ij = np.eye(s) - alpha * np.outer(d_hat, d_hat)

        # δ⁰[e_k, i] = I (identity restriction from vertex i)
        r0 = k_edge * s
        c_i = i * s
        c_j = j * s
        delta[r0:r0+s, c_i:c_i+s] = np.eye(s)
        delta[r0:r0+s, c_j:c_j+s] = -R_ij

    # 5. L_F = δ^T @ δ
    L = delta.T @ delta  # (n*s, n*s)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(np.abs(eigenvalues))

    tol = 1e-6
    nonzero = eigenvalues[eigenvalues > tol]
    gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0
    ker = int(np.sum(eigenvalues < tol))

    diagnostics = {
        "spectral_gap": gap,
        "kernel_dim": ker,
        "n_points": n,
        "n_edges": m,
        "stalk_dim": s,
        "L_size": L.shape[0],
    }

    return eigenvalues, diagnostics


def spectral_gap_from_cloud(points, k_neighbors=5, stalk_dim=8):
    """Convenience: compute spectral gap of sheaf Laplacian on point cloud."""
    _, diag = sheaf_laplacian_from_cloud(points, k_neighbors, stalk_dim)
    return diag["spectral_gap"], diag["kernel_dim"]


# ══════════════════════════════════════════════════════════════════════════════
# Base Map + Navigation (reuse from v2)
# ══════════════════════════════════════════════════════════════════════════════

def build_indexed_base_map(encoder, n_chunks=10000, chunk_words=30, stride_words=15):
    from datasets import load_dataset
    print(f"Building indexed base map ({n_chunks} chunks)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    chunks, sources = [], []
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
    d = embeddings.shape[1]
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    print(f"  FAISS index: {index.ntotal} vectors in R^{d}")
    return {"embeddings": embeddings, "embeddings_norm": emb_norm,
            "texts": chunks, "index": index, "sources": sources}


def embed_fine(text, encoder, window=5, stride=3):
    words = text.split()
    if len(words) <= window:
        chunks = [text]
    else:
        chunks = [" ".join(words[i:i+window])
                  for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def compute_h0_tightness(cloud):
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
    return float((finite[:, 1] - finite[:, 0]).mean())


# ══════════════════════════════════════════════════════════════════════════════
# v3 Three-Layer Navigator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class V3Result:
    query: str
    # Layer 1: Cosine (FAISS)
    cosine_best_text: str
    cosine_best_sim: float
    # Layer 2: H0 tightness
    h0_best_text: str
    h0_best_score: float
    h0_cosine_rank: int
    # Layer 3: Sheaf Laplacian
    sheaf_best_text: str
    sheaf_best_lambda1: float
    sheaf_best_kernel_dim: int
    sheaf_h0_rank: int
    # All candidates with full scores
    candidates: list[dict] = field(default_factory=list)


def navigate_v3(
    query: str,
    base_map: dict,
    encoder,
    k_faiss: int = 50,
    k_h0: int = 5,
    k_neighbors_sheaf: int = 4,
) -> V3Result:
    """Three-layer navigation: cosine → H0 → Sheaf Laplacian.

    1. FAISS: top k_faiss by cosine similarity
    2. H0: evaluate persistence tightness, take top k_h0
    3. Sheaf: construct L_F for each, pick lowest spectral gap
    """
    # ── Layer 1: FAISS cosine ────────────────────────────────────────
    q_emb = encoder.encode([query], convert_to_numpy=True)
    q_norm = q_emb.copy()
    faiss.normalize_L2(q_norm)

    scores, indices = base_map["index"].search(q_norm, k_faiss)
    scores, indices = scores[0], indices[0]

    cosine_best_text = base_map["texts"][indices[0]]
    cosine_best_sim = float(scores[0])

    # ── Layer 2: H0 tightness ────────────────────────────────────────
    q_cloud = embed_fine(query, encoder)

    h0_candidates = []
    for rank in range(min(20, len(indices))):  # evaluate top-20 for H0
        c_idx = int(indices[rank])
        c_text = base_map["texts"][c_idx]
        c_cloud = embed_fine(c_text, encoder)
        combined = np.vstack([q_cloud, c_cloud])
        h0 = compute_h0_tightness(combined)
        h0_candidates.append({
            "idx": c_idx, "text": c_text[:200], "cosine_rank": rank,
            "cosine_sim": float(scores[rank]), "h0": h0,
        })

    # Rank by H0 (ascending)
    h0_candidates.sort(key=lambda c: c["h0"])
    h0_best = h0_candidates[0]

    # ── Layer 3: Sheaf Laplacian on top k_h0 ─────────────────────────
    sheaf_candidates = h0_candidates[:k_h0]

    for cand in sheaf_candidates:
        c_cloud = embed_fine(cand["text"], encoder)
        combined = np.vstack([q_cloud, c_cloud])

        # Subsample if needed for Laplacian efficiency
        if len(combined) > 60:
            idx = np.random.choice(len(combined), 60, replace=False)
            combined = combined[idx]

        lambda1, ker_dim = spectral_gap_from_cloud(
            combined, k_neighbors=k_neighbors_sheaf, stalk_dim=8
        )

        cand["lambda1"] = lambda1
        cand["kernel_dim"] = ker_dim

    # Rank by lambda1 (ascending — lowest spectral gap = most consistent)
    sheaf_candidates.sort(key=lambda c: c["lambda1"])
    sheaf_best = sheaf_candidates[0]

    return V3Result(
        query=query,
        cosine_best_text=cosine_best_text[:200],
        cosine_best_sim=cosine_best_sim,
        h0_best_text=h0_best["text"][:200],
        h0_best_score=h0_best["h0"],
        h0_cosine_rank=h0_best["cosine_rank"],
        sheaf_best_text=sheaf_best["text"][:200],
        sheaf_best_lambda1=sheaf_best["lambda1"],
        sheaf_best_kernel_dim=sheaf_best["kernel_dim"],
        sheaf_h0_rank=sheaf_candidates.index(sheaf_best),
        candidates=sheaf_candidates,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_v3(n_base=10000, n_test=30):
    start = time.time()

    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    base_map = build_indexed_base_map(encoder, n_chunks=n_base)

    print(f"\nNavigating {n_test} test questions (3-layer: cosine → H0 → sheaf)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(ds.select(range(min(n_test, len(ds)))))

    results = []
    layer_disagree = {"cosine_vs_h0": 0, "h0_vs_sheaf": 0, "cosine_vs_sheaf": 0}

    for i, item in enumerate(test_items):
        q = item["question"]
        true_final = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else ""

        nav = navigate_v3(q, base_map, encoder)

        # Track disagreements between layers
        cosine_text = nav.cosine_best_text
        h0_text = nav.h0_best_text
        sheaf_text = nav.sheaf_best_text

        c_vs_h = cosine_text != h0_text
        h_vs_s = h0_text != sheaf_text
        c_vs_s = cosine_text != sheaf_text

        if c_vs_h: layer_disagree["cosine_vs_h0"] += 1
        if h_vs_s: layer_disagree["h0_vs_sheaf"] += 1
        if c_vs_s: layer_disagree["cosine_vs_sheaf"] += 1

        print(f"\n{'─'*70}")
        print(f"  Q{i}: {q[:80]}...")
        print(f"  True: #### {true_final}")
        print(f"  L1 Cosine:  {cosine_text[:80]}...")
        print(f"  L2 H0:     {h0_text[:80]}...")
        print(f"  L3 Sheaf:  {sheaf_text[:80]}...")
        print(f"  Sheaf λ₁={nav.sheaf_best_lambda1:.4f} ker={nav.sheaf_best_kernel_dim}")

        # Show all sheaf candidates
        print(f"  Sheaf ranking (top {len(nav.candidates)}):")
        for j, c in enumerate(nav.candidates):
            marker = " ◄" if j == 0 else ""
            print(f"    [{j}] λ₁={c['lambda1']:.4f} H0={c['h0']:.4f} "
                  f"cos_rank={c['cosine_rank']} ker={c['kernel_dim']}{marker}")

        results.append({
            "question": q[:200],
            "true_final": true_final,
            "cosine_text": cosine_text,
            "h0_text": h0_text,
            "sheaf_text": sheaf_text,
            "sheaf_lambda1": nav.sheaf_best_lambda1,
            "sheaf_kernel_dim": nav.sheaf_best_kernel_dim,
            "cosine_vs_h0": c_vs_h,
            "h0_vs_sheaf": h_vs_s,
            "cosine_vs_sheaf": c_vs_s,
            "candidates": nav.candidates,
        })

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*70}")
    print(f"  V3 SHEAF NAVIGATOR — RESULTS")
    print(f"{'='*70}")
    print(f"  {n_test} queries | {elapsed:.0f}s | base={n_base} chunks")
    print(f"")
    print(f"  Layer disagreements:")
    print(f"    Cosine vs H0:    {layer_disagree['cosine_vs_h0']}/{n_test} "
          f"({layer_disagree['cosine_vs_h0']/n_test*100:.0f}%)")
    print(f"    H0 vs Sheaf:     {layer_disagree['h0_vs_sheaf']}/{n_test} "
          f"({layer_disagree['h0_vs_sheaf']/n_test*100:.0f}%)")
    print(f"    Cosine vs Sheaf: {layer_disagree['cosine_vs_sheaf']}/{n_test} "
          f"({layer_disagree['cosine_vs_sheaf']/n_test*100:.0f}%)")
    print(f"")

    avg_lambda1 = np.mean([r["sheaf_lambda1"] for r in results])
    avg_ker = np.mean([r["sheaf_kernel_dim"] for r in results])
    print(f"  Avg spectral gap λ₁: {avg_lambda1:.4f}")
    print(f"  Avg kernel dimension: {avg_ker:.1f}")
    print(f"{'='*70}")

    log = {
        "experiment": "v3_sheaf_navigator",
        "elapsed": elapsed,
        "n_test": n_test,
        "layer_disagree": layer_disagree,
        "avg_lambda1": float(avg_lambda1),
        "avg_kernel_dim": float(avg_ker),
        "results": results,
    }
    with open(OUTPUT_DIR / "v3_results.json", "w") as f:
        json.dump(log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Saved → {OUTPUT_DIR / 'v3_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_base", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=30)
    args = parser.parse_args()
    run_v3(args.n_base, args.n_test)
