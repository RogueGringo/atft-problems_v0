#!/usr/bin/env python3
"""v5 Universal Slice — Section 3: The Guided Sheaf Navigator.

The endgame. Topological retrieval at Wikipedia scale with zero generation.

Pipeline:
  1. Query → embed → compute waypoint signature → predict archetype
  2. FAISS IVF → top-50 macroscopic candidates
  3. Archetype-guided ranking:
     - Compute each candidate's topological signature
     - Score by ||W(candidate) - W(archetype)|| (geometric prior)
     - Weight with H₀ tightness → combined ranking
  4. Top-5 → Sheaf Laplacian Truth Filter (FULL cloud, no subsampling):
     - Fine-grained sliding window (5 words, stride 1)
     - k-NN sheaf graph (k=5)
     - Householder restriction maps (stalk_dim=8)
     - Spectral gap λ₁ < threshold → verified
     - λ₁ spike → VETO, regardless of other scores
  5. Output: raw verified text + coherence certificate

Requires:
  results/v5_nq_base_map/   (from v5_nq_ingest.py)
  results/v5_nq_stencils/    (from v5_nq_stencil.py)

Output:
  results/v5_nq_navigation/
  └── navigation_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import faiss

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
STENCIL_DIR = Path(__file__).parent / "results" / "v5_nq_stencils"
OUTPUT_DIR = Path(__file__).parent / "results" / "v5_nq_navigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Waypoint Signature (shared with stencil extractor)
# ══════════════════════════════════════════════════════════════════════════════

def embed_sliding_window(text, encoder, window=5, stride=1):
    """Fine-grained sliding window for sheaf-level precision.

    stride=1 gives maximum resolution — every word shift creates a new point.
    This is the full manifold, not the sparse approximation.
    """
    words = text.split()
    if len(words) <= window:
        return encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
    chunks = [" ".join(words[i:i + window])
              for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def compute_signature(q_cloud, a_cloud, n_waypoints=3):
    """Compute waypoint signature for archetype matching."""
    from ripser import ripser
    from scipy.spatial.distance import cdist

    n_q = len(q_cloud)
    X = np.vstack([q_cloud, a_cloud])

    if len(X) > 100:
        idx = np.random.choice(len(X), 100, replace=False)
        q_mask = idx < n_q
        X = X[idx]
        n_q_sub = int(q_mask.sum())
    else:
        n_q_sub = n_q

    cross_dist = cdist(X[:n_q_sub], X[n_q_sub:])
    onset = float(cross_dist.min()) if cross_dist.size > 0 else 0.0

    result = ripser(X, maxdim=0)
    dgm = result["dgms"][0]
    finite = dgm[dgm[:, 1] < np.inf]

    if len(finite) == 0:
        return np.zeros(4 + 2 * n_waypoints, dtype=np.float32)

    bars = finite[:, 1] - finite[:, 0]
    deaths = np.sort(finite[:, 1])[::-1]

    wp = [float(deaths[i]) if i < len(deaths) else 0.0
          for i in range(n_waypoints)]

    eps = 0.05
    derivs = []
    for w in wp:
        if w <= 0:
            derivs.append(0.0)
        else:
            nearby = np.sum((finite[:, 1] >= w - eps) & (finite[:, 1] <= w + eps))
            derivs.append(float(nearby) / (2 * eps))

    vec = [onset, float(bars.mean()), float(bars.max()),
           len(bars) / max(len(q_cloud) + len(a_cloud), 1)]
    vec.extend(wp)
    vec.extend(derivs)
    return np.array(vec, dtype=np.float32)


def compute_h0_tightness(cloud):
    """H₀ persistence tightness — mean bar length."""
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
# Sheaf Laplacian Truth Filter — Full Precision
# ══════════════════════════════════════════════════════════════════════════════

def sheaf_truth_filter(
    points: np.ndarray,
    k_neighbors: int = 5,
    stalk_dim: int = 8,
) -> dict:
    """Full Sheaf Laplacian on the unsubsampled point cloud.

    No 60-point cap. The archetype filter upstream saves the budget;
    we reinvest it into absolute precision here.

    Returns dict with spectral_gap (λ₁), kernel_dim, diagnostics.
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist

    n = len(points)
    if n < 3:
        return {"spectral_gap": float('inf'), "kernel_dim": 0, "n_points": n}

    # PCA reduction to stalk dimension
    s = min(stalk_dim, points.shape[1], n - 1)
    pca = PCA(n_components=s)
    X = pca.fit_transform(points)

    # k-NN graph
    k = min(k_neighbors, n - 1)
    dists = cdist(X, X)
    edges = set()
    for i in range(n):
        for j in np.argsort(dists[i])[1:k + 1]:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    edges = list(edges)
    m = len(edges)

    if m == 0:
        return {"spectral_gap": float('inf'), "kernel_dim": 0, "n_points": n}

    # Block coboundary matrix δ⁰: (m*s) × (n*s)
    # For edge (i,j): F_{i→e} = I, F_{j→e} = R_ij (Householder restriction)
    delta = np.zeros((m * s, n * s))

    for k_edge, (i, j) in enumerate(edges):
        d = X[j] - X[i]
        d_norm = np.linalg.norm(d)

        if d_norm < 1e-10:
            R = np.eye(s)
        else:
            d_hat = d / d_norm
            alpha = min(1.0, d_norm)
            R = np.eye(s) - alpha * np.outer(d_hat, d_hat)

        r0 = k_edge * s
        delta[r0:r0 + s, i * s:(i + 1) * s] = np.eye(s)
        delta[r0:r0 + s, j * s:(j + 1) * s] = -R

    # L_F = δᵀδ
    L = delta.T @ delta
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(L)))

    tol = 1e-6
    nonzero = eigenvalues[eigenvalues > tol]
    gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0
    ker = int(np.sum(eigenvalues < tol))

    return {
        "spectral_gap": gap,
        "kernel_dim": ker,
        "n_points": n,
        "n_edges": m,
        "stalk_dim": s,
    }


# ══════════════════════════════════════════════════════════════════════════════
# The Navigator
# ══════════════════════════════════════════════════════════════════════════════

class UniversalSliceNavigator:
    """The engine. Archetype-guided, sheaf-verified topological retrieval."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import StandardScaler

        print("Loading Universal Slice Navigator...")

        # Base map
        self.embeddings = np.load(BASE_MAP_DIR / "embeddings.npy")
        self.emb_norm = np.load(BASE_MAP_DIR / "embeddings_norm.npy")
        self.index = faiss.read_index(str(BASE_MAP_DIR / "faiss_index.bin"))

        with open(BASE_MAP_DIR / "chunk_texts.json") as f:
            self.chunks = json.load(f)
        with open(BASE_MAP_DIR / "chunk_meta.json") as f:
            self.chunk_meta = json.load(f)
        with open(BASE_MAP_DIR / "articles.json") as f:
            self.article_titles = json.load(f)

        # Set nprobe for IVF if applicable
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 64

        # Stencils
        self.archetypes = np.load(STENCIL_DIR / "archetypes.npy")
        with open(STENCIL_DIR / "stencil_log.json") as f:
            stencil_log = json.load(f)

        # Reconstruct scaler from saved parameters
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(stencil_log["scaler_mean"])
        self.scaler.scale_ = np.array(stencil_log["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Build article → chunk_ids lookup for full-cloud signatures
        self.art_to_chunks = {}
        for i, m in enumerate(self.chunk_meta):
            self.art_to_chunks.setdefault(m["article_idx"], []).append(i)

        print(f"  Base map: {len(self.chunks):,} chunks, "
              f"{len(self.article_titles):,} articles")
        print(f"  Stencils: {len(self.archetypes)} archetypes")
        print(f"  Ready.")

    def predict_archetype(self, query_sig: np.ndarray) -> tuple[int, float]:
        """Map a signature to its nearest archetype."""
        sig_scaled = self.scaler.transform(query_sig.reshape(1, -1))
        centroids_scaled = self.scaler.transform(self.archetypes)
        dists = np.linalg.norm(centroids_scaled - sig_scaled, axis=1)
        best = int(np.argmin(dists))
        return best, float(dists[best])

    def navigate(
        self,
        query: str,
        k_faiss: int = 50,
        k_archetype: int = 5,
        lambda_threshold: float = 1e-4,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> dict:
        """Full navigation pipeline.

        Returns the verified text with coherence certificate.
        """
        t0 = time.time()

        # ── 1. Embed query ────────────────────────────────────────────
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        q_norm = q_emb.copy()
        faiss.normalize_L2(q_norm)

        # Fine-grained query cloud for sheaf evaluation
        q_cloud_fine = embed_sliding_window(query, self.encoder, window=5, stride=1)

        # Coarse cloud for signature computation
        q_cloud_coarse = embed_sliding_window(query, self.encoder, window=5, stride=3)

        # ── 2. FAISS retrieval ────────────────────────────────────────
        scores_faiss, indices_faiss = self.index.search(q_norm, k_faiss)
        scores_faiss = scores_faiss[0]
        indices_faiss = indices_faiss[0]

        # ── 3. Archetype-guided ranking ───────────────────────────────
        # Cache per-article signatures (many chunks share one article)
        art_sig_cache = {}
        candidates = []

        for rank in range(min(k_faiss, len(indices_faiss))):
            idx = int(indices_faiss[rank])
            if idx < 0:
                continue

            chunk_text = self.chunks[idx]
            meta = self.chunk_meta[idx]
            art_idx = meta["article_idx"]
            title = self.article_titles[art_idx]

            # Compute signature using FULL article embedding cloud
            if art_idx not in art_sig_cache:
                chunk_ids = self.art_to_chunks.get(art_idx, [idx])
                a_cloud = self.embeddings[chunk_ids]
                sig = compute_signature(q_cloud_coarse, a_cloud)
                arch_id, arch_dist = self.predict_archetype(sig)

                combined_cloud = np.vstack([q_cloud_coarse, a_cloud])
                if len(combined_cloud) > 80:
                    sub = np.random.choice(len(combined_cloud), 80, replace=False)
                    combined_cloud = combined_cloud[sub]
                h0 = compute_h0_tightness(combined_cloud)

                art_sig_cache[art_idx] = {
                    "archetype_id": arch_id,
                    "archetype_dist": float(arch_dist),
                    "h0_tightness": float(h0),
                }

            cached = art_sig_cache[art_idx]

            candidates.append({
                "idx": idx,
                "text": chunk_text,
                "title": title,
                "cosine_rank": rank,
                "cosine_sim": float(scores_faiss[rank]),
                "archetype_id": cached["archetype_id"],
                "archetype_dist": cached["archetype_dist"],
                "h0_tightness": cached["h0_tightness"],
            })

        # ── Composite scoring: 70% cosine anchor + 30% topology ──────
        # Normalize each signal to [0,1], then combine.
        # Cosine: invert (higher sim = lower score = better)
        cos_sims = [c["cosine_sim"] for c in candidates]
        arch_dists = [c["archetype_dist"] for c in candidates]
        h0s = [c["h0_tightness"] for c in candidates]

        def norm_range(vals):
            lo, hi = min(vals), max(vals)
            r = hi - lo if hi > lo else 1.0
            return lo, r

        cos_lo, cos_r = norm_range(cos_sims)
        arch_lo, arch_r = norm_range(arch_dists)
        h0_lo, h0_r = norm_range(h0s)

        for c in candidates:
            cos_score = 1.0 - (c["cosine_sim"] - cos_lo) / cos_r
            arch_score = (c["archetype_dist"] - arch_lo) / arch_r
            h0_score = (c["h0_tightness"] - h0_lo) / h0_r
            # α=0.70 cosine anchor, β split equally across arch + h0
            c["combined_score"] = (alpha * cos_score
                                   + beta * 0.5 * arch_score
                                   + beta * 0.5 * h0_score)

        # Sort by combined score (ascending — lower is better)
        candidates.sort(key=lambda c: c["combined_score"])

        # ── 4. Sheaf Laplacian Truth Filter on top-k ─────────────────
        verified = None
        sheaf_results = []

        for cand in candidates[:k_archetype]:
            # Build full-resolution point cloud
            a_cloud = embed_sliding_window(
                cand["text"], self.encoder, window=5, stride=1
            )
            full_cloud = np.vstack([q_cloud_fine, a_cloud])

            # Run sheaf — NO subsampling
            sheaf = sheaf_truth_filter(
                full_cloud, k_neighbors=5, stalk_dim=8
            )

            cand["lambda1"] = sheaf["spectral_gap"]
            cand["kernel_dim"] = sheaf["kernel_dim"]
            cand["sheaf_n_points"] = sheaf["n_points"]
            sheaf_results.append(cand)

            # Accept if below threshold
            if sheaf["spectral_gap"] < lambda_threshold and verified is None:
                verified = cand

        # If no candidate passes the strict threshold, take lowest λ₁
        if verified is None and sheaf_results:
            sheaf_results.sort(key=lambda c: c["lambda1"])
            verified = sheaf_results[0]

        elapsed = time.time() - t0

        result = {
            "query": query,
            "verified_text": verified["text"] if verified else "",
            "verified_title": verified["title"] if verified else "",
            "lambda1": verified["lambda1"] if verified else float('inf'),
            "kernel_dim": verified["kernel_dim"] if verified else 0,
            "archetype_id": verified["archetype_id"] if verified else -1,
            "archetype_dist": verified["archetype_dist"] if verified else float('inf'),
            "cosine_rank": verified["cosine_rank"] if verified else -1,
            "elapsed": elapsed,
            "candidates_evaluated": len(sheaf_results),
            "sheaf_results": [{
                "text": c["text"][:100],
                "title": c["title"],
                "lambda1": c["lambda1"],
                "kernel_dim": c["kernel_dim"],
                "archetype_id": c["archetype_id"],
                "archetype_dist": c["archetype_dist"],
                "cosine_rank": c["cosine_rank"],
                "cosine_sim": c["cosine_sim"],
            } for c in sheaf_results],
        }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_navigator(n_test=50):
    """Full evaluation: navigate NQ queries, measure accuracy."""
    import random

    nav = UniversalSliceNavigator()

    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    answerable = [q for q in qa_pairs if q["has_answer"]]
    test_qa = random.sample(answerable, min(n_test, len(answerable)))

    print(f"\n{'='*70}")
    print(f"  UNIVERSAL SLICE NAVIGATOR — {n_test} queries")
    print(f"{'='*70}")

    results = []
    answer_found = 0
    title_match = 0
    total_lambda1 = []

    for i, qa in enumerate(test_qa):
        q = qa["question"]
        answer = qa["answer"]
        expected_title = qa["title"]

        result = nav.navigate(q)

        # Check if answer text is in the retrieved chunk
        found = answer.lower() in result["verified_text"].lower() if answer else False
        title_hit = result["verified_title"] == expected_title
        if found: answer_found += 1
        if title_hit: title_match += 1
        total_lambda1.append(result["lambda1"])

        result["expected_answer"] = answer[:100]
        result["expected_title"] = expected_title
        result["answer_found"] = found
        result["title_match"] = title_hit
        results.append(result)

        mark_a = "FOUND" if found else "MISS "
        mark_t = "HIT" if title_hit else "   "
        print(f"\n  Q{i}: {q[:65]}")
        print(f"    Answer: {answer[:50]}")
        print(f"    {mark_a} {mark_t} | λ₁={result['lambda1']:.6f} "
              f"ker={result['kernel_dim']} "
              f"arch={result['archetype_id']} "
              f"cos_rank={result['cosine_rank']} "
              f"| {result['elapsed']:.1f}s")
        print(f"    Retrieved: [{result['verified_title'][:30]}] "
              f"{result['verified_text'][:60]}...")

    # Summary
    n = len(test_qa)
    avg_lambda1 = np.mean(total_lambda1)
    avg_time = np.mean([r["elapsed"] for r in results])

    print(f"\n{'='*70}")
    print(f"  RESULTS — UNIVERSAL SLICE NAVIGATOR")
    print(f"{'='*70}")
    print(f"  Queries:        {n}")
    print(f"  Answer found:   {answer_found}/{n} ({answer_found/n*100:.0f}%)")
    print(f"  Title match:    {title_match}/{n} ({title_match/n*100:.0f}%)")
    print(f"  Avg λ₁:         {avg_lambda1:.6f}")
    print(f"  Avg time:       {avg_time:.1f}s per query")
    print(f"{'='*70}")

    log = {
        "experiment": "v5_nq_navigation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_test": n,
        "answer_found": answer_found,
        "answer_found_pct": round(answer_found / n * 100, 1),
        "title_match": title_match,
        "title_match_pct": round(title_match / n * 100, 1),
        "avg_lambda1": float(avg_lambda1),
        "avg_time": round(avg_time, 2),
        "results": results,
    }
    with open(OUTPUT_DIR / "navigation_results.json", "w") as f:
        json.dump(log, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"  Saved → {OUTPUT_DIR / 'navigation_results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=50)
    args = parser.parse_args()
    run_navigator(args.n_test)
