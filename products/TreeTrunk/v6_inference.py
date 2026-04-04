#!/usr/bin/env python3
"""v6 Inference — Load trained sheaf projection into the navigator.

Replaces the frozen sentence-transformer embeddings with the learned
v6 projection that warps the manifold so factual truth = topological tightness.

Pipeline:
  1. Load v6 trained TextFeatureMap (384 → 128 projection)
  2. For each query: embed → project through trained head → sheaf evaluate
  3. Compare v5 (frozen embeddings) vs v6 (trained projection) on same queries
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import faiss

warnings.filterwarnings("ignore", message=".*more columns than rows.*")

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
STENCIL_DIR = Path(__file__).parent / "results" / "v5_nq_stencils"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"

from v6_topological_trainer import TextFeatureMap, DifferentiableSheafLaplacian


def sheaf_spectral_gap_torch(points: torch.Tensor, sheaf_lap: DifferentiableSheafLaplacian) -> float:
    """Compute spectral gap using the TRAINED sheaf Laplacian."""
    with torch.no_grad():
        eigs = sheaf_lap(points)
        # Extract λ₁: smallest eigenvalue above threshold
        nonzero = eigs[eigs > 1e-6]
        if len(nonzero) == 0:
            return 0.0
        return float(nonzero[0].item())


def sheaf_spectral_gap_numpy(points: np.ndarray, k_neighbors=5, stalk_dim=8) -> float:
    """Original v5 sheaf (numpy, no learned projection) for comparison."""
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist

    n = len(points)
    if n < 3:
        return 0.0
    s = min(stalk_dim, points.shape[1], n - 1)
    pca = PCA(n_components=s)
    X = pca.fit_transform(points)
    k = min(k_neighbors, n - 1)
    dists = cdist(X, X)
    edges = set()
    for i in range(n):
        for j in np.argsort(dists[i])[1:k+1]:
            edges.add((min(i,j), max(i,j)))
    edges = list(edges)
    m = len(edges)
    if m == 0:
        return 0.0
    delta = np.zeros((m * s, n * s))
    for k_e, (i, j) in enumerate(edges):
        d = X[j] - X[i]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            R = np.eye(s)
        else:
            d_hat = d / d_norm
            alpha = min(1.0, d_norm)
            R = np.eye(s) - alpha * np.outer(d_hat, d_hat)
        r0 = k_e * s
        delta[r0:r0+s, i*s:(i+1)*s] = np.eye(s)
        delta[r0:r0+s, j*s:(j+1)*s] = -R
    L = delta.T @ delta
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(L)))
    nonzero = eigenvalues[eigenvalues > 1e-6]
    return float(nonzero[0]) if len(nonzero) > 0 else 0.0


def embed_sliding_window(text, encoder, window=5, stride=1):
    """Fine-grained sliding window embedding."""
    words = text.split()
    if len(words) <= window:
        chunks = [text]
    else:
        chunks = [" ".join(words[i:i+window])
                  for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def run_comparison(queries: list[str] | None = None):
    """Run queries through both v5 (frozen) and v6 (trained) sheaf."""
    from sentence_transformers import SentenceTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if queries is None:
        queries = [
            "how many bones are in the human body",
            "who wrote the declaration of independence",
            "what is the speed of light in a vacuum",
            "when did the berlin wall fall",
            "what causes the northern lights",
        ]

    # ── Load base map ─────────────────────────────────────────────────
    print("Loading base map...")
    index = faiss.read_index(str(BASE_MAP_DIR / "faiss_index.bin"))
    if hasattr(index, 'nprobe'):
        index.nprobe = 64

    with open(BASE_MAP_DIR / "chunk_texts.json") as f:
        chunks = json.load(f)
    with open(BASE_MAP_DIR / "chunk_meta.json") as f:
        chunk_meta = json.load(f)
    with open(BASE_MAP_DIR / "articles.json") as f:
        article_titles = json.load(f)
    embeddings = np.load(BASE_MAP_DIR / "embeddings.npy", mmap_mode="r")

    # Article → chunks lookup
    art_to_chunks = {}
    for i, m in enumerate(chunk_meta):
        art_to_chunks.setdefault(m["article_idx"], []).append(i)

    # ── Load encoder ──────────────────────────────────────────────────
    print("Loading encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load trained v6 projection ────────────────────────────────────
    print("Loading v6 trained model...")
    feat_map = TextFeatureMap(out_dim=128, vocab_size=1, backbone_dim=384).to(device)
    sheaf_lap = DifferentiableSheafLaplacian(d_in=128, stalk_dim=8, k=4).to(device)

    checkpoint = torch.load(CHECKPOINT_DIR / "trainer_final.pt", map_location=device,
                            weights_only=False)

    # Extract the trained weights for feature_map and sheaf
    feat_state = {k.replace("feature_map.", ""): v
                  for k, v in checkpoint.items() if k.startswith("feature_map.")}
    sheaf_state = {k.replace("sheaf_loss_fn.sheaf.", ""): v
                   for k, v in checkpoint.items() if k.startswith("sheaf_loss_fn.sheaf.")}

    feat_map.load_state_dict(feat_state, strict=False)
    sheaf_lap.load_state_dict(sheaf_state, strict=False)
    feat_map.eval()
    sheaf_lap.eval()
    print(f"  Loaded trained weights on {device}")

    # ── Run queries ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  v5 vs v6 SHEAF COMPARISON — {len(queries)} queries")
    print(f"  v5: frozen sentence-transformer → numpy sheaf (PCA)")
    print(f"  v6: frozen ST → trained projection → torch sheaf (learned)")
    print(f"{'='*75}")

    k_faiss = 50
    k_eval = 5

    for q in queries:
        print(f"\n  Q: {q}")

        # Embed query
        q_emb_np = encoder.encode([q], convert_to_numpy=True)
        q_norm = q_emb_np.copy()
        faiss.normalize_L2(q_norm)
        q_cloud_np = embed_sliding_window(q, encoder, window=5, stride=1)

        # FAISS retrieval
        scores, indices = index.search(q_norm, k_faiss)

        # Evaluate top-5 by cosine with BOTH sheaf versions
        print(f"  {'Rank':>4s}  {'v5 λ₁':>8s}  {'v6 λ₁':>8s}  {'Δ':>7s}  {'Article':<30s}  Chunk")
        print(f"  {'─'*90}")

        for rank in range(min(k_eval, len(indices[0]))):
            idx = int(indices[0][rank])
            if idx < 0:
                continue

            chunk_text = chunks[idx]
            meta = chunk_meta[idx]
            title = article_titles[meta["article_idx"]]

            # Build answer cloud
            a_cloud_np = embed_sliding_window(chunk_text, encoder, window=5, stride=1)
            full_cloud_np = np.vstack([q_cloud_np, a_cloud_np])

            # v5: numpy sheaf on raw 384-dim embeddings
            gap_v5 = sheaf_spectral_gap_numpy(full_cloud_np, k_neighbors=5, stalk_dim=8)

            # v6: project through trained head, then trained sheaf
            full_cloud_t = torch.tensor(full_cloud_np, dtype=torch.float32, device=device)
            projected = feat_map(full_cloud_t)  # (n, 128) — trained projection
            gap_v6 = sheaf_spectral_gap_torch(projected, sheaf_lap)

            delta = gap_v6 - gap_v5
            marker = " ◄" if rank == 0 else ""
            print(f"  [{rank:>2d}]  {gap_v5:>8.5f}  {gap_v6:>8.5f}  {delta:>+7.4f}  "
                  f"{title[:30]:<30s}  {chunk_text[:35]}...{marker}")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    run_comparison()
