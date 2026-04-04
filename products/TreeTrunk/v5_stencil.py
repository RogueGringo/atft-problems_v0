#!/usr/bin/env python3
"""v5 Universal Slice — Section 2: Archetype Stencil Extractor.

Discovers the universal topological shapes of how questions route to answers.
This is the core innovation: instead of a hardcoded mean displacement vector,
we extract the actual geometric signature of each QA trajectory and cluster
them into archetypes.

Pipeline:
  1. Sample N QA pairs from Section 1's saved data
  2. For each pair: embed Q with sliding windows, retrieve A chunk embedding
  3. Combine into mini-cloud X = Q_cloud ∪ A_cloud
  4. Run H₀ persistence on X, extract waypoint signature W(C):
     - ε* (onset scale): distance where Q first merges with A
     - ε_{w,i} (waypoint scales): top-k persistence bar deaths
     - δ₁(ε_{w,i}) (topological derivatives): Betti-0 rate of change at waypoints
  5. Cluster signatures into k archetypes via k-means
  6. Save archetype centroids + metadata for Section 3 navigation

Requires: results/v5_base_map/ from Section 1

Output:
  results/v5_stencils/
  ├── signatures.npy         — (n_samples, signature_dim) raw signatures
  ├── archetypes.npy         — (k, signature_dim) cluster centroids
  ├── archetype_labels.npy   — (n_samples,) cluster assignments
  ├── archetype_meta.json    — per-archetype statistics and interpretation
  └── stencil_log.json       — timing and diagnostics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_base_map"
OUTPUT_DIR = Path(__file__).parent / "results" / "v5_stencils"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Waypoint Signature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def embed_sliding_window(
    text: str,
    encoder,
    window: int = 5,
    stride: int = 3,
) -> np.ndarray:
    """Embed text as a sliding-window point cloud."""
    words = text.split()
    if len(words) <= window:
        return encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)

    chunks = [" ".join(words[i:i + window])
              for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def extract_waypoint_signature(
    q_cloud: np.ndarray,
    a_cloud: np.ndarray,
    n_waypoints: int = 3,
) -> dict:
    """Extract W(C) = (ε*, ε_{w,i}, δ₁(ε_{w,i})) from a QA point cloud.

    The persistence diagram of X = Q_cloud ∪ A_cloud encodes:
    - When Q-points start merging with A-points (onset scale ε*)
    - The major structural merge events (waypoint scales)
    - How sharply the topology changes at each waypoint (derivatives)
    """
    from ripser import ripser
    from scipy.spatial.distance import cdist

    n_q = len(q_cloud)
    n_a = len(a_cloud)
    X = np.vstack([q_cloud, a_cloud])

    # Subsample if too large for persistence computation
    if len(X) > 100:
        idx = np.random.choice(len(X), 100, replace=False)
        # Preserve the Q/A boundary information
        q_mask = idx < n_q
        X = X[idx]
        n_q_sub = int(q_mask.sum())
        n_a_sub = len(X) - n_q_sub
    else:
        n_q_sub = n_q
        n_a_sub = n_a

    # ── Cross-cloud distance: minimum distance from any Q-point to any A-point
    cross_dist = cdist(X[:n_q_sub], X[n_q_sub:])
    onset_scale = float(cross_dist.min()) if cross_dist.size > 0 else 0.0

    # ── H₀ persistence
    result = ripser(X, maxdim=0)
    dgm = result["dgms"][0]

    # Finite bars only (exclude the infinite bar = connected component)
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) == 0:
        return {
            "onset_scale": onset_scale,
            "waypoint_scales": [0.0] * n_waypoints,
            "derivatives": [0.0] * n_waypoints,
            "mean_persistence": 0.0,
            "max_persistence": 0.0,
            "n_bars": 0,
            "n_q": n_q,
            "n_a": n_a,
        }

    # Bar lengths (death - birth)
    bars = finite[:, 1] - finite[:, 0]
    bars_sorted = np.sort(bars)[::-1]  # longest first

    # ── Waypoint scales: death times of the longest bars
    # These are the filtration values where major merges happen
    death_sorted = np.sort(finite[:, 1])[::-1]
    waypoint_scales = []
    for i in range(n_waypoints):
        if i < len(death_sorted):
            waypoint_scales.append(float(death_sorted[i]))
        else:
            waypoint_scales.append(0.0)

    # ── Topological derivatives: Betti-0 rate of change at waypoints
    # δ₁(ε) ≈ ΔBetti₀ / Δε near each waypoint
    # We approximate by counting how many bars die in a neighborhood of each waypoint
    derivatives = []
    eps_window = 0.05  # neighborhood width for derivative estimation
    for wp in waypoint_scales:
        if wp <= 0:
            derivatives.append(0.0)
            continue
        # Count bars dying in [wp - eps_window, wp + eps_window]
        nearby_deaths = np.sum(
            (finite[:, 1] >= wp - eps_window) &
            (finite[:, 1] <= wp + eps_window)
        )
        # Derivative = change in Betti-0 per unit filtration
        deriv = float(nearby_deaths) / (2 * eps_window) if eps_window > 0 else 0.0
        derivatives.append(deriv)

    return {
        "onset_scale": onset_scale,
        "waypoint_scales": waypoint_scales,
        "derivatives": derivatives,
        "mean_persistence": float(bars.mean()),
        "max_persistence": float(bars.max()),
        "n_bars": len(bars),
        "n_q": n_q,
        "n_a": n_a,
    }


def signature_to_vector(sig: dict, n_waypoints: int = 3) -> np.ndarray:
    """Flatten a waypoint signature into a fixed-length feature vector.

    Layout: [onset_scale, mean_pers, max_pers, n_bars_norm,
             wp_0, wp_1, wp_2, d_0, d_1, d_2]
    """
    vec = [
        sig["onset_scale"],
        sig["mean_persistence"],
        sig["max_persistence"],
        sig["n_bars"] / max(sig["n_q"] + sig["n_a"], 1),  # normalized bar count
    ]
    vec.extend(sig["waypoint_scales"][:n_waypoints])
    vec.extend(sig["derivatives"][:n_waypoints])
    return np.array(vec, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Archetype Clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_archetypes(
    signatures: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cluster signature vectors into k archetypes via k-means.

    Returns (centroids, labels, diagnostics).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Standardize features before clustering
    scaler = StandardScaler()
    sigs_scaled = scaler.fit_transform(signatures)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(sigs_scaled)

    # Transform centroids back to original scale
    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    # Per-cluster statistics
    cluster_stats = {}
    for c in range(k):
        mask = labels == c
        count = int(mask.sum())
        cluster_sigs = signatures[mask]
        cluster_stats[str(c)] = {
            "count": count,
            "fraction": round(count / len(labels), 3),
            "mean_onset": round(float(cluster_sigs[:, 0].mean()), 4),
            "mean_persistence": round(float(cluster_sigs[:, 1].mean()), 4),
            "mean_max_persistence": round(float(cluster_sigs[:, 2].mean()), 4),
            "mean_waypoint_0": round(float(cluster_sigs[:, 4].mean()), 4),
        }

    diagnostics = {
        "k": k,
        "inertia": float(km.inertia_),
        "cluster_stats": cluster_stats,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    return centroids, labels, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_section1_data():
    """Load saved base map data from Section 1."""
    print("Loading Section 1 base map...")

    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)
    with open(BASE_MAP_DIR / "chunks.json") as f:
        chunk_data = json.load(f)
    embeddings = np.load(BASE_MAP_DIR / "embeddings.npy")

    chunks = chunk_data["chunks"]
    metadata = chunk_data["metadata"]

    print(f"  {len(qa_pairs)} QA pairs, {len(chunks)} chunks, "
          f"embeddings {embeddings.shape}")

    return qa_pairs, chunks, metadata, embeddings


def run_stencil(
    n_samples: int = 5000,
    n_waypoints: int = 3,
    k_archetypes: int = 10,
):
    """Full Section 2 pipeline: extract signatures, cluster archetypes."""
    start = time.time()

    # ── Load ──────────────────────────────────────────────────────────
    qa_pairs, chunks, metadata, embeddings = load_section1_data()

    # Build context_id → chunk_id lookup
    ctx_to_chunks = {}
    for i, m in enumerate(metadata):
        ctx_to_chunks.setdefault(m["context_id"], []).append(i)

    # ── Sample QA pairs ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(qa_pairs), size=min(n_samples, len(qa_pairs)),
                            replace=False)
    sampled_qa = [qa_pairs[i] for i in sample_idx]
    print(f"\nSampled {len(sampled_qa)} QA pairs for stencil extraction")

    # ── Load encoder ─────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Extract signatures ───────────────────────────────────────────
    print(f"\nExtracting waypoint signatures (n_waypoints={n_waypoints})...")
    signatures = []
    raw_sigs = []
    skipped = 0

    for i, qa in enumerate(sampled_qa):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(sampled_qa)} signatures extracted...")

        ctx_id = qa["context_id"]
        chunk_ids = ctx_to_chunks.get(ctx_id, [])

        if not chunk_ids:
            skipped += 1
            continue

        # Q: embed as sliding-window point cloud
        q_cloud = embed_sliding_window(qa["question"], encoder)

        # A: use the pre-computed chunk embeddings for this context
        # Take the chunk closest to the answer (first chunk of the context
        # as approximation — Section 3 will use exact answer mapping)
        a_cloud = embeddings[chunk_ids]

        # Extract signature
        sig = extract_waypoint_signature(q_cloud, a_cloud, n_waypoints)
        sig["question"] = qa["question"][:100]
        sig["title"] = qa["title"]
        sig["context_id"] = ctx_id

        raw_sigs.append(sig)
        signatures.append(signature_to_vector(sig, n_waypoints))

    signatures = np.array(signatures)
    print(f"  {len(signatures)} signatures extracted ({skipped} skipped)")
    print(f"  Signature vector dim: {signatures.shape[1]}")

    # ── Cluster ──────────────────────────────────────────────────────
    print(f"\nClustering into {k_archetypes} archetypes...")
    centroids, labels, cluster_diag = cluster_archetypes(
        signatures, k=k_archetypes
    )

    # ── Interpret archetypes ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ARCHETYPE STENCILS — {k_archetypes} UNIVERSAL SHAPES")
    print(f"{'='*70}")

    archetype_meta = []
    for c in range(k_archetypes):
        mask = labels == c
        count = int(mask.sum())
        c_sigs = [raw_sigs[j] for j in range(len(labels)) if labels[j] == c]

        # Aggregate signature stats
        onsets = [s["onset_scale"] for s in c_sigs]
        mean_pers = [s["mean_persistence"] for s in c_sigs]
        max_pers = [s["max_persistence"] for s in c_sigs]
        derivs = [s["derivatives"][0] for s in c_sigs]

        # Collect titles for domain distribution
        titles = [s["title"] for s in c_sigs]
        title_counts = {}
        for t in titles:
            title_counts[t] = title_counts.get(t, 0) + 1
        top_titles = sorted(title_counts.items(), key=lambda x: -x[1])[:5]

        # Example questions
        examples = [s["question"] for s in c_sigs[:3]]

        meta = {
            "archetype_id": c,
            "count": count,
            "fraction": round(count / len(labels), 3),
            "onset_scale": {
                "mean": round(float(np.mean(onsets)), 4),
                "std": round(float(np.std(onsets)), 4),
                "min": round(float(np.min(onsets)), 4),
                "max": round(float(np.max(onsets)), 4),
            },
            "mean_persistence": round(float(np.mean(mean_pers)), 4),
            "max_persistence": round(float(np.mean(max_pers)), 4),
            "mean_derivative_0": round(float(np.mean(derivs)), 2),
            "top_domains": [{"title": t, "count": n} for t, n in top_titles],
            "example_questions": examples,
        }
        archetype_meta.append(meta)

        # Print summary
        onset_str = f"ε*={meta['onset_scale']['mean']:.3f}±{meta['onset_scale']['std']:.3f}"
        print(f"\n  Archetype {c} | n={count} ({meta['fraction']*100:.0f}%) | {onset_str}")
        print(f"    mean_pers={meta['mean_persistence']:.4f}  "
              f"max_pers={meta['max_persistence']:.4f}  "
              f"δ₁={meta['mean_derivative_0']:.1f}")
        print(f"    Top domains: {', '.join(t for t, _ in top_titles[:3])}")
        for ex in examples[:2]:
            print(f"    Q: {ex[:70]}")

    elapsed = time.time() - start

    # ── Save ─────────────────────────────────────────────────────────
    np.save(OUTPUT_DIR / "signatures.npy", signatures)
    np.save(OUTPUT_DIR / "archetypes.npy", centroids)
    np.save(OUTPUT_DIR / "archetype_labels.npy", labels)

    with open(OUTPUT_DIR / "archetype_meta.json", "w") as f:
        json.dump(archetype_meta, f, indent=2)

    log = {
        "experiment": "v5_stencil_extraction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_samples": len(signatures),
        "n_waypoints": n_waypoints,
        "k_archetypes": k_archetypes,
        "signature_dim": int(signatures.shape[1]),
        "skipped": skipped,
        "cluster_inertia": cluster_diag["inertia"],
        "scaler_mean": cluster_diag["scaler_mean"],
        "scaler_scale": cluster_diag["scaler_scale"],
    }
    with open(OUTPUT_DIR / "stencil_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  SECTION 2 COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  {len(signatures)} signatures → {k_archetypes} archetypes")
    print(f"  Saved to {OUTPUT_DIR}")
    print(f"{'='*70}")

    return centroids, labels, archetype_meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_stencil(args.n_samples, args.n_waypoints, args.k)
