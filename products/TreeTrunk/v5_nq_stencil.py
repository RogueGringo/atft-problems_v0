#!/usr/bin/env python3
"""v5 Universal Slice — NQ Scale: Archetype Stencil Extractor.

Runs the same waypoint signature extraction from v5_stencil.py but on the
NQ-scale base map. Tests whether the 10 SQuAD archetypes hold up when
exposed to 50K real Google search queries across 32K Wikipedia articles.

Key questions this answers:
  1. Do the SQuAD archetypes survive at NQ scale?
  2. Do new reasoning geometries emerge?
  3. Is the Archetype 3 "binary snap" (δ₁=34.9) a universal phenomenon?

Requires: results/v5_nq_base_map/ from v5_nq_ingest.py

Output:
  results/v5_nq_stencils/
  ├── signatures.npy         — (n_samples, 10) raw signatures
  ├── archetypes.npy         — (k, 10) cluster centroids
  ├── archetype_labels.npy   — (n_samples,) cluster assignments
  ├── archetype_meta.json    — per-archetype statistics
  └── stencil_log.json       — timing and diagnostics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
OUTPUT_DIR = Path(__file__).parent / "results" / "v5_nq_stencils"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Waypoint Signature (same math as SQuAD version)
# ══════════════════════════════════════════════════════════════════════════════

def embed_sliding_window(text, encoder, window=5, stride=3):
    """Embed text as a sliding-window point cloud."""
    words = text.split()
    if len(words) <= window:
        return encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
    chunks = [" ".join(words[i:i + window])
              for i in range(0, len(words) - window + 1, stride)]
    return encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def extract_waypoint_signature(q_cloud, a_cloud, n_waypoints=3):
    """Extract W(C) from the combined QA point cloud."""
    from ripser import ripser
    from scipy.spatial.distance import cdist

    n_q = len(q_cloud)
    X = np.vstack([q_cloud, a_cloud])

    # Subsample if needed
    if len(X) > 100:
        idx = np.random.choice(len(X), 100, replace=False)
        q_mask = idx < n_q
        X = X[idx]
        n_q_sub = int(q_mask.sum())
    else:
        n_q_sub = n_q

    # Onset scale: min cross-cloud distance
    cross_dist = cdist(X[:n_q_sub], X[n_q_sub:])
    onset_scale = float(cross_dist.min()) if cross_dist.size > 0 else 0.0

    # H₀ persistence
    result = ripser(X, maxdim=0)
    dgm = result["dgms"][0]
    finite = dgm[dgm[:, 1] < np.inf]

    if len(finite) == 0:
        return {
            "onset_scale": onset_scale,
            "waypoint_scales": [0.0] * n_waypoints,
            "derivatives": [0.0] * n_waypoints,
            "mean_persistence": 0.0,
            "max_persistence": 0.0,
            "n_bars": 0,
            "n_q": len(q_cloud),
            "n_a": len(a_cloud),
        }

    bars = finite[:, 1] - finite[:, 0]
    death_sorted = np.sort(finite[:, 1])[::-1]
    waypoint_scales = [float(death_sorted[i]) if i < len(death_sorted) else 0.0
                       for i in range(n_waypoints)]

    eps_window = 0.05
    derivatives = []
    for wp in waypoint_scales:
        if wp <= 0:
            derivatives.append(0.0)
            continue
        nearby = np.sum((finite[:, 1] >= wp - eps_window) &
                        (finite[:, 1] <= wp + eps_window))
        derivatives.append(float(nearby) / (2 * eps_window))

    return {
        "onset_scale": onset_scale,
        "waypoint_scales": waypoint_scales,
        "derivatives": derivatives,
        "mean_persistence": float(bars.mean()),
        "max_persistence": float(bars.max()),
        "n_bars": len(bars),
        "n_q": len(q_cloud),
        "n_a": len(a_cloud),
    }


def signature_to_vector(sig, n_waypoints=3):
    """Flatten signature to fixed-length feature vector."""
    vec = [
        sig["onset_scale"],
        sig["mean_persistence"],
        sig["max_persistence"],
        sig["n_bars"] / max(sig["n_q"] + sig["n_a"], 1),
    ]
    vec.extend(sig["waypoint_scales"][:n_waypoints])
    vec.extend(sig["derivatives"][:n_waypoints])
    return np.array(vec, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_archetypes(signatures, k=10):
    """Cluster signatures into k archetypes."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    sigs_scaled = scaler.fit_transform(signatures)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(sigs_scaled)
    centroids = scaler.inverse_transform(km.cluster_centers_)

    return centroids, labels, {
        "inertia": float(km.inertia_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_nq_stencil(n_samples=10000, n_waypoints=3, k_archetypes=10):
    """Extract stencils from NQ-scale base map."""
    start = time.time()

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading NQ base map...")
    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)
    with open(BASE_MAP_DIR / "chunk_meta.json") as f:
        chunk_meta = json.load(f)
    with open(BASE_MAP_DIR / "articles.json") as f:
        article_titles = json.load(f)
    embeddings = np.load(BASE_MAP_DIR / "embeddings.npy")

    print(f"  {len(qa_pairs):,} QA pairs, {len(chunk_meta):,} chunks, "
          f"embeddings {embeddings.shape}")

    # Build article → chunk lookup
    art_to_chunks = {}
    for i, m in enumerate(chunk_meta):
        art_to_chunks.setdefault(m["article_idx"], []).append(i)

    # Title → article_idx lookup
    title_to_idx = {t: i for i, t in enumerate(article_titles)}

    # ── Sample answerable QA pairs ───────────────────────────────────
    answerable = [q for q in qa_pairs if q["has_answer"]]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(answerable), size=min(n_samples, len(answerable)),
                            replace=False)
    sampled = [answerable[i] for i in sample_idx]
    print(f"\nSampled {len(sampled):,} answerable QA pairs")

    # ── Load encoder ─────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Extract signatures ───────────────────────────────────────────
    print(f"\nExtracting waypoint signatures...")
    signatures = []
    raw_sigs = []
    skipped = 0

    for i, qa in enumerate(sampled):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,}/{len(sampled):,} signatures...")

        art_idx = title_to_idx.get(qa["title"])
        if art_idx is None:
            skipped += 1
            continue
        chunk_ids = art_to_chunks.get(art_idx, [])
        if not chunk_ids:
            skipped += 1
            continue

        q_cloud = embed_sliding_window(qa["question"], encoder)
        a_cloud = embeddings[chunk_ids]

        sig = extract_waypoint_signature(q_cloud, a_cloud, n_waypoints)
        sig["question"] = qa["question"][:100]
        sig["title"] = qa["title"]

        raw_sigs.append(sig)
        signatures.append(signature_to_vector(sig, n_waypoints))

    signatures = np.array(signatures)
    print(f"  {len(signatures):,} signatures ({skipped} skipped)")

    # ── Cluster ──────────────────────────────────────────────────────
    print(f"\nClustering into {k_archetypes} archetypes...")
    centroids, labels, cluster_diag = cluster_archetypes(signatures, k_archetypes)

    # ── Interpret ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  NQ ARCHETYPE STENCILS — {k_archetypes} UNIVERSAL SHAPES")
    print(f"{'='*70}")

    archetype_meta = []
    for c in range(k_archetypes):
        mask = labels == c
        count = int(mask.sum())
        c_sigs = [raw_sigs[j] for j in range(len(labels)) if labels[j] == c]

        onsets = [s["onset_scale"] for s in c_sigs]
        mean_pers = [s["mean_persistence"] for s in c_sigs]
        max_pers = [s["max_persistence"] for s in c_sigs]
        derivs = [s["derivatives"][0] for s in c_sigs]

        titles = [s["title"] for s in c_sigs]
        title_counts = {}
        for t in titles:
            title_counts[t] = title_counts.get(t, 0) + 1
        top_titles = sorted(title_counts.items(), key=lambda x: -x[1])[:5]

        examples = [s["question"] for s in c_sigs[:3]]

        meta = {
            "archetype_id": c,
            "count": count,
            "fraction": round(count / len(labels), 3),
            "onset_scale": {
                "mean": round(float(np.mean(onsets)), 4),
                "std": round(float(np.std(onsets)), 4),
            },
            "mean_persistence": round(float(np.mean(mean_pers)), 4),
            "max_persistence": round(float(np.mean(max_pers)), 4),
            "mean_derivative_0": round(float(np.mean(derivs)), 2),
            "n_unique_articles": len(set(titles)),
            "top_domains": [{"title": t, "count": n} for t, n in top_titles],
            "example_questions": examples,
        }
        archetype_meta.append(meta)

        onset_str = f"ε*={meta['onset_scale']['mean']:.3f}±{meta['onset_scale']['std']:.3f}"
        print(f"\n  Archetype {c} | n={count} ({meta['fraction']*100:.0f}%) | {onset_str}")
        print(f"    mean_pers={meta['mean_persistence']:.4f}  "
              f"max_pers={meta['max_persistence']:.4f}  "
              f"δ₁={meta['mean_derivative_0']:.1f}")
        print(f"    {meta['n_unique_articles']} unique articles")
        print(f"    Top: {', '.join(t for t, _ in top_titles[:3])}")
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
        "experiment": "v5_nq_stencil",
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
    print(f"  NQ SECTION 2 COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  {len(signatures):,} signatures → {k_archetypes} archetypes")
    print(f"  Saved to {OUTPUT_DIR}")
    print(f"{'='*70}")

    return centroids, labels, archetype_meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_nq_stencil(args.n_samples, args.n_waypoints, args.k)
