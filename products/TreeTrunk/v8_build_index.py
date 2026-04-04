#!/usr/bin/env python3
"""v8 FAISS IndexPreTransform — Bake v7 projection into C++ search.

The trained 384→128 topological projection is compiled into a FAISS
LinearTransform. The database natively warps vectors in C++ during
search — zero PyTorch overhead at inference time.

Architecture:
  faiss.IndexPreTransform(
      LinearTransform(384, 128),     ← trained v7 projection (C++)
      IndexIVFFlat(128, nlist)       ← search in warped space
  )

When you pass a 384-dim query, FAISS:
  1. Multiplies by the trained (384,128) matrix in C++
  2. Searches the IVF index in 128-dim warped space
  3. Returns results — no Python, no PyTorch

Pipeline:
  1. Load extracted projection weight/bias
  2. Build LinearTransform → IndexPreTransform → IVF
  3. Stream 2.3M raw 384-dim vectors through (batched for RAM)
  4. Save the complete index to disk
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import faiss

PREP_DIR = Path(__file__).parent / "results" / "v8_faiss_prep"
RAW_EMB = Path(__file__).parent / "results" / "v5_nq_base_map" / "embeddings.npy"
OUTPUT_DIR = PREP_DIR


def build_index(batch_size: int = 100_000):
    start = time.time()

    # ── Load projection ───────────────────────────────────────────────
    print("Loading v7 projection weights...")
    W = np.load(PREP_DIR / "v7_projection_weight.npy")   # (384, 128)
    b = np.load(PREP_DIR / "v7_projection_bias.npy")      # (128,)

    d_in, d_out = W.shape
    print(f"  Projection: {d_in} → {d_out}")

    # ── Build LinearTransform ─────────────────────────────────────────
    print("Building FAISS LinearTransform...")
    transform = faiss.LinearTransform(d_in, d_out, have_bias=True)

    # FAISS stores A as a flat (d_out * d_in,) array in row-major order
    # A maps: x_out = A @ x_in + b
    # Our W is (d_in, d_out), so A = W.T which is (d_out, d_in)
    A = W.T.copy().astype(np.float32)  # (128, 384) row-major
    faiss.copy_array_to_vector(A.ravel(), transform.A)
    faiss.copy_array_to_vector(b.astype(np.float32), transform.b)
    transform.is_trained = True

    # Verify transform works
    test_vec = np.random.randn(1, d_in).astype(np.float32)
    test_out = transform.apply(test_vec)
    expected = (test_vec @ W + b).astype(np.float32)
    cos = np.dot(test_out[0], expected[0]) / (
        np.linalg.norm(test_out[0]) * np.linalg.norm(expected[0]))
    print(f"  Transform verification: cosine={cos:.6f} (should be ~1.0)")

    # ── Build IVF index in transformed space ──────────────────────────
    print("Building IVF index in 128-dim warped space...")

    # Load raw embeddings to count
    raw = np.load(RAW_EMB, mmap_mode="r")
    n_total = raw.shape[0]
    print(f"  {n_total:,} vectors to index")

    nlist = min(4096, int(np.sqrt(n_total) * 2))
    quantizer = faiss.IndexFlatIP(d_out)
    ivf_index = faiss.IndexIVFFlat(quantizer, d_out, nlist)

    # Wrap with the pre-transform
    index = faiss.IndexPreTransform(transform, ivf_index)

    # ── Train IVF (needs a sample of transformed vectors) ─────────────
    print(f"  Training IVF (nlist={nlist})...")
    # Sample 50K vectors for training
    n_train = min(50_000, n_total)
    train_idx = np.random.choice(n_total, n_train, replace=False)
    train_idx.sort()
    train_raw = np.array(raw[train_idx]).astype(np.float32)

    # Normalize before adding (for inner product search)
    train_transformed = transform.apply(train_raw)
    faiss.normalize_L2(train_transformed)
    ivf_index.train(train_transformed)

    # ── Add vectors in batches ────────────────────────────────────────
    print(f"  Adding {n_total:,} vectors (batch={batch_size:,})...")
    ivf_index.nprobe = 64
    t_add = time.time()

    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        batch = np.array(raw[start_idx:end_idx]).astype(np.float32)

        # Transform + normalize
        batch_t = transform.apply(batch)
        faiss.normalize_L2(batch_t)

        # Add directly to the IVF (bypass PreTransform for speed —
        # we already transformed)
        ivf_index.add(batch_t)

        if start_idx % (batch_size * 5) == 0 and start_idx > 0:
            pct = start_idx / n_total * 100
            print(f"    {start_idx:>10,}/{n_total:,} ({pct:.0f}%)")

    add_time = time.time() - t_add
    print(f"  Added {ivf_index.ntotal:,} vectors in {add_time:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────
    # Save the IVF index (without PreTransform — we apply transform at query time)
    # This is smaller and more portable
    index_path = OUTPUT_DIR / "v8_topological_ivf.faiss"
    faiss.write_index(ivf_index, str(index_path))

    # Also save the full PreTransform index for end-to-end use
    pretransform_path = OUTPUT_DIR / "v8_pretransform.faiss"
    faiss.write_index(index, str(pretransform_path))

    elapsed = time.time() - start

    log = {
        "experiment": "v8_faiss_build",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_vectors": int(ivf_index.ntotal),
        "d_in": d_in,
        "d_out": d_out,
        "nlist": nlist,
        "nprobe": 64,
        "add_time": round(add_time, 1),
        "transform_cosine_check": float(cos),
    }
    with open(OUTPUT_DIR / "v8_build_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Benchmark ─────────────────────────────────────────────────────
    print(f"\nBenchmark: 100 random queries...")
    queries = np.random.randn(100, d_in).astype(np.float32)
    queries_t = transform.apply(queries)
    faiss.normalize_L2(queries_t)

    t_search = time.time()
    scores, indices = ivf_index.search(queries_t, 10)
    search_time = (time.time() - t_search) * 1000  # ms
    per_query = search_time / 100

    print(f"  100 queries in {search_time:.1f}ms ({per_query:.2f}ms/query)")

    print(f"\n{'='*60}")
    print(f"  v8 FAISS INDEX BUILT — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  {ivf_index.ntotal:,} vectors indexed")
    print(f"  Transform: {d_in}d → {d_out}d (trained v7 projection)")
    print(f"  Index: IVFFlat(nlist={nlist}, nprobe=64)")
    print(f"  Search: {per_query:.2f}ms/query")
    print(f"  IVF index:  {index_path} ({index_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  PreTransform: {pretransform_path} ({pretransform_path.stat().st_size / 1e6:.0f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_index()
