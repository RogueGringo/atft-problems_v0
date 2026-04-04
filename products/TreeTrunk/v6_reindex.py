#!/usr/bin/env python3
"""v6 Re-Index — Project 2.3M chunks through the trained manifold.

The FAISS index currently holds raw 384-dim sentence-transformer vectors.
The trained v6 TextFeatureMap warps these into a 128-dim space where
factual truth = topological tightness. This script:

  1. Loads all 2.3M pre-computed 384-dim embeddings
  2. Projects them through trainer_final.pt's TextFeatureMap (384 → 128)
  3. Builds a new FAISS IVF index in the warped 128-dim space
  4. Saves the new index + embeddings for v6 navigation

Output:
  results/v6_base_map/
  ├── embeddings_128.npy      — (2.3M, 128) warped embeddings
  ├── embeddings_128_norm.npy — L2-normalized for FAISS
  ├── faiss_index_128.bin     — FAISS IVF in warped space
  └── reindex_log.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import faiss

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"
OUTPUT_DIR = Path(__file__).parent / "results" / "v6_base_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from v6_topological_trainer import TextFeatureMap


def reindex(batch_size: int = 4096):
    """Project all chunks through trained v6 and rebuild FAISS."""
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load trained projection ───────────────────────────────────────
    print("Loading trained v6 TextFeatureMap...")
    feat_map = TextFeatureMap(out_dim=128, vocab_size=1, backbone_dim=384).to(device)

    checkpoint = torch.load(CHECKPOINT_DIR / "trainer_final.pt",
                            map_location=device, weights_only=False)
    feat_state = {k.replace("feature_map.", ""): v
                  for k, v in checkpoint.items() if k.startswith("feature_map.")}
    feat_map.load_state_dict(feat_state, strict=False)
    feat_map.eval()
    print(f"  Loaded on {device}")

    # ── Load raw embeddings ───────────────────────────────────────────
    print("Loading raw 384-dim embeddings...")
    raw_emb = np.load(BASE_MAP_DIR / "embeddings.npy")  # full load, not mmap
    n_chunks, d_in = raw_emb.shape
    print(f"  {n_chunks:,} chunks × {d_in} dims ({raw_emb.nbytes / 1e9:.2f} GB)")

    # ── Project in batches ────────────────────────────────────────────
    print(f"Projecting through trained manifold (batch={batch_size})...")
    d_out = 128
    projected = np.zeros((n_chunks, d_out), dtype=np.float32)

    t_proj = time.time()
    with torch.no_grad():
        for start_idx in range(0, n_chunks, batch_size):
            end_idx = min(start_idx + batch_size, n_chunks)
            batch = torch.tensor(raw_emb[start_idx:end_idx],
                                 dtype=torch.float32, device=device)
            out = feat_map(batch)  # (batch, 128)
            projected[start_idx:end_idx] = out.cpu().numpy()

            if start_idx % (batch_size * 50) == 0 and start_idx > 0:
                pct = start_idx / n_chunks * 100
                print(f"  {start_idx:>10,}/{n_chunks:,} ({pct:.0f}%)")

    proj_time = time.time() - t_proj
    print(f"  Projection done: {proj_time:.1f}s "
          f"({n_chunks / proj_time:.0f} chunks/s)")
    print(f"  New shape: {projected.shape} "
          f"({projected.nbytes / 1e9:.2f} GB)")

    # ── Build FAISS index ─────────────────────────────────────────────
    print(f"\nBuilding FAISS IVF index in warped 128-dim space...")
    emb_norm = projected.copy()
    faiss.normalize_L2(emb_norm)

    nlist = min(4096, int(np.sqrt(n_chunks) * 2))
    quantizer = faiss.IndexFlatIP(d_out)
    index = faiss.IndexIVFFlat(quantizer, d_out, nlist)
    print(f"  Training IVF (nlist={nlist})...")
    index.train(emb_norm)
    index.nprobe = 64
    index.add(emb_norm)
    print(f"  {index.ntotal:,} vectors in R^{d_out}")

    # ── Save ──────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_DIR}/...")
    np.save(OUTPUT_DIR / "embeddings_128.npy", projected)
    np.save(OUTPUT_DIR / "embeddings_128_norm.npy", emb_norm)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss_index_128.bin"))

    # Symlink shared metadata from v5
    for name in ["chunk_texts.json", "chunk_meta.json", "articles.json", "qa_pairs.json"]:
        src = BASE_MAP_DIR / name
        dst = OUTPUT_DIR / name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    elapsed = time.time() - start

    log = {
        "experiment": "v6_reindex",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_chunks": n_chunks,
        "d_in": d_in,
        "d_out": d_out,
        "projection_time": round(proj_time, 1),
        "index_type": f"IndexIVFFlat(nlist={nlist}, nprobe=64)",
        "embedding_size_gb": round(projected.nbytes / 1e9, 2),
    }
    with open(OUTPUT_DIR / "reindex_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RE-INDEX COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  {n_chunks:,} chunks: {d_in}d → {d_out}d")
    print(f"  Projection: {proj_time:.1f}s")
    print(f"  Old embeddings: {raw_emb.nbytes / 1e9:.2f} GB (384-dim)")
    print(f"  New embeddings: {projected.nbytes / 1e9:.2f} GB (128-dim)")
    print(f"  Compression: {raw_emb.nbytes / projected.nbytes:.1f}x")
    print(f"  Index: {OUTPUT_DIR / 'faiss_index_128.bin'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    reindex()
