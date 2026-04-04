#!/usr/bin/env python3
"""v5 Universal Slice — Section 1: SQuAD Ingest & Base Map.

Builds the continuous point cloud from SQuAD's Wikipedia paragraphs.
This is the knowledge manifold that the Sheaf Navigator will traverse.

Pipeline:
  1. Load SQuAD v1.1 (train + validation)
  2. Deduplicate context paragraphs (~500 unique articles)
  3. Chunk with sliding window (30 words, stride 15)
  4. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  5. Build FAISS index (flat for 50K; swap to IVF at NQ scale)
  6. Save base map + QA pairs for stencil extraction (Section 2)

Output:
  results/v5_base_map/
  ├── embeddings.npy         — raw embedding matrix (n_chunks, 384)
  ├── embeddings_norm.npy    — L2-normalized for FAISS
  ├── faiss_index.bin        — FAISS index
  ├── chunks.json            — chunk texts + metadata (context_id, position)
  ├── contexts.json          — full deduplicated context paragraphs
  ├── qa_pairs.json          — all QA pairs with context pointers
  └── ingest_log.json        — statistics and timing
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import faiss

OUTPUT_DIR = Path(__file__).parent / "results" / "v5_base_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chunker
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    window: int = 30,
    stride: int = 15,
    min_words: int = 8,
) -> list[str]:
    """Sliding window chunker over word tokens.

    Overlapping windows ensure H₀ connectivity across chunk boundaries —
    adjacent chunks share half their words, so persistence bars bridge them.
    """
    words = text.split()
    if len(words) <= window:
        return [text] if len(words) >= min_words else []

    chunks = []
    for i in range(0, len(words) - window + 1, stride):
        chunk = " ".join(words[i:i + window])
        chunks.append(chunk)

    # Catch the tail if it wasn't covered by the last window
    tail_start = max(0, len(words) - window)
    tail = " ".join(words[tail_start:])
    if tail not in chunks and len(tail.split()) >= min_words:
        chunks.append(tail)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_squad() -> tuple[list[str], list[dict]]:
    """Load SQuAD v1.1, deduplicate contexts, extract QA pairs."""
    from datasets import load_dataset

    print("Loading SQuAD v1.1...")
    ds_train = load_dataset("rajpurkar/squad", split="train")
    ds_val = load_dataset("rajpurkar/squad", split="validation")

    contexts = OrderedDict()  # context_text → context_id
    qa_pairs = []

    for ds_split, split_name in [(ds_train, "train"), (ds_val, "validation")]:
        for item in ds_split:
            ctx = item["context"]
            if ctx not in contexts:
                contexts[ctx] = len(contexts)

            qa_pairs.append({
                "question": item["question"],
                "answers": item["answers"]["text"],
                "answer_start": item["answers"]["answer_start"],
                "context_id": contexts[ctx],
                "title": item["title"],
                "split": split_name,
            })

    unique_contexts = list(contexts.keys())
    titles = set(q["title"] for q in qa_pairs)

    print(f"  {len(qa_pairs)} QA pairs")
    print(f"  {len(unique_contexts)} unique context paragraphs")
    print(f"  {len(titles)} unique article titles")

    return unique_contexts, qa_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Base Map Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_base_map(
    contexts: list[str],
    window: int = 30,
    stride: int = 15,
) -> dict:
    """Chunk all contexts, embed, build FAISS index."""
    from sentence_transformers import SentenceTransformer

    # ── Chunk ─────────────────────────────────────────────────────────
    print(f"\nChunking {len(contexts)} contexts (window={window}, stride={stride})...")
    chunks = []
    chunk_meta = []

    for ctx_id, ctx in enumerate(contexts):
        ctx_chunks = chunk_text(ctx, window=window, stride=stride)
        for pos, chunk in enumerate(ctx_chunks):
            chunks.append(chunk)
            chunk_meta.append({"context_id": ctx_id, "position": pos})

    print(f"  {len(chunks)} chunks")
    print(f"  {len(chunks) / len(contexts):.1f} chunks/context avg")

    # ── Embed ─────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(chunks)} chunks with all-MiniLM-L6-v2...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=256,
    )
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size:  {embeddings.nbytes / 1e6:.1f} MB")

    # ── FAISS Index ───────────────────────────────────────────────────
    print(f"\nBuilding FAISS index...")
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)

    # Flat for ~50K vectors. At NQ scale (1M+), swap to:
    #   quantizer = faiss.IndexFlatIP(d)
    #   index = faiss.IndexIVFFlat(quantizer, d, nlist=1024)
    #   index.train(emb_norm)
    #   index.nprobe = 32
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(emb_norm)
    print(f"  {index.ntotal} vectors in R^{embeddings.shape[1]}")

    return {
        "chunks": chunks,
        "metadata": chunk_meta,
        "embeddings": embeddings,
        "embeddings_norm": emb_norm,
        "index": index,
        "encoder": encoder,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_base_map(
    base_map: dict,
    contexts: list[str],
    qa_pairs: list[dict],
    elapsed: float,
):
    """Persist everything Sections 2 and 3 need."""
    # Numeric arrays
    np.save(OUTPUT_DIR / "embeddings.npy", base_map["embeddings"])
    np.save(OUTPUT_DIR / "embeddings_norm.npy", base_map["embeddings_norm"])

    # FAISS index
    faiss.write_index(base_map["index"], str(OUTPUT_DIR / "faiss_index.bin"))

    # Chunk texts + metadata
    with open(OUTPUT_DIR / "chunks.json", "w") as f:
        json.dump({
            "chunks": base_map["chunks"],
            "metadata": base_map["metadata"],
        }, f)

    # Full context paragraphs (Section 2 needs these for stencil computation)
    with open(OUTPUT_DIR / "contexts.json", "w") as f:
        json.dump(contexts, f)

    # QA pairs (Section 2 stencil extraction input)
    with open(OUTPUT_DIR / "qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f)

    # Ingest log
    n_chunks = len(base_map["chunks"])
    titles = set(q["title"] for q in qa_pairs)
    log = {
        "experiment": "v5_squad_ingest",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_contexts": len(contexts),
        "n_chunks": n_chunks,
        "n_qa_pairs": len(qa_pairs),
        "n_titles": len(titles),
        "embedding_dim": int(base_map["embeddings"].shape[1]),
        "embedding_size_mb": round(base_map["embeddings"].nbytes / 1e6, 1),
        "chunks_per_context_avg": round(n_chunks / len(contexts), 1),
        "window": 30,
        "stride": 15,
    }
    with open(OUTPUT_DIR / "ingest_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size = p.stat().st_size
        if size < 1e6:
            print(f"  {p.name:30s} {size/1e3:>8.1f} KB")
        else:
            print(f"  {p.name:30s} {size/1e6:>8.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(base_map: dict, qa_pairs: list[dict], n_queries: int = 5):
    """Run a few test queries to verify the base map works."""
    import random

    print(f"\n{'='*70}")
    print(f"  SANITY CHECK — {n_queries} random queries")
    print(f"{'='*70}")

    encoder = base_map["encoder"]
    test_qa = random.sample(qa_pairs, n_queries)

    hits_at_1 = 0
    hits_at_5 = 0

    for qa in test_qa:
        q = qa["question"]
        expected_ctx = qa["context_id"]

        q_emb = encoder.encode([q], convert_to_numpy=True)
        q_norm = q_emb.copy()
        faiss.normalize_L2(q_norm)
        scores, indices = base_map["index"].search(q_norm, 5)

        top5_ctxs = [base_map["metadata"][int(idx)]["context_id"]
                     for idx in indices[0]]

        hit1 = top5_ctxs[0] == expected_ctx
        hit5 = expected_ctx in top5_ctxs
        if hit1:
            hits_at_1 += 1
        if hit5:
            hits_at_5 += 1

        print(f"\n  Q: {q[:80]}")
        print(f"  Gold answer: {qa['answers'][0][:60]}")
        print(f"  Expected context: {expected_ctx}")
        for rank in range(min(3, len(indices[0]))):
            idx = int(indices[0][rank])
            meta = base_map["metadata"][idx]
            chunk = base_map["chunks"][idx]
            hit = "HIT" if meta["context_id"] == expected_ctx else "   "
            print(f"    [{rank}] {hit} ctx={meta['context_id']:>4d} "
                  f"sim={scores[0][rank]:.3f} | {chunk[:60]}...")

    print(f"\n  Retrieval: {hits_at_1}/{n_queries} @1, "
          f"{hits_at_5}/{n_queries} @5")
    return hits_at_1, hits_at_5


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_ingest():
    """Full Section 1 pipeline."""
    start = time.time()

    # Load
    contexts, qa_pairs = load_squad()

    # Build
    base_map = build_base_map(contexts)

    elapsed = time.time() - start

    # Save
    save_base_map(base_map, contexts, qa_pairs, elapsed)

    # Verify
    sanity_check(base_map, qa_pairs)

    print(f"\n{'='*70}")
    print(f"  SECTION 1 COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  {len(base_map['chunks']):,} chunks from "
          f"{len(contexts):,} contexts")
    print(f"  {len(qa_pairs):,} QA pairs ready for Section 2")
    print(f"  Base map: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_ingest()
