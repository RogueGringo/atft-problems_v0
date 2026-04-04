#!/usr/bin/env python3
"""v5 Universal Slice — NQ Scale: Wikipedia-Scale Base Map.

Scale from SQuAD (152K chunks, 490 articles) to Natural Questions
(millions of chunks, thousands of Wikipedia articles).

The ocean.

Pipeline:
  1. Stream NQ dataset (307K questions, each with a full Wikipedia page)
  2. Deduplicate Wikipedia pages by title
  3. Strip HTML tokens → clean text
  4. Chunk with sliding window (30 words, stride 15)
  5. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  6. Build FAISS IVF index (clustered for sub-linear search)
  7. Save base map + QA pairs for stencil extraction

Output:
  results/v5_nq_base_map/
  ├── embeddings.npy         — (n_chunks, 384) float32
  ├── embeddings_norm.npy    — L2-normalized
  ├── faiss_index.bin        — FAISS IVF index
  ├── chunk_texts.json       — chunk text strings
  ├── chunk_meta.json        — chunk → article mapping
  ├── articles.json          — article title list
  ├── qa_pairs.json          — NQ QA pairs with article refs
  └── ingest_log.json        — statistics and timing
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import faiss

OUTPUT_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing
# ══════════════════════════════════════════════════════════════════════════════

def extract_clean_text(document: dict) -> str:
    """Strip HTML tokens from NQ document, return clean text."""
    tokens = document["tokens"]
    token_strs = tokens["token"]
    is_html = tokens["is_html"]

    clean = [t for t, h in zip(token_strs, is_html) if not h]
    text = " ".join(clean)

    # Clean up spacing around punctuation (NQ tokenization artifact)
    for p in [".", ",", ";", ":", "!", "?", "'s", "n't", "'re", "'ve", "'ll"]:
        text = text.replace(f" {p}", p)
    text = text.replace("( ", "(").replace(" )", ")")

    return text


def chunk_text(
    text: str,
    window: int = 30,
    stride: int = 15,
    min_words: int = 8,
) -> list[str]:
    """Sliding window chunker over word tokens."""
    words = text.split()
    if len(words) <= window:
        return [text] if len(words) >= min_words else []

    chunks = []
    for i in range(0, len(words) - window + 1, stride):
        chunks.append(" ".join(words[i:i + window]))

    tail_start = max(0, len(words) - window)
    tail = " ".join(words[tail_start:])
    if tail not in chunks and len(tail.split()) >= min_words:
        chunks.append(tail)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# NQ Streaming
# ══════════════════════════════════════════════════════════════════════════════

def stream_nq(n_questions: int = 50000) -> tuple[dict[str, str], list[dict]]:
    """Stream NQ dataset, extract unique articles and QA pairs.

    Returns:
        articles: {title: clean_text}
        qa_pairs: list of {question, answer, title, has_answer}
    """
    from datasets import load_dataset

    print(f"Streaming Natural Questions (target: {n_questions:,} questions)...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )

    articles = {}  # title → clean text
    qa_pairs = []
    n_with_answer = 0
    t0 = time.time()

    for i, item in enumerate(ds):
        if i >= n_questions:
            break

        if i % 2000 == 0 and i > 0:
            rate = i / (time.time() - t0)
            eta = (n_questions - i) / rate
            print(f"  {i:>6,}/{n_questions:,} | "
                  f"{len(articles):,} articles | "
                  f"{n_with_answer:,} answered | "
                  f"{rate:.0f} q/s | ETA {eta:.0f}s")

        # ── Extract article ───────────────────────────────────────────
        title = item["document"]["title"]
        if title not in articles:
            text = extract_clean_text(item["document"])
            if len(text.split()) >= 20:
                articles[title] = text

        # ── Extract QA pair ───────────────────────────────────────────
        question = item["question"]["text"]

        # Short answer (direct text from NQ annotations)
        sa = item["annotations"]["short_answers"][0]
        answer = sa["text"][0] if sa["text"] else ""

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "title": title,
            "has_answer": len(answer) > 0,
        })
        if answer:
            n_with_answer += 1

    elapsed = time.time() - t0
    print(f"\n  Streamed {len(qa_pairs):,} questions in {elapsed:.0f}s")
    print(f"  {len(articles):,} unique articles")
    print(f"  {n_with_answer:,} with short answers "
          f"({n_with_answer/len(qa_pairs)*100:.0f}%)")

    return articles, qa_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Base Map Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_nq_base_map(
    articles: dict[str, str],
    batch_size: int = 512,
    max_chunks_per_article: int = 75,
) -> dict:
    """Chunk all articles, embed, build FAISS IVF index.

    max_chunks_per_article caps memory: 32K articles × 75 = 2.4M chunks = ~3.7GB.
    At 75 chunks (30-word window, 15 stride), we cover ~1100 unique words per
    article — the intro and core sections where most answers live.
    """
    from sentence_transformers import SentenceTransformer

    # ── Chunk ─────────────────────────────────────────────────────────
    print(f"\nChunking {len(articles):,} articles "
          f"(max {max_chunks_per_article} chunks/article)...")
    chunks = []
    chunk_meta = []
    article_titles = list(articles.keys())
    capped = 0

    for art_idx, title in enumerate(article_titles):
        art_chunks = chunk_text(articles[title])
        if len(art_chunks) > max_chunks_per_article:
            art_chunks = art_chunks[:max_chunks_per_article]
            capped += 1
        for pos, chunk in enumerate(art_chunks):
            chunks.append(chunk)
            chunk_meta.append({"article_idx": art_idx, "position": pos})

    n_chunks = len(chunks)
    print(f"  {n_chunks:,} chunks ({capped:,} articles capped)")
    print(f"  {n_chunks / len(articles):.0f} chunks/article avg")
    print(f"  Estimated embedding size: {n_chunks * 384 * 4 / 1e9:.2f} GB")

    # ── Embed ─────────────────────────────────────────────────────────
    print(f"\nLoading encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {n_chunks:,} chunks...")
    t0 = time.time()
    embeddings = encoder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    emb_time = time.time() - t0
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size:  {embeddings.nbytes / 1e9:.2f} GB")
    print(f"  Time:  {emb_time:.0f}s ({n_chunks / emb_time:.0f} chunks/s)")

    # ── FAISS IVF Index ───────────────────────────────────────────────
    print(f"\nBuilding FAISS index...")
    d = embeddings.shape[1]
    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)

    if n_chunks > 500_000:
        nlist = min(4096, int(np.sqrt(n_chunks) * 2))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        print(f"  Training IVF (nlist={nlist})...")
        index.train(emb_norm)
        index.nprobe = 64
        index.add(emb_norm)
        index_type = f"IndexIVFFlat(nlist={nlist}, nprobe=64)"
    else:
        index = faiss.IndexFlatIP(d)
        index.add(emb_norm)
        index_type = "IndexFlatIP"

    print(f"  {index.ntotal:,} vectors in R^{d} [{index_type}]")

    return {
        "chunks": chunks,
        "metadata": chunk_meta,
        "embeddings": embeddings,
        "embeddings_norm": emb_norm,
        "index": index,
        "encoder": encoder,
        "article_titles": article_titles,
        "index_type": index_type,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save & Verify
# ══════════════════════════════════════════════════════════════════════════════

def save_nq_base_map(base_map, n_articles, qa_pairs, elapsed):
    """Persist everything for Sections 2 & 3."""
    print(f"\nSaving base map...")

    np.save(OUTPUT_DIR / "embeddings.npy", base_map["embeddings"])
    np.save(OUTPUT_DIR / "embeddings_norm.npy", base_map["embeddings_norm"])
    faiss.write_index(base_map["index"], str(OUTPUT_DIR / "faiss_index.bin"))

    with open(OUTPUT_DIR / "chunk_texts.json", "w") as f:
        json.dump(base_map["chunks"], f)
    with open(OUTPUT_DIR / "chunk_meta.json", "w") as f:
        json.dump(base_map["metadata"], f)
    with open(OUTPUT_DIR / "articles.json", "w") as f:
        json.dump(base_map["article_titles"], f)
    with open(OUTPUT_DIR / "qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f)

    n_chunks = len(base_map["chunks"])
    n_answered = sum(1 for q in qa_pairs if q["has_answer"])
    log = {
        "experiment": "v5_nq_ingest",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_articles": n_articles,
        "n_chunks": n_chunks,
        "n_qa_pairs": len(qa_pairs),
        "n_with_answer": n_answered,
        "embedding_dim": int(base_map["embeddings"].shape[1]),
        "embedding_size_gb": round(base_map["embeddings"].nbytes / 1e9, 2),
        "chunks_per_article_avg": round(n_chunks / max(n_articles, 1), 1),
        "index_type": base_map["index_type"],
    }
    with open(OUTPUT_DIR / "ingest_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size = p.stat().st_size
        if size < 1e6:
            print(f"  {p.name:30s} {size/1e3:>8.1f} KB")
        elif size < 1e9:
            print(f"  {p.name:30s} {size/1e6:>8.1f} MB")
        else:
            print(f"  {p.name:30s} {size/1e9:>8.2f} GB")


def sanity_check(base_map, qa_pairs, n_queries=8):
    """Retrieval test on random answerable queries."""
    import random

    print(f"\n{'='*70}")
    print(f"  SANITY CHECK — {n_queries} queries")
    print(f"{'='*70}")

    encoder = base_map["encoder"]
    answerable = [q for q in qa_pairs if q["has_answer"]]
    test_qa = random.sample(answerable, min(n_queries, len(answerable)))

    hits_at_1 = 0
    hits_at_5 = 0
    answer_in_chunk = 0

    for qa in test_qa:
        q = qa["question"]
        answer = qa["answer"]
        expected_title = qa["title"]

        q_emb = encoder.encode([q], convert_to_numpy=True)
        q_norm = q_emb.copy()
        faiss.normalize_L2(q_norm)
        scores, indices = base_map["index"].search(q_norm, 10)

        top_titles = []
        for idx in indices[0]:
            if idx < 0:
                continue
            t = base_map["article_titles"][base_map["metadata"][int(idx)]["article_idx"]]
            top_titles.append(t)

        hit1 = top_titles[0] == expected_title if top_titles else False
        hit5 = expected_title in top_titles[:5]
        if hit1: hits_at_1 += 1
        if hit5: hits_at_5 += 1

        # Check if answer text appears in any top-5 chunk
        for idx in indices[0][:5]:
            if idx >= 0 and answer.lower() in base_map["chunks"][int(idx)].lower():
                answer_in_chunk += 1
                break

        print(f"\n  Q: {q[:70]}")
        print(f"  A: {answer[:50]}")
        print(f"  Article: {expected_title[:40]}")
        for rank in range(min(3, len(indices[0]))):
            idx = int(indices[0][rank])
            if idx < 0: continue
            meta = base_map["metadata"][idx]
            title = base_map["article_titles"][meta["article_idx"]]
            chunk = base_map["chunks"][idx]
            mark = "HIT" if title == expected_title else "   "
            print(f"    [{rank}] {mark} [{title[:25]}] "
                  f"sim={scores[0][rank]:.3f} | {chunk[:45]}...")

    n = len(test_qa)
    print(f"\n  Article @1: {hits_at_1}/{n}  |  "
          f"Article @5: {hits_at_5}/{n}  |  "
          f"Answer in chunk @5: {answer_in_chunk}/{n}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_nq_ingest(n_questions: int = 50000):
    """Full NQ-scale ingest pipeline."""
    import gc
    start = time.time()

    articles, qa_pairs = stream_nq(n_questions)
    n_articles = len(articles)
    base_map = build_nq_base_map(articles)

    # Free raw article text (~4GB) — we have the chunks now
    del articles
    gc.collect()

    elapsed = time.time() - start
    save_nq_base_map(base_map, n_articles, qa_pairs, elapsed)
    sanity_check(base_map, qa_pairs)

    n_chunks = len(base_map["chunks"])
    n_answered = sum(1 for q in qa_pairs if q["has_answer"])
    print(f"\n{'='*70}")
    print(f"  NQ INGEST COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")
    print(f"  {n_articles:,} articles → {n_chunks:,} chunks")
    print(f"  {len(qa_pairs):,} QA pairs ({n_answered:,} answered)")
    print(f"  Embeddings: {base_map['embeddings'].nbytes / 1e9:.2f} GB")
    print(f"  Index: {base_map['index_type']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=50000,
                        help="Number of NQ questions to process")
    args = parser.parse_args()
    run_nq_ingest(args.n_questions)
