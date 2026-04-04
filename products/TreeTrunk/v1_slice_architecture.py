#!/usr/bin/env python3
"""v1 Slice Architecture — Topological Navigation, Not Generation.

The core insight: inference is structural alignment, not token prediction.

Components:
  1. Base Map: dense corpus embedded as point cloud in R^d (the knowledge)
  2. Stencil: learned Q→A trajectory shape from I/O data (the navigation pattern)
  3. Navigator: given input, follow the stencil through the base map
  4. Reader: read the text at the destination coordinate

No generation. No hallucination. Structural alignment in a space
that already contains the answer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from typing import Optional

OUTPUT_DIR = Path(__file__).parent / "results" / "v1_slice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: The Continuous Base Map
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaseMap:
    """Pre-embedded knowledge substrate."""
    embeddings: np.ndarray    # (N, d) — every chunk as a point in R^d
    texts: list[str]          # (N,) — the raw text for each chunk
    metadata: dict            # source info


def build_base_map(
    dataset_name: str = "wikipedia",
    topic: str = "mathematics",
    n_chunks: int = 5000,
    chunk_words: int = 50,
    overlap_words: int = 10,
    encoder_name: str = "all-MiniLM-L6-v2",
) -> BaseMap:
    """Embed a dense corpus into R^d.

    Uses a focused Wikipedia subset for density — every point
    should be semantically meaningful, not padding.
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"Building base map: {dataset_name}/{topic}, {n_chunks} chunks...")

    # Load corpus
    if dataset_name == "wikipedia":
        print("  Loading Wikipedia...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        # Filter for topic-relevant articles
        texts = []
        for item in ds:
            title = item.get("title", "").lower()
            text = item.get("text", "")
            if len(text) < 200:
                continue
            # Take articles related to topic, plus general articles for breadth
            if topic.lower() in title or len(texts) % 3 == 0:
                texts.append(text)
            if len(texts) >= n_chunks // 5:  # will expand via chunking
                break
        print(f"  Collected {len(texts)} articles")
    elif dataset_name == "gsm8k_solutions":
        print("  Loading GSM8K solutions as base map...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [item["question"] + "\n" + item["answer"] for item in ds]
        print(f"  Collected {len(texts)} solution chains")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Chunk into overlapping windows
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, max(1, len(words) - chunk_words + 1), chunk_words - overlap_words):
            chunk = " ".join(words[i:i + chunk_words])
            if len(chunk.split()) >= 10:  # skip tiny chunks
                chunks.append(chunk)
            if len(chunks) >= n_chunks:
                break
        if len(chunks) >= n_chunks:
            break

    chunks = chunks[:n_chunks]
    print(f"  {len(chunks)} chunks of ~{chunk_words} words")

    # Embed
    print(f"  Encoding with {encoder_name}...")
    encoder = SentenceTransformer(encoder_name)
    embeddings = encoder.encode(chunks, convert_to_numpy=True,
                                show_progress_bar=True, batch_size=128)

    print(f"  Base map: {embeddings.shape[0]} points in R^{embeddings.shape[1]}")

    return BaseMap(
        embeddings=embeddings,
        texts=chunks,
        metadata={
            "dataset": dataset_name,
            "topic": topic,
            "n_chunks": len(chunks),
            "dim": embeddings.shape[1],
            "encoder": encoder_name,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: The Stencil Extractor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stencil:
    """Learned Q→A trajectory shape."""
    mean_displacement: np.ndarray    # (d,) — average Q→A vector
    displacement_cov: np.ndarray     # (d, d) — covariance of displacements
    q_embeddings: np.ndarray         # (N_io, d) — question embeddings
    a_embeddings: np.ndarray         # (N_io, d) — answer embeddings
    displacements: np.ndarray        # (N_io, d) — individual Q→A vectors
    h0_bars_truth: list[float]       # H0 bar lengths for coherent Q→A paths
    metadata: dict


def extract_stencil(
    n_samples: int = 500,
    encoder_name: str = "all-MiniLM-L6-v2",
) -> Stencil:
    """Extract the Q→A trajectory shape from GSM8K.

    For each (question, answer) pair:
      - Embed Q as a single point
      - Embed A (full solution) as a single point
      - The displacement Q→A is one sample of the stencil

    The stencil = the DISTRIBUTION of these displacements.
    It encodes: "what direction and distance does a valid answer
    live from its question in embedding space?"

    Also computes H0 persistence on the combined Q+A path
    to establish the "topological tightness" of coherent pairs.
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from ripser import ripser

    print(f"Extracting stencil from GSM8K ({n_samples} pairs)...")

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(min(n_samples, len(ds))))

    questions = [item["question"] for item in ds]
    answers = [item["answer"] for item in ds]

    print(f"  Encoding {len(questions)} Q/A pairs...")
    encoder = SentenceTransformer(encoder_name)
    q_emb = encoder.encode(questions, convert_to_numpy=True,
                           show_progress_bar=True, batch_size=128)
    a_emb = encoder.encode(answers, convert_to_numpy=True,
                           show_progress_bar=True, batch_size=128)

    # Displacement vectors: Q → A
    displacements = a_emb - q_emb  # (N, d)
    mean_disp = displacements.mean(axis=0)  # (d,)
    cov_disp = np.cov(displacements.T)  # (d, d)

    print(f"  Mean displacement norm: {np.linalg.norm(mean_disp):.4f}")
    print(f"  Displacement std: {np.std(np.linalg.norm(displacements, axis=1)):.4f}")

    # H0 persistence on each Q→A path (embed Q steps + A steps together)
    print(f"  Computing H0 persistence on Q→A paths...")
    h0_bars = []
    for i in range(min(200, len(questions))):
        # Embed Q and A at finer granularity for persistence
        q_words = questions[i].split()
        a_words = answers[i].split()
        all_words = q_words + a_words

        # Sliding window
        window = 5
        stride = 3
        chunks = []
        for j in range(0, max(1, len(all_words) - window + 1), stride):
            chunks.append(" ".join(all_words[j:j + window]))

        if len(chunks) < 3:
            chunks = [questions[i], answers[i]]

        path_emb = encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        if len(path_emb) > 100:
            idx = np.random.choice(len(path_emb), 100, replace=False)
            path_emb = path_emb[idx]

        result = ripser(path_emb, maxdim=0)
        dgm = result["dgms"][0]
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) > 0:
            bars = finite[:, 1] - finite[:, 0]
            h0_bars.append(float(bars.mean()))
        else:
            h0_bars.append(0.0)

    print(f"  Mean H0 bar (truth paths): {np.mean(h0_bars):.4f}")

    return Stencil(
        mean_displacement=mean_disp,
        displacement_cov=cov_disp,
        q_embeddings=q_emb,
        a_embeddings=a_emb,
        displacements=displacements,
        h0_bars_truth=h0_bars,
        metadata={
            "n_pairs": len(questions),
            "dim": q_emb.shape[1],
            "mean_disp_norm": float(np.linalg.norm(mean_disp)),
            "encoder": encoder_name,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: The Navigator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NavigationResult:
    """Result of navigating from input to destination."""
    input_text: str
    input_embedding: np.ndarray
    destination_embedding: np.ndarray
    destination_text: str
    destination_idx: int
    distance: float
    h0_tightness: float           # H0 bar length of the path (lower = tighter)
    top_k_texts: list[str]        # nearest k chunks at destination
    top_k_distances: list[float]


def navigate(
    query: str,
    base_map: BaseMap,
    stencil: Stencil,
    encoder,
    k: int = 5,
) -> NavigationResult:
    """Navigate from query to answer via stencil.

    1. Embed query → q_point in R^d
    2. Apply stencil: destination = q_point + mean_displacement
    3. Find nearest chunks in base map to destination
    4. Measure H0 tightness of the path
    5. Return the text at the destination
    """
    from ripser import ripser

    # 1. Embed query
    q_emb = encoder.encode([query], convert_to_numpy=True)[0]

    # 2. Apply stencil — navigate to where the answer should be
    dest_emb = q_emb + stencil.mean_displacement

    # 3. Find nearest chunks in base map
    # Cosine similarity (normalized dot product)
    base_norms = np.linalg.norm(base_map.embeddings, axis=1, keepdims=True)
    dest_norm = np.linalg.norm(dest_emb)
    cosine_sims = (base_map.embeddings @ dest_emb) / (base_norms.squeeze() * dest_norm + 1e-8)

    top_k_idx = np.argsort(-cosine_sims)[:k]
    top_k_texts = [base_map.texts[i] for i in top_k_idx]
    top_k_dists = [float(1 - cosine_sims[i]) for i in top_k_idx]

    # 4. Measure H0 tightness of the Q→destination path
    path_points = np.vstack([
        q_emb.reshape(1, -1),
        dest_emb.reshape(1, -1),
        base_map.embeddings[top_k_idx],
    ])
    result = ripser(path_points, maxdim=0)
    dgm = result["dgms"][0]
    finite = dgm[dgm[:, 1] < np.inf]
    h0_tight = float((finite[:, 1] - finite[:, 0]).mean()) if len(finite) > 0 else 0.0

    return NavigationResult(
        input_text=query,
        input_embedding=q_emb,
        destination_embedding=dest_emb,
        destination_text=top_k_texts[0],
        destination_idx=int(top_k_idx[0]),
        distance=top_k_dists[0],
        h0_tightness=h0_tight,
        top_k_texts=top_k_texts,
        top_k_distances=top_k_dists,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_v1(n_base_chunks=5000, n_stencil_pairs=500, n_test=20):
    """Run the full slice architecture pipeline."""
    start = time.time()

    # Step 1: Build base map from GSM8K solutions
    # (so the answers actually exist in the base map)
    base_map = build_base_map(
        dataset_name="gsm8k_solutions",
        n_chunks=n_base_chunks,
        chunk_words=50,
        overlap_words=10,
    )

    # Step 2: Extract stencil
    stencil = extract_stencil(n_samples=n_stencil_pairs)

    # Step 3: Test navigation on held-out questions
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"\nTesting navigation on {n_test} held-out questions...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(ds.select(range(min(n_test, len(ds)))))

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    for i, item in enumerate(test_items):
        q = item["question"]
        true_answer = item["answer"]

        nav = navigate(q, base_map, stencil, encoder, k=5)

        # Check: does the navigated destination contain relevant content?
        # Simple relevance: does the destination text share key numbers/concepts?
        true_final = true_answer.split("####")[-1].strip() if "####" in true_answer else ""

        print(f"\n{'─'*70}")
        print(f"  Q{i}: {q[:100]}...")
        print(f"  True answer: ...#### {true_final}")
        print(f"  Navigator found (cosine dist={nav.distance:.4f}, H0={nav.h0_tightness:.4f}):")
        print(f"    → {nav.destination_text[:150]}...")
        print(f"  Top-5 destinations:")
        for j, (txt, dist) in enumerate(zip(nav.top_k_texts, nav.top_k_distances)):
            print(f"    [{j}] dist={dist:.4f}: {txt[:100]}...")

        results.append({
            "question": q,
            "true_final": true_final,
            "nav_text": nav.destination_text,
            "nav_distance": nav.distance,
            "nav_h0": nav.h0_tightness,
            "top_k": [{"text": t[:200], "dist": d}
                      for t, d in zip(nav.top_k_texts, nav.top_k_distances)],
        })

    elapsed = time.time() - start

    # Summary
    avg_dist = np.mean([r["nav_distance"] for r in results])
    avg_h0 = np.mean([r["nav_h0"] for r in results])

    print(f"\n{'='*70}")
    print(f"  V1 SLICE ARCHITECTURE — RESULTS")
    print(f"  {elapsed:.0f}s | base={base_map.metadata['n_chunks']} chunks | "
          f"stencil={stencil.metadata['n_pairs']} pairs | test={len(results)}")
    print(f"  Avg destination distance: {avg_dist:.4f}")
    print(f"  Avg H0 tightness: {avg_h0:.4f}")
    print(f"  Stencil displacement norm: {stencil.metadata['mean_disp_norm']:.4f}")
    print(f"{'='*70}")

    log = {
        "experiment": "v1_slice_architecture",
        "elapsed": elapsed,
        "base_map": base_map.metadata,
        "stencil": stencil.metadata,
        "avg_distance": float(avg_dist),
        "avg_h0": float(avg_h0),
        "results": results,
    }
    with open(OUTPUT_DIR / "v1_results.json", "w") as f:
        json.dump(log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Saved → {OUTPUT_DIR / 'v1_results.json'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_base", type=int, default=5000)
    parser.add_argument("--n_stencil", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=20)
    args = parser.parse_args()
    run_v1(args.n_base, args.n_stencil, args.n_test)
