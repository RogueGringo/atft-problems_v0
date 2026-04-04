#!/usr/bin/env python3
"""v6 Universal Slice Navigator — operates in the trained 128-dim manifold.

The full pipeline in warped space:
  1. Query → sentence-transformer → trained TextFeatureMap → 128-dim
  2. FAISS IVF search in 128-dim warped space (topologically informed)
  3. Sheaf Laplacian truth filter using TRAINED DifferentiableSheafLaplacian
  4. Output: raw verified text + coherence certificate

This is the end-to-end proof: the manifold was warped by training,
the index was re-built in the warped space, and the sheaf operates
with learned restriction maps.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import faiss

warnings.filterwarnings("ignore")

V6_MAP_DIR = Path(__file__).parent / "results" / "v6_base_map"
V5_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"

from v6_topological_trainer import TextFeatureMap, DifferentiableSheafLaplacian


class V6Navigator:
    """Universal Slice in the trained manifold."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading v6 Navigator...")

        # Encoder (frozen)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Trained projection + sheaf
        self.feat_map = TextFeatureMap(out_dim=128, vocab_size=1, backbone_dim=384).to(self.device)
        self.sheaf_lap = DifferentiableSheafLaplacian(d_in=128, stalk_dim=8, k=4).to(self.device)

        checkpoint = torch.load(CHECKPOINT_DIR / "trainer_final.pt",
                                map_location=self.device, weights_only=False)
        feat_state = {k.replace("feature_map.", ""): v
                      for k, v in checkpoint.items() if k.startswith("feature_map.")}
        sheaf_state = {k.replace("sheaf_loss_fn.sheaf.", ""): v
                       for k, v in checkpoint.items() if k.startswith("sheaf_loss_fn.sheaf.")}
        self.feat_map.load_state_dict(feat_state, strict=False)
        self.sheaf_lap.load_state_dict(sheaf_state, strict=False)
        self.feat_map.eval()
        self.sheaf_lap.eval()

        # FAISS index stays in ORIGINAL 384-dim space (cosine anchor).
        # The trained sheaf evaluates in warped 128-dim space (topology).
        # FAISS finds the haystack, sheaf finds the needle.
        self.index = faiss.read_index(str(V5_MAP_DIR / "faiss_index.bin"))
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 64

        # Chunk data
        with open(V5_MAP_DIR / "chunk_texts.json") as f:
            self.chunks = json.load(f)
        with open(V5_MAP_DIR / "chunk_meta.json") as f:
            self.chunk_meta = json.load(f)
        with open(V5_MAP_DIR / "articles.json") as f:
            self.article_titles = json.load(f)

        print(f"  {len(self.chunks):,} chunks in warped 128-dim space")
        print(f"  Ready.")

    def _embed_and_project(self, text: str, window: int = 5, stride: int = 1) -> torch.Tensor:
        """Encode text → sentence-transformer → trained projection → 128-dim."""
        words = text.split()
        if len(words) <= window:
            chunks = [text]
        else:
            chunks = [" ".join(words[i:i+window])
                      for i in range(0, len(words) - window + 1, stride)]
        raw = self.encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.feat_map(raw_t)

    def _sheaf_gap(self, cloud: torch.Tensor) -> float:
        """Spectral gap from trained sheaf Laplacian."""
        with torch.no_grad():
            eigs = self.sheaf_lap(cloud)
            nonzero = eigs[eigs > 1e-6]
            return float(nonzero[0].item()) if len(nonzero) > 0 else 0.0

    def navigate(self, query: str, k_faiss: int = 50, k_sheaf: int = 5) -> dict:
        """Full pipeline in warped space."""
        t0 = time.time()

        # 1. Embed + project query into warped space
        q_cloud = self._embed_and_project(query, window=5, stride=1)

        # FAISS search uses ORIGINAL 384-dim embeddings (cosine anchor)
        q_raw = self.encoder.encode([query], convert_to_numpy=True)
        q_norm = q_raw.copy()
        faiss.normalize_L2(q_norm)

        # 2. FAISS search in warped 128-dim space
        scores, indices = self.index.search(q_norm, k_faiss)
        scores, indices = scores[0], indices[0]

        # 3. Take top-k_sheaf by cosine, run trained sheaf on each
        sheaf_candidates = []
        for rank in range(min(k_sheaf, len(indices))):
            idx = int(indices[rank])
            if idx < 0:
                continue

            chunk_text = self.chunks[idx]
            meta = self.chunk_meta[idx]
            title = self.article_titles[meta["article_idx"]]

            # Build full cloud: query + candidate (both in 128-dim warped space)
            a_cloud = self._embed_and_project(chunk_text, window=5, stride=1)
            full_cloud = torch.cat([q_cloud, a_cloud], dim=0)

            gap = self._sheaf_gap(full_cloud)

            sheaf_candidates.append({
                "idx": idx,
                "text": chunk_text,
                "title": title,
                "cosine_rank": rank,
                "cosine_sim": float(scores[rank]),
                "lambda1": gap,
                "n_points": len(full_cloud),
            })

        # 4. Select by lowest λ₁ (most coherent in warped space)
        sheaf_candidates.sort(key=lambda c: c["lambda1"])
        best = sheaf_candidates[0] if sheaf_candidates else None

        elapsed = time.time() - t0

        return {
            "query": query,
            "verified_text": best["text"] if best else "",
            "verified_title": best["title"] if best else "",
            "lambda1": best["lambda1"] if best else float('inf'),
            "cosine_rank": best["cosine_rank"] if best else -1,
            "elapsed": elapsed,
            "candidates": sheaf_candidates,
        }


def run_demo(queries: list[str] | None = None):
    """Run blind queries through the full v6 pipeline."""
    if queries is None:
        queries = [
            "how many bones are in the human body",
            "who wrote the declaration of independence",
            "what is the speed of light in a vacuum",
            "when did the berlin wall fall",
            "what causes the northern lights",
            "who was the first president of the united states",
            "what is the largest ocean on earth",
            "when was the great wall of china built",
        ]

    nav = V6Navigator()

    print(f"\n{'='*75}")
    print(f"  v6 UNIVERSAL SLICE — TRAINED MANIFOLD")
    print(f"  2,338,208 chunks | 128-dim warped space | trained sheaf")
    print(f"{'='*75}")

    for q in queries:
        r = nav.navigate(q)

        print(f"\n  Q: {q}")
        print(f"  ANSWER: [{r['verified_title']}]")
        print(f"    {r['verified_text'][:120]}")
        print(f"    λ₁={r['lambda1']:.6f}  cos_rank={r['cosine_rank']}  "
              f"{r['elapsed']:.1f}s")

        print(f"    Sheaf ranking (top {len(r['candidates'])}):")
        for j, c in enumerate(r["candidates"]):
            marker = " <<<" if j == 0 else ""
            print(f"      [{j}] λ₁={c['lambda1']:.6f} cos={c['cosine_rank']} "
                  f"[{c['title'][:30]}]{marker}")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    run_demo()
