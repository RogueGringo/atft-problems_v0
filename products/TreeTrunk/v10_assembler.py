#!/usr/bin/env python3
"""v10 Topological Assembler — Active multi-hop reasoning via gradient descent on λ₁.

The engine transforms from static filter to active reasoning agent.

Algorithm:
  1. Parse query → Q_graph (dependency tree)
  2. FAISS retrieval → 50 candidate chunks
  3. Hop loop:
     - For each unused candidate, compute λ₁ if it were merged into
       the current assembled graph
     - Select the chunk that most REDUCES λ₁ (maximum gradient)
     - Add it to the assembly
     - Check convergence: if λ₁ plateaus or drops below threshold, stop
  4. Return the assembled multi-chunk graph

Each hop is a physical measurement: "which chunk completes the logical
structure?" No learned search policy — just gradient descent on the
spectral gap.
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

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
V9_CKPT = Path(__file__).parent / "results" / "v9_checkpoints" / "v9_final.pt"

from v9_causal_graph import (
    TextCausalGrapher, TypedSheafTopologyLoss, CausalGraph
)


class TopologicalAssembler:
    """Active multi-hop reasoning via λ₁ gradient descent."""

    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("Loading Topological Assembler...")

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.grapher = TextCausalGrapher("en_core_web_sm")

        # FAISS index
        self.index = faiss.read_index(str(BASE_MAP_DIR / "faiss_index.bin"))
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 64

        with open(BASE_MAP_DIR / "chunk_texts.json") as f:
            self.chunks = json.load(f)
        with open(BASE_MAP_DIR / "chunk_meta.json") as f:
            self.chunk_meta = json.load(f)
        with open(BASE_MAP_DIR / "articles.json") as f:
            self.article_titles = json.load(f)

        # Load trained v9 sheaf
        self.sheaf_loss = TypedSheafTopologyLoss(d_in=384, stalk_dim=8, margin=0.5).to(self.device)
        state = torch.load(V9_CKPT, map_location=self.device, weights_only=False)
        for key in state:
            if key.startswith("sheaf.restriction_maps."):
                map_id = key.split(".")[2]
                self.sheaf_loss.sheaf._get_restriction(map_id)
        self.sheaf_loss.load_state_dict(state)
        self.sheaf_loss.eval()

        print(f"  {self.index.ntotal:,} chunks in index")
        print(f"  {len(self.sheaf_loss.sheaf.restriction_maps)} trained restriction maps")
        print(f"  Ready.")

    def _encode_graph(self, graph: CausalGraph) -> torch.Tensor:
        """Encode graph nodes to 384-dim tensor."""
        if not graph.node_texts:
            return torch.zeros(1, 384, device=self.device)
        embs = self.encoder.encode(
            graph.node_texts, convert_to_numpy=True, show_progress_bar=False
        )
        return torch.tensor(embs, dtype=torch.float32, device=self.device)

    def _compute_gap(
        self, node_emb: torch.Tensor, edges: list[tuple[int, int, str]]
    ) -> float:
        """Compute spectral gap λ₁ via the trained typed sheaf."""
        if len(edges) == 0 or node_emb.shape[0] < 2:
            return 0.0
        with torch.no_grad():
            eigs = self.sheaf_loss.sheaf(node_emb, edges)
            nz = eigs[eigs > 1e-6]
            return float(nz[0].item()) if len(nz) > 0 else 0.0

    def _merge_graphs(
        self,
        graphs: list[CausalGraph],
    ) -> tuple[list[str], list[tuple[int, int, str]]]:
        """Merge multiple graphs with bridge edges between them."""
        merged_nodes = []
        merged_edges = []
        offset = 0

        for i, g in enumerate(graphs):
            merged_nodes.extend(g.node_texts)
            for s, d, t in g.edges:
                merged_edges.append((s + offset, d + offset, t))

            # Bridge edge to previous graph's last node
            if i > 0 and len(g.node_texts) > 0 and offset > 0:
                merged_edges.append((offset - 1, offset, "bridge"))

            offset += len(g.node_texts)

        return merged_nodes, merged_edges

    def assemble(
        self,
        query: str,
        max_hops: int = 5,
        k_faiss: int = 50,
        k_search_per_hop: int = 15,
        convergence_threshold: float = 1e-3,
        verbose: bool = True,
    ) -> dict:
        """Build a multi-hop answer graph via λ₁ gradient descent.

        Returns dict with assembled chunks, λ₁ trajectory, and metadata.
        """
        t0 = time.time()

        # ── Parse query ───────────────────────────────────────────────
        q_graph = self.grapher.parse(query)

        # ── FAISS retrieval ───────────────────────────────────────────
        q_emb_raw = self.encoder.encode([query], convert_to_numpy=True)
        q_norm = q_emb_raw.copy()
        faiss.normalize_L2(q_norm)
        scores, indices = self.index.search(q_norm, k_faiss)
        scores = scores[0]
        candidate_indices = [int(i) for i in indices[0] if i >= 0]

        # Pre-parse top candidates
        candidate_texts = [self.chunks[i] for i in candidate_indices]
        candidate_graphs = []
        for text in candidate_texts:
            candidate_graphs.append(self.grapher.parse(text))

        # ── Hop loop: gradient descent on λ₁ ──────────────────────────
        assembled_idx: list[int] = []
        assembled_graphs: list[CausalGraph] = [q_graph]
        lambda_trajectory = []
        hop_details = []

        # Initial λ₁: query graph alone
        q_nodes, q_edges = self._merge_graphs([q_graph])
        q_node_emb = self._encode_graph(q_graph)
        initial_gap = self._compute_gap(q_node_emb, q_edges)
        lambda_trajectory.append(initial_gap)

        if verbose:
            print(f"\n  Initial λ₁(query): {initial_gap:.6f}")

        for hop in range(max_hops):
            current_nodes, current_edges = self._merge_graphs(assembled_graphs)
            current_graph_as_cg = CausalGraph(current_nodes, current_edges)
            current_emb = self._encode_graph(current_graph_as_cg)
            current_gap = self._compute_gap(current_emb, current_edges)

            # Search only top-k_search_per_hop candidates for speed
            search_pool = [
                (i, candidate_indices[i]) for i in range(min(k_search_per_hop, len(candidate_indices)))
                if candidate_indices[i] not in assembled_idx
            ]

            if not search_pool:
                break

            # Measure marginal gap reduction for each candidate
            best_local_idx = None
            best_gap = float('inf')
            best_reduction = 0.0

            for local_i, chunk_idx in search_pool:
                cand_graph = candidate_graphs[local_i]

                # Try merging
                trial_graphs = assembled_graphs + [cand_graph]
                trial_nodes, trial_edges = self._merge_graphs(trial_graphs)
                trial_merged_cg = CausalGraph(trial_nodes, trial_edges)
                trial_emb = self._encode_graph(trial_merged_cg)
                trial_gap = self._compute_gap(trial_emb, trial_edges)

                reduction = current_gap - trial_gap
                if trial_gap < best_gap:
                    best_gap = trial_gap
                    best_local_idx = local_i
                    best_reduction = reduction

            if best_local_idx is None:
                break

            # Add the best chunk
            best_chunk_idx = candidate_indices[best_local_idx]
            best_cand_graph = candidate_graphs[best_local_idx]
            assembled_graphs.append(best_cand_graph)
            assembled_idx.append(best_chunk_idx)
            lambda_trajectory.append(best_gap)

            meta = self.chunk_meta[best_chunk_idx]
            title = self.article_titles[meta["article_idx"]]

            hop_detail = {
                "hop": hop + 1,
                "chunk_idx": best_chunk_idx,
                "title": title,
                "text_preview": self.chunks[best_chunk_idx][:80],
                "lambda1": best_gap,
                "reduction": best_reduction,
                "cosine_rank": best_local_idx,
            }
            hop_details.append(hop_detail)

            if verbose:
                print(f"  Hop {hop+1}: λ₁={best_gap:.6f} "
                      f"(Δ={best_reduction:+.6f}) "
                      f"[{title[:30]}] cos_rank={best_local_idx}")

            # Convergence checks
            if best_reduction < convergence_threshold:
                if verbose:
                    print(f"  Converged: Δ<{convergence_threshold}")
                break
            if best_gap < convergence_threshold:
                if verbose:
                    print(f"  Converged: λ₁<{convergence_threshold}")
                break

        elapsed = time.time() - t0

        return {
            "query": query,
            "assembled_chunks": [
                {"idx": idx, "title": self.article_titles[self.chunk_meta[idx]["article_idx"]],
                 "text": self.chunks[idx]}
                for idx in assembled_idx
            ],
            "lambda_trajectory": lambda_trajectory,
            "hops": hop_details,
            "n_hops": len(assembled_idx),
            "final_lambda1": lambda_trajectory[-1] if lambda_trajectory else 0,
            "elapsed_ms": elapsed * 1000,
        }


def run_demo(queries: list[str] | None = None):
    """Run assembler on demo queries."""
    if queries is None:
        queries = [
            "how many bones are in the human body",
            "who wrote the declaration of independence",
            "when did the berlin wall fall",
            "what causes the northern lights",
            "who was the first president of the united states",
        ]

    asm = TopologicalAssembler()

    print(f"\n{'='*75}")
    print(f"  v10 TOPOLOGICAL ASSEMBLER — Multi-hop reasoning")
    print(f"{'='*75}")

    for q in queries:
        print(f"\n  Q: {q}")
        result = asm.assemble(q, max_hops=5, verbose=True)

        print(f"\n  Assembled {result['n_hops']} chunks in {result['elapsed_ms']:.0f}ms")
        print(f"  λ₁ trajectory: {[f'{x:.4f}' for x in result['lambda_trajectory']]}")
        print(f"  Final λ₁: {result['final_lambda1']:.6f}")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    run_demo()
