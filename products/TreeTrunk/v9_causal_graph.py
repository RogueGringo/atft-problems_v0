#!/usr/bin/env python3
"""v9 Causal Graph Sheaf — Typed restriction maps over dependency graphs.

The sheaf no longer measures proximity. It calculates the physics of grammar.

Architecture:
  UniversalCausalGrapher  — abstract: any domain → (nodes, typed_edges)
  TextCausalGrapher       — spaCy dependency parse → typed edges
  MathCausalGrapher       — operator tree → typed edges [stub]
  CodeCausalGrapher       — Python AST → typed edges [stub]

  DependencyGraphSheafLaplacian — nn.Module
    - Accepts (node_embeddings, typed_edges)
    - Each edge type (nsubj, dobj, prep, ...) gets its own LEARNED
      restriction matrix via nn.ModuleDict
    - New edge types auto-instantiate on first encounter
    - L_F = δᵀδ with typed coboundary → eigendecomposition

  TypedSheafTopologyLoss — contrastive loss using typed sheaf
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. Universal Causal Grapher — domain-agnostic interface
# ══════════════════════════════════════════════════════════════════════════════

class CausalGraph:
    """The universal output: nodes + typed edges. Domain-agnostic."""
    __slots__ = ("node_texts", "edges")

    def __init__(self, node_texts: list[str], edges: list[tuple[int, int, str]]):
        """
        node_texts: list of text strings, one per node
        edges: list of (src_idx, dst_idx, edge_type) tuples
               edge_type is a string label (e.g., "nsubj", "dobj", "operand_left")
        """
        self.node_texts = node_texts
        self.edges = edges

    def __repr__(self):
        return f"CausalGraph({len(self.node_texts)} nodes, {len(self.edges)} edges)"


class UniversalCausalGrapher(ABC):
    """Abstract interface: raw input → CausalGraph."""

    @abstractmethod
    def parse(self, text: str) -> CausalGraph:
        """Convert domain-specific input into a universal CausalGraph."""
        ...


class TextCausalGrapher(UniversalCausalGrapher):
    """spaCy dependency parse → CausalGraph with syntactic typed edges.

    Each token becomes a node. Each dependency arc becomes a typed edge.
    Edge types: nsubj, dobj, amod, prep, pobj, det, aux, advmod, etc.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model)

    def parse(self, text: str) -> CausalGraph:
        doc = self.nlp(text)
        node_texts = [token.text for token in doc]
        edges = []
        for token in doc:
            if token.head != token:  # skip root self-loop
                edges.append((
                    token.head.i,  # source = head
                    token.i,       # target = dependent
                    token.dep_,    # edge type = dependency label
                ))
        return CausalGraph(node_texts, edges)


class MathCausalGrapher(UniversalCausalGrapher):
    """Operator tree for math expressions → CausalGraph.

    Edge types: operand_left, operand_right, equals, step_next.

    Parses GSM8K-style step-by-step solutions by splitting on
    sentence boundaries and connecting steps sequentially,
    with operator edges for mathematical expressions.

    TODO: Full symbolic math parser for equations.
    """

    def parse(self, text: str) -> CausalGraph:
        # Simple sentence-level graph with sequential edges
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".")
                     if s.strip()]
        if not sentences:
            return CausalGraph(["[empty]"], [])

        edges = []
        for i in range(len(sentences) - 1):
            edges.append((i, i + 1, "step_next"))

        # Check for mathematical operators within sentences
        for i, sent in enumerate(sentences):
            if "=" in sent or "+" in sent or "-" in sent or "*" in sent:
                if i > 0:
                    edges.append((i - 1, i, "computes"))

        return CausalGraph(sentences, edges)


class CodeCausalGrapher(UniversalCausalGrapher):
    """Python AST → CausalGraph.

    Edge types: function_def, return_val, loop_body, assignment, call.

    TODO: Full implementation with ast.parse.
    """

    def parse(self, text: str) -> CausalGraph:
        # Stub: line-level graph with sequential edges
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return CausalGraph(["[empty]"], [])

        edges = []
        for i in range(len(lines) - 1):
            edges.append((i, i + 1, "sequential"))

        # Detect basic patterns
        for i, line in enumerate(lines):
            if line.startswith("def "):
                for j in range(i + 1, min(i + 5, len(lines))):
                    edges.append((i, j, "function_body"))
            if line.startswith("return"):
                edges.append((i, 0, "return_val"))
            if "for " in line or "while " in line:
                if i + 1 < len(lines):
                    edges.append((i, i + 1, "loop_body"))

        return CausalGraph(lines, edges)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dependency Graph Sheaf Laplacian — typed restriction maps
# ══════════════════════════════════════════════════════════════════════════════

class TypedRestrictionMap(nn.Module):
    """Learnable restriction matrix for a specific edge type.

    Initialized as near-identity (small perturbation) so the sheaf
    starts measuring generic coherence, then learns type-specific
    geometric constraints through gradient pressure.
    """

    def __init__(self, stalk_dim: int):
        super().__init__()
        self.stalk_dim = stalk_dim
        # Initialize as identity + small random perturbation
        self.weight = nn.Parameter(
            torch.eye(stalk_dim) + 0.01 * torch.randn(stalk_dim, stalk_dim)
        )

    def forward(self, d_vec: torch.Tensor) -> torch.Tensor:
        """Compute the restriction map R for displacement vector d.

        Combines the learned type-specific rotation with the
        displacement-dependent Householder reflection.
        """
        d_norm = d_vec.norm() + 1e-12
        d_hat = d_vec / d_norm
        alpha = torch.clamp(d_norm, max=1.0)

        # Householder base: I - alpha * d_hat d_hat^T
        I = torch.eye(self.stalk_dim, device=d_vec.device, dtype=d_vec.dtype)
        householder = I - alpha * torch.outer(d_hat, d_hat)

        # Compose with learned type-specific rotation
        return self.weight @ householder


class DependencyGraphSheafLaplacian(nn.Module):
    """Sheaf Laplacian over a typed dependency graph.

    Unlike the v6 k-NN sheaf which uses generic edges, this sheaf:
    1. Receives a pre-built typed graph (from UniversalCausalGrapher)
    2. Applies a DIFFERENT learned restriction map per edge type
    3. New edge types auto-instantiate with fresh parameters

    The projection from backbone_dim to stalk_dim is also learned.
    """

    def __init__(self, d_in: int, stalk_dim: int = 8, eps: float = 1e-6):
        super().__init__()
        self.d_in = d_in
        self.stalk_dim = stalk_dim
        self.eps = eps

        # Learnable projection to stalk space
        self.proj = nn.Linear(d_in, stalk_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # Typed restriction maps — auto-expand on new edge types
        self.restriction_maps = nn.ModuleDict()

        # Default map for unknown edge types
        self.default_map = TypedRestrictionMap(stalk_dim)

    def _get_restriction(self, edge_type: str) -> TypedRestrictionMap:
        """Get or create the restriction map for an edge type."""
        # Sanitize key for nn.ModuleDict (no special chars)
        key = edge_type.replace(":", "_").replace("-", "_")
        if key not in self.restriction_maps:
            self.restriction_maps[key] = TypedRestrictionMap(self.stalk_dim).to(
                self.proj.weight.device
            )
        return self.restriction_maps[key]

    def forward(
        self,
        node_embeddings: torch.Tensor,
        typed_edges: list[tuple[int, int, str]],
    ) -> torch.Tensor:
        """Compute sheaf Laplacian eigenvalues on a typed graph.

        Args:
            node_embeddings: (n, d_in) node feature vectors
            typed_edges: list of (src, dst, type_str) tuples

        Returns:
            eigenvalues: sorted ascending eigenvalues of L_F
        """
        n = node_embeddings.shape[0]
        s = self.stalk_dim
        device = node_embeddings.device

        if n < 2 or len(typed_edges) == 0:
            return torch.zeros(max(n * s, 1), device=device)

        # Project to stalk space
        X = self.proj(node_embeddings)  # (n, s)

        # Build typed coboundary matrix
        m = len(typed_edges)
        delta = torch.zeros(m * s, n * s, device=device, dtype=X.dtype)
        I_s = torch.eye(s, device=device, dtype=X.dtype)

        for k_edge, (src, dst, etype) in enumerate(typed_edges):
            if src >= n or dst >= n:
                continue

            d_vec = X[dst] - X[src]  # displacement (differentiable)

            # Get the typed restriction map
            R = self._get_restriction(etype)(d_vec)  # (s, s)

            # Coboundary: δ[e, src] = I, δ[e, dst] = -R
            r0 = k_edge * s
            delta[r0:r0 + s, src * s:(src + 1) * s] = I_s
            delta[r0:r0 + s, dst * s:(dst + 1) * s] = -R

        # L_F = δᵀδ
        L = delta.T @ delta
        L = (L + L.T) * 0.5  # symmetrize

        eigenvalues = torch.linalg.eigh(L).eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=0.0)

        return eigenvalues


class TypedSheafTopologyLoss(nn.Module):
    """Contrastive loss using the typed dependency graph sheaf.

    Same relative triplet margin as v6, but operates on
    dependency-parsed graphs instead of k-NN point clouds.
    """

    def __init__(self, d_in: int, stalk_dim: int = 8, margin: float = 0.5,
                 tau: float = 0.01, eps_gap: float = 1e-4):
        super().__init__()
        self.sheaf = DependencyGraphSheafLaplacian(d_in, stalk_dim=stalk_dim)
        self.margin = margin
        self.tau = tau
        self.eps_gap = eps_gap

    def _spectral_gap(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Differentiable soft extraction of λ₁."""
        gates = torch.sigmoid((eigenvalues - self.eps_gap) / self.tau)
        return (eigenvalues * gates).sum() / (gates.sum() + 1e-12)

    def forward(
        self,
        pos_embeddings: torch.Tensor,
        pos_edges: list[tuple[int, int, str]],
        neg_embeddings: torch.Tensor,
        neg_edges: list[tuple[int, int, str]],
    ) -> tuple[torch.Tensor, dict]:
        """Contrastive loss: truth graph vs corrupted graph.

        Args:
            pos_embeddings: (n_pos, d) node embeddings for truth graph
            pos_edges: typed edges for truth graph
            neg_embeddings: (n_neg, d) node embeddings for corrupted graph
            neg_edges: typed edges for corrupted graph

        Returns:
            loss, info_dict
        """
        eigs_pos = self.sheaf(pos_embeddings, pos_edges)
        eigs_neg = self.sheaf(neg_embeddings, neg_edges)

        gap_pos = self._spectral_gap(eigs_pos)
        gap_neg = self._spectral_gap(eigs_neg)

        # Relative triplet: force gap_neg > gap_pos + margin
        triplet = F.relu(gap_pos - gap_neg + self.margin)
        direct = 0.1 * gap_pos
        loss = triplet + direct

        info = {
            "gap_pos": gap_pos.detach().item(),
            "gap_neg": gap_neg.detach().item(),
            "gap_delta": (gap_neg - gap_pos).detach().item(),
            "loss": loss.detach().item(),
        }
        return loss, info


# ══════════════════════════════════════════════════════════════════════════════
# 3. Proof of Concept
# ══════════════════════════════════════════════════════════════════════════════

def run_poc(n_chunks: int = 100, n_train_steps: int = 100):
    """Proof of concept: dependency-parsed sheaf vs category errors.

    1. Parse 100 NQ chunks with spaCy → typed graphs
    2. Create corrupted versions (swap nouns/verbs → category errors)
    3. Train the typed sheaf to separate truth from corruption
    4. Measure: does λ₁ separation emerge?
    """
    import time
    import json
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nv9 Proof of Concept — Dependency Graph Sheaf")
    print(f"  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    base_dir = Path(__file__).parent / "results" / "v5_nq_base_map"
    with open(base_dir / "chunk_texts.json") as f:
        all_chunks = json.load(f)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_chunks), n_chunks, replace=False)
    chunks = [all_chunks[i] for i in sample_idx]
    print(f"  {len(chunks)} chunks sampled")

    # ── Parse with spaCy ──────────────────────────────────────────────
    print("  Parsing with spaCy...")
    grapher = TextCausalGrapher("en_core_web_sm")
    graphs = [grapher.parse(c) for c in chunks]

    edge_types = set()
    for g in graphs:
        for _, _, t in g.edges:
            edge_types.add(t)
    print(f"  {len(edge_types)} unique edge types: {sorted(edge_types)[:10]}...")

    avg_nodes = np.mean([len(g.node_texts) for g in graphs])
    avg_edges = np.mean([len(g.edges) for g in graphs])
    print(f"  Avg {avg_nodes:.0f} nodes, {avg_edges:.0f} edges per graph")

    # ── Encode nodes ──────────────────────────────────────────────────
    print("  Encoding nodes with sentence-transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_graph(graph: CausalGraph) -> torch.Tensor:
        """Encode each node text into 384-dim."""
        if not graph.node_texts:
            return torch.zeros(1, 384, device=device)
        embs = encoder.encode(graph.node_texts, convert_to_numpy=True,
                              show_progress_bar=False)
        return torch.tensor(embs, dtype=torch.float32, device=device)

    # ── Create corrupted graphs (category errors) ─────────────────────
    print("  Creating category-error corruptions...")

    def corrupt_graph(graph: CausalGraph) -> CausalGraph:
        """Swap nouns and verbs to create category errors."""
        node_texts = list(graph.node_texts)
        n = len(node_texts)
        if n < 4:
            return graph

        # Randomly swap 30% of nodes
        n_swaps = max(1, n // 3)
        for _ in range(n_swaps):
            i, j = rng.integers(0, n, size=2)
            node_texts[i], node_texts[j] = node_texts[j], node_texts[i]

        # Keep edges but they now connect wrong nodes → category errors
        return CausalGraph(node_texts, graph.edges)

    corrupted = [corrupt_graph(g) for g in graphs]

    # ── Build model ───────────────────────────────────────────────────
    print(f"\n  Building typed sheaf model...")
    loss_fn = TypedSheafTopologyLoss(
        d_in=384, stalk_dim=8, margin=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"  Initial params: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n  Training {n_train_steps} steps...")
    print(f"  {'Step':>6s}  {'Loss':>8s}  {'GapPos':>8s}  {'GapNeg':>8s}  "
          f"{'Δ':>8s}  {'EdgeTypes':>10s}")
    print("  " + "─" * 60)

    for step in range(n_train_steps):
        idx = rng.integers(len(graphs))

        pos_graph = graphs[idx]
        neg_graph = corrupted[idx]

        pos_emb = encode_graph(pos_graph)
        neg_emb = encode_graph(neg_graph)

        optimizer.zero_grad()
        loss, info = loss_fn(
            pos_emb, pos_graph.edges,
            neg_emb, neg_graph.edges,
        )
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == n_train_steps - 1:
            n_types = len(loss_fn.sheaf.restriction_maps)
            print(f"  {step:>6d}  {info['loss']:>8.4f}  {info['gap_pos']:>8.4f}  "
                  f"{info['gap_neg']:>8.4f}  {info['gap_delta']:>+8.4f}  "
                  f"{n_types:>10d}")

    # ── Summary ───────────────────────────────────────────────────────
    final_params = sum(p.numel() for p in loss_fn.parameters())
    n_types = len(loss_fn.sheaf.restriction_maps)

    print(f"\n  {'='*60}")
    print(f"  v9 POC COMPLETE")
    print(f"  {'='*60}")
    print(f"  Edge types learned: {n_types}")
    print(f"  Final params: {final_params:,} (grew from {total_params:,})")
    print(f"  Restriction maps: {list(loss_fn.sheaf.restriction_maps.keys())[:15]}")
    print(f"  {'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=100)
    args = parser.parse_args()
    run_poc(args.n_chunks, args.n_steps)
