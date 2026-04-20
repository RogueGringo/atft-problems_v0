"""Typed-environment k-NN graph for local cosmic web.

Edge type = ordered pair (env_src, env_dst). Asymmetry is intentional: the restriction
map R_dst^t on an oriented edge depends on which environment is source vs destination,
so "void-wall" and "wall-void" are DIFFERENT edge types with different restriction maps.

The src->dst ordering of each stored edge is chosen deterministically by node index
(smaller first), NOT by environment. The type string reflects the environments in
that same src->dst index order, so a single undirected graph edge carries a single
oriented type consistent with the Laplacian's coboundary convention.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def oriented_edge_type_for_pair(src: Environment, dst: Environment) -> str:
    """Return 'void-wall' for src=VOID, dst=WALL. Order-sensitive by design."""
    return f"{src.value}-{dst.value}"


EDGE_TYPES: List[str] = sorted({
    oriented_edge_type_for_pair(a, b)
    for a in Environment
    for b in Environment
})


def build_typed_graph(
    web: LocalCosmicWeb,
    k: int = 8,
) -> Tuple[int, List[Tuple[int, int, str]]]:
    """Build undirected k-NN graph with canonically oriented edges (smaller idx first).

    Each undirected edge {u, v} is stored once as (min(u,v), max(u,v), oriented_type)
    where oriented_type uses environments in that same src->dst order.
    """
    n = web.positions.shape[0]
    tree = KDTree(web.positions)
    _, idx = tree.query(web.positions, k=k + 1)
    edges: List[Tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for src in range(n):
        for j in range(1, k + 1):
            dst = int(idx[src, j])
            if src == dst:
                continue
            s, d = (src, dst) if src < dst else (dst, src)
            pair = (s, d)
            if pair in seen:
                continue
            seen.add(pair)
            etype = oriented_edge_type_for_pair(
                web.environments[s], web.environments[d]
            )
            edges.append((s, d, etype))
    return n, edges


def to_adjacency(n: int, edges: List[Tuple[int, int, str]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int8)
    for s, d, _ in edges:
        A[s, d] = 1
        A[d, s] = 1
    return A
