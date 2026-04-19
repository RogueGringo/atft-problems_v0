"""Typed-environment k-NN graph for local cosmic web.

Edge type = canonical string of the two endpoint environments, e.g. "void-wall".
Symmetric by canonicalizing endpoint order on the enum .value strings.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def edge_type_for_pair(a: Environment, b: Environment) -> str:
    lo, hi = sorted([a.value, b.value])
    return f"{lo}-{hi}"


EDGE_TYPES: List[str] = sorted({
    edge_type_for_pair(a, b)
    for a in Environment
    for b in Environment
})


def build_typed_graph(
    web: LocalCosmicWeb,
    k: int = 8,
) -> Tuple[int, List[Tuple[int, int, str]]]:
    n = web.positions.shape[0]
    tree = KDTree(web.positions)
    _, idx = tree.query(web.positions, k=k + 1)   # col 0 is self
    edges: List[Tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for src in range(n):
        for j in range(1, k + 1):
            dst = int(idx[src, j])
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                continue
            seen.add(pair)
            etype = edge_type_for_pair(web.environments[src], web.environments[dst])
            edges.append((src, dst, etype))
    return n, edges


def to_adjacency(n: int, edges: List[Tuple[int, int, str]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int8)
    for s, d, _ in edges:
        A[s, d] = 1
        A[d, s] = 1
    return A
