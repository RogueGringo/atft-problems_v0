"""Cross-domain structural comparison — is the crystal universal?"""
from __future__ import annotations
import numpy as np
from arm.void.formats import Crystal

def crystal_distance(c1: Crystal, c2: Crystal, metric: str = "l1") -> float:
    v1 = np.array([c1.void_ratio, c1.identity_ratio, c1.prime_ratio])
    v2 = np.array([c2.void_ratio, c2.identity_ratio, c2.prime_ratio])
    if metric == "l1":
        return float(np.sum(np.abs(v1 - v2)))
    elif metric == "l2":
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    raise ValueError(f"Unknown metric: {metric}")

def universality_test(crystals: list[Crystal], threshold: float = 0.05) -> dict:
    n = len(crystals)
    if n < 2:
        return {"universal": True, "max_distance": 0.0, "pairs": [], "distances": []}
    distances, pairs = [], []
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = crystal_distance(crystals[i], crystals[j])
            distances.append(d)
            pairs.append((crystals[i].source, crystals[j].source))
            max_d = max(max_d, d)
    return {"universal": max_d < threshold, "max_distance": max_d, "pairs": pairs, "distances": distances}

def barcode_distance(bars1: np.ndarray, bars2: np.ndarray) -> float:
    if len(bars1) == 0 and len(bars2) == 0:
        return 0.0
    def _finite_persistence(bars):
        if len(bars) == 0:
            return np.array([])
        p = bars[:, 1] - bars[:, 0]
        return np.sort(p[np.isfinite(p)])[::-1]
    p1 = _finite_persistence(bars1)
    p2 = _finite_persistence(bars2)
    total_cost = 0.0
    n_match = min(len(p1), len(p2))
    for i in range(n_match):
        total_cost += abs(p1[i] - p2[i])
    for i in range(n_match, len(p1)):
        total_cost += p1[i] / 2.0
    for i in range(n_match, len(p2)):
        total_cost += p2[i] / 2.0
    return float(total_cost)
