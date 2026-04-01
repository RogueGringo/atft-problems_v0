"""Structural constant extraction — the primes of any signal."""
from __future__ import annotations
import numpy as np
from arm.void.formats import PersistenceDiagram, Crystal

def gini(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    values = np.sort(np.abs(values)).astype(np.float64)
    n = len(values)
    total = values.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * total) / (n * total))

def onset_scale(diagram: PersistenceDiagram) -> float:
    deaths = diagram.h0[:, 1]
    finite_deaths = deaths[np.isfinite(deaths)]
    if len(finite_deaths) == 0:
        return float('inf')
    return float(np.min(finite_deaths))

def effective_rank(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    sv = np.linalg.svd(data.astype(np.float64), compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))

def spectral_gap(eigenvalues: np.ndarray) -> float:
    ev = np.sort(np.abs(eigenvalues))[::-1]
    if len(ev) < 2 or ev[1] == 0:
        return float('inf')
    return float(ev[0] / ev[1])

def crystal_from_persistence(diagram: PersistenceDiagram) -> Crystal:
    bars = diagram.h0
    if len(bars) == 0:
        return Crystal(void_ratio=0, identity_ratio=0, prime_ratio=0, eff_rank=0, source="empty")
    persistence = bars[:, 1] - bars[:, 0]
    finite_p = persistence[np.isfinite(persistence)]
    if len(finite_p) == 0:
        return Crystal(void_ratio=0, identity_ratio=0, prime_ratio=1.0, eff_rank=0, source="all_infinite")
    mean_p = np.mean(finite_p)
    if mean_p == 0:
        return Crystal(void_ratio=1.0, identity_ratio=0, prime_ratio=0, eff_rank=0, source="zero_persistence")
    low_thresh = mean_p / 3.0
    high_thresh = mean_p * 2.0
    void_count = np.sum(persistence < low_thresh)
    prime_count = np.sum((persistence > high_thresh) | ~np.isfinite(persistence))
    identity_count = len(persistence) - void_count - prime_count
    total = len(persistence)
    return Crystal(
        void_ratio=float(void_count / total),
        identity_ratio=float(identity_count / total),
        prime_ratio=float(prime_count / total),
        eff_rank=effective_rank(bars),
        source="topology"
    )
