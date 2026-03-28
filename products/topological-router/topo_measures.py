"""Shared topology measurement functions used across all topological-router modules."""

import numpy as np
import torch


def effective_rank(hs: torch.Tensor) -> float:
    """Effective rank via entropy of normalized singular values.

    Lower value = more hierarchical (energy concentrated in fewer directions).
    """
    x = hs.float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def spectral_gap(hs: torch.Tensor) -> float:
    """Ratio of first to second singular value.

    Higher value = more hierarchical (dominant direction stands out).
    """
    x = hs.float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    return float(s[0] / s[1])


def norm_variance(hs: torch.Tensor) -> float:
    """Variance of L2 norms across tokens."""
    x = hs.float().cpu()
    norms = torch.norm(x, dim=1)
    return float(torch.var(norms).item())


def gini_fast(values: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    s = np.sort(values.astype(np.float64).ravel())
    n = len(s)
    if n == 0 or s.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * s)) / (n * s.sum()) - (n + 1.0) / n)


def h0_persistence(points: np.ndarray, max_n: int = 200) -> np.ndarray:
    """H0 persistence bars via GPU pairwise distance + CPU union-find.

    Returns an array of bar lengths (birth=0 for all, death=merge distance).
    The longest bar (infinite component) is excluded.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
    max_n  : subsample cap to keep computation tractable
    """
    if len(points) > max_n:
        idx = np.random.choice(len(points), max_n, replace=False)
        points = points[idx]

    n = len(points)
    t = torch.tensor(points, dtype=torch.float32)
    if torch.cuda.is_available():
        t = t.cuda()

    # pairwise distances
    with torch.no_grad():
        dists = torch.cdist(t, t)  # (n, n)
        # upper triangle, excluding diagonal
        rows, cols = torch.triu_indices(n, n, offset=1, device=t.device)
        edge_dists = dists[rows, cols].cpu().numpy()
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()

    # sort edges by distance
    order = np.argsort(edge_dists)
    edge_dists = edge_dists[order]
    rows = rows[order]
    cols = cols[order]

    # union-find with path compression and union by rank
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    birth = np.zeros(n, dtype=np.float64)  # birth time of each component (always 0)
    bars = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dist, r, c in zip(edge_dists, rows, cols):
        pr, pc = find(int(r)), find(int(c))
        if pr == pc:
            continue
        # younger component (higher birth) merges into older
        if rank[pr] < rank[pc]:
            pr, pc = pc, pr
        bars.append(float(dist))  # bar length = death - birth = dist - 0
        parent[pc] = pr
        if rank[pr] == rank[pc]:
            rank[pr] += 1

    return np.array(bars, dtype=np.float64)


def h0_gini(points: np.ndarray, max_n: int = 200) -> float:
    """Gini coefficient of H0 persistence bar lengths."""
    bars = h0_persistence(points, max_n=max_n)
    if len(bars) == 0:
        return 0.0
    return gini_fast(bars)


MEASURES = {
    "eff_rank": effective_rank,
    "spectral_gap": spectral_gap,
    "norm_var": norm_variance,
}
