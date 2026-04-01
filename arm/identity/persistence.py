"""The measuring tape — persistent homology via union-find (H₀)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from arm.void.formats import PointCloud, PersistenceDiagram

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def compute_h0(cloud: PointCloud, eps_max: float = 5.0, n_steps: int = 100) -> PersistenceDiagram:
    n = cloud.data.shape[0]
    if n == 0:
        return PersistenceDiagram(h0=np.empty((0, 2)), h1=np.empty((0, 2)), filtration_range=(0.0, eps_max))
    if n == 1:
        return PersistenceDiagram(h0=np.array([[0.0, float('inf')]]), h1=np.empty((0, 2)), filtration_range=(0.0, eps_max))

    dists = pdist(cloud.data.astype(np.float64))
    dist_matrix = squareform(dists)
    ii, jj = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[ii, jj]
    order = np.argsort(edge_dists)

    uf = UnionFind(n)
    death_times = {}

    for idx in order:
        d = edge_dists[idx]
        a, b = int(ii[idx]), int(jj[idx])
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            elder, younger = min(ra, rb), max(ra, rb)
            death_times[younger] = d
            uf.parent[younger] = elder

    bars = []
    for i in range(n):
        death = death_times.get(i, float('inf'))
        bars.append([0.0, death])
    bars.sort(key=lambda b: (b[1], b[0]))
    h0 = np.array(bars)

    return PersistenceDiagram(h0=h0, h1=np.empty((0, 2)), filtration_range=(0.0, eps_max))
