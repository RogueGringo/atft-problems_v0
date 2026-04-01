"""Internal representation boundaries — dataclasses, no logic, just shape."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass
class PointCloud:
    """A point cloud: N points in D dimensions."""
    data: np.ndarray
    source: str
    hash: str

    @classmethod
    def from_array(cls, data: np.ndarray, source: str) -> PointCloud:
        h = hashlib.sha256(data.tobytes()).hexdigest()
        return cls(data=data, source=source, hash=h)


@dataclass
class PersistenceDiagram:
    """Birth-death pairs from persistent homology."""
    h0: np.ndarray
    h1: np.ndarray
    filtration_range: tuple


@dataclass
class Crystal:
    """Structural ratios — the {0,1,3} decomposition of a signal."""
    void_ratio: float
    identity_ratio: float
    prime_ratio: float
    eff_rank: float
    source: str

    def validate(self):
        total = self.void_ratio + self.identity_ratio + self.prime_ratio
        assert abs(total - 1.0) < 1e-6, f"Crystal ratios sum to {total}, expected 1.0"


@dataclass
class ExperimentRecord:
    """One run of one experiment — the atomic unit of measurement."""
    id: str
    run: int
    series: str
    timestamp: str
    hypothesis: str
    protocol: str
    input_hash: str
    annotations: list
    result: dict
    comparison: dict
    verdict: str
    notes: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> ExperimentRecord:
        d = json.loads(s)
        return cls(**d)
