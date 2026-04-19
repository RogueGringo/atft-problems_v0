from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class Environment(Enum):
    VOID = "void"
    WALL = "wall"
    FILAMENT = "filament"
    NODE = "node"


@dataclass
class LocalCosmicWeb:
    positions: np.ndarray                     # (N, 3), Mpc
    environments: Sequence[Environment]       # length N

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3); got {self.positions.shape}")
        if len(self.environments) != self.positions.shape[0]:
            raise ValueError(
                f"environments length {len(self.environments)} != positions count {self.positions.shape[0]}"
            )


@dataclass
class VoidParameters:
    delta: float      # density contrast, must be <= 0 for an under-density
    R_mpc: float      # void radius, Mpc

    def __post_init__(self) -> None:
        if self.delta > 0:
            raise ValueError(f"void delta must be <= 0 (under-density); got {self.delta}")
        if self.R_mpc <= 0:
            raise ValueError(f"R_mpc must be positive; got {self.R_mpc}")


@dataclass
class SpectralSummary:
    spectrum: np.ndarray       # first k_spec smallest eigenvalues of L_F
    beta0: int                 # persistent H0 count
    beta1: int                 # persistent H1 count
    lambda_min: float          # smallest non-zero eigenvalue


@dataclass
class HubbleShift:
    delta_H0: float               # total predicted ΔH₀, km/s/Mpc
    kinematic_term: float         # c1 * δ contribution, km/s/Mpc
    topological_term: float       # α * f_topo contribution, km/s/Mpc

    def __post_init__(self) -> None:
        total = self.kinematic_term + self.topological_term
        if not np.isclose(self.delta_H0, total, atol=1e-9):
            raise ValueError(
                f"delta_H0 {self.delta_H0} != kinematic {self.kinematic_term} + topological {self.topological_term}"
            )
