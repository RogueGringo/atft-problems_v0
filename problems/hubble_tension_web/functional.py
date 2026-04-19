"""The 𝒦 operator and the (β0, β1, δ, R) functional wrapper.

Ansatz (phase-1 rework with corrected sign):
  ΔH0 = c1 * delta + alpha * f_topo(beta0, beta1, lambda_min, R)
  c1  = -H0_GLOBAL / 3.0                 # phase-1: sign corrected from +H0/3
  f_topo(beta0, beta1, lambda_min, R) =
      (beta1 / max(beta0, 1)) * (1.0 / max(lambda_min, 1e-6)) * (1.0 / R)

Sign convention: a local void (delta < 0) has less mass and therefore less
gravitational retardation, so the locally-inferred expansion rate is biased
UPWARD relative to the global (CMB) inference — ΔH0 > 0 for delta < 0.
The kinematic coefficient c1 = -H0/3 enforces this at leading order.

By construction:
  - f_topo vanishes when beta1 = 0 (smooth limit — not yet enforced by
    filtration; the full Opus rework introduces persistent beta1).
  - Kinematic term reduces to LTB for small delta.
  - alpha remains the one free coefficient; in phase-1 it is still fit
    against a circular reference curve (fixed in the full rework).
"""
from __future__ import annotations

import numpy as np

from problems.hubble_tension_web.types import (
    HubbleShift,
    LocalCosmicWeb,
    SpectralSummary,
    VoidParameters,
)
from problems.hubble_tension_web.graph import build_typed_graph
from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
from problems.hubble_tension_web.spectrum import summarize_spectrum

H0_GLOBAL: float = 67.4   # km/s/Mpc (Planck 2018)

# Sign convention: c1 = -H0/3. For a void (delta<0), c1*delta > 0, matching
# the observed tension direction (local H0 exceeds global H0). See REWORK spec §5.2.
C1: float = -H0_GLOBAL / 3.0

# alpha has units of km/s. f_topo has units of 1/Mpc.
# Product alpha * f_topo has units of km/s/Mpc, matching ΔH₀.
ALPHA_UNITS: str = "km/s"


def f_topo(beta0: int, beta1: int, lambda_min: float, R: float) -> float:
    return (beta1 / max(beta0, 1)) * (1.0 / max(lambda_min, 1e-6)) * (1.0 / R)


def kappa_operator(
    *,
    summary: SpectralSummary,
    delta: float,
    R: float,
    alpha: float,
) -> HubbleShift:
    kin = C1 * delta
    topo = alpha * f_topo(summary.beta0, summary.beta1, summary.lambda_min, R)
    return HubbleShift(
        delta_H0=kin + topo,
        kinematic_term=kin,
        topological_term=topo,
        delta=delta,
    )


def delta_H0(
    *,
    beta0: int,
    beta1: int,
    delta: float,
    R: float,
    lambda_min: float,
    alpha: float,
) -> HubbleShift:
    """Published external signature. Minimal inputs; reuses kappa_operator."""
    summary = SpectralSummary(
        spectrum=np.array([lambda_min]),   # placeholder; only lambda_min is used
        beta0=beta0,
        beta1=beta1,
        lambda_min=lambda_min,
    )
    return kappa_operator(summary=summary, delta=delta, R=R, alpha=alpha)


def predict_from_cosmic_web(
    *,
    web: LocalCosmicWeb,
    params: VoidParameters,
    alpha: float,
    k: int = 8,
    stalk_dim: int = 8,
    k_spec: int = 16,
    rng_seed: int = 0,
) -> HubbleShift:
    n, edges = build_typed_graph(web, k=k)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=rng_seed,
        environments=web.environments,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=k_spec)
    return kappa_operator(summary=summary, delta=params.delta, R=params.R_mpc, alpha=alpha)
