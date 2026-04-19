"""The 𝒦 operator and the (β0, β1, δ, R) functional wrapper.

Ansatz:
  ΔH0 = c1 * delta + alpha * f_topo(beta0, beta1, lambda_min, R)
  c1  = H0_GLOBAL / 3.0                  # LTB kinematic coefficient
  f_topo(beta0, beta1, lambda_min, R) =
      (beta1 / max(beta0, 1)) * (1.0 / max(lambda_min, 1e-6)) * (1.0 / R)

By construction:
  - f_topo vanishes when beta1 = 0 (smooth limit).
  - Kinematic term reduces to LTB for small delta.
  - alpha is the one free coefficient, fit by sim calibration.
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

H0_GLOBAL: float = 67.4   # km/s/Mpc (Planck)


def f_topo(beta0: int, beta1: int, lambda_min: float, R: float) -> float:
    return (beta1 / max(beta0, 1)) * (1.0 / max(lambda_min, 1e-6)) * (1.0 / R)


def kappa_operator(
    *,
    summary: SpectralSummary,
    delta: float,
    R: float,
    alpha: float,
) -> HubbleShift:
    c1 = H0_GLOBAL / 3.0
    kin = c1 * delta
    topo = alpha * f_topo(summary.beta0, summary.beta1, summary.lambda_min, R)
    return HubbleShift(delta_H0=kin + topo, kinematic_term=kin, topological_term=topo)


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
    stalk_dim: int = 4,
    k_spec: int = 16,
    rng_seed: int = 0,
) -> HubbleShift:
    n, edges = build_typed_graph(web, k=k)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=rng_seed
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=k_spec)
    return kappa_operator(summary=summary, delta=params.delta, R=params.R_mpc, alpha=alpha)
