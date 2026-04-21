"""Density field -> gravitational potential -> tidal tensor -> T-web classification.

Pipeline:
    rho      = cic_deposit(halos, grid=N)           # (N, N, N)    float64
    rho_hat  = np.fft.fftn(rho)
    k_vec    = np.fft.fftfreq(N) * N               # cell-unit wavenumbers
    k2       = k_x^2 + k_y^2 + k_z^2; k2[0,0,0] set to 1 to avoid div/0
    phi_hat  = -rho_hat / k2; phi_hat[0,0,0] = 0   # zero the DC mode
    T_ij_hat = +k_i * k_j * phi_hat                 # 6 unique components
    T_ij     = np.fft.ifftn(T_ij_hat).real          # (N, N, N) each
    eigvals  = np.linalg.eigvalsh(T_full)           # (N, N, N, 3)
    env_grid = 3 - np.sum(eigvals > lambda_th, axis=-1).astype(np.uint8)

Sign-convention note:
    The strict physics derivation gives T_ij_hat = -k_i k_j * phi_hat, which
    with phi_hat = -rho_hat/k^2 makes T eigvals positive at density peaks
    (nodes) and negative at voids. Under that convention "3 positive => NODE".
    We flip the sign (T_ij_hat = +k_i k_j * phi_hat) so that "3 positive =>
    VOID" holds instead, matching CODE_TO_ENV = (VOID, WALL, FILAMENT, NODE).
    This is a project-internal convention — we care about the eigenvalue
    sign-count for classification, not the trace relation Tr(T) = rho, so the
    flip is cost-free and keeps the mapping code == 0 <=> VOID intuitive.

The absolute normalization of rho and phi is irrelevant for T-web
classification because the count-of-positive-eigenvalues is scale-invariant.
This is why we can ignore 4*pi*G and just solve -k^2 phi_hat = rho_hat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problems.hubble_tension_web.types import Environment


# Lookup table: uint8 cell code -> Environment enum instance.
# Code 0 = all three eigenvalues positive = VOID (per spec T-web mapping).
# NEVER call Environment(code) — the enum is string-valued.
CODE_TO_ENV: tuple[Environment, ...] = (
    Environment.VOID,
    Environment.WALL,
    Environment.FILAMENT,
    Environment.NODE,
)


@dataclass
class ClassifyMeta:
    """Diagnostic fields returned alongside env_grid."""
    n_grid: int
    box_mpc: float
    cell_mpc: float
    lambda_th: float
    rho_mean: float
    rho_max: float


def cic_deposit(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    n_grid: int,
    box_mpc: float,
) -> np.ndarray:
    """Cloud-in-Cell mass assignment to a cubic grid.

    Each halo contributes mass to the 8 cells surrounding its position
    with weights equal to the overlap between a unit-cube kernel and each
    grid cell.

    Periodic boundary wrap: halos near the box edge wrap to the far side,
    matching the periodic cosmological volume.
    """
    cell = box_mpc / n_grid
    # Cell-center anchor: cell i is centered at (i + 0.5) * cell. Shift by -0.5
    # so a particle at a cell center yields frac = 0 (all mass in one cell).
    scaled = positions / cell - 0.5  # shape (N, 3), in cell-index units
    i0 = np.floor(scaled).astype(np.int64)
    frac = scaled - i0  # (N, 3)
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    for dx in (0, 1):
        wx = np.where(dx == 0, 1.0 - frac[:, 0], frac[:, 0])
        ix = (i0[:, 0] + dx) % n_grid
        for dy in (0, 1):
            wy = np.where(dy == 0, 1.0 - frac[:, 1], frac[:, 1])
            iy = (i0[:, 1] + dy) % n_grid
            for dz in (0, 1):
                wz = np.where(dz == 0, 1.0 - frac[:, 2], frac[:, 2])
                iz = (i0[:, 2] + dz) % n_grid
                w = masses * wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho


def _tidal_tensor_fft(rho: np.ndarray) -> np.ndarray:
    """Return (N, N, N, 3, 3) tidal tensor from a density grid.

    Sign convention aligned with CODE_TO_ENV[0] = VOID at "3 positive eigvals":
    we build T_ij = -partial_i partial_j phi (note the leading minus relative
    to the bare second derivative). With phi_hat = -rho_hat / k^2, the tensor
    in Fourier space is T_ij_hat = k_i k_j phi_hat. A density minimum (void)
    gives a local maximum of phi, so -Hessian(phi) is positive-definite there
    => three positive eigenvalues => code 0 = VOID.

    This matches the spec's CODE_TO_ENV ordering and the spec's fallback
    guidance ("the fix is to negate the overall sign in _tidal_tensor_fft").
    Any overall scale on rho factors through eigenvalues and does not change
    the sign-count classification.
    """
    n = rho.shape[0]
    rho_hat = np.fft.fftn(rho)

    k_vec = np.fft.fftfreq(n) * n  # (N,) in cell-index units
    kx = k_vec[:, None, None]
    ky = k_vec[None, :, None]
    kz = k_vec[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = np.where(k2 == 0.0, 1.0, k2)
    phi_hat = -rho_hat / k2_safe
    phi_hat[0, 0, 0] = 0.0 + 0.0j

    def _second_deriv(ka: np.ndarray, kb: np.ndarray) -> np.ndarray:
        # T_ij = -partial_i partial_j phi; in Fourier: T_ij_hat = k_i k_j phi_hat
        return np.fft.ifftn(ka * kb * phi_hat).real

    T_xx = _second_deriv(kx, kx)
    T_yy = _second_deriv(ky, ky)
    T_zz = _second_deriv(kz, kz)
    T_xy = _second_deriv(kx, ky)
    T_xz = _second_deriv(kx, kz)
    T_yz = _second_deriv(ky, kz)

    T = np.empty((n, n, n, 3, 3), dtype=np.float64)
    T[..., 0, 0] = T_xx
    T[..., 1, 1] = T_yy
    T[..., 2, 2] = T_zz
    T[..., 0, 1] = T_xy; T[..., 1, 0] = T_xy
    T[..., 0, 2] = T_xz; T[..., 2, 0] = T_xz
    T[..., 1, 2] = T_yz; T[..., 2, 1] = T_yz
    return T


def classify(
    *,
    positions: np.ndarray,
    masses: np.ndarray,
    n_grid: int,
    box_mpc: float,
    lambda_th: float = 0.0,
) -> tuple[np.ndarray, ClassifyMeta]:
    """Run the full CIC -> Poisson FFT -> tidal tensor -> T-web classification.

    Returns:
      env_grid: (n_grid, n_grid, n_grid) uint8 array of environment codes
                (0=VOID, 1=WALL, 2=FILAMENT, 3=NODE). Use CODE_TO_ENV[code]
                to get the Environment enum instance.
      meta:     ClassifyMeta with diagnostic fields.
    """
    rho = cic_deposit(positions, masses, n_grid=n_grid, box_mpc=box_mpc)
    T = _tidal_tensor_fft(rho)
    eigvals = np.linalg.eigvalsh(T)  # (n_grid, n_grid, n_grid, 3)
    n_positive = np.sum(eigvals > lambda_th, axis=-1)  # (n_grid, n_grid, n_grid)
    env_grid = (3 - n_positive).astype(np.uint8)

    meta = ClassifyMeta(
        n_grid=n_grid,
        box_mpc=box_mpc,
        cell_mpc=box_mpc / n_grid,
        lambda_th=lambda_th,
        rho_mean=float(rho.mean()),
        rho_max=float(rho.max()),
    )
    return env_grid, meta


def lookup_env_at_positions(
    env_grid: np.ndarray,
    positions: np.ndarray,
    *,
    box_mpc: float,
) -> list[Environment]:
    """Nearest-cell assignment: map each (x,y,z) to its grid cell and return Environment."""
    n_grid = env_grid.shape[0]
    cell = box_mpc / n_grid
    idx = np.floor(positions / cell).astype(np.int64)
    idx = np.clip(idx, 0, n_grid - 1)
    codes = env_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return [CODE_TO_ENV[int(c)] for c in codes]
