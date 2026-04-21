"""N-body halo-catalog ingestion for hubble_tension_web.

v1 scope:
- Parquet halo catalog (fixture or one-shot MDPL2 dump)
- Tidal-tensor T-web environment classification
- Simple local-minimum sphere-growing void finder
- LocalCosmicWeb assembly for the existing predict_from_cosmic_web path

The network fetch from CosmoSim MDPL2 is stubbed in v1 (see mdpl2_fetch.py);
tests and the nbody_kbc experiment operate against a local Parquet cache.
"""
from __future__ import annotations

# Default grid size for T-web classification. 128 keeps peak memory
# (3x3xN^3 float64 eigvalsh intermediate + complex FFT buffers) under
# ~500 MB on a 16 GB laptop. Override with ATFT_NBODY_GRID env var.
# Spec section "Risks & fallbacks" authorizes the 128 default.
DEFAULT_N_GRID: int = 128

# Default mass cut (M_sun). Spec targets 10^11.5 for galaxy-hosting halos.
DEFAULT_MASS_CUT: float = 10.0 ** 11.5

# Default tidal-tensor eigenvalue threshold. 0.0 per spec v1; tunable.
DEFAULT_LAMBDA_TH: float = 0.0


class NBodyDataNotAvailable(FileNotFoundError):
    """Raised when a halo cache is requested but neither local file nor network
    fetch can satisfy it. Subclasses FileNotFoundError so callers can
    `except FileNotFoundError` and skip gracefully (used by run_all.py)."""
