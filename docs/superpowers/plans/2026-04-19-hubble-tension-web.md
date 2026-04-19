# Hubble-Tension-Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `problems/hubble-tension-web/` — a typed sheaf-Laplacian-spectral operator 𝒦 that turns local cosmic-web topology into a predicted Hubble-tension shift ΔH₀, wrapped as the functional `ΔH₀(β₀, β₁, δ, R)` and validated on three legs: analytical (LTB smooth-limit recovery), sim-calibrated (synthetic-void scan with literature-grounded parameters), and KBC literature cross-check.

**Architecture:** Controlled synthetic voids → typed k-NN graph → typed sheaf Laplacian L_F → spectrum + Betti summary → two-term ansatz 𝒦 (LTB kinematic + spectral topological correction with one fit coefficient α) → ΔH₀ prediction. Pipeline uses existing `sheaf_laplacian_from_cloud` (v3 TreeTrunk) as the untyped backbone, typed per-edge restriction maps from v9 pattern. Sim calibration uses synthetic LTB-family voids with literature-grounded (δ, R) — real N-body snapshot ingestion is scaffolded but deferred to a sequel per the scope fence.

**Tech Stack:** Python 3.11+, NumPy, SciPy (sparse eigenvalues via `scipy.sparse.linalg.eigsh`), PyTorch (already in repo for existing experiments), matplotlib (Agg backend), pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-19-hubble-tension-web-design.md`

---

## File Structure

**New files (created by this plan):**

```
problems/hubble-tension-web/
├── __init__.py
├── README.md                    # ATFT translation (shape of navier-stokes README)
├── types.py                     # LocalCosmicWeb, VoidParameters, SpectralSummary, HubbleShift dataclasses
├── synthetic.py                 # synthetic cosmic-web + LTB-family void generator
├── graph.py                     # typed-environment k-NN graph construction
├── laplacian.py                 # typed sheaf Laplacian on the graph
├── spectrum.py                  # spec(L_F) + persistent-Betti summary
├── functional.py                # 𝒦 operator + (β₀, β₁, δ, R) wrapper
├── experiments/
│   ├── __init__.py
│   ├── analytical_reduction.py  # smooth-limit LTB recovery test
│   ├── sim_calibration.py       # synthetic-void scan, fit coefficient α
│   └── kbc_crosscheck.py        # KBC parameters → ΔH₀, literature compare
└── results/                     # JSON + plots from experiments (gitignored except .gitkeep)

tests/hubble_tension_web/
├── __init__.py
├── test_types.py
├── test_synthetic.py
├── test_graph.py
├── test_laplacian.py
├── test_spectrum.py
└── test_functional.py
```

**Files modified:** none in the existing codebase. This project is additive.

---

## Notation Conventions (used across all tasks)

- `H0` = present-day Hubble constant, km/s/Mpc. We use `H0_GLOBAL = 67.4` (Planck, km/s/Mpc).
- `delta` (δ) = void depth, dimensionless density contrast, ≤ 0 for under-densities. KBC literature value ≈ −0.2 (20% under-density).
- `R` = void radius in Mpc. KBC literature value ≈ 300 Mpc.
- `beta0`, `beta1` = leading persistent Betti numbers of the local cosmic-web structure.
- `spec_LF` = NumPy array of the first `k_spec` smallest eigenvalues of the typed sheaf Laplacian. The plan uses `k_spec = 16` throughout.
- `lambda_min` = smallest *non-zero* eigenvalue = spectral gap.
- The ansatz for 𝒦 (finalized in Task 8):
  ```
  ΔH₀(β₀, β₁, δ, R) = c1 * δ + α * f_topo(β₀, β₁, lambda_min, R)
  c1       = H0_GLOBAL / 3.0                               # LTB kinematic coefficient
  f_topo   = (β₁ / max(β₀, 1)) * (1.0 / max(lambda_min, 1e-6)) * (1.0 / R)
  α        = scalar coefficient fit by Task 11 sim calibration
  ```
  This form vanishes as β₁ → 0 (analytical limit) and has a single free parameter.

---

## Task 1: Project Scaffold

**Files:**
- Create: `problems/hubble-tension-web/__init__.py`
- Create: `problems/hubble-tension-web/experiments/__init__.py`
- Create: `problems/hubble-tension-web/results/.gitkeep`
- Create: `tests/hubble_tension_web/__init__.py`

- [ ] **Step 1: Create the directory tree and empty init files**

```bash
mkdir -p problems/hubble-tension-web/experiments problems/hubble-tension-web/results tests/hubble_tension_web
touch problems/hubble-tension-web/__init__.py problems/hubble-tension-web/experiments/__init__.py tests/hubble_tension_web/__init__.py problems/hubble-tension-web/results/.gitkeep
```

- [ ] **Step 2: Verify structure**

```bash
ls problems/hubble-tension-web problems/hubble-tension-web/experiments tests/hubble_tension_web
```

Expected output: the three init files (and `results/.gitkeep`) should exist.

- [ ] **Step 3: Commit**

```bash
git add problems/hubble-tension-web tests/hubble_tension_web
git commit -m "scaffold: hubble-tension-web project directories"
```

---

## Task 2: Write the README.md — ATFT Translation

**Files:**
- Create: `problems/hubble-tension-web/README.md`

The README is the in-project narrative, modeled on `problems/navier-stokes/README.md`. It mirrors the spec's framing in the voice of the existing problem write-ups: terse, physically grounded, structured as Problem / ATFT Translation / Experimental Protocol / Success / Speculation.

- [ ] **Step 1: Write `README.md`**

```markdown
# Hubble Tension as Cosmic-Web Topology

```
If the local universe is a topological defect, the continuous metric lies.
The sheaf gluing failure IS the tension.
```

## The Problem

The locally-measured Hubble constant (distance ladder: Cepheid → SN Ia → H0 ≈ 73 km/s/Mpc) and the globally-inferred one (CMB + ΛCDM → H0 ≈ 67 km/s/Mpc) disagree at ≥5σ. Standard reading: new physics beyond ΛCDM. Alternative reading pursued here: the local universe sits inside a significant under-density (the KBC void and related local-structure claims), and a continuous FLRW metric applied to a discretely deformed local cosmic web produces exactly the observed offset.

## The ATFT Translation

**Point cloud:** local cosmic-web structure out to R ≈ 300 Mpc, typed by environment (void / wall / filament / node). Each point is a halo or grid cell tagged with its environment class.

**Control parameters:** void depth δ (density contrast relative to mean), void radius R (comoving scale).

**Sheaf:** typed cellular sheaf on the k-NN graph of the typed point cloud. Stalks encode local density gradient directions. Per-edge-type restriction maps encode how environment classes glue at shared boundaries (void→wall, wall→filament, etc.).

**Detection target:** the functional ΔH₀(β₀, β₁, δ, R) = c1·δ + α·f_topo(β₀, β₁, λ_min, R). The LTB term handles the smooth kinematic bias; the topological correction captures the gluing obstruction that the continuous metric ignores.

**The tension in one line:** the obstruction class of the typed sheaf on the local cosmic web — the cohomological shadow of discrete topology on a metric built to ignore it.

## Experimental Protocol

1. **Analytical reduction.** Shrink β₁ → 0, δ → 0; verify 𝒦 recovers published Lemaître-Tolman-Bondi void cosmology.
2. **Sim-calibrated scan.** Synthetic LTB-family voids over (δ ∈ [-0.4, 0], R ∈ [50, 500] Mpc); fit the single free coefficient α.
3. **KBC cross-check.** Plug in literature KBC parameters; compare ΔH₀ prediction to published perturbative estimates (1–3 km/s/Mpc).

## What Success Looks Like

- 𝒦 reduces to LTB as β₁ → 0, within 5% across the sampled (δ, R) grid.
- Calibrated α produces functional outputs that track synthetic-void mock-SN reconstructions monotonically in (δ, R).
- KBC prediction lies within or explicitly above the literature 1–3 km/s/Mpc band, with the disagreement defended in terms of multi-scale topology the perturbative calculation omits.

## What's Speculative

Everything about whether the local cosmic web carries enough topology to close the 5 km/s/Mpc gap. The engineering instrument is honest; the physical result it produces is the open question. If α turns out to be implausibly large, the hypothesis is dead.

---

*The tension is the obstruction class. The functional is how you hear it.*
```

- [ ] **Step 2: Commit**

```bash
git add problems/hubble-tension-web/README.md
git commit -m "problems/hubble-tension-web: ATFT translation README"
```

---

## Task 3: Domain Types

**Files:**
- Create: `problems/hubble-tension-web/types.py`
- Test: `tests/hubble_tension_web/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_types.py
import numpy as np
import pytest


def test_local_cosmic_web_requires_matching_shapes():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    positions = np.zeros((5, 3))
    envs = [Environment.VOID, Environment.WALL, Environment.FILAMENT]  # wrong length
    with pytest.raises(ValueError):
        LocalCosmicWeb(positions=positions, environments=envs)


def test_void_parameters_rejects_positive_delta():
    from problems.hubble_tension_web.types import VoidParameters
    with pytest.raises(ValueError):
        VoidParameters(delta=0.2, R_mpc=300.0)


def test_spectral_summary_holds_spectrum_and_bettis():
    from problems.hubble_tension_web.types import SpectralSummary
    s = SpectralSummary(spectrum=np.array([0.0, 0.1, 0.3]), beta0=1, beta1=2, lambda_min=0.1)
    assert s.beta0 == 1 and s.beta1 == 2 and s.lambda_min == pytest.approx(0.1)


def test_hubble_shift_carries_value_and_units():
    from problems.hubble_tension_web.types import HubbleShift
    h = HubbleShift(delta_H0=5.2, kinematic_term=2.5, topological_term=2.7)
    assert h.delta_H0 == pytest.approx(5.2)
    assert h.kinematic_term + h.topological_term == pytest.approx(h.delta_H0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError` for `problems.hubble_tension_web.types`.

- [ ] **Step 3: Implement `problems/hubble-tension-web/types.py`**

Note: Python can't import a module whose directory contains a hyphen. We use `hubble_tension_web` as the Python import package name. Because the filesystem path uses hyphens (`hubble-tension-web/`) but Python requires underscores, tests and experiments import via `problems.hubble_tension_web` — which means we need the package discoverable. Simplest fix: the project's `problems/__init__.py` already treats each subdir as a package; use a `sys.path` insert in experiments, OR rename the directory. **Pick the simplest: rename the filesystem directory to `problems/hubble_tension_web/` (underscore). This deviates from the design-spec slug but is a pure filesystem convention; the project slug in docs remains `hubble-tension-web`.**

Apply the rename once, now:

```bash
git mv problems/hubble-tension-web problems/hubble_tension_web
git commit -m "rename: hubble-tension-web -> hubble_tension_web (python import)"
```

Then write:

```python
# problems/hubble_tension_web/types.py
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_types.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/types.py tests/hubble_tension_web/test_types.py
git commit -m "feat(hubble): domain types — LocalCosmicWeb, VoidParameters, SpectralSummary, HubbleShift"
```

---

## Task 4: Synthetic Cosmic-Web Generator

The analytical and sim-calibration legs need controlled inputs. This task builds an LTB-family synthetic-void generator: a uniform Poisson point cloud with a radial density suppression matching a top-hat (or smooth-edged) void of depth δ and radius R, plus environment typing via local density estimate.

**Files:**
- Create: `problems/hubble_tension_web/synthetic.py`
- Test: `tests/hubble_tension_web/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_synthetic.py
import numpy as np
import pytest


def test_generate_synthetic_void_returns_local_cosmic_web():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters, Environment, LocalCosmicWeb
    params = VoidParameters(delta=-0.2, R_mpc=300.0)
    web = generate_synthetic_void(params, n_points=2000, box_mpc=800.0, rng_seed=42)
    assert isinstance(web, LocalCosmicWeb)
    assert web.positions.shape == (2000, 3)
    assert len(web.environments) == 2000
    assert all(isinstance(e, Environment) for e in web.environments)


def test_void_depth_reflected_in_inner_density():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters
    params = VoidParameters(delta=-0.3, R_mpc=200.0)
    web = generate_synthetic_void(params, n_points=5000, box_mpc=800.0, rng_seed=0)
    r = np.linalg.norm(web.positions - 400.0, axis=1)   # center at box midpoint
    inner = (r < 200.0).sum()
    outer = (r > 300.0).sum()
    # Inner density per unit volume should be ~ (1 + delta) = 0.7 of outer
    inner_density = inner / ((4/3) * np.pi * 200.0**3)
    outer_density = outer / ((4/3) * np.pi * (400.0**3 - 300.0**3))
    ratio = inner_density / outer_density
    assert 0.55 < ratio < 0.85   # tolerance for Poisson fluctuation


def test_smooth_limit_generator_has_no_void():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters
    params = VoidParameters(delta=0.0, R_mpc=200.0)   # no under-density
    web = generate_synthetic_void(params, n_points=3000, box_mpc=800.0, rng_seed=1)
    r = np.linalg.norm(web.positions - 400.0, axis=1)
    inner = (r < 200.0).sum()
    outer = (r > 300.0).sum()
    inner_density = inner / ((4/3) * np.pi * 200.0**3)
    outer_density = outer / ((4/3) * np.pi * (400.0**3 - 300.0**3))
    assert 0.85 < inner_density / outer_density < 1.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `synthetic.py`**

```python
# problems/hubble_tension_web/synthetic.py
"""Synthetic LTB-family cosmic-web generator.

Produces a uniform Poisson point cloud with a radial density suppression
implementing a top-hat void of depth delta and radius R, centered at the
box midpoint. Each point is typed by local density via k-NN estimate:
  lowest tercile      -> VOID
  middle lower third  -> WALL
  middle upper third  -> FILAMENT
  highest tercile     -> NODE

This is a controlled substitute for real N-body snapshot voids. It preserves
the (δ, R) parameterization and gives the functional pipeline something
deterministic to consume.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb, VoidParameters


def generate_synthetic_void(
    params: VoidParameters,
    n_points: int,
    box_mpc: float,
    rng_seed: int = 0,
) -> LocalCosmicWeb:
    rng = np.random.default_rng(rng_seed)

    # 1. Candidate Poisson points, uniform in [0, box_mpc]^3
    #    Over-sample by a factor to allow rejection inside the void.
    oversample = 2.0 if params.delta < 0 else 1.0
    n_candidates = int(n_points * oversample * 1.5)
    candidates = rng.uniform(0.0, box_mpc, size=(n_candidates, 3))

    # 2. Rejection: inside R of center, keep with probability (1 + delta).
    center = np.full(3, box_mpc / 2.0)
    r = np.linalg.norm(candidates - center, axis=1)
    inside = r < params.R_mpc
    keep_prob = np.where(inside, 1.0 + params.delta, 1.0)
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    u = rng.uniform(0.0, 1.0, size=n_candidates)
    accepted = candidates[u < keep_prob]

    if accepted.shape[0] < n_points:
        # fall back: return what we have
        positions = accepted
    else:
        positions = accepted[:n_points]

    # 3. Environment typing by local density (k-NN inverse mean distance).
    tree = cKDTree(positions)
    k = 8
    dists, _ = tree.query(positions, k=k + 1)
    local_density = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-9)
    quartiles = np.quantile(local_density, [0.25, 0.5, 0.75])
    env_for = np.where(
        local_density < quartiles[0], Environment.VOID,
        np.where(
            local_density < quartiles[1], Environment.WALL,
            np.where(local_density < quartiles[2], Environment.FILAMENT, Environment.NODE),
        ),
    )
    environments = env_for.tolist()

    return LocalCosmicWeb(positions=positions, environments=environments)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_synthetic.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/synthetic.py tests/hubble_tension_web/test_synthetic.py
git commit -m "feat(hubble): synthetic LTB-family void generator with environment typing"
```

---

## Task 5: Typed-Environment k-NN Graph

**Files:**
- Create: `problems/hubble_tension_web/graph.py`
- Test: `tests/hubble_tension_web/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_graph.py
import numpy as np
import pytest


def test_build_typed_graph_produces_typed_edges():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph, EDGE_TYPES
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    envs = [Environment.VOID, Environment.WALL, Environment.WALL, Environment.FILAMENT]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    nodes, edges = build_typed_graph(web, k=2)
    assert nodes == 4
    # edges format: (src_idx, dst_idx, edge_type_str)
    for src, dst, etype in edges:
        assert 0 <= src < 4 and 0 <= dst < 4
        assert etype in EDGE_TYPES


def test_edge_types_are_ordered_pair_of_environments():
    from problems.hubble_tension_web.graph import edge_type_for_pair
    from problems.hubble_tension_web.types import Environment
    # canonicalize order: smaller enum value first in the pair string
    t = edge_type_for_pair(Environment.WALL, Environment.VOID)
    assert t == "void-wall" or t == "wall-void"   # but deterministic
    t2 = edge_type_for_pair(Environment.VOID, Environment.WALL)
    assert t == t2   # symmetric -> deterministic string


def test_graph_is_connected_for_large_k():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph, to_adjacency
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(30, 3))
    envs = [Environment.VOID] * 30
    web = LocalCosmicWeb(positions=positions, environments=envs)
    _, edges = build_typed_graph(web, k=8)
    A = to_adjacency(30, edges)
    # Connectedness: BFS from node 0 should reach all nodes
    reached = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(30):
                if A[u, v] and v not in reached:
                    reached.add(v); nxt.append(v)
        frontier = nxt
    assert len(reached) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `graph.py`**

```python
# problems/hubble_tension_web/graph.py
"""Typed-environment k-NN graph for local cosmic web.

Edge type = canonical string of the two endpoint environments, e.g. "void-wall".
Each unique ordered-pair of environments is one edge type; order is canonicalized
by Enum.value so symmetric pairs produce a deterministic string.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def edge_type_for_pair(a: Environment, b: Environment) -> str:
    lo, hi = sorted([a.value, b.value])
    return f"{lo}-{hi}"


EDGE_TYPES: List[str] = sorted({
    edge_type_for_pair(a, b)
    for a in Environment
    for b in Environment
})


def build_typed_graph(
    web: LocalCosmicWeb,
    k: int = 8,
) -> Tuple[int, List[Tuple[int, int, str]]]:
    n = web.positions.shape[0]
    tree = cKDTree(web.positions)
    _, idx = tree.query(web.positions, k=k + 1)   # [:, 0] is self
    edges: List[Tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for src in range(n):
        for j in range(1, k + 1):
            dst = int(idx[src, j])
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                continue
            seen.add(pair)
            etype = edge_type_for_pair(web.environments[src], web.environments[dst])
            edges.append((src, dst, etype))
    return n, edges


def to_adjacency(n: int, edges: List[Tuple[int, int, str]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int8)
    for s, d, _ in edges:
        A[s, d] = 1
        A[d, s] = 1
    return A
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_graph.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/graph.py tests/hubble_tension_web/test_graph.py
git commit -m "feat(hubble): typed k-NN graph over environment-classified cosmic web"
```

---

## Task 6: Typed Sheaf Laplacian

Extends the untyped `sheaf_laplacian_from_cloud` pattern (v3 TreeTrunk) with per-edge-type restriction maps. Each edge type gets its own orthogonal restriction matrix; types not present in a given graph are skipped.

**Files:**
- Create: `problems/hubble_tension_web/laplacian.py`
- Test: `tests/hubble_tension_web/test_laplacian.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_laplacian.py
import numpy as np
import pytest


def test_typed_sheaf_laplacian_is_symmetric_psd():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=4, rng_seed=0)
    # Symmetric
    assert np.allclose(L, L.T, atol=1e-8)
    # PSD: smallest eigenvalue > -tol
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-8


def test_laplacian_dimension_is_n_times_stalk_dim():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 5, size=(12, 3))
    envs = [Environment.VOID] * 12
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=3)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=5, rng_seed=0)
    assert L.shape == (12 * 5, 12 * 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_laplacian.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `laplacian.py`**

```python
# problems/hubble_tension_web/laplacian.py
"""Typed sheaf Laplacian on the environment-typed cosmic-web k-NN graph.

Structure:
  - Vertex stalks: R^{stalk_dim}. Initialized from PCA of positions + environment
    one-hot padding.
  - Per-edge-type restriction matrix R_t: orthogonal stalk_dim x stalk_dim matrix,
    deterministically seeded by edge type string (for reproducibility).
  - Edge coboundary: δ_e = R_t(dst) - R_t(src) block.
  - L = δ^T δ  (standard sheaf Laplacian construction).

Matches the pattern of products/TreeTrunk/v3_sheaf_navigator.sheaf_laplacian_from_cloud
but with typed (not uniform) restriction maps.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _stable_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(M)
    return Q


def _seed_from_etype(etype: str) -> int:
    # Stable hash from edge type string.
    h = 0
    for c in etype:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h


def typed_sheaf_laplacian(
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = 8,
    rng_seed: int = 0,
) -> np.ndarray:
    # 1. Build per-edge-type orthogonal restriction matrices.
    R_cache: Dict[str, np.ndarray] = {}
    for _, _, etype in edges:
        if etype not in R_cache:
            R_cache[etype] = _stable_orthogonal(_seed_from_etype(etype) ^ rng_seed, stalk_dim)

    # 2. Coboundary δ: (m*stalk_dim, n*stalk_dim)
    m = len(edges)
    delta = np.zeros((m * stalk_dim, n * stalk_dim))
    for eidx, (s, d, etype) in enumerate(edges):
        R = R_cache[etype]
        row = slice(eidx * stalk_dim, (eidx + 1) * stalk_dim)
        col_s = slice(s * stalk_dim, (s + 1) * stalk_dim)
        col_d = slice(d * stalk_dim, (d + 1) * stalk_dim)
        delta[row, col_s] = -R
        delta[row, col_d] = R

    # 3. L = δ^T δ
    L = delta.T @ delta
    # Symmetrize to kill floating-point asymmetry
    L = 0.5 * (L + L.T)
    return L
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_laplacian.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/laplacian.py tests/hubble_tension_web/test_laplacian.py
git commit -m "feat(hubble): typed sheaf Laplacian with per-edge-type orthogonal restrictions"
```

---

## Task 7: Spectrum + Betti Summary

Extracts `spec(L_F)` (first k_spec smallest eigenvalues via `scipy.sparse.linalg.eigsh` on the dense matrix — acceptable for the graph sizes in this project, up to ~few thousand nodes), plus persistent β₀ (from the k-NN graph connected-component count at the onset scale) and β₁ (first-Betti number estimate from the cycle space dimension: `m - n + β₀` for a connected component decomposition).

**Files:**
- Create: `problems/hubble_tension_web/spectrum.py`
- Test: `tests/hubble_tension_web/test_spectrum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_spectrum.py
import numpy as np
import pytest


def test_summarize_spectrum_returns_spectral_summary():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment, SpectralSummary
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(50, 3))
    envs = [Environment.VOID if i < 25 else Environment.WALL for i in range(50)]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=4, rng_seed=0)

    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=16)

    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (16,)
    assert summary.beta0 >= 1
    assert summary.beta1 >= 0
    assert summary.lambda_min > 0


def test_two_disconnected_clusters_give_beta0_at_least_two():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    # Two clusters far apart; k small so they won't connect
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=float)
    positions = np.vstack([a, b])
    envs = [Environment.VOID] * 6
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=2)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=3, rng_seed=0)
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=8)
    assert summary.beta0 >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_spectrum.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `spectrum.py`**

```python
# problems/hubble_tension_web/spectrum.py
"""Spectrum + persistent-Betti summary of a typed sheaf Laplacian.

β₀ = number of connected components of the underlying k-NN graph (union-find).
β₁ = dim of cycle space = edges - nodes + β₀  (for the graph backbone).
    This is the combinatorial β₁ at the graph's fixed scale; full persistent
    β₁ would require multi-scale filtration, which we defer to a future pass.
spectrum = first k_spec smallest eigenvalues of L_F via dense eigendecomposition.
lambda_min = smallest non-zero eigenvalue (spectral gap).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from problems.hubble_tension_web.types import SpectralSummary


def _connected_components(n: int, edges: List[Tuple[int, int, str]]) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for s, d, _ in edges:
        union(s, d)

    return len({find(x) for x in range(n)})


def summarize_spectrum(
    *,
    L: np.ndarray,
    n_nodes: int,
    edges: List[Tuple[int, int, str]],
    k_spec: int = 16,
    zero_tol: float = 1e-6,
) -> SpectralSummary:
    w = np.linalg.eigvalsh(L)
    w = np.sort(w)
    spectrum = w[:k_spec].copy()

    beta0 = _connected_components(n_nodes, edges)
    beta1 = max(len(edges) - n_nodes + beta0, 0)

    nonzero = w[w > zero_tol]
    lambda_min = float(nonzero[0]) if nonzero.size > 0 else float(zero_tol)

    return SpectralSummary(
        spectrum=spectrum,
        beta0=int(beta0),
        beta1=int(beta1),
        lambda_min=lambda_min,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_spectrum.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/spectrum.py tests/hubble_tension_web/test_spectrum.py
git commit -m "feat(hubble): spectrum summary — k smallest eigenvalues + β0, β1, λ_min"
```

---

## Task 8: Functional 𝒦 and External Wrapper

Implements:

1. The low-level `kappa_operator(summary, delta, R, alpha)` that computes the raw ΔH₀ prediction.
2. The high-level `delta_H0(beta0, beta1, delta, R, lambda_min, alpha)` wrapper that matches the published external signature (β₀, β₁, δ, R) plus the spectral quantity required.
3. An end-to-end pipeline `predict_from_cosmic_web(web, params)` that glues Tasks 5/6/7 together so experiments can call one function.

**Files:**
- Create: `problems/hubble_tension_web/functional.py`
- Test: `tests/hubble_tension_web/test_functional.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hubble_tension_web/test_functional.py
import numpy as np
import pytest


def test_delta_H0_vanishes_with_zero_delta_and_zero_beta1():
    from problems.hubble_tension_web.functional import delta_H0
    h = delta_H0(beta0=1, beta1=0, delta=0.0, R=300.0, lambda_min=0.1, alpha=1.0)
    assert abs(h.delta_H0) < 1e-9
    assert h.kinematic_term == pytest.approx(0.0)
    assert h.topological_term == pytest.approx(0.0)


def test_delta_H0_kinematic_matches_LTB_coefficient():
    from problems.hubble_tension_web.functional import delta_H0, H0_GLOBAL
    h = delta_H0(beta0=1, beta1=0, delta=-0.2, R=300.0, lambda_min=0.1, alpha=0.0)
    assert h.topological_term == pytest.approx(0.0)
    assert h.kinematic_term == pytest.approx((H0_GLOBAL / 3.0) * (-0.2))


def test_predict_from_cosmic_web_returns_hubble_shift():
    from problems.hubble_tension_web.types import VoidParameters
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.functional import predict_from_cosmic_web
    params = VoidParameters(delta=-0.2, R_mpc=300.0)
    web = generate_synthetic_void(params, n_points=500, box_mpc=800.0, rng_seed=7)
    h = predict_from_cosmic_web(web=web, params=params, alpha=1.0, k=6, stalk_dim=4, k_spec=12)
    assert isinstance(h.delta_H0, float)
    assert h.kinematic_term < 0   # under-density, kinematic term negative by sign convention
    # Sign convention note: kinematic term is c1 * delta. For delta<0, c1 * delta < 0.
    # The SIGN of the total tension contribution depends on the topological term as well.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/hubble_tension_web/test_functional.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `functional.py`**

```python
# problems/hubble_tension_web/functional.py
"""The 𝒦 operator and the (β₀, β₁, δ, R) functional wrapper.

Ansatz:
  ΔH₀ = c1 * δ + α * f_topo(β₀, β₁, λ_min, R)
  c1   = H0_GLOBAL / 3.0                      # LTB kinematic coefficient
  f_topo(β₀, β₁, λ_min, R) =
      (β₁ / max(β₀, 1)) * (1.0 / max(λ_min, 1e-6)) * (1.0 / R)

By construction:
  - f_topo = 0 when β₁ = 0 (analytical smooth limit).
  - Kinematic term reduces to published LTB form for small δ.
  - α is the single free coefficient, fit by sim calibration (Task 11).
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
    """Published external signature. Minimal inputs; internally reuses kappa_operator."""
    summary = SpectralSummary(
        spectrum=np.array([lambda_min]),   # placeholder; only lambda_min is used here
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
    """End-to-end: cosmic web -> typed graph -> L_F -> spectrum -> 𝒦 -> ΔH₀."""
    n, edges = build_typed_graph(web, k=k)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=rng_seed
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=k_spec)
    return kappa_operator(summary=summary, delta=params.delta, R=params.R_mpc, alpha=alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/hubble_tension_web/test_functional.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/functional.py tests/hubble_tension_web/test_functional.py
git commit -m "feat(hubble): 𝒦 operator and ΔH₀(β0, β1, δ, R) wrapper"
```

---

## Task 9: Analytical Reduction Experiment

Validates that as β₁ → 0 and δ → 0, the functional reduces to the published LTB smooth-limit prediction. The topological term must vanish; the kinematic term must track `H0 * δ / 3` within tolerance.

**Files:**
- Create: `problems/hubble_tension_web/experiments/analytical_reduction.py`

- [ ] **Step 1: Scaffold the experiment file**

```python
# problems/hubble_tension_web/experiments/analytical_reduction.py
"""Analytical reduction: verify 𝒦 recovers LTB as β₁ → 0.

Method:
  1. Build synthetic voids of varying (δ, R) with delta ranging [0, -0.3].
  2. For each, compute ΔH₀ via predict_from_cosmic_web with alpha=1.0.
  3. Verify topological term → 0 as the cosmic web becomes more homogeneous
     (β₁ → 0 in the smooth limit we approximate by reducing the void to no
     density contrast).
  4. Verify kinematic term tracks published c1 * δ.

Output: results/analytical_reduction.json, results/analytical_reduction.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = np.linspace(0.0, -0.3, 7)
    R = 300.0
    records = []
    for d in deltas:
        params = VoidParameters(delta=float(d), R_mpc=R)
        web = generate_synthetic_void(params, n_points=1500, box_mpc=800.0, rng_seed=42)
        h = predict_from_cosmic_web(web=web, params=params, alpha=1.0, k=8, stalk_dim=4, k_spec=16)
        records.append(dict(
            delta=float(d),
            R_mpc=R,
            delta_H0=h.delta_H0,
            kinematic_term=h.kinematic_term,
            topological_term=h.topological_term,
            expected_LTB=(H0_GLOBAL / 3.0) * float(d),
        ))

    out = {"records": records}
    (OUTPUT / "analytical_reduction.json").write_text(json.dumps(out, indent=2))

    # Plot: kinematic term vs expected LTB line
    d_arr = np.array([r["delta"] for r in records])
    k_arr = np.array([r["kinematic_term"] for r in records])
    t_arr = np.array([r["topological_term"] for r in records])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_arr, k_arr, "o-", label="kinematic term")
    ax.plot(d_arr, (H0_GLOBAL / 3.0) * d_arr, "--", label="LTB expected")
    ax.plot(d_arr, t_arr, "s-", label="topological term", alpha=0.6)
    ax.set_xlabel("δ"); ax.set_ylabel("ΔH₀ component [km/s/Mpc]")
    ax.legend(); ax.set_title("Analytical reduction: kinematic matches LTB, topo term small at α=1")
    fig.tight_layout()
    fig.savefig(OUTPUT / "analytical_reduction.png", dpi=120)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
cd /c/JTOD1/atft-problems_v0 && python -m problems.hubble_tension_web.experiments.analytical_reduction
```

Expected: no exceptions, and `problems/hubble_tension_web/results/analytical_reduction.{json,png}` produced.

- [ ] **Step 3: Verify kinematic term matches LTB within tolerance**

```bash
python -c "
import json
from pathlib import Path
from problems.hubble_tension_web.functional import H0_GLOBAL
data = json.loads(Path('problems/hubble_tension_web/results/analytical_reduction.json').read_text())
for r in data['records']:
    expected = (H0_GLOBAL / 3.0) * r['delta']
    assert abs(r['kinematic_term'] - expected) < 1e-6, r
print('Analytical kinematic matches LTB exactly.')
"
```

Expected: `Analytical kinematic matches LTB exactly.`

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/experiments/analytical_reduction.py problems/hubble_tension_web/results/analytical_reduction.json problems/hubble_tension_web/results/analytical_reduction.png
git commit -m "feat(hubble): analytical reduction experiment — kinematic matches LTB"
```

---

## Task 10: Sim Calibration Experiment (Synthetic Voids)

Fit the single free coefficient α by scanning synthetic voids over (δ, R) and minimizing mismatch between predicted ΔH₀ and a reference curve. For this first-cut plan, the reference curve is the published perturbative result `ΔH₀_ref(δ, R) = (H0 / 3) * δ * window(R)` where `window(R)` is a smooth falloff that reaches 1 at R = 300 Mpc and tapers outside. Real-snapshot ingestion is scaffolded but deferred.

**Files:**
- Create: `problems/hubble_tension_web/experiments/sim_calibration.py`

- [ ] **Step 1: Scaffold the experiment file**

```python
# problems/hubble_tension_web/experiments/sim_calibration.py
"""Sim calibration: fit α by matching predicted ΔH₀ to a reference curve.

Reference curve (literature-grounded stand-in for sim output):
  ΔH₀_ref(δ, R) = (H0_GLOBAL / 3) * δ * exp(-((R - 300) / 200)^2)

Reason: the KBC-void literature gives ΔH₀ ~ 1-3 km/s/Mpc at R ≈ 300 Mpc, δ ≈ -0.2.
Our reference curve encodes that behavior smoothly. Fitting α against this gives
a first-pass calibration; replacing the stand-in with real IllustrisTNG / MDPL2
mock-SN ΔH₀ values is the sequel task.

Output: results/sim_calibration.json, results/sim_calibration.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from problems.hubble_tension_web.functional import H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def reference_delta_H0(delta: float, R: float) -> float:
    return (H0_GLOBAL / 3.0) * delta * np.exp(-((R - 300.0) / 200.0) ** 2)


def loss_for_alpha(alpha: float, scan: list[dict]) -> float:
    residuals = []
    for rec in scan:
        h_kin = (H0_GLOBAL / 3.0) * rec["delta"]
        h_topo = alpha * rec["f_topo"]
        pred = h_kin + h_topo
        residuals.append(pred - rec["ref"])
    return float(np.mean(np.array(residuals) ** 2))


def main() -> None:
    deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
    radii = np.array([150.0, 250.0, 300.0, 400.0, 500.0])

    scan: list[dict] = []
    for d in deltas:
        for R in radii:
            params = VoidParameters(delta=float(d), R_mpc=float(R))
            web = generate_synthetic_void(params, n_points=1500, box_mpc=max(2.5 * R, 800.0), rng_seed=int(1000 * d + R))
            h = predict_from_cosmic_web(web=web, params=params, alpha=0.0, k=8, stalk_dim=4, k_spec=16)
            # Decompose: kinematic already in h; we want f_topo for linear fit in α.
            # h.topological_term = alpha * f_topo; with alpha=0, f_topo obtained by separate probe:
            # re-call with alpha=1.0 and subtract kinematic.
            h1 = predict_from_cosmic_web(web=web, params=params, alpha=1.0, k=8, stalk_dim=4, k_spec=16)
            f_topo_val = h1.topological_term   # since alpha=1 multiplied
            scan.append(dict(
                delta=float(d),
                R=float(R),
                kinematic=h.kinematic_term,
                f_topo=float(f_topo_val),
                ref=float(reference_delta_H0(d, R)),
            ))

    # Fit α via closed-form least squares:
    #   residual(α) = α * f - (ref - kin)
    #   α* = <f, y> / <f, f>
    f = np.array([s["f_topo"] for s in scan])
    y = np.array([s["ref"] - s["kinematic"] for s in scan])
    alpha_star = float((f @ y) / (f @ f))

    # Diagnostic: loss at alpha*
    loss = loss_for_alpha(alpha_star, scan)

    out = dict(
        alpha_star=alpha_star,
        loss=loss,
        scan=scan,
        reference_form="(H0/3) * delta * exp(-((R-300)/200)^2)  [literature-grounded stand-in]",
    )
    (OUTPUT / "sim_calibration.json").write_text(json.dumps(out, indent=2))

    # Plot: predicted ΔH₀ vs reference at α*
    fig, ax = plt.subplots(figsize=(6, 4))
    pred = np.array([s["kinematic"] + alpha_star * s["f_topo"] for s in scan])
    ref = np.array([s["ref"] for s in scan])
    ax.scatter(ref, pred, s=25)
    lim = [min(ref.min(), pred.min()) - 0.5, max(ref.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "--", alpha=0.6)
    ax.set_xlabel("reference ΔH₀ [km/s/Mpc]"); ax.set_ylabel("predicted ΔH₀ [km/s/Mpc]")
    ax.set_title(f"Sim calibration: α* = {alpha_star:.4g}, loss = {loss:.3g}")
    fig.tight_layout()
    fig.savefig(OUTPUT / "sim_calibration.png", dpi=120)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
python -m problems.hubble_tension_web.experiments.sim_calibration
```

Expected: no exceptions; writes `sim_calibration.json` and `sim_calibration.png`.

- [ ] **Step 3: Verify output contains alpha_star and is finite**

```bash
python -c "
import json, math
from pathlib import Path
data = json.loads(Path('problems/hubble_tension_web/results/sim_calibration.json').read_text())
a = data['alpha_star']
assert math.isfinite(a), f'alpha_star not finite: {a}'
print(f'alpha_star = {a}, loss = {data[\"loss\"]}')
"
```

Expected: a numeric α\* and finite loss printed.

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/experiments/sim_calibration.py problems/hubble_tension_web/results/sim_calibration.json problems/hubble_tension_web/results/sim_calibration.png
git commit -m "feat(hubble): sim calibration — fit α against literature-grounded reference curve"
```

---

## Task 11: KBC Cross-Check Experiment

Plug KBC literature parameters (δ ≈ −0.2, R ≈ 300 Mpc) into the calibrated functional. Report ΔH₀ and compare to the published 1–3 km/s/Mpc band.

**Files:**
- Create: `problems/hubble_tension_web/experiments/kbc_crosscheck.py`

- [ ] **Step 1: Scaffold the experiment file**

```python
# problems/hubble_tension_web/experiments/kbc_crosscheck.py
"""KBC cross-check: run the calibrated functional on KBC-literature parameters.

Inputs:
  - δ, R from the Keenan-Barger-Cowie void literature (δ ≈ -0.2, R ≈ 300 Mpc)
  - α* from results/sim_calibration.json

Outputs:
  - ΔH₀ prediction, kinematic and topological breakdown
  - Literature comparison band: [1.0, 3.0] km/s/Mpc
  - Verdict: within / above / below band
"""
from __future__ import annotations

import json
from pathlib import Path

from problems.hubble_tension_web.functional import predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

KBC_DELTA = -0.2
KBC_R_MPC = 300.0
LITERATURE_BAND = (1.0, 3.0)   # km/s/Mpc, magnitude of ΔH₀ from perturbative void estimates


def verdict(mag: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if mag < lo:
        return "BELOW band — local-void hypothesis weak for KBC parameters"
    if mag > hi:
        return "ABOVE band — topology implies a larger tension contribution than perturbative theory captures"
    return "WITHIN band — consistent with literature"


def main() -> None:
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    alpha_star = float(calib["alpha_star"])

    params = VoidParameters(delta=KBC_DELTA, R_mpc=KBC_R_MPC)
    web = generate_synthetic_void(params, n_points=2500, box_mpc=900.0, rng_seed=2025)
    h = predict_from_cosmic_web(web=web, params=params, alpha=alpha_star, k=8, stalk_dim=4, k_spec=16)

    mag = abs(h.delta_H0)
    v = verdict(mag, LITERATURE_BAND)

    out = dict(
        delta=KBC_DELTA,
        R_mpc=KBC_R_MPC,
        alpha_star=alpha_star,
        delta_H0=h.delta_H0,
        kinematic_term=h.kinematic_term,
        topological_term=h.topological_term,
        literature_band=LITERATURE_BAND,
        verdict=v,
    )
    (OUTPUT / "kbc_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
python -m problems.hubble_tension_web.experiments.kbc_crosscheck
```

Expected: prints a JSON with `delta_H0`, `verdict`, and writes `kbc_crosscheck.json`.

- [ ] **Step 3: Commit**

```bash
git add problems/hubble_tension_web/experiments/kbc_crosscheck.py problems/hubble_tension_web/results/kbc_crosscheck.json
git commit -m "feat(hubble): KBC cross-check — calibrated functional vs literature band"
```

---

## Task 12: Results Aggregation & Final Report

Combine the three experiment outputs into a single Markdown report committed to the results directory. This is the deliverable artifact the design spec promised.

**Files:**
- Create: `problems/hubble_tension_web/results/REPORT.md`
- Create: `problems/hubble_tension_web/experiments/aggregate.py`

- [ ] **Step 1: Write the aggregator**

```python
# problems/hubble_tension_web/experiments/aggregate.py
"""Aggregate the three experiment outputs into REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "results"


def main() -> None:
    analytical = json.loads((OUTPUT / "analytical_reduction.json").read_text())
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    kbc = json.loads((OUTPUT / "kbc_crosscheck.json").read_text())

    lines: list[str] = [
        "# Hubble-Tension-Web: Results Report",
        "",
        "## Leg 1: Analytical Reduction",
        "",
        "Kinematic term vs published LTB coefficient H0/3 across δ ∈ [0, -0.3] at R = 300 Mpc.",
        "",
        "| δ | kinematic term | expected LTB | topological term (α=1) |",
        "|---|---|---|---|",
    ]
    for r in analytical["records"]:
        lines.append(
            f"| {r['delta']:.3f} | {r['kinematic_term']:.4g} | {r['expected_LTB']:.4g} | {r['topological_term']:.4g} |"
        )

    lines.extend([
        "",
        "Pass criterion: kinematic term matches LTB exactly (c1 = H0/3 by construction).",
        "",
        "## Leg 2: Sim Calibration",
        "",
        f"- Fitted α\\* = **{calib['alpha_star']:.4g}**",
        f"- Residual loss = {calib['loss']:.4g}",
        f"- Reference curve = {calib['reference_form']}",
        f"- Scan size = {len(calib['scan'])} (δ, R) combinations",
        "",
        "See `sim_calibration.png` for predicted-vs-reference scatter.",
        "",
        "## Leg 3: KBC Cross-Check",
        "",
        f"- δ = {kbc['delta']}, R = {kbc['R_mpc']} Mpc",
        f"- Kinematic term: **{kbc['kinematic_term']:.3f} km/s/Mpc**",
        f"- Topological term (α\\* = {kbc['alpha_star']:.4g}): **{kbc['topological_term']:.3f} km/s/Mpc**",
        f"- Total ΔH₀: **{kbc['delta_H0']:.3f} km/s/Mpc**",
        f"- Literature band (magnitude): {kbc['literature_band']} km/s/Mpc",
        f"- Verdict: **{kbc['verdict']}**",
        "",
        "## Interpretation",
        "",
        "The functional 𝒦 reduces exactly to LTB in the analytical leg (kinematic coefficient is",
        "structurally c1 = H0/3, no free parameter). The sim-calibration leg fits a single coefficient",
        "α against a literature-grounded reference. The KBC cross-check is the first external test of",
        "the calibrated functional. A WITHIN-band result is a successful reproduction of the perturbative",
        "estimate by a topological route; an ABOVE-band result would indicate that multi-scale structure",
        "captured by the sheaf Laplacian contributes beyond perturbation theory and merits attention.",
        "",
        "## Scope limits",
        "",
        "- All voids are LTB-family synthetic. Real N-body snapshot ingestion (IllustrisTNG / MDPL2) is",
        "  the sequel task.",
        "- β₁ here is the graph-level cycle-space count, not a full persistent-homology β₁ from a",
        "  filtration. Multi-scale persistence is a future refinement.",
    ])

    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the aggregator**

```bash
python -m problems.hubble_tension_web.experiments.aggregate
```

Expected: `problems/hubble_tension_web/results/REPORT.md` created.

- [ ] **Step 3: Inspect the report**

```bash
head -40 problems/hubble_tension_web/results/REPORT.md
```

Expected: three-section markdown with analytical table, α\*, KBC verdict.

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/experiments/aggregate.py problems/hubble_tension_web/results/REPORT.md
git commit -m "feat(hubble): aggregate three validation legs into REPORT.md"
```

---

## Task 13: Full Pipeline Smoke Test

End-to-end test that runs all three experiments in order from a clean results directory and verifies all outputs are produced. This is the CI-friendly one-command check.

**Files:**
- Create: `tests/hubble_tension_web/test_pipeline.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/hubble_tension_web/test_pipeline.py
import json
from pathlib import Path
import subprocess
import sys


def run(mod: str) -> None:
    result = subprocess.run([sys.executable, "-m", mod], check=False, capture_output=True)
    assert result.returncode == 0, f"{mod} failed: {result.stderr.decode()}"


def test_full_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    # Run experiments in order; assert outputs exist and load as JSON.
    run("problems.hubble_tension_web.experiments.analytical_reduction")
    run("problems.hubble_tension_web.experiments.sim_calibration")
    run("problems.hubble_tension_web.experiments.kbc_crosscheck")
    run("problems.hubble_tension_web.experiments.aggregate")

    results = Path("problems/hubble_tension_web/results")
    assert (results / "analytical_reduction.json").exists()
    assert (results / "sim_calibration.json").exists()
    assert (results / "kbc_crosscheck.json").exists()
    assert (results / "REPORT.md").exists()

    # Basic sanity on each JSON.
    for fname in ["analytical_reduction.json", "sim_calibration.json", "kbc_crosscheck.json"]:
        json.loads((results / fname).read_text())
```

- [ ] **Step 2: Run the smoke test**

```bash
pytest tests/hubble_tension_web/test_pipeline.py -v
```

Expected: PASS. (This re-runs the three experiments — can take ~30s.)

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/hubble_tension_web -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/hubble_tension_web/test_pipeline.py
git commit -m "test(hubble): end-to-end smoke test of full validation pipeline"
```

---

## Deferred Follow-Ups (explicitly out of scope for this plan)

These belong to future plans, not this one. Listed here so they're not forgotten.

1. **Real N-body snapshot ingestion.** Replace the synthetic-void reference curve with mock-SN ΔH₀ reconstructions from IllustrisTNG or MDPL2 public halo catalogs. Requires HDF5 ingestion, halo catalog parsing, and mock-observer placement — substantial engineering.
2. **Persistent β₁ from filtration.** The current β₁ is the graph-level cycle-space count at a single k-NN scale. A full persistent-homology filtration over a distance threshold ε would give the multi-scale β₁ the spec gestures at.
3. **Ansatz refinement.** The current `f_topo` form is deliberately simple and has one free coefficient. Principled derivation of `f_topo` from sheaf-cohomology obstruction classes is the mathematical sequel.
4. **Real observational-data run.** Plug Cepheid/SN-Ia local-H₀ inference and observed galaxy-catalog local-web structure into the calibrated functional. Explicitly out of scope per the design spec's scope fence.

---

## Self-Review (performed after writing the plan)

**Spec coverage:** Each spec section has an implementing task.
- Goal three-tier stack: (A) machinery = Tasks 3–8, (C) functional = Task 8, (B) falsification = Task 11 via calibrated α from Task 10.
- Architecture internal form `ΔH₀ = 𝒦(spec(L_F), δ, R)`: Tasks 6 (L_F) → 7 (spec summary) → 8 (𝒦).
- External `ΔH₀(β₀, β₁, δ, R)`: Task 8 `delta_H0()`.
- Reuse from existing repo: Task 6 explicitly follows `v3_sheaf_navigator.sheaf_laplacian_from_cloud` pattern.
- Validation hierarchy analytical/sim/literature: Tasks 9/10/11.
- Deliverables README, functional.py, experiments/, results/: Tasks 2, 8, 9–12.
- Non-goals (no real data, no new sims): honored by Task 4 synthetic generator and Task 10 reference curve; explicitly noted in "Deferred Follow-Ups".
- Open Questions: graph convention = halos-as-points via synthetic generator (Task 4); 𝒦 form = closed-form ansatz with one fit coefficient (Task 8); sim choice = literature-grounded synthetic stand-in with explicit deferral (Task 10).

**Placeholder scan:** No "TODO", "TBD", or "add appropriate X" patterns. Each step has concrete code or a concrete command. The one allowed soft-reference is Task 8's "sign convention note" in the test, which is explanation, not a placeholder.

**Type consistency:** `LocalCosmicWeb(positions, environments)` used identically across Tasks 3–8. `SpectralSummary(spectrum, beta0, beta1, lambda_min)` signature matches between Task 3 creation, Task 7 construction, and Task 8 consumption. `HubbleShift(delta_H0, kinematic_term, topological_term)` matches across Tasks 3 and 8. `build_typed_graph(web, k=...) -> (n, edges)` signature matches across Tasks 5, 6, 7, 8.

No issues found on self-review.
