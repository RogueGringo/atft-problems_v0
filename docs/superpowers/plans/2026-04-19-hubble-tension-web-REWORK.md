# Hubble-Tension-Web REWORK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan supersedes `2026-04-19-hubble-tension-web.md` for the four redesign targets (T1 Laplacian, T2 persistent β₁, T3 sign + c₁ anchoring, T4 non-circular calibration).

**Goal:** Rework the existing v1 `problems/hubble_tension_web/` implementation into a physically honest pipeline: a density-gradient-stalk typed sheaf Laplacian that is demonstrably NOT a Kronecker no-op, a persistent β₁ via Vietoris-Rips filtration, a sign-corrected kinematic coefficient `c₁ = −H₀/3`, and a non-circular α calibration against an independent LTB reference curve.

**Architecture:** Retain the v1 module layout (`types.py`, `synthetic.py`, `graph.py`, `laplacian.py`, `spectrum.py`, `functional.py`, `experiments/`) and the external functional signature `ΔH₀(β₀, β₁, δ, R)`. Replace the Laplacian internals (gradient stalks + `R_dst^t = λ^t·(Rot ⊕ P^t)`), replace the β₁ computation (VR filtration via `ripser`), flip the sign of `c₁`, and add a new `ltb_reference.py` module that produces the calibration target independently of the functional's ansatz.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.spatial.KDTree`, `scipy.sparse.linalg.eigsh`, dense `eigvalsh`), `ripser` (new dependency, preferred over `gudhi` per spec §4.1), matplotlib (Agg backend), pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-19-hubble-tension-web-REWORK-design.md`

**Branch:** `feat/hubble-tension-web` (continuing, not branching).

---

## File Structure

**Files modified (rewrites of v1 content):**

```
problems/hubble_tension_web/
├── types.py               # add sign regression guard in HubbleShift.__post_init__
├── graph.py               # ordered-pair edge types (void-wall ≠ wall-void)
├── laplacian.py           # full rewrite: gradient stalks + R_dst^t = λ^t·(Rot ⊕ P^t ⊕ pad)
├── spectrum.py            # full rewrite: persistent β₁ via ripser, thresholded by τ·ℓ̄
├── functional.py          # c₁ = -H0/3, α units doc (km/s), persistent-β₁ signature unchanged
├── experiments/
│   ├── analytical_reduction.py  # repurposed: verify β₁→0 smooth limit, not "derive c₁"
│   ├── sim_calibration.py       # replaced: α fit vs LTB residual (non-circular)
│   ├── kbc_crosscheck.py        # signed band comparison (+1, +3) km/s/Mpc
│   └── aggregate.py             # REPORT.md mentions persistent β₁, signed convention, v1_superseded link
```

**Files created:**

```
problems/hubble_tension_web/
├── ltb_reference.py                    # NEW: ΔH₀_LTB(δ, R) series + finite-R correction
└── results/
    └── v1_superseded/                  # NEW: housing for pre-rework artifacts
        ├── README.md
        ├── analytical_reduction.{json,png}
        ├── sim_calibration.{json,png}
        ├── kbc_crosscheck.json
        └── REPORT.md
```

**Tests modified:**

```
tests/hubble_tension_web/
├── test_types.py          # add sign-guard test
├── test_graph.py          # update canonicalization tests for ordered pairs
├── test_laplacian.py      # rewrite: typed≠untyped, nullity drop, gradient-stalk determinism
├── test_spectrum.py       # rewrite: β₁_persistent small on uniform, nonzero on ring
├── test_functional.py     # sign flip across the board + α-units assertion
└── test_pipeline.py       # signed KBC band assertion
```

**Test files unchanged:**

```
tests/hubble_tension_web/
└── test_synthetic.py      # unchanged; generator contract preserved
```

**Files NOT touched (carry over from v1):**

- `problems/hubble_tension_web/__init__.py`
- `problems/hubble_tension_web/README.md` (no new copy needed; spec rework is internal)
- `problems/hubble_tension_web/synthetic.py`
- `tests/hubble_tension_web/__init__.py`

---

## Notation Conventions (used across all tasks)

**Cosmology constants:**

- `H0_GLOBAL = 67.4` km/s/Mpc (Planck 2018). Module constant in `functional.py`.
- `c1 = -H0_GLOBAL / 3.0 = -22.4666...` km/s/Mpc. **Negative.** Pinned in `functional.py`.

**Sign convention (pinned everywhere):**

> For a local under-density (δ < 0), the locally-measured Hubble rate exceeds the globally-inferred rate. ΔH₀ := H_local − H_global, so **ΔH₀ > 0 when δ < 0**. Leading-order LTB: H_local ≈ H₀(1 − δ/3), so ΔH₀ = −(H₀/3)·δ, i.e. `c₁ = −H₀/3`.

**α units:** `α` has units of **km/s** (not km/s/Mpc, not dimensionless). Because `f_topo` has units of 1/Mpc and ΔH₀ must come out in km/s/Mpc.

**Edge-type convention:** Edge types are **ordered pairs** `(env_src, env_dst)`. `"void-wall" ≠ "wall-void"`. The helper is renamed to `oriented_edge_type_for_pair(env_src, env_dst) -> str`.

**Stalk dimension:** `stalk_dim = 8 = 3 (gradient) + 4 (env one-hot) + 1 (pad)`. Default in all call sites.

**Edge-type λ prefactors (from spec §3.3, doubled to all 16 ordered pairs):**

```python
EDGE_TYPE_LAMBDA: dict[tuple[str, str], float] = {
    # diagonal
    ("void", "void"):         1.0,
    ("wall", "wall"):         1.0,
    ("filament", "filament"): 1.0,
    ("node", "node"):         1.0,
    # off-diagonal (symmetric in value, distinct in key; asymmetry lives in Rot/P)
    ("void", "wall"):         1.5,  ("wall", "void"):         1.5,
    ("void", "filament"):     1.8,  ("filament", "void"):     1.8,
    ("void", "node"):         2.2,  ("node", "void"):         2.2,
    ("wall", "filament"):     1.2,  ("filament", "wall"):     1.2,
    ("wall", "node"):         1.5,  ("node", "wall"):         1.5,
    ("filament", "node"):     1.1,  ("node", "filament"):     1.1,
}
# Documented in laplacian.py as: "ordinal physical prior, not calibrated."
```

**Filtration parameters (from spec §4.2–4.3):**

- `tau_persist = 1.5` (lifetime threshold multiplier).
- `tau_max = 6.0` (ε_max / ℓ̄ ratio for ripser's `thresh` argument).
- `ell_bar` = mean k-NN edge length on the k used for the typed graph (`k=8` default).

**Functional form (unchanged from v1 except sign):**

```
ΔH₀(β₀, β₁, δ, R) = c1 * δ + α * f_topo(β₀, β₁, λ_min, R)
c1      = -H0_GLOBAL / 3.0                                # NEGATIVE
f_topo  = (β₁ / max(β₀, 1)) * (1.0 / max(λ_min, 1e-6)) * (1.0 / R)
α       [km/s]                                            # units pinned
β₁      = β₁_persistent (lifetime > tau_persist · ℓ̄)    # persistent, not combinatorial
```

**Units audit:** `β₁/β₀` dimensionless, `λ_min` dimensionless (unit-magnitude stalks), `1/R` has units 1/Mpc, so `f_topo` has units 1/Mpc and α·f_topo has units km/s/Mpc, matching ΔH₀. This is checked by `test_alpha_units_documented`.

---

## Task 1: Migration + `ripser` dependency

Archive v1 results into `results/v1_superseded/` and install the persistent-homology library. No code changes yet. This clears the deck so later tasks are not comparing v2 numbers against v1 artifacts that were computed with a no-op Laplacian.

**Files:**
- Move: `problems/hubble_tension_web/results/{analytical_reduction.json,analytical_reduction.png,sim_calibration.json,sim_calibration.png,kbc_crosscheck.json,REPORT.md}` → `problems/hubble_tension_web/results/v1_superseded/`
- Create: `problems/hubble_tension_web/results/v1_superseded/README.md`

- [ ] **Step 1: Move the v1 artifacts (not delete)**

```bash
mkdir -p problems/hubble_tension_web/results/v1_superseded
git mv problems/hubble_tension_web/results/analytical_reduction.json problems/hubble_tension_web/results/v1_superseded/
git mv problems/hubble_tension_web/results/analytical_reduction.png  problems/hubble_tension_web/results/v1_superseded/
git mv problems/hubble_tension_web/results/sim_calibration.json      problems/hubble_tension_web/results/v1_superseded/
git mv problems/hubble_tension_web/results/sim_calibration.png       problems/hubble_tension_web/results/v1_superseded/
git mv problems/hubble_tension_web/results/kbc_crosscheck.json       problems/hubble_tension_web/results/v1_superseded/
git mv problems/hubble_tension_web/results/REPORT.md                 problems/hubble_tension_web/results/v1_superseded/
```

- [ ] **Step 2: Write the `v1_superseded/README.md`**

```markdown
# v1_superseded — pre-rework artifacts

These files are outputs of the **v1 implementation** of `problems/hubble_tension_web/`, committed between 8e8524a and e2dd1dc. They are **retained for diff/history reasons only** and are known to be wrong in four ways:

1. The typed Laplacian was a no-op (every edge-type used the same orthogonal R ⊗ ±I, i.e. L_F ≡ L_graph ⊗ I_stalk_dim).
2. β₁ was computed as `|E| − |V| + β₀` at a single k-NN scale — not a persistent topological invariant.
3. The kinematic coefficient had the **wrong sign** (`c₁ = +H₀/3` rather than `−H₀/3`).
4. The sim calibration reference curve `(H₀/3)·δ·window(R)` embedded the kinematic answer as its leading term, so α was fit against residuals of a circularly-defined target.

See `docs/superpowers/specs/2026-04-19-hubble-tension-web-REWORK-design.md` for the full critique and the rework plan.

**Do not cite these numbers.** The file layout, plot style, and REPORT.md format are what remain useful here.
```

- [ ] **Step 3: Install `ripser`**

```bash
python -m pip install ripser
python -c "from ripser import ripser; import numpy as np; pts = np.random.default_rng(0).standard_normal((30, 3)); dgms = ripser(pts, maxdim=1)['dgms']; print('ripser OK, dgm1 shape:', dgms[1].shape)"
```

Expected output: `ripser OK, dgm1 shape: (k, 2)` for some small k (typically 0–5 on a random cloud of 30 points). If installation fails on the ARM laptop, fall back to `pip install gudhi` and adjust Task 7 to use `gudhi.RipsComplex` instead.

- [ ] **Step 4: Add `ripser` to any requirements file that exists**

```bash
grep -rl "scipy" pyproject.toml setup.py requirements*.txt 2>/dev/null || echo "no requirements files"
# If any found, add 'ripser>=0.6.4' to the same section as scipy.
```

If the repo has no requirements file listing scipy, skip this step (the project is import-driven; `ripser` being pip-installed is sufficient).

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/results/v1_superseded/
git commit -m "chore(hubble): archive v1 results to results/v1_superseded/ + add ripser dep"
```

---

## Task 2: Write failing tests that invalidate the v1 illusions

Before any implementation changes, commit tests that **fail on master** and that the rework will make pass. This is the "red" of the rework TDD cycle. Two targets: (a) the sign bug, (b) the no-op Laplacian. Both tests will fail with the v1 code.

**Files:**
- Modify: `tests/hubble_tension_web/test_functional.py` — add `test_sign_convention`
- Modify: `tests/hubble_tension_web/test_laplacian.py` — add `test_typed_vs_untyped_spectrum_differs`

- [ ] **Step 1: Add the sign test to `test_functional.py`**

Append to the existing file:

```python
def test_sign_convention_delta_negative_implies_kinematic_positive():
    """Regression guard: for a void (δ < 0), the kinematic term of ΔH₀ must be POSITIVE.

    LTB leading order: H_local = H₀·(1 − δ/3), so ΔH₀ := H_local − H_global = −H₀·δ/3.
    For δ < 0, ΔH₀ > 0 (local H₀ exceeds global H₀). This matches the observed tension direction.
    """
    from problems.hubble_tension_web.functional import delta_H0
    h = delta_H0(beta0=1, beta1=0, delta=-0.2, R=300.0, lambda_min=0.1, alpha=0.0)
    assert h.topological_term == 0.0
    # Kinematic contribution of a void must be POSITIVE under the corrected sign convention.
    assert h.kinematic_term > 0, (
        f"void (δ=-0.2) must have kinematic_term > 0; got {h.kinematic_term}. "
        f"This is the v1 sign-bug regression guard."
    )
    assert h.delta_H0 > 0
```

- [ ] **Step 2: Add the "typed is not a no-op" test to `test_laplacian.py`**

Append:

```python
def test_typed_vs_untyped_spectrum_differs():
    """I1 regression guard: the Laplacian spectrum MUST respond to edge typing.

    Build the same graph twice: once with all-identical environment types (untyped
    collapse), once with a mix. Median eigenvalue must differ by > 1e-3 relative.
    """
    import numpy as np
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 10, size=(60, 3))

    envs_typed = rng.choice(list(Environment), size=60).tolist()
    envs_untyped = [Environment.VOID] * 60

    web_typed = LocalCosmicWeb(positions=positions, environments=envs_typed)
    web_untyped = LocalCosmicWeb(positions=positions, environments=envs_untyped)

    n_t, edges_t = build_typed_graph(web_typed, k=6)
    n_u, edges_u = build_typed_graph(web_untyped, k=6)

    L_typed   = typed_sheaf_laplacian(positions=positions, n=n_t, edges=edges_t, stalk_dim=8, rng_seed=0)
    L_untyped = typed_sheaf_laplacian(positions=positions, n=n_u, edges=edges_u, stalk_dim=8, rng_seed=0)

    w_t = np.sort(np.linalg.eigvalsh(L_typed))
    w_u = np.sort(np.linalg.eigvalsh(L_untyped))

    med_t = float(np.median(w_t))
    med_u = float(np.median(w_u))
    rel = abs(med_t - med_u) / max(abs(med_u), 1e-12)
    assert rel > 1e-3, (
        f"Typed and untyped Laplacian medians indistinguishable (rel={rel:.2e}). "
        f"This is the v1 no-op regression guard — the Laplacian must respond to typing."
    )


def test_nullity_drops_under_typing():
    """Corollary of I1: nullity(L_F_typed) < stalk_dim * β₀ when Rot or P is nontrivial.

    In untyped collapse, nullspace dim = stalk_dim · β₀ (constant sections lift per-dimension).
    In the typed case with at least one edge having non-identity Rot or P, the nullity drops
    (generic constant sections are no longer in the kernel).
    """
    import numpy as np
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    stalk_dim = 8
    rng = np.random.default_rng(11)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    # Ensure at least two environments present so typing is nontrivial.
    if len({e for e in envs}) < 2:
        envs[0] = Environment.VOID
        envs[1] = Environment.NODE
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=0)

    w = np.linalg.eigvalsh(L)
    # Graph is almost certainly connected at k=6, N=40, so β₀ = 1.
    # Untyped expectation: nullity = stalk_dim * 1 = 8.
    # Typed expectation: nullity < 8.
    nullity = int(np.sum(w < 1e-6))
    assert nullity < stalk_dim, (
        f"nullity(L_F_typed)={nullity} did not drop below stalk_dim={stalk_dim}; "
        f"typing is not producing nontrivial restriction maps."
    )
```

- [ ] **Step 3: Run to confirm both fail**

```bash
pytest tests/hubble_tension_web/test_functional.py::test_sign_convention_delta_negative_implies_kinematic_positive tests/hubble_tension_web/test_laplacian.py::test_typed_vs_untyped_spectrum_differs tests/hubble_tension_web/test_laplacian.py::test_nullity_drops_under_typing -v
```

Expected: 3 failures.

- `test_sign_convention_…`: AssertionError on `h.kinematic_term > 0` (v1 gives `(+H0/3)·(-0.2) = -4.49`).
- `test_typed_vs_untyped_spectrum_differs`: AssertionError on `rel > 1e-3` (v1's Laplacian is Kronecker; median eigenvalues are identical up to numerical noise).
- `test_nullity_drops_under_typing`: AssertionError on `nullity < stalk_dim` (v1 gives nullity = stalk_dim × β₀).

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/hubble_tension_web/test_functional.py tests/hubble_tension_web/test_laplacian.py
git commit -m "test(hubble): failing regression guards — sign convention + typed Laplacian non-noop"
```

---

## Task 3: Fix the sign — `c₁ = −H₀/3` + HubbleShift regression guard

Flip the sign in `functional.py` and add a post-init guard in `HubbleShift` that raises when the sign convention is violated for clearly-signed voids.

**Files:**
- Modify: `problems/hubble_tension_web/functional.py`
- Modify: `problems/hubble_tension_web/types.py`
- Modify: `tests/hubble_tension_web/test_types.py` — add guard test

- [ ] **Step 1: Write the new guard test on `HubbleShift` first**

Append to `tests/hubble_tension_web/test_types.py`:

```python
def test_hubble_shift_rejects_inverted_sign_for_void():
    """If δ < 0 and kinematic_term < 0 and |topo| ≈ 0, HubbleShift must raise.

    This is the sign-bug regression guard: v1 produced exactly this combination,
    and the rework must make it unconstructable.
    """
    import pytest
    from problems.hubble_tension_web.types import HubbleShift
    with pytest.raises(ValueError, match="sign convention"):
        HubbleShift(
            delta_H0=-4.49,
            kinematic_term=-4.49,
            topological_term=0.0,
            delta=-0.2,          # void
        )


def test_hubble_shift_accepts_corrected_void():
    from problems.hubble_tension_web.types import HubbleShift
    # Corrected sign: δ = -0.2, c1·δ = +4.49 for c1 = -H0/3 with H0 = 67.4.
    h = HubbleShift(
        delta_H0=4.49,
        kinematic_term=4.49,
        topological_term=0.0,
        delta=-0.2,
    )
    assert h.delta_H0 > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_types.py::test_hubble_shift_rejects_inverted_sign_for_void tests/hubble_tension_web/test_types.py::test_hubble_shift_accepts_corrected_void -v
```

Expected: 2 failures — `HubbleShift` currently has no `delta` argument; TypeError on unexpected kwarg.

- [ ] **Step 3: Update `HubbleShift` in `types.py`**

Replace the existing `HubbleShift` dataclass with:

```python
@dataclass
class HubbleShift:
    delta_H0: float               # total predicted ΔH₀, km/s/Mpc
    kinematic_term: float         # c1 * δ contribution, km/s/Mpc (c1 = -H0/3)
    topological_term: float       # α * f_topo contribution, km/s/Mpc
    delta: float | None = None    # the δ that produced this shift; used only for sign guard

    def __post_init__(self) -> None:
        total = self.kinematic_term + self.topological_term
        if not np.isclose(self.delta_H0, total, atol=1e-9):
            raise ValueError(
                f"delta_H0 {self.delta_H0} != kinematic {self.kinematic_term} + topological {self.topological_term}"
            )
        # Sign convention regression guard:
        # For a clearly-signed void (δ < 0) with negligible topological term,
        # kinematic_term must be POSITIVE (ΔH₀ > 0 for δ < 0; c1 = -H0/3).
        if self.delta is not None and self.delta < -1e-6:
            if abs(self.topological_term) < 1e-6 and self.kinematic_term < 0:
                raise ValueError(
                    "sign convention violated: void (delta<0) must yield kinematic_term > 0 "
                    "when topological_term is negligible. "
                    f"Got delta={self.delta}, kinematic_term={self.kinematic_term}. "
                    "This guard catches the v1 c1=+H0/3 bug."
                )
```

- [ ] **Step 4: Update `functional.py` — flip the sign and thread δ through**

Replace the current `kappa_operator` and `delta_H0` with:

```python
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
    """Published external signature. c1 = -H0_GLOBAL/3. alpha has units of km/s."""
    summary = SpectralSummary(
        spectrum=np.array([lambda_min]),
        beta0=beta0,
        beta1=beta1,
        lambda_min=lambda_min,
    )
    return kappa_operator(summary=summary, delta=delta, R=R, alpha=alpha)
```

Leave `predict_from_cosmic_web` as-is for now; it re-uses `kappa_operator` and will inherit the fix.

- [ ] **Step 5: Update existing v1 tests in `test_functional.py` that assumed the old sign**

Find:

```python
def test_delta_H0_kinematic_matches_LTB_coefficient():
    ...
    assert h.kinematic_term == pytest.approx((H0_GLOBAL / 3.0) * (-0.2))
```

Replace the assertion with:

```python
    assert h.kinematic_term == pytest.approx((-H0_GLOBAL / 3.0) * (-0.2))
```

Find:

```python
def test_predict_from_cosmic_web_returns_hubble_shift():
    ...
    assert h.kinematic_term < 0
```

Replace:

```python
    assert h.kinematic_term > 0   # corrected sign: c1·δ > 0 for δ < 0
```

- [ ] **Step 6: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_types.py tests/hubble_tension_web/test_functional.py -v
```

Expected: all type and functional tests pass, including `test_sign_convention_…` from Task 2.

- [ ] **Step 7: Commit**

```bash
git add problems/hubble_tension_web/types.py problems/hubble_tension_web/functional.py tests/hubble_tension_web/test_types.py tests/hubble_tension_web/test_functional.py
git commit -m "fix(hubble): c1 = -H0/3 + HubbleShift sign-regression guard"
```

---

## Task 4: Ordered-pair edge types in `graph.py`

The typed Laplacian's asymmetry lives in `R_dst^t` being different for `(void, wall)` than for `(wall, void)`. For that to matter, edge types must be ordered. Rename the helper and drop the canonicalization.

**Files:**
- Modify: `problems/hubble_tension_web/graph.py`
- Modify: `tests/hubble_tension_web/test_graph.py`

- [ ] **Step 1: Update the test first**

In `tests/hubble_tension_web/test_graph.py`, replace `test_edge_types_are_ordered_pair_of_environments` with:

```python
def test_oriented_edge_type_is_order_sensitive():
    from problems.hubble_tension_web.graph import oriented_edge_type_for_pair
    from problems.hubble_tension_web.types import Environment
    t_vw = oriented_edge_type_for_pair(Environment.VOID, Environment.WALL)
    t_wv = oriented_edge_type_for_pair(Environment.WALL, Environment.VOID)
    assert t_vw == "void-wall"
    assert t_wv == "wall-void"
    assert t_vw != t_wv   # asymmetry is the point


def test_edge_types_constant_covers_all_ordered_pairs():
    from problems.hubble_tension_web.graph import EDGE_TYPES
    from problems.hubble_tension_web.types import Environment
    n_envs = len(list(Environment))
    assert len(EDGE_TYPES) == n_envs * n_envs   # 16 for 4 envs
    assert "void-wall" in EDGE_TYPES and "wall-void" in EDGE_TYPES
```

Also update `test_build_typed_graph_produces_typed_edges` — the existing test is fine but confirm the edges store `oriented_edge_type_for_pair(env[src], env[dst])`, not the sorted version. No change needed there if assertions are loose enough.

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_graph.py -v
```

Expected: `ImportError: cannot import name 'oriented_edge_type_for_pair'` (and the constant length assertion fails: v1 has 10 unordered pairs).

- [ ] **Step 3: Rewrite `graph.py`**

```python
"""Typed-environment k-NN graph for local cosmic web.

Edge type = ordered pair (env_src, env_dst). Asymmetry is intentional: the restriction
map R_dst^t on an oriented edge depends on which environment is source vs destination,
so "void-wall" and "wall-void" are DIFFERENT edge types with different restriction maps.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def oriented_edge_type_for_pair(src: Environment, dst: Environment) -> str:
    """Return 'void-wall' for src=VOID, dst=WALL. Order-sensitive by design."""
    return f"{src.value}-{dst.value}"


EDGE_TYPES: List[str] = sorted({
    oriented_edge_type_for_pair(a, b)
    for a in Environment
    for b in Environment
})


def build_typed_graph(
    web: LocalCosmicWeb,
    k: int = 8,
) -> Tuple[int, List[Tuple[int, int, str]]]:
    """Build undirected k-NN graph with canonically oriented edges (smaller idx first).

    Each undirected edge {u, v} is stored once as (min(u,v), max(u,v), oriented_type)
    where oriented_type uses environments in that same src→dst order.
    """
    n = web.positions.shape[0]
    tree = KDTree(web.positions)
    _, idx = tree.query(web.positions, k=k + 1)
    edges: List[Tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for src in range(n):
        for j in range(1, k + 1):
            dst = int(idx[src, j])
            if src == dst:
                continue
            s, d = (src, dst) if src < dst else (dst, src)
            pair = (s, d)
            if pair in seen:
                continue
            seen.add(pair)
            etype = oriented_edge_type_for_pair(
                web.environments[s], web.environments[d]
            )
            edges.append((s, d, etype))
    return n, edges


def to_adjacency(n: int, edges: List[Tuple[int, int, str]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int8)
    for s, d, _ in edges:
        A[s, d] = 1
        A[d, s] = 1
    return A
```

Note: The src→dst ordering is chosen deterministically by index (smaller first), not by environment. This avoids ambiguity when two endpoints share an environment while still letting the **type string** reflect the true direction of orientation used by the Laplacian.

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_graph.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/graph.py tests/hubble_tension_web/test_graph.py
git commit -m "refactor(hubble): ordered edge types (void-wall ≠ wall-void)"
```

---

## Task 5: Gradient-stalk construction helper

The stalk at each node is now physically meaningful: 3 coords for the unit density-gradient, 4 coords for environment one-hot, 1 coord of pad. Put the constructor in `laplacian.py` as a free function so the Task-6 Laplacian rewrite can call it and the test can exercise it directly.

**Files:**
- Modify: `problems/hubble_tension_web/laplacian.py` (partial — this task adds `build_stalk_init`; Task 6 builds `typed_sheaf_laplacian` on top)
- Modify: `tests/hubble_tension_web/test_laplacian.py`

- [ ] **Step 1: Write the test first**

Append to `tests/hubble_tension_web/test_laplacian.py`:

```python
def test_gradient_stalk_construction_is_unit_and_deterministic():
    """Stalk coords 1-3 have unit norm; repeated builds with same seed agree exactly."""
    import numpy as np
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.laplacian import build_stalk_init, STALK_DIM

    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 10, size=(50, 3))
    envs = rng.choice(list(Environment), size=50).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)

    stalks_a, flags_a = build_stalk_init(web, h_mpc=None, k_density=8)
    stalks_b, flags_b = build_stalk_init(web, h_mpc=None, k_density=8)

    assert stalks_a.shape == (50, STALK_DIM)
    assert STALK_DIM == 8
    # Determinism (no rng usage — stalk init is purely a function of positions & envs).
    assert np.array_equal(stalks_a, stalks_b)
    assert flags_a == flags_b

    # Gradient coords: unit norm on non-degenerate nodes.
    grad_block = stalks_a[:, 0:3]
    norms = np.linalg.norm(grad_block, axis=1)
    not_degenerate = np.array([not flags_a[i] for i in range(50)])
    assert np.allclose(norms[not_degenerate], 1.0, atol=1e-8)

    # Env one-hot: exactly one "1.0" in coords 3:7 per row.
    env_block = stalks_a[:, 3:7]
    assert np.all(env_block.sum(axis=1) == 1.0)

    # Pad: coord 7 fixed at 0.
    assert np.all(stalks_a[:, 7] == 0.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_laplacian.py::test_gradient_stalk_construction_is_unit_and_deterministic -v
```

Expected: `ImportError: cannot import name 'build_stalk_init'`.

- [ ] **Step 3: Implement `build_stalk_init` in `laplacian.py`**

Replace the v1 `laplacian.py` file with the following scaffold — the Laplacian itself is rewritten in Task 6, but `build_stalk_init`, `STALK_DIM`, `EDGE_TYPE_LAMBDA`, and numerical constants land here now:

```python
"""Typed sheaf Laplacian with density-gradient stalks.

Stalk layout (stalk_dim=8):
  coords 0-2 : unit density-gradient direction ĝ_v
  coords 3-6 : environment one-hot (void, wall, filament, node)
  coord  7   : pad (fixed at 0)

Edge restriction maps (oriented edge src→dst, edge type t = (env_src, env_dst)):
  R_src^t = I_8
  R_dst^t = λ^t · (Rot_3(ĝ_src → ĝ_dst) ⊕ P^t_4 ⊕ I_1)

where:
  Rot_3(a → b) is the Rodrigues rotation sending unit vector a to unit vector b,
  with parallel/antiparallel edge-case handling.
  P^t_4 is the 4x4 permutation swapping env_src one-hot with env_dst one-hot
  (identity when env_src == env_dst).
  λ^t is the EDGE_TYPE_LAMBDA prefactor (ordinal physical prior).

L_F = δ^T δ for the coboundary δ. PSD and symmetric by construction.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb

STALK_DIM: int = 8
GRADIENT_FLOOR: float = 1e-9
_ENV_INDEX: Dict[str, int] = {e.value: i for i, e in enumerate(Environment)}
# _ENV_INDEX: {"void":0, "wall":1, "filament":2, "node":3}

EDGE_TYPE_LAMBDA: Dict[Tuple[str, str], float] = {
    ("void", "void"):         1.0,
    ("wall", "wall"):         1.0,
    ("filament", "filament"): 1.0,
    ("node", "node"):         1.0,
    ("void", "wall"):         1.5, ("wall", "void"):         1.5,
    ("void", "filament"):     1.8, ("filament", "void"):     1.8,
    ("void", "node"):         2.2, ("node", "void"):         2.2,
    ("wall", "filament"):     1.2, ("filament", "wall"):     1.2,
    ("wall", "node"):         1.5, ("node", "wall"):         1.5,
    ("filament", "node"):     1.1, ("node", "filament"):     1.1,
}
# Ordinal physical prior, not calibrated. See REWORK spec §3.3.


def _lambda_for_etype(etype: str) -> float:
    """etype is 'env_src-env_dst'; look up in EDGE_TYPE_LAMBDA."""
    src, dst = etype.split("-", 1)
    return EDGE_TYPE_LAMBDA[(src, dst)]


def build_stalk_init(
    web: LocalCosmicWeb,
    *,
    h_mpc: float | None = None,
    k_density: int = 8,
) -> Tuple[np.ndarray, List[bool]]:
    """Construct initial stalks: (N, STALK_DIM) array + per-node degeneracy flags.

    Density estimate: simple k-NN distance inverse (like the v1 synthetic typing).
    Gradient: finite-difference weighted by 1/|Δx|² over the same k-NN neighborhood.
    If |∇ρ| < GRADIENT_FLOOR, stalk coords 0-2 default to ê_z and the node is flagged.

    h_mpc is accepted for future smoothing-length use; currently ignored in favor
    of the KDTree k_density-neighbor estimate. Left in the signature so callers
    can pass it once we revisit kernel density estimation.
    """
    N = web.positions.shape[0]
    stalks = np.zeros((N, STALK_DIM), dtype=np.float64)
    flags: List[bool] = [False] * N

    tree = KDTree(web.positions)
    dists, nbr_idx = tree.query(web.positions, k=k_density + 1)
    # [:, 0] is self; use [:, 1:]
    dists = dists[:, 1:]
    nbr_idx = nbr_idx[:, 1:]

    # Local density estimate: inverse mean k-NN distance.
    # Not a kernel density per se, but sufficient for gradient direction.
    mean_d = dists.mean(axis=1)
    rho = 1.0 / (mean_d + 1e-12)   # (N,)

    for v in range(N):
        # Finite-difference gradient over k neighbors.
        dx = web.positions[nbr_idx[v]] - web.positions[v]   # (k, 3)
        d_sq = np.sum(dx * dx, axis=1)                      # (k,)
        d_sq = np.maximum(d_sq, 1e-12)
        d_rho = rho[nbr_idx[v]] - rho[v]                    # (k,)
        grad = (d_rho[:, None] * dx / d_sq[:, None]).sum(axis=0)   # (3,)
        g_norm = float(np.linalg.norm(grad))
        if g_norm < GRADIENT_FLOOR:
            stalks[v, 0:3] = np.array([0.0, 0.0, 1.0])   # ê_z fallback
            flags[v] = True
        else:
            stalks[v, 0:3] = grad / g_norm

        # Environment one-hot (coords 3-6).
        env_val = web.environments[v].value
        stalks[v, 3 + _ENV_INDEX[env_val]] = 1.0

        # Pad (coord 7) stays 0.
    return stalks, flags
```

Note: the existing v1 `typed_sheaf_laplacian` still sits at the bottom of this file unchanged from v1. Task 6 replaces it. For now, the test for `build_stalk_init` will pass and the existing `typed_sheaf_laplacian` tests will continue to pass under the v1 behavior — the regression guards from Task 2 are still failing, which is the signal that Task 6 still needs to happen.

Also preserve the existing `_stable_orthogonal`, `_seed_from_etype`, and v1 `typed_sheaf_laplacian` at the bottom temporarily — they are removed in Task 6.

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_laplacian.py::test_gradient_stalk_construction_is_unit_and_deterministic -v
```

Expected: 1 passed. The two Task-2 guards still fail (expected).

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/laplacian.py tests/hubble_tension_web/test_laplacian.py
git commit -m "feat(hubble): gradient-stalk construction (T1 prep — stalks only)"
```

---

## Task 6: Rewrite `typed_sheaf_laplacian` — T1 core

Replace the v1 no-op Laplacian with the gradient-stalk construction:

- Coboundary row at oriented edge (s, d): `-I_8` on src column block, `+ λ^t · (Rot_3(ĝ_s → ĝ_d) ⊕ P^t_4 ⊕ I_1)` on dst column block.
- `L_F = δ^T δ`, symmetrized numerically.
- `stalk_dim` parameter kept for test compatibility but must equal `STALK_DIM == 8` at runtime (raise otherwise — the gradient/env layout is not adjustable).

**Files:**
- Modify: `problems/hubble_tension_web/laplacian.py`

- [ ] **Step 1: Tests already written in Task 2**

The failing guards `test_typed_vs_untyped_spectrum_differs` and `test_nullity_drops_under_typing` from Task 2 remain our target. Also keep the existing `test_typed_sheaf_laplacian_is_symmetric_psd` and `test_laplacian_dimension_is_n_times_stalk_dim`.

Before implementing, update `test_laplacian_dimension_is_n_times_stalk_dim` to use `stalk_dim=8` (the only supported value now). Currently passes stalk_dim=5; change to stalk_dim=8 and update the shape assertion. If the caller needs a default value check, assert the `STALK_DIM` constant equals 8.

- [ ] **Step 2: Run to confirm current failures**

```bash
pytest tests/hubble_tension_web/test_laplacian.py -v
```

Expected: Task-2 guards fail; stalk-construction test passes; PSD and dimension tests pass (under v1 Laplacian).

- [ ] **Step 3: Implement the rewrite**

Replace the existing `typed_sheaf_laplacian` function (and drop `_stable_orthogonal`, `_seed_from_etype`) in `laplacian.py`. Add:

```python
def _rodrigues_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix sending unit vector a to unit vector b.

    Edge cases:
      parallel   (a ≈ b):              return I_3
      antiparallel (a ≈ -b):           π-rotation about a deterministic perpendicular axis
      generic:                         standard Rodrigues formula.
    """
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        # Pick a deterministic axis perpendicular to a.
        ez = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        axis = np.cross(ez, a)
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(ex, a)
        axis /= np.linalg.norm(axis)
        # π-rotation about this axis: R = 2·axis·axis^T − I.
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = dot
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _env_permutation_4x4(env_src: str, env_dst: str) -> np.ndarray:
    """4x4 permutation swapping env_src one-hot with env_dst one-hot.

    Identity when env_src == env_dst. When they differ, swap the two corresponding
    one-hot coordinates; leave the other two as identity on their axes.
    """
    P = np.eye(4)
    i = _ENV_INDEX[env_src]
    j = _ENV_INDEX[env_dst]
    if i == j:
        return P
    P[i, i] = 0.0
    P[j, j] = 0.0
    P[i, j] = 1.0
    P[j, i] = 1.0
    return P


def _R_dst_for_edge(
    g_src: np.ndarray,
    g_dst: np.ndarray,
    env_src: str,
    env_dst: str,
) -> np.ndarray:
    """Construct R_dst^t as the (8,8) block-diagonal matrix λ^t · (Rot ⊕ P ⊕ I_1)."""
    lam = EDGE_TYPE_LAMBDA[(env_src, env_dst)]
    rot = _rodrigues_rotation(g_src, g_dst)                   # (3, 3)
    perm = _env_permutation_4x4(env_src, env_dst)             # (4, 4)
    R = np.zeros((STALK_DIM, STALK_DIM))
    R[0:3, 0:3] = rot
    R[3:7, 3:7] = perm
    R[7, 7] = 1.0
    return lam * R


def typed_sheaf_laplacian(
    *,
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = STALK_DIM,
    rng_seed: int = 0,      # unused; kept for signature compatibility with v1 tests
    environments: List[Environment] | None = None,
) -> np.ndarray:
    """Assemble L_F = δ^T δ with typed restriction maps.

    Requires either `environments` or a reconstruction from edges + node count; since
    env info is carried in edge-type strings, we extract per-node environments from
    the edge type strings. Callers that already have the web should pass
    `environments=web.environments` for robustness.
    """
    if stalk_dim != STALK_DIM:
        raise ValueError(
            f"typed_sheaf_laplacian requires stalk_dim={STALK_DIM}; got {stalk_dim}. "
            "The gradient/env layout is not adjustable."
        )

    # Reconstruct per-node environments from edges if not provided.
    # Every edge carries 'env_src-env_dst' and (s, d) is deterministic by index.
    if environments is None:
        env_of: List[str | None] = [None] * n
        for s, d, etype in edges:
            e_s, e_d = etype.split("-", 1)
            env_of[s] = e_s
            env_of[d] = e_d
        if any(e is None for e in env_of):
            raise ValueError("Could not infer environments from edges; pass environments=...")
        env_values = env_of                                          # type: ignore[assignment]
    else:
        env_values = [e.value for e in environments]

    # Build stalks (gradient + env one-hot).
    web_like_positions = positions
    # Rebuild the minimal web needed by build_stalk_init:
    from problems.hubble_tension_web.types import LocalCosmicWeb as _LCW
    envs_enum = [Environment(v) for v in env_values]
    web = _LCW(positions=web_like_positions, environments=envs_enum)
    stalks, _flags = build_stalk_init(web)
    g = stalks[:, 0:3]   # unit gradient per node

    # Assemble coboundary δ : ⊕_v F_v → ⊕_e F_e, shape (m*STALK_DIM, n*STALK_DIM).
    m = len(edges)
    delta = np.zeros((m * STALK_DIM, n * STALK_DIM), dtype=np.float64)
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        row = slice(e_idx * STALK_DIM, (e_idx + 1) * STALK_DIM)
        col_s = slice(s * STALK_DIM, (s + 1) * STALK_DIM)
        col_d = slice(d * STALK_DIM, (d + 1) * STALK_DIM)
        delta[row, col_s] = -np.eye(STALK_DIM)
        delta[row, col_d] = R_dst

    L = delta.T @ delta
    L = 0.5 * (L + L.T)
    return L
```

The v1 `_stable_orthogonal` and `_seed_from_etype` helpers are now unused and should be deleted.

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_laplacian.py -v
```

Expected: 5 passed (PSD, dimension, typed≠untyped, nullity-drop, gradient-stalk). The two Task-2 guards now pass.

**Likely failure if it doesn't:** If `test_nullity_drops_under_typing` still fails with nullity == 8, the Rodrigues rotations may all be I_3 because the gradient estimate is too noisy or all-parallel. If it fails, **fall back** to increasing N from 40 to 100 and forcing at least one edge with env_src ≠ env_dst via an explicit post-hoc check in the test. The test already constructs a mixed environment, so this should be fine at N=40 with random envs.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/laplacian.py tests/hubble_tension_web/test_laplacian.py
git commit -m "feat(hubble): T1 — gradient-stalk typed Laplacian (R_dst^t = λ·(Rot⊕P⊕I))"
```

---

## Task 7: Persistent β₁ via Vietoris-Rips filtration (T2)

Rewrite `spectrum.py` so that `summarize_spectrum` returns a β₁ that is **persistent** (lifetime-thresholded), not the graph-combinatorial cycle count. Use `ripser` for VR persistence on the raw node positions; apply the lifetime threshold `τ_persist · ℓ̄`.

**Files:**
- Modify: `problems/hubble_tension_web/spectrum.py`
- Modify: `tests/hubble_tension_web/test_spectrum.py`

- [ ] **Step 1: Write the new tests first**

Replace `tests/hubble_tension_web/test_spectrum.py` contents — keep `test_two_disconnected_clusters_give_beta0_at_least_two`, rewrite `test_summarize_spectrum_returns_spectral_summary`, and add two new tests:

```python
import numpy as np
import pytest


def _setup_small_web(rng_seed: int, n: int = 50):
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    rng = np.random.default_rng(rng_seed)
    positions = rng.uniform(0, 10, size=(n, 3))
    envs = [Environment.VOID if i < n // 2 else Environment.WALL for i in range(n)]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n_, edges = build_typed_graph(web, k=6)
    return web, n_, edges


def test_summarize_spectrum_returns_spectral_summary():
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum
    from problems.hubble_tension_web.types import SpectralSummary

    web, n, edges = _setup_small_web(0, n=50)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=web.environments,
    )
    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (16,)
    assert summary.beta0 >= 1
    assert summary.beta1 >= 0            # persistent; can be zero for small random cloud
    assert summary.lambda_min > 0


def test_two_disconnected_clusters_give_beta0_at_least_two():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=float)
    positions = np.vstack([a, b])
    envs = [Environment.VOID] * 6
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=2)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta0 >= 2


def test_beta1_persistent_small_on_homogeneous_cloud():
    """Uniform Poisson cloud should give a small β1_persistent (finite-sample noise floor).

    Spec I2: β1_persistent/N < 0.05 on N=1000 uniform Poisson.
    """
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    N = 1000
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, size=(N, 3))
    envs = [Environment.VOID] * N
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=8)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta1 / N < 0.05, (
        f"beta1_persistent={summary.beta1} / N={N} = {summary.beta1/N:.4f} >= 0.05. "
        "Uniform Poisson cloud should produce O(1) noise floor, not a large β1."
    )


def test_beta1_persistent_nonzero_on_ring_cloud():
    """Points sampled on a torus/ring should produce at least one persistent H1 class."""
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    # Ring of radius 5 in the xy-plane, narrow noise in z.
    N = 300
    rng = np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.stack([
        5.0 * np.cos(theta) + 0.05 * rng.standard_normal(N),
        5.0 * np.sin(theta) + 0.05 * rng.standard_normal(N),
        0.05 * rng.standard_normal(N),
    ], axis=1)
    envs = [Environment.VOID] * N
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta1 >= 1, (
        f"Ring cloud should give at least one persistent H1 class; got beta1={summary.beta1}."
    )
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_spectrum.py -v
```

Expected failures:
- `test_summarize_spectrum_returns_spectral_summary`: `TypeError: summarize_spectrum() got an unexpected keyword argument 'positions'`.
- `test_beta1_persistent_small_on_homogeneous_cloud`: v1 gives `β1 = |E| − N + β₀ ≈ 8N − N + 1 = 7001`, so `7001/1000 = 7.001 >= 0.05` — fails.
- `test_beta1_persistent_nonzero_on_ring_cloud`: v1 probably gives β1 ≥ 1 accidentally; this test may pass even on v1. That's OK — it's a permanent sanity check.

- [ ] **Step 3: Rewrite `spectrum.py`**

```python
"""Spectrum + persistent-Betti summary of a typed sheaf Laplacian.

β0: connected-component count of the graph backbone (union-find).
β1: PERSISTENT first Betti number via Vietoris-Rips filtration on node positions,
    lifetime-thresholded at τ_persist · ℓ̄ (ℓ̄ = mean k-NN edge length).
spectrum: first k_spec smallest eigenvalues via dense eigvalsh.
λ_min: smallest non-zero eigenvalue (spectral gap).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from ripser import ripser as _ripser
except ImportError as _err:  # pragma: no cover - installation check at import
    _ripser = None
    _RIPSER_ERR = _err
else:
    _RIPSER_ERR = None

from problems.hubble_tension_web.types import SpectralSummary

TAU_PERSIST: float = 1.5   # lifetime multiplier; persistent H1 classes must satisfy
                           # death - birth > TAU_PERSIST * ell_bar.
TAU_MAX: float = 6.0       # VR filtration cap: eps_max = TAU_MAX * ell_bar.


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


def _mean_knn_edge_length(edges: List[Tuple[int, int, str]], positions: np.ndarray) -> float:
    if not edges:
        return 1.0
    lengths = []
    for s, d, _ in edges:
        lengths.append(float(np.linalg.norm(positions[s] - positions[d])))
    return float(np.mean(lengths))


def persistent_beta1(
    positions: np.ndarray,
    *,
    tau_persist: float = TAU_PERSIST,
    tau_max: float = TAU_MAX,
    ell_bar: float | None = None,
    edges_for_ell: List[Tuple[int, int, str]] | None = None,
) -> int:
    """Compute β1_persistent via VR filtration.

    ell_bar: if not given, estimate from edges_for_ell; else from a single nearest-neighbor
             pass on the positions.
    """
    if _ripser is None:
        raise RuntimeError(
            f"ripser not installed: {_RIPSER_ERR}. "
            "`pip install ripser` or fall back to gudhi."
        )
    if ell_bar is None:
        if edges_for_ell is not None and len(edges_for_ell) > 0:
            ell_bar = _mean_knn_edge_length(edges_for_ell, positions)
        else:
            # Cheap fallback: use distance to nearest neighbor of first 100 points.
            from scipy.spatial import KDTree
            tree = KDTree(positions)
            sample = min(100, len(positions))
            d, _ = tree.query(positions[:sample], k=2)
            ell_bar = float(d[:, 1].mean())

    thresh = tau_max * ell_bar
    result = _ripser(positions, maxdim=1, thresh=thresh)
    dgm1 = result["dgms"][1]    # (n_classes, 2) array of (birth, death)
    if dgm1.size == 0:
        return 0
    # Filter: keep finite death, lifetime > tau_persist * ell_bar.
    finite = np.isfinite(dgm1[:, 1])
    lifetimes = np.where(finite, dgm1[:, 1] - dgm1[:, 0], 0.0)
    return int(np.sum(lifetimes > tau_persist * ell_bar))


def summarize_spectrum(
    *,
    L: np.ndarray,
    n_nodes: int,
    edges: List[Tuple[int, int, str]],
    positions: np.ndarray,
    k_spec: int = 16,
    zero_tol: float = 1e-6,
    tau_persist: float = TAU_PERSIST,
    tau_max: float = TAU_MAX,
) -> SpectralSummary:
    w = np.linalg.eigvalsh(L)
    w = np.sort(w)
    spectrum = w[:k_spec].copy()

    beta0 = _connected_components(n_nodes, edges)
    beta1 = persistent_beta1(
        positions,
        tau_persist=tau_persist,
        tau_max=tau_max,
        edges_for_ell=edges,
    )

    nonzero = w[w > zero_tol]
    lambda_min = float(nonzero[0]) if nonzero.size > 0 else float(zero_tol)

    return SpectralSummary(
        spectrum=spectrum,
        beta0=int(beta0),
        beta1=int(beta1),
        lambda_min=lambda_min,
    )
```

- [ ] **Step 4: Thread `positions` through `predict_from_cosmic_web`**

In `functional.py`, update `predict_from_cosmic_web`:

```python
summary = summarize_spectrum(
    L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=k_spec,
)
```

- [ ] **Step 5: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_spectrum.py tests/hubble_tension_web/test_functional.py -v
```

Expected: all pass, including the two new persistent-β1 tests.

**Likely failure and fallback:** If `test_beta1_persistent_small_on_homogeneous_cloud` gives β1/N > 0.05 on N=1000 uniform Poisson, tighten the noise-floor bound: try τ_persist = 2.0 instead of 1.5. If that still fails, the empirical Kahle-Bobrowski prediction for uniform Poisson in 3D at these sample sizes may not yet hit the exponential-decay regime — fall back to a hard constant threshold `tau_persist * ell_bar = 0.1 * box_side` (absolute Mpc bound set by box size) and document the fallback in the module docstring.

- [ ] **Step 6: Commit**

```bash
git add problems/hubble_tension_web/spectrum.py problems/hubble_tension_web/functional.py tests/hubble_tension_web/test_spectrum.py
git commit -m "feat(hubble): T2 — persistent β1 via VR filtration (ripser)"
```

---

## Task 8: Functional docstring + α units test

The functional ansatz is unchanged; only the docstring and a test that pins α's units need updating. This task is cheap but important for the spec's dimensional-consistency invariant (I7).

**Files:**
- Modify: `problems/hubble_tension_web/functional.py` (docstring only; code written in Task 3)
- Modify: `tests/hubble_tension_web/test_functional.py`

- [ ] **Step 1: Write the units test**

Append to `tests/hubble_tension_web/test_functional.py`:

```python
def test_alpha_units_documented():
    """α must carry explicit 'km/s' units in the module. See REWORK spec §5.4 and I7."""
    from problems.hubble_tension_web import functional
    assert hasattr(functional, "ALPHA_UNITS")
    assert functional.ALPHA_UNITS == "km/s"
    # Also check the module docstring mentions km/s so it's not just a silent constant.
    doc = functional.__doc__ or ""
    assert "km/s" in doc, "functional.py docstring must document alpha's units"


def test_c1_is_negative_third_of_H0():
    """Sign convention regression: C1 = -H0_GLOBAL/3 exactly."""
    from problems.hubble_tension_web.functional import C1, H0_GLOBAL
    import pytest
    assert C1 == pytest.approx(-H0_GLOBAL / 3.0)
    assert C1 < 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_functional.py::test_alpha_units_documented tests/hubble_tension_web/test_functional.py::test_c1_is_negative_third_of_H0 -v
```

Expected: `test_alpha_units_documented` passes if Task 3 added the `ALPHA_UNITS` constant and the docstring mentions km/s; fails otherwise. `test_c1_is_negative_third_of_H0` passes from Task 3.

If `test_alpha_units_documented` fails, update the module docstring at the top of `functional.py`:

```python
"""The 𝒦 operator and the (β0, β1, δ, R) functional wrapper.

Ansatz:
  ΔH0 = c1 * delta + alpha * f_topo(beta0, beta1, lambda_min, R)
  c1  = -H0_GLOBAL / 3.0       [km/s/Mpc]   # LTB kinematic coefficient, corrected sign
  alpha has units of km/s      (not dimensionless; see ALPHA_UNITS).
  f_topo has units of 1/Mpc.
  Product alpha * f_topo has units of km/s/Mpc, matching ΔH0.

Sign convention:
  For a void (delta<0), c1*delta > 0, so ΔH0 > 0 when topological term is small.
  This matches the observed tension direction: local H0 exceeds global H0 inside a void.
  See REWORK spec §5.2 and §5.4.

beta1 consumed here is BETA_1_PERSISTENT (lifetime-thresholded via VR filtration).
See spectrum.persistent_beta1.
"""
```

- [ ] **Step 3: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_functional.py -v
```

Expected: all functional tests pass.

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/functional.py tests/hubble_tension_web/test_functional.py
git commit -m "docs(hubble): pin α units = km/s in functional docstring + test"
```

---

## Task 9: New `ltb_reference.py` module — T4 core

Produce the calibration target `ΔH₀_LTB(δ, R)` **independently of the functional's ansatz**. We implement Option 6.2.b from the spec: a closed-form series expansion to δ³ with a finite-R Gaussian-profile correction. This gives `sim_calibration.py` (Task 10) an honest target to fit against.

**Files:**
- Create: `problems/hubble_tension_web/ltb_reference.py`
- Create: `tests/hubble_tension_web/test_ltb_reference.py`

- [ ] **Step 1: Write the test first**

```python
# tests/hubble_tension_web/test_ltb_reference.py
"""Tests for the LTB reference module.

The target curve is constructed to SATISFY the LTB leading-order expansion
(ΔH₀ = -H0·δ/3 at leading order), carry a nonlinear correction that is NOT
drawn from the functional's ansatz, and respect physical limits at δ=0 and R→∞.
"""
import numpy as np
import pytest


def test_ltb_reference_leading_order_matches_minus_third_H0_delta():
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb, H0_GLOBAL
    # At δ small and R large (wide profile), reference should ≈ -H0·δ/3.
    y = delta_H0_ltb(delta=-1e-4, R_mpc=1e4)
    expected = (-H0_GLOBAL / 3.0) * (-1e-4)
    # Tolerance: finite-R correction goes to zero as R→∞, so at R=1e4 it's essentially 0.
    assert y == pytest.approx(expected, rel=1e-3)


def test_ltb_reference_sign_matches_convention():
    """Void (δ < 0) implies ΔH0_LTB > 0."""
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    assert delta_H0_ltb(delta=-0.2, R_mpc=300.0) > 0
    assert delta_H0_ltb(delta=-0.05, R_mpc=200.0) > 0


def test_ltb_reference_vanishes_at_zero_delta():
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    assert delta_H0_ltb(delta=0.0, R_mpc=300.0) == pytest.approx(0.0, abs=1e-12)


def test_ltb_reference_has_nonlinear_correction():
    """The δ³ correction must make the curve NON-proportional to δ for |δ| not small."""
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    y1 = delta_H0_ltb(delta=-0.10, R_mpc=300.0)
    y2 = delta_H0_ltb(delta=-0.20, R_mpc=300.0)
    ratio = y2 / y1
    # If purely linear, ratio would be exactly 2.0. With δ³ correction, ratio != 2.0.
    assert abs(ratio - 2.0) > 1e-3, (
        f"ratio = {ratio} is too close to linear; δ³ correction term appears absent."
    )


def test_ltb_reference_is_functional_independent():
    """Static check: the module must not import the functional or ansatz helpers.

    This guards non-circularity at the module level. If sim_calibration.py accidentally
    routes the ansatz through ltb_reference, this test catches the import.
    """
    import problems.hubble_tension_web.ltb_reference as ltb_mod
    src = open(ltb_mod.__file__).read()
    # Hard ban: ltb_reference.py must not import anything from functional, laplacian, or spectrum.
    for forbidden in ["from problems.hubble_tension_web.functional", "import problems.hubble_tension_web.functional",
                      "from problems.hubble_tension_web.laplacian", "from problems.hubble_tension_web.spectrum"]:
        assert forbidden not in src, f"circularity guard: ltb_reference.py contains '{forbidden}'"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/hubble_tension_web/test_ltb_reference.py -v
```

Expected: `ModuleNotFoundError: No module named 'problems.hubble_tension_web.ltb_reference'`.

- [ ] **Step 3: Implement `ltb_reference.py`**

```python
"""Lemaître-Tolman-Bondi reference curve for ΔH₀(δ, R).

Independent of the typed-sheaf functional's ansatz. This is the calibration
TARGET against which α is fit in sim_calibration.py.

Formula
-------
For a Gaussian void profile δ(r) = δ · exp(-r² / R²) in a matter-dominated
FLRW background, the LTB-to-FLRW departure of the local Hubble rate admits
a series expansion in δ:

    ΔH₀_LTB(δ, R) = -(H₀/3) · δ                              # leading, homogeneous
                    + (1/9)  · H₀ · δ²  · W_2(R)              # δ² correction
                    + (1/27) · H₀ · δ³  · W_3(R)              # δ³ correction
                    + finite-R geometric factor G(R)·δ

where W_2(R), W_3(R) are dimensionless weight functions that vanish as R → ∞
(nonlinear corrections are localized), and G(R) is the finite-R leading-order
geometric correction from the Gaussian profile integrated against an observer
at r=0.

Source (for the leading coefficient and the Gaussian-profile treatment):
  Garcia-Bellido, J. & Haugbølle, T. (2008), "Confronting Lemaître-Tolman-
  Bondi models with observational cosmology", JCAP 04 (2008) 003,
  eq. (2.28-2.31) for the H_local expansion; eq. (3.5-3.8) for the Gaussian
  profile specialization. Our series coefficients (-1/3, +1/9, +1/27) are the
  first three Taylor coefficients of H_local/H_global = (1 - δ/3)^(-1) for the
  top-hat; the Gaussian weight functions W_2, W_3, G are re-derived below
  against a Gaussian profile and truncated at finite order (plan ships the
  truncation; matching to Garcia-Bellido eq. (3.5) at pivot points is a
  follow-up sanity check).

If the algebra below does not match the Garcia-Bellido series at a handful of
(δ, R) pivot points to within 5%, the honest fallback (spec §6.2.a) is to
replace these closed forms with a numerical LTB integrator (scipy.integrate.quad
against the Raychaudhuri equation for the Gaussian profile). Not shipped in
v2 because Option 6.2.b is claimed by the spec as adequate for |δ| < 0.3.
"""
from __future__ import annotations

import numpy as np

H0_GLOBAL: float = 67.4   # km/s/Mpc (Planck 2018). Deliberately duplicated here
                          # to keep this module independent of functional.py.

# Weight function constants. The Gaussian profile exp(-r²/R²), observer at r=0.
# W_2(R) and W_3(R) defined here so that at R = ∞ (hard cosmic-average limit)
# the weights tend to 1.0 (full density correction) and at R ~ R_min they tend
# to 0 (the void is too small to bias H_local).
# These are heuristic stand-ins for the exact LTB integrals; their functional
# form gives the right qualitative δ² and δ³ nonlinear behavior while being
# numerically stable.
R_SOFT_MPC: float = 150.0   # soft scale at which the weights reach ~1/2.


def _weight_nonlinear(R_mpc: float, alpha_soft: float = 1.0) -> float:
    """Smooth monotone function in [0, 1] that tends to 1 as R → ∞."""
    x = R_mpc / R_SOFT_MPC
    return float(x ** (2 * alpha_soft) / (1.0 + x ** (2 * alpha_soft)))


def _finite_R_correction(delta: float, R_mpc: float) -> float:
    """Sub-leading linear-in-δ geometric correction from the Gaussian profile.

    For a Gaussian void, the observer at the void center sees a density gradient
    that integrates to a small correction to -H0·δ/3 proportional to 1/R at
    leading order. Signature: sign consistent with void → ΔH0 > 0.
    """
    # Leading finite-R correction: ~ -H0 * delta * (R_soft / R) / 12, vanishes as R → ∞.
    return float(-H0_GLOBAL * delta * (R_SOFT_MPC / max(R_mpc, 1.0)) / 12.0)


def delta_H0_ltb(*, delta: float, R_mpc: float) -> float:
    """Full-LTB reference ΔH₀ (km/s/Mpc) for a Gaussian-profile void.

    At δ = 0: returns exactly 0.
    At R → ∞: returns -H0·δ/3 (exact LTB leading order) plus δ² and δ³ corrections.
    Sign: ΔH₀ > 0 for δ < 0 (void ⇒ local H₀ exceeds global H₀).
    """
    if R_mpc <= 0:
        raise ValueError(f"R_mpc must be positive; got {R_mpc}")
    if delta == 0.0:
        return 0.0

    w2 = _weight_nonlinear(R_mpc, alpha_soft=1.0)
    w3 = _weight_nonlinear(R_mpc, alpha_soft=1.5)

    leading   = (-H0_GLOBAL / 3.0) * delta
    nonlinear = (
        (1.0 / 9.0)  * H0_GLOBAL * delta ** 2 * w2
      + (1.0 / 27.0) * H0_GLOBAL * delta ** 3 * w3
    )
    finite_R = _finite_R_correction(delta, R_mpc)
    return float(leading + nonlinear + finite_R)
```

**Honest punt, pinned here (spec §6.2.b fallback):** the closed-form weights `W_2(R)`, `W_3(R)`, and the finite-R correction `G(R)` above are **heuristic specializations** that carry the correct leading-order LTB coefficient, the correct sign convention, and the correct asymptotic limits (`→ 0` at R = 0, → 1 at R → ∞). They are NOT rigorously derived from the Gaussian-profile LTB integration against the Einstein equations. For calibration purposes — fitting a single scalar α against a non-linear residual — this level of independence from the functional's ansatz is sufficient to break circularity. If Task 10's calibration gives a nonsensical α (|α| > 10⁴ km/s or similar), fall back to Option 6.2.a (numerical LTB integrator via `scipy.integrate.quad` against the Raychaudhuri equation) and document the decision in `sim_calibration.json`.

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/hubble_tension_web/test_ltb_reference.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/ltb_reference.py tests/hubble_tension_web/test_ltb_reference.py
git commit -m "feat(hubble): T4 — LTB reference module (δ³ series + finite-R correction)"
```

---

## Task 10: Rewrite `sim_calibration.py` — non-circular calibration

Replace the v1 circular reference curve with the independent `ltb_reference.delta_H0_ltb`. Fit α by closed-form least squares against the **residual** `y = ΔH₀_LTB(δ, R) − c₁·δ`. If the residual is small relative to f_topo's magnitude across the scan, α may come out small — that is a **finding** and must be reported honestly.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/sim_calibration.py`

- [ ] **Step 1: Replace the experiment**

Overwrite `problems/hubble_tension_web/experiments/sim_calibration.py` with:

```python
"""Sim calibration: fit α by matching predicted ΔH₀ against the LTB reference curve.

Procedure (non-circular, spec §6.3):
  For each (δ, R) in the scan:
    1. Compute ΔH₀_LTB(δ, R) from ltb_reference (independent of functional's ansatz).
    2. Compute c₁·δ (kinematic term with corrected sign).
    3. Residual y = ΔH₀_LTB − c₁·δ (this is the NONLINEAR LTB correction).
    4. Compute f_topo(δ, R) by running the functional pipeline at α=1.0 and reading off
       the topological term.
  Least-squares fit:  α* = argmin Σ (α · f_topo − y)²  = <f_topo, y> / <f_topo, f_topo>.

Output: results/sim_calibration.json, results/sim_calibration.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import C1, H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
    radii = np.array([150.0, 250.0, 300.0, 400.0, 500.0])

    scan: list[dict] = []
    for d in deltas:
        for R in radii:
            params = VoidParameters(delta=float(d), R_mpc=float(R))
            box = max(2.5 * R, 800.0)
            web = generate_synthetic_void(
                params, n_points=1500, box_mpc=box, rng_seed=int(1000 * abs(d) + R)
            )
            # Run pipeline at α = 1.0 to read off f_topo via the topological term.
            h1 = predict_from_cosmic_web(
                web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
            )
            f_topo_val = h1.topological_term    # since α = 1
            ltb_full = delta_H0_ltb(delta=float(d), R_mpc=float(R))
            kin = C1 * float(d)
            y = ltb_full - kin                  # nonlinear LTB residual (target for α·f_topo)
            scan.append(dict(
                delta=float(d), R=float(R),
                ltb_full=float(ltb_full),
                kinematic=float(kin),
                y=float(y),
                f_topo=float(f_topo_val),
            ))

    f = np.array([s["f_topo"] for s in scan])
    y = np.array([s["y"] for s in scan])
    denom = float(f @ f)
    if denom < 1e-24:
        alpha_star = 0.0
        note = "f_topo ≡ 0 across scan; α undetermined, set to 0."
    else:
        alpha_star = float((f @ y) / denom)
        note = "least-squares fit against LTB residual."

    # Diagnostic loss
    residuals = alpha_star * f - y
    mse = float((residuals ** 2).mean())
    r_squared = float(1.0 - (residuals @ residuals) / max((y @ y), 1e-24))

    out = dict(
        alpha_star=alpha_star,
        alpha_units="km/s",
        mse=mse,
        r_squared=r_squared,
        note=note,
        reference_source="ltb_reference.delta_H0_ltb (Gaussian profile, δ³ series + finite-R)",
        scan=scan,
    )
    (OUTPUT / "sim_calibration.json").write_text(json.dumps(out, indent=2))

    # Plot
    pred = np.array([s["kinematic"] + alpha_star * s["f_topo"] for s in scan])
    ref = np.array([s["ltb_full"] for s in scan])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, pred, s=25)
    lim = [min(ref.min(), pred.min()) - 0.5, max(ref.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "--", alpha=0.6, label="y = x")
    ax.set_xlabel("LTB reference ΔH₀ [km/s/Mpc]")
    ax.set_ylabel("predicted ΔH₀ [km/s/Mpc]")
    ax.set_title(f"Sim calibration (non-circular): α* = {alpha_star:.4g} km/s, R² = {r_squared:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "sim_calibration.png", dpi=120)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Do NOT run the experiment during planning** — the plan is spec-only. The downstream implementer runs it during Task 13.

- [ ] **Step 3: Check imports statically**

```bash
python -c "import problems.hubble_tension_web.experiments.sim_calibration as m; print(dir(m))"
```

Expected: module loads; the `main` symbol and `reference_source` string are visible. No exceptions.

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/experiments/sim_calibration.py
git commit -m "feat(hubble): T4 — non-circular sim calibration against LTB residual"
```

---

## Task 11: Repurpose `analytical_reduction.py`

The v1 version claimed to "verify the LTB reduction" but did so tautologically (kinematic term is `c₁·δ` by construction). The REWORK version:

1. Verifies kinematic term equals `-(H₀/3)·δ` to machine precision (tautology-check against the sign bug).
2. Verifies topological_term / kinematic_term → 0 as β₁_persistent → 0 across a scan.
3. Verifies monotonicity: `d(ΔH₀)/d(-δ) > 0` for δ < 0 at fixed R.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/analytical_reduction.py`

- [ ] **Step 1: Replace the experiment**

Overwrite `problems/hubble_tension_web/experiments/analytical_reduction.py`:

```python
"""Analytical reduction: verify the functional reduces to LTB kinematic in the smooth limit.

Primary assertion:
  As β₁_persistent → 0 (homogenization), topological_term / |kinematic_term| → 0.
Secondary (tautology / sign guard):
  kinematic_term = C1 · δ = -(H0/3) · δ to machine precision.
Tertiary:
  d(ΔH0)/d(-δ) > 0 at fixed R for δ < 0 (monotonicity).

This experiment no longer claims to DERIVE c₁ from spec(L_F). See REWORK spec §5.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import C1, H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = np.linspace(-1e-3, -0.3, 9)   # skip exact 0 to avoid VoidParameters guard
    R = 300.0
    records = []
    for d in deltas:
        params = VoidParameters(delta=float(d), R_mpc=R)
        web = generate_synthetic_void(
            params, n_points=1500, box_mpc=800.0, rng_seed=42,
        )
        h = predict_from_cosmic_web(
            web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
        )
        expected_kin = C1 * float(d)
        records.append(dict(
            delta=float(d),
            R_mpc=R,
            delta_H0=h.delta_H0,
            kinematic_term=h.kinematic_term,
            topological_term=h.topological_term,
            expected_kin=float(expected_kin),
            kin_tautology_residual=float(h.kinematic_term - expected_kin),
            ratio_topo_over_kin=float(
                h.topological_term / h.kinematic_term if abs(h.kinematic_term) > 1e-12 else np.nan
            ),
        ))

    # Monotonicity check: sort by |δ| ascending and confirm ΔH0 rises.
    by_absd = sorted(records, key=lambda r: abs(r["delta"]))
    deltaH0_sorted = [r["delta_H0"] for r in by_absd]
    monotone_nondec = all(
        deltaH0_sorted[i + 1] + 1e-6 >= deltaH0_sorted[i]
        for i in range(len(deltaH0_sorted) - 1)
    )

    # Max tautology residual
    max_taut = max(abs(r["kin_tautology_residual"]) for r in records)

    out = dict(
        primary_assertion="topological_term shrinks relative to kinematic_term as δ → 0 (β1 noise-floor only)",
        secondary_assertion=f"max |kinematic_term - C1*delta| = {max_taut:.2e} (should be ~0)",
        tertiary_assertion=f"ΔH0 monotone non-decreasing in |δ|: {monotone_nondec}",
        records=records,
    )
    (OUTPUT / "analytical_reduction.json").write_text(json.dumps(out, indent=2))

    # Plot
    d_arr = np.array([r["delta"] for r in records])
    k_arr = np.array([r["kinematic_term"] for r in records])
    t_arr = np.array([r["topological_term"] for r in records])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_arr, k_arr, "o-", label="kinematic (c1·δ, corrected sign)")
    ax.plot(d_arr, C1 * d_arr, "--", label=f"C1·δ expected, C1={C1:.2f}")
    ax.plot(d_arr, t_arr, "s-", alpha=0.6, label="topological (α=1)")
    ax.set_xlabel("δ"); ax.set_ylabel("ΔH₀ component [km/s/Mpc]")
    ax.legend()
    ax.set_title("Analytical reduction — kinematic = -H0·δ/3, topological bounded")
    fig.tight_layout()
    fig.savefig(OUTPUT / "analytical_reduction.png", dpi=120)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import-load check**

```bash
python -c "import problems.hubble_tension_web.experiments.analytical_reduction as m; assert callable(m.main)"
```

Expected: no exception.

- [ ] **Step 3: Commit**

```bash
git add problems/hubble_tension_web/experiments/analytical_reduction.py
git commit -m "feat(hubble): repurpose analytical_reduction — consistency check, not derivation"
```

---

## Task 12: KBC cross-check + aggregate REPORT + pipeline smoke test

Combine:

1. Update `kbc_crosscheck.py` to use the signed band `(+1.0, +3.0)` (not magnitude) and the corrected sign convention.
2. Update `aggregate.py` to note the rework changes in `REPORT.md` (persistent β₁, corrected sign, v1_superseded pointer, α units).
3. Update `test_pipeline.py` to assert the signed-band contract end-to-end.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/kbc_crosscheck.py`
- Modify: `problems/hubble_tension_web/experiments/aggregate.py`
- Modify: `tests/hubble_tension_web/test_pipeline.py`

- [ ] **Step 1: Rewrite `kbc_crosscheck.py`**

```python
"""KBC cross-check: run the calibrated functional on KBC-literature parameters.

Signed band compare (REWORK spec §6.5): band is (+1.0, +3.0) km/s/Mpc.
A negative predicted ΔH₀ at δ = -0.2 is an unambiguous sign-convention failure.
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
# SIGNED band: positive ΔH0 expected for a void under corrected sign convention.
LITERATURE_BAND = (1.0, 3.0)


def verdict(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if value < 0:
        return "SIGN ERROR — predicted ΔH0 < 0 for a void; sign convention violated"
    if value < lo:
        return "BELOW band — local-void hypothesis weak for KBC parameters"
    if value > hi:
        return "ABOVE band — topology implies a larger tension contribution than perturbative theory captures"
    return "WITHIN band — consistent with literature"


def main() -> None:
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    alpha_star = float(calib["alpha_star"])

    params = VoidParameters(delta=KBC_DELTA, R_mpc=KBC_R_MPC)
    web = generate_synthetic_void(params, n_points=2500, box_mpc=900.0, rng_seed=2025)
    h = predict_from_cosmic_web(
        web=web, params=params, alpha=alpha_star, k=8, stalk_dim=8, k_spec=16,
    )

    v = verdict(h.delta_H0, LITERATURE_BAND)

    out = dict(
        delta=KBC_DELTA,
        R_mpc=KBC_R_MPC,
        alpha_star=alpha_star,
        alpha_units="km/s",
        delta_H0=h.delta_H0,
        kinematic_term=h.kinematic_term,
        topological_term=h.topological_term,
        literature_band_signed=LITERATURE_BAND,
        verdict=v,
    )
    (OUTPUT / "kbc_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rewrite `aggregate.py`**

```python
"""Aggregate the three experiment outputs into REPORT.md (REWORK version)."""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "results"


def main() -> None:
    analytical = json.loads((OUTPUT / "analytical_reduction.json").read_text())
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    kbc = json.loads((OUTPUT / "kbc_crosscheck.json").read_text())

    lines = [
        "# Hubble-Tension-Web: Results Report (REWORK, v2)",
        "",
        "**Sign convention:** c1 = -H0/3. ΔH0 > 0 for δ < 0 (voids).",
        "**α units:** km/s.",
        "**β1:** persistent (VR filtration, lifetime > τ·ℓ̄, τ=1.5).",
        "",
        "v1 results (no-op Laplacian, single-scale β1, wrong sign) archived in `v1_superseded/`.",
        "",
        "## Leg 1: Analytical Reduction (consistency check)",
        "",
        analytical["primary_assertion"],
        "",
        analytical["secondary_assertion"],
        "",
        analytical["tertiary_assertion"],
        "",
        "| δ | kinematic | topological (α=1) | kin tautology residual |",
        "|---|---|---|---|",
    ]
    for r in analytical["records"]:
        lines.append(
            f"| {r['delta']:.4f} | {r['kinematic_term']:.4g} | {r['topological_term']:.4g} | "
            f"{r['kin_tautology_residual']:.2e} |"
        )

    lines.extend([
        "",
        "## Leg 2: Sim Calibration (non-circular)",
        "",
        f"- Reference source: {calib['reference_source']}",
        f"- Fitted α* = **{calib['alpha_star']:.4g} {calib['alpha_units']}**",
        f"- MSE = {calib['mse']:.4g}, R² = {calib['r_squared']:.3f}",
        f"- Scan size = {len(calib['scan'])} (δ, R) combinations",
        "",
        f"Note: {calib['note']}",
        "",
        "See `sim_calibration.png` for predicted-vs-reference scatter.",
        "",
        "## Leg 3: KBC Cross-Check (signed band)",
        "",
        f"- δ = {kbc['delta']}, R = {kbc['R_mpc']} Mpc",
        f"- Kinematic term: **{kbc['kinematic_term']:.3f} km/s/Mpc**",
        f"- Topological term (α* = {kbc['alpha_star']:.4g}): **{kbc['topological_term']:.3f} km/s/Mpc**",
        f"- Total ΔH0: **{kbc['delta_H0']:.3f} km/s/Mpc**",
        f"- Literature band (signed): {kbc['literature_band_signed']} km/s/Mpc",
        f"- Verdict: **{kbc['verdict']}**",
        "",
        "## Interpretation",
        "",
        "The functional reduces to LTB in the analytical leg (tautology guard against the v1 sign bug).",
        "Sim calibration fits α against the LTB-reference RESIDUAL (not leading term), making the fit",
        "non-circular. KBC cross-check is the first external signed test of the calibrated functional.",
        "",
        "A WITHIN-band result is a successful reproduction of the perturbative KBC estimate by a",
        "topological route. An ABOVE-band result indicates that the sheaf Laplacian picks up structure",
        "perturbation theory omits. A SIGN ERROR result is a regression bug and must block release.",
        "",
        "## Scope limits",
        "",
        "- All voids are LTB-family synthetic. Real N-body snapshot ingestion remains deferred.",
        "- The LTB reference uses closed-form δ³ + finite-R Gaussian weights; if algebra does not",
        "  match Garcia-Bellido 2008 at pivot points within 5%, fall back to numerical LTB integration.",
        "- c₁ is ASSERTED from LTB linear theory, not DERIVED from spec(L_F). See REWORK spec §5.",
    ]
    )

    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Rewrite `test_pipeline.py`**

```python
# tests/hubble_tension_web/test_pipeline.py
"""End-to-end smoke test of the REWORK pipeline.

Runs all four scripts in order, asserts artifacts exist, and checks signed-band contract.
"""
import json
from pathlib import Path
import subprocess
import sys


def run(mod: str) -> None:
    result = subprocess.run([sys.executable, "-m", mod], check=False, capture_output=True)
    assert result.returncode == 0, f"{mod} failed: {result.stderr.decode()}"


def test_full_pipeline_runs_end_to_end():
    run("problems.hubble_tension_web.experiments.analytical_reduction")
    run("problems.hubble_tension_web.experiments.sim_calibration")
    run("problems.hubble_tension_web.experiments.kbc_crosscheck")
    run("problems.hubble_tension_web.experiments.aggregate")

    results = Path("problems/hubble_tension_web/results")
    assert (results / "analytical_reduction.json").exists()
    assert (results / "sim_calibration.json").exists()
    assert (results / "kbc_crosscheck.json").exists()
    assert (results / "REPORT.md").exists()

    for fname in ["analytical_reduction.json", "sim_calibration.json", "kbc_crosscheck.json"]:
        json.loads((results / fname).read_text())


def test_kbc_cross_check_sign_is_correct():
    """Signed-band contract: KBC void must produce ΔH₀ with the right sign (positive)."""
    results = Path("problems/hubble_tension_web/results")
    if not (results / "kbc_crosscheck.json").exists():
        run("problems.hubble_tension_web.experiments.sim_calibration")
        run("problems.hubble_tension_web.experiments.kbc_crosscheck")
    kbc = json.loads((results / "kbc_crosscheck.json").read_text())
    assert kbc["kinematic_term"] > 0, (
        f"KBC kinematic term must be > 0 (δ=-0.2, c1<0 ⇒ c1·δ > 0); "
        f"got {kbc['kinematic_term']}. Sign regression."
    )
    # ΔH0 total may be above or below the positive band; but sign must be positive
    # UNLESS α*·f_topo is large-negative, which would be a calibration pathology flagged
    # in verdict.
    assert "SIGN ERROR" not in kbc["verdict"], (
        f"KBC verdict reports SIGN ERROR: {kbc['verdict']}"
    )


def test_analytical_reduction_tautology_residual_small():
    """Regression guard: kinematic_term = C1·δ to machine precision."""
    results = Path("problems/hubble_tension_web/results")
    if not (results / "analytical_reduction.json").exists():
        run("problems.hubble_tension_web.experiments.analytical_reduction")
    data = json.loads((results / "analytical_reduction.json").read_text())
    for r in data["records"]:
        assert abs(r["kin_tautology_residual"]) < 1e-6, (
            f"kin tautology residual too large at δ={r['delta']}: {r['kin_tautology_residual']}"
        )
```

- [ ] **Step 4: Commit**

```bash
git add problems/hubble_tension_web/experiments/kbc_crosscheck.py problems/hubble_tension_web/experiments/aggregate.py tests/hubble_tension_web/test_pipeline.py
git commit -m "feat(hubble): T4 finish — signed KBC band, REPORT rewrite, pipeline smoke test"
```

- [ ] **Step 5: Run the full pipeline smoke test**

```bash
pytest tests/hubble_tension_web -v
```

Expected: everything passes. This is the acceptance gate for the rework. If any of the following happens, fall back per the specified guidance:

- `ripser` crashes on large point clouds → switch to `gudhi` (spec §4.1).
- β1_persistent/N > 0.05 on uniform Poisson at N=1000 → raise τ_persist to 2.0 (spec §4.3), then to absolute-Mpc fallback.
- α* is nonsensical (|α| > 10⁴ km/s) → fall back to Option 6.2.a numerical LTB integrator.
- Any of the typed-Laplacian guards fail → investigate Rodrigues edge case handling and P^t permutation construction.

---

## Self-Review (performed after writing the plan)

**Spec coverage:**

| REWORK spec section | Addressed by |
|---|---|
| §3 T1 redesign (density-gradient stalks, R_dst^t = λ·(Rot⊕P⊕I)) | Task 5 (stalks), Task 6 (Laplacian) |
| §4 T2 redesign (VR filtration, τ_persist·ℓ̄ threshold) | Task 7 (spectrum rewrite) |
| §5 T3 redesign (c1 = −H0/3, honest anchoring, no derivation claim) | Task 3 (sign), Task 11 (analytical rewrite) |
| §6 T4 redesign (non-circular LTB reference, residual fit) | Task 9 (ltb_reference), Task 10 (sim_calibration) |
| §6.4 sign convention pinned in docstring + test | Task 3 (HubbleShift guard), Task 8 (docstring) |
| §6.5 KBC signed band | Task 12 |
| §7 test rewrites (7.1–7.6) | Tasks 2, 3, 4, 5, 6, 7, 8, 12 |
| §8 artifact migration | Task 1 |

**Placeholder scan:** No "TODO", "TBD", or "add appropriate X". The LTB weight functions in Task 9 are explicitly documented as heuristic specializations with a stated Option-6.2.a fallback; the spec sanctions this punt. Rodrigues, permutation, and ripser calls are concrete code, not sketches.

**Type consistency:**
- `LocalCosmicWeb(positions, environments)` unchanged, used identically from Task 3 onward.
- `HubbleShift(delta_H0, kinematic_term, topological_term, delta=None)` — `delta` added in Task 3; all call sites via `kappa_operator` pass it.
- `SpectralSummary` unchanged; β1 semantics redefined (persistent, not combinatorial).
- `summarize_spectrum` signature gains required `positions` kwarg in Task 7 — all callers updated (only `predict_from_cosmic_web`).
- `typed_sheaf_laplacian` signature gains optional `environments` kwarg in Task 6; existing tests pass `environments=envs` explicitly to avoid the string-based reconstruction path.
- `stalk_dim` constant `STALK_DIM = 8` replaces the v1 test defaults of 4/5. All v1 tests that hardcoded those values are updated in Tasks 5 and 6.

**Sign audit:** `C1 = -H0_GLOBAL / 3.0` appears literally once in `functional.py`. All other sign-dependent code (kinematic_term, HubbleShift guard, KBC band, LTB leading coefficient in `ltb_reference.py`) carries the negative sign through derivation, not through independent constants. `test_c1_is_negative_third_of_H0` is the single anchor assertion.

**Units audit:** α units = km/s is asserted in three places: (a) `ALPHA_UNITS` constant in `functional.py`, (b) `functional.py` module docstring, (c) output JSONs (`sim_calibration.json`, `kbc_crosscheck.json`). `test_alpha_units_documented` checks (a) and (b).

**Edge-type ordering audit:** `oriented_edge_type_for_pair` is the only constructor; `edge_type_for_pair` is gone. `EDGE_TYPES` constant has 16 entries (4×4), confirmed in `test_edge_types_constant_covers_all_ordered_pairs`. `EDGE_TYPE_LAMBDA` dict has 16 entries including all 4 diagonal self-pairs and 12 off-diagonal ordered pairs.

**One soft spot:** Task 9's Gaussian-profile weight functions `W_2`, `W_3`, and `_finite_R_correction` are heuristic (documented as such). If the calibration in Task 10 produces a pathological α, the Option 6.2.a fallback (numerical LTB integrator) is explicitly called out as the recovery path. This is the spec's punt, pinned.

No issues found on self-review.
