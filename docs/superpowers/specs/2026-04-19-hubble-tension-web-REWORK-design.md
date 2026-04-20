# Hubble Tension Web — REWORK Design Spec

**Date:** 2026-04-19
**Status:** Rework spec. Approved path = Rework (not Reframe). Supersedes `2026-04-19-hubble-tension-web-design.md` for the four redesign targets.
**Branch:** `feat/hubble-tension-web`
**Author:** Blake (with Claude) after opus review
**Scope:** Specification only. No code changes in `problems/` or `tests/`. No experiment execution. Implementation plan is a downstream task.

---

## 1. Why we're reworking

The opus review surfaced five findings that, taken together, break the credibility chain of the original design:

1. **The typed sheaf Laplacian is a no-op.** Every edge type uses the same orthogonal `R` with `-R` on source and `+R` on destination, so `L_F ≡ L_graph ⊗ I_stalk_dim` exactly. Spectrum cannot respond to typing. The "typed" Laplacian is a misnomer.
2. **β₁ is not persistent.** `spectrum.py` computes β₁ = `|E| − |V| + β₀` at a single scale on a fixed k-NN graph. No filtration; no lifetime threshold. The reported β₁ = 5607 at δ = 0 is first-homology of the k-NN graph at one ε, not a topological invariant of the underlying point cloud.
3. **c₁ = H₀/3 is hardcoded.** The functional claims to "reduce to LTB" but the reduction is tautological: the LTB coefficient is asserted, not derived from `spec(L_F)`.
4. **Sim calibration is circular.** The reference curve `(H₀/3)·δ·exp(-((R-300)/200)²)` has `(H₀/3)·δ` as its leading term — i.e., the kinematic term we are supposedly calibrating against *is already the reference*. α is fit against residuals of a curve whose closed form embeds the answer.
5. **The sign is wrong.** `c1 = +H₀/3` implies ΔH₀ < 0 for δ < 0 (voids). But the observed tension has *local* H₀ > *global* H₀, i.e., a void should bias the locally-inferred rate *upward*: ΔH₀ > 0 for δ < 0. The LTB leading expansion is H_local ≈ H₀(1 − δ/3), so ΔH₀ = −H₀·δ/3. Current code has the opposite sign.

Tiers (A), (B), (C) of the original design are affected: (A) the machinery is hollow (T1); the spectral pipeline is single-scale (T2); (C) the functional is self-referential (T3, T4, sign). The rework addresses each.

## 2. Design principles for the rework

Invariants the rework MUST enforce:

- **I1 (sensitivity to typing).** The spectrum of `L_F` materially changes when edge-type assignments change. Collapsing all environments to a single type produces a spectrum whose bulk and tail both differ from the typed case by more than numerical tolerance.
- **I2 (smooth-limit topology).** β₁_persistent(cloud) → 0 (or O(1) noise floor set by finite sampling) as the point cloud approaches a uniform/homogeneous draw of increasing density. Single-scale β₁ is banned; only filtration-derived, lifetime-thresholded β₁ enters the functional.
- **I3 (LTB anchoring, honest).** c₁ is anchored to the LTB leading coefficient. Whether it is *derived* from the continuum limit of `L_F` or *asserted* from Einstein-equation integration is settled in T3 (we punt to asserted — see §5).
- **I4 (non-circular calibration).** The target curve against which α is fit does not contain `c₁·δ` as a closed-form leading term drawn from the same ansatz as the functional.
- **I5 (sign).** ΔH₀ > 0 for δ < 0 at leading order. The coefficient in front of δ in the kinematic term is **negative**. The spec and code must agree.
- **I6 (PSD and symmetry of L_F).** `L_F = δ^T δ` for some real coboundary δ — this makes symmetry and PSD automatic, regardless of restriction-map choice.
- **I7 (dimensional consistency).** The functional's output has units of velocity/length (km/s/Mpc). Each term must carry those units after its prefactor; α's units must be stated and verified.

---

## 3. T1 redesign — typed sheaf Laplacian with R_src ≠ R_dst

### 3.1 Choice of ansatz

**We pick the density-gradient-stalk ansatz** (candidate A), not the type-asymmetric non-orthogonal matrix ansatz (candidate B). Rationale: (a) the review explicitly prefers physical meaning, (b) candidate B is opaque about why type-pair `M^t` should take any particular value, (c) the gradient-stalk construction gives a direct interpretation of the spectrum as a measure of how smoothly local density-gradient directions glue across environment transitions, which is exactly the cohomological story the ATFT framing promises.

### 3.2 Stalks

At each node `v` (a galaxy, halo, or grid-cell — implementation decides, spec leaves open), the stalk is a real vector space `F_v = ℝ^d` with `d = stalk_dim` (default 6, rationale below). Its distinguished content:

- Coordinates 1–3: `ĝ_v`, the unit vector of the local density gradient at `v`, computed as follows:
  - Local density estimate `ρ_v` = kernel-density estimate on the point cloud using KDTree neighbors within a fixed physical radius `h` (scale parameter; plan picks value, default `h = 2·(mean k-NN distance)`).
  - Gradient `∇ρ_v` via finite differences against the same KDTree neighbors: `∇ρ_v ≈ Σ_w∈N(v) (ρ_w − ρ_v)·(x_w − x_v)/|x_w − x_v|²`. Normalize to get `ĝ_v = ∇ρ_v / |∇ρ_v|`. If `|∇ρ_v|` is below a numerical floor, set `ĝ_v` to a conventional axis (`ê_z`) and flag the node as "gradient-degenerate" for downstream diagnostics.
- Coordinates 4–7: environment one-hot, length 4 (void / wall / filament / node). (Total d = 7. We round up to d = 8 for alignment convenience — trailing coordinate fixed at 0, gives room for later extension.)

So `stalk_dim` defaults to 8: 3 (gradient) + 4 (env one-hot) + 1 (pad). `d` is small, dense eigvalsh stays tractable for `N·d` up to ~20k.

### 3.3 Restriction maps on each oriented edge

For an undirected edge `{u, v}` we choose a canonical orientation `u → v` (e.g., by smaller index first). Edge type `t = (env_u, env_v)` (now an **ordered** pair, not canonicalized — asymmetry is the point).

For each oriented edge:

- **`R_src^t` = I_d`** (identity on source stalk). Fixes a reference frame.
- **`R_dst^t` = λ^t · Rot(ĝ_u → ĝ_v) ⊕ P^t`** (block-diagonal):
  - `Rot(ĝ_u → ĝ_v)` acts on coordinates 1–3 as the Rodrigues rotation sending `ĝ_u` to `ĝ_v`. If `ĝ_u = ĝ_v` it is `I_3`; if anti-parallel it is any `π`-rotation in the perpendicular plane (deterministic tie-break: pick `ê_z × ĝ_u` if defined, else `ê_x × ĝ_u`).
  - `P^t` acts on coordinates 4–8 as the permutation that maps `env_u` one-hot to `env_v` one-hot (plus identity on pad). Unique up to the 1-cycle fixing the destination env; we use the unique permutation that swaps only `env_u` ↔ `env_v` in the 4-element block, identity elsewhere, and identity on the pad. For `env_u = env_v`, `P^t = I`.
  - `λ^t ∈ ℝ_+` is the **type-specific prefactor**, encoding how "costly" the environment transition is for the sheaf. Proposed default (plan picks exact numbers):
    - `λ^{void-void} = λ^{wall-wall} = λ^{filament-filament} = λ^{node-node} = 1.0`
    - `λ^{void-wall} = λ^{wall-void} = 1.5`
    - `λ^{void-filament} = λ^{filament-void} = 1.8`
    - `λ^{void-node} = λ^{node-void} = 2.2`
    - `λ^{wall-filament} = λ^{filament-wall} = 1.2`
    - `λ^{wall-node} = λ^{node-wall} = 1.5`
    - `λ^{filament-node} = λ^{node-filament} = 1.1`
  - These numbers are ordinal — strength of expected gluing obstruction across the transition. Not tuned parameters. Pinned in `functional.py` as a module constant `EDGE_TYPE_LAMBDA`, documented as "ordinal physical prior, not calibrated."

### 3.4 Coboundary and Laplacian

The oriented coboundary `δ : ⊕_v F_v → ⊕_e F_e` (where each `F_e ≅ ℝ^d`):

`(δ σ)_e = R_dst^{t(e)} σ_{dst(e)} − R_src^{t(e)} σ_{src(e)}`

i.e., the edge-block row at edge `e = u → v`:

- `col(src=u)`: `−R_src^t = −I_d`
- `col(dst=v)`: `+R_dst^t = +λ^t · (Rot ⊕ P^t ⊕ I_pad)`

Then `L_F := δ^T δ`, symmetrized numerically.

### 3.5 PSD proof sketch

`L_F = δ^T δ` for real `δ`. For any `x`, `x^T L_F x = ||δ x||² ≥ 0`. Symmetric by `(δ^T δ)^T = δ^T δ`. Done. No assumption on `R_src^t`, `R_dst^t` is required; they may be non-orthogonal (and `R_dst^t` is non-orthogonal here when `λ^t ≠ 1`).

### 3.6 Demonstration that typed and untyped cases differ

Consider the "untyped collapse" map: set `λ^t ≡ 1`, `Rot(·→·) ≡ I`, `P^t ≡ I` for all t. Then `R_src^t = R_dst^t = I`, and `L_F = (L_graph ⊗ I_d)`. The nontrivial spectrum of `L_F` is the spectrum of `L_graph` repeated `d` times; the nullity equals `d · β₀`.

Under the typed construction with at least one edge having `λ^t ≠ 1` or non-identity `Rot` or non-identity `P^t`, the edge block `+λ^t·(Rot ⊕ P^t)` differs from `+I_d`. The resulting `L_F` is not a Kronecker product: it is a block matrix whose off-diagonal `d×d` blocks (between node stalks) carry the per-edge-type signatures.

A cleanly checkable corollary: **rank of the nullspace of L_F.** In the untyped case, nullspace dim = `d · β₀` (any constant global section lifts). In the typed case with nontrivial `Rot` or `P^t`, the nullspace shrinks because a constant section `σ_v ≡ σ₀` must satisfy `(λ^t · M^t − I)σ₀ = 0` for every edge, which generically forces `σ₀ = 0` in the rotated/permuted subspaces. The nullity drops from `d·β₀` to `β₀` (only the pad coordinate remains globally flat) or lower. This rank change is the acceptance criterion: **I1 is verified by checking `nullity(L_F_typed) < nullity(L_F_untyped)` on any web with at least two environment types.**

A second check — **bulk spectrum shift:** the median eigenvalue of `L_F_typed` differs from the median of `L_F_untyped` by more than `1e-3` relative tolerance on a web of at least 50 nodes with mixed environments. This catches the case where only nullity changes.

### 3.7 Numerical safety

- Rodrigues rotation falls back to `I_3` when `|ĝ_u − ĝ_v| < 1e-9`.
- Near-antipodal case (`ĝ_u·ĝ_v < -1 + 1e-9`): deterministic perpendicular axis choice as in §3.3.
- Gradient-degenerate nodes (very low `|∇ρ_v|`): stalk coords 1–3 default to `ê_z`; flagged in a diagnostic dict emitted alongside `L_F`.

---

## 4. T2 redesign — persistent β₁ via filtration

### 4.1 Filtration structure

Vietoris–Rips complex over the Euclidean distance on node positions, restricted to the 1-skeleton and 2-skeleton (we need H₁, so we only need up through 2-simplices). Filtration parameter `ε` runs from 0 to a cap `ε_max`. We compute the barcode `B₁ = {(birth_i, death_i)}` of 1-dimensional persistence.

Implementation: use `gudhi` (preferred) or `ripser` (fallback). Both are mature, MIT/GPLv3 respectively. `ripser` is faster for pure H₁; `gudhi` integrates more cleanly with later typed-filtration extensions. Plan picks; default recommendation: `ripser` (lighter dependency, faster on 3D point clouds of size ~1500–5000).

### 4.2 Distance scale parameterization

`ε_max = τ_max · ℓ̄`, where `ℓ̄ = mean k-NN edge length` on the same k used in the typed graph construction, and `τ_max ∈ [4, 8]` (default 6). This ensures we capture loops whose characteristic size is up to a few mean spacings, which is where the cosmic web's persistent structures live; larger ε washes out into one big connected blob anyway.

### 4.3 Persistence threshold rule

β₁_persistent := `|{ i : death_i − birth_i > τ_persist · ℓ̄ }|`

with `τ_persist ∈ [1.0, 2.0]` (default 1.5). The threshold is a **multiple of the mean edge length**, not absolute Mpc — this makes it scale-adaptive across webs of different sampling density.

Rationale for `τ_persist > 1`: a Poisson point cloud at density ρ produces loops with birth ~ ρ^{-1/3} and death within a small O(1) factor of birth; lifetimes are O(ρ^{-1/3}). With `τ_persist > 1`, those random loops are filtered out; what remains is structural (the actual topology of the web). Smaller `τ_persist` includes noise; larger loses signal. Plan pins after sensitivity scan.

### 4.4 Smooth-limit behavior — back-of-envelope

For uniform-Poisson points at mean density ρ in ℝ³:

- Typical nearest-neighbor distance: `ℓ̄ ≈ (3/(4π ρ))^{1/3} ≈ 0.554 · ρ^{-1/3}`.
- In VR persistence of a homogeneous Poisson cloud, the expected number of persistent H₁ classes with lifetime > c · ℓ̄ decays exponentially in c for c > 1 (standard stochastic topology result; see e.g. Kahle 2011, Bobrowski–Kahle 2018). Concretely, `E[β₁_persistent(c)] ~ N · exp(−α(c))` with α(c) > 0 and growing in c.
- With our default `τ_persist = 1.5` and `N ∼ 1500`, `E[β₁_persistent]` is O(1) — a small noise floor, not the 5607 currently reported.
- In the strict smooth limit (`N → ∞`, Poisson): β₁_persistent / N → 0.

Therefore **I2 holds**: β₁_persistent vanishes (modulo an O(1) finite-sample noise floor) under homogenization. This is the key property that was missing.

### 4.5 β₁ as a function of (δ, R)

A void of depth δ and radius R creates a shell of density contrast at |x| ≈ R. Galaxies form a thin topologically nontrivial ring-like structure around that shell. β₁_persistent picks up these shell-scale loops whose lifetime exceeds `τ_persist · ℓ̄`. Qualitative scaling:

- Fixed R, deepening |δ|: sharper shell → fewer/cleaner persistent loops around the shell.
- Fixed δ, growing R: more mean-spacing-scale loops along the shell.

Plan emits a scan `β₁_persistent(δ, R)` surface as a diagnostic; it is the primary topological input to the functional (along with the spectrum).

### 4.6 What the functional consumes from T2

The functional signature stays `𝒦(spec(L_F), β₁, δ, R)`, but β₁ here means `β₁_persistent` (threshold τ_persist applied). β₀ stays as graph-connected-component count (β₀ via persistence gives the same answer at `ε > ε_connect`, no change in meaning). `spec(L_F)` is consumed via `λ_min` (spectral gap) and optionally the top-k low eigenvalues.

---

## 5. T3 redesign — c₁: honest punt

### 5.1 Decision: do NOT claim derivation from L_F's continuum limit

We explored two paths:

**Path A (claimed in original spec):** derive `c₁ = −H₀/3` from the continuum limit of `L_F` applied to a smooth density δ(x). The argument would run: (i) trivial section λ₀ = 0 corresponds to the global homogeneous mode; (ii) next mode carries δ-dependence with eigenvalue ~ |∇ρ|²/ρ² or similar; (iii) evaluated on a void profile, its contribution to 𝒦 is `c₁·δ + O(δ²)` with c₁ = −H₀/3.

**Why Path A fails as a derivation:** the Laplacian (sheaf or otherwise) is a second-order operator on a stalk bundle. The LTB coefficient −H₀/3 comes from integrating the Einstein field equations with an FLRW-perturbed metric and matching local comoving observer to global CMB rest frame. The Laplacian's continuum limit gives a *gradient structure* on the density field; it does not carry the Einstein-equation information about how density-contrast couples to the metric scale factor. Any "derivation" we write would secretly import −H₀/3 at a later step, which is exactly the circularity the review criticizes.

**Path B (honest path, chosen):** state plainly:

> c₁ is the LTB leading-order coefficient, anchored to Einstein-equation-derived cosmological perturbation theory, not derived from `spec(L_F)`. The typed sheaf functional's contribution is the **topological correction** on top of the LTB kinematic baseline. What the functional must *verify* — and this is the content of a replacement analytical experiment — is that its **topological term vanishes** in the smooth limit (β₁_persistent → 0, δ → 0, R → ∞), leaving the kinematic term alone to match LTB.

### 5.2 Sign

**c₁ = −H₀/3.** Derivation from LTB linear order: inside a homogeneous under-density of contrast δ < 0, mass conservation requires the local comoving volume to expand faster than the global average to conserve matter; the leading-order correction to the local Hubble rate is

H_local(t) = H₀(t) · (1 − δ/3) + O(δ²)

so

ΔH₀ := H_local − H_global = −(H₀/3) · δ + O(δ²)

For δ < 0 (a void), ΔH₀ > 0 (locally measured H₀ is *larger* than globally inferred), matching the observed tension direction.

**Current code has c₁ = +H₀/3. This is a sign bug and MUST be corrected in the rework.** The HubbleShift type invariant and all experiment expectations flip.

### 5.3 What replaces the current analytical_reduction experiment

New `experiments/analytical_reduction.py` intent (spec-level; plan/implementation fills in):

1. Scan δ ∈ [0, −0.3], R ∈ [150, 600] Mpc.
2. For each (δ, R), generate synthetic void, compute full functional including topology.
3. **Primary assertion:** `topological_term / kinematic_term → 0` as `β₁_persistent → 0` in a controlled subset where the void is well-sampled and clean (low-noise region of parameter space).
4. **Secondary assertion:** kinematic_term equals `−(H₀/3)·δ` to machine precision (this is a tautology-check that we didn't reintroduce the sign bug).
5. **Tertiary assertion:** at R → ∞, stalk_dim fixed, β₁_persistent → O(1) noise floor, so `topological_term → O(α/R) → 0`.

This experiment no longer claims derivation. It claims **consistency**: the functional reduces to LTB kinematic in the smooth limit, and the topological term vanishes in that limit. That's the honest content.

### 5.4 α's meaning and units

After the sign correction:

ΔH₀ = c₁·δ + α · f_topo(β₀, β₁, λ_min, R)

with c₁ = −H₀/3 (km/s/Mpc, since H₀ is in km/s/Mpc and δ is dimensionless).

Current `f_topo = (β₁/β₀) · (1/λ_min) · (1/R)`:

- β₁/β₀ dimensionless.
- λ_min has units of (stalk field)²/(length²)·(length²) or similar — since stalks are dimensionless real vectors, λ_min has units of (stalk vector magnitude)². In our construction with unit-vector gradients and unit one-hots, stalk magnitude is O(1) dimensionless, so **λ_min is dimensionless**.
- 1/R has units of 1/length = 1/Mpc.

So `f_topo` has units of 1/Mpc. To get ΔH₀ in km/s/Mpc we need α in units of **km/s**.

**The spec pins α's units explicitly at km/s.** Any calibration output MUST report α in km/s, and dimensional checks are added to tests.

---

## 6. T4 redesign — non-circular calibration + sign

### 6.1 Choice: analytical full-LTB comparison

We pick **option (b), full LTB solution comparison**, not N-body ingestion (heavy engineering, out of scope for this rework) and not group-theoretic isopycnic invariance (elegant but difficult to make concrete without substantially new math work).

**Why non-circular:** The full LTB solution for a given density profile `ρ(r)` is the solution of the Einstein equations plus continuity plus Bianchi identity. Its leading expansion in δ is `−H₀·δ/3`, which matches our kinematic term *by construction*. Its **full** form — including nonlinear δ, finite-R curvature, shell-crossing thresholds — is an independent function of (δ, R) that is NOT drawn from our ansatz. Fitting α against the residual (`ΔH₀_LTB_full(δ, R) − c₁·δ`) gives genuine signal.

### 6.2 Calibration target specification

Let `ΔH₀_LTB(δ, R; profile)` be the full LTB prediction for a standard void density profile (plan picks: top-hat, Gaussian `exp(-r²/R²)`, or Maxwell–Boltzmann-like; recommend Gaussian for smoothness). We compute this either:

- **Option 6.2.a:** from a published LTB solver (e.g., the numerical integrator in Garcia-Bellido & Haugbølle 2008, or the implementation in van Putten 2017 — plan identifies a reference implementation to port or use).
- **Option 6.2.b:** from a carefully re-derived series expansion to order δ³ with finite-R corrections. Order-δ³ is sufficient for the parameter range |δ| < 0.3 of interest; higher orders are cross-checked by option 6.2.a at a few pivot points.

Either way, the target curve `ΔH₀_LTB(δ, R)` is produced **before the calibration step**, in its own module `problems/hubble_tension_web/ltb_reference.py`, and has **no functional dependence on `L_F`, `β₁`, `λ_min`, or α**. This breaks the circularity.

### 6.3 Calibration procedure

Per (δ, R) sample:

1. Compute `ΔH₀_LTB(δ, R)` (independent).
2. Compute `c₁·δ` (kinematic, with corrected sign).
3. Residual `y(δ, R) = ΔH₀_LTB(δ, R) − c₁·δ`. This residual is the **nonlinear LTB correction** at that parameter.
4. Compute `f_topo(δ, R)` from the functional (requires `β₁_persistent`, `λ_min`, `β₀`, `R`).
5. Fit α by least squares: `α* = argmin Σ_{(δ,R)} (α·f_topo − y)²`.

If `α*` is small and `f_topo`-vs-y correlation is weak, the honest conclusion is that the topological term does NOT capture the nonlinear LTB residual, i.e., our hypothesis that β₁ explains the tension is weak. That's a *finding*, not a failure.

### 6.4 Sign convention — explicit statement

Pinned in README, functional.py docstring, and tests:

> **Sign convention.** For a local under-density (δ < 0), the locally-measured Hubble rate exceeds the globally-inferred rate. We define ΔH₀ := H_local − H_global, so **ΔH₀ > 0 when δ < 0**. The kinematic coefficient is c₁ = −H₀/3.

`HubbleShift.__post_init__` gains a sanity check: when `delta < 0` and `delta_H0 < 0` and `topological_term ≈ 0`, raise (sign bug regression guard). When `delta == 0` exactly, no sign expectation.

### 6.5 KBC cross-check after the sign fix

The current KBC cross-check magnitude-compares to a literature band `(1.0, 3.0)` km/s/Mpc. After the sign fix, the cross-check compares the **signed** value: the band becomes `(+1.0, +3.0)` (positive), and a negative predicted ΔH₀ at δ = −0.2 is an unambiguous failure rather than a magnitude near-miss.

---

## 7. Tests to rewrite

### 7.1 `tests/hubble_tension_web/test_laplacian.py`

- **Delete:** nothing unconditionally; keep PSD/symmetry test (still valid — `L = δ^T δ`).
- **Rewrite:** `test_laplacian_dimension_is_n_times_stalk_dim` — still valid shape-wise but stalk construction now requires positions and environments to produce meaningful gradient stalks. Update fixture.
- **Add:** `test_typed_vs_untyped_spectrum_differs` — build a web with two environment types and mixed edges; verify median eigenvalue of typed `L_F` differs from untyped by > 1e-3 relative. (Invariant I1.)
- **Add:** `test_nullity_drops_under_typing` — verify `nullity(L_F_typed) < d · β₀` when typing is nontrivial (rank check via small-eigenvalue count with tolerance).
- **Add:** `test_gradient_stalk_construction_is_unit_and_deterministic` — build a web twice with same seed; verify stalks match exactly. Verify gradient-coords have unit norm (except for gradient-degenerate flagged nodes).

### 7.2 `tests/hubble_tension_web/test_spectrum.py`

- **Delete:** `test_summarize_spectrum_returns_spectral_summary`'s β₁ expectation needs update — β₁ is now persistent, so the non-zero bound stays but value semantics change. Rewrite.
- **Add:** `test_beta1_persistent_small_on_homogeneous_cloud` — uniform Poisson cloud of N = 1000, verify `β₁_persistent / N < 0.05` (noise floor). Fail otherwise.
- **Add:** `test_beta1_persistent_nonzero_on_ring_cloud` — sample points on a torus or a ring with noise, verify `β₁_persistent ≥ 1`. Sanity check that the filtration picks up real loops.
- **Keep:** `test_two_disconnected_clusters_give_beta0_at_least_two` — β₀ still from connected components.

### 7.3 `tests/hubble_tension_web/test_functional.py`

- **Rewrite:** all sign expectations flip. Add a dedicated `test_sign_convention` that builds a void with δ = −0.1, runs through `predict_from_cosmic_web` with α = 0 (kinematic only), asserts `delta_H0 > 0`. Regression guard against the sign bug ever returning.
- **Add:** `test_alpha_units_documented` — inspect module-level constants or docstrings; assert α's unit string is present and equals "km/s".
- **Rewrite:** any test asserting `kinematic_term == (H₀/3)·δ` → now `== −(H₀/3)·δ`.

### 7.4 `tests/hubble_tension_web/test_graph.py`

- Edge-type representation changes from canonical-unordered to ordered pair (so `void-wall ≠ wall-void`). Update `edge_type_for_pair` test to match (if it enforces canonicalization) or rename function to `oriented_edge_type_for_pair`.

### 7.5 `tests/hubble_tension_web/test_pipeline.py`

- End-to-end signs flipped. KBC expected within positive band.

### 7.6 `tests/hubble_tension_web/test_synthetic.py`, `test_types.py`

- Types: add sign-regression guard in `HubbleShift.__post_init__`. Test the new guard.
- Synthetic: no change required unless generator needs tweaks for gradient estimation to be well-defined (some configurations may need density-gradient-degeneracy avoidance — implementation to flag).

---

## 8. Migration of existing artifacts

`problems/hubble_tension_web/results/` currently contains:
- `analytical_reduction.json`, `analytical_reduction.png`
- `sim_calibration.json`, `sim_calibration.png`
- `kbc_crosscheck.json`
- `REPORT.md`

All are computed with:
- No-op Laplacian (T1)
- Single-scale β₁ = 5607 (T2)
- Wrong-sign c₁ (§5.2)
- Circular calibration (T4)

**Decision: move to `problems/hubble_tension_web/results/v1_superseded/`** (not delete).

Rationale: the numbers themselves are wrong, but the shape of the outputs (file layout, REPORT.md format, plot style) is useful reference for the v2 implementation. Keeping them under `v1_superseded/` makes the history legible and provides a quick "look how much the sign fix changes things" diff for the implementation session.

Add a short `v1_superseded/README.md` explaining why they're superseded (link to this rework spec).

---

## 9. Non-goals

Carried from original design:
- **No running on real observational galaxy catalogs.**
- **No new N-body simulation runs.**
- **No replacement cosmology.**

New or clarified:
- **No N-body public-snapshot ingestion in this rework.** IllustrisTNG / MDPL2 was listed as a T4 calibration option; we chose LTB-analytical instead. A sequel project may revisit.
- **No derivation of c₁ from the sheaf Laplacian's continuum limit.** Explicitly punted (§5).
- **No generalization of the typed Laplacian beyond (env_src, env_dst) pairs** (e.g., multi-scale or redshift-shell typing). Deferred.
- **No learning-based calibration.** α is fit by closed-form least squares against the LTB reference; no neural net or GP regression in this rework.
- **No implementation decisions on persistent-homology library choice** (`ripser` vs. `gudhi`). Plan picks.

---

## 10. Effort estimate

Implementer-task-units (one unit ≈ a focused half-day with tests):

- **T1 (typed sheaf Laplacian with gradient stalks):** 3 units. Non-trivial because gradient estimation, Rodrigues edge cases, permutation construction, and new test coverage all land together.
- **T2 (persistent β₁ via VR filtration):** 2 units. Library integration (ripser/gudhi), threshold scanning, smooth-limit test.
- **T3 (c₁ sign fix + analytical_reduction rewrite):** 1 unit. Mostly a sign flip + test rewrite + updated docstring; the "rewrite experiment" is small because the new experiment is a consistency check, not a derivation.
- **T4 (LTB reference module + non-circular calibration):** 3 units. Series-expansion LTB solver (or port), calibration rewrite, KBC cross-check update.
- **Test refactor (7.1–7.6):** 1 unit on top of T1–T4 (the specific new tests above).
- **Artifact migration + REPORT rewrite:** 0.5 unit.

**Total: 10.5 implementer-task-units.** Roughly a week of focused work, or two weeks at part-time cadence.

### Dependencies added

- `ripser` (preferred) or `gudhi` — for VR persistent H₁. Lightweight, pip-installable. `ripser` has no heavy transitive deps; `gudhi` is larger but more featureful.
- Optional: a small vendored LTB series-expansion module (pure numpy, no new external dep required for Option 6.2.b).

### Physical consistency checks missed in the original design

Flagged here so the rework tests cover them:

- **Dimensional analysis** (§5.4): `α` has units of km/s. Tested.
- **Limiting case at δ = 0, R finite, β₁ > 0:** predicted ΔH₀ = α · f_topo, pure topological contribution with no under-density. Physical reading: a topologically complex region (lots of small loops) without a mean density deficit still biases local H₀. Whether this is a feature or a bug depends on whether one considers "topology without δ" physical. The spec requires a one-liner in the functional's docstring stating: "At δ = 0, the functional still predicts nonzero ΔH₀ when β₁_persistent > 0. This is the genuine topological-only contribution; it should be small in regions that have been smoothed."
- **Limiting case at R → ∞:** `1/R → 0`, so `f_topo → 0` regardless of β₁. Good — this enforces vanishing of the topological correction at cosmological scales, consistent with ΛCDM averaging.
- **Limiting case at R → 0:** `1/R → ∞`, so `f_topo` diverges. The functional is only defined for R ≥ R_min ~ (a few Mpc), the scale below which voids are not meaningful. Plan sets `R_min = 10 Mpc`; smaller inputs raise.
- **Monotonicity at fixed δ, increasing |δ|:** we require `d(ΔH₀)/d(−δ) > 0` for δ < 0. Tested on a small scan.

---

## Appendix — quick reference summary

| Target | Decision | Key artifact |
|---|---|---|
| T1 | Density-gradient stalks, `R_dst^t = λ^t·(Rot ⊕ P^t)`, `R_src^t = I` | `laplacian.py` rewrite |
| T2 | VR filtration + lifetime > τ·ℓ̄ (default τ=1.5) | `spectrum.py` rewrite |
| T3 | HONEST PUNT: c₁ anchored to LTB, not derived; sign = −H₀/3 | `functional.py` sign fix |
| T4 | Full-LTB analytical target, α fit against non-leading residual | `ltb_reference.py` new |
| Sign | c₁ = −H₀/3. ΔH₀ > 0 for δ < 0. | Pinned in docstring + test |
| Tests | 7 files updated, ~5 new test functions | `tests/hubble_tension_web/*` |
| Artifacts | Move v1 results to `results/v1_superseded/` | Migration task |
| Effort | 10.5 implementer-task-units | — |
