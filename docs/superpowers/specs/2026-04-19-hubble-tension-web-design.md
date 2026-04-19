# Hubble Tension as Cosmic-Web Topology — Design Spec

**Date:** 2026-04-19
**Status:** Approved design, ready for implementation plan.
**Author:** Blake Jones (with Claude)
**Project slug:** `problems/hubble-tension-web/`

## The Framing

The Hubble tension is the persistent ≥5σ disagreement between the locally-measured Hubble constant (distance ladder: Cepheid → SN Ia, H₀ ≈ 73 km/s/Mpc) and the globally-inferred one (CMB + ΛCDM, H₀ ≈ 67 km/s/Mpc). The standard reading is that the tension indicates new physics beyond ΛCDM. The alternative reading — this project's hypothesis — is that the tension is an artifact of *where we live*: if the local universe sits inside a significant under-density (the KBC void and related local-structure claims), then the locally-inferred expansion rate is biased by the void's topology, and a continuous FLRW metric applied to a discretely deformed local cosmic web produces exactly the observed offset.

In ATFT terms, this is a **sheaf gluing failure**. Local sections (distance-ladder H₀) and global sections (CMB H₀) fail to glue because the restriction maps between them pass through a topologically non-trivial local cosmic-web structure that the continuous metric does not see. The tension is the obstruction class — a cohomological shadow of discrete topology on a metric built to ignore it.

This project produces a functional that turns that shadow into a number.

## Goal — Three-Tier Stack

**(A) Machinery.** A typed sheaf-Laplacian-spectral operator 𝒦 over a graph built from local cosmic-web structure. The operator encodes the gluing of local probe sections (distance-ladder) onto global probe sections (CMB) across environment-typed (void / wall / filament / node) × redshift-shell categories.

**(C) Functional.** ΔH₀ = 𝒦(spec(L_F), δ, R), wrapped as an externally published `ΔH₀(β₀, β₁, δ, R)` interface. Betti numbers are the human-interpretable headline; the full spectrum is what the operator actually consumes.

**(B) Falsification test.** Run the functional on KBC-void parameters drawn from published sim-calibrated reconstructions of local structure. Output a number. If ΔH₀ ≈ 5 km/s/Mpc, the tension is topology. If ΔH₀ ≪ 1 km/s/Mpc, the local-void hypothesis is dead and ΛCDM needs new physics.

Each tier is a necessary component. (A) alone is a mathematical toy; (C) alone is an uncalibrated claim; (B) alone has nothing underneath.

## Architecture

- **Internal form.** ΔH₀ = 𝒦(spec(L_F), δ, R), where L_F is the typed sheaf Laplacian of a graph whose nodes are local cosmic-web elements (galaxies, halos, or grid cells — selected during plan) and whose typed fibers encode environment class. The spectrum's low eigenvalues correspond to large-scale gluing obstructions; the spectral gap at onset scale ε* is the quantity that maps into ΔH₀.
- **External interface.** `ΔH₀(β₀, β₁, δ, R)` — published signature. β₀, β₁ are the leading persistent Betti numbers of the local structure; δ is void depth; R is void radius. The wrapper exists so the result is communicable to the astrophysics community without demanding they learn sheaf cohomology.
- **Reuse from existing repo.**
  - `engine/topology` — typed sheaf Laplacian construction.
  - `products/TreeTrunk` / crystal kernel — `{0,1,3}` spectral reduction for the low-eigenvalue regime.
  - v11 cross-domain isomorphism analyzer — for comparing spectra across sim-injected voids of varying (δ, R) and asserting the functional's monotonicity.

## Validation Hierarchy

1. **Analytical anchor.** In the smooth limit (β₁ → 0, δ small, R large), 𝒦 must reduce to the Lemaître-Tolman-Bondi void prediction. Any deviation at this limit invalidates the operator before it touches data.
2. **Sim-calibrated scan.** Against public N-body snapshots (IllustrisTNG and/or MultiDark/MDPL2), inject or identify voids of known (δ, R), compute locally-inferred H₀ from mock SN catalogs placed inside each void, and compare to 𝒦's prediction across the (δ, R) scan. Pass = functional tracks sim ΔH₀ within stated error bars.
3. **Literature cross-check.** Published perturbative void-cosmology estimates put KBC-void ΔH₀ at roughly 1–3 km/s/Mpc. If 𝒦 outputs a different value, the disagreement must be defended explicitly (extra multi-scale structure captured by persistence that perturbation theory collapses, etc.) — not hidden.

All three legs are mandatory. The credibility chain is: analytical sanity → sim-quantitative accuracy → literature defensibility.

## Deliverables

Inside `problems/hubble-tension-web/`:

- `README.md` — the ATFT translation, in the shape of `problems/navier-stokes/README.md`: point cloud / control parameter / sheaf / detection target / experimental protocol / what success looks like / what's speculative.
- `functional.py` — the 𝒦 operator and the `(β₀, β₁, δ, R)` interface wrapper. Consumes sheaf Laplacian spectra; emits ΔH₀.
- `experiments/analytical_reduction.py` — smooth-limit test driving β₁ → 0, δ → 0, R → ∞; asserts convergence to LTB.
- `experiments/sim_calibration.py` — scan across public-snapshot voids, mock-SN local H₀ reconstruction, functional-vs-sim comparison.
- `experiments/kbc_crosscheck.py` — KBC-parameter input, literature-reference comparison, disagreement report.
- `results/` — persisted outputs of each experiment (JSON + plots).

## Non-Goals — Scope Fence

- **No running the functional on real observational galaxy catalogs.** Redshift-space distortions, peculiar-velocity corrections, catalog-specific systematics, and the full Cepheid/SN-Ia local-H₀ reconstruction pipeline belong to a sequel project. This project ends at sim-calibrated + literature cross-checked.
- **No new N-body simulation runs.** Public snapshots (IllustrisTNG, MDPL2, or equivalent) carry sufficient resolution. No custom sims.
- **No replacement cosmology.** This project tests one hypothesis by producing one number; it does not propose a new theory of gravity or a revised ΛCDM.
- **No implementation decisions in this spec.** Specific sim snapshot choice, graph-construction convention (galaxy-node vs. grid-cell-node vs. halo-node), exact form of 𝒦, choice of mock-SN reconstruction scheme — all deferred to the implementation plan produced by writing-plans.

## Open Questions Deferred to Planning

- Which sim suite (IllustrisTNG vs. MDPL2 vs. both) is the sim-calibration primary?
- Graph-construction convention at the local cosmic web: nodes = galaxies, halos, or grid cells?
- Form of 𝒦: closed-form map from spec(L_F) → ΔH₀, or learned from sim calibration and then checked against analytical limit?
- Error-bar budget for the sim-calibration pass: what does "tracks within error" mean quantitatively?

These are plan-level decisions, not design-level. They get resolved during writing-plans.
