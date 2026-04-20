# v1_superseded — pre-rework artifacts

These files are outputs of the **v1 implementation** of `problems/hubble_tension_web/` (and a phase-1 hotfix atop it), committed between `8e8524a` and `4b8e4bd`. They are **retained for diff/history reasons only** and are known to be wrong in four ways:

1. The typed Laplacian was a no-op in v1 (every edge-type used the same orthogonal `R ⊗ ±I`, i.e. `L_F ≡ L_graph ⊗ I_stalk_dim`). Phase-1 (`43eba25`) introduced asymmetric `R_src=I`, `R_dst=λ·Q` but still uses random-orthogonal `Q` rather than structured gradient stalks.
2. β₁ was computed as `|E| − |V| + β₀` at a single k-NN scale — not a persistent topological invariant.
3. The kinematic coefficient had the **wrong sign** (`c₁ = +H₀/3` rather than `−H₀/3`) in v1. Phase-1 flipped the sign in `functional.py` but the `sim_calibration.py` reference curve still uses the old convention, which drove α* negative (-0.185).
4. The sim calibration reference curve `(H₀/3)·δ·window(R)` embeds the kinematic answer as its leading term, so α is fit against residuals of a circularly-defined target.

See `docs/superpowers/specs/2026-04-19-hubble-tension-web-REWORK-design.md` for the full critique and the rework plan at `docs/superpowers/plans/2026-04-19-hubble-tension-web-REWORK.md`.

**Do not cite these numbers.** The file layout, plot style, and REPORT.md format are what remain useful here.
