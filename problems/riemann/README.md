# Riemann Hypothesis

```
21.5%. 16 sigma. Holonomy = 0 at sigma = 1/2.
The primes thread the zeros tighter than anything else we tested.
```

> Primary repo: [JTopo/Ti V0.1](https://github.com/RogueGringo/JTopo)
> This directory holds the migration summary. The full code, data, and paper live in JTopo.

## What Was Validated

The ATFT framework was validated against 7 predictions derived from the theoretical paper. Scorecard: **5 PASS / 1 FAIL / 1 PARTIAL**.

| # | Prediction | Verdict | Key Number |
|---|-----------|---------|------------|
| 1 | SU(2) confinement transition at beta_c = 2.30 | **PASS** | epsilon* drops 10x at exactly beta = 2.30 |
| 2 | Instanton discrimination (Q=+1 vs Q=-1) | **PARTIAL** | Vacuum vs instanton: KS = 1.0. Charge sign: FAIL (naive discretization kills q_uv) |
| 3 | LLM cross-model correlation r > 0.9 | **PASS** | r = 0.9998 across SmolLM2 + Qwen2.5 |
| 4 | ker(L_F) > 0 for on-shell configurations | **FAIL** | ker = 0 everywhere. Premium is continuous offset, not binary jump |
| 5 | QHO gap-bar correspondence | **PASS** | rho = 1.0 (tautological in R^1) |
| 6 | Betti curve onset scale discrimination | **PASS** | 21.1% onset difference between Zeta and GUE |
| 7 | Gini trajectory as quality predictor | **PASS** | Hierarchifying for structured sources, flattening for random |

The failure on Prediction 4 was informative: the data shows a **continuous spectral premium** rather than a binary kernel jump. The 21.5% arithmetic premium is a multiplicative constant that converges across K values (K=200: 21.5%, K=400: 21.6%). The paper's prediction was wrong in form but the underlying phenomenon is stronger — a continuous invariant carries more information than a binary detector.

## The Key Numbers

At K=200 (46 primes, 1000 Odlyzko zeros near the 10^20th zero):

| Source | S(sigma=0.5) | vs Zeta |
|--------|-------------|---------|
| **Zeta zeros** | **11.784** | — |
| Even spacing | 12.713 | +7.3% |
| GUE (10 D-E draws) | 14.970 +/- 0.198 | +21.5% |
| Poisson random | 22.087 | +87.4% |

**Z-score vs GUE ensemble: -16.06.** Edge-normalized per-edge premium: **15.3%.**

The hierarchy S(zeta) < S(even) < S(GUE) < S(random) holds at all 11 sigma values tested. It holds at K=100 and K=400. It holds after edge normalization.

The gauge connection holonomy at sigma = 1/2 is **exactly zero** (defect = 0.000000), confirming sigma = 1/2 as the unique unitary surface.

## Novelty Test

The 21.5% premium constitutes a **new invariant** — invisible to pair correlations (GUE and zeta share local statistics), invisible to spacing distributions, visible only through the sheaf Laplacian's multi-prime transport structure. Residual after controlling for all known spectral statistics: **33%** of the premium remains unexplained by standard RMT measures.

## Key Documents

| Document | Location |
|----------|----------|
| Paper (PDF) | `docs/paper/Computational Topology and the Riemann Hypothesis.pdf` |
| Technical audit | `docs/TECHNICAL_AUDIT.md` |
| Validation results | `docs/atft_validation_results/SUMMARY.md` |
| Full experimental results | `docs/RESULTS.md` |
| K=200 raw data | `output/phase3d_torch_k200_results.json` |
| GUE ensemble data | `output/phase3e_test2_rerun_results.json` |
| Publication figures | `output/figures/fig[1-5]_*.png` |

All paths relative to JTopo repo root.

## What's Next

- **K=800 anomaly investigation:** The K=400 premium converged at 21.6% (predicted: 27.7%). Is the convergence real, or does something change at higher K? K=800 will require the matrix-free engine and careful VRAM management.
- **Dirichlet L-function extension:** Replace the Riemann zeta zeros with zeros of Dirichlet L-functions L(s, chi). If the arithmetic premium persists for non-trivial characters, the phenomenon is not specific to zeta — it's a property of all L-functions on the critical line. This would be a strong statement about GRH.
- **Continuum limit:** Extrapolate the K -> infinity behavior. Does the spectral sum converge? Does the premium converge? The first three data points (K=100, 200, 400) suggest convergence at ~21.5%.

## Hardware

Everything ran on local hardware: i9-9900K (development) + RTX 5070 12GB (GPU sweeps). No cloud. K=200 took 12 hours across three tranches. K=400 ran with the matrix-free engine in a single session.

---

*The primes aren't a list. They're a field. The sheaf Laplacian is the instrument that reads the field. The zeros are where the reading is sharpest.*
