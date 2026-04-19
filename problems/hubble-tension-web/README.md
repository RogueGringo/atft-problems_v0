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
