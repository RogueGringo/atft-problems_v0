# Yang-Mills Mass Gap

```
The mass gap is not a number you calculate.
It's a discontinuity you detect.
```

## The Problem

Pure Yang-Mills theory — gauge fields with no quarks, no matter, just the field talking to itself — should have a **mass gap**: a minimum energy cost to excite anything above the vacuum. Proving this exists for SU(N) in 4D is one of the seven Millennium Problems. Nobody has done it. The lattice community has measured it numerically for decades, but measurement is not proof, and the theoretical mechanism behind confinement remains open.

## What We Already Have

We detected the SU(2) confinement-deconfinement transition at **beta_c = 2.30** on an 8^3x4 lattice using persistent homology alone. No Polyakov loop. No Wilson loop. The onset scale epsilon* dropped 10x at exactly the predicted critical coupling, with maximum topological derivative |d(epsilon*)/d(beta)| = 24.99. The confined phase has large onset scale (diffuse point cloud); the deconfined phase has small onset scale (compact point cloud). The transition is sharp, not gradual.

This is the **validation**. Yang-Mills SU(3) is the real target.

## The ATFT Translation

**Point cloud:** For each lattice gauge configuration at coupling beta, extract the plaquette field and map through a feature function phi: R^36 (8 generators of su(3), each contributing a 4D plaquette average + trace). Each lattice site becomes a point in R^36.

**Control parameter:** beta (inverse coupling). Sweep from beta = 5.5 (confined) through beta_c ~ 5.69 (SU(3) critical) to beta = 6.2 (deconfined).

**Sheaf:** Attach su(3)-valued fibers. The gauge connection on the Rips complex carries the local color structure between neighboring sites in feature space.

**Detection target:** The onset scale epsilon*(beta) should show a **discontinuity** at beta_c. In the confined phase, epsilon* is large — the gauge field is disordered, the point cloud is diffuse, and excitations above the vacuum cost energy (the mass gap is nonzero). In the deconfined phase, epsilon* is small — the field is ordered, the cloud is compact, and gluons propagate freely (the mass gap vanishes).

The mass gap IS the topological waypoint. It's the scale at which the persistence diagram changes character. If epsilon*(beta) has a sharp jump at beta_c that survives the continuum limit (larger lattices, finer spacing), that's the mass gap expressed as a topological invariant.

## Experimental Protocol

1. Generate SU(3) lattice configurations via heatbath + overrelaxation at 20 beta values in [5.5, 6.2]
2. Extract feature vectors phi in R^36 at each site
3. Build Rips complex, compute persistence, measure epsilon* and Gini trajectory at each beta
4. Identify the critical coupling as max |d(epsilon*)/d(beta)|
5. Repeat at 12^3x4 and 16^3x4 to check finite-size scaling

**Hardware:** i9-9900K + RTX 5070 12GB. SU(3) has 8 generators vs SU(2)'s 3, so the feature space is 3x larger. An 8^3x4 lattice has 2048 sites — manageable. 16^3x4 has 16384 sites — tight on VRAM but feasible with batched assembly (same trick that got K=200 Riemann running).

## What Success Looks Like

- epsilon*(beta) shows a sharp discontinuity near beta_c ~ 5.69, analogous to the SU(2) result at beta = 2.30
- The discontinuity sharpens with lattice volume (finite-size scaling consistent with first-order transition for SU(3))
- The Gini trajectory transitions from hierarchical (confined, structured vacuum) to flat (deconfined, perturbative regime)

## What's Speculative

Whether the topological waypoint survives the continuum limit. Lattice artifacts could shift or smear the transition. The feature map phi is a choice — different maps might see different things. The SU(2) validation gives us confidence, but SU(3) is a different beast: the deconfinement transition is first-order (vs second-order for SU(2)), which should make the discontinuity sharper, not weaker. That's a prediction we can test.

---

*The confined phase is where the topology is rich. The deconfined phase is where it's simple. The mass gap lives at the boundary.*
