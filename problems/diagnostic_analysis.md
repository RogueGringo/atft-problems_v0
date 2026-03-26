# SAT Phase Transition: Diagnostic Analysis (v1–v5)

**Date:** 2026-03-26  
**Status:** All five versions failed to detect α_c ≈ 4.27. Root causes identified.

---

## Summary

Five successive approaches attempted to detect the 3-SAT satisfiability phase transition at α_c ≈ 4.27 using topological methods (H₀ persistence, onset scale ε*, Gini coefficient). None produced a clean detection. The failures are not instrument failures — they are targeting failures. Every version pointed the topological instrument at the wrong observable.

---

## Version-by-Version Diagnosis

### v1: Spectral Embedding of Clause-Variable Graph

**Approach:** Embed the bipartite clause-variable graph via Laplacian eigenvectors, run H₀ persistence on the embedding.

**Result:** FAIL. Max |dε*/dα| at α = 2.00. Onset scale decays monotonically.

**Why it failed:** The clause-variable graph grows monotonically with α — more clauses means more edges, denser graph, smaller spectral gaps, smaller onset scale. There is no phase transition in the *problem structure* at fixed n. The transition lives in the *solution space*, which this embedding never touches.

### v2: Random Probe Satisfaction Patterns

**Approach:** Generate K=500 random assignments, evaluate clause satisfaction, use the binary satisfaction patterns as a point cloud in {0,1}^m, measure H₀ persistence via Hamming distance.

**Result:** FAIL. Satisfaction fraction locked at 7/8 = 0.875 for all α.

**Why it failed:** This is the most informative failure. Each clause in random 3-SAT is satisfied by a uniformly random assignment with probability exactly 7/8, independent of α. Random probes never approach the actual solution space — they sample a fixed-radius shell in Hamming space whose geometry is determined by the central limit theorem, not by the constraint structure. The point cloud is essentially the same object at every α.

**The 7/8 wall:** This is a fundamental limitation of random probing. The satisfying assignments are an exponentially small fraction of {0,1}^n, and random samples never get close enough to see the structure that changes at α_c.

### v3: Variable-Centric Constraint Geometry

**Approach:** Compute per-variable features (degree, polarity fraction, co-occurrence, frustration, connectivity), treat each variable as a point in R^6, run H₀ persistence.

**Result:** FAIL. All features monotone. Max derivative at α = 3.25 (onset), 2.00 (Gini), 2.25 (tension).

**Why it failed:** Per-variable statistics in random 3-SAT concentrate tightly at large n. Degree → Poisson(3α), polarity → Binomial, co-occurrence → deterministic function of α. The law of large numbers kills any transition signature. These features measure the *ensemble statistics* of the random formula, not the *geometry of its solution space*.

### v4: GPU-Parallel Greedy Walk Endpoints

**Approach:** Run 500 parallel greedy walks (flip variable in most unsatisfied clauses) for 200 steps, use endpoints as point cloud in Hamming space, run H₀ persistence.

**Result:** FAIL. Onset scale flat at ~0.375 for all α. Residual unsat grows monotonically from 0.063 to 0.091.

**Why it failed:** Two issues:

1. **Implementation bottleneck:** The flip loop iterates sequentially over walks (`for k in range(n_walks)`) inside each step, defeating GPU parallelism and limiting step count.

2. **Insufficient landscape differentiation:** Pure greedy descent on 100 variables with 200 steps doesn't produce meaningfully different endpoints. The walks all land at roughly the same Hamming distance from each other because the greedy heuristic (flip variable in most unsatisfied clauses) is deterministic given the starting point, and the starting points are random — so the endpoints form a uniform cloud whose radius is set by the greedy fixed-point structure, not the phase transition.

**The residual unsat never reaches zero** — even at α = 2.0, the greedy walk gets stuck at ~6.3% unsatisfied. This means the walks never find actual solutions, so they can't distinguish the SAT phase (solutions exist, walks find them) from the UNSAT phase (no solutions, walks get stuck). A noise parameter (as in WalkSAT) is essential.

### v5: Frustration Loops via Implication Graph

**Approach:** Build the 2N-literal implication graph, compute graph-level features (triangles, algebraic connectivity), spectrally embed and run H₀ persistence with an H₁ proxy (excess edges over triangles at onset scale).

**Result:** Nominal PASS (H₁ proxy max |d/dα| at α = 5.0 falls in [3.5, 5.5] window). Actual failure — the detection is noise.

**Why it's not real:**
- The onset scale has standard deviations of 0.09–0.16 on means of 0.44–0.54. The signal-to-noise ratio is <1.
- Algebraic connectivity λ₂ grows monotonically (2.78 → 20.28) — this is just the implication graph getting denser with more clauses. No transition.
- The H₁ proxy (edges − triangles at onset scale) fluctuates erratically with no clear discontinuity.
- The generous detection window [3.5, 5.5] admits spurious hits.
- Graph-level triangle counts grow as O(α³) by construction. This is combinatorics, not topology.

---

## The Core Structural Problem

All five versions share the same fundamental error: they measure properties of the **problem instance** (the clause-variable graph, the implication graph, the constraint statistics) rather than properties of the **solution space**.

The 3-SAT phase transition at α_c is a property of the solution set:
- Below α_c: solutions form one giant connected cluster (in Hamming space)
- At α_c: the cluster shatters into exponentially many disconnected sub-clusters
- Above α_c: no solutions exist

This shattering — called *replica symmetry breaking* in the statistical physics literature (Mézard, Parisi, Zecchina 2002) — is the phenomenon that the topological instrument needs to detect. The order parameter is the **overlap distribution** between solutions: the distribution of Hamming distances between pairs of satisfying (or near-satisfying) assignments.

The problem graph grows monotonically with α by construction. No amount of topological sophistication applied to the problem graph will produce a phase transition, because there is no phase transition in the problem graph.

---

## What Would Actually Work

### The Correct Observable: Solution Overlap Distribution

The point cloud should consist of **actual solutions or high-quality near-solutions**, not random probes or problem-graph embeddings.

**Protocol:**

1. **Generate solutions via WalkSAT with noise.** WalkSAT (Selman, Kautz, Cohen 1994) adds a noise parameter p: with probability p, flip a random variable in an unsatisfied clause; with probability 1−p, flip the variable that minimizes unsatisfied clauses. Multiple independent runs from random starts produce diverse solutions.

2. **Below α_c:** WalkSAT finds many distinct solutions. Collect K solutions. The pairwise Hamming overlap matrix defines the point cloud.

3. **At α_c:** WalkSAT struggles. Solutions are rare and clustered. The overlap distribution develops multiple peaks (cluster structure becomes visible).

4. **Above α_c:** No solutions exist. WalkSAT returns best-effort assignments that are stuck at local minima. The structure of these minima changes qualitatively.

5. **H₀ persistence on the overlap point cloud** detects the cluster shattering directly: one giant component fragments into many at α_c. This is the topological signature of replica symmetry breaking.

### Why This Is Different

- Random probes (v2) hit the 7/8 wall because they never approach the solution space. WalkSAT actually finds solutions (below α_c) or gets close.
- Greedy walks (v4) got stuck because pure greedy has no exploration. WalkSAT's noise parameter provides the stochasticity needed to escape local minima and find diverse solutions.
- Problem-graph embeddings (v1, v3, v5) measure the wrong object entirely. The overlap distribution measures the right object.

### Connection to the ATFT Framework

The README's core thesis is correct: the SAT phase transition is structurally analogous to the gauge theory confinement transition. In both cases, a global order parameter undergoes a discontinuity at a critical value of the control parameter. The ATFT instrument (sheaf Laplacian → onset scale → Gini) can detect this — but only if the point cloud encodes the order parameter.

- **Gauge theory:** point cloud = action density at lattice sites. The relevant observable.
- **Zeta zeros:** point cloud = spectral modes on the critical line. The relevant observable.
- **SAT:** point cloud = solution overlaps in Hamming space. **This is the relevant observable.** Not the clause-variable graph.

---

## Recommended v6 Design

```
1. For each α in sweep:
   a. Generate random 3-SAT instance (n=200, m=αn)
   b. Run K=1000 independent WalkSAT runs (p=0.57, T=10000 steps)
   c. Collect the best assignment from each run
   d. Point cloud: K points in {0,1}^n (or R^n)
   e. Distance matrix: pairwise Hamming / n
   f. H₀ persistence on distance matrix
   g. Record: ε*(α), Gini(α), number of components at scale δ, overlap distribution

2. Detection targets:
   - ε*(α) discontinuity at α_c
   - Component count spike at α_c
   - Overlap distribution: single peak → bimodal at α_c
   - Gini: low (democratic solutions) → high (hierarchical stuck states)
```

**Hardware note:** WalkSAT at n=200 with T=10000 steps runs in microseconds per instance. 1000 runs × 50 α values × 100 instances = 5M WalkSAT runs, each trivial. The persistence computation on K=1000 point clouds is the bottleneck, identical to existing pipeline.

---

*The instrument is sound. It was pointed at the wrong object. Point it at the solution space.*
