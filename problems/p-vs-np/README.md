# P vs NP

```
The hardest SAT instances live at a phase transition.
Phase transitions are what this instrument was built to detect.
```

## The Problem

P vs NP asks whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time. The Clay Institute version demands a proof one way or the other. Nobody has one. Most complexity theorists believe P != NP, but belief is not proof, and every proof attempt has hit barriers (relativization, natural proofs, algebrization).

We're not going to resolve P vs NP with a topological operator. But we can do something nobody has done: measure the **topological structure of the computational landscape** at the exact point where problems transition from easy to hard, and see if that structure explains the hardness.

## The Connection

Random k-SAT has a phase transition. For 3-SAT, at clause-to-variable ratio alpha_c ~ 4.27, instances flip from almost-all-satisfiable to almost-all-unsatisfiable. This is not gradual — it's a sharp threshold, proven rigorously for certain regimes. The hardest instances for SAT solvers cluster right at this transition.

This is structurally identical to what we've already measured:
- SU(2) lattice gauge: confinement transition at beta_c = 2.30 (detected via epsilon* discontinuity)
- Zeta zeros: arithmetic coherence peak at sigma = 0.500 (detected via spectral sum minimum)
- k-SAT: satisfiability transition at alpha_c ~ 4.27 (to be detected via... the same instrument)

The same operator. The same pipeline. Different point cloud.

## The ATFT Translation

**Point cloud:** Each k-SAT instance defines a clause-variable structure. Embed it as a point cloud by either:
- (a) Treating each clause as a point in R^k (coordinates = variable indices), or
- (b) Embedding the clause-variable bipartite graph via spectral methods (Laplacian eigenvectors) into R^d

Option (b) is better — it captures the global structure of the instance, not just local clause content.

**Control parameter:** alpha = m/n (clause-to-variable ratio). Sweep from alpha = 3.0 (easy SAT) through alpha_c ~ 4.27 (transition) to alpha = 5.5 (easy UNSAT).

**Sheaf:** Attach Z_2-valued or R-valued fibers encoding variable assignment information. The transport map on the Rips complex carries the constraint structure between neighboring clauses in the embedded space.

**Detection target:** The onset scale epsilon*(alpha) should show a **discontinuity** at alpha_c. Below the transition, the solution space is large and connected — the point cloud has simple topology. Above the transition, there are no solutions — the constraint structure is rigid and the topology is again simple (but different). AT the transition, the solution space shatters into exponentially many clusters — the topology is maximally complex.

The Gini trajectory tells a parallel story:
- **SAT phase (alpha < alpha_c):** Flat Gini — many balanced solutions, democratic structure
- **Transition (alpha ~ alpha_c):** Rapidly hierarchifying — the solution space is fragmenting
- **UNSAT phase (alpha > alpha_c):** High Gini, stable — the proof structure dominates, few critical clauses control everything

## Experimental Protocol

1. Generate random 3-SAT instances: n = 200 variables, sweep alpha from 3.0 to 5.5 in 50 steps, 100 instances per alpha value
2. Embed each instance via spectral method (graph Laplacian of clause-variable bipartite graph, top-d eigenvectors)
3. Build Rips complex on the embedded point cloud, compute persistence, measure epsilon* and Gini
4. Average over instances at each alpha. Plot epsilon*(alpha) and Gini(alpha)
5. Identify the topological transition point and compare to known alpha_c

**Hardware:** i9-9900K + RTX 5070. Random 3-SAT generation is trivial (numpy). Spectral embedding of a 200-variable, ~850-clause instance takes milliseconds. The persistent homology is the bottleneck, but n=200 produces point clouds of O(1000) points — well within capacity. The 5000 total instances (50 alpha values x 100 instances) will take hours, not days.

## What Success Looks Like

- epsilon*(alpha) shows a sharp discontinuity near alpha_c ~ 4.27, analogous to the gauge theory transition
- The topological complexity (total persistence, number of persistent features) peaks at the transition
- The Gini trajectory transitions from flat to hierarchical across the threshold
- Hardest instances for SAT solvers (measured by DPLL backtrack count) correlate with maximal topological complexity

## The Deep Question

If the topology of the SAT landscape at the phase transition has a specific, measurable character — and if that character is the same kind of discontinuity we see in gauge theory confinement — then computational hardness might be a topological phenomenon. Not metaphorically. Literally. The same sheaf Laplacian that detects the mass gap detects the satisfiability threshold.

We're not claiming this proves P != NP. We're claiming it puts a new instrument on the problem: one that reads the geometry of the constraint space at the exact point where easy becomes hard. Whatever the answer to P vs NP is, the topology at alpha_c is part of the story.

---

*The hard instances aren't random. They live at a phase transition. The topology knows where the transition is.*
