# Hodge Conjecture — Algebraic Cycles as Persistent Features

> The most abstract of the seven. Can persistent homology distinguish
> algebraic from non-algebraic cohomology classes?

## Status: PLANNED (high risk, far from current capability)

## The Conjecture

On a smooth projective algebraic variety, every Hodge class is a rational
linear combination of classes of algebraic cycles.

## ATFT Translation

**Point cloud:** Sample N points from a projective variety V ⊂ CP^n.
**Feature map:** Algebraic coordinates (real and imaginary parts of homogeneous coordinates).
**Persistence:** H_k persistence on the Rips complex of the sample.

**Prediction:** Algebraic subvarieties (which ARE algebraic cycles) should produce
persistence features that are:
1. Longer-lived (more geometrically rigid)
2. More stable under perturbation
3. Distinguishable from non-algebraic Hodge classes (if any exist)

## Why It's Hard

1. Sampling from projective varieties in CP^n requires algebraic geometry tools
2. Non-algebraic Hodge classes have never been constructed (that's the conjecture)
3. The connection between persistence and Hodge theory is purely speculative
4. Numerical precision in projective coordinates is challenging

## Feasibility

LOW with current tools. Would need:
- A library for sampling from algebraic varieties (sage, magma, or custom)
- A clear theoretical connection between persistence and Hodge decomposition
- A test case where the answer is known (surfaces where Hodge = algebraic is proved)

## First Step

Test on a K3 surface (4-dimensional real manifold where the Hodge conjecture
is known to hold). Sample points, compute persistence, verify that all
long-lived features correspond to known algebraic cycles.
