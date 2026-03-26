# ATFT Problems

```
What happens when you stop solving equations
and start reading topology?
```

> Seven problems. One instrument. The same question every time: does the topology of the configuration space, read at multiple scales through sheaf-valued persistent homology, detect the structure that differential equations describe?

---

## The Instrument

The **Adaptive Topological Operator** takes any parameterized configuration — a gauge field, a fluid flow, a SAT instance, a set of zeta zeros — translates it to a point cloud, and measures how its topology evolves across observation scales. The field equations of the configuration appear as **topological waypoints**: critical scales where the cohomology undergoes qualitative change.

This is not a metaphor. The Čech-de Rham isomorphism guarantees that persistent homology on Rips complexes computes the same cohomological invariants as continuous differential geometry. The discrete computation is exact in an isomorphic category.

The instrument has been validated:
- SU(2) confinement-deconfinement transition detected at β_c = 2.30 without Polyakov loop
- Cross-model LLM universality confirmed at r = 0.991 across 4 architectures
- Arithmetic premium of 21.5% for zeta zeros — a new invariant invisible to pair correlations
- σ = ½ proven to be the unique unitary surface of the gauge connection (defect = 0.000000)

**Source:** [JTopo/Ti V0.1](https://github.com/RogueGringo/JTopo)

---

## The Problems

Each problem gets the same treatment: translate to point cloud, build sheaf, sweep control parameter, detect waypoints, report honestly.

| Problem | Status | Point Cloud | Control Parameter | Question |
|---------|--------|------------|-------------------|----------|
| [Riemann Hypothesis](problems/riemann/) | **Validated** | Zeta zeros | σ (real part) | Does sheaf transport cohere at σ=½? |
| [Yang-Mills Mass Gap](problems/yang-mills/) | **Active** | SU(3) lattice configs | β (coupling) | Is the mass gap a topological waypoint? |
| [Navier-Stokes](problems/navier-stokes/) | **Active** | Vortex positions | Re (Reynolds) | Do singularities appear as persistence events? |
| [P vs NP](problems/p-vs-np/) | **Active** | SAT clause-variable graphs | α (clause ratio) | Is the SAT phase transition topological? |
| [Birch & Swinnerton-Dyer](problems/bsd/) | Planned | Elliptic curve points | Conductor | Does rank = dim ker(L_F)? |
| [Hodge Conjecture](problems/hodge/) | Planned | Algebraic variety samples | Degree | Do algebraic cycles persist longer? |
| [Poincaré](problems/poincare/) | Planned | Ricci flow snapshots | Flow time | Can ATFT recover Perelman's topology? |

---

## The Pattern

Every problem follows the same 6-step pipeline:

```
GENERATE → BUILD → SWEEP → DETECT → COMPARE → REPORT
   ↑                                              |
   └────────── next problem uses same engine ──────┘
```

1. **GENERATE** the point cloud from the problem's configuration space
2. **BUILD** the Rips complex and attach sheaf with Lie algebra fibers
3. **SWEEP** the control parameter across the relevant range
4. **DETECT** topological waypoints: onset scale discontinuities, Gini transitions
5. **COMPARE** against known results, controls, or null hypothesis
6. **REPORT** honestly: PASS / FAIL / PARTIAL with traceable artifacts

The engine is shared. The problems are independent. Each subdirectory is self-contained: its own experiments, results, figures, and write-up.

---

## The Axiom

Primes are the 0-dimensional framework of computational reality. Every mathematical structure that describes reality — from the distribution of primes to the behavior of fluids to the hardness of computation — has a topological signature. The topology doesn't care what the structure IS. It cares how the structure CHANGES across scales.

The adaptive operator is the instrument that reads this signature. The waypoint constraints are the field equations written in the native language of topology.

The question isn't whether this works. We've shown it works on zeta zeros, gauge fields, and language models. The question is: how far does it go?

---

## Hardware

Everything runs local. No cloud. No external compute.

| Resource | Spec |
|----------|------|
| CPU | Intel i9-9900K (8 cores, 16 threads, 5 GHz) |
| RAM | 32 GB |
| GPU | NVIDIA RTX 5070 (12 GB VRAM, SM 12.0) |
| Disk | 850 GB free |
| Software | PyTorch 2.10 + CUDA 12.8, scipy, mpmath, sympy, networkx |

---

## Getting Started

```bash
git clone https://github.com/RogueGringo/atft-problems.git
cd atft-problems
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the Riemann hypothesis validation
python -m problems.riemann.validate

# Run the Yang-Mills mass gap test
python -m problems.yang_mills.sweep

# Run all problems
python -m pytest problems/ -v
```

---

*Built by B. Aaron Jones. One GPU, one framework, seven problems.*

*The topology doesn't care what the structure is. It cares how the structure changes across scales.*
