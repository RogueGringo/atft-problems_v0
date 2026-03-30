# The Simplicial Sheaf Architecture: A Unified Theory

> Extension of THESIS_013 ({0,1,3}). Emerged from first experimental results
> and theoretical synthesis during the full campaign run.

## The One-Sentence Version

A neural network is a simplicial if-then loop, geometricized for sheaf-explicit
operation on primes, with the derivative zeta zeros as the operational
convergence function.

## Decomposition

### 1. Simplicial If-Then Loop

The weight matrix IS a simplicial complex:
- Each nonzero weight = a simplex (connection between neurons)
- Weight = 0: no simplex (void)
- Weight = 1: 1-simplex (edge, identity transport)
- Weight = 3: rigid simplex (triangle-completing, irreducible amplification)
- H₀ of the weight topology = connected components (independent circuits)
- H₁ of the weight topology = loops (consistency constraints)

The "if-then" IS the computation:
```
if weight == 0: skip          (structured absence)
if weight == 1: pass through  (identity transport)
if weight == 3: amplify       (prime operation)
```

This is computation reduced to its simplicial skeleton.

### 2. Geometricized

Weight values encode geometric dimension:
- 0 = point (0-dim, void)
- 1 = edge (1-dim, identity)
- √2 = face diagonal (2-dim, rotation/orthogonal coupling)
- 3 = prime (irreducible, structure-generating)

The program IS the shape. Logic becomes geometry.

### 3. Sheaf-Explicit Operation on Primes

The network is a sheaf:
- Open sets = layers (or neighborhoods of neurons)
- Sections = weight transport maps at each position
- Restriction maps = how sections compose across layers
- Global sections = ker(L_F) = what the network actually computes

The operation is ON primes because:
- The 3-weights are the load-bearing structure
- The 1-weights are scaffolding (identity, structural but passive)
- The 0-weights are the topology (structured absence carrying information)
- Everything is organized around the irreducible elements

### 4. Derivative Zeta Zeros as Convergence Function

In number theory:
- ζ(s) zeros encode WHERE primes live (their distribution)
- ζ'(s) zeros encode WHERE prime density CHANGES (inflection points)

In the network:
- Weight zeros encode the computational topology (87-201% degradation on permutation)
- The pattern of activation density across layers = the network's zeta function
- Convergence of iterative inference = the point where ζ'(activation) = 0
- Fixed point = no region gaining or losing computational density = equilibrium

## Experimental Evidence (First Results)

### From quick comparison (1000 steps, 4 weight sets):

| Finding | Evidence |
|---------|----------|
| Ternary models converge iteratively | 3 iterations vs fp16 NEVER (21+) |
| Zero positions carry information | 87-201% ppl degradation on permutation |
| {0,1,3} forces democratic representations | Eff rank 168 vs fp16's 82 |
| {-1,0,1} most democratic | Eff rank 213 (confirms BitNet finding) |
| Weight stability in ternary | 99.9% stable vs fp16 0.0% stable |

### From full campaign (5000 steps, 5 weight sets, in progress):

| Finding | Evidence |
|---------|----------|
| {0,1,2,3} minimal overfitting | Gen gap 1.02x (test ≈ train) |
| Iterative convergence universal in ternary | All ternary sets converge in 3 iters |

## Key Predictions

1. **Training speed should increase** for ternary models as weights crystallize
   (fewer positions to update → less computation per step)

2. **{0,1,2,3} is the full minimal basis**: void + identity + bisection + triangulation
   Encodes all fundamental geometric operations in 2 bits

3. **Generalization gap will be smallest for {0,1,3}** because 3 forces the hardest
   structural commitments (no smooth interpolation through 2)

4. **A small {0,1,3} model will match a much larger fp16 model** on generalization
   tasks because 100% of its parameters are structurally committed

5. **The convergence criterion for iterative inference** is entropy-derivative = 0
   across the activation distribution (the ζ' zero analog)

## The Communication Theory Connection

When a message is transmitted:
- The receiver has a priori dimensional knowledge
- The FORM of the communication carries as much information as the CONTENT
- The syntax (arrangement) is the structured absence (zeros)
- The vocabulary (values) is the committed structure (1s and 3s)
- Coherence = convergence of the receiver's internal model (fixed point reached)

A {0,1,3} network communicates the same way:
- Zeros = syntax (arrangement carries information)
- 1s and 3s = vocabulary (committed transport maps)
- Iterative convergence = comprehension (fixed point = understanding)
- Non-convergence = incoherent input (honest failure, not hallucination)

## The Hallucination Diagnosis

fp16 models hallucinate because:
1. Weight stability = 0% — no weight has committed to a value
2. Every inference path routes through UNCOMMITTED parameters
3. The model improvises a path through noise = dreaming
4. Coherent output (legal text, etc.) is simulated form, not structural truth
5. It fools observers because form IS most of communication — but it's not real

{0,1,3} models cannot hallucinate the same way because:
1. Weight stability = 99.9% — all weights are committed
2. Inference routes through CRYSTALLIZED structure
3. The model produces output from stable topology = truth
4. If the input doesn't converge, the model REPORTS non-convergence
5. Structural truth, not simulated form

## Long Run Results (50K steps, 50M params)

### The Compression Vector

Every discrete weight model trains by compression (eff rank decreases, hierarchy builds).
fp16 trains by diffusion (eff rank increases, spreads out). They go in opposite directions.

### {0,1,3} at 50K steps:

| Phase | Steps | Eff Rank | Spectral Gap | What Happened |
|-------|-------|----------|-------------|---------------|
| Exploration | 0-3K | 167→111 | 10→28 | Everything active, random search |
| Crystallization | 3K-7K | 111→97 | 28→66 | First compression, peak rigidity |
| Annealing | 7K-15K | 97→89 | 66→39 | Crystal relaxes to optimal form |
| Second compression | 15K-25K | 89→80 | 39→39 | Deeper structure discovered |
| Saturation | 25K-50K | 80→77 | 39→38 | Harmonic baseline, manifold settled |

### Final crystal: 77 dimensions, spectral gap 38, PPL 80, gen gap 1.01

The 77-dimensional manifold is the intrinsic structure of TinyStories narrative.
54% of the initial representational space was pruned as structurally unnecessary.
The model cannot memorize (gen gap ≤ 1.01 throughout training).

### The quantization tax

PPL 80 vs fp16's PPL 6 at same size. The gap is syntax, not semantics:
- {0,1,3} at 50M captures WHAT (vocabulary, names, actions, story structure)
- fp16 at 50M additionally captures HOW (long-range grammar, syntax)
- More parameters (200M+) should close the syntax gap by providing more
  routing layers within the same 77-dimensional manifold

### Weight drift during training

Zeros: 22.2% → 22.7% (slowly increasing sparsity)
Ones: 41.7% → 43.1% (more identity transport)
Threes: 36.1% → 34.2% (fewer amplifiers)

The network learns that more connections should pass signal through unchanged
and fewer need amplification. The crystal becomes more transparent over time.

## What Remains

- Full campaign results (running)
- {0, 1, √2, 3} experiment (diagonal transport)
- Scale comparison: at what {0,1,3} size do you match fp16?
- Topological LoRA: adapt only low-persistence positions
- The convergence criterion: implement ζ'-analog measurement
- Longer training: does {0,1,3} eventually match fp16 on perplexity
  while maintaining superior generalization?

## Origin

This synthesis emerged from experimental results during the first {0,1,3}
training campaign. The simplicial-sheaf-zeta connection was articulated
during live analysis of the 4-way weight set comparison, which showed
that ternary models converge iteratively while fp16 does not, and that
zero positions carry massive structural information.
