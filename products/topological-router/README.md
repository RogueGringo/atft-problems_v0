# Topological Token Router

> Temperature is dead. Gini-guided sampling is the replacement.

## The Problem with Temperature

Temperature is a single scalar applied uniformly to the entire logit
distribution at every token position. It knows nothing about:
- Whether the model is confident or guessing
- Whether the current token is in a reasoning chain or a creative fill
- Whether the hidden state topology is hierarchifying or flattening
- Which attention heads are certain and which are noise

It's a volume knob on a symphony orchestra. It makes everything
louder or softer. It can't make the violins louder and the drums softer.

## The Replacement: Gini-Guided Topological Sampling

At each token position, BEFORE sampling:
1. Read the hidden state topology (Gini of the attention pattern)
2. If hierarchifying (high Gini): sample deterministically (the model knows)
3. If flattening (low Gini): sample broadly (the model is uncertain — explore)
4. If mixed (some heads confident, others not): per-head weighted sampling

The "temperature" becomes a FUNCTION of the model's internal state.
Not a constant. Not set by the user. Computed from the topology.

## Multi-Model Braided Generation

Token-by-token routing across N models:
1. All N models process the same context in parallel
2. Each generates candidate next token + hidden state
3. The detector measures Gini of each candidate's hidden state
4. The token from the highest-Gini model is selected
5. ALL models receive this token as the next context
6. Repeat

The output is a BRAID — woven from whichever model is most coherent
at each position. No single model dominates. The topology decides.

## Architecture

```
Context → [Model A, Model B, Model C, Model D]  (parallel inference)
              ↓          ↓          ↓          ↓
         [hidden_A]  [hidden_B]  [hidden_C]  [hidden_D]
              ↓          ↓          ↓          ↓
         [Gini_A]    [Gini_B]    [Gini_C]    [Gini_D]  (53ms each)
              ↓          ↓          ↓          ↓
         [Route: select token from argmax(Gini)]
              ↓
         Selected token → append to ALL model contexts
              ↓
         Repeat
```

## Why Token Chaining is Now a Strength

The "weakness" of autoregressive models: each token depends on all
previous tokens. One bad token corrupts the chain.

The strength: each token's hidden state encodes the ENTIRE chain's
topology. The detector can read the full history at every position.
The moment one model's chain degrades, switch to another — and the
new model inherits the GOOD tokens from the shared context.

The chain doesn't corrupt. It ROUTES AROUND corruption.

## Semantic Prime Calibration

Before trusting any model on a real query, probe it with the 59
semantic primes. Measure its topology on KNOWN ground truth.
If the model's semantic prime Gini deviates from the established
0.945 fixed point — it's drifting. Downweight it in the routing.

The primes are the health check. The topology is the router.
The trust is in the math, not the model.
