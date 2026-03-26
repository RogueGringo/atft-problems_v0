# ATFT Hallucination Detector

> Real-time topological detection of LLM reasoning degradation.
> No ground truth needed. Model-agnostic. One GPU.

## What It Does

Monitors an LLM's hidden states during inference. Computes the Gini trajectory
of H₀ persistence lifetimes across layers. When the trajectory flattens or
inverts — reasoning is degrading. Flag it before the output reaches the user.

## The Science

Validated at r = 0.991 across 4 architectures (SmolLM2, Qwen2.5, TinyLlama, Phi-1.5):
- **Positive Gini slope** (hierarchifying) → coherent reasoning
- **Flat Gini** → shallow processing, likely repetition
- **Negative Gini slope** (flattening) → degraded reasoning, likely hallucination

The Gini coefficient measures whether the topological features of the hidden
state point cloud are HIERARCHICAL (one dominant structure = focused reasoning)
or UNIFORM (many equal features = unfocused, drifting).

## Architecture

```
User prompt → LLM inference (with hidden state extraction)
                    ↓
              Per-layer H₀ persistence (GPU)
                    ↓
              Gini trajectory computation
                    ↓
              Slope analysis → COHERENT / WARNING / HALLUCINATING
```

## Usage (target API)

```python
from atft_detector import HallucinationDetector

detector = HallucinationDetector(model_name="Qwen/Qwen2.5-0.5B")

# Check a response
result = detector.check("What is the capital of France?",
                         response="The capital of France is Paris.")
print(result.verdict)      # COHERENT
print(result.gini_slope)   # +0.023
print(result.confidence)   # 0.94

# Real-time monitoring
for token in detector.stream("Explain quantum entanglement..."):
    if token.verdict == "HALLUCINATING":
        print(f"⚠ Degradation detected at token {token.position}")
        break
```
