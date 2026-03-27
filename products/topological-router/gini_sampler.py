#!/usr/bin/env python3
"""Gini-Guided Token Sampling — topology replaces temperature.

Instead of a fixed scalar temperature, the sampling distribution is
shaped by the model's internal topology at each token position.

High Gini (hierarchical hidden state) → low temperature (confident)
Low Gini (flat hidden state) → high temperature (exploring)
Mixed → per-head weighted sampling

This is the core primitive for the topological router.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial.distance import pdist


def gini_fast(values: np.ndarray) -> float:
    """Fast Gini coefficient."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    s = np.sort(values)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(i * s)) / (n * np.sum(s)) - (n + 1) / n)


def hidden_state_gini(hidden_state: torch.Tensor) -> float:
    """Compute Gini of H₀ persistence on a single hidden state.

    Args:
        hidden_state: (seq_len, hidden_dim) tensor from one layer

    Returns:
        Gini coefficient of persistence lifetimes
    """
    h = hidden_state.cpu().float().numpy()
    if h.shape[0] < 3:
        return 0.0

    # Fast PCA via SVD
    d = min(15, h.shape[0] - 1, h.shape[1])
    h_c = h - h.mean(0)
    try:
        _, _, Vt = np.linalg.svd(h_c, full_matrices=False)
        h_r = h_c @ Vt[:d].T
    except np.linalg.LinAlgError:
        h_r = h[:, :d]

    # Fast H₀ persistence
    dists = pdist(h_r)
    n = len(h_r)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = np.argsort(dists)
    ii, jj = np.triu_indices(n, k=1)
    bars = []
    for k in order:
        a, b = int(ii[k]), int(jj[k])
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            bars.append(float(dists[k]))

    return gini_fast(np.array(bars)) if bars else 0.0


def gini_to_temperature(gini_value: float,
                         t_min: float = 0.1,
                         t_max: float = 1.5,
                         midpoint: float = 0.3) -> float:
    """Convert Gini score to temperature.

    High Gini (hierarchical, confident) → low temperature
    Low Gini (flat, uncertain) → high temperature

    Uses sigmoid mapping centered at midpoint.
    """
    # Sigmoid: maps [0, 1] → [t_max, t_min]
    # High Gini → low temp (confident → deterministic)
    scale = 10.0  # sharpness of transition
    sigmoid = 1.0 / (1.0 + np.exp(-scale * (gini_value - midpoint)))
    temperature = t_max - sigmoid * (t_max - t_min)
    return float(temperature)


def gini_guided_sample(logits: torch.Tensor,
                        hidden_state: torch.Tensor,
                        top_k: int = 50) -> tuple[int, float, float]:
    """Sample next token using Gini-guided temperature.

    Args:
        logits: (vocab_size,) raw logits from model
        hidden_state: (seq_len, hidden_dim) from the last layer

    Returns:
        (token_id, gini_value, temperature_used)
    """
    g = hidden_state_gini(hidden_state)
    temp = gini_to_temperature(g)

    # Apply temperature
    scaled_logits = logits / temp

    # Top-k filtering
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(scaled_logits, top_k)
        probs = torch.softmax(topk_vals, dim=-1)
        selected = torch.multinomial(probs, 1).item()
        token_id = topk_idx[selected].item()
    else:
        probs = torch.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()

    return token_id, g, temp


def demo():
    """Demo: generate text with Gini-guided sampling vs fixed temperature."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="cuda",
        output_hidden_states=True,
    )
    model.eval()

    prompt = "The relationship between prime numbers and topology is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\nPrompt: {prompt}")
    print(f"\n{'Token':<20} {'Gini':>8} {'Temp':>8} {'Method'}")
    print("-" * 50)

    # Generate 30 tokens with Gini-guided sampling
    generated_ids = inputs["input_ids"][0].tolist()
    for step in range(30):
        input_ids = torch.tensor([generated_ids], device="cuda")
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        logits = outputs.logits[0, -1, :]  # last token logits
        hidden = outputs.hidden_states[-1][0]  # last layer hidden state

        token_id, g, temp = gini_guided_sample(logits, hidden)
        generated_ids.append(token_id)

        token_str = tokenizer.decode([token_id])
        print(f"  {token_str:<20} {g:>8.4f} {temp:>8.3f} {'→det' if temp < 0.5 else '→exp' if temp > 1.0 else '→mid'}")

    full_text = tokenizer.decode(generated_ids)
    print(f"\nGenerated: {full_text}")


if __name__ == "__main__":
    demo()
