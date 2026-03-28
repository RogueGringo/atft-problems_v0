#!/usr/bin/env python3
"""Topology-Regularized Training — Penalize flat hidden states.

The model already uses 99.5% of its representational basis.
The bottleneck isn't alignment — it's HIERARCHY QUALITY.

Standard loss: minimize next-token prediction error.
Our loss: minimize next-token error + maximize hidden state hierarchy.

Loss = alpha * LM_loss + (1 - alpha) * flatness_penalty

Where flatness_penalty = 1 - (spectral_gap / spectral_gap_target)
  - High spectral gap (hierarchical) → low penalty
  - Low spectral gap (flat) → high penalty

The model learns to maintain hierarchical hidden states while still
predicting tokens correctly. The topology regularization shapes HOW
the model represents information, not just WHAT it predicts.

Logged: every metric at every step for full trajectory analysis.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results" / "topo_regularized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))


# ── Topology regularization ────────────────────────────────────────────────

def hierarchy_penalty(hidden_states: torch.Tensor,
                      target_concentration: float = 0.5) -> torch.Tensor:
    """Differentiable penalty for flat hidden states. NO SVD needed.

    Uses variance concentration: what fraction of total variance is in the
    top principal direction? Computed via power iteration (1 step, cheap).

    High concentration → hierarchical → low penalty
    Low concentration → flat → high penalty

    Returns penalty in [0, 1].
    """
    if hidden_states.dim() == 3:
        h = hidden_states[0]  # first batch element
    else:
        h = hidden_states

    h = h.float()
    if h.shape[0] < 3:
        return torch.tensor(0.0, device=h.device, requires_grad=True)

    # Center
    h = h - h.mean(dim=0)

    # Total variance
    total_var = (h ** 2).sum()

    # Variance in top direction via power iteration (1 step)
    # v = H^T @ H @ random_vec, normalized
    # Then top_var = ||H @ v||^2
    torch.manual_seed(42)  # deterministic for reproducibility
    v = torch.randn(h.shape[1], device=h.device, dtype=h.dtype)
    v = v / (v.norm() + 1e-10)

    Hv = h @ v                    # (seq_len,)
    HTHv = h.T @ Hv               # (hidden_dim,)
    v_top = HTHv / (HTHv.norm() + 1e-10)
    top_var = (h @ v_top).pow(2).sum()

    # Concentration = top_var / total_var (fraction of variance in top direction)
    concentration = top_var / (total_var + 1e-10)

    # Penalty: 1 when flat (concentration ≈ 1/dim), 0 when hierarchical
    penalty = 1.0 - torch.clamp(concentration / target_concentration, 0.0, 1.0)

    return penalty


# ── LoRA (same as topo_distill but with device-aware init) ──────────────

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        device = original.weight.device
        # LoRA params in fp32 for training stability (model stays fp16)
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x):
        base = self.original(x)
        # Cast x to fp32 for LoRA computation, cast result back
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora_out.to(base.dtype)


def apply_lora(model, target_layer: int, rank: int = 8) -> list[nn.Parameter]:
    """Apply LoRA to attention projections near the target layer."""
    n_layers = model.config.num_hidden_layers
    start = max(0, target_layer - 2)
    end = min(n_layers, target_layer + 3)

    params = []
    for li in range(start, end):
        layer = model.model.layers[li]
        for proj in ["q_proj", "k_proj", "v_proj"]:
            orig = getattr(layer.self_attn, proj)
            lora = LoRALinear(orig, rank=rank)
            setattr(layer.self_attn, proj, lora)
            params.extend([lora.lora_A, lora.lora_B])

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: layers {start}-{end-1}, {n_train:,} trainable ({n_train/n_total*100:.3f}%)")
    return params


# ── Dataset ─────────────────────────────────────────────────────────────────

class TrainingData(Dataset):
    def __init__(self, tokenizer, max_length=128, n_samples=2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Diverse prompts
        from prompts.loader import load_prompts
        for p in load_prompts():
            self.samples.append(p["text"])

        # MMLU Q&A pairs
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "college_physics", split="test")
            for r in ds:
                correct = r["choices"][int(r["answer"])]
                self.samples.append(f"Q: {r['question']}\nA: {correct}")
        except Exception:
            pass

        if len(self.samples) < n_samples:
            self.samples = (self.samples * ((n_samples // len(self.samples)) + 1))[:n_samples]
        np.random.default_rng(42).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.samples[idx], return_tensors="pt",
                            truncation=True, max_length=self.max_length,
                            padding="max_length")
        return {k: v.squeeze(0) for k, v in enc.items()}


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model, tokenizer, target_layer, device, n_questions=20):
    """Quick eval: LM loss + spectral gap on physics questions."""
    from prompts.loader import load_prompts
    physics = [p["text"] for p in load_prompts("physics")][:n_questions]

    lm_losses = []
    gaps = []

    for text in physics:
        inp = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128).to(device)
        out = model(input_ids=inp["input_ids"], output_hidden_states=True)

        logits = out.logits[:, :-1, :]
        labels = inp["input_ids"][:, 1:]
        lm = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1), reduction="mean")
        lm_losses.append(lm.item())

        hs = out.hidden_states[target_layer][0].float()
        hs_c = hs - hs.mean(0)
        s = torch.linalg.svdvals(hs_c)
        if len(s) >= 2:
            gaps.append((s[0] / (s[1] + 1e-10)).item())

    return {
        "lm_loss": float(np.mean(lm_losses)),
        "spectral_gap": float(np.mean(gaps)),
        "spectral_gap_std": float(np.std(gaps)),
    }


# ── Training ────────────────────────────────────────────────────────────────

def train(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
          n_epochs: int = 2,
          batch_size: int = 1,
          lr: float = 1e-4,
          alpha: float = 0.7,         # weight for LM loss
          target_gap: float = 50.0,   # spectral gap target
          lora_rank: int = 8,
          max_length: int = 128,      # shorter sequences = less memory
          eval_every: int = 100,
          log_every: int = 20,
          device: str = "cuda"):

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  TOPOLOGY-REGULARIZED TRAINING")
    print(f"  {model_name}")
    print(f"  Loss = {alpha}*LM + {1-alpha}*hierarchy_penalty")
    print(f"  target_gap={target_gap}, lora_rank={lora_rank}")
    print(f"  {ts}")
    print("=" * 65)

    # Load model
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float16, device_map=device,
        output_hidden_states=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 3
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{n_layers}L, target={target_layer}, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # LoRA
    trainable = apply_lora(model, target_layer, rank=lora_rank)
    model.train()

    # Data
    dataset = TrainingData(tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = n_epochs * len(loader)
    print(f"  Data: {len(dataset)} samples, {total_steps} steps")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # Pre-eval
    print(f"\n  Pre-training eval...")
    model.eval()
    pre = quick_eval(model, tokenizer, target_layer, device)
    print(f"    lm_loss={pre['lm_loss']:.4f} sg={pre['spectral_gap']:.1f} +/- {pre['spectral_gap_std']:.1f}")
    model.train()

    # Training log
    log = {
        "config": {"model": model_name, "alpha": alpha, "target_gap": target_gap,
                   "lora_rank": lora_rank, "lr": lr, "n_epochs": n_epochs,
                   "batch_size": batch_size, "max_length": max_length, "timestamp": ts},
        "pre_eval": pre,
        "steps": [],
        "evals": [],
    }

    # Training loop
    print(f"\n  Training...")
    global_step = 0

    for epoch in range(n_epochs):
        epoch_lm = []
        epoch_topo = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)

            # LM loss
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, 1:].float()
            lm_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1), reduction="none")
            lm_loss = (lm_loss.view_as(labels) * mask).sum() / (mask.sum() + 1e-10)

            # Topology penalty at target layer
            hs = outputs.hidden_states[target_layer]
            topo_penalty = hierarchy_penalty(hs, target_concentration=target_gap)

            # Combined
            total = alpha * lm_loss + (1 - alpha) * topo_penalty

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            global_step += 1
            epoch_lm.append(lm_loss.item())
            epoch_topo.append(topo_penalty.item())

            if global_step % log_every == 0:
                step_data = {
                    "step": global_step,
                    "lm_loss": lm_loss.item(),
                    "topo_penalty": topo_penalty.item(),
                    "total_loss": total.item(),
                    "epoch": epoch,
                }
                log["steps"].append(step_data)
                print(f"  step {global_step:4d} | "
                      f"lm={lm_loss.item():.4f} topo={topo_penalty.item():.4f} "
                      f"total={total.item():.4f}")

            if global_step % eval_every == 0:
                model.eval()
                ev = quick_eval(model, tokenizer, target_layer, device)
                log["evals"].append({"step": global_step, **ev})
                print(f"  EVAL {global_step}: lm={ev['lm_loss']:.4f} "
                      f"sg={ev['spectral_gap']:.1f} +/- {ev['spectral_gap_std']:.1f}")
                model.train()

        print(f"  Epoch {epoch+1}: lm={np.mean(epoch_lm):.4f} topo={np.mean(epoch_topo):.4f}")

    # Post-eval
    print(f"\n  Post-training eval...")
    model.eval()
    post = quick_eval(model, tokenizer, target_layer, device)
    log["post_eval"] = post
    print(f"    lm_loss={post['lm_loss']:.4f} sg={post['spectral_gap']:.1f} +/- {post['spectral_gap_std']:.1f}")

    # Summary
    print(f"\n{'='*65}")
    print("  PRE vs POST")
    print(f"{'='*65}")
    for k in ["lm_loss", "spectral_gap"]:
        delta = post[k] - pre[k]
        print(f"  {k:<16}: {pre[k]:.2f} → {post[k]:.2f} ({'+' if delta>0 else ''}{delta:.2f})")

    sg_improved = post["spectral_gap"] > pre["spectral_gap"]
    lm_preserved = post["lm_loss"] < pre["lm_loss"] * 1.1  # allow 10% degradation
    print(f"\n  Spectral gap improved: {'YES' if sg_improved else 'NO'}")
    print(f"  LM quality preserved:  {'YES' if lm_preserved else 'NO'}")
    if sg_improved and lm_preserved:
        print(f"  VERDICT: Topology regularization WORKS — hierarchy up, quality stable")
    elif sg_improved:
        print(f"  VERDICT: Hierarchy improved but LM degraded — reduce (1-alpha)")
    else:
        print(f"  VERDICT: No improvement — adjust target_gap or increase epochs")

    # Save
    log_path = OUTPUT_DIR / f"training_log_{ts.replace(':','-')}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Log: {log_path}")

    # Save LoRA
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k.lower()}
    lora_path = OUTPUT_DIR / f"lora_weights_{ts.replace(':','-')}.pt"
    torch.save({"lora_state_dict": lora_state, "config": log["config"]}, lora_path)
    print(f"  LoRA: {lora_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return log


if __name__ == "__main__":
    train(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        n_epochs=2,
        batch_size=1,
        lr=5e-5,          # lower for stability
        alpha=0.9,        # 90% LM, 10% topology (gentle regularization)
        target_gap=0.3,   # target_concentration=0.3 (30% variance in top direction)
        lora_rank=8,
        max_length=128,
        eval_every=200,
        log_every=50,
    )
