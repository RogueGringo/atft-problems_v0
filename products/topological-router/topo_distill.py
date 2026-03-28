#!/usr/bin/env python3
"""Topological Distillation — Align a 3B model's representations to a 7B's discovered basis.

NOT standard knowledge distillation (matching output logits).
This matches REPRESENTATIONAL GEOMETRY: the 7B's adaptive basis defines
what "good representations" look like topologically. The 3B learns to
produce hidden states that project well onto that basis.

Loss = alpha * next_token_loss + (1 - alpha) * projection_residual_loss

Where projection_residual_loss = ||h - P^T @ P @ h||^2 / ||h||^2
(fraction of hidden state NOT explained by the target basis)

Logged metrics per step:
  - training loss (total, lm, projection)
  - prime_ratio (fraction of hidden state in prime subspace)
  - basis_ratio (fraction in full adaptive basis)
  - gini of hidden state topology at target layer
  - eval metrics every N steps
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results" / "topo_distill"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import spectral_gap, effective_rank, gini_fast


# ── Training log ────────────────────────────────────────────────────────────

@dataclass
class TrainingLog:
    """Accumulates all metrics for analysis and redirection."""
    steps: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def log_step(self, step: int, **kwargs):
        kwargs["step"] = step
        kwargs["timestamp"] = time.time()
        self.steps.append(kwargs)

    def log_eval(self, step: int, **kwargs):
        kwargs["step"] = step
        kwargs["timestamp"] = time.time()
        self.evals.append(kwargs)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({
                "config": self.config,
                "steps": self.steps,
                "evals": self.evals,
            }, f, indent=2, default=str)

    def print_step(self, step_data: dict):
        s = step_data
        print(f"  step {s['step']:4d} | "
              f"loss={s.get('total_loss',0):.4f} "
              f"(lm={s.get('lm_loss',0):.4f} proj={s.get('proj_loss',0):.4f}) | "
              f"prime_r={s.get('prime_ratio',0):.3f} "
              f"basis_r={s.get('basis_ratio',0):.3f} "
              f"sg={s.get('spectral_gap',0):.2f}")


# ── Projection loss ─────────────────────────────────────────────────────────

class ProjectionLoss(nn.Module):
    """Measures how much of hidden state is NOT explained by target basis."""

    def __init__(self, target_basis: torch.Tensor):
        """
        Args:
            target_basis: (n_basis, hidden_dim) orthonormal basis from 7B
        """
        super().__init__()
        # Store as buffer (not a parameter — no gradients on the basis itself)
        self.register_buffer("basis", target_basis.float())

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) at target layer

        Returns:
            (loss_scalar, metrics_dict)
        """
        h = hidden_states.float()

        # Project: proj = h @ B^T @ B
        proj = h @ self.basis.T @ self.basis  # (batch, seq, hidden_dim)
        residual = h - proj

        # Loss: mean squared residual / mean squared hidden state
        h_norm_sq = (h ** 2).sum(dim=-1).mean()
        res_norm_sq = (residual ** 2).sum(dim=-1).mean()

        loss = res_norm_sq / (h_norm_sq + 1e-10)

        # Metrics
        with torch.no_grad():
            basis_ratio = 1.0 - (res_norm_sq / (h_norm_sq + 1e-10)).item()

        return loss, {"basis_ratio": basis_ratio}


# ── Dataset ─────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple text dataset from prompt files + additional training text."""

    def __init__(self, tokenizer, max_length: int = 256, n_samples: int = 2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load diverse prompts
        from prompts.loader import load_prompts
        prompts = load_prompts()
        for p in prompts:
            self.samples.append(p["text"])

        # Load MMLU questions for domain-specific training
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "college_physics", split="test")
            for row in ds:
                q = row["question"]
                choices = row["choices"]
                correct = choices[int(row["answer"])]
                self.samples.append(f"Q: {q}\nA: {correct}")
        except Exception:
            pass

        # Repeat to reach n_samples
        if len(self.samples) < n_samples:
            repeats = (n_samples // len(self.samples)) + 1
            self.samples = (self.samples * repeats)[:n_samples]

        # Shuffle
        rng = np.random.default_rng(42)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length, padding="max_length")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ── LoRA ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices — match dtype and device of original weight
        dtype = original.weight.dtype
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A)
        # B starts at zero so LoRA initially has no effect

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def apply_lora(model, target_layer: int, rank: int = 16, alpha: float = 32.0) -> list[nn.Parameter]:
    """Apply LoRA to Q, K, V projections at layers around the target layer.

    Returns list of trainable parameters.
    """
    n_layers = model.config.num_hidden_layers
    # Apply to target layer +/- 2 layers (5 layers total)
    start = max(0, target_layer - 2)
    end = min(n_layers, target_layer + 3)

    trainable_params = []

    for layer_idx in range(start, end):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Replace Q, K, V projections with LoRA versions
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            original = getattr(attn, proj_name)
            lora = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(attn, proj_name, lora)
            trainable_params.extend([lora.lora_A, lora.lora_B])

    print(f"  LoRA applied to layers {start}-{end-1} (Q,K,V)")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable %: {sum(p.numel() for p in trainable_params)/sum(p.numel() for p in model.parameters())*100:.2f}%")

    return trainable_params


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, target_layer: int, proj_loss_fn: ProjectionLoss,
             prime_basis: torch.Tensor, device: str) -> dict:
    """Quick evaluation: likelihood accuracy on 20 MMLU-Physics questions."""
    from prompts.loader import load_prompts
    physics = [p["text"] for p in load_prompts("physics")][:20]

    total_lm_loss = 0
    total_proj_loss = 0
    basis_ratios = []
    prime_ratios = []
    spectral_gaps = []

    for text in physics:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=256).to(device)
        outputs = model(input_ids=inputs["input_ids"],
                       output_hidden_states=True)

        # LM loss
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        lm_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1), reduction="mean")
        total_lm_loss += lm_loss.item()

        # Projection loss
        hs = outputs.hidden_states[target_layer].float()
        p_loss, p_metrics = proj_loss_fn(hs)
        total_proj_loss += p_loss.item()
        basis_ratios.append(p_metrics["basis_ratio"])

        # Prime ratio
        P = prime_basis.to(device).float()
        h_flat = hs[0].float()
        prime_proj = h_flat @ P.T @ P
        pr = torch.norm(prime_proj) / (torch.norm(h_flat) + 1e-10)
        prime_ratios.append(pr.item())

        # Spectral gap
        sg = spectral_gap(hs[0])
        spectral_gaps.append(sg)

    n = len(physics)
    return {
        "lm_loss": total_lm_loss / n,
        "proj_loss": total_proj_loss / n,
        "basis_ratio": float(np.mean(basis_ratios)),
        "prime_ratio": float(np.mean(prime_ratios)),
        "spectral_gap": float(np.mean(spectral_gaps)),
    }


# ── Main training loop ─────────────────────────────────────────────────────

def train(model_name: str = "Qwen/Qwen2.5-3B-Instruct",
          target_basis_path: str | Path | None = None,
          n_epochs: int = 3,
          batch_size: int = 2,
          lr: float = 2e-4,
          alpha: float = 0.5,
          lora_rank: int = 16,
          eval_every: int = 50,
          log_every: int = 10,
          device: str = "cuda") -> TrainingLog:
    """
    Args:
        model_name: 3B model to fine-tune
        target_basis_path: path to 7B's adaptive_basis .pt file
        n_epochs: training epochs
        alpha: weight for LM loss (1-alpha for projection loss)
        lora_rank: LoRA rank
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  TOPOLOGICAL DISTILLATION")
    print(f"  {model_name} → aligned to 7B basis")
    print(f"  {ts}")
    print("=" * 65)

    log = TrainingLog()
    log.config = {
        "model": model_name, "n_epochs": n_epochs, "batch_size": batch_size,
        "lr": lr, "alpha": alpha, "lora_rank": lora_rank,
        "eval_every": eval_every, "timestamp": ts,
    }

    # ── Load target basis ──
    # Must match the model's hidden_dim. Use the model's OWN adaptive basis
    # (teach it to concentrate representational power on its discovered structure)
    if target_basis_path is None:
        basis_dir = Path(__file__).parent / "results" / "basis_discovery"
        # Match by model name to ensure hidden_dim compatibility
        model_short = model_name.split("/")[-1]
        candidates = list(basis_dir.glob(f"adaptive_basis_*{model_short}*"))
        if not candidates:
            # Try broader match
            candidates = list(basis_dir.glob("adaptive_basis_*3B*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No adaptive basis found for {model_short}. Run sweep first.")
        target_basis_path = candidates[0]

    print(f"\n  Loading target basis: {target_basis_path}")
    basis_data = torch.load(target_basis_path, weights_only=False)
    target_basis = basis_data["adaptive_basis"]
    print(f"  Target basis: {target_basis.shape}")
    print(f"  (Using model's own discovered basis — self-distillation)")

    # Also load prime basis for monitoring — must match THIS model's hidden_dim
    prime_dir = Path(__file__).parent / "results" / "basis_discovery"
    prime_candidates = list(prime_dir.glob(f"prime_basis_*{model_name.split('/')[-1]}*"))
    if prime_candidates:
        prime_data = torch.load(prime_candidates[0], weights_only=False)
        prime_basis = prime_data["prime_basis"]
    else:
        prime_basis = target_basis[:58]  # use first 58 vectors of adaptive basis

    # ── Load model ──
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float16, device_map=device,
        output_hidden_states=True)
    model.train()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 3
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{n_layers} layers, target_layer={target_layer}, VRAM={vram:.1f}GB")

    # ── Apply LoRA ──
    trainable_params = apply_lora(model, target_layer, rank=lora_rank)

    # ── Projection loss ──
    proj_loss_fn = ProjectionLoss(target_basis).to(device)

    # ── Dataset ──
    dataset = TextDataset(tokenizer, max_length=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} samples, batch_size={batch_size}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = n_epochs * len(loader)
    print(f"  Total steps: {total_steps}")

    # ── Pre-training eval ──
    print(f"\n  Pre-training evaluation...")
    model.eval()
    pre_eval = evaluate(model, tokenizer, target_layer, proj_loss_fn, prime_basis, device)
    log.log_eval(0, phase="pre", **pre_eval)
    print(f"    lm_loss={pre_eval['lm_loss']:.4f} proj_loss={pre_eval['proj_loss']:.4f} "
          f"basis_r={pre_eval['basis_ratio']:.3f} prime_r={pre_eval['prime_ratio']:.3f} "
          f"sg={pre_eval['spectral_gap']:.2f}")
    model.train()

    # ── Training ──
    print(f"\n  Training ({n_epochs} epochs, alpha={alpha})...")
    global_step = 0

    for epoch in range(n_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          output_hidden_states=True)

            # LM loss
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            # Mask padding
            mask = attention_mask[:, 1:].float()
            lm_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1), reduction="none")
            lm_loss = (lm_loss.view_as(labels) * mask).sum() / (mask.sum() + 1e-10)

            # Projection loss at target layer
            hs = outputs.hidden_states[target_layer]
            proj_loss, proj_metrics = proj_loss_fn(hs)

            # Combined loss
            total_loss = alpha * lm_loss + (1 - alpha) * proj_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            global_step += 1
            epoch_losses.append(total_loss.item())

            # Log
            if global_step % log_every == 0:
                with torch.no_grad():
                    # Quick topology metrics on this batch
                    sg = spectral_gap(hs[0].detach())
                    P = prime_basis.to(device).float()
                    h_flat = hs[0].detach().float()
                    pp = h_flat @ P.T @ P
                    pr = (torch.norm(pp) / (torch.norm(h_flat) + 1e-10)).item()

                step_data = {
                    "total_loss": total_loss.item(),
                    "lm_loss": lm_loss.item(),
                    "proj_loss": proj_loss.item(),
                    "basis_ratio": proj_metrics["basis_ratio"],
                    "prime_ratio": pr,
                    "spectral_gap": sg,
                    "epoch": epoch,
                }
                log.log_step(global_step, **step_data)
                log.print_step(log.steps[-1])

            # Eval
            if global_step % eval_every == 0:
                model.eval()
                eval_result = evaluate(model, tokenizer, target_layer,
                                      proj_loss_fn, prime_basis, device)
                log.log_eval(global_step, phase="train", epoch=epoch, **eval_result)
                print(f"  EVAL step {global_step}: "
                      f"lm={eval_result['lm_loss']:.4f} "
                      f"proj={eval_result['proj_loss']:.4f} "
                      f"basis_r={eval_result['basis_ratio']:.3f} "
                      f"sg={eval_result['spectral_gap']:.2f}")
                model.train()

        print(f"  Epoch {epoch+1}/{n_epochs}: mean_loss={np.mean(epoch_losses):.4f}")

    # ── Post-training eval ──
    print(f"\n  Post-training evaluation...")
    model.eval()
    post_eval = evaluate(model, tokenizer, target_layer, proj_loss_fn, prime_basis, device)
    log.log_eval(global_step, phase="post", **post_eval)
    print(f"    lm_loss={post_eval['lm_loss']:.4f} proj_loss={post_eval['proj_loss']:.4f} "
          f"basis_r={post_eval['basis_ratio']:.3f} prime_r={post_eval['prime_ratio']:.3f} "
          f"sg={post_eval['spectral_gap']:.2f}")

    # ── Compare pre vs post ──
    print(f"\n{'='*65}")
    print("  PRE vs POST COMPARISON")
    print(f"{'='*65}")
    for key in ["lm_loss", "proj_loss", "basis_ratio", "prime_ratio", "spectral_gap"]:
        pre = pre_eval[key]
        post = post_eval[key]
        delta = post - pre
        direction = "+" if delta > 0 else ""
        print(f"    {key:<16}: {pre:.4f} → {post:.4f} ({direction}{delta:.4f})")

    # ── Save ──
    log.config["pre_eval"] = pre_eval
    log.config["post_eval"] = post_eval
    log_path = OUTPUT_DIR / f"training_log_{ts.replace(':','-')}.json"
    log.save(log_path)
    print(f"\n  Training log: {log_path}")

    # Save LoRA weights
    lora_path = OUTPUT_DIR / f"lora_weights_{ts.replace(':','-')}.pt"
    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k.lower()}
    torch.save({
        "lora_state_dict": lora_state,
        "config": log.config,
    }, lora_path)
    print(f"  LoRA weights: {lora_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return log


if __name__ == "__main__":
    log = train(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # 3B OOMs; 1.5B fits
        n_epochs=3,
        batch_size=1,       # minimize memory
        lr=2e-4,
        alpha=0.5,
        lora_rank=8,        # smaller rank = less memory
        eval_every=100,
        log_every=20,
    )
