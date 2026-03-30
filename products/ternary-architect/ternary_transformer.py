#!/usr/bin/env python3
"""Minimal GPT with TernaryLinear layers.

Decoder-only transformer where all linear projections (Q, K, V, O, FFN)
use the configurable weight set {0,1,3}, {-1,0,1}, {0,1,2}, or fp16.

Embeddings and LayerNorm stay fp32 — they're a tiny fraction of params
and need precision for position encoding and normalization.

Two configs:
  small:  ~50M  (d=512, 6 layers, 8 heads)
  medium: ~120M (d=768, 8 layers, 12 heads)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ternary_linear import make_linear


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 512  # max context length
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    weight_set: str = "013"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def param_count_estimate(self) -> int:
        """Rough parameter count (millions)."""
        d = self.n_embd
        # embeddings: vocab * d + block_size * d
        emb = self.vocab_size * d + self.block_size * d
        # per layer: 4*d*d (QKV+O) + 2*4d*d (FFN) + 2*d (layernorms)
        per_layer = 4 * d * d + 2 * 4 * d * d + 2 * d
        # final LN + LM head (tied with embedding)
        final = d + self.vocab_size * d
        total = emb + self.n_layer * per_layer + final
        return total


CONFIGS = {
    "small": GPTConfig(n_layer=6, n_head=8, n_embd=512),      # ~52M
    "medium": GPTConfig(n_layer=8, n_head=12, n_embd=768),     # ~125M
    # Depth-first: same width as small, 8x deeper
    # The 77-dim manifold is the data's intrinsic structure.
    # More layers = more sequential processing = long-range syntax.
    "deep": GPTConfig(n_layer=48, n_head=8, n_embd=512),      # ~184M
}


# ── Attention ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # Q, K, V projections — all ternary
        self.q_proj = make_linear(config.n_embd, config.n_embd,
                                  bias=False, weight_set=config.weight_set)
        self.k_proj = make_linear(config.n_embd, config.n_embd,
                                  bias=False, weight_set=config.weight_set)
        self.v_proj = make_linear(config.n_embd, config.n_embd,
                                  bias=False, weight_set=config.weight_set)
        # Output projection
        self.o_proj = make_linear(config.n_embd, config.n_embd,
                                  bias=False, weight_set=config.weight_set)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(out))


# ── FFN ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up = make_linear(config.n_embd, 4 * config.n_embd,
                              weight_set=config.weight_set)
        self.down = make_linear(4 * config.n_embd, config.n_embd,
                                weight_set=config.weight_set)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x))))


# ── Transformer block ────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────

class TernaryGPT(nn.Module):
    """Decoder-only GPT with configurable weight quantization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head — tied with tok_emb (shared weight, saves params)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                return_hidden: bool = False):
        """
        Parameters
        ----------
        idx : (B, T) token indices
        targets : (B, T) target indices for loss computation
        return_hidden : if True, also return per-layer hidden states

        Returns
        -------
        logits : (B, T, vocab_size)
        loss : scalar if targets provided
        hidden_states : list of (B, T, d) tensors if return_hidden
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x.detach())

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        if return_hidden:
            return logits, loss, hidden_states
        return logits, loss

    def count_params(self) -> dict:
        """Count parameters by category."""
        ternary = 0
        continuous = 0
        for name, p in self.named_parameters():
            if any(k in name for k in ["tok_emb", "pos_emb", "ln", "lm_head"]):
                continuous += p.numel()
            else:
                ternary += p.numel()
        return {
            "ternary": ternary,
            "continuous": continuous,
            "total": ternary + continuous,
            "ternary_pct": ternary / (ternary + continuous) * 100,
        }

    def weight_distributions(self) -> dict[str, dict]:
        """Per-layer weight distribution diagnostic."""
        dists = {}
        for name, module in self.named_modules():
            if hasattr(module, "weight_distribution"):
                dists[name] = module.weight_distribution()
        return dists

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new):
            # Crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx


def build_model(size: str = "small", weight_set: str = "013",
                vocab_size: int = 32000) -> TernaryGPT:
    """Build a model from preset config."""
    config = GPTConfig(**{**CONFIGS[size].__dict__, "weight_set": weight_set,
                          "vocab_size": vocab_size})
    model = TernaryGPT(config)

    # Deep networks need identity init — start as transparent crystal,
    # let training carve the structure from the top down
    if config.n_layer >= 24:
        from ternary_linear import TernaryLinear
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                module.reset_parameters(init_mode="identity")

    return model
