#!/usr/bin/env python3
"""HarmonicStack — Spectral decomposition engine for structured information.

Two-stage architecture:
  prism: wide, shallow {0,1,3} transformer — learn the crystal structure
  full:  frozen prism + SpectralRouter + BandAnalyzers — read the bands

The prism crystallises input into three spectral bands:
  void     (w=0): structured absence, dark dimensions
  identity (w=1): signal highways, transparent dimensions
  prime    (w=3): amplified, irreducible structure

Stage 2 reads these bands independently and recombines for output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ternary_linear import make_linear, TernaryLinear
from ternary_transformer import GPTConfig, Block


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class HarmonicConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_prism_layers: int = 1       # 1 or 3 for experiments
    n_embd: int = 2048            # wide
    n_head: int = 16              # 2048/16 = 128 head_dim
    dropout: float = 0.0
    weight_set: str = "013"
    analyzer_width: int = 512


# ── SpectralRouter ─────────────────────────────────────────────────────────

class SpectralRouter(nn.Module):
    """Static router based on the crystallized weight structure.

    Reads a TernaryLinear weight matrix and assigns each output dimension
    to one of three bands (void / identity / prime) based on which weight
    value dominates that column.

    The routing table is fixed at build time — routing is structural,
    not learned.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd

        # Placeholder buffers — populated by build_routing_table()
        self.register_buffer("void_dims",     torch.zeros(0, dtype=torch.long))
        self.register_buffer("identity_dims", torch.zeros(0, dtype=torch.long))
        self.register_buffer("prime_dims",    torch.zeros(0, dtype=torch.long))

    def build_routing_table(self, prism_layer: TernaryLinear) -> None:
        """Assign each output dimension to its dominant band.

        For output dim j, count the fraction of inputs that pass through
        w=0, w=1, or w=3.  The band with the highest count owns that dim.

        Parameters
        ----------
        prism_layer : TernaryLinear
            The layer whose crystallized weights define the routing.
            Shape: (out_features, in_features)
        """
        with torch.no_grad():
            w_q = prism_layer.get_quantized_weight()   # (out, in)
            out_features = w_q.shape[0]

            # Count occurrences of each weight value per output dim
            n0 = (w_q == 0.0).sum(dim=1).float()   # (out,)
            n1 = (w_q == 1.0).sum(dim=1).float()
            n3 = (w_q == 3.0).sum(dim=1).float()

            counts = torch.stack([n0, n1, n3], dim=1)   # (out, 3)
            dominant = counts.argmax(dim=1)              # (out,)  0/1/2

            void_dims     = (dominant == 0).nonzero(as_tuple=True)[0]
            identity_dims = (dominant == 1).nonzero(as_tuple=True)[0]
            prime_dims    = (dominant == 2).nonzero(as_tuple=True)[0]

        self.void_dims     = void_dims
        self.identity_dims = identity_dims
        self.prime_dims    = prime_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split (B, T, d) into three tensors by band membership.

        Returns
        -------
        void_x     : (B, T, |void_dims|)
        identity_x : (B, T, |identity_dims|)
        prime_x    : (B, T, |prime_dims|)
        """
        return (
            x[:, :, self.void_dims],
            x[:, :, self.identity_dims],
            x[:, :, self.prime_dims],
        )

    def band_sizes(self) -> dict[str, int]:
        return {
            "void":     self.void_dims.numel(),
            "identity": self.identity_dims.numel(),
            "prime":    self.prime_dims.numel(),
        }


# ── BandAnalyzer ───────────────────────────────────────────────────────────

class BandAnalyzer(nn.Module):
    """Single-layer spectral analyzer for one band.

    TernaryLinear projection → LayerNorm → GELU
    Maps band_width → analyzer_width features.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_set: str = "013"):
        super().__init__()
        self.proj = make_linear(in_features, out_features,
                                bias=False, weight_set=weight_set)
        self.ln   = nn.LayerNorm(out_features)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(self.proj(x)))


# ── HarmonicStack ──────────────────────────────────────────────────────────

class HarmonicStack(nn.Module):
    """Spectral decomposition engine.

    stage="prism"
        Wide shallow ternary transformer.  Trains normally.
        forward → (logits, loss) or (logits, loss, hidden_states)

    stage="full"
        Prism is frozen.  SpectralRouter + BandAnalyzers + new LM head.
        Call build_stage2(prism_ternary_layer) to wire up the router.
        forward → (logits, loss) or (logits, loss, hidden_states)
    """

    def __init__(self, config: HarmonicConfig, stage: str = "prism"):
        super().__init__()
        assert stage in ("prism", "full"), f"Unknown stage: {stage}"
        self.config = config
        self.stage  = stage

        # ── Prism (shared between both stages) ──────────────────────────
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop    = nn.Dropout(config.dropout)

        # Build a GPTConfig that matches our HarmonicConfig
        gpt_cfg = GPTConfig(
            vocab_size  = config.vocab_size,
            block_size  = config.block_size,
            n_layer     = config.n_prism_layers,
            n_head      = config.n_head,
            n_embd      = config.n_embd,
            dropout     = config.dropout,
            weight_set  = config.weight_set,
        )
        self.prism_blocks = nn.ModuleList(
            [Block(gpt_cfg) for _ in range(config.n_prism_layers)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        # ── Stage-specific heads ─────────────────────────────────────────
        if stage == "prism":
            # LM head tied with tok_emb
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight

            # Stage-2 components — not built yet
            self.router        = None
            self.band_analyzers= None

        else:  # stage == "full"
            # Prism weights will be frozen after build_stage2
            # Router and analyzers start empty — call build_stage2()
            self.router         = SpectralRouter(config.n_embd)
            self.band_analyzers = None   # built by build_stage2
            # LM head will be created by build_stage2 (3*analyzer_width → vocab)
            self.lm_head = None

        self.apply(self._init_weights)

    # ── Weight init ────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Stage-2 wiring ─────────────────────────────────────────────────────

    def get_last_prism_ternary(self) -> TernaryLinear | None:
        """Return the last prism block's mlp.down if it is a TernaryLinear."""
        last_block = self.prism_blocks[-1]
        down = last_block.mlp.down
        if isinstance(down, TernaryLinear):
            return down
        return None

    def build_stage2(self, prism_ternary_layer: TernaryLinear) -> None:
        """Wire up router + analyzers from a crystallized prism layer.

        Freezes all prism parameters, then builds:
          - SpectralRouter routing table
          - Three BandAnalyzers (one per band, input width = band_size)
          - New LM head: 3*analyzer_width → vocab_size  (NOT tied)
        """
        assert self.stage == "full", "build_stage2 only valid in stage='full'"

        # 1. Freeze prism
        for p in self.tok_emb.parameters():
            p.requires_grad = False
        for p in self.pos_emb.parameters():
            p.requires_grad = False
        for block in self.prism_blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False

        # 2. Build routing table
        self.router.build_routing_table(prism_ternary_layer)
        sizes = self.router.band_sizes()

        # 3. Build three BandAnalyzers
        aw = self.config.analyzer_width
        ws = self.config.weight_set
        self.band_analyzers = nn.ModuleList([
            BandAnalyzer(max(1, sizes["void"]),     aw, weight_set=ws),
            BandAnalyzer(max(1, sizes["identity"]), aw, weight_set=ws),
            BandAnalyzer(max(1, sizes["prime"]),    aw, weight_set=ws),
        ])

        # 4. New LM head (not tied — different dimension)
        self.lm_head = nn.Linear(3 * aw, self.config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        idx:           torch.Tensor,
        targets:       torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """
        Parameters
        ----------
        idx     : (B, T) token indices
        targets : (B, T) or None — for loss computation
        return_hidden : if True, return per-block hidden states too

        Returns
        -------
        stage="prism":
            (logits, loss) or (logits, loss, hidden_states)
        stage="full":
            same signature — hidden_states are from the prism blocks
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence {T} > block_size {self.config.block_size}"
        )

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x   = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states = []
        for block in self.prism_blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x.detach())

        x = self.ln_f(x)   # (B, T, n_embd)

        if self.stage == "prism":
            logits = self.lm_head(x)   # (B, T, vocab_size)

        else:  # stage == "full"
            assert self.band_analyzers is not None and self.lm_head is not None, (
                "Call build_stage2() before running in stage='full'"
            )
            void_x, ident_x, prime_x = self.router(x)
            analyzed = torch.cat([
                self.band_analyzers[0](void_x),
                self.band_analyzers[1](ident_x),
                self.band_analyzers[2](prime_x),
            ], dim=-1)   # (B, T, 3*analyzer_width)
            logits = self.lm_head(analyzed)   # (B, T, vocab_size)

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

    # ── Diagnostics ────────────────────────────────────────────────────────

    def count_params(self) -> dict:
        """Count parameters by category (mirrors TernaryGPT.count_params)."""
        ternary    = 0
        continuous = 0
        for name, p in self.named_parameters():
            if any(k in name for k in
                   ["tok_emb", "pos_emb", "ln", "lm_head"]):
                continuous += p.numel()
            else:
                ternary += p.numel()
        total = ternary + continuous
        return {
            "ternary":    ternary,
            "continuous": continuous,
            "total":      total,
            "ternary_pct": ternary / total * 100 if total > 0 else 0.0,
        }

    def weight_distributions(self) -> dict[str, dict]:
        """Per-layer weight distribution diagnostic."""
        dists = {}
        for name, module in self.named_modules():
            if hasattr(module, "weight_distribution"):
                dists[name] = module.weight_distribution()
        return dists

    # ── Generation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,
        max_new:     int   = 100,
        temperature: float = 0.8,
        top_k:       int   = 50,
    ) -> torch.Tensor:
        """Autoregressive generation (same interface as TernaryGPT)."""
        for _ in range(max_new):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

        return idx
