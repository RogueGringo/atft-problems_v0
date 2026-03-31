# Harmonic Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spectral decomposition engine that validates whether the 22/42/36 crystal constant holds in wide+shallow architectures, then tests whether spectral bands have independent internal structure.

**Architecture:** Two-stage Harmonic Stack. Stage 1 (Prism) is a wide shallow {0,1,3} transformer. Stage 2 (Analyzers) is three parallel {0,1,3} layers receiving routed spectral bands. Reuses existing `TernaryLinear` and measurement infrastructure.

**Tech Stack:** PyTorch, existing `ternary_linear.py` (`TernaryLinear`, `make_linear`), existing `ternary_transformer.py` (reuse `CausalSelfAttention`, `MLP`, `Block`), HuggingFace `datasets` + `transformers` (GPT-2 tokenizer), existing `topo_measures.py` (`effective_rank`, `spectral_gap`, `gini_fast`), existing `measure.py` (`iterative_inference`, `zero_mask_topology`).

---

## File Structure

```
products/ternary-architect/
  harmonic_stack.py      (NEW)  — HarmonicStack model: prism + router + analyzers + branch
  run_harmonic.py        (NEW)  — Training script for Experiments 1 and 2
  ternary_linear.py      (NO CHANGES)
  ternary_transformer.py (NO CHANGES)
  run_long.py            (NO CHANGES)
```

- `harmonic_stack.py`: Model class with configurable prism width/depth, static router, three analyzer layers, and language modeling branch head. Imports `make_linear` from `ternary_linear.py` and reuses `CausalSelfAttention`, `MLP`, `Block` from `ternary_transformer.py`.
- `run_harmonic.py`: Training loop with two modes (`--stage prism` for Experiment 1, `--stage full` for Experiment 2). Reuses `TextChunked`, `split_dataset`, `measure_perplexity`, `measure_topology`, `weight_stats` from `run_long.py`.

---

### Task 1: Build the HarmonicStack model (prism-only mode)

**Files:**
- Create: `products/ternary-architect/harmonic_stack.py`

This task builds the model class that supports both Experiment 1 (prism only) and Experiment 2 (full stack). For Experiment 1, only the prism and branch head are active.

- [ ] **Step 1: Create `harmonic_stack.py` with HarmonicStack class**

```python
#!/usr/bin/env python3
"""Harmonic Stack — spectral decomposition engine.

Two-stage architecture:
  Stage 1 (Prism): Wide shallow {0,1,3} transformer. Produces the crystal.
  Router: Static — reads crystallized weights, routes dims by band (0/1/3).
  Stage 2 (Analyzers): Three parallel {0,1,3} layers per spectral band.
  Branch: Application head (language modeling for PPL validation).

The crystal IS the output. PPL validates the measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ternary_linear import TernaryLinear, make_linear
from ternary_transformer import GPTConfig, Block


@dataclass
class HarmonicConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_prism_layers: int = 1      # 1 or 3 for Experiment 1
    n_embd: int = 2048           # wide
    n_head: int = 16             # 2048 / 16 = 128 head_dim
    dropout: float = 0.0
    weight_set: str = "013"
    # Stage 2 (only used when stage="full")
    analyzer_width: int = 512    # output width of each analyzer


# ── Router ───────────────────────────────────────────────────────────────

class SpectralRouter(nn.Module):
    """Static router — reads the prism crystal and routes dims by band.

    For each output dimension j of the last prism block's MLP down-projection,
    count what fraction of incoming weights are 0, 1, or 3.
    Assign dim j to the band with the highest fraction.

    No learned parameters. The crystal IS the routing table.
    """

    def __init__(self):
        super().__init__()
        # These get populated by build_routing_table()
        self.register_buffer("void_dims", torch.zeros(0, dtype=torch.long))
        self.register_buffer("identity_dims", torch.zeros(0, dtype=torch.long))
        self.register_buffer("prime_dims", torch.zeros(0, dtype=torch.long))

    def build_routing_table(self, prism_layer: TernaryLinear):
        """Read a crystallized TernaryLinear and assign each output dim to a band."""
        wq = prism_layer.get_quantized_weight()  # (out_features, in_features)
        n_out, n_in = wq.shape

        void_idx, identity_idx, prime_idx = [], [], []
        for j in range(n_out):
            col = wq[j]
            n0 = (col == 0).sum().item()
            n1 = (col == 1).sum().item()
            n3 = (col == 3).sum().item()
            # Assign to dominant band
            if n3 >= n1 and n3 >= n0:
                prime_idx.append(j)
            elif n0 >= n1:
                void_idx.append(j)
            else:
                identity_idx.append(j)

        self.void_dims = torch.tensor(void_idx, dtype=torch.long)
        self.identity_dims = torch.tensor(identity_idx, dtype=torch.long)
        self.prime_dims = torch.tensor(prime_idx, dtype=torch.long)

    def forward(self, x: torch.Tensor):
        """Route (B, T, d) into three bands based on dim assignment."""
        return (
            x[:, :, self.void_dims],      # (B, T, n_void)
            x[:, :, self.identity_dims],   # (B, T, n_identity)
            x[:, :, self.prime_dims],      # (B, T, n_prime)
        )

    def band_sizes(self) -> dict[str, int]:
        return {
            "void": len(self.void_dims),
            "identity": len(self.identity_dims),
            "prime": len(self.prime_dims),
        }


# ── Analyzers ────────────────────────────────────────────────────────────

class BandAnalyzer(nn.Module):
    """Single {0,1,3} layer that analyzes one spectral band."""

    def __init__(self, in_features: int, out_features: int,
                 weight_set: str = "013"):
        super().__init__()
        self.layer = make_linear(in_features, out_features,
                                 bias=False, weight_set=weight_set)
        self.ln = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(self.layer(x)))


# ── Full Model ───────────────────────────────────────────────────────────

class HarmonicStack(nn.Module):
    """Spectral decomposition engine.

    stage="prism": Only Stage 1 + branch head (Experiment 1).
    stage="full":  Stage 1 (frozen) + router + analyzers + branch (Experiment 2).
    """

    def __init__(self, config: HarmonicConfig, stage: str = "prism"):
        super().__init__()
        self.config = config
        self.stage = stage

        # Transducer (text): embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Stage 1: Prism — wide shallow transformer blocks
        prism_config = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_prism_layers,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            weight_set=config.weight_set,
        )
        self.prism = nn.ModuleList([Block(prism_config)
                                    for _ in range(config.n_prism_layers)])
        self.ln_prism = nn.LayerNorm(config.n_embd)

        # Stage 2: Router + Analyzers (only built for stage="full")
        self.router = SpectralRouter()
        self.void_analyzer = None
        self.identity_analyzer = None
        self.prime_analyzer = None

        if stage == "full":
            # Analyzer widths will be set after routing table is built
            # Placeholder — call build_stage2() after loading prism checkpoint
            pass

        # Branch head: LM projection
        # For prism stage: project from n_embd to vocab
        # For full stage: project from 3 * analyzer_width to vocab
        if stage == "prism":
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # weight tying
        else:
            self.lm_head = None  # built by build_stage2

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_stage2(self, prism_ternary_layer: TernaryLinear):
        """Build router and analyzers from a crystallized prism layer.

        Call this after loading a prism checkpoint for Experiment 2.
        Reads the crystal, builds routing table, constructs analyzers
        with correct input widths.
        """
        self.router.build_routing_table(prism_ternary_layer)
        sizes = self.router.band_sizes()
        aw = self.config.analyzer_width

        self.void_analyzer = BandAnalyzer(sizes["void"], aw,
                                          self.config.weight_set)
        self.identity_analyzer = BandAnalyzer(sizes["identity"], aw,
                                              self.config.weight_set)
        self.prime_analyzer = BandAnalyzer(sizes["prime"], aw,
                                           self.config.weight_set)

        # LM head from concatenated analyzer outputs
        self.lm_head = nn.Linear(3 * aw, self.config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                return_hidden: bool = False):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states = []
        for block in self.prism:
            x = block(x)
            if return_hidden:
                hidden_states.append(x.detach())

        x = self.ln_prism(x)

        if self.stage == "full" and self.void_analyzer is not None:
            # Route through spectral bands
            void_x, ident_x, prime_x = self.router(x)
            void_out = self.void_analyzer(void_x)
            ident_out = self.identity_analyzer(ident_x)
            prime_out = self.prime_analyzer(prime_x)
            x = torch.cat([void_out, ident_out, prime_out], dim=-1)

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
            "ternary_pct": ternary / max(1, ternary + continuous) * 100,
        }

    def get_last_prism_ternary(self) -> TernaryLinear | None:
        """Return the last TernaryLinear in the prism (for router building)."""
        last_block = self.prism[-1]
        return last_block.mlp.down if isinstance(last_block.mlp.down, TernaryLinear) else None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new):
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
```

- [ ] **Step 2: Verify module imports work**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems/products/ternary-architect && python3 -c "
from harmonic_stack import HarmonicStack, HarmonicConfig
cfg = HarmonicConfig(n_prism_layers=1, n_embd=2048, n_head=16)
model = HarmonicStack(cfg, stage='prism')
print(model.count_params())
x = __import__('torch').randint(0, 100, (1, 32))
logits, loss = model(x)
print('logits:', logits.shape)
print('OK')
"
```

Expected: prints param counts, logits shape `(1, 32, 50257)`, and "OK".

- [ ] **Step 3: Commit**

```bash
git add products/ternary-architect/harmonic_stack.py
git commit -m "feat: HarmonicStack model — prism + router + analyzers"
```

---

### Task 2: Build the run_harmonic.py training script (Experiment 1)

**Files:**
- Create: `products/ternary-architect/run_harmonic.py`

Reuses dataset loading, measurement, and training infrastructure patterns from `run_long.py` but adapted for the HarmonicStack model.

- [ ] **Step 1: Create `run_harmonic.py`**

```python
#!/usr/bin/env python3
"""Harmonic Stack experiments.

Experiment 1 (--stage prism): Validate wide+shallow crystal.
  Does 22/42/36 hold at width 2048 with 1-3 layers?

Experiment 2 (--stage full): Router + Analyzers.
  Do the three spectral bands have different internal structure?

Usage:
    python run_harmonic.py --stage prism --width 2048 --prism_layers 1 --dataset wikitext
    python run_harmonic.py --stage full --prism_checkpoint results/harmonic_prism_1L_2048/model.pt --dataset wikitext
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear
from harmonic_stack import HarmonicStack, HarmonicConfig
from run_long import TextChunked, split_dataset, measure_perplexity, weight_stats
from topo_measures import effective_rank, spectral_gap, gini_fast


def measure_topology_harmonic(model, sample_batch, device):
    """Topology measurements for HarmonicStack."""
    model.eval()
    with torch.no_grad():
        x = sample_batch[0][:1].to(device)
        _, _, hs = model(x, return_hidden=True)
    results = {}
    for i, h in enumerate(hs):
        h0 = h[0]
        results[f"prism_layer_{i}"] = {
            "eff_rank": effective_rank(h0),
            "spectral_gap": spectral_gap(h0),
            "gini_sv": gini_fast(torch.linalg.svdvals(h0.float().cpu()).numpy()),
        }
    for k in ["eff_rank", "spectral_gap", "gini_sv"]:
        vals = [v[k] for v in results.values()]
        results[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
    model.train()
    return results


def per_layer_crystal(model):
    """Per-layer weight distribution for the prism."""
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, TernaryLinear):
            wq = m.get_quantized_weight()
            n = wq.numel()
            layers[name] = {
                "w0": (wq == 0).sum().item() / n,
                "w1": (wq == 1).sum().item() / n,
                "w3": (wq == 3).sum().item() / n,
            }
    return layers


def analyzer_crystals(model):
    """Per-analyzer weight distribution (Experiment 2 only)."""
    crystals = {}
    for label, analyzer in [("void", model.void_analyzer),
                             ("identity", model.identity_analyzer),
                             ("prime", model.prime_analyzer)]:
        if analyzer is None:
            continue
        if isinstance(analyzer.layer, TernaryLinear):
            wq = analyzer.layer.get_quantized_weight()
            n = wq.numel()
            crystals[label] = {
                "w0": (wq == 0).sum().item() / n,
                "w1": (wq == 1).sum().item() / n,
                "w3": (wq == 3).sum().item() / n,
            }
    return crystals


def get_output_dir(args):
    tag = f"harmonic_{args.stage}_{args.prism_layers}L_{args.width}"
    d = Path(__file__).parent / "results" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser(description="Harmonic Stack experiments")
    parser.add_argument("--stage", choices=["prism", "full"], default="prism",
                        help="prism = Experiment 1, full = Experiment 2")
    parser.add_argument("--width", type=int, default=2048,
                        help="Prism width (n_embd)")
    parser.add_argument("--n_heads", type=int, default=16,
                        help="Number of attention heads in prism")
    parser.add_argument("--prism_layers", type=int, default=1,
                        help="Number of transformer blocks in the prism")
    parser.add_argument("--analyzer_width", type=int, default=512,
                        help="Output width of each analyzer (Experiment 2)")
    parser.add_argument("--prism_checkpoint", type=str, default=None,
                        help="Path to trained prism model.pt (required for --stage full)")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["tinystories", "wikitext", "kant", "sep"])
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ternary_decay", type=float, default=0.01)
    args = parser.parse_args()

    OUTPUT_DIR = get_output_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    full_ds = TextChunked(tokenizer, max_length=args.max_length,
                          n_samples=args.n_samples,
                          dataset_name=args.dataset)
    train_ds, test_ds = split_dataset(full_ds)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)
    sample_batch = next(iter(train_loader))

    # Model
    config = HarmonicConfig(
        vocab_size=tokenizer.vocab_size,
        n_prism_layers=args.prism_layers,
        n_embd=args.width,
        n_head=args.n_heads,
        analyzer_width=args.analyzer_width,
    )

    if args.stage == "prism":
        model = HarmonicStack(config, stage="prism").to(device)
    else:
        assert args.prism_checkpoint, "--prism_checkpoint required for --stage full"
        # Load prism, freeze it, build stage 2
        model = HarmonicStack(config, stage="full").to(device)
        prism_state = torch.load(args.prism_checkpoint, map_location=device,
                                 weights_only=True)
        # Load only prism + embedding weights (not lm_head which differs)
        missing, unexpected = model.load_state_dict(prism_state, strict=False)
        print(f"  Loaded prism: {len(missing)} missing, {len(unexpected)} unexpected")
        # Freeze prism
        for name, p in model.named_parameters():
            if "prism" in name or "tok_emb" in name or "pos_emb" in name or "ln_prism" in name:
                p.requires_grad_(False)
        # Build router and analyzers from crystallized prism
        prism_ternary = model.get_last_prism_ternary()
        assert prism_ternary is not None, "Could not find prism TernaryLinear"
        model.build_stage2(prism_ternary)
        # Move new modules to device
        model.void_analyzer = model.void_analyzer.to(device)
        model.identity_analyzer = model.identity_analyzer.to(device)
        model.prime_analyzer = model.prime_analyzer.to(device)
        model.lm_head = model.lm_head.to(device)
        sizes = model.router.band_sizes()
        print(f"  Router bands: void={sizes['void']} identity={sizes['identity']} prime={sizes['prime']}")

    params = model.count_params()
    print(f"\n{params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")

    # Optimizer — separate ternary and continuous params
    ternary_params = []
    continuous_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["tok_emb", "pos_emb", "ln", "lm_head"]):
            continuous_params.append(p)
        else:
            ternary_params.append(p)

    optim_groups = []
    if ternary_params:
        optim_groups.append({"params": ternary_params, "weight_decay": args.ternary_decay})
        print(f"  Ternary params (L2={args.ternary_decay}): {sum(p.numel() for p in ternary_params):,}")
    optim_groups.append({"params": continuous_params, "weight_decay": 0.01})
    print(f"  Continuous params (L2=0.01): {sum(p.numel() for p in continuous_params):,}")

    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
    accum = max(1, args.effective_batch // args.batch_size)

    # LR schedule: warmup + cosine decay
    warmup_steps = min(2000, args.max_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    log = {
        "experiment": f"harmonic_{args.stage}",
        "config": vars(args),
        "params": params,
        "checkpoints": [],
        "steps": [],
    }

    step = 0
    start = time.time()
    print(f"\nTraining {args.max_steps} steps. Checkpoint every 1000.\n")
    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for bx, by in train_loader:
            if step >= args.max_steps:
                break
            bx, by = bx.to(device), by.to(device)
            _, loss = model(bx, targets=by)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Quick log
            if step % 200 == 0:
                elapsed = time.time() - start
                ppl = torch.exp(loss).item()
                ppl_s = f"{ppl:.1f}" if ppl < 1e5 else f"{ppl:.0e}"
                log["steps"].append({"step": step, "loss": loss.item(),
                                     "ppl": min(ppl, 1e7), "elapsed": elapsed})
                print(f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl_s:>10s} | "
                      f"{elapsed:.0f}s", flush=True)

            # Full checkpoint
            if step > 0 and step % 1000 == 0:
                train_ppl = measure_perplexity(model, train_loader, device)
                test_ppl = measure_perplexity(model, test_loader, device)
                topo = measure_topology_harmonic(model, sample_batch, device)
                wstats = weight_stats(model)
                gen_gap = test_ppl["ppl"] / train_ppl["ppl"] if train_ppl["ppl"] > 0 else 999

                ckpt = {
                    "step": step,
                    "train_ppl": train_ppl,
                    "test_ppl": test_ppl,
                    "gen_gap": gen_gap,
                    "eff_rank": topo.get("mean_eff_rank", 0),
                    "spectral_gap": topo.get("mean_spectral_gap", 0),
                    "weight_dist": wstats,
                    "per_layer_crystal": per_layer_crystal(model),
                    "elapsed": time.time() - start,
                }

                # Add analyzer crystals for Experiment 2
                if args.stage == "full":
                    ckpt["analyzer_crystals"] = analyzer_crystals(model)

                log["checkpoints"].append(ckpt)

                print(f"\n    CKPT {step:6d} | train {train_ppl['ppl']:.1f} | "
                      f"test {test_ppl['ppl']:.1f} | gap {gen_gap:.3f}")
                print(f"    eff_rank {topo.get('mean_eff_rank', 0):.1f} | "
                      f"spec_gap {topo.get('mean_spectral_gap', 0):.1f}")
                print(f"    weights: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} "
                      f"3={wstats['three']:.3f}")
                if args.stage == "full":
                    ac = analyzer_crystals(model)
                    for band, c in ac.items():
                        print(f"    {band:>10s} sub-crystal: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
                print(flush=True)

                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(log, f, indent=2, default=str)

            step += 1

    # ── Final ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    train_ppl = measure_perplexity(model, train_loader, device)
    test_ppl = measure_perplexity(model, test_loader, device)
    topo = measure_topology_harmonic(model, sample_batch, device)
    wstats = weight_stats(model)

    log["final"] = {
        "elapsed": elapsed,
        "train_ppl": train_ppl,
        "test_ppl": test_ppl,
        "gen_gap": test_ppl["ppl"] / train_ppl["ppl"],
        "topology": {
            "eff_rank": topo.get("mean_eff_rank", 0),
            "spectral_gap": topo.get("mean_spectral_gap", 0),
        },
        "weight_dist": wstats,
        "per_layer_crystal": per_layer_crystal(model),
    }

    if args.stage == "full":
        log["final"]["analyzer_crystals"] = analyzer_crystals(model)
        log["final"]["router_bands"] = model.router.band_sizes()

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print(f"\n{'='*60}")
    print(f"  HARMONIC STACK — {args.stage} — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  train_ppl={train_ppl['ppl']:.1f} | test_ppl={test_ppl['ppl']:.1f}")
    print(f"  gen_gap={test_ppl['ppl']/train_ppl['ppl']:.3f}")
    print(f"  eff_rank={topo.get('mean_eff_rank', 0):.1f}")
    print(f"  crystal: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} 3={wstats['three']:.3f}")
    if args.stage == "full":
        ac = analyzer_crystals(model)
        for band, c in ac.items():
            print(f"  {band} sub-crystal: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — verify the script starts and runs a few steps**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems/products/ternary-architect && python3 run_harmonic.py --stage prism --width 512 --n_heads 8 --prism_layers 1 --dataset tinystories --max_steps 10 --n_samples 1000 --batch_size 4 --effective_batch 4
```

Expected: Loads data, prints param count, runs 10 steps without crashing, prints loss values.

- [ ] **Step 3: Commit**

```bash
git add products/ternary-architect/run_harmonic.py
git commit -m "feat: run_harmonic.py — training script for Harmonic Stack experiments"
```

---

### Task 3: Run Experiment 1A — 1-layer prism on WikiText

**Files:**
- No code changes. This is an experiment run.

- [ ] **Step 1: Launch 1-layer x 2048 prism on WikiText**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems/products/ternary-architect && python3 run_harmonic.py --stage prism --width 2048 --n_heads 16 --prism_layers 1 --dataset wikitext --max_steps 20000 --batch_size 8 --effective_batch 32 --ternary_decay 0.01
```

Expected: ~30-60 min on RTX 5070. Crystal checkpoints every 1000 steps. Results saved to `results/harmonic_prism_1L_2048/`.

- [ ] **Step 2: Verify the crystal**

After training completes, check:
```bash
python3 -c "
import json
with open('results/harmonic_prism_1L_2048/training_log.json') as f:
    log = json.load(f)
final = log['final']
w = final['weight_dist']
print(f'Crystal: 0={w[\"zero\"]:.3f} 1={w[\"one\"]:.3f} 3={w[\"three\"]:.3f}')
print(f'PPL: {final[\"test_ppl\"][\"ppl\"]:.1f}')
print(f'eff_rank: {final[\"topology\"][\"eff_rank\"]:.1f}')
print(f'gen_gap: {final[\"gen_gap\"]:.3f}')
# Check prediction: crystal within 2% of 22/42/36
w0, w1, w3 = w['zero'], w['one'], w['three']
in_range = abs(w0 - 0.22) < 0.02 and abs(w1 - 0.42) < 0.02 and abs(w3 - 0.36) < 0.02
print(f'Crystal matches 22/42/36: {\"YES\" if in_range else \"NO — INVESTIGATE\"} ')
"
```

Success: crystal within 2% of 22/42/36. PPL will be higher than the 6x512 control (77.9) due to less depth — that's expected.

- [ ] **Step 3: Commit results**

```bash
git add results/harmonic_prism_1L_2048/training_log.json
git commit -m "experiment: 1L x 2048 prism on WikiText — crystal validation"
```

---

### Task 4: Run Experiment 1B — 3-layer prism on WikiText

**Files:**
- No code changes. Experiment run.

- [ ] **Step 1: Launch 3-layer x 2048 prism on WikiText**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems/products/ternary-architect && python3 run_harmonic.py --stage prism --width 2048 --n_heads 16 --prism_layers 3 --dataset wikitext --max_steps 20000 --batch_size 8 --effective_batch 32 --ternary_decay 0.01
```

Expected: ~45-90 min. Results saved to `results/harmonic_prism_3L_2048/`.

- [ ] **Step 2: Compare 1L vs 3L vs 6x512 control**

```bash
python3 -c "
import json

results = {}
for tag, path in [
    ('1L x 2048', 'results/harmonic_prism_1L_2048/training_log.json'),
    ('3L x 2048', 'results/harmonic_prism_3L_2048/training_log.json'),
    ('6L x 512 (control)', 'results/long_run_small_decay0.01/training_log.json'),
]:
    with open(path) as f:
        log = json.load(f)
    final = log['final']
    w = final['weight_dist']
    results[tag] = {
        'w0': w.get('zero', 0), 'w1': w.get('one', 0), 'w3': w.get('three', 0),
        'ppl': final['test_ppl']['ppl'],
        'eff_rank': final.get('topology', {}).get('eff_rank', 0),
    }

print(f'{\"Config\":>20s} | {\"w0\":>6s} {\"w1\":>6s} {\"w3\":>6s} | {\"PPL\":>8s} | {\"eff_rank\":>8s}')
print('-' * 70)
for tag, r in results.items():
    print(f'{tag:>20s} | {r[\"w0\"]:6.3f} {r[\"w1\"]:6.3f} {r[\"w3\"]:6.3f} | {r[\"ppl\"]:8.1f} | {r[\"eff_rank\"]:8.1f}')
"
```

Success: All three show 22/42/36 (within 2%). PPL ordering expected: 6x512 < 3L x 2048 < 1L x 2048.

- [ ] **Step 3: Commit results**

```bash
git add results/harmonic_prism_3L_2048/training_log.json
git commit -m "experiment: 3L x 2048 prism on WikiText — depth comparison"
```

---

### Task 5: Run Experiment 2 — Router + Analyzers

**Files:**
- No code changes. Experiment run using best prism checkpoint.

- [ ] **Step 1: Launch full Harmonic Stack with frozen prism**

Use the better prism model from Experiment 1 (likely 3L based on PPL):

```bash
cd /home/wb1/Desktop/Dev/atft-problems/products/ternary-architect && python3 run_harmonic.py --stage full --width 2048 --n_heads 16 --prism_layers 3 --prism_checkpoint results/harmonic_prism_3L_2048/model.pt --analyzer_width 512 --dataset wikitext --max_steps 20000 --batch_size 8 --effective_batch 32 --ternary_decay 0.01
```

Expected: Loads prism, prints router band sizes (~450 void / ~860 identity / ~738 prime), trains analyzers. Results saved to `results/harmonic_full_3L_2048/`.

- [ ] **Step 2: Analyze sub-crystals — the critical test**

```bash
python3 -c "
import json
with open('results/harmonic_full_3L_2048/training_log.json') as f:
    log = json.load(f)
final = log['final']

print('=== SUB-CRYSTAL ANALYSIS ===')
print('Question: do the three bands have different internal structure?')
print()
ac = final.get('analyzer_crystals', {})
for band in ['void', 'identity', 'prime']:
    c = ac.get(band, {})
    print(f'{band:>10s}: 0={c.get(\"w0\", 0):.3f} 1={c.get(\"w1\", 0):.3f} 3={c.get(\"w3\", 0):.3f}')

print()
bands = final.get('router_bands', {})
print(f'Router: void={bands.get(\"void\", 0)} identity={bands.get(\"identity\", 0)} prime={bands.get(\"prime\", 0)}')
print(f'Combined PPL: {final[\"test_ppl\"][\"ppl\"]:.1f}')

# Check if sub-crystals differ
if ac:
    v = ac.get('void', {})
    i = ac.get('identity', {})
    p = ac.get('prime', {})
    diff_vi = abs(v.get('w3', 0) - i.get('w3', 0))
    diff_vp = abs(v.get('w3', 0) - p.get('w3', 0))
    diff_ip = abs(i.get('w3', 0) - p.get('w3', 0))
    max_diff = max(diff_vi, diff_vp, diff_ip)
    print(f'Max w3 difference between bands: {max_diff:.3f}')
    print(f'Bands carry independent info: {\"YES\" if max_diff > 0.02 else \"NO — IDENTICAL\"}')
"
```

Success: Three sub-crystals DIFFER (max w3 difference > 2%). Prime band shows higher w3 than identity band.

- [ ] **Step 3: Commit results and summary**

```bash
git add results/harmonic_full_3L_2048/training_log.json
git commit -m "experiment: Harmonic Stack full — spectral band independence test"
```

---

### Task 6: Update BRIDGE.md with experimental results

**Files:**
- Modify: `docs/BRIDGE.md` (Part VII section)

- [ ] **Step 1: Update BRIDGE.md with actual results from Experiments 1 and 2**

After both experiments complete, update the Harmonic Stack section in BRIDGE.md with the actual numbers — replace predictions with measurements. Include the comparison table from Task 4 Step 2 and the sub-crystal analysis from Task 5 Step 2.

- [ ] **Step 2: Commit**

```bash
git add docs/BRIDGE.md
git commit -m "results: Harmonic Stack experiments — crystal validation and band analysis"
```

- [ ] **Step 3: Push to GitHub**

```bash
git push origin master
```
