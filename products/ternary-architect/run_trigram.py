#!/usr/bin/env python3
"""Trigram Prism — 27-state structural training with topology BitFlip.

Structure-first language model. Primary vocabulary: 27 structural states
(3³ ternary characters). Secondary channel: 8192 content hash buckets.
Target: predict next structural state, not next BPE token.

BitFlip from identity init with topology-informed persistence thresholds.
The ASYMMETRY metric (|void - prime|) is the key crystal measurement.

Architecture: 1-layer transformer at width 4096 (207M params).
97% ternary via BitFlipLinear.

Usage:
    python run_trigram.py
    python run_trigram.py --dataset wikitext --max_steps 50000
    python run_trigram.py --dataset tinystories --width 2048 --batch_size 16
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ── Path setup ────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import BitFlipLinear, make_linear
from ternary_transformer import GPTConfig, Block
from topology_bitflip import TopologyBitFlipEngine
from trigram_transducer import classify_char, structural_trigram_id, load_dataset_text
from topo_measures import effective_rank, spectral_gap, gini_fast


# ── Content hash ──────────────────────────────────────────────────────────────

def content_hash(c1: str, c2: str, c3: str, n_buckets: int = 8192) -> int:
    """Hash a raw character trigram into a content bucket.

    Provides a vocabulary-free content signal alongside structural identity.
    8192 buckets ≈ 13 bits — enough to distinguish common character clusters
    while staying far below BPE vocabulary size.
    """
    return hash((c1, c2, c3)) % n_buckets


# ── Dataset ───────────────────────────────────────────────────────────────────

class DualChannelTrigramDataset(Dataset):
    """Produces (structural_ids, content_hashes) pairs for each text chunk.

    For every overlapping character trigram in text:
      structural_id = structural_trigram_id(classify(c1), classify(c2), classify(c3))
      content_id    = content_hash(c1, c2, c3)

    Chunks into fixed-length sequences. __getitem__ returns:
      (struct[:-1], content[:-1]), struct[1:]
    Target is next structural state only — structure predicts structure.
    """

    def __init__(self, text: str, chunk_size: int = 512):
        struct_ids = []
        content_ids = []

        for i in range(len(text) - 2):
            c1, c2, c3 = text[i], text[i + 1], text[i + 2]
            s_id = structural_trigram_id(classify_char(c1), classify_char(c2), classify_char(c3))
            c_id = content_hash(c1, c2, c3)
            struct_ids.append(s_id)
            content_ids.append(c_id)

        self.chunks_struct = []
        self.chunks_content = []

        # Chunk into sequences of chunk_size (keep +1 for target shift)
        stride = chunk_size
        for i in range(0, len(struct_ids) - chunk_size, stride):
            self.chunks_struct.append(struct_ids[i : i + chunk_size + 1])
            self.chunks_content.append(content_ids[i : i + chunk_size + 1])

        print(
            f"  DualChannelTrigramDataset: {len(text)} chars → "
            f"{len(struct_ids)} trigrams → {len(self.chunks_struct)} chunks of {chunk_size}"
        )

    def __len__(self) -> int:
        return len(self.chunks_struct)

    def __getitem__(self, idx: int):
        s = torch.tensor(self.chunks_struct[idx], dtype=torch.long)
        c = torch.tensor(self.chunks_content[idx], dtype=torch.long)
        # Input: [:-1], Target: struct[1:] only
        return (s[:-1], c[:-1]), s[1:]


# ── Model ─────────────────────────────────────────────────────────────────────

class TrigramPrism(nn.Module):
    """Structure-first model.

    Primary input:  nn.Embedding(27, d_model)       — structural states
    Secondary input: nn.Embedding(8192, content_dim) → Linear → d_model
    Position:       nn.Embedding(block_size, d_model)
    All three summed → dropout → 1 transformer Block (BitFlip) → LN → Linear(d_model, 27)

    Target: next structural state.
    """

    def __init__(
        self,
        n_struct: int = 27,
        n_content: int = 8192,
        d_model: int = 4096,
        n_head: int = 16,
        block_size: int = 512,
        content_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.n_struct = n_struct

        # Primary: structural state embedding
        self.struct_emb = nn.Embedding(n_struct, d_model)

        # Secondary: content hash embedding + projection
        self.content_emb = nn.Embedding(n_content, content_dim)
        self.content_proj = nn.Linear(content_dim, d_model, bias=False)

        # Positional embedding
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.drop = nn.Dropout(dropout)

        # One transformer Block — all linear layers are BitFlipLinear
        cfg = GPTConfig(
            vocab_size=n_struct,
            block_size=block_size,
            n_layer=1,
            n_head=n_head,
            n_embd=d_model,
            dropout=dropout,
            weight_set="bitflip",
        )
        self.block = Block(cfg)

        self.ln_f = nn.LayerNorm(d_model)

        # Output head: d_model → 27 structural states (small, no ternary needed)
        self.lm_head = nn.Linear(d_model, n_struct, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.struct_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.content_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.content_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Identity init for all BitFlipLinear weights
        for m in self.modules():
            if isinstance(m, BitFlipLinear):
                m.reset_parameters(init_mode="identity")

    def forward(
        self,
        struct_ids: torch.Tensor,
        content_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """
        Parameters
        ----------
        struct_ids  : (B, T) in [0, 26]
        content_ids : (B, T) in [0, 8191]
        targets     : (B, T) structural state targets
        return_hidden : if True, return hidden states after the block

        Returns
        -------
        logits : (B, T, 27)
        loss   : scalar if targets provided
        hidden : (B, T, d_model) if return_hidden
        """
        B, T = struct_ids.shape
        assert T <= self.block_size, f"Sequence {T} > block_size {self.block_size}"

        pos = torch.arange(T, device=struct_ids.device).unsqueeze(0)  # (1, T)

        x_struct = self.struct_emb(struct_ids)                        # (B, T, d)
        x_content = self.content_proj(self.content_emb(content_ids))  # (B, T, d)
        x_pos = self.pos_emb(pos)                                      # (1, T, d)

        x = self.drop(x_struct + x_content + x_pos)

        x = self.block(x)

        hidden = x if return_hidden else None

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, 27)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.n_struct),
                targets.view(-1),
            )

        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def count_params(self) -> dict:
        ternary = continuous = 0
        for name, p in self.named_parameters():
            is_bitflip_weight = False
            for mname, m in self.named_modules():
                if isinstance(m, BitFlipLinear) and name == f"{mname}.weight":
                    is_bitflip_weight = True
                    break
            if is_bitflip_weight:
                ternary += p.numel()
            else:
                continuous += p.numel()
        total = ternary + continuous
        return {
            "ternary": ternary,
            "continuous": continuous,
            "total": total,
            "ternary_pct": ternary / total * 100 if total > 0 else 0.0,
        }

    def weight_distributions(self) -> dict[str, dict]:
        dists = {}
        for name, m in self.named_modules():
            if isinstance(m, BitFlipLinear):
                dists[name] = m.weight_distribution()
        return dists


# ── Measurement helpers ───────────────────────────────────────────────────────

def weight_stats(model: TrigramPrism) -> dict:
    """Aggregate crystal: void/identity/prime fractions across all BitFlipLinear."""
    total_void = total_one = total_three = total = 0
    for m in model.modules():
        if isinstance(m, BitFlipLinear):
            wq = m.weight.data
            total_void += (wq == 0).sum().item()
            total_one += (wq == 1).sum().item()
            total_three += (wq == 3).sum().item()
            total += wq.numel()
    if total == 0:
        return {"void": 0.0, "identity": 0.0, "prime": 0.0, "asymmetry": 0.0}
    v = total_void / total
    i = total_one / total
    p = total_three / total
    return {
        "void": v,
        "identity": i,
        "prime": p,
        "asymmetry": abs(v - p),  # KEY METRIC
    }


def per_layer_crystal(model: TrigramPrism) -> dict[str, dict]:
    """Per-layer crystal distribution for diagnostic output."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, BitFlipLinear):
            wq = m.weight.data
            tot = wq.numel()
            v = (wq == 0).sum().item() / tot
            i = (wq == 1).sum().item() / tot
            p = (wq == 3).sum().item() / tot
            out[name] = {"void": v, "identity": i, "prime": p, "asymmetry": abs(v - p)}
    return out


def measure_eff_rank(model: TrigramPrism, sample_batch, device: torch.device) -> float:
    """Compute effective rank of hidden states after the transformer block."""
    model.eval()
    with torch.no_grad():
        (s_ids, c_ids), _ = sample_batch
        s_ids = s_ids[:1].to(device)
        c_ids = c_ids[:1].to(device)
        _, _, hidden = model(s_ids, c_ids, return_hidden=True)
        h = hidden[0]  # (T, d_model)
        rank = effective_rank(h)
    model.train()
    return rank


def measure_ppl(model: TrigramPrism, loader: DataLoader, device: torch.device,
                max_batches: int = 100) -> dict:
    """Cross-entropy and PPL-equivalent on 27-class structural prediction."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, ((s_ids, c_ids), targets) in enumerate(loader):
            if i >= max_batches:
                break
            s_ids = s_ids.to(device)
            c_ids = c_ids.to(device)
            targets = targets.to(device)
            _, loss = model(s_ids, c_ids, targets=targets)
            n = targets.numel()
            total_loss += loss.item() * n
            total_tokens += n
    model.train()
    avg = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return {"loss": avg, "ppl": float(min(np.exp(avg), 1e7))}


def split_dataset(dataset, train_frac: float = 0.9):
    n = len(dataset)
    n_train = int(n * train_frac)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    return (
        Subset(dataset, indices[:n_train].tolist()),
        Subset(dataset, indices[n_train:].tolist()),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trigram Prism — structure-first ternary training"
    )
    parser.add_argument(
        "--dataset",
        default="tinystories",
        choices=["tinystories", "wikitext", "kant", "animalfarm", "sep",
                 "korean", "chinese", "arabic"],
        help="Training dataset",
    )
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--width", type=int, default=4096,
                        help="Model width (d_model). Default 4096.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_samples", type=int, default=200000,
                        help="Max text samples for streaming datasets")
    args = parser.parse_args()

    # Output directory
    OUTPUT_DIR = Path(__file__).parent / "results" / f"trigram_{args.dataset}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}...")
    text = load_dataset_text(args.dataset, n_samples=args.n_samples)
    print(f"  Raw text: {len(text):,} chars")

    full_ds = DualChannelTrigramDataset(text, chunk_size=args.chunk_size)
    del text
    gc.collect()

    train_ds, test_ds = split_dataset(full_ds)
    print(f"  Train: {len(train_ds)} chunks | Test: {len(test_ds)} chunks")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    sample_batch = next(iter(train_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    n_heads = max(1, args.width // 256)  # head_dim = 256
    model = TrigramPrism(
        n_struct=27,
        n_content=8192,
        d_model=args.width,
        n_head=n_heads,
        block_size=args.chunk_size,
        content_dim=256,
    ).to(device)

    params = model.count_params()
    print(f"\nTrigramPrism: {params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")
    print(f"  d_model={args.width} n_head={n_heads} block_size={args.chunk_size}")

    # ── Optimizer — exclude BitFlipLinear weights ──────────────────────────
    bitflip_weight_ids = set()
    for mname, m in model.named_modules():
        if isinstance(m, BitFlipLinear):
            bitflip_weight_ids.add(id(m.weight))

    continuous_params = [
        p for p in model.parameters()
        if id(p) not in bitflip_weight_ids
    ]

    optimizer = torch.optim.AdamW(
        [{"params": continuous_params, "weight_decay": 0.01}],
        lr=args.lr,
    )

    # LR warmup (2000 steps) + cosine decay
    warmup_steps = min(2000, args.max_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1.0 + np.cos(np.pi * progress)) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── TopologyBitFlipEngine ──────────────────────────────────────────────
    engine = TopologyBitFlipEngine(
        model,
        base_flip_pct=0.0001,
        cycle_steps=100,
        warmup_steps=2000,
        gravity=0.0,
        persistence_cycle=2000,
        persistence_scale=5.0,
        n_row_sample=200,
    )
    print(
        f"  TopologyBitFlipEngine: flip_pct=0.001 cycle=100 warmup=500 "
        f"persistence_cycle=2000 scale=5.0"
    )

    # ── Training log ──────────────────────────────────────────────────────
    init_wstats = weight_stats(model)
    log = {
        "experiment": "trigram_prism",
        "dataset": args.dataset,
        "config": {
            "width": args.width,
            "n_head": n_heads,
            "block_size": args.chunk_size,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "flip_pct": 0.001,
            "flip_cycle": 100,
            "warmup": 500,
            "cooldown": 5000,
            "persistence_cycle": 2000,
            "persistence_scale": 5.0,
            "init_mode": "identity",
        },
        "params": params,
        "init_weight_dist": init_wstats,
        "hypothesis": (
            "Structure-first: 27 structural states + 8192 content hashes. "
            "BitFlip from identity init. ASYMMETRY = |void - prime| is the key metric. "
            "Crystal should diverge from uniform as topology forms."
        ),
        "checkpoints": [],
        "steps": [],
    }

    # ── Training loop ─────────────────────────────────────────────────────
    step = 0
    start = time.time()
    cooldown_start = args.max_steps - 5000

    print(f"\nTraining {args.max_steps} steps. Checkpoint every 1000.\n")
    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for (s_ids, c_ids), targets in train_loader:
            if step >= args.max_steps:
                break

            s_ids = s_ids.to(device)
            c_ids = c_ids.to(device)
            targets = targets.to(device)

            _, loss = model(s_ids, c_ids, targets=targets)

            # NaN recovery — freeze and stabilize
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    *** NaN/Inf at step {step} — stabilizing 500 steps ***", flush=True)
                optimizer.zero_grad()
                # Skip this batch, continue training
                step += 1
                continue

            loss.backward()

            # Accumulate gradient signal for BitFlip engine
            engine.accumulate()

            # Gradient clip on continuous params, then optimizer step
            torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # BitFlip step (with cooldown)
            if step < cooldown_start:
                flip_stats = engine.maybe_flip(step)
                if flip_stats is not None and flip_stats["total"] > 0:
                    print(
                        f"    FLIP {step}: 0→1={flip_stats['0→1']} "
                        f"1→3={flip_stats['1→3']} 3→1={flip_stats['3→1']} "
                        f"1→0={flip_stats['1→0']} total={flip_stats['total']}",
                        flush=True,
                    )

            # Quick log every 200 steps
            if step % 200 == 0:
                elapsed = time.time() - start
                ppl_val = float(np.exp(min(loss.item(), 20.0)))
                log["steps"].append({
                    "step": step,
                    "loss": loss.item(),
                    "ppl": ppl_val,
                    "elapsed": elapsed,
                })
                print(
                    f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl_val:8.2f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

            # Full checkpoint every 1000 steps
            if step > 0 and step % 1000 == 0:
                train_ppl = measure_ppl(model, train_loader, device)
                test_ppl = measure_ppl(model, test_loader, device)
                eff_rank_val = measure_eff_rank(model, sample_batch, device)
                wstats = weight_stats(model)
                layer_crystal = per_layer_crystal(model)

                ckpt = {
                    "step": step,
                    "train": train_ppl,
                    "test": test_ppl,
                    "gen_gap": (
                        test_ppl["ppl"] / train_ppl["ppl"]
                        if train_ppl["ppl"] > 0 else 999.0
                    ),
                    "eff_rank": eff_rank_val,
                    "crystal": wstats,
                    "asymmetry": wstats["asymmetry"],
                    "per_layer_crystal": layer_crystal,
                    "elapsed": time.time() - start,
                }
                log["checkpoints"].append(ckpt)

                print(
                    f"\n    CKPT {step:6d} | train_ppl {train_ppl['ppl']:.2f} | "
                    f"test_ppl {test_ppl['ppl']:.2f}"
                )
                print(
                    f"    eff_rank {eff_rank_val:.1f} | "
                    f"ASYMMETRY {wstats['asymmetry']:.4f}"
                )
                print(
                    f"    crystal: void={wstats['void']:.3f} "
                    f"identity={wstats['identity']:.3f} "
                    f"prime={wstats['prime']:.3f}",
                    flush=True,
                )
                print()

                # Save intermediate log
                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(log, f, indent=2, default=str)

            step += 1

    # ── Final measurements ────────────────────────────────────────────────
    elapsed = time.time() - start
    train_ppl = measure_ppl(model, train_loader, device)
    test_ppl = measure_ppl(model, test_loader, device)
    eff_rank_val = measure_eff_rank(model, sample_batch, device)
    wstats = weight_stats(model)
    layer_crystal = per_layer_crystal(model)

    log["final"] = {
        "elapsed": elapsed,
        "train": train_ppl,
        "test": test_ppl,
        "gen_gap": test_ppl["ppl"] / train_ppl["ppl"] if train_ppl["ppl"] > 0 else 999.0,
        "eff_rank": eff_rank_val,
        "crystal": wstats,
        "asymmetry": wstats["asymmetry"],
        "per_layer_crystal": layer_crystal,
    }

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print(f"\n{'=' * 60}")
    print(f"  TRIGRAM PRISM — {args.dataset} — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  train_ppl={train_ppl['ppl']:.2f} | test_ppl={test_ppl['ppl']:.2f}")
    print(f"  gen_gap={log['final']['gen_gap']:.3f}")
    print(f"  eff_rank={eff_rank_val:.1f}")
    print(
        f"  crystal: void={wstats['void']:.3f} "
        f"identity={wstats['identity']:.3f} "
        f"prime={wstats['prime']:.3f}"
    )
    print(f"  ASYMMETRY={wstats['asymmetry']:.4f}  (|void - prime|)")
    print(f"  Saved → {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
