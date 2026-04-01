#!/usr/bin/env python3
"""Drilling data transducer — cross-modality crystal test.

Parses EDR/MWD SQL dump (WITS channel data at 3-second intervals),
normalizes multi-channel time series, and feeds it to a {0,1,3} TernaryLinear
prism for crystal measurement.

Research question: Does the ternary crystal ({0,1,3} weight distribution)
emerge from physical drilling signal the same way it emerges from language?
If 22/42/36 is universal, it should appear here too.

Architecture: DrillingPrism — no transformer, just ternary linear layers.
  Input: (B, T, n_channels) normalized drilling data
  Encoder: Linear(n_channels, hidden) — expand to prism width
  Prism: TernaryLinear(hidden, hidden) — the crystal
  Decoder: Linear(hidden, n_channels) — predict next timestep

Task: autoregressive next-timestep prediction.

Usage:
    python run_drilling.py
    python run_drilling.py --sql_path /path/to/dump.sql --max_steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ── Path setup ────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "topological-router"))

from ternary_linear import TernaryLinear, make_linear
from topo_measures import effective_rank, spectral_gap, gini_fast

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_SQL = Path.home() / "Downloads" / "Oilfield_EDR_SQL_Depth_and_Time" / \
              "SQL_Time" / "172.26.69.100_timedata_1760755485077.sql"

# Key numeric WITS channels — confirmed active in this dataset
KEY_CHANNELS = [
    "0108",  # Bit Depth
    "0110",  # Hole Depth
    "0112",  # Hook Load
    "0113",  # ROP (rate of penetration, average)
    "0115",  # WOB (weight on bit)
    "0117",  # Torque
    "0118",  # RPM
    "0119",  # SPP (standpipe pressure)
    "0120",  # Flow Rate
    "0121",  # Pump Pressure
    "0140",  # ROP instantaneous
    "0824",  # Gamma Ray (sparse: ~3-4% coverage — forward-fill)
]

CHANNEL_NAMES = {
    "0108": "BitDepth",
    "0110": "HoleDepth",
    "0112": "HookLoad",
    "0113": "ROP_avg",
    "0115": "WOB",
    "0117": "Torque",
    "0118": "RPM",
    "0119": "SPP",
    "0120": "FlowRate",
    "0121": "PumpPressure",
    "0140": "ROP_inst",
    "0824": "GammaRay",
}


# ── Data loading ──────────────────────────────────────────────────────────

def parse_sql_dump(sql_path: Path, channels: list[str],
                   max_rows: int | None = None) -> np.ndarray:
    """Parse PostgreSQL COPY format. Returns (n_rows, n_channels) float32 array.

    Missing channel values are forward-filled (last known value), then
    zero-filled for the leading NaN segment.

    Returns raw (unnormalized) values — caller normalizes per-channel.
    """
    print(f"Parsing SQL dump: {sql_path}")
    ch_idx = {ch: i for i, ch in enumerate(channels)}
    n_ch = len(channels)

    rows = []
    in_copy = False

    with open(sql_path, "r", errors="replace") as f:
        for line in f:
            if "COPY public.timedata" in line:
                in_copy = True
                continue
            if in_copy and line.strip() == "\\.":
                break
            if not in_copy:
                continue

            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue

            row = np.full(n_ch, np.nan, dtype=np.float32)
            for pair in parts[1].split(","):
                kv = pair.split("=", 1)
                if len(kv) != 2:
                    continue
                k = kv[0].strip()
                if k not in ch_idx:
                    continue
                try:
                    row[ch_idx[k]] = float(kv[1].strip())
                except ValueError:
                    pass
            rows.append(row)

            if max_rows is not None and len(rows) >= max_rows:
                break

    data = np.stack(rows, axis=0)  # (N, n_channels)
    print(f"  {len(data):,} rows x {n_ch} channels loaded")

    # Forward-fill NaN values per channel
    for c in range(n_ch):
        col = data[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            print(f"  WARNING: channel {channels[c]} is all-NaN — filling with 0")
            data[:, c] = 0.0
            continue
        # Propagate last valid observation forward
        last_val = col[~nan_mask][0]  # first non-NaN for leading segment
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = last_val
            else:
                last_val = col[i]
        data[:, c] = col
        if nan_mask.any():
            fill_pct = nan_mask.sum() / len(col) * 100
            print(f"  channel {channels[c]} ({CHANNEL_NAMES.get(channels[c], '?')}): "
                  f"forward-filled {fill_pct:.1f}% NaN")

    return data


def normalize_channels(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize each channel to [0, 1].

    Returns (normalized, mins, maxs). Flat channels (max==min) become 0.
    """
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid divide-by-zero for constant channels
    normalized = (data - mins) / ranges
    return normalized.astype(np.float32), mins, maxs


# ── Dataset ───────────────────────────────────────────────────────────────

class DrillingDataset(Dataset):
    """Windowed multi-channel drilling time series.

    Each sample: (x, y) where
      x = (T, n_channels) — input window
      y = (T, n_channels) — target = x shifted by 1 (next timestep)

    Autoregressive: predict row t+1 from rows 0..t.
    """

    def __init__(self, data: np.ndarray, window_size: int = 256, stride: int = 64):
        """
        Parameters
        ----------
        data : (N, n_channels) normalized float32
        window_size : timesteps per window (sequence length T)
        stride : hop between windows — use < window_size for overlap
        """
        self.data = torch.from_numpy(data)
        self.window_size = window_size
        self.stride = stride

        # Build window start indices — need window_size + 1 rows for x+y
        n = len(data)
        self.starts = list(range(0, n - window_size - 1, stride))
        print(f"  {len(self.starts):,} windows "
              f"(T={window_size}, stride={stride}, N={n:,})")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        e = s + self.window_size + 1  # +1 for the target shift
        chunk = self.data[s:e]        # (T+1, n_channels)
        x = chunk[:-1]                # (T, n_channels)
        y = chunk[1:]                 # (T, n_channels) — next timestep
        return x, y


def split_dataset(dataset: Dataset, train_frac: float = 0.9):
    n = len(dataset)
    n_train = int(n * train_frac)
    # Chronological split — no shuffling for time series
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# ── Model ─────────────────────────────────────────────────────────────────

class DrillingPrism(nn.Module):
    """Ternary prism for drilling data.

    Architecture:
      1. Encoder: fp32 Linear(n_channels → hidden)   — lift to prism width
      2. Prism:   TernaryLinear(hidden → hidden)      — the {0,1,3} crystal
      3. Norm:    LayerNorm(hidden)                   — stabilize
      4. Decoder: fp32 Linear(hidden → n_channels)   — project back

    Simple and clean — isolates the ternary prism's contribution.
    No transformer, no attention. Just the crystal.
    """

    def __init__(self, n_channels: int, expansion: int = 16,
                 weight_set: str = "013", init_mode: str = "mixed"):
        super().__init__()
        hidden = n_channels * expansion
        self.n_channels = n_channels
        self.hidden = hidden

        self.encoder = nn.Linear(n_channels, hidden)
        self.prism = make_linear(hidden, hidden, bias=True, weight_set=weight_set)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = nn.Linear(hidden, n_channels)

        # Initialize prism weights
        if isinstance(self.prism, TernaryLinear):
            self.prism.reset_parameters(init_mode=init_mode)

        # Initialize encoder/decoder with small weights
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)
        nn.init.zeros_(self.decoder.bias)

        print(f"DrillingPrism: {n_channels}ch x{expansion} → {hidden}d hidden")
        print(f"  prism type: {weight_set}, init: {init_mode}")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  total params: {n_params:,}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, n_channels) normalized drilling data

        Returns
        -------
        pred : (B, T, n_channels) predicted next-timestep values
        loss : scalar MSE loss (or 0 if no targets passed externally)
        """
        # Encoder: lift to hidden space
        h = self.encoder(x)           # (B, T, hidden)
        h = F.gelu(h)

        # Prism: the crystal layer — ternary weights
        h = self.prism(h)             # (B, T, hidden)
        h = self.norm(h)
        h = F.gelu(h)

        # Decoder: project back to channel space
        pred = self.decoder(h)        # (B, T, n_channels)
        return pred

    def weight_distribution(self) -> dict[str, float]:
        """Get prism weight distribution."""
        if isinstance(self.prism, TernaryLinear):
            return self.prism.weight_distribution()
        return {"type": "fp16"}

    def count_params(self) -> dict[str, int]:
        ternary_params = sum(p.numel() for name, p in self.named_parameters()
                             if "prism" in name and "weight" in name)
        total_params = sum(p.numel() for p in self.parameters())
        return {"total": total_params, "ternary": ternary_params,
                "ternary_pct": 100 * ternary_params / total_params}


# ── Measurements ──────────────────────────────────────────────────────────

def measure_topology(model: DrillingPrism, sample_x: torch.Tensor,
                     device: torch.device) -> dict:
    """Measure effective rank and spectral gap of hidden representations."""
    model.eval()
    with torch.no_grad():
        x = sample_x[:1].to(device)  # (1, T, n_channels)
        # Get hidden states from prism layer
        h = model.encoder(x)
        h = F.gelu(h)
        h_prism = model.prism(h)     # (1, T, hidden)
        h_prism_flat = h_prism[0]    # (T, hidden)

        svs = torch.linalg.svdvals(h_prism_flat.float().cpu()).numpy()
        er = effective_rank(h_prism_flat)
        sg = spectral_gap(h_prism_flat)
        gini = gini_fast(svs)

    model.train()
    return {"eff_rank": er, "spectral_gap": sg, "gini_sv": gini}


def measure_channel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                              channels: list[str],
                              threshold: float = 0.05) -> dict[str, float]:
    """Per-channel prediction accuracy: fraction of timesteps within threshold.

    Uses absolute normalized error. Since data is in [0,1], threshold=0.05
    means 5% of the channel's full range.
    """
    with torch.no_grad():
        err = (pred - target).abs()  # (B, T, n_channels)
        accurate = (err < threshold).float().mean(dim=(0, 1))  # (n_channels,)
        result = {}
        for i, ch in enumerate(channels):
            name = CHANNEL_NAMES.get(ch, ch)
            result[f"{ch}_{name}"] = accurate[i].item()
        result["mean"] = accurate.mean().item()
    return result


# ── Main training loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drilling data transducer — ternary crystal test")
    parser.add_argument("--sql_path", type=str, default=str(DEFAULT_SQL),
                        help="Path to SQL dump file")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window_size", type=int, default=256,
                        help="Timesteps per window (sequence length T)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Hop between windows (overlap = window_size - stride)")
    parser.add_argument("--expansion", type=int, default=16,
                        help="Hidden dim = n_channels * expansion")
    parser.add_argument("--weight_set", type=str, default="013",
                        choices=["013", "n101", "fp16"],
                        help="Weight set for prism layer")
    parser.add_argument("--init_mode", type=str, default="mixed",
                        choices=["mixed", "void", "identity", "uniform"],
                        help="Ternary weight initialization")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Cap dataset rows (for quick tests)")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    # ── Output ──────────────────────────────────────────────────────────
    output_dir = HERE / "results" / "drilling_crystal"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.json"
    print(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────
    sql_path = Path(args.sql_path)
    if not sql_path.exists():
        print(f"ERROR: SQL file not found: {sql_path}")
        sys.exit(1)

    raw = parse_sql_dump(sql_path, KEY_CHANNELS, max_rows=args.max_rows)
    normalized, ch_mins, ch_maxs = normalize_channels(raw)

    # Report channel ranges
    print("\nChannel statistics (after normalization):")
    for i, ch in enumerate(KEY_CHANNELS):
        name = CHANNEL_NAMES.get(ch, ch)
        std = normalized[:, i].std()
        print(f"  {ch} {name:15s}: raw=[{ch_mins[i]:.2f}, {ch_maxs[i]:.2f}] "
              f"  norm_std={std:.4f}")

    full_ds = DrillingDataset(normalized, window_size=args.window_size,
                              stride=args.stride)
    train_ds, test_ds = split_dataset(full_ds, train_frac=0.9)
    print(f"\nTrain: {len(train_ds):,} windows | Test: {len(test_ds):,} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"),
                             drop_last=True)

    # ── Model ────────────────────────────────────────────────────────────
    n_channels = len(KEY_CHANNELS)
    model = DrillingPrism(
        n_channels=n_channels,
        expansion=args.expansion,
        weight_set=args.weight_set,
        init_mode=args.init_mode,
    ).to(device)

    params = model.count_params()
    print(f"\nParams: {params['total']:,} total, "
          f"{params['ternary']:,} ternary ({params['ternary_pct']:.1f}%)")

    # ── Optimizer — two param groups ─────────────────────────────────────
    ternary_params = []
    continuous_params = []
    for name, p in model.named_parameters():
        if "prism.weight" in name and isinstance(model.prism, TernaryLinear):
            ternary_params.append(p)
        else:
            continuous_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": ternary_params,   "weight_decay": 0.0},   # L2=0 on ternary
        {"params": continuous_params, "weight_decay": 0.01},  # L2=0.01 on continuous
    ], lr=args.lr)

    # LR warmup + cosine decay
    warmup_steps = min(1000, args.max_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ─────────────────────────────────────────────────────────
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0]  # keep on CPU for topology measurements

    log = {
        "experiment": "drilling_crystal",
        "weight_set": args.weight_set,
        "config": vars(args),
        "channels": KEY_CHANNELS,
        "channel_names": CHANNEL_NAMES,
        "channel_stats": {
            ch: {"min": float(ch_mins[i]), "max": float(ch_maxs[i])}
            for i, ch in enumerate(KEY_CHANNELS)
        },
        "params": params,
        "steps": [],
    }

    # Initial crystal snapshot
    init_dist = model.weight_distribution()
    log["init_weight_dist"] = init_dist
    print(f"\nInit crystal: {init_dist}")

    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    accumulated_loss = 0.0
    accumulated_steps = 0

    print(f"\nTraining {args.max_steps:,} steps...")
    model.train()

    while step < args.max_steps:
        # Fetch next batch (cycle iterator)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)  # (B, T, n_channels)
        y = y.to(device)  # (B, T, n_channels)

        optimizer.zero_grad()
        pred = model(x)           # (B, T, n_channels)
        loss = F.mse_loss(pred, y)
        loss.backward()

        # Gradient clipping on continuous params only
        nn.utils.clip_grad_norm_(continuous_params, max_norm=1.0)

        optimizer.step()
        scheduler.step()

        accumulated_loss += loss.item()
        accumulated_steps += 1
        step += 1

        # ── Logging ──────────────────────────────────────────────────────
        if step % args.log_every == 0:
            train_loss = accumulated_loss / accumulated_steps
            accumulated_loss = 0.0
            accumulated_steps = 0

            # Test loss
            model.eval()
            test_loss = 0.0
            test_steps = 0
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for tx, ty in test_loader:
                    tx, ty = tx.to(device), ty.to(device)
                    tp = model(tx)
                    test_loss += F.mse_loss(tp, ty).item()
                    test_steps += 1
                    if test_steps <= 5:
                        test_preds.append(tp.cpu())
                        test_targets.append(ty.cpu())
                    if test_steps >= 50:
                        break
            test_loss /= max(1, test_steps)

            # Channel accuracy on test set
            if test_preds:
                all_pred = torch.cat(test_preds, dim=0)
                all_tgt = torch.cat(test_targets, dim=0)
                ch_acc = measure_channel_accuracy(all_pred, all_tgt, KEY_CHANNELS)
            else:
                ch_acc = {}

            # Crystal weight distribution
            wdist = model.weight_distribution()

            # Topology of prism hidden states
            topo = measure_topology(model, sample_x, device)

            # PPL equivalent: exp(MSE * scale) — MSE is in normalized [0,1] space
            # Use reconstruction-equivalent: compare to naive persistence baseline
            # Naive: predict mean (MSE = variance of test data)
            ppl_equiv = float(np.exp(min(test_loss * 10, 20)))  # scaled for readability

            elapsed = time.time() - t0
            lr_current = scheduler.get_last_lr()[0]

            entry = {
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "ppl_equiv": ppl_equiv,
                "lr": lr_current,
                "elapsed_s": elapsed,
                "crystal": wdist,
                "topology": topo,
                "channel_accuracy": ch_acc,
            }
            log["steps"].append(entry)

            # Print
            void_pct = wdist.get("w=0", 0) * 100
            unit_pct = wdist.get("w=1", 0) * 100
            prime_pct = wdist.get("w=3", 0) * 100
            print(
                f"step {step:6d} | "
                f"train={train_loss:.5f} test={test_loss:.5f} ppl~={ppl_equiv:.1f} | "
                f"crystal: 0={void_pct:.0f}% 1={unit_pct:.0f}% 3={prime_pct:.0f}% | "
                f"eff_rank={topo['eff_rank']:.1f} gini={topo['gini_sv']:.3f} | "
                f"ch_acc={ch_acc.get('mean', 0):.3f} | "
                f"lr={lr_current:.2e} | {elapsed:.0f}s"
            )

            model.train()

            # Save log checkpoint
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    final_dist = model.weight_distribution()
    log["final_weight_dist"] = final_dist

    # Final test loss
    model.eval()
    final_test_loss = 0.0
    final_steps = 0
    with torch.no_grad():
        for tx, ty in test_loader:
            tx, ty = tx.to(device), ty.to(device)
            tp = model(tx)
            final_test_loss += F.mse_loss(tp, ty).item()
            final_steps += 1
            if final_steps >= 100:
                break
    final_test_loss /= max(1, final_steps)

    # Final topology
    final_topo = measure_topology(model, sample_x, device)
    log["final"] = {
        "test_loss": final_test_loss,
        "ppl_equiv": float(np.exp(min(final_test_loss * 10, 20))),
        "weight_dist": final_dist,
        "topology": final_topo,
    }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # Console summary
    void_pct = final_dist.get("w=0", 0) * 100
    unit_pct = final_dist.get("w=1", 0) * 100
    prime_pct = final_dist.get("w=3", 0) * 100
    print("\n" + "=" * 70)
    print("DRILLING CRYSTAL — FINAL RESULTS")
    print("=" * 70)
    print(f"Test MSE:     {final_test_loss:.6f}")
    print(f"PPL equiv:    {log['final']['ppl_equiv']:.2f}")
    print(f"Crystal:      void={void_pct:.1f}% unit={unit_pct:.1f}% prime={prime_pct:.1f}%")
    print(f"  Ref (lang): void~22%  unit~42%  prime~36%")
    print(f"  Match?      {'YES — crystal is universal!' if abs(void_pct - 22) < 8 and abs(prime_pct - 36) < 8 else 'Different from language baseline'}")
    print(f"Eff rank:     {final_topo['eff_rank']:.2f}")
    print(f"Spectral gap: {final_topo['spectral_gap']:.3f}")
    print(f"Gini (SVs):   {final_topo['gini_sv']:.4f}")
    print(f"Log saved:    {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
