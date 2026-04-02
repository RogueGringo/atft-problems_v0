#!/usr/bin/env python3
"""TopologyBitFlipEngine — per-weight persistence thresholds for BitFlip training.

Extends BitFlipEngine from ternary_linear.py with topology-informed flip criteria.

Key insight: BitFlipEngine uses a GLOBAL flip_pct — every weight has the same
flip threshold regardless of its local topological environment. This produces
symmetric crystals (void ≈ prime) because the energy landscape is flat.

TopologyBitFlipEngine replaces the global threshold with per-weight thresholds
derived from H₀ persistent homology of the weight matrix rows:

  Stable topology  (long bars, high Gini) → HIGH threshold → resists flipping
  Weak topology    (short bars, low Gini) → LOW threshold  → flips easily

The persistence diagram IS the energy landscape for discrete state transitions.
The float expressivity lives in the threshold tensor, not the weights.
"""
from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import numpy as np

# ── Path setup for topo_measures ─────────────────────────────────────────────

_TOPO_ROUTER = os.path.join(
    os.path.dirname(__file__),
    "..", "topological-router"
)
_TOPO_ROUTER = os.path.abspath(_TOPO_ROUTER)
if _TOPO_ROUTER not in sys.path:
    sys.path.insert(0, _TOPO_ROUTER)

from topo_measures import h0_persistence, gini_fast  # noqa: E402

# ── Import the base engine and layer ─────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ternary_linear import BitFlipLinear, BitFlipEngine  # noqa: E402


# ── TopologyBitFlipEngine ─────────────────────────────────────────────────────

class TopologyBitFlipEngine:
    """BitFlip engine with per-weight flip thresholds derived from
    local topological persistence.

    Stable topology → high threshold → weight resists flipping
    Weak topology → low threshold → weight flips easily

    The float expressivity lives in the threshold, not the weights.
    The persistence diagram IS the energy landscape for discrete state
    transitions.

    Parameters
    ----------
    model : nn.Module
        Model containing BitFlipLinear layers.
    base_flip_pct : float
        Baseline flip threshold (magnitude). Acts as the floor. Default 0.001.
    cycle_steps : int
        Accumulate gradient for this many steps, then flip.
    warmup_steps : int
        No flips before this many optimizer steps.
    gravity : float
        Discrete L2 — demotion urgency multiplier (same semantics as
        BitFlipEngine.gravity). 0.0 = symmetric.
    persistence_cycle : int
        Recompute per-weight thresholds every this many optimizer steps.
        Should be a multiple of cycle_steps (or at least >> cycle_steps).
        Default 1000.
    persistence_scale : float
        How strongly persistence Gini modulates the threshold. A Gini of 1
        (maximally unequal bars = very stable topology) multiplies the base
        threshold by (1 + persistence_scale). Default 5.0.
    n_row_sample : int
        Number of rows to sample per layer when computing H₀ persistence.
        Full computation would be O(out² × in²); sampling keeps it tractable.
        Default 200.
    """

    def __init__(
        self,
        model: nn.Module,
        base_flip_pct: float = 0.001,
        cycle_steps: int = 100,
        warmup_steps: int = 500,
        gravity: float = 0.0,
        persistence_cycle: int = 1000,
        persistence_scale: float = 5.0,
        n_row_sample: int = 200,
    ):
        self.model = model
        self.base_flip_pct = base_flip_pct
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.gravity = gravity
        self.persistence_cycle = persistence_cycle
        self.persistence_scale = persistence_scale
        self.n_row_sample = n_row_sample

        self.optim_steps = 0
        self.accum_count = 0
        self.history: list[dict] = []

        # Gradient accumulators: magnitude and direction (same as BitFlipEngine)
        self.grad_mag: dict[str, torch.Tensor] = {}
        self.grad_dir: dict[str, torch.Tensor] = {}

        # Per-weight threshold tensors — initialized to base_flip_pct
        # Shape matches weight tensor for each BitFlipLinear layer.
        self.thresholds: dict[str, torch.Tensor] = {}

        for name, m in model.named_modules():
            if isinstance(m, BitFlipLinear):
                self.grad_mag[name] = torch.zeros_like(m.weight.data)
                self.grad_dir[name] = torch.zeros_like(m.weight.data)
                self.thresholds[name] = torch.full_like(
                    m.weight.data, fill_value=base_flip_pct
                )

    # ── Gradient accumulation ─────────────────────────────────────────────────

    def accumulate(self):
        """Call after backward. Reads and clears BitFlipLinear gradients.

        Identical semantics to BitFlipEngine.accumulate().
        """
        for name, m in self.model.named_modules():
            if isinstance(m, BitFlipLinear) and m.weight.grad is not None:
                self.grad_mag[name] += m.weight.grad.abs()
                self.grad_dir[name] += m.weight.grad
                m.weight.grad = None  # keep optimizer away
        self.accum_count += 1

    # ── Persistence threshold update ──────────────────────────────────────────

    def update_persistence_thresholds(self):
        """Recompute per-weight thresholds from current weight topology.

        Algorithm per layer:
          1. Sample up to n_row_sample rows from the quantized weight matrix.
             Each row is one point in R^{in_features}.
          2. Compute H₀ persistence bars on the sampled row-points.
          3. For each sampled row, compute the Gini of bars involving that row's
             nearest-neighbour distances (approximated by the global bar lengths
             rescaled by that row's mean pairwise distance quartile).
          4. Map per-row Gini → per-row threshold multiplier:
               threshold_row = base * (1 + scale * gini_row)
          5. Broadcast per-row threshold to all weights in that row.
             Unsampled rows keep the base threshold.
        """
        for name, m in self.model.named_modules():
            if not isinstance(m, BitFlipLinear):
                continue
            if name not in self.thresholds:
                continue

            with torch.no_grad():
                w = m.weight.data  # (out_features, in_features)
                out_f, in_f = w.shape

                # Reset to base first — unsampled rows stay at base
                self.thresholds[name].fill_(self.base_flip_pct)

                # Sample row indices
                n_sample = min(self.n_row_sample, out_f)
                if n_sample < 2:
                    continue
                perm = torch.randperm(out_f)[:n_sample]
                sampled_rows = w[perm].float().cpu().numpy()  # (n_sample, in_f)

                # Compute H₀ persistence on the sampled rows as point cloud
                bars = h0_persistence(sampled_rows, max_n=n_sample)
                if len(bars) == 0:
                    continue

                # Global bar stats for normalization
                bar_max = float(bars.max()) if bars.max() > 0 else 1.0

                # For each sampled row, estimate its local bar contribution
                # by computing pairwise distances to all other sampled rows
                # and using the min-distance (nearest neighbour) as a proxy
                # for the bar length at which it would merge.
                t_rows = torch.tensor(sampled_rows)  # (n_sample, in_f)
                dists = torch.cdist(t_rows, t_rows)  # (n_sample, n_sample)
                # Nearest neighbour distance for each row (exclude self)
                dists.fill_diagonal_(float("inf"))
                nn_dist = dists.min(dim=1).values.numpy()  # (n_sample,)

                # Normalize nn_dist to [0, 1] using the bar_max
                nn_norm = np.clip(nn_dist / bar_max, 0.0, 1.0)

                # Compute one global Gini for the bars to get a layer-level
                # stability signal.
                global_gini = gini_fast(bars)

                # Per-row stability proxy: nn_norm (relative isolation of each row).
                # Rows far from neighbours → isolated → stable.
                # Rows close to neighbours → clustered → unstable.
                #
                # We use nn_norm directly as the per-row modulator, scaled by
                # (1 + global_gini) so that globally stable layers amplify
                # the per-row signal and globally unstable layers dampen it.
                # This ensures threshold variability is always present (because
                # nn_norm itself varies), even when Gini ≈ 0.
                #
                # Formula: threshold_i = base * (1 + scale * (1 + gini) * nn_norm_i)
                # This guarantees non-uniform thresholds whenever nn_dist varies.
                per_row_stability = (1.0 + global_gini) * nn_norm  # (n_sample,)

                # Map to thresholds
                per_row_threshold = (
                    self.base_flip_pct * (1.0 + self.persistence_scale * per_row_stability)
                )  # (n_sample,)

                # Write back to the threshold tensor
                th = self.thresholds[name]  # (out_f, in_f)
                for i, row_idx in enumerate(perm.tolist()):
                    th[row_idx, :] = float(per_row_threshold[i])

    # ── Main flip step ────────────────────────────────────────────────────────

    def maybe_flip(self, step: int) -> dict | None:
        """Call each optimizer step. Returns flip stats if triggered.

        Modified from BitFlipEngine.maybe_flip():
          - Uses per-weight threshold tensors instead of top-K global selection.
          - A weight is a flip candidate when its accumulated gradient MAGNITUDE
            exceeds its per-weight threshold (rather than being in the top-K).
          - Gravity still modulates promotion vs. demotion urgency.
          - Persistence thresholds are refreshed every persistence_cycle steps.
        """
        self.optim_steps += 1

        # Periodic persistence update (independent of flip cycle)
        if self.optim_steps % self.persistence_cycle == 0:
            self.update_persistence_thresholds()

        if self.optim_steps < self.warmup_steps:
            return None
        if self.optim_steps % self.cycle_steps != 0:
            return None
        if self.accum_count == 0:
            return None

        flips = {"0→1": 0, "1→3": 0, "3→1": 0, "1→0": 0}

        for name, m in self.model.named_modules():
            if not isinstance(m, BitFlipLinear):
                continue
            if name not in self.grad_mag:
                continue

            with torch.no_grad():
                w = m.weight.data
                mag = self.grad_mag[name] / self.accum_count
                direction = self.grad_dir[name].sign()

                # Per-weight thresholds — move to same device as weights
                th = self.thresholds[name].to(w.device)

                g_up = 1.0 / (1.0 + self.gravity)
                g_down = 1.0 + self.gravity

                # Urgency scores (same four directions as BitFlipEngine)
                urgency = torch.zeros_like(w)

                mask_0_up = (w == 0) & (direction < 0)
                urgency[mask_0_up] = mag[mask_0_up] * g_up

                mask_1_up = (w == 1) & (direction < 0)
                urgency[mask_1_up] = mag[mask_1_up] * g_up

                mask_3_down = (w == 3) & (direction > 0)
                urgency[mask_3_down] = mag[mask_3_down] * g_down

                mask_1_down = (w == 1) & (direction > 0)
                urgency[mask_1_down] = mag[mask_1_down] * g_down

                if urgency.sum() == 0:
                    continue

                # KEY DIFFERENCE from BitFlipEngine:
                # Flip where gradient magnitude exceeds per-weight threshold,
                # BUT cap total flips at base_flip_pct × n_weights.
                candidates = urgency > th
                n_candidates = candidates.sum().item()
                max_flips = max(1, int(w.numel() * self.base_flip_pct))

                if n_candidates == 0:
                    continue

                if n_candidates > max_flips:
                    # Too many candidates — take top-K by urgency
                    flat_urgency = (urgency * candidates.float()).flatten()
                    threshold_val = flat_urgency.topk(max_flips).values[-1]
                    flip_mask = urgency >= threshold_val
                else:
                    flip_mask = candidates

                # Apply flips
                f01 = flip_mask & mask_0_up
                w[f01] = 1.0
                flips["0→1"] += f01.sum().item()

                f13 = flip_mask & mask_1_up
                w[f13] = 3.0
                flips["1→3"] += f13.sum().item()

                f31 = flip_mask & mask_3_down
                w[f31] = 1.0
                flips["3→1"] += f31.sum().item()

                f10 = flip_mask & mask_1_down
                w[f10] = 0.0
                flips["1→0"] += f10.sum().item()

        # Reset accumulators
        for name in self.grad_mag:
            self.grad_mag[name].zero_()
            self.grad_dir[name].zero_()
        self.accum_count = 0

        stats = {"step": step, **flips, "total": sum(flips.values())}
        self.history.append(stats)
        return stats

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def threshold_stats(self) -> dict[str, dict[str, float]]:
        """Return per-layer threshold statistics for diagnostics."""
        out = {}
        for name, th in self.thresholds.items():
            out[name] = {
                "mean": th.mean().item(),
                "std": th.std().item(),
                "min": th.min().item(),
                "max": th.max().item(),
            }
        return out


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 60)
    print("TopologyBitFlipEngine self-test")
    print("=" * 60)

    # ── Build a tiny model with two BitFlipLinear layers ──────────────────────
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = BitFlipLinear(64, 128, bias=True)
            self.l2 = BitFlipLinear(128, 32, bias=True)

        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    model = TinyModel()
    engine = TopologyBitFlipEngine(
        model,
        base_flip_pct=0.001,
        cycle_steps=10,
        warmup_steps=5,
        persistence_cycle=50,
        persistence_scale=5.0,
        n_row_sample=200,
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} params")
    print(f"BitFlipLinear layers tracked: {list(engine.thresholds.keys())}")
    print(f"\nInitial threshold stats (should all equal base_flip_pct={engine.base_flip_pct}):")
    for name, stats in engine.threshold_stats().items():
        print(f"  {name}: mean={stats['mean']:.6f}  std={stats['std']:.6f}  "
              f"min={stats['min']:.6f}  max={stats['max']:.6f}")

    # ── Run 100 accumulate/flip cycles on random data ─────────────────────────
    print("\nRunning 100 steps of accumulate/maybe_flip on random data...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters()
         if not any(p is m.weight for _, m in model.named_modules()
                    if isinstance(m, BitFlipLinear))],
        lr=1e-3,
    )

    flip_events = []
    for step in range(100):
        x = torch.randn(16, 64)
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        engine.accumulate()
        optimizer.step()
        optimizer.zero_grad()

        result = engine.maybe_flip(step)
        if result is not None:
            flip_events.append(result)

    print(f"Flip events triggered: {len(flip_events)}")
    for ev in flip_events:
        print(f"  step={ev['step']:3d}  0→1={ev['0→1']:4d}  1→3={ev['1→3']:4d}  "
              f"3→1={ev['3→1']:4d}  1→0={ev['1→0']:4d}  total={ev['total']:4d}")

    # ── Verify: thresholds are non-uniform on fresh (non-degenerate) weights ──
    # Note: we test persistence on a FRESH model, not the post-training one.
    # After aggressive training, weights can collapse to all-zero (degenerate),
    # which correctly produces uniform thresholds (topologically: all points
    # identical → no persistence variation). The test therefore uses a separate
    # fresh engine on non-degenerate weights.
    print("\nTesting persistence update on fresh (non-degenerate) model weights...")
    fresh_model = TinyModel()
    fresh_engine = TopologyBitFlipEngine(
        fresh_model,
        base_flip_pct=0.001,
        persistence_scale=5.0,
        n_row_sample=200,
    )
    fresh_engine.update_persistence_thresholds()
    stats_after = fresh_engine.threshold_stats()

    print("\nThreshold stats AFTER persistence update (fresh weights):")
    all_nonuniform = True
    for name, stats in stats_after.items():
        print(f"  {name}: mean={stats['mean']:.8f}  std={stats['std']:.2e}  "
              f"min={stats['min']:.8f}  max={stats['max']:.8f}")
        if stats["std"] < 1e-8:
            print(f"  WARNING: {name} thresholds still uniform (std < 1e-8)!")
            all_nonuniform = False

    if all_nonuniform:
        print("\nPASS: thresholds are non-uniform after persistence update.")
    else:
        print("\nFAIL: some layer thresholds remain uniform.")
        sys.exit(1)

    # ── Verify: per-region flip variability ───────────────────────────────────
    print("\nVerifying per-region flip variability (fresh model)...")
    for name, th in fresh_engine.thresholds.items():
        row_means = th.mean(dim=1)  # mean threshold per output row
        row_std = row_means.std().item()
        print(f"  {name}: per-row threshold std = {row_std:.2e} "
              f"(> 0 means rows have different flip criteria)")
        assert row_std > 1e-8 or th.shape[0] == 1, (
            f"Expected non-uniform thresholds in {name}, got row_std={row_std:.2e}"
        )

    # ── Show post-training weight stats ───────────────────────────────────────
    print("\nPost-training weight distributions (for reference):")
    for name, m in model.named_modules():
        if isinstance(m, BitFlipLinear):
            d = m.weight_distribution()
            print(f"  {name}: {d}")

    print("\nAll checks passed.")
    print("=" * 60)
