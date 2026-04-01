#!/usr/bin/env python3
"""TernaryLinear — The {0, 1, 3} linear layer.

Each weight is one of three values:
  0 (void)   — no connection, structured absence
  1 (unit)   — identity transport, connection exists
  3 (prime)  — irreducible amplification, generates structure

Training: latent fp32 weights + straight-through estimator (STE).
Inference: no floating-point multiplier needed.
  x * 0 = skip (FREE)
  x * 1 = pass (FREE)
  x * 3 = (x << 1) + x (CHEAP)

2 bits per weight. Same as BitNet b1.58.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Quantization functions ────────────────────────────────────────────────

import math

SQRT2 = math.sqrt(2)

WEIGHT_SETS = {
    "013": torch.tensor([0.0, 1.0, 3.0]),
    "n101": torch.tensor([-1.0, 0.0, 1.0]),  # BitNet
    "012": torch.tensor([0.0, 1.0, 2.0]),     # ablation
    "015": torch.tensor([0.0, 1.0, 5.0]),     # ablation
    "017": torch.tensor([0.0, 1.0, 7.0]),     # ablation
    "0123": torch.tensor([0.0, 1.0, 2.0, 3.0]),  # 4-level ablation
    # The prime sequence — transport maps ARE the primes
    # 0=void, 1=unit, √2=diagonal(2 geometricized), 3,5,7,11=irreducible amplifiers
    "primes": torch.tensor([0.0, 1.0, SQRT2, 3.0, 5.0, 7.0, 11.0]),
}


def quantize_to_set(w: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Snap each weight to the nearest value in the set.

    This is the forward quantization — deterministic, differentiable via STE.
    """
    values = values.to(w.device, w.dtype)
    # Expand for broadcasting: w is (...,), values is (k,)
    # distances: (..., k)
    dists = (w.unsqueeze(-1) - values).abs()
    indices = dists.argmin(dim=-1)
    return values[indices]


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator for discrete quantization.

    Forward: snap to nearest value in set.
    Backward: gradient passes through as if quantization didn't happen.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return quantize_to_set(w, values)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradient through unchanged
        return grad_output, None


# ── TernaryLinear layer ───────────────────────────────────────────────────

class TernaryLinear(nn.Module):
    """Linear layer with ternary weights from a configurable value set.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
        Whether to include a learnable bias (fp32, not quantized).
    weight_set : str
        Key into WEIGHT_SETS. Default "013" = {0, 1, 3}.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, weight_set: str = "013"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_set = weight_set

        # Register the value set as a buffer (not a parameter)
        self.register_buffer("values", WEIGHT_SETS[weight_set].clone())

        # Latent weights — what the optimizer sees (fp32, continuous)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self, init_mode: str = "uniform"):
        """Initialize latent weights.

        Modes:
          uniform: spread across the value set (original, good for shallow nets)
          identity: bias toward 1 (identity transport) — critical for deep nets.
          mixed: 70% near 1, 15% near 0, 15% near 3. Gives the crystal
                 all three building blocks from birth. Let training decide.
          void: all weights start centered at 0 (void-biased) with enough
                thermal noise (std=0.4) that ~40% can cross the 0→1 boundary.
                The model starts mostly-void but CAN activate connections.
                It must decide which to keep, which to amplify to 3, and which
                to leave dark. Self-organization, not hand-designed ratios.
                v1 (std=0.01) was too cold — 100% stuck at zero, never activated.
                v2 (std=0.4) gives the STE room to explore.
                Hypothesis: model discovers its own 0/1/3 ratio rather than
                inheriting one from init. Compare to mixed (15/70/15 frozen)
                and the old run (binary collapse to 0/100/0 w3).
        """
        v = self.values
        if init_mode == "void":
            # Born from void — biased toward zero but with thermal noise
            # std=0.4 puts ~40% of weights past the 0→1 boundary (at 0.5)
            # so the network can activate connections if gradients demand it.
            # v1 had std=0.01 → everything trapped at zero forever.
            nn.init.normal_(self.weight, mean=0.0, std=0.4)
        elif init_mode == "mixed" and 1.0 in v.tolist() and 3.0 in v.tolist():
            # Mixed: seed all three regions so the crystal has amplifiers from birth
            # 70% identity highways, 15% voids, 15% amplifiers
            n = self.weight.numel()
            flat = torch.empty(n)
            n_void = int(0.15 * n)
            n_prime = int(0.15 * n)
            n_unit = n - n_void - n_prime
            # Each group centered on its target with tight spread
            flat[:n_unit] = torch.normal(mean=1.0, std=0.2, size=(n_unit,))
            flat[n_unit:n_unit+n_void] = torch.normal(mean=0.0, std=0.15, size=(n_void,))
            flat[n_unit+n_void:] = torch.normal(mean=3.0, std=0.3, size=(n_prime,))
            # Shuffle so they're not in blocks
            flat = flat[torch.randperm(n)]
            self.weight.data = flat.view_as(self.weight)
        elif init_mode == "identity" and 1.0 in v.tolist():
            nn.init.normal_(self.weight, mean=1.0, std=0.3)
        else:
            lo, hi = v.min().item(), v.max().item()
            nn.init.uniform_(self.weight, lo - 0.3, hi + 0.3)

    def get_quantized_weight(self) -> torch.Tensor:
        """Return the quantized weight matrix (for inference / diagnostics)."""
        return STEQuantize.apply(self.weight, self.values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = STEQuantize.apply(self.weight, self.values)
        return F.linear(x, w_q, self.bias)

    def weight_distribution(self) -> dict[str, float]:
        """Return fraction of weights at each value (diagnostic)."""
        with torch.no_grad():
            w_q = quantize_to_set(self.weight, self.values)
            total = w_q.numel()
            dist = {}
            for v in self.values:
                count = (w_q == v.item()).sum().item()
                dist[f"w={v.item():.0f}"] = count / total
            return dist

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, set={self.weight_set} "
                f"values={self.values.tolist()}")


class FP16Linear(nn.Module):
    """Standard linear layer (fp16/fp32) as baseline control."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_set: str = "fp16"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_set = weight_set

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def weight_distribution(self) -> dict[str, str]:
        return {"type": "continuous", "std": f"{self.linear.weight.std().item():.4f}"}

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def get_quantized_weight(self) -> torch.Tensor:
        return self.linear.weight.data

    def extra_repr(self) -> str:
        return self.linear.extra_repr()


# ── BitFlip: Quake III-style discrete training ──────────────────────────

class _BitFlipSTE(torch.autograd.Function):
    """Identity forward, STE backward — for discrete weights.

    The weight IS discrete ({0,1,3}). Forward passes it through unchanged.
    Backward passes gradient through for the BitFlipEngine to read.
    This prevents autograd from building a full computation graph for
    the weight tensor (which would OOM on large models).
    """
    @staticmethod
    def forward(ctx, w):
        # Pass through unchanged — weight is already {0,1,3}
        return w

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitFlipLinear(nn.Module):
    """Linear layer with TRULY discrete {0,1,3} weights. No latent floats.

    Quake III fast inverse sqrt insight: don't solve the problem in the
    wrong domain. STE tries to cross quantization boundaries in continuous
    space — expensive, fails. BitFlip operates directly on discrete weights.

    In 2 bits:
      00 = 0 (void)    — no connection
      01 = 1 (unit)    — identity transport
      11 = 3 (amplifier) — irreducible amplification
      10 = 2 (unused)  — unstable transition state

    Training = gradient-informed bit flips, not gradient descent on latents.
    The gradient tells us WHICH bit to flip and WHEN.
    The 'magic constant' (flip threshold) controls approximation quality.

    Weights are nn.Parameter for autograd, but NEVER updated by optimizer.
    BitFlipEngine reads .grad, accumulates signal, flips bits.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, weight_set: str = "bitflip"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_set = weight_set

        # Discrete weights — always exactly {0, 1, 3}
        # Parameter so autograd flows, but excluded from optimizer
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self, init_mode: str = "mixed"):
        """Initialize discrete weights.

        Unlike TernaryLinear, there are no latent floats to set —
        weights are placed directly at {0, 1, 3}.
        """
        with torch.no_grad():
            n = self.weight.numel()
            if init_mode == "mixed":
                # 70% one, 15% zero, 15% three — same as TernaryLinear mixed
                codes = torch.zeros(n, dtype=torch.long)
                n_void = int(0.15 * n)
                n_prime = int(0.15 * n)
                n_unit = n - n_void - n_prime
                codes[:n_unit] = 1        # identity
                codes[n_unit:n_unit+n_void] = 0  # void
                codes[n_unit+n_void:] = 2  # will map to 3
                codes = codes[torch.randperm(n)]
                values = torch.tensor([0.0, 1.0, 3.0])
                self.weight.data = values[codes].view_as(self.weight)
            elif init_mode == "void":
                self.weight.data.zero_()
            elif init_mode == "identity":
                self.weight.data.fill_(1.0)
            elif init_mode == "uniform":
                codes = torch.randint(0, 3, (n,))
                values = torch.tensor([0.0, 1.0, 3.0])
                self.weight.data = values[codes].view_as(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Discrete weights through STE wrapper — prevents OOM from
        # autograd building full computation graph on weight tensor.
        # The weight IS {0,1,3}. No quantization needed.
        w = _BitFlipSTE.apply(self.weight)
        return F.linear(x, w, self.bias)

    def get_quantized_weight(self) -> torch.Tensor:
        return self.weight.data

    def weight_distribution(self) -> dict[str, float]:
        with torch.no_grad():
            w = self.weight.data
            total = w.numel()
            return {
                "w=0": (w == 0).sum().item() / total,
                "w=1": (w == 1).sum().item() / total,
                "w=3": (w == 3).sum().item() / total,
            }

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, BITFLIP discrete")


class BitFlipEngine:
    """Quake III-style training engine for discrete ternary weights.

    Replaces STE + optimizer for ternary params entirely.
    Accumulates gradient signal, then makes discrete bit-flip decisions.

    The gradient tells us two things:
      - SIGN: which direction the weight wants to move
        negative grad → want weight bigger (0→1→3)
        positive grad → want weight smaller (3→1→0)
      - MAGNITUDE: how urgently

    Flip decisions (one step at a time, like Newton refinement):
      0 → 1  (activate)    when grad strongly negative
      1 → 3  (amplify)     when grad strongly negative
      3 → 1  (de-amplify)  when grad strongly positive
      1 → 0  (deactivate)  when grad strongly positive

    The magic constant: flip_pct — what fraction of weights can flip per
    cycle. Like 0x5f3759df, it encodes the approximation quality.
    Too high = chaotic (Quake with bad constant = wrong answer).
    Too low = stuck (never converges).

    Parameters
    ----------
    flip_pct : float
        Max fraction of weights to flip per cycle. THE MAGIC CONSTANT.
    cycle_steps : int
        Accumulate gradient for this many optimizer steps, then flip.
    warmup_steps : int
        Don't flip before this many optimizer steps.
    gravity : float
        Discrete L2 — bias toward zero. Demotion urgency is multiplied by
        (1 + gravity), promotion urgency divided by (1 + gravity).
        0.0 = symmetric (no bias). 1.0 = demotions are 2x easier than
        promotions. This is weight decay in discrete space — active weights
        must justify their existence with higher gradient signal.
    """

    def __init__(self, model: nn.Module,
                 flip_pct: float = 0.001,
                 cycle_steps: int = 100,
                 warmup_steps: int = 500,
                 gravity: float = 0.0):
        self.model = model
        self.flip_pct = flip_pct
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.gravity = gravity
        self.optim_steps = 0
        self.accum_count = 0
        self.history: list[dict] = []

        # Gradient accumulators: magnitude and direction
        self.grad_mag: dict[str, torch.Tensor] = {}
        self.grad_dir: dict[str, torch.Tensor] = {}

        for name, m in model.named_modules():
            if isinstance(m, BitFlipLinear):
                self.grad_mag[name] = torch.zeros_like(m.weight.data)
                self.grad_dir[name] = torch.zeros_like(m.weight.data)

    def accumulate(self):
        """Call after backward. Reads and clears BitFlipLinear gradients."""
        for name, m in self.model.named_modules():
            if isinstance(m, BitFlipLinear) and m.weight.grad is not None:
                self.grad_mag[name] += m.weight.grad.abs()
                self.grad_dir[name] += m.weight.grad  # sign accumulates
                m.weight.grad = None  # don't let optimizer touch it
        self.accum_count += 1

    def maybe_flip(self, step: int) -> dict | None:
        """Call each optimizer step. Returns flip stats if triggered."""
        self.optim_steps += 1
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
                direction = self.grad_dir[name].sign()  # net direction

                n_flips = max(1, int(w.numel() * self.flip_pct))

                # Candidates: urgency score for each possible flip
                # Gravity: promotions (→1, →3) need more urgency,
                # demotions (→0, 3→1) need less. Discrete weight decay.
                g_up = 1.0 / (1.0 + self.gravity)    # promotion dampened
                g_down = 1.0 + self.gravity            # demotion boosted
                urgency = torch.zeros_like(w)

                # 0 → 1: activate where grad says "increase" (negative grad)
                mask_0_up = (w == 0) & (direction < 0)
                urgency[mask_0_up] = mag[mask_0_up] * g_up

                # 1 → 3: amplify where grad says "increase more"
                mask_1_up = (w == 1) & (direction < 0)
                urgency[mask_1_up] = mag[mask_1_up] * g_up

                # 3 → 1: de-amplify where grad says "decrease"
                mask_3_down = (w == 3) & (direction > 0)
                urgency[mask_3_down] = mag[mask_3_down] * g_down

                # 1 → 0: deactivate where grad says "decrease"
                mask_1_down = (w == 1) & (direction > 0)
                urgency[mask_1_down] = mag[mask_1_down] * g_down

                if urgency.sum() == 0:
                    continue

                # Top n_flips by urgency
                flat = urgency.flatten()
                k = min(n_flips, (flat > 0).sum().item())
                if k == 0:
                    continue
                threshold = flat.topk(k).values[-1]
                flip_mask = urgency >= threshold

                # Apply flips — one step at a time (the Newton refinement)
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

        stats = {"step": step, **flips,
                 "total": sum(flips.values())}
        self.history.append(stats)
        return stats


# ── Mutation Engine ──────────────────────────────────────────────────────

class TernaryMutator:
    """Four-direction boundary crossing for STE-based ternary training.

    The STE handles smooth within-basin optimization (what it's good at).
    This mutator handles boundary crossing (what STE can't do).
    Best of both worlds: STE + QuakeFlip.

    Uses gradient DIRECTION (sign) not just magnitude:
      0 → 1: activate (set latent to 1.0) — grad says "need signal here"
      1 → 3: amplify (set latent to 3.0) — grad says "need MORE signal"
      3 → 1: de-amplify (set latent to 1.0) — grad says "too much"
      1 → 0: deactivate (set latent to -0.3) — grad says "don't need this"

    Sets latent weights to basin CENTERS so STE can immediately fine-tune
    from a stable position. One Newton step per flip.

    Parameters
    ----------
    flip_pct : float
        Fraction of weights to flip per cycle. Gentle = 0.0005.
    cycle_steps : int
        Optimizer steps between flip cycles.
    warmup_steps : int
        No flips before this many optimizer steps.
    """

    def __init__(self, model: nn.Module,
                 flip_pct: float = 0.0005,
                 cycle_steps: int = 500,
                 warmup_steps: int = 1000):
        self.model = model
        self.flip_pct = flip_pct
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.optim_steps = 0
        self.accum_steps = 0
        self.history: list[dict] = []

        self.grad_mag: dict[str, torch.Tensor] = {}
        self.grad_dir: dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                self.grad_mag[name] = torch.zeros_like(module.weight.data)
                self.grad_dir[name] = torch.zeros_like(module.weight.data)

    def accumulate(self):
        """Call after each backward pass."""
        for name, module in self.model.named_modules():
            if isinstance(module, TernaryLinear) and module.weight.grad is not None:
                self.grad_mag[name] += module.weight.grad.abs()
                self.grad_dir[name] += module.weight.grad
        self.accum_steps += 1

    def maybe_mutate(self, step: int) -> dict | None:
        """Four-direction boundary crossing on latent weights."""
        self.optim_steps += 1
        if self.optim_steps < self.warmup_steps:
            return None
        if self.optim_steps % self.cycle_steps != 0:
            return None
        if self.accum_steps == 0:
            return None

        flips = {"0→1": 0, "1→3": 0, "3→1": 0, "1→0": 0}

        for name, module in self.model.named_modules():
            if not isinstance(module, TernaryLinear):
                continue
            if name not in self.grad_mag:
                continue

            with torch.no_grad():
                w = module.weight.data
                w_q = quantize_to_set(w, module.values)
                mag = self.grad_mag[name] / max(1, self.accum_steps)
                direction = self.grad_dir[name].sign()

                n_flips = max(1, int(w.numel() * self.flip_pct))
                urgency = torch.zeros_like(w)

                # Four directions — same logic as BitFlipEngine
                mask_0_up = (w_q == 0) & (direction < 0)
                urgency[mask_0_up] = mag[mask_0_up]

                mask_1_up = (w_q == 1) & (direction < 0)
                urgency[mask_1_up] = mag[mask_1_up]

                mask_3_down = (w_q == 3) & (direction > 0)
                urgency[mask_3_down] = mag[mask_3_down]

                mask_1_down = (w_q == 1) & (direction > 0)
                urgency[mask_1_down] = mag[mask_1_down]

                if urgency.sum() == 0:
                    continue

                flat = urgency.flatten()
                k = min(n_flips, int((flat > 0).sum().item()))
                if k == 0:
                    continue
                threshold = flat.topk(k).values[-1]
                flip_mask = urgency >= threshold

                # Set latent weights to basin centers — STE can fine-tune from here
                f01 = flip_mask & mask_0_up
                w[f01] = 1.0   # center of 1-basin
                flips["0→1"] += f01.sum().item()

                f13 = flip_mask & mask_1_up
                w[f13] = 3.0   # center of 3-basin
                flips["1→3"] += f13.sum().item()

                f31 = flip_mask & mask_3_down
                w[f31] = 1.0   # center of 1-basin
                flips["3→1"] += f31.sum().item()

                f10 = flip_mask & mask_1_down
                w[f10] = -0.3  # firmly in 0-basin (quantizes to 0)
                flips["1→0"] += f10.sum().item()

        for name in self.grad_mag:
            self.grad_mag[name].zero_()
            self.grad_dir[name].zero_()
        self.accum_steps = 0

        total = sum(flips.values())
        stats = {"step": step, **flips, "total": total}
        self.history.append(stats)
        return stats


def make_linear(in_features: int, out_features: int,
                bias: bool = True, weight_set: str = "013") -> nn.Module:
    """Factory: create the right linear layer for a given weight set."""
    if weight_set == "fp16":
        return FP16Linear(in_features, out_features, bias=bias)
    if weight_set == "bitflip":
        return BitFlipLinear(in_features, out_features, bias=bias)
    return TernaryLinear(in_features, out_features, bias=bias,
                         weight_set=weight_set)
