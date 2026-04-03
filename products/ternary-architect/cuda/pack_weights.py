#!/usr/bin/env python3
"""Pack and unpack {0,1,3} ternary weights to/from 2-bit packed uint32.

Encoding:
  0 → 00   (void)
  1 → 01   (unit)
  3 → 11   (prime)
  2 → 01   (defensive — maps to 1 if encountered)

16 weights per uint32 (2 bits × 16 = 32 bits).
8× memory compression vs fp32, 4× vs fp16.
"""
from __future__ import annotations

import torch


# ── bit layout ────────────────────────────────────────────────────────────
# Weight k occupies bits [2k+1 : 2k] of the uint32 (LSB = weight 0).
# That is: packed >>= (2 * position_within_word) & 0x3

WEIGHTS_PER_WORD = 16


def pack_ternary(weights: torch.Tensor) -> torch.Tensor:
    """Pack (N, K) tensor with values {0,1,3} into (N, K//16) uint32.

    K must be a multiple of 16.

    Encoding:
      0 → 0b00   skip
      1 → 0b01   copy
      3 → 0b11   triple-add
      2 → 0b01   defensive (maps to 1)

    Returns a CPU uint32 tensor of shape (N, K // 16).
    """
    assert weights.ndim == 2, f"Expected 2D tensor, got shape {weights.shape}"
    N, K = weights.shape
    assert K % WEIGHTS_PER_WORD == 0, (
        f"K={K} must be divisible by {WEIGHTS_PER_WORD}"
    )

    # Work on CPU int32 (torch has no uint32 dtype in older versions)
    w = weights.detach().cpu().float().round().long()  # (N, K), values in {0,1,2,3}

    # Defensive: clamp to 0-3, remap 2→1
    w = w.clamp(0, 3)
    w[w == 2] = 1

    # Encode: 0→0, 1→1, 3→3  (already matches 2-bit: 0=00, 1=01, 3=11)
    # (value 2=10 is excluded by the defensive remap above)

    # Reshape to (N, K//16, 16) for bit packing
    w = w.view(N, K // WEIGHTS_PER_WORD, WEIGHTS_PER_WORD)  # (N, W, 16)

    # Pack 16 × 2-bit values into one uint32
    packed = torch.zeros(N, K // WEIGHTS_PER_WORD, dtype=torch.int32)
    for bit_pos in range(WEIGHTS_PER_WORD):
        packed |= (w[:, :, bit_pos].to(torch.int32) << (2 * bit_pos))

    return packed  # (N, K//16) int32 (bit-identical to uint32)


def unpack_ternary(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack (N, K//16) int32/uint32 → (N, K) float tensor with values {0,1,3}.

    Parameters
    ----------
    packed : torch.Tensor
        Shape (N, K//16), dtype int32 (or int64 — truncated to 32 bits).
    K : int
        Original number of columns. Must be a multiple of 16.

    Returns
    -------
    torch.Tensor
        Shape (N, K), dtype float32, values in {0.0, 1.0, 3.0}.
    """
    assert packed.ndim == 2
    N, W = packed.shape
    assert W == K // WEIGHTS_PER_WORD, (
        f"packed width {W} != K // {WEIGHTS_PER_WORD} = {K // WEIGHTS_PER_WORD}"
    )

    packed = packed.cpu().to(torch.int32)  # ensure 32-bit

    # Extract 16 × 2-bit fields per word
    out = torch.zeros(N, W, WEIGHTS_PER_WORD, dtype=torch.int32)
    for bit_pos in range(WEIGHTS_PER_WORD):
        out[:, :, bit_pos] = (packed >> (2 * bit_pos)) & 0x3

    # out values: 0→0.0, 1→1.0, 3→3.0, 2→clip to 1.0 (defensive)
    out_f = out.float()
    out_f[out == 2] = 1.0

    return out_f.view(N, K)  # (N, K)


# ── self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    N, K = 64, 128

    # Random ternary weight matrix
    idx = torch.randint(0, 3, (N, K))
    values = torch.tensor([0.0, 1.0, 3.0])
    W = values[idx]  # (N, K)

    packed = pack_ternary(W)
    print(f"Original shape: {W.shape}  ({W.numel() * 4} bytes fp32)")
    print(f"Packed shape:   {packed.shape}  ({packed.numel() * 4} bytes uint32)")
    print(f"Compression:    {W.numel() * 4 / (packed.numel() * 4):.1f}x")

    W2 = unpack_ternary(packed, K)
    max_err = (W - W2).abs().max().item()
    print(f"Max roundtrip error: {max_err}")
    assert max_err == 0.0, "ROUNDTRIP FAILED"
    print("Pack/unpack roundtrip: PASS")

    # Edge: all zeros
    W_zero = torch.zeros(N, K)
    p = pack_ternary(W_zero)
    assert (p == 0).all(), "Zero pack failed"
    assert (unpack_ternary(p, K) == 0).all(), "Zero unpack failed"
    print("All-zeros test: PASS")

    # Edge: all threes
    W_three = torch.full((N, K), 3.0)
    p = pack_ternary(W_three)
    u = unpack_ternary(p, K)
    assert (u == 3.0).all(), f"All-3 unpack failed, got {u.unique()}"
    print("All-threes test: PASS")

    print("ALL TESTS PASSED")
