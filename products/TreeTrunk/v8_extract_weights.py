#!/usr/bin/env python3
"""v8 Weight Extraction — PyTorch → NumPy air gap.

Extracts the trained v7 TextFeatureMap projection weights, detaches
from the gradient graph, and saves as pure NumPy arrays for FAISS
LinearTransform ingestion.

Critical: FAISS LinearTransform expects weight as (d_in, d_out) = (384, 128).
PyTorch nn.Linear stores weight as (d_out, d_in) = (128, 384).
We MUST transpose.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

CHECKPOINT = Path(__file__).parent / "results" / "v7_checkpoints" / "v7_trainer_final.pt"
OUTPUT_DIR = Path(__file__).parent / "results" / "v8_faiss_prep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract():
    print("Loading v7 checkpoint...")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    # ── Inventory all keys ────────────────────────────────────────────
    print(f"\nCheckpoint keys ({len(checkpoint)}):")
    feat_keys = [k for k in checkpoint if k.startswith("feature_map.")]
    sheaf_keys = [k for k in checkpoint if k.startswith("sheaf_loss_fn.")]
    router_keys = [k for k in checkpoint if k.startswith("gauge_router.")]

    print(f"  feature_map: {len(feat_keys)} tensors")
    print(f"  sheaf_loss:  {len(sheaf_keys)} tensors")
    print(f"  gauge_router: {len(router_keys)} tensors")

    # ── Extract TextFeatureMap projection head ────────────────────────
    # The head is: Linear(384, 256) → GELU → Linear(256, 128)
    # We need the full two-layer projection, not just one linear.
    # For FAISS LinearTransform, we need a single (384, 128) matrix.
    # We'll extract both layers and compose them.

    print(f"\nTextFeatureMap layers:")
    for k in feat_keys:
        shape = checkpoint[k].shape
        print(f"  {k}: {shape}")

    # The head structure from v6_topological_trainer.py:
    #   self.head = nn.Sequential(
    #       nn.Linear(backbone_dim, out_dim * 2),  # 384 → 256
    #       nn.GELU(),
    #       nn.Linear(out_dim * 2, out_dim),        # 256 → 128
    #   )
    # Keys: feature_map.proj_head.0.weight, .0.bias, .2.weight, .2.bias

    W1 = checkpoint["feature_map.proj_head.0.weight"].detach().float().numpy()  # (256, 384)
    b1 = checkpoint["feature_map.proj_head.0.bias"].detach().float().numpy()    # (256,)
    W2 = checkpoint["feature_map.proj_head.2.weight"].detach().float().numpy()  # (128, 256)
    b2 = checkpoint["feature_map.proj_head.2.bias"].detach().float().numpy()    # (128,)

    print(f"\n  W1: {W1.shape} (384 → 256)")
    print(f"  b1: {b1.shape}")
    print(f"  W2: {W2.shape} (256 → 128)")
    print(f"  b2: {b2.shape}")

    # For a linear-only FAISS transform, we'd compose: W_total = W2 @ W1
    # But GELU is nonlinear — we can't collapse to a single matrix.
    # Save both layers separately. The FAISS pipeline will need a
    # custom two-stage PreTransform, or we accept the linear approximation.

    # Option A: Save individual layers for custom pipeline
    np.save(OUTPUT_DIR / "v7_W1.npy", W1)
    np.save(OUTPUT_DIR / "v7_b1.npy", b1)
    np.save(OUTPUT_DIR / "v7_W2.npy", W2)
    np.save(OUTPUT_DIR / "v7_b2.npy", b2)

    # Option B: Linear approximation — compose W2 @ W1, b2 + W2 @ b1
    # This drops the GELU nonlinearity but gives a single (384, 128) matrix
    # that FAISS LinearTransform can ingest directly.
    W_linear = W2 @ W1                     # (128, 384) — PyTorch layout
    b_linear = b2 + W2 @ b1                # (128,)

    # FAISS wants (d_in, d_out) = (384, 128) — TRANSPOSE
    W_faiss = W_linear.T                   # (384, 128)
    b_faiss = b_linear                     # (128,)

    np.save(OUTPUT_DIR / "v7_projection_weight.npy", W_faiss)
    np.save(OUTPUT_DIR / "v7_projection_bias.npy", b_faiss)

    print(f"\n  W_faiss (linear approx): {W_faiss.shape}  ← (384, 128) for FAISS")
    print(f"  b_faiss: {b_faiss.shape}")

    # ── Also extract sheaf Laplacian projection ───────────────────────
    sheaf_proj = checkpoint["sheaf_loss_fn.sheaf.proj.weight"].detach().float().numpy()
    print(f"\n  Sheaf projection: {sheaf_proj.shape}")  # (8, 128)
    np.save(OUTPUT_DIR / "v7_sheaf_proj.npy", sheaf_proj)

    # ── Verify ────────────────────────────────────────────────────────
    print(f"\nSaved to {OUTPUT_DIR}/:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.suffix == ".npy":
            arr = np.load(p)
            print(f"  {p.name:35s} {str(arr.shape):>15s}  {arr.dtype}")

    # Quick sanity: pass a random 384-dim vector through both paths
    x = np.random.randn(384).astype(np.float32)

    # PyTorch path (with GELU)
    h = W1 @ x + b1
    h = h * 0.5 * (1 + np.vectorize(lambda v: float(torch.erf(torch.tensor(v / np.sqrt(2)))))(h))
    y_torch = W2 @ h + b2

    # Linear approx path (no GELU)
    y_linear = W_linear @ x + b_linear

    cosine = np.dot(y_torch, y_linear) / (np.linalg.norm(y_torch) * np.linalg.norm(y_linear))
    rmse = np.sqrt(np.mean((y_torch - y_linear) ** 2))

    print(f"\n  Linear approx quality:")
    print(f"    Cosine(torch, linear): {cosine:.4f}")
    print(f"    RMSE:                  {rmse:.4f}")
    print(f"    Note: cosine < 0.95 means GELU matters — use two-stage pipeline")


if __name__ == "__main__":
    extract()
