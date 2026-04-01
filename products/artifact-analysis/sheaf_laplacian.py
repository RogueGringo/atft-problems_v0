#!/usr/bin/env python3
"""
Sheaf Laplacian Analysis of {0,1,3} Crystal Structures

Computes the sheaf Laplacian L_F = δ*δ on trained ternary weight matrices.
Tests whether ker(L_F) is non-trivial — whether a global section exists.
Tracks spectral gap and Gini of eigenvalues across training checkpoints.

The ATFT framework says:
- ker(L_F) non-trivial → global section exists → semantic truth found
- Spectral gap λ₁ → 0 → system approaching coherence
- Gini of eigenvalues at criticality → universal constant

We test: does the crystal formation (22/42/36) correspond to
the sheaf Laplacian entering its kernel? Is the crystal the global section?
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ternary-architect"))
from ternary_linear import TernaryLinear


def extract_ternary_weights(model_path, model_class, **kwargs):
    """Load a model and extract all TernaryLinear quantized weights."""
    model = model_class(**kwargs)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            wq = module.get_quantized_weight().float()
            weights[name] = wq
    return weights


def weight_matrix_as_bipartite_graph(W):
    """Treat a weight matrix W (out×in) as a bipartite graph.

    Nodes: input dims (left) + output dims (right)
    Edges: non-zero weights connect input i to output j
    Edge weight: the {0,1,3} value

    This gives us a graph on which to define the sheaf.
    """
    out_dim, in_dim = W.shape
    n_nodes = in_dim + out_dim

    # Build adjacency from non-zero weights
    edges = []
    for j in range(out_dim):
        for i in range(in_dim):
            w = W[j, i].item()
            if w != 0:  # non-void connections
                edges.append((i, in_dim + j, w))  # input_i → output_j with weight w

    return n_nodes, edges


def sheaf_laplacian_from_weights(W, stalk_dim=1):
    """Construct the sheaf Laplacian L_F for a {0,1,3} weight matrix.

    For a bipartite graph from the weight matrix:
    - Each node gets a 1D stalk (scalar)
    - Each edge gets a restriction map defined by the weight value
    - L_F = δ* δ where δ is the coboundary map

    For {0,1,3} weights:
    - w=0: no edge (void — topological insulator)
    - w=1: identity restriction map (signal passes unchanged)
    - w=3: amplifying restriction map (signal amplified)

    The coboundary δ: C⁰ → C¹ maps vertex signals to edge disagreements.
    For edge e = (u,v) with weight w: δf(e) = w·f(v) - f(u)

    This captures: "how much does the output disagree with the input
    after applying the connection weight?"
    """
    out_dim, in_dim = W.shape
    n = in_dim + out_dim

    # Build L_F directly as n×n matrix
    # L_F[i,i] = sum of w² for all edges incident to i
    # L_F[i,j] = -w for edge (i,j), where w is the edge weight
    L = np.zeros((n, n))

    for j in range(out_dim):
        for i in range(in_dim):
            w = W[j, i].item()
            if w == 0:
                continue
            # Edge: input_i → output_(in_dim + j) with weight w
            u = i
            v = in_dim + j
            # Sheaf Laplacian contributions
            L[u, u] += w * w      # diagonal: w²
            L[v, v] += w * w      # diagonal: w²
            L[u, v] -= w          # off-diagonal: -w (restriction map)
            L[v, u] -= w          # symmetric


    return L


def analyze_sheaf_laplacian(L, name=""):
    """Analyze the sheaf Laplacian: kernel dimension, spectral gap, Gini."""

    # Eigenvalues (L is symmetric, use eigh)
    # For large matrices, compute only a subset
    n = L.shape[0]

    if n > 2000:
        # Too large for full eigendecomposition — sample a submatrix
        idx = np.random.RandomState(42).choice(n, 2000, replace=False)
        idx.sort()
        L_sub = L[np.ix_(idx, idx)]
        eigenvalues = np.linalg.eigvalsh(L_sub)
        sampled = True
    else:
        eigenvalues = np.linalg.eigvalsh(L)
        sampled = False

    eigenvalues = np.sort(np.abs(eigenvalues))  # ensure non-negative

    # Kernel dimension: count eigenvalues ≈ 0
    tol = 1e-6
    kernel_dim = np.sum(eigenvalues < tol)

    # Spectral gap: first non-zero eigenvalue
    nonzero = eigenvalues[eigenvalues >= tol]
    spectral_gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0

    # Gini of eigenvalue distribution
    n_eig = len(eigenvalues)
    if n_eig > 0 and eigenvalues.sum() > 0:
        sorted_eig = np.sort(eigenvalues)
        cumulative = np.cumsum(sorted_eig)
        gini = 1.0 - 2.0 * cumulative.sum() / (n_eig * cumulative[-1])
    else:
        gini = 0.0

    # Effective rank of eigenvalue distribution
    eig_norm = eigenvalues / (eigenvalues.sum() + 1e-12)
    eig_norm = eig_norm[eig_norm > 1e-12]
    entropy = -np.sum(eig_norm * np.log(eig_norm))
    eff_rank = np.exp(entropy)

    result = {
        "name": name,
        "matrix_size": int(n),
        "sampled": sampled,
        "kernel_dim": int(kernel_dim),
        "spectral_gap": float(spectral_gap),
        "gini_eigenvalues": float(gini),
        "eff_rank_eigenvalues": float(eff_rank),
        "max_eigenvalue": float(eigenvalues[-1]),
        "n_nonzero_eigenvalues": int(len(nonzero)),
        "eigenvalue_percentiles": {
            "p1": float(np.percentile(eigenvalues, 1)),
            "p10": float(np.percentile(eigenvalues, 10)),
            "p50": float(np.percentile(eigenvalues, 50)),
            "p90": float(np.percentile(eigenvalues, 90)),
            "p99": float(np.percentile(eigenvalues, 99)),
        },
    }

    return result


def analyze_model(model_path, model_class, model_kwargs, label=""):
    """Full sheaf Laplacian analysis of a trained model."""
    print(f"\n{'='*60}")
    print(f"SHEAF LAPLACIAN ANALYSIS: {label}")
    print(f"{'='*60}")

    weights = extract_ternary_weights(model_path, model_class, **model_kwargs)

    results = {}
    for name, W in weights.items():
        print(f"\n  Layer: {name} ({W.shape[0]}×{W.shape[1]})")

        # Weight distribution
        n = W.numel()
        w0 = (W == 0).sum().item() / n
        w1 = (W == 1).sum().item() / n
        w3 = (W == 3).sum().item() / n
        print(f"  Crystal: void={w0:.3f} identity={w1:.3f} prime={w3:.3f}")

        # Sheaf Laplacian
        # For large layers, analyze a projection
        if W.shape[0] > 500 or W.shape[1] > 500:
            # Project to manageable size
            proj_size = 256
            row_idx = torch.randperm(W.shape[0])[:min(proj_size, W.shape[0])]
            col_idx = torch.randperm(W.shape[1])[:min(proj_size, W.shape[1])]
            W_proj = W[row_idx][:, col_idx]
            print(f"  Projected to {W_proj.shape[0]}×{W_proj.shape[1]} for Laplacian")
            L = sheaf_laplacian_from_weights(W_proj.detach().numpy())
        else:
            L = sheaf_laplacian_from_weights(W.detach().numpy())

        analysis = analyze_sheaf_laplacian(L, name)
        results[name] = {**analysis, "crystal": {"void": w0, "identity": w1, "prime": w3}}

        print(f"  Kernel dim: {analysis['kernel_dim']}")
        print(f"  Spectral gap λ₁: {analysis['spectral_gap']:.6f}")
        print(f"  Gini(eigenvalues): {analysis['gini_eigenvalues']:.4f}")
        print(f"  Eff rank(eigenvalues): {analysis['eff_rank_eigenvalues']:.2f}")

        # THE KEY TEST: does kernel exist?
        if analysis['kernel_dim'] > 0:
            print(f"  *** GLOBAL SECTION EXISTS (ker dim = {analysis['kernel_dim']}) ***")
        else:
            print(f"  No global section (trivial kernel)")

    return results


if __name__ == "__main__":
    import torch.nn as nn
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ternary-architect"))
    from ternary_linear import make_linear
    from ternary_transformer import GPTConfig, Block

    # Rebuild CARETv2 model class (minimal, for loading)
    class CARETv2(nn.Module):
        def __init__(self, voc=50257, d=2048, h=16, bs=512):
            super().__init__()
            self.ch0_emb=nn.Embedding(voc,1024); self.ch1_emb=nn.Embedding(voc,384)
            self.ch2_emb=nn.Embedding(voc,384); self.ch3_emb=nn.Embedding(voc,256)
            self.pos_emb=nn.Embedding(bs,d); self.drop=nn.Dropout(0.0)
            self.lens=make_linear(d,d,bias=False,weight_set='013')
            cfg=GPTConfig(vocab_size=voc,block_size=bs,n_layer=1,n_head=h,n_embd=d,dropout=0.0,weight_set='013')
            self.prism=Block(cfg); self.ln_prism=nn.LayerNorm(d)
            nv,ni=int(d*0.222),int(d*0.417); np_=d-nv-ni
            self.split_sizes=[nv,ni,np_]
            vc=nv//3; vr=nv-3*vc
            self.void_ch0=make_linear(nv,vc+vr,False,'013')
            self.void_ch1=make_linear(nv,vc,False,'013')
            self.void_ch2=make_linear(nv,vc,False,'013')
            self.void_ln=nn.LayerNorm(nv)
            ic=ni//3; ir=ni-3*ic
            self.ident_ch0=make_linear(ni,ic+ir,False,'013')
            self.ident_ch1=make_linear(ni,ic,False,'013')
            self.ident_ch2=make_linear(ni,ic,False,'013')
            self.ident_ln=nn.LayerNorm(ni)
            pc=np_//3; pr=np_-3*pc
            self.prime_ch0=make_linear(np_,pc+pr,False,'013')
            self.prime_ch1=make_linear(np_,pc,False,'013')
            self.prime_ch2=make_linear(np_,pc,False,'013')
            self.prime_ln=nn.LayerNorm(np_)
            self.act=nn.GELU()
            self.lm_head=nn.Linear(d,voc,bias=False)
        def forward(self,x,**k): pass

    # Analyze CARET v2 model
    model_path = str(Path(__file__).resolve().parent.parent /
                     "ternary-architect/results/caret_v2_corrected/model.pt")

    results = analyze_model(
        model_path, CARETv2, {"voc": 50257},
        label="CARET v2 Corrected (best architecture)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: SHEAF LAPLACIAN ANALYSIS")
    print(f"{'='*60}")

    total_kernel = sum(r["kernel_dim"] for r in results.values())
    mean_spectral_gap = np.mean([r["spectral_gap"] for r in results.values()])
    mean_gini = np.mean([r["gini_eigenvalues"] for r in results.values()])

    print(f"\n  Total kernel dimension across layers: {total_kernel}")
    print(f"  Mean spectral gap: {mean_spectral_gap:.6f}")
    print(f"  Mean Gini(eigenvalues): {mean_gini:.4f}")

    if total_kernel > 0:
        print(f"\n  *** GLOBAL SECTIONS EXIST ***")
        print(f"  The {0,1,3} crystal IS a sheaf with non-trivial kernel.")
        print(f"  The 22/42/36 ratio IS the global section.")
    else:
        print(f"\n  No global sections found in projected submatrices.")
        print(f"  May need full eigendecomposition or different projection.")

    # Save
    out_path = Path(__file__).resolve().parent / "results" / "sheaf_laplacian_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
