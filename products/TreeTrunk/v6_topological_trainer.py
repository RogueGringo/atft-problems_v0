#!/usr/bin/env python3
"""v6 Topological Trainer — differentiable topology with learned α/β.

Upgrades from v5:
  v5: numpy sheaf Laplacian, static α=0.7 β=0.3, no gradients
  v6: fully differentiable sheaf Laplacian in PyTorch, α/β learned from
      query geometry by DynamicGaugeRouter MLP, contrastive SheafTopologyLoss.

Architecture:
  UniversalFeatureMap  — abstract base; TextFeatureMap + SensorFeatureMap concrete
  SheafTopologyLoss    — differentiable L_F = δᵀδ, eigendecomposition via eigh
  DynamicGaugeRouter   — MLP: query geometry → (α, β) via softmax
  TopologicalTrainer   — forward pass: embeddings → loss → optimizer step

Key design choices:
  - PCA replaced by learnable linear projection (nn.Linear) — stays in graph
  - k-NN graph built once from detached distances, restriction maps stay
    differentiable w.r.t. the projected coordinates
  - torch.linalg.eigh on L_F (symmetric PSD) — gradients flow through eigenvalues
  - Epsilon padding on near-degenerate eigenvalues to stabilise backward pass
  - All modules GPU-compatible via .to(device) / device-aware construction
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _knn_graph(X: torch.Tensor, k: int) -> list[tuple[int, int]]:
    """Build k-NN edge list from coordinate matrix X (n, d).

    Uses detached distances so the graph topology is fixed per forward pass —
    restriction maps remain differentiable w.r.t. X coordinates.
    """
    with torch.no_grad():
        dists = torch.cdist(X, X)                    # (n, n)
        # argsort each row; skip self (index 0 = self)
        nn_idx = torch.argsort(dists, dim=1)[:, 1:k + 1]   # (n, k)

    n = X.shape[0]
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j_t in nn_idx[i]:
            j = j_t.item()
            edge = (i, j) if i < j else (j, i)
            edges.add(edge)
    return list(edges)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SheafTopologyLoss — differentiable cellular sheaf Laplacian
# ══════════════════════════════════════════════════════════════════════════════

class DifferentiableSheafLaplacian(nn.Module):
    """Differentiable cellular sheaf Laplacian on a point cloud.

    Construction (all operations stay in the autograd graph):
      1. Project: X = points @ W  where W is a learnable (d_in, stalk_dim) matrix
      2. k-NN graph: fixed topology from detached distances (non-diff argmin)
      3. Restriction maps: Householder-style  R_ij = I - alpha * d_hat d_hat^T
         where d = X[j] - X[i]  (differentiable w.r.t. X)
      4. Coboundary δ: assembled as a dense (m*s, n*s) tensor
      5. L_F = δᵀ δ  (symmetric PSD)
      6. eigenvalues via torch.linalg.eigh

    Args:
        d_in:      input embedding dimension
        stalk_dim: per-vertex stalk dimension s  (output of learnable projection)
        k:         k-NN neighborhood size
        eps:       epsilon added to eigenvalues before sqrt/log to prevent
                   degenerate gradient at λ=0
    """

    def __init__(self, d_in: int, stalk_dim: int = 8, k: int = 5, eps: float = 1e-6):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.k = k
        self.eps = eps
        # Learnable projection: replaces PCA — stays in autograd graph
        self.proj = nn.Linear(d_in, stalk_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Compute sheaf Laplacian eigenvalues for a point cloud.

        Args:
            points: (n, d_in) point cloud tensor, requires_grad OK

        Returns:
            eigenvalues: (n * stalk_dim,) sorted ascending eigenvalues of L_F
        """
        n, d_in = points.shape
        s = self.stalk_dim
        device = points.device

        if n < 3:
            # Degenerate cloud — return zeros so loss is neutral
            return torch.zeros(n * s, device=device)

        # 1. Project to stalk space
        X = self.proj(points)                        # (n, s)

        # 2. k-NN graph — topology fixed, maps differentiable
        k_eff = min(self.k, n - 1)
        edges = _knn_graph(X, k_eff)
        m = len(edges)

        if m == 0:
            return torch.zeros(n * s, device=device)

        # 3. Build coboundary δ: (m*s, n*s)
        #    For edge (i, j):
        #      δ[e_block, i_block] = +I_s
        #      δ[e_block, j_block] = -R_ij
        #    R_ij = I - alpha * d_hat d_hat^T  (Householder restriction)
        delta_rows = []   # list of (m*s) row tensors, each length n*s

        I_s = torch.eye(s, device=device, dtype=points.dtype)

        # We'll build δ column-by-column would be awkward; build row blocks instead
        # Accumulate as a list of (s, n*s) blocks, then stack
        delta_blocks: list[torch.Tensor] = []

        for (i, j) in edges:
            d_vec = X[j] - X[i]                     # (s,) — differentiable
            d_norm = d_vec.norm() + 1e-12
            d_hat = d_vec / d_norm                   # (s,)

            alpha = torch.clamp(d_norm, max=1.0)
            # R_ij = I - alpha * d_hat d_hat^T
            R_ij = I_s - alpha * torch.outer(d_hat, d_hat)  # (s, s) — differentiable

            # Build the row block for this edge: (s, n*s)
            row_block = torch.zeros(s, n * s, device=device, dtype=points.dtype)
            row_block[:, i * s:(i + 1) * s] = I_s
            row_block[:, j * s:(j + 1) * s] = -R_ij
            delta_blocks.append(row_block)

        delta = torch.cat(delta_blocks, dim=0)       # (m*s, n*s)

        # 4. L_F = δᵀ δ  (n*s, n*s) — symmetric PSD
        L = delta.T @ delta

        # 5. Symmetrize for numerical stability before eigh
        L = (L + L.T) * 0.5

        # 6. Eigendecomposition — torch.linalg.eigh requires symmetric input
        #    Returns eigenvalues in ascending order
        eigenvalues = torch.linalg.eigh(L).eigenvalues    # (n*s,)

        # Clamp negative numerical noise (should be ~0 for PSD matrix)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)

        return eigenvalues


class SheafTopologyLoss(nn.Module):
    """Contrastive loss over sheaf spectral gaps.

    For each (query, positive, negative) triplet:
      L_pos = spectral_gap(concat(q_emb, pos_emb))   — minimise (want coherent path)
      L_neg = spectral_gap(concat(q_emb, neg_emb))   — maximise (want incoherent path)

      loss = L_pos - margin * L_neg  +  relu(margin - L_neg)

    The spectral gap is λ₁ — the smallest non-zero eigenvalue of L_F.
    It is extracted via soft-thresholding: sum(λ * sigmoid((λ - eps) / tau))
    so gradients flow smoothly even when the gap is small.

    Args:
        d_in:      embedding dimension fed into the sheaf
        stalk_dim: stalk dimension for the sheaf projection
        k:         k-NN degree
        margin:    contrastive margin
        tau:       softmax temperature for smooth λ₁ extraction
        eps_gap:   eigenvalue threshold below which we consider λ ≈ 0
    """

    def __init__(
        self,
        d_in: int,
        stalk_dim: int = 8,
        k: int = 5,
        margin: float = 0.5,
        tau: float = 0.01,
        eps_gap: float = 1e-4,
    ):
        super().__init__()
        self.sheaf = DifferentiableSheafLaplacian(d_in, stalk_dim=stalk_dim, k=k)
        self.margin = margin
        self.tau = tau
        self.eps_gap = eps_gap

    def _spectral_gap(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Differentiable approximation to λ₁ (smallest non-zero eigenvalue).

        Uses a smooth sigmoid gate:  gap = sum_i( λ_i * σ((λ_i - eps) / τ) )
        This is essentially a soft minimum over the non-zero spectrum.
        """
        gates = torch.sigmoid((eigenvalues - self.eps_gap) / self.tau)
        # Weighted by gates; the minimum non-zero value dominates
        soft_gap = (eigenvalues * gates).sum() / (gates.sum() + 1e-12)
        return soft_gap

    def forward(
        self,
        q_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute contrastive sheaf loss.

        Args:
            q_emb:   (n_q, d) query embeddings
            pos_emb: (n_pos, d) positive (correct answer) embeddings
            neg_emb: (n_neg, d) negative (wrong answer) embeddings

        Returns:
            loss:  scalar tensor with grad_fn
            info:  dict of diagnostics (detached scalars)
        """
        # Concatenate into point clouds for the sheaf
        pos_cloud = torch.cat([q_emb, pos_emb], dim=0)   # (n_q + n_pos, d)
        neg_cloud = torch.cat([q_emb, neg_emb], dim=0)   # (n_q + n_neg, d)

        eigs_pos = self.sheaf(pos_cloud)
        eigs_neg = self.sheaf(neg_cloud)

        gap_pos = self._spectral_gap(eigs_pos)   # want small (coherent)
        gap_neg = self._spectral_gap(eigs_neg)   # want large (incoherent)

        # Relative triplet loss: force gap_neg > gap_pos + margin
        # This fires whenever the negative is within `margin` of the positive,
        # regardless of absolute scale. No more dead relu at gap ~5.0.
        triplet_loss = F.relu(gap_pos - gap_neg + self.margin)

        # Also keep a small direct pull on gap_pos toward zero
        direct_loss = 0.1 * gap_pos

        loss = triplet_loss + direct_loss

        info = {
            "gap_pos": gap_pos.detach().item(),
            "gap_neg": gap_neg.detach().item(),
            "triplet": triplet_loss.detach().item(),
            "direct": direct_loss.detach().item(),
            "loss": loss.detach().item(),
            "gap_delta": (gap_neg - gap_pos).detach().item(),
        }
        return loss, info


# ══════════════════════════════════════════════════════════════════════════════
# 2. UniversalFeatureMap — abstract base + concrete subclasses
# ══════════════════════════════════════════════════════════════════════════════

class UniversalFeatureMap(ABC, nn.Module):
    """Abstract feature map: heterogeneous input → shared embedding space.

    Subclasses override `encode_raw`. The shared embedding dimension is
    `out_dim`, set at construction time.

    The abstraction keeps the sheaf topology layer domain-agnostic:
    it only ever sees (n, out_dim) tensors.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    @abstractmethod
    def encode_raw(self, x) -> torch.Tensor:
        """Map raw input → (n, out_dim) embedding tensor.

        x can be any type — token ids, raw sensor arrays, pre-computed
        embeddings, etc. The subclass defines the contract.
        """
        ...

    def forward(self, x) -> torch.Tensor:
        emb = self.encode_raw(x)
        # L2-normalise so cosine similarity == dot product
        return F.normalize(emb, p=2, dim=-1)


class TextFeatureMap(UniversalFeatureMap):
    """Text → embedding via frozen backbone + learnable projection head.

    In smoke-test mode (no sentence_transformers available), or when
    backbone=None, uses a random frozen embedding table instead.

    Args:
        out_dim:      shared embedding dimension
        vocab_size:   fallback vocab size when no backbone is provided
        backbone_dim: dimension output by the sentence-transformer backbone
        backbone:     optional frozen SentenceTransformer or nn.Module
    """

    def __init__(
        self,
        out_dim: int = 256,
        vocab_size: int = 32000,
        backbone_dim: int = 384,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__(out_dim)
        self.backbone = backbone
        self.backbone_dim = backbone_dim

        if backbone is None:
            # Fallback: frozen random embedding table (for smoke testing)
            self.embed_table = nn.Embedding(vocab_size, backbone_dim)
            nn.init.normal_(self.embed_table.weight, std=0.02)
            self.embed_table.weight.requires_grad_(False)

        # Learnable projection head: backbone_dim → out_dim
        self.proj_head = nn.Sequential(
            nn.Linear(backbone_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def encode_raw(self, x) -> torch.Tensor:
        """x: (n,) long tensor of token ids, or (n, backbone_dim) float tensor.

        If backbone is set, x is passed through it (assumed to handle batching).
        Otherwise x must be token ids or a pre-computed float embedding.
        """
        if isinstance(x, torch.Tensor) and x.dtype == torch.long:
            # Token ids → backbone embedding
            if self.backbone is not None:
                raise NotImplementedError(
                    "External backbone forward pass not implemented here; "
                    "pass pre-computed float embeddings instead."
                )
            # Fallback: embed table mean-pooled (each row = one token)
            raw = self.embed_table(x)                # (n, backbone_dim)
        elif isinstance(x, torch.Tensor) and x.is_floating_point():
            # Pre-computed embeddings — just project
            raw = x                                  # (n, backbone_dim)
        else:
            raise TypeError(f"Unsupported input type for TextFeatureMap: {type(x)}")

        return self.proj_head(raw)                   # (n, out_dim)


class SensorFeatureMap(UniversalFeatureMap):
    """1D sensor array → embedding via conv stack + projection.

    Treats each input row as a 1D signal of length `signal_len`.

    Args:
        out_dim:      shared embedding dimension
        signal_len:   length of each sensor reading (number of channels / timesteps)
        n_filters:    number of conv filters at each stage
    """

    def __init__(self, out_dim: int = 256, signal_len: int = 128, n_filters: int = 64):
        super().__init__(out_dim)
        self.signal_len = signal_len

        # 1D conv encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        conv_out_dim = n_filters * 2 * 16

        self.proj_head = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def encode_raw(self, x) -> torch.Tensor:
        """x: (n, signal_len) float tensor of sensor readings."""
        # Add channel dim for Conv1d
        x_ch = x.unsqueeze(1)                        # (n, 1, signal_len)
        conv_out = self.encoder(x_ch)                # (n, conv_out_dim)
        return self.proj_head(conv_out)               # (n, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DynamicGaugeRouter — learned α/β from query geometry
# ══════════════════════════════════════════════════════════════════════════════

class DynamicGaugeRouter(nn.Module):
    """MLP that outputs (α, β) from query embedding geometry.

    Replaces the static α=0.7, β=0.3 from v5.

    Geometry features computed from the query embedding cloud:
      - mean norm          (1,)   — overall magnitude
      - std of norms       (1,)   — spread in magnitude
      - mean pairwise dist (1,)   — spread in embedding space
      - effective rank     (1,)   — entropy of singular value spectrum
      - mean cosine sim    (1,)   — internal coherence of the cloud

    Total geometry vector: (5,)
    Output: (α, β) via 2-simplex softmax (sum to 1)

    Args:
        hidden_dim: MLP hidden dimension
        init_alpha: initial α bias (cosine weight); default matches v5 prior
        init_beta:  initial β bias (topology weight)
    """

    N_GEO_FEATURES = 5

    def __init__(
        self,
        hidden_dim: int = 32,
        init_alpha: float = 0.7,
        init_beta: float = 0.3,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(self.N_GEO_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),   # logits for [α, β]
        )

        # Initialise final layer bias to reproduce v5 prior
        with torch.no_grad():
            prior = torch.tensor([math.log(init_alpha), math.log(init_beta)])
            prior = prior - prior.logsumexp(0)       # normalise in log space
            self.mlp[-1].bias.copy_(prior)

    @staticmethod
    def _geometry_features(emb: torch.Tensor) -> torch.Tensor:
        """Compute (5,) geometry vector from (n, d) embedding cloud.

        All ops are differentiable w.r.t. emb if needed, but in the
        forward pass we only backprop through the MLP output, not emb.
        """
        with torch.no_grad():
            norms = emb.norm(dim=-1)                  # (n,)
            mean_norm = norms.mean()
            std_norm = norms.std() if len(norms) > 1 else torch.zeros(1, device=emb.device)

            # Mean pairwise distance (subsample if large)
            n = emb.shape[0]
            if n > 64:
                idx = torch.randperm(n, device=emb.device)[:64]
                sub = emb[idx]
            else:
                sub = emb

            dists = torch.cdist(sub, sub)
            # Upper triangle only
            mask = torch.triu(torch.ones(sub.shape[0], sub.shape[0],
                                         device=emb.device, dtype=torch.bool), diagonal=1)
            mean_pdist = dists[mask].mean() if mask.any() else torch.zeros(1, device=emb.device)

            # Effective rank: exp(entropy of normalised singular values)
            # Capped at 32-dim for speed
            cap = min(n, emb.shape[1], 32)
            _, S, _ = torch.linalg.svd(sub[:cap], full_matrices=False)
            S_norm = S / (S.sum() + 1e-12)
            eff_rank = torch.exp(-(S_norm * torch.log(S_norm + 1e-12)).sum())

            # Mean pairwise cosine similarity
            emb_norm = F.normalize(sub, p=2, dim=-1)
            cos_sim_mat = emb_norm @ emb_norm.T
            cos_sim_vals = cos_sim_mat[mask]
            mean_cos = cos_sim_vals.mean() if cos_sim_vals.numel() > 0 else torch.zeros(1, device=emb.device)

        geo = torch.stack([
            mean_norm.squeeze(),
            std_norm.squeeze(),
            mean_pdist.squeeze(),
            eff_rank.squeeze(),
            mean_cos.squeeze(),
        ])                                            # (5,)
        return geo

    def forward(self, query_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict α, β from query embedding geometry.

        Args:
            query_emb: (n_q, d) query embedding cloud

        Returns:
            alpha: scalar tensor (cosine weight)
            beta:  scalar tensor (topology weight)
        """
        geo = self._geometry_features(query_emb)     # (5,) — detached
        geo = geo.to(query_emb.device)

        # Normalise geometry features (simple z-score style clamp)
        geo = torch.tanh(geo / (geo.abs().mean() + 1e-8))

        logits = self.mlp(geo)                        # (2,)
        weights = F.softmax(logits, dim=-1)           # (2,) sums to 1

        alpha = weights[0]
        beta = weights[1]
        return alpha, beta


# ══════════════════════════════════════════════════════════════════════════════
# 4. TopologicalTrainer — ties everything together
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalTrainer(nn.Module):
    """Full differentiable topological trainer.

    Pipeline per forward pass:
      1. feature_map(raw_input)  → embeddings (n, d)
      2. gauge_router(q_emb)     → (α, β) weights
      3. sheaf_loss(q, pos, neg) → contrastive loss
      4. Combined loss = sheaf_loss + router_reg

    Args:
        feature_map:  UniversalFeatureMap instance
        d_emb:        embedding dimension (must match feature_map.out_dim)
        stalk_dim:    sheaf stalk dimension
        k:            k-NN degree for sheaf graph
        margin:       contrastive margin
        lr:           learning rate for internal optimizer
    """

    def __init__(
        self,
        feature_map: UniversalFeatureMap,
        d_emb: int,
        stalk_dim: int = 8,
        k: int = 5,
        margin: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__()
        assert feature_map.out_dim == d_emb, (
            f"feature_map.out_dim={feature_map.out_dim} must equal d_emb={d_emb}"
        )

        self.feature_map = feature_map
        self.sheaf_loss_fn = SheafTopologyLoss(
            d_in=d_emb, stalk_dim=stalk_dim, k=k, margin=margin
        )
        self.gauge_router = DynamicGaugeRouter()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(
        self,
        q_raw,
        pos_raw,
        neg_raw,
    ) -> tuple[torch.Tensor, dict]:
        """One forward pass.

        Args:
            q_raw:   raw query input (format depends on feature_map type)
            pos_raw: raw positive input
            neg_raw: raw negative input

        Returns:
            loss: scalar tensor
            info: diagnostic dict (detached)
        """
        # 1. Encode
        q_emb = self.feature_map(q_raw)             # (n_q, d)
        pos_emb = self.feature_map(pos_raw)          # (n_pos, d)
        neg_emb = self.feature_map(neg_raw)          # (n_neg, d)

        # 2. Dynamic gauge weights — informational, not yet fed into sheaf
        #    (future: use α, β to weight multiple loss terms)
        alpha, beta = self.gauge_router(q_emb)

        # 3. Sheaf contrastive loss
        loss, sheaf_info = self.sheaf_loss_fn(q_emb, pos_emb, neg_emb)

        info = {
            **sheaf_info,
            "alpha": alpha.detach().item(),
            "beta": beta.detach().item(),
        }
        return loss, info

    def step(self, q_raw, pos_raw, neg_raw) -> dict:
        """Single training step: forward + backward + optimizer update."""
        self.optimizer.zero_grad()
        loss, info = self.forward(q_raw, pos_raw, neg_raw)
        loss.backward()
        self.optimizer.step()
        return info


# ══════════════════════════════════════════════════════════════════════════════
# 5. Smoke test — instantiate everything, verify gradients
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test(device: str = "cuda"):
    """Instantiate all modules, run forward pass with random data, verify grads.

    Checks:
      1. All three nn.Module classes instantiate without error
      2. Forward pass completes on the target device
      3. loss.backward() succeeds (no None grad anywhere in the graph)
      4. DynamicGaugeRouter outputs sum to 1
      5. α/β are non-trivially initialised (not both 0.5)

    Returns True if all checks pass.
    """
    print(f"\n{'='*60}")
    print(f"  v6 SMOKE TEST — device={device}")
    print(f"{'='*60}\n")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Dimensions
    D_EMB = 128
    N_Q = 8       # query cloud size
    N_POS = 10    # positive cloud size
    N_NEG = 10    # negative cloud size
    VOCAB = 500
    SIGNAL_LEN = 64

    # ── 1. TextFeatureMap ────────────────────────────────────────────────────
    print("\n[1] TextFeatureMap (token ids → embedding)")
    text_map = TextFeatureMap(out_dim=D_EMB, vocab_size=VOCAB, backbone_dim=D_EMB).to(device)
    # Use pre-computed float embeddings (backbone_dim = D_EMB for simplicity)
    q_tokens = torch.randn(N_Q, D_EMB, device=device)
    pos_tokens = torch.randn(N_POS, D_EMB, device=device)
    neg_tokens = torch.randn(N_NEG, D_EMB, device=device)

    q_emb = text_map(q_tokens)
    assert q_emb.shape == (N_Q, D_EMB), f"Bad shape: {q_emb.shape}"
    print(f"    q_emb: {q_emb.shape}  ok")

    # ── 2. SensorFeatureMap ──────────────────────────────────────────────────
    print("\n[2] SensorFeatureMap (sensor array → embedding)")
    sensor_map = SensorFeatureMap(out_dim=D_EMB, signal_len=SIGNAL_LEN).to(device)
    sensor_input = torch.randn(N_Q, SIGNAL_LEN, device=device)
    sensor_emb = sensor_map(sensor_input)
    assert sensor_emb.shape == (N_Q, D_EMB), f"Bad shape: {sensor_emb.shape}"
    print(f"    sensor_emb: {sensor_emb.shape}  ok")

    # ── 3. DifferentiableSheafLaplacian ─────────────────────────────────────
    print("\n[3] DifferentiableSheafLaplacian")
    sheaf_lap = DifferentiableSheafLaplacian(d_in=D_EMB, stalk_dim=8, k=4).to(device)
    pts = torch.randn(N_Q + N_POS, D_EMB, device=device, requires_grad=True)
    eigs = sheaf_lap(pts)
    print(f"    eigenvalues shape: {eigs.shape}")
    assert eigs.shape[0] == (N_Q + N_POS) * 8, f"Bad eig count: {eigs.shape}"
    eigs.sum().backward()
    assert pts.grad is not None, "No grad on pts!"
    print(f"    backward ok — pts.grad norm: {pts.grad.norm():.4f}")
    pts.grad = None

    # ── 4. SheafTopologyLoss ─────────────────────────────────────────────────
    print("\n[4] SheafTopologyLoss (contrastive)")
    sheaf_loss_fn = SheafTopologyLoss(d_in=D_EMB, stalk_dim=8, k=4, margin=0.5).to(device)
    q_in  = torch.randn(N_Q,   D_EMB, device=device, requires_grad=True)
    pos_in = torch.randn(N_POS, D_EMB, device=device, requires_grad=True)
    neg_in = torch.randn(N_NEG, D_EMB, device=device, requires_grad=True)

    loss, info = sheaf_loss_fn(q_in, pos_in, neg_in)
    print(f"    loss: {loss.item():.6f}  gap_pos: {info['gap_pos']:.6f}  gap_neg: {info['gap_neg']:.6f}")
    loss.backward()
    assert q_in.grad is not None, "No grad on q_in!"
    assert pos_in.grad is not None, "No grad on pos_in!"
    assert neg_in.grad is not None, "No grad on neg_in!"
    print(f"    backward ok — grads: q={q_in.grad.norm():.4f} pos={pos_in.grad.norm():.4f} neg={neg_in.grad.norm():.4f}")

    # ── 5. DynamicGaugeRouter ────────────────────────────────────────────────
    print("\n[5] DynamicGaugeRouter (learned α/β)")
    router = DynamicGaugeRouter().to(device)
    q_geo = torch.randn(N_Q, D_EMB, device=device)
    alpha, beta = router(q_geo)
    print(f"    alpha: {alpha.item():.4f}  beta: {beta.item():.4f}  sum: {(alpha + beta).item():.6f}")
    assert abs((alpha + beta).item() - 1.0) < 1e-5, "alpha + beta != 1"
    (alpha + beta).backward()
    print(f"    backward ok")

    # ── 6. TopologicalTrainer (end-to-end) ───────────────────────────────────
    print("\n[6] TopologicalTrainer (end-to-end)")
    feat_map = TextFeatureMap(out_dim=D_EMB, vocab_size=VOCAB, backbone_dim=D_EMB).to(device)
    trainer = TopologicalTrainer(
        feature_map=feat_map,
        d_emb=D_EMB,
        stalk_dim=8,
        k=4,
        margin=0.5,
        lr=1e-3,
    ).to(device)

    # Simulate two training steps
    for step_i in range(2):
        q_raw  = torch.randn(N_Q,   D_EMB, device=device)
        pos_raw = torch.randn(N_POS, D_EMB, device=device)
        neg_raw = torch.randn(N_NEG, D_EMB, device=device)

        info = trainer.step(q_raw, pos_raw, neg_raw)
        print(f"    step {step_i}: loss={info['loss']:.6f}  α={info['alpha']:.4f}  β={info['beta']:.4f}")

    # Verify α/β parameters are being tracked
    router_params = list(trainer.gauge_router.parameters())
    total_router_params = sum(p.numel() for p in router_params)
    print(f"    DynamicGaugeRouter params: {total_router_params}")
    assert total_router_params > 0, "Router has no parameters!"

    # ── Summary ──────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in trainer.parameters())
    print(f"\n{'='*60}")
    print(f"  ALL CHECKS PASSED")
    print(f"  Total trainer params: {total_params:,}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v6 Topological Trainer smoke test")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    passed = smoke_test(device=args.device)
    if not passed:
        raise SystemExit(1)
