"""Module 1: Prime Basis Seed

Computes a prime subspace from Wierzbicka's 59 semantic primes.
For each prime phrase, run it through the model, extract the last-token
hidden state at the target layer (n_layers // 3), build an orthonormal
basis via SVD, then project the full vocabulary to measure how
"prime-aligned" each token is.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from semantic_primes import ALL_PRIMES, PRIME_TO_CATEGORY  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results" / "basis_discovery"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. compute_prime_basis
# ---------------------------------------------------------------------------

def compute_prime_basis(model, tokenizer, device="cuda") -> dict:
    """Extract hidden states for all 59 primes and build an orthonormal basis.

    Returns
    -------
    dict with keys:
        prime_basis      : torch.Tensor (59, hidden_dim)  — Vt (orthonormal rows)
        singular_values  : torch.Tensor (59,)
        prime_vectors    : torch.Tensor (59, hidden_dim)  — raw last-token hs
        target_layer     : int
        hidden_dim       : int
    """
    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 3
    print(f"[prime_basis] n_layers={n_layers}, target_layer={target_layer}")

    model.eval()
    prime_vectors = []

    with torch.no_grad():
        for prime in ALL_PRIMES:
            inputs = tokenizer(prime, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
            hs = outputs.hidden_states[target_layer][0, -1, :].float().cpu()
            prime_vectors.append(hs)

    prime_vectors_t = torch.stack(prime_vectors, dim=0)  # (59, hidden_dim)
    hidden_dim = prime_vectors_t.shape[1]
    print(f"[prime_basis] prime_vectors shape: {prime_vectors_t.shape}")

    # Mean-centre then SVD
    prime_mean = prime_vectors_t.mean(dim=0)  # (hidden_dim,)
    centered = prime_vectors_t - prime_mean.unsqueeze(0)
    # torch.linalg.svd returns (U, S, Vh) where Vh is already the transpose
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Vh shape: (59, hidden_dim) — rows are orthonormal basis vectors
    # Drop the first singular vector: it corresponds to the dominant mean-direction
    # shared by virtually all tokens (sv[0] >> sv[1..]).  Keeping it causes
    # prime_ratio ≈ 1 for every token since that direction is in ALL hidden states.
    Vh_trimmed = Vh[1:]   # (58, hidden_dim)
    S_trimmed  = S[1:]    # (58,)

    print(f"[prime_basis] SVD complete. Basis shape (trimmed): {Vh_trimmed.shape}, "
          f"top-3 singular values (after trim): {S_trimmed[:3].tolist()}")

    return {
        "prime_basis": Vh_trimmed,       # (58, hidden_dim)
        "singular_values": S_trimmed,    # (58,)
        "prime_vectors": prime_vectors_t,  # (59, hidden_dim) raw
        "prime_mean": prime_mean,          # (hidden_dim,)  centroid for projection
        "target_layer": target_layer,
        "hidden_dim": hidden_dim,
    }


# ---------------------------------------------------------------------------
# 2. project_vocabulary
# ---------------------------------------------------------------------------

def project_vocabulary(
    model,
    tokenizer,
    prime_basis: torch.Tensor,
    target_layer: int,
    device: str = "cuda",
    batch_size: int = 64,
    prime_mean: torch.Tensor = None,
) -> dict:
    """Project every vocabulary token onto the prime subspace.

    Parameters
    ----------
    prime_basis : (59, hidden_dim) orthonormal rows (Vt from SVD)

    Returns
    -------
    dict with keys:
        prime_ratios    : np.ndarray (vocab_size,)  — ||proj|| / ||hs||
        residual_norms  : np.ndarray (vocab_size,)
        mean, std, median : float statistics of prime_ratios
        vocab_size      : int
    """
    vocab_size = tokenizer.vocab_size
    print(f"[project_vocab] vocab_size={vocab_size}, batch_size={batch_size}")

    P = prime_basis.float().to(device)  # (58, hidden_dim) — trimmed basis
    # If a prime_mean centroid is provided, subtract it from token hidden states
    # before projecting so that the global mean direction (stripped from P) is
    # not counted toward prime_ratio.
    mean_vec = None
    if prime_mean is not None:
        mean_vec = prime_mean.float().to(device).unsqueeze(0)  # (1, hidden_dim)
    # Projection matrix: P.T @ P — projects onto prime subspace
    # proj = hs @ (P.T @ P)  but computed as (hs @ P.T) @ P for efficiency
    # We compute per-batch: proj = (hs @ P.T) @ P

    model.eval()
    prime_ratios = np.zeros(vocab_size, dtype=np.float32)
    residual_norms = np.zeros(vocab_size, dtype=np.float32)

    n_batches = (vocab_size + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, vocab_size)
            token_ids = list(range(start, end))

            # Build (batch, 1) input tensor
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(device)
            # attention_mask: all ones
            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # hidden_states[target_layer]: (batch, 1, hidden_dim)
            hs = outputs.hidden_states[target_layer][:, 0, :].float()  # (batch, hidden_dim)

            # Mean-centre token hidden states the same way prime vectors were centred
            if mean_vec is not None:
                hs_centered = hs - mean_vec
            else:
                hs_centered = hs

            # Project onto prime subspace (mean-centred coords)
            # coords shape: (batch, 58)
            coords = hs_centered @ P.T
            # projected vector in hidden_dim space: (batch, hidden_dim)
            proj = coords @ P

            proj_norms = torch.norm(proj, dim=1)                     # (batch,)
            hs_norms   = torch.norm(hs_centered, dim=1)              # (batch,)
            res_norms  = torch.norm(hs_centered - proj, dim=1)       # (batch,)

            ratios = (proj_norms / (hs_norms + 1e-10)).cpu().numpy()
            prime_ratios[start:end] = ratios
            residual_norms[start:end] = res_norms.cpu().numpy()

            if batch_idx % 100 == 0:
                print(f"[project_vocab] batch {batch_idx}/{n_batches} "
                      f"(tokens {start}-{end-1}), "
                      f"mean_ratio={ratios.mean():.4f}")

    stats = {
        "prime_ratios": prime_ratios,
        "residual_norms": residual_norms,
        "mean": float(prime_ratios.mean()),
        "std": float(prime_ratios.std()),
        "median": float(np.median(prime_ratios)),
        "vocab_size": vocab_size,
    }
    print(f"[project_vocab] Done. prime_ratio mean={stats['mean']:.4f}, "
          f"std={stats['std']:.4f}, median={stats['median']:.4f}")
    return stats


# ---------------------------------------------------------------------------
# 3. run_module1
# ---------------------------------------------------------------------------

def run_module1(model_name: str, device: str = "cuda", use_awq: bool = False) -> dict:
    """Load model, compute prime basis, project vocabulary, save results.

    Returns combined results dict.
    """
    print(f"[run_module1] model={model_name}, device={device}, use_awq={use_awq}")

    # --- Load model ---
    if use_awq or "AWQ" in model_name:
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
        model = awq.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            output_hidden_states=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Module 1a: prime basis ---
    basis_results = compute_prime_basis(model, tokenizer, device=device)

    # --- Module 1b: vocabulary projection ---
    vocab_results = project_vocabulary(
        model,
        tokenizer,
        prime_basis=basis_results["prime_basis"],
        target_layer=basis_results["target_layer"],
        device=device,
        prime_mean=basis_results.get("prime_mean"),
    )

    # --- Save ---
    # Build a short model name for the filename (last path component, no slashes)
    model_short = model_name.replace("/", "_").replace("\\", "_")
    out_path = RESULTS_DIR / f"prime_basis_{model_short}.pt"

    save_dict = {
        "model_name": model_name,
        "target_layer": basis_results["target_layer"],
        "hidden_dim": basis_results["hidden_dim"],
        "prime_basis": basis_results["prime_basis"],
        "singular_values": basis_results["singular_values"],
        "prime_vectors": basis_results["prime_vectors"],
        "prime_mean": basis_results.get("prime_mean"),
        "prime_ratios": torch.from_numpy(vocab_results["prime_ratios"]),
        "residual_norms": torch.from_numpy(vocab_results["residual_norms"]),
        "stats": {
            "mean": vocab_results["mean"],
            "std": vocab_results["std"],
            "median": vocab_results["median"],
            "vocab_size": vocab_results["vocab_size"],
        },
        "all_primes": ALL_PRIMES,
        "prime_to_category": PRIME_TO_CATEGORY,
    }
    torch.save(save_dict, out_path)
    print(f"[run_module1] Saved results to {out_path}")

    # --- Cleanup ---
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined = {**basis_results, **vocab_results, "save_path": str(out_path)}
    return combined


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Module 1 — Prime Basis Seed on {model_name} (device={device})")

    t0 = time.time()
    results = run_module1(model_name, device=device)
    elapsed = time.time() - t0

    print("\n=== Module 1 Summary ===")
    print(f"  target_layer     : {results['target_layer']}")
    print(f"  hidden_dim       : {results['hidden_dim']}")
    print(f"  prime_basis shape: {results['prime_basis'].shape}")
    print(f"  singular_values  : {results['singular_values'][:5].tolist()} ...")
    print(f"  vocab_size       : {results['vocab_size']}")
    print(f"  prime_ratio mean : {results['mean']:.4f}")
    print(f"  prime_ratio std  : {results['std']:.4f}")
    print(f"  prime_ratio median: {results['median']:.4f}")
    print(f"  saved to         : {results['save_path']}")
    print(f"  elapsed          : {elapsed:.1f}s")
