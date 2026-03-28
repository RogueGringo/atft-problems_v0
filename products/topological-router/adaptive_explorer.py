"""Module 2: Adaptive Basis Explorer

Starts from the prime basis (Module 1), iteratively probes a model's hidden
states with diverse prompts, discovers directions NOT covered by the current
basis, adds them, and tracks convergence.  Uses a bandit-style policy:
sample from whichever cognitive mode had the highest residual last iteration.
"""

import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import gini_fast  # noqa: E402
from prompts.loader import load_by_mode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GINI_EPSILON = 0.005        # convergence threshold on Gini change
CONVERGENCE_PATIENCE = 3    # consecutive stable iterations to declare convergence
MAX_ITERATIONS = 25
RESIDUAL_THRESHOLD = 0.01   # min singular-value ratio to keep a new direction
NEW_VECTORS_PER_ITER = 5    # max new basis vectors added per iteration

OUTPUT_DIR = Path(__file__).parent / "results" / "basis_discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. extract_hidden_states
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    target_layer: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Tokenize prompts, run forward passes, extract & mean-center hidden states.

    Returns
    -------
    torch.Tensor  (total_tokens, hidden_dim) — cpu float32, mean-centered
    """
    all_hs = []
    model.eval()

    with torch.no_grad():
        for text in prompts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[target_layer] → (1, seq_len, hidden_dim)
            hs = outputs.hidden_states[target_layer][0].float().cpu()
            all_hs.append(hs)

    # Concatenate all tokens → (total_tokens, hidden_dim)
    combined = torch.cat(all_hs, dim=0)
    # Mean-center
    combined = combined - combined.mean(dim=0, keepdim=True)
    return combined


# ---------------------------------------------------------------------------
# 2. compute_residual
# ---------------------------------------------------------------------------

def compute_residual(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Project hidden states onto basis and return the residual.

    Parameters
    ----------
    hidden_states : (N, D)
    basis         : (K, D)  — orthonormal rows

    Returns
    -------
    (residual, mean_residual_norm)
        residual : (N, D)
        mean_residual_norm : float — average L2 norm of residual vectors
    """
    # proj = hidden_states @ basis.T @ basis   →  (N, D)
    coords = hidden_states @ basis.T        # (N, K)
    proj = coords @ basis                   # (N, D)
    residual = hidden_states - proj
    mean_residual_norm = float(torch.norm(residual, dim=1).mean().item())
    return residual, mean_residual_norm


# ---------------------------------------------------------------------------
# 3. expand_basis
# ---------------------------------------------------------------------------

def expand_basis(
    basis: torch.Tensor,
    residual: torch.Tensor,
    n_new: int = NEW_VECTORS_PER_ITER,
) -> torch.Tensor:
    """Add new orthogonal directions from the residual to the basis.

    Steps:
        1. SVD of residual → top n_new right singular vectors
        2. Filter by magnitude: keep only those whose singular value
           exceeds RESIDUAL_THRESHOLD * S[0]
        3. Concatenate with existing basis
        4. QR decomposition to re-orthogonalize

    Returns
    -------
    torch.Tensor  (K_new, D) — expanded orthonormal basis
    """
    U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
    # Vt[:n_new] are the top new directions; S[:n_new] their magnitudes
    n_candidates = min(n_new, len(S))
    threshold = S[0].item() * RESIDUAL_THRESHOLD

    keep = []
    for i in range(n_candidates):
        if S[i].item() > threshold:
            keep.append(Vt[i])
    if len(keep) == 0:
        return basis  # nothing significant to add

    new_dirs = torch.stack(keep, dim=0)  # (n_keep, D)
    combined = torch.cat([basis, new_dirs], dim=0)  # (K + n_keep, D)

    # QR to re-orthogonalize (QR on transposed → columns become ortho rows)
    Q, R = torch.linalg.qr(combined.T, mode="reduced")  # Q: (D, K_new)
    # Rows of Q.T are orthonormal basis vectors
    new_basis = Q.T  # (K_new, D)
    return new_basis


# ---------------------------------------------------------------------------
# 4. adaptive_explore  (main loop)
# ---------------------------------------------------------------------------

def adaptive_explore(
    model,
    tokenizer,
    prime_basis: torch.Tensor,
    target_layer: int,
    device: str = "cuda",
) -> dict:
    """Bandit-guided iterative basis expansion.

    Returns
    -------
    dict with keys:
        adaptive_basis, convergence_trajectory, residual_history,
        basis_growth_log, converged, final_gini, final_basis_size
    """
    # Load prompt bank grouped by cognitive mode
    mode_prompts = load_by_mode()
    mode_names = sorted(mode_prompts.keys())
    print(f"[adaptive_explore] Loaded modes: {mode_names} "
          f"({sum(len(v) for v in mode_prompts.values())} total prompts)")

    basis = prime_basis.clone().float()

    convergence_trajectory = []   # list of gini values per iteration
    residual_history = []         # list of mean-residual-norm per iteration
    basis_growth_log = []         # list of dicts per iteration
    mode_residuals = {m: 0.0 for m in mode_names}
    stable_count = 0
    prev_gini = None
    converged = False

    for iteration in range(MAX_ITERATIONS):
        t0 = time.time()

        # ---- (a) Bandit prompt selection -----------------------------------
        if iteration == 0:
            # First pass: one prompt from each mode
            selected = []
            selected_mode = "all"
            for m in mode_names:
                selected.append(random.choice(mode_prompts[m])["text"])
        else:
            # Pick mode with highest residual
            best_mode = max(mode_residuals, key=mode_residuals.get)
            selected_mode = best_mode
            pool = mode_prompts[best_mode]
            k = min(5, len(pool))
            selected = [p["text"] for p in random.sample(pool, k)]

        # ---- (b) Extract hidden states ------------------------------------
        hs = extract_hidden_states(model, tokenizer, selected, target_layer, device)

        # ---- (c) Compute residual against current basis -------------------
        residual, mean_res = compute_residual(hs, basis)

        # Track per-mode residual for bandit policy
        if iteration == 0:
            # Assign aggregate residual to each mode (uniform first pass)
            for m in mode_names:
                mode_residuals[m] = mean_res
        else:
            mode_residuals[selected_mode] = mean_res

        # ---- (d) Expand basis if residual is significant ------------------
        old_size = basis.shape[0]
        if mean_res > RESIDUAL_THRESHOLD:
            basis = expand_basis(basis, residual, n_new=NEW_VECTORS_PER_ITER)
        n_added = basis.shape[0] - old_size

        # ---- (e) Convergence check via Gini of projected SVD spectrum -----
        # Project hidden states onto current basis, then SVD of the projection
        coords = hs @ basis.T                           # (N, K)
        sv = torch.linalg.svdvals(coords)               # (min(N,K),)
        gini = gini_fast(sv.numpy())

        if prev_gini is not None and abs(gini - prev_gini) < GINI_EPSILON:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= CONVERGENCE_PATIENCE:
            converged = True

        # ---- (f) Logging --------------------------------------------------
        elapsed = time.time() - t0
        log_entry = {
            "iteration": iteration,
            "n_added": n_added,
            "basis_size": basis.shape[0],
            "gini": gini,
            "mean_residual": mean_res,
            "mode": selected_mode,
            "stable_count": stable_count,
            "elapsed_s": round(elapsed, 2),
        }
        convergence_trajectory.append(gini)
        residual_history.append(mean_res)
        basis_growth_log.append(log_entry)
        prev_gini = gini

        # ---- (g) Print progress -------------------------------------------
        print(f"  iter {iteration:>2d} | basis {basis.shape[0]:>4d} | "
              f"+{n_added:<3d} | gini {gini:.4f} | "
              f"residual {mean_res:.4f} | mode={selected_mode} | "
              f"stable={stable_count} | {elapsed:.1f}s")

        if converged:
            print(f"[adaptive_explore] Converged at iteration {iteration} "
                  f"(Gini stable for {CONVERGENCE_PATIENCE} iterations)")
            break

    return {
        "adaptive_basis": basis,
        "convergence_trajectory": convergence_trajectory,
        "residual_history": residual_history,
        "basis_growth_log": basis_growth_log,
        "converged": converged,
        "final_gini": convergence_trajectory[-1] if convergence_trajectory else 0.0,
        "final_basis_size": basis.shape[0],
    }


# ---------------------------------------------------------------------------
# 5. run_module2
# ---------------------------------------------------------------------------

def run_module2(
    model_name: str,
    prime_basis_path: Path | str,
    device: str = "cuda",
) -> dict:
    """Load prime basis, load model, run adaptive exploration, save results."""
    prime_basis_path = Path(prime_basis_path)
    print(f"[run_module2] model={model_name}, prime_basis={prime_basis_path}")

    # --- Load prime basis from Module 1 ---
    saved = torch.load(prime_basis_path, map_location="cpu", weights_only=True)
    prime_basis = saved["prime_basis"].float()
    target_layer = int(saved["target_layer"])
    print(f"[run_module2] Loaded prime basis: shape={prime_basis.shape}, "
          f"target_layer={target_layer}")

    # --- Load model ---
    if "AWQ" in model_name:
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

    # --- Run adaptive exploration ---
    print(f"\n{'='*60}")
    print(f"  Module 2 — Adaptive Basis Explorer")
    print(f"{'='*60}")

    t0 = time.time()
    results = adaptive_explore(model, tokenizer, prime_basis, target_layer, device)
    elapsed = time.time() - t0

    # --- Save ---
    model_short = model_name.replace("/", "_").replace("\\", "_")
    # Strip common prefixes for shorter filenames
    for prefix in ["Qwen_", ""]:
        if model_short.startswith(prefix):
            short_name = model_short[len(prefix):] if prefix else model_short
            break
    out_path = OUTPUT_DIR / f"adaptive_basis_{model_short}.pt"

    save_dict = {
        "adaptive_basis": results["adaptive_basis"],
        "convergence_trajectory": results["convergence_trajectory"],
        "residual_history": results["residual_history"],
        "basis_growth_log": results["basis_growth_log"],
        "metadata": {
            "model_name": model_name,
            "prime_basis_path": str(prime_basis_path),
            "target_layer": target_layer,
            "prime_basis_size": prime_basis.shape[0],
            "final_basis_size": results["final_basis_size"],
            "converged": results["converged"],
            "final_gini": results["final_gini"],
            "n_iterations": len(results["basis_growth_log"]),
            "elapsed_s": round(elapsed, 1),
        },
    }
    torch.save(save_dict, out_path)

    print(f"\n{'='*60}")
    print(f"  Module 2 Summary")
    print(f"{'='*60}")
    print(f"  prime_basis_size  : {prime_basis.shape[0]}")
    print(f"  final_basis_size  : {results['final_basis_size']}")
    print(f"  converged         : {results['converged']}")
    print(f"  final_gini        : {results['final_gini']:.4f}")
    print(f"  iterations        : {len(results['basis_growth_log'])}")
    print(f"  elapsed           : {elapsed:.1f}s")
    print(f"  saved to          : {out_path}")

    # --- Cleanup ---
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    basis_path = OUTPUT_DIR / "prime_basis_Qwen2.5-0.5B.pt"
    if not basis_path.exists():
        # Try alternate naming from Module 1
        for p in OUTPUT_DIR.glob("prime_basis_*Qwen*0.5B*.pt"):
            basis_path = p
            break
    if basis_path.exists():
        run_module2("Qwen/Qwen2.5-0.5B", basis_path)
    else:
        print("Run Module 1 first: python3 prime_basis.py")
