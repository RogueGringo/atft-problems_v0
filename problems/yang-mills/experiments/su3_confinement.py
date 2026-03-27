#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""SU(3) Yang-Mills confinement-deconfinement experiment.

Extends the SU(2) lattice gauge experiment to SU(3) using Metropolis-Hastings
with random-near-identity proposals.  Applies the parity-complete feature map
and H₀ persistence to track the confinement transition across β.

Lattice : 8³×4
β sweep : [5.0, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 6.0, 6.5, 7.0]
Expected β_c (SU(3), Nt=4) ≈ 5.69

Protocol
--------
For each β:
  1. Thermalize 100 sweeps (Metropolis-Hastings).
  2. Collect 5 configs (1 sweep apart to save time).
  3. Compute parity-complete feature map φ(x) = (s_μν, q_μν).
  4. Run H₀ persistence on 500-point subsample.
  5. Record ε*(β) and Gini(β).

SU(3) specifics
---------------
- 3×3 complex unitary matrices, det=1.
- Random SU(3) near identity via expm(i ε H) with H traceless Hermitian.
- Plaquette: (1/3) Re Tr P_μν.
- Feature map: s_μν = 1 - (1/3) Re Tr P_μν, q_μν = (1/3) Im Tr P_μν.
- Metropolis acceptance: exp(-β ΔS), ΔS = -ΔRe Tr(U × Staple†) / 3.

Outputs
-------
  results/su3_confinement.json
  results/figures/su3_transition.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.spatial.distance import pdist

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.parent  # problems/yang-mills/
RESULTS_DIR = SCRIPT_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot styling ──────────────────────────────────────────────────────────────

COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "bg":    "#0f0d08",
    "text":  "#d6d0be",
    "muted": "#817a66",
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.facecolor":     COLORS["bg"],
    "figure.facecolor":   COLORS["bg"],
    "text.color":         COLORS["text"],
    "axes.labelcolor":    COLORS["text"],
    "xtick.color":        COLORS["muted"],
    "ytick.color":        COLORS["muted"],
    "axes.edgecolor":     COLORS["muted"],
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.dpi":        200,
})

# ── Experiment parameters ─────────────────────────────────────────────────────

LATTICE      = (8, 8, 8, 4)
BETA_VALUES  = [5.0, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 6.0, 6.5, 7.0]
N_THERM      = 100   # thermalization sweeps
N_CONFIGS    = 5     # configs to collect per β
N_SKIP       = 1     # sweeps between saved configs
EPSILON_MH   = 0.20  # step size for Metropolis proposals
N_ACCEPT_LOG = 5000  # log acceptance rate every N link updates

# ── SU(3) primitives ──────────────────────────────────────────────────────────

def project_su3(M: np.ndarray) -> np.ndarray:
    """Project a 3×3 complex matrix to SU(3) via QR + det fix."""
    Q, R = np.linalg.qr(M)
    # Fix phases so R has positive diagonal (standard QR convention)
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phases[np.newaxis, :]
    # Fix determinant to +1
    d = np.linalg.det(Q)
    Q[:, 0] /= d  # divide first column by det (|det|=1 by construction)
    return Q


def random_su3(rng: np.random.Generator) -> np.ndarray:
    """Haar-random SU(3) matrix."""
    M = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return project_su3(M)


def random_su3_near_identity(rng: np.random.Generator, epsilon: float = EPSILON_MH) -> np.ndarray:
    """Random SU(3) close to identity via R = exp(i ε H), H traceless Hermitian."""
    H = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H = (H + H.conj().T) / 2          # Hermitian
    H -= np.trace(H) / 3 * np.eye(3)  # traceless
    R = expm(1j * epsilon * H)
    # Numerically project back to SU(3) for safety
    return project_su3(R)


# ── Lattice helpers ───────────────────────────────────────────────────────────

def shift(site: list[int], mu: int, lattice_shape: tuple[int, ...], step: int = 1) -> tuple[int, ...]:
    """Return site shifted by ±1 in direction mu with periodic BC."""
    s = list(site)
    s[mu] = (s[mu] + step) % lattice_shape[mu]
    return tuple(s)


def staple_sum(links: dict[int, np.ndarray], site: tuple[int, ...],
               mu: int, lattice_shape: tuple[int, ...]) -> np.ndarray:
    """Compute the staple sum A_μ(x) for SU(3).

    A_μ(x) = Σ_{ν≠μ} [U_ν(x+μ) U_μ†(x+ν) U_ν†(x)          (upper)
                       + U_ν†(x+μ-ν) U_μ†(x-ν) U_ν(x-ν)]   (lower)
    """
    ndim = len(lattice_shape)
    A = np.zeros((3, 3), dtype=np.complex128)

    for nu in range(ndim):
        if nu == mu:
            continue

        x_mu   = shift(list(site), mu, lattice_shape, +1)
        x_nu   = shift(list(site), nu, lattice_shape, +1)
        x_mnu  = shift(list(site), nu, lattice_shape, -1)

        # upper staple: U_ν(x+μ) @ U_μ†(x+ν) @ U_ν†(x)
        A += (links[nu][x_mu] @
              links[mu][x_nu].conj().T @
              links[nu][tuple(site)].conj().T)

        # lower staple: U_ν†(x+μ-ν) @ U_μ†(x-ν) @ U_ν(x-ν)
        x_mu_mnu = shift(list(x_mu), nu, lattice_shape, -1)
        A += (links[nu][x_mu_mnu].conj().T @
              links[mu][x_mnu].conj().T @
              links[nu][x_mnu])

    return A


def plaquette_su3(links: dict[int, np.ndarray], site: tuple[int, ...],
                  mu: int, nu: int, lattice_shape: tuple[int, ...]) -> np.ndarray:
    """Return plaquette matrix P_μν(x) = U_μ U_ν U_μ† U_ν†."""
    x_mu = shift(list(site), mu, lattice_shape, +1)
    x_nu = shift(list(site), nu, lattice_shape, +1)
    return (links[mu][tuple(site)] @
            links[nu][x_mu] @
            links[mu][x_nu].conj().T @
            links[nu][tuple(site)].conj().T)


def average_plaquette_su3(links: dict[int, np.ndarray],
                          lattice_shape: tuple[int, ...]) -> float:
    """Average (1/3) Re Tr P_μν over all sites and μ<ν pairs."""
    ndim  = len(lattice_shape)
    total = 0.0
    count = 0
    for idx in np.ndindex(*lattice_shape):
        for mu in range(ndim):
            for nu in range(mu + 1, ndim):
                P     = plaquette_su3(links, idx, mu, nu, lattice_shape)
                total += np.real(np.trace(P)) / 3.0
                count += 1
    return total / count if count > 0 else 0.0


# ── Metropolis-Hastings update ────────────────────────────────────────────────

def metropolis_su3_sweep(links: dict[int, np.ndarray], beta: float,
                          lattice_shape: tuple[int, ...],
                          rng: np.random.Generator,
                          epsilon: float = EPSILON_MH) -> float:
    """One full Metropolis-Hastings sweep.  Returns acceptance rate."""
    ndim       = len(lattice_shape)
    n_accepted = 0
    n_total    = 0

    for mu in range(ndim):
        for site in np.ndindex(*lattice_shape):
            U_old = links[mu][site]
            A     = staple_sum(links, site, mu, lattice_shape)

            # Propose U' = R × U_old, R near identity
            R     = random_su3_near_identity(rng, epsilon)
            U_new = R @ U_old

            # ΔS = -(1/3) Re Tr[(U_new - U_old) A]
            # S = -(β/3) Re Tr[U @ Σ] where Σ = staple sum (no conjugate)
            delta_s = -np.real(np.trace((U_new - U_old) @ A)) / 3.0

            # Accept with min(1, exp(-β ΔS))
            if delta_s <= 0 or rng.random() < np.exp(-beta * delta_s):
                links[mu][site] = U_new
                n_accepted += 1
            n_total += 1

    return n_accepted / n_total if n_total > 0 else 0.0


# ── Config generation ─────────────────────────────────────────────────────────

def generate_configs_su3(beta: float, lattice_shape: tuple[int, ...],
                          n_therm: int = N_THERM, n_configs: int = N_CONFIGS,
                          n_skip: int = N_SKIP, seed: int = 42,
                          epsilon: float = EPSILON_MH) -> list[dict]:
    """Generate SU(3) lattice configs via Metropolis-Hastings."""
    rng  = np.random.default_rng(seed)
    ndim = len(lattice_shape)

    # Cold start: all links = identity
    links: dict[int, np.ndarray] = {}
    for mu in range(ndim):
        links[mu] = np.tile(np.eye(3, dtype=np.complex128), (*lattice_shape, 1, 1))

    # Thermalization
    print(f"    Thermalizing {n_therm} sweeps ...", flush=True)
    for sweep in range(n_therm):
        acc = metropolis_su3_sweep(links, beta, lattice_shape, rng, epsilon)
        if (sweep + 1) % 20 == 0:
            plaq = average_plaquette_su3(links, lattice_shape)
            print(f"      therm {sweep+1}/{n_therm}: <P>={plaq:.4f}, acc={acc:.3f}",
                  flush=True)

    # Collect configs
    configs = []
    print(f"    Collecting {n_configs} configs ...", flush=True)
    for c_idx in range(n_configs):
        for _ in range(n_skip):
            metropolis_su3_sweep(links, beta, lattice_shape, rng, epsilon)
        config = {mu: links[mu].copy() for mu in range(ndim)}
        plaq   = average_plaquette_su3(links, lattice_shape)
        configs.append(config)
        print(f"      config {c_idx+1}/{n_configs}: <P>={plaq:.4f}", flush=True)

    return configs


# ── Parity-complete feature map (SU(3)) ──────────────────────────────────────

def parity_complete_feature_map_su3(config: dict, lattice_shape: tuple[int, ...]) -> np.ndarray:
    """φ(x) = (s_μν, q_μν) ∈ R¹² per site.

    s_μν = 1 - (1/3) Re Tr P_μν   (action density, parity-even)
    q_μν = (1/3) Im Tr P_μν        (topological charge density, parity-odd)

    For 4D: 6 μ<ν pairs → φ ∈ R¹².
    """
    ndim    = len(lattice_shape)
    n_pairs = ndim * (ndim - 1) // 2
    vol     = int(np.prod(lattice_shape))

    features = np.zeros((vol, 2 * n_pairs))
    flat_idx = 0

    for site in np.ndindex(*lattice_shape):
        pair_idx = 0
        for mu in range(ndim):
            for nu in range(mu + 1, ndim):
                P    = plaquette_su3(config, site, mu, nu, lattice_shape)
                tr_P = np.trace(P)
                s    = 1.0 - np.real(tr_P) / 3.0
                q    = np.imag(tr_P) / 3.0
                features[flat_idx, pair_idx]           = s
                features[flat_idx, n_pairs + pair_idx] = q
                pair_idx += 1
        flat_idx += 1

    return features


# ── TDA: H₀ persistence + observables ────────────────────────────────────────

def h0_persistence_subsample(points: np.ndarray, n_sample: int = 500,
                              seed: int = 42) -> np.ndarray:
    """H₀ Vietoris-Rips persistence via Kruskal MST on subsampled cloud."""
    rng = np.random.default_rng(seed)
    n   = len(points)
    if n > n_sample:
        idx    = rng.choice(n, n_sample, replace=False)
        points = points[idx]
        n      = n_sample

    dists = pdist(points)

    parent   = list(range(n))
    rank_uf  = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x         = parent[x]
        return x

    edges = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[k], i, j))
            k += 1
    edges.sort()

    bars: list[float] = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(float(dist))

    return np.array(bars)


def onset_scale(bars: np.ndarray, percentile: float = 95) -> float:
    """ε* = 95th percentile of H₀ persistence bar lengths."""
    if len(bars) == 0:
        return 0.0
    return float(np.percentile(bars, percentile))


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index    = np.arange(1, n + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 72)
    print("  SU(3) Yang-Mills: confinement-deconfinement experiment")
    print(f"  Lattice: {LATTICE}   β sweep: {BETA_VALUES}")
    print(f"  {timestamp}")
    print("=" * 72, flush=True)

    results_per_beta: dict[str, dict] = {}

    for beta in BETA_VALUES:
        print(f"\n{'='*60}")
        print(f"  β = {beta:.2f}")
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        configs = generate_configs_su3(
            beta        = beta,
            lattice_shape = LATTICE,
            n_therm     = N_THERM,
            n_configs   = N_CONFIGS,
            n_skip      = N_SKIP,
            seed        = int(beta * 1000),
            epsilon     = EPSILON_MH,
        )
        elapsed_gen = time.time() - t0
        print(f"  Generated {len(configs)} configs in {elapsed_gen:.0f}s", flush=True)

        onset_scales: list[float] = []
        gini_values:  list[float] = []
        plaquettes:   list[float] = []

        for c_idx, config in enumerate(configs):
            plaq = average_plaquette_su3(config, LATTICE)
            plaquettes.append(plaq)

            features = parity_complete_feature_map_su3(config, LATTICE)
            bars     = h0_persistence_subsample(features, n_sample=500,
                                                seed=c_idx + int(beta * 100))

            eps_star = onset_scale(bars)
            g        = gini(bars)
            onset_scales.append(eps_star)
            gini_values.append(g)

            print(f"    config {c_idx+1}: <P>={plaq:.4f}, "
                  f"ε*={eps_star:.4f}, G={g:.4f}", flush=True)

        results_per_beta[str(beta)] = {
            "beta":           beta,
            "n_configs":      len(configs),
            "plaquettes":     plaquettes,
            "mean_plaquette": float(np.mean(plaquettes)) if plaquettes else 0.0,
            "onset_scales":   onset_scales,
            "mean_onset":     float(np.mean(onset_scales)) if onset_scales else 0.0,
            "std_onset":      float(np.std(onset_scales)) if len(onset_scales) > 1 else 0.0,
            "gini_values":    gini_values,
            "mean_gini":      float(np.mean(gini_values)) if gini_values else 0.0,
            "gen_time_s":     elapsed_gen,
        }

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  ANALYSIS")
    print(f"{'='*72}")

    betas_sorted = sorted(results_per_beta.keys(), key=float)
    betas      = [results_per_beta[k]["beta"]           for k in betas_sorted]
    mean_onsets = [results_per_beta[k]["mean_onset"]    for k in betas_sorted]
    mean_ginis  = [results_per_beta[k]["mean_gini"]     for k in betas_sorted]
    mean_plaqs  = [results_per_beta[k]["mean_plaquette"] for k in betas_sorted]

    for k in betas_sorted:
        r = results_per_beta[k]
        print(f"  β={r['beta']:.2f}: <P>={r['mean_plaquette']:.4f}, "
              f"ε*={r['mean_onset']:.4f}±{r['std_onset']:.4f}, "
              f"G={r['mean_gini']:.4f}")

    betas_arr  = np.array(betas)
    onsets_arr = np.array(mean_onsets)

    detected_transition = False
    transition_beta     = 0.0
    max_deriv           = 0.0

    if len(betas_arr) >= 3:
        d_onset          = np.gradient(onsets_arr, betas_arr)
        max_deriv_idx    = int(np.argmax(np.abs(d_onset)))
        transition_beta  = float(betas_arr[max_deriv_idx])
        max_deriv        = float(d_onset[max_deriv_idx])
        detected_transition = 5.4 <= transition_beta <= 6.0
        print(f"\n  Maximum |dε*/dβ| at β = {transition_beta:.2f}  "
              f"(derivative = {max_deriv:.4f})")
        print(f"  Transition in [5.4, 6.0]: {detected_transition}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(betas, mean_plaqs, "o-", color=COLORS["gold"], linewidth=2, markersize=8)
    ax.axvline(5.69, color=COLORS["red"], linestyle="--", alpha=0.6, label=r"$\beta_c \approx 5.69$")
    ax.set_xlabel(r"$\beta$ (coupling)")
    ax.set_ylabel(r"$\langle P \rangle$")
    ax.set_title("Order Parameter", color=COLORS["gold"])
    ax.legend(labelcolor=COLORS["text"])
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.plot(betas, mean_onsets, "o-", color=COLORS["teal"], linewidth=2, markersize=8)
    ax.axvline(5.69, color=COLORS["red"], linestyle="--", alpha=0.6, label=r"$\beta_c \approx 5.69$")
    if detected_transition:
        ax.axvline(transition_beta, color=COLORS["gold"], linestyle=":",
                   alpha=0.7, label=f"detected β={transition_beta:.2f}")
    ax.set_xlabel(r"$\beta$ (coupling)")
    ax.set_ylabel(r"$\varepsilon^*(\beta)$")
    ax.set_title("Topological Onset Scale", color=COLORS["teal"])
    ax.legend(labelcolor=COLORS["text"])
    ax.grid(True, alpha=0.15)

    ax = axes[2]
    ax.plot(betas, mean_ginis, "o-", color=COLORS["red"], linewidth=2, markersize=8)
    ax.axvline(5.69, color=COLORS["red"], linestyle="--", alpha=0.6, label=r"$\beta_c \approx 5.69$")
    ax.set_xlabel(r"$\beta$ (coupling)")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Eigenvalue Hierarchy", color=COLORS["red"])
    ax.legend(labelcolor=COLORS["text"])
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"SU(3) Yang-Mills confinement-deconfinement  "
        f"({LATTICE[0]}³×{LATTICE[3]}, Metropolis-Hastings)",
        color=COLORS["gold"], fontsize=14, y=1.02,
    )
    fig_path = FIGS_DIR / "su3_transition.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = "PASS" if detected_transition else "PARTIAL"
    print(f"\n{'='*72}")
    print(f"  VERDICT: {verdict}")
    if detected_transition:
        print(f"  Transition detected at β = {transition_beta:.2f}  (expected ≈5.69)")
    else:
        print("  No clean transition detected in [5.4, 6.0] — may need more sweeps.")
    print(f"{'='*72}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment":        "SU(3) Yang-Mills confinement-deconfinement",
        "timestamp":         timestamp,
        "lattice":           list(LATTICE),
        "beta_values":       BETA_VALUES,
        "n_therm":           N_THERM,
        "n_configs":         N_CONFIGS,
        "epsilon_mh":        EPSILON_MH,
        "results_per_beta":  results_per_beta,
        "transition_beta":   float(transition_beta) if detected_transition else None,
        "max_derivative":    max_deriv,
        "detected_in_range": detected_transition,
        "verdict":           verdict,
        "summary": (
            f"SU(3) on {LATTICE[0]}³×{LATTICE[3]} lattice, Metropolis-Hastings. "
            f"Transition {'detected' if detected_transition else 'NOT detected'} "
            f"at β={transition_beta:.2f} (expected ≈5.69).  Verdict: {verdict}."
        ),
    }

    out_path = RESULTS_DIR / "su3_confinement.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
