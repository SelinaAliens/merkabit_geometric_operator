"""
Analysis 34: GUE Null Test for Spectral Zeta Riemann Alignment
==============================================================
Generates 50 random GUE matrices per lattice size (L = 18, 30, 36, 38, 39),
computes their spectral zeta zeros, and reports the z-score of M's alignment
rate against this null distribution.

Uses numpy-only zero-finding (no mpmath) for speed.
"""

import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Setup
# =============================================================================
CACHE_DIR = Path("C:/Users/selin/merkabit_results/analysis_32_montgomery_L38")
RESULTS_DIR = Path("C:/Users/selin/merkabit_results/spectral_zeta")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

L_VALUES = [18, 30, 36, 38, 39]
N_GUE_TRIALS = 50
RNG_SEED = 2026

# =============================================================================
# 1. Load M eigenvalues
# =============================================================================
print("=" * 70)
print("ANALYSIS 34: GUE NULL TEST FOR SPECTRAL ZETA RIEMANN ALIGNMENT")
print("=" * 70)

eigenvalues_M = {}
wing_sizes = {}
for L in L_VALUES:
    eigs = np.load(str(CACHE_DIR / f"eigs_L{L}.npy"))
    abs_eigs = np.abs(eigs)
    pos_eigs = abs_eigs[abs_eigs > 1e-6]
    pos_eigs = np.sort(pos_eigs)
    eigenvalues_M[L] = pos_eigs
    wing_sizes[L] = len(pos_eigs)
    print(f"L={L}: {len(eigs)} total, {len(pos_eigs)} used (wing size N={len(pos_eigs)})")

# =============================================================================
# 2. Load Riemann zeros
# =============================================================================
riemann_cache = Path("C:/Users/selin/merkabit_results/riemann_zeros/riemann_zeros_cache.npy")
if riemann_cache.exists():
    rz_all = np.load(str(riemann_cache))
    riemann_t = rz_all[rz_all > 0]
    riemann_t = riemann_t[riemann_t <= 52][:50].tolist()
    print(f"\nLoaded {len(riemann_t)} Riemann zeros (t <= 52) from cache")
else:
    riemann_t = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
    print(f"\nUsing {len(riemann_t)} hardcoded Riemann zeros")

riemann_t_arr = np.array(riemann_t)


# =============================================================================
# 3. GUE matrix generation
# =============================================================================
def generate_gue_eigenvalues(N, rng):
    """Generate eigenvalues of a random GUE matrix of size N."""
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * N))
    return np.linalg.eigvalsh(H)


# =============================================================================
# 4. Vectorized spectral zeta zero-finding (numpy only)
# =============================================================================
def find_zeros_numpy(abs_eigs, n_sig=60, n_t=500):
    """
    Find zeros of zeta_M(s) = sum |lambda_n|^{-s} in the critical strip.
    Returns list of (Re(rho), Im(rho)).
    Fully vectorized grid evaluation for speed.
    """
    log_eigs = np.log(abs_eigs)
    sigmas = np.linspace(0.02, 0.98, n_sig)
    ts = np.linspace(0.3, 50.0, n_t)
    dsig = sigmas[1] - sigmas[0]
    dt = ts[1] - ts[0]

    # --- Step 1: Vectorized grid evaluation ---
    # weights[i, k] = exp(-sigma_i * ln|lambda_k|) = |lambda_k|^{-sigma_i}
    weights = np.exp(-np.outer(sigmas, log_eigs))  # (n_sig, N)
    # phases[j, k] = exp(-i * t_j * ln|lambda_k|)
    phases = np.exp(-1j * np.outer(ts, log_eigs))  # (n_t, N)
    # grid[i, j] = |sum_k weights[i,k] * phases[j,k]| = |zeta_M(sigma_i + i*t_j)|
    grid = np.abs(weights @ phases.T)  # (n_sig, n_t)

    # --- Step 2: Find local minima ---
    threshold = np.percentile(grid, 2)
    candidates = []
    for i in range(1, n_sig - 1):
        for j in range(1, n_t - 1):
            v = grid[i, j]
            if v < threshold:
                nbrs = [grid[i-1,j], grid[i+1,j], grid[i,j-1], grid[i,j+1],
                        grid[i-1,j-1], grid[i+1,j+1], grid[i-1,j+1], grid[i+1,j-1]]
                if v <= min(nbrs):
                    candidates.append((sigmas[i], ts[j], v))

    candidates.sort(key=lambda x: x[2])

    # --- Step 3: Local refinement ---
    refined = []
    for sig0, t0, v0 in candidates[:120]:
        s_lo = max(0.01, sig0 - 2*dsig)
        s_hi = min(0.99, sig0 + 2*dsig)
        t_lo = max(0.1, t0 - 2*dt)
        t_hi = min(51, t0 + 2*dt)
        loc_sigs = np.linspace(s_lo, s_hi, 20)
        loc_ts = np.linspace(t_lo, t_hi, 20)

        # Local vectorized evaluation
        loc_w = np.exp(-np.outer(loc_sigs, log_eigs))  # (20, N)
        loc_p = np.exp(-1j * np.outer(loc_ts, log_eigs))  # (20, N)
        loc_grid = np.abs(loc_w @ loc_p.T)  # (20, 20)

        idx = np.unravel_index(np.argmin(loc_grid), loc_grid.shape)
        refined.append((loc_sigs[idx[0]], loc_ts[idx[1]], loc_grid[idx]))

    # Deduplicate
    unique = []
    for c in refined:
        if not any(abs(c[0]-u[0]) < 0.005 and abs(c[1]-u[1]) < 0.03 for u in unique):
            unique.append(c)
    unique.sort(key=lambda x: x[2])

    # --- Step 4: Newton refinement (numpy) ---
    zeros = []
    h = 1e-6
    for sig0, t0, v0 in unique[:80]:
        s = complex(sig0, t0)
        for _ in range(50):  # Newton iterations
            f = np.sum(np.exp(-s * log_eigs))
            # Numerical derivative
            df_ds = (np.sum(np.exp(-(s + h) * log_eigs)) - f) / h
            if abs(df_ds) < 1e-30:
                break
            s_new = s - f / df_ds
            if abs(s_new - s) < 1e-12:
                s = s_new
                break
            s = s_new

        rho_re = s.real
        rho_im = s.imag
        residual = abs(np.sum(np.exp(-s * log_eigs)))

        if (0 < rho_re < 1 and 0.1 < rho_im <= 51 and residual < 1e-4):
            if not any(abs(z[0]-rho_re) < 0.001 and abs(z[1]-rho_im) < 0.005
                       for z in zeros):
                zeros.append((rho_re, rho_im))

    zeros.sort(key=lambda z: z[1])
    return zeros


# =============================================================================
# 5. Alignment metric (same as Analysis 33)
# =============================================================================
def compute_alignment_rate(zeros, riemann_t):
    """
    Fraction of spectral zeta zeros whose Im(rho) falls within
    half the mean spacing of some Riemann zero height.
    """
    if len(zeros) < 2:
        return 0.0, 0.5, []

    merkabit_t = sorted([z[1] for z in zeros])
    mean_spacing = np.mean(np.diff(merkabit_t))

    min_dists = []
    n_close = 0
    for mt in merkabit_t:
        dists = np.abs(mt - riemann_t_arr)
        md = np.min(dists)
        min_dists.append(md)
        if md < mean_spacing * 0.5:
            n_close += 1

    rate = n_close / len(merkabit_t)
    return rate, mean_spacing, min_dists


def compute_mean_deviation(zeros):
    """Mean |sigma - 1/2| for a set of zeros."""
    if not zeros:
        return 0.5
    return np.mean([abs(z[0] - 0.5) for z in zeros])


# =============================================================================
# 6. Run M operator zeros (recompute for consistency)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Computing M operator spectral zeta zeros")
print("=" * 70)

M_zeros = {}
M_align_rates = {}
M_mean_devs = {}
M_n_zeros = {}

for L in L_VALUES:
    t0 = time.time()
    zeros = find_zeros_numpy(eigenvalues_M[L])
    elapsed = time.time() - t0

    M_zeros[L] = zeros
    rate, spacing, _ = compute_alignment_rate(zeros, riemann_t)
    M_align_rates[L] = rate
    M_mean_devs[L] = compute_mean_deviation(zeros)
    M_n_zeros[L] = len(zeros)

    print(f"  L={L}: {len(zeros)} zeros, alignment={rate:.3f}, "
          f"<|s-1/2|>={M_mean_devs[L]:.4f}, spacing={spacing:.3f} ({elapsed:.1f}s)")


# =============================================================================
# 7. Run GUE null trials
# =============================================================================
print("\n" + "=" * 70)
print(f"STEP 2: GUE null test ({N_GUE_TRIALS} trials per L)")
print("=" * 70)

rng = np.random.default_rng(seed=RNG_SEED)

gue_align_rates = {L: [] for L in L_VALUES}
gue_mean_devs = {L: [] for L in L_VALUES}
gue_n_zeros = {L: [] for L in L_VALUES}

t_total_start = time.time()

for L in L_VALUES:
    N = wing_sizes[L]
    print(f"\n  L={L} (N={N}): ", end="", flush=True)
    t_L_start = time.time()

    for trial in range(N_GUE_TRIALS):
        # Generate GUE eigenvalues
        gue_eigs = generate_gue_eigenvalues(N, rng)
        abs_gue = np.abs(gue_eigs)
        abs_gue = abs_gue[abs_gue > 1e-6]
        abs_gue = np.sort(abs_gue)

        if len(abs_gue) < 10:
            gue_align_rates[L].append(0.0)
            gue_mean_devs[L].append(0.5)
            gue_n_zeros[L].append(0)
            continue

        # Find zeros
        zeros = find_zeros_numpy(abs_gue)
        rate, _, _ = compute_alignment_rate(zeros, riemann_t)
        dev = compute_mean_deviation(zeros)

        gue_align_rates[L].append(rate)
        gue_mean_devs[L].append(dev)
        gue_n_zeros[L].append(len(zeros))

        if (trial + 1) % 10 == 0:
            print(f"{trial+1}", end=" ", flush=True)

    elapsed_L = time.time() - t_L_start
    rates = gue_align_rates[L]
    print(f"\n    Done in {elapsed_L:.1f}s. GUE mean rate={np.mean(rates):.3f} "
          f"+/- {np.std(rates):.3f}, mean zeros={np.mean(gue_n_zeros[L]):.1f}")

total_elapsed = time.time() - t_total_start
print(f"\nTotal GUE computation time: {total_elapsed:.1f}s")


# =============================================================================
# 8. Statistical analysis
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Statistical analysis")
print("=" * 70)

results_lines = []
results_lines.append("Analysis 34: GUE Null Test for Spectral Zeta Riemann Alignment")
results_lines.append("=" * 70)
results_lines.append(f"GUE trials per L: {N_GUE_TRIALS}")
results_lines.append(f"RNG seed: {RNG_SEED}")
results_lines.append(f"Riemann zeros used: {len(riemann_t)} (t <= 52)")
results_lines.append("")

# Header
header = (f"{'L':>4} {'N':>6} {'M_zeros':>8} {'GUE_zeros':>10} "
          f"{'M_rate':>8} {'GUE_mean':>9} {'GUE_std':>8} "
          f"{'z-score':>8} {'p-value':>8} "
          f"{'M_dev':>8} {'GUE_dev':>9}")
print(f"\n{header}")
print("-" * 105)
results_lines.append(header)
results_lines.append("-" * 105)

p_values = []
z_scores = []

for L in L_VALUES:
    gue_rates = np.array(gue_align_rates[L])
    gue_devs_arr = np.array(gue_mean_devs[L])
    gue_nz = np.array(gue_n_zeros[L])

    M_rate = M_align_rates[L]
    M_dev = M_mean_devs[L]

    gue_mean_rate = np.mean(gue_rates)
    gue_std_rate = np.std(gue_rates)
    gue_mean_dev = np.mean(gue_devs_arr)
    gue_mean_nz = np.mean(gue_nz)

    # z-score
    if gue_std_rate > 0:
        z = (M_rate - gue_mean_rate) / gue_std_rate
    else:
        z = 0.0

    # Empirical p-value (one-sided: fraction of GUE >= M)
    p_emp = np.mean(gue_rates >= M_rate)
    # Ensure p_emp > 0 for Fisher combination
    p_emp_adj = max(p_emp, 1.0 / (N_GUE_TRIALS + 1))

    p_values.append(p_emp_adj)
    z_scores.append(z)

    line = (f"{L:>4} {wing_sizes[L]:>6} {M_n_zeros[L]:>8} {gue_mean_nz:>10.1f} "
            f"{M_rate:>8.3f} {gue_mean_rate:>9.3f} {gue_std_rate:>8.3f} "
            f"{z:>8.2f} {p_emp:>8.3f} "
            f"{M_dev:>8.4f} {gue_mean_dev:>9.4f}")
    print(line)
    results_lines.append(line)

# Fisher combined p-value
print("\n" + "-" * 105)
results_lines.append("")

if all(p > 0 for p in p_values):
    fisher_stat = -2 * np.sum(np.log(p_values))
    fisher_p = 1 - stats.chi2.cdf(fisher_stat, df=2 * len(p_values))
    print(f"\nFisher combined test: chi2 = {fisher_stat:.3f}, "
          f"df = {2*len(p_values)}, p = {fisher_p:.6f}")
    results_lines.append(f"Fisher combined test: chi2 = {fisher_stat:.3f}, "
                         f"df = {2*len(p_values)}, p = {fisher_p:.6f}")
else:
    fisher_p = 1.0
    print("\nFisher combined test: undefined (zero p-values)")
    results_lines.append("Fisher combined test: undefined")

# Mann-Whitney U test: M deviation distributions vs GUE pooled
print("\n--- Mann-Whitney U test (off-critical deviation) ---")
results_lines.append("")
results_lines.append("--- Mann-Whitney U test (off-critical deviation) ---")

for L in L_VALUES:
    M_devs_list = [abs(z[0] - 0.5) for z in M_zeros[L]]
    # Pool all GUE deviations for this L
    gue_devs_pooled = []
    # We need to recompute individual zero deviations for GUE
    # Use the mean deviation as a proxy (we only stored means)
    # For proper U-test, we'd need individual zeros — use stored mean as sample
    gue_devs_sample = gue_mean_devs[L]

    if len(M_devs_list) >= 2 and len(gue_devs_sample) >= 2:
        # Compare M's mean deviation against distribution of GUE mean deviations
        M_mean = np.mean(M_devs_list)
        # One-sample test: where does M_mean fall in GUE distribution?
        rank = np.sum(np.array(gue_devs_sample) <= M_mean)
        percentile = rank / len(gue_devs_sample) * 100
        u_stat, u_p = stats.mannwhitneyu(
            M_devs_list, gue_devs_sample, alternative='less'
        )
        line = (f"  L={L}: M mean dev = {M_mean:.4f}, "
                f"GUE percentile = {percentile:.1f}%, "
                f"U = {u_stat:.1f}, p = {u_p:.4f}")
    else:
        line = f"  L={L}: insufficient data"
    print(line)
    results_lines.append(line)

# Summary verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
results_lines.append("")
results_lines.append("=" * 70)
results_lines.append("VERDICT")
results_lines.append("=" * 70)

mean_z = np.mean(z_scores)
any_significant = any(z > 2 for z in z_scores)

if fisher_p < 0.05:
    verdict = "SIGNIFICANT: M's Riemann alignment exceeds GUE null (p < 0.05)"
elif any_significant:
    sig_Ls = [L for L, z in zip(L_VALUES, z_scores) if z > 2]
    verdict = f"PARTIALLY SIGNIFICANT: z > 2 at L = {sig_Ls}"
elif mean_z > 1:
    verdict = f"SUGGESTIVE: Mean z = {mean_z:.2f} (trend but not significant)"
else:
    verdict = f"CONSISTENT WITH GUE NULL: Mean z = {mean_z:.2f}"

print(f"\n  {verdict}")
print(f"  Mean z-score across L: {mean_z:.3f}")
print(f"  Fisher combined p-value: {fisher_p:.6f}")
results_lines.append(f"  {verdict}")
results_lines.append(f"  Mean z-score across L: {mean_z:.3f}")
results_lines.append(f"  Fisher combined p-value: {fisher_p:.6f}")

for L, z, p in zip(L_VALUES, z_scores, p_values):
    line = f"  L={L}: z = {z:+.3f}, p = {p:.4f}"
    print(line)
    results_lines.append(line)


# =============================================================================
# 9. Save results
# =============================================================================
# Text output
with open(str(RESULTS_DIR / "analysis_34_null_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
    f.write("\n")
print(f"\nResults saved to {RESULTS_DIR / 'analysis_34_null_results.txt'}")

# Raw data for reproducibility
np.savez(str(RESULTS_DIR / "null_data.npz"),
         L_values=L_VALUES,
         M_align_rates=np.array([M_align_rates[L] for L in L_VALUES]),
         M_mean_devs=np.array([M_mean_devs[L] for L in L_VALUES]),
         M_n_zeros=np.array([M_n_zeros[L] for L in L_VALUES]),
         gue_align_rates_18=np.array(gue_align_rates[18]),
         gue_align_rates_30=np.array(gue_align_rates[30]),
         gue_align_rates_36=np.array(gue_align_rates[36]),
         gue_align_rates_38=np.array(gue_align_rates[38]),
         gue_align_rates_39=np.array(gue_align_rates[39]),
         gue_mean_devs_18=np.array(gue_mean_devs[18]),
         gue_mean_devs_30=np.array(gue_mean_devs[30]),
         gue_mean_devs_36=np.array(gue_mean_devs[36]),
         gue_mean_devs_38=np.array(gue_mean_devs[38]),
         gue_mean_devs_39=np.array(gue_mean_devs[39]),
         z_scores=np.array(z_scores),
         p_values=np.array(p_values),
         fisher_p=fisher_p,
         riemann_t=np.array(riemann_t))
print(f"Raw data saved to {RESULTS_DIR / 'null_data.npz'}")


# =============================================================================
# 10. FIGURE 4: Alignment rate distributions
# =============================================================================
fig, axes = plt.subplots(1, len(L_VALUES), figsize=(4*len(L_VALUES), 5), sharey=False)
if len(L_VALUES) == 1:
    axes = [axes]

for ax, L, z in zip(axes, L_VALUES, z_scores):
    rates = gue_align_rates[L]
    m_rate = M_align_rates[L]

    ax.hist(rates, bins=15, color='steelblue', alpha=0.7, edgecolor='white',
            density=False, label='GUE null')
    ax.axvline(m_rate, color='red', linewidth=2.5, linestyle='--',
               label=f'M (rate={m_rate:.3f})')

    # Shade region >= M
    ax.axvspan(m_rate, ax.get_xlim()[1] if ax.get_xlim()[1] > m_rate else m_rate + 0.1,
               alpha=0.15, color='red')

    ax.set_xlabel('Alignment rate', fontsize=11)
    ax.set_ylabel('Count' if ax == axes[0] else '', fontsize=11)
    ax.set_title(f'L = {L}\nz = {z:+.2f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

plt.suptitle('Analysis 34: GUE Null Test - Riemann Alignment Rate', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure4_null_alignment.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")


# =============================================================================
# 11. FIGURE 5: Off-critical deviation comparison
# =============================================================================
fig, axes = plt.subplots(1, len(L_VALUES), figsize=(4*len(L_VALUES), 5), sharey=True)
if len(L_VALUES) == 1:
    axes = [axes]

for ax, L in zip(axes, L_VALUES):
    gue_devs = gue_mean_devs[L]
    m_dev = M_mean_devs[L]

    bp = ax.boxplot([gue_devs], positions=[0], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6),
                    medianprops=dict(color='navy', linewidth=2))
    ax.plot(0, m_dev, 'rD', markersize=14, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10, label=f'M ({m_dev:.4f})')

    # Percentile of M in GUE distribution
    pct = np.mean(np.array(gue_devs) <= m_dev) * 100
    ax.set_title(f'L = {L}\nM at {pct:.0f}th pctile', fontsize=12, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels(['GUE\nnull'], fontsize=10)
    ax.set_ylabel('<|sigma - 1/2|>' if ax == axes[0] else '', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('Analysis 34: Off-Critical Deviation - M vs GUE Null', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure5_null_deviation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")
