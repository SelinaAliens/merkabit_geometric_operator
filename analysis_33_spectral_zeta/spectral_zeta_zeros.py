"""
Analysis 33: Spectral Zeta Function ζ_M(s) = Σ |λₙ|⁻ˢ — Zero-Finding
=========================================================================
Using cached eigenvalues from Analysis 32 (L = 18, 30, 36, 38, 39).

Constructs the finite spectral zeta function at 50-digit precision,
finds zeros in the critical strip 0 < Re(s) < 1, Im(s) ∈ [0, 50],
and compares with Riemann zeta zeros.
"""

import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import mpmath
from mpmath import mpf, mpc, log, exp, fsum, findroot, re, im, fabs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

mpmath.mp.dps = 50  # 50-digit precision

# =============================================================================
# Setup
# =============================================================================
CACHE_DIR = Path("C:/Users/selin/merkabit_results/analysis_32_montgomery_L38")
RESULTS_DIR = Path("C:/Users/selin/merkabit_results/spectral_zeta")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

L_VALUES = [18, 30, 36, 38, 39]

# Load eigenvalues — use |λₙ| for all non-zero eigenvalues
eigenvalues = {}
for L in L_VALUES:
    eigs = np.load(str(CACHE_DIR / f"eigs_L{L}.npy"))
    abs_eigs = np.abs(eigs)
    # Exclude near-zero eigenvalues (would dominate and diverge)
    pos_eigs = abs_eigs[abs_eigs > 1e-6]
    pos_eigs = np.sort(pos_eigs)
    eigenvalues[L] = pos_eigs
    print(f"L={L}: {len(eigs)} total, {len(pos_eigs)} used, "
          f"range [{pos_eigs.min():.6f}, {pos_eigs.max():.6f}], "
          f"ln-range [{np.log(pos_eigs.min()):.3f}, {np.log(pos_eigs.max()):.3f}]")


# =============================================================================
# 1. Fast numpy evaluation for grid search
# =============================================================================
def zeta_numpy_grid(sigmas, ts, log_eigs):
    """Evaluate |ζ_M(σ+it)| on 2D grid using numpy vectorization."""
    n_sig, n_t = len(sigmas), len(ts)
    grid = np.zeros((n_sig, n_t))
    # Precompute exp(-σ * ln|λ|) for each σ
    for i, sig in enumerate(sigmas):
        weights = np.exp(-sig * log_eigs)  # shape (N,)
        for j, t in enumerate(ts):
            # ζ_M(σ+it) = Σ |λ|^(-σ) exp(-it ln|λ|)
            phases = np.exp(-1j * t * log_eigs)
            val = np.sum(weights * phases)
            grid[i, j] = abs(val)
    return grid


def zeta_numpy_point(s, log_eigs):
    """Evaluate ζ_M(s) at a single complex point."""
    return np.sum(np.exp(-s * log_eigs))


# =============================================================================
# 2. mpmath high-precision evaluation
# =============================================================================
def zeta_mp(s, log_eigs_mp):
    """ζ_M(s) at 50-digit precision using precomputed ln|λ|."""
    s = mpc(s) if not isinstance(s, mpc) else s
    return fsum(exp(-s * le) for le in log_eigs_mp)


# =============================================================================
# 3. Zero-finding pipeline
# =============================================================================
def find_zeros_for_L(L, eig_array):
    """Complete zero-finding: grid → refine → mpmath confirm."""
    print(f"\n{'='*60}")
    print(f"L = {L}: {len(eig_array)} eigenvalues")
    print(f"{'='*60}")
    t_start = time.time()

    log_eigs = np.log(eig_array)

    # --- Step 1: Coarse grid search (numpy) ---
    n_sig, n_t = 60, 500
    sigmas = np.linspace(0.02, 0.98, n_sig)
    ts = np.linspace(0.3, 50.0, n_t)
    dt = ts[1] - ts[0]
    dsig = sigmas[1] - sigmas[0]

    print(f"  Coarse grid ({n_sig}×{n_t}, Δσ={dsig:.4f}, Δt={dt:.4f})...")
    grid = zeta_numpy_grid(sigmas, ts, log_eigs)

    # --- Step 2: Find local minima ---
    # Adaptive threshold: bottom 2% of values
    threshold = np.percentile(grid, 2)
    candidates = []
    for i in range(1, n_sig - 1):
        for j in range(1, n_t - 1):
            v = grid[i, j]
            if v < threshold:
                # 8-neighbor local minimum check
                nbrs = [grid[i-1,j], grid[i+1,j], grid[i,j-1], grid[i,j+1],
                         grid[i-1,j-1], grid[i+1,j+1], grid[i-1,j+1], grid[i+1,j-1]]
                if v <= min(nbrs):
                    candidates.append((sigmas[i], ts[j], v))

    candidates.sort(key=lambda x: x[2])
    print(f"  {len(candidates)} coarse minima (threshold={threshold:.4f})")

    # --- Step 3: Local numpy refinement ---
    refined = []
    for sig0, t0, v0 in candidates[:120]:
        # Fine 20×20 grid around candidate
        s_lo = max(0.01, sig0 - 2*dsig)
        s_hi = min(0.99, sig0 + 2*dsig)
        t_lo = max(0.1, t0 - 2*dt)
        t_hi = min(51, t0 + 2*dt)
        loc_sigs = np.linspace(s_lo, s_hi, 20)
        loc_ts = np.linspace(t_lo, t_hi, 20)
        loc_grid = zeta_numpy_grid(loc_sigs, loc_ts, log_eigs)
        idx = np.unravel_index(np.argmin(loc_grid), loc_grid.shape)
        refined.append((loc_sigs[idx[0]], loc_ts[idx[1]], loc_grid[idx]))

    # Deduplicate
    unique = []
    for c in refined:
        if not any(abs(c[0]-u[0]) < 0.005 and abs(c[1]-u[1]) < 0.03 for u in unique):
            unique.append(c)
    unique.sort(key=lambda x: x[2])
    print(f"  {len(unique)} unique candidates after local refinement")

    # --- Step 4: mpmath precision refinement ---
    log_eigs_mp = [log(mpf(str(float(e)))) for e in eig_array]

    zeros = []
    n_tried = 0
    for sig0, t0, v0 in unique[:80]:
        n_tried += 1
        try:
            result = findroot(
                lambda s: zeta_mp(s, log_eigs_mp),
                mpc(sig0, t0),
                tol=mpf('1e-25'),
                maxsteps=300
            )
            rho_re = float(re(result))
            rho_im = float(im(result))
            residual = float(fabs(zeta_mp(result, log_eigs_mp)))

            if (0 < rho_re < 1 and 0.1 < rho_im <= 51 and residual < 1e-6):
                if not any(abs(z[0]-rho_re) < 0.001 and abs(z[1]-rho_im) < 0.005
                           for z in zeros):
                    zeros.append((rho_re, rho_im, residual))
        except Exception:
            continue

    zeros.sort(key=lambda z: z[1])
    elapsed = time.time() - t_start

    print(f"  {len(zeros)} confirmed zeros ({n_tried} tried, {elapsed:.1f}s)")
    for rho_re, rho_im, res in zeros[:20]:
        dev = abs(rho_re - 0.5)
        print(f"    ρ = {rho_re:.10f} + {rho_im:.10f}i  |σ-½|={dev:.8f}  |ζ|={res:.2e}")
    if len(zeros) > 20:
        print(f"    ... ({len(zeros)-20} more)")

    return zeros, grid, sigmas, ts


# =============================================================================
# 4. Run for all L values
# =============================================================================
all_zeros = {}
all_grids = {}
for L in L_VALUES:
    zeros, grid, sigmas, ts = find_zeros_for_L(L, eigenvalues[L])
    all_zeros[L] = zeros
    all_grids[L] = (grid, sigmas, ts)

# Save zeros
with open(str(RESULTS_DIR / "zeros_data.txt"), "w", encoding="utf-8") as f:
    f.write("Spectral Zeta Function zeta_M(s) = Sum |lambda_n|^(-s)  --  Zeros in Critical Strip\n")
    f.write("=" * 70 + "\n\n")
    for L in L_VALUES:
        zeros = all_zeros[L]
        f.write(f"L = {L}  ({len(eigenvalues[L])} eigenvalues, {len(zeros)} zeros)\n")
        f.write(f"{'#':>4} {'Re(ρ)':>14} {'Im(ρ)':>14} {'|σ-½|':>12} {'|ζ_M(ρ)|':>12}\n")
        f.write("-" * 60 + "\n")
        for k, (rho_re, rho_im, res) in enumerate(zeros):
            f.write(f"{k+1:>4} {rho_re:>14.10f} {rho_im:>14.10f} "
                    f"{abs(rho_re-0.5):>12.8f} {res:>12.2e}\n")
        if zeros:
            devs = [abs(z[0] - 0.5) for z in zeros]
            f.write(f"\n  Mean |σ-½| = {np.mean(devs):.8f}\n")
            f.write(f"  Min  |σ-½| = {np.min(devs):.8f}\n")
            f.write(f"  Std  |σ-½| = {np.std(devs):.8f}\n")
        f.write("\n")

# =============================================================================
# 5. Load Riemann zeros for comparison
# =============================================================================
riemann_cache = Path("C:/Users/selin/merkabit_results/riemann_zeros/riemann_zeros_cache.npy")
if riemann_cache.exists():
    rz_all = np.load(str(riemann_cache))
    riemann_t = rz_all[rz_all > 0][:50].tolist()
    print(f"\nLoaded {len(riemann_t)} Riemann zeros from cache")
else:
    # First 29 non-trivial Riemann zeta zeros (imaginary parts)
    riemann_t = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
    print(f"\nUsing {len(riemann_t)} hardcoded Riemann zeros")

# =============================================================================
# FIGURE 1: Zeros in complex plane at L=39
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

L_plot = 39
grid_39, sigmas_39, ts_39 = all_grids[L_plot]

# Left panel: heatmap of log|ζ_M(s)|
ax = axes[0]
extent = [sigmas_39[0], sigmas_39[-1], ts_39[0], ts_39[-1]]
log_grid = np.log10(grid_39.T + 1e-20)
vmin, vmax = np.percentile(log_grid, [1, 99])
im_plot = ax.imshow(log_grid, aspect='auto', origin='lower', extent=extent,
                     cmap='inferno_r', vmin=vmin, vmax=vmax)
ax.axvline(x=0.5, color='cyan', linewidth=2, linestyle='--', alpha=0.8,
           label='Re(s) = ½')
zeros_39 = all_zeros[L_plot]
if zeros_39:
    zr = [z[0] for z in zeros_39]
    zi = [z[1] for z in zeros_39]
    ax.scatter(zr, zi, c='lime', s=60, marker='o', edgecolors='white',
               linewidths=0.8, label=f'{len(zeros_39)} zeros', zorder=5)
ax.set_xlabel('Re(s) = σ', fontsize=12)
ax.set_ylabel('Im(s) = t', fontsize=12)
ax.set_title(f'L = {L_plot}: log₁₀|ζ_M(s)| with zeros', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
plt.colorbar(im_plot, ax=ax, label='log₁₀|ζ_M(s)|', shrink=0.8)

# Right panel: zeros as scatter + deviation lines
ax = axes[1]
ax.axvline(x=0.5, color='red', linewidth=2, linestyle='--', alpha=0.7,
           label='Critical line σ=½')
ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.3)
ax.axvline(x=1, color='gray', linewidth=0.5, alpha=0.3)
ax.fill_betweenx([0, 52], 0, 1, alpha=0.04, color='blue')
if zeros_39:
    for rho_re, rho_im, _ in zeros_39:
        ax.plot([0.5, rho_re], [rho_im, rho_im], '-', color='orange',
                alpha=0.5, linewidth=1)
    ax.scatter([z[0] for z in zeros_39], [z[1] for z in zeros_39],
               c='blue', s=50, marker='o', zorder=5, label=f'{len(zeros_39)} zeros')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 52)
ax.set_xlabel('Re(s) = σ', fontsize=12)
ax.set_ylabel('Im(s) = t', fontsize=12)
ax.set_title(f'L = {L_plot}: ζ_M(s) zeros in critical strip', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure1_zeros_L39.png"), dpi=150)
plt.close()
print("Figure 1 saved.")

# =============================================================================
# FIGURE 2: Mean |σ - 1/2| vs L  +  convergence analysis
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Collect statistics
Ls_plot, mean_devs, std_devs, n_zeros_list = [], [], [], []
all_devs = {}
for L in L_VALUES:
    zeros = all_zeros[L]
    if len(zeros) > 0:
        devs = [abs(z[0] - 0.5) for z in zeros]
        Ls_plot.append(L)
        mean_devs.append(np.mean(devs))
        std_devs.append(np.std(devs) / np.sqrt(len(devs)))
        n_zeros_list.append(len(zeros))
        all_devs[L] = devs

# Left panel: mean deviation vs L
ax = axes[0]
gamma = None
if len(Ls_plot) >= 2:
    ax.errorbar(Ls_plot, mean_devs, yerr=std_devs, fmt='o-', color='blue',
                markersize=10, capsize=5, linewidth=2, label='⟨|σ - ½|⟩')

    # Power-law fit if enough points
    if len(Ls_plot) >= 3 and all(d > 0 for d in mean_devs):
        log_L = np.log(np.array(Ls_plot, dtype=float))
        log_dev = np.log(np.array(mean_devs))
        coeffs = np.polyfit(log_L, log_dev, 1)
        gamma = -coeffs[0]
        L_fit = np.linspace(min(Ls_plot) * 0.8, max(Ls_plot) * 1.2, 200)
        dev_fit = np.exp(coeffs[1]) * L_fit ** coeffs[0]
        ax.plot(L_fit, dev_fit, '--', color='red', alpha=0.7, linewidth=1.5,
                label=f'Fit: L$^{{-{gamma:.2f}}}$')

    ax.set_xlabel('L (lattice size)', fontsize=12)
    ax.set_ylabel('⟨|σ - ½|⟩', fontsize=12)
    ax.set_title('Off-critical deviation vs lattice size', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

# Right panel: box plot of deviations per L
ax = axes[1]
if all_devs:
    box_data = [all_devs[L] for L in Ls_plot]
    bp = ax.boxplot(box_data, positions=list(range(len(Ls_plot))),
                     widths=0.5, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(Ls_plot)))
    ax.set_xticklabels([str(L) for L in Ls_plot])
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('|σ - ½|', fontsize=12)
    ax.set_title('Distribution of off-critical deviations', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure2_deviation_vs_L.png"), dpi=150)
plt.close()
print("Figure 2 saved.")

# =============================================================================
# FIGURE 3: Comparison with Riemann zeros
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: overlay all L zeros with Riemann zero positions
ax = axes[0]
for rt in riemann_t:
    if rt <= 50:
        ax.axhline(y=rt, color='red', alpha=0.2, linewidth=1)
ax.plot([], [], '-', color='red', alpha=0.5, linewidth=2, label='Riemann ζ zeros')
ax.axvline(x=0.5, color='black', linewidth=1.5, linestyle='--', alpha=0.4)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_VALUES)))
for idx, L in enumerate(L_VALUES):
    zeros = all_zeros[L]
    if zeros:
        ax.scatter([z[0] for z in zeros], [z[1] for z in zeros],
                   c=[colors[idx]], s=35, marker='o', alpha=0.8, zorder=5,
                   label=f'L={L} ({len(zeros)})')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 52)
ax.set_xlabel('Re(s)', fontsize=12)
ax.set_ylabel('Im(s)', fontsize=12)
ax.set_title('ζ_M(s) zeros vs Riemann ζ(s) zeros', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)

# Right: nearest-neighbor distance to Riemann zeros
ax = axes[1]
for idx, L in enumerate(L_VALUES):
    zeros = all_zeros[L]
    if not zeros:
        continue
    merkabit_t = sorted([z[1] for z in zeros])
    # For each ζ_M zero, find min distance to any Riemann zero
    min_dists = []
    for mt in merkabit_t:
        dists = [abs(mt - rt) for rt in riemann_t if rt <= 52]
        if dists:
            min_dists.append(min(dists))
    if min_dists:
        ax.scatter([L] * len(min_dists), min_dists, c=[colors[idx]],
                   s=20, alpha=0.5)
        ax.plot(L, np.mean(min_dists), 'D', color=colors[idx],
                markersize=10, markeredgecolor='black', markeredgewidth=0.5,
                zorder=6)
        # Mean spacing of ζ_M zeros for reference
        if len(merkabit_t) >= 2:
            mean_sp = np.mean(np.diff(merkabit_t))
            ax.plot([L-0.3, L+0.3], [mean_sp, mean_sp], '-',
                    color=colors[idx], linewidth=2, alpha=0.7)

ax.set_xlabel('L', fontsize=12)
ax.set_ylabel('Min |t_M - t_Riemann|', fontsize=12)
ax.set_title('Nearest-neighbor distance to Riemann zeros', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure3_riemann_comparison.png"), dpi=150)
plt.close()
print("Figure 3 saved.")

# =============================================================================
# 6. Detailed Riemann comparison
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON WITH RIEMANN ZETA ZEROS")
print("=" * 70)

for L in L_VALUES:
    zeros = all_zeros[L]
    if not zeros:
        print(f"\nL={L}: No zeros found")
        continue

    merkabit_t_vals = [z[1] for z in zeros]
    print(f"\nL={L} ({len(zeros)} ζ_M zeros):")
    print(f"  {'ζ_M Im(ρ)':>12} {'Re(ρ)':>10} {'Near Riem':>12} {'Δt':>8} {'Match?':>8}")
    print("  " + "-" * 55)

    matches = []
    for z in zeros:
        mt, mr = z[1], z[0]
        diffs = [(abs(mt - rt), rt) for rt in riemann_t if rt <= 52]
        if diffs:
            best_diff, best_rt = min(diffs)
            matches.append((mt, mr, best_rt, best_diff))

    # Mean spacing of ζ_M zeros
    if len(merkabit_t_vals) >= 2:
        mean_spacing = np.mean(np.diff(sorted(merkabit_t_vals)))
    else:
        mean_spacing = 50.0

    for mt, mr, rt, dt in sorted(matches, key=lambda m: m[0]):
        flag = " <<<" if dt < mean_spacing * 0.5 else ""
        print(f"  {mt:>12.6f} {mr:>10.6f} {rt:>12.6f} {dt:>8.4f}{flag}")

    n_close = sum(1 for m in matches if m[3] < mean_spacing * 0.5)
    print(f"\n  Mean ζ_M zero spacing: {mean_spacing:.4f}")
    print(f"  Matches within ½ spacing: {n_close}/{len(matches)}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n{'L':>4} {'N_eig':>6} {'N_zeros':>8} {'⟨|σ-½|⟩':>12} "
      f"{'min|σ-½|':>12} {'med|σ-½|':>12} {'trend':>8}")
print("-" * 68)

prev_dev = None
for L in L_VALUES:
    zeros = all_zeros[L]
    n_z = len(zeros)
    if n_z > 0:
        devs = [abs(z[0] - 0.5) for z in zeros]
        mean_dev = np.mean(devs)
        min_dev = np.min(devs)
        med_dev = np.median(devs)
        if prev_dev is not None:
            trend = "↓" if mean_dev < prev_dev else "↑"
        else:
            trend = "—"
        prev_dev = mean_dev
    else:
        mean_dev = min_dev = med_dev = float('nan')
        trend = "N/A"
    print(f"{L:>4} {len(eigenvalues[L]):>6} {n_z:>8} {mean_dev:>12.8f} "
          f"{min_dev:>12.8f} {med_dev:>12.8f} {trend:>8}")

if gamma is not None:
    print(f"\nPower-law fit: ⟨|σ-½|⟩ ~ L^(-{gamma:.4f})")
    if gamma > 0:
        print(f"  → Deviation DECREASES with L (rate γ = {gamma:.4f})")
        print(f"  → Zeros migrate toward critical line as L → ∞")
    else:
        print(f"  → No convergence detected (γ = {gamma:.4f} ≤ 0)")

print(f"\nAll results saved to: {RESULTS_DIR}")
