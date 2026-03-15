"""
Analysis 36: Extended Spacing Null Test + L=72
===============================================
Tests whether M's spectral zeta zero spacing match with Riemann is
genuine or just what any GUE matrix produces.

Key improvements over Analysis 35:
  - Search window Im(s) in [0.3, 200] (was [0.3, 50]) -> ~100+ zeros per L
  - 79 Riemann zeros (78 spacings) instead of 10 (9 spacings) -> real statistical power
  - GUE null test: 50 trials per L, KS(GUE_spacings, Riemann_spacings) distribution
  - L=72 added (N=5184)
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
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = Path("C:/Users/selin/merkabit_results/spectral_zeta")
CACHE_32 = Path("C:/Users/selin/merkabit_results/analysis_32_montgomery_L38")
CACHE_30 = Path("C:/Users/selin/merkabit_results/analysis_30_dedekind_L42")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_L = [18, 30, 36, 37, 38, 39, 42, 45, 48, 54, 60, 72]
N_GUE_TRIALS = 50
T_MAX = 200.0
N_T = 2000  # grid points in t direction

# Operator parameters
XI = 3.0
PHI = 1.0 / 6
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

print("=" * 70)
print("ANALYSIS 36: EXTENDED SPACING NULL TEST + L=72")
print(f"Search window: Im(s) in [0.3, {T_MAX}]")
print(f"GUE null trials: {N_GUE_TRIALS} per L")
print("=" * 70)


# =============================================================================
# Operator construction (from montgomery_L38.py)
# =============================================================================
class EisensteinTorus:
    def __init__(self, L):
        self.L = L
        self.nodes = []
        self.node_index = {}
        for a in range(L):
            for b in range(L):
                idx = len(self.nodes)
                self.nodes.append((a, b))
                self.node_index[(a, b)] = idx
        self.num_nodes = len(self.nodes)
        self.edges = []
        self.neighbours = defaultdict(list)
        edge_set = set()
        for i, (a, b) in enumerate(self.nodes):
            for da, db in UNIT_VECTORS_AB:
                na = (a + da) % L
                nb = (b + db) % L
                j = self.node_index[(na, nb)]
                self.neighbours[i].append(j)
                edge = (min(i, j), max(i, j))
                if edge not in edge_set and i != j:
                    edge_set.add(edge)
                    self.edges.append(edge)
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            if s == 0: self.chirality.append(0)
            elif s == 1: self.chirality.append(+1)
            else: self.chirality.append(-1)


def assign_spinors_geometric(torus):
    N = torus.num_nodes
    z_coords = [a + b * OMEGA_EISEN for (a, b) in torus.nodes]
    L_max = max(abs(z) for z in z_coords) if N > 1 else 1.0
    u = np.zeros((N, 2), dtype=complex)
    v = np.zeros((N, 2), dtype=complex)
    omega = np.zeros(N)
    for i, (a, b) in enumerate(torus.nodes):
        z = z_coords[i]
        r = abs(z) / (L_max + 1e-10)
        theta = np.pi * (a - b) / 6.0
        u_i = np.exp(1j * theta) * np.array([
            np.cos(np.pi * r / 2), 1j * np.sin(np.pi * r / 2)
        ], dtype=complex)
        u_i /= np.linalg.norm(u_i)
        v_i = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
        u[i] = u_i
        v[i] = v_i
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega


def build_M(torus, u, v, omega):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / XI)
    L = torus.L
    for (i, j) in torus.edges:
        a_i, b_i = torus.nodes[i]
        a_j, b_j = torus.nodes[j]
        da = a_j - a_i
        db = b_j - b_i
        if da > L // 2: da -= L
        if da < -(L // 2): da += L
        if db > L // 2: db -= L
        if db < -(L // 2): db += L
        A_ij = PHI * (2 * a_i + da) / 2.0 * db
        coupling = decay * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)
    M = (M + M.conj().T) / 2.0
    return M


def compute_eigenvalues(L):
    print(f"    Building L={L} (N={L*L})...", end=" ", flush=True)
    t0 = time.time()
    torus = EisensteinTorus(L)
    u, v, omega = assign_spinors_geometric(torus)
    M_mat = build_M(torus, u, v, omega)
    eigs = np.linalg.eigvalsh(M_mat)
    print(f"done ({time.time()-t0:.1f}s)")
    return eigs


# =============================================================================
# Extended spectral zeta zero-finding (Im(s) up to T_MAX)
# =============================================================================
def find_zeros_extended(abs_eigs, n_sig=60, n_t=N_T, t_max=T_MAX):
    """Find zeros of zeta_M(s) in extended critical strip [0, t_max]."""
    log_eigs = np.log(abs_eigs)
    sigmas = np.linspace(0.02, 0.98, n_sig)
    ts = np.linspace(0.3, t_max, n_t)
    dsig = sigmas[1] - sigmas[0]
    dt = ts[1] - ts[0]

    # Vectorized grid evaluation
    weights = np.exp(-np.outer(sigmas, log_eigs))  # (n_sig, N)
    phases = np.exp(-1j * np.outer(ts, log_eigs))  # (n_t, N)
    grid = np.abs(weights @ phases.T)  # (n_sig, n_t)

    # Find local minima (bottom 2%)
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

    # Local refinement (top 300 candidates for extended window)
    n_refine = min(300, len(candidates))
    refined = []
    for sig0, t0, v0 in candidates[:n_refine]:
        s_lo = max(0.01, sig0 - 2*dsig)
        s_hi = min(0.99, sig0 + 2*dsig)
        t_lo = max(0.1, t0 - 2*dt)
        t_hi = min(t_max + 1, t0 + 2*dt)
        loc_sigs = np.linspace(s_lo, s_hi, 20)
        loc_ts = np.linspace(t_lo, t_hi, 20)
        loc_w = np.exp(-np.outer(loc_sigs, log_eigs))
        loc_p = np.exp(-1j * np.outer(loc_ts, log_eigs))
        loc_grid = np.abs(loc_w @ loc_p.T)
        idx = np.unravel_index(np.argmin(loc_grid), loc_grid.shape)
        refined.append((loc_sigs[idx[0]], loc_ts[idx[1]], loc_grid[idx]))

    # Deduplicate
    unique = []
    for c in refined:
        if not any(abs(c[0]-u[0]) < 0.005 and abs(c[1]-u[1]) < 0.03 for u in unique):
            unique.append(c)
    unique.sort(key=lambda x: x[2])

    # Newton refinement (top 200 for extended window)
    zeros = []
    h = 1e-6
    for sig0, t0, v0 in unique[:200]:
        s = complex(sig0, t0)
        for _ in range(50):
            f = np.sum(np.exp(-s * log_eigs))
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
        if (0 < rho_re < 1 and 0.1 < rho_im <= t_max + 1 and residual < 1e-4):
            if not any(abs(z[0]-rho_re) < 0.001 and abs(z[1]-rho_im) < 0.005
                       for z in zeros):
                zeros.append((rho_re, rho_im))
    zeros.sort(key=lambda z: z[1])
    return zeros


def generate_gue_eigenvalues(N, rng):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / (2 * np.sqrt(2 * N))
    return np.linalg.eigvalsh(H)


def compute_normalized_spacings(zeros):
    """Compute normalized nearest-neighbor spacings from zero list."""
    if len(zeros) < 3:
        return np.array([])
    t_vals = np.sort([z[1] for z in zeros])
    spacings = np.diff(t_vals)
    mean_sp = np.mean(spacings)
    if mean_sp < 1e-10:
        return np.array([])
    return spacings / mean_sp


# =============================================================================
# STEP 0: Load / compute eigenvalues
# =============================================================================
print("\n" + "=" * 70)
print("STEP 0: Loading eigenvalues")
print("=" * 70)

eigenvalues = {}
wing_sizes = {}

for L in ALL_L:
    cache_32 = CACHE_32 / f"eigs_L{L}.npy"
    cache_30 = CACHE_30 / f"eigs_L{L}.npy"
    local_cache = RESULTS_DIR / f"eigs_L{L}.npy"

    if cache_32.exists():
        eigs = np.load(str(cache_32))
    elif cache_30.exists():
        eigs = np.load(str(cache_30))
    elif local_cache.exists():
        eigs = np.load(str(local_cache))
    else:
        eigs = compute_eigenvalues(L)
        np.save(str(local_cache), eigs)

    abs_eigs = np.abs(eigs)
    pos_eigs = abs_eigs[abs_eigs > 1e-6]
    pos_eigs = np.sort(pos_eigs)
    eigenvalues[L] = pos_eigs
    wing_sizes[L] = len(pos_eigs)
    print(f"  L={L}: N={len(eigs)}, N_used={len(pos_eigs)}, "
          f"max|eig|={np.max(abs_eigs):.4f}")


# =============================================================================
# STEP 1: Load Riemann zeros (extended)
# =============================================================================
riemann_cache = Path("C:/Users/selin/merkabit_results/riemann_zeros/riemann_zeros_cache.npy")
rz_all = np.load(str(riemann_cache))
rz_pos = rz_all[rz_all > 0]
riemann_t = rz_pos[rz_pos <= T_MAX + 1]
print(f"\nRiemann zeros with t <= {T_MAX}: {len(riemann_t)}")
riemann_t_arr = np.array(riemann_t)

# Riemann normalized spacings
riemann_spacings = np.diff(np.sort(riemann_t_arr))
riemann_mean_sp = np.mean(riemann_spacings)
riemann_norm_sp = riemann_spacings / riemann_mean_sp
print(f"Riemann spacings: {len(riemann_norm_sp)} values, mean spacing = {riemann_mean_sp:.4f}")


# =============================================================================
# STEP 2: M operator spectral zeta zeros (extended window)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: M operator zeros in extended window [0.3, 200]")
print("=" * 70)

M_zeros = {}
M_norm_spacings = {}
M_mean_devs = {}

for L in ALL_L:
    t0 = time.time()
    zeros = find_zeros_extended(eigenvalues[L])
    elapsed = time.time() - t0
    M_zeros[L] = zeros

    norm_sp = compute_normalized_spacings(zeros)
    M_norm_spacings[L] = norm_sp

    if zeros:
        devs = [abs(z[0] - 0.5) for z in zeros]
        M_mean_devs[L] = np.mean(devs)
        print(f"  L={L}: {len(zeros)} zeros, {len(norm_sp)} spacings, "
              f"<|s-1/2|>={np.mean(devs):.4f}, min={np.min(devs):.4f} ({elapsed:.1f}s)")
    else:
        M_mean_devs[L] = 0.5
        print(f"  L={L}: 0 zeros ({elapsed:.1f}s)")


# =============================================================================
# STEP 3: KS tests for M spacings
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: M spacing statistics vs Riemann (78 spacings)")
print("=" * 70)

# GUE Wigner surmise CDF
def gue_wigner_cdf(s):
    from scipy.integrate import quad
    pdf = lambda x: (32.0 / np.pi**2) * x**2 * np.exp(-4 * x**2 / np.pi)
    result, _ = quad(pdf, 0, s)
    return result

M_ks_riemann = {}
header = f"{'L':>4} {'N_zeros':>8} {'N_sp':>6} {'KS(Riem)':>10} {'p(Riem)':>10} {'KS(GUE_W)':>10} {'p(GUE_W)':>10} {'KS(Pois)':>10} {'p(Pois)':>10}"
print(f"\n{header}")
print("-" * 90)

for L in ALL_L:
    norm_sp = M_norm_spacings[L]
    n_z = len(M_zeros[L])

    if len(norm_sp) < 4:
        print(f"{L:>4} {n_z:>8} {len(norm_sp):>6} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        M_ks_riemann[L] = 1.0
        continue

    # KS vs Riemann
    ks_r, p_r = stats.ks_2samp(norm_sp, riemann_norm_sp)
    M_ks_riemann[L] = ks_r

    # KS vs GUE Wigner
    gue_cdf_vals = np.array([gue_wigner_cdf(s) for s in np.sort(norm_sp)])
    ecdf_vals = np.arange(1, len(norm_sp) + 1) / len(norm_sp)
    ks_g = np.max(np.abs(ecdf_vals - gue_cdf_vals))
    p_g = stats.ksone.sf(ks_g, len(norm_sp)) * 2

    # KS vs Poisson
    ks_p, p_p = stats.kstest(norm_sp, 'expon', args=(0, 1))

    print(f"{L:>4} {n_z:>8} {len(norm_sp):>6} {ks_r:>10.4f} {p_r:>10.4f} "
          f"{ks_g:>10.4f} {p_g:>10.4f} {ks_p:>10.4f} {p_p:>10.4f}")


# =============================================================================
# STEP 4: GUE null test for spacing (CORE TEST)
# =============================================================================
print("\n" + "=" * 70)
print(f"STEP 4: GUE NULL TEST ({N_GUE_TRIALS} trials per L)")
print("=" * 70)

rng = np.random.default_rng(seed=2026)
gue_ks_distributions = {L: [] for L in ALL_L}
gue_n_zeros_dist = {L: [] for L in ALL_L}

t_total = time.time()

for L in ALL_L:
    N = wing_sizes[L]
    print(f"\n  L={L} (N={N}): ", end="", flush=True)
    t_L = time.time()

    for trial in range(N_GUE_TRIALS):
        gue_eigs = generate_gue_eigenvalues(N, rng)
        abs_gue = np.abs(gue_eigs)
        abs_gue = abs_gue[abs_gue > 1e-6]
        abs_gue = np.sort(abs_gue)

        if len(abs_gue) < 10:
            gue_ks_distributions[L].append(1.0)
            gue_n_zeros_dist[L].append(0)
            continue

        zeros = find_zeros_extended(abs_gue)
        gue_n_zeros_dist[L].append(len(zeros))
        norm_sp = compute_normalized_spacings(zeros)

        if len(norm_sp) >= 4:
            ks_val, _ = stats.ks_2samp(norm_sp, riemann_norm_sp)
            gue_ks_distributions[L].append(ks_val)
        else:
            gue_ks_distributions[L].append(1.0)

        if (trial + 1) % 10 == 0:
            print(f"{trial+1}", end=" ", flush=True)

    elapsed_L = time.time() - t_L
    ks_arr = np.array(gue_ks_distributions[L])
    valid = ks_arr[ks_arr < 1.0]
    if len(valid) > 0:
        print(f"\n    {elapsed_L:.1f}s. GUE KS(Riem): mean={np.mean(valid):.4f} "
              f"+/- {np.std(valid):.4f}, mean zeros={np.mean(gue_n_zeros_dist[L]):.1f}")
    else:
        print(f"\n    {elapsed_L:.1f}s. No valid GUE trials")

total_elapsed = time.time() - t_total
print(f"\nTotal GUE time: {total_elapsed:.1f}s")


# =============================================================================
# STEP 5: Statistical comparison
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Statistical comparison — M vs GUE null")
print("=" * 70)

results_lines = []
results_lines.append("Analysis 36: Extended Spacing Null Test + L=72")
results_lines.append("=" * 70)
results_lines.append(f"Search window: Im(s) in [0.3, {T_MAX}]")
results_lines.append(f"Riemann zeros: {len(riemann_t)} (t <= {T_MAX}), {len(riemann_norm_sp)} spacings")
results_lines.append(f"GUE trials: {N_GUE_TRIALS} per L")
results_lines.append("")

header_s = (f"{'L':>4} {'M_zeros':>8} {'M_KS':>8} {'GUE_mean':>9} {'GUE_std':>8} "
            f"{'z-score':>8} {'p-value':>8} {'M_dev':>8}")
print(f"\n{header_s}")
print("-" * 75)
results_lines.append(header_s)
results_lines.append("-" * 75)

z_scores = []
p_values = []

for L in ALL_L:
    m_ks = M_ks_riemann[L]
    gue_ks = np.array(gue_ks_distributions[L])
    valid_gue = gue_ks[gue_ks < 1.0]

    if len(valid_gue) < 5:
        line = f"{L:>4} {len(M_zeros[L]):>8} {m_ks:>8.4f} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {M_mean_devs[L]:>8.4f}"
        print(line)
        results_lines.append(line)
        continue

    gue_mean = np.mean(valid_gue)
    gue_std = np.std(valid_gue)

    # z-score: negative = M is CLOSER to Riemann than GUE
    if gue_std > 0:
        z = (m_ks - gue_mean) / gue_std
    else:
        z = 0.0

    # Empirical p-value: fraction of GUE with KS <= M's KS (lower KS = better)
    p_emp = np.mean(valid_gue <= m_ks)
    p_emp_adj = max(p_emp, 1.0 / (len(valid_gue) + 1))

    z_scores.append(z)
    p_values.append(p_emp_adj)

    line = (f"{L:>4} {len(M_zeros[L]):>8} {m_ks:>8.4f} {gue_mean:>9.4f} {gue_std:>8.4f} "
            f"{z:>8.2f} {p_emp:>8.3f} {M_mean_devs[L]:>8.4f}")
    print(line)
    results_lines.append(line)

# Fisher combined p-value
print("\n" + "-" * 75)
results_lines.append("")

if len(p_values) >= 2:
    fisher_stat = -2 * np.sum(np.log(p_values))
    from scipy.stats import chi2
    fisher_p = 1 - chi2.cdf(fisher_stat, df=2 * len(p_values))
    mean_z = np.mean(z_scores)

    print(f"\nFisher combined: chi2={fisher_stat:.3f}, df={2*len(p_values)}, p={fisher_p:.6f}")
    print(f"Mean z-score: {mean_z:.3f}")
    results_lines.append(f"Fisher combined: chi2={fisher_stat:.3f}, df={2*len(p_values)}, p={fisher_p:.6f}")
    results_lines.append(f"Mean z-score: {mean_z:.3f}")
else:
    mean_z = 0
    fisher_p = 1.0

# Verdict
print("\n" + "=" * 70)
print("VERDICT: Spacing Null Test")
print("=" * 70)
results_lines.append("")
results_lines.append("=" * 70)
results_lines.append("VERDICT")
results_lines.append("=" * 70)

if mean_z < -2 and fisher_p < 0.05:
    verdict = "SIGNIFICANT: M's spacing matches Riemann BETTER than GUE null"
elif mean_z < -1:
    verdict = f"SUGGESTIVE: Mean z = {mean_z:.2f} (trend toward Riemann match)"
elif abs(mean_z) < 1:
    verdict = f"CONSISTENT WITH GUE NULL: Mean z = {mean_z:.2f} (M not special)"
else:
    verdict = f"M WORSE than GUE: Mean z = {mean_z:.2f} (GUE matches Riemann better)"

print(f"\n  {verdict}")
print(f"  Fisher p = {fisher_p:.6f}")
results_lines.append(f"  {verdict}")
results_lines.append(f"  Fisher p = {fisher_p:.6f}")

for L, z, p in zip([L for L in ALL_L if L in M_ks_riemann],
                    z_scores, p_values):
    line = f"  L={L}: z = {z:+.3f}, p = {p:.4f}"
    print(line)
    results_lines.append(line)


# =============================================================================
# STEP 6: Extended convergence (Part A update with L=72)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Extended convergence with L=72")
print("=" * 70)
results_lines.append("")
results_lines.append("EXTENDED CONVERGENCE (Part A update)")
results_lines.append("-" * 50)

Ls_valid = []
mean_devs = []
min_devs = []

for L in ALL_L:
    zeros = M_zeros[L]
    if len(zeros) >= 2:
        devs = [abs(z[0] - 0.5) for z in zeros]
        Ls_valid.append(L)
        mean_devs.append(np.mean(devs))
        min_devs.append(np.min(devs))
        line = f"  L={L}: {len(zeros)} zeros, <|s-1/2|>={np.mean(devs):.4f}, min={np.min(devs):.6f}"
        print(line)
        results_lines.append(line)

Ls_arr = np.array(Ls_valid, dtype=float)
mean_arr = np.array(mean_devs)
min_arr = np.array(min_devs)

# Power-law fit
log_L = np.log(Ls_arr)
log_dev = np.log(mean_arr)
coeffs = np.polyfit(log_L, log_dev, 1)
gamma = -coeffs[0]
A_fit = np.exp(coeffs[1])

# Bootstrap CI
gammas_boot = []
residuals = log_dev - (coeffs[0] * log_L + coeffs[1])
boot_rng = np.random.default_rng(2026)
for _ in range(1000):
    boot_resid = boot_rng.choice(residuals, size=len(residuals), replace=True)
    boot_log_dev = coeffs[0] * log_L + coeffs[1] + boot_resid
    boot_coeffs = np.polyfit(log_L, boot_log_dev, 1)
    gammas_boot.append(-boot_coeffs[0])
gammas_boot = np.array(gammas_boot)
gamma_lo = np.percentile(gammas_boot, 2.5)
gamma_hi = np.percentile(gammas_boot, 97.5)

print(f"\n  gamma = {gamma:.4f} [{gamma_lo:.4f}, {gamma_hi:.4f}] (12 L values)")
print(f"  Best min|s-1/2| = {np.min(min_arr):.6f} at L={Ls_arr[np.argmin(min_arr)]:.0f}")
results_lines.append(f"\n  gamma = {gamma:.4f} [{gamma_lo:.4f}, {gamma_hi:.4f}]")
results_lines.append(f"  Best min|s-1/2| = {np.min(min_arr):.6f} at L={Ls_arr[np.argmin(min_arr)]:.0f}")


# =============================================================================
# Save results
# =============================================================================
with open(str(RESULTS_DIR / "analysis_36_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines) + "\n")
print(f"\nResults saved to {RESULTS_DIR / 'analysis_36_results.txt'}")


# =============================================================================
# FIGURE 9: Spacing null test histograms
# =============================================================================
plot_Ls = [L for L in [18, 39, 48, 60, 72] if L in gue_ks_distributions]
fig, axes = plt.subplots(1, len(plot_Ls), figsize=(4*len(plot_Ls), 5), sharey=False)
if len(plot_Ls) == 1:
    axes = [axes]

for ax, L in zip(axes, plot_Ls):
    gue_ks = np.array(gue_ks_distributions[L])
    valid = gue_ks[gue_ks < 1.0]
    m_ks = M_ks_riemann[L]

    if len(valid) > 0:
        ax.hist(valid, bins=15, color='steelblue', alpha=0.7, edgecolor='white',
                density=False, label='GUE null')
    ax.axvline(m_ks, color='red', linewidth=2.5, linestyle='--',
               label=f'M (KS={m_ks:.3f})')

    # Find z-score for this L
    z_val = 0
    if len(valid) > 0:
        z_val = (m_ks - np.mean(valid)) / (np.std(valid) + 1e-10)

    ax.set_xlabel('KS(spacings, Riemann)', fontsize=11)
    ax.set_ylabel('Count' if ax == axes[0] else '', fontsize=11)
    ax.set_title(f'L = {L}\nz = {z_val:+.2f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

plt.suptitle('Analysis 36: GUE Null Test for Spacing Match with Riemann', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure9_spacing_null.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved.")


# =============================================================================
# FIGURE 10: CDF comparison with 78 Riemann spacings
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: CDF comparison
ax = axes[0]
s_range = np.linspace(0, 4, 200)

# GUE Wigner surmise CDF
gue_cdf_curve = np.array([gue_wigner_cdf(s) for s in s_range])
ax.plot(s_range, gue_cdf_curve, 'k-', linewidth=2, label='GUE Wigner')

# Poisson CDF
ax.plot(s_range, 1 - np.exp(-s_range), 'k:', linewidth=1.5, alpha=0.5, label='Poisson')

# Riemann CDF (78 spacings)
riem_sorted = np.sort(riemann_norm_sp)
riem_ecdf = np.arange(1, len(riem_sorted) + 1) / len(riem_sorted)
ax.step(riem_sorted, riem_ecdf, color='red', linewidth=2.5, alpha=0.8,
        label=f'Riemann ({len(riemann_norm_sp)} sp)', where='post')

# M for select L values
colors_m = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
select_Ls = [L for L in [39, 54, 60, 72] if L in M_norm_spacings and len(M_norm_spacings[L]) >= 4]
for idx, L in enumerate(select_Ls):
    sp = np.sort(M_norm_spacings[L])
    ecdf = np.arange(1, len(sp) + 1) / len(sp)
    ax.step(sp, ecdf, color=colors_m[idx], linewidth=1.5, alpha=0.8,
            label=f'M L={L} ({len(sp)} sp)', where='post')

ax.set_xlabel('Normalized spacing s', fontsize=12)
ax.set_ylabel('CDF', fontsize=12)
ax.set_title(f'Spacing CDF: 78 Riemann Spacings', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 4)

# Panel 2: KS(M, Riemann) vs L with GUE null band
ax = axes[1]
Ls_ks = []
m_ks_vals = []
gue_means = []
gue_stds = []

for L in ALL_L:
    if M_ks_riemann.get(L, 1.0) < 1.0:
        valid = np.array(gue_ks_distributions[L])
        valid = valid[valid < 1.0]
        if len(valid) >= 5:
            Ls_ks.append(L)
            m_ks_vals.append(M_ks_riemann[L])
            gue_means.append(np.mean(valid))
            gue_stds.append(np.std(valid))

if Ls_ks:
    Ls_ks = np.array(Ls_ks)
    m_ks_vals = np.array(m_ks_vals)
    gue_means = np.array(gue_means)
    gue_stds = np.array(gue_stds)

    ax.fill_between(Ls_ks, gue_means - 2*gue_stds, gue_means + 2*gue_stds,
                    alpha=0.15, color='steelblue', label='GUE null 2-sigma')
    ax.fill_between(Ls_ks, gue_means - gue_stds, gue_means + gue_stds,
                    alpha=0.3, color='steelblue', label='GUE null 1-sigma')
    ax.plot(Ls_ks, gue_means, 'o-', color='steelblue', markersize=6, label='GUE mean')
    ax.plot(Ls_ks, m_ks_vals, 'D-', color='red', markersize=8, markeredgecolor='black',
            markeredgewidth=1, linewidth=1.5, label='M operator', zorder=5)

ax.set_xlabel('L', fontsize=12)
ax.set_ylabel('KS(spacings, Riemann)', fontsize=12)
ax.set_title('KS Distance to Riemann: M vs GUE Null', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Analysis 36: Extended Riemann Spacing Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure10_extended_cdf.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10 saved.")


# =============================================================================
# FIGURE 11: Updated convergence with L=72
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Mean deviation
ax = axes[0]
std_errs = [np.std([abs(z[0]-0.5) for z in M_zeros[L]]) / np.sqrt(len(M_zeros[L]))
            for L in Ls_valid]
ax.errorbar(Ls_arr, mean_arr, yerr=std_errs, fmt='o', color='blue',
            markersize=8, capsize=4, linewidth=1.5, zorder=5)
L_fit = np.linspace(15, 85, 200)
ax.plot(L_fit, A_fit * L_fit**(-gamma), '--', color='red', linewidth=1.5,
        label=f'Fit: L^(-{gamma:.3f})')
ax.set_xlabel('L', fontsize=12)
ax.set_ylabel('Mean |sigma - 1/2|', fontsize=12)
ax.set_title(f'Extended Convergence (12 L values)\ngamma = {gamma:.3f} [{gamma_lo:.3f}, {gamma_hi:.3f}]',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Min deviation
ax = axes[1]
ax.plot(Ls_arr, min_arr, 'o-', color='darkgreen', markersize=8, linewidth=1.5)
ax.axhline(y=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('L', fontsize=12)
ax.set_ylabel('Min |sigma - 1/2|', fontsize=12)
ax.set_title(f'Closest Zero to Critical Line\nBest = {np.min(min_arr):.6f} at L={Ls_arr[np.argmin(min_arr)]:.0f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-0.002)

plt.suptitle('Analysis 36: Convergence Update with L=72', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "figure11_convergence_L72.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 11 saved.")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")
