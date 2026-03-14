#!/usr/bin/env python3
"""
MONTGOMERY PAIR CORRELATION — Analysis 20
==========================================
Direct comparison: M eigenvalue pair correlation vs Riemann zeros vs Montgomery formula.

g(r) = 1 - (sin(pi*r)/(pi*r))^2   [Montgomery 1973, Odlyzko 1987]

M operator: E6 McKay on Eisenstein torus Z[omega], geometric Hopf-paired spinors,
Peierls flux Phi=1/6 (P gate phase), no resonance.

Pipeline: exact Analysis 19 (positive wing >20th pct, degree-10 polynomial unfolding).
"""
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"
os.makedirs(OUT, exist_ok=True)

OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1)]
XI = 3.0

# ============================================================================
# EISENSTEIN TORUS
# ============================================================================
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

# ============================================================================
# GEOMETRIC SPINORS (exact Analysis 19)
# ============================================================================
def assign_spinors_geometric(torus):
    """
    Fully deterministic geometric Hopf-paired spinors.
    u[i] = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]
    v[i] = [-conj(u[1]), conj(u[0])]   (SU(2) Hopf antipode)
    """
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
        u_i = np.exp(1j * theta) * np.array([np.cos(np.pi * r / 2),
                                               1j * np.sin(np.pi * r / 2)],
                                              dtype=complex)
        u_i /= np.linalg.norm(u_i)
        v_i = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
        u[i] = u_i
        v[i] = v_i
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega

# ============================================================================
# M CONSTRUCTION (exact Analysis 19, no resonance, Peierls phase)
# ============================================================================
def build_M(torus, u, v, omega, Phi, xi=XI, use_resonance=False):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = torus.L
    for (i, j) in torus.edges:
        a_i, b_i = torus.nodes[i]
        a_j, b_j = torus.nodes[j]
        if use_resonance:
            resonance = np.exp(-(omega[i] + omega[j]) ** 2 / 0.1)
        else:
            resonance = 1.0
        da = a_j - a_i
        db = b_j - b_i
        if da >  L // 2: da -= L
        if da < -(L // 2): da += L
        if db >  L // 2: db -= L
        if db < -(L // 2): db += L
        A_ij = Phi * (2 * a_i + da) / 2.0 * db
        coupling = decay * resonance * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)
    M = (M + M.conj().T) / 2.0
    return M

# ============================================================================
# SPECTRAL ANALYSIS (exact Analysis 19 pipeline)
# ============================================================================
def unfold_spectrum(eigenvalues, poly_degree=10):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        spacings = np.diff(evals)
        if len(spacings) > 0 and np.mean(spacings) > 0:
            spacings /= np.mean(spacings)
        return evals, spacings
    cdf = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 10))
    coeffs = np.polyfit(evals, cdf * N, deg)
    N_smooth = np.polyval(coeffs, evals)
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings

def extract_positive_wing(eigenvalues, threshold_pct=20):
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    return pos_eigs[pos_eigs > cutoff]

def get_unfolded_wing(eigs):
    """Full Analysis 19 pipeline: positive wing + polynomial unfolding.
    Returns (unfolded_eigenvalues, spacings)."""
    eigs = np.sort(np.real(eigs))
    wing = extract_positive_wing(eigs)
    if len(wing) < 10:
        return wing, np.array([])
    unfolded, spacings = unfold_spectrum(wing)
    return unfolded, spacings

# ============================================================================
# PAIR CORRELATION
# ============================================================================
def pair_correlation_fast(unfolded_eigs, r_vals, bandwidth=0.5):
    """Pair correlation g(r) via Gaussian kernel density estimation."""
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5:
        return np.ones(len(r_vals))

    # All pairwise differences
    diffs = eigs[:, None] - eigs[None, :]  # (N, N)
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]  # N*(N-1) differences

    g = np.zeros(len(r_vals))
    for k, r in enumerate(r_vals):
        g[k] = np.sum(np.exp(-0.5 * ((diffs_flat - r) / bandwidth) ** 2))

    g /= (N * bandwidth * np.sqrt(2 * np.pi))

    # Normalize to 1 at large r
    large_r = g[r_vals > 3.0]
    if len(large_r) > 3:
        norm = np.mean(large_r)
        if norm > 0:
            g /= norm

    return g

def montgomery_formula(r_vals):
    """g(r) = 1 - (sin(pi*r)/(pi*r))^2  [Montgomery 1973]"""
    g = np.ones_like(r_vals, dtype=float)
    nonzero = r_vals > 1e-10
    g[nonzero] = 1 - (np.sin(np.pi * r_vals[nonzero]) /
                       (np.pi * r_vals[nonzero])) ** 2
    g[~nonzero] = 0.0  # g(0) = 0 (level repulsion)
    return g

# ============================================================================
# RIEMANN ZEROS
# ============================================================================
def get_riemann_zeros(n_zeros=1000):
    """Compute Riemann zeros using mpmath."""
    cache_file = os.path.join(OUT, f"riemann_zeros_{n_zeros}.npy")
    if os.path.exists(cache_file):
        gammas = np.load(cache_file)
        print(f"  Loaded {len(gammas)} cached Riemann zeros")
        return gammas

    try:
        from mpmath import mp, zetazero
        mp.dps = 20
        print(f"  Computing {n_zeros} Riemann zeros via mpmath...")
        t0 = time.time()
        gammas = np.array([float(zetazero(n).imag) for n in range(1, n_zeros + 1)])
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")
        print(f"  gamma_1 = {gammas[0]:.6f} (expect 14.134725)")
        print(f"  gamma_{n_zeros} = {gammas[-1]:.4f}")
        np.save(cache_file, gammas)
        return gammas
    except ImportError:
        print("  WARNING: mpmath not available")
        return None

def unfold_riemann_zeros(gammas):
    """Unfold Riemann zeros using Riemann-von Mangoldt formula.
    N(T) ~ T/(2pi) * log(T/(2pi*e)) + 7/8
    """
    T = gammas
    N_smooth = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings

# ============================================================================
# KS TESTS (exact Analysis 19)
# ============================================================================
def make_wigner_cdf(beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b_coeff = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x ** beta * np.exp(-b_coeff * x ** 2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for idx, si in enumerate(s):
            result[idx], _ = quad(lambda x: a * x ** beta * np.exp(-b_coeff * x ** 2), 0, max(si, 0))
        return result
    return cdf

def ks_tests(spacings):
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {k: (1.0, 0.0) for k in ['goe', 'gue', 'poi']}
    results = {}
    for name, beta_val in [('goe', 1), ('gue', 2)]:
        cdf = make_wigner_cdf(beta_val)
        ks_stat, p = stats.kstest(pos, cdf)
        results[name] = (float(ks_stat), float(p))
    ks_stat, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (float(ks_stat), float(p))
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("MONTGOMERY PAIR CORRELATION — Analysis 20")
    print("M operator (E6/Eisenstein/Hopf) vs Riemann zeros vs Montgomery formula")
    print("=" * 70)

    r_vals = np.linspace(0.01, 4.0, 200)

    # ------------------------------------------------------------------
    # STEP 1: Build M at L=18, Phi=1/6
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Build M operator ---")
    L = 18
    Phi = 1.0 / 6
    torus = EisensteinTorus(L)
    u, v, omega = assign_spinors_geometric(torus)
    M = build_M(torus, u, v, omega, Phi, use_resonance=False)
    eigs = np.linalg.eigvalsh(M)
    print(f"  Torus L={L}, N={torus.num_nodes}, edges={len(torus.edges)}")
    print(f"  Eigenvalue range: [{eigs[0]:.6f}, {eigs[-1]:.6f}]")
    print(f"  n_positive = {np.sum(eigs > 0)}, n_negative = {np.sum(eigs < 0)}")

    # Extract positive wing + unfold (Analysis 19 pipeline)
    unfolded_M, spacings_M = get_unfolded_wing(eigs)
    print(f"  Positive wing (>20th pct): {len(unfolded_M)} eigenvalues")

    # KS tests on spacings (confirm GUE)
    ks_M = ks_tests(spacings_M)
    print(f"  KS(GOE) = {ks_M['goe'][0]:.4f}, p = {ks_M['goe'][1]:.4f}")
    print(f"  KS(GUE) = {ks_M['gue'][0]:.4f}, p = {ks_M['gue'][1]:.4f}")
    print(f"  KS(Poi) = {ks_M['poi'][0]:.4f}, p = {ks_M['poi'][1]:.4f}")

    # Pair correlation of M
    print(f"\n  Computing M pair correlation...")
    g_M = pair_correlation_fast(unfolded_M, r_vals, bandwidth=0.4)

    # ------------------------------------------------------------------
    # STEP 2: Montgomery formula
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Montgomery formula ---")
    g_montgomery = montgomery_formula(r_vals)
    print(f"  g(0.01) = {g_montgomery[0]:.6f}")
    print(f"  g(1.00) = {g_montgomery[r_vals.searchsorted(1.0)]:.6f}")
    print(f"  g(2.00) = {g_montgomery[r_vals.searchsorted(2.0)]:.6f}")

    # ------------------------------------------------------------------
    # STEP 3: Riemann zero pair correlation
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Riemann zeros ---")
    gammas = get_riemann_zeros(n_zeros=1000)
    if gammas is not None:
        unfolded_riem, spacings_riem = unfold_riemann_zeros(gammas)
        g_riemann = pair_correlation_fast(unfolded_riem, r_vals, bandwidth=0.4)

        # KS tests on Riemann spacings
        ks_riem = ks_tests(spacings_riem)
        print(f"  Riemann KS(GOE) = {ks_riem['goe'][0]:.4f}, p = {ks_riem['goe'][1]:.4f}")
        print(f"  Riemann KS(GUE) = {ks_riem['gue'][0]:.4f}, p = {ks_riem['gue'][1]:.4f}")
    else:
        g_riemann = None

    # ------------------------------------------------------------------
    # STEP 4: Quantitative comparison
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Quantitative comparison ---")

    # RMS residuals vs Montgomery
    residuals_M = g_M - g_montgomery
    rms_M = np.sqrt(np.mean(residuals_M ** 2))
    print(f"  M vs Montgomery:       RMS = {rms_M:.4f}")

    if g_riemann is not None:
        residuals_riem = g_riemann - g_montgomery
        rms_riem = np.sqrt(np.mean(residuals_riem ** 2))
        print(f"  Riemann vs Montgomery: RMS = {rms_riem:.4f}")
        ratio = rms_M / rms_riem if rms_riem > 0 else float('inf')
        print(f"  Ratio M/Riemann:       {ratio:.3f}x")
        print(f"  (< 2.0 = comparable, < 1.0 = M fits better)")

        # Direct M vs Riemann
        residuals_MR = g_M - g_riemann
        rms_MR = np.sqrt(np.mean(residuals_MR ** 2))
        print(f"  M vs Riemann (direct): RMS = {rms_MR:.4f}")

    # ------------------------------------------------------------------
    # STEP 5: Correlation hole
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Correlation hole ---")
    small_mask = r_vals < 0.3
    g_M_small = g_M[small_mask]
    g_mont_small = g_montgomery[small_mask]
    r_small = r_vals[small_mask]

    print(f"  M operator:      g(r=0.01) = {g_M[0]:.4f}")
    print(f"  Montgomery:      g(r=0.01) = {g_montgomery[0]:.4f}")
    if g_riemann is not None:
        print(f"  Riemann:         g(r=0.01) = {g_riemann[0]:.4f}")
    print(f"  Poisson:         g(r=0.01) = 1.000")

    # Measure hole depth: average g(r) for r < 0.3
    hole_M = np.mean(g_M_small)
    hole_mont = np.mean(g_mont_small)
    print(f"\n  Mean g(r) for r < 0.3:")
    print(f"  M operator:      {hole_M:.4f}")
    print(f"  Montgomery:      {hole_mont:.4f}")
    if g_riemann is not None:
        hole_riem = np.mean(g_riemann[small_mask])
        print(f"  Riemann:         {hole_riem:.4f}")

    # ------------------------------------------------------------------
    # STEP 6: Phi sweep — Montgomery RMS vs Phi
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Montgomery RMS vs Phi (L=18) ---")
    print(f"  {'Phi':>8} | {'RMS vs Montgomery':>18} | {'KS_GUE':>7} | {'note'}")
    print(f"  {'-'*55}")

    phi_sweep_results = []
    for Phi_test in [0.0, 1/12, 1/6, 1/4, 1/3, 1/2]:
        M_test = build_M(torus, u, v, omega, Phi_test, use_resonance=False)
        eigs_test = np.linalg.eigvalsh(M_test)
        unf_test, sp_test = get_unfolded_wing(eigs_test)

        if len(unf_test) < 10:
            continue

        g_test = pair_correlation_fast(unf_test, r_vals, bandwidth=0.4)
        rms_test = np.sqrt(np.mean((g_test - g_montgomery) ** 2))

        ks_gue_test = ks_tests(sp_test)['gue']

        flag = " <- P-gate" if abs(Phi_test - 1/6) < 0.001 else ""
        print(f"  {Phi_test:>8.4f} | {rms_test:>18.4f} | {ks_gue_test[0]:>7.4f} |{flag}")

        phi_sweep_results.append({
            'phi': Phi_test, 'rms': rms_test,
            'ks_gue': ks_gue_test[0], 'p_gue': ks_gue_test[1],
            'g': g_test.copy()
        })

    # ------------------------------------------------------------------
    # STEP 7: Size scaling — pair correlation at multiple L
    # ------------------------------------------------------------------
    print("\n--- STEP 7: Size scaling (L=12,15,18,21,24) ---")
    print(f"  {'L':>4} | {'N':>5} | {'wing':>5} | {'RMS vs Montgomery':>18} | {'KS_GUE':>7}")
    print(f"  {'-'*60}")

    for L_test in [12, 15, 18, 21, 24]:
        t_test = EisensteinTorus(L_test)
        u_t, v_t, om_t = assign_spinors_geometric(t_test)
        M_t = build_M(t_test, u_t, v_t, om_t, 1.0/6, use_resonance=False)
        eigs_t = np.linalg.eigvalsh(M_t)
        unf_t, sp_t = get_unfolded_wing(eigs_t)
        if len(unf_t) < 10:
            continue
        g_t = pair_correlation_fast(unf_t, r_vals, bandwidth=0.4)
        rms_t = np.sqrt(np.mean((g_t - g_montgomery) ** 2))
        ks_t = ks_tests(sp_t)['gue']
        print(f"  {L_test:>4} | {t_test.num_nodes:>5} | {len(unf_t):>5} | {rms_t:>18.4f} | {ks_t[0]:>7.4f}")

    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")

    # Figure 1: THE KEY FIGURE — three curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(r_vals, g_M, 'b-', linewidth=2, label='M operator (L=18, Φ=1/6)')
    if g_riemann is not None:
        ax.plot(r_vals, g_riemann, 'r--', linewidth=1.5, label=f'Riemann zeros (first 1000)')
    ax.plot(r_vals, g_montgomery, 'k:', linewidth=2, label='Montgomery: 1-(sin πr/πr)²')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('r (units of mean spacing)', fontsize=12)
    ax.set_ylabel('g(r) pair correlation', fontsize=12)
    ax.set_title('Pair Correlation: M Operator vs Riemann Zeros vs Montgomery', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.5)
    ax.grid(True, alpha=0.3)

    # Inset: correlation hole detail
    ax_inset = ax.inset_axes([0.45, 0.55, 0.35, 0.35])
    mask_zoom = r_vals < 1.5
    ax_inset.plot(r_vals[mask_zoom], g_M[mask_zoom], 'b-', linewidth=2)
    if g_riemann is not None:
        ax_inset.plot(r_vals[mask_zoom], g_riemann[mask_zoom], 'r--', linewidth=1.5)
    ax_inset.plot(r_vals[mask_zoom], g_montgomery[mask_zoom], 'k:', linewidth=2)
    ax_inset.set_title('Correlation hole detail', fontsize=8)
    ax_inset.set_xlim(0, 1.5)
    ax_inset.set_ylim(-0.1, 1.2)
    ax_inset.grid(True, alpha=0.3)

    # Right panel: residuals
    ax2 = axes[1]
    ax2.plot(r_vals, residuals_M, 'b-', linewidth=1.5, label=f'M - Montgomery (RMS={rms_M:.4f})')
    if g_riemann is not None:
        ax2.plot(r_vals, residuals_riem, 'r--', linewidth=1.5,
                 label=f'Riemann - Montgomery (RMS={rms_riem:.4f})')
    ax2.axhline(0, color='k', ls='-', alpha=0.3)
    ax2.set_xlabel('r (units of mean spacing)', fontsize=12)
    ax2.set_ylabel('Residual g(r) - g_Montgomery(r)', fontsize=12)
    ax2.set_title('Residuals from Montgomery Formula', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/pair_correlation_main.png", dpi=200)
    print(f"  Saved pair_correlation_main.png")
    plt.close()

    # Figure 2: Phi sweep — g(r) at multiple Phi values
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    colors = ['gray', 'orange', 'blue', 'green', 'red', 'purple']
    for idx, res in enumerate(phi_sweep_results):
        lw = 2.5 if abs(res['phi'] - 1/6) < 0.001 else 1.0
        ls = '-' if abs(res['phi'] - 1/6) < 0.001 else '--'
        ax.plot(r_vals, res['g'], color=colors[idx % len(colors)],
                ls=ls, linewidth=lw,
                label=f"Φ={res['phi']:.4f} (RMS={res['rms']:.4f})")
    ax.plot(r_vals, g_montgomery, 'k:', linewidth=2, label='Montgomery')
    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('Pair Correlation vs Phi (L=18, geometric spinors)', fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.5)
    ax.grid(True, alpha=0.3)

    # Right panel: RMS vs Phi
    ax2 = axes[1]
    phis = [r['phi'] for r in phi_sweep_results]
    rmss = [r['rms'] for r in phi_sweep_results]
    ax2.plot(phis, rmss, 'bo-', markersize=8, linewidth=2)
    ax2.axvline(1/6, color='green', ls=':', alpha=0.7, label='Φ=1/6 (P gate)')
    if g_riemann is not None:
        ax2.axhline(rms_riem, color='red', ls='--', alpha=0.7,
                     label=f'Riemann zeros RMS={rms_riem:.4f}')
    ax2.set_xlabel('Phi', fontsize=12)
    ax2.set_ylabel('RMS residual vs Montgomery', fontsize=12)
    ax2.set_title('Montgomery Agreement vs Peierls Flux', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/pair_correlation_vs_phi.png", dpi=200)
    print(f"  Saved pair_correlation_vs_phi.png")
    plt.close()

    # ------------------------------------------------------------------
    # SAVE DATA
    # ------------------------------------------------------------------
    print("\n--- Saving data files ---")

    # Residuals file
    with open(f"{OUT}/residuals.txt", 'w') as f:
        f.write("# r  g_M  g_montgomery")
        if g_riemann is not None:
            f.write("  g_riemann  res_M  res_riemann")
        f.write("\n")
        for k in range(len(r_vals)):
            line = f"{r_vals[k]:.6f}  {g_M[k]:.6f}  {g_montgomery[k]:.6f}"
            if g_riemann is not None:
                line += f"  {g_riemann[k]:.6f}  {residuals_M[k]:.6f}  {residuals_riem[k]:.6f}"
            f.write(line + "\n")
    print(f"  Saved residuals.txt")

    # Montgomery RMS vs Phi
    with open(f"{OUT}/montgomery_rms_vs_phi.txt", 'w') as f:
        f.write("# Phi  RMS_vs_Montgomery  KS_GUE  p_GUE\n")
        for res in phi_sweep_results:
            f.write(f"{res['phi']:.6f}  {res['rms']:.6f}  {res['ks_gue']:.6f}  {res['p_gue']:.6f}\n")
    print(f"  Saved montgomery_rms_vs_phi.txt")

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY — Analysis 20: Montgomery Pair Correlation")
    print("=" * 70)
    summary_lines = []
    summary_lines.append("MONTGOMERY PAIR CORRELATION — Analysis 20")
    summary_lines.append(f"Operator: M on EisensteinTorus(L={L}), geometric Hopf spinors, Phi=1/6")
    summary_lines.append(f"Pipeline: positive wing (>20th pct), degree-10 polynomial unfolding")
    summary_lines.append(f"Wing eigenvalues used: {len(unfolded_M)}")
    summary_lines.append("")
    summary_lines.append("NEAREST-NEIGHBOR SPACING (KS tests):")
    summary_lines.append(f"  M: KS(GOE)={ks_M['goe'][0]:.4f} p={ks_M['goe'][1]:.4f}, "
                         f"KS(GUE)={ks_M['gue'][0]:.4f} p={ks_M['gue'][1]:.4f}, "
                         f"KS(Poi)={ks_M['poi'][0]:.4f} p={ks_M['poi'][1]:.4f}")
    if gammas is not None:
        summary_lines.append(f"  Riemann: KS(GOE)={ks_riem['goe'][0]:.4f} p={ks_riem['goe'][1]:.4f}, "
                             f"KS(GUE)={ks_riem['gue'][0]:.4f} p={ks_riem['gue'][1]:.4f}")
    summary_lines.append("")
    summary_lines.append("PAIR CORRELATION (RMS vs Montgomery formula):")
    summary_lines.append(f"  M operator:       RMS = {rms_M:.4f}")
    if gammas is not None:
        summary_lines.append(f"  Riemann zeros:    RMS = {rms_riem:.4f}")
        summary_lines.append(f"  Ratio M/Riemann:  {ratio:.3f}x")
        summary_lines.append(f"  M vs Riemann:     RMS = {rms_MR:.4f}")
    summary_lines.append("")
    summary_lines.append("CORRELATION HOLE:")
    summary_lines.append(f"  M g(0.01):        {g_M[0]:.4f}")
    summary_lines.append(f"  Montgomery g(0.01): {g_montgomery[0]:.4f}")
    if g_riemann is not None:
        summary_lines.append(f"  Riemann g(0.01):  {g_riemann[0]:.4f}")
    summary_lines.append("")
    summary_lines.append("PHI SWEEP (L=18):")
    for res in phi_sweep_results:
        flag = " <- P-gate" if abs(res['phi'] - 1/6) < 0.001 else ""
        summary_lines.append(f"  Phi={res['phi']:.4f}: RMS={res['rms']:.4f}, "
                             f"KS(GUE)={res['ks_gue']:.4f}{flag}")

    summary = "\n".join(summary_lines)
    print(summary)

    with open(f"{OUT}/SUMMARY.txt", 'w') as f:
        f.write(summary)
    print(f"\n  Saved SUMMARY.txt")
    print(f"\nAll output in: {OUT}")
