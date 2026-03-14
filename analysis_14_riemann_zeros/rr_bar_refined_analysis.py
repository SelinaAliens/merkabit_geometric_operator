#!/usr/bin/env python3
"""
R/R̄ MERGER OPERATOR — REFINED ANALYSIS
========================================

Key findings from initial run:
1. GUE KS distance DECREASES with N (convergence signature)
2. Beta ~ 1.0 (GOE, not GUE) — investigate why
3. Large kernel (~10% near zero) contaminates statistics
4. High spacing variance from near-degenerate eigenvalues

This refined analysis:
- Separates kernel from bulk spectrum
- Tests bulk-only spacing statistics
- Investigates GOE vs GUE (real-symmetry breaking)
- Adds the Floquet Hamiltonian H_F channel coupling
- Tests whether the FULL operator (M + Floquet modulation) gives beta=2
"""

import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\riemann_zeros")

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
XI = 3.0
F_RETURN = 0.696778
OMEGA_EISEN = np.exp(2j * np.pi / 3)
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = 5

RIEMANN_ZEROS_KNOWN = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846
])


# ============================================================================
# LATTICE AND SPINOR CONSTRUCTION
# ============================================================================

def build_eisenstein_lattice(N_sites):
    R_max = int(np.sqrt(N_sites)) + 2
    sites = []
    for a in range(-R_max, R_max + 1):
        for b in range(-R_max, R_max + 1):
            z = a + b * OMEGA_EISEN
            sites.append(z)
    sites = sorted(sites, key=abs)[:N_sites]
    return np.array(sites)


def assign_spinors_e6(sites, L_scale=None):
    N = len(sites)
    if L_scale is None:
        L_scale = np.max(np.abs(sites)) + 1.0
    U = np.zeros((N, 2), dtype=complex)
    V = np.zeros((N, 2), dtype=complex)
    for i, z in enumerate(sites):
        b = np.imag(z) / (np.sqrt(3)/2) if abs(np.imag(z)) > 1e-10 else 0.0
        a = np.real(z) + b/2
        phase = np.pi * (a - b) / RANK_E6
        r = abs(z) / L_scale
        theta = np.pi * r
        u = np.array([
            np.cos(theta/2) * np.exp(1j * phase),
            1j * np.sin(theta/2) * np.exp(-1j * phase)
        ], dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)
        U[i] = u
        V[i] = v
    return U, V


def coupling_strength(z_i, z_j, xi=XI):
    r = abs(z_i - z_j)
    if r < 1e-10:
        return 0.0
    return np.exp(-r / xi)


# ============================================================================
# OPERATOR CONSTRUCTION — THREE VARIANTS
# ============================================================================

def build_RRbar_basic(sites, U, V, xi=XI):
    """Basic R/R-bar merger: M_ij = J(r) * [<u_i|v_j> + <v_i|u_j>]"""
    N = len(sites)
    M = np.zeros((N, N), dtype=complex)
    max_range = 3.0 * xi
    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))
        for j in range(i+1, N):
            r = abs(sites[i] - sites[j])
            if r > max_range:
                continue
            J = coupling_strength(sites[i], sites[j], xi)
            uv = np.vdot(U[i], V[j])
            vu = np.vdot(V[i], U[j])
            coupling = J * (uv + vu)
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)
    return M


def build_RRbar_floquet_modulated(sites, U, V, xi=XI):
    """
    R/R-bar with Floquet modulation from the ouroboros cycle.

    The key insight: the bare R/R-bar operator has an accidental
    time-reversal symmetry (GOE, beta=1). The ouroboros cycle BREAKS
    this symmetry via the asymmetric P gate, which applies opposite
    phases to u and v.

    M_Floquet = sum_{k=0}^{11} U_k^dag * M * U_k / 12

    where U_k is the k-th step of the ouroboros cycle.
    This time-averages the modulated operator over the full period.
    """
    N = len(sites)
    M_base = build_RRbar_basic(sites, U, V, xi)

    # Build the single-site gate matrices for each ouroboros step
    # and apply them to modulate the coupling
    M_floquet = np.zeros((N, N), dtype=complex)

    for k in range(COXETER_H):
        # Get gate angles for step k
        absent = k % NUM_GATES
        p_angle = STEP_PHASE
        sym_base = STEP_PHASE / 3
        omega_k = 2 * np.pi * k / COXETER_H
        rx_a = sym_base * (1.0 + 0.5 * np.cos(omega_k))
        rz_a = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))

        gl = OUROBOROS_GATES[absent]
        if gl == 'S': rz_a *= 0.4; rx_a *= 1.3
        elif gl == 'R': rx_a *= 0.4; rz_a *= 1.3
        elif gl == 'T': rx_a *= 0.7; rz_a *= 0.7
        elif gl == 'P': p_angle *= 0.6; rx_a *= 1.8; rz_a *= 1.5

        # Phase modulation from P gate (asymmetric!)
        # Forward phase: +p/2, Inverse phase: -p/2
        # This BREAKS time-reversal symmetry
        phase_fwd = p_angle / 2
        phase_inv = -p_angle / 2

        # Build modulated coupling matrix
        for i in range(N):
            for j in range(N):
                if abs(M_base[i, j]) < 1e-15:
                    continue
                # Site-dependent phases from Rz rotation
                b_i = np.imag(sites[i]) / (np.sqrt(3)/2) if abs(np.imag(sites[i])) > 1e-10 else 0.0
                a_i = np.real(sites[i]) + b_i/2
                b_j = np.imag(sites[j]) / (np.sqrt(3)/2) if abs(np.imag(sites[j])) > 1e-10 else 0.0
                a_j = np.real(sites[j]) + b_j/2

                # Floquet phase: depends on lattice position and step
                phi_i = rz_a * (a_i - b_i) / RANK_E6
                phi_j = rz_a * (a_j - b_j) / RANK_E6

                # P gate phase difference (time-reversal breaking)
                p_phase = np.exp(1j * (phase_fwd - phase_inv) * (a_i - a_j) / RANK_E6)

                M_floquet[i, j] += M_base[i, j] * p_phase * np.exp(1j * (phi_i - phi_j))

    M_floquet /= COXETER_H

    # Hermitianise (should be nearly Hermitian already)
    M_floquet = (M_floquet + M_floquet.conj().T) / 2
    return M_floquet


def build_RRbar_complex_phase(sites, U, V, xi=XI):
    """
    R/R-bar with explicit complex phase from Eisenstein lattice structure.

    Key: to get GUE (beta=2), the operator must have COMPLEX off-diagonal
    elements that cannot be made real by a gauge transformation.

    On the Eisenstein lattice, the natural gauge field is the
    Peierls phase from the magnetic flux through each hexagonal plaquette:
        A_ij = arg(z_j - z_i) / 6

    This gives: M_ij = J(r) * exp(i*A_ij) * [<u_i|v_j> + <v_i|u_j>]
    """
    N = len(sites)
    M = np.zeros((N, N), dtype=complex)
    max_range = 3.0 * xi

    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))
        for j in range(i+1, N):
            r = abs(sites[i] - sites[j])
            if r > max_range:
                continue
            J = coupling_strength(sites[i], sites[j], xi)
            uv = np.vdot(U[i], V[j])
            vu = np.vdot(V[i], U[j])

            # Eisenstein lattice Peierls phase
            dz = sites[j] - sites[i]
            peierls_phase = np.exp(1j * np.angle(dz) / RANK_E6)

            coupling = J * peierls_phase * (uv + vu)
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)

    return M


# ============================================================================
# SPECTRUM ANALYSIS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=12):
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)
    cdf_empirical = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 20))
    coeffs = np.polyfit(evals_sorted, cdf_empirical * N, deg)
    N_smooth = np.polyval(coeffs, evals_sorted)
    unfolded = N_smooth
    spacings = np.diff(unfolded)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return unfolded, spacings


def unfold_riemann_zeros(zeros):
    T = np.sort(zeros)
    N_smooth = (T / (2*np.pi)) * np.log(T / (2*np.pi)) - T / (2*np.pi) + 7.0/8
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def wigner_surmise(s, beta):
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta+2)/2))**((beta+1)) / (gamma_fn((beta+1)/2))**((beta+2))
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2
    return a * s**beta * np.exp(-b * s**2)


def make_wigner_cdf_func(beta):
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta+2)/2))**((beta+1)) / (gamma_fn((beta+1)/2))**((beta+2))
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2
    def cdf_func(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for i, si in enumerate(s):
            result[i], _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(si, 0))
        return result
    return cdf_func


def fit_level_repulsion_beta(spacings):
    small = spacings[(spacings > 0.01) & (spacings < 0.8)]
    if len(small) < 10:
        small = spacings[(spacings > 0.01) & (spacings < 1.5)]
    if len(small) < 5:
        return 0.0, 0.0
    n_bins = min(25, len(small) // 3)
    if n_bins < 3:
        return 0.0, 0.0
    hist, bin_edges = np.histogram(small, bins=n_bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0
    log_s = np.log(centers[mask])
    log_p = np.log(hist[mask])
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = coeffs[0]
    predicted = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - predicted)**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def pair_correlation(unfolded, r_max=3.0, n_bins=60):
    N = len(unfolded)
    diffs = []
    for i in range(N):
        for j in range(i+1, min(i+50, N)):
            d = abs(unfolded[j] - unfolded[i])
            if d < r_max:
                diffs.append(d)
    if len(diffs) == 0:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)
    diffs = np.array(diffs)
    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(diffs, bins=bins, density=True)
    r_centers = (bins[:-1] + bins[1:]) / 2
    # Normalise to unit mean density
    hist = hist * r_max
    return r_centers, hist


def montgomery_formula(r):
    return np.where(np.abs(r) < 1e-10, 0.0, 1 - (np.sin(np.pi * r) / (np.pi * r))**2)


# ============================================================================
# MAIN REFINED ANALYSIS
# ============================================================================

def analyze_variant(name, M, sites, riemann_spacings, verbose=True):
    """Analyze one operator variant with kernel separation."""
    N = M.shape[0]

    # Verify Hermiticity
    herm_err = np.max(np.abs(M - M.conj().T))
    assert herm_err < 1e-8, f"Not Hermitian: {herm_err}"

    # Check if operator has significant imaginary part
    real_frac = np.linalg.norm(np.real(M)) / (np.linalg.norm(M) + 1e-15)
    imag_frac = np.linalg.norm(np.imag(M)) / (np.linalg.norm(M) + 1e-15)

    eigenvalues = np.linalg.eigvalsh(M)

    if verbose:
        print(f"\n  {name}:")
        print(f"    Hermiticity: {herm_err:.2e}")
        print(f"    Real/Imag fraction: {real_frac:.4f} / {imag_frac:.4f}")
        print(f"    Spectrum: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

    results = {}

    # Analyze FULL spectrum
    _, spacings_full = unfold_spectrum(eigenvalues)
    pos_spacings_full = spacings_full[spacings_full > 0]

    # Analyze BULK only (remove kernel: eigenvalues far from zero)
    # Use Marchenko-Pastur-style edge: keep eigenvalues with |lambda| > threshold
    # Threshold: keep eigenvalues outside the central 20%
    abs_evals = np.abs(eigenvalues)
    threshold = np.percentile(abs_evals, 20)
    bulk_mask = abs_evals > threshold
    bulk_evals = eigenvalues[bulk_mask]

    if len(bulk_evals) > 20:
        _, spacings_bulk = unfold_spectrum(bulk_evals)
        pos_spacings_bulk = spacings_bulk[spacings_bulk > 0]
    else:
        pos_spacings_bulk = pos_spacings_full

    # Analyze POSITIVE eigenvalues only (one "wing" of the spectrum)
    pos_evals = eigenvalues[eigenvalues > threshold]
    if len(pos_evals) > 20:
        _, spacings_pos = unfold_spectrum(pos_evals)
        pos_spacings_wing = spacings_pos[spacings_pos > 0]
    else:
        pos_spacings_wing = pos_spacings_full

    for label, spacings in [("full", pos_spacings_full),
                             ("bulk", pos_spacings_bulk),
                             ("positive_wing", pos_spacings_wing)]:
        if len(spacings) < 10:
            continue

        # KS tests
        goe_cdf = make_wigner_cdf_func(1)
        gue_cdf = make_wigner_cdf_func(2)
        gse_cdf = make_wigner_cdf_func(4)

        ks_goe, p_goe = stats.kstest(spacings, goe_cdf)
        ks_gue, p_gue = stats.kstest(spacings, gue_cdf)
        ks_gse, p_gse = stats.kstest(spacings, gse_cdf)
        ks_poi, p_poi = stats.kstest(spacings, lambda s: 1 - np.exp(-s))
        ks_riem, p_riem = stats.ks_2samp(spacings, riemann_spacings)

        beta_fit, beta_r2 = fit_level_repulsion_beta(spacings)

        if verbose:
            print(f"    [{label}] N_spacings={len(spacings)}")
            print(f"      KS(GOE b=1): {ks_goe:.4f} (p={p_goe:.4f})")
            print(f"      KS(GUE b=2): {ks_gue:.4f} (p={p_gue:.4f})")
            print(f"      KS(GSE b=4): {ks_gse:.4f} (p={p_gse:.4f})")
            print(f"      KS(Poisson): {ks_poi:.4f} (p={p_poi:.4f})")
            print(f"      KS(Riemann): {ks_riem:.4f} (p={p_riem:.4f})")
            print(f"      beta = {beta_fit:.3f} (R2={beta_r2:.3f})")

        results[label] = {
            'ks_goe': ks_goe, 'p_goe': p_goe,
            'ks_gue': ks_gue, 'p_gue': p_gue,
            'ks_gse': ks_gse, 'p_gse': p_gse,
            'ks_poi': ks_poi, 'p_poi': p_poi,
            'ks_riem': ks_riem, 'p_riem': p_riem,
            'beta': beta_fit, 'beta_r2': beta_r2,
            'spacings': spacings,
        }

    results['eigenvalues'] = eigenvalues
    results['real_frac'] = real_frac
    results['imag_frac'] = imag_frac
    results['n_near_zero'] = np.sum(np.abs(eigenvalues) < 0.01)
    return results


def main():
    t_start = time.time()

    print("="*76)
    print("  R/R-BAR MERGER — REFINED ANALYSIS")
    print("  Separating kernel from bulk, testing three operator variants")
    print("="*76)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load Riemann zeros
    zeros_file = RESULTS_DIR / "riemann_zeros_cache.npy"
    if zeros_file.exists():
        riemann_zeros = np.load(zeros_file)
    else:
        riemann_zeros = RIEMANN_ZEROS_KNOWN
    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    pos_spacings_riem = spacings_riem[spacings_riem > 0]
    print(f"  Riemann zeros: {len(riemann_zeros)}")

    # Test at N = 1000 (good balance of size and speed)
    N = 1000
    print(f"\n  Building Eisenstein lattice with N={N} sites...")
    sites = build_eisenstein_lattice(N)
    U, V = assign_spinors_e6(sites)

    print(f"\n{'='*76}")
    print("  VARIANT 1: BARE R/R-BAR (from initial run — expected GOE)")
    print(f"{'='*76}")
    M_bare = build_RRbar_basic(sites, U, V)
    res_bare = analyze_variant("M_bare", M_bare, sites, pos_spacings_riem)

    print(f"\n{'='*76}")
    print("  VARIANT 2: FLOQUET-MODULATED R/R-BAR (time-reversal breaking)")
    print(f"{'='*76}")
    M_floquet = build_RRbar_floquet_modulated(sites, U, V)
    res_floquet = analyze_variant("M_floquet", M_floquet, sites, pos_spacings_riem)

    print(f"\n{'='*76}")
    print("  VARIANT 3: PEIERLS-PHASE R/R-BAR (Eisenstein gauge field)")
    print(f"{'='*76}")
    M_peierls = build_RRbar_complex_phase(sites, U, V)
    res_peierls = analyze_variant("M_peierls", M_peierls, sites, pos_spacings_riem)

    # ====================================================================
    # GENERATE COMPARISON FIGURE
    # ====================================================================
    print(f"\n  Generating comparison figures...")

    s_range = np.linspace(0.01, 4.0, 200)
    goe_curve = np.array([wigner_surmise(s, 1) for s in s_range])
    gue_curve = np.array([wigner_surmise(s, 2) for s in s_range])
    gse_curve = np.array([wigner_surmise(s, 4) for s in s_range])
    poisson_curve = np.exp(-s_range)

    # Figure: 3x1 grid comparing variants (bulk spacings)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (vname, res) in zip(axes, [("Bare M", res_bare),
                                         ("Floquet M", res_floquet),
                                         ("Peierls M", res_peierls)]):
        # Use bulk spacings
        sp_key = 'bulk' if 'bulk' in res else 'full'
        spacings = res[sp_key]['spacings']

        ax.hist(spacings, bins=50, density=True, alpha=0.5, color='steelblue',
                edgecolor='navy', linewidth=0.5, label=f'M ({sp_key})')
        ax.hist(pos_spacings_riem, bins=50, density=True, alpha=0.3, color='coral',
                edgecolor='darkred', linewidth=0.5, label='Riemann')
        ax.plot(s_range, goe_curve, 'b--', linewidth=2, label='GOE (b=1)')
        ax.plot(s_range, gue_curve, 'r-', linewidth=2.5, label='GUE (b=2)')
        ax.plot(s_range, gse_curve, 'g--', linewidth=2, label='GSE (b=4)')
        ax.plot(s_range, poisson_curve, 'k:', linewidth=1.5, label='Poisson')

        beta = res[sp_key]['beta']
        ks_gue = res[sp_key]['ks_gue']
        ax.set_title(f'{vname}\nbeta={beta:.2f}, KS(GUE)={ks_gue:.3f}', fontsize=12)
        ax.set_xlabel('s', fontsize=12)
        ax.set_ylabel('P(s)', fontsize=12)
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'R/R-bar Operator Variants: Spacing Distributions (N={N})', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure7_variant_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    figure7_variant_comparison.png saved")

    # Figure: Pair correlation comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    r_theory = np.linspace(0.01, 3.0, 200)
    g_mont = montgomery_formula(r_theory)

    for ax, (vname, res) in zip(axes, [("Bare M", res_bare),
                                         ("Floquet M", res_floquet),
                                         ("Peierls M", res_peierls)]):
        sp_key = 'bulk' if 'bulk' in res else 'full'
        # Get unfolded from bulk eigenvalues
        abs_evals = np.abs(res['eigenvalues'])
        threshold = np.percentile(abs_evals, 20)
        bulk_evals = res['eigenvalues'][abs_evals > threshold]
        if len(bulk_evals) > 20:
            unfolded, _ = unfold_spectrum(bulk_evals)
            r_c, g_r = pair_correlation(unfolded)
            ax.plot(r_c, g_r, 'b-', linewidth=1.5, alpha=0.7, label=f'{vname}')
        ax.plot(r_theory, g_mont, 'r-', linewidth=2, label='Montgomery')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('r', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title(f'{vname}: Pair Correlation', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure8_pair_correlation_variants.png', dpi=150)
    plt.close(fig)
    print("    figure8_pair_correlation_variants.png saved")

    # ====================================================================
    # SCALING FOR BEST VARIANT
    # ====================================================================
    # Find best variant
    variants = {"bare": res_bare, "floquet": res_floquet, "peierls": res_peierls}
    best_name = None
    best_ks_gue = 1.0
    for vname, res in variants.items():
        sp_key = 'bulk' if 'bulk' in res else 'full'
        ks = res[sp_key]['ks_gue']
        if ks < best_ks_gue:
            best_ks_gue = ks
            best_name = vname

    print(f"\n  Best variant at N={N}: {best_name} (KS(GUE)={best_ks_gue:.4f})")

    # Scaling analysis for best variant
    print(f"\n{'='*76}")
    print(f"  SCALING ANALYSIS: {best_name} variant")
    print(f"{'='*76}")

    N_values = [100, 200, 500, 1000, 2000]
    scaling_results = []

    for N_s in N_values:
        print(f"\n  N = {N_s}...")
        sites_s = build_eisenstein_lattice(N_s)
        U_s, V_s = assign_spinors_e6(sites_s)

        if best_name == "bare":
            M_s = build_RRbar_basic(sites_s, U_s, V_s)
        elif best_name == "floquet":
            M_s = build_RRbar_floquet_modulated(sites_s, U_s, V_s)
        else:
            M_s = build_RRbar_complex_phase(sites_s, U_s, V_s)

        eigenvalues_s = np.linalg.eigvalsh(M_s)

        # Bulk analysis
        abs_ev = np.abs(eigenvalues_s)
        thresh = np.percentile(abs_ev, 20)
        bulk_ev = eigenvalues_s[abs_ev > thresh]

        if len(bulk_ev) > 20:
            _, spacings_s = unfold_spectrum(bulk_ev)
            pos_sp = spacings_s[spacings_s > 0]
        else:
            _, spacings_s = unfold_spectrum(eigenvalues_s)
            pos_sp = spacings_s[spacings_s > 0]

        if len(pos_sp) < 5:
            continue

        gue_cdf = make_wigner_cdf_func(2)
        goe_cdf = make_wigner_cdf_func(1)
        ks_gue, p_gue = stats.kstest(pos_sp, gue_cdf)
        ks_goe, p_goe = stats.kstest(pos_sp, goe_cdf)
        ks_poi, p_poi = stats.kstest(pos_sp, lambda s: 1 - np.exp(-s))
        ks_riem, p_riem = stats.ks_2samp(pos_sp, pos_spacings_riem)
        beta, r2 = fit_level_repulsion_beta(pos_sp)

        scaling_results.append({
            'N': N_s, 'ks_goe': ks_goe, 'ks_gue': ks_gue,
            'ks_poi': ks_poi, 'ks_riem': ks_riem,
            'p_goe': p_goe, 'p_gue': p_gue, 'p_riem': p_riem,
            'beta': beta, 'n_bulk': len(pos_sp)
        })
        print(f"    KS(GOE)={ks_goe:.4f} KS(GUE)={ks_gue:.4f} "
              f"KS(Poi)={ks_poi:.4f} KS(Riem)={ks_riem:.4f} beta={beta:.3f}")

    # Scaling figure
    if scaling_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        Ns = [r['N'] for r in scaling_results]
        axes[0].plot(Ns, [r['ks_goe'] for r in scaling_results], 'b^-', ms=8, lw=2, label='KS(GOE)')
        axes[0].plot(Ns, [r['ks_gue'] for r in scaling_results], 'ro-', ms=8, lw=2, label='KS(GUE)')
        axes[0].plot(Ns, [r['ks_poi'] for r in scaling_results], 'ks-', ms=8, lw=2, label='KS(Poisson)')
        axes[0].plot(Ns, [r['ks_riem'] for r in scaling_results], 'g*-', ms=10, lw=2, label='KS(Riemann)')
        axes[0].set_xlabel('N', fontsize=13)
        axes[0].set_ylabel('KS statistic', fontsize=13)
        axes[0].set_title(f'{best_name.title()} R/R-bar: KS vs N (bulk)', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(Ns, [r['beta'] for r in scaling_results], 'mo-', ms=8, lw=2)
        axes[1].axhline(y=1, color='b', ls='--', label='GOE (b=1)')
        axes[1].axhline(y=2, color='r', ls='--', label='GUE (b=2)')
        axes[1].axhline(y=4, color='g', ls='--', label='GSE (b=4)')
        axes[1].set_xlabel('N', fontsize=13)
        axes[1].set_ylabel('beta', fontsize=13)
        axes[1].set_title('Level Repulsion vs N (bulk)', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'figure9_scaling_refined.png', dpi=150)
        plt.close(fig)
        print("\n    figure9_scaling_refined.png saved")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print(f"\n{'='*76}")
    print("  REFINED ANALYSIS SUMMARY")
    print(f"{'='*76}")

    print(f"\n  Operator Variants (N={N}, bulk spectrum):")
    for vname, res in [("Bare", res_bare), ("Floquet", res_floquet), ("Peierls", res_peierls)]:
        sp_key = 'bulk' if 'bulk' in res else 'full'
        r = res[sp_key]
        print(f"    {vname:10s}: beta={r['beta']:.2f}  "
              f"KS(GOE)={r['ks_goe']:.3f}  KS(GUE)={r['ks_gue']:.3f}  "
              f"KS(Poi)={r['ks_poi']:.3f}  KS(Riem)={r['ks_riem']:.3f}  "
              f"Im/Re={res['imag_frac']:.3f}/{res['real_frac']:.3f}")

    print(f"\n  Key observations:")
    print(f"    1. Bare M is real-symmetric => GOE (beta=1) by construction")
    print(f"    2. Floquet modulation adds complex phase => should shift toward GUE")
    print(f"    3. Peierls gauge field adds lattice-geometric phase")
    print(f"    4. Best variant: {best_name}")

    if scaling_results:
        final = scaling_results[-1]
        print(f"\n  Scaling at largest N = {final['N']}:")
        print(f"    beta = {final['beta']:.3f}")
        print(f"    KS(GUE) = {final['ks_gue']:.4f} (p={final['p_gue']:.4f})")
        print(f"    KS(Riemann) = {final['ks_riem']:.4f} (p={final['p_riem']:.4f})")

    # Save summary
    with open(RESULTS_DIR / "REFINED_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write("R/R-BAR MERGER OPERATOR — REFINED ANALYSIS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for vname, res in [("Bare", res_bare), ("Floquet", res_floquet), ("Peierls", res_peierls)]:
            sp_key = 'bulk' if 'bulk' in res else 'full'
            r = res[sp_key]
            f.write(f"{vname}: beta={r['beta']:.3f} KS(GOE)={r['ks_goe']:.4f} "
                    f"KS(GUE)={r['ks_gue']:.4f} KS(Poi)={r['ks_poi']:.4f} "
                    f"KS(Riem)={r['ks_riem']:.4f}\n")
        if scaling_results:
            f.write(f"\nScaling ({best_name}):\n")
            for r in scaling_results:
                f.write(f"N={r['N']}: beta={r['beta']:.3f} KS(GUE)={r['ks_gue']:.4f} "
                        f"KS(Riem)={r['ks_riem']:.4f}\n")

    elapsed = time.time() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
