#!/usr/bin/env python3
"""
PEIERLS FLUX SWEEP: Breaking Time-Reversal Symmetry in the R/R̄ Operator
=========================================================================

The bare M = |u><v| + |v><u| is purely imaginary (eigenvalues in ±pairs).
This forces GOE (beta=1). To reach GUE (beta=2), we need a magnetic flux
that breaks time-reversal symmetry.

On the Eisenstein lattice, the natural flux is a Peierls phase:
    M_ij -> M_ij * exp(i * phi * A_ij)

where A_ij is the directed area enclosed between sites i, j and the origin.
phi = magnetic flux per hexagonal plaquette.

Sweep phi from 0 to pi and find phi* where:
    - beta -> 2 (GUE level repulsion)
    - KS(M, GUE) is minimised
    - KS(M, Riemann) is minimised

Key prediction: if phi* = pi/6 = 2*pi/12 = one Coxeter step,
that is a structural result linking the Floquet cycle to RMT.
"""

import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.integrate import quad

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\riemann_zeros")
np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # pi/6
XI = 3.0
OMEGA_EISEN = np.exp(2j * np.pi / 3)


# ============================================================================
# LATTICE AND SPINORS (from previous runs)
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
        b = np.imag(z) / (np.sqrt(3) / 2) if abs(np.imag(z)) > 1e-10 else 0.0
        a = np.real(z) + b / 2
        phase = np.pi * (a - b) / RANK_E6
        r = abs(z) / L_scale
        theta = np.pi * r
        u = np.array([
            np.cos(theta / 2) * np.exp(1j * phase),
            1j * np.sin(theta / 2) * np.exp(-1j * phase)
        ], dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)
        U[i] = u
        V[i] = v
    return U, V


# ============================================================================
# DIRECTED AREA (PEIERLS PHASE)
# ============================================================================

def directed_area(z_i, z_j):
    """
    Directed area of the triangle (origin, z_i, z_j).

    A_ij = (1/2) * Im(conj(z_i) * z_j)

    This is the signed area: positive for counterclockwise (i -> j),
    negative for clockwise. On the hexagonal lattice, a full plaquette
    has area sqrt(3)/2.

    The Peierls phase exp(i * phi * A_ij / A_plaquette) assigns
    one flux quantum phi per plaquette.
    """
    return 0.5 * np.imag(np.conj(z_i) * z_j)


def hexagonal_plaquette_area():
    """Area of one hexagonal plaquette on the Eisenstein lattice.
    The unit hexagon has vertices at the 6 nearest neighbours of the origin.
    Area = 6 * (equilateral triangle area) = 6 * sqrt(3)/4 = 3*sqrt(3)/2.
    But each plaquette is ONE equilateral triangle (the fundamental domain),
    with area sqrt(3)/4 for unit-length edges.

    Actually, the Eisenstein lattice has TWO triangular plaquettes per
    hexagonal cell. The fundamental plaquette area is sqrt(3)/4.
    """
    # Two nearest-neighbour vectors: (1, 0) and omega = (-1/2, sqrt(3)/2)
    # Triangle area = |Im(conj(1) * omega)| / 2 = sqrt(3)/4
    return np.sqrt(3) / 4


# ============================================================================
# BUILD M WITH PEIERLS FLUX
# ============================================================================

def build_M_with_flux(sites, U, V, phi, xi=XI):
    """
    Build R/R-bar operator with Peierls phase phi per plaquette.

    M_ij = J(r_ij) * exp(i * phi * A_ij / A_plaq) * [<u_i|v_j> + <v_i|u_j>]

    At phi=0: bare operator (purely imaginary, GOE)
    At phi=pi/6: one Coxeter step per plaquette
    At phi=pi/3: one flux quantum per plaquette (strong breaking)
    """
    N = len(sites)
    M = np.zeros((N, N), dtype=complex)
    max_range = 3.0 * xi
    A_plaq = hexagonal_plaquette_area()

    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))  # on-site (always real)

        for j in range(i + 1, N):
            r = abs(sites[i] - sites[j])
            if r > max_range:
                continue

            J = np.exp(-r / xi)

            # Base R/R-bar coupling
            uv = np.vdot(U[i], V[j])
            vu = np.vdot(V[i], U[j])
            base_coupling = uv + vu

            # Peierls phase from directed area
            A_ij = directed_area(sites[i], sites[j])
            peierls = np.exp(1j * phi * A_ij / A_plaq)

            coupling = J * peierls * base_coupling
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)  # Hermitian

    return M


# ============================================================================
# SPECTRUM ANALYSIS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=12):
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)
    cdf_emp = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 20))
    coeffs = np.polyfit(evals_sorted, cdf_emp * N, deg)
    N_smooth = np.polyval(coeffs, evals_sorted)
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def unfold_riemann_zeros(zeros):
    T = np.sort(zeros)
    N_smooth = (T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7.0 / 8
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def make_wigner_cdf_func(beta):
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** ((beta + 1)) / (gamma_fn((beta + 1) / 2)) ** ((beta + 2))
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2

    def cdf_func(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for i, si in enumerate(s):
            result[i], _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(si, 0))
        return result

    return cdf_func


def fit_beta(spacings):
    """Fit level repulsion exponent: P(s) ~ s^beta for small s."""
    small = spacings[(spacings > 0.02) & (spacings < 0.8)]
    if len(small) < 10:
        small = spacings[(spacings > 0.01) & (spacings < 1.5)]
    if len(small) < 5:
        return 0.0, 0.0

    n_bins = min(25, max(5, len(small) // 3))
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
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def analyze_spectrum(M, riemann_spacings):
    """Full statistical analysis of one operator."""
    eigenvalues = np.linalg.eigvalsh(M)

    # Compute real vs imaginary content
    real_norm = np.linalg.norm(np.real(M))
    imag_norm = np.linalg.norm(np.imag(M))
    total_norm = np.linalg.norm(M)
    real_frac = real_norm / (total_norm + 1e-15)
    imag_frac = imag_norm / (total_norm + 1e-15)

    # Full spectrum analysis
    _, spacings_full = unfold_spectrum(eigenvalues)
    pos_full = spacings_full[spacings_full > 0]

    # Positive wing: eigenvalues above 20th percentile of |lambda|
    abs_ev = np.abs(eigenvalues)
    thresh = np.percentile(abs_ev, 20)
    pos_wing = eigenvalues[eigenvalues > thresh]

    if len(pos_wing) > 30:
        _, spacings_wing = unfold_spectrum(pos_wing)
        pos_sp = spacings_wing[spacings_wing > 0]
    else:
        pos_sp = pos_full

    # KS tests on positive wing (cleanest signal)
    gue_cdf = make_wigner_cdf_func(2)
    goe_cdf = make_wigner_cdf_func(1)

    ks_gue, p_gue = stats.kstest(pos_sp, gue_cdf)
    ks_goe, p_goe = stats.kstest(pos_sp, goe_cdf)
    ks_poi, p_poi = stats.kstest(pos_sp, lambda s: 1 - np.exp(-s))
    ks_riem, p_riem = stats.ks_2samp(pos_sp, riemann_spacings)

    beta, beta_r2 = fit_beta(pos_sp)

    # Also fit beta on full spectrum
    beta_full, _ = fit_beta(pos_full)

    # Spectrum symmetry: how close are eigenvalues to +-pairs?
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)
    if N % 2 == 0:
        pair_asym = np.mean(np.abs(evals_sorted[:N // 2] + evals_sorted[N // 2:][::-1]))
    else:
        pair_asym = np.mean(np.abs(evals_sorted[:N // 2] + evals_sorted[N // 2 + 1:][::-1]))

    return {
        'ks_gue': ks_gue, 'p_gue': p_gue,
        'ks_goe': ks_goe, 'p_goe': p_goe,
        'ks_poi': ks_poi, 'p_poi': p_poi,
        'ks_riem': ks_riem, 'p_riem': p_riem,
        'beta_wing': beta, 'beta_wing_r2': beta_r2,
        'beta_full': beta_full,
        'real_frac': real_frac, 'imag_frac': imag_frac,
        'pair_asymmetry': pair_asym,
        'spacings_wing': pos_sp,
        'eigenvalues': eigenvalues,
        'n_wing': len(pos_sp),
    }


# ============================================================================
# MAIN: PEIERLS FLUX SWEEP
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 76)
    print("  PEIERLS FLUX SWEEP: Time-Reversal Breaking in R/R-bar Operator")
    print("=" * 76)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load Riemann zeros
    zeros_file = RESULTS_DIR / "riemann_zeros_cache.npy"
    if zeros_file.exists():
        riemann_zeros = np.load(zeros_file)
    else:
        from rr_bar_riemann_analysis import RIEMANN_ZEROS_KNOWN
        riemann_zeros = RIEMANN_ZEROS_KNOWN
    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    pos_spacings_riem = spacings_riem[spacings_riem > 0]
    print(f"  Riemann zeros: {len(riemann_zeros)}")

    # Build lattice
    N = 1000
    print(f"  Building Eisenstein lattice N={N}...")
    sites = build_eisenstein_lattice(N)
    U, V = assign_spinors_e6(sites)

    # Plaquette area
    A_plaq = hexagonal_plaquette_area()
    print(f"  Hexagonal plaquette area: {A_plaq:.6f}")
    print(f"  Expected: sqrt(3)/4 = {np.sqrt(3)/4:.6f}")

    # ====================================================================
    # PHASE 1: COARSE SWEEP phi = 0 to pi
    # ====================================================================
    print(f"\n{'='*76}")
    print("  PHASE 1: COARSE SWEEP phi in [0, pi]")
    print(f"{'='*76}")

    # Structural landmarks
    phi_landmarks = {
        0: "bare (no flux)",
        np.pi / 12: "pi/12",
        np.pi / 6: "pi/6 = 2pi/12 (ONE COXETER STEP)",
        np.pi / 4: "pi/4",
        np.pi / 3: "pi/3 = 2pi/6 (one flux quantum / hexagon)",
        5 * np.pi / 12: "5pi/12",
        np.pi / 2: "pi/2",
        2 * np.pi / 3: "2pi/3",
        5 * np.pi / 6: "5pi/6",
        np.pi: "pi",
    }

    phi_coarse = np.linspace(0, np.pi, 25)
    # Insert structural landmarks
    for phi_l in phi_landmarks.keys():
        if not any(abs(phi_coarse - phi_l) < 0.01):
            phi_coarse = np.sort(np.append(phi_coarse, phi_l))

    print(f"  Sweeping {len(phi_coarse)} flux values...")
    print(f"\n  {'phi':>8s} {'phi/pi':>8s} {'beta_w':>8s} {'beta_f':>8s} "
          f"{'KS_GUE':>8s} {'p_GUE':>8s} {'KS_GOE':>8s} {'p_GOE':>8s} "
          f"{'KS_Riem':>8s} {'p_Riem':>8s} {'Re/Im':>10s} {'asym':>8s} {'note'}")
    print("  " + "-" * 120)

    sweep_results = []

    for phi in phi_coarse:
        M = build_M_with_flux(sites, U, V, phi)

        # Verify Hermitian
        assert np.max(np.abs(M - M.conj().T)) < 1e-9

        res = analyze_spectrum(M, pos_spacings_riem)
        res['phi'] = phi
        sweep_results.append(res)

        # Check if this is a landmark
        note = ""
        for phi_l, label in phi_landmarks.items():
            if abs(phi - phi_l) < 0.01:
                note = label
                break

        print(f"  {phi:8.4f} {phi/np.pi:8.4f} {res['beta_wing']:8.3f} {res['beta_full']:8.3f} "
              f"{res['ks_gue']:8.4f} {res['p_gue']:8.4f} {res['ks_goe']:8.4f} {res['p_goe']:8.4f} "
              f"{res['ks_riem']:8.4f} {res['p_riem']:8.4f} "
              f"{res['real_frac']:4.2f}/{res['imag_frac']:4.2f} "
              f"{res['pair_asymmetry']:8.4f} {note}")

    # ====================================================================
    # PHASE 2: FINE SWEEP around best phi
    # ====================================================================
    # Find phi* where KS(GUE) is minimised
    ks_gue_vals = [r['ks_gue'] for r in sweep_results]
    best_idx = np.argmin(ks_gue_vals)
    phi_best_coarse = sweep_results[best_idx]['phi']

    print(f"\n  Coarse optimum: phi* = {phi_best_coarse:.4f} = {phi_best_coarse/np.pi:.4f}*pi")
    print(f"    KS(GUE) = {sweep_results[best_idx]['ks_gue']:.4f}")
    print(f"    beta_wing = {sweep_results[best_idx]['beta_wing']:.3f}")

    # Fine sweep around best
    phi_lo = max(0, phi_best_coarse - np.pi / 6)
    phi_hi = min(np.pi, phi_best_coarse + np.pi / 6)
    phi_fine = np.linspace(phi_lo, phi_hi, 30)

    print(f"\n{'='*76}")
    print(f"  PHASE 2: FINE SWEEP phi in [{phi_lo/np.pi:.3f}*pi, {phi_hi/np.pi:.3f}*pi]")
    print(f"{'='*76}")

    print(f"\n  {'phi':>8s} {'phi/pi':>8s} {'beta_w':>8s} {'KS_GUE':>8s} {'p_GUE':>8s} "
          f"{'KS_Riem':>8s} {'p_Riem':>8s} {'Re/Im':>10s}")
    print("  " + "-" * 80)

    fine_results = []
    for phi in phi_fine:
        M = build_M_with_flux(sites, U, V, phi)
        res = analyze_spectrum(M, pos_spacings_riem)
        res['phi'] = phi
        fine_results.append(res)

        print(f"  {phi:8.4f} {phi/np.pi:8.4f} {res['beta_wing']:8.3f} "
              f"{res['ks_gue']:8.4f} {res['p_gue']:8.4f} "
              f"{res['ks_riem']:8.4f} {res['p_riem']:8.4f} "
              f"{res['real_frac']:4.2f}/{res['imag_frac']:4.2f}")

    # Find fine optimum
    ks_gue_fine = [r['ks_gue'] for r in fine_results]
    best_fine_idx = np.argmin(ks_gue_fine)
    phi_star = fine_results[best_fine_idx]['phi']
    best_res = fine_results[best_fine_idx]

    # ====================================================================
    # PHASE 3: SCALING AT phi*
    # ====================================================================
    print(f"\n{'='*76}")
    print(f"  PHASE 3: SCALING ANALYSIS AT phi* = {phi_star:.4f} = {phi_star/np.pi:.4f}*pi")
    print(f"{'='*76}")

    N_values = [200, 500, 1000, 2000]
    scaling = []

    for Ns in N_values:
        print(f"  N = {Ns}...", end=" ", flush=True)
        sites_s = build_eisenstein_lattice(Ns)
        U_s, V_s = assign_spinors_e6(sites_s)
        M_s = build_M_with_flux(sites_s, U_s, V_s, phi_star)
        res_s = analyze_spectrum(M_s, pos_spacings_riem)
        res_s['N'] = Ns
        scaling.append(res_s)
        print(f"beta={res_s['beta_wing']:.3f} KS(GUE)={res_s['ks_gue']:.4f} "
              f"p(GUE)={res_s['p_gue']:.4f} KS(Riem)={res_s['ks_riem']:.4f}")

    # ====================================================================
    # GENERATE FIGURES
    # ====================================================================
    print(f"\n  Generating figures...")

    all_sweep = sweep_results  # use coarse for main sweep plots

    phi_vals = np.array([r['phi'] for r in all_sweep])
    beta_w_vals = np.array([r['beta_wing'] for r in all_sweep])
    beta_f_vals = np.array([r['beta_full'] for r in all_sweep])
    ks_gue_all = np.array([r['ks_gue'] for r in all_sweep])
    ks_goe_all = np.array([r['ks_goe'] for r in all_sweep])
    ks_riem_all = np.array([r['ks_riem'] for r in all_sweep])
    ks_poi_all = np.array([r['ks_poi'] for r in all_sweep])
    p_gue_all = np.array([r['p_gue'] for r in all_sweep])
    p_riem_all = np.array([r['p_riem'] for r in all_sweep])
    real_frac_all = np.array([r['real_frac'] for r in all_sweep])
    asym_all = np.array([r['pair_asymmetry'] for r in all_sweep])

    # --- Figure 10: Main sweep (3 panels) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Panel 1: beta(phi)
    ax = axes[0]
    ax.plot(phi_vals / np.pi, beta_w_vals, 'bo-', ms=5, lw=2, label='beta (positive wing)')
    ax.plot(phi_vals / np.pi, beta_f_vals, 'cs--', ms=4, lw=1.5, alpha=0.6, label='beta (full)')
    ax.axhline(y=2, color='r', ls='--', lw=2, label='GUE (beta=2)')
    ax.axhline(y=1, color='b', ls=':', lw=1.5, label='GOE (beta=1)')
    ax.axhline(y=4, color='g', ls='--', lw=1.5, alpha=0.5, label='GSE (beta=4)')
    ax.axhline(y=0, color='k', ls=':', lw=1, alpha=0.5, label='Poisson (beta=0)')
    # Mark landmarks
    for phi_l, label in phi_landmarks.items():
        if 'COXETER' in label:
            ax.axvline(x=phi_l / np.pi, color='red', ls='-', alpha=0.4, lw=2)
            ax.text(phi_l / np.pi, ax.get_ylim()[1] * 0.95, 'pi/6', fontsize=10,
                    color='red', ha='center', va='top')
        elif 'flux quantum' in label:
            ax.axvline(x=phi_l / np.pi, color='orange', ls='-', alpha=0.4, lw=2)
            ax.text(phi_l / np.pi, ax.get_ylim()[1] * 0.85, 'pi/3', fontsize=10,
                    color='orange', ha='center', va='top')
    ax.axvline(x=phi_star / np.pi, color='magenta', ls='-', lw=2, alpha=0.7)
    ax.text(phi_star / np.pi + 0.02, 2.5, f'phi*={phi_star/np.pi:.3f}pi', fontsize=10,
            color='magenta')
    ax.set_ylabel('Level repulsion beta', fontsize=13)
    ax.set_title(f'Peierls Flux Sweep: R/R-bar Operator on Eisenstein Lattice (N={N})', fontsize=14)
    ax.legend(fontsize=9, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: KS distances
    ax = axes[1]
    ax.plot(phi_vals / np.pi, ks_gue_all, 'ro-', ms=5, lw=2, label='KS(M, GUE)')
    ax.plot(phi_vals / np.pi, ks_goe_all, 'b^-', ms=5, lw=2, label='KS(M, GOE)')
    ax.plot(phi_vals / np.pi, ks_riem_all, 'g*-', ms=7, lw=2, label='KS(M, Riemann)')
    ax.plot(phi_vals / np.pi, ks_poi_all, 'ks-', ms=4, lw=1.5, alpha=0.5, label='KS(M, Poisson)')
    ax.axhline(y=0.05, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(0.02, 0.06, 'KS=0.05', fontsize=9, color='gray')
    ax.axvline(x=phi_star / np.pi, color='magenta', ls='-', lw=2, alpha=0.7)
    for phi_l, label in phi_landmarks.items():
        if 'COXETER' in label:
            ax.axvline(x=phi_l / np.pi, color='red', ls='-', alpha=0.4, lw=2)
        elif 'flux quantum' in label:
            ax.axvline(x=phi_l / np.pi, color='orange', ls='-', alpha=0.4, lw=2)
    ax.set_ylabel('KS statistic', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: p-values and symmetry breaking
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(phi_vals / np.pi, p_gue_all, 'ro-', ms=5, lw=2, label='p(GUE)')
    ax.plot(phi_vals / np.pi, p_riem_all, 'g*-', ms=7, lw=2, label='p(Riemann)')
    ax.axhline(y=0.05, color='gray', ls='--', lw=1.5, label='p=0.05 threshold')
    ax2.plot(phi_vals / np.pi, real_frac_all, 'c--', ms=3, lw=1.5, alpha=0.6, label='Re fraction')
    ax2.plot(phi_vals / np.pi, asym_all, 'm:', ms=3, lw=1.5, alpha=0.6, label='Pair asymmetry')
    ax.axvline(x=phi_star / np.pi, color='magenta', ls='-', lw=2, alpha=0.7)
    for phi_l, label in phi_landmarks.items():
        if 'COXETER' in label:
            ax.axvline(x=phi_l / np.pi, color='red', ls='-', alpha=0.4, lw=2)
        elif 'flux quantum' in label:
            ax.axvline(x=phi_l / np.pi, color='orange', ls='-', alpha=0.4, lw=2)
    ax.set_xlabel('phi / pi', fontsize=13)
    ax.set_ylabel('p-value', fontsize=13)
    ax2.set_ylabel('Symmetry measures', fontsize=13, color='teal')
    ax.legend(fontsize=10, loc='upper left')
    ax2.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure10_peierls_sweep.png', dpi=150)
    plt.close(fig)
    print("    figure10_peierls_sweep.png saved")

    # --- Figure 11: P(s) at phi* vs GUE vs Riemann ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    from scipy.special import gamma as gamma_fn

    def wigner_pdf(s, beta):
        a = 2.0 * (gamma_fn((beta + 2) / 2)) ** ((beta + 1)) / (gamma_fn((beta + 1) / 2)) ** ((beta + 2))
        b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
        return a * s ** beta * np.exp(-b * s ** 2)

    s_range = np.linspace(0.01, 4.0, 200)

    # Left: P(s) at phi*
    ax = axes[0]
    ax.hist(best_res['spacings_wing'], bins=50, density=True, alpha=0.5,
            color='steelblue', edgecolor='navy', linewidth=0.5,
            label=f'M at phi*={phi_star/np.pi:.3f}pi')
    ax.hist(pos_spacings_riem, bins=50, density=True, alpha=0.3,
            color='coral', edgecolor='darkred', linewidth=0.5, label='Riemann zeros')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE (b=1)')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2.5, label='GUE (b=2)')
    ax.plot(s_range, [wigner_pdf(s, 4) for s in s_range], 'g--', lw=2, label='GSE (b=4)')
    ax.plot(s_range, np.exp(-s_range), 'k:', lw=1.5, label='Poisson')
    ax.set_xlabel('s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.set_title(f'Spacing Distribution at phi* = {phi_star/np.pi:.3f}*pi\n'
                 f'beta={best_res["beta_wing"]:.2f}, KS(GUE)={best_res["ks_gue"]:.3f}, '
                 f'p(GUE)={best_res["p_gue"]:.3f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: Scaling at phi*
    ax = axes[1]
    Ns_sc = [r['N'] for r in scaling]
    ax.plot(Ns_sc, [r['ks_gue'] for r in scaling], 'ro-', ms=8, lw=2, label='KS(GUE)')
    ax.plot(Ns_sc, [r['ks_goe'] for r in scaling], 'b^-', ms=8, lw=2, label='KS(GOE)')
    ax.plot(Ns_sc, [r['ks_riem'] for r in scaling], 'g*-', ms=10, lw=2, label='KS(Riemann)')
    ax.plot(Ns_sc, [r['ks_poi'] for r in scaling], 'ks-', ms=6, lw=1.5, label='KS(Poisson)')
    ax.axhline(y=0.05, color='gray', ls=':', lw=1)
    ax.set_xlabel('N (lattice sites)', fontsize=13)
    ax.set_ylabel('KS statistic', fontsize=13)
    ax.set_title(f'Scaling at phi* = {phi_star/np.pi:.3f}*pi', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure11_phi_star_analysis.png', dpi=150)
    plt.close(fig)
    print("    figure11_phi_star_analysis.png saved")

    # ====================================================================
    # FINAL REPORT
    # ====================================================================
    print(f"\n{'='*76}")
    print("  FINAL REPORT: PEIERLS FLUX SWEEP")
    print(f"{'='*76}")

    print(f"\n  OPTIMAL FLUX:")
    print(f"    phi* = {phi_star:.6f} rad = {phi_star/np.pi:.6f} * pi")
    print(f"    phi* / (pi/6) = {phi_star / (np.pi/6):.4f} (ratio to Coxeter step)")
    print(f"    phi* / (pi/3) = {phi_star / (np.pi/3):.4f} (ratio to flux quantum)")

    # Check if phi* is near structural values
    structural_tests = [
        ("pi/6 (Coxeter step = 2pi/12)", np.pi / 6),
        ("pi/3 (flux quantum = 2pi/6)", np.pi / 3),
        ("pi/4", np.pi / 4),
        ("pi/12", np.pi / 12),
        ("pi/2", np.pi / 2),
        ("2pi/DIM_E6 = 2pi/78", 2 * np.pi / DIM_E6),
        ("2pi/COXETER = 2pi/12", 2 * np.pi / COXETER_H),
        ("2pi/POSITIVE_ROOTS = 2pi/36", 2 * np.pi / 36),
        ("arctan(1/3)", np.arctan(1.0 / 3)),
    ]

    print(f"\n  STRUCTURAL TESTS:")
    for label, phi_test in structural_tests:
        dist = abs(phi_star - phi_test)
        match = "<<<" if dist < 0.05 else ("<<" if dist < 0.1 else ("<" if dist < 0.2 else ""))
        print(f"    |phi* - {label}| = {dist:.4f}  ({dist/np.pi:.4f}*pi)  {match}")

    print(f"\n  STATISTICS AT phi*:")
    print(f"    beta (positive wing) = {best_res['beta_wing']:.3f} (R2={best_res['beta_wing_r2']:.3f})")
    print(f"    KS(M, GUE)  = {best_res['ks_gue']:.4f}  (p = {best_res['p_gue']:.4f})")
    print(f"    KS(M, GOE)  = {best_res['ks_goe']:.4f}  (p = {best_res['p_goe']:.4f})")
    print(f"    KS(M, Poi)  = {best_res['ks_poi']:.4f}  (p = {best_res['p_poi']:.4f})")
    print(f"    KS(M, Riem) = {best_res['ks_riem']:.4f}  (p = {best_res['p_riem']:.4f})")
    print(f"    Re/Im       = {best_res['real_frac']:.3f}/{best_res['imag_frac']:.3f}")

    print(f"\n  SCALING AT phi*:")
    for r in scaling:
        print(f"    N={r['N']:5d}: beta={r['beta_wing']:.3f} "
              f"KS(GUE)={r['ks_gue']:.4f} p(GUE)={r['p_gue']:.4f} "
              f"KS(Riem)={r['ks_riem']:.4f} p(Riem)={r['p_riem']:.4f}")

    # Convergence check
    if len(scaling) >= 2:
        ks_trend = [r['ks_gue'] for r in scaling]
        decreasing = all(ks_trend[i] >= ks_trend[i + 1] - 0.02 for i in range(len(ks_trend) - 1))
        print(f"\n    GUE convergence (decreasing KS): {'YES' if decreasing else 'NO'}")
        print(f"    KS(GUE) trend: {['%.4f' % x for x in ks_trend]}")

    # Classification
    print(f"\n  CLASSIFICATION:")
    if best_res['p_gue'] > 0.05 and best_res['p_riem'] > 0.05:
        print(f"    STRONG POSITIVE: Consistent with GUE AND Riemann zeros")
    elif best_res['p_gue'] > 0.05:
        print(f"    GUE MATCH: Consistent with GUE, differs from Riemann")
    elif best_res['beta_wing'] > 1.5:
        print(f"    APPROACHING GUE: beta > 1.5 but KS not yet converged")
    elif best_res['ks_gue'] < best_res['ks_poi']:
        print(f"    PARTIAL: Closer to GUE than Poisson, level repulsion present")
    else:
        print(f"    NULL: No clear GUE signature")

    # Save report
    with open(RESULTS_DIR / "PEIERLS_SWEEP_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(f"PEIERLS FLUX SWEEP REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"N = {N}\n\n")
        f.write(f"phi* = {phi_star:.6f} = {phi_star/np.pi:.6f}*pi\n")
        f.write(f"beta = {best_res['beta_wing']:.3f}\n")
        f.write(f"KS(GUE) = {best_res['ks_gue']:.4f} (p={best_res['p_gue']:.4f})\n")
        f.write(f"KS(Riemann) = {best_res['ks_riem']:.4f} (p={best_res['p_riem']:.4f})\n\n")
        f.write("COARSE SWEEP:\n")
        f.write(f"{'phi/pi':>8s} {'beta_w':>8s} {'KS_GUE':>8s} {'p_GUE':>8s} {'KS_Riem':>8s}\n")
        for r in sweep_results:
            f.write(f"{r['phi']/np.pi:8.4f} {r['beta_wing']:8.3f} "
                    f"{r['ks_gue']:8.4f} {r['p_gue']:8.4f} {r['ks_riem']:8.4f}\n")
        f.write(f"\nSCALING AT phi*:\n")
        for r in scaling:
            f.write(f"N={r['N']}: beta={r['beta_wing']:.3f} "
                    f"KS(GUE)={r['ks_gue']:.4f} KS(Riem)={r['ks_riem']:.4f}\n")

    elapsed = time.time() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
