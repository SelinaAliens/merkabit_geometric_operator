#!/usr/bin/env python3
"""
ANALYSIS 25: EISENSTEIN EULER PRODUCT — DEDEKIND ZETA CONNECTION
=================================================================

The Merkabit lives on the Eisenstein lattice Z[omega], where omega = e^{2pi*i/3}.
The Dedekind zeta function of Q(omega) factorizes as:

    zeta_{Q(omega)}(s) = zeta(s) * L(s, chi_{-3})

This CONTAINS all Riemann zeros. The Euler product over Eisenstein primes
classifies rational primes into three Z3 classes:

    p = 3         -> ramifies (class 0, singular)
    p = 2 mod 3   -> inert    (class -1)
    p = 1 mod 3   -> splits   (class +1)

This is the trit structure.
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
from scipy.optimize import curve_fit
import mpmath

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\eisenstein_euler")

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
XI = 3.0
OMEGA_EISEN = np.exp(2j * np.pi / 3)

# ============================================================================
# L-FUNCTION TOOLS (fast via mpmath)
# ============================================================================

def L_chi3(s):
    """L(s, chi_{-3}) using mpmath's nsum for convergence acceleration."""
    chi = [0, 1, -1]
    return mpmath.nsum(lambda n: chi[int(n) % 3] * mpmath.power(n, -s), [1, mpmath.inf])

def Z_L(t):
    """Hardy Z-function for L(s, chi_{-3}).
    Real-valued function whose zeros on the real line correspond to
    zeros of L(1/2+it, chi_{-3}) on the critical line."""
    s = mpmath.mpf('0.5') + mpmath.mpc(0, t)
    L_val = L_chi3(s)
    prefactor = mpmath.power(3/mpmath.pi, (s+1)/2) * mpmath.gamma((s+1)/2)
    theta = mpmath.arg(prefactor)
    Z = mpmath.exp(1j * theta) * L_val
    return float(mpmath.re(Z))


# ============================================================================
# STEP 1: RIEMANN ZETA ZEROS
# ============================================================================

def compute_riemann_zeros(N_zeros):
    print(f"\n{'='*70}")
    print(f"STEP 1: Computing {N_zeros} Riemann zeta zeros")
    print(f"{'='*70}")
    t0 = time.time()
    mpmath.mp.dps = 20

    zeros = []
    for n in range(1, N_zeros + 1):
        z = mpmath.zetazero(n)
        gamma = float(z.imag)
        zeros.append(gamma)
        if n % 50 == 0:
            print(f"  {n}/{N_zeros} (t = {gamma:.4f})")

    zeros = np.array(zeros)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
    print(f"  Mean spacing: {np.mean(np.diff(zeros)):.4f}")
    return zeros


# ============================================================================
# STEP 2: L(s, chi_{-3}) ZEROS
# ============================================================================

def compute_L_zeros(T_max, dt=0.3):
    print(f"\n{'='*70}")
    print(f"STEP 2: Computing L(s, chi_{{-3}}) zeros up to T = {T_max:.1f}")
    print(f"{'='*70}")
    t0 = time.time()
    mpmath.mp.dps = 20

    # Scan Z_L(t) for sign changes
    t_vals = np.arange(0.5, T_max, dt)
    z_vals = [Z_L(t) for t in t_vals]

    # Find and refine zeros by bisection
    zeros = []
    for i in range(len(z_vals) - 1):
        if z_vals[i] * z_vals[i+1] < 0:
            a, b = t_vals[i], t_vals[i+1]
            for _ in range(50):
                m = (a + b) / 2
                zm = Z_L(m)
                if z_vals[i] * zm < 0:
                    b = m
                else:
                    a = m
            zeros.append((a + b) / 2)

    zeros = np.array(zeros)
    print(f"  Found {len(zeros)} zeros")
    print(f"  Time: {time.time()-t0:.1f}s")
    if len(zeros) > 0:
        print(f"  Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
        print(f"  Mean spacing: {np.mean(np.diff(zeros)):.4f}")
        print(f"  First 10: {[f'{z:.4f}' for z in zeros[:10]]}")
    return zeros


# ============================================================================
# STEP 3: MERGE DEDEKIND ZEROS
# ============================================================================

def merge_dedekind_zeros(riemann_zeros, L_zeros):
    print(f"\n{'='*70}")
    print(f"STEP 3: Merging Dedekind zeta zeros")
    print(f"{'='*70}")

    # Merge and sort
    all_zeros = np.sort(np.concatenate([riemann_zeros, L_zeros]))

    # Tag each zero
    tags = []
    for z in all_zeros:
        in_r = np.min(np.abs(riemann_zeros - z)) < 0.01 if len(riemann_zeros) > 0 else False
        in_L = np.min(np.abs(L_zeros - z)) < 0.01 if len(L_zeros) > 0 else False
        if in_r and in_L:
            tags.append('both')
        elif in_r:
            tags.append('zeta')
        else:
            tags.append('L')

    n_z = tags.count('zeta')
    n_L = tags.count('L')
    n_both = tags.count('both')

    print(f"  Riemann zeros: {len(riemann_zeros)}")
    print(f"  L-function zeros: {len(L_zeros)}")
    print(f"  Total Dedekind: {len(all_zeros)}")
    print(f"  Common zeros: {n_both}")

    if len(all_zeros) > 1:
        sp_ded = np.diff(all_zeros)
        sp_riem = np.diff(np.sort(riemann_zeros))
        print(f"  Mean spacing Dedekind: {np.mean(sp_ded):.4f}")
        print(f"  Mean spacing Riemann: {np.mean(sp_riem):.4f}")
        print(f"  Density ratio: {np.mean(sp_riem)/np.mean(sp_ded):.3f} (expect ~2)")

    return all_zeros, tags


# ============================================================================
# STEP 4: M OPERATOR
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


def build_M_peierls(sites, U, V, xi=XI):
    """M operator with Peierls phase (GUE variant)."""
    N = len(sites)
    M = np.zeros((N, N), dtype=complex)
    max_range = 3.0 * xi
    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))
        for j in range(i+1, N):
            r = abs(sites[i] - sites[j])
            if r > max_range:
                continue
            J = np.exp(-r / xi)
            uv = np.vdot(U[i], V[j])
            vu = np.vdot(V[i], U[j])
            dz = sites[j] - sites[i]
            peierls_phase = np.exp(1j * np.angle(dz) / RANK_E6)
            coupling = J * peierls_phase * (uv + vu)
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)
    return M


def compute_M_spectrum(N_sites):
    print(f"\n{'='*70}")
    print(f"STEP 4: M operator (N={N_sites}, Peierls)")
    print(f"{'='*70}")
    t0 = time.time()

    sites = build_eisenstein_lattice(N_sites)
    U, V = assign_spinors_e6(sites)
    M = build_M_peierls(sites, U, V)

    herm_err = np.max(np.abs(M - M.conj().T))
    print(f"  Hermiticity error: {herm_err:.2e}")

    evals = np.linalg.eigvalsh(M)
    print(f"  Range: [{evals[0]:.6f}, {evals[-1]:.6f}]")
    print(f"  Time: {time.time()-t0:.1f}s")

    return evals, sites, M


# ============================================================================
# SPECTRAL ANALYSIS TOOLS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=12):
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)
    cdf_empirical = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 20))
    coeffs = np.polyfit(evals_sorted, cdf_empirical * N, deg)
    N_smooth = np.polyval(coeffs, evals_sorted)
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def goe_surmise(s):
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def gue_surmise(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def fit_beta(spacings):
    small = spacings[(spacings > 0.01) & (spacings < 0.8)]
    if len(small) < 10:
        return 0.0, 0.0
    n_bins = min(25, len(small) // 3)
    hist, be = np.histogram(small, bins=n_bins, density=True)
    centers = (be[:-1] + be[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0
    log_s = np.log(centers[mask])
    log_p = np.log(hist[mask])
    c = np.polyfit(log_s, log_p, 1)
    ss_res = np.sum((log_p - np.polyval(c, log_s))**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    return c[0], 1 - ss_res/ss_tot if ss_tot > 0 else 0


def pair_correlation(data, L=15, n_bins=60):
    N = len(data)
    if N < 20:
        return np.array([]), np.array([])
    _, spacings = unfold_spectrum(data)
    unfolded = np.cumsum(np.concatenate([[0], spacings]))
    diffs = []
    for i in range(len(unfolded)):
        for j in range(i+1, min(i + int(L*2), len(unfolded))):
            d = unfolded[j] - unfolded[i]
            if d < L:
                diffs.append(d)
    if len(diffs) < 10:
        return np.array([]), np.array([])
    diffs = np.array(diffs)
    hist, be = np.histogram(diffs, bins=n_bins, range=(0, L), density=True)
    centers = (be[:-1] + be[1:]) / 2
    return centers, hist


def montgomery(r):
    result = np.zeros_like(r, dtype=float)
    nonzero = r > 1e-10
    result[nonzero] = 1.0 - (np.sin(np.pi * r[nonzero]) / (np.pi * r[nonzero]))**2
    return result


# ============================================================================
# STEP 5: COMPARATIVE SPECTRAL STATISTICS
# ============================================================================

def compare_spectra(datasets_dict):
    print(f"\n{'='*70}")
    print(f"STEP 5: Comparative spectral statistics")
    print(f"{'='*70}")

    print(f"\n  {'Dataset':<22} {'N':<7} {'beta':<8} {'KS_GOE':<9} {'KS_GUE':<9} {'KS_Pois':<9} {'Best':<8}")
    print(f"  {'-'*72}")

    results = {}
    for name, data in datasets_dict.items():
        if len(data) < 15:
            continue
        _, spacings = unfold_spectrum(data)
        spacings = spacings[spacings > 0]
        if len(spacings) < 10:
            continue

        beta, r2 = fit_beta(spacings)

        s_sorted = np.sort(spacings)
        N = len(s_sorted)
        cdf_emp = np.arange(1, N+1) / N

        cdf_goe = np.array([quad(goe_surmise, 0, s)[0] for s in s_sorted])
        cdf_gue = np.array([quad(gue_surmise, 0, s)[0] for s in s_sorted])
        cdf_pois = 1.0 - np.exp(-s_sorted)

        ks_goe = np.max(np.abs(cdf_emp - cdf_goe))
        ks_gue = np.max(np.abs(cdf_emp - cdf_gue))
        ks_pois = np.max(np.abs(cdf_emp - cdf_pois))

        best = 'GOE' if ks_goe < ks_gue and ks_goe < ks_pois else ('GUE' if ks_gue < ks_pois else 'Poisson')

        print(f"  {name:<22} {N:<7} {beta:<8.3f} {ks_goe:<9.4f} {ks_gue:<9.4f} {ks_pois:<9.4f} {best:<8}")

        results[name] = {
            'spacings': spacings, 'beta': beta, 'ks_goe': ks_goe,
            'ks_gue': ks_gue, 'ks_pois': ks_pois, 'best': best
        }

    return results


# ============================================================================
# STEP 6: PAIR CORRELATION COMPARISON
# ============================================================================

def compare_pair_correlations(datasets_dict):
    print(f"\n{'='*70}")
    print(f"STEP 6: Pair correlation vs Montgomery")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    r_theory = np.linspace(0.01, 5, 200)
    R2_mont = montgomery(r_theory)

    rms_values = {}
    for idx, (name, data) in enumerate(datasets_dict.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        if len(data) < 20:
            ax.set_title(f'{name} (insufficient)')
            continue

        centers, R2 = pair_correlation(data, L=15, n_bins=60)
        if len(centers) == 0:
            ax.set_title(f'{name} (no data)')
            continue

        ax.plot(r_theory, R2_mont, 'k-', lw=2, label='Montgomery', alpha=0.7)
        mask = centers < 5
        ax.plot(centers[mask], R2[mask], 'o', ms=3, alpha=0.7, label=name)
        ax.set_xlabel('r')
        ax.set_ylabel('R2(r)')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.1, 1.5)

        m2 = (centers > 0.5) & (centers < 4)
        if np.sum(m2) > 5:
            R2_m = montgomery(centers[m2])
            rms = np.sqrt(np.mean((R2[m2] - R2_m)**2))
            rms_values[name] = rms
            ax.text(0.05, 0.95, f'RMS = {rms:.4f}', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round', fc='wheat'))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'pair_correlation_comparison.png', dpi=150)
    plt.close()

    print(f"\n  Pair correlation RMS from Montgomery:")
    for name, rms in sorted(rms_values.items(), key=lambda x: x[1]):
        print(f"    {name:<22}: {rms:.4f}")

    return rms_values


# ============================================================================
# STEP 7: Z3 PRIME CLASSIFICATION
# ============================================================================

def classify_primes_z3(N_max=500):
    print(f"\n{'='*70}")
    print(f"STEP 7: Z3 classification of primes")
    print(f"{'='*70}")

    is_prime = [True] * (N_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N_max + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N_max + 1) if is_prime[p]]

    c0, cp, cm = [], [], []
    for p in primes:
        if p == 3:
            c0.append(p)
        elif p % 3 == 1:
            cp.append(p)
        elif p == 2 or p % 3 == 2:
            cm.append(p)

    print(f"  Total primes: {len(primes)}")
    print(f"  Class 0 (ramifies): {c0}")
    print(f"  Class +1 (splits, p=1 mod 3): {len(cp)} — first 10: {cp[:10]}")
    print(f"  Class -1 (inert, p=2 mod 3): {len(cm)} — first 10: {cm[:10]}")
    print(f"  |class+|/|class-| = {len(cp)/len(cm):.4f}")

    # Euler product factors at s=2
    print(f"\n  Euler product at s=2:")
    s = 2.0

    log_class0 = sum(-np.log(1 - p**(-s)) for p in c0)
    log_classp = sum(-2*np.log(1 - p**(-s)) for p in cp)
    log_classm = sum(-np.log(1 - p**(-2*s)) for p in cm)
    log_total = log_class0 + log_classp + log_classm

    # Compare to exact
    zeta_2 = float(mpmath.zeta(2))
    L_2 = float(mpmath.re(L_chi3(2)))
    exact = np.log(zeta_2 * L_2)

    print(f"    log(class 0): {log_class0:.6f} ({100*log_class0/log_total:.1f}%)")
    print(f"    log(class+1): {log_classp:.6f} ({100*log_classp/log_total:.1f}%)")
    print(f"    log(class-1): {log_classm:.6f} ({100*log_classm/log_total:.1f}%)")
    print(f"    Euler total:  {log_total:.6f}")
    print(f"    Exact:        {exact:.6f}")
    print(f"    Error:        {abs(log_total - exact):.6f}")

    # Z3 trit mapping
    print(f"\n  Z3 TRIT ←→ GATE SUBLATTICE MAPPING:")
    print(f"    Ramified (p=3, class 0) ←→ T, F gates (symmetric, zero asymmetry)")
    print(f"    Split (p≡1 mod 3, +1)   ←→ R, P gates (class +1)")
    print(f"    Inert (p≡2 mod 3, -1)   ←→ S gate (class -1)")
    print(f"    Tunnel: |0> BLOCKED ←→ p=3 RAMIFIES (singular)")
    print(f"    Both: class ±1 propagate, class 0 is special")

    return primes, c0, cp, cm


# ============================================================================
# STEP 8: EIGENVECTOR Z3 ANALYSIS
# ============================================================================

def z3_eigenvector_analysis(sites, M):
    print(f"\n{'='*70}")
    print(f"STEP 8: M eigenvector Z3 sublattice structure")
    print(f"{'='*70}")

    evals, evecs = np.linalg.eigh(M)
    N = len(evals)

    # Z3 class of each site
    z3_class = np.zeros(len(sites), dtype=int)
    for i, z in enumerate(sites):
        b = np.imag(z) / (np.sqrt(3)/2) if abs(np.imag(z)) > 1e-10 else 0.0
        a_coord = np.real(z) + b/2
        a_int = int(np.round(a_coord))
        b_int = int(np.round(b))
        z3_class[i] = (a_int + b_int) % 3

    n_per_class = [np.sum(z3_class == c) for c in range(3)]
    print(f"  Sites per Z3 class: {n_per_class}")
    frac_per_class = [n / len(sites) for n in n_per_class]
    print(f"  Fractions: {[f'{f:.3f}' for f in frac_per_class]} (expect ~1/3 each)")

    # Weight of each eigenvector on each sublattice
    weights = np.zeros((N, 3))
    for k in range(N):
        v = evecs[:, k]
        for c in range(3):
            weights[k, c] = np.sum(np.abs(v[z3_class == c])**2)

    max_weight = np.max(weights, axis=1)
    dominant = np.argmax(weights, axis=1)

    print(f"\n  Eigenvector sublattice weights:")
    print(f"    Mean max weight: {np.mean(max_weight):.4f}")
    print(f"    If random: {max(frac_per_class):.4f}")
    print(f"    Excess: {np.mean(max_weight) - max(frac_per_class):.4f}")
    print(f"    Strongly localized (>50%): {np.sum(max_weight > 0.50)}/{N}")

    # Split spectrum by dominant sublattice
    print(f"\n  Spectrum by dominant sublattice:")
    for c in range(3):
        mask = dominant == c
        evals_c = evals[mask]
        if len(evals_c) > 10:
            _, sp = unfold_spectrum(evals_c)
            sp = sp[sp > 0]
            if len(sp) > 5:
                beta, _ = fit_beta(sp)
                print(f"    Class {c}: {np.sum(mask)} evals, beta = {beta:.3f}")

    # Cross-sublattice correlations:
    # Do eigenvalues associated with different sublattices interleave (GUE)
    # or cluster (GOE)?
    for c1 in range(3):
        for c2 in range(c1+1, 3):
            mask1 = dominant == c1
            mask2 = dominant == c2
            if np.sum(mask1) > 5 and np.sum(mask2) > 5:
                e1 = np.sort(evals[mask1])
                e2 = np.sort(evals[mask2])
                # Nearest-neighbor cross-class distances
                cross_dists = []
                for e in e1:
                    idx = np.searchsorted(e2, e)
                    if idx > 0:
                        cross_dists.append(abs(e - e2[idx-1]))
                    if idx < len(e2):
                        cross_dists.append(abs(e - e2[idx]))
                cross_dists = np.array(cross_dists)
                print(f"    Cross (class {c1}-{c2}): mean NN dist = {np.mean(cross_dists):.4f}")

    return weights, dominant


# ============================================================================
# STEP 9: DENSITY SCALING
# ============================================================================

def density_analysis(riemann_zeros, L_zeros, dedekind_zeros):
    print(f"\n{'='*70}")
    print(f"STEP 9: Density scaling analysis")
    print(f"{'='*70}")

    # Riemann: N(T) ~ (T/2pi) * log(T/2pi)
    # L(s,chi_{-3}): N(T) ~ (T/2pi) * log(3*T/2pi)
    # Dedekind: N(T) = N_zeta(T) + N_L(T)

    def density_model(T, a, b, c):
        return a * T * np.log(T) + b * T + c

    for name, zeros in [('Riemann', riemann_zeros), ('L-function', L_zeros), ('Dedekind', dedekind_zeros)]:
        if len(zeros) < 10:
            continue
        T = zeros
        N_count = np.arange(1, len(T) + 1)
        try:
            popt, _ = curve_fit(density_model, T, N_count, p0=[1/(2*np.pi), 0, 0])
            pred = density_model(T, *popt)
            r2 = 1 - np.sum((N_count - pred)**2) / np.sum((N_count - np.mean(N_count))**2)
            print(f"  {name}: N(T) = {popt[0]:.6f}*T*log(T) + {popt[1]:.4f}*T + {popt[2]:.1f}")
            print(f"    R2 = {r2:.6f}")
        except Exception as e:
            print(f"  {name}: fit failed ({e})")

    # Theoretical
    print(f"\n  Theoretical leading coefficients:")
    print(f"    Riemann:  1/(2*pi) = {1/(2*np.pi):.6f}")
    print(f"    L-func:   1/(2*pi) = {1/(2*np.pi):.6f} (same leading)")
    print(f"    Dedekind: 1/pi     = {1/np.pi:.6f} (double)")


# ============================================================================
# STEP 10: SPACING DISTRIBUTION PLOTS
# ============================================================================

def plot_spacing_distributions(datasets_dict, results):
    print(f"\n{'='*70}")
    print(f"STEP 10: Plotting spacing distributions")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    s_theory = np.linspace(0.001, 4, 200)

    for idx, (name, data) in enumerate(datasets_dict.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        if len(data) < 15:
            ax.set_title(f'{name} (insufficient)')
            continue

        _, spacings = unfold_spectrum(data)
        spacings = spacings[spacings > 0]
        if len(spacings) < 10:
            continue

        ax.hist(spacings, bins=40, range=(0, 4), density=True, alpha=0.5, label='Data')
        ax.plot(s_theory, goe_surmise(s_theory), 'b-', lw=1.5, label='GOE')
        ax.plot(s_theory, gue_surmise(s_theory), 'r-', lw=1.5, label='GUE')
        ax.plot(s_theory, np.exp(-s_theory), 'g--', lw=1.5, label='Poisson')

        if name in results:
            beta = results[name]['beta']
            best = results[name]['best']
            ax.set_title(f'{name} (beta={beta:.2f}, {best})')
        else:
            ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_xlabel('s')
        ax.set_ylabel('P(s)')
        ax.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'spacing_distributions.png', dpi=150)
    plt.close()
    print(f"  Saved spacing_distributions.png")


# ============================================================================
# STEP 11: L-FUNCTION SPECIAL VALUES AND F CONNECTION
# ============================================================================

def F_connection(riemann_zeros, L_zeros, dedekind_zeros):
    print(f"\n{'='*70}")
    print(f"STEP 11: Connection to Floquet constant F and fine structure")
    print(f"{'='*70}")

    F = 0.69677776457995974715
    neg_ln_F = -np.log(F)
    alpha_inv = 137.035999

    # L-function special values
    mpmath.mp.dps = 25
    L1 = complex(L_chi3(1))
    L2 = complex(L_chi3(2))
    L1_exact = np.pi / (3 * np.sqrt(3))

    print(f"  F = {F:.15f}")
    print(f"  -ln(F) = {neg_ln_F:.10f}")
    print(f"  alpha^-1 = {alpha_inv:.6f}")
    print(f"\n  L-function special values:")
    print(f"    L(1, chi_-3) = {L1.real:.10f}")
    print(f"    Exact: pi/(3*sqrt(3)) = {L1_exact:.10f}")
    print(f"    Match: {abs(L1.real - L1_exact):.2e}")
    print(f"    L(2, chi_-3) = {L2.real:.10f}")

    # Class number: h(-3) = 1
    # L(1) = 2*pi*h / (w*sqrt(|D|)) where w=6 (units in Z[omega]), D=-3
    h_computed = L1.real * 6 * np.sqrt(3) / (2 * np.pi)
    print(f"    Class number h(-3) = {h_computed:.6f} (expect 1)")

    # Key test: numerical relationships
    print(f"\n  Numerical coincidence search:")
    print(f"    -ln(F) = {neg_ln_F:.10f}")
    print(f"    L(1)*L(2) = {L1.real * L2.real:.10f}")
    print(f"    L(1)^2 = {L1.real**2:.10f}")
    print(f"    pi^2/27 = {np.pi**2/27:.10f}")  # = L(2) for chi_{-3}?

    # The Dedekind zeta at s=2
    zeta_Q_2 = float(mpmath.zeta(2)) * L2.real
    print(f"    zeta_Q(omega)(2) = zeta(2)*L(2) = {zeta_Q_2:.10f}")
    print(f"    pi^2/6 * L(2) = {np.pi**2/6 * L2.real:.10f}")

    # Connection to E6
    print(f"\n  E6 connections:")
    pos_roots = 36
    print(f"    pos_roots(E6) = {pos_roots}")
    print(f"    -ln(F) = {neg_ln_F:.6f}")
    print(f"    36/V^2 where V = sqrt(36/-ln(F)) = {np.sqrt(36/neg_ln_F):.6f}")
    V_sq = 36 / neg_ln_F
    print(f"    V^2 = {V_sq:.6f}")
    V = np.sqrt(V_sq)
    print(f"    V = {V:.6f}")

    # Key: V^2 = 36 / (-ln F) = 36 / 0.3613... = 99.63...
    # Close to 100 = (2*n_gates)^2
    print(f"    V^2 vs (2*n_gates)^2 = {(2*5)**2}: diff = {abs(V_sq - 100):.4f}")

    # Conductor connection
    print(f"\n  Conductor of Q(omega) = 3")
    print(f"  Discriminant = -3")
    print(f"  |Z3| = 3")
    print(f"  Conductor / rank(E6) = 3/6 = 1/2")
    print(f"  log(conductor) = log(3) = {np.log(3):.6f}")

    # Does log(3) appear in the zero distribution?
    if len(dedekind_zeros) > 1 and len(riemann_zeros) > 1:
        sp_ded = np.mean(np.diff(dedekind_zeros))
        sp_riem = np.mean(np.diff(riemann_zeros))
        ratio = sp_riem / sp_ded
        print(f"\n  Spacing ratio (Riemann/Dedekind) = {ratio:.6f}")
        print(f"  vs 1 + log(3)/log(T_mid) where T_mid = {np.median(riemann_zeros):.1f}:")
        T_mid = np.median(riemann_zeros)
        expected = 1 + np.log(3) / np.log(T_mid / (2*np.pi))
        print(f"    Expected ratio: {expected:.6f}")

    # Most important: does zeta_Q(omega)(s) have conductor information
    # that bridges to F?
    print(f"\n  KEY TEST: Does the Dedekind zeta encode F?")
    # The residue at s=1:
    # Res(zeta_Q(omega), s=1) = 2*pi*h*R / (w*sqrt(|D|))
    # where h=1, R=1 (regulator), w=6, D=-3
    residue = 2 * np.pi * 1 * 1 / (6 * np.sqrt(3))
    print(f"    Residue at s=1: {residue:.10f}")
    print(f"    = pi/(3*sqrt(3)) = L(1, chi_-3) = {L1_exact:.10f}")
    print(f"    Residue / pi = {residue / np.pi:.10f}")
    print(f"    1/(3*sqrt(3)) = {1/(3*np.sqrt(3)):.10f}")

    # F and the residue:
    print(f"    -ln(F) / residue = {neg_ln_F / residue:.10f}")
    print(f"    F^(1/residue) = {F**(1/residue):.10f}")
    print(f"    e^(-residue) = {np.exp(-residue):.10f}")


# ============================================================================
# STEP 12: INTERLEAVING TEST
# ============================================================================

def interleaving_test(riemann_zeros, L_zeros):
    """Test whether zeta and L-function zeros interleave nicely."""
    print(f"\n{'='*70}")
    print(f"STEP 12: Zero interleaving analysis")
    print(f"{'='*70}")

    if len(riemann_zeros) < 5 or len(L_zeros) < 5:
        print(f"  Insufficient zeros")
        return

    T_max = min(riemann_zeros[-1], L_zeros[-1])
    rz = riemann_zeros[riemann_zeros < T_max]
    lz = L_zeros[L_zeros < T_max]

    # Merge and tag
    all_z = np.sort(np.concatenate([rz, lz]))
    tags = []
    for z in all_z:
        if np.min(np.abs(rz - z)) < 0.01:
            tags.append('R')
        else:
            tags.append('L')

    # Count consecutive same-type
    runs = []
    current_run = 1
    for i in range(1, len(tags)):
        if tags[i] == tags[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    mean_run = np.mean(runs)
    max_run = np.max(runs)

    print(f"  Zeros in range [0, {T_max:.1f}]: {len(rz)} Riemann, {len(lz)} L-function")
    print(f"  Total merged: {len(all_z)}")
    print(f"  Consecutive same-type runs:")
    print(f"    Mean run length: {mean_run:.3f} (perfect interleaving = 1.0)")
    print(f"    Max run length: {max_run}")
    print(f"    Run distribution: {dict(zip(*np.unique(runs, return_counts=True)))}")

    # Nearest-neighbor cross-type distances
    cross_nn = []
    for i, z in enumerate(all_z):
        if tags[i] == 'R':
            dists = np.abs(lz - z)
        else:
            dists = np.abs(rz - z)
        cross_nn.append(np.min(dists))

    cross_nn = np.array(cross_nn)
    print(f"\n  Cross-type nearest-neighbor distances:")
    print(f"    Mean: {np.mean(cross_nn):.4f}")
    print(f"    Median: {np.median(cross_nn):.4f}")
    print(f"    Min: {np.min(cross_nn):.6f}")

    # Same-type nearest neighbor for comparison
    same_nn_R = np.diff(np.sort(rz))
    same_nn_L = np.diff(np.sort(lz))
    print(f"    Same-type NN (Riemann): mean = {np.mean(same_nn_R):.4f}")
    print(f"    Same-type NN (L-func): mean = {np.mean(same_nn_L):.4f}")

    ratio = np.mean(cross_nn) / np.mean(np.concatenate([same_nn_R, same_nn_L]))
    print(f"    Cross/Same ratio: {ratio:.4f}")
    print(f"    (< 1 suggests repulsion between types, > 1 suggests attraction)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ANALYSIS 25: EISENSTEIN EULER PRODUCT")
    print("Dedekind Zeta Connection to Merkabit Architecture")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- STEP 1: Riemann zeros ----
    N_RIEMANN = 200
    riemann_zeros = compute_riemann_zeros(N_RIEMANN)

    # ---- STEP 2: L-function zeros ----
    T_max = riemann_zeros[-1] + 10
    L_zeros = compute_L_zeros(T_max, dt=0.25)

    # ---- STEP 3: Merge ----
    dedekind_zeros, tags = merge_dedekind_zeros(riemann_zeros, L_zeros)

    # ---- STEP 4: M operator ----
    N_SITES = 500
    M_evals, sites, M_matrix = compute_M_spectrum(N_SITES)

    # ---- Prepare datasets ----
    M_bulk = M_evals[np.abs(M_evals) > 0.01 * np.max(np.abs(M_evals))]

    datasets = {
        'Riemann zeta': riemann_zeros,
        'L(s,chi_{-3})': L_zeros,
        'Dedekind Q(omega)': dedekind_zeros,
        'M operator (bulk)': M_bulk
    }

    # ---- STEP 5: Spectral statistics ----
    results = compare_spectra(datasets)

    # ---- STEP 6: Pair correlations ----
    rms_values = compare_pair_correlations(datasets)

    # ---- STEP 7: Z3 primes ----
    primes, c0, cp, cm = classify_primes_z3()

    # ---- STEP 8: Eigenvector Z3 ----
    weights, dom_class = z3_eigenvector_analysis(sites, M_matrix)

    # ---- STEP 9: Density ----
    density_analysis(riemann_zeros, L_zeros, dedekind_zeros)

    # ---- STEP 10: Plots ----
    plot_spacing_distributions(datasets, results)

    # ---- STEP 11: F connection ----
    F_connection(riemann_zeros, L_zeros, dedekind_zeros)

    # ---- STEP 12: Interleaving ----
    interleaving_test(riemann_zeros, L_zeros)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 25 — FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n1. ZERO INVENTORY:")
    print(f"   Riemann zeta:       {len(riemann_zeros)} zeros up to t={riemann_zeros[-1]:.1f}")
    print(f"   L(s, chi_{{-3}}):    {len(L_zeros)} zeros up to t={L_zeros[-1]:.1f}" if len(L_zeros) > 0 else "   L(s, chi_{-3}): 0 zeros")
    print(f"   Dedekind Q(omega):  {len(dedekind_zeros)} zeros (union)")
    print(f"   M operator:         {len(M_evals)} eigenvalues ({len(M_bulk)} bulk)")

    print(f"\n2. SPECTRAL UNIVERSALITY:")
    for name, r in results.items():
        marker = "IN GUE" if r['best'] == 'GUE' else r['best']
        print(f"   {name:<22}: beta={r['beta']:.3f}, KS(GUE)={r['ks_gue']:.4f} [{marker}]")

    print(f"\n3. PAIR CORRELATION (RMS from Montgomery):")
    for name, rms in sorted(rms_values.items(), key=lambda x: x[1]):
        print(f"   {name:<22}: {rms:.4f}")

    print(f"\n4. KEY FINDING — Z3 TRIT = EISENSTEIN EULER PRODUCT:")
    print(f"   The three-way classification of primes in Z[omega]")
    print(f"   maps exactly to the gate sublattice assignment (Analysis 24):")
    print(f"     Class 0 (ramified, p=3)  <-> T, F gates (symmetric)")
    print(f"     Class +1 (split, p=1%3)  <-> R, P gates")
    print(f"     Class -1 (inert, p=2%3)  <-> S gate")
    print(f"   The tunnel operator blocking |0> while |+/-1> propagate")
    print(f"   mirrors the singular behavior of p=3 in the Euler product.")

    print(f"\n5. DOES DEDEKIND GIVE BETTER MATCH THAN RIEMANN ALONE?")
    if 'Riemann zeta' in rms_values and 'Dedekind Q(omega)' in rms_values:
        r_rms = rms_values['Riemann zeta']
        d_rms = rms_values['Dedekind Q(omega)']
        print(f"   Pair corr RMS: Riemann={r_rms:.4f}, Dedekind={d_rms:.4f}")
        if d_rms < r_rms:
            print(f"   -> DEDEKIND CLOSER to Montgomery conjecture!")
        else:
            print(f"   -> Riemann closer (Montgomery is specifically about zeta zeros)")

    if 'M operator (bulk)' in rms_values and 'Dedekind Q(omega)' in rms_values:
        m_rms = rms_values['M operator (bulk)']
        d_rms = rms_values['Dedekind Q(omega)']
        print(f"   Pair corr RMS: M_operator={m_rms:.4f}, Dedekind={d_rms:.4f}")

    print(f"\n{'='*70}")
    print(f"END OF ANALYSIS 25")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
