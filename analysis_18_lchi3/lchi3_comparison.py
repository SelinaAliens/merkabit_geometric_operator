#!/usr/bin/env python3
"""
PATH A — L(s, chi_{-3}) ZEROS vs EISENSTEIN M EIGENVALUE SPACINGS
=================================================================

The Lattice Encodes Its Own L-Function

The Eisenstein lattice Z[omega], omega = exp(2pi*i/3), is the ring of integers
of the imaginary quadratic field Q(sqrt(-3)).

The Dedekind zeta function of Q(sqrt(-3)) factorises as:
    zeta_{Q(sqrt(-3))}(s) = zeta(s) * L(s, chi_{-3})

where chi_{-3} is the Kronecker symbol (./−3), a real primitive Dirichlet
character of conductor 3.

This analysis compares:
    M eigenvalue spacings (Eisenstein lattice operator)
    vs
    L(s, chi_{-3}) zero spacings (the field's own L-function)

Steps:
  1. Compute zeros of L(s, chi_{-3}) on the critical line
  2. Verify RMT statistics of chi_{-3} zeros
  3. Build M on Eisenstein lattice (three variants)
  4. Primary KS comparison: M spacings vs chi_{-3} spacings
  5. Pair correlation comparison
  6. Spectral density matching
  7. F-connection via L(1/2, chi_{-3})
"""

import numpy as np
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn, loggamma

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings('ignore', message='.*Polyfit.*poorly conditioned.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\lchi3_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
XI = 3.0
F_RETURN = 0.696778
LN_F = -np.log(F_RETURN)  # 0.36129
OMEGA_EISEN = np.exp(2j * np.pi / 3)

# Eisenstein unit vectors (6-fold coordination)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

RADII = [1, 2, 3, 5, 7, 10]


# ============================================================================
# EISENSTEIN CELL (from lattice_scaling_simulation.py)
# ============================================================================

class EisensteinCell:
    """Eisenstein lattice cell of arbitrary radius."""

    def __init__(self, radius):
        self.radius = radius
        self.r_sq = radius * radius

        self.nodes = []
        for a in range(-radius - 1, radius + 2):
            for b in range(-radius - 1, radius + 2):
                if a * a - a * b + b * b <= self.r_sq:
                    self.nodes.append((a, b))

        self.num_nodes = len(self.nodes)
        self.node_index = {n: i for i, n in enumerate(self.nodes)}

        self.edges = []
        self.neighbours = defaultdict(list)
        node_set = set(self.nodes)

        for i, (a1, b1) in enumerate(self.nodes):
            for da, db in UNIT_VECTORS_AB:
                nb = (a1 + da, b1 + db)
                if nb in node_set:
                    j = self.node_index[nb]
                    if j > i:
                        self.edges.append((i, j))
                    self.neighbours[i].append(j)

        self.is_interior = []
        self.interior_nodes = []
        self.boundary_nodes = []

        for i, (a, b) in enumerate(self.nodes):
            all_nbrs_present = True
            for da, db in UNIT_VECTORS_AB:
                if (a + da, b + db) not in node_set:
                    all_nbrs_present = False
                    break
            self.is_interior.append(all_nbrs_present)
            if all_nbrs_present:
                self.interior_nodes.append(i)
            else:
                self.boundary_nodes.append(i)

        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            if s == 0:
                self.chirality.append(0)
            elif s == 1:
                self.chirality.append(+1)
            else:
                self.chirality.append(-1)


# ============================================================================
# SPINOR ASSIGNMENT
# ============================================================================

def assign_spinors_eisenstein(cell):
    """Assign dual spinors to each node of an EisensteinCell."""
    u_list, v_list, omega_list = [], [], []

    z_coords = [a + b * OMEGA_EISEN for (a, b) in cell.nodes]
    L = max(abs(z) for z in z_coords) if len(z_coords) > 1 else 1.0

    for i, (a, b) in enumerate(cell.nodes):
        z = z_coords[i]
        r = abs(z) / (L + 1e-10)
        theta = np.pi * (a - b) / 6.0

        u = np.exp(1j * theta) * np.array([np.cos(np.pi * r / 2),
                                            1j * np.sin(np.pi * r / 2)],
                                           dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)

        omega = cell.chirality[i] * 1.0
        u_list.append(u)
        v_list.append(v)
        omega_list.append(omega)

    return {'u': u_list, 'v': v_list, 'omega': omega_list}


# ============================================================================
# BUILD M OPERATORS (three variants)
# ============================================================================

def build_M_with_resonance(cell, spinors, xi=XI):
    """Build M WITH resonance condition (bipartite structure)."""
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)

    for (i, j) in cell.edges:
        omega_i = spinors['omega'][i]
        omega_j = spinors['omega'][j]
        resonance = np.exp(-(omega_i + omega_j) ** 2 / 0.1)
        coupling = decay * resonance * np.vdot(spinors['u'][i], spinors['v'][j])
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)

    M = (M + M.conj().T) / 2.0
    return M


def build_M_no_resonance(cell, spinors, xi=XI):
    """Build M WITHOUT resonance — all edges active."""
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)

    for (i, j) in cell.edges:
        coupling = decay * np.vdot(spinors['u'][i], spinors['v'][j])
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)

    M = (M + M.conj().T) / 2.0
    return M


def build_M_peierls(cell, spinors, xi=XI, phi=np.pi/3):
    """Build M with Peierls phase — all edges + gauge field."""
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)

    for (i, j) in cell.edges:
        a_i, b_i = cell.nodes[i]
        a_j, b_j = cell.nodes[j]
        peierls = np.exp(1j * phi * (b_i + b_j) / 2.0 * (a_j - a_i))
        coupling = decay * peierls * np.vdot(spinors['u'][i], spinors['v'][j])
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)

    M = (M + M.conj().T) / 2.0
    return M


# ============================================================================
# SPECTRAL ANALYSIS TOOLS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=10):
    """Unfold eigenvalues using polynomial fit to CDF."""
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


def make_wigner_cdf(beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2

    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for i, si in enumerate(s):
            result[i], _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(si, 0))
        return result
    return cdf


def wigner_pdf(s, beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    return a * s ** beta * np.exp(-b * s ** 2)


def fit_beta(spacings):
    """Fit level repulsion exponent: P(s) ~ s^beta for small s."""
    pos = spacings[spacings > 0.02]
    if len(pos) < 8:
        pos = spacings[spacings > 0.005]
    if len(pos) < 5:
        return 0.0, 0.0

    small = pos[pos < np.percentile(pos, 40)]
    if len(small) < 5:
        small = pos[pos < np.median(pos)]
    if len(small) < 3:
        return 0.0, 0.0

    n_bins = min(20, max(4, len(small) // 3))
    try:
        hist, edges = np.histogram(small, bins=n_bins, density=True)
    except ValueError:
        return 0.0, 0.0
    centres = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0

    log_s = np.log(centres[mask])
    log_p = np.log(hist[mask])
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = coeffs[0]
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def ks_tests(spacings, reference_spacings=None):
    """KS tests against GOE, GUE, Poisson, and optional reference."""
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {k: (1.0, 0.0) for k in ['goe', 'gue', 'poi', 'ref']}

    results = {}
    for name, beta in [('goe', 1), ('gue', 2)]:
        cdf = make_wigner_cdf(beta)
        ks, p = stats.kstest(pos, cdf)
        results[name] = (ks, p)

    ks, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (ks, p)

    if reference_spacings is not None and len(reference_spacings) > 10:
        ks, p = stats.ks_2samp(pos, reference_spacings[reference_spacings > 0])
        results['ref'] = (ks, p)
    else:
        results['ref'] = (1.0, 0.0)

    return results


def extract_positive_wing(eigenvalues, threshold_pct=20):
    """Extract positive wing: eigenvalues above threshold percentile, positive half."""
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    wing = pos_eigs[pos_eigs > cutoff]
    return wing


def analyse_spectrum(eigenvalues, ref_spacings=None, use_wing=True):
    """Full spectral analysis: unfold, KS tests, beta fit."""
    if use_wing:
        wing = extract_positive_wing(eigenvalues)
    else:
        wing = eigenvalues

    if len(wing) < 5:
        return {
            'n_eigs': len(wing), 'beta': 0.0, 'r2': 0.0,
            'ks_goe': 1.0, 'p_goe': 0.0, 'ks_gue': 1.0, 'p_gue': 0.0,
            'ks_poi': 1.0, 'p_poi': 0.0, 'ks_ref': 1.0, 'p_ref': 0.0,
            'spacings': np.array([])
        }

    _, spacings = unfold_spectrum(wing)
    spacings = spacings[spacings > 0]

    if len(spacings) < 4:
        return {
            'n_eigs': len(wing), 'beta': 0.0, 'r2': 0.0,
            'ks_goe': 1.0, 'p_goe': 0.0, 'ks_gue': 1.0, 'p_gue': 0.0,
            'ks_poi': 1.0, 'p_poi': 0.0, 'ks_ref': 1.0, 'p_ref': 0.0,
            'spacings': spacings
        }

    beta, r2 = fit_beta(spacings)
    ks = ks_tests(spacings, ref_spacings)

    return {
        'n_eigs': len(wing),
        'beta': beta, 'r2': r2,
        'ks_goe': ks['goe'][0], 'p_goe': ks['goe'][1],
        'ks_gue': ks['gue'][0], 'p_gue': ks['gue'][1],
        'ks_poi': ks['poi'][0], 'p_poi': ks['poi'][1],
        'ks_ref': ks['ref'][0], 'p_ref': ks['ref'][1],
        'spacings': spacings
    }


# ============================================================================
# L(s, chi_{-3}) COMPUTATION
# ============================================================================

def chi_m3(n):
    """
    Kronecker symbol chi_{-3}(n) = (-3/n).

    For n coprime to 3:
        chi(n) = +1 if n = 1 mod 3
        chi(n) = -1 if n = 2 mod 3
    For 3|n:
        chi(n) = 0
    """
    r = n % 3
    if r == 0:
        return 0
    if r == 1:
        return 1
    return -1  # r == 2


def theta_chi3(t):
    """
    Phase function Theta(t) for Hardy Z-function of L(s, chi_{-3}).

    For odd character (a=1), conductor q=3:
        Theta(t) = (t/2) log(3/pi) + Im(log Gamma(3/4 + it/2))

    Z(t) = exp(i*Theta(t)) * L(1/2 + it, chi_{-3})  is REAL.
    """
    lg = loggamma(0.75 + 0.5j * t)
    return t / 2 * np.log(3.0 / np.pi) + np.imag(lg)


def weyl_count_chi3(T):
    """
    Smooth zero counting function N_0(T) for L(s, chi_{-3}).

    N_0(T) = (1/pi) * Theta(T) + 1

    Asymptotically: N_0(T) ~ (T / 2pi) * log(3T / 2pi*e)
    """
    return theta_chi3(T) / np.pi + 1.0


def compute_L_chi3_on_critical_line(t_array, N_terms=30000):
    """
    Compute L(1/2 + it, chi_{-3}) for an array of t values.

    Uses direct partial sum with numpy vectorization over n.
    Error: O(N^{-1/2}) ~ 0.006 for N=30000.
    """
    n = np.arange(1, N_terms + 1)
    chi = np.zeros(N_terms)
    for k in range(N_terms):
        nk = k + 1
        r = nk % 3
        if r == 1:
            chi[k] = 1.0
        elif r == 2:
            chi[k] = -1.0
        # else: 0 (multiples of 3)

    # Only keep nonzero chi values
    mask = chi != 0
    n_nz = n[mask]
    chi_nz = chi[mask]
    n_inv_sqrt = 1.0 / np.sqrt(n_nz)
    log_n_nz = np.log(n_nz)
    weights = chi_nz * n_inv_sqrt

    L_vals = np.zeros(len(t_array), dtype=complex)
    for idx, t in enumerate(t_array):
        phases = np.exp(-1j * t * log_n_nz)
        L_vals[idx] = np.sum(weights * phases)

    return L_vals


def compute_Z_chi3(t_array, N_terms=30000):
    """
    Compute Hardy Z-function Z(t) = Re(exp(i*Theta(t)) * L(1/2+it, chi_{-3})).

    Z(t) is real; its sign changes locate the zeros.
    """
    L_vals = compute_L_chi3_on_critical_line(t_array, N_terms)
    theta_vals = np.array([theta_chi3(t) for t in t_array])
    Z_vals = np.real(np.exp(1j * theta_vals) * L_vals)
    return Z_vals


def find_zeros_chi3(n_zeros=500, T_max=700.0, scan_step=0.03, N_terms=30000):
    """
    Find zeros of L(s, chi_{-3}) on the critical line.

    Method:
    1. Scan Z(t) for sign changes on grid
    2. Refine each zero with bisection using higher N_terms
    """
    print(f"    Scanning Z(t) for t in [1, {T_max}], step={scan_step}...")
    t_scan = np.arange(1.0, T_max, scan_step)

    # Compute Z in chunks to manage memory
    chunk_size = 2000
    Z_scan = np.zeros(len(t_scan))
    for start in range(0, len(t_scan), chunk_size):
        end = min(start + chunk_size, len(t_scan))
        Z_scan[start:end] = compute_Z_chi3(t_scan[start:end], N_terms)
        if start % 10000 == 0 and start > 0:
            print(f"      ... scanned {start}/{len(t_scan)} points")

    # Find sign changes
    sign_changes = []
    for i in range(len(Z_scan) - 1):
        if Z_scan[i] * Z_scan[i + 1] < 0:
            sign_changes.append((t_scan[i], t_scan[i + 1]))
        if len(sign_changes) >= n_zeros + 50:  # extra buffer
            break

    print(f"    Found {len(sign_changes)} sign changes")

    # Refine with bisection (using more terms for accuracy)
    N_refine = min(50000, N_terms * 2)
    zeros = []
    for t_lo, t_hi in sign_changes:
        for _ in range(40):  # 40 bisection steps -> ~1e-12 precision
            t_mid = (t_lo + t_hi) / 2
            Z_lo = compute_Z_chi3(np.array([t_lo]), N_refine)[0]
            Z_mid = compute_Z_chi3(np.array([t_mid]), N_refine)[0]
            if Z_lo * Z_mid < 0:
                t_hi = t_mid
            else:
                t_lo = t_mid
        zeros.append((t_lo + t_hi) / 2)
        if len(zeros) >= n_zeros:
            break

    zeros = np.array(zeros[:n_zeros])
    print(f"    Refined {len(zeros)} zeros")
    return zeros


def unfold_lchi3_zeros(zeros):
    """
    Unfold L(s, chi_{-3}) zeros using the smooth Weyl counting function.

    N_smooth(gamma_n) should give approximately n.
    Spacings s_n = N_smooth(gamma_{n+1}) - N_smooth(gamma_n) have mean ~1.
    """
    N_smooth = np.array([weyl_count_chi3(t) for t in zeros])
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


# ============================================================================
# RIEMANN ZEROS (for comparison)
# ============================================================================

def unfold_riemann_zeros(zeros):
    """Unfold Riemann zeta zeros using standard Weyl law."""
    T = np.sort(zeros)
    N_smooth = (T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7.0 / 8
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


# ============================================================================
# PAIR CORRELATION
# ============================================================================

def pair_correlation(unfolded_levels, r_max=3.0, n_bins=60):
    """
    Compute pair correlation function g(r) from unfolded levels.

    g(r) measures the density of level pairs at separation r.
    For Poisson: g(r) = 1 (flat).
    For GUE: g(r) = 1 - sinc^2(pi*r).
    For GOE: g(r) = 1 - sinc^2(pi*r) - correction.
    """
    N = len(unfolded_levels)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    dr = r_edges[1] - r_edges[0]
    hist = np.zeros(n_bins)

    levels_sorted = np.sort(unfolded_levels)

    for i in range(N):
        for j in range(i + 1, min(i + 30, N)):  # consider up to 30th neighbor
            r = levels_sorted[j] - levels_sorted[i]
            if r > r_max:
                break
            bin_idx = int(r / dr)
            if 0 <= bin_idx < n_bins:
                hist[bin_idx] += 1

    # Normalization: each bin should have N*dr pairs for Poisson density 1
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    g = hist / (N * dr) if N * dr > 0 else hist

    return r_centers, g


def pair_corr_gue(r):
    """GUE pair correlation: 1 - sinc^2(pi*r)"""
    return 1 - np.sinc(r)**2


def pair_corr_goe(r):
    """
    GOE pair correlation (Wigner surmise approximation):
    Uses the 2x2 GOE result as an approximation.
    Exact GOE has: R_2(r) = 1 - s(r)^2 - s'(r) * integral, but we use simplified.
    """
    # For GOE, the pair correlation has WEAKER repulsion than GUE
    # Approximate: interpolate between Poisson and GUE
    # Exact: involves sine integral, but for visualization the Wigner surmise is fine
    sinc_val = np.sinc(r)
    return 1 - sinc_val**2 + 0.0  # Simplified; exact GOE is more complex


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 76)
    print("  PATH A: L(s, chi_{-3}) ZEROS vs EISENSTEIN M EIGENVALUE SPACINGS")
    print("  The Lattice Encodes Its Own L-Function")
    print("=" * 76)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Architecture: E6 (rank={RANK_E6}, dim={DIM_E6}, h={COXETER_H})")
    print(f"  Field: Q(sqrt(-3)), ring of integers = Z[omega]")
    print(f"  Character: chi_{{-3}} = Kronecker (./−3), conductor 3, REAL")
    print(f"  Prediction: L(s, chi_{{-3}}) zeros -> GOE (beta=1)")
    print()

    # ==================================================================
    # STEP 1: ACQUIRE L(s, chi_{-3}) ZEROS
    # ==================================================================
    print(f"{'='*76}")
    print("  STEP 1: COMPUTING ZEROS OF L(s, chi_{{-3}})")
    print(f"{'='*76}\n")

    cache_file = RESULTS_DIR / "lchi3_zeros.npy"

    if cache_file.exists():
        zeros_chi3 = np.load(cache_file)
        print(f"  Loaded {len(zeros_chi3)} cached zeros from {cache_file.name}")
    else:
        print("  Computing zeros from scratch...")
        print("  Method: Hardy Z-function, partial sum N=30000, bisection refinement")
        print()

        zeros_chi3 = find_zeros_chi3(n_zeros=500, T_max=750.0,
                                      scan_step=0.025, N_terms=30000)
        np.save(cache_file, zeros_chi3)
        print(f"\n  Saved {len(zeros_chi3)} zeros to {cache_file.name}")

    print(f"\n  First 10 zeros of L(s, chi_{{-3}}):")
    for i, z in enumerate(zeros_chi3[:10]):
        print(f"    gamma_{i+1:3d} = {z:.8f}")

    print(f"\n  Total zeros: {len(zeros_chi3)}")
    print(f"  Range: [{zeros_chi3[0]:.4f}, {zeros_chi3[-1]:.4f}]")
    print(f"  Expected count at T={zeros_chi3[-1]:.0f}: "
          f"N_0 = {weyl_count_chi3(zeros_chi3[-1]):.1f}")

    # Verify Weyl law
    actual_count = np.arange(1, len(zeros_chi3) + 1)
    predicted = np.array([weyl_count_chi3(t) for t in zeros_chi3])
    weyl_error = np.max(np.abs(actual_count - predicted))
    print(f"  Max Weyl deviation: {weyl_error:.2f}")

    # ==================================================================
    # STEP 2: VERIFY RMT STATISTICS OF chi_{-3} ZEROS
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 2: RMT STATISTICS OF L(s, chi_{{-3}}) ZEROS")
    print(f"{'='*76}\n")

    N_smooth_chi3, spacings_chi3 = unfold_lchi3_zeros(zeros_chi3)
    sp_chi3 = spacings_chi3[spacings_chi3 > 0]

    print(f"  Unfolded spacings: {len(sp_chi3)} (from {len(zeros_chi3)} zeros)")
    print(f"  Mean spacing: {np.mean(sp_chi3):.6f} (should be ~1.0)")
    print(f"  Std spacing:  {np.std(sp_chi3):.6f}")

    beta_chi3, r2_chi3 = fit_beta(sp_chi3)
    print(f"\n  Level repulsion: beta = {beta_chi3:.4f} (R2 = {r2_chi3:.4f})")
    print(f"    GOE: beta=1, GUE: beta=2, Poisson: beta=0")

    ks_chi3 = ks_tests(sp_chi3)
    print(f"\n  KS tests:")
    print(f"    KS(GOE) = {ks_chi3['goe'][0]:.4f}  p = {ks_chi3['goe'][1]:.4f}  "
          f"{'<-- MATCH' if ks_chi3['goe'][1] > 0.05 else ''}")
    print(f"    KS(GUE) = {ks_chi3['gue'][0]:.4f}  p = {ks_chi3['gue'][1]:.4f}  "
          f"{'<-- MATCH' if ks_chi3['gue'][1] > 0.05 else ''}")
    print(f"    KS(Poi) = {ks_chi3['poi'][0]:.4f}  p = {ks_chi3['poi'][1]:.4f}")

    # Also load Riemann zeros for comparison
    riemann_file = Path(r"C:\Users\selin\merkabit_results\riemann_zeros\riemann_zeros_cache.npy")
    if riemann_file.exists():
        riemann_zeros = np.load(riemann_file)
        _, sp_riem_full = unfold_riemann_zeros(riemann_zeros)
        sp_riem = sp_riem_full[sp_riem_full > 0]
        print(f"\n  Riemann zeros loaded: {len(riemann_zeros)}")

        ks_chi3_vs_riem = stats.ks_2samp(sp_chi3, sp_riem[:len(sp_chi3)])
        print(f"  KS(chi_{{-3}}, Riemann) = {ks_chi3_vs_riem.statistic:.4f}  "
              f"p = {ks_chi3_vs_riem.pvalue:.4f}")
    else:
        riemann_zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876,
                                  32.935062, 37.586178, 40.918719, 43.327073])
        _, sp_riem_full = unfold_riemann_zeros(riemann_zeros)
        sp_riem = sp_riem_full[sp_riem_full > 0]
        print(f"\n  Using fallback Riemann zeros: {len(riemann_zeros)}")

    # Determine which universality class chi_{-3} zeros belong to
    if ks_chi3['goe'][1] > 0.05 and ks_chi3['gue'][1] < 0.05:
        chi3_class = "GOE"
    elif ks_chi3['gue'][1] > 0.05 and ks_chi3['goe'][1] < 0.05:
        chi3_class = "GUE"
    elif ks_chi3['goe'][1] > 0.05 and ks_chi3['gue'][1] > 0.05:
        chi3_class = "BOTH_CONSISTENT"
    else:
        chi3_class = "NEITHER"
    print(f"\n  VERDICT: chi_{{-3}} zeros are {chi3_class}-like")
    print(f"  (Katz-Sarnak: individual L-functions -> GUE in bulk)")
    print(f"  (Real character family statistics -> orthogonal at low-lying)")

    # HIGH-ZERO SUBSET: use only zeros with gamma > 100 (deeper into bulk)
    high_mask = zeros_chi3 > 100
    if np.sum(high_mask) > 50:
        zeros_high = zeros_chi3[high_mask]
        _, sp_high = unfold_lchi3_zeros(zeros_high)
        sp_high = sp_high[sp_high > 0]
        beta_high, _ = fit_beta(sp_high)
        ks_high = ks_tests(sp_high)
        print(f"\n  HIGH-ZERO SUBSET (gamma > 100, N={len(zeros_high)}):")
        print(f"    beta = {beta_high:.4f}")
        print(f"    KS(GOE) = {ks_high['goe'][0]:.4f}  p = {ks_high['goe'][1]:.4f}")
        print(f"    KS(GUE) = {ks_high['gue'][0]:.4f}  p = {ks_high['gue'][1]:.4f}")
        if ks_high['gue'][1] > 0.05:
            print(f"    --> GUE CONSISTENT in bulk (Katz-Sarnak confirmed)")
        elif ks_high['goe'][1] > 0.05:
            print(f"    --> GOE CONSISTENT in bulk")

    # Save chi3 spacing data
    np.save(RESULTS_DIR / "lchi3_spacings_unfolded.npy", sp_chi3)

    with open(RESULTS_DIR / "lchi3_goe_verification.txt", 'w') as f:
        f.write("L(s, chi_{-3}) Zero Spacing Statistics\n")
        f.write(f"N zeros: {len(zeros_chi3)}\n")
        f.write(f"N spacings: {len(sp_chi3)}\n")
        f.write(f"Beta: {beta_chi3:.6f} (R2={r2_chi3:.4f})\n")
        f.write(f"KS(GOE): {ks_chi3['goe'][0]:.6f} p={ks_chi3['goe'][1]:.6f}\n")
        f.write(f"KS(GUE): {ks_chi3['gue'][0]:.6f} p={ks_chi3['gue'][1]:.6f}\n")
        f.write(f"KS(Poi): {ks_chi3['poi'][0]:.6f} p={ks_chi3['poi'][1]:.6f}\n")
        f.write(f"Class: {chi3_class}\n")

    # ==================================================================
    # STEP 3: BUILD M ON EISENSTEIN LATTICE (ALL VARIANTS)
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 3: M OPERATOR ON EISENSTEIN LATTICE — ALL VARIANTS")
    print(f"{'='*76}\n")

    # Store all M results
    M_results = {}  # key: (radius, variant)

    for variant_name, build_fn in [
        ("resonance", build_M_with_resonance),
        ("no_resonance", build_M_no_resonance),
        ("peierls", build_M_peierls)
    ]:
        print(f"\n  Variant: {variant_name}")
        print(f"  {'Radius':>6} {'N':>5} {'N_wing':>7} {'beta':>7} "
              f"{'KS(GOE)':>8} {'KS(GUE)':>8} {'KS(chi3)':>9} "
              f"{'p(chi3)':>8} {'KS(Riem)':>9}")
        print("  " + "-" * 85)

        for radius in RADII:
            cell = EisensteinCell(radius)
            spinors = assign_spinors_eisenstein(cell)

            M = build_fn(cell, spinors)
            eigenvalues = np.linalg.eigvalsh(M)

            # Analyse with chi3 spacings as reference
            res = analyse_spectrum(eigenvalues, sp_chi3)

            # Also test against Riemann
            ks_riem_test = ks_tests(res['spacings'], sp_riem) if len(res['spacings']) > 4 else {'ref': (1.0, 0.0)}

            M_results[(radius, variant_name)] = {
                'eigenvalues': eigenvalues,
                'spacings': res['spacings'],
                'beta': res['beta'], 'r2': res['r2'],
                'ks_goe': res['ks_goe'], 'p_goe': res['p_goe'],
                'ks_gue': res['ks_gue'], 'p_gue': res['p_gue'],
                'ks_chi3': res['ks_ref'], 'p_chi3': res['p_ref'],
                'ks_riem': ks_riem_test['ref'][0], 'p_riem': ks_riem_test['ref'][1],
                'n_eigs': res['n_eigs'],
                'n_total': cell.num_nodes
            }

            r = M_results[(radius, variant_name)]
            print(f"  {radius:6d} {r['n_total']:5d} {r['n_eigs']:7d} "
                  f"{r['beta']:7.3f} {r['ks_goe']:8.4f} {r['ks_gue']:8.4f} "
                  f"{r['ks_chi3']:9.4f} {r['p_chi3']:8.4f} {r['ks_riem']:9.4f}")

    # ==================================================================
    # STEP 4: PRIMARY COMPARISON — M vs chi_{-3} vs GOE
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 4: PRIMARY COMPARISON TABLE")
    print("  Does M specifically match L(s, chi_{{-3}}) better than generic GOE?")
    print(f"{'='*76}\n")

    print(f"  {'Analysis':>20} | {'KS(chi3)':>9} {'p':>7} | {'KS(GOE)':>8} {'p':>7} | "
          f"{'KS(Riem)':>9} {'p':>7} | {'Verdict':>12}")
    print("  " + "-" * 95)

    best_results = {}
    for variant in ["resonance", "no_resonance", "peierls"]:
        for radius in RADII:
            key = (radius, variant)
            if key not in M_results:
                continue
            r = M_results[key]
            if r['n_eigs'] < 10:
                continue

            label = f"r={radius} {variant[:6]}"

            # Verdict
            if r['ks_chi3'] < r['ks_goe']:
                verdict = "chi3 BETTER"
            elif abs(r['ks_chi3'] - r['ks_goe']) < 0.02:
                verdict = "SIMILAR"
            else:
                verdict = "GOE better"

            print(f"  {label:>20} | {r['ks_chi3']:9.4f} {r['p_chi3']:7.4f} | "
                  f"{r['ks_goe']:8.4f} {r['p_goe']:7.4f} | "
                  f"{r['ks_riem']:9.4f} {r['p_riem']:7.4f} | {verdict:>12}")

            best_results[key] = r

    # Highlight best M result
    best_key = min(best_results.keys(),
                   key=lambda k: best_results[k]['ks_chi3'] if best_results[k]['n_eigs'] > 10 else 999)
    br = best_results[best_key]
    print(f"\n  BEST M result: {best_key}")
    print(f"    KS(M, chi_{{-3}}) = {br['ks_chi3']:.4f}  p = {br['p_chi3']:.4f}")
    print(f"    KS(M, GOE)      = {br['ks_goe']:.4f}  p = {br['p_goe']:.4f}")
    print(f"    KS(M, Riemann)  = {br['ks_riem']:.4f}  p = {br['p_riem']:.4f}")

    # Decision
    if br['ks_chi3'] < br['ks_goe'] and br['p_chi3'] > 0.05:
        decision = "STRONG: M specifically matches L(s, chi_{-3})"
    elif br['p_chi3'] > 0.05:
        decision = "POSITIVE: Cannot reject M ~ L(s, chi_{-3})"
    elif abs(br['ks_chi3'] - br['ks_goe']) < 0.03:
        decision = "NEUTRAL: M is GOE-class, not specifically chi_{-3}"
    else:
        decision = "NEGATIVE: M does not match L(s, chi_{-3})"
    print(f"\n  DECISION: {decision}")

    # Save comparison table
    with open(RESULTS_DIR / "M_vs_lchi3_ks_table.txt", 'w') as f:
        f.write("M EIGENVALUE SPACINGS vs L(s, chi_{-3}) ZERO SPACINGS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Analysis':>20} KS(chi3) p(chi3) KS(GOE) p(GOE) KS(Riem) p(Riem)\n")
        f.write("-" * 80 + "\n")
        for key in sorted(best_results.keys()):
            r = best_results[key]
            label = f"r={key[0]} {key[1][:6]}"
            f.write(f"{label:>20} {r['ks_chi3']:.5f} {r['p_chi3']:.5f} "
                    f"{r['ks_goe']:.5f} {r['p_goe']:.5f} "
                    f"{r['ks_riem']:.5f} {r['p_riem']:.5f}\n")
        f.write(f"\nBest: {best_key}\n")
        f.write(f"Decision: {decision}\n")

    # ==================================================================
    # STEP 5: PAIR CORRELATION COMPARISON
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 5: PAIR CORRELATION g(r)")
    print(f"{'='*76}\n")

    # Compute pair correlation for chi_{-3} zeros
    r_pc, g_chi3 = pair_correlation(N_smooth_chi3, r_max=3.0, n_bins=60)
    g_gue_theory = pair_corr_gue(r_pc)
    g_goe_theory = pair_corr_goe(r_pc)

    # Pair correlation for best M
    best_eigs = M_results[best_key]['eigenvalues']
    best_wing = extract_positive_wing(best_eigs)
    if len(best_wing) > 10:
        N_sm_M, _ = unfold_spectrum(best_wing)
        r_pc_M, g_M = pair_correlation(N_sm_M, r_max=3.0, n_bins=60)
    else:
        r_pc_M, g_M = r_pc, np.ones_like(r_pc)

    # Also for no-resonance r=10
    nores_eigs = M_results[(10, 'no_resonance')]['eigenvalues']
    nores_wing = extract_positive_wing(nores_eigs)
    if len(nores_wing) > 10:
        N_sm_nr, _ = unfold_spectrum(nores_wing)
        r_pc_nr, g_nr = pair_correlation(N_sm_nr, r_max=3.0, n_bins=60)
    else:
        r_pc_nr, g_nr = r_pc, np.ones_like(r_pc)

    # Compare g(r) values at r=1 (expected: GUE -> 1-sinc^2(pi) = 1,
    # GOE slightly different)
    idx_r1 = np.argmin(np.abs(r_pc - 1.0))
    print(f"  Pair correlation at r = {r_pc[idx_r1]:.2f}:")
    print(f"    chi_{{-3}} zeros: g = {g_chi3[idx_r1]:.4f}")
    print(f"    GUE theory:     g = {g_gue_theory[idx_r1]:.4f}")
    print(f"    GOE theory:     g = {g_goe_theory[idx_r1]:.4f}")
    if len(best_wing) > 10:
        print(f"    M (best):       g = {g_M[idx_r1]:.4f}")
    if len(nores_wing) > 10:
        print(f"    M (no-res r10): g = {g_nr[idx_r1]:.4f}")

    # Save
    with open(RESULTS_DIR / "pair_correlation_comparison.txt", 'w') as f:
        f.write("Pair Correlation g(r)\n")
        f.write(f"{'r':>8} {'chi3':>8} {'GUE':>8} {'GOE':>8} {'M_best':>8} {'M_nr10':>8}\n")
        for i in range(len(r_pc)):
            f.write(f"{r_pc[i]:8.4f} {g_chi3[i]:8.4f} {g_gue_theory[i]:8.4f} "
                    f"{g_goe_theory[i]:8.4f}")
            if len(g_M) > i:
                f.write(f" {g_M[i]:8.4f}")
            if len(g_nr) > i:
                f.write(f" {g_nr[i]:8.4f}")
            f.write("\n")

    # ==================================================================
    # STEP 6: SPECTRAL DENSITY MATCHING
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 6: SPECTRAL DENSITY MATCHING")
    print(f"{'='*76}\n")

    print("  Does M eigenvalue density match L(s, chi_{{-3}}) zero density?")
    print()
    print(f"  {'Variant':>12} {'Radius':>6} {'N_pos':>5} {'range':>8} {'density_M':>10} "
          f"{'T_eff':>8} {'density_chi3(T)':>15}")
    print("  " + "-" * 75)

    density_data = []
    for variant in ["no_resonance", "peierls", "resonance"]:
        for radius in RADII:
            key = (radius, variant)
            if key not in M_results:
                continue
            eigs = M_results[key]['eigenvalues']
            pos_eigs = eigs[eigs > 1e-8]
            if len(pos_eigs) < 3:
                continue

            spectral_range = pos_eigs.max() - pos_eigs.min()
            if spectral_range < 1e-10:
                continue
            density_M = len(pos_eigs) / spectral_range

            # What effective T makes L(s,chi_{-3}) have the same density?
            # Density of chi3 zeros at height T: d(T) = (1/2pi) * log(3T/(2pi*e))
            # Solve: d(T) = density_M
            # T = (2pi/3) * exp(2*pi*density_M + 1)  [approximate]
            try:
                T_eff = (2 * np.pi / 3) * np.exp(2 * np.pi * density_M + 1)
                if T_eff > 1e15:
                    T_eff = np.inf
            except OverflowError:
                T_eff = np.inf

            if np.isfinite(T_eff) and T_eff > 1:
                density_chi3_at_T = np.log(3 * T_eff / (2 * np.pi * np.e)) / (2 * np.pi)
            else:
                density_chi3_at_T = np.inf

            density_data.append({
                'variant': variant, 'radius': radius,
                'n_pos': len(pos_eigs), 'range': spectral_range,
                'density_M': density_M, 'T_eff': T_eff,
                'density_chi3': density_chi3_at_T
            })

            T_str = f"{T_eff:.1f}" if np.isfinite(T_eff) and T_eff < 1e10 else "inf"
            d_str = f"{density_chi3_at_T:.4f}" if np.isfinite(density_chi3_at_T) else "inf"
            print(f"  {variant[:12]:>12} {radius:6d} {len(pos_eigs):5d} "
                  f"{spectral_range:8.4f} {density_M:10.4f} "
                  f"{T_str:>8} {d_str:>15}")

    # Save density data
    with open(RESULTS_DIR / "density_scaling.txt", 'w') as f:
        f.write("Spectral Density Matching\n")
        f.write("variant radius n_pos range density_M T_eff density_chi3\n")
        for d in density_data:
            f.write(f"{d['variant']} {d['radius']} {d['n_pos']} "
                    f"{d['range']:.6f} {d['density_M']:.6f} "
                    f"{d['T_eff']:.2f} {d['density_chi3']:.6f}\n")

    # ==================================================================
    # STEP 7: F-CONNECTION VIA L(1/2, chi_{-3})
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 7: F-CONNECTION VIA L(1/2, chi_{{-3}})")
    print(f"{'='*76}\n")

    # Compute L(1/2, chi_{-3}) — the central value
    # Using the Chowla-Selberg formula:
    #   L(1, chi_{-3}) = pi / (3*sqrt(3))
    # For L(1/2, chi_{-3}), use partial sum with many terms
    print("  Computing L(1/2, chi_{{-3}}) via partial sum...")
    N_terms_central = 100000
    n = np.arange(1, N_terms_central + 1)
    chi_vals = np.zeros(N_terms_central)
    for k in range(N_terms_central):
        nk = k + 1
        r = nk % 3
        if r == 1:
            chi_vals[k] = 1.0
        elif r == 2:
            chi_vals[k] = -1.0

    L_half = np.sum(chi_vals / np.sqrt(n))
    # Richardson extrapolation: L_N - (L_N - L_{N/2}) * 2 / (2-1)
    # For better accuracy, also compute with N/2
    half_N = N_terms_central // 2
    L_half_2 = np.sum(chi_vals[:half_N] / np.sqrt(n[:half_N]))
    L_half_richardson = 2 * L_half - L_half_2  # Richardson extrapolation

    # Also compute L(1, chi_{-3}) exactly for validation
    L_one_exact = np.pi / (3 * np.sqrt(3))

    # Try mpmath for high precision if available
    try:
        import mpmath
        mpmath.mp.dps = 30
        s_half = mpmath.mpf('0.5')
        h1 = mpmath.hurwitz(s_half, mpmath.mpf(1) / 3)
        h2 = mpmath.hurwitz(s_half, mpmath.mpf(2) / 3)
        L_half_exact = float(mpmath.power(3, -s_half) * (h1 - h2))
        print(f"  L(1/2, chi_{{-3}}) = {L_half_exact:.10f}  (mpmath, 30 digits)")
        L_half_value = L_half_exact

        # L(1, chi_{-3}) via digamma function (Hurwitz pole at s=1)
        # L(1, chi) = -1/q * sum_{a=1}^{q-1} chi(a) * psi(a/q)
        # For q=3: L(1, chi_{-3}) = -1/3 * [chi(1)*psi(1/3) + chi(2)*psi(2/3)]
        # = -1/3 * [psi(1/3) - psi(2/3)]
        psi_1_3 = float(mpmath.digamma(mpmath.mpf(1) / 3))
        psi_2_3 = float(mpmath.digamma(mpmath.mpf(2) / 3))
        L_one_mp = -1.0 / 3 * (psi_1_3 - psi_2_3)
        print(f"  L(1, chi_{{-3}})   = {L_one_mp:.10f}  (mpmath digamma)")
        print(f"  L(1, chi_{{-3}})   = {L_one_exact:.10f}  (exact: pi/(3*sqrt(3)))")
        print(f"    Match: |diff| = {abs(L_one_mp - L_one_exact):.2e}")
    except ImportError:
        L_half_value = L_half_richardson
        print(f"  L(1/2, chi_{{-3}}) = {L_half_value:.10f}  (partial sum + Richardson)")
        print(f"  (mpmath not available for high-precision check)")

    print()
    print(f"  F-CONNECTION COMPARISON:")
    print(f"  -ln(F)            = {LN_F:.10f}")
    print(f"  36/100 (E6 roots) = 0.3600000000")
    print(f"  L(1/2, chi_{{-3}}) = {L_half_value:.10f}")
    print(f"  L(1/2)^2          = {L_half_value**2:.10f}")
    print()
    print(f"  |L(1/2) - (-ln(F))|         = {abs(L_half_value - LN_F):.10f}")
    print(f"  |L(1/2)^2 - (-ln(F))|       = {abs(L_half_value**2 - LN_F):.10f}")
    print(f"  |L(1/2)^2 - 2*(-ln(F))|     = {abs(L_half_value**2 - 2*LN_F):.10f}")
    print(f"  |L(1/2) - 36/100|           = {abs(L_half_value - 0.36):.10f}")
    print(f"  |L(1/2)^2/pi - (-ln(F))|    = {abs(L_half_value**2/np.pi - LN_F):.10f}")
    print(f"  |-ln(L(1/2)) - 1/3|         = {abs(-np.log(abs(L_half_value)) - 1/3):.10f}")

    # Check positive eigenvalue fractions
    print(f"\n  Positive eigenvalue fraction vs -ln(F):")
    print(f"  {'Radius':>6} {'no-res frac':>11} {'res frac':>9} {'-ln(F)':>8} "
          f"{'|delta_nr|':>10}")
    print("  " + "-" * 55)

    for radius in RADII:
        for variant in ["no_resonance", "resonance"]:
            key = (radius, variant)
            if key not in M_results:
                continue
            eigs = M_results[key]['eigenvalues']
            n_pos = np.sum(eigs > 1e-8)
            frac = n_pos / len(eigs)
            delta = abs(frac - LN_F)
            if variant == "no_resonance":
                nr_frac, nr_delta = frac, delta
            else:
                print(f"  {radius:6d} {nr_frac:11.5f} {frac:9.5f} "
                      f"{LN_F:8.5f} {nr_delta:10.6f}")

    # Save F-connection
    with open(RESULTS_DIR / "f_connection_lchi3.txt", 'w') as f:
        f.write("F-CONNECTION via L(1/2, chi_{-3})\n")
        f.write(f"-ln(F) = {LN_F:.12f}\n")
        f.write(f"36/100 = 0.360000\n")
        f.write(f"L(1/2, chi_{{-3}}) = {L_half_value:.12f}\n")
        f.write(f"L(1/2)^2 = {L_half_value**2:.12f}\n")
        f.write(f"|L(1/2) - (-ln(F))| = {abs(L_half_value - LN_F):.12f}\n")
        f.write(f"|L(1/2)^2 - (-ln(F))| = {abs(L_half_value**2 - LN_F):.12f}\n")

    # ==================================================================
    # FIGURES
    # ==================================================================
    print(f"\n{'='*76}")
    print("  GENERATING FIGURES")
    print(f"{'='*76}\n")

    s_range = np.linspace(0.001, 4.0, 200)

    # --- Figure 1: P(s) comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: chi_{-3} zeros spacing distribution
    ax = axes[0]
    nb = max(10, min(30, len(sp_chi3) // 8))
    ax.hist(sp_chi3, bins=nb, density=True, alpha=0.5,
            color='gold', edgecolor='darkgoldenrod', label=r'$L(s,\chi_{-3})$ zeros')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.plot(s_range, np.exp(-s_range), 'g:', lw=2, label='Poisson')
    ax.set_title(f'$L(s, \\chi_{{-3}})$ Zero Spacings\n'
                 f'$\\beta$={beta_chi3:.2f}, KS(GOE)={ks_chi3["goe"][0]:.3f}',
                 fontsize=12)
    ax.set_xlabel('s (unfolded spacing)', fontsize=12)
    ax.set_ylabel('P(s)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Panel 2: Best M spacing distribution
    ax = axes[1]
    best_sp = M_results[best_key]['spacings']
    if len(best_sp) > 5:
        nb_m = max(5, min(25, len(best_sp) // 4))
        try:
            ax.hist(best_sp, bins=nb_m, density=True, alpha=0.5,
                    color='coral', edgecolor='darkred', label=f'M ({best_key[1][:6]} r={best_key[0]})')
        except ValueError:
            ax.hist(best_sp, bins='auto', density=True, alpha=0.5,
                    color='coral', edgecolor='darkred', label=f'M ({best_key[1][:6]} r={best_key[0]})')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.hist(sp_chi3, bins=30, density=True, alpha=0.2,
            color='gold', edgecolor='darkgoldenrod', label=r'$\chi_{-3}$')
    bk = M_results[best_key]
    ax.set_title(f'Best M Spacings\n'
                 f'$\\beta$={bk["beta"]:.2f}, KS($\\chi_{{-3}}$)={bk["ks_chi3"]:.3f}',
                 fontsize=12)
    ax.set_xlabel('s', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Panel 3: Overlay all three — M, chi3, Riemann
    ax = axes[2]
    ax.hist(sp_chi3, bins=30, density=True, alpha=0.35,
            color='gold', edgecolor='darkgoldenrod', label=r'$L(s,\chi_{-3})$')
    if len(sp_riem) > 30:
        ax.hist(sp_riem[:500], bins=30, density=True, alpha=0.25,
                color='purple', edgecolor='purple', label=r'$\zeta(s)$ (Riemann)')
    if len(best_sp) > 5:
        try:
            ax.hist(best_sp, bins=nb_m, density=True, alpha=0.35,
                    color='coral', edgecolor='darkred', label=f'M (best)')
        except ValueError:
            pass
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.set_title('Three-Way Overlay', fontsize=12)
    ax.set_xlabel('s', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure1_spacing_comparison.png', dpi=150)
    plt.close(fig)
    print("  figure1_spacing_comparison.png saved")

    # --- Figure 2: Pair correlation ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(r_pc, g_chi3, 'o-', ms=3, color='goldenrod', lw=1.5,
            label=r'$L(s,\chi_{-3})$ zeros')
    ax.plot(r_pc, g_gue_theory, 'r-', lw=2, label='GUE: $1-\\mathrm{sinc}^2(\\pi r)$')
    ax.plot(r_pc, g_goe_theory, 'b--', lw=2, label='GOE (approx)')
    if len(g_M) == len(r_pc_M):
        ax.plot(r_pc_M, g_M, 's-', ms=3, color='coral', lw=1.2, label='M (best)')
    if len(g_nr) == len(r_pc_nr):
        ax.plot(r_pc_nr, g_nr, '^-', ms=3, color='steelblue', lw=1.2, label='M (no-res r10)')
    ax.axhline(y=1, color='green', ls=':', lw=1, alpha=0.5, label='Poisson')
    ax.set_xlabel('r (unfolded separation)', fontsize=14)
    ax.set_ylabel('g(r)', fontsize=14)
    ax.set_title('Pair Correlation Function', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure2_pair_correlation.png', dpi=150)
    plt.close(fig)
    print("  figure2_pair_correlation.png saved")

    # --- Figure 3: KS improvement plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: KS(chi3) vs KS(GOE) for no-resonance variant
    ax = axes[0]
    radii_plot = [r for r in RADII if (r, 'no_resonance') in M_results
                  and M_results[(r, 'no_resonance')]['n_eigs'] > 5]
    ks_chi3_vals = [M_results[(r, 'no_resonance')]['ks_chi3'] for r in radii_plot]
    ks_goe_vals = [M_results[(r, 'no_resonance')]['ks_goe'] for r in radii_plot]
    ks_riem_vals = [M_results[(r, 'no_resonance')]['ks_riem'] for r in radii_plot]

    ax.plot(radii_plot, ks_chi3_vals, 'o-', ms=8, color='goldenrod', lw=2,
            label=r'KS(M, $\chi_{-3}$)')
    ax.plot(radii_plot, ks_goe_vals, 's-', ms=8, color='blue', lw=2,
            label='KS(M, GOE)')
    ax.plot(radii_plot, ks_riem_vals, '^-', ms=8, color='purple', lw=2,
            label=r'KS(M, $\zeta$)')
    ax.axhline(y=0.05, color='gray', ls=':', alpha=0.5, label='p=0.05 threshold')
    ax.set_xlabel('Radius', fontsize=14)
    ax.set_ylabel('KS statistic', fontsize=14)
    ax.set_title('No-Resonance: Does $\\chi_{-3}$ beat GOE?', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: KS delta = KS(chi3) - KS(GOE)
    ax = axes[1]
    delta_vals = [ks_chi3_vals[i] - ks_goe_vals[i] for i in range(len(radii_plot))]
    colors = ['green' if d < 0 else 'red' for d in delta_vals]
    ax.bar(radii_plot, delta_vals, color=colors, alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='black', lw=1)
    ax.set_xlabel('Radius', fontsize=14)
    ax.set_ylabel(r'$\Delta$KS = KS($\chi_{-3}$) - KS(GOE)', fontsize=13)
    ax.set_title(r'Negative = $\chi_{-3}$ BETTER than generic GOE', fontsize=13)
    for i, (r, d) in enumerate(zip(radii_plot, delta_vals)):
        ax.text(r, d + 0.005 * np.sign(d), f"{d:+.3f}",
                ha='center', va='bottom' if d > 0 else 'top', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure3_ks_improvement.png', dpi=150)
    plt.close(fig)
    print("  figure3_ks_improvement.png saved")

    # --- Figure 4: F-connection three-way ---
    fig, ax = plt.subplots(figsize=(10, 7))

    radii_frac = []
    fracs_res, fracs_nores = [], []
    for radius in RADII:
        if (radius, 'resonance') in M_results and (radius, 'no_resonance') in M_results:
            e_r = M_results[(radius, 'resonance')]['eigenvalues']
            e_nr = M_results[(radius, 'no_resonance')]['eigenvalues']
            radii_frac.append(radius)
            fracs_res.append(np.sum(e_r > 1e-8) / len(e_r))
            fracs_nores.append(np.sum(e_nr > 1e-8) / len(e_nr))

    ax.plot(radii_frac, fracs_res, 'o-', ms=8, color='coral', lw=2,
            label='M (with resonance)')
    ax.plot(radii_frac, fracs_nores, 's-', ms=8, color='steelblue', lw=2,
            label='M (no resonance)')
    ax.axhline(y=LN_F, color='red', ls='--', lw=2,
               label=f'-ln(F) = {LN_F:.5f}')
    ax.axhline(y=0.36, color='green', ls=':', lw=2,
               label='36/100 (E6 roots)')
    ax.axhline(y=L_half_value, color='goldenrod', ls='-.',  lw=2,
               label=f'L(1/2, $\\chi_{{-3}}$) = {L_half_value:.5f}')
    ax.set_xlabel('Eisenstein Cell Radius', fontsize=14)
    ax.set_ylabel('Positive Eigenvalue Fraction', fontsize=14)
    ax.set_title('F-Connection: Three Constants', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure4_f_connection.png', dpi=150)
    plt.close(fig)
    print("  figure4_f_connection.png saved")

    # ==================================================================
    # FULL REPORT
    # ==================================================================
    print(f"\n{'='*76}")
    print("  STEP 7: FULL REPORT")
    print(f"{'='*76}\n")

    elapsed = time.time() - t_start

    print("  THE CHAIN:")
    print("    Z[omega], omega = exp(2pi*i/3)  =  ring of integers of Q(sqrt(-3))")
    print("    zeta_{Q(sqrt(-3))}(s) = zeta(s) * L(s, chi_{-3})")
    print("    chi_{-3} = Kronecker symbol (./−3), conductor 3, REAL")
    print()
    print(f"  chi_{{-3}} zeros: {chi3_class}")
    print(f"    beta = {beta_chi3:.3f}")
    print(f"    KS(GOE) = {ks_chi3['goe'][0]:.4f}  p = {ks_chi3['goe'][1]:.4f}")
    print(f"    KS(GUE) = {ks_chi3['gue'][0]:.4f}  p = {ks_chi3['gue'][1]:.4f}")
    print()
    print(f"  BEST M COMPARISON:")
    print(f"    {best_key}")
    print(f"    KS(M, chi_{{-3}}) = {br['ks_chi3']:.4f}  p = {br['p_chi3']:.4f}")
    print(f"    KS(M, GOE)      = {br['ks_goe']:.4f}  p = {br['p_goe']:.4f}")
    print(f"    KS(M, Riemann)  = {br['ks_riem']:.4f}  p = {br['p_riem']:.4f}")
    print()
    print(f"  F-CONNECTION:")
    print(f"    -ln(F)            = {LN_F:.8f}")
    print(f"    L(1/2, chi_{{-3}}) = {L_half_value:.8f}")
    print(f"    36/100            = 0.36000000")
    print(f"    pos_frac(r=10)    = {fracs_res[-1]:.8f}  (resonance)")
    print(f"    pos_frac(r=10)    = {fracs_nores[-1]:.8f}  (no resonance)")
    print()
    print(f"  DECISION: {decision}")
    print()
    print(f"  Elapsed: {elapsed:.1f}s")

    # Write full report
    with open(RESULTS_DIR / "FULL_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write("PATH A -- L(s, chi_{-3}) ZEROS vs EISENSTEIN M EIGENVALUE SPACINGS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("THE CHAIN:\n")
        f.write("  Z[omega] = ring of integers of Q(sqrt(-3))\n")
        f.write("  zeta_{Q(sqrt(-3))}(s) = zeta(s) * L(s, chi_{-3})\n")
        f.write("  chi_{-3} = Kronecker (./-3), conductor 3, real\n\n")

        f.write(f"chi_{{-3}} ZERO STATISTICS ({len(zeros_chi3)} zeros):\n")
        f.write(f"  Beta = {beta_chi3:.6f}\n")
        f.write(f"  KS(GOE) = {ks_chi3['goe'][0]:.6f}  p = {ks_chi3['goe'][1]:.6f}\n")
        f.write(f"  KS(GUE) = {ks_chi3['gue'][0]:.6f}  p = {ks_chi3['gue'][1]:.6f}\n")
        f.write(f"  KS(Poi) = {ks_chi3['poi'][0]:.6f}  p = {ks_chi3['poi'][1]:.6f}\n")
        f.write(f"  Class: {chi3_class}\n\n")

        f.write("M vs chi_{-3} COMPARISON TABLE:\n")
        f.write(f"{'Analysis':>20} KS(chi3) p(chi3) KS(GOE) p(GOE) KS(Riem) p(Riem)\n")
        f.write("-" * 80 + "\n")
        for key in sorted(best_results.keys()):
            r = best_results[key]
            label = f"r={key[0]} {key[1][:6]}"
            f.write(f"{label:>20} {r['ks_chi3']:.5f} {r['p_chi3']:.5f} "
                    f"{r['ks_goe']:.5f} {r['p_goe']:.5f} "
                    f"{r['ks_riem']:.5f} {r['p_riem']:.5f}\n")

        f.write(f"\nBest M: {best_key}\n")
        f.write(f"  KS(M, chi_{{-3}}) = {br['ks_chi3']:.6f}  p = {br['p_chi3']:.6f}\n")
        f.write(f"  KS(M, GOE) = {br['ks_goe']:.6f}  p = {br['p_goe']:.6f}\n")
        f.write(f"  KS(M, Riem) = {br['ks_riem']:.6f}  p = {br['p_riem']:.6f}\n\n")

        f.write("F-CONNECTION:\n")
        f.write(f"  -ln(F) = {LN_F:.12f}\n")
        f.write(f"  L(1/2, chi_{{-3}}) = {L_half_value:.12f}\n")
        f.write(f"  L(1/2)^2 = {L_half_value**2:.12f}\n")
        f.write(f"  36/100 = 0.360000\n")
        f.write(f"  |L(1/2) - (-ln(F))| = {abs(L_half_value - LN_F):.12f}\n\n")

        f.write(f"DECISION: {decision}\n\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")

    print(f"\n  All output in: {RESULTS_DIR}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
