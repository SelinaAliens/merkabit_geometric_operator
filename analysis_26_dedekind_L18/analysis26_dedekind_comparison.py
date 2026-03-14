#!/usr/bin/env python3
"""
ANALYSIS 26: DEDEKIND ZETA COMPARISON WITH CORRECT PEIERLS FORMULA
===================================================================

Uses Formula A (Landau gauge on EisensteinTorus) — the CORRECT Peierls
construction from Analysis 19, which gave KS(GUE)=0.052 at L=18, phi=1/6.

Compares M operator pair correlation against:
  1. Riemann zeta zeros
  2. L(s, chi_{-3}) zeros
  3. Dedekind zeta zeros (merged 1+2)
  4. Montgomery conjecture 1 - (sin(pi*r)/(pi*r))^2

Key questions:
  Q1: Is RMS(M, Dedekind) < RMS(M, Riemann)?
  Q2: Do split eigenvectors match Riemann, inert match L(s,chi_{-3})?
  Q3: What is the RMS at L=18 with correct formula?
"""

import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn
import mpmath

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\dedekind_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H   # pi/6
XI = 3.0
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]


# ============================================================================
# EISENSTEIN TORUS (from Analysis 19 — Formula A construction)
# ============================================================================

class EisensteinTorus:
    """
    Periodic Eisenstein lattice (flat torus).
    Node (a,b) identified with (a mod L, b mod L). L^2 nodes, coord 6.
    Sublattice: (a+b) mod 3 -> chirality {0, +1, -1}.
    """
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
# GEOMETRIC HOPF SPINORS
# ============================================================================

def assign_spinors_geometric(cell):
    """
    Geometric Hopf-paired spinors on torus.
    u[i] = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]
    v[i] = SU(2) time-reverse of u[i]: [-conj(u1), conj(u0)]
    """
    u_list, v_list, omega_list = [], [], []
    z_coords = [a + b * OMEGA_EISEN for (a, b) in cell.nodes]
    L_scale = max(abs(z) for z in z_coords) if len(z_coords) > 1 else 1.0

    for i, (a, b) in enumerate(cell.nodes):
        z = z_coords[i]
        r = abs(z) / (L_scale + 1e-10)
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
# FORMULA A: LANDAU GAUGE PEIERLS PHASE (from Analysis 19)
# ============================================================================

def build_M_formula_A(cell, spinors, phi=1.0/6, xi=XI):
    """
    Build M operator on Eisenstein torus with Landau gauge Peierls flux.

    Formula A: A_ij = phi * (2*a_i + da)/2 * db
    Coupling: decay * <u_i|v_j> * exp(2j*pi*A_ij)

    phi = 1/6 -> physical flux pi/6 per triangle = one P gate step.
    NO resonance factor (as per task brief).
    """
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = cell.L

    for (i, j) in cell.edges:
        a_i, b_i = cell.nodes[i]
        a_j, b_j = cell.nodes[j]

        u_i = spinors['u'][i]
        v_j = spinors['v'][j]
        u_j = spinors['u'][j]
        v_i = spinors['v'][i]

        # Minimal image displacement (handle torus wrapping)
        da = a_j - a_i
        db = b_j - b_i
        if da > L // 2: da -= L
        if da < -(L // 2): da += L
        if db > L // 2: db -= L
        if db < -(L // 2): db += L

        # Peierls phase (Landau gauge): A_ij = phi * (2*a_i + da)/2 * db
        A_ij = phi * (2 * a_i + da) / 2.0 * db

        # Cross-chirality coupling with Peierls phase
        coupling_ij = decay * np.vdot(u_i, v_j) * np.exp(2j * np.pi * A_ij)

        M[i, j] = coupling_ij
        M[j, i] = np.conj(coupling_ij)

    # Hermitianise
    M = (M + M.conj().T) / 2.0
    return M


# ============================================================================
# SPECTRAL ANALYSIS TOOLS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=12):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        return evals, np.diff(evals) / max(np.mean(np.diff(evals)), 1e-10)
    cdf = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 20))
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


def wigner_surmise_pdf(s, beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    return a * s ** beta * np.exp(-b * s ** 2)


def fit_beta(spacings):
    pos = spacings[spacings > 0.02]
    if len(pos) < 8:
        pos = spacings[spacings > 0.005]
    if len(pos) < 5:
        return 0.0, 0.0
    small = pos[pos < np.percentile(pos, 40)]
    if len(small) < 5:
        return 0.0, 0.0
    log_s = np.log(small)
    log_p = np.log(np.arange(1, len(small) + 1) / len(pos))
    if len(log_s) > 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_s, log_p)
        return max(0, slope), std_err
    return 0.0, 0.0


def ks_test_gue(spacings):
    """KS test against GUE (beta=2) Wigner surmise."""
    a_gue = 32 / np.pi**2
    b_gue = 4 / np.pi

    def gue_cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a_gue * x**2 * np.exp(-b_gue * x**2), 0, max(s, 0))
            return val
        return np.array([gue_cdf(si) for si in s])

    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return 1.0, 0.0
    emp_cdf = np.arange(1, len(pos) + 1) / len(pos)
    theo_cdf = gue_cdf(np.sort(pos))
    ks_stat = np.max(np.abs(emp_cdf - theo_cdf))
    # Approximate p-value
    n = len(pos)
    p_val = 2 * np.exp(-2 * n * ks_stat**2) if n > 0 else 0
    return ks_stat, p_val


def ks_test_goe(spacings):
    """KS test against GOE (beta=1) Wigner surmise."""
    def goe_cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: (np.pi/2)*x*np.exp(-np.pi*x**2/4), 0, max(s, 0))
            return val
        return np.array([goe_cdf(si) for si in s])

    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return 1.0, 0.0
    emp_cdf = np.arange(1, len(pos) + 1) / len(pos)
    theo_cdf = goe_cdf(np.sort(pos))
    ks_stat = np.max(np.abs(emp_cdf - theo_cdf))
    n = len(pos)
    p_val = 2 * np.exp(-2 * n * ks_stat**2) if n > 0 else 0
    return ks_stat, p_val


# ============================================================================
# PAIR CORRELATION FUNCTIONS
# ============================================================================

def pair_correlation(spacings, r_max=3.0, n_bins=60):
    """Compute pair correlation g(r) from unfolded spacings."""
    # From unfolded eigenvalues, compute all pair separations
    pos = spacings[spacings > 0]
    if len(pos) < 10:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)

    # Build unfolded eigenvalue positions from spacings
    unfolded = np.cumsum(np.concatenate([[0], pos]))

    # Compute all pair distances
    N = len(unfolded)
    distances = []
    for i in range(N):
        for j in range(i + 1, min(i + 50, N)):  # local pairs only
            d = unfolded[j] - unfolded[i]
            if d < r_max:
                distances.append(d)

    if len(distances) < 10:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)

    distances = np.array(distances)
    bins = np.linspace(0, r_max, n_bins + 1)
    counts, _ = np.histogram(distances, bins=bins)
    dr = bins[1] - bins[0]
    r_centers = (bins[:-1] + bins[1:]) / 2

    # Normalize: expected count in each bin = N * (N-1)/2 * dr / L_total * 2
    L_total = unfolded[-1] - unfolded[0]
    density = N / L_total if L_total > 0 else 1
    expected = density * dr * N  # expected per bin
    g = counts / max(expected, 1)

    return r_centers, g


def montgomery_pair(r):
    """Montgomery pair correlation: 1 - (sin(pi*r)/(pi*r))^2"""
    result = np.ones_like(r, dtype=float)
    nonzero = r > 1e-10
    result[nonzero] = 1.0 - (np.sin(np.pi * r[nonzero]) / (np.pi * r[nonzero]))**2
    result[~nonzero] = 0.0
    return result


def rms_vs_montgomery(r, g):
    """RMS deviation of pair correlation from Montgomery formula."""
    g_mont = montgomery_pair(r)
    mask = r > 0.1  # avoid r=0 singularity
    if np.sum(mask) < 3:
        return np.inf
    return np.sqrt(np.mean((g[mask] - g_mont[mask])**2))


def pair_correlation_from_zeros(zeros, r_max=3.0, n_bins=60):
    """Compute pair correlation directly from a list of zeros."""
    zeros = np.sort(zeros)
    N = len(zeros)

    # Unfold: use smooth part of counting function
    # For Riemann zeros: N(T) ~ T/(2pi) * log(T/(2pi*e))
    # For simplicity, use polynomial unfolding
    if N < 10:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)

    _, spacings = unfold_spectrum(zeros)
    return pair_correlation(spacings, r_max, n_bins)


# ============================================================================
# L-FUNCTION ZEROS
# ============================================================================

def L_chi3(s):
    """L(s, chi_{-3}) using mpmath's nsum for convergence acceleration."""
    chi = [0, 1, -1]
    return mpmath.nsum(lambda n: chi[int(n) % 3] * mpmath.power(n, -s), [1, mpmath.inf])


def Z_L(t):
    """Hardy Z-function for L(s, chi_{-3})."""
    s = mpmath.mpf('0.5') + mpmath.mpc(0, t)
    L_val = L_chi3(s)
    prefactor = mpmath.power(3 / mpmath.pi, (s + 1) / 2) * mpmath.gamma((s + 1) / 2)
    theta = mpmath.arg(prefactor)
    Z = mpmath.exp(1j * theta) * L_val
    return float(mpmath.re(Z))


def compute_L_zeros_fast(T_max, dt=0.3):
    """Find zeros of L(s, chi_{-3}) on critical line using Hardy Z-function."""
    print(f"\n  Computing L(s,chi_-3) zeros up to T={T_max}...")
    mpmath.mp.dps = 20
    t0 = time.time()

    zeros = []
    t = 1.0
    prev_Z = Z_L(t)

    while t < T_max:
        t += dt
        curr_Z = Z_L(t)
        if prev_Z * curr_Z < 0:
            # Bisect to refine
            lo, hi = t - dt, t
            for _ in range(30):
                mid = (lo + hi) / 2
                Z_mid = Z_L(mid)
                if Z_mid * Z_L(lo) < 0:
                    hi = mid
                else:
                    lo = mid
            zeros.append((lo + hi) / 2)
        prev_Z = curr_Z

    zeros = np.array(zeros)
    print(f"  Found {len(zeros)} L-function zeros in {time.time()-t0:.1f}s")
    if len(zeros) > 0:
        print(f"  Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
    return zeros


# ============================================================================
# STEP 1: COMPUTE THREE ZERO SETS
# ============================================================================

def step1_zero_sets():
    print("=" * 70)
    print("STEP 1: COMPUTE THREE ZERO SETS")
    print("=" * 70)

    # 1a. Riemann zeros (500)
    N_riemann = 500
    print(f"\n  Computing {N_riemann} Riemann zeta zeros...")
    t0 = time.time()
    mpmath.mp.dps = 20
    riemann_zeros = []
    for n in range(1, N_riemann + 1):
        z = mpmath.zetazero(n)
        riemann_zeros.append(float(z.imag))
        if n % 100 == 0:
            print(f"    {n}/{N_riemann} (t = {riemann_zeros[-1]:.2f})")
    riemann_zeros = np.array(riemann_zeros)
    print(f"  Riemann: {len(riemann_zeros)} zeros in {time.time()-t0:.1f}s")
    print(f"    Range: [{riemann_zeros[0]:.4f}, {riemann_zeros[-1]:.4f}]")

    # 1b. L-function zeros — try cache first
    cache_path = Path(r"C:\Users\selin\merkabit_results\lchi3_comparison\lchi3_zeros.npy")
    if cache_path.exists():
        L_zeros = np.load(cache_path)
        print(f"\n  L-function: loaded {len(L_zeros)} cached zeros")
        print(f"    Range: [{L_zeros[0]:.4f}, {L_zeros[-1]:.4f}]")
    else:
        T_max = riemann_zeros[-1] * 1.2  # match range
        L_zeros = compute_L_zeros_fast(T_max)

    # Extend L-function zeros if needed to match Riemann range
    if len(L_zeros) > 0 and L_zeros[-1] < riemann_zeros[-1]:
        print(f"  Extending L-zeros from {L_zeros[-1]:.1f} to {riemann_zeros[-1]:.1f}...")
        extra = compute_L_zeros_fast(riemann_zeros[-1] + 10, dt=0.3)
        # Merge, avoiding duplicates
        combined = np.sort(np.unique(np.concatenate([L_zeros, extra])))
        L_zeros = combined
        print(f"  L-function total: {len(L_zeros)} zeros")

    # 1c. Dedekind zeros (merged)
    dedekind_zeros = np.sort(np.concatenate([riemann_zeros, L_zeros]))
    # Remove near-duplicates (within 0.001)
    unique_ded = [dedekind_zeros[0]]
    for z in dedekind_zeros[1:]:
        if z - unique_ded[-1] > 0.001:
            unique_ded.append(z)
    dedekind_zeros = np.array(unique_ded)

    print(f"\n  ZERO SET SUMMARY:")
    print(f"    Riemann:  {len(riemann_zeros)} zeros, range [{riemann_zeros[0]:.2f}, {riemann_zeros[-1]:.2f}]")
    print(f"    L-func:   {len(L_zeros)} zeros, range [{L_zeros[0]:.2f}, {L_zeros[-1]:.2f}]")
    print(f"    Dedekind: {len(dedekind_zeros)} zeros (merged)")

    # Density check
    T_range = min(riemann_zeros[-1], L_zeros[-1])
    d_r = len(riemann_zeros[riemann_zeros <= T_range]) / T_range
    d_L = len(L_zeros[L_zeros <= T_range]) / T_range
    d_d = len(dedekind_zeros[dedekind_zeros <= T_range]) / T_range
    print(f"\n  Density (up to T={T_range:.1f}):")
    print(f"    Riemann:  {d_r:.6f}  (expected ~1/(2pi) = {1/(2*np.pi):.6f})")
    print(f"    L-func:   {d_L:.6f}  (expected ~1/(2pi))")
    print(f"    Dedekind: {d_d:.6f}  (expected ~1/pi = {1/np.pi:.6f})")

    return riemann_zeros, L_zeros, dedekind_zeros


# ============================================================================
# STEP 2: M EIGENVALUES AT L=18, phi=1/6 (FORMULA A)
# ============================================================================

def step2_M_eigenvalues(L=18, phi=1.0/6):
    print(f"\n{'='*70}")
    print(f"STEP 2: M OPERATOR — L={L}, phi={phi:.6f} (Formula A, Landau gauge)")
    print(f"{'='*70}")

    t0 = time.time()
    cell = EisensteinTorus(L)
    print(f"  Torus: {cell.num_nodes} nodes, {len(cell.edges)} edges")

    # Sublattice counts
    sub_counts = [0, 0, 0]
    for s in cell.sublattice:
        sub_counts[s] += 1
    print(f"  Sublattice: s=0: {sub_counts[0]}, s=1: {sub_counts[1]}, s=2: {sub_counts[2]}")

    spinors = assign_spinors_geometric(cell)
    print(f"  Spinors assigned ({time.time()-t0:.1f}s)")

    # Verify plaquette flux
    print(f"\n  Plaquette flux verification (phi={phi:.6f}):")
    # Type A triangle: (0,0) -> (1,0) -> (1,1) -> (0,0)
    A01 = phi * (2 * 0 + 1) / 2.0 * 0   # da=1, db=0
    A12 = phi * (2 * 1 + 0) / 2.0 * 1   # da=0, db=1
    A20 = phi * (2 * 1 + (-1)) / 2.0 * (-1)  # da=-1, db=-1
    flux_A = A01 + A12 + A20
    print(f"    Triangle A flux: {flux_A:.6f} (expect {phi/2:.6f})")
    print(f"    Physical flux: {2*np.pi*flux_A:.6f} rad = {2*np.pi*flux_A/np.pi:.6f} * pi")

    # Build M
    print(f"\n  Building M operator...")
    M = build_M_formula_A(cell, spinors, phi, xi=XI)
    print(f"  M built: {M.shape} ({time.time()-t0:.1f}s)")

    # Verify Hermiticity and complexity
    herm_err = np.max(np.abs(M - M.conj().T))
    real_frac = np.sum(np.abs(np.imag(M[np.abs(M) > 1e-10])) < 1e-10) / max(np.sum(np.abs(M) > 1e-10), 1)
    print(f"  Hermiticity error: {herm_err:.2e}")
    print(f"  Fraction of purely real entries: {real_frac:.4f}")
    print(f"  M is {'REAL (GOE expected)' if real_frac > 0.99 else 'COMPLEX (GUE possible)'}")

    # Diagonalize
    print(f"\n  Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    print(f"  Eigenvalues: {len(eigenvalues)}, range [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")
    print(f"  Diagonalization done ({time.time()-t0:.1f}s)")

    # Positive wing extraction
    pos_wing = extract_positive_wing(eigenvalues, threshold_pct=20)
    print(f"  Positive wing: {len(pos_wing)} eigenvalues (after 20% threshold)")

    # Unfold and compute spacings
    _, spacings = unfold_spectrum(pos_wing)
    print(f"  Unfolded spacings: {len(spacings)}")

    # RMT statistics
    beta, beta_err = fit_beta(spacings)
    ks_gue, p_gue = ks_test_gue(spacings)
    ks_goe, p_goe = ks_test_goe(spacings)

    print(f"\n  RMT STATISTICS (positive wing):")
    print(f"    beta = {beta:.3f} +/- {beta_err:.3f}")
    print(f"    KS(GUE) = {ks_gue:.4f}, p = {p_gue:.4f}")
    print(f"    KS(GOE) = {ks_goe:.4f}, p = {p_goe:.4f}")
    print(f"    Classification: {'GUE' if ks_gue < ks_goe else 'GOE'}")

    return eigenvalues, eigenvectors, spacings, cell


# ============================================================================
# STEP 3: PAIR CORRELATION COMPARISON
# ============================================================================

def step3_pair_correlation(spacings_M, riemann_zeros, L_zeros, dedekind_zeros):
    print(f"\n{'='*70}")
    print("STEP 3: PAIR CORRELATION COMPARISON")
    print(f"{'='*70}")

    r_max = 3.0
    n_bins = 60

    # M operator pair correlation
    r_M, g_M = pair_correlation(spacings_M, r_max, n_bins)

    # Zero set pair correlations
    r_R, g_R = pair_correlation_from_zeros(riemann_zeros, r_max, n_bins)
    r_L, g_L = pair_correlation_from_zeros(L_zeros, r_max, n_bins)
    r_D, g_D = pair_correlation_from_zeros(dedekind_zeros, r_max, n_bins)

    # Montgomery reference
    r_ref = np.linspace(0.01, r_max, 200)
    g_mont = montgomery_pair(r_ref)

    # RMS deviations from Montgomery
    rms_M = rms_vs_montgomery(r_M, g_M)
    rms_R = rms_vs_montgomery(r_R, g_R)
    rms_L = rms_vs_montgomery(r_L, g_L)
    rms_D = rms_vs_montgomery(r_D, g_D)

    print(f"\n  RMS vs Montgomery:")
    print(f"    M operator:     {rms_M:.6f}")
    print(f"    Riemann zeros:  {rms_R:.6f}")
    print(f"    L-func zeros:   {rms_L:.6f}")
    print(f"    Dedekind zeros: {rms_D:.6f}")

    # Direct RMS between M and each zero family
    # Interpolate to common grid
    r_common = np.linspace(0.1, r_max, 50)
    g_M_interp = np.interp(r_common, r_M, g_M)
    g_R_interp = np.interp(r_common, r_R, g_R)
    g_L_interp = np.interp(r_common, r_L, g_L)
    g_D_interp = np.interp(r_common, r_D, g_D)

    rms_MR = np.sqrt(np.mean((g_M_interp - g_R_interp)**2))
    rms_ML = np.sqrt(np.mean((g_M_interp - g_L_interp)**2))
    rms_MD = np.sqrt(np.mean((g_M_interp - g_D_interp)**2))

    print(f"\n  Direct RMS (M vs zero family):")
    print(f"    M vs Riemann:   {rms_MR:.6f}")
    print(f"    M vs L-func:    {rms_ML:.6f}")
    print(f"    M vs Dedekind:  {rms_MD:.6f}")
    print(f"\n  Q1 ANSWER: RMS(M,Dedekind) {'<' if rms_MD < rms_MR else '>='} RMS(M,Riemann)")
    print(f"    Ratio: RMS(M,Ded)/RMS(M,Riem) = {rms_MD/max(rms_MR,1e-10):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: M vs Montgomery
    ax = axes[0, 0]
    ax.plot(r_M, g_M, 'b-', lw=2, label='M operator (L=18, phi=1/6)', alpha=0.8)
    ax.plot(r_ref, g_mont, 'k--', lw=2, label='Montgomery conjecture')
    ax.set_xlabel('r (unfolded spacing)')
    ax.set_ylabel('g(r)')
    ax.set_title(f'M operator pair correlation (RMS={rms_M:.4f})')
    ax.legend()
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.1, 1.5)

    # Top-right: All zero families
    ax = axes[0, 1]
    ax.plot(r_R, g_R, 'r-', lw=1.5, label=f'Riemann (RMS={rms_R:.4f})', alpha=0.7)
    ax.plot(r_L, g_L, 'g-', lw=1.5, label=f'L(s,chi_-3) (RMS={rms_L:.4f})', alpha=0.7)
    ax.plot(r_D, g_D, 'm-', lw=1.5, label=f'Dedekind (RMS={rms_D:.4f})', alpha=0.7)
    ax.plot(r_ref, g_mont, 'k--', lw=2, label='Montgomery')
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title('Zero family pair correlations')
    ax.legend(fontsize=9)
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.1, 1.5)

    # Bottom-left: M vs Riemann and Dedekind
    ax = axes[1, 0]
    ax.plot(r_M, g_M, 'b-', lw=2, label=f'M (RMS_Mont={rms_M:.4f})', alpha=0.8)
    ax.plot(r_R, g_R, 'r--', lw=1.5, label=f'Riemann (RMS_M={rms_MR:.4f})')
    ax.plot(r_D, g_D, 'm--', lw=1.5, label=f'Dedekind (RMS_M={rms_MD:.4f})')
    ax.plot(r_ref, g_mont, 'k:', lw=1, alpha=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title(f'M vs Riemann vs Dedekind (Q1)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.1, 1.5)

    # Bottom-right: Spacing distributions
    ax = axes[1, 1]
    s_grid = np.linspace(0, 4, 100)
    goe_pdf = wigner_surmise_pdf(s_grid, 1)
    gue_pdf = wigner_surmise_pdf(s_grid, 2)
    poisson_pdf = np.exp(-s_grid)
    ax.hist(spacings_M[spacings_M > 0], bins=30, density=True, alpha=0.5, color='blue', label='M spacings')
    ax.plot(s_grid, goe_pdf, 'r-', lw=2, label='GOE (beta=1)')
    ax.plot(s_grid, gue_pdf, 'g-', lw=2, label='GUE (beta=2)')
    ax.plot(s_grid, poisson_pdf, 'k--', lw=1.5, label='Poisson')
    ax.set_xlabel('s (unfolded spacing)')
    ax.set_ylabel('P(s)')
    ax.set_title('M operator spacing distribution')
    ax.legend()
    ax.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'pair_correlation_comparison.png', dpi=150)
    plt.close()
    print(f"\n  Saved: pair_correlation_comparison.png")

    return {
        'rms_M_mont': rms_M, 'rms_R_mont': rms_R, 'rms_L_mont': rms_L, 'rms_D_mont': rms_D,
        'rms_MR': rms_MR, 'rms_ML': rms_ML, 'rms_MD': rms_MD,
        'r_M': r_M, 'g_M': g_M, 'r_R': r_R, 'g_R': g_R, 'r_D': r_D, 'g_D': g_D,
    }


# ============================================================================
# STEP 4: Z3 SUBLATTICE EIGENVECTOR CLASSIFICATION
# ============================================================================

def step4_sublattice_classification(eigenvalues, eigenvectors, cell, spacings_M):
    print(f"\n{'='*70}")
    print("STEP 4: Z3 SUBLATTICE EIGENVECTOR CLASSIFICATION")
    print(f"{'='*70}")

    N = cell.num_nodes
    sublattice = np.array(cell.sublattice)

    # For each eigenvector, compute weight on each sublattice
    # sublattice s in {0, 1, 2} -> Z3 class: 0=ramified, 1=split, 2=inert
    print(f"\n  Z3 prime -> sublattice mapping:")
    print(f"    s=0 (ramified): p=3, gates T,F")
    print(f"    s=1 (split):    p=1 mod 3, gates R,P")
    print(f"    s=2 (inert):    p=2 mod 3, gate S")

    weights = np.zeros((len(eigenvalues), 3))
    dominant_sub = np.zeros(len(eigenvalues), dtype=int)

    for k in range(len(eigenvalues)):
        v = eigenvectors[:, k]
        prob = np.abs(v)**2
        for s in range(3):
            mask = sublattice == s
            weights[k, s] = np.sum(prob[mask])
        dominant_sub[k] = np.argmax(weights[k])

    # Overall distribution
    print(f"\n  Eigenvector sublattice weights (all {len(eigenvalues)} eigenvectors):")
    print(f"    Mean weight s=0 (ramified): {np.mean(weights[:, 0]):.4f}")
    print(f"    Mean weight s=1 (split):    {np.mean(weights[:, 1]):.4f}")
    print(f"    Mean weight s=2 (inert):    {np.mean(weights[:, 2]):.4f}")
    print(f"    Expected uniform: {1/3:.4f}")

    # Count dominant sublattice
    counts = [np.sum(dominant_sub == s) for s in range(3)]
    print(f"\n  Dominant sublattice counts:")
    print(f"    s=0 (ramified): {counts[0]} ({100*counts[0]/len(eigenvalues):.1f}%)")
    print(f"    s=1 (split):    {counts[1]} ({100*counts[1]/len(eigenvalues):.1f}%)")
    print(f"    s=2 (inert):    {counts[2]} ({100*counts[2]/len(eigenvalues):.1f}%)")

    # Positive wing analysis — split into sublattice groups
    pos_mask = eigenvalues > np.percentile(eigenvalues[eigenvalues > 0], 20)
    pos_indices = np.where(pos_mask)[0]

    print(f"\n  POSITIVE WING sublattice analysis ({len(pos_indices)} eigenvectors):")

    for s, label in [(0, 'ramified'), (1, 'split'), (2, 'inert')]:
        sub_mask = dominant_sub[pos_indices] == s
        sub_evals = eigenvalues[pos_indices[sub_mask]]
        if len(sub_evals) > 10:
            _, sub_spacings = unfold_spectrum(sub_evals)
            sub_beta, sub_err = fit_beta(sub_spacings)
            sub_ks_gue, sub_p_gue = ks_test_gue(sub_spacings)
            sub_ks_goe, sub_p_goe = ks_test_goe(sub_spacings)
            print(f"\n    s={s} ({label}): {len(sub_evals)} eigenvalues")
            print(f"      beta = {sub_beta:.3f} +/- {sub_err:.3f}")
            print(f"      KS(GUE) = {sub_ks_gue:.4f}, KS(GOE) = {sub_ks_goe:.4f}")
            print(f"      -> {'GUE' if sub_ks_gue < sub_ks_goe else 'GOE'}")
        else:
            print(f"\n    s={s} ({label}): {len(sub_evals)} eigenvalues (too few for RMT)")

    # Q2: Do split eigenvectors match Riemann, inert match L(s,chi_{-3})?
    # Test via pair correlation of sublattice-separated eigenvalues
    print(f"\n  Q2 ASSESSMENT:")
    print(f"    Split (s=1) should correlate with Riemann zeros (GOE/GUE)")
    print(f"    Inert (s=2) should correlate with L-function zeros")
    print(f"    Ramified (s=0) should be special (p=3 only)")

    # Max weight analysis
    max_weights = np.max(weights, axis=1)
    print(f"\n  Max sublattice weight statistics:")
    print(f"    Mean: {np.mean(max_weights):.4f}")
    print(f"    Std:  {np.std(max_weights):.4f}")
    print(f"    Min:  {np.min(max_weights):.4f}")
    print(f"    Max:  {np.max(max_weights):.4f}")
    strong_loc = np.sum(max_weights > 0.5)
    print(f"    Strongly localized (>0.5): {strong_loc} ({100*strong_loc/len(max_weights):.1f}%)")

    # Sublattice pair correlation comparison
    print(f"\n  Sublattice-resolved pair correlations:")
    r_max = 3.0
    for s, label in [(1, 'split'), (2, 'inert')]:
        sub_mask = dominant_sub[pos_indices] == s
        sub_evals = eigenvalues[pos_indices[sub_mask]]
        if len(sub_evals) > 20:
            _, sub_spacings = unfold_spectrum(sub_evals)
            r_sub, g_sub = pair_correlation(sub_spacings, r_max, 40)
            rms_sub = rms_vs_montgomery(r_sub, g_sub)
            print(f"    s={s} ({label}): RMS(Montgomery) = {rms_sub:.4f}")

    return weights, dominant_sub


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ANALYSIS 26: DEDEKIND ZETA COMPARISON WITH CORRECT PEIERLS FORMULA")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nFormula A (Landau gauge): A_ij = phi * (2*a_i + da)/2 * db")
    print(f"Coupling: decay * <u_i|v_j> * exp(2j*pi*A_ij)")
    print(f"Parameters: L=18, phi=1/6, xi={XI}, no resonance")

    # Step 1
    riemann_zeros, L_zeros, dedekind_zeros = step1_zero_sets()

    # Step 2
    eigenvalues, eigenvectors, spacings_M, cell = step2_M_eigenvalues(L=18, phi=1.0/6)

    # Step 3
    pc_results = step3_pair_correlation(spacings_M, riemann_zeros, L_zeros, dedekind_zeros)

    # Step 4
    weights, dominant_sub = step4_sublattice_classification(eigenvalues, eigenvectors, cell, spacings_M)

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("ANALYSIS 26 — FINAL ANSWERS")
    print(f"{'='*70}")

    print(f"\n  Q1: Is RMS(M, Dedekind) < RMS(M, Riemann)?")
    print(f"      RMS(M, Riemann)  = {pc_results['rms_MR']:.6f}")
    print(f"      RMS(M, Dedekind) = {pc_results['rms_MD']:.6f}")
    if pc_results['rms_MD'] < pc_results['rms_MR']:
        print(f"      ANSWER: YES — M is closer to Dedekind than to Riemann alone")
    else:
        print(f"      ANSWER: NO — M is closer to Riemann than to Dedekind")

    print(f"\n  Q2: Do split eigenvectors match Riemann, inert match L(s,chi_-3)?")
    print(f"      (See sublattice classification above)")

    print(f"\n  Q3: What is the RMS at L=18 with correct formula?")
    print(f"      RMS(M, Montgomery) = {pc_results['rms_M_mont']:.6f}")

    # Save eigenvalues
    np.save(RESULTS_DIR / 'eigenvalues_L18_phi016.npy', eigenvalues)
    np.save(RESULTS_DIR / 'eigenvectors_L18_phi016.npy', eigenvectors)
    np.save(RESULTS_DIR / 'riemann_zeros_500.npy', riemann_zeros)
    np.save(RESULTS_DIR / 'L_zeros.npy', L_zeros)
    np.save(RESULTS_DIR / 'dedekind_zeros.npy', dedekind_zeros)
    np.save(RESULTS_DIR / 'sublattice_weights.npy', weights)
    print(f"\n  Saved all data to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
