#!/usr/bin/env python3
"""
ANALYSIS 27: L=30 DEDEKIND COMPARISON — ANALYSIS 20 PIPELINE
=============================================================

Q1 at L=30: Does M fit zeta_{Q(omega)} better than zeta(s)?

Uses EXACT Analysis 20 pipeline:
  - Gaussian KDE pair correlation on ALL N*(N-1) pairwise diffs
  - Riemann-von Mangoldt unfolding for Riemann zeros
  - Polynomial unfolding (degree 10) for M eigenvalues
  - r_vals = linspace(0.01, 4.0, 200)
  - bandwidth = 0.4
  - positive wing > 20th percentile

Parameters identical to Analysis 20/21:
  L=30, phi=1/6 (Formula A), xi=3.0, no resonance
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
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\dedekind_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
XI = 3.0
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

# Analysis 20 parameters
R_VALS = np.linspace(0.01, 4.0, 200)
BANDWIDTH = 0.4
POLY_DEG = 10
PCT_THRESH = 20


# ============================================================================
# EISENSTEIN TORUS (exact Analysis 20)
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
# GEOMETRIC SPINORS (exact Analysis 20)
# ============================================================================

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
# M CONSTRUCTION (exact Analysis 20 — Formula A, no resonance)
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
        if da > L // 2: da -= L
        if da < -(L // 2): da += L
        if db > L // 2: db -= L
        if db < -(L // 2): db += L
        A_ij = Phi * (2 * a_i + da) / 2.0 * db
        coupling = decay * resonance * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)
    M = (M + M.conj().T) / 2.0
    return M


# ============================================================================
# SPECTRAL ANALYSIS (exact Analysis 20)
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=POLY_DEG):
    """Polynomial unfolding for M eigenvalues (Analysis 20 method)."""
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


def extract_positive_wing(eigenvalues, threshold_pct=PCT_THRESH):
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    return pos_eigs[pos_eigs > cutoff]


def get_unfolded_wing(eigs):
    """Full Analysis 19/20 pipeline: positive wing + polynomial unfolding."""
    eigs = np.sort(np.real(eigs))
    wing = extract_positive_wing(eigs)
    if len(wing) < 10:
        return wing, np.array([])
    unfolded, spacings = unfold_spectrum(wing)
    return unfolded, spacings


# ============================================================================
# UNFOLDING FOR ZERO FAMILIES (Analysis 20 method)
# ============================================================================

def unfold_riemann_zeros(gammas):
    """
    Unfold Riemann zeros using Riemann-von Mangoldt formula.
    N(T) = T/(2pi) * log(T/(2pi*e)) + 7/8
    This is the Analysis 20 method. Do NOT use polynomial fit.
    """
    T = np.asarray(gammas, dtype=float)
    N_smooth = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_smooth


def unfold_lchi3_zeros(gammas):
    """
    Unfold L(s,chi_{-3}) zeros.
    Same leading density as Riemann: N_L(T) ~ T/(2pi)*log(q*T/(2pi*e))
    For conductor q=3: N_L(T) = T/(2pi)*log(3*T/(2pi*e)) + correction
    Use the same form as Riemann-von Mangoldt with conductor correction.
    """
    T = np.asarray(gammas, dtype=float)
    N_smooth = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_smooth


def unfold_dedekind_zeros(gammas):
    """
    Unfold Dedekind zeros using double-density formula.
    N_ded(T) = sum of Riemann + L-function counting:
    N(T) = T/pi * log(T/(2pi*e)) + correction from conductor
    Approximate: N_ded(T) ~ T/(2pi)*log(T/(2pi*e)) + T/(2pi)*log(3T/(2pi*e)) + 7/4
    """
    T = np.asarray(gammas, dtype=float)
    N_riem = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    N_L = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_riem + N_L


# ============================================================================
# PAIR CORRELATION (exact Analysis 20 — Gaussian KDE, all pairwise diffs)
# ============================================================================

def pair_correlation_kde(unfolded_eigs, r_vals, bandwidth=BANDWIDTH):
    """
    Gaussian KDE on ALL N*(N-1) pairwise differences.
    This is the Analysis 20 method. Do not substitute histogram binning.
    """
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5:
        return np.ones(len(r_vals))

    # All pairwise differences
    diffs = eigs[:, None] - eigs[None, :]  # (N, N)
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]  # N*(N-1) differences

    # KDE evaluation
    g = np.zeros(len(r_vals))
    for k, r in enumerate(r_vals):
        g[k] = np.sum(np.exp(-0.5 * ((diffs_flat - r) / bandwidth) ** 2))

    g /= (N * bandwidth * np.sqrt(2 * np.pi))

    # Normalize to 1 at large r
    large_r_mask = r_vals > 3.0
    if large_r_mask.sum() > 3:
        g /= np.mean(g[large_r_mask])

    return g


def montgomery_formula(r_vals):
    """g(r) = 1 - (sin(pi*r)/(pi*r))^2"""
    g = np.ones_like(r_vals, dtype=float)
    mask = r_vals > 1e-10
    g[mask] = 1 - (np.sin(np.pi * r_vals[mask]) /
                   (np.pi * r_vals[mask])) ** 2
    g[~mask] = 0.0
    return g


def rms_vs_montgomery(g, r_vals):
    """Full range [0.01, 4.0], 200 points — Analysis 20 standard."""
    g_mont = montgomery_formula(r_vals)
    return float(np.sqrt(np.mean((g - g_mont) ** 2)))


# ============================================================================
# KS TESTS (exact Analysis 20)
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

def main():
    print("=" * 70)
    print("ANALYSIS 27: L=30 DEDEKIND COMPARISON")
    print("Pipeline: Analysis 20 KDE method, Riemann-von Mangoldt unfolding")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Build M at L=30, Phi=1/6
    # ==================================================================
    print("\n--- STEP 1: Build M operator (L=30, Phi=1/6) ---")
    L = 30
    Phi = 1.0 / 6
    t0 = time.time()

    torus = EisensteinTorus(L)
    u, v, omega = assign_spinors_geometric(torus)
    M = build_M(torus, u, v, omega, Phi, use_resonance=False)
    eigs = np.linalg.eigvalsh(M)

    print(f"  Torus L={L}, N={torus.num_nodes}, edges={len(torus.edges)}")
    print(f"  Eigenvalue range: [{eigs[0]:.6f}, {eigs[-1]:.6f}]")
    print(f"  n_positive = {np.sum(eigs > 0)}, n_negative = {np.sum(eigs < 0)}")

    # Hermiticity and complexity check
    herm_err = np.max(np.abs(M - M.conj().T))
    nonzero_mask = np.abs(M) > 1e-10
    real_frac = np.sum(np.abs(np.imag(M[nonzero_mask])) < 1e-10) / max(np.sum(nonzero_mask), 1)
    print(f"  Hermiticity error: {herm_err:.2e}")
    print(f"  Purely real entries: {100*real_frac:.1f}%")
    print(f"  Build time: {time.time()-t0:.1f}s")

    # Extract positive wing + unfold (Analysis 19/20 pipeline)
    unfolded_M, spacings_M = get_unfolded_wing(eigs)
    print(f"  Positive wing (>20th pct): {len(unfolded_M)} eigenvalues")

    # KS tests
    ks_M = ks_tests(spacings_M)
    print(f"  KS(GOE) = {ks_M['goe'][0]:.4f}, p = {ks_M['goe'][1]:.4f}")
    print(f"  KS(GUE) = {ks_M['gue'][0]:.4f}, p = {ks_M['gue'][1]:.4f}")
    print(f"  KS(Poi) = {ks_M['poi'][0]:.4f}, p = {ks_M['poi'][1]:.4f}")

    # ==================================================================
    # STEP 2: Get zero sets
    # ==================================================================
    print("\n--- STEP 2: Zero sets ---")

    # 2a. Riemann zeros — 1000
    riemann_cache = Path(r"C:\Users\selin\merkabit_results\montgomery_comparison\riemann_zeros_1000.npy")
    if riemann_cache.exists():
        riemann_gammas = np.load(riemann_cache)
        print(f"  Riemann: loaded {len(riemann_gammas)} cached zeros")
    else:
        print(f"  Computing 1000 Riemann zeros...")
        from mpmath import mp, zetazero
        mp.dps = 20
        t0 = time.time()
        riemann_gammas = np.array([float(zetazero(n).imag) for n in range(1, 1001)])
        print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Riemann range: [{riemann_gammas[0]:.4f}, {riemann_gammas[-1]:.4f}]")

    # Unfold with Riemann-von Mangoldt
    unfolded_riem = unfold_riemann_zeros(riemann_gammas)
    spacings_riem = np.diff(unfolded_riem)
    mean_s_riem = np.mean(spacings_riem)
    if mean_s_riem > 0:
        spacings_riem /= mean_s_riem
        unfolded_riem = (unfolded_riem - unfolded_riem[0]) / mean_s_riem

    # 2b. L(s,chi_{-3}) zeros — load from Analysis 18
    lchi3_cache = Path(r"C:\Users\selin\merkabit_results\lchi3_comparison\lchi3_zeros.npy")
    if lchi3_cache.exists():
        lchi3_gammas = np.load(lchi3_cache)
        print(f"  L(s,chi_-3): loaded {len(lchi3_gammas)} cached zeros")
    else:
        print("  ERROR: L-function zeros not found!")
        return
    print(f"  L-func range: [{lchi3_gammas[0]:.4f}, {lchi3_gammas[-1]:.4f}]")

    # Use first 500 for even comparison with Riemann
    lchi3_500 = lchi3_gammas[:500]

    # Unfold L-function zeros with conductor-corrected formula
    unfolded_lchi3 = unfold_lchi3_zeros(lchi3_500)
    spacings_lchi3 = np.diff(unfolded_lchi3)
    mean_s_lchi3 = np.mean(spacings_lchi3)
    if mean_s_lchi3 > 0:
        spacings_lchi3 /= mean_s_lchi3
        unfolded_lchi3 = (unfolded_lchi3 - unfolded_lchi3[0]) / mean_s_lchi3

    # 2c. Dedekind zeros — sorted union of first 500 Riemann + first 500 L-func
    riemann_500 = riemann_gammas[:500]
    dedekind_gammas = np.sort(np.concatenate([riemann_500, lchi3_500]))

    # Remove near-duplicates (within 0.01)
    unique_ded = [dedekind_gammas[0]]
    for z in dedekind_gammas[1:]:
        if z - unique_ded[-1] > 0.01:
            unique_ded.append(z)
    dedekind_gammas = np.array(unique_ded)

    print(f"  Dedekind: {len(dedekind_gammas)} zeros (500 Riemann + 500 L-func, deduped)")
    print(f"  Dedekind range: [{dedekind_gammas[0]:.4f}, {dedekind_gammas[-1]:.4f}]")

    # Unfold Dedekind with double-density formula
    unfolded_ded = unfold_dedekind_zeros(dedekind_gammas)
    spacings_ded = np.diff(unfolded_ded)
    mean_s_ded = np.mean(spacings_ded)
    if mean_s_ded > 0:
        spacings_ded /= mean_s_ded
        unfolded_ded = (unfolded_ded - unfolded_ded[0]) / mean_s_ded

    # Verify unfolding quality: mean spacing should be ~1
    print(f"\n  Unfolding verification (mean spacing, should be ~1.0):")
    sp_riem_check = np.diff(unfolded_riem)
    sp_lchi3_check = np.diff(unfolded_lchi3)
    sp_ded_check = np.diff(unfolded_ded)
    print(f"    Riemann:  {np.mean(sp_riem_check):.4f}")
    print(f"    L-func:   {np.mean(sp_lchi3_check):.4f}")
    print(f"    Dedekind: {np.mean(sp_ded_check):.4f}")

    # ==================================================================
    # STEP 3: Pair correlations (Analysis 20 KDE method)
    # ==================================================================
    print("\n--- STEP 3: Pair correlations (KDE, bandwidth=0.4) ---")
    r_vals = R_VALS

    print("  Computing M pair correlation...")
    t0 = time.time()
    g_M = pair_correlation_kde(unfolded_M, r_vals)
    print(f"    Done ({time.time()-t0:.1f}s, {len(unfolded_M)} eigenvalues)")

    print("  Computing Riemann pair correlation...")
    t0 = time.time()
    g_riem = pair_correlation_kde(unfolded_riem, r_vals)
    print(f"    Done ({time.time()-t0:.1f}s, {len(unfolded_riem)} zeros)")

    print("  Computing L(s,chi_-3) pair correlation...")
    t0 = time.time()
    g_lchi3 = pair_correlation_kde(unfolded_lchi3, r_vals)
    print(f"    Done ({time.time()-t0:.1f}s, {len(unfolded_lchi3)} zeros)")

    print("  Computing Dedekind pair correlation...")
    t0 = time.time()
    g_ded = pair_correlation_kde(unfolded_ded, r_vals)
    print(f"    Done ({time.time()-t0:.1f}s, {len(unfolded_ded)} zeros)")

    g_mont = montgomery_formula(r_vals)

    # RMS vs Montgomery
    rms_M = rms_vs_montgomery(g_M, r_vals)
    rms_riem = rms_vs_montgomery(g_riem, r_vals)
    rms_lchi3 = rms_vs_montgomery(g_lchi3, r_vals)
    rms_ded = rms_vs_montgomery(g_ded, r_vals)

    print(f"\n  Pair correlation RMS vs Montgomery:")
    print(f"    M operator (L=30, phi=1/6):  {rms_M:.4f}  (cf. Analysis 21: 0.118)")
    print(f"    Riemann zeros (N=1000):      {rms_riem:.4f}  (cf. Analysis 20: 0.082)")
    print(f"    L(s,chi_-3) zeros (N=500):   {rms_lchi3:.4f}")
    print(f"    Dedekind zeta_Q(w) (N={len(dedekind_gammas)}):  {rms_ded:.4f}")

    # Direct M vs each family
    rms_M_vs_riem = float(np.sqrt(np.mean((g_M - g_riem) ** 2)))
    rms_M_vs_lchi3 = float(np.sqrt(np.mean((g_M - g_lchi3) ** 2)))
    rms_M_vs_ded = float(np.sqrt(np.mean((g_M - g_ded) ** 2)))

    print(f"\n  Direct M vs each zero family:")
    print(f"    M vs Riemann:        {rms_M_vs_riem:.4f}")
    print(f"    M vs L(s,chi_-3):    {rms_M_vs_lchi3:.4f}")
    print(f"    M vs Dedekind:       {rms_M_vs_ded:.4f}")

    # Q1 answer
    print(f"\n  Q1 answer: RMS(M,Ded) < RMS(M,Riem)? ", end="")
    if rms_M_vs_ded < rms_M_vs_riem:
        print("YES")
        improvement = (rms_M_vs_riem - rms_M_vs_ded) / rms_M_vs_riem * 100
        print(f"    Improvement: {improvement:.1f}%")
    else:
        print("NO")
        deficit = (rms_M_vs_ded - rms_M_vs_riem) / rms_M_vs_riem * 100
        print(f"    Deficit: {deficit:.1f}%")

    # Ratio M/Riemann
    ratio_MR = rms_M / rms_riem if rms_riem > 0 else float('inf')
    print(f"\n  M/Riemann ratio (vs Montgomery): {ratio_MR:.3f}x  (cf. L=18: 1.49)")

    # ==================================================================
    # STEP 4: Correlation hole
    # ==================================================================
    print(f"\n--- STEP 4: Correlation hole ---")
    g_M_0 = g_M[0]
    g_riem_0 = g_riem[0]
    g_lchi3_0 = g_lchi3[0]
    g_ded_0 = g_ded[0]
    g_mont_0 = g_mont[0]

    print(f"  Correlation hole g(r=0.01):")
    print(f"    M operator:        {g_M_0:.4f}")
    print(f"    Riemann zeros:     {g_riem_0:.4f}")
    print(f"    L(s,chi_-3):       {g_lchi3_0:.4f}")
    print(f"    Dedekind:          {g_ded_0:.4f}")
    print(f"    Montgomery (theo): {g_mont_0:.4f}")

    # Mean g(r) for r < 0.3
    small_mask = r_vals < 0.3
    print(f"\n  Mean g(r) for r < 0.3:")
    print(f"    M operator:        {np.mean(g_M[small_mask]):.4f}")
    print(f"    Riemann:           {np.mean(g_riem[small_mask]):.4f}")
    print(f"    L(s,chi_-3):       {np.mean(g_lchi3[small_mask]):.4f}")
    print(f"    Dedekind:          {np.mean(g_ded[small_mask]):.4f}")
    print(f"    Montgomery:        {np.mean(g_mont[small_mask]):.4f}")

    # ==================================================================
    # STEP 5: KS tests on Riemann and L-func spacings
    # ==================================================================
    print(f"\n--- STEP 5: KS tests on zero family spacings ---")
    ks_riem = ks_tests(spacings_riem)
    print(f"  Riemann: KS(GOE)={ks_riem['goe'][0]:.4f} p={ks_riem['goe'][1]:.4f}, "
          f"KS(GUE)={ks_riem['gue'][0]:.4f} p={ks_riem['gue'][1]:.4f}")
    ks_lchi3 = ks_tests(spacings_lchi3)
    print(f"  L-func:  KS(GOE)={ks_lchi3['goe'][0]:.4f} p={ks_lchi3['goe'][1]:.4f}, "
          f"KS(GUE)={ks_lchi3['gue'][0]:.4f} p={ks_lchi3['gue'][1]:.4f}")
    ks_ded = ks_tests(spacings_ded)
    print(f"  Dedekind: KS(GOE)={ks_ded['goe'][0]:.4f} p={ks_ded['goe'][1]:.4f}, "
          f"KS(GUE)={ks_ded['gue'][0]:.4f} p={ks_ded['gue'][1]:.4f}")

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n--- Generating figures ---")

    # Figure 1: THE KEY FIGURE — M vs all zero families vs Montgomery
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(r_vals, g_M, 'b-', linewidth=2, label=f'M operator (L=30, RMS={rms_M:.4f})')
    ax.plot(r_vals, g_riem, 'r--', linewidth=1.5, label=f'Riemann (N=1000, RMS={rms_riem:.4f})')
    ax.plot(r_vals, g_lchi3, 'g-.', linewidth=1.5, label=f'L(s,chi_-3) (N=500, RMS={rms_lchi3:.4f})')
    ax.plot(r_vals, g_ded, 'm:', linewidth=2, label=f'Dedekind (N={len(dedekind_gammas)}, RMS={rms_ded:.4f})')
    ax.plot(r_vals, g_mont, 'k:', linewidth=2, alpha=0.5, label='Montgomery')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('r (units of mean spacing)', fontsize=12)
    ax.set_ylabel('g(r) pair correlation', fontsize=12)
    ax.set_title('Analysis 27: M (L=30) vs Zero Families vs Montgomery', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.5)
    ax.grid(True, alpha=0.3)

    # Right: residuals from Montgomery
    ax2 = axes[1]
    ax2.plot(r_vals, g_M - g_mont, 'b-', linewidth=1.5, label=f'M - Mont (RMS={rms_M:.4f})')
    ax2.plot(r_vals, g_riem - g_mont, 'r--', linewidth=1.5, label=f'Riem - Mont (RMS={rms_riem:.4f})')
    ax2.plot(r_vals, g_lchi3 - g_mont, 'g-.', linewidth=1.5, label=f'L-func - Mont (RMS={rms_lchi3:.4f})')
    ax2.plot(r_vals, g_ded - g_mont, 'm:', linewidth=2, label=f'Ded - Mont (RMS={rms_ded:.4f})')
    ax2.axhline(0, color='k', ls='-', alpha=0.3)
    ax2.set_xlabel('r', fontsize=12)
    ax2.set_ylabel('Residual g(r) - Montgomery', fontsize=12)
    ax2.set_title('Residuals from Montgomery Formula', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'analysis27_pair_correlation.png', dpi=200)
    print(f"  Saved analysis27_pair_correlation.png")
    plt.close()

    # Figure 2: M vs Riemann vs Dedekind (direct comparison for Q1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_vals, g_M - g_riem, 'r-', linewidth=2, label=f'M - Riemann (RMS={rms_M_vs_riem:.4f})')
    ax.plot(r_vals, g_M - g_ded, 'm--', linewidth=2, label=f'M - Dedekind (RMS={rms_M_vs_ded:.4f})')
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('r (units of mean spacing)', fontsize=12)
    ax.set_ylabel('Difference in g(r)', fontsize=12)
    ax.set_title(f'Q1: M closer to Riemann or Dedekind? (L=30)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'analysis27_Q1_comparison.png', dpi=200)
    print(f"  Saved analysis27_Q1_comparison.png")
    plt.close()

    # ==================================================================
    # SAVE DATA
    # ==================================================================
    print("\n--- Saving data ---")
    np.save(RESULTS_DIR / 'g_M_L30_A27.npy', g_M)
    np.save(RESULTS_DIR / 'g_riem_A27.npy', g_riem)
    np.save(RESULTS_DIR / 'g_lchi3_A27.npy', g_lchi3)
    np.save(RESULTS_DIR / 'g_ded_A27.npy', g_ded)
    np.save(RESULTS_DIR / 'eigs_L30_A27.npy', eigs)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 27 -- L=30 DEDEKIND COMPARISON")
    print("=" * 70)
    print(f"Pipeline: Analysis 20 KDE method, Riemann-von Mangoldt unfolding")
    print(f"\nM eigenvalues: {len(unfolded_M)} (positive wing of {torus.num_nodes})")
    print(f"Riemann zeros: 1000")
    print(f"L(s,chi_-3) zeros: 500")
    print(f"Dedekind zeros: {len(dedekind_gammas)} (combined)")
    print(f"\nPAIR CORRELATION RMS vs MONTGOMERY:")
    print(f"  M operator:   {rms_M:.4f}  (cf. Analysis 21: 0.118)")
    print(f"  Riemann:      {rms_riem:.4f}  (cf. Analysis 20: 0.082)")
    print(f"  L(s,chi_-3):  {rms_lchi3:.4f}")
    print(f"  Dedekind:     {rms_ded:.4f}")
    print(f"\nDIRECT M vs ZERO FAMILIES:")
    print(f"  M vs Riemann:   {rms_M_vs_riem:.4f}")
    print(f"  M vs L(s,chi_-3): {rms_M_vs_lchi3:.4f}")
    print(f"  M vs Dedekind:  {rms_M_vs_ded:.4f}")

    q1_answer = "YES" if rms_M_vs_ded < rms_M_vs_riem else "NO"
    if rms_M_vs_ded < rms_M_vs_riem:
        pct = (rms_M_vs_riem - rms_M_vs_ded) / rms_M_vs_riem * 100
        print(f"\nQ1: M fits Dedekind better than Riemann? {q1_answer}")
        print(f"    Improvement: {pct:.1f}%")
    else:
        pct = (rms_M_vs_ded - rms_M_vs_riem) / rms_M_vs_riem * 100
        print(f"\nQ1: M fits Dedekind better than Riemann? {q1_answer}")
        print(f"    Deficit: {pct:.1f}%")

    print(f"\nCORRELATION HOLE:")
    print(f"  g(r=0.01): M={g_M_0:.4f}, Riemann={g_riem_0:.4f}, Dedekind={g_ded_0:.4f}")
    print(f"\nCONVERGENCE (all L=30 results):")
    print(f"  KS(GUE):    {ks_M['gue'][0]:.4f}  (cf. L=18: 0.052)")
    print(f"  RMS/Riem:   {ratio_MR:.3f}  (cf. L=18: 1.49)")

    # Save summary
    summary_file = RESULTS_DIR / 'analysis27_L30.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSIS 27 -- L=30 DEDEKIND COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(f"Pipeline: Analysis 20 KDE method\n")
        f.write(f"\nM eigenvalues: {len(unfolded_M)}\n")
        f.write(f"Riemann zeros: 1000\n")
        f.write(f"L(s,chi_-3) zeros: 500\n")
        f.write(f"Dedekind zeros: {len(dedekind_gammas)}\n")
        f.write(f"\nPAIR CORRELATION RMS vs MONTGOMERY:\n")
        f.write(f"  M operator:   {rms_M:.4f}  (cf. Analysis 21: 0.118)\n")
        f.write(f"  Riemann:      {rms_riem:.4f}  (cf. Analysis 20: 0.082)\n")
        f.write(f"  L(s,chi_-3):  {rms_lchi3:.4f}\n")
        f.write(f"  Dedekind:     {rms_ded:.4f}\n")
        f.write(f"\nDIRECT M vs ZERO FAMILIES:\n")
        f.write(f"  M vs Riemann:    {rms_M_vs_riem:.4f}\n")
        f.write(f"  M vs L(s,chi_-3): {rms_M_vs_lchi3:.4f}\n")
        f.write(f"  M vs Dedekind:   {rms_M_vs_ded:.4f}\n")
        f.write(f"\nQ1: M fits Dedekind better than Riemann? {q1_answer}\n")
        if rms_M_vs_ded < rms_M_vs_riem:
            f.write(f"    Improvement: {pct:.1f}%\n")
        else:
            f.write(f"    Deficit: {pct:.1f}%\n")
        f.write(f"\nCORRELATION HOLE:\n")
        f.write(f"  g(r=0.01): M={g_M_0:.4f}, Riem={g_riem_0:.4f}, Ded={g_ded_0:.4f}\n")
        f.write(f"\nCONVERGENCE:\n")
        f.write(f"  KS(GUE):  {ks_M['gue'][0]:.4f}  (cf. L=18: 0.052)\n")
        f.write(f"  KS(GOE):  {ks_M['goe'][0]:.4f}\n")
        f.write(f"  RMS/Riem: {ratio_MR:.3f}\n")
        f.write(f"\nKS TESTS ON ZERO FAMILIES:\n")
        f.write(f"  Riemann:  KS(GUE)={ks_riem['gue'][0]:.4f} p={ks_riem['gue'][1]:.4f}\n")
        f.write(f"  L-func:   KS(GUE)={ks_lchi3['gue'][0]:.4f} p={ks_lchi3['gue'][1]:.4f}\n")
        f.write(f"  Dedekind: KS(GUE)={ks_ded['gue'][0]:.4f} p={ks_ded['gue'][1]:.4f}\n")

    print(f"\n  Saved: {summary_file.name}")
    print(f"\nAll output in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
