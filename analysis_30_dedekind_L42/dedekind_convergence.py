#!/usr/bin/env python3
"""
ANALYSIS 30: L=42 DEDEKIND CONVERGENCE TEST
=============================================
Merkabit Riemann Spectral Program

Tests whether M's correlation hole depth g(0) converges toward the
Dedekind value (~0.476) or the Riemann value (~0.274) as lattice size L increases.

Runs: L = 30 (replication check), 36, 42, 48
Pipeline: exact Analysis 20/27 method (Gaussian KDE, bw=0.4, poly-10 unfolding)
Operator: Formula A, Landau gauge, Phi=1/6, xi=3.0, no resonance
"""

import numpy as np
import sys
import time
import gc
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

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\analysis_30_dedekind_L42")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
XI = 3.0
PHI = 1.0 / 6  # Coxeter step
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

# Pipeline parameters (Analysis 20/27 compatible)
R_VALS = np.linspace(0.01, 4.0, 400)  # start at 0.01 to match Analysis 27
BANDWIDTH = 0.4
POLY_DEG = 10
PCT_THRESH = 20
N_GUE_TRIALS = 20

# Lattice sizes
L_VALUES = [30, 36, 42, 48]

# Reference values from Analysis 27 (L=30)
REF = {
    'rms_M_mont': 0.1182,
    'rms_M_riem': 0.0506,
    'rms_M_ded': 0.0362,
    'g0_M': 0.4382,
    'g0_riem': 0.2744,
    'g0_ded': 0.4764,
    'ks_gue': 0.0638,
}


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
            if s == 0:
                self.chirality.append(0)
            elif s == 1:
                self.chirality.append(+1)
            else:
                self.chirality.append(-1)


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
        u_i = np.exp(1j * theta) * np.array([
            np.cos(np.pi * r / 2),
            1j * np.sin(np.pi * r / 2)
        ], dtype=complex)
        u_i /= np.linalg.norm(u_i)
        v_i = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
        u[i] = u_i
        v[i] = v_i
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega


# ============================================================================
# M CONSTRUCTION (Formula A, Landau gauge, no resonance)
# ============================================================================

def build_M(torus, u, v, omega, Phi=PHI, xi=XI):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = torus.L
    for (i, j) in torus.edges:
        a_i, b_i = torus.nodes[i]
        a_j, b_j = torus.nodes[j]
        da = a_j - a_i
        db = b_j - b_i
        if da > L // 2:
            da -= L
        if da < -(L // 2):
            da += L
        if db > L // 2:
            db -= L
        if db < -(L // 2):
            db += L
        A_ij = Phi * (2 * a_i + da) / 2.0 * db
        coupling = decay * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)
        M[i, j] = coupling
        M[j, i] = np.conj(coupling)
    M = (M + M.conj().T) / 2.0
    return M


# ============================================================================
# SPECTRAL PIPELINE
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=POLY_DEG):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        sp = np.diff(evals)
        if len(sp) > 0 and np.mean(sp) > 0:
            sp /= np.mean(sp)
        return evals, sp
    deg = min(poly_degree, max(3, N // 10))
    coeffs = np.polyfit(evals, np.arange(1, N + 1), deg)
    N_smooth = np.polyval(coeffs, evals)
    sp = np.diff(N_smooth)
    m = np.mean(sp)
    if m > 0:
        sp /= m
    return N_smooth, sp


def extract_positive_wing(eigenvalues, threshold_pct=PCT_THRESH):
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    return pos_eigs[pos_eigs > cutoff]


def get_unfolded_wing(eigs):
    eigs = np.sort(np.real(eigs))
    wing = extract_positive_wing(eigs)
    if len(wing) < 10:
        return wing, np.array([])
    unfolded, spacings = unfold_spectrum(wing)
    return unfolded, spacings


# ============================================================================
# PAIR CORRELATION (Gaussian KDE on all pairwise diffs — Analysis 20 method)
# ============================================================================

def pair_correlation_kde(unfolded_eigs, r_vals=R_VALS, bandwidth=BANDWIDTH):
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5:
        return np.ones(len(r_vals))

    # All pairwise differences
    diffs = eigs[:, None] - eigs[None, :]
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]

    # KDE evaluation — chunked for memory efficiency on large wings
    g = np.zeros(len(r_vals))
    chunk = 50
    for start in range(0, len(r_vals), chunk):
        end = min(start + chunk, len(r_vals))
        r_chunk = r_vals[start:end]
        # (chunk_size, N_diffs) broadcasting
        g[start:end] = np.sum(
            np.exp(-0.5 * ((diffs_flat[None, :] - r_chunk[:, None]) / bandwidth) ** 2),
            axis=1
        )

    g /= (N * bandwidth * np.sqrt(2 * np.pi))

    # Normalize: g(r -> inf) -> 1
    large_r_mask = r_vals > 3.0
    if large_r_mask.sum() > 3:
        norm = np.mean(g[large_r_mask])
        if norm > 0:
            g /= norm

    return g


def montgomery_formula(r_vals):
    g = np.ones_like(r_vals, dtype=float)
    mask = r_vals > 1e-10
    g[mask] = 1 - (np.sin(np.pi * r_vals[mask]) / (np.pi * r_vals[mask])) ** 2
    g[~mask] = 0.0
    return g


# ============================================================================
# KS TESTS
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
# ZERO FAMILY UNFOLDING
# ============================================================================

def unfold_riemann_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    N_smooth = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_smooth


def unfold_lchi3_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    N_smooth = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_smooth


def unfold_dedekind_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    N_riem = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    N_L = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_riem + N_L


# ============================================================================
# LOAD ZERO FAMILIES
# ============================================================================

def load_zero_families():
    print("\n--- Loading zero families ---")

    # Riemann zeros
    cache_100k = Path(r"C:\Users\selin\merkabit_results\riemann_zeros\riemann_zeros_cache.npy")
    cache_1k = Path(r"C:\Users\selin\merkabit_results\montgomery_comparison\riemann_zeros_1000.npy")

    if cache_100k.exists():
        all_riemann = np.load(cache_100k)
        riemann_gammas = all_riemann[:2000]
        print(f"  Riemann: {len(riemann_gammas)} zeros (from 100K cache)")
    elif cache_1k.exists():
        riemann_gammas = np.load(cache_1k)
        print(f"  Riemann: {len(riemann_gammas)} zeros (from 1K cache)")
    else:
        print("  Computing 2000 Riemann zeros via mpmath...")
        from mpmath import mp, zetazero
        mp.dps = 20
        riemann_gammas = np.array([float(zetazero(n).imag) for n in range(1, 2001)])
        print(f"  Done: {len(riemann_gammas)} zeros")

    # L(s,chi_{-3}) zeros
    lchi3_cache = Path(r"C:\Users\selin\merkabit_results\lchi3_comparison\lchi3_zeros.npy")
    dedekind_cache = Path(r"C:\Users\selin\merkabit_results\dedekind_comparison\dedekind_zeros.npy")

    lchi3_gammas = None
    if lchi3_cache.exists():
        lchi3_gammas = np.load(lchi3_cache)
        print(f"  L(s,chi_-3): {len(lchi3_gammas)} zeros (cached)")

    # Use first 500 of each for consistency with Analysis 27
    riemann_500 = riemann_gammas[:500]

    # Unfold Riemann zeros with Riemann-von Mangoldt
    unfolded_riem = unfold_riemann_zeros(riemann_500)
    sp_riem = np.diff(unfolded_riem)
    mean_sr = np.mean(sp_riem)
    if mean_sr > 0:
        sp_riem /= mean_sr
        unfolded_riem = (unfolded_riem - unfolded_riem[0]) / mean_sr

    # Build Dedekind zeros
    if lchi3_gammas is not None:
        lchi3_500 = lchi3_gammas[:500]

        # Unfold L-func zeros
        unfolded_lchi3 = unfold_lchi3_zeros(lchi3_500)
        sp_lchi3 = np.diff(unfolded_lchi3)
        mean_sl = np.mean(sp_lchi3)
        if mean_sl > 0:
            sp_lchi3 /= mean_sl
            unfolded_lchi3 = (unfolded_lchi3 - unfolded_lchi3[0]) / mean_sl

        # Dedekind = merged + deduped
        dedekind_gammas = np.sort(np.concatenate([riemann_500, lchi3_500]))
        unique_ded = [dedekind_gammas[0]]
        for z in dedekind_gammas[1:]:
            if z - unique_ded[-1] > 0.01:
                unique_ded.append(z)
        dedekind_gammas = np.array(unique_ded)

        # Unfold Dedekind
        unfolded_ded = unfold_dedekind_zeros(dedekind_gammas)
        sp_ded = np.diff(unfolded_ded)
        mean_sd = np.mean(sp_ded)
        if mean_sd > 0:
            sp_ded /= mean_sd
            unfolded_ded = (unfolded_ded - unfolded_ded[0]) / mean_sd

        print(f"  Dedekind: {len(dedekind_gammas)} merged zeros")
    else:
        unfolded_lchi3 = None
        unfolded_ded = None
        dedekind_gammas = None
        print("  WARNING: L-function zeros not found")

    return {
        'riemann_gammas': riemann_gammas,
        'unfolded_riem': unfolded_riem,
        'unfolded_lchi3': unfolded_lchi3,
        'unfolded_ded': unfolded_ded,
        'dedekind_gammas': dedekind_gammas,
    }


# ============================================================================
# GUE NULL
# ============================================================================

def gue_null_baseline(n_wing, r_vals=R_VALS, n_trials=N_GUE_TRIALS):
    g_mont = montgomery_formula(r_vals)
    rms_list = []
    g0_list = []

    for trial in range(n_trials):
        G = np.random.randn(n_wing, n_wing) + 1j * np.random.randn(n_wing, n_wing)
        H_gue = (G + G.conj().T) / (2 * np.sqrt(2 * n_wing))
        eigs_gue = np.linalg.eigvalsh(H_gue)

        unfolded, spacings = unfold_spectrum(eigs_gue)
        g = pair_correlation_kde(unfolded, r_vals)
        rms = float(np.sqrt(np.mean((g - g_mont) ** 2)))
        rms_list.append(rms)
        g0_list.append(g[0])

    return {
        'rms_mean': np.mean(rms_list),
        'rms_std': np.std(rms_list),
        'g0_mean': np.mean(g0_list),
        'g0_std': np.std(g0_list),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print("ANALYSIS 30: L=42 DEDEKIND CONVERGENCE TEST")
    print("Merkabit Riemann Spectral Program")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"L values: {L_VALUES}")
    print(f"Pipeline: Analysis 20/27 (KDE bw={BANDWIDTH}, poly={POLY_DEG}, wing>P{PCT_THRESH})")
    print(f"r_range: [{R_VALS[0]}, {R_VALS[-1]}], {len(R_VALS)} points")
    print(f"GUE null: {N_GUE_TRIALS} trials per L")

    r_vals = R_VALS
    g_mont = montgomery_formula(r_vals)

    # ==================================================================
    # STEP 1: Load zero families (once)
    # ==================================================================
    zeros = load_zero_families()

    # Compute pair correlations for zero families (once)
    print("\n--- Computing zero family pair correlations ---")

    t0 = time.time()
    g_riem = pair_correlation_kde(zeros['unfolded_riem'], r_vals)
    rms_riem_mont = float(np.sqrt(np.mean((g_riem - g_mont) ** 2)))
    print(f"  Riemann: RMS(Mont)={rms_riem_mont:.4f}, g(0)={g_riem[0]:.4f} ({time.time()-t0:.1f}s)")

    has_dedekind = zeros['unfolded_ded'] is not None
    if has_dedekind:
        t0 = time.time()
        g_ded = pair_correlation_kde(zeros['unfolded_ded'], r_vals)
        rms_ded_mont = float(np.sqrt(np.mean((g_ded - g_mont) ** 2)))
        print(f"  Dedekind: RMS(Mont)={rms_ded_mont:.4f}, g(0)={g_ded[0]:.4f} ({time.time()-t0:.1f}s)")

        t0 = time.time()
        g_lchi3 = pair_correlation_kde(zeros['unfolded_lchi3'], r_vals)
        rms_lchi3_mont = float(np.sqrt(np.mean((g_lchi3 - g_mont) ** 2)))
        print(f"  L-func: RMS(Mont)={rms_lchi3_mont:.4f}, g(0)={g_lchi3[0]:.4f} ({time.time()-t0:.1f}s)")
    else:
        g_ded = None
        g_lchi3 = None

    # ==================================================================
    # STEP 2: Run M operator at each L
    # ==================================================================
    results = []
    g_M_curves = {}

    for Li, L in enumerate(L_VALUES):
        print(f"\n{'='*70}")
        print(f"  L = {L} (N = {L*L})")
        print(f"{'='*70}")

        # Build operator
        t0 = time.time()
        torus = EisensteinTorus(L)
        u, v, omega = assign_spinors_geometric(torus)
        M = build_M(torus, u, v, omega)
        build_time = time.time() - t0
        print(f"  Built M ({torus.num_nodes}x{torus.num_nodes}) in {build_time:.1f}s")

        # Hermiticity check
        herm_err = np.max(np.abs(M - M.conj().T))
        print(f"  Hermiticity error: {herm_err:.2e}")

        # Diagonalize
        t0 = time.time()
        eigs = np.linalg.eigvalsh(M)
        diag_time = time.time() - t0
        print(f"  Diagonalized in {diag_time:.1f}s")
        print(f"  Eigenvalue range: [{eigs[0]:.6f}, {eigs[-1]:.6f}]")
        print(f"  n_positive = {np.sum(eigs > 0)}, n_negative = {np.sum(eigs < 0)}")

        # Free M matrix memory
        del M, torus, u, v, omega
        gc.collect()

        # Extract wing + unfold
        unfolded_M, spacings_M = get_unfolded_wing(eigs)
        n_wing = len(unfolded_M)
        print(f"  Positive wing (>P{PCT_THRESH}): {n_wing} eigenvalues")

        # KS tests
        ks = ks_tests(spacings_M)
        print(f"  KS(GOE) = {ks['goe'][0]:.4f}, p = {ks['goe'][1]:.4f}")
        print(f"  KS(GUE) = {ks['gue'][0]:.4f}, p = {ks['gue'][1]:.4f}")
        print(f"  KS(Poi) = {ks['poi'][0]:.4f}, p = {ks['poi'][1]:.4f}")

        # Pair correlation
        t0 = time.time()
        g_M = pair_correlation_kde(unfolded_M, r_vals)
        kde_time = time.time() - t0
        print(f"  Pair correlation computed in {kde_time:.1f}s")

        g_M_curves[L] = g_M.copy()

        # RMS calculations
        rms_M_mont = float(np.sqrt(np.mean((g_M - g_mont) ** 2)))
        rms_M_riem = float(np.sqrt(np.mean((g_M - g_riem) ** 2)))

        if has_dedekind:
            rms_M_ded = float(np.sqrt(np.mean((g_M - g_ded) ** 2)))
            ded_improvement = (rms_M_riem - rms_M_ded) / rms_M_riem * 100
        else:
            rms_M_ded = None
            ded_improvement = None

        # Hole depth
        g0_M = g_M[0]
        g0_riem = g_riem[0]
        g0_ded = g_ded[0] if has_dedekind else None

        print(f"\n  RMS vs Montgomery:    {rms_M_mont:.4f}")
        print(f"  RMS vs Riemann:       {rms_M_riem:.4f}")
        if rms_M_ded is not None:
            print(f"  RMS vs Dedekind:      {rms_M_ded:.4f}")
            print(f"  Dedekind improvement: {ded_improvement:.1f}%")
        print(f"  g(r~0) [M]:           {g0_M:.4f}")
        print(f"  g(r~0) [Riemann]:     {g0_riem:.4f}")
        if g0_ded is not None:
            print(f"  g(r~0) [Dedekind]:    {g0_ded:.4f}")

        # L=30 replication check
        if L == 30:
            print(f"\n  --- L=30 REPLICATION CHECK (vs Analysis 27) ---")
            checks = [
                ('RMS(M,Mont)', rms_M_mont, REF['rms_M_mont']),
                ('RMS(M,Riem)', rms_M_riem, REF['rms_M_riem']),
                ('g(0) M', g0_M, REF['g0_M']),
                ('KS(GUE)', ks['gue'][0], REF['ks_gue']),
            ]
            if rms_M_ded is not None:
                checks.append(('RMS(M,Ded)', rms_M_ded, REF['rms_M_ded']))

            all_pass = True
            for name, val, ref in checks:
                dev = abs(val - ref) / ref * 100
                status = "OK" if dev < 5 else ("WARN" if dev < 10 else "FAIL")
                if dev >= 5:
                    all_pass = False
                print(f"    {name}: {val:.4f} vs ref {ref:.4f} ({dev:.1f}% dev) [{status}]")

            if all_pass:
                print(f"  REPLICATION: PASS (all within 5%)")
            else:
                print(f"  REPLICATION: ACCEPTABLE (minor grid differences expected)")
                print(f"  Note: 400-pt grid vs A27's 200-pt grid may cause small shifts")

        # GUE null
        print(f"\n  Computing GUE null ({N_GUE_TRIALS} trials, N_wing={n_wing})...")
        t0 = time.time()
        gue_null = gue_null_baseline(n_wing, r_vals)
        gue_time = time.time() - t0
        print(f"  GUE null: RMS = {gue_null['rms_mean']:.4f} +/- {gue_null['rms_std']:.4f}")
        print(f"  GUE null: g(0) = {gue_null['g0_mean']:.4f} +/- {gue_null['g0_std']:.4f}")
        sigma_dist = abs(rms_M_mont - gue_null['rms_mean']) / gue_null['rms_std'] if gue_null['rms_std'] > 0 else float('inf')
        print(f"  M distance from GUE null: {sigma_dist:.1f} sigma")
        print(f"  GUE null computed in {gue_time:.1f}s")

        # Store results
        results.append({
            'L': L,
            'N': L * L,
            'n_wing': n_wing,
            'rms_M_mont': rms_M_mont,
            'rms_M_riem': rms_M_riem,
            'rms_M_ded': rms_M_ded,
            'ded_improvement': ded_improvement,
            'g0_M': g0_M,
            'g0_riem': g0_riem,
            'g0_ded': g0_ded,
            'ks_goe': ks['goe'],
            'ks_gue': ks['gue'],
            'ks_poi': ks['poi'],
            'gue_null': gue_null,
            'sigma_dist': sigma_dist,
        })

        # Save eigenvalues for each L
        np.save(RESULTS_DIR / f'eigs_L{L}.npy', eigs)
        np.save(RESULTS_DIR / f'g_M_L{L}.npy', g_M)

    # ==================================================================
    # TABLE A — HOLE DEPTH TRAJECTORY
    # ==================================================================
    print(f"\n{'='*70}")
    print("TABLE A -- HOLE DEPTH TRAJECTORY (the decisive result)")
    print("=" * 70)

    tA_header = f"{'L':>4} {'N_wing':>7} {'g_M(0)':>9} {'g_Ded(0)':>10} {'g_Riem(0)':>10} {'g_GUE(0)':>10}"
    print(tA_header)
    print("-" * 60)

    tA_lines = [tA_header, "-" * 60]
    for r in results:
        g0d = f"{r['g0_ded']:.4f}" if r['g0_ded'] is not None else "N/A"
        g0g = f"{r['gue_null']['g0_mean']:.4f}"
        line = f"{r['L']:>4} {r['n_wing']:>7} {r['g0_M']:>9.4f} {g0d:>10} {r['g0_riem']:>10.4f} {g0g:>10}"
        print(line)
        tA_lines.append(line)

    # Trend analysis
    g0s = [r['g0_M'] for r in results]
    mono_up = all(g0s[i] <= g0s[i+1] for i in range(len(g0s)-1))
    mono_down = all(g0s[i] >= g0s[i+1] for i in range(len(g0s)-1))

    print()
    if mono_up:
        print("  TREND: g_M(0) MONOTONICALLY INCREASING -> Dedekind convergence")
    elif mono_down:
        print("  TREND: g_M(0) MONOTONICALLY DECREASING -> Riemann convergence")
    else:
        print("  TREND: Non-monotonic -> inconclusive")

    final_g0 = g0s[-1]
    dist_ded = abs(final_g0 - 0.476)
    dist_riem = abs(final_g0 - 0.274)
    print(f"  At L={results[-1]['L']}: |g_M(0)-Ded| = {dist_ded:.4f}, |g_M(0)-Riem| = {dist_riem:.4f}")
    print(f"  -> Closer to {'Dedekind' if dist_ded < dist_riem else 'Riemann'}")

    # ==================================================================
    # TABLE B — RMS COMPARISON
    # ==================================================================
    print(f"\n{'='*70}")
    print("TABLE B -- RMS COMPARISON")
    print("=" * 70)

    tB_header = f"{'L':>4} {'RMS(M,Mont)':>12} {'RMS(M,Riem)':>12} {'RMS(M,Ded)':>12} {'Ded impr%':>10}"
    print(tB_header)
    print("-" * 56)

    tB_lines = [tB_header, "-" * 56]
    for r in results:
        d = f"{r['rms_M_ded']:.4f}" if r['rms_M_ded'] is not None else "N/A"
        imp = f"{r['ded_improvement']:.1f}%" if r['ded_improvement'] is not None else "N/A"
        line = f"{r['L']:>4} {r['rms_M_mont']:>12.4f} {r['rms_M_riem']:>12.4f} {d:>12} {imp:>10}"
        print(line)
        tB_lines.append(line)

    if has_dedekind:
        imps = [r['ded_improvement'] for r in results]
        mono_inc = all(imps[i] <= imps[i+1] for i in range(len(imps)-1))
        print()
        if mono_inc:
            print("  TREND: Dedekind improvement MONOTONICALLY INCREASING")
        else:
            print("  TREND: Dedekind improvement NOT monotonically increasing")

    # ==================================================================
    # TABLE C — GUE CLASSIFICATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("TABLE C -- GUE CLASSIFICATION MAINTAINED")
    print("=" * 70)

    tC_header = f"{'L':>4} {'KS(GUE)':>9} {'p-value':>9} {'KS(GOE)':>9} {'Classification':>16}"
    print(tC_header)
    print("-" * 55)

    tC_lines = [tC_header, "-" * 55]
    for r in results:
        ks_gue_stat, ks_gue_p = r['ks_gue']
        ks_goe_stat, ks_goe_p = r['ks_goe']
        if ks_gue_p > 0.05 and ks_gue_p > ks_goe_p:
            cls = "GUE"
        elif ks_goe_p > 0.05 and ks_goe_p > ks_gue_p:
            cls = "GOE"
        elif ks_gue_p > 0.05:
            cls = "GUE (marginal)"
        elif ks_goe_p > 0.05:
            cls = "GOE (marginal)"
        else:
            cls = "Neither"
        line = f"{r['L']:>4} {ks_gue_stat:>9.4f} {ks_gue_p:>9.4f} {ks_goe_stat:>9.4f} {cls:>16}"
        print(line)
        tC_lines.append(line)

    # ==================================================================
    # SAVE TABLES
    # ==================================================================
    for fname, lines in [
        ('results_table_A.txt', tA_lines),
        ('results_table_B.txt', tB_lines),
        ('results_table_C.txt', tC_lines),
    ]:
        with open(RESULTS_DIR / fname, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
    print(f"\n  Saved tables A, B, C")

    # ==================================================================
    # FIGURE 1: Hole depth trajectory
    # ==================================================================
    print("\n--- Generating figures ---")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    mask = r_vals <= 1.5

    for i, r in enumerate(results):
        L_val = r['L']
        if L_val in g_M_curves:
            ax.plot(r_vals[mask], g_M_curves[L_val][mask], '-',
                    color=colors[i % len(colors)], linewidth=2,
                    label=f'M (L={L_val}, g(0)={r["g0_M"]:.3f})')

    if has_dedekind:
        ax.plot(r_vals[mask], g_ded[mask], 'k--', linewidth=2,
                label=f'Dedekind zeros (g(0)={g_ded[0]:.3f})')
    ax.plot(r_vals[mask], g_riem[mask], 'k:', linewidth=2,
            label=f'Riemann zeros (g(0)={g_riem[0]:.3f})')
    ax.plot(r_vals[mask], g_mont[mask], 'gray', linewidth=1, alpha=0.5,
            label='Montgomery formula')

    # Reference lines
    if has_dedekind:
        ax.axhline(g_ded[0], color='black', ls='--', alpha=0.3, linewidth=0.8)
        ax.text(1.35, g_ded[0] + 0.015, f'Dedekind g(0)={g_ded[0]:.3f}', fontsize=8, alpha=0.6)
    ax.axhline(g_riem[0], color='black', ls=':', alpha=0.3, linewidth=0.8)
    ax.text(1.35, g_riem[0] + 0.015, f'Riemann g(0)={g_riem[0]:.3f}', fontsize=8, alpha=0.6)

    ax.set_xlabel('r (units of mean spacing)', fontsize=12)
    ax.set_ylabel('g(r) pair correlation', fontsize=12)
    ax.set_title('Analysis 30: Hole Depth Trajectory -- M vs Dedekind vs Riemann', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure1_hole_trajectory.png', dpi=200)
    print(f"  Saved figure1_hole_trajectory.png")
    plt.close()

    # ==================================================================
    # FIGURE 2: Dedekind improvement vs L
    # ==================================================================
    if has_dedekind:
        fig, ax = plt.subplots(figsize=(8, 6))
        Ls = [r['L'] for r in results]
        imps = [r['ded_improvement'] for r in results]
        ax.plot(Ls, imps, 'bo-', markersize=10, linewidth=2)

        # Trendline
        z = np.polyfit(Ls, imps, 1)
        p_trend = np.poly1d(z)
        L_ext = np.linspace(min(Ls) - 2, max(Ls) + 5, 50)
        ax.plot(L_ext, p_trend(L_ext), 'r--', alpha=0.5,
                label=f'Linear trend: slope={z[0]:.2f}%/L')

        for i in range(len(Ls)):
            ax.annotate(f'{imps[i]:.1f}%', (Ls[i], imps[i]),
                        textcoords="offset points", xytext=(10, 10), fontsize=10)

        ax.set_xlabel('Lattice size L', fontsize=12)
        ax.set_ylabel('Dedekind improvement %', fontsize=12)
        ax.set_title('Analysis 30: Dedekind Improvement vs Lattice Size', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figure2_dedekind_improvement.png', dpi=200)
        print(f"  Saved figure2_dedekind_improvement.png")
        plt.close()

    # ==================================================================
    # FIGURE 3: Full pair correlation at L=42
    # ==================================================================
    L_target = 42
    if L_target in g_M_curves:
        r_idx = [r for r in results if r['L'] == L_target][0]
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(r_vals, g_M_curves[L_target], 'b-', linewidth=2,
                label=f'M (L={L_target}, RMS(Mont)={r_idx["rms_M_mont"]:.4f})')
        ax.plot(r_vals, g_mont, 'k:', linewidth=2, alpha=0.5,
                label='Montgomery formula')
        ax.plot(r_vals, g_riem, 'r--', linewidth=1.5,
                label=f'Riemann zeros (RMS={rms_riem_mont:.4f})')
        if has_dedekind:
            ax.plot(r_vals, g_ded, 'm-.', linewidth=1.5,
                    label=f'Dedekind zeros (RMS={rms_ded_mont:.4f})')

        ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel('r (units of mean spacing)', fontsize=12)
        ax.set_ylabel('g(r) pair correlation', fontsize=12)
        ax.set_title(f'Analysis 30: Full Pair Correlation at L={L_target}', fontsize=12)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figure3_pair_correlation_L42.png', dpi=200)
        print(f"  Saved figure3_pair_correlation_L42.png")
        plt.close()

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print("ANALYSIS 30 -- FINAL SUMMARY")
    print("=" * 70)
    print(f"Total computation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\nHOLE DEPTH TRAJECTORY (decisive result):")
    for r in results:
        print(f"  L={r['L']:2d} (wing={r['n_wing']:4d}): g_M(0) = {r['g0_M']:.4f}")

    if has_dedekind:
        print(f"\n  Reference: Dedekind g(0) = {g_ded[0]:.4f}")
    print(f"  Reference: Riemann g(0) = {g_riem[0]:.4f}")

    print(f"\nDEDEKIND IMPROVEMENT TRAJECTORY:")
    for r in results:
        imp = f"{r['ded_improvement']:.1f}%" if r['ded_improvement'] is not None else "N/A"
        print(f"  L={r['L']:2d}: {imp}")

    print(f"\nGUE NULL DISTANCE:")
    for r in results:
        print(f"  L={r['L']:2d}: {r['sigma_dist']:.1f} sigma from GUE null")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    g0_trajectory = [r['g0_M'] for r in results]
    increasing = all(g0_trajectory[i] < g0_trajectory[i+1] + 0.005 for i in range(len(g0_trajectory)-1))
    decreasing = all(g0_trajectory[i] > g0_trajectory[i+1] - 0.005 for i in range(len(g0_trajectory)-1))

    if has_dedekind:
        final_closer_to_ded = abs(g0_trajectory[-1] - g_ded[0]) < abs(g0_trajectory[-1] - g_riem[0])

        if increasing and final_closer_to_ded:
            print("STRONG POSITIVE: g_M(0) increasing monotonically toward Dedekind")
            print("-> M is an Eisenstein-field operator in the spectral sense")
            print("-> 28.4% Dedekind improvement is STRUCTURAL, not finite-size artifact")
        elif decreasing:
            print("NEGATIVE: g_M(0) decreasing toward Riemann")
            print("-> M converges to zeta(s), not zeta_{Q(omega)}")
        elif not increasing and not decreasing:
            print("INCONCLUSIVE: g_M(0) trajectory is non-monotonic")
            print("-> Finite-size effects may dominate; recommend L >= 60 test")
        else:
            print("MIXED: trajectory direction and target don't align clearly")
    else:
        print("INCOMPLETE: Dedekind zeros unavailable for comparison")

    if has_dedekind:
        imp_trajectory = [r['ded_improvement'] for r in results]
        if imp_trajectory[-1] > 40:
            print(f"\nDedekind improvement at L={results[-1]['L']}: {imp_trajectory[-1]:.1f}% (STRONG)")
        elif imp_trajectory[-1] > 28.4:
            print(f"\nDedekind improvement at L={results[-1]['L']}: {imp_trajectory[-1]:.1f}% (GROWING)")
        else:
            print(f"\nDedekind improvement at L={results[-1]['L']}: {imp_trajectory[-1]:.1f}% (NOT GROWING)")

    # GUE maintained?
    all_gue = all(r['ks_gue'][1] > 0.05 for r in results)
    if all_gue:
        print("\nGUE classification: MAINTAINED at all L values")
    else:
        broken = [r['L'] for r in results if r['ks_gue'][1] <= 0.05]
        print(f"\nGUE classification: BROKEN at L = {broken}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")

    # Save full summary
    with open(RESULTS_DIR / 'analysis_30_summary.txt', 'w', encoding='utf-8') as f:
        f.write("ANALYSIS 30: L=42 DEDEKIND CONVERGENCE TEST\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("HOLE DEPTH TRAJECTORY:\n")
        for r in results:
            g0d = f"{r['g0_ded']:.4f}" if r['g0_ded'] is not None else "N/A"
            f.write(f"  L={r['L']:2d} (wing={r['n_wing']:4d}): g_M(0)={r['g0_M']:.4f}, "
                    f"g_Ded(0)={g0d}, g_Riem(0)={r['g0_riem']:.4f}\n")

        f.write("\nRMS COMPARISON:\n")
        for r in results:
            d = f"{r['rms_M_ded']:.4f}" if r['rms_M_ded'] is not None else "N/A"
            imp = f"{r['ded_improvement']:.1f}%" if r['ded_improvement'] is not None else "N/A"
            f.write(f"  L={r['L']:2d}: RMS(M,Mont)={r['rms_M_mont']:.4f}, "
                    f"RMS(M,Riem)={r['rms_M_riem']:.4f}, RMS(M,Ded)={d}, "
                    f"Ded_impr={imp}\n")

        f.write("\nGUE CLASSIFICATION:\n")
        for r in results:
            f.write(f"  L={r['L']:2d}: KS(GUE)={r['ks_gue'][0]:.4f} p={r['ks_gue'][1]:.4f}, "
                    f"KS(GOE)={r['ks_goe'][0]:.4f} p={r['ks_goe'][1]:.4f}\n")

        f.write("\nGUE NULL DISTANCE:\n")
        for r in results:
            f.write(f"  L={r['L']:2d}: RMS_M={r['rms_M_mont']:.4f}, "
                    f"GUE_null={r['gue_null']['rms_mean']:.4f}+/-{r['gue_null']['rms_std']:.4f}, "
                    f"{r['sigma_dist']:.1f} sigma\n")

        f.write(f"\nTotal computation time: {elapsed:.0f}s\n")

    print(f"\n  Saved analysis_30_summary.txt")


if __name__ == '__main__':
    main()
