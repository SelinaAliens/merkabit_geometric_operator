#!/usr/bin/env python3
"""
ANALYSIS 32: FULL MONTGOMERY PIPELINE AT THE RESONANT ISLAND (L=36-39)
======================================================================
Merkabit Riemann Spectral Program

Runs the complete Analysis 20/27 pair-correlation pipeline at every size
in the GUE resonance island L = 36, 37, 38, 39, plus anchor sizes L = 18
and L = 30 for convergence comparison.

Key questions answered:
  1. Does RMS(M, Riemann) improve at resonant sizes vs L=30?
  2. Is GUE strength anti-correlated with Riemann correspondence
     within the island?
  3. What is the full spectral profile across L = h*xi to (h+1)*xi?

Operator: Formula A, Landau gauge, Phi=1/6, xi=3.0, no resonance.
Pipeline: exact Analysis 20/27 (Gaussian KDE, bw=0.4, poly-10 unfolding).
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
from scipy import stats as sp_stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\analysis_32_montgomery_L38")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

XI = 3.0
PHI = 1.0 / 6
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

R_VALS = np.linspace(0.01, 4.0, 400)
BANDWIDTH = 0.4
POLY_DEG = 10
PCT_THRESH = 20
N_GUE_TRIALS = 50

# All sizes: anchors + full island
ALL_L = [18, 30, 36, 37, 38, 39]
ISLAND_L = [36, 37, 38, 39]

# Reference values
REF_L38_KS_GUE = 0.045
REF_L38_P_GUE = 0.206
REF_L30_RMS_RIEM = 0.0511
REF_L30_RMS_MONT = 0.118


# ============================================================================
# OPERATOR CONSTRUCTION — identical to Analysis 20/27/30/31
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


# ============================================================================
# SPECTRAL PIPELINE — full version, identical to Analysis 20/27
# ============================================================================

def unfold_spectrum(eigenvalues):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        sp = np.diff(evals)
        if len(sp) > 0 and np.mean(sp) > 0: sp /= np.mean(sp)
        return evals, sp
    deg = min(POLY_DEG, max(3, N // 10))
    coeffs = np.polyfit(evals, np.arange(1, N + 1), deg)
    N_smooth = np.polyval(coeffs, evals)
    sp = np.diff(N_smooth)
    m = np.mean(sp)
    if m > 0: sp /= m
    return N_smooth, sp


def extract_positive_wing(eigenvalues):
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4: return pos_eigs
    cutoff = np.percentile(pos_eigs, PCT_THRESH)
    return pos_eigs[pos_eigs > cutoff]


def get_unfolded_wing(eigs):
    eigs = np.sort(np.real(eigs))
    wing = extract_positive_wing(eigs)
    if len(wing) < 10: return wing, np.array([])
    unfolded, spacings = unfold_spectrum(wing)
    return unfolded, spacings


def pair_correlation_kde(unfolded_eigs, r_vals=R_VALS):
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5: return np.ones(len(r_vals))
    diffs = eigs[:, None] - eigs[None, :]
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]
    g = np.zeros(len(r_vals))
    chunk = 50
    for start in range(0, len(r_vals), chunk):
        end = min(start + chunk, len(r_vals))
        r_chunk = r_vals[start:end]
        g[start:end] = np.sum(
            np.exp(-0.5 * ((diffs_flat[None, :] - r_chunk[:, None]) / BANDWIDTH) ** 2),
            axis=1)
    g /= (N * BANDWIDTH * np.sqrt(2 * np.pi))
    large_r_mask = r_vals > 3.0
    if large_r_mask.sum() > 3:
        norm = np.mean(g[large_r_mask])
        if norm > 0: g /= norm
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
        ks_stat, p = sp_stats.kstest(pos, cdf)
        results[name] = (float(ks_stat), float(p))
    ks_stat, p = sp_stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (float(ks_stat), float(p))
    return results


# ============================================================================
# ZERO FAMILIES
# ============================================================================

def unfold_riemann_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0

def unfold_lchi3_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    return T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0

def unfold_dedekind_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    N_r = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    N_L = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_r + N_L


def load_and_process_zeros():
    """Load Riemann, L-func, Dedekind zeros; compute their pair correlations."""
    cache_100k = Path(r"C:\Users\selin\merkabit_results\riemann_zeros\riemann_zeros_cache.npy")
    cache_1k = Path(r"C:\Users\selin\merkabit_results\montgomery_comparison\riemann_zeros_1000.npy")
    if cache_100k.exists():
        riemann_gammas = np.load(cache_100k)[:2000]
    elif cache_1k.exists():
        riemann_gammas = np.load(cache_1k)
    else:
        from mpmath import mp, zetazero
        mp.dps = 20
        riemann_gammas = np.array([float(zetazero(n).imag) for n in range(1, 2001)])

    lchi3_cache = Path(r"C:\Users\selin\merkabit_results\lchi3_comparison\lchi3_zeros.npy")
    lchi3_gammas = np.load(lchi3_cache) if lchi3_cache.exists() else None

    # Process Riemann (first 500)
    r500 = riemann_gammas[:500]
    uf_r = unfold_riemann_zeros(r500)
    sp_r = np.diff(uf_r)
    m_r = np.mean(sp_r)
    if m_r > 0:
        sp_r /= m_r
        uf_r = (uf_r - uf_r[0]) / m_r

    # Process Dedekind
    uf_d = None
    if lchi3_gammas is not None:
        l500 = lchi3_gammas[:500]
        ded_raw = np.sort(np.concatenate([r500, l500]))
        unique = [ded_raw[0]]
        for z in ded_raw[1:]:
            if z - unique[-1] > 0.01: unique.append(z)
        ded_gammas = np.array(unique)
        uf_d = unfold_dedekind_zeros(ded_gammas)
        sp_d = np.diff(uf_d)
        m_d = np.mean(sp_d)
        if m_d > 0:
            sp_d /= m_d
            uf_d = (uf_d - uf_d[0]) / m_d

    return uf_r, uf_d


# ============================================================================
# GUE NULL BASELINE
# ============================================================================

def gue_null_baseline(n_wing, r_vals, g_mont, n_trials=N_GUE_TRIALS):
    rms_list = []
    g0_list = []
    g_curves = []
    for _ in range(n_trials):
        G = np.random.randn(n_wing, n_wing) + 1j * np.random.randn(n_wing, n_wing)
        H = (G + G.conj().T) / (2 * np.sqrt(2 * n_wing))
        eigs = np.linalg.eigvalsh(H)
        uf, sp = unfold_spectrum(eigs)
        g = pair_correlation_kde(uf, r_vals)
        rms = float(np.sqrt(np.mean((g - g_mont) ** 2)))
        rms_list.append(rms)
        g0_list.append(g[0])
        g_curves.append(g)
    return {
        'rms_mean': np.mean(rms_list), 'rms_std': np.std(rms_list),
        'g0_mean': np.mean(g0_list), 'g0_std': np.std(g0_list),
        'g_mean': np.mean(g_curves, axis=0), 'g_std': np.std(g_curves, axis=0),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    total_t0 = time.time()

    print("=" * 78)
    print("ANALYSIS 32: FULL MONTGOMERY PIPELINE — RESONANT ISLAND L=36-39")
    print("Merkabit Riemann Spectral Program")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print(f"Sizes: {ALL_L}  (island: {ISLAND_L})")
    print(f"Pipeline: Analysis 20/27 (KDE bw={BANDWIDTH}, poly-{POLY_DEG}, wing>P{PCT_THRESH})")
    print(f"GUE null: {N_GUE_TRIALS} trials per size")

    r_vals = R_VALS
    g_mont = montgomery_formula(r_vals)

    # ==========================================================
    # STEP 1: Zero families (computed once)
    # ==========================================================
    print("\n--- Loading zero families ---")
    uf_riem, uf_ded = load_and_process_zeros()
    g_riem = pair_correlation_kde(uf_riem, r_vals)
    rms_riem_mont = float(np.sqrt(np.mean((g_riem - g_mont) ** 2)))
    print(f"  Riemann: RMS(Mont)={rms_riem_mont:.4f}, g(0)={g_riem[0]:.4f}")

    has_ded = uf_ded is not None
    if has_ded:
        g_ded = pair_correlation_kde(uf_ded, r_vals)
        rms_ded_mont = float(np.sqrt(np.mean((g_ded - g_mont) ** 2)))
        print(f"  Dedekind: RMS(Mont)={rms_ded_mont:.4f}, g(0)={g_ded[0]:.4f}")
    else:
        g_ded = None

    # Riemann RMS sanity check
    if abs(rms_riem_mont - 0.082) > 0.010:
        print(f"  WARNING: Riemann RMS={rms_riem_mont:.4f} differs from expected 0.082")

    # ==========================================================
    # STEP 2: Build M and run full pipeline at each L
    # ==========================================================
    results = {}
    g_M_curves = {}

    for L in ALL_L:
        print(f"\n{'='*78}")
        print(f"  L = {L}  (N = {L*L}, L/xi = {L/XI:.2f}, L/h = {L/12:.3f})")
        print(f"{'='*78}")

        t0 = time.time()
        torus = EisensteinTorus(L)
        u, v, omega = assign_spinors_geometric(torus)
        M_mat = build_M(torus, u, v, omega)
        eigs = np.linalg.eigvalsh(M_mat)
        del M_mat, torus, u, v, omega
        gc.collect()
        build_t = time.time() - t0
        print(f"  Built + diagonalised in {build_t:.1f}s")
        print(f"  Eigenvalue range: [{eigs[0]:.6f}, {eigs[-1]:.6f}]")

        # Wing + unfold
        unfolded_M, spacings_M = get_unfolded_wing(eigs)
        n_wing = len(unfolded_M)
        print(f"  Positive wing (>P{PCT_THRESH}): {n_wing}")

        # KS tests
        ks = ks_tests(spacings_M)
        gue_s, gue_p = ks['gue']
        goe_s, goe_p = ks['goe']
        cls = "GUE" if gue_p > 0.05 and gue_p > goe_p else ("GOE" if goe_p > 0.05 else "Neither")
        print(f"  KS(GUE) = {gue_s:.4f}, p = {gue_p:.4f}")
        print(f"  KS(GOE) = {goe_s:.4f}, p = {goe_p:.4f}")
        print(f"  Classification: {cls}")

        # L=38 replication check
        if L == 38:
            dev_ks = abs(gue_s - REF_L38_KS_GUE) / REF_L38_KS_GUE * 100
            dev_p = abs(gue_p - REF_L38_P_GUE) / REF_L38_P_GUE * 100
            print(f"  Replication: KS={gue_s:.4f} vs ref {REF_L38_KS_GUE} ({dev_ks:.1f}% dev)")
            print(f"               p={gue_p:.4f} vs ref {REF_L38_P_GUE} ({dev_p:.1f}% dev)")
            if dev_ks > 10 or dev_p > 20:
                print(f"  *** WARNING: L=38 replication deviation exceeds threshold ***")

        # Pair correlation
        t0 = time.time()
        g_M = pair_correlation_kde(unfolded_M, r_vals)
        kde_t = time.time() - t0
        print(f"  Pair correlation: {kde_t:.1f}s")

        g_M_curves[L] = g_M.copy()

        # RMS calculations
        rms_M_mont = float(np.sqrt(np.mean((g_M - g_mont) ** 2)))
        rms_M_riem = float(np.sqrt(np.mean((g_M - g_riem) ** 2)))
        rms_M_ded = float(np.sqrt(np.mean((g_M - g_ded) ** 2))) if has_ded else None
        g0_M = g_M[0]

        print(f"  RMS(M, Montgomery): {rms_M_mont:.4f}")
        print(f"  RMS(M, Riemann):    {rms_M_riem:.4f}")
        if rms_M_ded is not None:
            print(f"  RMS(M, Dedekind):   {rms_M_ded:.4f}")
        print(f"  g_M(0):             {g0_M:.4f}")

        # GUE null
        print(f"  GUE null ({N_GUE_TRIALS} trials, N={n_wing})...", end="", flush=True)
        t0 = time.time()
        gue_null = gue_null_baseline(n_wing, r_vals, g_mont)
        gue_t = time.time() - t0
        sigma = abs(rms_M_mont - gue_null['rms_mean']) / gue_null['rms_std'] if gue_null['rms_std'] > 0 else float('inf')
        print(f" {gue_t:.0f}s")
        print(f"  GUE null RMS: {gue_null['rms_mean']:.4f} +/- {gue_null['rms_std']:.4f}")
        print(f"  GUE null g(0): {gue_null['g0_mean']:.4f} +/- {gue_null['g0_std']:.4f}")
        print(f"  M distance: {sigma:.1f} sigma above GUE null")

        # Store
        results[L] = {
            'L': L, 'N': L*L, 'n_wing': n_wing,
            'ks_gue': (gue_s, gue_p), 'ks_goe': (goe_s, goe_p),
            'cls': cls,
            'rms_M_mont': rms_M_mont, 'rms_M_riem': rms_M_riem,
            'rms_M_ded': rms_M_ded,
            'g0_M': g0_M, 'g0_riem': g_riem[0],
            'g0_ded': g_ded[0] if has_ded else None,
            'gue_null': gue_null, 'sigma': sigma,
            'spacings': spacings_M,
        }
        np.save(RESULTS_DIR / f'eigs_L{L}.npy', eigs)
        np.save(RESULTS_DIR / f'g_M_L{L}.npy', g_M)

    # ==========================================================
    # PRIMARY OUTPUT TABLE
    # ==========================================================
    print(f"\n{'='*78}")
    print("PRIMARY OUTPUT TABLE")
    print("=" * 78)

    hdr = (f"{'L':>4} {'wing':>5} {'KS(GUE)':>8} {'p(GUE)':>7} {'cls':>5} "
           f"{'RMS(Mont)':>10} {'RMS(Riem)':>10} {'RMS(Ded)':>9} "
           f"{'g_M(0)':>7} {'sigma':>6}")
    print(hdr)
    print("-" * 82)
    for L in ALL_L:
        r = results[L]
        ded_s = f"{r['rms_M_ded']:.4f}" if r['rms_M_ded'] is not None else "N/A"
        print(f"{L:>4} {r['n_wing']:>5} {r['ks_gue'][0]:>8.4f} {r['ks_gue'][1]:>7.4f} {r['cls']:>5} "
              f"{r['rms_M_mont']:>10.4f} {r['rms_M_riem']:>10.4f} {ded_s:>9} "
              f"{r['g0_M']:>7.4f} {r['sigma']:>6.1f}")

    print(f"\n  Reference: RMS(Riemann, Montgomery) = {rms_riem_mont:.4f}")
    print(f"  Reference: g_Riemann(0) = {g_riem[0]:.4f}")
    if has_ded:
        print(f"  Reference: g_Dedekind(0) = {g_ded[0]:.4f}")

    # ==========================================================
    # CONVERGENCE COMPARISON
    # ==========================================================
    print(f"\n{'='*78}")
    print("CONVERGENCE vs L=30")
    print("=" * 78)
    r30 = results[30]
    for L in [18, 36, 37, 38, 39]:
        r = results[L]
        delta_mont = r['rms_M_mont'] - r30['rms_M_mont']
        delta_riem = r['rms_M_riem'] - r30['rms_M_riem']
        label_m = "BETTER" if delta_mont < -0.001 else ("WORSE" if delta_mont > 0.001 else "SAME")
        label_r = "BETTER" if delta_riem < -0.001 else ("WORSE" if delta_riem > 0.001 else "SAME")
        print(f"  L={L:2d}: RMS(Mont)={r['rms_M_mont']:.4f} ({label_m:6s} by {abs(delta_mont):.4f}), "
              f"RMS(Riem)={r['rms_M_riem']:.4f} ({label_r:6s} by {abs(delta_riem):.4f})")

    # ==========================================================
    # ADDITION 2: ANTI-CORRELATION TEST
    # ==========================================================
    print(f"\n{'='*78}")
    print("ANTI-CORRELATION TEST: p(GUE) vs RMS(M, Riemann) within island")
    print("=" * 78)

    island_p_gue = [results[L]['ks_gue'][1] for L in ISLAND_L]
    island_rms_riem = [results[L]['rms_M_riem'] for L in ISLAND_L]
    island_rms_mont = [results[L]['rms_M_mont'] for L in ISLAND_L]
    island_g0 = [results[L]['g0_M'] for L in ISLAND_L]

    print(f"\n  Island profile:")
    print(f"  {'L':>4} {'p(GUE)':>8} {'RMS(Riem)':>10} {'RMS(Mont)':>10} {'g_M(0)':>8}")
    print(f"  {'-'*42}")
    for i, L in enumerate(ISLAND_L):
        print(f"  {L:>4} {island_p_gue[i]:>8.4f} {island_rms_riem[i]:>10.4f} "
              f"{island_rms_mont[i]:>10.4f} {island_g0[i]:>8.4f}")

    # Pearson correlations
    if len(ISLAND_L) >= 4:
        corr_riem, p_corr_riem = sp_stats.pearsonr(island_p_gue, island_rms_riem)
        corr_mont, p_corr_mont = sp_stats.pearsonr(island_p_gue, island_rms_mont)
        corr_g0, p_corr_g0 = sp_stats.pearsonr(island_p_gue, island_g0)

        print(f"\n  Pearson correlation within island (N=4):")
        print(f"    p(GUE) vs RMS(M, Riemann):    r = {corr_riem:+.4f}  (p = {p_corr_riem:.4f})")
        print(f"    p(GUE) vs RMS(M, Montgomery):  r = {corr_mont:+.4f}  (p = {p_corr_mont:.4f})")
        print(f"    p(GUE) vs g_M(0):              r = {corr_g0:+.4f}  (p = {p_corr_g0:.4f})")

        if corr_riem > 0.5:
            print(f"\n  RESULT: POSITIVE correlation (r={corr_riem:+.3f})")
            print(f"  -> Stronger GUE => WORSE Riemann match (ANTI-CORRELATION CONFIRMED)")
            print(f"  -> Symmetry-breaking and arithmetic correspondence are COMPETING")
        elif corr_riem < -0.5:
            print(f"\n  RESULT: NEGATIVE correlation (r={corr_riem:+.3f})")
            print(f"  -> Stronger GUE => BETTER Riemann match (ALIGNED)")
        else:
            print(f"\n  RESULT: WEAK correlation (r={corr_riem:+.3f})")
            print(f"  -> No clear relationship within the island")

        # Spearman rank (robust to N=4)
        rs_riem, ps_riem = sp_stats.spearmanr(island_p_gue, island_rms_riem)
        print(f"\n  Spearman rank: rho = {rs_riem:+.4f} (p = {ps_riem:.4f})")
    else:
        corr_riem = None
        print("  (Not enough island points for correlation)")

    # ==========================================================
    # FIGURES
    # ==========================================================
    print(f"\n--- Generating figures ---")

    # ----------------------------------------------------------
    # Figure 1: Pair correlation g(r) at L=38
    # ----------------------------------------------------------
    r38 = results[38]
    gue38 = r38['gue_null']

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(r_vals, g_M_curves[38], 'b-', linewidth=2.5,
            label=f'M (L=38, RMS(Mont)={r38["rms_M_mont"]:.4f})')
    ax.plot(r_vals, g_mont, 'k--', linewidth=2, alpha=0.6, label='Montgomery formula')
    ax.plot(r_vals, g_riem, 'r-', linewidth=1.5, alpha=0.8,
            label=f'Riemann zeros (RMS(Mont)={rms_riem_mont:.4f})')
    ax.fill_between(r_vals,
                     gue38['g_mean'] - gue38['g_std'],
                     gue38['g_mean'] + gue38['g_std'],
                     color='gray', alpha=0.2, label=f'GUE null (N=577, 50 trials)')
    ax.plot(r_vals, gue38['g_mean'], 'gray', linewidth=1, alpha=0.5)

    # g(0) markers
    ax.plot(0.01, r38['g0_M'], 'bo', markersize=12, zorder=5)
    ax.plot(0.01, g_riem[0], 'ro', markersize=12, zorder=5)
    ax.annotate(f'g_M(0) = {r38["g0_M"]:.3f}', (0.01, r38['g0_M']),
                xytext=(0.3, r38['g0_M'] + 0.02), fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    ax.annotate(f'g_Riem(0) = {g_riem[0]:.3f}', (0.01, g_riem[0]),
                xytext=(0.3, g_riem[0] - 0.05), fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('r (units of mean spacing)', fontsize=13)
    ax.set_ylabel('g(r) pair correlation', fontsize=13)
    ax.set_title('Analysis 32: Pair Correlation at L=38 (Peak GUE, Resonant Island)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 1.4)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure1_pair_correlation_L38.png', dpi=200)
    print(f"  Saved figure1_pair_correlation_L38.png")
    plt.close()

    # ----------------------------------------------------------
    # Figure 2: RMS convergence across GUE sizes
    # ----------------------------------------------------------
    gue_Ls = [18, 30, 36, 37, 38, 39]
    fig, ax = plt.subplots(figsize=(10, 6))

    rms_riem_vals = [results[L]['rms_M_riem'] for L in gue_Ls]
    rms_mont_vals = [results[L]['rms_M_mont'] for L in gue_Ls]

    ax.plot(gue_Ls, rms_riem_vals, 'ro-', markersize=10, linewidth=2,
            label='RMS(M, Riemann zeros)', zorder=5)
    ax.plot(gue_Ls, rms_mont_vals, 'bs-', markersize=8, linewidth=1.5,
            label='RMS(M, Montgomery)', zorder=4)
    ax.axhline(rms_riem_mont, color='red', ls=':', linewidth=1.5, alpha=0.5,
               label=f'Riemann self-RMS = {rms_riem_mont:.4f} (asymptotic target)')

    # Mark island
    ax.axvspan(35.5, 39.5, alpha=0.08, color='green', label='Resonant island (h*xi)')

    for i, L in enumerate(gue_Ls):
        ax.annotate(f'{rms_riem_vals[i]:.4f}', (L, rms_riem_vals[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color='red')

    ax.set_xlabel('Lattice size L (GUE-confirmed)', fontsize=12)
    ax.set_ylabel('RMS', fontsize=12)
    ax.set_title('Analysis 32: RMS(M, Riemann) Convergence across GUE Sizes', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure2_rms_convergence.png', dpi=200)
    print(f"  Saved figure2_rms_convergence.png")
    plt.close()

    # ----------------------------------------------------------
    # Figure 3: Hole depth convergence
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    g0_vals = [results[L]['g0_M'] for L in gue_Ls]
    ax.plot(gue_Ls, g0_vals, 'bo-', markersize=10, linewidth=2,
            label='g_M(0)', zorder=5)
    ax.axhline(g_riem[0], color='red', ls='--', linewidth=1.5,
               label=f'Riemann g(0) = {g_riem[0]:.3f}')
    ax.axhline(results[38]['gue_null']['g0_mean'], color='gray', ls='--', linewidth=1.5,
               label=f'GUE null g(0) = {results[38]["gue_null"]["g0_mean"]:.3f}')
    if has_ded:
        ax.axhline(g_ded[0], color='purple', ls=':', linewidth=1,
                   label=f'Dedekind g(0) = {g_ded[0]:.3f}')

    ax.axvspan(35.5, 39.5, alpha=0.08, color='green')

    for i, L in enumerate(gue_Ls):
        ax.annotate(f'{g0_vals[i]:.3f}', (L, g0_vals[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel('Lattice size L (GUE-confirmed)', fontsize=12)
    ax.set_ylabel('g_M(0) — correlation hole depth', fontsize=12)
    ax.set_title('Analysis 32: Hole Depth across GUE Sizes', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure3_hole_depth_convergence.png', dpi=200)
    print(f"  Saved figure3_hole_depth_convergence.png")
    plt.close()

    # ----------------------------------------------------------
    # Figure 4: Spacing distribution at L=38
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    sp38 = results[38]['spacings']
    sp_pos = sp38[sp38 > 0]
    ax.hist(sp_pos, bins=np.arange(0, 4.2, 0.2), density=True, alpha=0.5,
            color='skyblue', edgecolor='black', linewidth=0.5, label='M (L=38) spacings')

    s_th = np.linspace(0, 4, 500)
    # GOE Wigner surmise
    goe_pdf = (np.pi / 2) * s_th * np.exp(-np.pi * s_th**2 / 4)
    ax.plot(s_th, goe_pdf, 'r--', linewidth=2, label='GOE Wigner surmise')
    # GUE Wigner surmise
    gue_pdf = (32 / np.pi**2) * s_th**2 * np.exp(-4 * s_th**2 / np.pi)
    ax.plot(s_th, gue_pdf, 'b-', linewidth=2, label='GUE Wigner surmise')

    ax.set_xlabel('s (normalised spacing)', fontsize=12)
    ax.set_ylabel('P(s)', fontsize=12)
    ax.set_title(f'Analysis 32: Spacing Distribution at L=38 (KS(GUE) p={results[38]["ks_gue"][1]:.3f})', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure4_spacing_distribution_L38.png', dpi=200)
    print(f"  Saved figure4_spacing_distribution_L38.png")
    plt.close()

    # ----------------------------------------------------------
    # Figure 5 (Addition): Island profile — anti-correlation
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: dual-axis island profile
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(ISLAND_L, island_p_gue, 'b^-', markersize=12, linewidth=2, label='p(GUE)')
    ax1_twin.plot(ISLAND_L, island_rms_riem, 'rs-', markersize=12, linewidth=2, label='RMS(M, Riem)')

    ax1.set_xlabel('Lattice size L (resonant island)', fontsize=12)
    ax1.set_ylabel('p(GUE)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('RMS(M, Riemann)', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax1.set_title('Island Profile: GUE Strength vs Riemann Correspondence', fontsize=11)
    ax1.grid(True, alpha=0.2)

    # Right: scatter plot
    ax2 = axes[1]
    ax2.scatter(island_p_gue, island_rms_riem, s=150, c=[36, 37, 38, 39],
                cmap='coolwarm', edgecolors='black', linewidth=1, zorder=5)
    for i, L in enumerate(ISLAND_L):
        ax2.annotate(f'L={L}', (island_p_gue[i], island_rms_riem[i]),
                     textcoords="offset points", xytext=(8, 8), fontsize=11, fontweight='bold')

    if corr_riem is not None:
        # Regression line
        z = np.polyfit(island_p_gue, island_rms_riem, 1)
        p_line = np.poly1d(z)
        x_ext = np.linspace(min(island_p_gue) - 0.02, max(island_p_gue) + 0.02, 50)
        ax2.plot(x_ext, p_line(x_ext), 'k--', alpha=0.5,
                 label=f'r = {corr_riem:+.3f} (Pearson)')

    ax2.set_xlabel('p(GUE)', fontsize=12)
    ax2.set_ylabel('RMS(M, Riemann)', fontsize=12)
    ax2.set_title('Anti-Correlation: Symmetry vs Arithmetic', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure5_island_anticorrelation.png', dpi=200)
    print(f"  Saved figure5_island_anticorrelation.png")
    plt.close()

    # ==========================================================
    # SAVE TEXT OUTPUTS
    # ==========================================================
    # Full results
    with open(RESULTS_DIR / 'results_L38_full.txt', 'w', encoding='utf-8') as f:
        f.write("ANALYSIS 32 — FULL MONTGOMERY PIPELINE: RESONANT ISLAND L=36-39\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for L in ALL_L:
            r = results[L]
            f.write(f"L = {L}  (N = {r['N']}, wing = {r['n_wing']})\n")
            f.write(f"  KS(GUE) = {r['ks_gue'][0]:.4f}, p = {r['ks_gue'][1]:.4f}\n")
            f.write(f"  KS(GOE) = {r['ks_goe'][0]:.4f}, p = {r['ks_goe'][1]:.4f}\n")
            f.write(f"  Class:    {r['cls']}\n")
            f.write(f"  RMS(M, Montgomery):  {r['rms_M_mont']:.4f}\n")
            f.write(f"  RMS(M, Riemann):     {r['rms_M_riem']:.4f}\n")
            ded_s = f"{r['rms_M_ded']:.4f}" if r['rms_M_ded'] is not None else "N/A"
            f.write(f"  RMS(M, Dedekind):    {ded_s}\n")
            f.write(f"  g_M(0):              {r['g0_M']:.4f}\n")
            f.write(f"  GUE null RMS:        {r['gue_null']['rms_mean']:.4f} +/- {r['gue_null']['rms_std']:.4f}\n")
            f.write(f"  GUE null g(0):       {r['gue_null']['g0_mean']:.4f} +/- {r['gue_null']['g0_std']:.4f}\n")
            f.write(f"  sigma above null:    {r['sigma']:.1f}\n\n")

        f.write(f"Reference: RMS(Riemann, Montgomery) = {rms_riem_mont:.4f}\n")
        f.write(f"Reference: g_Riemann(0) = {g_riem[0]:.4f}\n")
        if has_ded:
            f.write(f"Reference: g_Dedekind(0) = {g_ded[0]:.4f}\n")

        f.write(f"\nANTI-CORRELATION TEST:\n")
        if corr_riem is not None:
            f.write(f"  Pearson r(p_GUE, RMS_Riem) = {corr_riem:+.4f}  (p = {p_corr_riem:.4f})\n")
            f.write(f"  Spearman rho              = {rs_riem:+.4f}  (p = {ps_riem:.4f})\n")

    # Convergence table
    with open(RESULTS_DIR / 'convergence_table.txt', 'w', encoding='utf-8') as f:
        f.write("CONVERGENCE TABLE — ALL GUE-CONFIRMED SIZES\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'L':>4} {'wing':>5} {'RMS(Mont)':>10} {'RMS(Riem)':>10} {'g_M(0)':>8} {'sigma':>7} {'Source':>10}\n")
        f.write("-" * 60 + "\n")
        sources = {18: 'An.19/32', 30: 'An.27/32', 36: 'An.32', 37: 'An.32', 38: 'An.32', 39: 'An.32'}
        for L in gue_Ls:
            r = results[L]
            f.write(f"{L:>4} {r['n_wing']:>5} {r['rms_M_mont']:>10.4f} {r['rms_M_riem']:>10.4f} "
                    f"{r['g0_M']:>8.4f} {r['sigma']:>7.1f} {sources[L]:>10}\n")

    print(f"\n  Saved results_L38_full.txt")
    print(f"  Saved convergence_table.txt")

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================
    elapsed = time.time() - total_t0

    print(f"\n{'='*78}")
    print("ANALYSIS 32 — FINAL SUMMARY")
    print("=" * 78)
    print(f"Computation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n--- CONVERGENCE TABLE ---")
    print(f"{'L':>4} {'RMS(Mont)':>10} {'RMS(Riem)':>10} {'g_M(0)':>8} {'sigma':>7}")
    for L in gue_Ls:
        r = results[L]
        print(f"{L:>4} {r['rms_M_mont']:>10.4f} {r['rms_M_riem']:>10.4f} "
              f"{r['g0_M']:>8.4f} {r['sigma']:>7.1f}")

    print(f"\n--- ISLAND PROFILE ---")
    best_riem_L = ISLAND_L[np.argmin(island_rms_riem)]
    worst_riem_L = ISLAND_L[np.argmax(island_rms_riem)]
    peak_gue_L = ISLAND_L[np.argmax(island_p_gue)]

    print(f"  Peak GUE:            L={peak_gue_L} (p={max(island_p_gue):.4f})")
    print(f"  Best Riemann match:  L={best_riem_L} (RMS={min(island_rms_riem):.4f})")
    print(f"  Worst Riemann match: L={worst_riem_L} (RMS={max(island_rms_riem):.4f})")

    if peak_gue_L == worst_riem_L:
        print(f"\n  CONFIRMED: Peak GUE coincides with worst Riemann match")
    if corr_riem is not None and corr_riem > 0.5:
        print(f"  CONFIRMED: Anti-correlation r = {corr_riem:+.3f}")
        print(f"  -> Symmetry-breaking and arithmetic correspondence are COMPETING")

    # Verdict
    print(f"\n--- VERDICT ---")
    r30_rms = results[30]['rms_M_riem']
    best_island_rms = min(island_rms_riem)
    if best_island_rms < r30_rms * 0.95:
        print(f"  RMS(M, Riem) at L={best_riem_L}: {best_island_rms:.4f} < {r30_rms:.4f} (L=30)")
        print(f"  -> CONVERGENCE toward zeta(s) at resonant scale CONFIRMED")
    elif best_island_rms < r30_rms * 1.05:
        print(f"  RMS(M, Riem) at L={best_riem_L}: {best_island_rms:.4f} ~ {r30_rms:.4f} (L=30)")
        print(f"  -> FLAT: stable Riemann affinity, no convergence")
    else:
        print(f"  RMS(M, Riem) at L={best_riem_L}: {best_island_rms:.4f} > {r30_rms:.4f} (L=30)")
        print(f"  -> Island does not improve Riemann correspondence uniformly")

    print(f"\n  All outputs: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
