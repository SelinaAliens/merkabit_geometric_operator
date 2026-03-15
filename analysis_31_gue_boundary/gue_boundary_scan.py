#!/usr/bin/env python3
"""
ANALYSIS 30 — PART 2: GUE BOUNDARY & CROSSOVER CHARACTERISATION
================================================================

Priority 1: GUE boundary scan at L = 31, 33, 35, 37, 39, 41
Priority 3: Fine crossover scan L = 30..48 step 1
Priority 2: Dedekind test at confirmed GUE-class sizes

Reuses exact Analysis 20/27 pipeline from dedekind_convergence.py
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
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\analysis_30_dedekind_L42")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
XI = 3.0
PHI = 1.0 / 6
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

# Pipeline parameters
R_VALS = np.linspace(0.01, 4.0, 400)
BANDWIDTH = 0.4
POLY_DEG = 10
PCT_THRESH = 20


# ============================================================================
# EISENSTEIN TORUS (identical to dedekind_convergence.py)
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


def pair_correlation_kde(unfolded_eigs, r_vals=R_VALS, bandwidth=BANDWIDTH):
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5:
        return np.ones(len(r_vals))
    diffs = eigs[:, None] - eigs[None, :]
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]
    g = np.zeros(len(r_vals))
    chunk = 50
    for start in range(0, len(r_vals), chunk):
        end = min(start + chunk, len(r_vals))
        r_chunk = r_vals[start:end]
        g[start:end] = np.sum(
            np.exp(-0.5 * ((diffs_flat[None, :] - r_chunk[:, None]) / bandwidth) ** 2),
            axis=1
        )
    g /= (N * bandwidth * np.sqrt(2 * np.pi))
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
# ZERO FAMILY LOADING (for Dedekind test)
# ============================================================================

def unfold_riemann_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0


def unfold_lchi3_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    return T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0


def unfold_dedekind_zeros(gammas):
    T = np.asarray(gammas, dtype=float)
    N_riem = T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0
    N_L = T / (2 * np.pi) * np.log(3 * T / (2 * np.pi * np.e)) + 7.0 / 8.0
    return N_riem + N_L


def load_zero_families():
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

    riemann_500 = riemann_gammas[:500]
    unfolded_riem = unfold_riemann_zeros(riemann_500)
    sp_riem = np.diff(unfolded_riem)
    mean_sr = np.mean(sp_riem)
    if mean_sr > 0:
        sp_riem /= mean_sr
        unfolded_riem = (unfolded_riem - unfolded_riem[0]) / mean_sr

    if lchi3_gammas is not None:
        lchi3_500 = lchi3_gammas[:500]
        unfolded_lchi3 = unfold_lchi3_zeros(lchi3_500)
        sp_lchi3 = np.diff(unfolded_lchi3)
        mean_sl = np.mean(sp_lchi3)
        if mean_sl > 0:
            sp_lchi3 /= mean_sl
            unfolded_lchi3 = (unfolded_lchi3 - unfolded_lchi3[0]) / mean_sl

        dedekind_gammas = np.sort(np.concatenate([riemann_500, lchi3_500]))
        unique_ded = [dedekind_gammas[0]]
        for z in dedekind_gammas[1:]:
            if z - unique_ded[-1] > 0.01:
                unique_ded.append(z)
        dedekind_gammas = np.array(unique_ded)

        unfolded_ded = unfold_dedekind_zeros(dedekind_gammas)
        sp_ded = np.diff(unfolded_ded)
        mean_sd = np.mean(sp_ded)
        if mean_sd > 0:
            sp_ded /= mean_sd
            unfolded_ded = (unfolded_ded - unfolded_ded[0]) / mean_sd
    else:
        unfolded_lchi3 = None
        unfolded_ded = None

    return {
        'unfolded_riem': unfolded_riem,
        'unfolded_lchi3': unfolded_lchi3,
        'unfolded_ded': unfolded_ded,
    }


# ============================================================================
# COMPUTE M SPECTRAL STATS AT GIVEN L
# ============================================================================

def compute_M_stats(L, verbose=True):
    """Build M, diagonalise, return KS stats + wing info."""
    t0 = time.time()
    torus = EisensteinTorus(L)
    u, v, omega = assign_spinors_geometric(torus)
    M_mat = build_M(torus, u, v, omega)
    eigs = np.linalg.eigvalsh(M_mat)
    del M_mat, torus, u, v, omega
    gc.collect()

    unfolded, spacings = get_unfolded_wing(eigs)
    n_wing = len(unfolded)
    ks = ks_tests(spacings)
    elapsed = time.time() - t0

    if verbose:
        gue_s, gue_p = ks['gue']
        goe_s, goe_p = ks['goe']
        cls = "GUE" if (gue_p > 0.05 and gue_p > goe_p) else ("GOE" if goe_p > 0.05 else "Neither")
        print(f"  L={L:2d}  N={L*L:5d}  wing={n_wing:4d}  "
              f"KS(GUE)={gue_s:.4f} p={gue_p:.4f}  "
              f"KS(GOE)={goe_s:.4f} p={goe_p:.4f}  "
              f"[{cls}]  ({elapsed:.1f}s)")

    return {
        'L': L,
        'N': L * L,
        'n_wing': n_wing,
        'ks_gue': ks['gue'],
        'ks_goe': ks['goe'],
        'ks_poi': ks['poi'],
        'eigs': eigs,
        'unfolded': unfolded,
        'spacings': spacings,
        'elapsed': elapsed,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    total_start = time.time()

    print("=" * 78)
    print("ANALYSIS 30 — PART 2: GUE BOUNDARY & CROSSOVER CHARACTERISATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    r_vals = R_VALS
    g_mont = montgomery_formula(r_vals)

    # ==================================================================
    # PRIORITY 1: GUE boundary scan at L = 31, 33, 35, 37, 39, 41
    # ==================================================================
    print("\n" + "=" * 78)
    print("PRIORITY 1: GUE BOUNDARY SCAN")
    print("Testing L = 31, 33, 35, 37, 39, 41 (avoiding multiples of h=12)")
    print("=" * 78)

    priority1_Ls = [31, 33, 35, 37, 39, 41]
    p1_results = []
    for L in priority1_Ls:
        res = compute_M_stats(L)
        p1_results.append(res)

    # Also include L=30 and L=42 from Part 1 for context
    print("\n  Reference points from Part 1:")
    ref30 = compute_M_stats(30)
    ref42 = compute_M_stats(42)

    # Summary
    all_p1 = [ref30] + p1_results + [ref42]
    all_p1.sort(key=lambda x: x['L'])

    print(f"\n  --- GUE BOUNDARY SUMMARY ---")
    gue_sizes = []
    goe_sizes = []
    for r in all_p1:
        gue_p = r['ks_gue'][1]
        goe_p = r['ks_goe'][1]
        if gue_p > 0.05 and gue_p > goe_p:
            cls = "GUE"
            gue_sizes.append(r['L'])
        elif goe_p > 0.05:
            cls = "GOE"
            goe_sizes.append(r['L'])
        else:
            cls = "Neither"
            goe_sizes.append(r['L'])  # lump with GOE
        print(f"    L={r['L']:2d}: KS(GUE) p={gue_p:.4f}, KS(GOE) p={goe_p:.4f} -> {cls}")

    print(f"\n  GUE-class sizes: {gue_sizes}")
    print(f"  GOE-class sizes: {goe_sizes}")

    if len(gue_sizes) > 0:
        last_gue = max(gue_sizes)
        print(f"  Last GUE-class size: L = {last_gue}")
    else:
        print(f"  No GUE-class sizes found in scan range")

    # ==================================================================
    # PRIORITY 3: Fine crossover scan L = 30..48 step 1
    # ==================================================================
    print("\n" + "=" * 78)
    print("PRIORITY 3: FINE CROSSOVER SCAN L = 30..48")
    print("=" * 78)

    fine_Ls = list(range(30, 49))
    fine_results = {}

    # Reuse already-computed results
    already_done = {r['L']: r for r in all_p1}
    for L in fine_Ls:
        if L in already_done:
            fine_results[L] = already_done[L]
        else:
            res = compute_M_stats(L)
            fine_results[L] = res

    # Crossover table
    print(f"\n  --- FINE CROSSOVER TABLE ---")
    print(f"  {'L':>4} {'N_wing':>6} {'KS(GUE)':>8} {'p(GUE)':>8} {'KS(GOE)':>8} {'p(GOE)':>8} {'Class':>8} {'L mod 3':>7} {'L mod 6':>7} {'L mod 12':>8}")
    print(f"  {'-'*85}")

    crossover_data = []
    for L in fine_Ls:
        r = fine_results[L]
        gue_s, gue_p = r['ks_gue']
        goe_s, goe_p = r['ks_goe']
        if gue_p > 0.05 and gue_p > goe_p:
            cls = "GUE"
        elif goe_p > 0.05:
            cls = "GOE"
        elif gue_p > goe_p:
            cls = "gue?"
        else:
            cls = "goe?"
        print(f"  {L:>4} {r['n_wing']:>6} {gue_s:>8.4f} {gue_p:>8.4f} {goe_s:>8.4f} {goe_p:>8.4f} {cls:>8} {L%3:>7} {L%6:>7} {L%12:>8}")
        crossover_data.append({
            'L': L, 'n_wing': r['n_wing'],
            'ks_gue_stat': gue_s, 'ks_gue_p': gue_p,
            'ks_goe_stat': goe_s, 'ks_goe_p': goe_p,
            'cls': cls,
        })

    # Find crossover point
    Ls_arr = np.array([d['L'] for d in crossover_data])
    gue_ps = np.array([d['ks_gue_p'] for d in crossover_data])
    goe_ps = np.array([d['ks_goe_p'] for d in crossover_data])

    # Crossover = where gue_p and goe_p swap dominance
    gue_dominant = gue_ps > goe_ps
    transitions = []
    for i in range(len(gue_dominant) - 1):
        if gue_dominant[i] != gue_dominant[i + 1]:
            transitions.append((Ls_arr[i], Ls_arr[i + 1]))

    print(f"\n  Classification transitions:")
    for L1, L2 in transitions:
        print(f"    {L1} -> {L2}")

    # Check for patterns
    gue_class_Ls = [d['L'] for d in crossover_data if d['cls'] == 'GUE']
    goe_class_Ls = [d['L'] for d in crossover_data if d['cls'] in ('GOE', 'goe?')]

    # Modular analysis
    print(f"\n  --- MODULAR STRUCTURE ANALYSIS ---")
    for mod in [2, 3, 4, 6, 12]:
        print(f"\n  mod {mod}:")
        for residue in range(mod):
            Ls_with_res = [d['L'] for d in crossover_data if d['L'] % mod == residue]
            gue_count = sum(1 for L in Ls_with_res if L in gue_class_Ls)
            total = len(Ls_with_res)
            pct = 100 * gue_count / total if total > 0 else 0
            label = f"L mod {mod} = {residue}"
            print(f"    {label:18s}: {gue_count}/{total} GUE ({pct:.0f}%)  Ls={Ls_with_res}")

    # Correlation with xi = 3
    print(f"\n  --- DECAY LENGTH ANALYSIS ---")
    print(f"  xi = {XI}, L/xi ratio at crossover:")
    for d in crossover_data:
        ratio = d['L'] / XI
        print(f"    L={d['L']:2d}: L/xi = {ratio:.2f}, class = {d['cls']}")

    # ==================================================================
    # FIGURE 4: Crossover plot
    # ==================================================================
    print("\n--- Generating crossover figures ---")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: KS p-values
    ax = axes[0]
    ax.plot(Ls_arr, gue_ps, 'b.-', linewidth=1.5, markersize=8, label='p(GUE)')
    ax.plot(Ls_arr, goe_ps, 'r.-', linewidth=1.5, markersize=8, label='p(GOE)')
    ax.axhline(0.05, color='gray', ls='--', alpha=0.5, label='p = 0.05 threshold')
    ax.fill_between(Ls_arr, 0, 0.05, alpha=0.05, color='red')

    # Mark special L values
    for L_special, label, color in [
        (30, 'L=30 (5h/2)', 'green'),
        (36, 'L=36 (3h)', 'orange'),
        (42, 'L=42 (7h/2)', 'purple'),
        (48, 'L=48 (4h)', 'brown'),
    ]:
        ax.axvline(L_special, color=color, ls=':', alpha=0.4, linewidth=1)
        ax.text(L_special + 0.2, max(gue_ps) * 0.9, label, fontsize=7,
                color=color, rotation=90, va='top')

    # Mark multiples of 3
    for L_m3 in range(30, 49, 3):
        ax.axvline(L_m3, color='gray', ls=':', alpha=0.15)

    ax.set_ylabel('KS p-value', fontsize=12)
    ax.set_title('Analysis 30 Part 2: GUE/GOE Crossover — KS p-values vs L', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, max(max(gue_ps), max(goe_ps)) * 1.15)
    ax.grid(True, alpha=0.2)

    # Bottom: KS statistics
    ax2 = axes[1]
    gue_stats = np.array([d['ks_gue_stat'] for d in crossover_data])
    goe_stats = np.array([d['ks_goe_stat'] for d in crossover_data])
    ax2.plot(Ls_arr, gue_stats, 'b.-', linewidth=1.5, markersize=8, label='KS stat (GUE)')
    ax2.plot(Ls_arr, goe_stats, 'r.-', linewidth=1.5, markersize=8, label='KS stat (GOE)')

    # Colour background by classification
    for i, d in enumerate(crossover_data):
        if d['cls'] == 'GUE':
            ax2.axvspan(d['L'] - 0.4, d['L'] + 0.4, alpha=0.15, color='blue')
        elif d['cls'] in ('GOE', 'goe?'):
            ax2.axvspan(d['L'] - 0.4, d['L'] + 0.4, alpha=0.10, color='red')

    ax2.set_xlabel('Lattice size L', fontsize=12)
    ax2.set_ylabel('KS statistic', fontsize=12)
    ax2.set_title('KS Statistics with Classification (blue=GUE, red=GOE)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure4_gue_crossover.png', dpi=200)
    print(f"  Saved figure4_gue_crossover.png")
    plt.close()

    # ==================================================================
    # PRIORITY 2: Dedekind test at confirmed GUE-class sizes
    # ==================================================================
    print("\n" + "=" * 78)
    print("PRIORITY 2: DEDEKIND TEST AT CONFIRMED GUE-CLASS SIZES")
    print("=" * 78)

    # Load zero families
    print("  Loading zero families...")
    zeros = load_zero_families()
    g_riem = pair_correlation_kde(zeros['unfolded_riem'], r_vals)
    has_ded = zeros['unfolded_ded'] is not None
    if has_ded:
        g_ded = pair_correlation_kde(zeros['unfolded_ded'], r_vals)
    else:
        g_ded = None

    # Identify all GUE-class L values above 18
    gue_test_Ls = sorted([d['L'] for d in crossover_data if d['cls'] == 'GUE'])
    # Also include some borderline cases (p > 0.04)
    borderline_Ls = sorted([d['L'] for d in crossover_data
                            if d['cls'] != 'GUE' and d['ks_gue_p'] > 0.04
                            and d['L'] not in gue_test_Ls])

    print(f"  GUE-class sizes: {gue_test_Ls}")
    print(f"  Borderline sizes (p>0.04): {borderline_Ls}")

    test_Ls = gue_test_Ls + borderline_Ls
    test_Ls.sort()

    dedekind_results = []

    for L in test_Ls:
        r = fine_results[L]
        g_M = pair_correlation_kde(r['unfolded'], r_vals)

        rms_M_mont = float(np.sqrt(np.mean((g_M - g_mont) ** 2)))
        rms_M_riem = float(np.sqrt(np.mean((g_M - g_riem) ** 2)))
        g0_M = g_M[0]

        if has_ded:
            rms_M_ded = float(np.sqrt(np.mean((g_M - g_ded) ** 2)))
            ded_impr = (rms_M_riem - rms_M_ded) / rms_M_riem * 100
        else:
            rms_M_ded = None
            ded_impr = None

        cls_label = "GUE" if L in gue_class_Ls else "borderline"
        ded_str = f"{rms_M_ded:.4f}" if rms_M_ded is not None else "N/A"
        imp_str = f"{ded_impr:.1f}%" if ded_impr is not None else "N/A"
        print(f"  L={L:2d} [{cls_label:10s}]: RMS(M,Riem)={rms_M_riem:.4f}, "
              f"RMS(M,Ded)={ded_str}, Ded_impr={imp_str}, g(0)={g0_M:.4f}")

        dedekind_results.append({
            'L': L, 'n_wing': r['n_wing'],
            'rms_M_mont': rms_M_mont, 'rms_M_riem': rms_M_riem,
            'rms_M_ded': rms_M_ded, 'ded_impr': ded_impr,
            'g0_M': g0_M,
            'g0_riem': g_riem[0],
            'g0_ded': g_ded[0] if has_ded else None,
            'cls': cls_label,
            'ks_gue_p': r['ks_gue'][1],
        })

    # Dedekind table for GUE-class sizes
    print(f"\n  --- DEDEKIND RESULTS AT GUE-CLASS SIZES ---")
    print(f"  {'L':>4} {'wing':>5} {'p(GUE)':>7} {'RMS(Riem)':>10} {'RMS(Ded)':>9} {'Ded%':>7} {'g_M(0)':>8} {'class':>10}")
    print(f"  {'-'*70}")
    for d in dedekind_results:
        ded_str = f"{d['rms_M_ded']:.4f}" if d['rms_M_ded'] is not None else "N/A"
        imp_str = f"{d['ded_impr']:.1f}%" if d['ded_impr'] is not None else "N/A"
        print(f"  {d['L']:>4} {d['n_wing']:>5} {d['ks_gue_p']:>7.4f} {d['rms_M_riem']:>10.4f} {ded_str:>9} {imp_str:>7} {d['g0_M']:>8.4f} {d['cls']:>10}")

    # Filter GUE-only for clean Dedekind analysis
    gue_only = [d for d in dedekind_results if d['cls'] == 'GUE']
    if len(gue_only) >= 2 and has_ded:
        print(f"\n  --- CLEAN DEDEKIND IMPROVEMENT (GUE-class only) ---")
        gue_Ls = [d['L'] for d in gue_only]
        gue_imps = [d['ded_impr'] for d in gue_only]
        gue_g0s = [d['g0_M'] for d in gue_only]

        for d in gue_only:
            print(f"    L={d['L']:2d}: Ded improvement = {d['ded_impr']:.1f}%, g_M(0) = {d['g0_M']:.4f}")

        if len(gue_Ls) >= 2:
            mono_up_imp = all(gue_imps[i] <= gue_imps[i+1] for i in range(len(gue_imps)-1))
            mono_up_g0 = all(gue_g0s[i] <= gue_g0s[i+1] for i in range(len(gue_g0s)-1))
            print(f"\n    Ded improvement monotonic increasing? {'YES' if mono_up_imp else 'NO'}")
            print(f"    g_M(0) monotonic increasing? {'YES' if mono_up_g0 else 'NO'}")

    # ==================================================================
    # FIGURE 5: Dedekind at GUE sizes
    # ==================================================================
    if has_ded and len(dedekind_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: g_M(0) vs L, colour by class
        ax = axes[0]
        for d in dedekind_results:
            color = 'blue' if d['cls'] == 'GUE' else 'orange'
            marker = 'o' if d['cls'] == 'GUE' else 's'
            ax.plot(d['L'], d['g0_M'], marker, color=color, markersize=10)
            ax.annotate(f"{d['g0_M']:.3f}", (d['L'], d['g0_M']),
                        textcoords="offset points", xytext=(5, 8), fontsize=7)

        ax.axhline(g_ded[0], color='purple', ls='--', linewidth=1.5,
                    label=f'Dedekind g(0) = {g_ded[0]:.3f}')
        ax.axhline(g_riem[0], color='red', ls=':', linewidth=1.5,
                    label=f'Riemann g(0) = {g_riem[0]:.3f}')

        ax.plot([], [], 'bo', markersize=8, label='GUE-class')
        ax.plot([], [], 's', color='orange', markersize=8, label='Borderline')

        ax.set_xlabel('Lattice size L', fontsize=12)
        ax.set_ylabel('g_M(0)', fontsize=12)
        ax.set_title('Hole Depth at GUE-class vs Borderline Sizes', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: Dedekind improvement vs L
        ax2 = axes[1]
        for d in dedekind_results:
            if d['ded_impr'] is not None:
                color = 'blue' if d['cls'] == 'GUE' else 'orange'
                marker = 'o' if d['cls'] == 'GUE' else 's'
                ax2.plot(d['L'], d['ded_impr'], marker, color=color, markersize=10)
                ax2.annotate(f"{d['ded_impr']:.1f}%", (d['L'], d['ded_impr']),
                             textcoords="offset points", xytext=(5, 8), fontsize=7)

        ax2.axhline(0, color='gray', ls='-', alpha=0.3)
        ax2.set_xlabel('Lattice size L', fontsize=12)
        ax2.set_ylabel('Dedekind improvement %', fontsize=12)
        ax2.set_title('Dedekind Improvement (blue=GUE, orange=borderline)', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'figure5_dedekind_gue_only.png', dpi=200)
        print(f"  Saved figure5_dedekind_gue_only.png")
        plt.close()

    # ==================================================================
    # CROSSOVER CHARACTERISATION
    # ==================================================================
    print(f"\n{'='*78}")
    print("CROSSOVER CHARACTERISATION")
    print("=" * 78)

    # Compute the "GUE preference" = log(p_GUE / p_GOE)
    log_ratios = []
    for d in crossover_data:
        gue_p = max(d['ks_gue_p'], 1e-10)
        goe_p = max(d['ks_goe_p'], 1e-10)
        log_ratios.append(np.log10(gue_p / goe_p))
    log_ratios = np.array(log_ratios)

    print(f"\n  GUE preference = log10(p_GUE / p_GOE):")
    print(f"  Positive = GUE preferred, Negative = GOE preferred")
    for i, d in enumerate(crossover_data):
        bar = "=" * int(abs(log_ratios[i]) * 20)
        side = "GUE" if log_ratios[i] > 0 else "GOE"
        print(f"    L={d['L']:2d}: {log_ratios[i]:>+7.3f} {'>' if log_ratios[i] > 0 else '<'} {bar} ({side})")

    # Find zero-crossing of log-ratio
    sign_changes = []
    for i in range(len(log_ratios) - 1):
        if log_ratios[i] * log_ratios[i + 1] < 0:
            # Linear interpolation
            L_cross = Ls_arr[i] + (Ls_arr[i+1] - Ls_arr[i]) * (-log_ratios[i]) / (log_ratios[i+1] - log_ratios[i])
            sign_changes.append(L_cross)

    print(f"\n  GUE/GOE crossover points (linear interpolation):")
    for Lc in sign_changes:
        print(f"    L* = {Lc:.2f}  (L*/xi = {Lc/XI:.2f}, L*/h = {Lc/12:.2f})")

    # Check against structural numbers
    print(f"\n  --- STRUCTURAL NUMBER CHECK ---")
    print(f"  h (Coxeter number E6) = 12")
    print(f"  xi (decay length) = {XI}")
    print(f"  3h = 36, 3h+1 = 37, h*xi = {12*XI}")
    for Lc in sign_changes:
        for name, val in [('h', 12), ('2h', 24), ('3h', 36), ('xi*h', XI*12),
                          ('6*xi', 6*XI), ('10*xi', 10*XI), ('12*xi', 12*XI),
                          ('h+xi', 12+XI)]:
            if abs(Lc - val) < 2:
                print(f"    L* = {Lc:.2f} ~ {name} = {val}")

    # ==================================================================
    # FIGURE 6: GUE preference (log ratio)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    colors_bar = ['blue' if lr > 0 else 'red' for lr in log_ratios]
    ax.bar(Ls_arr, log_ratios, color=colors_bar, alpha=0.7, width=0.8)
    ax.axhline(0, color='black', linewidth=1)

    for Lc in sign_changes:
        ax.axvline(Lc, color='green', ls='--', linewidth=2, alpha=0.7,
                   label=f'Crossover L* = {Lc:.1f}')

    # Mark structural values
    for L_struct, label in [(36, '3h=36'), (12*XI, f'h*xi={12*XI:.0f}')]:
        ax.axvline(L_struct, color='purple', ls=':', alpha=0.4)
        ax.text(L_struct + 0.3, max(log_ratios) * 0.8, label, fontsize=8, color='purple')

    ax.set_xlabel('Lattice size L', fontsize=12)
    ax.set_ylabel('log10(p_GUE / p_GOE)', fontsize=12)
    ax.set_title('GUE Preference Index (blue=GUE, red=GOE)', fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure6_gue_preference.png', dpi=200)
    print(f"  Saved figure6_gue_preference.png")
    plt.close()

    # ==================================================================
    # SAVE CROSSOVER DATA
    # ==================================================================
    crossover_file = RESULTS_DIR / 'crossover_data.txt'
    with open(crossover_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSIS 30 PART 2 — GUE/GOE CROSSOVER DATA\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("FINE CROSSOVER TABLE (L = 30..48)\n")
        f.write(f"{'L':>4} {'wing':>6} {'KS(GUE)':>8} {'p(GUE)':>8} {'KS(GOE)':>8} {'p(GOE)':>8} {'class':>8} {'log_ratio':>10}\n")
        f.write("-" * 75 + "\n")
        for i, d in enumerate(crossover_data):
            f.write(f"{d['L']:>4} {d['n_wing']:>6} {d['ks_gue_stat']:>8.4f} {d['ks_gue_p']:>8.4f} "
                    f"{d['ks_goe_stat']:>8.4f} {d['ks_goe_p']:>8.4f} {d['cls']:>8} {log_ratios[i]:>+10.4f}\n")

        f.write(f"\nCrossover points: {[f'{Lc:.2f}' for Lc in sign_changes]}\n")

        if has_ded:
            f.write(f"\nDEDEKIND TEST AT GUE-CLASS SIZES:\n")
            for d in dedekind_results:
                imp = f"{d['ded_impr']:.1f}%" if d['ded_impr'] is not None else "N/A"
                f.write(f"  L={d['L']:2d} [{d['cls']:10s}]: Ded_impr={imp:>7}, g_M(0)={d['g0_M']:.4f}\n")

    print(f"\n  Saved crossover_data.txt")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    elapsed = time.time() - total_start
    print(f"\n{'='*78}")
    print("FINAL SUMMARY — ANALYSIS 30 PART 2")
    print("=" * 78)
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n1. GUE BOUNDARY:")
    print(f"   GUE-class sizes (L=30..48): {gue_class_Ls}")
    if len(gue_class_Ls) > 0:
        print(f"   Last confirmed GUE: L = {max(gue_class_Ls)}")

    print(f"\n2. CROSSOVER CHARACTERISATION:")
    for Lc in sign_changes:
        print(f"   Crossover at L* = {Lc:.2f} (L*/xi = {Lc/XI:.2f}, L*/h = {Lc/12:.3f})")

    print(f"\n3. DEDEKIND AT GUE SIZES:")
    if has_ded and len(gue_only) > 0:
        for d in gue_only:
            print(f"   L={d['L']:2d}: Ded improvement = {d['ded_impr']:.1f}%, g_M(0) = {d['g0_M']:.4f}")
    else:
        print(f"   (no GUE-class sizes above 30 or no Dedekind data)")

    print(f"\n  All outputs: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
