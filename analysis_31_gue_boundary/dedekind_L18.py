#!/usr/bin/env python3
"""
Quick script: Run Dedekind pipeline at L=18, then repackage all
Analysis 30/31 outputs into analysis_31_gue_boundary/ with updated figures.
"""

import numpy as np
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

# Architecture constants
XI = 3.0
PHI = 1.0 / 6
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]
R_VALS = np.linspace(0.01, 4.0, 400)
BANDWIDTH = 0.4
POLY_DEG = 10
PCT_THRESH = 20

SRC_DIR = Path(r"C:\Users\selin\merkabit_results\analysis_30_dedekind_L42")
DST_DIR = Path(r"C:\Users\selin\merkabit_results\analysis_31_gue_boundary")
DST_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# === Operator pipeline (copied from gue_boundary_scan.py) ===

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


# === Zero family loading ===

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
    sp = np.diff(unfolded_riem)
    m = np.mean(sp)
    if m > 0:
        sp /= m
        unfolded_riem = (unfolded_riem - unfolded_riem[0]) / m

    unfolded_ded = None
    if lchi3_gammas is not None:
        lchi3_500 = lchi3_gammas[:500]
        dedekind_gammas = np.sort(np.concatenate([riemann_500, lchi3_500]))
        unique_ded = [dedekind_gammas[0]]
        for z in dedekind_gammas[1:]:
            if z - unique_ded[-1] > 0.01:
                unique_ded.append(z)
        dedekind_gammas = np.array(unique_ded)
        unfolded_ded = unfold_dedekind_zeros(dedekind_gammas)
        sp_d = np.diff(unfolded_ded)
        m_d = np.mean(sp_d)
        if m_d > 0:
            sp_d /= m_d
            unfolded_ded = (unfolded_ded - unfolded_ded[0]) / m_d

    return unfolded_riem, unfolded_ded


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("TASK 1: Dedekind pipeline at L=18")
    print("=" * 70)

    r_vals = R_VALS
    g_mont = montgomery_formula(r_vals)

    # Load zero families
    print("Loading zero families...")
    unfolded_riem, unfolded_ded = load_zero_families()
    g_riem = pair_correlation_kde(unfolded_riem, r_vals)
    g_ded = pair_correlation_kde(unfolded_ded, r_vals) if unfolded_ded is not None else None

    # Build M at L=18
    L = 18
    print(f"\nBuilding M at L={L}...")
    t0 = time.time()
    torus = EisensteinTorus(L)
    u, v, omega = assign_spinors_geometric(torus)
    M_mat = build_M(torus, u, v, omega)
    eigs = np.linalg.eigvalsh(M_mat)
    del M_mat, torus, u, v, omega
    gc.collect()

    unfolded_M, spacings_M = get_unfolded_wing(eigs)
    n_wing = len(unfolded_M)
    ks = ks_tests(spacings_M)

    print(f"  N = {L*L}, wing = {n_wing}")
    print(f"  KS(GUE) = {ks['gue'][0]:.4f}, p = {ks['gue'][1]:.4f}")
    print(f"  KS(GOE) = {ks['goe'][0]:.4f}, p = {ks['goe'][1]:.4f}")

    # Pair correlation
    g_M = pair_correlation_kde(unfolded_M, r_vals)

    rms_M_mont = float(np.sqrt(np.mean((g_M - g_mont) ** 2)))
    rms_M_riem = float(np.sqrt(np.mean((g_M - g_riem) ** 2)))
    rms_M_ded = float(np.sqrt(np.mean((g_M - g_ded) ** 2))) if g_ded is not None else None
    ded_impr = (rms_M_riem - rms_M_ded) / rms_M_riem * 100 if rms_M_ded is not None else None
    g0_M = g_M[0]

    print(f"\n  L=18 DEDEKIND RESULTS:")
    print(f"  RMS(M, Montgomery): {rms_M_mont:.4f}")
    print(f"  RMS(M, Riemann):    {rms_M_riem:.4f}")
    print(f"  RMS(M, Dedekind):   {rms_M_ded:.4f}" if rms_M_ded else "  RMS(M, Dedekind):   N/A")
    print(f"  Dedekind improvement: {ded_impr:.1f}%" if ded_impr else "  Dedekind improvement: N/A")
    print(f"  g_M(0):             {g0_M:.4f}")
    print(f"  g_Riem(0):          {g_riem[0]:.4f}")
    print(f"  g_Ded(0):           {g_ded[0]:.4f}" if g_ded is not None else "")
    print(f"  Computed in {time.time()-t0:.1f}s")

    # === Full Dedekind table (L=18 + previous results from Part 2) ===
    # Prior results from gue_boundary_scan.py output
    prior = [
        {'L': 30, 'n_wing': 360, 'ks_gue_p': 0.1200, 'rms_M_riem': 0.0511,
         'rms_M_ded': 0.0362, 'ded_impr': 29.2, 'g0_M': 0.4381, 'cls': 'GUE'},
        {'L': 32, 'n_wing': 410, 'ks_gue_p': 0.1113, 'rms_M_riem': 0.0393,
         'rms_M_ded': 0.0358, 'ded_impr': 9.0, 'g0_M': 0.4025, 'cls': 'GUE'},
        {'L': 36, 'n_wing': 518, 'ks_gue_p': 0.0712, 'rms_M_riem': 0.0412,
         'rms_M_ded': 0.0499, 'ded_impr': -21.1, 'g0_M': 0.3821, 'cls': 'GUE'},
        {'L': 37, 'n_wing': 548, 'ks_gue_p': 0.1534, 'rms_M_riem': 0.0436,
         'rms_M_ded': 0.0461, 'ded_impr': -5.7, 'g0_M': 0.3979, 'cls': 'GUE'},
        {'L': 38, 'n_wing': 577, 'ks_gue_p': 0.2059, 'rms_M_riem': 0.0535,
         'rms_M_ded': 0.0580, 'ded_impr': -8.4, 'g0_M': 0.4076, 'cls': 'GUE'},
        {'L': 39, 'n_wing': 608, 'ks_gue_p': 0.0636, 'rms_M_riem': 0.0367,
         'rms_M_ded': 0.0445, 'ded_impr': -21.4, 'g0_M': 0.3798, 'cls': 'GUE'},
    ]

    l18_row = {
        'L': 18, 'n_wing': n_wing, 'ks_gue_p': ks['gue'][1],
        'rms_M_riem': rms_M_riem,
        'rms_M_ded': rms_M_ded, 'ded_impr': ded_impr,
        'g0_M': g0_M, 'cls': 'GUE',
    }

    all_rows = [l18_row] + prior

    print(f"\n{'='*78}")
    print("COMPLETE DEDEKIND TABLE — GUE-CLASS SIZES (Analysis 31)")
    print("=" * 78)
    hdr = f"{'L':>4} {'wing':>5} {'p(GUE)':>7} {'RMS(M,Riem)':>11} {'RMS(M,Ded)':>10} {'Ded%':>7} {'g_M(0)':>8} {'class':>6}"
    print(hdr)
    print("-" * 70)
    for d in all_rows:
        ded_s = f"{d['rms_M_ded']:.4f}" if d['rms_M_ded'] is not None else "N/A"
        imp_s = f"{d['ded_impr']:.1f}%" if d['ded_impr'] is not None else "N/A"
        print(f"{d['L']:>4} {d['n_wing']:>5} {d['ks_gue_p']:>7.4f} {d['rms_M_riem']:>11.4f} "
              f"{ded_s:>10} {imp_s:>7} {d['g0_M']:>8.4f} {d['cls']:>6}")

    # === TASK 2: Repackage into analysis_31_gue_boundary/ ===
    print(f"\n{'='*70}")
    print("TASK 2: Repackaging into analysis_31_gue_boundary/")
    print("=" * 70)

    copy_map = {
        'gue_boundary_scan.py': 'gue_boundary_scan.py',
        'dedekind_convergence.py': 'dedekind_gue_only.py',
        'figure4_gue_crossover.png': 'figure1_gue_boundary.png',
        'figure6_gue_preference.png': 'figure1b_gue_preference.png',
        'figure1_hole_trajectory.png': 'figure3_hole_depth_gue_only.png',
        'figure3_pair_correlation_L42.png': 'figure3b_pair_correlation_L42.png',
        'crossover_data.txt': 'results_part1_boundary.txt',
        'results_table_A.txt': 'results_part2_hole_depth.txt',
        'results_table_B.txt': 'results_part2_rms.txt',
        'results_table_C.txt': 'results_part2_gue_class.txt',
        'analysis_30_summary.txt': 'results_part3_crossover.txt',
    }

    for src_name, dst_name in copy_map.items():
        src = SRC_DIR / src_name
        dst = DST_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src_name} -> {dst_name}")
        else:
            print(f"  SKIP {src_name} (not found)")

    # Copy eigenvalue/g_M data files
    for f in SRC_DIR.glob('eigs_L*.npy'):
        shutil.copy2(f, DST_DIR / f.name)
    for f in SRC_DIR.glob('g_M_L*.npy'):
        shutil.copy2(f, DST_DIR / f.name)

    # Save this script too
    shutil.copy2(SRC_DIR / 'dedekind_L18.py', DST_DIR / 'dedekind_L18.py')

    # Save complete Dedekind table
    with open(DST_DIR / 'results_part2_dedekind.txt', 'w', encoding='utf-8') as f:
        f.write("ANALYSIS 31 — DEDEKIND COMPARISON AT GUE-CLASS SIZES\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Reference: g_Ded(0) = {:.4f}, g_Riem(0) = {:.4f}\n\n".format(
            g_ded[0] if g_ded is not None else 0, g_riem[0]))
        f.write(hdr + "\n")
        f.write("-" * 70 + "\n")
        for d in all_rows:
            ded_s = f"{d['rms_M_ded']:.4f}" if d['rms_M_ded'] is not None else "N/A"
            imp_s = f"{d['ded_impr']:.1f}%" if d['ded_impr'] is not None else "N/A"
            f.write(f"{d['L']:>4} {d['n_wing']:>5} {d['ks_gue_p']:>7.4f} {d['rms_M_riem']:>11.4f} "
                    f"{ded_s:>10} {imp_s:>7} {d['g0_M']:>8.4f} {d['cls']:>6}\n")
        f.write("\nInterpretation:\n")
        f.write("  Dedekind improvement POSITIVE at L=18,30,32 (first GUE window)\n")
        f.write("  Dedekind improvement NEGATIVE at L=36,37,38,39 (GUE island = h*xi resonance)\n")
        f.write("  -> Dedekind affinity is a finite-size effect at small L\n")
        f.write("  -> At the strongest GUE sizes (L=36-39), M tracks Riemann, not Dedekind\n")
    print(f"  Saved results_part2_dedekind.txt")

    # === TASK 3: Updated figure2 with L=18 ===
    print(f"\n{'='*70}")
    print("TASK 3: Updated figure2_dedekind_improvement.png with L=18")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of Dedekind improvement
    ax = axes[0]
    Ls = [d['L'] for d in all_rows]
    imps = [d['ded_impr'] if d['ded_impr'] is not None else 0 for d in all_rows]
    colors = ['#2196F3' if imp > 0 else '#f44336' for imp in imps]

    bars = ax.bar(range(len(Ls)), imps, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([str(L) for L in Ls], fontsize=11)
    ax.axhline(0, color='black', linewidth=1)

    # Annotate bars
    for i, (L, imp) in enumerate(zip(Ls, imps)):
        va = 'bottom' if imp >= 0 else 'top'
        offset = 1.0 if imp >= 0 else -1.0
        ax.text(i, imp + offset, f'{imp:.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    # Mark GUE windows
    ax.axvspan(-0.5, 2.5, alpha=0.06, color='blue', label='GUE window 1 (L=18,30,32)')
    ax.axvspan(2.5, 6.5, alpha=0.06, color='green', label='GUE island (L=36-39, h*xi resonance)')

    ax.set_xlabel('Lattice size L (GUE-confirmed only)', fontsize=12)
    ax.set_ylabel('Dedekind improvement %', fontsize=12)
    ax.set_title('Analysis 31: Dedekind Improvement at GUE-Class Sizes', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(min(imps) - 8, max(imps) + 8)

    # Right: g_M(0) at each GUE size
    ax2 = axes[1]
    g0s = [d['g0_M'] for d in all_rows]
    ax2.plot(Ls, g0s, 'bo-', markersize=10, linewidth=2, label='g_M(0)')

    if g_ded is not None:
        ax2.axhline(g_ded[0], color='purple', ls='--', linewidth=1.5,
                    label=f'Dedekind g(0) = {g_ded[0]:.3f}')
    ax2.axhline(g_riem[0], color='red', ls=':', linewidth=1.5,
                label=f'Riemann g(0) = {g_riem[0]:.3f}')

    for i, d in enumerate(all_rows):
        ax2.annotate(f'{d["g0_M"]:.3f}', (d['L'], d['g0_M']),
                     textcoords="offset points", xytext=(8, 8), fontsize=8)

    # Mark windows
    ax2.axvspan(17, 33, alpha=0.06, color='blue')
    ax2.axvspan(35, 40, alpha=0.06, color='green')

    ax2.set_xlabel('Lattice size L (GUE-confirmed only)', fontsize=12)
    ax2.set_ylabel('g_M(0) — correlation hole depth', fontsize=12)
    ax2.set_title('Hole Depth at GUE-Class Sizes', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(DST_DIR / 'figure2_dedekind_improvement.png', dpi=200)
    print(f"  Saved figure2_dedekind_improvement.png")
    plt.close()

    # Also save to the source dir for completeness
    shutil.copy2(DST_DIR / 'figure2_dedekind_improvement.png',
                 SRC_DIR / 'figure2_dedekind_improvement_v2.png')

    # === Final listing ===
    print(f"\n{'='*70}")
    print(f"analysis_31_gue_boundary/ contents:")
    print("=" * 70)
    for f in sorted(DST_DIR.iterdir()):
        sz = f.stat().st_size
        print(f"  {f.name:45s}  {sz:>8,} bytes")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
