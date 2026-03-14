#!/usr/bin/env python3
"""
GUE Null Comparison
====================
Generate random GUE matrices at each wing size N that the M operator
produces, compute Montgomery pair correlation RMS, and compare.

Question: Is M's RMS = 0.118 at N=360 better, worse, or indistinguishable
from a generic GUE matrix of the same size?

Uses the EXACT same pipeline as L24_convergence.py:
  - degree-10 polynomial unfolding
  - Gaussian KDE pair correlation (bandwidth 0.4)
  - Montgomery formula comparison over r in [0.01, 4.0], 200 points
"""
import numpy as np
from scipy import stats
import time
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"

# ── Exact pipeline functions from L24_convergence.py ──

def unfold_spectrum(eigenvalues, poly_degree=10):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        sp = np.diff(evals)
        if len(sp) > 0 and np.mean(sp) > 0: sp /= np.mean(sp)
        return evals, sp
    deg = min(poly_degree, max(3, N // 10))
    coeffs = np.polyfit(evals, np.arange(1, N+1), deg)
    N_smooth = np.polyval(coeffs, evals)
    sp = np.diff(N_smooth)
    m = np.mean(sp)
    if m > 0: sp /= m
    return N_smooth, sp

def pair_correlation_fast(unfolded_eigs, r_vals, bandwidth=0.4):
    eigs = np.sort(unfolded_eigs)
    N = len(eigs)
    if N < 5: return np.ones(len(r_vals))
    diffs = eigs[:, None] - eigs[None, :]
    np.fill_diagonal(diffs, np.nan)
    diffs_flat = diffs[~np.isnan(diffs)]
    g = np.zeros(len(r_vals))
    for k, r in enumerate(r_vals):
        g[k] = np.sum(np.exp(-0.5 * ((diffs_flat - r) / bandwidth)**2))
    g /= (N * bandwidth * np.sqrt(2 * np.pi))
    large_r = g[r_vals > 3.0]
    if len(large_r) > 3:
        norm = np.mean(large_r)
        if norm > 0: g /= norm
    return g

def montgomery_formula(r_vals):
    g = np.ones_like(r_vals, dtype=float)
    nz = r_vals > 1e-10
    g[nz] = 1 - (np.sin(np.pi * r_vals[nz]) / (np.pi * r_vals[nz]))**2
    g[~nz] = 0.0
    return g

# ── GUE matrix generator ──

def random_gue(N):
    """Generate an N x N matrix from GUE: H = (A + A^dag) / sqrt(2N)"""
    A = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    H = (A + A.conj().T) / 2.0
    return H

# ============================================================================
if __name__ == '__main__':
    np.random.seed(42)

    r_vals = np.linspace(0.01, 4.0, 200)
    g_mont = montgomery_formula(r_vals)

    # Wing sizes that M produces at each L
    wing_sizes = {
        'L=12':  58,
        'L=15':  89,
        'L=18': 130,
        'L=21': 176,
        'L=24': 230,
        'L=27': 291,
        'L=30': 360,
    }

    # M operator RMS values (from L24_convergence.py output)
    m_rms = {
        'L=12': 0.1631,
        'L=15': 0.1247,
        'L=18': 0.1351,
        'L=21': 0.1226,
        'L=24': 0.1223,
        'L=27': 0.1245,
        'L=30': 0.1182,
    }

    # Also test Riemann benchmark size
    wing_sizes['Riemann'] = 1000
    m_rms['Riemann'] = 0.0821

    N_TRIALS = 100

    print("=" * 80)
    print("GUE NULL COMPARISON")
    print(f"Generating {N_TRIALS} random GUE matrices at each wing size")
    print("Same pipeline: poly-10 unfolding, KDE bw=0.4, r in [0.01, 4.0]")
    print("=" * 80)

    results = {}

    for label in ['L=12', 'L=15', 'L=18', 'L=21', 'L=24', 'L=27', 'L=30', 'Riemann']:
        N = wing_sizes[label]
        t0 = time.time()
        rms_list = []
        hole_list = []

        for trial in range(N_TRIALS):
            H = random_gue(N)
            eigs = np.linalg.eigvalsh(H)

            # Unfold with same polynomial method
            unf, sp = unfold_spectrum(eigs)

            # Pair correlation with same KDE
            g = pair_correlation_fast(unf, r_vals, 0.4)

            # RMS vs Montgomery
            rms = np.sqrt(np.mean((g - g_mont)**2))
            rms_list.append(rms)
            hole_list.append(g[0])  # g(r~0.01)

        rms_arr = np.array(rms_list)
        hole_arr = np.array(hole_list)
        dt = time.time() - t0

        results[label] = {
            'N': N,
            'mean_rms': np.mean(rms_arr),
            'std_rms': np.std(rms_arr),
            'median_rms': np.median(rms_arr),
            'min_rms': np.min(rms_arr),
            'max_rms': np.max(rms_arr),
            'p5_rms': np.percentile(rms_arr, 5),
            'p95_rms': np.percentile(rms_arr, 95),
            'mean_hole': np.mean(hole_arr),
            'std_hole': np.std(hole_arr),
            'M_rms': m_rms[label],
        }

        # How many GUE trials beat M?
        n_better = np.sum(rms_arr < m_rms[label])
        results[label]['n_better'] = n_better
        results[label]['percentile_M'] = 100.0 * np.sum(rms_arr < m_rms[label]) / N_TRIALS

        print(f"\n  {label} (N={N:4d}): GUE RMS = {np.mean(rms_arr):.4f} +/- {np.std(rms_arr):.4f}  "
              f"[{dt:.1f}s]")
        print(f"    M operator RMS = {m_rms[label]:.4f}")
        print(f"    GUE trials better than M: {n_better}/{N_TRIALS} ({results[label]['percentile_M']:.0f}%)")
        print(f"    GUE hole g(0): {np.mean(hole_arr):.4f} +/- {np.std(hole_arr):.4f}")

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("SUMMARY: M OPERATOR vs GUE NULL")
    print("=" * 80)
    print(f"  {'Size':>8} | {'N':>5} | {'GUE RMS mean+/-std':>20} | {'M RMS':>8} | "
          f"{'M vs GUE':>10} | {'GUE better':>10}")
    print(f"  {'-' * 78}")

    for label in ['L=12', 'L=15', 'L=18', 'L=21', 'L=24', 'L=27', 'L=30', 'Riemann']:
        r = results[label]
        # Sigma distance
        if r['std_rms'] > 0:
            sigma = (r['M_rms'] - r['mean_rms']) / r['std_rms']
            sigma_str = f"{sigma:+.1f}sigma"
        else:
            sigma_str = "N/A"
        print(f"  {label:>8} | {r['N']:>5} | {r['mean_rms']:.4f} +/- {r['std_rms']:.4f}   | "
              f"{r['M_rms']:.4f} | {sigma_str:>10} | {r['n_better']:>4}/{N_TRIALS}")

    # ── Interpretation ──
    r30 = results['L=30']
    print(f"\n{'=' * 80}")
    print("INTERPRETATION (L=30, N=360)")
    print(f"{'=' * 80}")
    print(f"  M operator:      RMS = {r30['M_rms']:.4f}")
    print(f"  GUE null:        RMS = {r30['mean_rms']:.4f} +/- {r30['std_rms']:.4f}")
    print(f"  GUE 90% CI:      [{r30['p5_rms']:.4f}, {r30['p95_rms']:.4f}]")

    sigma30 = (r30['M_rms'] - r30['mean_rms']) / r30['std_rms'] if r30['std_rms'] > 0 else 0
    if abs(sigma30) < 2:
        print(f"  Verdict: M is CONSISTENT with GUE null ({sigma30:+.1f} sigma)")
        print(f"           M's pair correlation matches what ANY GUE matrix of this size gives.")
        print(f"           The result is: M has the CORRECT SYMMETRY CLASS (GUE), which is itself")
        print(f"           nontrivial for a deterministic operator with zero free parameters.")
    elif sigma30 < -2:
        print(f"  Verdict: M OUTPERFORMS generic GUE ({sigma30:+.1f} sigma)")
        print(f"           M produces BETTER Montgomery agreement than random GUE matrices.")
        print(f"           This suggests M has additional arithmetic structure beyond GUE universality.")
    else:
        print(f"  Verdict: M UNDERPERFORMS generic GUE ({sigma30:+.1f} sigma)")
        print(f"           M produces WORSE Montgomery agreement than random GUE matrices.")
        print(f"           Finite-size effects or non-universal features may be present.")

    # ── Save ──
    fname = os.path.join(OUT, "gue_null_comparison.txt")
    with open(fname, 'w') as f:
        f.write("GUE Null Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"N_TRIALS = {N_TRIALS}\n")
        f.write(f"Pipeline: poly-10 unfolding, KDE bw=0.4, r in [0.01, 4.0]\n\n")
        f.write(f"{'Size':>8} | {'N':>5} | {'GUE RMS':>10} | {'GUE std':>8} | {'M RMS':>8} | "
                f"{'sigma':>8} | {'GUE<M':>6}\n")
        f.write("-" * 70 + "\n")
        for label in ['L=12', 'L=15', 'L=18', 'L=21', 'L=24', 'L=27', 'L=30', 'Riemann']:
            r = results[label]
            sig = (r['M_rms'] - r['mean_rms']) / r['std_rms'] if r['std_rms'] > 0 else 0
            f.write(f"  {label:>6} | {r['N']:>5} | {r['mean_rms']:>10.4f} | {r['std_rms']:>8.4f} | "
                    f"{r['M_rms']:>8.4f} | {sig:>+8.2f} | {r['n_better']:>3}/{N_TRIALS}\n")
    print(f"\n  Saved: {fname}")
