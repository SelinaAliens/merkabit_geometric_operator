#!/usr/bin/env python3
"""
GUE Null Wing Check
====================
Validate that the GUE null comparison is fair by testing two approaches:

Approach A (original): Generate N_wing x N_wing GUE, use ALL eigenvalues
Approach B (wing-matched): Generate N_total x N_total GUE, extract positive
                           wing the SAME way as M operator

If both give similar RMS, the comparison is fair.
If B gives higher RMS, the wing extraction introduces a bias.
"""
import numpy as np
import time

# ── Exact pipeline functions ──

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

def extract_positive_wing(evals, pct=20):
    pos = evals[evals > 0]
    if len(pos) < 4: return pos
    return pos[pos > np.percentile(pos, pct)]

def get_unfolded_wing(eigs):
    wing = extract_positive_wing(np.sort(np.real(eigs)))
    if len(wing) < 10: return wing, np.array([])
    return unfold_spectrum(wing)

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

def random_gue(N):
    A = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    H = (A + A.conj().T) / 2.0
    return H

# ============================================================================
if __name__ == '__main__':
    np.random.seed(42)
    r_vals = np.linspace(0.01, 4.0, 200)
    g_mont = montgomery_formula(r_vals)
    N_TRIALS = 50

    # Test at L=30: total N=900, wing N~360
    # and at L=18: total N=324, wing N~130
    test_cases = [
        {'label': 'L=18', 'N_total': 324, 'N_wing_expected': 130},
        {'label': 'L=30', 'N_total': 900, 'N_wing_expected': 360},
    ]

    print("=" * 80)
    print("GUE NULL WING CHECK: Is the comparison fair?")
    print(f"N_TRIALS = {N_TRIALS}")
    print("=" * 80)

    for tc in test_cases:
        N_total = tc['N_total']
        N_wing_exp = tc['N_wing_expected']

        print(f"\n{'='*60}")
        print(f"  {tc['label']}: N_total = {N_total}, expected wing ~ {N_wing_exp}")
        print(f"{'='*60}")

        # ── Approach A: N_wing x N_wing GUE, all eigenvalues ──
        rms_A = []
        hole_A = []
        for _ in range(N_TRIALS):
            H = random_gue(N_wing_exp)
            eigs = np.linalg.eigvalsh(H)
            unf, sp = unfold_spectrum(eigs)
            g = pair_correlation_fast(unf, r_vals, 0.4)
            rms_A.append(np.sqrt(np.mean((g - g_mont)**2)))
            hole_A.append(g[0])
        rms_A = np.array(rms_A)
        hole_A = np.array(hole_A)

        # ── Approach B: N_total x N_total GUE, extract positive wing ──
        rms_B = []
        hole_B = []
        wing_sizes_B = []
        for _ in range(N_TRIALS):
            H = random_gue(N_total)
            eigs = np.linalg.eigvalsh(H)
            # Same wing extraction as M operator
            unf, sp = get_unfolded_wing(eigs)
            n_wing = len(unf)
            wing_sizes_B.append(n_wing)
            g = pair_correlation_fast(unf, r_vals, 0.4)
            rms_B.append(np.sqrt(np.mean((g - g_mont)**2)))
            hole_B.append(g[0])
        rms_B = np.array(rms_B)
        hole_B = np.array(hole_B)
        wing_sizes_B = np.array(wing_sizes_B)

        print(f"\n  Approach A (full {N_wing_exp}x{N_wing_exp} GUE, all eigs):")
        print(f"    RMS = {np.mean(rms_A):.4f} +/- {np.std(rms_A):.4f}")
        print(f"    g(0) = {np.mean(hole_A):.4f} +/- {np.std(hole_A):.4f}")

        print(f"\n  Approach B ({N_total}x{N_total} GUE, positive wing extraction):")
        print(f"    Wing size: {np.mean(wing_sizes_B):.0f} +/- {np.std(wing_sizes_B):.0f} "
              f"(expected {N_wing_exp})")
        print(f"    RMS = {np.mean(rms_B):.4f} +/- {np.std(rms_B):.4f}")
        print(f"    g(0) = {np.mean(hole_B):.4f} +/- {np.std(hole_B):.4f}")

        diff = np.mean(rms_B) - np.mean(rms_A)
        pooled_std = np.sqrt(np.std(rms_A)**2/N_TRIALS + np.std(rms_B)**2/N_TRIALS)
        t_stat = diff / pooled_std if pooled_std > 0 else 0

        print(f"\n  Difference (B - A): {diff:+.4f}  ({t_stat:+.1f} sigma)")
        if abs(t_stat) < 2:
            print(f"  VERDICT: No significant difference. Original comparison is FAIR.")
        elif t_stat > 2:
            print(f"  VERDICT: Wing extraction gives HIGHER RMS. Original comparison BIASED")
            print(f"           (GUE null too favorable — M's gap to GUE is SMALLER than reported).")
        else:
            print(f"  VERDICT: Wing extraction gives LOWER RMS. Original comparison CONSERVATIVE")
            print(f"           (GUE null too harsh — M's gap to GUE is LARGER than reported).")
