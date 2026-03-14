#!/usr/bin/env python3
"""
ANALYSIS 23: E6 Root Projection onto A4^(1) Eigenspace
=======================================================
Deriving gate modulation values {4,13,7,10,15}/10 and {13,4,7,10,18}/10

The affine A4^(1) pentagon has Z5 symmetry -- the abstract algebra cannot
distinguish nodes. But E6 BREAKS this symmetry. The two "extra" E6 nodes
(beyond the A4 subchain) project onto the A4^(1) eigenspace with SPECIFIC
DIRECTIONS.

For each gate removal i, the modulation factors are:
  rz_i = |proj of d_Rz onto eigenspace of remaining A4|
  rx_i = |proj of d_Rx onto eigenspace of remaining A4|

Target values (in units of 1/10):
  S: rz=4,  rx=13
  R: rz=13, rx=4
  T: rz=7,  rx=7
  F: rz=10, rx=10
  P: rz=15, rx=18
"""
import numpy as np
from itertools import permutations
from fractions import Fraction
import os

OUT = "C:/Users/selin/merkabit_results/e6_modulation"

# ============================================================================
# E6 CARTAN MATRIX
# ============================================================================
# Standard labeling (1-indexed in physics, 0-indexed here):
#   0 - 1 - 2 - 3 - 4
#               |
#               5
A_E6 = np.array([
    [ 2, -1,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0, -1],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2,  0],
    [ 0, -1,  0,  0,  0,  2]
], dtype=float)

GATE_NAMES = ['S', 'R', 'T', 'F', 'P']

# Target values
TARGET_RZ = np.array([0.4, 1.3, 0.7, 1.0, 1.5])  # S, R, T, F, P
TARGET_RX = np.array([1.3, 0.4, 0.7, 1.0, 1.8])

# E6 inverse Cartan = fundamental weight inner product matrix
A_E6_inv = np.linalg.inv(A_E6)

print("=" * 70)
print("ANALYSIS 23: E6 PROJECTION ONTO A4^(1) EIGENSPACE")
print("=" * 70)

# ============================================================================
# STEP 1: Fundamental Weight Projection
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: FUNDAMENTAL WEIGHT PROJECTIONS")
print("=" * 70)

print("\nE6 inverse Cartan matrix (fundamental weight metric):")
for i in range(6):
    row_str = " ".join([f"{A_E6_inv[i,j]:7.4f}" for j in range(6)])
    print(f"  [{row_str}]")

print("\nFundamental weights (rows of A_E6^{-1}):")
for i in range(6):
    print(f"  omega_{i+1} = {A_E6_inv[i, :]}")

# ============================================================================
# STEP 2: Try all embeddings
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: SCAN ALL EMBEDDINGS")
print("=" * 70)

print("\nFor each choice of 5 pentagon nodes from the 6 E6 nodes,")
print("and each assignment of weight directions to Rz/Rx:")

best_score = np.inf
best_config = None
all_results = []

# Which 5 E6 nodes form the pentagon? Try leaving out each one.
for omitted in range(6):
    pentagon = [n for n in range(6) if n != omitted]
    extra_node = omitted

    # The "extra" node provides ONE direction. But we need TWO (Rz and Rx).
    # Use the FUNDAMENTAL WEIGHT of the extra node projected differently.
    # Or: use the extra node's connections to the pentagon to define TWO directions.

    # Approach A: Use the fundamental weight omega_{extra} projected onto
    # the subspace of each remaining A4 (4 nodes after gate removal).
    # This gives only ONE modulation value per gate, not two.

    # Approach B: Use TWO fundamental weights from the full E6.
    # For each pair (omega_i, omega_j) with i,j in {1,...,6}:
    for w_rz_idx in range(6):
        for w_rx_idx in range(6):
            if w_rz_idx == w_rx_idx:
                continue

            w_rz = A_E6_inv[w_rz_idx, :]
            w_rx = A_E6_inv[w_rx_idx, :]

            mags_rz = []
            mags_rx = []
            valid = True

            for gate_idx in range(5):
                # Remove this gate's node from the pentagon
                remaining = [pentagon[k] for k in range(5) if k != gate_idx]
                idx = remaining

                # 4x4 Gram sub-matrix
                G_sub = A_E6[np.ix_(idx, idx)]
                try:
                    G_sub_inv = np.linalg.inv(G_sub)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                # Components of weight vectors at remaining node indices
                w_rz_sub = w_rz[idx]
                w_rx_sub = w_rx[idx]

                # Projection magnitude: ||proj||^2 = w_sub^T G_sub^{-1} w_sub
                mag2_rz = w_rz_sub @ G_sub_inv @ w_rz_sub
                mag2_rx = w_rx_sub @ G_sub_inv @ w_rx_sub

                if mag2_rz < -1e-10 or mag2_rx < -1e-10:
                    valid = False
                    break

                mags_rz.append(np.sqrt(max(0, mag2_rz)))
                mags_rx.append(np.sqrt(max(0, mag2_rx)))

            if not valid:
                continue

            mags_rz = np.array(mags_rz)
            mags_rx = np.array(mags_rx)

            # Normalize so that F gate (index 3) = 1.0
            if mags_rz[3] < 1e-10 or mags_rx[3] < 1e-10:
                continue

            rz_n = mags_rz / mags_rz[3]
            rx_n = mags_rx / mags_rx[3]

            # Score against targets
            score = np.sum((rz_n - TARGET_RZ)**2) + np.sum((rx_n - TARGET_RX)**2)

            all_results.append({
                'omitted': omitted,
                'w_rz': w_rz_idx,
                'w_rx': w_rx_idx,
                'rz': rz_n,
                'rx': rx_n,
                'score': score,
                'mags_rz_raw': mags_rz,
                'mags_rx_raw': mags_rx,
            })

            if score < best_score:
                best_score = score
                best_config = all_results[-1]

# Sort by score
all_results.sort(key=lambda x: x['score'])

print(f"\nTotal configurations tested: {len(all_results)}")
print(f"\nTop 10 best matches:")
print(f"{'Rank':>5} | {'Omit':>4} | {'w_rz':>5} | {'w_rx':>5} | {'Score':>8} |"
      f" {'S_rz':>5} {'S_rx':>5} | {'R_rz':>5} {'R_rx':>5} | {'T_rz':>5} {'T_rx':>5} |"
      f" {'F_rz':>5} {'F_rx':>5} | {'P_rz':>5} {'P_rx':>5}")
print("-" * 120)
for rank, r in enumerate(all_results[:10]):
    print(f"{rank+1:>5} | {r['omitted']:>4} | w{r['w_rz']+1:>4} | w{r['w_rx']+1:>4} | {r['score']:>8.4f} |"
          f" {r['rz'][0]:>5.3f} {r['rx'][0]:>5.3f} | {r['rz'][1]:>5.3f} {r['rx'][1]:>5.3f} |"
          f" {r['rz'][2]:>5.3f} {r['rx'][2]:>5.3f} | {r['rz'][3]:>5.3f} {r['rx'][3]:>5.3f} |"
          f" {r['rz'][4]:>5.3f} {r['rx'][4]:>5.3f}")

print(f"\nTarget: {' '*35}|"
      f" {TARGET_RZ[0]:>5.3f} {TARGET_RX[0]:>5.3f} | {TARGET_RZ[1]:>5.3f} {TARGET_RX[1]:>5.3f} |"
      f" {TARGET_RZ[2]:>5.3f} {TARGET_RX[2]:>5.3f} | {TARGET_RZ[3]:>5.3f} {TARGET_RX[3]:>5.3f} |"
      f" {TARGET_RZ[4]:>5.3f} {TARGET_RX[4]:>5.3f}")

if best_config:
    bc = best_config
    print(f"\n*** BEST CONFIGURATION ***")
    print(f"  Pentagon omits E6 node {bc['omitted']+1}")
    print(f"  Rz direction = omega_{bc['w_rz']+1}")
    print(f"  Rx direction = omega_{bc['w_rx']+1}")
    print(f"  Score = {bc['score']:.6f}")
    print(f"\n  {'Gate':>6} | {'rz':>8} | {'rx':>8} | {'rz_target':>10} | {'rx_target':>10} | {'rz_err':>8} | {'rx_err':>8}")
    print(f"  {'-'*70}")
    for i, g in enumerate(GATE_NAMES):
        print(f"  {g:>6} | {bc['rz'][i]:>8.4f} | {bc['rx'][i]:>8.4f} |"
              f" {TARGET_RZ[i]:>10.1f} | {TARGET_RX[i]:>10.1f} |"
              f" {bc['rz'][i]-TARGET_RZ[i]:>+8.4f} | {bc['rx'][i]-TARGET_RX[i]:>+8.4f}")
    print(f"\n  Raw magnitudes (before normalization):")
    for i, g in enumerate(GATE_NAMES):
        print(f"    {g}: rz_raw = {bc['mags_rz_raw'][i]:.6f}, rx_raw = {bc['mags_rx_raw'][i]:.6f}")

# ============================================================================
# STEP 3: Also try SIMPLE ROOT projections (not just fund. weights)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SIMPLE ROOT PROJECTIONS")
print("=" * 70)

best_score_sr = np.inf
best_config_sr = None
sr_results = []

for omitted in range(6):
    pentagon = [n for n in range(6) if n != omitted]

    for sr_rz_idx in range(6):
        for sr_rx_idx in range(6):
            if sr_rz_idx == sr_rx_idx:
                continue

            # Simple roots: just the standard basis in Dynkin label space
            # projected via the Cartan matrix
            alpha_rz = A_E6[sr_rz_idx, :]
            alpha_rx = A_E6[sr_rx_idx, :]

            mags_rz = []
            mags_rx = []
            valid = True

            for gate_idx in range(5):
                remaining = [pentagon[k] for k in range(5) if k != gate_idx]
                idx = remaining

                G_sub = A_E6[np.ix_(idx, idx)]
                try:
                    G_sub_inv = np.linalg.inv(G_sub)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                a_rz_sub = alpha_rz[idx]
                a_rx_sub = alpha_rx[idx]

                mag2_rz = a_rz_sub @ G_sub_inv @ a_rz_sub
                mag2_rx = a_rx_sub @ G_sub_inv @ a_rx_sub

                if mag2_rz < -1e-10 or mag2_rx < -1e-10:
                    valid = False
                    break

                mags_rz.append(np.sqrt(max(0, mag2_rz)))
                mags_rx.append(np.sqrt(max(0, mag2_rx)))

            if not valid:
                continue

            mags_rz = np.array(mags_rz)
            mags_rx = np.array(mags_rx)

            if mags_rz[3] < 1e-10 or mags_rx[3] < 1e-10:
                continue

            rz_n = mags_rz / mags_rz[3]
            rx_n = mags_rx / mags_rx[3]

            score = np.sum((rz_n - TARGET_RZ)**2) + np.sum((rx_n - TARGET_RX)**2)

            sr_results.append({
                'omitted': omitted,
                'a_rz': sr_rz_idx,
                'a_rx': sr_rx_idx,
                'rz': rz_n,
                'rx': rx_n,
                'score': score,
            })

            if score < best_score_sr:
                best_score_sr = score
                best_config_sr = sr_results[-1]

sr_results.sort(key=lambda x: x['score'])

print(f"\nTop 5 simple root configurations:")
for rank, r in enumerate(sr_results[:5]):
    print(f"  {rank+1}. Omit node {r['omitted']+1}, alpha_{r['a_rz']+1}->Rz, alpha_{r['a_rx']+1}->Rx,"
          f" score={r['score']:.4f}")
    for i, g in enumerate(GATE_NAMES):
        print(f"     {g}: rz={r['rz'][i]:.4f}, rx={r['rx'][i]:.4f}")

# ============================================================================
# STEP 4: Try MIXED: one fundamental weight, one simple root
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MIXED (WEIGHT + ROOT) PROJECTIONS")
print("=" * 70)

best_score_mix = np.inf
best_config_mix = None
mix_results = []

for omitted in range(6):
    pentagon = [n for n in range(6) if n != omitted]

    for w_idx in range(6):
        for a_idx in range(6):
            # Try omega_w as Rz, alpha_a as Rx (and vice versa)
            for swap in [False, True]:
                if swap:
                    d_rz = A_E6[a_idx, :]  # simple root
                    d_rx = A_E6_inv[w_idx, :]  # fund weight
                    label = f"alpha_{a_idx+1}->Rz, omega_{w_idx+1}->Rx"
                else:
                    d_rz = A_E6_inv[w_idx, :]  # fund weight
                    d_rx = A_E6[a_idx, :]  # simple root
                    label = f"omega_{w_idx+1}->Rz, alpha_{a_idx+1}->Rx"

                mags_rz = []
                mags_rx = []
                valid = True

                for gate_idx in range(5):
                    remaining = [pentagon[k] for k in range(5) if k != gate_idx]
                    idx = remaining

                    G_sub = A_E6[np.ix_(idx, idx)]
                    try:
                        G_sub_inv = np.linalg.inv(G_sub)
                    except np.linalg.LinAlgError:
                        valid = False
                        break

                    d_rz_sub = d_rz[idx]
                    d_rx_sub = d_rx[idx]

                    mag2_rz = d_rz_sub @ G_sub_inv @ d_rz_sub
                    mag2_rx = d_rx_sub @ G_sub_inv @ d_rx_sub

                    if mag2_rz < -1e-10 or mag2_rx < -1e-10:
                        valid = False
                        break

                    mags_rz.append(np.sqrt(max(0, mag2_rz)))
                    mags_rx.append(np.sqrt(max(0, mag2_rx)))

                if not valid:
                    continue

                mags_rz = np.array(mags_rz)
                mags_rx = np.array(mags_rx)

                if mags_rz[3] < 1e-10 or mags_rx[3] < 1e-10:
                    continue

                rz_n = mags_rz / mags_rz[3]
                rx_n = mags_rx / mags_rx[3]

                score = np.sum((rz_n - TARGET_RZ)**2) + np.sum((rx_n - TARGET_RX)**2)

                if score < best_score_mix:
                    best_score_mix = score
                    best_config_mix = {
                        'omitted': omitted, 'label': label,
                        'rz': rz_n, 'rx': rx_n, 'score': score,
                    }

print(f"  Best mixed score: {best_score_mix:.4f}")
if best_config_mix:
    bm = best_config_mix
    print(f"  Config: omit node {bm['omitted']+1}, {bm['label']}")
    for i, g in enumerate(GATE_NAMES):
        print(f"    {g}: rz={bm['rz'][i]:.4f} (target {TARGET_RZ[i]}), rx={bm['rx'][i]:.4f} (target {TARGET_RX[i]})")

# ============================================================================
# STEP 5: Try ARBITRARY 6-vectors for Rz and Rx
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: OPTIMAL Rz/Rx DIRECTIONS VIA LEAST-SQUARES")
print("=" * 70)

# For the best pentagon (omitted node), find the optimal 6D vectors d_rz, d_rx
# that minimize || projected_mags - targets ||

from scipy.optimize import minimize

def objective(params, pentagon, target_rz, target_rx, normalize_idx=3):
    """Given 12 parameters (two 6-vectors), compute projection magnitudes and score."""
    d_rz = params[:6]
    d_rx = params[6:]

    mags_rz = np.zeros(5)
    mags_rx = np.zeros(5)

    for gate_idx in range(5):
        remaining = [pentagon[k] for k in range(5) if k != gate_idx]
        idx = remaining

        G_sub = A_E6[np.ix_(idx, idx)]
        try:
            G_sub_inv = np.linalg.inv(G_sub)
        except:
            return 1e10

        d_rz_sub = d_rz[idx]
        d_rx_sub = d_rx[idx]

        mag2_rz = d_rz_sub @ G_sub_inv @ d_rz_sub
        mag2_rx = d_rx_sub @ G_sub_inv @ d_rx_sub

        mags_rz[gate_idx] = np.sqrt(max(0, mag2_rz))
        mags_rx[gate_idx] = np.sqrt(max(0, mag2_rx))

    if mags_rz[normalize_idx] < 1e-10 or mags_rx[normalize_idx] < 1e-10:
        return 1e10

    rz_n = mags_rz / mags_rz[normalize_idx]
    rx_n = mags_rx / mags_rx[normalize_idx]

    return np.sum((rz_n - target_rz)**2) + np.sum((rx_n - target_rx)**2)

for omitted in range(6):
    pentagon = [n for n in range(6) if n != omitted]

    best_opt_score = np.inf
    best_opt_result = None

    for trial in range(30):
        np.random.seed(trial * 7 + omitted * 37)
        x0 = np.random.randn(12) * 0.5
        res = minimize(objective, x0, args=(pentagon, TARGET_RZ, TARGET_RX),
                       method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_opt_score:
            best_opt_score = res.fun
            best_opt_result = res

    if best_opt_score < 0.01:
        d_rz_opt = best_opt_result.x[:6]
        d_rx_opt = best_opt_result.x[6:]
        # Compute the actual values
        pentagon_arr = pentagon
        mags_rz = np.zeros(5)
        mags_rx = np.zeros(5)
        for gate_idx in range(5):
            remaining = [pentagon_arr[k] for k in range(5) if k != gate_idx]
            idx = remaining
            G_sub = A_E6[np.ix_(idx, idx)]
            G_sub_inv = np.linalg.inv(G_sub)
            mags_rz[gate_idx] = np.sqrt(max(0, d_rz_opt[idx] @ G_sub_inv @ d_rz_opt[idx]))
            mags_rx[gate_idx] = np.sqrt(max(0, d_rx_opt[idx] @ G_sub_inv @ d_rx_opt[idx]))

        rz_n = mags_rz / mags_rz[3]
        rx_n = mags_rx / mags_rx[3]

        print(f"\n  *** MATCH: Omit node {omitted+1}, score = {best_opt_score:.8f} ***")
        for i, g in enumerate(GATE_NAMES):
            print(f"    {g}: rz={rz_n[i]:.6f} (target {TARGET_RZ[i]}), rx={rx_n[i]:.6f} (target {TARGET_RX[i]})")
        print(f"    d_rz = {d_rz_opt}")
        print(f"    d_rx = {d_rx_opt}")

        # Express in terms of fundamental weights
        print(f"\n    d_rz in omega basis: {A_E6 @ d_rz_opt}")
        print(f"    d_rx in omega basis: {A_E6 @ d_rx_opt}")
    else:
        print(f"  Omit node {omitted+1}: best score = {best_opt_score:.6f} (no match < 0.01)")

# ============================================================================
# STEP 6: Casimir approach
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: CASIMIR EIGENVALUES")
print("=" * 70)

# SU(5) = A4 quadratic Casimir for fundamental representations
# For SU(N), the quadratic Casimir of the k-th antisymmetric rep (dim = C(N,k)):
# C_2(omega_k) = k(N-k)(N+1) / (2N)
# For SU(5), N=5:
print("\nSU(5) = A4 quadratic Casimir for fundamental reps:")
N = 5
for k in range(1, N):
    C2 = k * (N - k) * (N + 1) / (2 * N)
    dim = 1
    for j in range(k):
        dim = dim * (N - j) // (j + 1)
    print(f"  omega_{k} ({dim}-dim rep): C_2 = {k}*{N-k}*{N+1}/(2*{N}) = {C2:.4f} = {Fraction(C2).limit_denominator(100)}")

# ============================================================================
# STEP 7: Rational structure analysis
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: RATIONAL STRUCTURE OF MODULATION VALUES")
print("=" * 70)

h = 12  # Coxeter number of E6
r = 6   # rank of E6

vals_rz = [4, 13, 7, 10, 15]
vals_rx = [13, 4, 7, 10, 18]

print("\nAll values in units of 1/10:")
print(f"  {'Gate':>5} | {'rz*10':>6} | {'rx*10':>6} | {'sum':>5} | {'product':>8} | {'rz frac':>10} | {'rx frac':>10}")
print(f"  {'-'*65}")
for i, g in enumerate(GATE_NAMES):
    s = vals_rz[i] + vals_rx[i]
    p = vals_rz[i] * vals_rx[i]
    fz = Fraction(vals_rz[i], 10)
    fx = Fraction(vals_rx[i], 10)
    print(f"  {g:>5} | {vals_rz[i]:>6} | {vals_rx[i]:>6} | {s:>5} | {p:>8} | {fz:>10} | {fx:>10}")

total_rz = sum(vals_rz)
total_rx = sum(vals_rx)
grand = total_rz + total_rx
print(f"\n  Total rz*10 = {total_rz}")
print(f"  Total rx*10 = {total_rx}")
print(f"  Grand total = {grand}")
print(f"  101 is prime: {all(101 % k != 0 for k in range(2, 11))}")

print(f"\nE6 connections:")
print(f"  h = {h} (Coxeter number)")
print(f"  r = {r} (rank)")
print(f"  dim(E6) = 78")
print(f"  |W(E6)| = 51840")

print(f"\nNumerator interpretations:")
interpretations = {
    4: [f"r-2 = {r-2}", f"2r/3 = {2*r//3}", "dim(A1) = 3+1?"],
    7: [f"r+1 = {r+1}", f"dim(G2) Coxeter = 6+1?"],
    10: [f"dim(A4) pos roots = {4*5//2}", f"reference (F)"],
    13: [f"h+1 = {h+1}", f"dim(B2) = 10+3?"],
    15: [f"h+3 = {h+3}", f"dim(A5) pos roots = {5*6//2}",
         "*** ROUTE C CROSSING STEP ***"],
    18: [f"3*r = 3*{r} = {3*r}", f"3h/2 = {3*h//2}",
         f"dim(D3) pos roots = {3*(3-1)//2 * 2}... no, = {3*2} no"],
}

for val, interps in sorted(interpretations.items()):
    print(f"\n  {val}/10 = {Fraction(val, 10)}:")
    for interp in interps:
        print(f"    - {interp}")

print(f"\n  KEY: 15 = h+3 = Coxeter + Eisenstein face count")
print(f"  Route C: alpha^-1 = 137.036 at step 15 of Berry phase accumulation")
print(f"  P gate modulation rz = 15/10 = 3/2")
print(f"  If 15 = h+3, then crossing step = P gate coupling constant")

# Product structure
print(f"\n\nProduct structure (rz * rx in units of 1/100):")
for i, g in enumerate(GATE_NAMES):
    p = vals_rz[i] * vals_rx[i]
    print(f"  {g}: {vals_rz[i]} * {vals_rx[i]} = {p} = {Fraction(p, 100)}")
print(f"  Note: S*R products = 52 = 4*13")
print(f"  T product = 49 = 7^2")
print(f"  F product = 100 = 10^2")
print(f"  P product = 270 = 15*18 = 2*3^3*5")

# Sum structure
print(f"\n\nSum structure (rz + rx in units of 1/10):")
sums = []
for i, g in enumerate(GATE_NAMES):
    s = vals_rz[i] + vals_rx[i]
    sums.append(s)
    print(f"  {g}: {vals_rz[i]} + {vals_rx[i]} = {s} = {Fraction(s, 10)}")
print(f"\n  S,R sums = 17 = prime")
print(f"  T sum = 14 = 2*7")
print(f"  F sum = 20 = 4*5")
print(f"  P sum = 33 = 3*11")

# Difference structure (rz - rx)
print(f"\n\nDifference structure (rz - rx in units of 1/10):")
for i, g in enumerate(GATE_NAMES):
    d = vals_rz[i] - vals_rx[i]
    print(f"  {g}: {vals_rz[i]} - {vals_rx[i]} = {d:+d}")
print(f"  S: -9, R: +9, T: 0, F: 0, P: -3")
print(f"  Note: S+R differences cancel: -9+9 = 0")
print(f"  P difference = -3 = -Eisenstein face count")
print(f"  The asymmetry is 9 = 3^2 for S,R and 3 = 3^1 for P")

# ============================================================================
# STEP 8: Check if values relate to A4 weight lattice
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: A4 WEIGHT LATTICE CHECK")
print("=" * 70)

# The A4 = SU(5) weight lattice has fundamental weights omega_1,...,omega_4
# In the orthogonal basis (e_i - e_{i+1}), the weights of the 5-dim rep are:
# e_1, e_2, e_3, e_4, e_5 (with sum = 0 constraint)
# e_i = (4-i)/5 * delta_1 + ... (complicated in Dynkin label basis)

# The 4 fundamental weights of A4:
A_A4 = np.array([
    [ 2, -1,  0,  0],
    [-1,  2, -1,  0],
    [ 0, -1,  2, -1],
    [ 0,  0, -1,  2]
], dtype=float)

A_A4_inv = np.linalg.inv(A_A4)
print(f"\nA4 inverse Cartan (metric on weight space):")
for row in A_A4_inv:
    print(f"  [{', '.join([f'{x:6.3f}' for x in row])}]")

print(f"\n  Diagonal: {np.diag(A_A4_inv)}")
print(f"  Row sums: {np.sum(A_A4_inv, axis=1)}")

# For A4 = SU(5): A^{-1}_{ij} = min(i,j)(5-max(i,j))/5 (1-indexed)
print(f"\n  A4 metric formula: A^(-1)_{{ij}} = min(i,j)*(5-max(i,j))/5")
for i in range(4):
    for j in range(4):
        formula = (min(i+1,j+1) * (5 - max(i+1,j+1))) / 5
        print(f"    ({i+1},{j+1}): formula = {formula:.3f}, actual = {A_A4_inv[i,j]:.3f}, "
              f"match = {np.isclose(formula, A_A4_inv[i,j])}")

# Check: are any modulation values equal to A4^{-1} diagonal elements?
print(f"\n  A4^(-1) diagonal = {np.diag(A_A4_inv)}")
print(f"  Values * 5 = {np.diag(A_A4_inv) * 5}")
print(f"  These are: 4/5, 6/5, 6/5, 4/5")
print(f"  Note: 4/5 = 0.8, 6/5 = 1.2")
print(f"  From Analysis 22: |V2(alpha_4)|^2 = 6/5, |V2(alpha_5)|^2 = 4/5 !!!")
print(f"  SAME numbers appear in the A4 inverse Cartan!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Best fundamental weight projection:
  Score = {best_score:.6f} {'(MATCH!)' if best_score < 0.1 else '(no match)'}

Best simple root projection:
  Score = {best_score_sr:.6f} {'(MATCH!)' if best_score_sr < 0.1 else '(no match)'}

Best mixed projection:
  Score = {best_score_mix:.6f} {'(MATCH!)' if best_score_mix < 0.1 else '(no match)'}

Rational structure of values (denominator 10):
  S: 4/10, 13/10 | R: 13/10, 4/10 | T: 7/10, 7/10 | F: 1, 1 | P: 15/10, 18/10

  4  = rank(E6) - 2
  7  = rank(E6) + 1
  13 = h(E6) + 1
  15 = h(E6) + 3 = ROUTE C CROSSING STEP
  18 = 3 * rank(E6)

  The S<->R swap (4 <-> 13) has difference 9 = 3^2
  The P asymmetry (15 vs 18) has difference 3 = 3^1
  All differences are powers of 3 (Eisenstein!)

  Grand total = 101 (prime)
""")

# Save output
with open(f"{OUT}/analysis23_projection.txt", 'w') as f:
    f.write("Analysis 23 output saved. See console output.\n")

print(f"All output in: {OUT}")
