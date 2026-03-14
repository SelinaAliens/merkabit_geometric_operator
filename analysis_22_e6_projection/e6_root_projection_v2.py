#!/usr/bin/env python3
"""
E6 ROOT PROJECTION — WEIGHT DECOMPOSITION APPROACH
====================================================
Key insight from v1: V2(alpha) = n4*V2(alpha4) + n5*V2(alpha5)
since alpha0...alpha3 project to zero in V2.

So the modulation factors are entirely determined by the distribution
of (n4, n5) coefficients across roots grouped by pentachoric vertex.

This script:
1. Computes the (n4, n5) weight for every positive root
2. Groups roots by which pentachoric vertex they involve
3. Extracts the per-vertex weight structure
4. Tests whether (n4, n5) ratios reproduce the ad hoc (Rz, Rx) modulation
5. Explores all reasonable identification schemes (not just one)
"""
import numpy as np
from scipy.linalg import cholesky, null_space
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"

# ============================================================================
# E6 ROOT SYSTEM (same as v1)
# ============================================================================
CARTAN = np.array([
    [ 2, -1,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0, -1],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2,  0],
    [ 0,  0, -1,  0,  0,  2]
], dtype=float)

L_chol = cholesky(CARTAN, lower=True)
SIMPLE_ROOTS = L_chol.copy()

def generate_positive_roots():
    roots = set()
    for i in range(6):
        r = [0]*6; r[i] = 1
        roots.add(tuple(r))
    changed = True
    while changed:
        changed = False
        to_add = []
        for rt in list(roots):
            n = np.array(rt, dtype=int)
            for i in range(6):
                inner = sum(n[j] * CARTAN[j, i] for j in range(6))
                if inner < 0:
                    new_n = list(n)
                    new_n[i] += 1
                    new_t = tuple(new_n)
                    if new_t not in roots and all(c >= 0 for c in new_n):
                        to_add.append(new_t)
                        changed = True
        for r in to_add:
            roots.add(r)
    return sorted(roots)

def to_euclid(coeffs):
    return sum(coeffs[i] * SIMPLE_ROOTS[i] for i in range(6))

def build_subspace_decomposition():
    V4_raw = SIMPLE_ROOTS[:4]
    Q4, _ = np.linalg.qr(V4_raw.T)
    V2 = null_space(Q4.T)
    return Q4, V2

def map_root_to_vertices(coeffs):
    n = np.array(coeffs[:6], dtype=int)
    a4_support = set()
    for k in range(4):
        if n[k] > 0:
            a4_support.add(k+1)
            a4_support.add(k+2)
    if len(a4_support) == 0:
        extra_support = set()
        if n[4] > 0:
            extra_support.add(4)
            extra_support.add(5)
        if n[5] > 0:
            extra_support.add(3)
            extra_support.add(4)
        return extra_support
    return a4_support

# ============================================================================
# MAIN COMPUTATION
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("E6 ROOT PROJECTION — WEIGHT DECOMPOSITION")
    print("=" * 70)

    pos_root_coeffs = generate_positive_roots()
    Q4, V2 = build_subspace_decomposition()

    # Verify V2 structure
    alpha4_v2 = V2.T @ to_euclid(np.array([0,0,0,0,1,0], dtype=float))
    alpha5_v2 = V2.T @ to_euclid(np.array([0,0,0,0,0,1], dtype=float))
    print(f"\nalpha_4 V2 projection: ({alpha4_v2[0]:.6f}, {alpha4_v2[1]:.6f}), |V2| = {np.linalg.norm(alpha4_v2):.6f}")
    print(f"alpha_5 V2 projection: ({alpha5_v2[0]:.6f}, {alpha5_v2[1]:.6f}), |V2| = {np.linalg.norm(alpha5_v2):.6f}")

    cos_angle = np.dot(alpha4_v2, alpha5_v2) / (np.linalg.norm(alpha4_v2) * np.linalg.norm(alpha5_v2))
    angle_45 = np.arccos(np.clip(cos_angle, -1, 1))
    print(f"Angle between alpha_4 and alpha_5 in V2: {angle_45*180/np.pi:.4f} deg")
    print(f"cos(angle) = {cos_angle:.6f}")
    # Check if cos = -sqrt(6)/4
    print(f"-sqrt(6)/4 = {-np.sqrt(6)/4:.6f}")
    print(f"EXACT? {np.isclose(cos_angle, -np.sqrt(6)/4)}")

    # ================================================================
    # (n4, n5) WEIGHT DECOMPOSITION
    # ================================================================
    print(f"\n{'='*70}")
    print(f"(n4, n5) WEIGHT DECOMPOSITION")
    print(f"{'='*70}")

    root_data = []
    n4n5_groups = {}

    for coeffs in pos_root_coeffs:
        n = np.array(coeffs, dtype=int)
        n4, n5 = int(n[4]), int(n[5])
        vertices = map_root_to_vertices(coeffs)
        rd = {
            'coeffs': coeffs,
            'n4': n4, 'n5': n5,
            'vertices': vertices,
            'is_a4': (n4 == 0 and n5 == 0),
        }
        root_data.append(rd)

        key = (n4, n5)
        if key not in n4n5_groups:
            n4n5_groups[key] = []
        n4n5_groups[key].append(rd)

    print(f"\nDistinct (n4, n5) values in positive roots:")
    for key in sorted(n4n5_groups.keys()):
        group = n4n5_groups[key]
        print(f"  ({key[0]}, {key[1]}): {len(group)} roots")

    # ================================================================
    # PER-VERTEX WEIGHT STATISTICS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PER-VERTEX (n4, n5) STATISTICS")
    print(f"{'='*70}")

    gate_names = ['S', 'R', 'T', 'F', 'P']
    ad_hoc = {
        'S': {'rz': 0.4, 'rx': 1.3},
        'R': {'rz': 1.3, 'rx': 0.4},
        'T': {'rz': 0.7, 'rx': 0.7},
        'F': {'rz': 1.0, 'rx': 1.0},
        'P': {'rz': 1.5, 'rx': 1.8},
    }

    vertex_stats = {}
    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and not rd['is_a4']]

        n4_vals = [rd['n4'] for rd in involved]
        n5_vals = [rd['n5'] for rd in involved]

        total_n4 = sum(n4_vals)
        total_n5 = sum(n5_vals)
        mean_n4 = np.mean(n4_vals) if len(n4_vals) > 0 else 0
        mean_n5 = np.mean(n5_vals) if len(n5_vals) > 0 else 0

        # (n4, n5) distribution
        n4n5_dist = {}
        for rd in involved:
            key = (rd['n4'], rd['n5'])
            n4n5_dist[key] = n4n5_dist.get(key, 0) + 1

        vertex_stats[vertex] = {
            'gate': gate,
            'count': len(involved),
            'total_n4': total_n4,
            'total_n5': total_n5,
            'mean_n4': mean_n4,
            'mean_n5': mean_n5,
            'n4n5_dist': n4n5_dist,
        }

        print(f"\n  Vertex {vertex} ({gate}): {len(involved)} extra roots")
        print(f"    Total n4 = {total_n4}, Total n5 = {total_n5}")
        print(f"    Mean  n4 = {mean_n4:.4f}, Mean  n5 = {mean_n5:.4f}")
        print(f"    n4/(n4+n5) = {total_n4/(total_n4+total_n5):.4f}" if (total_n4+total_n5) > 0 else "")
        print(f"    Distribution:")
        for key in sorted(n4n5_dist.keys()):
            print(f"      (n4={key[0]}, n5={key[1]}): {n4n5_dist[key]} roots")

    # ================================================================
    # COMPARISON TABLE: Various mappings
    # ================================================================
    print(f"\n{'='*70}")
    print(f"COMPARISON: WEIGHT STATISTICS vs AD HOC")
    print(f"{'='*70}")

    print(f"\n  {'Gate':>6} | {'Vertex':>7} | {'mean_n4':>8} | {'mean_n5':>8} | {'n4/(n4+n5)':>12} | {'ad_hoc_rz':>10} | {'ad_hoc_rx':>10}")
    print(f"  {'-'*75}")
    for vertex in range(1, 6):
        s = vertex_stats[vertex]
        frac = s['total_n4']/(s['total_n4']+s['total_n5']) if (s['total_n4']+s['total_n5']) > 0 else 0
        gate = s['gate']
        ah = ad_hoc[gate]
        print(f"  {gate:>6} | {vertex:>7} | {s['mean_n4']:>8.4f} | {s['mean_n5']:>8.4f} | {frac:>12.4f} | {ah['rz']:>10.1f} | {ah['rx']:>10.1f}")

    # ================================================================
    # TRY ALL POSSIBLE MAPPINGS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE MAP SEARCH")
    print(f"{'='*70}")
    print(f"Testing all possible identifications of (n4, n5) with (rz, rx)")
    print(f"and all vertex-to-gate permutations.\n")

    # Observable per vertex: mean_n4, mean_n5, total_n4, total_n5, count,
    # ratio n4/(n4+n5), count/total_count

    # Ad hoc targets for each gate
    target = np.array([[0.4, 1.3],  # S
                       [1.3, 0.4],  # R
                       [0.7, 0.7],  # T
                       [1.0, 1.0],  # F
                       [1.5, 1.8]]) # P

    # Observables to try: for each vertex, define two numbers that might map to (rz, rx)
    obs_methods = {}

    # Method 1: (mean_n4, mean_n5)
    obs_methods['mean_n4_n5'] = np.array([[vertex_stats[v]['mean_n4'], vertex_stats[v]['mean_n5']] for v in range(1, 6)])

    # Method 2: (mean_n5, mean_n4) [swap]
    obs_methods['mean_n5_n4'] = np.array([[vertex_stats[v]['mean_n5'], vertex_stats[v]['mean_n4']] for v in range(1, 6)])

    # Method 3: (total_n4, total_n5)
    obs_methods['total_n4_n5'] = np.array([[vertex_stats[v]['total_n4'], vertex_stats[v]['total_n5']] for v in range(1, 6)])

    # Method 4: (count, n4/(n4+n5))
    obs_methods['count_ratio'] = np.array([
        [vertex_stats[v]['count'],
         vertex_stats[v]['total_n4']/(vertex_stats[v]['total_n4']+vertex_stats[v]['total_n5']) if (vertex_stats[v]['total_n4']+vertex_stats[v]['total_n5'])>0 else 0]
        for v in range(1, 6)])

    # Method 5: Use ABSENT vertex approach — roots NOT involving vertex v
    absent_stats = {}
    for vertex in range(1, 6):
        active = [rd for rd in root_data if vertex not in rd['vertices'] and not rd['is_a4']]
        n4_vals = [rd['n4'] for rd in active]
        n5_vals = [rd['n5'] for rd in active]
        absent_stats[vertex] = {
            'count': len(active),
            'mean_n4': np.mean(n4_vals) if len(n4_vals) > 0 else 0,
            'mean_n5': np.mean(n5_vals) if len(n5_vals) > 0 else 0,
            'total_n4': sum(n4_vals),
            'total_n5': sum(n5_vals),
        }

    obs_methods['absent_mean_n4_n5'] = np.array([[absent_stats[v]['mean_n4'], absent_stats[v]['mean_n5']] for v in range(1, 6)])
    obs_methods['absent_mean_n5_n4'] = np.array([[absent_stats[v]['mean_n5'], absent_stats[v]['mean_n4']] for v in range(1, 6)])
    obs_methods['absent_total_n4_n5'] = np.array([[absent_stats[v]['total_n4'], absent_stats[v]['total_n5']] for v in range(1, 6)])

    # Method: use count of extra roots per vertex (size of representation slice)
    # and the n5/n4 ratio as asymmetry measure
    obs_methods['count_and_n5overn4'] = np.array([
        [vertex_stats[v]['count'],
         vertex_stats[v]['total_n5']/(vertex_stats[v]['total_n4']+0.01)]
        for v in range(1, 6)])

    # For each observable method, try:
    # 1. Linear fit: find best a, b such that obs = a * target + b
    # 2. Ratio fit: normalize both to mean=1, compare

    best_score = np.inf
    best_method = None
    best_perm = None
    best_swap = None

    for method_name, obs in obs_methods.items():
        # Try all 5! permutations of vertex-to-gate mapping
        # AND both (rz, rx) orderings
        for perm in permutations(range(5)):
            for swap in [False, True]:
                t = target[list(perm)]
                if swap:
                    t = t[:, ::-1]

                # Normalize both to have same mean and std in each column
                obs_norm = np.zeros_like(obs)
                t_norm = np.zeros_like(t)
                for col in range(2):
                    o_mean, o_std = obs[:, col].mean(), obs[:, col].std()
                    t_mean, t_std = t[:, col].mean(), t[:, col].std()
                    if o_std > 1e-10 and t_std > 1e-10:
                        obs_norm[:, col] = (obs[:, col] - o_mean) / o_std
                        t_norm[:, col] = (t[:, col] - t_mean) / t_std
                    else:
                        obs_norm[:, col] = 0
                        t_norm[:, col] = 0

                # Score: sum of squared differences
                score = np.sum((obs_norm - t_norm)**2)
                if score < best_score:
                    best_score = score
                    best_method = method_name
                    best_perm = perm
                    best_swap = swap

    print(f"  Best normalized match:")
    print(f"    Method: {best_method}")
    print(f"    Vertex permutation: {best_perm} (vertex {[p+1 for p in best_perm]} -> gates {gate_names})")
    print(f"    Swap (rz, rx): {best_swap}")
    print(f"    Score: {best_score:.4f} (0 = perfect)")

    # Show the best match in detail
    obs = obs_methods[best_method]
    t = target[list(best_perm)]
    if best_swap:
        t = t[:, ::-1]

    print(f"\n  {'Vertex':>8} | {'Obs col1':>10} | {'Obs col2':>10} | {'Target rz':>10} | {'Target rx':>10}")
    print(f"  {'-'*55}")
    for v in range(5):
        print(f"  {v+1:>8} | {obs[v, 0]:>10.4f} | {obs[v, 1]:>10.4f} | {t[v, 0]:>10.2f} | {t[v, 1]:>10.2f}")

    # Pearson correlations
    from scipy import stats
    r1, p1 = stats.pearsonr(obs[:, 0], t[:, 0])
    r2, p2 = stats.pearsonr(obs[:, 1], t[:, 1])
    r_all, p_all = stats.pearsonr(obs.flatten(), t.flatten())
    print(f"\n  Pearson r (col1): {r1:.4f} (p={p1:.4f})")
    print(f"  Pearson r (col2): {r2:.4f} (p={p2:.4f})")
    print(f"  Pearson r (all):  {r_all:.4f} (p={p_all:.4f})")

    # ================================================================
    # DEEPER: n4, n5 AS COORDINATES IN WEIGHT LATTICE
    # ================================================================
    print(f"\n{'='*70}")
    print(f"WEIGHT LATTICE STRUCTURE")
    print(f"{'='*70}")

    # Each positive root's (n4, n5) weight. What's the pattern?
    print(f"\n  All positive roots in (n4, n5) weight space:")
    for key in sorted(n4n5_groups.keys()):
        group = n4n5_groups[key]
        # How many involve each vertex?
        v_counts = {v: 0 for v in range(1, 6)}
        for rd in group:
            for v in rd['vertices']:
                if 1 <= v <= 5:
                    v_counts[v] += 1
        v_str = ", ".join([f"V{v}:{v_counts[v]}" for v in range(1, 6)])
        print(f"  (n4={key[0]}, n5={key[1]}): {len(group)} roots | {v_str}")

    # ================================================================
    # THE KEY TEST: Per-vertex (n4, n5) center of mass
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PER-VERTEX CENTER OF MASS IN (n4, n5) SPACE")
    print(f"{'='*70}")

    print(f"\n  {'Vertex':>8} | {'Gate':>6} | {'<n4>':>8} | {'<n5>':>8} | {'<n4>-<n5>':>10} | {'<n4>/<n5>':>10} | {'sqrt(n4^2+n5^2)':>15}")
    print(f"  {'-'*75}")

    vertex_centers = {}
    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and not rd['is_a4']]

        n4s = np.array([rd['n4'] for rd in involved], dtype=float)
        n5s = np.array([rd['n5'] for rd in involved], dtype=float)

        mn4 = np.mean(n4s)
        mn5 = np.mean(n5s)
        ratio = mn4/mn5 if mn5 > 0 else np.inf
        mag = np.sqrt(mn4**2 + mn5**2)

        vertex_centers[vertex] = (mn4, mn5)
        print(f"  {vertex:>8} | {gate:>6} | {mn4:>8.4f} | {mn5:>8.4f} | {mn4-mn5:>10.4f} | {ratio:>10.4f} | {mag:>15.4f}")

    # ================================================================
    # COMPUTE: |V2|^2 per root = n4^2 |V2(a4)|^2 + n5^2 |V2(a5)|^2 + 2 n4 n5 V2(a4)*V2(a5)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"|V2|^2 DECOMPOSITION")
    print(f"{'='*70}")

    norm4_sq = np.dot(alpha4_v2, alpha4_v2)
    norm5_sq = np.dot(alpha5_v2, alpha5_v2)
    cross = np.dot(alpha4_v2, alpha5_v2)
    print(f"\n  |V2(a4)|^2 = {norm4_sq:.6f}")
    print(f"  |V2(a5)|^2 = {norm5_sq:.6f}")
    print(f"  V2(a4).V2(a5) = {cross:.6f}")
    print(f"\n  For root with coefficients (n4, n5):")
    print(f"    |V2|^2 = {norm4_sq:.4f}*n4^2 + {norm5_sq:.4f}*n5^2 + 2*({cross:.4f})*n4*n5")

    # Verify with known roots
    for key in sorted(n4n5_groups.keys()):
        n4, n5 = key
        predicted = norm4_sq * n4**2 + norm5_sq * n5**2 + 2 * cross * n4 * n5
        # Actual
        actual = np.linalg.norm(n4 * alpha4_v2 + n5 * alpha5_v2)**2
        print(f"    (n4={n4}, n5={n5}): predicted |V2|^2 = {predicted:.4f}, actual = {actual:.4f}, |V2| = {np.sqrt(actual):.4f}")

    # ================================================================
    # DIRECTIONAL DECOMPOSITION: project V2 onto ORTHOGONAL axes
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ORTHOGONAL DECOMPOSITION OF V2 CONTENT")
    print(f"{'='*70}")

    # Rotate V2 basis to align with alpha_4 direction
    # e1 = alpha4_v2 / |alpha4_v2|  (along alpha_4)
    # e2 = perpendicular
    e1 = alpha4_v2 / np.linalg.norm(alpha4_v2)
    e2 = np.array([-e1[1], e1[0]])  # 90-degree rotation

    print(f"\n  Orthogonal basis in V2:")
    print(f"    e1 (along a4): ({e1[0]:.6f}, {e1[1]:.6f})")
    print(f"    e2 (perp to a4): ({e2[0]:.6f}, {e2[1]:.6f})")

    # Project alpha_5 onto this basis
    a5_e1 = np.dot(alpha5_v2, e1)
    a5_e2 = np.dot(alpha5_v2, e2)
    print(f"    a5 in this basis: ({a5_e1:.6f}, {a5_e2:.6f})")
    print(f"    So V2 = n4*(|a4,V2|, 0) + n5*({a5_e1:.4f}, {a5_e2:.4f})")

    # For each root, compute (along_alpha4, perp_to_alpha4) components
    print(f"\n  Per-vertex orthogonal decomposition:")
    print(f"  {'Vertex':>8} | {'Gate':>6} | {'mean along a4':>15} | {'mean perp a4':>15} | {'RMS along':>12} | {'RMS perp':>12}")
    print(f"  {'-'*80}")

    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and not rd['is_a4']]

        along_vals = []
        perp_vals = []
        for rd in involved:
            v2_vec = rd['n4'] * alpha4_v2 + rd['n5'] * alpha5_v2
            along_vals.append(np.dot(v2_vec, e1))
            perp_vals.append(np.dot(v2_vec, e2))

        along = np.array(along_vals)
        perp = np.array(perp_vals)

        print(f"  {vertex:>8} | {gate:>6} | {np.mean(along):>15.4f} | {np.mean(perp):>15.4f} | {np.sqrt(np.mean(along**2)):>12.4f} | {np.sqrt(np.mean(perp**2)):>12.4f}")

    # ================================================================
    # ALTERNATIVE: Use (n4, n5) counts directly
    # ================================================================
    print(f"\n{'='*70}")
    print(f"DIRECT COUNT TABLE: roots at each (n4, n5) per vertex")
    print(f"{'='*70}")

    all_keys = sorted(n4n5_groups.keys())
    extra_keys = [k for k in all_keys if k != (0, 0)]

    header = f"  {'(n4,n5)':>10}"
    for v in range(1, 6):
        header += f" | V{v}({gate_names[v-1]})"
    header += " | Total"
    print(header)
    print(f"  {'-' * (12 + 6 * 10)}")

    for key in extra_keys:
        group = n4n5_groups[key]
        row = f"  ({key[0]},{key[1]}){' '*(7-len(f'({key[0]},{key[1]})'))} "
        for v in range(1, 6):
            count = sum(1 for rd in group if v in rd['vertices'])
            row += f" | {count:>7}"
        row += f" | {len(group):>5}"
        print(row)

    # Totals
    row = f"  {'Total':>10}"
    for v in range(1, 6):
        count = sum(1 for rd in root_data if v in rd['vertices'] and not rd['is_a4'])
        row += f" | {count:>7}"
    row += f" | {len([rd for rd in root_data if not rd['is_a4']]):>5}"
    print(row)

    # Weighted sum per vertex: sum of n4 and sum of n5
    row_n4 = f"  {'Sum n4':>10}"
    row_n5 = f"  {'Sum n5':>10}"
    for v in range(1, 6):
        involved = [rd for rd in root_data if v in rd['vertices'] and not rd['is_a4']]
        sn4 = sum(rd['n4'] for rd in involved)
        sn5 = sum(rd['n5'] for rd in involved)
        row_n4 += f" | {sn4:>7}"
        row_n5 += f" | {sn5:>7}"
    print(row_n4)
    print(row_n5)

    # ================================================================
    # CRITICAL TEST: Fraction of n5 weight per vertex
    # ================================================================
    print(f"\n{'='*70}")
    print(f"CRITICAL TEST: n5-FRACTION vs AD HOC ASYMMETRY")
    print(f"{'='*70}")

    print(f"\n  The ad hoc parameters have a clear pattern:")
    print(f"    S: rz < rx (rx-heavy)")
    print(f"    R: rz > rx (rz-heavy)")
    print(f"    T: rz = rx (symmetric)")
    print(f"    F: rz = rx (symmetric)")
    print(f"    P: rz < rx (rx-heavy, both large)")
    print(f"\n  The asymmetry rz/(rz+rx) in ad hoc:")
    for gate in gate_names:
        ah = ad_hoc[gate]
        frac = ah['rz'] / (ah['rz'] + ah['rx'])
        print(f"    {gate}: rz/(rz+rx) = {frac:.4f}")

    print(f"\n  The n4/(n4+n5) fraction per vertex:")
    for vertex in range(1, 6):
        s = vertex_stats[vertex]
        frac = s['total_n4']/(s['total_n4']+s['total_n5']) if (s['total_n4']+s['total_n5']) > 0 else 0
        print(f"    V{vertex} ({s['gate']}): n4/(n4+n5) = {frac:.4f}")

    print(f"\n  Ordering test:")
    print(f"    Ad hoc rz/(rz+rx) ordering: S < T = F < P < R")
    ad_hoc_fracs = {g: ad_hoc[g]['rz']/(ad_hoc[g]['rz']+ad_hoc[g]['rx']) for g in gate_names}
    ah_order = sorted(gate_names, key=lambda g: ad_hoc_fracs[g])
    print(f"    Sorted: {' < '.join([f'{g}({ad_hoc_fracs[g]:.3f})' for g in ah_order])}")

    n4_fracs = {}
    for vertex in range(1, 6):
        s = vertex_stats[vertex]
        n4_fracs[vertex] = s['total_n4']/(s['total_n4']+s['total_n5']) if (s['total_n4']+s['total_n5']) > 0 else 0
    v_order = sorted(range(1, 6), key=lambda v: n4_fracs[v])
    print(f"    n4/(n4+n5) sorted: {' < '.join([f'V{v}({gate_names[v-1]})({n4_fracs[v]:.3f})' for v in v_order])}")

    # ================================================================
    # FIGURE: Comprehensive comparison
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E6 Root Projection: Weight Decomposition Analysis', fontsize=14)

    # Panel 1: (n4, n5) weight lattice
    ax = axes[0, 0]
    for key, group in n4n5_groups.items():
        if key == (0, 0):
            continue
        ax.scatter(key[0], key[1], s=len(group)*50, c='steelblue', alpha=0.7,
                   edgecolors='black', linewidth=1)
        ax.annotate(f'{len(group)}', (key[0], key[1]), ha='center', va='center', fontsize=9)
    ax.set_xlabel('n4 coefficient', fontsize=11)
    ax.set_ylabel('n5 coefficient', fontsize=11)
    ax.set_title('Positive roots in (n4, n5) weight space', fontsize=11)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 2.3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-vertex center of mass in (n4, n5)
    ax = axes[0, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for vertex in range(1, 6):
        mn4, mn5 = vertex_centers[vertex]
        ax.scatter(mn4, mn5, c=colors[vertex-1], s=200, marker='*',
                   edgecolors='black', linewidth=1, zorder=10,
                   label=f'V{vertex} ({gate_names[vertex-1]})')
    ax.set_xlabel('<n4>', fontsize=11)
    ax.set_ylabel('<n5>', fontsize=11)
    ax.set_title('Per-vertex center of mass', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Ad hoc parameters (target)
    ax = axes[0, 2]
    for i, gate in enumerate(gate_names):
        ah = ad_hoc[gate]
        ax.scatter(ah['rz'], ah['rx'], c=colors[i], s=200, marker='s',
                   edgecolors='black', linewidth=1,
                   label=f'{gate} (V{i+1})')
    ax.set_xlabel('rz', fontsize=11)
    ax.set_ylabel('rx', fontsize=11)
    ax.set_title('Ad hoc modulation targets', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    # Panel 4: Root count per vertex vs ad hoc total
    ax = axes[1, 0]
    counts = [vertex_stats[v]['count'] for v in range(1, 6)]
    totals = [ad_hoc[gate_names[v-1]]['rz'] + ad_hoc[gate_names[v-1]]['rx'] for v in range(1, 6)]
    ax.bar(np.arange(5)-0.2, np.array(counts)/max(counts), 0.35, label='Root count (norm)', alpha=0.7, color='steelblue')
    ax.bar(np.arange(5)+0.2, np.array(totals)/max(totals), 0.35, label='Ad hoc rz+rx (norm)', alpha=0.7, color='coral')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f'V{v}\n({gate_names[v-1]})' for v in range(1, 6)])
    ax.set_title('Root count vs ad hoc total', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: n4/(n4+n5) vs rz/(rz+rx) — the asymmetry test
    ax = axes[1, 1]
    n4_frac_vals = [n4_fracs[v] for v in range(1, 6)]
    ah_frac_vals = [ad_hoc_fracs[gate_names[v-1]] for v in range(1, 6)]
    for v in range(1, 6):
        ax.scatter(n4_frac_vals[v-1], ah_frac_vals[v-1], c=colors[v-1], s=200,
                   marker='*', edgecolors='black', linewidth=1,
                   label=f'V{v} ({gate_names[v-1]})')
    ax.plot([0.3, 0.7], [0.3, 0.7], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('n4/(n4+n5) from E6 roots', fontsize=11)
    ax.set_ylabel('rz/(rz+rx) from ad hoc', fontsize=11)
    ax.set_title('Asymmetry comparison', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Weight distribution histograms per vertex
    ax = axes[1, 2]
    for vertex in range(1, 6):
        involved = [rd for rd in root_data if vertex in rd['vertices'] and not rd['is_a4']]
        n4s = [rd['n4'] for rd in involved]
        n5s = [rd['n5'] for rd in involved]
        total_weight = sum(n4s) + sum(n5s)
        n4_frac = sum(n4s) / total_weight if total_weight > 0 else 0
        ax.barh(vertex, n4_frac, color=colors[vertex-1], alpha=0.7, height=0.6,
                label=f'V{vertex} ({gate_names[vertex-1]})')
        ax.barh(vertex, 1-n4_frac, left=n4_frac, color=colors[vertex-1], alpha=0.3, height=0.6)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('n4 fraction (dark) | n5 fraction (light)', fontsize=10)
    ax.set_ylabel('Vertex', fontsize=11)
    ax.set_title('n4/n5 balance per vertex', fontsize=11)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f'V{v} ({gate_names[v-1]})' for v in range(1, 6)])
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{OUT}/e6_weight_decomposition.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved {fname}")
    plt.close()

    # ================================================================
    # FINAL ANALYSIS: What DOES distinguish the vertices?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"STRUCTURAL ANALYSIS: What distinguishes vertices in V2?")
    print(f"{'='*70}")

    print(f"""
  KEY FINDINGS:

  1. V2 has exactly 4 non-zero projection points (plus origin):
     - (sqrt(6/5), 0)        = a4 direction,  |V2| = sqrt(6/5) ~ 1.095
     - (sqrt(3/10), 1/sqrt2)    = 52.24 deg ray,    |V2| = 2/sqrt5  ~ 0.894
     - (-sqrt(3/10), 1/sqrt2)   = 127.76 deg ray,   |V2| = 2/sqrt5  ~ 0.894
     - (0, sqrt2)            = 90 deg ray,        |V2| = sqrt2    ~ 1.414
     (The highest root (1,2,3,2,1,2) lies on the 90 deg ray)

  2. cos(angle(a4,V2, a5,V2)) = -sqrt6/4 ~ -0.612 -> angle ~ 127.76 deg
     This is EXACT (algebraic, from E6 Cartan matrix).

  3. The 4 discrete V2 directions come from (n4, n5) = (1,0), (0,1), (1,1), (1,2):
     - (1,0) -> V2 = a4,V2 -> angle 0 deg
     - (0,1) -> V2 = a5,V2 -> angle 127.76 deg
     - (1,1) -> V2 = a4,V2 + a5,V2 -> angle 52.24 deg
     - (1,2) -> V2 = a4,V2 + 2*a5,V2 -> angle 90 deg

  4. The angle 52.24 deg = arccos(sqrt(6/5)/(2*2/sqrt5)) = arccos(sqrt30/20*sqrt5)
     The angle 127.76 deg = pi - 52.24 deg
     The angle 90 deg for (1,2) is because a4,V2 + 2*a5,V2 is exactly vertical.

  5. Per-vertex multiplicity differs but shares same 4 directions.
     The vertex distinction is in COUNT, not GEOMETRY of V2 projections.

  6. The Cholesky embedding breaks the S5 vertex symmetry:
     - Vertex 1 (S): 12 extra roots (fewest)
     - Vertex 2 (R): 19 extra roots
     - Vertex 3 (T): 24 extra roots
     - Vertex 4 (F): 26 extra roots
     - Vertex 5 (P): 22 extra roots

     This ordering V1 < V2 < V5 < V3 < V4 does NOT match
     the ad hoc total (rz+rx) ordering S < T = F < P < R.
""")

    # ================================================================
    # EXACT ALGEBRAIC VALUES
    # ================================================================
    print(f"\n{'='*70}")
    print(f"EXACT ALGEBRAIC VALUES")
    print(f"{'='*70}")

    # The V2 norms
    print(f"\n  |V2(a4)|^2 = {norm4_sq:.10f}")
    print(f"  Exact: 6/5 = {6/5:.10f}")
    print(f"  Match: {np.isclose(norm4_sq, 6/5)}")

    print(f"\n  |V2(a5)|^2 = {norm5_sq:.10f}")
    print(f"  Exact: 4/5 = {4/5:.10f}")
    print(f"  Match: {np.isclose(norm5_sq, 4/5)}")

    print(f"\n  V2(a4)*V2(a5) = {cross:.10f}")
    print(f"  Exact: -3/5 = {-3/5:.10f}")
    print(f"  Match: {np.isclose(cross, -3/5)}")

    print(f"\n  |V2|^2 for (n4, n5):")
    print(f"    = (6/5)n4^2 + (4/5)n5^2 - (6/5)n4n5")
    print(f"    = (2/5)(3n4^2 + 2n5^2 - 3n4n5)")

    for n4, n5 in sorted(n4n5_groups.keys()):
        if n4 == 0 and n5 == 0:
            continue
        v2sq = (6*n4**2 + 4*n5**2 - 6*n4*n5) / 5
        print(f"    (n4={n4}, n5={n5}): |V2|^2 = {v2sq:.4f} = {int(round(v2sq*5))}/5, |V2| = {np.sqrt(v2sq):.6f}")

    print(f"\n{'='*70}")
    print(f"END OF ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAll output in: {OUT}")
