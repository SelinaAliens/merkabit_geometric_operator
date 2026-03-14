#!/usr/bin/env python3
"""
E6 ROOT PROJECTION ONTO PENTACHORIC SYMMETRY PLANES
=====================================================
The open problem: derive the Rx/Rz modulation parameters from E6 root geometry.

Method:
1. Construct E6 root system (72 roots in R^6)
2. Identify A4 sub-root system (nodes {1,2,3,4} of E6 Dynkin diagram)
3. Decompose R^6 = V4 (A4 subspace) ⊕ V2 (complement)
4. Project all E6 roots onto V2 — these are the "internal" directions
5. Map each root to pentachoric vertices via its A4 projection
6. For each absent vertex, compute active root V2 projections
7. Compare with ad hoc modulation parameters: S(0.4,1.3), R(0.4,1.3),
   T(0.7,0.7), F(1.0,1.0), P(0.6,1.8,1.5)

The A4 sub-root system uses E6 nodes {1,2,3,4} (a straight chain).
E6 Dynkin diagram:
    1 - 2 - 3 - 4 - 5
                |
                6

The extra E6 directions (nodes 5 and 6) map to the 2D internal space.
Node 5 connects to node 4 (one end of A4).
Node 6 connects to node 3 (middle of A4).
This BREAKS the A4 S5 symmetry — different vertices see different V2 content.
"""
import numpy as np
from scipy.linalg import cholesky, null_space
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"

# ============================================================================
# E6 ROOT SYSTEM CONSTRUCTION
# ============================================================================

# E6 Cartan matrix
# Labeling: nodes 0-5 (Python indexing)
# Dynkin:  0 - 1 - 2 - 3 - 4
#                      |
#                      5
CARTAN = np.array([
    [ 2, -1,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0, -1],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2,  0],
    [ 0,  0, -1,  0,  0,  2]
], dtype=float)

# Euclidean embedding: Cholesky L such that L L^T = Cartan
# Then simple_root_i (in R^6) = L[i, :]
L_chol = cholesky(CARTAN, lower=True)
SIMPLE_ROOTS = L_chol.copy()  # (6, 6), each row is a simple root in R^6

def generate_positive_roots():
    """Generate all positive roots of E6 by iterative addition of simple roots."""
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
                # <root, alpha_i> in root basis = sum_j n_j * A_{ji}
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
    """Convert root from simple root basis to Euclidean R^6."""
    return sum(coeffs[i] * SIMPLE_ROOTS[i] for i in range(6))

# ============================================================================
# A4 SUBSPACE DECOMPOSITION
# ============================================================================

def build_subspace_decomposition():
    """
    V4 = span(alpha_0, alpha_1, alpha_2, alpha_3)  [A4 subspace]
    V2 = orthogonal complement [internal space for Rx/Rz]
    """
    V4_raw = SIMPLE_ROOTS[:4]  # (4, 6)
    # Orthonormal basis for V4
    Q4, _ = np.linalg.qr(V4_raw.T)  # (6, 4)
    # V2 = null space of Q4^T (the 2D complement)
    V2 = null_space(Q4.T)  # (6, 2)

    # Verify orthogonality
    assert np.allclose(Q4.T @ V2, 0, atol=1e-10), "V4 and V2 not orthogonal!"

    return Q4, V2

def project_to_V2(root_euclid, V2_basis):
    """Project a root onto the 2D internal space. Returns 2D coordinates."""
    return V2_basis.T @ root_euclid

def project_to_V4(root_euclid, Q4):
    """Project a root onto the 4D A4 subspace. Returns 4D coordinates."""
    return Q4.T @ root_euclid

# ============================================================================
# PENTACHORIC VERTEX MAPPING
# ============================================================================

def map_root_to_vertices(coeffs):
    """
    Map an E6 root to the pentachoric vertices it involves.

    The A4 sub-root system (in the permutation representation) has roots
    e_i - e_j where i,j are pentachoric vertices (1-indexed, 1 to 5).

    Simple roots: alpha_k = e_{k+1} - e_{k+2} (k=0,1,2,3 in 0-indexed E6)

    A general A4 positive root is a contiguous sum alpha_i + ... + alpha_j
    = e_{i+1} - e_{j+2}, involving vertices i+1 and j+2.

    For E6 roots with extra components (n4, n5), we track:
    - Which A4 simple roots are in the support (nonzero coefficients n0-n3)
    - This determines the "footprint" on the pentachoric vertices

    Returns: set of vertex indices (1-5) that the root's A4 projection involves
    """
    n = np.array(coeffs[:6], dtype=int)

    # A4 simple root support: which of alpha_0,...,alpha_3 appear?
    a4_support = set()
    for k in range(4):
        if n[k] > 0:
            # Simple root alpha_k = e_{k+1} - e_{k+2}
            a4_support.add(k+1)  # vertex k+1
            a4_support.add(k+2)  # vertex k+2

    # If the root is purely in the extra directions (n0=n1=n2=n3=0),
    # it doesn't directly involve any pentachoric vertex
    if len(a4_support) == 0:
        # Pure alpha_4 and/or alpha_5 root
        # alpha_4 connects to alpha_3, which involves vertices 4,5
        # alpha_5 connects to alpha_2, which involves vertices 3,4
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
    print("E6 ROOT PROJECTION ONTO PENTACHORIC SYMMETRY PLANES")
    print("=" * 70)

    # Generate roots
    pos_root_coeffs = generate_positive_roots()
    print(f"\nE6 positive roots: {len(pos_root_coeffs)}")
    assert len(pos_root_coeffs) == 36, f"Expected 36, got {len(pos_root_coeffs)}"

    pos_roots_euclid = [to_euclid(np.array(r, dtype=float)) for r in pos_root_coeffs]

    # Verify root lengths
    lengths = [np.linalg.norm(r) for r in pos_roots_euclid]
    print(f"Root lengths: min={min(lengths):.4f}, max={max(lengths):.4f} (should all be sqrt(2)={np.sqrt(2):.4f})")

    # Build subspace decomposition
    Q4, V2 = build_subspace_decomposition()
    print(f"\nV4 basis: {Q4.shape} (A4 subspace)")
    print(f"V2 basis: {V2.shape} (internal space)")

    # Classify roots
    a4_roots = [r for r in pos_root_coeffs if r[4] == 0 and r[5] == 0]
    extra_roots = [r for r in pos_root_coeffs if r[4] != 0 or r[5] != 0]
    print(f"\nA4 positive roots: {len(a4_roots)} (should be 10)")
    print(f"Extra positive roots: {len(extra_roots)} (should be 26)")

    # Project all roots onto V2
    print(f"\n{'='*70}")
    print(f"ROOT PROJECTIONS ONTO V2 (internal space)")
    print(f"{'='*70}")
    print(f"\n{'Coefficients':>30} | {'V2 proj':>20} | {'|V2|':>8} | {'|V4|':>8} | {'Vertices':>15}")
    print(f"{'-'*90}")

    root_data = []
    for i, coeffs in enumerate(pos_root_coeffs):
        r_euclid = pos_roots_euclid[i]
        v2_proj = project_to_V2(r_euclid, V2)
        v4_proj = project_to_V4(r_euclid, Q4)
        vertices = map_root_to_vertices(coeffs)

        root_data.append({
            'coeffs': coeffs,
            'euclid': r_euclid,
            'v2': v2_proj,
            'v4': v4_proj,
            'v2_norm': np.linalg.norm(v2_proj),
            'v4_norm': np.linalg.norm(v4_proj),
            'vertices': vertices,
        })

        coeff_str = str(coeffs)
        v2_str = f"({v2_proj[0]:+.4f}, {v2_proj[1]:+.4f})"
        vert_str = str(sorted(vertices))
        print(f"{coeff_str:>30} | {v2_str:>20} | {root_data[-1]['v2_norm']:>8.4f} | {root_data[-1]['v4_norm']:>8.4f} | {vert_str:>15}")

    # Verify: A4 roots should have zero V2 projection
    a4_norms = [f'{rd["v2_norm"]:.6f}' for rd in root_data if rd['coeffs'][4]==0 and rd['coeffs'][5]==0]
    print(f"\nA4 roots V2 norms: {a4_norms}")

    # Statistics by vertex involvement
    print(f"\n{'='*70}")
    print(f"V2 PROJECTIONS BY ABSENT VERTEX")
    print(f"{'='*70}")

    # For each vertex v (1-5), compute the V2 projections of roots NOT involving v
    for vertex in range(1, 6):
        active = [rd for rd in root_data if vertex not in rd['vertices']]
        inactive = [rd for rd in root_data if vertex in rd['vertices']]

        active_v2 = np.array([rd['v2'] for rd in active])
        inactive_v2 = np.array([rd['v2'] for rd in inactive])

        active_v2_norms = np.array([rd['v2_norm'] for rd in active])
        inactive_v2_norms = np.array([rd['v2_norm'] for rd in inactive])

        print(f"\n  Vertex {vertex} ABSENT:")
        print(f"    Active roots: {len(active)}, Inactive roots: {len(inactive)}")
        if len(active) > 0:
            print(f"    Active V2 mean norm: {np.mean(active_v2_norms):.6f}")
            print(f"    Active V2 mean: ({np.mean(active_v2[:, 0]):.6f}, {np.mean(active_v2[:, 1]):.6f})")
            print(f"    Active V2 RMS:  ({np.sqrt(np.mean(active_v2[:, 0]**2)):.6f}, {np.sqrt(np.mean(active_v2[:, 1]**2)):.6f})")
        if len(inactive) > 0:
            print(f"    Inactive V2 mean norm: {np.mean(inactive_v2_norms):.6f}")

    # Also compute: for roots INVOLVING vertex v, what are their V2 projections?
    print(f"\n{'='*70}")
    print(f"V2 CONTENT BY VERTEX (roots involving each vertex)")
    print(f"{'='*70}")

    vertex_v2_data = {}
    for vertex in range(1, 6):
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            norms = np.array([rd['v2_norm'] for rd in involved])
            mean_v2 = np.mean(v2s, axis=0)
            rms_v2 = np.sqrt(np.mean(v2s**2, axis=0))
            vertex_v2_data[vertex] = {
                'mean': mean_v2,
                'rms': rms_v2,
                'mean_norm': np.mean(norms),
                'count': len(involved),
            }
            print(f"\n  Vertex {vertex}: {len(involved)} extra roots")
            print(f"    Mean V2: ({mean_v2[0]:.6f}, {mean_v2[1]:.6f})")
            print(f"    RMS V2:  ({rms_v2[0]:.6f}, {rms_v2[1]:.6f})")
            print(f"    Mean |V2|: {np.mean(norms):.6f}")
        else:
            print(f"\n  Vertex {vertex}: 0 extra roots")

    # ================================================================
    # COMPARE WITH AD HOC MODULATION PARAMETERS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH AD HOC MODULATION PARAMETERS")
    print(f"{'='*70}")

    # The ad hoc parameters from the ouroboros gate code:
    # Gate absent -> (rz_mult, rx_mult) applied to sym_base angles
    ad_hoc = {
        'S': {'rz': 0.4, 'rx': 1.3},  # step k%5 == 0
        'R': {'rz': 1.3, 'rx': 0.4},  # step k%5 == 1
        'T': {'rz': 0.7, 'rx': 0.7},  # step k%5 == 2
        'F': {'rz': 1.0, 'rx': 1.0},  # step k%5 == 3 (identity multipliers)
        'P': {'rz': 1.5, 'rx': 1.8},  # step k%5 == 4 (also p *= 0.6)
    }

    gate_names = ['S', 'R', 'T', 'F', 'P']
    print(f"\n  {'Gate':>6} | {'Vertex':>7} | {'Ad hoc (rz, rx)':>20} | {'V2 RMS (c1, c2)':>20} | {'V2 mean norm':>12}")
    print(f"  {'-'*75}")

    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            rms = np.sqrt(np.mean(v2s**2, axis=0))
            mnorm = np.mean([rd['v2_norm'] for rd in involved])
        else:
            rms = np.array([0, 0])
            mnorm = 0

        ah = ad_hoc[gate]
        print(f"  {gate:>6} | {vertex:>7} | ({ah['rz']:.1f}, {ah['rx']:.1f}){' '*10} | ({rms[0]:.4f}, {rms[1]:.4f}){' '*5} | {mnorm:>12.6f}")

    # Try ratio normalization
    print(f"\n  RATIO ANALYSIS (normalized to gate F = vertex 4):")
    ref_vertex = 4  # Gate F has multiplier (1.0, 1.0)
    ref_involved = [rd for rd in root_data if ref_vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
    if len(ref_involved) > 0:
        ref_v2s = np.array([rd['v2'] for rd in ref_involved])
        ref_rms = np.sqrt(np.mean(ref_v2s**2, axis=0))
        ref_norm = np.mean([rd['v2_norm'] for rd in ref_involved])
    else:
        ref_rms = np.array([1, 1])
        ref_norm = 1

    print(f"  Reference (vertex 4 = gate F): RMS = ({ref_rms[0]:.4f}, {ref_rms[1]:.4f}), norm = {ref_norm:.4f}")
    print(f"\n  {'Gate':>6} | {'Ad hoc ratio (rz/1, rx/1)':>25} | {'V2 ratio (c1/ref1, c2/ref2)':>30} | {'Match?':>8}")
    print(f"  {'-'*80}")

    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            rms = np.sqrt(np.mean(v2s**2, axis=0))
        else:
            rms = np.array([0, 0])

        ah = ad_hoc[gate]
        ratio_ah = (ah['rz'] / 1.0, ah['rx'] / 1.0)  # normalized to F
        ratio_v2 = (rms[0] / ref_rms[0] if ref_rms[0] > 0 else 0,
                     rms[1] / ref_rms[1] if ref_rms[1] > 0 else 0)

        match = "~" if (abs(ratio_ah[0] - ratio_v2[0]) < 0.2 and abs(ratio_ah[1] - ratio_v2[1]) < 0.2) else "NO"
        print(f"  {gate:>6} | ({ratio_ah[0]:.2f}, {ratio_ah[1]:.2f}){' '*13} | ({ratio_v2[0]:.4f}, {ratio_v2[1]:.4f}){' '*13} | {match:>8}")

    # ================================================================
    # ALTERNATIVE: Use ABSENT vertex's V2 content (complement approach)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ALTERNATIVE: V2 CONTENT OF ROOTS NOT INVOLVING EACH VERTEX")
    print(f"{'='*70}")
    print(f"  (When vertex is absent, these roots remain active)")

    print(f"\n  {'Gate':>6} | {'Vertex':>7} | {'Active count':>12} | {'Active V2 RMS':>25} | {'Active mean |V2|':>15}")
    print(f"  {'-'*85}")

    active_data = {}
    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        active = [rd for rd in root_data if vertex not in rd['vertices']]
        if len(active) > 0:
            v2s = np.array([rd['v2'] for rd in active])
            rms = np.sqrt(np.mean(v2s**2, axis=0))
            mnorm = np.mean([rd['v2_norm'] for rd in active])
        else:
            rms = np.array([0, 0])
            mnorm = 0
        active_data[vertex] = {'rms': rms, 'norm': mnorm, 'count': len(active)}
        print(f"  {gate:>6} | {vertex:>7} | {len(active):>12} | ({rms[0]:.6f}, {rms[1]:.6f}){' '*3} | {mnorm:>15.6f}")

    # ================================================================
    # FIGURE: V2 projections of all E6 roots, colored by vertex
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('E6 Root Projections onto V2 (Internal Space)', fontsize=13)

    # Panel 1: All roots colored by V2 norm
    ax = axes[0]
    v2_projs = np.array([rd['v2'] for rd in root_data])
    v2_norms = np.array([rd['v2_norm'] for rd in root_data])
    # Also include negative roots
    neg_v2_projs = -v2_projs
    all_v2 = np.vstack([v2_projs, neg_v2_projs])
    all_norms = np.concatenate([v2_norms, v2_norms])

    sc = ax.scatter(all_v2[:, 0], all_v2[:, 1], c=all_norms, cmap='viridis',
                    s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='|V2|')
    ax.set_xlabel('V2 component 1', fontsize=10)
    ax.set_ylabel('V2 component 2', fontsize=10)
    ax.set_title('All 72 E6 roots projected onto V2', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: Roots colored by vertex involvement
    ax = axes[1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for vertex in range(1, 6):
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            ax.scatter(v2s[:, 0], v2s[:, 1], c=colors[vertex-1], s=40,
                       alpha=0.6, label=f'V{vertex} ({gate_names[vertex-1]})')
            # Also negative
            ax.scatter(-v2s[:, 0], -v2s[:, 1], c=colors[vertex-1], s=40, alpha=0.3)
    ax.set_xlabel('V2 component 1', fontsize=10)
    ax.set_ylabel('V2 component 2', fontsize=10)
    ax.set_title('Extra roots by pentachoric vertex', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: V2 RMS by vertex (bar chart comparison with ad hoc)
    ax = axes[2]
    x = np.arange(5)
    width = 0.35

    # V2 RMS component 1
    v2_c1 = []
    v2_c2 = []
    ah_rz = []
    ah_rx = []
    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            rms = np.sqrt(np.mean(v2s**2, axis=0))
        else:
            rms = np.array([0, 0])
        v2_c1.append(rms[0])
        v2_c2.append(rms[1])
        ah_rz.append(ad_hoc[gate]['rz'])
        ah_rx.append(ad_hoc[gate]['rx'])

    # Normalize both to sum to same total for visual comparison
    v2_total = np.array(v2_c1) + np.array(v2_c2)
    ah_total = np.array(ah_rz) + np.array(ah_rx)
    scale = np.mean(ah_total) / np.mean(v2_total) if np.mean(v2_total) > 0 else 1

    bars1 = ax.bar(x - width/2, np.array(v2_c1) * scale, width, label='V2_c1 (scaled)', alpha=0.7, color='steelblue')
    bars2 = ax.bar(x - width/2, np.array(v2_c2) * scale, width, bottom=np.array(v2_c1)*scale, label='V2_c2 (scaled)', alpha=0.7, color='lightblue')
    bars3 = ax.bar(x + width/2, ah_rz, width, label='Ad hoc Rz', alpha=0.7, color='coral')
    bars4 = ax.bar(x + width/2, ah_rx, width, bottom=ah_rz, label='Ad hoc Rx', alpha=0.7, color='lightsalmon')

    ax.set_xticks(x)
    ax.set_xticklabels([f'V{v}\n({gate_names[v-1]})' for v in range(1, 6)])
    ax.set_ylabel('Magnitude', fontsize=10)
    ax.set_title('V2 RMS vs Ad hoc modulation', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{OUT}/e6_root_projection.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved {fname}")
    plt.close()

    # ================================================================
    # DEEPER ANALYSIS: Angle structure in V2
    # ================================================================
    print(f"\n{'='*70}")
    print(f"V2 ANGLE STRUCTURE (polar coordinates)")
    print(f"{'='*70}")

    for vertex in range(1, 6):
        gate = gate_names[vertex - 1]
        involved = [rd for rd in root_data if vertex in rd['vertices'] and rd['v2_norm'] > 1e-10]
        if len(involved) > 0:
            v2s = np.array([rd['v2'] for rd in involved])
            angles = np.arctan2(v2s[:, 1], v2s[:, 0])
            norms = np.array([rd['v2_norm'] for rd in involved])
            print(f"\n  Vertex {vertex} ({gate}): {len(involved)} roots")
            for j, rd in enumerate(involved):
                print(f"    {str(rd['coeffs']):>30}  |V2|={norms[j]:.4f}  angle={angles[j]*180/np.pi:>8.2f} deg")

    # ================================================================
    # CONNECTION TO RX/RZ: which V2 direction is Rx, which is Rz?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"V2 DIRECTION IDENTIFICATION")
    print(f"{'='*70}")
    print(f"""
  The two V2 directions come from alpha_4 and alpha_5 (E6 nodes beyond A4).
  - alpha_4 connects to A4 node 3 (bottom of chain) -> one end of pentachoron
  - alpha_5 connects to A4 node 2 (branching node) -> middle of pentachoron

  alpha_4's V2 projection:""")
    alpha4_euclid = SIMPLE_ROOTS[4]
    alpha4_v2 = project_to_V2(alpha4_euclid, V2)
    alpha4_v4 = project_to_V4(alpha4_euclid, Q4)
    print(f"    V2 = ({alpha4_v2[0]:.6f}, {alpha4_v2[1]:.6f}), |V2| = {np.linalg.norm(alpha4_v2):.6f}")
    print(f"    V4 = {alpha4_v4}, |V4| = {np.linalg.norm(alpha4_v4):.6f}")

    print(f"\n  alpha_5's V2 projection:")
    alpha5_euclid = SIMPLE_ROOTS[5]
    alpha5_v2 = project_to_V2(alpha5_euclid, V2)
    alpha5_v4 = project_to_V4(alpha5_euclid, Q4)
    print(f"    V2 = ({alpha5_v2[0]:.6f}, {alpha5_v2[1]:.6f}), |V2| = {np.linalg.norm(alpha5_v2):.6f}")
    print(f"    V4 = {alpha5_v4}, |V4| = {np.linalg.norm(alpha5_v4):.6f}")

    print(f"\n  Angle between alpha_4 and alpha_5 V2 projections: "
          f"{np.arccos(np.clip(np.dot(alpha4_v2, alpha5_v2) / (np.linalg.norm(alpha4_v2) * np.linalg.norm(alpha5_v2) + 1e-30), -1, 1)) * 180 / np.pi:.2f} deg")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"""
E6 root system: 36 positive roots, 72 total
A4 sub-system: 10 positive roots (all in V4, V2 projection = 0)
Extra roots: 26 positive roots (with nonzero V2 projection)

The V2 internal space encodes the Rx/Rz modulation directions.
The distribution of V2 projections across pentachoric vertices
determines how each absent gate modifies the symmetric rotation angles.

Comparison with ad hoc parameters above shows whether the E6 root
geometry reproduces the hand-tuned modulation factors.
""")

    print(f"\nAll output in: {OUT}")
