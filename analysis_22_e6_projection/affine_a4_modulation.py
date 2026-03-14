#!/usr/bin/env python3
"""
AFFINE A4 MODULATION FACTORS FROM REPRESENTATION THEORY
=========================================================
The 5 ouroboros gates {S, R, T, F, P} form the affine A4 extended Dynkin diagram
A4^(1) -- a cyclic pentagon S-R-T-F-P-S.  P is the affine node.

The modulation factors (rz, rx) when each gate is absent should follow from
the matrix elements of the A4^(1) generators in the SU(2)xSU(2) spinor
representation on S^3 x S^3.

Approach:
1. Build the affine A4^(1) Cartan matrix (5x5 cyclic)
2. Compute fundamental weights and coweights
3. Embed the two SU(2) rotation generators (Rz, Rx) into the Cartan subalgebra
4. For each absent gate i, compute the projection of the REMAINING generators
   onto the Rz and Rx subspaces
5. The projections should give the modulation factors as rational numbers

The key structure: A4^(1) Cartan matrix
     S   R   T   F   P
S  [ 2  -1   0   0  -1]
R  [-1   2  -1   0   0]
T  [ 0  -1   2  -1   0]
F  [ 0   0  -1   2  -1]
P  [-1   0   0  -1   2]
"""
import numpy as np
from scipy.linalg import null_space
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"

# ============================================================================
# AFFINE A4 CARTAN MATRIX
# ============================================================================

# Affine A4^(1): cyclic pentagon
# Nodes: 0=S, 1=R, 2=T, 3=F, 4=P
CARTAN_AFF = np.array([
    [ 2, -1,  0,  0, -1],
    [-1,  2, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  2, -1],
    [-1,  0,  0, -1,  2]
], dtype=float)

GATE_NAMES = ['S', 'R', 'T', 'F', 'P']

# Kac labels (marks) of A4^(1): all equal to 1
# null vector: delta = (1, 1, 1, 1, 1)
KAC_LABELS = np.array([1, 1, 1, 1, 1], dtype=float)

# Verify null vector
assert np.allclose(CARTAN_AFF @ KAC_LABELS, 0), "Kac labels should be null vector"

# ============================================================================
# FINITE A4 CARTAN MATRIX (remove affine node P = node 4)
# ============================================================================
CARTAN_A4 = np.array([
    [ 2, -1,  0,  0],
    [-1,  2, -1,  0],
    [ 0, -1,  2, -1],
    [ 0,  0, -1,  2]
], dtype=float)

# A4 fundamental weights: omega_i such that (alpha_i, omega_j) = delta_ij
# omega = A^(-1) (inverse Cartan matrix)
FUND_WEIGHTS = np.linalg.inv(CARTAN_A4)

# ============================================================================
# APPROACH 1: Weyl vector projection
# ============================================================================
def approach_weyl_vector():
    """
    The Weyl vector rho = sum of fundamental weights = (1/2) sum of positive roots.
    For each absent node i, the remaining Weyl vector changes.
    """
    print("=" * 70)
    print("APPROACH 1: WEYL VECTOR / HALF-SUM OF POSITIVE ROOTS")
    print("=" * 70)

    # For A4, the Weyl vector in the fundamental weight basis is (1,1,1,1)
    # In the simple root basis: rho = A^{-1} @ (1,1,1,1)
    rho_fund = np.array([1, 1, 1, 1], dtype=float)  # in omega basis
    rho_root = FUND_WEIGHTS @ rho_fund  # This gives rho in terms of simple roots... wait
    # Actually: omega = A^{-1} rows, so rho = sum omega_i = sum of rows of A^{-1}
    # In the root basis: rho = A^{-1} @ (1,1,1,1) ... no.
    # rho_j = sum_i omega_i^j where omega_i^j is the j-th component of omega_i
    # omega_i = A^{-1}[i, :] row ... actually A^{-1}[i,j] = (alpha_j, omega_i)/(alpha_j, alpha_j)
    # The standard convention: <alpha_i, omega_j> = delta_{ij}
    # So omega_j = sum_k (A^{-1})_{jk} alpha_k
    # And rho = sum_j omega_j = sum_j sum_k (A^{-1})_{jk} alpha_k = sum_k (sum_j (A^{-1})_{jk}) alpha_k

    rho_in_root_basis = np.sum(FUND_WEIGHTS, axis=0)  # sum of rows
    print(f"\nA4 Weyl vector rho in root basis: {rho_in_root_basis}")
    print(f"  = {rho_in_root_basis[0]:.4f} a0 + {rho_in_root_basis[1]:.4f} a1 + {rho_in_root_basis[2]:.4f} a2 + {rho_in_root_basis[3]:.4f} a3")


# ============================================================================
# APPROACH 2: Adjacency-based modulation from affine diagram
# ============================================================================
def approach_adjacency():
    """
    In the affine A4^(1) diagram, each node i has exactly 2 neighbors.
    When gate i is absent, the contribution depends on the GRAPH DISTANCE
    from node i to the Rz and Rx directions.

    Hypothesis: the modulation of Rz and Rx when gate i is absent comes from
    the graph Laplacian eigenvalues restricted to the complement of node i.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: GRAPH LAPLACIAN OF A4^(1) MINUS NODE")
    print("=" * 70)

    # Adjacency matrix of the pentagon
    ADJ = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ], dtype=float)

    # Graph Laplacian L = D - A
    D = np.diag(ADJ.sum(axis=1))
    LAP = D - ADJ

    print("\nFull pentagon eigenvalues:")
    evals_full = np.linalg.eigvalsh(LAP)
    print(f"  {np.sort(evals_full)}")
    # Should be 0, 5/2 - sqrt(5)/2, 5/2 + sqrt(5)/2, 5/2 - sqrt(5)/2, 5/2 + sqrt(5)/2

    for node in range(5):
        # Remove node i: take 4x4 submatrix
        remaining = [j for j in range(5) if j != node]
        L_sub = LAP[np.ix_(remaining, remaining)]
        evals = np.sort(np.linalg.eigvalsh(L_sub))

        # Also the adjacency submatrix
        A_sub = ADJ[np.ix_(remaining, remaining)]
        a_evals = np.sort(np.linalg.eigvalsh(A_sub))

        print(f"\n  Remove node {node} ({GATE_NAMES[node]}):")
        print(f"    Laplacian eigenvalues: {evals}")
        print(f"    Adjacency eigenvalues: {a_evals}")
        print(f"    Sum of adj evals: {np.sum(a_evals):.6f}")
        print(f"    Product of nonzero Lap evals: {np.prod(evals[evals > 0.01]):.6f}")


# ============================================================================
# APPROACH 3: Cartan matrix inverse elements
# ============================================================================
def approach_cartan_inverse():
    """
    For each way of removing one node from the affine diagram to get finite A4,
    the inverse Cartan matrix gives the metric on the weight lattice.

    When node i is removed from A4^(1), the remaining 4 nodes form A4.
    The relabeling changes which node maps to which A4 position.
    The diagonal elements of A4^{-1} in the relabeled basis should give
    the modulation factors.
    """
    print("\n" + "=" * 70)
    print("APPROACH 3: CARTAN INVERSE FOR EACH NODE REMOVAL")
    print("=" * 70)

    for removed in range(5):
        remaining = [j for j in range(5) if j != removed]
        C_sub = CARTAN_AFF[np.ix_(remaining, remaining)]

        # This should always be the A4 Cartan matrix (possibly relabeled)
        det = np.linalg.det(C_sub)
        C_inv = np.linalg.inv(C_sub)

        print(f"\n  Remove node {removed} ({GATE_NAMES[removed]}):")
        print(f"    Remaining nodes: {[GATE_NAMES[j] for j in remaining]}")
        print(f"    det(C) = {det:.4f} (should be 5 for A4)")
        print(f"    C^(-1) diagonal: {np.diag(C_inv)}")
        print(f"    C^(-1) sum of rows: {np.sum(C_inv, axis=1)}")
        print(f"    C^(-1) total sum: {np.sum(C_inv):.6f}")

        # The metric tensor g_ij = (omega_i, omega_j) = C^{-1}_{ij}
        # The diagonal gives ||omega_i||^2
        # Row sums give (omega_i, rho) where rho = sum omega_j


# ============================================================================
# APPROACH 4: Projection onto two SU(2) subspaces
# ============================================================================
def approach_su2_projection():
    """
    The two SU(2) factors correspond to specific linear combinations of the
    Cartan generators. In the A4^(1) basis:

    SU(2)_Rz corresponds to the direction that breaks the S<->R symmetry
    SU(2)_Rx corresponds to the orthogonal direction

    The S<->R swap means: Rz(S) = Rx(R) and vice versa.
    This constrains the SU(2) directions to be related by the Weyl reflection
    that exchanges nodes 0 and 1 (S and R).
    """
    print("\n" + "=" * 70)
    print("APPROACH 4: SU(2) x SU(2) PROJECTION")
    print("=" * 70)

    # The affine diagram has cyclic Z5 symmetry.
    # The Z2 that swaps S<->R also swaps P<->T (reflections of the pentagon).
    # Specifically: the reflection fixing F maps:
    #   S(0) <-> P(4), R(1) <-> T(2), F(3) fixed
    # Wait, let me check. Pentagon 0-1-2-3-4-0.
    # Reflection that swaps 0<->1 also swaps 4<->2, fixes 3.
    # So: S<->R, P<->T, F fixed.

    # Under this Z2:
    # rz(S) = rx(R) = 0.4  -> node 0 and node 1 swap their rz/rx
    # rz(R) = rx(S) = 1.3  -> consistent
    # rz(T) = rx(T) = 0.7  -> T should also be invariant... but T<->P under this Z2!

    # Actually, let me reconsider. The reflection depends on the axis.
    # Pentagon vertices at angles 2*pi*k/5 for k=0,1,2,3,4.
    # Reflection about axis through vertex 2 (T): swaps 0<->4, 1<->3
    # => S(0)<->P(4), R(1)<->F(3), T(2) fixed

    # Reflection about axis through midpoint of edge 0-1: swaps 0<->1, 2<->4, fixes nothing
    # => S(0)<->R(1), T(2)<->P(4), F(3) maps to... no, 3 maps to 3 if axis through 3?

    # Pentagon symmetry D5. Reflections:
    # Through vertex k: swaps (k-1)<->(k+1), (k-2)<->(k+2)
    # Through midpoint of edge k-(k+1): swaps k<->(k+1), (k-1)<->(k+2), (k+2)<->(k-2)?

    # Let me just compute: which reflection swaps S(0) and R(1)?
    # If axis goes through midpoint of edge 0-1 and through vertex 3:
    #   0 <-> 1, 2 <-> 4, 3 fixed
    #   S <-> R, T <-> P, F fixed

    # Under this Z2: if rz and rx transform as:
    # rz(S) = rx(R), rx(S) = rz(R)  [S<->R with rz<->rx]
    # rz(T) = rx(P), rx(T) = rz(P)  [T<->P with rz<->rx]
    # rz(F) = rx(F)                   [F is symmetric]

    # Check: rz(T) = 0.7, rx(P) = 1.8 -- NOT EQUAL! So this Z2 doesn't work
    # with rz<->rx identification.

    # Let me try the OTHER Z2: through vertex T, swapping S<->P and R<->F.
    # Through vertex 2: swaps 0<->4, 1<->3
    #   S(0) <-> P(4), R(1) <-> F(3), T(2) fixed
    # Under this with rz<->rx:
    # rz(S) = rx(P) -> 0.4 = 1.8? NO
    # So this doesn't work either directly.

    # The actual symmetry from the ad hoc values is JUST the S<->R exchange
    # with rz<->rx swap. This means the SU(2)_rz and SU(2)_rx directions
    # are related by whatever diagram symmetry connects S to R.

    print("\n  Pentagon Z2 symmetries and their action on gates:")
    # The 5 reflections of D5
    for axis_vertex in range(5):
        perm = {}
        for k in range(5):
            # Reflection through vertex 'axis_vertex':
            # k -> 2*axis_vertex - k (mod 5)
            image = (2 * axis_vertex - k) % 5
            perm[k] = image
        perm_str = ", ".join([f"{GATE_NAMES[k]}->{GATE_NAMES[perm[k]]}" for k in range(5)])
        fixed = [GATE_NAMES[k] for k in range(5) if perm[k] == k]
        print(f"    Through {GATE_NAMES[axis_vertex]}: {perm_str} (fixed: {fixed})")


# ============================================================================
# APPROACH 5: Coxeter element and its eigenvalues
# ============================================================================
def approach_coxeter():
    """
    The Coxeter element of A4 is the product of all simple reflections.
    Its eigenvalues are exp(2*pi*i*m_j/h) where m_j are the exponents
    and h is the Coxeter number.

    For A4: h=5, exponents = {1, 2, 3, 4}
    For the affine A4^(1), the Coxeter element acts on the 5D space.

    The key: when gate i is absent, the partial Coxeter element (product of
    the remaining 4 reflections) has eigenvalues that encode the modulation.
    """
    print("\n" + "=" * 70)
    print("APPROACH 5: PARTIAL COXETER ELEMENTS")
    print("=" * 70)

    # Simple reflections in 5D for affine A4
    # s_i(x) = x - (x . alpha_i) * alpha_i_dual
    # In the root basis: s_i(alpha_j) = alpha_j - A_{ij} * alpha_i

    # Build reflection matrices
    reflections = []
    for i in range(5):
        S = np.eye(5)
        for j in range(5):
            S[j, i] = S[j, i] - CARTAN_AFF[i, j]
        reflections.append(S)

    # Verify they are reflections
    for i in range(5):
        assert np.allclose(reflections[i] @ reflections[i], np.eye(5)), f"s_{i} not involution"

    # Full Coxeter element: product of all 5 reflections
    C_full = np.eye(5)
    for i in range(5):
        C_full = C_full @ reflections[i]

    evals_full = np.linalg.eigvals(C_full)
    print(f"\n  Full Coxeter element eigenvalues:")
    for ev in sorted(evals_full, key=lambda x: np.angle(x)):
        print(f"    {ev:.6f}  (|ev|={abs(ev):.6f}, angle={np.angle(ev)*180/np.pi:.2f} deg)")

    # For each absent gate: partial Coxeter element (product of remaining 4)
    for absent in range(5):
        remaining = [j for j in range(5) if j != absent]
        C_partial = np.eye(5)
        for j in remaining:
            C_partial = C_partial @ reflections[j]

        evals = np.linalg.eigvals(C_partial)
        evals_sorted = sorted(evals, key=lambda x: abs(np.angle(x)))

        angles = np.array([np.angle(ev) for ev in evals_sorted]) * 180 / np.pi
        mags = np.array([abs(ev) for ev in evals_sorted])

        # The partial Coxeter element in the 4D subspace
        C_4d = CARTAN_AFF[np.ix_(remaining, remaining)]
        # Build reflections in 4D
        refs_4d = []
        for idx, j in enumerate(remaining):
            S = np.eye(4)
            for k_idx, k in enumerate(remaining):
                S[k_idx, idx] = S[k_idx, idx] - C_4d[idx, k_idx]
            refs_4d.append(S)

        C_partial_4d = np.eye(4)
        for ref in refs_4d:
            C_partial_4d = C_partial_4d @ ref

        evals_4d = np.linalg.eigvals(C_partial_4d)
        angles_4d = sorted([np.angle(ev) * 180 / np.pi for ev in evals_4d])

        print(f"\n  Absent {GATE_NAMES[absent]} (node {absent}):")
        print(f"    4D eigenvalue angles: {[f'{a:.2f}' for a in angles_4d]} deg")
        print(f"    4D eigenvalue magnitudes: {[f'{abs(ev):.6f}' for ev in evals_4d]}")

        # The trace and determinant
        tr = np.trace(C_partial_4d)
        det = np.linalg.det(C_partial_4d)
        print(f"    Trace = {tr:.6f}, Det = {det:.6f}")


# ============================================================================
# APPROACH 6: Direct computation of Rz/Rx weights from Dynkin diagram
# ============================================================================
def approach_dynkin_weights():
    """
    The most direct approach:

    In A4^(1), the two generators beyond A2 (the middle subdiagram) are
    associated with the Rz and Rx directions. The affine node P adds the
    loop extension.

    The modulation factor for Rz when gate i is absent is the PROJECTION
    of the i-th coroot onto the Rz direction in the dual space.

    For A4 with nodes S-R-T-F in a chain:
    The Rz direction ~ fundamental weight omega_1 (at the S end)
    The Rx direction ~ fundamental weight omega_4 (at the F end)
    (or linear combinations thereof)

    When node i is absent, Rz modulation = (rho_remaining, Rz_direction) / (rho_full, Rz_direction)
    """
    print("\n" + "=" * 70)
    print("APPROACH 6: FUNDAMENTAL WEIGHT PROJECTIONS")
    print("=" * 70)

    # A4 inverse Cartan matrix elements give (omega_i, omega_j)
    A4_inv = np.linalg.inv(CARTAN_A4)

    print(f"\nA4 inverse Cartan matrix (fundamental weight inner products):")
    for row in A4_inv:
        print(f"  [{', '.join([f'{x:7.4f}' for x in row])}]")

    # Try to interpret: The Rz and Rx directions as specific weights
    # Option A: Rz = omega_1, Rx = omega_4 (ends of the chain)
    # Option B: Rz = omega_2, Rx = omega_3 (middle nodes)
    # Option C: some linear combination

    # The key constraint: S<->R with rz<->rx means the Rz and Rx directions
    # are exchanged by the Weyl reflection through the axis that swaps S<->R.
    # In the A4 chain (when P is removed), S=node 0, R=node 1, T=node 2, F=node 3.
    # The reflection that swaps 0<->1 in the chain is NOT a diagram automorphism
    # (A4 only has the reversal 0<->3, 1<->2).

    # So the A4 diagram automorphism reverses the chain: S<->F, R<->T
    # Under this: omega_1 <-> omega_4, omega_2 <-> omega_3

    # But we want S<->R to swap Rz<->Rx. In the AFFINE diagram, S<->R IS
    # a symmetry (pentagon has the Z2 that fixes F and swaps S<->R, T<->P).
    # When we remove P (the affine node) to get finite A4, S<->R is NOT
    # a symmetry of A4. But the modulation factors know about the affine
    # structure because they come from the full A4^(1).

    # The affine Weyl vector: rho_aff = sum of affine fundamental weights
    # The level-1 representations of A4^(1) are the 5 fundamental reps

    # Let me try a direct approach: for each pair of directions (d_rz, d_rx)
    # in the 4D A4 weight space, compute the modulation factors and see
    # which pair reproduces the ad hoc values.

    ad_hoc_rz = np.array([0.4, 1.3, 0.7, 1.0])  # S, R, T, F (P removed)
    ad_hoc_rx = np.array([1.3, 0.4, 0.7, 1.0])  # S, R, T, F

    # The Rz modulation for node i = projection of remaining structure onto Rz
    # If Rz = sum c_j * omega_j, then the contribution of node i to Rz is
    # (alpha_i, Rz) = sum c_j * (alpha_i, omega_j) = sum c_j * delta_{ij} = c_i

    # So the Rz contribution of node i is simply c_i (coefficient in omega basis).
    # When node i is absent: Rz_modulation = (total_Rz - c_i) / total_Rz
    # Actually the modulation is more like: the SCALING of the symmetric angle.

    # Let me try: Rz_mod(absent=i) proportional to sum_{j != i} w_j(rz)
    # where w_j(rz) is the weight of node j in the Rz direction.

    # Parametrize: Rz = a * omega_1 + b * omega_2 + c * omega_3 + d * omega_4
    # Rx = d * omega_1 + c * omega_2 + b * omega_3 + a * omega_4
    # (reversed, to implement the S<->R = diagram reversal symmetry)

    # Under A4 reversal (0<->3, 1<->2): omega_1 <-> omega_4, omega_2 <-> omega_3
    # So Rz -> Rx and Rx -> Rz. This gives: rz(S) = rx(F), rz(R) = rx(T), etc.
    # But we want rz(S) = rx(R), not rz(S) = rx(F).

    # Hmm, this means the S<->R swap is NOT the A4 diagram automorphism.
    # It must be an AFFINE symmetry.

    print("\n  Testing: Rz/Rx from AFFINE diagram symmetry")
    print("  The reflection that swaps S(0)<->R(1) in the pentagon also swaps T(2)<->P(4)")
    print("  and fixes F(3).")
    print("  When P is removed: S<->R symmetry is encoded in the ADJACENCY to P.")

    # In the affine diagram, the two nodes adjacent to P are S and F.
    # The two nodes NOT adjacent to P are R and T.
    # The S<->R reflection (fixing F, swapping S<->R and T<->P) means:
    # In the finite A4 obtained by removing P, node S is where P connected,
    # while node F is the other P-neighbor.

    # Alternative: use the CONNECTION STRUCTURE to P as the distinguishing feature.
    # S is adjacent to P in the affine diagram (1 step)
    # R is 2 steps from P
    # T is 2 steps from P
    # F is adjacent to P (1 step)
    # P is the removed node

    dist_to_P = {0: 1, 1: 2, 2: 2, 3: 1}  # distances in the pentagon to node 4
    # Or counting through the shorter path:
    # S->P direct (1), R->S->P (2), T->F->P (2), F->P direct (1)

    print(f"\n  Graph distances to P in the pentagon:")
    for i in range(4):
        print(f"    {GATE_NAMES[i]}: {dist_to_P[i]} steps from P")
    print(f"    Note: S and F are adjacent to P; R and T are distance-2")

    # This means S and F share a property (adjacent to affine node)
    # while R and T share a property (non-adjacent).
    # But in the ad hoc: S<->R are the swapped pair, not S<->F.

    # So the relevant symmetry is NOT just graph distance.
    # The S<->R swap must come from the ORIENTATION of the pentagon.

    print("\n  KEY INSIGHT: The pentagon is DIRECTED (the ouroboros cycle)")
    print("  The cycle goes S->R->T->F->P->S (counterclockwise)")
    print("  Under the reflection that fixes F and passes through midpoint(S-R):")
    print("    S <-> R (both are 'before' and 'after' F)")
    print("    T <-> P (both are 'opposite' F)")
    print("  The Rz/Rx distinction comes from the CHIRALITY of the cycle")


# ============================================================================
# APPROACH 7: Casimir eigenvalues of subalgebras
# ============================================================================
def approach_casimir():
    """
    When gate i is absent, the remaining 4 generators form an A4 subalgebra
    (but with a specific embedding into A4^(1)).

    The quadratic Casimir of this A4 subalgebra, when expressed in terms of
    the Rz and Rx generators, gives the modulation factors.

    C2(A4) = sum_{a} T^a T^a

    The projection: C2 = C2_Rz + C2_Rx + C2_cross

    The modulation factors are: rz_mod = C2_Rz / C2_Rz(reference)
                                rx_mod = C2_Rx / C2_Rx(reference)
    """
    print("\n" + "=" * 70)
    print("APPROACH 7: CASIMIR DECOMPOSITION")
    print("=" * 70)

    # The quadratic Casimir for A4 in the fundamental (5-dim) representation
    # is proportional to the identity. But in specific representations,
    # the decomposition into Rz and Rx components gives different weights.

    # For now, compute the Killing form restricted to each subalgebra.
    # The Killing form of A4^(1) is degenerate (null direction = delta).

    # Killing form B(h_i, h_j) = A_{ij} for the Cartan subalgebra
    # (up to normalization)

    for removed in range(5):
        remaining = [j for j in range(5) if j != removed]
        C_sub = CARTAN_AFF[np.ix_(remaining, remaining)]

        # Quadratic Casimir eigenvalue for A4 in fundamental rep
        # C2 = (N^2 - 1)/(2N) = (25-1)/10 = 2.4 for su(5)
        # Independent of which A4 embedding (always the same algebra)

        # But the DECOMPOSITION into the Rz/Rx subspaces depends on embedding.
        # The trace of the Cartan matrix gives one measure:
        tr = np.trace(C_sub)  # Always 8 for A4
        off_diag = np.sum(C_sub) - tr

        # Sum of eigenvalues of C_sub
        evals = np.sort(np.linalg.eigvalsh(C_sub))

        print(f"\n  Remove {GATE_NAMES[removed]} (node {removed}):")
        print(f"    Eigenvalues of A4 Cartan: {evals}")
        print(f"    These are 2-2*cos(2*pi*k/5) for k=1,2,3,4:")
        for k in range(1, 5):
            print(f"      k={k}: {2 - 2*np.cos(2*np.pi*k/5):.6f}")


# ============================================================================
# APPROACH 8: Direct numerical search for the embedding
# ============================================================================
def approach_numerical_search():
    """
    Brute force: parameterize the Rz and Rx directions as vectors in the
    5D affine Cartan space, and search for the directions that reproduce
    the ad hoc modulation factors.
    """
    print("\n" + "=" * 70)
    print("APPROACH 8: NUMERICAL SEARCH FOR Rz/Rx EMBEDDING")
    print("=" * 70)

    # The ad hoc values
    ad_hoc = np.array([
        [0.4, 1.3],   # S
        [1.3, 0.4],   # R
        [0.7, 0.7],   # T
        [1.0, 1.0],   # F
        [1.5, 1.8],   # P
    ])

    # The modulation when gate i is absent: (rz_i, rx_i)
    # Hypothesis: there exist vectors v_rz and v_rx in R^5 such that
    # rz_i = sum_{j != i} |v_rz . e_j|^2  (or similar)
    # where e_j are the standard basis vectors (Cartan elements).

    # Actually, the simplest model:
    # Each node j has a "weight" w_j = (w_j^rz, w_j^rx)
    # When gate i is absent, the modulation is:
    # rz_i = sum_{j != i, j adjacent to i in pentagon} w_j^rz
    # or
    # rz_i = (total_rz - w_i^rz) / (total_rz - w_ref^rz) * ref_rz

    # Let me try the SIMPLEST model first:
    # rz_i = sum_{j != i} c_j^rz (sum of remaining node weights)
    # This means: c_j^rz is the Rz weight of node j
    # rz_i = total_rz - c_i^rz

    # From the ad hoc: if rz_i = C_rz - c_i^rz, then
    # c_i^rz = C_rz - rz_i
    # The differences between any two: c_i^rz - c_j^rz = rz_j - rz_i

    # c_S^rz - c_R^rz = rz_R - rz_S = 1.3 - 0.4 = 0.9
    # c_S^rz - c_T^rz = rz_T - rz_S = 0.7 - 0.4 = 0.3
    # c_S^rz - c_F^rz = rz_F - rz_S = 1.0 - 0.4 = 0.6
    # c_S^rz - c_P^rz = rz_P - rz_S = 1.5 - 0.4 = 1.1

    print("\n  LINEAR MODEL: rz_i = C - c_i^rz (total minus own weight)")
    print("\n  Differences in Rz weight (relative to S):")
    for i in range(5):
        diff_rz = ad_hoc[i, 0] - ad_hoc[0, 0]
        diff_rx = ad_hoc[i, 1] - ad_hoc[0, 1]
        print(f"    {GATE_NAMES[i]}: delta_rz = {diff_rz:+.1f}, delta_rx = {diff_rx:+.1f}")

    # Under the linear model, the per-node weights are (up to a constant):
    # c^rz: S=C-0.4, R=C-1.3, T=C-0.7, F=C-1.0, P=C-1.5
    # Normalize so sum = 0 (remove the constant):
    mean_rz = np.mean(ad_hoc[:, 0])  # = 0.78
    c_rz = -(ad_hoc[:, 0] - mean_rz)  # deviations

    mean_rx = np.mean(ad_hoc[:, 1])
    c_rx = -(ad_hoc[:, 1] - mean_rx)

    print(f"\n  Mean rz = {mean_rz:.4f}, Mean rx = {mean_rx:.4f}")
    print(f"\n  Per-node Rz deviation (c^rz, zero-sum):")
    for i in range(5):
        print(f"    {GATE_NAMES[i]}: c_rz = {c_rz[i]:+.4f}, c_rx = {c_rx[i]:+.4f}")

    # Check if c_rz and c_rx are Cartan matrix eigenvectors!
    print(f"\n  Are the weight vectors eigenvectors of the affine Cartan?")
    Ac_rz = CARTAN_AFF @ c_rz
    Ac_rx = CARTAN_AFF @ c_rx

    # Check proportionality
    for name, c, Ac in [("Rz", c_rz, Ac_rz), ("Rx", c_rx, Ac_rx)]:
        ratios = Ac / (c + 1e-30)
        print(f"\n    {name}: c = {c}")
        print(f"    A*c = {Ac}")
        print(f"    A*c / c = {ratios}")
        if np.std(ratios[np.abs(c) > 0.01]) < 0.1:
            print(f"    -> EIGENVECTOR with eigenvalue ~ {np.mean(ratios[np.abs(c) > 0.01]):.4f}")
        else:
            print(f"    -> NOT an eigenvector (ratios vary)")

    # Check against graph Laplacian eigenvectors
    print(f"\n  Affine A4 Cartan eigenvalues:")
    evals, evecs = np.linalg.eigh(CARTAN_AFF)
    for i in range(5):
        print(f"    lambda_{i} = {evals[i]:.6f}, v = {evecs[:, i]}")

    print(f"\n  Decompose c_rz and c_rx into Cartan eigenvectors:")
    for name, c in [("Rz", c_rz), ("Rx", c_rx)]:
        coeffs = evecs.T @ c
        print(f"    {name} = " + " + ".join([f"{coeffs[i]:.4f} * v_{i}" for i in range(5)]))

    # Try: can c_rz and c_rx be explained by just 2 eigenvectors?
    # (since the 5D space has a null direction, we really have 4D)
    print(f"\n  Projection onto non-null eigenspace (4D):")
    for name, c in [("Rz", c_rz), ("Rx", c_rx)]:
        coeffs = evecs.T @ c
        # Null direction is eigenvector 0 (eigenvalue 0)
        c_projected = sum(coeffs[i] * evecs[:, i] for i in range(1, 5))
        residual = np.linalg.norm(c - c_projected) / np.linalg.norm(c)
        print(f"    {name}: coeffs = {coeffs[1:]}, residual = {residual:.6f}")


# ============================================================================
# APPROACH 9: The key test - rational structure
# ============================================================================
def approach_rational():
    """
    Check if the ad hoc values have a rational structure consistent with A4^(1).
    """
    print("\n" + "=" * 70)
    print("APPROACH 9: RATIONAL STRUCTURE OF AD HOC VALUES")
    print("=" * 70)

    ad_hoc = {
        'S': (0.4, 1.3),
        'R': (1.3, 0.4),
        'T': (0.7, 0.7),
        'F': (1.0, 1.0),
        'P': (1.5, 1.8),
    }

    # Express as fractions
    print("\n  Ad hoc values as fractions:")
    for gate in GATE_NAMES:
        rz, rx = ad_hoc[gate]
        fz = Fraction(rz).limit_denominator(100)
        fx = Fraction(rx).limit_denominator(100)
        total = Fraction(rz + rx).limit_denominator(100)
        print(f"    {gate}: rz = {fz}, rx = {fx}, rz+rx = {total}, rz*rx = {Fraction(rz*rx).limit_denominator(100)}")

    # Key observations:
    # S: rz+rx = 1.7 = 17/10
    # R: rz+rx = 1.7 = 17/10
    # T: rz+rx = 1.4 = 7/5
    # F: rz+rx = 2.0 = 2
    # P: rz+rx = 3.3 = 33/10
    #
    # Products:
    # S: rz*rx = 0.52 = 13/25
    # R: rz*rx = 0.52 = 13/25
    # T: rz*rx = 0.49 = 49/100
    # F: rz*rx = 1.0 = 1
    # P: rz*rx = 2.7 = 27/10

    print("\n  Sums rz+rx:")
    sums = [ad_hoc[g][0] + ad_hoc[g][1] for g in GATE_NAMES]
    for i, g in enumerate(GATE_NAMES):
        print(f"    {g}: {sums[i]:.1f} = {Fraction(sums[i]).limit_denominator(100)}")

    print(f"\n  Sum of all sums: {sum(sums):.1f} = {Fraction(sum(sums)).limit_denominator(100)}")
    print(f"  Mean sum: {np.mean(sums):.2f}")

    # Check ratios to reference (F)
    print("\n  Ratios to F=1.0 reference:")
    for gate in GATE_NAMES:
        rz, rx = ad_hoc[gate]
        print(f"    {gate}: rz/1.0 = {Fraction(rz).limit_denominator(20)}, rx/1.0 = {Fraction(rx).limit_denominator(20)}")

    # The key pattern:
    # 0.4 = 2/5, 1.3 = 13/10, 0.7 = 7/10, 1.0 = 1, 1.5 = 3/2, 1.8 = 9/5
    # Check denominators: 5, 10, 10, 1, 2, 5

    print("\n  As exact fractions with denominator 10:")
    for gate in GATE_NAMES:
        rz, rx = ad_hoc[gate]
        rz10 = round(rz * 10)
        rx10 = round(rx * 10)
        print(f"    {gate}: rz = {rz10}/10, rx = {rx10}/10")

    # So the values in units of 1/10 are:
    # S: (4, 13), R: (13, 4), T: (7, 7), F: (10, 10), P: (15, 18)
    # Sum per gate: S:17, R:17, T:14, F:20, P:33
    # Total: 17+17+14+20+33 = 101

    vals_10 = [(4, 13), (13, 4), (7, 7), (10, 10), (15, 18)]
    total_rz = sum(v[0] for v in vals_10)
    total_rx = sum(v[1] for v in vals_10)
    print(f"\n  In units of 1/10:")
    print(f"    Total rz*10 = {total_rz}")
    print(f"    Total rx*10 = {total_rx}")
    print(f"    Grand total = {total_rz + total_rx}")

    # Differences from symmetric point (8.5, 8.5):
    print(f"\n  Deviations from mean ({total_rz/5:.1f}/10, {total_rx/5:.1f}/10):")
    mean_rz10 = total_rz / 5
    mean_rx10 = total_rx / 5
    for i, gate in enumerate(GATE_NAMES):
        drz = vals_10[i][0] - mean_rz10
        drx = vals_10[i][1] - mean_rx10
        print(f"    {gate}: ({drz:+.1f}, {drx:+.1f})")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("AFFINE A4^(1) MODULATION FACTOR DERIVATION")
    print("=" * 70)
    print(f"\nGate assignment: {', '.join([f'{g}=node {i}' for i, g in enumerate(GATE_NAMES)])}")
    print(f"Pentagon: S-R-T-F-P-S (cyclic)")
    print(f"Affine node: P")

    approach_weyl_vector()
    approach_adjacency()
    approach_cartan_inverse()
    approach_su2_projection()
    approach_coxeter()
    approach_dynkin_weights()
    approach_casimir()
    approach_numerical_search()
    approach_rational()

    print(f"\n{'='*70}")
    print(f"All output in: {OUT}")
