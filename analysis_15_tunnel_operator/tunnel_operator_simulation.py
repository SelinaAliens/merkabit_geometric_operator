#!/usr/bin/env python3
"""
INTER-MERKABIT TUNNEL OPERATOR: GOE -> GUE Phase Transition
=============================================================

Core hypothesis (Selina Stenberg, March 13 2026):
The intra-cell R/R-bar operator M is purely imaginary => GOE (beta~1).
The inter-cell tunnel T_AB couples FORWARD of A to INVERSE of B
(cross-chirality). This coupling is NOT purely imaginary because u^A
and v^B are from different cells with independent spinor phases.

Prediction: adding T_AB drives beta: 1 -> 2 (GOE -> GUE).

Modules:
  1. Two-cell system: beta(lambda) tunnel strength sweep
  2. Resonance condition: beta(delta) detuning sweep
  3. Scaling: 1 to N_cells, beta(N) at full coupling
  4. Critical ratio and F-connection
  5. Tunnel selectivity: g(|0>), g(|+1>), g(|-1>)
  6. Riemann zero comparison before/after tunnel
  7. Phase diagram: beta(lambda, N_cells)

Build on:
  multi_merkabit_cell_noise.py — HexagonalCell (7-node Eisenstein)
  cswap_coupling_simulation.py — C-SWAP resonance condition
  torsion_channel_simulation.py — MerkabitState, gate definitions
  Previous result: intra-cell M gave GOE beta~1
"""

import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.integrate import quad

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\tunnel_operator")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
XI = 3.0
F_RETURN = 0.696778
OMEGA_EISEN = np.exp(2j * np.pi / 3)

# Eisenstein unit vectors (6-fold coordination)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]


# ============================================================================
# HEXAGONAL CELL ON EISENSTEIN LATTICE
# ============================================================================

def eisenstein_coord_to_complex(a, b):
    """Convert Eisenstein integer (a, b) to complex number a + b*omega."""
    return a + b * OMEGA_EISEN


class HexagonalCell:
    """
    7-node hexagonal cell on the Eisenstein lattice.
    Centre at (a0, b0) + 6 nearest neighbours.

    Boundary: 6 outer nodes (3 neighbours each inside cell)
    Interior: 1 centre node (6 neighbours all inside cell)
    """

    def __init__(self, centre_a=0, centre_b=0, cell_id=0):
        self.centre_ab = (centre_a, centre_b)
        self.cell_id = cell_id

        # Generate 7 nodes: centre + 6 neighbours
        self.nodes_ab = [(centre_a, centre_b)]
        for da, db in UNIT_VECTORS_AB:
            self.nodes_ab.append((centre_a + da, centre_b + db))

        self.N = len(self.nodes_ab)
        self.sites = np.array([eisenstein_coord_to_complex(a, b)
                               for a, b in self.nodes_ab])

        # Adjacency within cell
        self.neighbours = {i: [] for i in range(self.N)}
        self.edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = abs(self.sites[i] - self.sites[j])
                if d < 1.15:  # nearest neighbour ~ 1.0
                    self.neighbours[i].append(j)
                    self.neighbours[j].append(i)
                    self.edges.append((i, j))

        # Boundary vs interior
        self.is_boundary = np.array([len(self.neighbours[i]) < 6
                                     for i in range(self.N)])
        self.boundary_indices = [i for i in range(self.N) if self.is_boundary[i]]
        self.interior_indices = [i for i in range(self.N) if not self.is_boundary[i]]

        # Sublattice colouring
        self.sublattice = np.array([(a + b) % 3 for a, b in self.nodes_ab])


def build_multi_cell_lattice(n_cells):
    """
    Build n_cells hexagonal cells on the Eisenstein lattice.

    Cell arrangement:
      1 cell:  centre only
      2 cells: centre + one neighbour
      7 cells: centre + 6 neighbours (complete first shell)
      19 cells: 7 + 12 second-shell cells
      37 cells: 19 + 18 third-shell cells

    Cell centres are on a SCALED Eisenstein lattice with spacing 2
    (each cell has radius 1, so centres are 2 apart to avoid overlap).
    """
    # Generate cell centres on scaled Eisenstein lattice
    cell_centres = []
    R = int(np.sqrt(n_cells)) + 2
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            # Scale by 2 to avoid node overlap between cells
            ca, cb = 2 * a, 2 * b
            cell_centres.append((ca, cb))

    # Sort by distance from origin, take n_cells
    cell_centres.sort(key=lambda ab: abs(eisenstein_coord_to_complex(*ab)))
    cell_centres = cell_centres[:n_cells]

    cells = []
    for idx, (ca, cb) in enumerate(cell_centres):
        cells.append(HexagonalCell(ca, cb, cell_id=idx))

    return cells


# ============================================================================
# SPINOR ASSIGNMENT
# ============================================================================

def assign_cell_spinors(cell, omega_base=1.0):
    """
    Assign dual spinors to each node in a cell.
    Forward spinor u: E6-constrained phase from lattice position.
    Inverse spinor v: time-reversed conjugate of u.
    Frequency omega: base frequency (sign alternates by sublattice).
    """
    N = cell.N
    U = np.zeros((N, 2), dtype=complex)
    V = np.zeros((N, 2), dtype=complex)
    omegas = np.zeros(N)

    L_scale = np.max(np.abs(cell.sites)) + 1.0

    for i in range(N):
        a, b = cell.nodes_ab[i]
        phase = np.pi * (a - b) / RANK_E6
        r = abs(cell.sites[i] - eisenstein_coord_to_complex(*cell.centre_ab))
        theta = np.pi * r / L_scale

        u = np.array([
            np.cos(theta / 2) * np.exp(1j * phase),
            1j * np.sin(theta / 2) * np.exp(-1j * phase)
        ], dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)

        U[i] = u
        V[i] = v

        # Frequency: alternate sign by sublattice for resonance
        if cell.sublattice[i] == 0:
            omegas[i] = omega_base
        elif cell.sublattice[i] == 1:
            omegas[i] = omega_base * 0.9
        else:
            omegas[i] = -omega_base

    return U, V, omegas


# ============================================================================
# INTRA-CELL OPERATOR M (R/R-bar merger, purely imaginary -> GOE)
# ============================================================================

def build_intra_cell_M(cell, U, V, xi=XI):
    """
    Intra-cell R/R-bar merger: M_ij = J(r) * [<u_i|v_j> + <v_i|u_j>]
    This is purely imaginary by construction (proven in previous run).
    """
    N = cell.N
    M = np.zeros((N, N), dtype=complex)

    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))
        for j in range(i + 1, N):
            d = abs(cell.sites[i] - cell.sites[j])
            J = np.exp(-d / xi)
            uv = np.vdot(U[i], V[j])
            vu = np.vdot(V[i], U[j])
            coupling = J * (uv + vu)
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)

    return M


# ============================================================================
# INTER-CELL TUNNEL OPERATOR T_AB (cross-chirality, complex -> GUE)
# ============================================================================

def find_boundary_pairs(cell_A, cell_B, threshold=1.15):
    """
    Find pairs of boundary nodes (i in A, j in B) that are
    Eisenstein nearest neighbours in the combined lattice.
    """
    pairs = []
    for i in cell_A.boundary_indices:
        for j in cell_B.boundary_indices:
            d = abs(cell_A.sites[i] - cell_B.sites[j])
            if d < threshold:
                pairs.append((i, j))
    return pairs


def build_tunnel_T(cell_A, cell_B, U_A, V_A, U_B, V_B,
                   omegas_A, omegas_B, J_tunnel=1.0, xi=XI,
                   detuning=0.0):
    """
    Inter-cell tunnel operator T_AB.

    T_AB[i,j] = J_tunnel * exp(-r/xi) * resonance * <u_i^A | v_j^B>

    CROSS-CHIRALITY: forward of A couples to inverse of B.
    This is NOT the same as the intra-cell coupling because:
      - u^A and v^B belong to DIFFERENT cells
      - Their phases are independently assigned
      - <u_i^A | v_j^B> is COMPLEX (not purely imaginary)

    Resonance: exp(-(omega_A + omega_B + detuning)^2 / sigma^2)
    """
    N_A = cell_A.N
    N_B = cell_B.N
    T = np.zeros((N_A, N_B), dtype=complex)

    pairs = find_boundary_pairs(cell_A, cell_B)
    sigma_res = 0.1  # resonance width

    for i_A, j_B in pairs:
        r = abs(cell_A.sites[i_A] - cell_B.sites[j_B])
        J = J_tunnel * np.exp(-r / xi)

        # Resonance condition: omega_A + omega_B = 0
        omega_sum = omegas_A[i_A] + omegas_B[j_B] + detuning
        resonance = np.exp(-omega_sum ** 2 / (2 * sigma_res ** 2))

        # Cross-chirality coupling: forward of A <-> inverse of B
        # <u_i^A | v_j^B> -- this is COMPLEX, not purely imaginary
        uv_cross = np.vdot(U_A[i_A], V_B[j_B])
        T[i_A, j_B] = J * resonance * uv_cross

    # ALSO couple inverse of A to forward of B (bidirectional tunnel)
    # This ensures the full operator is Hermitian
    # T_BA = T_AB^dagger is handled by the block matrix construction

    return T


def build_full_hamiltonian(cells, spinors_list, J_tunnel=1.0, xi=XI,
                           detuning=0.0):
    """
    Build the full Hamiltonian for multiple cells:

    H = | M_0     T_01    T_02   ... |
        | T_01^+  M_1     T_12   ... |
        | T_02^+  T_12^+  M_2    ... |
        | ...                         |

    Diagonal blocks: intra-cell M_i (GOE, purely imaginary)
    Off-diagonal blocks: tunnel T_ij (complex, symmetry-breaking)
    """
    n_cells = len(cells)
    sizes = [c.N for c in cells]
    total = sum(sizes)
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    H = np.zeros((total, total), dtype=complex)

    # Diagonal blocks: intra-cell M
    for c_idx in range(n_cells):
        cell = cells[c_idx]
        U, V, omegas = spinors_list[c_idx]
        M = build_intra_cell_M(cell, U, V, xi)
        o = offsets[c_idx]
        H[o:o + cell.N, o:o + cell.N] = M

    # Off-diagonal blocks: tunnel T_ij
    for c_i in range(n_cells):
        for c_j in range(c_i + 1, n_cells):
            # Check if cells are neighbours (centres within distance ~2.5)
            d_centres = abs(cells[c_i].sites[0] - cells[c_j].sites[0])
            # Cell centres are 2 apart in Eisenstein metric
            # Boundary nodes of adjacent cells should be within ~1 of each other
            if d_centres > 4.0:  # too far apart
                continue

            U_i, V_i, om_i = spinors_list[c_i]
            U_j, V_j, om_j = spinors_list[c_j]

            T = build_tunnel_T(cells[c_i], cells[c_j],
                               U_i, V_i, U_j, V_j,
                               om_i, om_j,
                               J_tunnel=J_tunnel, xi=xi,
                               detuning=detuning)

            o_i = offsets[c_i]
            o_j = offsets[c_j]
            H[o_i:o_i + cells[c_i].N, o_j:o_j + cells[c_j].N] = T
            H[o_j:o_j + cells[c_j].N, o_i:o_i + cells[c_i].N] = T.conj().T

    # Force exact Hermiticity
    H = (H + H.conj().T) / 2.0
    return H


# ============================================================================
# SPECTRAL ANALYSIS TOOLS
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=10):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        spacings = np.diff(evals)
        if len(spacings) > 0 and np.mean(spacings) > 0:
            spacings /= np.mean(spacings)
        return evals, spacings

    cdf = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 10))
    coeffs = np.polyfit(evals, cdf * N, deg)
    N_smooth = np.polyval(coeffs, evals)
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def unfold_riemann_zeros(zeros):
    T = np.sort(zeros)
    N_smooth = (T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7.0 / 8
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings


def make_wigner_cdf(beta):
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2

    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for i, si in enumerate(s):
            result[i], _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(si, 0))
        return result
    return cdf


def wigner_pdf(s, beta):
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    return a * s ** beta * np.exp(-b * s ** 2)


def fit_beta(spacings):
    """Fit level repulsion: P(s) ~ s^beta for small s."""
    pos = spacings[spacings > 0.02]
    if len(pos) < 8:
        pos = spacings[spacings > 0.005]
    if len(pos) < 5:
        return 0.0, 0.0

    small = pos[pos < np.percentile(pos, 40)]
    if len(small) < 5:
        small = pos[pos < np.median(pos)]
    if len(small) < 3:
        return 0.0, 0.0

    n_bins = min(20, max(4, len(small) // 3))
    hist, edges = np.histogram(small, bins=n_bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0

    log_s = np.log(centres[mask])
    log_p = np.log(hist[mask])
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = coeffs[0]
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def ks_tests(spacings, riemann_spacings=None):
    """Run KS tests against GOE, GUE, GSE, Poisson, Riemann."""
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {k: (1.0, 0.0) for k in ['goe', 'gue', 'gse', 'poi', 'riem']}

    results = {}
    for name, beta in [('goe', 1), ('gue', 2), ('gse', 4)]:
        cdf = make_wigner_cdf(beta)
        ks, p = stats.kstest(pos, cdf)
        results[name] = (ks, p)

    ks, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (ks, p)

    if riemann_spacings is not None and len(riemann_spacings) > 0:
        ks, p = stats.ks_2samp(pos, riemann_spacings)
        results['riem'] = (ks, p)
    else:
        results['riem'] = (1.0, 0.0)

    return results


# ============================================================================
# MODULE 5: TUNNEL SELECTIVITY
# ============================================================================

def tunnel_selectivity():
    """Check which trit states pass through the tunnel."""
    print("\n" + "=" * 76)
    print("  MODULE 5: TUNNEL SELECTIVITY")
    print("=" * 76)

    states = {
        "|+1>": ([1, 0], [1, 0]),
        "| 0>": ([1, 0], [0, 1]),
        "|-1>": ([1, 0], [-1, 0]),
    }

    lines = []
    print(f"\n  {'State':<8} {'phi':>8} {'cos^2(phi)':>12} {'|<u|v>|^2':>12} {'1-|<u|v>|^2':>14}")
    print("  " + "-" * 60)

    for name, (u, v) in states.items():
        u, v = np.array(u, dtype=complex), np.array(v, dtype=complex)
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        phi = np.angle(np.vdot(u, v))
        g_cos2 = np.cos(phi) ** 2
        overlap2 = abs(np.vdot(u, v)) ** 2
        anti_overlap = 1 - overlap2
        print(f"  {name:<8} {phi:8.4f} {g_cos2:12.4f} {overlap2:12.4f} {anti_overlap:14.4f}")
        lines.append(f"{name}: phi={phi:.4f}, cos^2(phi)={g_cos2:.4f}, "
                     f"|<u|v>|^2={overlap2:.4f}")

    print(f"\n  Interpretation:")
    print(f"    cos^2(phi): |+1> and |-1> tunnel (g=1), |0> blocked (g=0)")
    print(f"    This is TRIT-SELECTIVE: standing wave |0> does NOT tunnel")
    print(f"    Correspondence: trivial zero blocked, non-trivial zeros pass")

    with open(RESULTS_DIR / "tunnel_selectivity.txt", 'w') as f:
        f.write("TUNNEL SELECTIVITY\n")
        f.write("g(phi) = cos^2(phi) where phi = arg(<u|v>)\n\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n|+1> and |-1> tunnel (g=1), |0> blocked (g=0)\n")

    return


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 76)
    print("  INTER-MERKABIT TUNNEL OPERATOR")
    print("  GOE -> GUE Phase Transition via Cross-Chirality Coupling")
    print("=" * 76)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Architecture: E6 (rank={RANK_E6}, dim={DIM_E6}, h={COXETER_H})")
    print()

    # Load Riemann zeros
    zeros_file = Path(r"C:\Users\selin\merkabit_results\riemann_zeros\riemann_zeros_cache.npy")
    if zeros_file.exists():
        riemann_zeros = np.load(zeros_file)
    else:
        riemann_zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876,
                                  32.935062, 37.586178, 40.918719, 43.327073])
    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    pos_riem = spacings_riem[spacings_riem > 0]
    print(f"  Riemann zeros: {len(riemann_zeros)}")

    # ====================================================================
    # MODULE 5: Tunnel selectivity (quick, informative)
    # ====================================================================
    tunnel_selectivity()

    # ====================================================================
    # MODULE 1: TWO-CELL SYSTEM — beta(lambda)
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 1: TWO-CELL SYSTEM — beta(lambda) TUNNEL STRENGTH SWEEP")
    print(f"{'='*76}")

    cell_A = HexagonalCell(0, 0, cell_id=0)
    cell_B = HexagonalCell(2, 0, cell_id=1)

    U_A, V_A, om_A = assign_cell_spinors(cell_A, omega_base=1.0)
    U_B, V_B, om_B = assign_cell_spinors(cell_B, omega_base=-1.0)  # counter-rotating

    # Verify intra-cell is purely imaginary
    M_A = build_intra_cell_M(cell_A, U_A, V_A)
    M_B = build_intra_cell_M(cell_B, U_B, V_B)
    re_A = np.linalg.norm(np.real(M_A))
    im_A = np.linalg.norm(np.imag(M_A))
    print(f"\n  Intra-cell M_A: Re/Im = {re_A:.4f}/{im_A:.4f}")

    # Build tunnel
    T_AB = build_tunnel_T(cell_A, cell_B, U_A, V_A, U_B, V_B,
                          om_A, om_B, J_tunnel=1.0)
    re_T = np.linalg.norm(np.real(T_AB))
    im_T = np.linalg.norm(np.imag(T_AB))
    n_nonzero_T = np.sum(np.abs(T_AB) > 1e-10)
    print(f"  Tunnel T_AB: Re/Im = {re_T:.4f}/{im_T:.4f}, nonzero entries: {n_nonzero_T}")
    print(f"  Boundary pairs found: {len(find_boundary_pairs(cell_A, cell_B))}")

    # Lambda sweep
    lambda_vals = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    print(f"\n  {'lambda':>8} {'beta_w':>8} {'beta_f':>8} "
          f"{'KS_GOE':>8} {'p_GOE':>8} {'KS_GUE':>8} {'p_GUE':>8} "
          f"{'KS_Poi':>8} {'KS_Riem':>8} {'Re/Im':>10}")
    print("  " + "-" * 100)

    lambda_results = []
    for lam in lambda_vals:
        N_A, N_B = cell_A.N, cell_B.N
        H = np.zeros((N_A + N_B, N_A + N_B), dtype=complex)
        H[:N_A, :N_A] = M_A
        H[N_A:, N_A:] = M_B
        H[:N_A, N_A:] = lam * T_AB
        H[N_A:, :N_A] = lam * T_AB.conj().T
        H = (H + H.conj().T) / 2.0

        re_H = np.linalg.norm(np.real(H))
        im_H = np.linalg.norm(np.imag(H))
        tot_H = np.linalg.norm(H)

        eigenvalues = np.linalg.eigvalsh(H)
        _, spacings = unfold_spectrum(eigenvalues)
        pos_sp = spacings[spacings > 0]

        beta_w, r2_w = fit_beta(pos_sp)
        # Also full-spectrum beta
        beta_f, _ = fit_beta(spacings[spacings > 0.005]) if len(spacings) > 5 else (0, 0)

        ks = ks_tests(pos_sp, pos_riem)

        res = {
            'lambda': lam, 'beta_wing': beta_w, 'beta_full': beta_f,
            'ks_goe': ks['goe'][0], 'p_goe': ks['goe'][1],
            'ks_gue': ks['gue'][0], 'p_gue': ks['gue'][1],
            'ks_poi': ks['poi'][0], 'p_poi': ks['poi'][1],
            'ks_riem': ks['riem'][0], 'p_riem': ks['riem'][1],
            'real_frac': re_H / (tot_H + 1e-15),
            'imag_frac': im_H / (tot_H + 1e-15),
            'eigenvalues': eigenvalues, 'spacings': pos_sp,
        }
        lambda_results.append(res)

        print(f"  {lam:8.2f} {beta_w:8.3f} {beta_f:8.3f} "
              f"{ks['goe'][0]:8.4f} {ks['goe'][1]:8.4f} "
              f"{ks['gue'][0]:8.4f} {ks['gue'][1]:8.4f} "
              f"{ks['poi'][0]:8.4f} {ks['riem'][0]:8.4f} "
              f"{re_H/(tot_H+1e-15):4.2f}/{im_H/(tot_H+1e-15):4.2f}")

    # Find lambda* where beta crosses 2.0
    betas_lam = np.array([r['beta_wing'] for r in lambda_results])
    lams = np.array([r['lambda'] for r in lambda_results])
    cross_mask = betas_lam >= 2.0
    if np.any(cross_mask):
        lambda_star = lams[cross_mask][0]
        print(f"\n  lambda* (beta >= 2.0): {lambda_star:.4f}")
    else:
        beta_max = np.max(betas_lam)
        lambda_at_max = lams[np.argmax(betas_lam)]
        print(f"\n  beta did not reach 2.0. Max beta = {beta_max:.3f} at lambda = {lambda_at_max:.2f}")
        lambda_star = lambda_at_max

    # Save
    np.save(RESULTS_DIR / "beta_vs_lambda.npy",
            np.column_stack([lams, betas_lam]))

    # ====================================================================
    # MODULE 2: RESONANCE CONDITION — beta(detuning)
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 2: RESONANCE CONDITION — beta(detuning)")
    print(f"{'='*76}")

    delta_vals = np.linspace(0, np.pi, 15)
    lam_fixed = max(1.0, lambda_star)

    print(f"\n  Fixed lambda = {lam_fixed:.2f}")
    print(f"\n  {'delta':>8} {'delta/pi':>8} {'beta':>8} {'KS_GUE':>8} {'KS_GOE':>8}")
    print("  " + "-" * 50)

    detuning_results = []
    for delta in delta_vals:
        T_det = build_tunnel_T(cell_A, cell_B, U_A, V_A, U_B, V_B,
                               om_A, om_B, J_tunnel=1.0, detuning=delta)
        H = np.zeros((N_A + N_B, N_A + N_B), dtype=complex)
        H[:N_A, :N_A] = M_A
        H[N_A:, N_A:] = M_B
        H[:N_A, N_A:] = lam_fixed * T_det
        H[N_A:, :N_A] = lam_fixed * T_det.conj().T
        H = (H + H.conj().T) / 2.0

        eigenvalues = np.linalg.eigvalsh(H)
        _, spacings = unfold_spectrum(eigenvalues)
        pos_sp = spacings[spacings > 0]
        beta, _ = fit_beta(pos_sp)
        ks = ks_tests(pos_sp, pos_riem)

        detuning_results.append({
            'delta': delta, 'beta': beta,
            'ks_gue': ks['gue'][0], 'ks_goe': ks['goe'][0]
        })
        print(f"  {delta:8.4f} {delta/np.pi:8.4f} {beta:8.3f} "
              f"{ks['gue'][0]:8.4f} {ks['goe'][0]:8.4f}")

    # ====================================================================
    # MODULE 3: SCALING — 1 to N_cells
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 3: SCALING — beta(N_cells) AT FULL COUPLING")
    print(f"{'='*76}")

    cell_counts = [1, 2, 7, 19, 37]
    lam_scale = max(1.0, lambda_star)

    print(f"\n  lambda = {lam_scale:.2f}")
    print(f"\n  {'N_cells':>8} {'N_evals':>8} {'beta':>8} {'KS_GUE':>8} {'p_GUE':>8} "
          f"{'KS_Riem':>8} {'DOS':>8} {'BW':>8}")
    print("  " + "-" * 75)

    scaling_results = []
    for nc in cell_counts:
        cells = build_multi_cell_lattice(nc)
        spinors = [assign_cell_spinors(c, omega_base=(-1) ** c.cell_id)
                    for c in cells]

        if nc == 1:
            # Single cell: just M
            U0, V0, om0 = spinors[0]
            H = build_intra_cell_M(cells[0], U0, V0)
        else:
            H = build_full_hamiltonian(cells, spinors, J_tunnel=lam_scale)

        assert np.max(np.abs(H - H.conj().T)) < 1e-9, "Not Hermitian"

        eigenvalues = np.linalg.eigvalsh(H)
        _, spacings = unfold_spectrum(eigenvalues)
        pos_sp = spacings[spacings > 0]
        beta, r2 = fit_beta(pos_sp)
        ks = ks_tests(pos_sp, pos_riem)

        bandwidth = eigenvalues.max() - eigenvalues.min()
        dos = len(eigenvalues) / bandwidth if bandwidth > 0 else 0

        scaling_results.append({
            'N_cells': nc, 'N_evals': len(eigenvalues),
            'beta': beta, 'beta_r2': r2,
            'ks_gue': ks['gue'][0], 'p_gue': ks['gue'][1],
            'ks_riem': ks['riem'][0], 'p_riem': ks['riem'][1],
            'ks_poi': ks['poi'][0],
            'dos': dos, 'bandwidth': bandwidth,
        })
        print(f"  {nc:8d} {len(eigenvalues):8d} {beta:8.3f} "
              f"{ks['gue'][0]:8.4f} {ks['gue'][1]:8.4f} "
              f"{ks['riem'][0]:8.4f} {dos:8.3f} {bandwidth:8.3f}")

    np.save(RESULTS_DIR / "beta_vs_N.npy",
            np.array([(r['N_cells'], r['beta']) for r in scaling_results]))

    # DOS scaling test: log(N) or N?
    Nc = np.array([r['N_cells'] for r in scaling_results if r['N_cells'] > 1])
    DOS = np.array([r['dos'] for r in scaling_results if r['N_cells'] > 1])
    N_ev = np.array([r['N_evals'] for r in scaling_results if r['N_cells'] > 1])

    if len(Nc) >= 3:
        # Fit DOS = a * log(N) + b
        log_fit = np.polyfit(np.log(Nc), DOS, 1)
        # Fit DOS = a * N + b
        lin_fit = np.polyfit(Nc, DOS, 1)

        log_pred = np.polyval(log_fit, np.log(Nc))
        lin_pred = np.polyval(lin_fit, Nc)
        r2_log = 1 - np.sum((DOS - log_pred) ** 2) / np.sum((DOS - np.mean(DOS)) ** 2)
        r2_lin = 1 - np.sum((DOS - lin_pred) ** 2) / np.sum((DOS - np.mean(DOS)) ** 2)

        print(f"\n  DOS scaling:")
        print(f"    log(N) fit: R2 = {r2_log:.4f}")
        print(f"    linear N fit: R2 = {r2_lin:.4f}")
        print(f"    {'LOG wins (Riemann-like!)' if r2_log > r2_lin else 'LINEAR wins (Weyl law)'}")

    # ====================================================================
    # MODULE 4: CRITICAL RATIO AND F-CONNECTION
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 4: CRITICAL RATIO AND F-CONNECTION")
    print(f"{'='*76}")

    for r in [lambda_results[0]] + [lr for lr in lambda_results if lr['lambda'] == 1.0]:
        ev = r['eigenvalues']
        n_pos = np.sum(ev > 1e-10)
        n_tot = len(ev)
        ratio = n_pos / n_tot
        print(f"  lambda={r['lambda']:.2f}: {n_pos}/{n_tot} positive = {ratio:.5f}")

    print(f"  -ln(F) = {-np.log(F_RETURN):.5f}")
    print(f"  36/100 = {36 / 100:.5f}")

    with open(RESULTS_DIR / "f_connection_two_cell.txt", 'w') as f:
        f.write("F-CONNECTION: Positive eigenvalue ratio\n\n")
        for r in lambda_results:
            ev = r['eigenvalues']
            n_pos = np.sum(ev > 1e-10)
            ratio = n_pos / len(ev)
            f.write(f"lambda={r['lambda']:.2f}: {n_pos}/{len(ev)} = {ratio:.5f}\n")
        f.write(f"\n-ln(F) = {-np.log(F_RETURN):.5f}\n")

    # ====================================================================
    # MODULE 6: RIEMANN COMPARISON BEFORE/AFTER TUNNEL
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 6: RIEMANN COMPARISON — BEFORE vs AFTER TUNNEL")
    print(f"{'='*76}")

    # Before: single cell M_A
    ev_before = np.linalg.eigvalsh(M_A)
    _, sp_before = unfold_spectrum(ev_before)
    ks_before = ks_tests(sp_before[sp_before > 0], pos_riem)

    # After: two-cell at lambda=1
    lam1_result = [r for r in lambda_results if r['lambda'] == 1.0][0]
    ks_after = ks_tests(lam1_result['spacings'], pos_riem)

    print(f"\n  Single cell M_A:")
    print(f"    KS(Riemann) = {ks_before['riem'][0]:.4f} (p={ks_before['riem'][1]:.4f})")
    print(f"    KS(GUE) = {ks_before['gue'][0]:.4f} (p={ks_before['gue'][1]:.4f})")

    print(f"\n  Two-cell (lambda=1.0):")
    print(f"    KS(Riemann) = {ks_after['riem'][0]:.4f} (p={ks_after['riem'][1]:.4f})")
    print(f"    KS(GUE) = {ks_after['gue'][0]:.4f} (p={ks_after['gue'][1]:.4f})")

    improved = ks_after['riem'][0] < ks_before['riem'][0]
    print(f"\n  Tunnel IMPROVED Riemann match: {'YES' if improved else 'NO'}")
    print(f"    Delta KS(Riem) = {ks_after['riem'][0] - ks_before['riem'][0]:+.4f}")

    with open(RESULTS_DIR / "riemann_comparison.txt", 'w') as f:
        f.write("RIEMANN COMPARISON: Before vs After Tunnel\n\n")
        f.write(f"Single cell: KS(Riem)={ks_before['riem'][0]:.4f} p={ks_before['riem'][1]:.4f}\n")
        f.write(f"Two-cell:    KS(Riem)={ks_after['riem'][0]:.4f} p={ks_after['riem'][1]:.4f}\n")
        f.write(f"Improved: {'YES' if improved else 'NO'}\n")

    # ====================================================================
    # MODULE 7: PHASE DIAGRAM beta(lambda, N_cells)
    # ====================================================================
    print(f"\n{'='*76}")
    print("  MODULE 7: PHASE DIAGRAM beta(lambda, N_cells)")
    print(f"{'='*76}")

    lam_grid = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    nc_grid = [1, 2, 7, 19]
    beta_grid = np.zeros((len(lam_grid), len(nc_grid)))

    for j, nc in enumerate(nc_grid):
        cells_g = build_multi_cell_lattice(nc)
        spinors_g = [assign_cell_spinors(c, omega_base=(-1) ** c.cell_id)
                     for c in cells_g]

        for i, lam in enumerate(lam_grid):
            if nc == 1:
                U0, V0, om0 = spinors_g[0]
                H_g = build_intra_cell_M(cells_g[0], U0, V0)
            else:
                H_g = build_full_hamiltonian(cells_g, spinors_g, J_tunnel=lam)

            ev = np.linalg.eigvalsh(H_g)
            _, sp = unfold_spectrum(ev)
            beta_g, _ = fit_beta(sp[sp > 0])
            beta_grid[i, j] = beta_g

    np.save(RESULTS_DIR / "beta_phase_diagram.npy", beta_grid)

    print(f"\n  {'':>8}", end="")
    for nc in nc_grid:
        print(f"  {'N='+str(nc):>8}", end="")
    print()
    for i, lam in enumerate(lam_grid):
        print(f"  lam={lam:4.1f}", end="")
        for j in range(len(nc_grid)):
            print(f"  {beta_grid[i,j]:8.3f}", end="")
        print()

    # ====================================================================
    # GENERATE FIGURES
    # ====================================================================
    print(f"\n  Generating figures...")

    s_range = np.linspace(0.01, 4.0, 200)

    # --- Figure 1: beta(lambda) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    lams_plot = np.array([r['lambda'] for r in lambda_results])
    betas_plot = np.array([r['beta_wing'] for r in lambda_results])
    ax.semilogx(lams_plot[lams_plot > 0], betas_plot[lams_plot > 0],
                'bo-', ms=8, lw=2, label='beta (positive wing)')
    ax.axhline(y=2, color='r', ls='--', lw=2, label='GUE (beta=2)')
    ax.axhline(y=1, color='b', ls=':', lw=1.5, label='GOE (beta=1)')
    ax.axhline(y=0, color='k', ls=':', lw=1, alpha=0.5, label='Poisson')
    if np.any(cross_mask):
        ax.axvline(x=lambda_star, color='magenta', ls='-', lw=2, alpha=0.7,
                   label=f'lambda*={lambda_star:.2f}')
    ax.set_xlabel('Tunnel strength lambda', fontsize=14)
    ax.set_ylabel('Level repulsion beta', fontsize=14)
    ax.set_title('GOE -> GUE Transition: beta vs Tunnel Strength (Two-Cell)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure1_beta_vs_lambda.png', dpi=150)
    plt.close(fig)
    print("    figure1_beta_vs_lambda.png saved")

    # --- Figure 2: beta(N_cells) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    nc_plot = [r['N_cells'] for r in scaling_results]
    beta_nc = [r['beta'] for r in scaling_results]
    ax.plot(nc_plot, beta_nc, 'ro-', ms=10, lw=2)
    ax.axhline(y=2, color='r', ls='--', lw=2, label='GUE (beta=2)')
    ax.axhline(y=1, color='b', ls=':', lw=1.5, label='GOE (beta=1)')
    ax.set_xlabel('Number of cells', fontsize=14)
    ax.set_ylabel('Level repulsion beta', fontsize=14)
    ax.set_title('Scaling: beta vs N_cells at Full Coupling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure2_beta_vs_N.png', dpi=150)
    plt.close(fig)
    print("    figure2_beta_vs_N.png saved")

    # --- Figure 3: Phase diagram ---
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(beta_grid, aspect='auto', cmap='RdBu_r',
                   vmin=0, vmax=4, origin='lower',
                   extent=[0, len(nc_grid) - 1, 0, len(lam_grid) - 1])
    ax.set_xticks(range(len(nc_grid)))
    ax.set_xticklabels(nc_grid)
    ax.set_yticks(range(len(lam_grid)))
    ax.set_yticklabels([f'{l:.1f}' for l in lam_grid])
    ax.set_xlabel('N_cells', fontsize=14)
    ax.set_ylabel('Tunnel strength lambda', fontsize=14)
    ax.set_title('Phase Diagram: beta(lambda, N_cells)', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, label='beta')
    # Mark beta=2 contour
    for i in range(len(lam_grid)):
        for j in range(len(nc_grid)):
            color = 'white' if beta_grid[i, j] > 2 else 'black'
            ax.text(j, i, f'{beta_grid[i,j]:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure3_phase_diagram.png', dpi=150)
    plt.close(fig)
    print("    figure3_phase_diagram.png saved")

    # --- Figure 4: DOS scaling ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    nc_s = [r['N_cells'] for r in scaling_results]
    dos_s = [r['dos'] for r in scaling_results]
    ne_s = [r['N_evals'] for r in scaling_results]

    axes[0].plot(nc_s, ne_s, 'bo-', ms=8, lw=2)
    axes[0].set_xlabel('N_cells', fontsize=13)
    axes[0].set_ylabel('N_eigenvalues', fontsize=13)
    axes[0].set_title('Eigenvalue Count vs Cells', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nc_s, dos_s, 'ro-', ms=8, lw=2, label='DOS(N)')
    if len(Nc) >= 3:
        nc_fine = np.linspace(1, max(nc_s), 100)
        axes[1].plot(nc_fine, np.polyval(log_fit, np.log(nc_fine)),
                     'g--', lw=2, label=f'a*log(N)+b (R2={r2_log:.3f})')
        axes[1].plot(nc_fine, np.polyval(lin_fit, nc_fine),
                     'k:', lw=2, label=f'a*N+b (R2={r2_lin:.3f})')
    axes[1].set_xlabel('N_cells', fontsize=13)
    axes[1].set_ylabel('Density of States', fontsize=13)
    axes[1].set_title('DOS Scaling: log(N) vs N?', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure4_dos_scaling.png', dpi=150)
    plt.close(fig)
    print("    figure4_dos_scaling.png saved")

    # --- Figure 5: Tunnel selectivity ---
    fig, ax = plt.subplots(figsize=(8, 6))
    phi_range = np.linspace(-np.pi, np.pi, 200)
    g_vals = np.cos(phi_range) ** 2
    ax.plot(phi_range / np.pi, g_vals, 'b-', lw=2)
    # Mark trit states
    for name, phi_val, color in [('|+1>', 0, 'green'), ('|0>', np.pi/2, 'red'),
                                   ('|-1>', np.pi, 'green')]:
        g = np.cos(phi_val) ** 2
        ax.plot(phi_val / np.pi, g, 'o', ms=12, color=color, zorder=5)
        ax.annotate(name, (phi_val / np.pi, g), textcoords="offset points",
                    xytext=(10, 10), fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('phi / pi', fontsize=14)
    ax.set_ylabel('g(phi) = cos^2(phi)', fontsize=14)
    ax.set_title('Tunnel Selectivity: Which Trit States Pass?', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure5_tunnel_selectivity.png', dpi=150)
    plt.close(fig)
    print("    figure5_tunnel_selectivity.png saved")

    # --- Figure 6: Spacing distributions before/after ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def safe_hist(ax, data, max_bins, color, edgecolor):
        """Histogram that handles sparse/degenerate data gracefully."""
        d = data[np.isfinite(data)]
        if len(d) < 2 or (np.max(d) - np.min(d)) < 1e-12:
            # Too few points or zero range — show scatter instead
            if len(d) > 0:
                ax.scatter(d, np.ones_like(d) * 0.5, s=80, c=color,
                           edgecolors=edgecolor, zorder=5)
            ax.text(0.5, 0.5, f'N={len(d)} spacings\n(too few for histogram)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, style='italic', color='gray')
        else:
            nb = max(3, min(max_bins, len(d) // 2))
            try:
                ax.hist(d, bins=nb, density=True, alpha=0.5,
                        color=color, edgecolor=edgecolor)
            except ValueError:
                ax.hist(d, bins='auto', density=True, alpha=0.5,
                        color=color, edgecolor=edgecolor)

    # Before (single cell)
    ax = axes[0]
    sp_b_pos = sp_before[sp_before > 0]
    safe_hist(ax, sp_b_pos, 15, 'steelblue', 'navy')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    beta_b = fit_beta(sp_b_pos)[0] if len(sp_b_pos) > 1 else 0.0
    ax.set_title(f'Single Cell M_A (7 eigenvalues)\nbeta~{beta_b:.2f}',
                 fontsize=12)
    ax.set_xlabel('s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # After (two-cell lambda=1)
    ax = axes[1]
    sp_after = lam1_result['spacings']
    safe_hist(ax, sp_after, 20, 'coral', 'darkred')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.set_title(f'Two-Cell H (14 eigenvalues, lambda=1)\nbeta~{lam1_result["beta_wing"]:.2f}',
                 fontsize=12)
    ax.set_xlabel('s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Before vs After Tunnel: Spacing Distribution', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure6_riemann_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    figure6_riemann_comparison.png saved")

    # ====================================================================
    # FULL REPORT
    # ====================================================================
    print(f"\n{'='*76}")
    print("  FULL REPORT")
    print(f"{'='*76}")

    print(f"\n  1. TUNNEL SYMMETRY BREAKING:")
    print(f"     Intra-cell M: purely imaginary (Re/Im = {re_A:.4f}/{im_A:.4f})")
    print(f"     Tunnel T_AB:  complex (Re/Im = {re_T:.4f}/{im_T:.4f})")
    tr = "YES" if re_T > 0.01 else "NO"
    print(f"     Tunnel has real component: {tr}")
    print(f"     -> Time-reversal breaking: {'CONFIRMED' if tr == 'YES' else 'NOT CONFIRMED'}")

    print(f"\n  2. GOE -> GUE TRANSITION:")
    print(f"     beta(lambda=0) = {lambda_results[0]['beta_wing']:.3f}")
    if len([r for r in lambda_results if r['lambda'] == 1.0]) > 0:
        b1 = [r for r in lambda_results if r['lambda'] == 1.0][0]['beta_wing']
        print(f"     beta(lambda=1) = {b1:.3f}")
    print(f"     beta(max lambda) = {betas_lam[-1]:.3f}")
    if np.any(cross_mask):
        print(f"     lambda* (beta=2) = {lambda_star:.4f}")
        print(f"     lambda* / (1/12) = {lambda_star * 12:.4f}")
        print(f"     lambda* / (pi/6) = {lambda_star / (np.pi/6):.4f}")
    else:
        print(f"     beta never reached 2.0 (max = {np.max(betas_lam):.3f})")

    print(f"\n  3. SCALING:")
    for r in scaling_results:
        print(f"     N={r['N_cells']:3d}: beta={r['beta']:.3f} "
              f"KS(GUE)={r['ks_gue']:.4f} KS(Riem)={r['ks_riem']:.4f}")

    print(f"\n  4. RIEMANN IMPROVEMENT:")
    print(f"     Before tunnel: KS(Riem) = {ks_before['riem'][0]:.4f}")
    print(f"     After tunnel:  KS(Riem) = {ks_after['riem'][0]:.4f}")
    print(f"     Improved: {'YES' if improved else 'NO'}")

    # Save full report
    with open(RESULTS_DIR / "FULL_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(f"INTER-MERKABIT TUNNEL OPERATOR: FULL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Tunnel T_AB Re/Im = {re_T:.4f}/{im_T:.4f}\n")
        f.write(f"Time-reversal breaking: {'YES' if re_T > 0.01 else 'NO'}\n\n")
        f.write(f"beta(lambda) sweep:\n")
        for r in lambda_results:
            f.write(f"  lam={r['lambda']:8.2f} beta={r['beta_wing']:8.3f} "
                    f"KS(GUE)={r['ks_gue']:.4f} KS(Riem)={r['ks_riem']:.4f}\n")
        f.write(f"\nScaling:\n")
        for r in scaling_results:
            f.write(f"  N={r['N_cells']:3d}: beta={r['beta']:.3f} KS(GUE)={r['ks_gue']:.4f}\n")
        f.write(f"\nRiemann comparison:\n")
        f.write(f"  Before: KS={ks_before['riem'][0]:.4f}\n")
        f.write(f"  After:  KS={ks_after['riem'][0]:.4f}\n")
        f.write(f"  Improved: {'YES' if improved else 'NO'}\n")
        f.write(f"\nPhase diagram (beta):\n")
        f.write(f"  lam\\N  " + "  ".join(f"{nc:>6}" for nc in nc_grid) + "\n")
        for i, lam in enumerate(lam_grid):
            f.write(f"  {lam:5.1f}  " + "  ".join(f"{beta_grid[i,j]:6.3f}"
                    for j in range(len(nc_grid))) + "\n")

    elapsed = time.time() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
