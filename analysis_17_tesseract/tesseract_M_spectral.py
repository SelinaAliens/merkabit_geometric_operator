#!/usr/bin/env python3
"""
4-SPINOR TESSERACT M OPERATOR — SPECTRAL ANALYSIS
=====================================================

Breaking the Bipartite Barrier via the Cross Gate

The 2x2 Eisenstein M (radius 10, 367 nodes) showed:
  KS(GOE)=0.145 — best result but still GOE (beta~1)
  Bipartite block: resonance kills sub0<->sub1/sub2 edges
  +/-lambda eigenvalue pairing prevents RMT universality

The insight: Sub0 is the ROTATION AXIS (R-gate absent position).
At the 4x4 level (S^7 x S^7, two counter-rotating cubes): the cross gate
gate_cross_asym_4 activates inter-sector coupling. Sub0 is no longer
static — it participates in the upper<->lower mixing. The bipartite
partition is BROKEN by the cross coupling.

Prediction: M_tesseract on the Eisenstein lattice will show beta -> 2 (GUE)
because the cross-sector coupling breaks the bipartite +/-lambda symmetry.

Dimensional hierarchy being tested:
  2x2 (S^3 x S^3, E6, h=12): Sub0 static pivot -> GOE (beta~1)
  4x4 (S^7 x S^7, E7?, h=18?): Sub0 activates via cross gate -> GUE (beta=2)?
"""

import numpy as np
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings('ignore', message='.*Polyfit.*poorly conditioned.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\tesseract_M_spectral")
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
NUM_GATES = 5
COXETER_PHASE = 2 * np.pi / COXETER_H  # pi/6

# Eisenstein unit vectors
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

RADII = [1, 2, 3, 5, 7, 10]
CROSS_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


# ============================================================================
# EISENSTEIN CELL (from lattice_scaling_simulation.py)
# ============================================================================

class EisensteinCell:
    def __init__(self, radius):
        self.radius = radius
        self.r_sq = radius * radius
        self.nodes = []
        for a in range(-radius - 1, radius + 2):
            for b in range(-radius - 1, radius + 2):
                if a * a - a * b + b * b <= self.r_sq:
                    self.nodes.append((a, b))
        self.num_nodes = len(self.nodes)
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.edges = []
        self.neighbours = defaultdict(list)
        node_set = set(self.nodes)
        for i, (a1, b1) in enumerate(self.nodes):
            for da, db in UNIT_VECTORS_AB:
                nb = (a1 + da, b1 + db)
                if nb in node_set:
                    j = self.node_index[nb]
                    if j > i:
                        self.edges.append((i, j))
                    self.neighbours[i].append(j)
        self.is_interior = []
        self.interior_nodes = []
        self.boundary_nodes = []
        for i, (a, b) in enumerate(self.nodes):
            all_nbrs = all((a + da, b + db) in node_set for da, db in UNIT_VECTORS_AB)
            self.is_interior.append(all_nbrs)
            if all_nbrs:
                self.interior_nodes.append(i)
            else:
                self.boundary_nodes.append(i)
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = [0 if s == 0 else (+1 if s == 1 else -1) for s in self.sublattice]
        self.coordination = [len(self.neighbours[i]) for i in range(self.num_nodes)]


# ============================================================================
# STEP 1: 4-SPINOR ASSIGNMENT ON EISENSTEIN LATTICE
# ============================================================================

def assign_spinors_4spinor(cell):
    """
    Assign 4-component dual spinors to each Eisenstein lattice node.

    Upper sector: same as 2-spinor M construction.
    Lower sector: phase-shifted by pi/6 (one Coxeter step).
    Full 4-spinor: stack [upper, lower], normalise.

    The pi/6 phase offset encodes the dual pentachoric structure
    (pentachoron + its pi/6-rotated dual) within a single lattice site.
    """
    z_coords = [a + b * OMEGA_EISEN for (a, b) in cell.nodes]
    L = max(abs(z) for z in z_coords) if len(z_coords) > 1 else 1.0

    u4_list, v4_list, omega_list = [], [], []

    for i, (a, b) in enumerate(cell.nodes):
        r = abs(z_coords[i]) / (L + 1e-10)
        theta = np.pi * (a - b) / 6.0

        # Upper sector
        u_up = np.exp(1j * theta) * np.array(
            [np.cos(np.pi * r / 2), 1j * np.sin(np.pi * r / 2)], dtype=complex)
        u_up /= np.linalg.norm(u_up)
        v_up = np.array([-np.conj(u_up[1]), np.conj(u_up[0])], dtype=complex)

        # Lower sector: pi/6 Coxeter phase shift
        theta_lo = theta + COXETER_PHASE
        u_lo = np.exp(1j * theta_lo) * np.array(
            [np.cos(np.pi * r / 2), 1j * np.sin(np.pi * r / 2)], dtype=complex)
        u_lo /= np.linalg.norm(u_lo)
        v_lo = np.array([-np.conj(u_lo[1]), np.conj(u_lo[0])], dtype=complex)

        # Stack into 4-spinors
        u4 = np.concatenate([u_up, u_lo])
        v4 = np.concatenate([v_up, v_lo])
        u4 /= np.linalg.norm(u4)
        v4 /= np.linalg.norm(v4)

        u4_list.append(u4)
        v4_list.append(v4)
        omega_list.append(cell.chirality[i] * 1.0)

    return {'u': u4_list, 'v': v4_list, 'omega': omega_list}


# ============================================================================
# STEP 2: BUILD M_TESSERACT — SCALAR FORM (N x N)
# ============================================================================

def build_M_tesseract(cell, spinors4, xi=XI, cross_strength=0.3):
    """
    Build 4-spinor R/R-bar merger operator on Eisenstein lattice (scalar form).

    For each edge (i,j), the 4-spinor coupling has FOUR terms:
      1. Upper-Upper: <u_up_i | v_up_j>  (same as 2x2 M)
      2. Lower-Lower: <u_lo_i | v_lo_j>  (second pentachoric dual)
      3. Upper-Lower: <u_up_i | v_lo_j>  (CROSS coupling)
      4. Lower-Upper: <u_lo_i | v_up_j>  (CROSS coupling)

    M_scalar[i,j] = (1-cs) * (UU + LL)/2 + cs * (UL + LU)/2
    """
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)

    for (i, j) in cell.edges:
        u4i, v4j = spinors4['u'][i], spinors4['v'][j]

        omega_i = spinors4['omega'][i]
        omega_j = spinors4['omega'][j]
        resonance = np.exp(-(omega_i + omega_j) ** 2 / 0.1)

        # Sector decomposition
        u_up_i, u_lo_i = u4i[:2], u4i[2:]
        v_up_j, v_lo_j = v4j[:2], v4j[2:]

        # Intra-sector (diagonal blocks)
        uu = np.vdot(u_up_i, v_up_j)
        ll = np.vdot(u_lo_i, v_lo_j)

        # Cross-sector (off-diagonal blocks)
        ul = np.vdot(u_up_i, v_lo_j)
        lu = np.vdot(u_lo_i, v_up_j)

        intra = (uu + ll) / 2.0
        cross = (ul + lu) / 2.0

        coupling = decay * resonance * (
            (1 - cross_strength) * intra + cross_strength * cross
        )

        M[i, j] = coupling
        M[j, i] = np.conj(coupling)

    M = (M + M.conj().T) / 2.0
    return M


# ============================================================================
# BUILD M_TESSERACT — FULL 4N x 4N BLOCK FORM
# ============================================================================

def build_M_tesseract_full(cell, spinors4, xi=XI, cross_strength=0.3):
    """
    Full 4N x 4N M operator where each site has 4 spinor components.

    M_full[4i:4i+4, 4j:4j+4] = coupling_block(i, j)
    where coupling_block is a 4x4 matrix encoding intra + cross sector.
    """
    N = cell.num_nodes
    M = np.zeros((4 * N, 4 * N), dtype=complex)
    decay = np.exp(-1.0 / xi)

    for (i, j) in cell.edges:
        u4i, v4j = spinors4['u'][i], spinors4['v'][j]

        omega_i = spinors4['omega'][i]
        omega_j = spinors4['omega'][j]
        resonance = np.exp(-(omega_i + omega_j) ** 2 / 0.1)

        # Build the 4x4 coupling block
        # block[a, b] = decay * resonance * weight * <u_i[a] | v_j[b]>
        # where a, b index the 4 spinor components
        block = np.zeros((4, 4), dtype=complex)
        for a in range(4):
            for b in range(4):
                # Determine sector: 0,1 = upper; 2,3 = lower
                a_sector = 0 if a < 2 else 1
                b_sector = 0 if b < 2 else 1
                if a_sector == b_sector:
                    weight = (1 - cross_strength)
                else:
                    weight = cross_strength
                block[a, b] = decay * resonance * weight * np.conj(u4i[a]) * v4j[b]

        M[4*i:4*i+4, 4*j:4*j+4] = block
        M[4*j:4*j+4, 4*i:4*i+4] = block.conj().T

    M = (M + M.conj().T) / 2.0
    return M


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
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for idx, si in enumerate(s):
            result[idx], _ = quad(lambda x: a * x ** beta * np.exp(-b * x ** 2), 0, max(si, 0))
        return result
    return cdf


def wigner_pdf(s, beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    return a * s ** beta * np.exp(-b * s ** 2)


def fit_beta(spacings):
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
    try:
        hist, edges = np.histogram(small, bins=n_bins, density=True)
    except ValueError:
        return 0.0, 0.0
    centres = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0
    log_s = np.log(centres[mask])
    log_p = np.log(hist[mask])
    coeffs = np.polyfit(log_s, log_p, 1)
    beta_val = coeffs[0]
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta_val, r2


def ks_tests(spacings, riemann_spacings=None):
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


def analyse_spectrum(eigenvalues, riemann_spacings, use_wing=True):
    if use_wing:
        pos_eigs = eigenvalues[eigenvalues > 0]
        if len(pos_eigs) > 4:
            cutoff = np.percentile(pos_eigs, 20)
            wing = pos_eigs[pos_eigs > cutoff]
        else:
            wing = pos_eigs
    else:
        wing = eigenvalues
    if len(wing) < 5:
        return {'n_eigs': len(wing), 'beta': 0.0, 'r2': 0.0,
                'ks_goe': 1.0, 'p_goe': 0.0, 'ks_gue': 1.0, 'p_gue': 0.0,
                'ks_gse': 1.0, 'p_gse': 0.0,
                'ks_poi': 1.0, 'p_poi': 0.0, 'ks_riem': 1.0, 'p_riem': 0.0,
                'spacings': np.array([])}
    _, spacings = unfold_spectrum(wing)
    spacings = spacings[spacings > 0]
    if len(spacings) < 4:
        return {'n_eigs': len(wing), 'beta': 0.0, 'r2': 0.0,
                'ks_goe': 1.0, 'p_goe': 0.0, 'ks_gue': 1.0, 'p_gue': 0.0,
                'ks_gse': 1.0, 'p_gse': 0.0,
                'ks_poi': 1.0, 'p_poi': 0.0, 'ks_riem': 1.0, 'p_riem': 0.0,
                'spacings': spacings}
    beta_val, r2 = fit_beta(spacings)
    ks = ks_tests(spacings, riemann_spacings)
    return {
        'n_eigs': len(wing), 'beta': beta_val, 'r2': r2,
        'ks_goe': ks['goe'][0], 'p_goe': ks['goe'][1],
        'ks_gue': ks['gue'][0], 'p_gue': ks['gue'][1],
        'ks_gse': ks['gse'][0], 'p_gse': ks['gse'][1],
        'ks_poi': ks['poi'][0], 'p_poi': ks['poi'][1],
        'ks_riem': ks['riem'][0], 'p_riem': ks['riem'][1],
        'spacings': spacings
    }


# ============================================================================
# STEP 4: BIPARTITE BREAKING DIAGNOSTIC
# ============================================================================

def check_pm_pairing(eigenvalues, tol_rel=0.05):
    """What fraction of eigenvalues are still in +/-lambda pairs?"""
    pos = eigenvalues[eigenvalues > 1e-6]
    neg = -eigenvalues[eigenvalues < -1e-6]
    if len(pos) == 0:
        return 1.0
    paired = 0
    for lp in pos:
        if len(neg) > 0 and np.any(np.abs(neg - lp) / (lp + 1e-15) < tol_rel):
            paired += 1
    return paired / len(pos)


def check_im_re_ratio(M):
    """Im/Re ratio of off-diagonal elements."""
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    re_m = np.mean(np.abs(np.real(M[off])))
    im_m = np.mean(np.abs(np.imag(M[off])))
    return im_m / (re_m + 1e-15)


# ============================================================================
# STEP 8: COXETER PERIOD CHECK (4-spinor ouroboros)
# ============================================================================

def gate_P_4(u4, v4, theta):
    """Phase gate on 4-spinor: Pf tensor Pi (asymmetric)."""
    Pf = np.diag([np.exp(1j * theta), np.exp(-1j * theta),
                  np.exp(1j * theta), np.exp(-1j * theta)])
    Pi = np.diag([np.exp(-1j * theta), np.exp(1j * theta),
                  np.exp(-1j * theta), np.exp(1j * theta)])
    return Pf @ u4, Pi @ v4


def gate_Rz_4(u4, v4, theta):
    """Rz gate on each 2-spinor sector."""
    Rz = np.zeros((4, 4), dtype=complex)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    for sec in [0, 2]:
        Rz[sec, sec] = c - 1j * s
        Rz[sec + 1, sec + 1] = c + 1j * s
    return Rz @ u4, Rz.conj() @ v4


def gate_Rx_4(u4, v4, theta):
    """Rx gate on each 2-spinor sector."""
    Rx = np.zeros((4, 4), dtype=complex)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    for sec in [0, 2]:
        Rx[sec, sec] = c
        Rx[sec, sec + 1] = -1j * s
        Rx[sec + 1, sec] = -1j * s
        Rx[sec + 1, sec + 1] = c
    return Rx @ u4, Rx.conj() @ v4


def gate_cross_asym_4(u4, v4, theta):
    """Asymmetric cross gate: counter-rotating cubes (tesseract torsion)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    Cf = np.array([
        [c, 0, -s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, s, 0, c],
    ], dtype=complex)
    Ci = np.array([
        [c, 0, s, 0],
        [0, c, 0, s],
        [-s, 0, c, 0],
        [0, -s, 0, c],
    ], dtype=complex)
    return Cf @ u4, Ci @ v4


def ouroboros_step_4(u4, v4, step_index, cross_strength=0.3):
    """One ouroboros step for 4-spinor: P -> cross -> Rz -> Rx."""
    k = step_index
    theta = STEP_PHASE
    omega_k = 2 * np.pi * k / COXETER_H
    sym_base = theta / 3
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2 * np.pi / 3))
    cross_angle = cross_strength * theta * (1.0 + 0.5 * np.cos(omega_k + 4 * np.pi / 3))

    u, v = gate_P_4(u4, v4, theta)
    u, v = gate_cross_asym_4(u, v, cross_angle)
    u, v = gate_Rz_4(u, v, rz_angle)
    u, v = gate_Rx_4(u, v, rx_angle)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return u, v


def compute_berry_phase_4(trajectory_u):
    """Berry phase from sequence of u-spinors."""
    gamma = 0.0
    for k in range(len(trajectory_u) - 1):
        overlap = np.vdot(trajectory_u[k], trajectory_u[k + 1])
        gamma += np.angle(overlap)
    # Close the loop
    overlap = np.vdot(trajectory_u[-1], trajectory_u[0])
    gamma += np.angle(overlap)
    return gamma


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 76)
    print("  4-SPINOR TESSERACT M OPERATOR — SPECTRAL ANALYSIS")
    print("  Breaking the Bipartite Barrier via the Cross Gate")
    print("=" * 76)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Architecture: E6 (rank={RANK_E6}, dim={DIM_E6}, h={COXETER_H})")
    print(f"  Radii: {RADII}  |  Cross sweep: {CROSS_VALS}")
    print(f"  Node counts: ", end="")
    for r in RADII:
        print(f"{EisensteinCell(r).num_nodes}", end=" ")
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
    # STEP 3: SWEEP cross_strength FROM 0 TO 1
    # ====================================================================
    print(f"\n{'='*76}")
    print("  STEP 3: BETA vs CROSS_STRENGTH — THE KEY EXPERIMENT")
    print(f"{'='*76}")

    sweep_results = {}
    for radius in [3, 5, 7, 10]:
        cell = EisensteinCell(radius)
        spinors4 = assign_spinors_4spinor(cell)
        N = cell.num_nodes

        print(f"\n  Radius {radius} ({N} nodes, {len(cell.edges)} edges):")
        print(f"  {'cross':>6} | {'beta':>7} | {'KS_GOE':>8} {'p_GOE':>7} | "
              f"{'KS_GUE':>8} {'p_GUE':>7} | {'KS_GSE':>8} | "
              f"{'KS_Riem':>8} {'p_Riem':>7} | {'Im/Re':>6} | {'paired':>7}")
        print("  " + "-" * 100)

        radius_results = []
        for cs in CROSS_VALS:
            M = build_M_tesseract(cell, spinors4, cross_strength=cs)
            eigs = np.linalg.eigvalsh(M)

            # Bipartite pairing check
            paired = check_pm_pairing(eigs)

            # Im/Re ratio
            im_re = check_im_re_ratio(M)

            # Full analysis (positive wing)
            res = analyse_spectrum(eigs, pos_riem)
            res['cross'] = cs
            res['paired'] = paired
            res['im_re'] = im_re
            res['eigenvalues'] = eigs
            radius_results.append(res)

            print(f"  {cs:>6.2f} | {res['beta']:>7.3f} | "
                  f"{res['ks_goe']:>8.4f} {res['p_goe']:>7.4f} | "
                  f"{res['ks_gue']:>8.4f} {res['p_gue']:>7.4f} | "
                  f"{res['ks_gse']:>8.4f} | "
                  f"{res['ks_riem']:>8.4f} {res['p_riem']:>7.4f} | "
                  f"{im_re:>6.3f} | {paired:>6.1%}")

        sweep_results[radius] = radius_results

    # Find cross* where beta is maximized
    print(f"\n  CROSS* IDENTIFICATION:")
    cross_star_data = {}
    for radius in [3, 5, 7, 10]:
        rr = sweep_results[radius]
        betas = [r['beta'] for r in rr]
        best_idx = np.argmax(betas)
        cs_best = rr[best_idx]['cross']
        beta_best = betas[best_idx]
        cross_star_data[radius] = {'cross_star': cs_best, 'beta_star': beta_best}
        print(f"    r={radius}: cross*={cs_best:.2f}, beta*={beta_best:.3f}")

    # Architectural constant checks
    print(f"\n  Architectural constant candidates:")
    print(f"    1/sqrt(3) = {1/np.sqrt(3):.4f}  (triality)")
    print(f"    1/2       = 0.5000  (equal intra/cross)")
    print(f"    sin(pi/12) = {np.sin(np.pi/12):.4f}  (half Coxeter step)")
    print(f"    1/3       = 0.3333  (zero-point constant)")

    # ====================================================================
    # STEP 3b: FULL 4N x 4N BLOCK FORM AT RADIUS 5
    # ====================================================================
    print(f"\n{'='*76}")
    print("  STEP 3b: FULL 4N x 4N BLOCK FORM (radius 5)")
    print(f"{'='*76}\n")

    cell5 = EisensteinCell(5)
    spinors4_5 = assign_spinors_4spinor(cell5)
    N5 = cell5.num_nodes

    print(f"  Matrix size: {4*N5} x {4*N5} ({4*N5} eigenvalues)")
    print(f"  {'cross':>6} | {'beta':>7} | {'KS_GOE':>8} {'p_GOE':>7} | "
          f"{'KS_GUE':>8} {'p_GUE':>7} | {'KS_GSE':>8} | "
          f"{'KS_Riem':>8} {'p_Riem':>7}")
    print("  " + "-" * 80)

    full_results = {}
    for cs in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        M_full = build_M_tesseract_full(cell5, spinors4_5, cross_strength=cs)
        eigs_full = np.linalg.eigvalsh(M_full)
        res_full = analyse_spectrum(eigs_full, pos_riem)

        full_results[cs] = res_full
        print(f"  {cs:>6.2f} | {res_full['beta']:>7.3f} | "
              f"{res_full['ks_goe']:>8.4f} {res_full['p_goe']:>7.4f} | "
              f"{res_full['ks_gue']:>8.4f} {res_full['p_gue']:>7.4f} | "
              f"{res_full['ks_gse']:>8.4f} | "
              f"{res_full['ks_riem']:>8.4f} {res_full['p_riem']:>7.4f}")

    # Also radius 7 full form
    print(f"\n  Full 4N x 4N at radius 7:")
    cell7 = EisensteinCell(7)
    spinors4_7 = assign_spinors_4spinor(cell7)
    N7 = cell7.num_nodes
    print(f"  Matrix size: {4*N7} x {4*N7}")

    full_results_r7 = {}
    for cs in [0.0, 0.3, 0.5, 1.0]:
        M_full7 = build_M_tesseract_full(cell7, spinors4_7, cross_strength=cs)
        eigs_full7 = np.linalg.eigvalsh(M_full7)
        res7 = analyse_spectrum(eigs_full7, pos_riem)
        full_results_r7[cs] = res7
        print(f"  cs={cs:.1f}: beta={res7['beta']:.3f}, KS(GUE)={res7['ks_gue']:.4f} "
              f"(p={res7['p_gue']:.4f}), KS(Riem)={res7['ks_riem']:.4f}")

    # ====================================================================
    # STEP 5: F-CONNECTION AT TESSERACT LEVEL
    # ====================================================================
    print(f"\n{'='*76}")
    print("  STEP 5: F-CONNECTION AT TESSERACT LEVEL")
    print(f"{'='*76}\n")

    ln_F = -np.log(F_RETURN)
    print(f"  -ln(F) = {ln_F:.6f}")
    print(f"  36/100 = 0.360000\n")

    print(f"  {'Radius':>6} {'cross':>6} {'N':>5} {'N_pos':>5} {'frac':>8} {'|d(lnF)|':>10}")
    print("  " + "-" * 48)

    f_data = {}
    for radius in RADII:
        cell = EisensteinCell(radius)
        spinors4 = assign_spinors_4spinor(cell)
        f_data[radius] = {}

        for cs in [0.0, 0.3, 1.0]:
            M = build_M_tesseract(cell, spinors4, cross_strength=cs)
            eigs = np.linalg.eigvalsh(M)
            n_pos = np.sum(eigs > 1e-8)
            frac = n_pos / len(eigs)
            delta = abs(frac - ln_F)
            f_data[radius][cs] = {'n_pos': n_pos, 'N': len(eigs), 'frac': frac, 'delta': delta}
            print(f"  {radius:6d} {cs:6.1f} {len(eigs):5d} {n_pos:5d} {frac:8.5f} {delta:10.6f}")

    # ====================================================================
    # STEP 8: COXETER PERIOD CHECK
    # ====================================================================
    print(f"\n{'='*76}")
    print("  STEP 8: COXETER PERIOD CHECK (4-spinor ouroboros)")
    print(f"{'='*76}\n")

    # Initial state: |0> tesseract (u perp v in C^4)
    u0 = np.array([1, 0, 0, 0], dtype=complex)
    v0 = np.array([0, 0, 0, 1], dtype=complex)

    print(f"  {'Period':>6} {'cs':>4} {'F_return':>10} {'Berry/pi':>10}")
    print("  " + "-" * 35)

    coxeter_results = []
    for cs in [0.0, 0.3, 0.5, 1.0]:
        for period in [12, 18, 24, 30]:
            u, v = u0.copy(), v0.copy()
            traj_u = [u.copy()]
            for step in range(period):
                u, v = ouroboros_step_4(u, v, step, cross_strength=cs)
                traj_u.append(u.copy())

            # Return fidelity
            F = abs(np.vdot(
                np.concatenate([u0, v0]),
                np.concatenate([u, v])
            )) ** 2

            # Berry phase
            gamma = compute_berry_phase_4(traj_u[:-1])

            coxeter_results.append({'cs': cs, 'period': period, 'F': F, 'gamma': gamma})
            print(f"  {period:6d} {cs:4.1f} {F:10.6f} {gamma/np.pi:10.6f}")

    # ====================================================================
    # FIGURES
    # ====================================================================
    print(f"\n  Generating figures...")
    s_range = np.linspace(0.001, 4.0, 200)

    # --- Figure 1: beta vs cross_strength (THE KEY PLOT) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for radius in [3, 5, 7, 10]:
        rr = sweep_results[radius]
        cs_arr = [r['cross'] for r in rr]
        b_arr = [r['beta'] for r in rr]
        ax.plot(cs_arr, b_arr, 'o-', lw=2, markersize=6,
                label=f'r={radius} (N={rr[0]["n_eigs"]}+)')
    ax.axhline(y=1, color='blue', ls='--', alpha=0.4, label='GOE (beta=1)')
    ax.axhline(y=2, color='red', ls='--', alpha=0.4, label='GUE (beta=2)')
    ax.axhline(y=4, color='purple', ls=':', alpha=0.3, label='GSE (beta=4)')
    ax.set_xlabel('Cross Strength', fontsize=13)
    ax.set_ylabel('Level Repulsion beta', fontsize=13)
    ax.set_title('Scalar M: beta vs Cross Coupling', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for radius in [3, 5, 7, 10]:
        rr = sweep_results[radius]
        cs_arr = [r['cross'] for r in rr]
        paired_arr = [r['paired'] for r in rr]
        ax.plot(cs_arr, paired_arr, 's-', lw=2, markersize=6,
                label=f'r={radius}')
    ax.set_xlabel('Cross Strength', fontsize=13)
    ax.set_ylabel('Fraction Still Paired (+/-lambda)', fontsize=13)
    ax.set_title('Bipartite Dissolution', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('4-Spinor Tesseract M: Cross Gate Breaks Bipartite Structure',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure1_beta_vs_cross.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    figure1_beta_vs_cross.png saved")

    # --- Figure 2: Pairing dissolution ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for radius in [3, 5, 7, 10]:
        rr = sweep_results[radius]
        cs_arr = [r['cross'] for r in rr]
        ks_gue = [r['ks_gue'] for r in rr]
        ks_goe = [r['ks_goe'] for r in rr]
        ax.plot(cs_arr, ks_gue, 'o-', lw=2, label=f'KS(GUE) r={radius}')
        ax.plot(cs_arr, ks_goe, 's--', lw=1, alpha=0.5, label=f'KS(GOE) r={radius}')
    ax.set_xlabel('Cross Strength', fontsize=13)
    ax.set_ylabel('KS Distance', fontsize=13)
    ax.set_title('GOE/GUE Distance vs Cross Coupling', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure2_pairing_dissolution.png', dpi=150)
    plt.close(fig)
    print("    figure2_pairing_dissolution.png saved")

    # --- Figure 3: P(s) at best cross for r=10 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scalar form r=10 at cross*
    best10 = sweep_results[10]
    betas10 = [r['beta'] for r in best10]
    best_idx = np.argmax(betas10)
    best_r10 = best10[best_idx]
    sp_best = best_r10['spacings']

    ax = axes[0]
    if len(sp_best) > 3:
        nb = max(5, min(25, len(sp_best) // 4))
        try:
            ax.hist(sp_best, bins=nb, density=True, alpha=0.5,
                    color='coral', edgecolor='darkred', label='Data')
        except ValueError:
            ax.hist(sp_best, bins='auto', density=True, alpha=0.5,
                    color='coral', edgecolor='darkred', label='Data')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.plot(s_range, [wigner_pdf(s, 4) for s in s_range], 'm:', lw=2, label='GSE')
    ax.hist(pos_riem[:300], bins=30, density=True, alpha=0.2, color='purple', label='Riemann')
    ax.set_title(f'Scalar r=10, cross={best_r10["cross"]:.2f}\n'
                 f'beta={best_r10["beta"]:.3f}', fontsize=12)
    ax.set_xlabel('s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Full form r=5 at best cross
    best_full_beta = max(full_results.values(), key=lambda x: x['beta'])
    best_cs_full = [cs for cs, v in full_results.items() if v is best_full_beta][0]
    sp_full = best_full_beta['spacings']

    ax = axes[1]
    if len(sp_full) > 3:
        nb = max(5, min(30, len(sp_full) // 4))
        try:
            ax.hist(sp_full, bins=nb, density=True, alpha=0.5,
                    color='steelblue', edgecolor='navy', label='Data')
        except ValueError:
            ax.hist(sp_full, bins='auto', density=True, alpha=0.5,
                    color='steelblue', edgecolor='navy', label='Data')
    ax.plot(s_range, [wigner_pdf(s, 1) for s in s_range], 'b--', lw=2, label='GOE')
    ax.plot(s_range, [wigner_pdf(s, 2) for s in s_range], 'r-', lw=2, label='GUE')
    ax.plot(s_range, [wigner_pdf(s, 4) for s in s_range], 'm:', lw=2, label='GSE')
    ax.hist(pos_riem[:300], bins=30, density=True, alpha=0.2, color='purple', label='Riemann')
    ax.set_title(f'Full 4N r=5, cross={best_cs_full:.2f}\n'
                 f'beta={best_full_beta["beta"]:.3f} ({4*N5} evals)',
                 fontsize=12)
    ax.set_xlabel('s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Spacing Distributions at Optimal Cross Coupling', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure3_spacing_at_crossstar.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    figure3_spacing_at_crossstar.png saved")

    # --- Figure 4: F-connection 4-spinor ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for cs in [0.0, 0.3, 1.0]:
        radii_arr = RADII
        fracs = [f_data[r][cs]['frac'] for r in radii_arr]
        Ns_arr = [f_data[r][cs]['N'] for r in radii_arr]
        ax.plot(Ns_arr, fracs, 'o-', lw=2, markersize=8, label=f'cross={cs:.1f}')
    ax.axhline(y=ln_F, color='red', ls='--', lw=2, label=f'-ln(F)={ln_F:.4f}')
    ax.axhline(y=0.36, color='gray', ls=':', alpha=0.5, label='36/100')
    ax.set_xlabel('N (nodes)', fontsize=14)
    ax.set_ylabel('Positive Eigenvalue Fraction', fontsize=14)
    ax.set_title('F-Connection: pos_frac vs Lattice Size (4-Spinor)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure4_f_connection_4spinor.png', dpi=150)
    plt.close(fig)
    print("    figure4_f_connection_4spinor.png saved")

    # --- Figure 5: Coxeter period check ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    periods = [12, 18, 24, 30]

    for cs in [0.0, 0.3, 0.5, 1.0]:
        Fs = [r['F'] for r in coxeter_results if r['cs'] == cs]
        gammas = [r['gamma'] / np.pi for r in coxeter_results if r['cs'] == cs]
        axes[0].plot(periods, Fs, 'o-', lw=2, label=f'cs={cs:.1f}')
        axes[1].plot(periods, gammas, 's-', lw=2, label=f'cs={cs:.1f}')

    axes[0].set_xlabel('Period T', fontsize=13)
    axes[0].set_ylabel('Return Fidelity F', fontsize=13)
    axes[0].set_title('F(T) — Which Period Closes?', fontsize=13)
    axes[0].axhline(y=F_RETURN, color='red', ls='--', alpha=0.5, label=f'F_2spinor={F_RETURN:.4f}')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Period T', fontsize=13)
    axes[1].set_ylabel('Berry Phase / pi', fontsize=13)
    axes[1].set_title('Berry Phase vs Period', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Coxeter Period Check: E6(12) vs E7(18) vs E8(30)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure5_coxeter_period.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    figure5_coxeter_period.png saved")

    # ====================================================================
    # FULL REPORT
    # ====================================================================
    print(f"\n{'='*76}")
    print("  FULL REPORT")
    print(f"{'='*76}")

    print(f"\n  1. BETA vs CROSS (scalar, positive wing):")
    for radius in [5, 7, 10]:
        rr = sweep_results[radius]
        best = max(rr, key=lambda x: x['beta'])
        print(f"     r={radius}: best beta={best['beta']:.3f} at cross={best['cross']:.2f}")

    print(f"\n  2. FULL 4N FORM (r=5, {4*N5} evals):")
    for cs, res in sorted(full_results.items()):
        print(f"     cs={cs:.1f}: beta={res['beta']:.3f}, KS(GUE)={res['ks_gue']:.4f}")
    best_full = max(full_results.items(), key=lambda x: x[1]['beta'])
    print(f"     BEST: cs={best_full[0]:.1f}, beta={best_full[1]['beta']:.3f}")

    print(f"\n  3. F-CONNECTION:")
    for cs in [0.0, 0.3, 1.0]:
        d10 = f_data[10][cs]
        print(f"     cs={cs:.1f}, r=10: pos_frac={d10['frac']:.5f}, "
              f"|delta|={d10['delta']:.6f}")

    print(f"\n  4. COXETER PERIOD:")
    for cs in [0.0, 0.3]:
        print(f"     cs={cs:.1f}:")
        for r in coxeter_results:
            if r['cs'] == cs:
                print(f"       T={r['period']:2d}: F={r['F']:.6f}")

    # Decision matrix
    all_betas_sweep = []
    for radius in [3, 5, 7, 10]:
        for r in sweep_results[radius]:
            all_betas_sweep.append(r['beta'])
    all_betas_full = [v['beta'] for v in full_results.values()]
    all_betas_full7 = [v['beta'] for v in full_results_r7.values()]
    beta_max = max(all_betas_sweep + all_betas_full + all_betas_full7)

    print(f"\n  5. DECISION MATRIX:")
    print(f"     Max beta across all: {beta_max:.3f}")
    if beta_max >= 1.8:
        print(f"     ** GUE CONFIRMED: beta >= 1.8 — cross gate breaks bipartite!")
    elif beta_max >= 1.3:
        print(f"     ** TRANSITIONAL: beta={beta_max:.2f}, GOE->GUE crossover in progress")
    elif beta_max >= 0.7:
        print(f"     ** GOE-LIKE: beta ~ 1, bipartite not fully broken at this level")
    elif beta_max >= 3.5:
        print(f"     ** GSE: beta >= 4, quaternionic structure, symplectic L-functions!")
    else:
        print(f"     ** INCONCLUSIVE: beta={beta_max:.2f}")

    # Save FULL_REPORT.txt
    with open(RESULTS_DIR / "FULL_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write("4-SPINOR TESSERACT M OPERATOR: FULL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write("Beta vs cross_strength (scalar form, positive wing):\n")
        for radius in [3, 5, 7, 10]:
            f.write(f"\n  Radius {radius}:\n")
            for r in sweep_results[radius]:
                f.write(f"    cs={r['cross']:.2f}: beta={r['beta']:.4f}, "
                        f"KS(GUE)={r['ks_gue']:.4f}, paired={r['paired']:.3f}\n")

        f.write(f"\nFull 4N form (r=5, {4*N5} evals):\n")
        for cs, res in sorted(full_results.items()):
            f.write(f"  cs={cs:.1f}: beta={res['beta']:.4f}, "
                    f"KS(GUE)={res['ks_gue']:.4f}, KS(Riem)={res['ks_riem']:.4f}\n")

        f.write(f"\nFull 4N form (r=7, {4*N7} evals):\n")
        for cs, res in sorted(full_results_r7.items()):
            f.write(f"  cs={cs:.1f}: beta={res['beta']:.4f}, "
                    f"KS(GUE)={res['ks_gue']:.4f}\n")

        f.write(f"\nF-connection:\n")
        for radius in RADII:
            for cs in [0.0, 0.3, 1.0]:
                d = f_data[radius][cs]
                f.write(f"  r={radius}, cs={cs:.1f}: frac={d['frac']:.5f}, "
                        f"|delta|={d['delta']:.6f}\n")

        f.write(f"\nCoxeter period:\n")
        for r in coxeter_results:
            f.write(f"  cs={r['cs']:.1f}, T={r['period']:2d}: "
                    f"F={r['F']:.6f}, gamma/pi={r['gamma']/np.pi:.6f}\n")

        f.write(f"\nMax beta: {beta_max:.4f}\n")

    # Save data arrays
    for radius in [3, 5, 7, 10]:
        for r in sweep_results[radius]:
            np.save(RESULTS_DIR / f"eigs_r{radius}_cs{r['cross']:.2f}.npy",
                    r['eigenvalues'])

    with open(RESULTS_DIR / "beta_vs_cross.txt", 'w') as f:
        f.write("radius  cross  beta  ks_goe  ks_gue  ks_gse  ks_riem  paired  im_re\n")
        for radius in [3, 5, 7, 10]:
            for r in sweep_results[radius]:
                f.write(f"{radius}  {r['cross']:.2f}  {r['beta']:.4f}  "
                        f"{r['ks_goe']:.5f}  {r['ks_gue']:.5f}  {r['ks_gse']:.5f}  "
                        f"{r['ks_riem']:.5f}  {r['paired']:.4f}  {r['im_re']:.4f}\n")

    elapsed = time.time() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
