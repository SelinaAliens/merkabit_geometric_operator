"""
Path B: Peierls Flux Sweep on the Eisenstein Torus
===================================================

Thread uniform magnetic flux Phi through each hexagonal plaquette of the
Eisenstein torus via Peierls phases, breaking time-reversal symmetry.

Prediction: GOE -> GUE transition at Phi* = pi/6 = 2*pi/h(E6).

The P gate is the time-reversal-odd element of the pentachoric architecture.
Its phase advance per Floquet step is pi/6 = 2*pi/h(E6).
Threading this exact flux per hexagonal plaquette inserts one P gate step
into each hopping term of M.

Steps:
  1. Build M with Peierls phase on EisensteinTorus
  2. Coarse Phi sweep (0 -> 1)
  3. Fine sweep around Phi = 1/6
  4. Full RMT comparison at Phi*
  5. F-connection under flux
  6. Hofstadter butterfly
  7. Torus size scaling (L=6,9,12,15,18)
"""

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

# ── Constants ──────────────────────────────────────────────────────────
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H   # pi/6
XI = 3.0
F_RETURN = 0.696778
NEG_LN_F = -np.log(F_RETURN)          # 0.36129...
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

OUT = "C:/Users/selin/merkabit_results/peierls_flux"
os.makedirs(OUT, exist_ok=True)

# ── EisensteinTorus Class ─────────────────────────────────────────────
class EisensteinTorus:
    """
    Periodic Eisenstein lattice (flat torus).
    Node (a,b) identified with (a mod L, b mod L). L^2 nodes, all coord 6.
    Sublattice: (a+b) mod 3 -> chirality {0, +1, -1}.
    """
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

        self.is_interior = [True] * self.num_nodes
        self.interior_nodes = list(range(self.num_nodes))
        self.boundary_nodes = []

        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            if s == 0: self.chirality.append(0)
            elif s == 1: self.chirality.append(+1)
            else: self.chirality.append(-1)

        self.coordination = [len(self.neighbours[i]) for i in range(self.num_nodes)]


# ── Spinor Assignment ─────────────────────────────────────────────────
def assign_spinors_torus(cell):
    """Assign dual spinors on torus. Same construction as EisensteinCell."""
    u_list, v_list, omega_list = [], [], []
    z_coords = [a + b * OMEGA_EISEN for (a, b) in cell.nodes]
    L = max(abs(z) for z in z_coords) if len(z_coords) > 1 else 1.0

    for i, (a, b) in enumerate(cell.nodes):
        z = z_coords[i]
        r = abs(z) / (L + 1e-10)
        theta = np.pi * (a - b) / 6.0

        u = np.exp(1j * theta) * np.array([np.cos(np.pi * r / 2),
                                            1j * np.sin(np.pi * r / 2)],
                                           dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)

        omega = cell.chirality[i] * 1.0
        u_list.append(u)
        v_list.append(v)
        omega_list.append(omega)

    return {'u': u_list, 'v': v_list, 'omega': omega_list}


# ── Peierls Phase (Landau Gauge) ──────────────────────────────────────
def peierls_phase(a_i, b_i, a_j, b_j, phi):
    """
    Peierls phase for hopping i -> j on Eisenstein lattice.

    Landau gauge: A_ij = phi * (a_i + a_j)/2 * (b_j - b_i)

    This ensures the sum around any elementary hexagonal plaquette
    equals exactly phi (the flux quantum per plaquette).

    phi = 1/6 corresponds to Phi = pi/6 = 2*pi/h(E6) = one P gate step.
    """
    return phi * (a_i + a_j) / 2.0 * (b_j - b_i)


def verify_plaquette_flux(cell, phi):
    """
    Verify Peierls flux through elementary plaquettes.

    On the Eisenstein lattice, elementary plaquettes are TRIANGLES (not hexagons).
    Two types:
      Type A (upward): (a,b) -> (a+1,b) -> (a+1,b+1) -> (a,b)
        displacements: (1,0), (0,1), (-1,-1)
      Type B (downward): (a,b) -> (a+1,b+1) -> (a,b+1) -> (a,b)
        displacements: (1,1), (-1,0), (0,-1)

    Each triangle should have gauge flux = phi/2.
    Physical flux per triangle = 2*pi * phi/2 = pi*phi.
    At phi = 1/6: flux = pi/6 (one P gate step).

    A hexagon around a central node contains 6 triangles -> flux 3*phi.
    """
    L = cell.L
    results = {'triangleA': [], 'triangleB': [], 'hexagon': []}

    # Type A triangles at several positions
    for a0, b0 in [(0, 0), (1, 0), (0, 1), (3, 2), (5, 4)]:
        # (a,b) -> (a+1,b) -> (a+1,b+1) -> (a,b)
        p1 = peierls_phase(a0, b0, (a0+1)%L, b0, phi)
        p2 = peierls_phase((a0+1)%L, b0, (a0+1)%L, (b0+1)%L, phi)
        p3 = peierls_phase((a0+1)%L, (b0+1)%L, a0, b0, phi)
        results['triangleA'].append(p1 + p2 + p3)

    # Type B triangles
    for a0, b0 in [(0, 0), (1, 0), (0, 1), (3, 2), (5, 4)]:
        # (a,b) -> (a+1,b+1) -> (a,b+1) -> (a,b)
        p1 = peierls_phase(a0, b0, (a0+1)%L, (b0+1)%L, phi)
        p2 = peierls_phase((a0+1)%L, (b0+1)%L, a0, (b0+1)%L, phi)
        p3 = peierls_phase(a0, (b0+1)%L, a0, b0, phi)
        results['triangleB'].append(p1 + p2 + p3)

    # Hexagonal plaquette around (0,0): 6 nearest neighbors in cyclic order
    # (1,0), (1,1), (0,1), (-1,0), (-1,-1), (0,-1)
    hex_verts = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]
    hex_phase = 0.0
    for k in range(6):
        a1, b1 = hex_verts[k]
        a2, b2 = hex_verts[(k + 1) % 6]
        # Use modular coords on torus
        hex_phase += peierls_phase((a1 % L), (b1 % L), (a2 % L), (b2 % L), phi)
    results['hexagon'].append(hex_phase)

    return results


# ── Build M with Peierls Flux ─────────────────────────────────────────
def build_M_peierls(cell, spinors, phi, xi=XI, use_resonance=True):
    """
    Build M operator on Eisenstein torus with Peierls flux.

    M_ij = exp(-1/xi) * resonance(omega_i, omega_j) * <u_i|v_j> * exp(i * A_ij)

    phi = flux per plaquette in units of 2*pi.
    phi = 1/6 -> Phi = pi/6 (one P gate step per hop).

    When phi = 0, this reduces to the standard M operator.
    When phi != 0, time-reversal symmetry is broken: M is still Hermitian
    but M != M* (not real-symmetric), which should drive GOE -> GUE.
    """
    N = cell.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = cell.L

    for (i, j) in cell.edges:
        a_i, b_i = cell.nodes[i]
        a_j, b_j = cell.nodes[j]

        u_i = spinors['u'][i]
        v_j = spinors['v'][j]

        omega_i = spinors['omega'][i]
        omega_j = spinors['omega'][j]

        if use_resonance:
            resonance = np.exp(-(omega_i + omega_j) ** 2 / 0.1)
        else:
            resonance = 1.0

        # Cross-chirality coupling with Peierls phase
        # Need to handle torus wrapping for gauge consistency
        # Use minimal image convention for phase computation
        da = a_j - a_i
        db = b_j - b_i
        # Wrap to nearest image
        if da > L // 2: da -= L
        if da < -(L // 2): da += L
        if db > L // 2: db -= L
        if db < -(L // 2): db += L

        # Peierls phase using actual displacement (not wrapped coords)
        A_ij = phi * (2 * a_i + da) / 2.0 * db

        coupling_ij = decay * resonance * np.vdot(u_i, v_j) * np.exp(2j * np.pi * A_ij)

        M[i, j] = coupling_ij
        M[j, i] = np.conj(coupling_ij)

    M = (M + M.conj().T) / 2.0
    return M


# ── Spectral Analysis Tools ──────────────────────────────────────────
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


def extract_positive_wing(eigenvalues, threshold_pct=20):
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    return pos_eigs[pos_eigs > cutoff]


def wigner_pdf(s, beta):
    a = 2.0 * (gamma_fn((beta + 2) / 2)) ** (beta + 1) / (gamma_fn((beta + 1) / 2)) ** (beta + 2)
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2)) ** 2
    return a * s ** beta * np.exp(-b * s ** 2)


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
    beta = coeffs[0]
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def ks_tests(spacings):
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {k: (1.0, 0.0) for k in ['goe', 'gue', 'gse', 'poi']}
    results = {}
    for name, beta_val in [('goe', 1), ('gue', 2), ('gse', 4)]:
        cdf = make_wigner_cdf(beta_val)
        ks, p = stats.kstest(pos, cdf)
        results[name] = (ks, p)
    ks, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (ks, p)
    return results


def spectrum_analysis(evals, label=""):
    """Full spectral analysis: unfold, fit beta, KS tests, positive fraction."""
    evals = np.sort(np.real(evals))
    n_pos = np.sum(evals > 0)
    n_neg = np.sum(evals < 0)
    n_zero = np.sum(np.abs(evals) < 1e-10)
    pos_frac = n_pos / len(evals) if len(evals) > 0 else 0

    # Full spectrum
    _, spacings_full = unfold_spectrum(evals)
    beta_full, r2_full = fit_beta(spacings_full) if len(spacings_full) > 10 else (0, 0)
    ks_full = ks_tests(spacings_full) if len(spacings_full) > 10 else {}

    # Positive wing
    wing = extract_positive_wing(evals)
    if len(wing) > 10:
        _, spacings_wing = unfold_spectrum(wing)
        beta_wing, r2_wing = fit_beta(spacings_wing)
        ks_wing = ks_tests(spacings_wing)
    else:
        spacings_wing = np.array([])
        beta_wing, r2_wing = 0, 0
        ks_wing = {}

    return {
        'label': label,
        'n_evals': len(evals),
        'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
        'pos_frac': pos_frac,
        'beta_full': beta_full, 'r2_full': r2_full,
        'ks_full': ks_full,
        'beta_wing': beta_wing, 'r2_wing': r2_wing,
        'ks_wing': ks_wing,
        'spacings_full': spacings_full,
        'spacings_wing': spacings_wing,
        'evals': evals,
    }


# ── Step 1: Build & Verify ───────────────────────────────────────────
def step1_build_and_verify(L=9):
    """Build EisensteinTorus, assign spinors, verify plaquette flux."""
    print("=" * 60)
    print("STEP 1: Build Eisenstein Torus with Peierls Flux")
    print("=" * 60)

    cell = EisensteinTorus(L)
    spinors = assign_spinors_torus(cell)

    print(f"  Torus T_{L}: {cell.num_nodes} nodes, {len(cell.edges)} edges")
    print(f"  All coordination 6: {set(cell.coordination) == {6}}")
    chi_counts = Counter(cell.chirality)
    print(f"  Chirality distribution: {dict(chi_counts)}")

    # Verify plaquette flux at phi = 1/6
    phi_test = 1.0 / 6
    pf = verify_plaquette_flux(cell, phi_test)
    print(f"\n  Plaquette flux verification (phi=1/6):")
    print(f"    Expected gauge flux per triangle: phi/2 = {phi_test/2:.6f}")
    print(f"    Physical flux per triangle: pi*phi = {np.pi*phi_test:.6f}")
    print(f"    Type A triangles: {[f'{x:.6f}' for x in pf['triangleA']]}")
    print(f"    Type B triangles: {[f'{x:.6f}' for x in pf['triangleB']]}")
    print(f"    Hexagon (6 triangles): {[f'{x:.6f}' for x in pf['hexagon']]} (expect 3*phi={3*phi_test:.6f})")

    # Time-reversal symmetry analysis
    # T-invariance for Hermitian matrix: M = M^T (transpose, not conjugate)
    # M real symmetric -> GOE (T-invariant, T^2 = +1)
    # M complex Hermitian but M != M^T -> GUE (T broken)
    print(f"\n  Time-reversal analysis (T: M = M^T):")

    for phi_val, label in [(0.0, "phi=0"), (phi_test, "phi=1/6")]:
        for use_res, res_label in [(True, "with res"), (False, "no res")]:
            M = build_M_peierls(cell, spinors, phi_val, use_resonance=use_res)
            # T-symmetry: M = M^T
            tr = np.max(np.abs(M - M.T))
            # Hermiticity: M = M^dag
            herm = np.max(np.abs(M - M.conj().T))
            # Fraction of imaginary content
            im_frac = np.sum(np.abs(np.imag(M))) / (np.sum(np.abs(M)) + 1e-30)
            T_status = "T-inv" if tr < 1e-10 else "T-broken"
            print(f"    {label} ({res_label}): max|M-M^T|={tr:.2e} ({T_status}), "
                  f"Herm={herm<1e-12}, Im_frac={im_frac:.4f}")

    return cell, spinors


# ── Step 2: Coarse Phi Sweep ──────────────────────────────────────────
def step2_coarse_sweep(cell, spinors):
    """Sweep phi from 0 to 1, track beta and KS statistics."""
    print("\n" + "=" * 60)
    print("STEP 2: Coarse Flux Sweep (phi = 0 to 1)")
    print("=" * 60)

    phi_values = [0.0, 1/24, 1/12, 1/8, 1/6, 5/24, 1/4, 1/3, 5/12, 1/2, 2/3, 5/6, 1.0]
    results_res = []
    results_nores = []

    print("  --- With resonance ---")
    for phi in phi_values:
        M = build_M_peierls(cell, spinors, phi, use_resonance=True)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals, label=f"res phi={phi:.4f}")
        results_res.append((phi, sa))

        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        print(f"  phi={phi:7.4f}: beta={sa['beta_wing']:+7.4f}, "
              f"KS(GOE)={ks_goe[0]:.4f} p={ks_goe[1]:.3f}, "
              f"KS(GUE)={ks_gue[0]:.4f} p={ks_gue[1]:.3f}, "
              f"pf={sa['pos_frac']:.4f}")

    print("\n  --- Without resonance (all edges couple) ---")
    for phi in phi_values:
        M = build_M_peierls(cell, spinors, phi, use_resonance=False)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals, label=f"nores phi={phi:.4f}")
        results_nores.append((phi, sa))

        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        favor = "GOE" if ks_goe[0] < ks_gue[0] else "GUE"
        print(f"  phi={phi:7.4f}: beta={sa['beta_wing']:+7.4f}, "
              f"KS(GOE)={ks_goe[0]:.4f}, KS(GUE)={ks_gue[0]:.4f} -> {favor}, "
              f"pf={sa['pos_frac']:.4f}")

    return results_res, results_nores


# ── Step 3: Fine Sweep around Phi = 1/6 ──────────────────────────────
def step3_fine_sweep(cell, spinors):
    """Fine sweep around phi = 1/6 to locate the transition."""
    print("\n" + "=" * 60)
    print("STEP 3: Fine Sweep around phi = 1/6")
    print("=" * 60)

    phi_fine = np.linspace(0.10, 0.25, 31)
    results_res = []
    results_nores = []

    print("  --- With resonance ---")
    print(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}  {'pf':>6s}")
    for phi in phi_fine:
        M = build_M_peierls(cell, spinors, phi, use_resonance=True)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals)
        results_res.append((phi, sa))
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        marker = " <--" if abs(phi - 1/6) < 0.003 else ""
        print(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}  "
              f"{sa['pos_frac']:6.4f}{marker}")

    print("\n  --- Without resonance ---")
    print(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}  {'favor':>5s}")
    for phi in phi_fine:
        M = build_M_peierls(cell, spinors, phi, use_resonance=False)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals)
        results_nores.append((phi, sa))
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        favor = "GOE" if ks_goe[0] < ks_gue[0] else "GUE"
        marker = " <--" if abs(phi - 1/6) < 0.003 else ""
        print(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}  "
              f"{favor:>5s}{marker}")

    return results_res, results_nores


# ── Step 4: Full RMT at Phi* ──────────────────────────────────────────
def step4_full_rmt_at_phi_star(cell, spinors):
    """Detailed RMT analysis at phi = 0 and phi = 1/6."""
    print("\n" + "=" * 60)
    print("STEP 4: Full RMT Comparison at Phi* = 1/6")
    print("=" * 60)

    comparisons = {}
    for phi, label in [(0.0, "phi=0 (no flux)"), (1.0/6, "phi=1/6 (P gate)")]:
        M = build_M_peierls(cell, spinors, phi)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals, label=label)
        comparisons[label] = sa

        print(f"\n  {label}:")
        print(f"    N_evals = {sa['n_evals']}, pos_frac = {sa['pos_frac']:.4f}")
        print(f"    Full spectrum: beta = {sa['beta_full']:+.4f}")
        print(f"    Positive wing: beta = {sa['beta_wing']:+.4f}")

        for wing_name, ks_dict in [("full", sa['ks_full']), ("wing", sa['ks_wing'])]:
            if ks_dict:
                print(f"    KS ({wing_name}):")
                for k in ['goe', 'gue', 'gse', 'poi']:
                    if k in ks_dict:
                        ks, p = ks_dict[k]
                        print(f"      {k.upper()}: KS={ks:.4f}, p={p:.4f}")

    # Also test without resonance
    print("\n  --- Without resonance condition ---")
    for phi, label in [(0.0, "no-res phi=0"), (1.0/6, "no-res phi=1/6")]:
        M = build_M_peierls(cell, spinors, phi, use_resonance=False)
        evals = np.linalg.eigvalsh(M)
        sa = spectrum_analysis(evals, label=label)
        comparisons[label] = sa

        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        print(f"  {label}: beta_wing={sa['beta_wing']:+.4f}, "
              f"KS(GOE)={ks_goe[0]:.4f}, KS(GUE)={ks_gue[0]:.4f}, "
              f"pos_frac={sa['pos_frac']:.4f}")

    return comparisons


# ── Step 5: F-Connection Under Flux ───────────────────────────────────
def step5_f_connection(cell, spinors):
    """Track pos_frac vs phi: does it shift from 0.36 toward something else?"""
    print("\n" + "=" * 60)
    print("STEP 5: F-Connection Under Flux")
    print("=" * 60)
    print(f"  -ln(F) = {NEG_LN_F:.6f}")

    phi_vals = np.linspace(0, 1, 51)
    pos_fracs_res = []
    pos_fracs_nores = []

    for phi in phi_vals:
        # With resonance
        M = build_M_peierls(cell, spinors, phi, use_resonance=True)
        evals = np.linalg.eigvalsh(M)
        pf = np.sum(evals > 0) / len(evals)
        pos_fracs_res.append(pf)

        # Without resonance
        M = build_M_peierls(cell, spinors, phi, use_resonance=False)
        evals = np.linalg.eigvalsh(M)
        pf = np.sum(evals > 0) / len(evals)
        pos_fracs_nores.append(pf)

    print(f"\n  With resonance:")
    for phi, pf in zip([0, 1/6, 1/3, 1/2, 1.0],
                       [pos_fracs_res[0], pos_fracs_res[8], pos_fracs_res[17],
                        pos_fracs_res[25], pos_fracs_res[50]]):
        diff = abs(pf - NEG_LN_F)
        print(f"    phi={phi:.4f}: pos_frac={pf:.6f}, |pf - (-ln F)| = {diff:.6f}")

    print(f"\n  Without resonance:")
    for phi, pf in zip([0, 1/6, 1/3, 1/2, 1.0],
                       [pos_fracs_nores[0], pos_fracs_nores[8], pos_fracs_nores[17],
                        pos_fracs_nores[25], pos_fracs_nores[50]]):
        diff = abs(pf - NEG_LN_F)
        print(f"    phi={phi:.4f}: pos_frac={pf:.6f}, |pf - (-ln F)| = {diff:.6f}")

    return phi_vals, pos_fracs_res, pos_fracs_nores


# ── Step 6: Hofstadter Butterfly ──────────────────────────────────────
def step6_hofstadter(cell, spinors, n_phi=201):
    """
    Compute Hofstadter butterfly: eigenvalues vs phi.
    Plot all eigenvalues for phi in [0, 1].
    """
    print("\n" + "=" * 60)
    print("STEP 6: Hofstadter Butterfly")
    print("=" * 60)

    phi_vals = np.linspace(0, 1, n_phi)
    all_evals = []

    for phi in phi_vals:
        M = build_M_peierls(cell, spinors, phi)
        evals = np.linalg.eigvalsh(M)
        all_evals.append(evals)

    all_evals = np.array(all_evals)
    print(f"  Computed {n_phi} phi values, {all_evals.shape[1]} evals each")
    print(f"  E range: [{all_evals.min():.4f}, {all_evals.max():.4f}]")

    # Check if structure changes at rational phi
    for p, q, label in [(1, 6, "1/6 (P gate)"), (1, 3, "1/3"), (1, 2, "1/2"),
                         (1, 4, "1/4"), (1, 12, "1/12 (Coxeter)")]:
        idx = int(round(p / q * (n_phi - 1)))
        e = all_evals[idx]
        gaps = np.diff(np.sort(e))
        max_gap = np.max(gaps) if len(gaps) > 0 else 0
        print(f"  phi={label}: max gap = {max_gap:.6f}")

    return phi_vals, all_evals


# ── Step 7: Torus Size Scaling ────────────────────────────────────────
def step7_size_scaling():
    """
    Scale torus size L = 6, 9, 12, 15, 18.
    At each size: compare phi=0 vs phi=1/6.
    Check if GOE->GUE transition strengthens with size.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Torus Size Scaling")
    print("=" * 60)

    L_values = [6, 9, 12, 15, 18]
    scaling_results = []

    for L in L_values:
        t0 = time.time()
        cell = EisensteinTorus(L)
        spinors = assign_spinors_torus(cell)
        N = cell.num_nodes

        results_L = {'L': L, 'N': N}

        for use_res, res_tag in [(True, 'res'), (False, 'nores')]:
            for phi, phi_tag in [(0.0, 'zero'), (1.0/6, 'pgate')]:
                tag = f'{res_tag}_{phi_tag}'
                M = build_M_peierls(cell, spinors, phi, use_resonance=use_res)
                evals = np.linalg.eigvalsh(M)
                sa = spectrum_analysis(evals)
                results_L[f'beta_{tag}'] = sa['beta_wing']
                results_L[f'pf_{tag}'] = sa['pos_frac']
                results_L[f'ks_goe_{tag}'] = sa['ks_wing'].get('goe', (1, 0))
                results_L[f'ks_gue_{tag}'] = sa['ks_wing'].get('gue', (1, 0))

        dt = time.time() - t0
        scaling_results.append(results_L)

        # Print no-resonance results (where the signal is)
        goe_0 = results_L['ks_goe_nores_zero'][0]
        gue_0 = results_L['ks_gue_nores_zero'][0]
        goe_p = results_L['ks_goe_nores_pgate'][0]
        gue_p = results_L['ks_gue_nores_pgate'][0]
        fav0 = "GOE" if goe_0 < gue_0 else "GUE"
        favp = "GOE" if goe_p < gue_p else "GUE"
        print(f"  L={L:2d} (N={N:4d}): "
              f"nores phi=0 KS_GOE={goe_0:.3f} KS_GUE={gue_0:.3f} ({fav0}) | "
              f"nores phi=1/6 KS_GOE={goe_p:.3f} KS_GUE={gue_p:.3f} ({favp}) "
              f"[{dt:.1f}s]")

    # Scaling summary: no-resonance
    print("\n  Scaling summary (no-resonance):")
    print(f"  {'L':>4s} {'N':>5s}  {'KS_GOE(0)':>10s} {'KS_GUE(0)':>10s} {'fav0':>5s}  "
          f"{'KS_GOE(P)':>10s} {'KS_GUE(P)':>10s} {'favP':>5s}  {'beta(0)':>8s} {'beta(P)':>8s}")
    for r in scaling_results:
        goe_0 = r['ks_goe_nores_zero'][0]
        gue_0 = r['ks_gue_nores_zero'][0]
        goe_p = r['ks_goe_nores_pgate'][0]
        gue_p = r['ks_gue_nores_pgate'][0]
        fav0 = "GOE" if goe_0 < gue_0 else "GUE"
        favp = "GOE" if goe_p < gue_p else "GUE"
        print(f"  {r['L']:4d} {r['N']:5d}  {goe_0:10.4f} {gue_0:10.4f} {fav0:>5s}  "
              f"{goe_p:10.4f} {gue_p:10.4f} {favp:>5s}  "
              f"{r['beta_nores_zero']:+8.4f} {r['beta_nores_pgate']:+8.4f}")

    # Also show resonance for comparison
    print("\n  Scaling summary (with resonance):")
    print(f"  {'L':>4s} {'N':>5s}  {'beta(0)':>8s} {'beta(P)':>8s} {'KS_GOE(P)':>10s} {'KS_GUE(P)':>10s}")
    for r in scaling_results:
        print(f"  {r['L']:4d} {r['N']:5d}  {r['beta_res_zero']:+8.4f} {r['beta_res_pgate']:+8.4f} "
              f"{r['ks_goe_res_pgate'][0]:10.4f} {r['ks_gue_res_pgate'][0]:10.4f}")

    return scaling_results


# ── Figures ───────────────────────────────────────────────────────────
def make_figures(coarse_results, fine_results, comparisons, phi_f, pf_res, pf_nores,
                 hof_phi, hof_evals, scaling_results,
                 coarse_nores=None, fine_nores=None):
    """Generate all figures."""

    # ── Figure 1: KS statistics vs Phi (no-resonance — main signal) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if coarse_nores:
        phi_c = [r[0] for r in coarse_nores]
        goe_c = [r[1]['ks_wing'].get('goe', (1,0))[0] for r in coarse_nores]
        gue_c = [r[1]['ks_wing'].get('gue', (1,0))[0] for r in coarse_nores]
        axes[0].plot(phi_c, goe_c, 'bs-', markersize=6, label='KS(GOE)')
        axes[0].plot(phi_c, gue_c, 'ro-', markersize=6, label='KS(GUE)')
        axes[0].axvline(1/6, color='green', ls=':', alpha=0.7, label='phi=1/6')
        axes[0].set_xlabel('phi')
        axes[0].set_ylabel('KS statistic (lower = better fit)')
        axes[0].set_title('Coarse Sweep (no resonance)')
        axes[0].legend(fontsize=8)

    if fine_nores:
        phi_fn = [r[0] for r in fine_nores]
        goe_fn = [r[1]['ks_wing'].get('goe', (1,0))[0] for r in fine_nores]
        gue_fn = [r[1]['ks_wing'].get('gue', (1,0))[0] for r in fine_nores]
        axes[1].plot(phi_fn, goe_fn, 'bs-', markersize=4, label='KS(GOE)')
        axes[1].plot(phi_fn, gue_fn, 'ro-', markersize=4, label='KS(GUE)')
        axes[1].axvline(1/6, color='green', ls=':', alpha=0.7, label='1/6')
        axes[1].set_xlabel('phi')
        axes[1].set_ylabel('KS statistic')
        axes[1].set_title('Fine Sweep (no resonance)')
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_ks_vs_phi_nores.png", dpi=150)
    plt.close()

    # ── Figure 2: Spacing distributions at phi=0 and phi=1/6 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    s_grid = np.linspace(0, 4, 200)

    for ax, (label, sa) in zip(axes, [
        ("phi=0 (no flux)", comparisons.get("phi=0 (no flux)", {})),
        ("phi=1/6 (P gate)", comparisons.get("phi=1/6 (P gate)", {}))
    ]):
        sp = sa.get('spacings_wing', np.array([]))
        if len(sp) > 5:
            ax.hist(sp, bins=30, density=True, alpha=0.5, color='gray', label='Data')
        ax.plot(s_grid, wigner_pdf(s_grid, 1), 'b-', label='GOE', alpha=0.7)
        ax.plot(s_grid, wigner_pdf(s_grid, 2), 'r-', label='GUE', alpha=0.7)
        ax.plot(s_grid, np.exp(-s_grid), 'g--', label='Poisson', alpha=0.5)
        ax.set_xlabel('s (normalised spacing)')
        ax.set_ylabel('P(s)')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_spacing_distributions.png", dpi=150)
    plt.close()

    # ── Figure 3: F-connection (pos_frac vs phi) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_f, pf_res, 'b.-', label='With resonance', markersize=3)
    ax.plot(phi_f, pf_nores, 'r.-', label='No resonance', markersize=3)
    ax.axhline(NEG_LN_F, color='green', ls='--', alpha=0.7, label=f'-ln(F) = {NEG_LN_F:.4f}')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='1/2')
    ax.axvline(1/6, color='orange', ls=':', alpha=0.7, label='phi=1/6')
    ax.set_xlabel('phi')
    ax.set_ylabel('pos_frac')
    ax.set_title('F-Connection Under Flux')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_f_connection.png", dpi=150)
    plt.close()

    # ── Figure 4: Hofstadter butterfly ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(hof_evals.shape[1]):
        ax.plot(hof_phi, hof_evals[:, i], 'k,', markersize=0.3)
    ax.axvline(1/6, color='red', ls='--', alpha=0.5, label='phi=1/6')
    ax.axvline(1/3, color='blue', ls='--', alpha=0.3, label='phi=1/3')
    ax.axvline(1/2, color='green', ls='--', alpha=0.3, label='phi=1/2')
    ax.set_xlabel('phi (flux/plaquette)')
    ax.set_ylabel('E (eigenvalue)')
    ax.set_title(f'Hofstadter Butterfly (Eisenstein Torus L={hof_evals.shape[1]})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_hofstadter.png", dpi=150)
    plt.close()

    # ── Figure 5: Size scaling (no-resonance — where the signal is) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ns = [r['N'] for r in scaling_results]

    # No-resonance KS statistics
    ks_goe_zero = [r['ks_goe_nores_zero'][0] for r in scaling_results]
    ks_gue_zero = [r['ks_gue_nores_zero'][0] for r in scaling_results]
    ks_goe_pgate = [r['ks_goe_nores_pgate'][0] for r in scaling_results]
    ks_gue_pgate = [r['ks_gue_nores_pgate'][0] for r in scaling_results]

    axes[0].plot(Ns, ks_goe_zero, 'b^--', label='phi=0 KS(GOE)')
    axes[0].plot(Ns, ks_gue_zero, 'bv--', label='phi=0 KS(GUE)')
    axes[0].plot(Ns, ks_goe_pgate, 'r^-', label='phi=1/6 KS(GOE)')
    axes[0].plot(Ns, ks_gue_pgate, 'rv-', label='phi=1/6 KS(GUE)')
    axes[0].set_xlabel('N (nodes)')
    axes[0].set_ylabel('KS statistic (lower = better)')
    axes[0].set_title('KS Statistics (no resonance)')
    axes[0].legend(fontsize=7)

    # Beta comparison
    beta_nores_zero = [r['beta_nores_zero'] for r in scaling_results]
    beta_nores_pgate = [r['beta_nores_pgate'] for r in scaling_results]
    beta_res_zero = [r['beta_res_zero'] for r in scaling_results]
    beta_res_pgate = [r['beta_res_pgate'] for r in scaling_results]

    axes[1].plot(Ns, beta_nores_zero, 'b^--', label='nores phi=0')
    axes[1].plot(Ns, beta_nores_pgate, 'rv-', label='nores phi=1/6')
    axes[1].plot(Ns, beta_res_zero, 'bs:', alpha=0.4, label='res phi=0')
    axes[1].plot(Ns, beta_res_pgate, 'ro:', alpha=0.4, label='res phi=1/6')
    axes[1].axhline(1, color='blue', ls='--', alpha=0.3, label='GOE')
    axes[1].axhline(2, color='red', ls='--', alpha=0.3, label='GUE')
    axes[1].set_xlabel('N (nodes)')
    axes[1].set_ylabel('beta')
    axes[1].set_title('Beta vs Torus Size')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_size_scaling.png", dpi=150)
    plt.close()

    print(f"  Figures saved to {OUT}/")


# ── Full Report ───────────────────────────────────────────────────────
def write_report(coarse_res, coarse_nores, fine_res, fine_nores, comparisons,
                 phi_f, pf_res, pf_nores, hof_phi, hof_evals, scaling, cell_L):
    """Write FULL_REPORT.txt."""
    lines = []
    lines.append("PATH B: PEIERLS FLUX SWEEP ON EISENSTEIN TORUS")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Coarse sweep — with resonance
    lines.append("\nStep 2a: Coarse Sweep WITH resonance (positive wing)")
    lines.append(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}  {'pos_frac':>9s}")
    for phi, sa in coarse_res:
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        lines.append(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}  "
                     f"{sa['pos_frac']:9.4f}")

    # Coarse sweep — no resonance (KEY SIGNAL)
    lines.append("\nStep 2b: Coarse Sweep WITHOUT resonance (KEY - GOE->GUE transition)")
    lines.append(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}  {'favor':>5s}  {'pos_frac':>9s}")
    for phi, sa in coarse_nores:
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        favor = "GOE" if ks_goe[0] < ks_gue[0] else "GUE"
        lines.append(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}  "
                     f"{favor:>5s}  {sa['pos_frac']:9.4f}")

    # Fine sweep
    lines.append("\nStep 3a: Fine Sweep WITH resonance near phi=1/6")
    lines.append(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}")
    for phi, sa in fine_res:
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        marker = " <--" if abs(phi - 1/6) < 0.003 else ""
        lines.append(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}{marker}")

    lines.append("\nStep 3b: Fine Sweep WITHOUT resonance near phi=1/6 (KEY)")
    lines.append(f"  {'phi':>8s}  {'beta':>8s}  {'KS_GOE':>8s}  {'KS_GUE':>8s}  {'favor':>5s}")
    for phi, sa in fine_nores:
        ks_goe = sa['ks_wing'].get('goe', (1, 0))
        ks_gue = sa['ks_wing'].get('gue', (1, 0))
        favor = "GOE" if ks_goe[0] < ks_gue[0] else "GUE"
        marker = " <--" if abs(phi - 1/6) < 0.003 else ""
        lines.append(f"  {phi:8.5f}  {sa['beta_wing']:+8.4f}  {ks_goe[0]:8.4f}  {ks_gue[0]:8.4f}  "
                     f"{favor:>5s}{marker}")

    # Full RMT
    lines.append("\nStep 4: Full RMT at Phi*")
    for label, sa in comparisons.items():
        lines.append(f"\n  {label}:")
        lines.append(f"    N_evals={sa['n_evals']}, pos_frac={sa['pos_frac']:.6f}")
        lines.append(f"    beta_full={sa['beta_full']:+.4f}, beta_wing={sa['beta_wing']:+.4f}")
        for wing_name, ks_dict in [("full", sa['ks_full']), ("wing", sa['ks_wing'])]:
            if ks_dict:
                for k in ['goe', 'gue', 'poi']:
                    if k in ks_dict:
                        ks, p = ks_dict[k]
                        lines.append(f"    KS_{wing_name}({k.upper()})={ks:.4f}, p={p:.4f}")

    # F-connection
    lines.append("\nStep 5: F-Connection Under Flux")
    lines.append(f"  -ln(F) = {NEG_LN_F:.6f}")
    key_phis = [0, 8, 17, 25, 50]  # indices
    labels = ["0.000", "1/6", "1/3", "1/2", "1.000"]
    for lbl, idx in zip(labels, key_phis):
        if idx < len(pf_res):
            lines.append(f"  phi={lbl}: res_pf={pf_res[idx]:.6f}, nores_pf={pf_nores[idx]:.6f}")

    # Scaling — no-resonance (KEY)
    lines.append("\nStep 7a: Size Scaling (no resonance — KEY SIGNAL)")
    lines.append(f"  {'L':>4s} {'N':>5s}  {'KS_GOE(0)':>10s} {'KS_GUE(0)':>10s} {'fav0':>5s}  "
                f"{'KS_GOE(P)':>10s} {'KS_GUE(P)':>10s} {'favP':>5s}")
    for r in scaling:
        g0 = r['ks_goe_nores_zero'][0]
        u0 = r['ks_gue_nores_zero'][0]
        gp = r['ks_goe_nores_pgate'][0]
        up = r['ks_gue_nores_pgate'][0]
        f0 = "GOE" if g0 < u0 else "GUE"
        fp = "GOE" if gp < up else "GUE"
        lines.append(f"  {r['L']:4d} {r['N']:5d}  {g0:10.4f} {u0:10.4f} {f0:>5s}  "
                     f"{gp:10.4f} {up:10.4f} {fp:>5s}")

    # Scaling — with resonance
    lines.append("\nStep 7b: Size Scaling (with resonance)")
    lines.append(f"  {'L':>4s} {'N':>5s}  {'beta(0)':>8s} {'beta(P)':>8s} {'KS_GOE(P)':>10s} {'KS_GUE(P)':>10s}")
    for r in scaling:
        lines.append(f"  {r['L']:4d} {r['N']:5d}  {r['beta_res_zero']:+8.4f} {r['beta_res_pgate']:+8.4f} "
                     f"{r['ks_goe_res_pgate'][0]:10.4f} {r['ks_gue_res_pgate'][0]:10.4f}")

    # Decision
    lines.append("\n" + "=" * 60)
    lines.append("DECISION MATRIX")
    lines.append("=" * 60)

    # Count GOE->GUE flips in no-resonance scaling
    n_goe_zero = sum(1 for r in scaling
                     if r['ks_goe_nores_zero'][0] < r['ks_gue_nores_zero'][0])
    n_gue_pgate = sum(1 for r in scaling
                      if r['ks_gue_nores_pgate'][0] < r['ks_goe_nores_pgate'][0])

    lines.append(f"\n  No-resonance phi=0: {n_goe_zero}/{len(scaling)} favor GOE")
    lines.append(f"  No-resonance phi=1/6: {n_gue_pgate}/{len(scaling)} favor GUE")
    if n_gue_pgate > n_goe_zero:
        lines.append(f"  TRANSITION DETECTED: flux shifts preference from GOE to GUE")
    elif n_gue_pgate == len(scaling):
        lines.append(f"  STRONG TRANSITION: all sizes favor GUE at phi=1/6")

    # Largest torus detail
    if scaling:
        largest = scaling[-1]
        goe_0 = largest['ks_goe_nores_zero'][0]
        gue_0 = largest['ks_gue_nores_zero'][0]
        goe_p = largest['ks_goe_nores_pgate'][0]
        gue_p = largest['ks_gue_nores_pgate'][0]
        lines.append(f"\n  Largest torus (L={largest['L']}, N={largest['N']}):")
        lines.append(f"    phi=0:   KS(GOE)={goe_0:.4f}, KS(GUE)={gue_0:.4f} -> {'GOE' if goe_0<gue_0 else 'GUE'}")
        lines.append(f"    phi=1/6: KS(GOE)={goe_p:.4f}, KS(GUE)={gue_p:.4f} -> {'GOE' if goe_p<gue_p else 'GUE'}")

    # F-connection
    pf_phi0 = pf_res[0]
    pf_phi16 = pf_res[8] if len(pf_res) > 8 else 0
    lines.append(f"\n  F-connection:")
    lines.append(f"    pos_frac(phi=0) = {pf_phi0:.6f}")
    lines.append(f"    pos_frac(phi=1/6) = {pf_phi16:.6f}")
    lines.append(f"    -ln(F) = {NEG_LN_F:.6f}")

    report = "\n".join(lines)
    with open(f"{OUT}/FULL_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  Report written to {OUT}/FULL_REPORT.txt")

    return report


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("PATH B: PEIERLS FLUX SWEEP ON EISENSTEIN TORUS")
    print(f"  Prediction: GOE -> GUE at Phi* = pi/6 = 2pi/h(E6)")
    print(f"  P gate phase advance = pi/6 per Floquet step")
    print()

    L = 12  # Default torus size for steps 1-6

    # Step 1
    cell, spinors = step1_build_and_verify(L)

    # Step 2
    coarse_res, coarse_nores = step2_coarse_sweep(cell, spinors)

    # Step 3
    fine_res, fine_nores = step3_fine_sweep(cell, spinors)

    # Step 4
    comparisons = step4_full_rmt_at_phi_star(cell, spinors)

    # Step 5
    phi_f, pf_res, pf_nores = step5_f_connection(cell, spinors)

    # Step 6
    hof_phi, hof_evals = step6_hofstadter(cell, spinors, n_phi=201)

    # Step 7
    scaling = step7_size_scaling()

    # Figures
    make_figures(coarse_res, fine_res, comparisons, phi_f, pf_res, pf_nores,
                 hof_phi, hof_evals, scaling,
                 coarse_nores=coarse_nores, fine_nores=fine_nores)

    # Report
    write_report(coarse_res, coarse_nores, fine_res, fine_nores, comparisons,
                 phi_f, pf_res, pf_nores, hof_phi, hof_evals, scaling, L)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
