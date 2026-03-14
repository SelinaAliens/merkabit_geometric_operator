#!/usr/bin/env python3
"""
PHI MINIMUM CHECK — Is the shift from Phi=1/6 to Phi=1/4 at L=24 bandwidth-dependent?

Geometric spinors are DETERMINISTIC (no seeds). The 0.001 RMS difference is exact.
But the pair correlation kernel bandwidth (sigma) affects the RMS estimate.
If the minimum shifts back to 1/6 at smaller bandwidth, the shift is a smoothing artefact.

Also: reduced bandwidth at L=30 to check if the correlation hole deepens
and convergence accelerates.
"""
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
from collections import defaultdict
import time
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1)]
XI = 3.0

class EisensteinTorus:
    def __init__(self, L):
        self.L = L
        self.nodes = [(a, b) for a in range(L) for b in range(L)]
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        self.edges = []
        self.neighbours = defaultdict(list)
        edge_set = set()
        for i, (a, b) in enumerate(self.nodes):
            for da, db in UNIT_VECTORS_AB:
                j = self.node_index[((a+da)%L, (b+db)%L)]
                self.neighbours[i].append(j)
                edge = (min(i,j), max(i,j))
                if edge not in edge_set and i != j:
                    edge_set.add(edge)
                    self.edges.append(edge)
        self.sublattice = [(a+b)%3 for (a,b) in self.nodes]
        self.chirality = [0 if s==0 else (1 if s==1 else -1) for s in self.sublattice]

def assign_spinors_geometric(torus):
    N = torus.num_nodes
    z_coords = [a + b * OMEGA_EISEN for (a, b) in torus.nodes]
    L_max = max(abs(z) for z in z_coords) if N > 1 else 1.0
    u = np.zeros((N, 2), dtype=complex)
    v = np.zeros((N, 2), dtype=complex)
    omega = np.zeros(N)
    for i, (a, b) in enumerate(torus.nodes):
        r = abs(z_coords[i]) / (L_max + 1e-10)
        theta = np.pi * (a - b) / 6.0
        u_i = np.exp(1j * theta) * np.array([np.cos(np.pi*r/2), 1j*np.sin(np.pi*r/2)], dtype=complex)
        u_i /= np.linalg.norm(u_i)
        u[i] = u_i
        v[i] = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega

def build_M(torus, u, v, omega, Phi, xi=XI):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = torus.L
    for (i, j) in torus.edges:
        a_i, b_i = torus.nodes[i]
        a_j, b_j = torus.nodes[j]
        da = a_j - a_i; db = b_j - b_i
        if da >  L//2: da -= L
        if da < -(L//2): da += L
        if db >  L//2: db -= L
        if db < -(L//2): db += L
        A_ij = Phi * (2*a_i + da) / 2.0 * db
        c = decay * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)
        M[i, j] = c; M[j, i] = np.conj(c)
    return (M + M.conj().T) / 2.0

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

if __name__ == '__main__':
    r_vals = np.linspace(0.01, 4.0, 200)
    g_mont = montgomery_formula(r_vals)

    # ================================================================
    # TEST 1: Phi minimum vs bandwidth at L=24
    # ================================================================
    print("=" * 70)
    print("TEST 1: PHI MINIMUM vs BANDWIDTH at L=24")
    print("=" * 70)
    print("Geometric spinors are deterministic. The only free parameter is bandwidth.")
    print()

    torus24 = EisensteinTorus(24)
    u24, v24, om24 = assign_spinors_geometric(torus24)

    # Pre-compute eigenvalues at each Phi (deterministic, do it once)
    phi_vals = [0.0, 1/12, 1/6, 1/4, 1/3, 1/2]
    eigs_by_phi = {}
    unf_by_phi = {}
    for Phi in phi_vals:
        M = build_M(torus24, u24, v24, om24, Phi)
        eigs = np.linalg.eigvalsh(M)
        unf, sp = get_unfolded_wing(eigs)
        eigs_by_phi[Phi] = eigs
        unf_by_phi[Phi] = unf

    bandwidths = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

    print(f"  {'bw':>6} |", end="")
    for Phi in phi_vals:
        label = f"Phi={Phi:.4f}"
        print(f" {label:>12} |", end="")
    print(f" {'minimum':>10}")
    print(f"  {'-'*105}")

    for bw in bandwidths:
        rms_vals = {}
        for Phi in phi_vals:
            g = pair_correlation_fast(unf_by_phi[Phi], r_vals, bandwidth=bw)
            rms = np.sqrt(np.mean((g - g_mont)**2))
            rms_vals[Phi] = rms

        min_phi = min(rms_vals, key=rms_vals.get)
        print(f"  {bw:>6.2f} |", end="")
        for Phi in phi_vals:
            flag = " *" if Phi == min_phi else "  "
            print(f" {rms_vals[Phi]:>10.4f}{flag} |", end="")
        print(f" Phi={min_phi:.4f}")

    # ================================================================
    # TEST 2: Same test at L=18 (should always be 1/6)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 2: PHI MINIMUM vs BANDWIDTH at L=18 (reference)")
    print("=" * 70)

    torus18 = EisensteinTorus(18)
    u18, v18, om18 = assign_spinors_geometric(torus18)

    eigs18_by_phi = {}
    unf18_by_phi = {}
    for Phi in phi_vals:
        M = build_M(torus18, u18, v18, om18, Phi)
        eigs = np.linalg.eigvalsh(M)
        unf, sp = get_unfolded_wing(eigs)
        eigs18_by_phi[Phi] = eigs
        unf18_by_phi[Phi] = unf

    print(f"  {'bw':>6} |", end="")
    for Phi in phi_vals:
        label = f"Phi={Phi:.4f}"
        print(f" {label:>12} |", end="")
    print(f" {'minimum':>10}")
    print(f"  {'-'*105}")

    for bw in bandwidths:
        rms_vals = {}
        for Phi in phi_vals:
            g = pair_correlation_fast(unf18_by_phi[Phi], r_vals, bandwidth=bw)
            rms = np.sqrt(np.mean((g - g_mont)**2))
            rms_vals[Phi] = rms

        min_phi = min(rms_vals, key=rms_vals.get)
        print(f"  {bw:>6.2f} |", end="")
        for Phi in phi_vals:
            flag = " *" if Phi == min_phi else "  "
            print(f" {rms_vals[Phi]:>10.4f}{flag} |", end="")
        print(f" Phi={min_phi:.4f}")

    # ================================================================
    # TEST 3: Reduced bandwidth convergence (L=12 to L=30)
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 3: CONVERGENCE vs BANDWIDTH (Phi=1/6)")
    print("=" * 70)
    print("Does smaller bandwidth resolve the correlation hole and accelerate convergence?")
    print()

    L_vals = [12, 15, 18, 21, 24, 27, 30]
    bw_test = [0.2, 0.3, 0.4]

    # Load Riemann benchmark
    cache = os.path.join(OUT, "riemann_zeros_1000.npy")
    gammas = np.load(cache)
    T = gammas
    N_smooth_r = T / (2*np.pi) * np.log(T / (2*np.pi*np.e)) + 7.0/8.0

    print(f"  Riemann benchmark (1000 zeros):")
    for bw in bw_test:
        g_r = pair_correlation_fast(N_smooth_r, r_vals, bw)
        rms_r = np.sqrt(np.mean((g_r - g_mont)**2))
        print(f"    bw={bw:.1f}: RMS={rms_r:.4f}, g(0)={g_r[0]:.4f}")

    print(f"\n  {'L':>4} | {'wing':>5} |", end="")
    for bw in bw_test:
        print(f" {'RMS(bw='+str(bw)+')':>12} {'g(0)':>7} |", end="")
    print()
    print(f"  {'-'*75}")

    for L in L_vals:
        torus = EisensteinTorus(L)
        u, v, omega = assign_spinors_geometric(torus)
        M = build_M(torus, u, v, omega, 1.0/6)
        eigs = np.linalg.eigvalsh(M)
        unf, sp = get_unfolded_wing(eigs)
        n_wing = len(unf)

        print(f"  {L:>4} | {n_wing:>5} |", end="")
        for bw in bw_test:
            g = pair_correlation_fast(unf, r_vals, bw)
            rms = np.sqrt(np.mean((g - g_mont)**2))
            hole = g[0]
            print(f" {rms:>12.4f} {hole:>7.4f} |", end="")
        print()

    # ================================================================
    # TEST 4: Fine Phi grid at L=24, bw=0.3
    # ================================================================
    print(f"\n{'='*70}")
    print("TEST 4: FINE PHI GRID at L=24, bw=0.3")
    print("=" * 70)
    print("Is the true minimum at 1/6, between 1/6 and 1/4, or at 1/4?")
    print()

    fine_phis = np.linspace(0.10, 0.30, 21)
    print(f"  {'Phi':>8} | {'RMS(bw=0.3)':>12} | {'RMS(bw=0.4)':>12} | note")
    print(f"  {'-'*55}")

    for Phi_test in fine_phis:
        M_t = build_M(torus24, u24, v24, om24, Phi_test)
        eigs_t = np.linalg.eigvalsh(M_t)
        unf_t, _ = get_unfolded_wing(eigs_t)

        g3 = pair_correlation_fast(unf_t, r_vals, 0.3)
        rms3 = np.sqrt(np.mean((g3 - g_mont)**2))

        g4 = pair_correlation_fast(unf_t, r_vals, 0.4)
        rms4 = np.sqrt(np.mean((g4 - g_mont)**2))

        flags = []
        if abs(Phi_test - 1/6) < 0.005: flags.append("1/6")
        if abs(Phi_test - 1/4) < 0.005: flags.append("1/4")
        if abs(Phi_test - 0.2) < 0.005: flags.append("1/5")
        flag = " <- " + ", ".join(flags) if flags else ""
        print(f"  {Phi_test:>8.4f} | {rms3:>12.4f} | {rms4:>12.4f} |{flag}")

    print(f"\nAll checks complete.")
