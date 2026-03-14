#!/usr/bin/env python3
"""
Fine Phi Sweep
===============
The coarse sweep tested {0, 1/12, 1/6, 1/4, 1/3, 1/2}.
At L=18 the minimum was at 1/6; at L=24 it shifted to 1/4.
Is 1/4 exactly the minimum, or is it between 1/6 and 1/4?

Run a fine sweep at L=18, 24, 30 with 60 phi values from 0 to 0.5
to locate the true continuous minimum.
"""
import numpy as np
from collections import defaultdict
import time

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
    for i, (a, b) in enumerate(torus.nodes):
        r = abs(z_coords[i]) / (L_max + 1e-10)
        theta = np.pi * (a - b) / 6.0
        u_i = np.exp(1j * theta) * np.array([np.cos(np.pi*r/2), 1j*np.sin(np.pi*r/2)], dtype=complex)
        u_i /= np.linalg.norm(u_i)
        u[i] = u_i
        v[i] = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
    return u, v

def build_M(torus, u, v, Phi, xi=XI):
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

# ============================================================================
if __name__ == '__main__':
    r_vals = np.linspace(0.01, 4.0, 200)
    g_mont = montgomery_formula(r_vals)

    # Fine sweep: 61 points from 0 to 0.5
    phi_values = np.linspace(0.0, 0.5, 61)

    # Key fractions to mark
    key_fracs = {
        '1/12': 1/12,
        '1/6':  1/6,
        '1/4':  1/4,
        '1/3':  1/3,
    }

    print("=" * 80)
    print("FINE PHI SWEEP")
    print("=" * 80)

    for L in [18, 24, 30]:
        t0 = time.time()
        torus = EisensteinTorus(L)
        u, v = assign_spinors_geometric(torus)

        rms_results = []
        hole_results = []

        for Phi in phi_values:
            M = build_M(torus, u, v, Phi)
            eigs = np.linalg.eigvalsh(M)
            wing = extract_positive_wing(np.sort(np.real(eigs)))
            if len(wing) < 10:
                rms_results.append(np.nan)
                hole_results.append(np.nan)
                continue
            unf, sp = unfold_spectrum(wing)
            g = pair_correlation_fast(unf, r_vals, 0.4)
            rms = np.sqrt(np.mean((g - g_mont)**2))
            rms_results.append(rms)
            hole_results.append(g[0])

        rms_arr = np.array(rms_results)
        hole_arr = np.array(hole_results)

        # Find minimum
        valid = ~np.isnan(rms_arr)
        idx_min = np.nanargmin(rms_arr)
        phi_min = phi_values[idx_min]
        rms_min = rms_arr[idx_min]

        dt = time.time() - t0

        print(f"\n{'='*60}")
        print(f"  L = {L}  (N = {torus.num_nodes}, wing ~ {len(wing)})  [{dt:.1f}s]")
        print(f"{'='*60}")
        print(f"  RMS minimum at Phi = {phi_min:.4f}  (RMS = {rms_min:.4f})")
        print(f"  As fraction: Phi_min = {phi_min:.4f} = 1/{1/phi_min:.2f}" if phi_min > 0 else "")

        # Compare to key fractions
        print(f"\n  Key fractions:")
        for name, frac in sorted(key_fracs.items(), key=lambda x: x[1]):
            idx = np.argmin(np.abs(phi_values - frac))
            rms_at = rms_arr[idx]
            delta = rms_at - rms_min
            marker = " ← MINIMUM" if abs(phi_values[idx] - phi_min) < 0.005 else ""
            print(f"    Phi = {name:>4} = {frac:.4f}: RMS = {rms_at:.4f}  (delta = +{delta:.4f}){marker}")

        # Nearest exact fractions to minimum
        print(f"\n  Nearest simple fractions to minimum (Phi = {phi_min:.4f}):")
        for n, d in [(1,4), (1,5), (1,6), (2,9), (2,11), (3,16), (1,7)]:
            f = n/d
            print(f"    {n}/{d} = {f:.4f}  (diff = {abs(phi_min - f):.4f})")

        # Print fine detail around minimum
        print(f"\n  Detail around minimum:")
        for i in range(max(0, idx_min-5), min(len(phi_values), idx_min+6)):
            marker = " <--" if i == idx_min else ""
            print(f"    Phi = {phi_values[i]:.4f}: RMS = {rms_arr[i]:.4f}{marker}")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("  If minimum is at 1/6 for all L: hexagonal (Eisenstein) structure dominates")
    print("  If minimum shifts to 1/4: tetrahedral structure (4 vertices, A4 subgroup)")
    print("  If minimum drifts continuously: finite-size effect, no locked fraction")
