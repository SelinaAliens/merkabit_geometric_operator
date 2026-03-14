#!/usr/bin/env python3
"""
Ultra-Fine Phi Sweep
=====================
The fine sweep found the minimum near Phi = 0.15 = 3/20 at L=24,30.
Now sweep Phi in [0.12, 0.20] with 200 points to locate the true
continuous minimum, and test whether it locks to an exact fraction.

Candidate fractions in this range:
  1/7   = 0.142857...
  3/20  = 0.150000
  2/13  = 0.153846...
  1/6   = 0.166667...
  3/17  = 0.176471...

E6-derived candidates:
  3/20  = Z3 / (dim * n_gates)
  r/(r+h+dim+... ) = various
  (h-r)/(h*r-h) = 6/60 = 1/10 (too low)
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
        edge_set = set()
        for i, (a, b) in enumerate(self.nodes):
            for da, db in UNIT_VECTORS_AB:
                j = self.node_index[((a+da)%L, (b+db)%L)]
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

    # Ultra-fine: 200 points from 0.12 to 0.20
    phi_values = np.linspace(0.12, 0.20, 201)
    step = phi_values[1] - phi_values[0]

    # Candidate exact fractions
    candidates = [
        ('1/7',    1/7),
        ('10/69',  10/69),
        ('3/20',   3/20),
        ('2/13',   2/13),
        ('4/25',   4/25),
        ('5/31',   5/31),
        ('3/19',   3/19),
        ('1/6',    1/6),
        ('6/37',   6/37),
        ('5/33',   5/33),
        ('7/46',   7/46),
        ('3/h',    3/12),
        ('r/h²',   6/144),
        ('(r-1)/(h*3)', 5/36),
        ('n/(h+3+n+r+h)', 5/36),
        ('3/(4*n)', 3/20),  # = Z3/(dim*n_gates)
        ('1/(2h/r+1)', 1/(2*12/6+1)),  # = 1/5
        ('r/(h*(r-1))', 6/60),  # = 1/10
        ('3/(h+r+2)', 3/20),
    ]

    print("=" * 80)
    print("ULTRA-FINE PHI SWEEP")
    print(f"Range: [{phi_values[0]:.4f}, {phi_values[-1]:.4f}], step = {step:.5f}")
    print("=" * 80)

    for L in [24, 30]:
        t0 = time.time()
        torus = EisensteinTorus(L)
        u, v = assign_spinors_geometric(torus)

        rms_results = []
        for Phi in phi_values:
            M = build_M(torus, u, v, Phi)
            eigs = np.linalg.eigvalsh(M)
            wing = extract_positive_wing(np.sort(np.real(eigs)))
            if len(wing) < 10:
                rms_results.append(np.nan)
                continue
            unf, sp = unfold_spectrum(wing)
            g = pair_correlation_fast(unf, r_vals, 0.4)
            rms = np.sqrt(np.mean((g - g_mont)**2))
            rms_results.append(rms)

        rms_arr = np.array(rms_results)
        idx_min = np.nanargmin(rms_arr)
        phi_min = phi_values[idx_min]
        rms_min = rms_arr[idx_min]

        # Parabolic interpolation around minimum
        if 1 <= idx_min <= len(phi_values) - 2:
            x0, x1, x2 = phi_values[idx_min-1], phi_values[idx_min], phi_values[idx_min+1]
            y0, y1, y2 = rms_arr[idx_min-1], rms_arr[idx_min], rms_arr[idx_min+1]
            # Vertex of parabola through 3 points
            denom = 2 * ((x1-x0)*(y1-y2) - (x1-x2)*(y1-y0))
            if abs(denom) > 1e-15:
                phi_parab = x1 - ((x1-x0)**2*(y1-y2) - (x1-x2)**2*(y1-y0)) / denom
                # Evaluate parabolic RMS at vertex
                A = ((y0-y1)*(x1-x2) - (y1-y2)*(x0-x1)) / ((x0-x1)*(x1-x2)*(x0-x2))
                rms_parab = y1 - (x1 - phi_parab)**2 * A if A != 0 else y1
            else:
                phi_parab = phi_min
                rms_parab = rms_min
        else:
            phi_parab = phi_min
            rms_parab = rms_min

        dt = time.time() - t0

        print(f"\n{'='*70}")
        print(f"  L = {L}  [{dt:.1f}s]")
        print(f"{'='*70}")
        print(f"  Grid minimum:     Phi = {phi_min:.6f}  (RMS = {rms_min:.6f})")
        print(f"  Parabolic interp: Phi = {phi_parab:.6f}  (RMS ~ {rms_parab:.6f})")
        print(f"  1/Phi_parab = {1/phi_parab:.6f}")

        # Distance to candidate fractions
        print(f"\n  Distance to candidate fractions:")
        dists = []
        for name, frac in candidates:
            d = abs(phi_parab - frac)
            dists.append((d, name, frac))
        dists.sort()
        seen = set()
        for d, name, frac in dists[:15]:
            if frac not in seen:
                seen.add(frac)
                # RMS at this exact fraction (nearest grid point)
                idx_frac = np.argmin(np.abs(phi_values - frac))
                rms_frac = rms_arr[idx_frac] if 0 <= idx_frac < len(rms_arr) else np.nan
                print(f"    {name:>12} = {frac:.6f}  |  diff = {d:.6f}  |  RMS = {rms_frac:.6f}")

        # Detail around minimum: 20 points
        print(f"\n  Detail around minimum:")
        for i in range(max(0, idx_min-10), min(len(phi_values), idx_min+11)):
            marker = " <-- grid min" if i == idx_min else ""
            print(f"    Phi = {phi_values[i]:.5f}: RMS = {rms_arr[i]:.6f}{marker}")

    # ── Final analysis ──
    print(f"\n{'='*80}")
    print("FRACTION ANALYSIS")
    print(f"{'='*80}")
    print("  Key architectural fractions and their meaning:")
    print(f"    1/7   = {1/7:.6f}  — 7 = r+1 (rank+1)")
    print(f"    3/20  = {3/20:.6f}  — 3/(4*5) = Z3/(dim*n_gates)")
    print(f"    2/13  = {2/13:.6f}  — 2/(h+1) = 2/13")
    print(f"    1/6   = {1/6:.6f}  — 1/r = Coxeter step 2pi/12")
    print(f"    3/19  = {3/19:.6f}  — 3/(h+r+1)")
    print(f"    3/18  = {3/18:.6f}  — 1/(r) = 3/3r")
    print(f"    5/33  = {5/33:.6f}  — n/(h+h+r+3)")
