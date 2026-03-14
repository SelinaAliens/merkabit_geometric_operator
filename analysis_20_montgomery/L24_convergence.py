#!/usr/bin/env python3
"""
L=24 Montgomery Convergence Check
==================================
Single focused run: does RMS(L=24) continue toward RMS(Riemann) = 0.082?
Exact same pipeline as Analysis 20 L=18.
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

# ── Eisenstein Torus ──
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

# ── Geometric Spinors (exact Analysis 19) ──
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

# ── M Construction ──
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

# ── Spectral Pipeline (exact Analysis 19/20) ──
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

# ── Pair Correlation ──
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

# ── KS Tests ──
def make_wigner_cdf(beta):
    a = 2.0 * gamma_fn((beta+2)/2)**(beta+1) / gamma_fn((beta+1)/2)**(beta+2)
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2
    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for idx, si in enumerate(s):
            result[idx], _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(si, 0))
        return result
    return cdf

def ks_tests(spacings):
    pos = spacings[spacings > 0]
    if len(pos) < 5: return {k: (1.0, 0.0) for k in ['goe','gue','poi']}
    results = {}
    for name, bv in [('goe',1), ('gue',2)]:
        ks, p = stats.kstest(pos, make_wigner_cdf(bv))
        results[name] = (float(ks), float(p))
    ks, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (float(ks), float(p))
    return results

# ── Riemann Zeros ──
def get_riemann_pair_correlation(r_vals, bandwidth=0.4, n_zeros=1000):
    cache = os.path.join(OUT, f"riemann_zeros_{n_zeros}.npy")
    if os.path.exists(cache):
        gammas = np.load(cache)
        print(f"  Loaded {len(gammas)} cached Riemann zeros")
    else:
        from mpmath import mp, zetazero
        mp.dps = 20
        print(f"  Computing {n_zeros} zeros...")
        gammas = np.array([float(zetazero(n).imag) for n in range(1, n_zeros+1)])
        np.save(cache, gammas)
        print(f"  Saved to cache")
    T = gammas
    N_smooth = T / (2*np.pi) * np.log(T / (2*np.pi*np.e)) + 7.0/8.0
    sp = np.diff(N_smooth)
    sp /= np.mean(sp)
    g = pair_correlation_fast(N_smooth, r_vals, bandwidth)
    rms = np.sqrt(np.mean((g - montgomery_formula(r_vals))**2))
    return g, rms, gammas

# ============================================================================
if __name__ == '__main__':
    r_vals = np.linspace(0.01, 4.0, 200)
    g_mont = montgomery_formula(r_vals)

    print("=" * 65)
    print("L=24 MONTGOMERY CONVERGENCE CHECK")
    print("=" * 65)

    # ── Riemann benchmark ──
    print("\n--- Riemann benchmark ---")
    g_riem, rms_riem, gammas = get_riemann_pair_correlation(r_vals)
    print(f"  RMS(Riemann) = {rms_riem:.4f}")
    print(f"  g_riem(0.01) = {g_riem[0]:.4f}")

    # ── Run L=12, 18, 24, 27, 30 ──
    results = []
    for L in [12, 15, 18, 21, 24, 27, 30]:
        t0 = time.time()
        torus = EisensteinTorus(L)
        u, v, omega = assign_spinors_geometric(torus)
        M = build_M(torus, u, v, omega, 1.0/6)
        eigs = np.linalg.eigvalsh(M)
        unf, sp = get_unfolded_wing(eigs)
        n_wing = len(unf)
        g = pair_correlation_fast(unf, r_vals, 0.4)
        rms = np.sqrt(np.mean((g - g_mont)**2))
        ks = ks_tests(sp)
        hole = g[0]
        dt = time.time() - t0

        results.append(dict(L=L, N=torus.num_nodes, n_wing=n_wing,
                            rms=rms, hole=hole,
                            ks_gue=ks['gue'][0], p_gue=ks['gue'][1],
                            g=g.copy(), dt=dt))

        print(f"\n  L={L:2d} (N={torus.num_nodes:4d}, wing={n_wing:3d}): "
              f"RMS={rms:.4f}, g(0)={hole:.4f}, "
              f"KS(GUE)={ks['gue'][0]:.4f} p={ks['gue'][1]:.3f}  [{dt:.1f}s]")

    # ── Convergence table ──
    print("\n" + "=" * 65)
    print("CONVERGENCE TABLE")
    print("=" * 65)
    print(f"  {'L':>4} | {'N':>5} | {'wing':>5} | {'RMS vs Montgomery':>18} | {'g(r~0)':>8} | {'KS(GUE)':>8}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['L']:>4} | {r['N']:>5} | {r['n_wing']:>5} | {r['rms']:>18.4f} | {r['hole']:>8.4f} | {r['ks_gue']:>8.4f}")
    print(f"  {'Riem':>4} | {'1000':>5} | {'1000':>5} | {rms_riem:>18.4f} | {g_riem[0]:>8.4f} | {'0.0433':>8}")
    print(f"  {'Mont':>4} | {'inf':>5} | {'inf':>5} | {'0.0000':>18} | {'0.0000':>8} | {'--':>8}")

    # ── Phi sweep at L=24 ──
    print(f"\n{'='*65}")
    print("PHI SWEEP AT L=24")
    print(f"{'='*65}")
    torus24 = EisensteinTorus(24)
    u24, v24, om24 = assign_spinors_geometric(torus24)
    print(f"  {'Phi':>8} | {'RMS vs Montgomery':>18} | {'KS(GUE)':>8} | {'note'}")
    print(f"  {'-'*55}")
    phi_results = []
    for Phi_test in [0.0, 1/12, 1/6, 1/4, 1/3, 1/2]:
        M_t = build_M(torus24, u24, v24, om24, Phi_test)
        eigs_t = np.linalg.eigvalsh(M_t)
        unf_t, sp_t = get_unfolded_wing(eigs_t)
        g_t = pair_correlation_fast(unf_t, r_vals, 0.4)
        rms_t = np.sqrt(np.mean((g_t - g_mont)**2))
        ks_t = ks_tests(sp_t)['gue']
        flag = " <- P-gate" if abs(Phi_test - 1/6) < 0.001 else ""
        print(f"  {Phi_test:>8.4f} | {rms_t:>18.4f} | {ks_t[0]:>8.4f} |{flag}")
        phi_results.append(dict(phi=Phi_test, rms=rms_t, ks_gue=ks_t[0]))

    # ── Find RMS minimum ──
    min_phi = min(phi_results, key=lambda x: x['rms'])
    print(f"\n  RMS minimum at Phi={min_phi['phi']:.4f} (RMS={min_phi['rms']:.4f})")

    # ── Verdict ──
    r24 = [r for r in results if r['L'] == 24][0]
    r18 = [r for r in results if r['L'] == 18][0]
    r12 = [r for r in results if r['L'] == 12][0]

    print(f"\n{'='*65}")
    print("VERDICT")
    print(f"{'='*65}")
    print(f"  L=12 RMS: {r12['rms']:.4f}")
    print(f"  L=18 RMS: {r18['rms']:.4f}")
    print(f"  L=24 RMS: {r24['rms']:.4f}")
    print(f"  Riemann:  {rms_riem:.4f}")

    if r24['rms'] < r18['rms']:
        if r24['rms'] < r12['rms'] * 0.85:
            print(f"  --> CONVERGING (accelerating)")
        else:
            print(f"  --> CONVERGING (steady)")
    else:
        print(f"  --> STALLED or DIVERGING")

    ratio24 = r24['rms'] / rms_riem
    print(f"  L=24 ratio to Riemann: {ratio24:.3f}x")

    # Extrapolate
    # Fit RMS ~ a / sqrt(N) + b
    Ns = np.array([r['n_wing'] for r in results])
    RMSs = np.array([r['rms'] for r in results])
    # Simple: log(RMS) vs log(N)
    mask = Ns > 20
    if mask.sum() >= 3:
        coeffs = np.polyfit(np.log(Ns[mask]), np.log(RMSs[mask]), 1)
        slope = coeffs[0]
        print(f"\n  Power law fit: RMS ~ N^{slope:.3f}")
        # Extrapolate to RMS = 0.082
        N_target = np.exp((np.log(0.082) - coeffs[1]) / coeffs[0])
        print(f"  Extrapolated N for RMS=0.082: ~{N_target:.0f} eigenvalues")
        L_target = np.sqrt(N_target / 0.4)  # rough: wing ~ 0.4 * L^2
        print(f"  Estimated L for RMS=0.082: ~{L_target:.0f}")

    # ── Save arrays for Paper 7 ──
    np.save(os.path.join(OUT, "r_vals.npy"), r_vals)
    np.save(os.path.join(OUT, "g_montgomery.npy"), g_mont)
    np.save(os.path.join(OUT, "g_riemann.npy"), g_riem)
    for r in results:
        np.save(os.path.join(OUT, f"g_M_L{r['L']}.npy"), r['g'])

    # ── Save text report ──
    fname = os.path.join(OUT, "L24_convergence.txt")
    with open(fname, 'w') as f:
        f.write("L=24 Montgomery Convergence Check\n")
        f.write("=" * 50 + "\n\n")
        f.write("Convergence table:\n")
        for r in results:
            f.write(f"  L={r['L']:2d} (wing={r['n_wing']:3d}): "
                    f"RMS={r['rms']:.4f}, g(0)={r['hole']:.4f}, "
                    f"KS(GUE)={r['ks_gue']:.4f}, p={r['p_gue']:.3f}\n")
        f.write(f"  Riemann (1000): RMS={rms_riem:.4f}, g(0)={g_riem[0]:.4f}\n")
        f.write(f"  Montgomery: RMS=0.0000, g(0)=0.0000\n\n")
        f.write("Phi sweep at L=24:\n")
        for pr in phi_results:
            flag = " <- P-gate" if abs(pr['phi'] - 1/6) < 0.001 else ""
            f.write(f"  Phi={pr['phi']:.4f}: RMS={pr['rms']:.4f}, KS(GUE)={pr['ks_gue']:.4f}{flag}\n")
        f.write(f"\nRMS minimum at Phi={min_phi['phi']:.4f}\n")
        f.write(f"\nVerdict: {'CONVERGING' if r24['rms'] < r18['rms'] else 'STALLED'}\n")
        f.write(f"L=24 ratio to Riemann: {ratio24:.3f}x\n")
    print(f"\n  Saved: {fname}")
    print(f"  Saved: g_M_L*.npy, g_riemann.npy, g_montgomery.npy, r_vals.npy")
