#!/usr/bin/env python3
"""
COMMENSURABILITY TEST -- GEOMETRIC SPINORS (EXACT ANALYSIS 19 CONSTRUCTION)
===========================================================================
Previous tests used random base spinors with deterministic phases -> GOE everywhere.
The original Analysis 19 GUE signal used FULLY GEOMETRIC spinors:
  u[i] = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]
  v[i] = [-conj(u[1]), conj(u[0])]   (SU(2) time-reverse)

Both u and v are deterministic functions of the lattice coordinates (a, b).
v is NOT an independent vector -- it is the time-reverse of u.

This test uses the EXACT spinor construction from peierls_flux.py to verify
whether the commensurability prediction (GUE iff 6|L) holds with the
construction that originally produced KS(GUE) = 0.052 at L=18.

Prediction: GUE iff 6|L (6 = h(E6)/2)
Output: C:/Users/selin/merkabit_results/peierls_flux_corrected/commensurability_geometric.txt
"""
import numpy as np
from collections import defaultdict
import time
import os

OMEGA_EISEN = np.exp(2j * np.pi / 3)

# ============================================================================
# EISENSTEIN TORUS
# ============================================================================
class EisensteinTorus:
    UNIT_VECTORS = [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1)]
    def __init__(self, L):
        self.L = L
        self.nodes = [(a, b) for a in range(L) for b in range(L)]
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        self.edges = []
        self.neighbours = defaultdict(list)
        edge_set = set()
        for i, (a, b) in enumerate(self.nodes):
            for da, db in self.UNIT_VECTORS:
                na = (a + da) % L
                nb = (b + db) % L
                j = self.node_index[(na, nb)]
                self.neighbours[i].append(j)
                edge = (min(i, j), max(i, j))
                if edge not in edge_set and i != j:
                    edge_set.add(edge)
                    self.edges.append(edge)
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            if s == 0: self.chirality.append(0)
            elif s == 1: self.chirality.append(+1)
            else: self.chirality.append(-1)

# ============================================================================
# GEOMETRIC SPINORS -- EXACT COPY FROM peierls_flux.py assign_spinors_torus
# ============================================================================
def assign_spinors_geometric(torus):
    """
    Fully deterministic geometric spinors from Analysis 19.
    u[i] = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]
    v[i] = [-conj(u[1]), conj(u[0])]   (SU(2) time-reverse)
    No random numbers. No seed. Entirely determined by lattice geometry.
    """
    N = torus.num_nodes
    z_coords = [a + b * OMEGA_EISEN for (a, b) in torus.nodes]
    L_max = max(abs(z) for z in z_coords) if N > 1 else 1.0

    u = np.zeros((N, 2), dtype=complex)
    v = np.zeros((N, 2), dtype=complex)
    omega = np.zeros(N)

    for i, (a, b) in enumerate(torus.nodes):
        z = z_coords[i]
        r = abs(z) / (L_max + 1e-10)
        theta = np.pi * (a - b) / 6.0

        u_i = np.exp(1j * theta) * np.array([np.cos(np.pi * r / 2),
                                               1j * np.sin(np.pi * r / 2)],
                                              dtype=complex)
        u_i /= np.linalg.norm(u_i)
        v_i = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)

        u[i] = u_i
        v[i] = v_i
        omega[i] = torus.chirality[i] * 1.0

    return u, v, omega

# ============================================================================
# RANDOM SPINORS (CONTROL)
# ============================================================================
def assign_spinors_random(torus, seed=42):
    """Random spinors -- no geometric structure. Control."""
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    u = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    v = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    omega = np.zeros(N)
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        v[i] /= np.linalg.norm(v[i])
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega

# ============================================================================
# CASCADE SPINORS (Z3 PHASES ON RANDOM BASE)
# ============================================================================
def assign_spinors_cascade(torus, seed=42):
    """Z3 phases on random spinors. From commensurability_cascade.py."""
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    u = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    v = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    omega = np.zeros(N)
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        s = torus.sublattice[i]
        gate_step = s * 4
        theta = gate_step * np.pi / 6
        v[i] = np.exp(1j * theta) * v[i] / np.linalg.norm(v[i])
        omega[i] = torus.chirality[i] * 1.0
    return u, v, omega

# ============================================================================
# PEIERLS + M CONSTRUCTION (EXACT COPY FROM peierls_flux.py)
# ============================================================================
def build_M(torus, u, v, omega, Phi, xi=3.0, use_resonance=False):
    """
    Build M with Peierls phase. Exact copy of peierls_flux.py build_M_peierls.
    use_resonance=False matches the no-resonance variant that gave GUE.
    """
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    L = torus.L

    for (i, j) in torus.edges:
        a_i, b_i = torus.nodes[i]
        a_j, b_j = torus.nodes[j]

        if use_resonance:
            resonance = np.exp(-(omega[i] + omega[j]) ** 2 / 0.1)
        else:
            resonance = 1.0

        da = a_j - a_i
        db = b_j - b_i
        if da >  L // 2: da -= L
        if da < -(L // 2): da += L
        if db >  L // 2: db -= L
        if db < -(L // 2): db += L

        A_ij = Phi * (2 * a_i + da) / 2.0 * db
        coupling = decay * resonance * np.vdot(u[i], v[j]) * np.exp(2j * np.pi * A_ij)

        M[i, j] = coupling
        M[j, i] = np.conj(coupling)

    M = (M + M.conj().T) / 2.0
    return M

# ============================================================================
# RMT STATISTICS
# ============================================================================
def goe_cdf(s): return 1 - np.exp(-np.pi/4 * s**2)
def gue_cdf(s): return 1 - np.exp(-4/np.pi * s**2) * (1 + 4/np.pi * s**2)

def ks(spacings, cdf):
    s = np.sort(spacings)
    n = len(s)
    emp = np.arange(1, n+1) / n
    th  = cdf(s)
    stat = np.max(np.abs(emp - th))
    t = stat * np.sqrt(n)
    p = float(2 * np.exp(-2 * t**2))
    return float(stat), p

def get_spacings(eigs):
    s = np.diff(np.sort(eigs))
    s = s[s > 1e-8]
    if len(s) == 0: return s
    return s / s.mean()

def fit_beta(spacings, n_bins=30):
    hist, edges = np.histogram(spacings, bins=n_bins, range=(0.01, 2.5), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = (centers < 0.8) & (hist > 0)
    if mask.sum() < 3: return np.nan
    return float(np.polyfit(np.log(centers[mask]), np.log(hist[mask]), 1)[0])

# ============================================================================
# MAIN TEST
# ============================================================================
def run_test(L_vals, Phi, spinor_fn, label, needs_seed=False, n_seeds=5,
             use_resonance=False):
    """Run commensurability test for one spinor construction."""
    print(f"\n{'='*70}")
    print(f"SPINOR CONSTRUCTION: {label}")
    print(f"Phi = {Phi:.6f}, resonance = {use_resonance}")
    print(f"Prediction: GUE iff 6|L")
    print(f"{'='*70}")
    print(f"  {'L':>4} | {'N':>5} | {'L%6':>4} | {'predict':>7} | "
          f"{'beta':>6} | {'KS_GOE':>7} | {'KS_GUE':>7} | "
          f"{'p_GOE':>7} | {'p_GUE':>7} | {'winner':>6} | {'call':>10}")
    print(f"  {'-'*95}")

    results = []
    for L in L_vals:
        torus = EisensteinTorus(L)
        lmod  = L % 6
        pred  = "GUE" if lmod == 0 else "GOE"
        t0 = time.time()

        betas, kgs, kus, pgs, pus = [], [], [], [], []

        if needs_seed:
            seeds = range(n_seeds)
        else:
            seeds = [None]  # geometric spinors are deterministic

        for seed in seeds:
            if needs_seed:
                u, v, omega = spinor_fn(torus, seed)
            else:
                u, v, omega = spinor_fn(torus)

            M = build_M(torus, u, v, omega, Phi, use_resonance=use_resonance)
            eigs = np.linalg.eigvalsh(M)
            sp = get_spacings(eigs)
            if len(sp) < 5: continue

            betas.append(fit_beta(sp))
            kg, pg = ks(sp, goe_cdf)
            ku, pu = ks(sp, gue_cdf)
            kgs.append(kg); kus.append(ku)
            pgs.append(pg); pus.append(pu)

        b   = float(np.nanmean(betas))
        kg  = float(np.nanmean(kgs))
        ku  = float(np.nanmean(kus))
        pg  = float(np.nanmean(pgs))
        pu  = float(np.nanmean(pus))
        win = "GUE" if ku < kg else "GOE"
        correct = (pred == win)
        call = "correct" if correct else "wrong"
        dt  = time.time() - t0

        print(f"  {L:>4} | {torus.num_nodes:>5} | {lmod:>4} | {pred:>7} | "
              f"{b:>6.3f} | {kg:>7.4f} | {ku:>7.4f} | "
              f"{pg:>7.4f} | {pu:>7.4f} | {win:>6} | {call}  ({dt:.0f}s)")

        results.append(dict(L=L, N=torus.num_nodes, Lmod6=lmod, pred=pred,
                            beta=b, ks_goe=kg, ks_gue=ku, p_goe=pg, p_gue=pu,
                            winner=win, correct=correct))

    n_correct = sum(r['correct'] for r in results)
    print(f"\n  Prediction accuracy: {n_correct}/{len(results)}")
    return results

def run_phi_comparison(L, spinor_fn, label, needs_seed=False, n_seeds=5,
                       use_resonance=False):
    """Phi sweep at fixed L."""
    print(f"\n  Phi sweep at L={L} -- {label}")
    print(f"  {'Phi':>7} | {'KS_GOE':>7} | {'KS_GUE':>7} | {'p_GOE':>7} | {'p_GUE':>7} | {'winner':>6}")
    print(f"  {'-'*55}")
    torus = EisensteinTorus(L)

    for Phi in [0.0, 1/12, 1/6, 1/4, 1/3]:
        kgs, kus, pgs, pus = [], [], [], []

        if needs_seed:
            seeds = range(n_seeds)
        else:
            seeds = [None]

        for seed in seeds:
            if needs_seed:
                u, v, omega = spinor_fn(torus, seed)
            else:
                u, v, omega = spinor_fn(torus)

            M = build_M(torus, u, v, omega, Phi, use_resonance=use_resonance)
            eigs = np.linalg.eigvalsh(M)
            sp = get_spacings(eigs)
            if len(sp) < 5: continue
            kg, pg = ks(sp, goe_cdf)
            ku, pu = ks(sp, gue_cdf)
            kgs.append(kg); kus.append(ku)
            pgs.append(pg); pus.append(pu)

        kg = float(np.nanmean(kgs))
        ku = float(np.nanmean(kus))
        pg = float(np.nanmean(pgs))
        pu = float(np.nanmean(pus))
        win = "GUE" if ku < kg else "GOE"
        flag = " <- P-gate" if abs(Phi - 1/6) < 0.001 else ""
        print(f"  {Phi:>7.4f} | {kg:>7.4f} | {ku:>7.4f} | {pg:>7.4f} | {pu:>7.4f} | {win:>6}{flag}")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    OUTPUT = "C:/Users/selin/merkabit_results/peierls_flux_corrected/"
    os.makedirs(OUTPUT, exist_ok=True)

    L_VALS = [9, 12, 15, 18, 21, 24]
    PHI    = 1/6

    # ---- 1. GEOMETRIC spinors (exact Analysis 19 construction) ----
    r_geo = run_test(L_VALS, PHI,
                     assign_spinors_geometric,
                     "GEOMETRIC (exact Analysis 19: u from (a,b), v = time-reverse)",
                     needs_seed=False, use_resonance=False)

    # ---- 2. GEOMETRIC with resonance (for comparison) ----
    r_geo_res = run_test(L_VALS, PHI,
                         assign_spinors_geometric,
                         "GEOMETRIC + RESONANCE",
                         needs_seed=False, use_resonance=True)

    # ---- 3. CASCADE (Z3 phases on random spinors -- already falsified) ----
    r_cascade = run_test(L_VALS, PHI,
                         assign_spinors_cascade,
                         "CASCADE (Z3 phases on RANDOM spinors)",
                         needs_seed=True, n_seeds=5, use_resonance=False)

    # ---- 4. RANDOM control ----
    r_rand = run_test(L_VALS, PHI,
                      assign_spinors_random,
                      "RANDOM (control)",
                      needs_seed=True, n_seeds=5, use_resonance=False)

    # ---- Phi sweep for geometric spinors ----
    print(f"\n{'='*70}")
    print("PHI SWEEP -- geometric spinors, no resonance")
    print(f"{'='*70}")
    for L in [12, 18, 24]:
        run_phi_comparison(L, assign_spinors_geometric,
                           f"GEOMETRIC L={L}",
                           needs_seed=False, use_resonance=False)

    # ---- Save summary ----
    fname = os.path.join(OUTPUT, "commensurability_geometric.txt")
    with open(fname, 'w') as f:
        f.write("COMMENSURABILITY TEST -- GEOMETRIC SPINORS\n")
        f.write("u[i] = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]\n")
        f.write("v[i] = [-conj(u[1]), conj(u[0])]  (SU(2) time-reverse)\n")
        f.write(f"Phi=1/6, prediction: GUE iff 6|L\n\n")
        for label, results in [("GEOMETRIC (no res)", r_geo),
                                ("GEOMETRIC (with res)", r_geo_res),
                                ("CASCADE (random base)", r_cascade),
                                ("RANDOM", r_rand)]:
            f.write(f"\n[{label}]\n")
            f.write("L N L%6 pred beta KS_GOE KS_GUE p_GOE p_GUE winner correct\n")
            for r in results:
                f.write(f"{r['L']} {r['N']} {r['Lmod6']} {r['pred']} "
                        f"{r['beta']:.4f} {r['ks_goe']:.4f} {r['ks_gue']:.4f} "
                        f"{r['p_goe']:.4f} {r['p_gue']:.4f} {r['winner']} {r['correct']}\n")
            n = sum(x['correct'] for x in results)
            f.write(f"Accuracy: {n}/{len(results)}\n")

    print(f"\nSaved: {fname}")
    print("\n" + "="*70)
    print("KEY QUESTIONS ANSWERED:")
    print("  1. Does GEOMETRIC reproduce Analysis 19's GUE signal?")
    print("     -> If yes: the lattice geometry (not random seed) drives GUE")
    print("  2. Does commensurability (6|L) hold for geometric spinors?")
    print("     -> 5/6 or 6/6 correct = confirmed; 3/6 = falsified")
    print("  3. Does resonance matter for geometric spinors?")
    print("     -> Compare GEOMETRIC vs GEOMETRIC+RESONANCE")
    print("="*70)
