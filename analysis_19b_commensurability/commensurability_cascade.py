#!/usr/bin/env python3
"""
COMMENSURABILITY TEST -- CASCADE SPINORS
=========================================
Corrected spinor construction from gate cascade order.
Previous error: random spinors destroy cascade phase correlations.
Analysis 19 error: chirality * pi/3 approximates but misses Z3 structure.
Correct construction:
  Sublattice s in {0,1,2} maps to gate step s*4 in the 12-step ouroboros.
  Accumulated phase at step k: theta_k = k * pi/6 (one Coxeter step)

  s=0: gate step 0  -> phase 0           = 1
  s=1: gate step 4  -> phase 4*pi/6      = exp(i * 2pi/3)   (cube root of unity)
  s=2: gate step 8  -> phase 8*pi/6      = exp(i * 4pi/3)   (cube root of unity)
These are the Z3 cube roots -- the correct group structure of Z[omega].
The P gate is the gate that separates u and v with opposite phase.
It acts coherently only on spinors that carry the correct cascade phase.
Random spinors have no cascade memory -> Peierls flux has nothing to resonate with.
Prediction (unchanged):
  GUE iff 6|L   (6 = h(E6)/2)
  L=21: 21%6=3 -> GOE
  L=24: 24%6=0 -> GUE, KS(GUE) < 0.052
Output: C:/Users/selin/merkabit_results/peierls_flux_corrected/commensurability_cascade.txt
"""
import numpy as np
from collections import defaultdict
import time
import os
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
        # sublattice: (a+b) % 3 in {0,1,2}
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
# ============================================================================
# CASCADE SPINORS -- CORRECT CONSTRUCTION
# ============================================================================
def assign_spinors_cascade(torus, seed=42):
    """
    Spinors assigned by gate cascade position.
    The 12-step ouroboros has three sublattice positions equally spaced:
      s=0: step 0  -> phase = 0
      s=1: step 4  -> phase = 4 * pi/6 = 2*pi/3
      s=2: step 8  -> phase = 8 * pi/6 = 4*pi/3
    These are the primitive cube roots of unity: 1, omega, omega^2
    where omega = exp(2*pi*i/3) -- exactly the Eisenstein lattice generator.
    v[i] carries this phase relative to u[i].
    The P gate at phi=1/6 then acts coherently across the sublattice,
    because each node's v-spinor is already at the correct cascade phase.
    """
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    u = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    v = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        s = torus.sublattice[i]          # 0, 1, 2
        gate_step = s * 4                # 0, 4, 8  (equally spaced in 12-step cycle)
        theta = gate_step * np.pi / 6   # 0, 2pi/3, 4pi/3  (cube roots of unity)
        v[i] = np.exp(1j * theta) * v[i] / np.linalg.norm(v[i])
    return u, v
def assign_spinors_analysis19(torus, seed=42):
    """
    Original Analysis 19 construction for direct comparison.
    phase = chirality * pi/3 where chirality in {-1, 0, +1}
    """
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    chirality = [1 if s==1 else (-1 if s==2 else 0) for s in torus.sublattice]
    u = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    v = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        theta = chirality[i] * np.pi / 3
        v[i] = np.exp(1j * theta) * v[i] / np.linalg.norm(v[i])
    return u, v
def assign_spinors_random(torus, seed=42):
    """Random spinors -- no cascade structure. Should give GOE always."""
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    u = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    v = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        v[i] /= np.linalg.norm(v[i])
    return u, v
# ============================================================================
# PEIERLS + M CONSTRUCTION
# ============================================================================
def peierls_phase(site_i, site_j, L, Phi):
    a_i, b_i = site_i
    a_j, b_j = site_j
    da = a_j - a_i
    db = b_j - b_i
    if da >  L//2: da -= L
    if da < -L//2: da += L
    if db >  L//2: db -= L
    if db < -L//2: db += L
    return 2 * np.pi * Phi * (a_i + da/2.0) * db
def build_M(torus, u, v, Phi, xi=3.0):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    for (i, j) in torus.edges:
        theta = peierls_phase(torus.nodes[i], torus.nodes[j], torus.L, Phi)
        pf = np.exp(1j * theta)
        c = decay * pf * np.vdot(u[i], v[j])
        M[i, j] = c
        M[j, i] = np.conj(c)
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
def pos_frac(eigs):
    return float(np.sum(eigs > 1e-8 * np.max(np.abs(eigs))) / len(eigs))
# ============================================================================
# MAIN TEST
# ============================================================================
def run_test(L_vals, Phi, spinor_fn, label, n_seeds=5):
    """Run commensurability test for one spinor construction."""
    print(f"\n{'='*70}")
    print(f"SPINOR CONSTRUCTION: {label}")
    print(f"Phi = {Phi:.6f}  ({Phi} = {Phi:.4f})")
    print(f"Prediction: GUE iff 6|L")
    print(f"{'='*70}")
    print(f"  {'L':>4} | {'N':>5} | {'L%6':>4} | {'predict':>7} | "
          f"{'beta':>6} | {'KS_GOE':>7} | {'KS_GUE':>7} | "
          f"{'p_GUE':>7} | {'winner':>6} | {'call':>10}")
    print(f"  {'-'*80}")
    results = []
    for L in L_vals:
        torus = EisensteinTorus(L)
        lmod  = L % 6
        pred  = "GUE" if lmod == 0 else "GOE"
        t0 = time.time()
        betas, kgs, kus, pus = [], [], [], []
        for seed in range(n_seeds):
            u, v = spinor_fn(torus, seed)
            M    = build_M(torus, u, v, Phi)
            eigs = np.linalg.eigvalsh(M)
            sp   = get_spacings(eigs)
            if len(sp) < 5: continue
            betas.append(fit_beta(sp))
            kg, _  = ks(sp, goe_cdf)
            ku, pu = ks(sp, gue_cdf)
            kgs.append(kg); kus.append(ku); pus.append(pu)
        b   = float(np.nanmean(betas))
        kg  = float(np.nanmean(kgs))
        ku  = float(np.nanmean(kus))
        pu  = float(np.nanmean(pus))
        win = "GUE" if ku < kg else "GOE"
        correct = (pred == win)
        call = "correct" if correct else "wrong"
        dt  = time.time() - t0
        print(f"  {L:>4} | {torus.num_nodes:>5} | {lmod:>4} | {pred:>7} | "
              f"{b:>6.3f} | {kg:>7.4f} | {ku:>7.4f} | "
              f"{pu:>7.3f} | {win:>6} | {call}  ({dt:.0f}s)")
        results.append(dict(L=L, N=torus.num_nodes, Lmod6=lmod, pred=pred,
                            beta=b, ks_goe=kg, ks_gue=ku, p_gue=pu,
                            winner=win, correct=correct))
    n_correct = sum(r['correct'] for r in results)
    print(f"\n  Prediction accuracy: {n_correct}/{len(results)}")
    return results
def run_phi_comparison(L, spinor_fn, label, n_seeds=5):
    """
    At fixed L, compare phi=0 vs phi=1/6.
    Confirms the flux effect is real and not a spinor artifact.
    """
    print(f"\n  Phi sweep at L={L} -- {label}")
    print(f"  {'Phi':>7} | {'KS_GOE':>7} | {'KS_GUE':>7} | {'winner':>6}")
    print(f"  {'-'*35}")
    torus = EisensteinTorus(L)
    for Phi in [0.0, 1/12, 1/6, 1/4]:
        kgs, kus = [], []
        for seed in range(n_seeds):
            u, v = spinor_fn(torus, seed)
            M    = build_M(torus, u, v, Phi)
            eigs = np.linalg.eigvalsh(M)
            sp   = get_spacings(eigs)
            if len(sp) < 5: continue
            kg, _ = ks(sp, goe_cdf)
            ku, _ = ks(sp, gue_cdf)
            kgs.append(kg); kus.append(ku)
        kg = float(np.nanmean(kgs))
        ku = float(np.nanmean(kus))
        win = "GUE" if ku < kg else "GOE"
        flag = " <- P-gate" if abs(Phi - 1/6) < 0.001 else ""
        print(f"  {Phi:>7.4f} | {kg:>7.4f} | {ku:>7.4f} | {win:>6}{flag}")
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    OUTPUT = "C:/Users/selin/merkabit_results/peierls_flux_corrected/"
    os.makedirs(OUTPUT, exist_ok=True)
    L_VALS = [9, 12, 15, 18, 21, 24]
    PHI    = 1/6    # P gate flux -- one Eisenstein face angle
    # ---- Three spinor constructions side by side ----
    # 1. Correct cascade (Z3 cube roots of unity)
    r_cascade = run_test(L_VALS, PHI,
                         assign_spinors_cascade,
                         "CASCADE (Z3: phases 0, 2pi/3, 4pi/3)")
    # 2. Analysis 19 original (chirality * pi/3 -- approximate)
    r_a19 = run_test(L_VALS, PHI,
                     assign_spinors_analysis19,
                     "ANALYSIS 19 (chirality * pi/3 -- approximate)")
    # 3. Random (control -- should give GOE everywhere)
    r_rand = run_test(L_VALS, PHI,
                      assign_spinors_random,
                      "RANDOM (control -- no cascade structure)")
    # ---- Phi sweep at L=18 and L=24 for cascade spinors ----
    print(f"\n{'='*70}")
    print("PHI SWEEP -- cascade spinors, confirming flux drives transition")
    print(f"{'='*70}")
    for L in [18, 24]:
        run_phi_comparison(L, assign_spinors_cascade,
                           f"CASCADE spinors L={L}")
    # ---- Save summary ----
    fname = os.path.join(OUTPUT, "commensurability_cascade.txt")
    with open(fname, 'w') as f:
        f.write("COMMENSURABILITY TEST -- CASCADE SPINORS\n")
        f.write(f"Phi=1/6, prediction: GUE iff 6|L\n\n")
        for label, results in [("CASCADE", r_cascade),
                                ("ANALYSIS19", r_a19),
                                ("RANDOM", r_rand)]:
            f.write(f"\n[{label}]\n")
            f.write("L N L%6 pred beta KS_GOE KS_GUE p_GUE winner correct\n")
            for r in results:
                f.write(f"{r['L']} {r['N']} {r['Lmod6']} {r['pred']} "
                        f"{r['beta']:.4f} {r['ks_goe']:.4f} {r['ks_gue']:.4f} "
                        f"{r['p_gue']:.4f} {r['winner']} {r['correct']}\n")
            n = sum(x['correct'] for x in results)
            f.write(f"Accuracy: {n}/{len(results)}\n")
    print(f"\nSaved: {fname}")
    print("\n" + "="*70)
    print("WHAT TO LOOK FOR:")
    print("  CASCADE correct 5/6 or 6/6 -> commensurability confirmed")
    print("  CASCADE correct 3/6        -> commensurability not the rule")
    print("  ANALYSIS19 matches CASCADE -> original construction was adequate")
    print("  RANDOM correct  3/6        -> confirms random is chance level")
    print("  Phi sweep: GUE at phi=1/6 only -> flux effect is real")
    print("="*70)
