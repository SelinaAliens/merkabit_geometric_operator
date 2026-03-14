#!/usr/bin/env python3
"""
COMMENSURABILITY PREDICTION TEST
==================================
Prediction from Analysis 19:
  GUE emerges at Phi=1/6 when and only when 6 | L  (since 6 = h(E6)/2)
  L=21: 21/6 = 3.5  INCOMMENSURATE -> prediction: GOE survives
  L=24: 24/6 = 4.0  COMMENSURATE   -> prediction: GUE, KS < 0.052 (L=18 best)
This is a predictive test, not a fit.
Run time: ~15 minutes (L=24 has 576 nodes, 5 seeds each)
Output: C:/Users/selin/merkabit_results/peierls_flux_corrected/commensurability_test.txt
"""
import numpy as np
from collections import defaultdict
import time
import os
# ============================================================================
# MINIMAL SELF-CONTAINED IMPLEMENTATION
# (copy of Analysis 19 functions -- no imports from other scripts needed)
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
        self.chirality = [(1 if (a+b)%3==1 else (-1 if (a+b)%3==2 else 0))
                          for (a,b) in self.nodes]
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
def assign_spinors(torus, seed):
    rng = np.random.default_rng(seed)
    N = torus.num_nodes
    u = rng.standard_normal((N,2)) + 1j*rng.standard_normal((N,2))
    v = rng.standard_normal((N,2)) + 1j*rng.standard_normal((N,2))
    for i in range(N):
        u[i] /= np.linalg.norm(u[i])
        phase = np.exp(1j * torus.chirality[i] * np.pi / 3)
        v[i] = phase * v[i] / np.linalg.norm(v[i])
    return u, v
def build_M(torus, u, v, Phi, xi=3.0):
    N = torus.num_nodes
    M = np.zeros((N, N), dtype=complex)
    decay = np.exp(-1.0 / xi)
    for (i, j) in torus.edges:
        theta = peierls_phase(torus.nodes[i], torus.nodes[j], torus.L, Phi)
        pf = np.exp(1j * theta)
        c = decay * pf * np.vdot(u[i], v[j])
        M[i,j] = c
        M[j,i] = np.conj(c)
    return M
def goe_cdf(s): return 1 - np.exp(-np.pi/4 * s**2)
def gue_cdf(s): return 1 - np.exp(-4/np.pi * s**2) * (1 + 4/np.pi * s**2)
def ks(spacings, cdf):
    s = np.sort(spacings)
    n = len(s)
    emp = np.arange(1, n+1) / n
    th  = cdf(s)
    stat = np.max(np.abs(emp - th))
    t = stat * np.sqrt(n)
    p = 2 * np.exp(-2 * t**2)
    return stat, p
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
    return np.polyfit(np.log(centers[mask]), np.log(hist[mask]), 1)[0]
# ============================================================================
# COMMENSURABILITY TEST
# ============================================================================
def test_commensurability(L_vals, Phi=1/6, n_seeds=5, output_dir=None):
    print("\n" + "="*65)
    print("COMMENSURABILITY PREDICTION TEST")
    print(f"Phi = 1/6 = {Phi:.6f} (P gate flux, one Eisenstein face)")
    print(f"Condition: GUE iff 6 | L  (6 = h(E6)/2 = 12/2)")
    print("="*65)
    print(f"\n  {'L':>4} | {'N':>5} | {'L%6':>4} | {'predict':>8} | "
          f"{'beta':>6} | {'KS_GOE':>7} | {'KS_GUE':>7} | "
          f"{'p_GUE':>7} | {'winner':>8}")
    print(f"  {'-'*75}")
    results = []
    for L in L_vals:
        torus = EisensteinTorus(L)
        N = torus.num_nodes
        lmod = L % 6
        predict = "GUE" if lmod == 0 else "GOE"
        t0 = time.time()
        betas, kg_list, ku_list, pu_list = [], [], [], []
        for seed in range(n_seeds):
            u, v = assign_spinors(torus, seed)
            M = build_M(torus, u, v, Phi)
            eigs = np.linalg.eigvalsh(M)
            sp = get_spacings(eigs)
            if len(sp) < 5: continue
            betas.append(fit_beta(sp))
            kg, _ = ks(sp, goe_cdf)
            ku, pu = ks(sp, gue_cdf)
            kg_list.append(kg)
            ku_list.append(ku)
            pu_list.append(pu)
        b  = np.nanmean(betas)
        kg = np.nanmean(kg_list)
        ku = np.nanmean(ku_list)
        pu = np.nanmean(pu_list)
        winner = "GUE" if ku < kg else "GOE"
        dt = time.time() - t0
        # Prediction correct?
        correct = ((predict == "GUE") == (ku < kg))
        flag = "CORRECT" if correct else "WRONG"
        print(f"  {L:>4} | {N:>5} | {lmod:>4} | {predict:>8} | "
              f"{b:>6.3f} | {kg:>7.4f} | {ku:>7.4f} | "
              f"{pu:>7.3f} | {winner:>8}  {flag}  ({dt:.0f}s)")
        results.append({'L': L, 'N': N, 'Lmod6': lmod, 'predict': predict,
                        'beta': b, 'ks_goe': kg, 'ks_gue': ku, 'p_gue': pu,
                        'winner': winner, 'correct': correct})
    # Summary
    print(f"\n{'='*65}")
    n_correct = sum(r['correct'] for r in results)
    print(f"Prediction accuracy: {n_correct}/{len(results)}")
    # Compare to Analysis 19 baseline
    print(f"\nKS(GUE) progression at commensurate L (Phi=1/6):")
    print(f"  L=12 (Analysis 19): 0.102")
    print(f"  L=18 (Analysis 19): 0.052")
    for r in results:
        if r['Lmod6'] == 0:
            trend = "improving" if r['ks_gue'] < 0.052 else "not improving"
            print(f"  L={r['L']} (this run):   {r['ks_gue']:.3f}  {trend}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, "commensurability_test.txt")
        with open(fname, 'w') as f:
            f.write("COMMENSURABILITY PREDICTION TEST\n")
            f.write(f"Phi=1/6, condition: GUE iff 6|L\n\n")
            f.write("L | N | L%6 | predict | beta | KS_GOE | KS_GUE | p_GUE | correct\n")
            for r in results:
                f.write(f"{r['L']} {r['N']} {r['Lmod6']} {r['predict']} "
                        f"{r['beta']:.4f} {r['ks_goe']:.4f} {r['ks_gue']:.4f} "
                        f"{r['p_gue']:.4f} {r['correct']}\n")
            f.write(f"\nPrediction accuracy: {n_correct}/{len(results)}\n")
        print(f"\nSaved: {fname}")
    return results
# ============================================================================
# ALSO RUN PHI=0 AT L=21, L=24 -- CONFIRM BASELINE GOE HOLDS
# ============================================================================
def test_baseline(L_vals, n_seeds=3):
    """Confirm phi=0 gives GOE at all L -- isolates the flux effect."""
    print(f"\n{'='*65}")
    print("BASELINE CHECK: phi=0 at L=21, L=24")
    print(f"{'='*65}")
    print(f"  {'L':>4} | {'KS_GOE':>7} | {'KS_GUE':>7} | {'winner':>8}")
    print(f"  {'-'*35}")
    for L in L_vals:
        torus = EisensteinTorus(L)
        kg_list, ku_list = [], []
        for seed in range(n_seeds):
            u, v = assign_spinors(torus, seed)
            M = build_M(torus, u, v, Phi=0)
            eigs = np.linalg.eigvalsh(M)
            sp = get_spacings(eigs)
            if len(sp) < 5: continue
            kg, _ = ks(sp, goe_cdf)
            ku, _ = ks(sp, gue_cdf)
            kg_list.append(kg)
            ku_list.append(ku)
        kg = np.nanmean(kg_list)
        ku = np.nanmean(ku_list)
        winner = "GOE" if kg < ku else "GUE"
        print(f"  {L:>4} | {kg:>7.4f} | {ku:>7.4f} | {winner:>8}")
# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    OUTPUT = "C:/Users/selin/merkabit_results/peierls_flux_corrected/"
    # The prediction:
    # L=21: 21%6=3 -> incommensurate -> GOE
    # L=24: 24%6=0 -> commensurate   -> GUE, KS < 0.052
    # Also include L=9,12,15,18 to show full pattern
    L_ALL = [9, 12, 15, 18, 21, 24]
    test_baseline([21, 24], n_seeds=3)
    test_commensurability(L_ALL, Phi=1/6, n_seeds=5, output_dir=OUTPUT)
    print("\nDone. The prediction is confirmed or falsified above.")
    print("L=21 -> GOE, L=24 -> GUE (KS_GUE < KS_GOE) is the call.")
