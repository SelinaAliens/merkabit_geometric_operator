#!/usr/bin/env python3
"""
COMMENSURABILITY TEST -- FINAL (EXACT ANALYSIS 19 PIPELINE)
============================================================
Uses the EXACT spectral analysis pipeline from peierls_flux.py:
  1. Positive wing extraction (threshold_pct=20)
  2. Polynomial unfolding (degree 10)
  3. scipy.stats.kstest with numerically integrated Wigner CDF

Tests four spinor constructions:
  A. GEOMETRIC (u from lattice coords, v = SU(2) time-reverse) -- EXACT Analysis 19
  B. CASCADE (Z3 phases on random base)
  C. RANDOM (control)

At L=9,12,15,18,21,24 with Phi=1/6, no resonance.

Prediction: GUE iff 6|L (6 = h(E6)/2)
"""
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
from collections import defaultdict
import time
import os

OMEGA_EISEN = np.exp(2j * np.pi / 3)

# ============================================================================
# EISENSTEIN TORUS (exact copy from peierls_flux.py)
# ============================================================================
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

class EisensteinTorus:
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
        self.sublattice = [(a + b) % 3 for (a, b) in self.nodes]
        self.chirality = []
        for s in self.sublattice:
            if s == 0: self.chirality.append(0)
            elif s == 1: self.chirality.append(+1)
            else: self.chirality.append(-1)

# ============================================================================
# SPINOR CONSTRUCTIONS
# ============================================================================
def assign_spinors_geometric(torus):
    """Fully deterministic geometric spinors (exact Analysis 19)."""
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

def assign_spinors_cascade(torus, seed=42):
    """Z3 phases on random base spinors."""
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

def assign_spinors_random(torus, seed=42):
    """Random spinors (control)."""
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
# M CONSTRUCTION (exact copy from peierls_flux.py build_M_peierls)
# ============================================================================
def build_M(torus, u, v, omega, Phi, xi=3.0, use_resonance=False):
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
# SPECTRAL ANALYSIS (exact copy from peierls_flux.py)
# ============================================================================
def unfold_spectrum(eigenvalues, poly_degree=10):
    """Polynomial unfolding of eigenvalues."""
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
    """Extract positive eigenvalues above threshold percentile."""
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) < 4:
        return pos_eigs
    cutoff = np.percentile(pos_eigs, threshold_pct)
    return pos_eigs[pos_eigs > cutoff]

def make_wigner_cdf(beta):
    """Construct Wigner surmise CDF for given beta via numerical integration."""
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
    """Fit level repulsion exponent beta from small-s behavior."""
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

def ks_tests(spacings):
    """KS tests against GOE, GUE, Poisson using scipy.stats.kstest."""
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {k: (1.0, 0.0) for k in ['goe', 'gue', 'poi']}
    results = {}
    for name, beta_val in [('goe', 1), ('gue', 2)]:
        cdf = make_wigner_cdf(beta_val)
        ks_stat, p = stats.kstest(pos, cdf)
        results[name] = (float(ks_stat), float(p))
    ks_stat, p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (float(ks_stat), float(p))
    return results

def analyze_spectrum(evals):
    """Full Analysis 19 pipeline: positive wing + unfold + KS."""
    evals = np.sort(np.real(evals))
    pos_frac = float(np.sum(evals > 0) / len(evals))

    # Positive wing analysis (THE key metric from Analysis 19)
    wing = extract_positive_wing(evals)
    if len(wing) > 10:
        _, spacings_wing = unfold_spectrum(wing)
        beta_wing, _ = fit_beta(spacings_wing)
        ks_wing = ks_tests(spacings_wing)
    else:
        beta_wing = 0.0
        ks_wing = {'goe': (1.0, 0.0), 'gue': (1.0, 0.0), 'poi': (1.0, 0.0)}

    return {
        'pos_frac': pos_frac,
        'beta': beta_wing,
        'ks_goe': ks_wing['goe'][0],
        'p_goe': ks_wing['goe'][1],
        'ks_gue': ks_wing['gue'][0],
        'p_gue': ks_wing['gue'][1],
        'n_wing': len(wing) if len(wing) > 10 else 0,
    }

# ============================================================================
# MAIN TEST
# ============================================================================
def run_test(L_vals, Phi, spinor_fn, label, needs_seed=False, n_seeds=5):
    print(f"\n{'='*80}")
    print(f"SPINOR: {label}")
    print(f"Phi = {Phi:.6f}, no resonance, POSITIVE WING + POLYNOMIAL UNFOLDING")
    print(f"Prediction: GUE iff 6|L")
    print(f"{'='*80}")
    print(f"  {'L':>4} | {'N':>5} | {'wing':>5} | {'L%6':>4} | {'pred':>5} | "
          f"{'beta':>6} | {'KS_GOE':>7} | {'KS_GUE':>7} | "
          f"{'p_GOE':>7} | {'p_GUE':>7} | {'win':>4} | {'call':>8}")
    print(f"  {'-'*100}")

    results = []
    for L in L_vals:
        torus = EisensteinTorus(L)
        lmod = L % 6
        pred = "GUE" if lmod == 0 else "GOE"
        t0 = time.time()

        all_stats = []
        if needs_seed:
            seeds = range(n_seeds)
        else:
            seeds = [None]

        for seed in seeds:
            if needs_seed:
                u, v, omega = spinor_fn(torus, seed)
            else:
                u, v, omega = spinor_fn(torus)
            M = build_M(torus, u, v, omega, Phi, use_resonance=False)
            eigs = np.linalg.eigvalsh(M)
            sa = analyze_spectrum(eigs)
            if sa['n_wing'] > 0:
                all_stats.append(sa)

        if not all_stats:
            print(f"  {L:>4} | {torus.num_nodes:>5} |     0 | {lmod:>4} | {pred:>5} | "
                  f"{'N/A':>6} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | "
                  f"{'N/A':>4} | {'skip':>8}")
            continue

        b  = float(np.mean([s['beta'] for s in all_stats]))
        kg = float(np.mean([s['ks_goe'] for s in all_stats]))
        ku = float(np.mean([s['ks_gue'] for s in all_stats]))
        pg = float(np.mean([s['p_goe'] for s in all_stats]))
        pu = float(np.mean([s['p_gue'] for s in all_stats]))
        nw = int(np.mean([s['n_wing'] for s in all_stats]))

        win = "GUE" if ku < kg else "GOE"
        correct = (pred == win)
        call = "correct" if correct else "WRONG"
        dt = time.time() - t0

        print(f"  {L:>4} | {torus.num_nodes:>5} | {nw:>5} | {lmod:>4} | {pred:>5} | "
              f"{b:>6.3f} | {kg:>7.4f} | {ku:>7.4f} | "
              f"{pg:>7.4f} | {pu:>7.4f} | {win:>4} | {call:>8}  ({dt:.0f}s)")

        results.append(dict(L=L, N=torus.num_nodes, n_wing=nw, Lmod6=lmod,
                            pred=pred, beta=b, ks_goe=kg, ks_gue=ku,
                            p_goe=pg, p_gue=pu, winner=win, correct=correct))

    n_correct = sum(r['correct'] for r in results)
    print(f"\n  Prediction accuracy: {n_correct}/{len(results)}")
    return results

def run_phi_sweep(L, spinor_fn, label, needs_seed=False, n_seeds=5):
    print(f"\n  Phi sweep L={L} -- {label} (positive wing + unfolding)")
    print(f"  {'Phi':>7} | {'KS_GOE':>7} | {'KS_GUE':>7} | {'p_GOE':>7} | {'p_GUE':>7} | {'win':>4}")
    print(f"  {'-'*55}")
    torus = EisensteinTorus(L)
    for Phi in [0.0, 1/12, 1/6, 1/4, 1/3]:
        all_stats = []
        if needs_seed:
            seeds = range(n_seeds)
        else:
            seeds = [None]
        for seed in seeds:
            if needs_seed:
                u, v, omega = spinor_fn(torus, seed)
            else:
                u, v, omega = spinor_fn(torus)
            M = build_M(torus, u, v, omega, Phi, use_resonance=False)
            eigs = np.linalg.eigvalsh(M)
            sa = analyze_spectrum(eigs)
            if sa['n_wing'] > 0:
                all_stats.append(sa)
        if not all_stats:
            continue
        kg = float(np.mean([s['ks_goe'] for s in all_stats]))
        ku = float(np.mean([s['ks_gue'] for s in all_stats]))
        pg = float(np.mean([s['p_goe'] for s in all_stats]))
        pu = float(np.mean([s['p_gue'] for s in all_stats]))
        win = "GUE" if ku < kg else "GOE"
        flag = " <- P-gate" if abs(Phi - 1/6) < 0.001 else ""
        print(f"  {Phi:>7.4f} | {kg:>7.4f} | {ku:>7.4f} | {pg:>7.4f} | {pu:>7.4f} | {win:>4}{flag}")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    OUTPUT = "C:/Users/selin/merkabit_results/peierls_flux_corrected/"
    os.makedirs(OUTPUT, exist_ok=True)

    L_VALS = [9, 12, 15, 18, 21, 24]
    PHI = 1/6

    print("ANALYSIS 19 PIPELINE: positive wing (>20th pct) + poly unfold + scipy KS")
    print("This is the EXACT pipeline that produced KS(GUE)=0.052 at L=18")

    # 1. GEOMETRIC (the construction that originally produced GUE signal)
    r_geo = run_test(L_VALS, PHI, assign_spinors_geometric,
                     "GEOMETRIC (u from (a,b), v = time-reverse)",
                     needs_seed=False)

    # 2. CASCADE (Z3 phases on random)
    r_cas = run_test(L_VALS, PHI, assign_spinors_cascade,
                     "CASCADE (Z3 phases on random base)",
                     needs_seed=True, n_seeds=5)

    # 3. RANDOM (control)
    r_rnd = run_test(L_VALS, PHI, assign_spinors_random,
                     "RANDOM (control)",
                     needs_seed=True, n_seeds=5)

    # Phi sweeps
    print(f"\n{'='*80}")
    print("PHI SWEEPS (positive wing + unfolding)")
    print(f"{'='*80}")
    for L in [12, 18, 24]:
        run_phi_sweep(L, assign_spinors_geometric,
                      f"GEOMETRIC L={L}", needs_seed=False)
    for L in [18, 24]:
        run_phi_sweep(L, assign_spinors_cascade,
                      f"CASCADE L={L}", needs_seed=True, n_seeds=5)

    # Save
    fname = os.path.join(OUTPUT, "commensurability_final.txt")
    with open(fname, 'w') as f:
        f.write("COMMENSURABILITY FINAL -- EXACT ANALYSIS 19 PIPELINE\n")
        f.write("Positive wing (>20th pct) + polynomial unfolding + scipy KS\n")
        f.write(f"Phi=1/6, prediction: GUE iff 6|L\n\n")
        for label, results in [("GEOMETRIC", r_geo),
                                ("CASCADE", r_cas),
                                ("RANDOM", r_rnd)]:
            f.write(f"\n[{label}]\n")
            f.write("L N wing L%6 pred beta KS_GOE KS_GUE p_GOE p_GUE winner correct\n")
            for r in results:
                f.write(f"{r['L']} {r['N']} {r['n_wing']} {r['Lmod6']} {r['pred']} "
                        f"{r['beta']:.4f} {r['ks_goe']:.4f} {r['ks_gue']:.4f} "
                        f"{r['p_goe']:.4f} {r['p_gue']:.4f} {r['winner']} {r['correct']}\n")
            n = sum(x['correct'] for x in results)
            f.write(f"Accuracy: {n}/{len(results)}\n")
    print(f"\nSaved: {fname}")
