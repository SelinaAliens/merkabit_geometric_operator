#!/usr/bin/env python3
"""
COHERENCE-EIGENVALUE MAPPING — Analysis 21
============================================
Test: do eigenmode structural properties correlate with eigenvalue sign?

Framework prediction (chaotic boundary dynamics):
  - Positive eigenvalues: modes tilted toward COHERENCE (standing wave)
  - Negative eigenvalues: modes tilted toward DECOHERENCE (dispersed)
  - Near-zero eigenvalues: chaotic BOUNDARY

If correct, at least one of these observables should show a sigmoid
(or monotonic trend) when plotted against eigenvalue:

  1. IPR (localization) — coherent modes should be delocalized
  2. Sublattice polarization — coherent modes should project onto sub-0
  3. Phase alignment — coherent modes should match the lattice Hopf phase
  4. Radial concentration — coherent vs decoherent may differ in radial profile
  5. Neighbor coupling strength — coherent modes should maximize M coupling

TECHNICAL NOTE: <u_i|v_i> = 0 identically (Hopf orthogonality), so the
on-site overlap vector is trivially zero. The coherence signal must be in
the INTER-SITE structure of eigenvectors.

Operator: M on EisensteinTorus, geometric Hopf spinors, Phi=1/6, no resonance.
"""
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"
os.makedirs(OUT, exist_ok=True)

OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1)]
XI = 3.0

# ============================================================================
# EISENSTEIN TORUS (exact copy from montgomery_comparison.py)
# ============================================================================
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
    r_vals = np.zeros(N)  # store radial coordinate
    theta_vals = np.zeros(N)  # store angular phase
    for i, (a, b) in enumerate(torus.nodes):
        r = abs(z_coords[i]) / (L_max + 1e-10)
        theta = np.pi * (a - b) / 6.0
        u_i = np.exp(1j * theta) * np.array([np.cos(np.pi*r/2), 1j*np.sin(np.pi*r/2)], dtype=complex)
        u_i /= np.linalg.norm(u_i)
        u[i] = u_i
        v[i] = np.array([-np.conj(u_i[1]), np.conj(u_i[0])], dtype=complex)
        omega[i] = torus.chirality[i] * 1.0
        r_vals[i] = r
        theta_vals[i] = theta
    return u, v, omega, r_vals, theta_vals

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

# ============================================================================
# EIGENMODE OBSERVABLES
# ============================================================================
def compute_observables(evals, evecs, torus, u, v, r_site, theta_site, M):
    """
    For each eigenmode psi_n (column of evecs), compute structural observables.

    Returns dict of arrays, each length N_modes.
    """
    N = torus.num_nodes
    n_modes = len(evals)

    obs = {
        'eigenvalue': evals.copy(),
        'ipr': np.zeros(n_modes),           # Inverse participation ratio
        'sub0_weight': np.zeros(n_modes),    # Weight on sublattice 0
        'sub1_weight': np.zeros(n_modes),    # Weight on sublattice 1
        'sub2_weight': np.zeros(n_modes),    # Weight on sublattice 2
        'chirality_moment': np.zeros(n_modes),  # Sum |psi|^2 * chirality
        'radial_moment': np.zeros(n_modes),  # Sum |psi|^2 * r_i
        'phase_alignment': np.zeros(n_modes),  # |Sum psi_i^* exp(i*theta_i)|^2
        'neighbor_coherence': np.zeros(n_modes),  # Mean |M_ij * psi_j| over neighbors
        'amplitude_entropy': np.zeros(n_modes),  # Shannon entropy of |psi|^2
        'coupling_projection': np.zeros(n_modes),  # |<psi|row_sum(|M|)>|^2
        'spatial_smoothness': np.zeros(n_modes),  # Gradient measure
    }

    # Pre-compute some quantities
    sub0_mask = np.array([1.0 if torus.sublattice[i]==0 else 0.0 for i in range(N)])
    sub1_mask = np.array([1.0 if torus.sublattice[i]==1 else 0.0 for i in range(N)])
    sub2_mask = np.array([1.0 if torus.sublattice[i]==2 else 0.0 for i in range(N)])
    chi = np.array([torus.chirality[i] for i in range(N)], dtype=float)

    # Coupling strength per site: sum of |M_ij| over j
    coupling_strength = np.sum(np.abs(M), axis=1)
    coupling_strength /= np.linalg.norm(coupling_strength)  # normalize

    # Phase pattern from geometric spinors
    phase_pattern = np.exp(1j * theta_site)

    for n in range(n_modes):
        psi = evecs[:, n]  # n-th eigenmode (column)
        prob = np.abs(psi)**2  # probability density

        # 1. IPR (localization measure, 1/N for uniform, 1 for localized)
        obs['ipr'][n] = np.sum(prob**2)

        # 2-4. Sublattice weights
        obs['sub0_weight'][n] = np.dot(prob, sub0_mask)
        obs['sub1_weight'][n] = np.dot(prob, sub1_mask)
        obs['sub2_weight'][n] = np.dot(prob, sub2_mask)

        # 5. Chirality moment (signed sublattice measure)
        obs['chirality_moment'][n] = np.dot(prob, chi)

        # 6. Radial moment (where on lattice the mode lives)
        obs['radial_moment'][n] = np.dot(prob, r_site)

        # 7. Phase alignment with lattice Hopf phase
        obs['phase_alignment'][n] = np.abs(np.sum(np.conj(psi) * phase_pattern))**2

        # 8. Neighbor coherence: how much does the mode exploit M coupling?
        # <psi|M|M|psi> / <psi|M|psi> is just lambda_n...
        # Instead: correlation between |psi_i|^2 and coupling_strength_i
        obs['neighbor_coherence'][n] = np.dot(prob, coupling_strength)

        # 9. Shannon entropy of probability distribution
        pos_prob = prob[prob > 1e-30]
        obs['amplitude_entropy'][n] = -np.sum(pos_prob * np.log(pos_prob))

        # 10. Coupling projection: overlap of mode with coupling pattern
        obs['coupling_projection'][n] = np.abs(np.dot(np.conj(psi), coupling_strength))**2

        # 11. Spatial smoothness: mean |psi_i - psi_j|^2 over neighbors
        grad_sum = 0.0
        n_pairs = 0
        for (i, j) in torus.edges:
            grad_sum += np.abs(psi[i] - psi[j])**2
            n_pairs += 1
        obs['spatial_smoothness'][n] = grad_sum / max(n_pairs, 1)

    return obs

# ============================================================================
# ANALYSIS: CORRELATION WITH EIGENVALUE
# ============================================================================
def analyze_correlations(obs, label=""):
    """Compute correlation of each observable with eigenvalue."""
    evals = obs['eigenvalue']
    n = len(evals)

    print(f"\n{'='*70}")
    print(f"COHERENCE-EIGENVALUE CORRELATIONS {label}")
    print(f"{'='*70}")
    print(f"  N_modes = {n}")
    print(f"  Eigenvalue range: [{evals[0]:.4f}, {evals[-1]:.4f}]")
    print(f"  n_positive = {np.sum(evals > 0)}, n_negative = {np.sum(evals < 0)}")
    print()

    results = {}

    obs_names = [
        ('ipr', 'IPR (localization)'),
        ('sub0_weight', 'Sublattice-0 weight'),
        ('sub1_weight', 'Sublattice-1 weight'),
        ('sub2_weight', 'Sublattice-2 weight'),
        ('chirality_moment', 'Chirality moment'),
        ('radial_moment', 'Radial moment'),
        ('phase_alignment', 'Phase alignment'),
        ('neighbor_coherence', 'Neighbor coherence'),
        ('amplitude_entropy', 'Amplitude entropy'),
        ('coupling_projection', 'Coupling projection'),
        ('spatial_smoothness', 'Spatial smoothness'),
    ]

    print(f"  {'Observable':>25} | {'Pearson r':>10} | {'p-value':>10} | {'Spearman rho':>12} | {'p-value':>10} | {'pos mean':>10} | {'neg mean':>10} | {'t-test p':>10}")
    print(f"  {'-'*120}")

    pos_mask = evals > 0
    neg_mask = evals < 0

    for key, name in obs_names:
        y = obs[key]

        # Pearson correlation with eigenvalue
        r_p, p_p = stats.pearsonr(evals, y)

        # Spearman rank correlation
        r_s, p_s = stats.spearmanr(evals, y)

        # t-test: positive vs negative eigenvalue modes
        y_pos = y[pos_mask]
        y_neg = y[neg_mask]
        if len(y_pos) > 1 and len(y_neg) > 1:
            t_stat, t_p = stats.ttest_ind(y_pos, y_neg)
        else:
            t_stat, t_p = 0, 1.0

        flag = ""
        if p_p < 0.001: flag += " ***"
        elif p_p < 0.01: flag += " **"
        elif p_p < 0.05: flag += " *"

        print(f"  {name:>25} | {r_p:>10.4f} | {p_p:>10.2e} | {r_s:>12.4f} | {p_s:>10.2e} | {np.mean(y_pos):>10.4f} | {np.mean(y_neg):>10.4f} | {t_p:>10.2e}{flag}")

        results[key] = {
            'pearson_r': r_p, 'pearson_p': p_p,
            'spearman_rho': r_s, 'spearman_p': p_s,
            'pos_mean': np.mean(y_pos), 'neg_mean': np.mean(y_neg),
            'ttest_p': t_p
        }

    # Summary: which observables show significant correlation?
    print(f"\n  SIGNIFICANT CORRELATIONS (p < 0.05):")
    any_sig = False
    for key, name in obs_names:
        r = results[key]
        if r['pearson_p'] < 0.05 or r['spearman_p'] < 0.05 or r['ttest_p'] < 0.05:
            any_sig = True
            print(f"    {name}: Pearson r={r['pearson_r']:.3f} (p={r['pearson_p']:.2e}), "
                  f"Spearman rho={r['spearman_rho']:.3f} (p={r['spearman_p']:.2e}), "
                  f"t-test p={r['ttest_p']:.2e}")
    if not any_sig:
        print(f"    NONE — eigenvalue sign is UNCORRELATED with all structural observables")

    return results

# ============================================================================
# POSITIVE WING ANALYSIS
# ============================================================================
def analyze_positive_wing(obs, label=""):
    """Repeat analysis restricted to positive eigenvalues only (the wing used for RMT)."""
    evals = obs['eigenvalue']
    pos = evals > 0
    wing = evals > np.percentile(evals[pos], 20) if np.sum(pos) > 10 else pos

    print(f"\n{'='*70}")
    print(f"POSITIVE WING ANALYSIS {label}")
    print(f"{'='*70}")
    n_wing = np.sum(wing)
    print(f"  Wing eigenvalues: {n_wing} (from {np.sum(pos)} positive)")

    if n_wing < 10:
        print(f"  Too few wing eigenvalues for analysis")
        return None

    wing_evals = evals[wing]

    # Within the positive wing, do observables correlate with eigenvalue magnitude?
    print(f"  Wing eigenvalue range: [{wing_evals[0]:.4f}, {wing_evals[-1]:.4f}]")

    obs_names = [
        ('ipr', 'IPR'),
        ('sub0_weight', 'Sub-0 weight'),
        ('chirality_moment', 'Chirality'),
        ('phase_alignment', 'Phase align'),
        ('spatial_smoothness', 'Smoothness'),
        ('amplitude_entropy', 'Entropy'),
    ]

    print(f"\n  {'Observable':>20} | {'Pearson r':>10} | {'p-value':>10} | {'bottom-25% mean':>15} | {'top-25% mean':>15}")
    print(f"  {'-'*85}")

    q25 = np.percentile(wing_evals, 25)
    q75 = np.percentile(wing_evals, 75)
    bot_mask = wing_evals < q25
    top_mask = wing_evals > q75

    for key, name in obs_names:
        y = obs[key][wing]
        r_p, p_p = stats.pearsonr(wing_evals, y)
        bot_mean = np.mean(y[bot_mask]) if np.sum(bot_mask) > 0 else 0
        top_mean = np.mean(y[top_mask]) if np.sum(top_mask) > 0 else 0
        flag = " *" if p_p < 0.05 else ""
        print(f"  {name:>20} | {r_p:>10.4f} | {p_p:>10.2e} | {bot_mean:>15.4f} | {top_mean:>15.4f}{flag}")

    return wing

# ============================================================================
# BOUNDARY ZONE ANALYSIS
# ============================================================================
def analyze_boundary(obs, label=""):
    """Check if near-zero eigenvalues have distinct properties (the 'chaotic boundary')."""
    evals = obs['eigenvalue']

    print(f"\n{'='*70}")
    print(f"BOUNDARY ZONE ANALYSIS {label}")
    print(f"{'='*70}")

    # Define zones: |lambda| < 10th percentile of |lambda| = boundary
    abs_evals = np.abs(evals)
    threshold_10 = np.percentile(abs_evals, 10)
    threshold_25 = np.percentile(abs_evals, 25)

    boundary = abs_evals < threshold_10
    interior = abs_evals > threshold_25

    n_boundary = np.sum(boundary)
    n_interior = np.sum(interior)

    print(f"  Boundary zone (|lambda| < {threshold_10:.4f}): {n_boundary} modes")
    print(f"  Interior zone (|lambda| > {threshold_25:.4f}): {n_interior} modes")

    if n_boundary < 3 or n_interior < 3:
        print(f"  Too few modes for comparison")
        return

    obs_names = [
        ('ipr', 'IPR'),
        ('sub0_weight', 'Sub-0 weight'),
        ('chirality_moment', 'Chirality'),
        ('phase_alignment', 'Phase align'),
        ('spatial_smoothness', 'Smoothness'),
        ('amplitude_entropy', 'Entropy'),
        ('coupling_projection', 'Coupling proj'),
    ]

    print(f"\n  {'Observable':>20} | {'Boundary mean':>14} | {'Interior mean':>14} | {'MW p-value':>12} | {'Distinct?':>10}")
    print(f"  {'-'*80}")

    for key, name in obs_names:
        y = obs[key]
        y_bound = y[boundary]
        y_inter = y[interior]

        # Mann-Whitney U test (non-parametric)
        if len(y_bound) >= 3 and len(y_inter) >= 3:
            u_stat, mw_p = stats.mannwhitneyu(y_bound, y_inter, alternative='two-sided')
        else:
            mw_p = 1.0

        distinct = "YES" if mw_p < 0.05 else "no"
        flag = " ***" if mw_p < 0.001 else (" **" if mw_p < 0.01 else (" *" if mw_p < 0.05 else ""))
        print(f"  {name:>20} | {np.mean(y_bound):>14.4f} | {np.mean(y_inter):>14.4f} | {mw_p:>12.2e} | {distinct:>10}{flag}")

# ============================================================================
# FIGURE: COMPREHENSIVE SCATTER PLOTS
# ============================================================================
def make_figures(obs, L, Phi, label_suffix=""):
    """Generate the key figures."""
    evals = obs['eigenvalue']

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Coherence-Eigenvalue Mapping (L={L}, Phi={Phi:.4f})', fontsize=14)

    plots = [
        ('ipr', 'IPR (localization)', 'red'),
        ('sub0_weight', 'Sublattice-0 weight', 'green'),
        ('sub1_weight', 'Sublattice-1 weight', 'blue'),
        ('sub2_weight', 'Sublattice-2 weight', 'orange'),
        ('chirality_moment', 'Chirality moment', 'purple'),
        ('radial_moment', 'Radial moment', 'brown'),
        ('phase_alignment', 'Phase alignment', 'teal'),
        ('neighbor_coherence', 'Neighbor coherence', 'navy'),
        ('amplitude_entropy', 'Amplitude entropy', 'darkred'),
        ('coupling_projection', 'Coupling projection', 'darkgreen'),
        ('spatial_smoothness', 'Spatial smoothness', 'gray'),
    ]

    for idx, (key, name, color) in enumerate(plots):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]

        y = obs[key]
        ax.scatter(evals, y, s=3, alpha=0.5, color=color)
        ax.axvline(0, color='black', ls=':', alpha=0.5)

        # Running average (binned)
        n_bins = 30
        bin_edges = np.linspace(evals.min(), evals.max(), n_bins + 1)
        bin_means_x = []
        bin_means_y = []
        for b in range(n_bins):
            mask = (evals >= bin_edges[b]) & (evals < bin_edges[b+1])
            if np.sum(mask) > 0:
                bin_means_x.append(np.mean(evals[mask]))
                bin_means_y.append(np.mean(y[mask]))
        if len(bin_means_x) > 2:
            ax.plot(bin_means_x, bin_means_y, 'k-', linewidth=2, alpha=0.8)

        # Correlation
        r_p, p_p = stats.pearsonr(evals, y)
        ax.set_title(f'{name}\nr={r_p:.3f}, p={p_p:.1e}', fontsize=9)
        ax.set_xlabel('Eigenvalue', fontsize=8)
        ax.tick_params(labelsize=7)

    # Last panel: eigenvalue histogram
    ax = axes[2, 3]
    ax.hist(evals, bins=50, color='steelblue', alpha=0.7, density=True)
    ax.axvline(0, color='red', ls='--', linewidth=2)
    ax.set_title('Eigenvalue distribution', fontsize=9)
    ax.set_xlabel('Eigenvalue', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)

    plt.tight_layout()
    fname = f"{OUT}/coherence_mapping_L{L}{label_suffix}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close()

def make_sigmoid_figure(obs, L, Phi, label_suffix=""):
    """Focus on the most correlated observable — check for sigmoid shape."""
    evals = obs['eigenvalue']

    # Find the observable with strongest Pearson correlation
    obs_names = ['ipr', 'sub0_weight', 'sub1_weight', 'sub2_weight',
                 'chirality_moment', 'radial_moment', 'phase_alignment',
                 'neighbor_coherence', 'amplitude_entropy', 'coupling_projection',
                 'spatial_smoothness']

    best_key = None
    best_r = 0
    for key in obs_names:
        r, p = stats.pearsonr(evals, obs[key])
        if abs(r) > abs(best_r):
            best_r = r
            best_key = key

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Best Coherence Observable vs Eigenvalue (L={L}, Phi={Phi:.4f})', fontsize=13)

    # Panel 1: Scatter with sigmoid fit attempt
    ax = axes[0]
    y = obs[best_key]
    ax.scatter(evals, y, s=5, alpha=0.3, color='steelblue')

    # Binned average
    sort_idx = np.argsort(evals)
    n_bins = 40
    bin_size = max(1, len(evals) // n_bins)
    bin_x = []
    bin_y = []
    bin_err = []
    for b in range(0, len(evals) - bin_size + 1, bin_size):
        chunk = sort_idx[b:b+bin_size]
        bin_x.append(np.mean(evals[chunk]))
        bin_y.append(np.mean(y[chunk]))
        bin_err.append(np.std(y[chunk]) / np.sqrt(len(chunk)))
    bin_x = np.array(bin_x)
    bin_y = np.array(bin_y)
    bin_err = np.array(bin_err)

    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt='ro-', markersize=4, linewidth=2,
                label=f'Binned mean (r={best_r:.3f})')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Eigenvalue', fontsize=11)
    ax.set_ylabel(best_key, fontsize=11)
    ax.set_title(f'Best observable: {best_key}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: CDF-like plot — cumulative mean of observable sorted by eigenvalue
    ax = axes[1]
    sorted_evals = evals[sort_idx]
    sorted_y = y[sort_idx]
    cumul = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
    ax.plot(sorted_evals, cumul, 'b-', linewidth=1.5)
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Eigenvalue', fontsize=11)
    ax.set_ylabel(f'Cumulative mean of {best_key}', fontsize=11)
    ax.set_title('Cumulative trend (sigmoid test)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Box-whisker by eigenvalue quintile
    ax = axes[2]
    n_quint = 5
    quintile_edges = np.percentile(evals, np.linspace(0, 100, n_quint + 1))
    box_data = []
    box_labels = []
    for q in range(n_quint):
        mask = (evals >= quintile_edges[q]) & (evals < quintile_edges[q+1] + 1e-10)
        if np.sum(mask) > 0:
            box_data.append(y[mask])
            mid = (quintile_edges[q] + quintile_edges[q+1]) / 2
            box_labels.append(f'{mid:.2f}')

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors_bp = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_quint))
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
    ax.set_xlabel('Eigenvalue quintile (center)', fontsize=11)
    ax.set_ylabel(best_key, fontsize=11)
    ax.set_title('Distribution by eigenvalue quintile', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{OUT}/coherence_sigmoid_L{L}{label_suffix}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ANALYSIS 21: COHERENCE-EIGENVALUE MAPPING")
    print("Does eigenvalue sign predict eigenmode structure?")
    print("=" * 70)

    # ==================================================================
    # PRIMARY TEST: L=18, Phi=1/6 (best GUE, no resonance issues)
    # ==================================================================
    for L in [18, 24, 30]:
        Phi = 1.0 / 6

        print(f"\n{'#'*70}")
        print(f"# L={L}, Phi=1/6, N={L*L}")
        print(f"{'#'*70}")

        torus = EisensteinTorus(L)
        u, v, omega, r_site, theta_site = assign_spinors_geometric(torus)

        # Verify Hopf orthogonality
        hopf_overlaps = np.array([np.vdot(u[i], v[i]) for i in range(torus.num_nodes)])
        print(f"  Hopf orthogonality check: max|<u_i|v_i>| = {np.max(np.abs(hopf_overlaps)):.2e}")

        M = build_M(torus, u, v, omega, Phi)

        # Full eigen-decomposition (eigenvectors + eigenvalues)
        print(f"  Computing full eigen-decomposition (N={torus.num_nodes})...")
        evals, evecs = np.linalg.eigh(M)
        print(f"  Eigenvalue range: [{evals[0]:.4f}, {evals[-1]:.4f}]")

        # Compute observables
        print(f"  Computing {11} structural observables for {len(evals)} modes...")
        obs = compute_observables(evals, evecs, torus, u, v, r_site, theta_site, M)

        # Correlations
        results = analyze_correlations(obs, label=f"(L={L}, Phi=1/6)")

        # Positive wing analysis
        analyze_positive_wing(obs, label=f"(L={L})")

        # Boundary analysis
        analyze_boundary(obs, label=f"(L={L})")

        # Figures
        make_figures(obs, L, Phi)
        make_sigmoid_figure(obs, L, Phi)

    # ==================================================================
    # CONTROL: Phi=0 (GOE regime) at L=18
    # ==================================================================
    print(f"\n{'#'*70}")
    print(f"# CONTROL: L=18, Phi=0 (GOE regime)")
    print(f"{'#'*70}")

    torus18 = EisensteinTorus(18)
    u18, v18, om18, r18, th18 = assign_spinors_geometric(torus18)
    M0 = build_M(torus18, u18, v18, om18, 0.0)
    evals0, evecs0 = np.linalg.eigh(M0)
    obs0 = compute_observables(evals0, evecs0, torus18, u18, v18, r18, th18, M0)
    results0 = analyze_correlations(obs0, label="(L=18, Phi=0 CONTROL)")
    make_figures(obs0, 18, 0.0, label_suffix="_phi0")

    # ==================================================================
    # COMPARISON: Phi=1/6 vs Phi=0 correlations
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"PHI=1/6 vs PHI=0 CORRELATION COMPARISON (L=18)")
    print(f"{'='*70}")

    # Re-run L=18 Phi=1/6
    M16 = build_M(torus18, u18, v18, om18, 1.0/6)
    evals16, evecs16 = np.linalg.eigh(M16)
    obs16 = compute_observables(evals16, evecs16, torus18, u18, v18, r18, th18, M16)

    obs_names = ['ipr', 'sub0_weight', 'chirality_moment', 'phase_alignment',
                 'spatial_smoothness', 'amplitude_entropy']

    print(f"\n  {'Observable':>25} | {'r (Phi=1/6)':>12} | {'r (Phi=0)':>12} | {'Difference':>12}")
    print(f"  {'-'*70}")
    for key in obs_names:
        r16, _ = stats.pearsonr(evals16, obs16[key])
        r0, _ = stats.pearsonr(evals0, obs0[key])
        diff = abs(r16) - abs(r0)
        flag = " <--" if diff > 0.05 else ""
        print(f"  {key:>25} | {r16:>12.4f} | {r0:>12.4f} | {diff:>12.4f}{flag}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — Analysis 21: Coherence-Eigenvalue Mapping")
    print(f"{'='*70}")
    print(f"""
Key question: Do structural properties of eigenmodes correlate with eigenvalue sign?

Framework prediction:
  Positive lambda -> coherence-like modes (delocalized, phase-aligned)
  Negative lambda -> decoherence-like modes (localized, phase-random)
  Near-zero lambda -> chaotic boundary (distinct structure)

Observables tested (11 total):
  IPR, sublattice weights (0,1,2), chirality moment, radial moment,
  phase alignment, neighbor coherence, amplitude entropy, coupling
  projection, spatial smoothness

Results reported above for L=18, 24, 30 at Phi=1/6, and L=18 Phi=0 control.
""")

    print(f"\nAll output in: {OUT}")
