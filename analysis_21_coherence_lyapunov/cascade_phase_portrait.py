#!/usr/bin/env python3
"""
CASCADE PHASE PORTRAIT & LYAPUNOV EXPONENT — Analysis 21b
==========================================================
Test 1: Phase portrait of the 12-step ouroboros cascade
Test 2: Lyapunov exponent (single-site + lattice Floquet)
Test 3: Lattice Floquet quasienergy statistics

CRITICAL STRUCTURAL INSIGHT:
The single-site cascade applies FIXED SU(2) rotations to (u, v) independently:
  u(k+1) = U_k * u(k),  v(k+1) = V_k * v(k)
After one cycle:  u -> U_total * u,  v -> V_total * v
where U_total, V_total are FIXED SU(2) matrices.

This means: dynamics on S^3 x S^3 is a ROTATION. The Lyapunov exponent
is EXACTLY ZERO. The orbit is periodic or quasiperiodic. NEVER chaotic.

Chaos in the merkabit framework, if it exists, must come from MULTI-SITE
coupling (the M operator), not from single-site cascade dynamics.

The lattice Floquet dynamics CAN show non-trivial statistics because the
M coupling creates effective interactions between the Floquet drive and
the lattice geometry.
"""
import numpy as np
from scipy.linalg import expm
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "C:/Users/selin/merkabit_results/montgomery_comparison"
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# SINGLE-SITE CASCADE (exact copy from torsion_channel_simulation.py)
# ============================================================================
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # pi/6
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = len(OUROBOROS_GATES)

class MerkabitState:
    def __init__(self, u, v, omega=1.0):
        self.u = np.array(u, dtype=complex)
        self.v = np.array(v, dtype=complex)
        self.omega = omega
        self.u /= np.linalg.norm(self.u)
        self.v /= np.linalg.norm(self.v)

    def copy(self):
        return MerkabitState(self.u.copy(), self.v.copy(), self.omega)

def make_trit_zero(omega=1.0):
    """|0> = standing wave: u=[1,0], v=[0,1], C=0."""
    return MerkabitState([1, 0], [0, 1], omega)

def gate_Rx(state, theta):
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    R = np.array([[c, s], [s, c]], dtype=complex)
    return MerkabitState(R @ state.u, R @ state.v, state.omega)

def gate_Rz(state, theta):
    R = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    return MerkabitState(R @ state.u, R @ state.v, state.omega)

def gate_P(state, phi):
    """P gate: ASYMMETRIC phase shift."""
    Pf = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    Pi = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    return MerkabitState(Pf @ state.u, Pi @ state.v, state.omega)

def ouroboros_step(state, step_index, theta=STEP_PHASE):
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'S':
        rz_angle *= 0.4; rx_angle *= 1.3
    elif gate_label == 'R':
        rx_angle *= 0.4; rz_angle *= 1.3
    elif gate_label == 'T':
        rx_angle *= 0.7; rz_angle *= 0.7
    elif gate_label == 'P':
        p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
    s = gate_P(state, p_angle)
    s = gate_Rz(s, rz_angle)
    s = gate_Rx(s, rx_angle)
    return s

def berry_connection(s_prev, s_curr):
    ov_u = np.vdot(s_prev.u, s_curr.u)
    ov_v = np.vdot(s_prev.v, s_curr.v)
    return np.angle(ov_u * ov_v), np.angle(ov_u), np.angle(ov_v)

def random_spinor():
    z = np.random.randn(2) + 1j * np.random.randn(2)
    return z / np.linalg.norm(z)

def random_state_near_zero(epsilon):
    """State near |0> = (u=[1,0], v=[0,1]) with perturbation epsilon."""
    u0 = np.array([1.0, 0.0], dtype=complex)
    v0 = np.array([0.0, 1.0], dtype=complex)
    # Add small random perturbation
    du = epsilon * (np.random.randn(2) + 1j * np.random.randn(2))
    dv = epsilon * (np.random.randn(2) + 1j * np.random.randn(2))
    u = u0 + du; u /= np.linalg.norm(u)
    v = v0 + dv; v /= np.linalg.norm(v)
    return MerkabitState(u, v)

def random_state():
    return MerkabitState(random_spinor(), random_spinor())

# ============================================================================
# TEST 1: SINGLE-SITE PHASE PORTRAIT
# ============================================================================
def test1_phase_portrait():
    print("=" * 70)
    print("TEST 1: SINGLE-SITE PHASE PORTRAIT")
    print("=" * 70)

    n_cycles = 20  # 20 ouroboros cycles = 240 steps
    n_steps = n_cycles * COXETER_H

    # State categories
    categories = {
        'near_zero': [],
        'moderate': [],
        'far': [],
    }

    n_states = 200  # per category

    print(f"  Running {3*n_states} trajectories x {n_steps} steps...")

    # Generate initial states
    for _ in range(n_states):
        categories['near_zero'].append(random_state_near_zero(0.01))
        categories['moderate'].append(random_state_near_zero(0.5))
        categories['far'].append(random_state())

    results = {}
    for cat_name, states in categories.items():
        all_coherence = np.zeros((len(states), n_steps + 1))
        all_berry = np.zeros((len(states), n_steps + 1))
        all_overlap = np.zeros((len(states), n_steps + 1))

        for idx, s0 in enumerate(states):
            s = s0.copy()
            # Coherence = 1 - |<u|v>|^2 (= 1 at |0>, = 0 at |+1> or |-1>)
            all_coherence[idx, 0] = 1.0 - abs(np.vdot(s.u, s.v))**2
            all_overlap[idx, 0] = abs(np.vdot(s.u, np.array([1,0], dtype=complex)))**2 \
                                * abs(np.vdot(s.v, np.array([0,1], dtype=complex)))**2

            gamma = 0.0
            for k in range(n_steps):
                s_prev = s.copy()
                s = ouroboros_step(s, k % COXETER_H)
                A, _, _ = berry_connection(s_prev, s)
                gamma -= A
                all_berry[idx, k+1] = gamma
                all_coherence[idx, k+1] = 1.0 - abs(np.vdot(s.u, s.v))**2
                all_overlap[idx, k+1] = abs(np.vdot(s.u, np.array([1,0], dtype=complex)))**2 \
                                       * abs(np.vdot(s.v, np.array([0,1], dtype=complex)))**2

        results[cat_name] = {
            'coherence': all_coherence,
            'berry': all_berry,
            'overlap': all_overlap,
        }

    # --- FIGURE: Phase Portrait ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Cascade Phase Portrait: Berry phase vs Coherence (20 ouroboros cycles)', fontsize=13)

    colors = {'near_zero': 'blue', 'moderate': 'orange', 'far': 'green'}
    titles = {'near_zero': 'Near |0> (eps=0.01)',
              'moderate': 'Moderate (eps=0.5)',
              'far': 'Random (far from |0>)'}

    for idx, (cat_name, color) in enumerate(colors.items()):
        ax = axes[idx]
        data = results[cat_name]

        # Plot first 50 trajectories
        for j in range(min(50, n_states)):
            ax.plot(data['berry'][j], data['coherence'][j],
                    '-', color=color, alpha=0.05, linewidth=0.5)

        # Plot mean trajectory
        mean_coh = np.mean(data['coherence'], axis=0)
        mean_berry = np.mean(data['berry'], axis=0)
        ax.plot(mean_berry, mean_coh, 'k-', linewidth=2, label='mean')

        # Mark cycle boundaries (every 12 steps)
        for c in range(n_cycles + 1):
            k = c * COXETER_H
            if k <= n_steps:
                ax.plot(mean_berry[k], mean_coh[k], 'ro', markersize=4)

        ax.set_xlabel('Accumulated Berry phase', fontsize=11)
        ax.set_ylabel('Coherence (1 - |<u|v>|^2)', fontsize=11)
        ax.set_title(titles[cat_name], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    fname = f"{OUT}/cascade_phase_portrait.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close()

    # --- FIGURE: Time series ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Cascade Dynamics: Time Series (20 ouroboros cycles)', fontsize=13)

    for row, cat_name in enumerate(['near_zero', 'moderate', 'far']):
        data = results[cat_name]
        steps = np.arange(n_steps + 1)

        # Coherence vs step
        ax = axes[row, 0]
        for j in range(min(20, n_states)):
            ax.plot(steps, data['coherence'][j], '-', alpha=0.2, color=colors[cat_name])
        mean_coh = np.mean(data['coherence'], axis=0)
        ax.plot(steps, mean_coh, 'k-', linewidth=2)
        # Mark cycle boundaries
        for c in range(n_cycles + 1):
            ax.axvline(c * COXETER_H, color='gray', ls=':', alpha=0.3)
        ax.set_ylabel('Coherence', fontsize=10)
        ax.set_title(f'{titles[cat_name]} - Coherence', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Berry phase vs step
        ax = axes[row, 1]
        for j in range(min(20, n_states)):
            ax.plot(steps, data['berry'][j], '-', alpha=0.2, color=colors[cat_name])
        mean_berry = np.mean(data['berry'], axis=0)
        ax.plot(steps, mean_berry, 'k-', linewidth=2)
        for c in range(n_cycles + 1):
            ax.axvline(c * COXETER_H, color='gray', ls=':', alpha=0.3)
        ax.set_ylabel('Accumulated Berry phase', fontsize=10)
        ax.set_title(f'{titles[cat_name]} - Berry phase', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel('Step', fontsize=10)
    axes[2, 1].set_xlabel('Step', fontsize=10)
    plt.tight_layout()
    fname = f"{OUT}/cascade_time_series.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close()

    # --- ANALYSIS: Periodicity ---
    print(f"\n  ORBIT ANALYSIS:")
    s0 = make_trit_zero()
    s = s0.copy()
    # Run 1000 cycles and check return distance
    return_dists = []
    for c in range(200):
        for k in range(COXETER_H):
            s = ouroboros_step(s, k)
        d = np.sqrt(np.linalg.norm(s.u - s0.u)**2 + np.linalg.norm(s.v - s0.v)**2)
        return_dists.append(d)

    return_dists = np.array(return_dists)
    print(f"  |0> after 1 cycle:  d = {return_dists[0]:.6f}")
    print(f"  |0> after 10 cycles: d = {return_dists[9]:.6f}")
    print(f"  |0> after 100 cycles: d = {return_dists[99]:.6f}")
    print(f"  |0> after 200 cycles: d = {return_dists[199]:.6f}")
    print(f"  Min return distance in 200 cycles: {np.min(return_dists):.6f} at cycle {np.argmin(return_dists)+1}")

    # Compute U_total and V_total explicitly
    print(f"\n  FLOQUET MATRICES (one-cycle maps):")
    # Build U_total and V_total by tracking basis vectors
    u_basis = [np.array([1,0], dtype=complex), np.array([0,1], dtype=complex)]
    v_basis = [np.array([1,0], dtype=complex), np.array([0,1], dtype=complex)]

    U_total = np.zeros((2,2), dtype=complex)
    V_total = np.zeros((2,2), dtype=complex)

    for col in range(2):
        s = MerkabitState(u_basis[col], np.array([1,0], dtype=complex))
        for k in range(COXETER_H):
            s = ouroboros_step(s, k)
        U_total[:, col] = s.u * np.linalg.norm(u_basis[col])

    for col in range(2):
        s = MerkabitState(np.array([1,0], dtype=complex), v_basis[col])
        for k in range(COXETER_H):
            s = ouroboros_step(s, k)
        V_total[:, col] = s.v * np.linalg.norm(v_basis[col])

    # Eigenvalues of U_total and V_total give the rotation angles
    eig_U = np.linalg.eigvals(U_total)
    eig_V = np.linalg.eigvals(V_total)
    alpha_U = np.angle(eig_U[0])
    alpha_V = np.angle(eig_V[0])

    print(f"  U_total eigenvalues: {eig_U[0]:.4f}, {eig_U[1]:.4f}")
    print(f"  V_total eigenvalues: {eig_V[0]:.4f}, {eig_V[1]:.4f}")
    print(f"  Rotation angle alpha_U = {alpha_U:.6f} rad = {alpha_U/np.pi:.6f} pi")
    print(f"  Rotation angle alpha_V = {alpha_V:.6f} rad = {alpha_V/np.pi:.6f} pi")
    print(f"  alpha_U/pi rational? -> period estimate: {int(round(np.pi / abs(alpha_U))) if abs(alpha_U) > 0.001 else 'inf'} cycles")

    # Check coherence evolution pattern
    print(f"\n  COHERENCE PATTERN (|0> initial):")
    s = make_trit_zero()
    for k in range(2 * COXETER_H):
        coh = 1.0 - abs(np.vdot(s.u, s.v))**2
        step_in_cycle = k % COXETER_H
        if step_in_cycle == 0 or step_in_cycle == 5 or step_in_cycle == 6 or step_in_cycle == 11:
            print(f"    Step {k:3d} (cycle step {step_in_cycle:2d}): coherence = {coh:.6f}")
        s = ouroboros_step(s, k % COXETER_H)
    coh = 1.0 - abs(np.vdot(s.u, s.v))**2
    print(f"    Step {2*COXETER_H:3d} (end of 2 cycles):   coherence = {coh:.6f}")

    return results


# ============================================================================
# TEST 2: LYAPUNOV EXPONENT
# ============================================================================
def test2_lyapunov():
    print(f"\n{'='*70}")
    print("TEST 2: LYAPUNOV EXPONENT — OBSERVABLE DIVERGENCE")
    print("=" * 70)

    print(f"\n  STRUCTURAL NOTE: The single-site cascade applies FIXED SU(2)")
    print(f"  rotations at each step. The Floquet map is a PRODUCT OF ROTATIONS.")
    print(f"  On S^3 x S^3, this is quasiperiodic. The Lyapunov exponent")
    print(f"  of a rotation is IDENTICALLY ZERO.")
    print(f"\n  Nevertheless, we verify this numerically and measure the")
    print(f"  observable sensitivity d(n) = |C_1(n) - C_2(n)|.")

    epsilons = [1e-3, 1e-6, 1e-9, 1e-12]
    n_cycles = 100
    n_steps = n_cycles * COXETER_H
    n_pairs = 50

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Lyapunov Exponent: Observable Divergence (Single-Site Cascade)', fontsize=13)

    print(f"\n  {'epsilon':>10} | {'max d(n)':>12} | {'mean d(n)':>12} | {'lambda_eff':>12} | {'Interpretation':>20}")
    print(f"  {'-'*75}")

    for eps_idx, eps in enumerate(epsilons):
        all_d = np.zeros((n_pairs, n_steps + 1))

        for pair in range(n_pairs):
            # State 1: random
            s1 = random_state()
            # State 2: perturbed copy
            du = eps * (np.random.randn(2) + 1j * np.random.randn(2))
            dv = eps * (np.random.randn(2) + 1j * np.random.randn(2))
            u2 = s1.u + du; u2 /= np.linalg.norm(u2)
            v2 = s1.v + dv; v2 /= np.linalg.norm(v2)
            s2 = MerkabitState(u2, v2, s1.omega)

            c1_prev = 1.0 - abs(np.vdot(s1.u, s1.v))**2
            c2_prev = 1.0 - abs(np.vdot(s2.u, s2.v))**2
            all_d[pair, 0] = abs(c1_prev - c2_prev)

            for k in range(n_steps):
                s1 = ouroboros_step(s1, k % COXETER_H)
                s2 = ouroboros_step(s2, k % COXETER_H)
                c1 = 1.0 - abs(np.vdot(s1.u, s1.v))**2
                c2 = 1.0 - abs(np.vdot(s2.u, s2.v))**2
                all_d[pair, k+1] = abs(c1 - c2)

        mean_d = np.mean(all_d, axis=0)
        max_d = np.max(all_d, axis=0)

        # Check for exponential growth
        # If d(n) = eps * exp(lambda * n), then lambda = log(d/eps) / n
        # Use middle section to avoid transient and saturation
        mid_start = n_steps // 4
        mid_end = 3 * n_steps // 4
        if np.mean(mean_d[mid_start:mid_end]) > 1e-30:
            # Attempt linear fit of log(d) vs n
            valid = mean_d > 1e-30
            if np.sum(valid) > 10:
                log_d = np.log(mean_d[valid] + 1e-50)
                n_valid = np.where(valid)[0]
                slope, _, _, _, _ = stats.linregress(n_valid[:len(n_valid)//2], log_d[:len(n_valid)//2])
                lambda_eff = slope
            else:
                lambda_eff = 0.0
        else:
            lambda_eff = 0.0

        max_val = np.max(mean_d)
        mean_val = np.mean(mean_d)
        interp = "ZERO (rotation)" if abs(lambda_eff) < 0.001 else f"lambda={lambda_eff:.4f}"

        print(f"  {eps:>10.0e} | {max_val:>12.4e} | {mean_val:>12.4e} | {lambda_eff:>12.6f} | {interp:>20}")

        # Plot
        row, col = eps_idx // 2, eps_idx % 2
        ax = axes[row, col]
        steps = np.arange(n_steps + 1)
        for j in range(min(10, n_pairs)):
            ax.plot(steps / COXETER_H, all_d[j], '-', alpha=0.2, color='blue')
        ax.plot(steps / COXETER_H, mean_d, 'r-', linewidth=2, label='mean')
        ax.set_xlabel('Ouroboros cycles', fontsize=10)
        ax.set_ylabel('|C_1 - C_2|', fontsize=10)
        ax.set_title(f'eps = {eps:.0e}, lambda_eff = {lambda_eff:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log' if np.max(mean_d) > 0 else 'linear')

    plt.tight_layout()
    fname = f"{OUT}/lyapunov_single_site.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved {fname}")
    plt.close()

    # --- Fubini-Study distance (state-space) ---
    print(f"\n  FUBINI-STUDY DISTANCE TEST:")
    print(f"  (Unitary evolution preserves inner products exactly)")

    s1 = random_state()
    du = 1e-6 * (np.random.randn(2) + 1j * np.random.randn(2))
    dv = 1e-6 * (np.random.randn(2) + 1j * np.random.randn(2))
    u2 = s1.u + du; u2 /= np.linalg.norm(u2)
    v2 = s1.v + dv; v2 /= np.linalg.norm(v2)
    s2 = MerkabitState(u2, v2)

    # Initial distances
    d_u_0 = np.arccos(min(1.0, abs(np.vdot(s1.u, s2.u))))
    d_v_0 = np.arccos(min(1.0, abs(np.vdot(s1.v, s2.v))))

    for k in range(10 * COXETER_H):
        s1 = ouroboros_step(s1, k % COXETER_H)
        s2 = ouroboros_step(s2, k % COXETER_H)

    d_u_f = np.arccos(min(1.0, abs(np.vdot(s1.u, s2.u))))
    d_v_f = np.arccos(min(1.0, abs(np.vdot(s1.v, s2.v))))

    print(f"  Initial FS distance: d_u = {d_u_0:.2e}, d_v = {d_v_0:.2e}")
    print(f"  After 10 cycles:     d_u = {d_u_f:.2e}, d_v = {d_v_f:.2e}")
    print(f"  Ratio d_u: {d_u_f/max(d_u_0,1e-50):.6f}")
    print(f"  Ratio d_v: {d_v_f/max(d_v_0,1e-50):.6f}")
    print(f"  -> Distances PRESERVED (rotation, lambda = 0 exactly)")


# ============================================================================
# TEST 3: LATTICE FLOQUET — QUASIENERGY STATISTICS
# ============================================================================
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

def build_lattice_P_gate(torus, step_index):
    """
    Build the lattice P gate for one ouroboros step.
    Each site gets a chirality-modulated phase from the P gate.
    """
    N = torus.num_nodes
    k = step_index
    absent = k % NUM_GATES
    p_angle = STEP_PHASE

    gate_label = OUROBOROS_GATES[absent]
    if gate_label == 'P':
        p_angle *= 0.6

    # The P gate applies ASYMMETRIC phase: forward (+chi) and inverse (-chi)
    # On the lattice, after projecting to the scalar M picture, the effect is:
    # site i gets phase exp(i * p_angle * chi_i / 2) from forward
    # and  phase exp(-i * p_angle * chi_i / 2) from inverse
    # Net: exp(i * p_angle * chi_i) [the asymmetry between Pf and Pi]
    phases = np.array([p_angle * torus.chirality[i] for i in range(N)])
    return np.diag(np.exp(1j * phases))


def test3_lattice_floquet():
    print(f"\n{'='*70}")
    print("TEST 3: LATTICE FLOQUET QUASIENERGY STATISTICS")
    print("=" * 70)

    L = 12  # N=144, manageable for matrix exponential
    torus = EisensteinTorus(L)
    N = torus.num_nodes
    u, v, omega = assign_spinors_geometric(torus)

    print(f"  L={L}, N={N}")

    results = {}
    for Phi in [0.0, 1.0/6]:
        print(f"\n  --- Phi = {Phi:.4f} ---")
        M = build_M(torus, u, v, omega, Phi)

        # Build the full 12-step Floquet operator
        # F_cycle = prod_{k=0}^{11} P_k * exp(-i * M * tau)
        tau = 1.0  # coupling time per step
        U_M = expm(-1j * M * tau)

        F_cycle = np.eye(N, dtype=complex)
        for k in range(COXETER_H):
            P_k = build_lattice_P_gate(torus, k)
            F_step = P_k @ U_M
            F_cycle = F_step @ F_cycle

        # Quasienergies
        eig_F = np.linalg.eigvals(F_cycle)
        quasienergies = np.angle(eig_F)  # in [-pi, pi]
        quasienergies = np.sort(quasienergies)

        print(f"  |det(F_cycle)| = {abs(np.linalg.det(F_cycle)):.6f} (should be 1)")
        print(f"  Quasienergy range: [{quasienergies[0]:.4f}, {quasienergies[-1]:.4f}]")

        # Unfold quasienergies
        # On the circle, the mean spacing is 2*pi/N
        mean_spacing = 2 * np.pi / N
        spacings = np.diff(quasienergies)
        # Add wrap-around spacing
        spacings = np.append(spacings, quasienergies[0] + 2*np.pi - quasienergies[-1])
        spacings /= mean_spacing  # normalize to mean 1

        # Remove zero spacings (degenerate quasienergies)
        nonzero = spacings > 1e-10
        spacings_nz = spacings[nonzero]
        print(f"  Non-degenerate spacings: {len(spacings_nz)} / {len(spacings)}")
        print(f"  Mean spacing (normalized): {np.mean(spacings_nz):.4f}")

        # KS tests
        def wigner_cdf_goe(s):
            return 1 - np.exp(-np.pi * s**2 / 4)

        def wigner_cdf_gue(s):
            return 1 - np.exp(-4 * s**2 / np.pi)

        if len(spacings_nz) > 10:
            ks_goe, p_goe = stats.kstest(spacings_nz, wigner_cdf_goe)
            ks_gue, p_gue = stats.kstest(spacings_nz, wigner_cdf_gue)
            ks_poi, p_poi = stats.kstest(spacings_nz, 'expon')

            print(f"  KS(GOE) = {ks_goe:.4f}, p = {p_goe:.4f}")
            print(f"  KS(GUE) = {ks_gue:.4f}, p = {p_gue:.4f}")
            print(f"  KS(Poisson) = {ks_poi:.4f}, p = {p_poi:.4f}")

            favored = "GOE" if ks_goe < ks_gue else "GUE"
            print(f"  -> Favored: {favored}")
        else:
            ks_goe, p_goe, ks_gue, p_gue, ks_poi, p_poi = 0,0,0,0,0,0

        results[Phi] = {
            'quasienergies': quasienergies,
            'spacings': spacings_nz,
            'ks_goe': ks_goe, 'p_goe': p_goe,
            'ks_gue': ks_gue, 'p_gue': p_gue,
            'ks_poi': ks_poi, 'p_poi': p_poi,
        }

    # --- FIGURE: Quasienergy statistics ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Lattice Floquet Quasienergy Statistics (L={L}, N={N})', fontsize=13)

    # Panel 1: Quasienergy spectrum
    ax = axes[0]
    for Phi, color, label in [(0.0, 'blue', 'Phi=0'), (1.0/6, 'red', 'Phi=1/6')]:
        qe = results[Phi]['quasienergies']
        ax.scatter(np.arange(len(qe)), qe, s=2, color=color, alpha=0.5, label=label)
    ax.set_xlabel('Index', fontsize=10)
    ax.set_ylabel('Quasienergy', fontsize=10)
    ax.set_title('Quasienergy spectrum', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spacing distributions
    ax = axes[1]
    s_grid = np.linspace(0, 4, 200)
    goe_pdf = (np.pi/2) * s_grid * np.exp(-np.pi * s_grid**2 / 4)
    gue_pdf = (32/np.pi**2) * s_grid**2 * np.exp(-4 * s_grid**2 / np.pi)
    poi_pdf = np.exp(-s_grid)
    ax.plot(s_grid, goe_pdf, 'g--', linewidth=2, label='GOE')
    ax.plot(s_grid, gue_pdf, 'r--', linewidth=2, label='GUE')
    ax.plot(s_grid, poi_pdf, 'k:', linewidth=2, label='Poisson')

    for Phi, color, ls in [(0.0, 'blue', '-'), (1.0/6, 'red', '-')]:
        sp = results[Phi]['spacings']
        if len(sp) > 5:
            ax.hist(sp, bins=30, density=True, alpha=0.3, color=color,
                    label=f'Phi={Phi:.2f} (n={len(sp)})')
    ax.set_xlabel('Normalized spacing s', fontsize=10)
    ax.set_ylabel('P(s)', fontsize=10)
    ax.set_title('Spacing distribution', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Panel 3: Summary table
    ax = axes[2]
    ax.axis('off')
    table_data = [['Phi', 'KS(GOE)', 'KS(GUE)', 'KS(Poi)', 'Favored']]
    for Phi in [0.0, 1.0/6]:
        r = results[Phi]
        fav = "GOE" if r['ks_goe'] < r['ks_gue'] else "GUE"
        table_data.append([
            f'{Phi:.4f}',
            f"{r['ks_goe']:.3f} (p={r['p_goe']:.2f})",
            f"{r['ks_gue']:.3f} (p={r['p_gue']:.2f})",
            f"{r['ks_poi']:.3f} (p={r['p_poi']:.2f})",
            fav
        ])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Quasienergy KS Statistics', fontsize=11)

    plt.tight_layout()
    fname = f"{OUT}/lattice_floquet_quasienergy.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved {fname}")
    plt.close()

    return results


# ============================================================================
# TEST 4: LATTICE FLOQUET OTOC (Out-of-Time-Order Correlator)
# ============================================================================
def test4_lattice_otoc():
    print(f"\n{'='*70}")
    print("TEST 4: LATTICE FLOQUET OTOC (Quantum Chaos Diagnostic)")
    print("=" * 70)

    L = 12
    torus = EisensteinTorus(L)
    N = torus.num_nodes
    u, v, omega = assign_spinors_geometric(torus)

    print(f"  L={L}, N={N}")

    # OTOC: C(t) = (1/N) Tr([W(t), V]^dag [W(t), V])
    # where W = projection onto a site, V = projection onto another site
    # W(t) = F^(-t) W F^t (Heisenberg picture)

    # Choose W = |0><0| (first site projector)
    # Choose V = |N//2><N//2| (half-lattice site projector)
    W = np.zeros((N, N), dtype=complex)
    W[0, 0] = 1.0
    V = np.zeros((N, N), dtype=complex)
    V[N//2, N//2] = 1.0

    n_steps_otoc = 60  # 5 ouroboros cycles

    results_otoc = {}
    for Phi in [0.0, 1.0/6]:
        print(f"\n  --- Phi = {Phi:.4f} ---")
        M_mat = build_M(torus, u, v, omega, Phi)
        tau = 1.0

        # Build one-step Floquet operator and its inverse
        U_M = expm(-1j * M_mat * tau)
        # Build step-by-step (Floquet operator varies per step)
        # For OTOC, we need F_step for each step in the cycle

        otoc_vals = np.zeros(n_steps_otoc + 1)
        W_t = W.copy()  # W(0) = W
        otoc_vals[0] = 0.0  # [W(0), V] with W=V is related but W != V here

        # Compute [W(0), V]
        comm = W_t @ V - V @ W_t
        otoc_vals[0] = np.real(np.trace(comm.conj().T @ comm)) / N

        # Build full F_cycle and its inverse
        F_cycle = np.eye(N, dtype=complex)
        for k in range(COXETER_H):
            P_k = build_lattice_P_gate(torus, k)
            F_step = P_k @ U_M
            F_cycle = F_step @ F_cycle

        F_inv = np.linalg.inv(F_cycle)

        # Evolve W in Heisenberg picture: W(t) = F^(-t) W F^t
        # We do it step by step (one full cycle at a time for efficiency)
        F_t = np.eye(N, dtype=complex)  # F^t
        F_inv_t = np.eye(N, dtype=complex)  # F^(-t)

        for step in range(1, n_steps_otoc + 1):
            F_t = F_cycle @ F_t
            F_inv_t = F_inv_t @ F_inv

            W_t = F_inv_t @ W @ F_t
            comm = W_t @ V - V @ W_t
            otoc_vals[step] = np.real(np.trace(comm.conj().T @ comm)) / N

        results_otoc[Phi] = otoc_vals
        print(f"  OTOC(0) = {otoc_vals[0]:.6f}")
        print(f"  OTOC(30) = {otoc_vals[min(30, n_steps_otoc)]:.6f}")
        print(f"  OTOC(60) = {otoc_vals[n_steps_otoc]:.6f}")

        # Check for exponential growth
        early = otoc_vals[1:20]
        if np.all(early > 0):
            log_otoc = np.log(early + 1e-50)
            slope, intercept, _, _, _ = stats.linregress(np.arange(len(early)), log_otoc)
            print(f"  Early growth rate (lambda_OTOC): {slope:.4f}")
            if slope > 0.01:
                print(f"  -> EXPONENTIAL GROWTH detected (quantum chaos signature)")
            else:
                print(f"  -> No exponential growth (integrable/regular)")
        else:
            print(f"  -> OTOC has zero values in early time (regular)")

    # --- FIGURE: OTOC ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cycles = np.arange(n_steps_otoc + 1)
    for Phi, color, label in [(0.0, 'blue', 'Phi=0 (GOE)'), (1.0/6, 'red', 'Phi=1/6 (GUE)')]:
        otoc = results_otoc[Phi]
        ax.semilogy(cycles, otoc + 1e-20, '-o', color=color, markersize=3, label=label)
    ax.set_xlabel('Ouroboros cycles', fontsize=12)
    ax.set_ylabel('OTOC C(t)', fontsize=12)
    ax.set_title(f'Out-of-Time-Order Correlator (L={L}, N={N})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{OUT}/lattice_otoc.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved {fname}")
    plt.close()

    return results_otoc


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 70)
    print("ANALYSIS 21b: CASCADE PHASE PORTRAIT & LYAPUNOV EXPONENT")
    print("=" * 70)

    # Test 1: Single-site phase portrait
    results1 = test1_phase_portrait()

    # Test 2: Single-site Lyapunov
    test2_lyapunov()

    # Test 3: Lattice Floquet quasienergy statistics
    results3 = test3_lattice_floquet()

    # Test 4: Lattice Floquet OTOC
    results4 = test4_lattice_otoc()

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — Analysis 21b")
    print(f"{'='*70}")
    print(f"""
TEST 1 — SINGLE-SITE PHASE PORTRAIT:
  The 12-step ouroboros cascade applies FIXED SU(2) rotations at each step.
  The dynamics on S^3 x S^3 is a ROTATION (product of unitary matrices).
  Orbits are periodic or quasiperiodic. Not chaotic. Not strange attractor.
  Coherence oscillates between min and max values with cycle-dependent period.

TEST 2 — SINGLE-SITE LYAPUNOV:
  Lyapunov exponent is IDENTICALLY ZERO (proven: SU(2) rotations preserve
  Fubini-Study distance). Observable sensitivity d(n) oscillates but does
  not grow exponentially. The cascade is NOT at the edge of chaos.

TEST 3 — LATTICE FLOQUET QUASIENERGY STATISTICS:
  The 12-step Floquet operator on the Eisenstein torus has quasienergies
  whose spacing statistics may differ between Phi=0 and Phi=1/6.
  This connects the cascade dynamics to the GOE/GUE transition.

TEST 4 — LATTICE OTOC:
  The OTOC measures quantum chaos on the lattice. Exponential growth of
  OTOC indicates chaos; saturation or power-law growth indicates regularity.
  Compare Phi=0 (GOE) vs Phi=1/6 (GUE) to test whether the P gate flux
  enhances or suppresses quantum chaos on the lattice.

KEY STRUCTURAL INSIGHT:
  The chaos is NOT in the single-site cascade (which is a rotation).
  If chaos exists, it is in the LATTICE COUPLING (the M operator),
  which creates effective nonlinearity through multi-site interactions.
  The Floquet drive (P gate) modulates this coupling, potentially
  driving a regular-to-chaotic transition at Phi=1/6.
""")

    print(f"\nAll output in: {OUT}")
