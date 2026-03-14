#!/usr/bin/env python3
"""
R/R̄ MERGER OPERATOR vs RIEMANN ZEROS: GUE EIGENVALUE SPACING COMPARISON
=========================================================================

Core insight (Selina Stenberg, March 13 2026):
The missing Hermitian operator whose eigenvalues are the Riemann zeros is not
the full Floquet return operator U^12. It is the R/R̄ coupling operator at the
standing wave |0⟩ — the intra-merkabit torsion channel itself.

    M = |u⟩⟨v| + |v⟩⟨u| = R + R̄

Hermitian by construction. Natural Berry-Keating structure.
Standing wave |0⟩ in its kernel.
Built on Eisenstein lattice Z[ω], E₆ symmetry via McKay constraint.

Modules:
  1. Build Eisenstein lattice and assign E₆-constrained spinors
  2. Construct R/R̄ merger operator M and verify Hermiticity
  3. Extract and unfold eigenvalue spectrum
  4. Load Riemann zeros
  5. Statistical comparison (KS, pair correlation, level variance, β)
  6. Standing wave kernel analysis
  7. Scaling analysis (N = 100..2000)
  8. Berry-Keating projection
  9. Direct eigenvalue-to-zero matching

Output: C:/Users/selin/merkabit_results/riemann_zeros/
"""

import numpy as np
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Safe UTF-8 output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import interp1d

RESULTS_DIR = Path(r"C:\Users\selin\merkabit_results\riemann_zeros")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================================
# ARCHITECTURAL CONSTANTS (zero free parameters)
# ============================================================================

RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12          # Floquet cycle length
POSITIVE_ROOTS = 36
STEP_PHASE = 2 * np.pi / COXETER_H  # pi/6
F_SPECTRAL_GAP = 1.0/3  # 24-cell spectral gap
XI = 3.0                 # Coherence length (Xiang DTC)
R_COOP = 311.0/100       # Cooperative parameter
F_RETURN = 0.696778      # Berry phase return fidelity

OMEGA_EISEN = np.exp(2j * np.pi / 3)  # primitive cube root of unity
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
NUM_GATES = len(OUROBOROS_GATES)

# Riemann zeros (first 50, hardcoded fallback)
RIEMANN_ZEROS_KNOWN = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846
])


# ============================================================================
# MODULE 1: EISENSTEIN LATTICE CONSTRUCTION
# ============================================================================

def build_eisenstein_lattice(N_sites):
    """
    Generate N_sites points of the Eisenstein lattice Z[omega],
    omega = exp(2*pi*i/3). Coordination number = 6 (hexagonal).
    Sort by distance from origin and take the N_sites closest.
    """
    R_max = int(np.sqrt(N_sites)) + 2
    sites = []
    for a in range(-R_max, R_max + 1):
        for b in range(-R_max, R_max + 1):
            z = a + b * OMEGA_EISEN
            sites.append(z)
    sites = sorted(sites, key=abs)[:N_sites]
    return np.array(sites)


def assign_spinors_e6(sites, L_scale=None):
    """
    Assign dual spinors to each lattice site under E6 McKay constraint.

    u_z = exp(i*pi*(a-b)/6) * [cos(pi*|z|/L), i*sin(pi*|z|/L)]^T
    v_z = [-conj(u_z[1]), conj(u_z[0])]^T  (time-reversed conjugate)

    This ensures:
    - <u_z|v_z> = 0 at all sites (standing wave condition)
    - Spinors transform under P_24 (binary tetrahedral group)
    """
    N = len(sites)
    if L_scale is None:
        L_scale = np.max(np.abs(sites)) + 1.0

    U = np.zeros((N, 2), dtype=complex)
    V = np.zeros((N, 2), dtype=complex)

    for i, z in enumerate(sites):
        # Decompose z = a + b*omega into integer coordinates
        # z = a + b*(-1/2 + i*sqrt(3)/2) => Im(z) = b*sqrt(3)/2
        b = np.imag(z) / (np.sqrt(3)/2) if abs(np.imag(z)) > 1e-10 else 0.0
        a = np.real(z) + b/2

        # E6 phase from lattice position (mod 12 = Coxeter h)
        phase = np.pi * (a - b) / RANK_E6  # /6 = /rank(E6)

        # Radial dependence
        r = abs(z) / L_scale
        theta = np.pi * r  # maps [0, L] -> [0, pi]

        # Forward spinor with E6 phase
        u = np.array([
            np.cos(theta/2) * np.exp(1j * phase),
            1j * np.sin(theta/2) * np.exp(-1j * phase)
        ], dtype=complex)
        u /= np.linalg.norm(u)

        # Inverse spinor: time-reversed conjugate (ensures <u|v> = 0)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)

        U[i] = u
        V[i] = v

    return U, V


def coupling_strength(z_i, z_j, xi=XI):
    """
    Coupling between sites i and j.
    Decays as exp(-|z_i - z_j| / xi) where xi = 3.0.
    """
    r = abs(z_i - z_j)
    if r < 1e-10:
        return 0.0  # no self-coupling in off-diagonal
    return np.exp(-r / xi)


# ============================================================================
# MODULE 2: BUILD R/R-BAR MERGER OPERATOR
# ============================================================================

def build_RRbar_operator(sites, U, V, xi=XI, max_range=None):
    """
    Build the R/R-bar coupling operator M on the Eisenstein lattice.

        M_ij = J(r_ij) * [<u_i|v_j> + <v_i|u_j>]    (i != j)
        M_ii = Re(<u_i|v_i>)                           (on-site coherence)

    M is Hermitian by construction:
        M_ij* = J(r_ij)* * [<u_i|v_j>* + <v_i|u_j>*]
              = J(r_ij) * [<v_j|u_i> + <u_j|v_i>]
              = M_ji

    Parameters
    ----------
    sites : array of complex, shape (N,)
    U, V : arrays of shape (N, 2), forward and inverse spinors
    xi : float, coupling decay length
    max_range : float, cutoff distance (None = use 3*xi)

    Returns
    -------
    M : ndarray, shape (N, N), Hermitian
    """
    N = len(sites)
    if max_range is None:
        max_range = 3.0 * xi  # beyond 3*xi, coupling < exp(-3) ~ 0.05

    M = np.zeros((N, N), dtype=complex)

    for i in range(N):
        # On-site: standing wave coherence
        M[i, i] = np.real(np.vdot(U[i], V[i]))

        for j in range(i+1, N):
            r = abs(sites[i] - sites[j])
            if r > max_range:
                continue

            J = coupling_strength(sites[i], sites[j], xi)

            # R/R-bar coupling: <u_i|v_j> + <v_i|u_j>
            uv = np.vdot(U[i], V[j])  # <u_i|v_j>
            vu = np.vdot(V[i], U[j])  # <v_i|u_j>
            coupling = J * (uv + vu)

            M[i, j] = coupling
            M[j, i] = np.conj(coupling)  # Hermitian

    return M


# ============================================================================
# MODULE 3: SPECTRUM UNFOLDING
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=12):
    """
    Unfold eigenvalues to uniform mean density = 1.

    Method: fit smooth polynomial to cumulative eigenvalue density,
    map each eigenvalue to its unfolded position.
    Poly degree 12 = Coxeter number (natural for E6 architecture).
    """
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)

    # Empirical CDF
    cdf_empirical = np.arange(1, N + 1) / N

    # Fit smooth polynomial to CDF
    # Use Chebyshev basis for numerical stability
    deg = min(poly_degree, max(3, N // 20))
    coeffs = np.polyfit(evals_sorted, cdf_empirical * N, deg)
    N_smooth = np.polyval(coeffs, evals_sorted)

    # Unfolded levels
    unfolded = N_smooth

    # Nearest-neighbour spacings
    spacings = np.diff(unfolded)

    # Normalise to mean spacing = 1
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s

    return unfolded, spacings


def unfold_riemann_zeros(zeros):
    """
    Unfold Riemann zeros using the smooth Riemann-von Mangoldt formula:
    N_smooth(T) = (T/2pi)*log(T/2pi) - T/2pi + 7/8 + O(1/T)
    """
    T = np.sort(zeros)
    N_smooth = (T / (2*np.pi)) * np.log(T / (2*np.pi)) - T / (2*np.pi) + 7.0/8

    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s

    return N_smooth, spacings


# ============================================================================
# MODULE 4: LOAD RIEMANN ZEROS
# ============================================================================

def load_riemann_zeros():
    """Load Riemann zeros from Odlyzko dataset or hardcoded fallback."""
    # Try downloading first 10000
    zeros_file = RESULTS_DIR / "riemann_zeros_cache.npy"
    if zeros_file.exists():
        zeros = np.load(zeros_file)
        print(f"  Loaded {len(zeros)} cached Riemann zeros")
        return zeros

    try:
        import urllib.request
        url = "https://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1"
        print(f"  Downloading Riemann zeros from Odlyzko...")
        raw_path = RESULTS_DIR / "zeros1_raw.txt"
        urllib.request.urlretrieve(url, str(raw_path))
        # Parse: file has one zero per line (imaginary parts)
        zeros = []
        with open(raw_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        zeros.append(float(line))
                    except ValueError:
                        continue
        if len(zeros) > 50:
            zeros = np.array(zeros)
            np.save(zeros_file, zeros)
            print(f"  Downloaded {len(zeros)} Riemann zeros")
            return zeros
    except Exception as e:
        print(f"  Download failed: {e}")

    print(f"  Using hardcoded {len(RIEMANN_ZEROS_KNOWN)} zeros")
    return RIEMANN_ZEROS_KNOWN


# ============================================================================
# MODULE 5: STATISTICAL DISTRIBUTIONS
# ============================================================================

def gue_pdf(s):
    """GUE Wigner-Dyson distribution (beta=2)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def gse_pdf(s):
    """GSE Wigner surmise (beta=4)."""
    C = (2**18) / (3**6 * np.pi**3)
    return C * s**4 * np.exp(-64 * s**2 / (9 * np.pi))

def goe_pdf(s):
    """GOE Wigner-Dyson distribution (beta=1)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    # Note: GOE surmise is same functional form for P(s) but different for
    # higher-order statistics. Use exact GOE surmise:
    # Actually: P_GOE(s) = (pi/2)*s*exp(-pi*s^2/4) is the Wigner surmise for beta=1
    # The Wigner surmise for beta=2 (GUE) is:
    # P_GUE(s) = (32/pi^2)*s^2*exp(-4*s^2/pi)
    # Let me use the exact Wigner surmises:

def wigner_surmise(s, beta):
    """
    Exact Wigner surmise for 2x2 matrices:
    P_beta(s) = a_beta * s^beta * exp(-b_beta * s^2)
    """
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta+2)/2))**((beta+1)) / (gamma_fn((beta+1)/2))**((beta+2))
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2
    return a * s**beta * np.exp(-b * s**2)


def wigner_surmise_cdf(s_vals, beta):
    """CDF of Wigner surmise via numerical integration."""
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta+2)/2))**((beta+1)) / (gamma_fn((beta+1)/2))**((beta+2))
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2

    cdf = np.zeros_like(s_vals, dtype=float)
    for i, s in enumerate(s_vals):
        val, _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, s)
        cdf[i] = val
    return cdf


def make_wigner_cdf_func(beta):
    """Return a callable CDF function for KS test."""
    from scipy.special import gamma as gamma_fn
    a = 2.0 * (gamma_fn((beta+2)/2))**((beta+1)) / (gamma_fn((beta+1)/2))**((beta+2))
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2

    def cdf_func(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(s, 0))
            return val
        result = np.zeros_like(s, dtype=float)
        for i, si in enumerate(s):
            result[i], _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(si, 0))
        return result
    return cdf_func


def pair_correlation(unfolded, r_max=3.0, n_bins=100):
    """
    Compute pair correlation function g(r) of unfolded eigenvalues.
    Montgomery's formula for GUE/Riemann: g(r) = 1 - (sin(pi*r)/(pi*r))^2
    """
    N = len(unfolded)
    diffs = []
    for i in range(N):
        for j in range(i+1, min(i+50, N)):  # local pairs only
            d = abs(unfolded[j] - unfolded[i])
            if d < r_max:
                diffs.append(d)

    if len(diffs) == 0:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)

    diffs = np.array(diffs)
    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(diffs, bins=bins, density=True)

    # Normalise: mean density should give g(r)->1 at large r
    r_centers = (bins[:-1] + bins[1:]) / 2
    # Normalise by expected uniform density
    mean_density = len(diffs) / r_max
    if mean_density > 0:
        hist = hist * r_max  # rough normalisation

    return r_centers, hist


def montgomery_formula(r):
    """Montgomery's pair correlation: g(r) = 1 - (sin(pi*r)/(pi*r))^2"""
    result = np.where(np.abs(r) < 1e-10, 0.0, 1 - (np.sin(np.pi * r) / (np.pi * r))**2)
    return result


def level_number_variance(unfolded, L_vals):
    """
    Level number variance Sigma^2(L).
    GUE: Sigma^2(L) ~ (2/pi^2)*log(2*pi*L) + const for large L
    Poisson: Sigma^2(L) = L
    """
    N = len(unfolded)
    variances = []
    for L in L_vals:
        counts = []
        for start in np.linspace(unfolded[0], unfolded[-1] - L, min(200, N//2)):
            n_in = np.sum((unfolded >= start) & (unfolded < start + L))
            counts.append(n_in)
        if len(counts) > 1:
            variances.append(np.var(counts))
        else:
            variances.append(0.0)
    return np.array(variances)


def spectral_rigidity_delta3(unfolded, L_vals):
    """
    Dyson-Mehta Delta_3(L) statistic (spectral rigidity).
    GUE: Delta_3(L) ~ (1/pi^2)*log(2*pi*L) + const
    Poisson: Delta_3(L) = L/15
    """
    N = len(unfolded)
    delta3 = []
    for L in L_vals:
        d3_samples = []
        for start_idx in range(0, N - max(10, int(L)), max(1, N//100)):
            # Get eigenvalues in window [a, a+L]
            a = unfolded[start_idx]
            in_window = unfolded[(unfolded >= a) & (unfolded < a + L)]
            n = len(in_window)
            if n < 3:
                continue
            # Best-fit straight line: minimise integral of (N(x) - Ax - B)^2
            x = in_window - a
            # Least squares fit
            A = np.arange(1, n + 1)
            slope = np.sum(x * A) / np.sum(x**2) if np.sum(x**2) > 0 else 0
            intercept = np.mean(A) - slope * np.mean(x)
            residuals = A - (slope * x + intercept)
            d3_val = np.mean(residuals**2) / max(n, 1)
            d3_samples.append(d3_val)
        delta3.append(np.mean(d3_samples) if d3_samples else 0.0)
    return np.array(delta3)


def fit_level_repulsion_beta(spacings):
    """
    Fit level repulsion exponent beta from P(s) ~ s^beta at small s.
    """
    # Use spacings in range [0, 0.5]
    small = spacings[spacings < 0.5]
    if len(small) < 10:
        small = spacings[spacings < 1.0]
    if len(small) < 5:
        return 0.0, 0.0

    # Histogram
    n_bins = min(20, len(small) // 3)
    if n_bins < 3:
        return 0.0, 0.0
    hist, bin_edges = np.histogram(small, bins=n_bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Only fit where hist > 0
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0

    log_s = np.log(centers[mask])
    log_p = np.log(hist[mask])

    # Linear fit: log(P) = beta*log(s) + const
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = coeffs[0]
    # R^2
    predicted = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - predicted)**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta, r2


# ============================================================================
# MODULE 6: BERRY-KEATING PROJECTION
# ============================================================================

def build_momentum_operator(sites):
    """
    Build momentum operator on Eisenstein lattice via finite differences.
    P = -i * (discrete gradient) using nearest-neighbour finite differences.
    """
    N = len(sites)
    P = np.zeros((N, N), dtype=complex)

    for i in range(N):
        # Find nearest neighbours (distance ~ 1)
        dists = np.abs(sites - sites[i])
        nn_mask = (dists > 1e-10) & (dists < 1.5)
        nn_indices = np.where(nn_mask)[0]

        if len(nn_indices) == 0:
            continue

        for j in nn_indices:
            dz = sites[j] - sites[i]
            # Momentum along Re direction
            P[i, j] = -1j * np.real(dz) / abs(dz)**2

    # Symmetrise to make Hermitian (p = (P + P†)/2)
    P = (P + P.conj().T) / 2
    return P


def berry_keating_projection(M, sites):
    """
    Project M onto position-momentum quadratures.
    Berry-Keating conjecture: H = xp + px (symmetrised).

    Check: is M ~ lambda * (XP + PX) for some lambda?
    """
    N = len(sites)
    X = np.diag(np.real(sites))
    P_op = build_momentum_operator(sites)

    XP = X @ P_op
    XP_sym = (XP + XP.conj().T) / 2  # Hermitian symmetrisation

    # Remove any zero operator case
    norm_XP = np.linalg.norm(XP_sym)
    norm_M = np.linalg.norm(M)
    if norm_XP < 1e-10 or norm_M < 1e-10:
        return 0.0, 1.0

    # Best fit: min ||M - lambda * XP_sym||_F
    lambda_fit = np.real(np.trace(M.conj().T @ XP_sym)) / norm_XP**2
    residual = np.linalg.norm(M - lambda_fit * XP_sym) / norm_M

    return lambda_fit, residual


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_single_N(N_sites, riemann_zeros, report_lines, verbose=True):
    """Run full analysis for a single lattice size N."""
    prefix = f"  [N={N_sites}]"

    if verbose:
        print(f"\n{'='*76}")
        print(f"  ANALYSIS FOR N = {N_sites} SITES")
        print(f"{'='*76}")

    # --- Module 1: Build lattice ---
    t0 = time.time()
    sites = build_eisenstein_lattice(N_sites)
    U, V = assign_spinors_e6(sites)
    t_lattice = time.time() - t0
    if verbose:
        print(f"\n{prefix} Eisenstein lattice built in {t_lattice:.2f}s")
        print(f"{prefix} Lattice radius: {np.max(np.abs(sites)):.2f}")

    # Verify orthogonality <u_i|v_i> = 0
    on_site_overlaps = np.array([np.vdot(U[i], V[i]) for i in range(N_sites)])
    max_overlap = np.max(np.abs(on_site_overlaps))
    if verbose:
        print(f"{prefix} Max on-site |<u|v>|: {max_overlap:.2e} (should be ~0)")

    # --- Module 2: Build M ---
    t0 = time.time()
    M = build_RRbar_operator(sites, U, V, xi=XI)
    t_build = time.time() - t0
    if verbose:
        print(f"{prefix} R/R-bar operator built in {t_build:.2f}s")

    # Verify Hermiticity
    herm_err = np.max(np.abs(M - M.conj().T))
    if verbose:
        print(f"{prefix} Hermiticity check: max|M - M^dag| = {herm_err:.2e}")
    assert herm_err < 1e-10, f"M is not Hermitian! Error: {herm_err}"

    # --- Eigenvalue decomposition ---
    t0 = time.time()
    eigenvalues = np.linalg.eigvalsh(M)
    t_eig = time.time() - t0
    if verbose:
        print(f"{prefix} Eigendecomposition in {t_eig:.2f}s")
        print(f"{prefix} Spectral range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        print(f"{prefix} Mean eigenvalue: {np.mean(eigenvalues):.6f}")

    # Save eigenvalues
    np.save(RESULTS_DIR / f"M_eigenvalues_N{N_sites}.npy", eigenvalues)

    # --- Module 3: Unfold spectrum ---
    unfolded_M, spacings_M = unfold_spectrum(eigenvalues, poly_degree=12)
    mean_spacing = np.mean(spacings_M)
    if verbose:
        print(f"{prefix} Mean unfolded spacing: {mean_spacing:.4f} (target: 1.0)")
        print(f"{prefix} Std of spacings: {np.std(spacings_M):.4f}")

    # --- Module 6: Standing wave kernel ---
    near_zero = eigenvalues[np.abs(eigenvalues) < 0.01]
    n_near_zero = len(near_zero)
    min_abs_eval = np.min(np.abs(eigenvalues))
    if verbose:
        print(f"\n{prefix} KERNEL ANALYSIS:")
        print(f"{prefix}   Eigenvalues with |lambda| < 0.01: {n_near_zero}")
        print(f"{prefix}   Minimum |lambda|: {min_abs_eval:.6e}")
        if n_near_zero > 0:
            print(f"{prefix}   Near-zero eigenvalues: {near_zero[:10]}")

    # --- Module 5: Statistical comparison ---
    if verbose:
        print(f"\n{prefix} STATISTICAL COMPARISON:")

    # KS tests against GUE (beta=2), GSE (beta=4), Poisson
    positive_spacings = spacings_M[spacings_M > 0]

    # Poisson CDF: 1 - exp(-s)
    ks_poi, p_poi = stats.kstest(positive_spacings, lambda s: 1 - np.exp(-s))

    # GUE (beta=2) via Wigner surmise CDF
    gue_cdf_func = make_wigner_cdf_func(2)
    ks_gue, p_gue = stats.kstest(positive_spacings, gue_cdf_func)

    # GSE (beta=4) via Wigner surmise CDF
    gse_cdf_func = make_wigner_cdf_func(4)
    ks_gse, p_gse = stats.kstest(positive_spacings, gse_cdf_func)

    if verbose:
        print(f"{prefix}   KS(M, GUE  beta=2): stat={ks_gue:.4f}, p={p_gue:.6f}")
        print(f"{prefix}   KS(M, GSE  beta=4): stat={ks_gse:.4f}, p={p_gse:.6f}")
        print(f"{prefix}   KS(M, Poisson):     stat={ks_poi:.4f}, p={p_poi:.6f}")

    # Level repulsion exponent
    beta_fit, beta_r2 = fit_level_repulsion_beta(positive_spacings)
    if verbose:
        print(f"{prefix}   Level repulsion beta = {beta_fit:.3f} (R2={beta_r2:.3f})")
        print(f"{prefix}     GUE predicts beta=2, GSE predicts beta=4, Poisson predicts beta=0")

    # Pair correlation
    r_centers, g_r = pair_correlation(unfolded_M, r_max=3.0, n_bins=50)

    # Level number variance
    L_vals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    sigma2 = level_number_variance(unfolded_M, L_vals)

    # Spectral rigidity
    delta3 = spectral_rigidity_delta3(unfolded_M, L_vals)

    # Riemann zero comparison
    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    positive_spacings_riem = spacings_riem[spacings_riem > 0]

    # KS test: M spacings vs Riemann zero spacings
    # Use 2-sample KS test
    ks_riem, p_riem = stats.ks_2samp(positive_spacings, positive_spacings_riem)
    if verbose:
        print(f"{prefix}   KS(M, Riemann):     stat={ks_riem:.4f}, p={p_riem:.6f}")

    # Berry-Keating projection (only for smaller N due to cost)
    bk_lambda, bk_residual = 0.0, 1.0
    if N_sites <= 500:
        bk_lambda, bk_residual = berry_keating_projection(M, sites)
        if verbose:
            print(f"\n{prefix} BERRY-KEATING PROJECTION:")
            print(f"{prefix}   lambda_fit = {bk_lambda:.6f}")
            print(f"{prefix}   Residual ||M - lambda*XP|| / ||M|| = {bk_residual:.4f}")

    # Direct eigenvalue matching
    n_match = min(len(positive_spacings), len(positive_spacings_riem), 40)
    if n_match > 5:
        M_s = np.sort(positive_spacings)[-n_match:]
        R_s = np.sort(positive_spacings_riem)[:n_match]
        # Normalise both
        M_norm = (M_s - M_s.min()) / (M_s.max() - M_s.min() + 1e-15) * n_match
        R_norm = (R_s - R_s.min()) / (R_s.max() - R_s.min() + 1e-15) * n_match
        rmsd = np.sqrt(np.mean((M_norm - R_norm)**2))
        random_baseline = np.sqrt(n_match / 12)
    else:
        rmsd = float('nan')
        random_baseline = float('nan')

    if verbose and n_match > 5:
        print(f"\n{prefix} DIRECT EIGENVALUE MATCHING (n={n_match}):")
        print(f"{prefix}   RMSD(M, Riemann) = {rmsd:.4f}")
        print(f"{prefix}   Random baseline  = {random_baseline:.3f}")

    # Collect results
    results = {
        'N': N_sites,
        'ks_gue': ks_gue, 'p_gue': p_gue,
        'ks_gse': ks_gse, 'p_gse': p_gse,
        'ks_poi': ks_poi, 'p_poi': p_poi,
        'ks_riem': ks_riem, 'p_riem': p_riem,
        'beta_fit': beta_fit, 'beta_r2': beta_r2,
        'n_near_zero': n_near_zero,
        'min_abs_eval': min_abs_eval,
        'bk_lambda': bk_lambda, 'bk_residual': bk_residual,
        'rmsd': rmsd,
        'spacings': positive_spacings,
        'unfolded': unfolded_M,
        'eigenvalues': eigenvalues,
        'r_centers': r_centers, 'g_r': g_r,
        'L_vals': L_vals, 'sigma2': sigma2, 'delta3': delta3,
    }

    # Append to report
    report_lines.append(f"\n{'='*76}")
    report_lines.append(f"N = {N_sites}")
    report_lines.append(f"{'='*76}")
    report_lines.append(f"Spectral range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
    report_lines.append(f"Hermiticity error: {herm_err:.2e}")
    report_lines.append(f"Max on-site |<u|v>|: {max_overlap:.2e}")
    report_lines.append(f"KS(M, GUE  beta=2): stat={ks_gue:.4f}, p={p_gue:.6f}")
    report_lines.append(f"KS(M, GSE  beta=4): stat={ks_gse:.4f}, p={p_gse:.6f}")
    report_lines.append(f"KS(M, Poisson):     stat={ks_poi:.4f}, p={p_poi:.6f}")
    report_lines.append(f"KS(M, Riemann):     stat={ks_riem:.4f}, p={p_riem:.6f}")
    report_lines.append(f"Level repulsion beta = {beta_fit:.3f} (R2={beta_r2:.3f})")
    report_lines.append(f"Near-zero eigenvalues: {n_near_zero}")
    report_lines.append(f"Min |eigenvalue|: {min_abs_eval:.6e}")
    report_lines.append(f"Berry-Keating: lambda={bk_lambda:.6f}, residual={bk_residual:.4f}")
    report_lines.append(f"RMSD eigenvalue match: {rmsd:.4f}")

    return results


def generate_figures(all_results, riemann_zeros):
    """Generate all publication figures."""
    print("\n  Generating figures...")

    # Use the largest N result for primary figures
    primary = all_results[-1]
    spacings_M = primary['spacings']
    unfolded_M = primary['unfolded']

    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    positive_spacings_riem = spacings_riem[spacings_riem > 0]

    s_range = np.linspace(0.01, 4.0, 200)

    # ---- Figure 1: Spacing distributions ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Histogram of M spacings
    ax.hist(spacings_M, bins=50, density=True, alpha=0.5, color='steelblue',
            label=f'Merkabit M (N={primary["N"]})', edgecolor='navy', linewidth=0.5)

    # Histogram of Riemann spacings
    ax.hist(positive_spacings_riem, bins=30, density=True, alpha=0.4, color='coral',
            label='Riemann zeros', edgecolor='darkred', linewidth=0.5)

    # GUE (beta=2) curve
    gue_curve = np.array([wigner_surmise(s, 2) for s in s_range])
    ax.plot(s_range, gue_curve, 'r-', linewidth=2.5, label='GUE (beta=2)')

    # GSE (beta=4) curve
    gse_curve = np.array([wigner_surmise(s, 4) for s in s_range])
    ax.plot(s_range, gse_curve, 'g--', linewidth=2.5, label='GSE (beta=4)')

    # Poisson
    poisson_curve = np.exp(-s_range)
    ax.plot(s_range, poisson_curve, 'k:', linewidth=2, label='Poisson')

    ax.set_xlabel('Normalised spacing s', fontsize=14)
    ax.set_ylabel('P(s)', fontsize=14)
    ax.set_title('Nearest-Neighbour Spacing Distribution: Merkabit M vs RMT', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure1_spacing_distributions.png', dpi=150)
    plt.close(fig)
    print("    figure1_spacing_distributions.png saved")

    # ---- Figure 2: Pair correlation ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    r_centers = primary['r_centers']
    g_r = primary['g_r']

    ax.plot(r_centers, g_r, 'b-', linewidth=1.5, alpha=0.7, label=f'Merkabit M (N={primary["N"]})')

    # Montgomery formula
    r_theory = np.linspace(0.01, 3.0, 200)
    g_montgomery = montgomery_formula(r_theory)
    ax.plot(r_theory, g_montgomery, 'r-', linewidth=2.5, label='Montgomery (GUE): 1-(sin(pi*r)/(pi*r))^2')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('r (normalised distance)', fontsize=14)
    ax.set_ylabel('g(r)', fontsize=14)
    ax.set_title('Pair Correlation Function: Merkabit M vs Montgomery Formula', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure2_pair_correlation.png', dpi=150)
    plt.close(fig)
    print("    figure2_pair_correlation.png saved")

    # ---- Figure 3: Scaling ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    Ns = [r['N'] for r in all_results]
    ks_gue_vals = [r['ks_gue'] for r in all_results]
    ks_gse_vals = [r['ks_gse'] for r in all_results]
    ks_poi_vals = [r['ks_poi'] for r in all_results]
    ks_riem_vals = [r['ks_riem'] for r in all_results]
    beta_vals = [r['beta_fit'] for r in all_results]

    axes[0].plot(Ns, ks_gue_vals, 'ro-', linewidth=2, markersize=8, label='KS(M, GUE)')
    axes[0].plot(Ns, ks_gse_vals, 'gs-', linewidth=2, markersize=8, label='KS(M, GSE)')
    axes[0].plot(Ns, ks_poi_vals, 'k^-', linewidth=2, markersize=8, label='KS(M, Poisson)')
    axes[0].plot(Ns, ks_riem_vals, 'b*-', linewidth=2, markersize=10, label='KS(M, Riemann)')
    axes[0].set_xlabel('N (lattice sites)', fontsize=13)
    axes[0].set_ylabel('KS statistic', fontsize=13)
    axes[0].set_title('KS Distance vs Lattice Size', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Ns, beta_vals, 'mo-', linewidth=2, markersize=8)
    axes[1].axhline(y=2, color='r', linestyle='--', label='GUE (beta=2)')
    axes[1].axhline(y=4, color='g', linestyle='--', label='GSE (beta=4)')
    axes[1].axhline(y=0, color='k', linestyle=':', label='Poisson (beta=0)')
    axes[1].set_xlabel('N (lattice sites)', fontsize=13)
    axes[1].set_ylabel('Level repulsion beta', fontsize=13)
    axes[1].set_title('Level Repulsion Exponent vs N', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure3_scaling.png', dpi=150)
    plt.close(fig)
    print("    figure3_scaling.png saved")

    # ---- Figure 4: Direct match scatter ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Use spacings for comparison
    n_plot = min(40, len(spacings_M), len(positive_spacings_riem))
    M_sorted = np.sort(spacings_M)[:n_plot]
    R_sorted = np.sort(positive_spacings_riem)[:n_plot]

    ax.scatter(R_sorted, M_sorted, c='steelblue', s=40, alpha=0.7, edgecolors='navy')
    lims = [0, max(np.max(R_sorted), np.max(M_sorted)) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('Riemann zero spacings (sorted)', fontsize=13)
    ax.set_ylabel('Merkabit M spacings (sorted)', fontsize=13)
    ax.set_title('Q-Q Plot: Merkabit vs Riemann Spacings', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure4_direct_match.png', dpi=150)
    plt.close(fig)
    print("    figure4_direct_match.png saved")

    # ---- Figure 5: Level number variance ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    L_vals = primary['L_vals']
    sigma2 = primary['sigma2']
    delta3 = primary['delta3']

    # Sigma^2(L)
    axes[0].plot(L_vals, sigma2, 'bo-', linewidth=2, markersize=8, label=f'Merkabit M (N={primary["N"]})')
    L_fine = np.linspace(0.5, 10, 100)
    gue_sigma2 = (2/np.pi**2) * np.log(2*np.pi*L_fine) + 0.44  # GUE + constant offset
    axes[0].plot(L_fine, gue_sigma2, 'r-', linewidth=2, label='GUE: (2/pi^2)ln(2piL)+c')
    axes[0].plot(L_fine, L_fine, 'k:', linewidth=2, label='Poisson: L')
    axes[0].set_xlabel('L', fontsize=13)
    axes[0].set_ylabel('Sigma^2(L)', fontsize=13)
    axes[0].set_title('Level Number Variance', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Delta_3(L)
    axes[1].plot(L_vals, delta3, 'bo-', linewidth=2, markersize=8, label=f'Merkabit M (N={primary["N"]})')
    gue_delta3 = (1/np.pi**2) * np.log(2*np.pi*L_fine) + 0.06  # GUE + constant
    axes[1].plot(L_fine, gue_delta3, 'r-', linewidth=2, label='GUE: (1/pi^2)ln(2piL)+c')
    axes[1].plot(L_fine, L_fine/15, 'k:', linewidth=2, label='Poisson: L/15')
    axes[1].set_xlabel('L', fontsize=13)
    axes[1].set_ylabel('Delta_3(L)', fontsize=13)
    axes[1].set_title('Spectral Rigidity (Dyson-Mehta)', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure5_spectral_rigidity.png', dpi=150)
    plt.close(fig)
    print("    figure5_spectral_rigidity.png saved")

    # ---- Figure 6: Spectrum near zero (kernel) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    evals = primary['eigenvalues']
    axes[0].hist(evals, bins=80, color='steelblue', edgecolor='navy', alpha=0.7)
    axes[0].set_xlabel('Eigenvalue lambda', fontsize=13)
    axes[0].set_ylabel('Count', fontsize=13)
    axes[0].set_title(f'Full Spectrum of M (N={primary["N"]})', fontsize=13)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='lambda=0 (kernel)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Zoom near zero
    near_zero_evals = evals[np.abs(evals) < 0.5]
    if len(near_zero_evals) > 0:
        axes[1].hist(near_zero_evals, bins=40, color='coral', edgecolor='darkred', alpha=0.7)
    axes[1].set_xlabel('Eigenvalue lambda', fontsize=13)
    axes[1].set_ylabel('Count', fontsize=13)
    axes[1].set_title(f'Spectrum Near Zero (|lambda| < 0.5)', fontsize=13)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='lambda=0')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'figure6_kernel.png', dpi=150)
    plt.close(fig)
    print("    figure6_kernel.png saved")


# ============================================================================
# F-CONNECTION CHECK
# ============================================================================

def check_F_connection():
    """
    Check numerical connections between F = 0.696778 and Riemann zeros.
    """
    F = F_RETURN
    gamma1 = 14.134725  # first Riemann zero

    print("\n" + "="*76)
    print("  F-CONNECTION CHECK: Return Fidelity vs Riemann Zeros")
    print("="*76)

    print(f"\n  F = {F:.6f}")
    print(f"  -ln(F) = {-np.log(F):.6f}")
    print(f"  gamma_1 = {gamma1:.6f}")
    print(f"  gamma_1 / (2*pi) = {gamma1/(2*np.pi):.6f}")
    print(f"  gamma_1 / 10 = {gamma1/10:.6f}")
    print(f"  -ln(F) / (2*pi) = {-np.log(F)/(2*np.pi):.6f}")

    # Route C: alpha^-1 = 137 - ln(F)/10
    alpha_inv = 137 - np.log(F)/10
    print(f"\n  Route C: alpha^-1 = 137 - ln(F)/10 = {alpha_inv:.6f}")
    print(f"  NIST alpha^-1 = 137.035999...")

    # Check: does -ln(F) relate to gamma_1?
    ratio = -np.log(F) / gamma1
    print(f"\n  -ln(F) / gamma_1 = {ratio:.6f}")
    print(f"  1/(4*pi) = {1/(4*np.pi):.6f}")
    print(f"  1/12 = {1/12:.6f}")

    # Check spectral gap connection
    print(f"\n  Spectral gap connection:")
    print(f"  -ln(F) = {-np.log(F):.6f}")
    print(f"  36/100 = {36/100:.6f}  (positive_roots / 100)")
    print(f"  -ln(F) - 36/100 = {-np.log(F) - 36/100:.6f}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    t_start = time.time()

    print("="*76)
    print("  R/R-BAR MERGER OPERATOR vs RIEMANN ZEROS")
    print("  Merkabit Torsion Channel: GUE Eigenvalue Spacing Comparison")
    print("="*76)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Architecture: E6 (rank={RANK_E6}, dim={DIM_E6}, h={COXETER_H})")
    print(f"  Coherence length xi = {XI}")
    print(f"  Return fidelity F = {F_RETURN}")
    print()

    # Load Riemann zeros
    print("-"*76)
    print("  MODULE 4: LOADING RIEMANN ZEROS")
    print("-"*76)
    riemann_zeros = load_riemann_zeros()
    _, spacings_riem = unfold_riemann_zeros(riemann_zeros)
    print(f"  Using {len(riemann_zeros)} Riemann zeros")
    print(f"  Range: [{riemann_zeros.min():.4f}, {riemann_zeros.max():.4f}]")
    print(f"  Mean unfolded spacing: {np.mean(spacings_riem[spacings_riem>0]):.4f}")

    # F-connection check
    check_F_connection()

    # Scaling analysis
    report_lines = []
    report_lines.append("R/R-BAR MERGER OPERATOR vs RIEMANN ZEROS: FULL REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Architecture: E6 (rank={RANK_E6}, dim={DIM_E6}, h={COXETER_H})")
    report_lines.append(f"Coherence length xi = {XI}")
    report_lines.append(f"Riemann zeros: {len(riemann_zeros)}")

    # Run for multiple N values
    N_values = [100, 200, 500, 1000, 2000]
    all_results = []

    for N in N_values:
        try:
            result = run_single_N(N, riemann_zeros, report_lines, verbose=True)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR at N={N}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n  NO RESULTS — all sizes failed.")
        return

    # Generate figures
    generate_figures(all_results, riemann_zeros)

    # Save spacings
    primary = all_results[-1]
    np.save(RESULTS_DIR / "unfolded_spacings_M.npy", primary['spacings'])
    np.save(RESULTS_DIR / "unfolded_spacings_riemann.npy",
            spacings_riem[spacings_riem > 0])

    # ====================================================================
    # DECISION MATRIX
    # ====================================================================
    print("\n" + "="*76)
    print("  DECISION MATRIX")
    print("="*76)

    p = primary
    best_class = "UNDETERMINED"
    # Which distribution fits best?
    ks_values = {
        'GUE (beta=2)': p['ks_gue'],
        'GSE (beta=4)': p['ks_gse'],
        'Poisson': p['ks_poi'],
    }
    best_match = min(ks_values, key=ks_values.get)

    print(f"\n  BEST FIT: {best_match} (KS = {ks_values[best_match]:.4f})")
    print(f"\n  Detailed results at N = {p['N']}:")
    print(f"    KS(M, GUE):     {p['ks_gue']:.4f}  (p = {p['p_gue']:.6f})")
    print(f"    KS(M, GSE):     {p['ks_gse']:.4f}  (p = {p['p_gse']:.6f})")
    print(f"    KS(M, Poisson): {p['ks_poi']:.4f}  (p = {p['p_poi']:.6f})")
    print(f"    KS(M, Riemann): {p['ks_riem']:.4f}  (p = {p['p_riem']:.6f})")
    print(f"    Beta:           {p['beta_fit']:.3f}  (R2 = {p['beta_r2']:.3f})")
    print(f"    Near-zero evals: {p['n_near_zero']}")
    print(f"    BK residual:    {p['bk_residual']:.4f}")
    print(f"    RMSD match:     {p['rmsd']:.4f}")

    # Scaling trend
    if len(all_results) > 2:
        ks_gue_trend = [r['ks_gue'] for r in all_results]
        ks_poi_trend = [r['ks_poi'] for r in all_results]
        gue_decreasing = all(ks_gue_trend[i] >= ks_gue_trend[i+1] - 0.02
                            for i in range(len(ks_gue_trend)-1))
        poi_increasing = all(ks_poi_trend[i] <= ks_poi_trend[i+1] + 0.02
                            for i in range(len(ks_poi_trend)-1))

        print(f"\n  SCALING TRENDS:")
        print(f"    KS(GUE) trend:     {['%.4f' % x for x in ks_gue_trend]}")
        print(f"    KS(Poisson) trend: {['%.4f' % x for x in ks_poi_trend]}")
        print(f"    GUE convergence (decreasing KS): {'YES' if gue_decreasing else 'NO'}")
        print(f"    Poisson divergence (increasing): {'YES' if poi_increasing else 'NO'}")

    # Classification
    print(f"\n  CLASSIFICATION:")
    if p['p_gue'] > 0.05 and p['p_riem'] > 0.05:
        verdict = "STRONG POSITIVE: M consistent with GUE AND Riemann zeros"
    elif p['p_gue'] > 0.05:
        verdict = "MODERATE POSITIVE: M consistent with GUE but differs from Riemann"
    elif p['p_gse'] > 0.05:
        verdict = "GSE CLASS: M encodes symplectic L-function family"
    elif p['ks_gue'] < p['ks_poi']:
        verdict = "PARTIAL: M closer to GUE than Poisson — level repulsion present"
    else:
        verdict = "NULL: M spacings match Poisson — no level repulsion"

    print(f"    {verdict}")

    if p['beta_fit'] > 1.5 and p['beta_fit'] < 2.5:
        print(f"    Beta ~ 2 confirms GUE universality class")
    elif p['beta_fit'] > 3.5:
        print(f"    Beta ~ 4 confirms GSE (quaternionic/symplectic) class")
    elif p['beta_fit'] < 0.5:
        print(f"    Beta ~ 0 confirms Poisson (no correlations)")
    else:
        print(f"    Beta = {p['beta_fit']:.2f} — intermediate (possible crossover)")

    # Save full report
    report_lines.append(f"\n{'='*76}")
    report_lines.append("DECISION MATRIX")
    report_lines.append(f"{'='*76}")
    report_lines.append(f"Best fit: {best_match}")
    report_lines.append(verdict)
    report_lines.append(f"Beta = {p['beta_fit']:.3f}")

    with open(RESULTS_DIR / "FULL_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Full report saved to {RESULTS_DIR / 'FULL_REPORT.txt'}")

    # Save KS statistics
    with open(RESULTS_DIR / "ks_test_results.txt", 'w', encoding='utf-8') as f:
        f.write("N\tKS_GUE\tp_GUE\tKS_GSE\tp_GSE\tKS_Poisson\tp_Poisson\tKS_Riemann\tp_Riemann\tbeta\n")
        for r in all_results:
            f.write(f"{r['N']}\t{r['ks_gue']:.6f}\t{r['p_gue']:.6f}\t"
                    f"{r['ks_gse']:.6f}\t{r['p_gse']:.6f}\t"
                    f"{r['ks_poi']:.6f}\t{r['p_poi']:.6f}\t"
                    f"{r['ks_riem']:.6f}\t{r['p_riem']:.6f}\t"
                    f"{r['beta_fit']:.4f}\n")
    print(f"  KS results saved to {RESULTS_DIR / 'ks_test_results.txt'}")

    # Save kernel analysis
    with open(RESULTS_DIR / "kernel_analysis.txt", 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(f"N={r['N']}: near-zero eigenvalues={r['n_near_zero']}, "
                    f"min|lambda|={r['min_abs_eval']:.6e}\n")
    print(f"  Kernel analysis saved to {RESULTS_DIR / 'kernel_analysis.txt'}")

    # Save Berry-Keating
    with open(RESULTS_DIR / "berry_keating_projection.txt", 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(f"N={r['N']}: lambda={r['bk_lambda']:.6f}, "
                    f"residual={r['bk_residual']:.4f}\n")
    print(f"  Berry-Keating results saved")

    # Save beta repulsion
    with open(RESULTS_DIR / "beta_repulsion.txt", 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(f"N={r['N']}: beta={r['beta_fit']:.4f}, R2={r['beta_r2']:.4f}\n")
    print(f"  Beta repulsion saved")

    # Save scaling
    with open(RESULTS_DIR / "scaling_results.txt", 'w', encoding='utf-8') as f:
        f.write("N\tKS_GUE\tKS_GSE\tKS_Poisson\tKS_Riemann\tbeta\n")
        for r in all_results:
            f.write(f"{r['N']}\t{r['ks_gue']:.6f}\t{r['ks_gse']:.6f}\t"
                    f"{r['ks_poi']:.6f}\t{r['ks_riem']:.6f}\t{r['beta_fit']:.4f}\n")

    # Save RMSD
    with open(RESULTS_DIR / "rmsd_eigenvalue_matching.txt", 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(f"N={r['N']}: RMSD={r['rmsd']:.4f}\n")

    elapsed = time.time() - t_start
    print(f"\n{'='*76}")
    print(f"  COMPLETE. Total runtime: {elapsed:.1f}s")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
