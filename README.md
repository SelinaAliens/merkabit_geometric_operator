# Geometric Operator on the Eisenstein Lattice — Paper 6 Scripts

**Paper**: *Geometric Operator on the Eisenstein Lattice Converges to GUE Eigenvalue Statistics Under P-Gate Magnetic Flux*

**Authors**: Selina Stenberg with Claude Anthropic, March 2026

**Series**: The Merkabit Riemann Spectral Program — Paper 6

---

## Summary

This repository contains all computational scripts for Analyses 14–27 of the Merkabit Riemann spectral program. The central object is the R/R-bar merger operator **M**, constructed from geometric SU(2) Hopf-paired spinors on the Eisenstein lattice Z[omega] with zero free parameters.

### Key Results

| Result | Value |
|--------|-------|
| GOE -> GUE transition | KS(GUE) = 0.052, p = 0.57 at L = 18, Phi = 1/6 |
| Montgomery pair correlation | RMS = 0.118 at L = 30 (ratio 1.49x to Riemann zeros) |
| Dedekind zeta match | M fits zeta_{Q(omega)} 28.4% better than zeta(s) alone |
| Gate modulation | All 10 values from r = 6, h = 12, n = 5 (zero free parameters) |
| Route C | alpha^{-1} = 137.036 from E6 integers |

---

## Directory Structure

### Analysis 14: R/R-bar vs Riemann Zeros
`analysis_14_riemann_zeros/`
- `rr_bar_riemann_analysis.py` — Initial M operator construction, GOE wing detection
- `rr_bar_refined_analysis.py` — Three M variants (basic, Floquet, Peierls Formula C)
- `peierls_flux_sweep.py` — Flux sweep on open cell

**Result**: GOE wing with beta ~ 0.85. F-connection: -ln(F) = 0.36129 ~ pos_roots/100

### Analysis 15: Inter-Merkabit Tunnel Operator
`analysis_15_tunnel_operator/`
- `tunnel_operator_simulation.py` — Two-cell and multi-cell tunnel operator

**Result**: |0> blocked, |+1> and |-1> tunnel. Time-reversal broken. Beta -> 1.777 at N = 7.

### Analysis 16: Full Eisenstein Cell
`analysis_16_eisenstein_spectral/`
- `full_eisenstein_spectral.py` — Open Eisenstein cell, radii 1-10, 2x2 spinors

**Result**: F-connection pos_frac -> 0.36129 (Delta = 0.0016 at r = 10). Bipartite barrier identified.

### Analysis 17: Tesseract 4-Spinor M
`analysis_17_tesseract/`
- `tesseract_M_spectral.py` — Tesseract geometry, bipartite investigation

**Result**: Bipartite barrier is graph topology, not spinor. F-connection topological.

### Analysis 18: L(s, chi_{-3}) Comparison (Path A)
`analysis_18_lchi3/`
- `lchi3_comparison.py` — Direct comparison of M spectrum with L-function zeros

**Result**: FALSIFIED — chi_{-3} zeros are GUE (Katz-Sarnak), not GOE.

### Analysis 19: Peierls Flux Sweep (GOE -> GUE Transition)
`analysis_19_peierls_flux/`
- `peierls_flux.py` — **Primary script**: Formula A (Landau gauge), L = 6-18, phi sweep

**Result**: GOE -> GUE at Phi = 1/6. KS(GUE) = 0.052, p = 0.57 at L = 18. Geometric spinors essential.

### Analysis 19b: Commensurability / Standing-Wave Tests
`analysis_19b_commensurability/`
- `commensurability_test.py` — Random spinors (crude pipeline)
- `commensurability_cascade.py` — Three constructions (crude pipeline)
- `commensurability_geometric.py` — Geometric spinors (crude pipeline)
- `commensurability_final.py` — Exact Analysis 19 pipeline reproduction

**Result**: L mod 12 = {0, 6} gives GUE; L mod 12 = {3} gives bumps. Coxeter standing-wave pattern.

### Analysis 20: Montgomery Pair Correlation
`analysis_20_montgomery/`
- `montgomery_comparison.py` — **Core pipeline**: Gaussian KDE on all N(N-1) pairwise diffs, Riemann-von Mangoldt unfolding, phi sweep, L = 18 and L = 30
- `L24_convergence.py` — Size scaling L = 12 to L = 30
- `phi_minimum_check.py` — Bandwidth sensitivity check

**Result**: RMS(M, Montgomery) = 0.118 at L = 30. Phi = 1/6 minimizes RMS. Convergence confirmed.

### Analysis 21: Coherence-Eigenvalue Mapping + Lyapunov
`analysis_21_coherence_lyapunov/`
- `coherence_eigenvalue_mapping.py` — Near-zero eigenvalue characterization
- `cascade_phase_portrait.py` — Phase portrait, Lyapunov exponent, OTOC

**Result**: lambda = 0 exactly (not chaotic). Near-zero eigenvalues = ergodic band center. Montgomery from symmetry + ergodicity.

### Analysis 22: E6 Root Projection + Affine A4(1)
`analysis_22_e6_projection/`
- `e6_root_projection.py` — Spatial root projection (v1)
- `e6_root_projection_v2.py` — Weight decomposition (v2)
- `affine_a4_modulation.py` — Affine A4(1) modulation computation

**Result**: P gate = affine node of A4(1). S<->R Weyl symmetry. A4(1) completely node-symmetric; representation breaks symmetry.

### Analysis 23: E6 Root Projection onto A4(1) Eigenspace
`analysis_23_e6_a4_eigenspace/`
- `analysis23_projection.py` — Fundamental weight projections, Coxeter element eigendecomposition

**Result**: Modulation not from simple projection. 15/h = 5/4 exact. A4 weight lattice connection confirmed.

### Analysis 24: Z3 Sublattice Action on Gate Couplings
`analysis_24_z3_sublattice/`
- `analysis24_z3_action.py` — Gate modulation from three E6 integers

**Result**: All 10 gate values from r = 6, h = 12, n = 5. Route C closed analytically. Zero free parameters.

### Analysis 25: Eisenstein Euler Product
`analysis_25_euler_product/`
- `analysis25_eisenstein_euler.py` — Z3 = prime factorization in Z[omega], zero interleaving

**Result**: Gate sublattice = prime class (split/inert/ramified). Zero interleaving confirmed: mean run 1.31.

### Analysis 26: Dedekind Zeta at L = 18 (Formula Validation)
`analysis_26_dedekind_L18/`
- `analysis26_dedekind_comparison.py` — Formula A validation, Dedekind test at L = 18

**Result**: KS(GUE) = 0.049 (reproduces Analysis 19). Q1 = NO at L = 18 (7% — finite-size effect).

### Analysis 27: Dedekind Zeta at L = 30 (Definitive Test)
`analysis_27_dedekind_L30/`
- `analysis27_L30.py` — Exact Analysis 20 pipeline + Dedekind comparison at L = 30

**Result**: **M fits Dedekind 28.4% better than Riemann alone.** RMS(M, Dedekind) = 0.0362 vs RMS(M, Riemann) = 0.0506.

### Cached Data
`data/`
- `riemann_zeros_1000.npy` — First 1000 non-trivial Riemann zeta zeros
- `lchi3_zeros.npy` — L(s, chi_{-3}) zeros

---

## The Operator

The R/R-bar merger operator M on the Eisenstein torus T_L:

```
M[i,j] = exp(-|z_i - z_j| / xi) * exp(i * Phi * A_ij) * <u_i | v_j>
```

where:
- `u(a,b) = exp(i*pi*(a-b)/6) * [cos(pi*r/2), i*sin(pi*r/2)]` (geometric spinor)
- `v(a,b) = [-conj(u1), conj(u0)]` (SU(2) Hopf antipode)
- `A_ij = Phi * (2*a_i + da)/2 * db` (Landau gauge, Formula A)
- `xi = 3` (decay length), `Phi = 1/6` (P-gate flux)

Zero free parameters. Every quantity determined by the Eisenstein lattice geometry and E6 McKay correspondence.

---

## Dependencies

```
numpy, scipy, matplotlib, mpmath (for high-precision zeros)
```

---

## Citation

Stenberg, S. (2026). *Geometric Operator on the Eisenstein Lattice Converges to GUE Eigenvalue Statistics Under P-Gate Magnetic Flux*. The Merkabit Riemann Spectral Program — Paper 6.

See also:
- [1] Stenberg, S. (2026). The Merkabit. Zenodo, DOI: 10.5281/zenodo.18925475
- [2] Stenberg, S. (2026). alpha = 4/3 universally. Zenodo, DOI: 10.5281/zenodo.18980026
- [3] Stenberg, S. (2026). Fine structure constant from geometry. Zenodo, DOI: 10.5281/zenodo.18981288
