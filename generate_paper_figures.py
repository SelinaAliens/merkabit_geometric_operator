#!/usr/bin/env python3
"""
Generate 5 publication-quality figures for Paper 6.
Figs 1, 2, 5: regenerated from cached .npy data.
Figs 3, 4: copied from validated analysis scripts (peierls_flux, montgomery_comparison).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, shutil

OUT = "C:/Users/selin/merkabit_results/paper_figures"
MONT_DIR = "C:/Users/selin/merkabit_results/montgomery_comparison"
DED_DIR = "C:/Users/selin/merkabit_results/dedekind_comparison"
PEIERLS_DIR = "C:/Users/selin/merkabit_results/peierls_flux"
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# COMMON STYLE
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette
C_M = '#1f77b4'      # blue - M operator
C_RIEM = '#d62728'    # red - Riemann zeros
C_MONT = '#333333'    # dark grey - Montgomery formula
C_DED = '#9467bd'     # purple - Dedekind
C_LFUN = '#2ca02c'    # green - L-function

def montgomery_formula(r):
    """Montgomery pair correlation conjecture."""
    g = np.ones_like(r)
    mask = r > 0
    g[mask] = 1.0 - (np.sin(np.pi * r[mask]) / (np.pi * r[mask]))**2
    return g


# ============================================================================
# FIGURE 1: g(r) at L=30 - M operator vs Riemann zeros vs Montgomery
# ============================================================================
def fig1_pair_correlation():
    print("Generating Figure 1: Pair correlation at L=30...")
    r = np.load(os.path.join(MONT_DIR, "r_vals.npy"))
    g_M = np.load(os.path.join(DED_DIR, "g_M_L30_A27.npy"))
    g_riem = np.load(os.path.join(DED_DIR, "g_riem_A27.npy"))
    g_mont = montgomery_formula(r)

    # Compute RMS values
    rms_M = np.sqrt(np.mean((g_M - g_mont)**2))
    rms_R = np.sqrt(np.mean((g_riem - g_mont)**2))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(r, g_mont, color=C_MONT, ls='--', lw=2.0, label='Montgomery conjecture', zorder=1)
    ax.plot(r, g_riem, color=C_RIEM, ls='-.', lw=1.5, alpha=0.8,
            label=f'Riemann zeros (RMS = {rms_R:.4f})', zorder=2)
    ax.plot(r, g_M, color=C_M, ls='-', lw=2.0,
            label=f'M operator, L = 30 (RMS = {rms_M:.4f})', zorder=3)

    ax.set_xlabel(r'$r$ (units of mean spacing)')
    ax.set_ylabel(r'$g(r)$  pair correlation')
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.05, 1.4)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.axhline(1, color='grey', ls=':', lw=0.5, alpha=0.5)

    # Ratio annotation
    ax.text(0.03, 0.97, f'Ratio M/Riemann = {rms_M/rms_R:.2f}x',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.savefig(os.path.join(OUT, "fig1_pair_correlation.png"))
    plt.close(fig)
    print(f"  Saved. RMS: M={rms_M:.4f}, Riemann={rms_R:.4f}, ratio={rms_M/rms_R:.2f}x")


# ============================================================================
# FIGURE 2: Convergence RMS vs L
# ============================================================================
def fig2_convergence():
    print("Generating Figure 2: Convergence scaling...")
    r = np.load(os.path.join(MONT_DIR, "r_vals.npy"))
    g_mont = montgomery_formula(r)

    Ls = [12, 15, 18, 21, 24, 27, 30]
    rms_vals = []
    for L in Ls:
        g_M = np.load(os.path.join(MONT_DIR, f"g_M_L{L}.npy"))
        rms = np.sqrt(np.mean((g_M - g_mont)**2))
        rms_vals.append(rms)
        print(f"  L={L}: RMS = {rms:.4f}")

    rms_vals = np.array(rms_vals)
    Ls_arr = np.array(Ls, dtype=float)

    # Power-law fit: RMS ~ a * L^b
    log_L = np.log(Ls_arr)
    log_rms = np.log(rms_vals)
    coeffs = np.polyfit(log_L, log_rms, 1)
    slope = coeffs[0]
    fit_line = np.exp(np.polyval(coeffs, log_L))

    # Riemann reference
    g_riem = np.load(os.path.join(MONT_DIR, "g_riemann.npy"))
    rms_riem = np.sqrt(np.mean((g_riem - g_mont)**2))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.semilogy(Ls_arr, rms_vals, 'o-', color=C_M, lw=2, ms=8, mfc=C_M, mec='white', mew=1.5,
                label='M operator RMS', zorder=3)
    ax.semilogy(Ls_arr, fit_line, '--', color='grey', lw=1.2,
                label=f'Power-law fit: $L^{{{slope:.2f}}}$', zorder=2)
    ax.axhline(rms_riem, color=C_RIEM, ls=':', lw=1.5,
               label=f'Riemann zeros (N=1000): {rms_riem:.4f}', zorder=1)

    ax.set_xlabel(r'Lattice size $L$')
    ax.set_ylabel('RMS vs Montgomery')
    ax.set_xlim(10, 32)
    ax.set_ylim(0.05, 0.25)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks(Ls)

    # Annotate ratio at L=30
    ratio_30 = rms_vals[-1] / rms_riem
    ax.annotate(f'{ratio_30:.2f}x',
                xy=(30, rms_vals[-1]), xytext=(27, rms_vals[-1]*1.3),
                arrowprops=dict(arrowstyle='->', color='grey'),
                fontsize=10, color='grey')

    fig.savefig(os.path.join(OUT, "fig2_convergence.png"))
    plt.close(fig)
    print(f"  Saved. Slope = {slope:.2f}, ratio at L=30 = {ratio_30:.2f}x")


# ============================================================================
# FIGURE 3: Copy validated phi sweep from peierls_flux analysis
# ============================================================================
def fig3_phi_sweep():
    src = os.path.join(MONT_DIR, "pair_correlation_vs_phi.png")
    dst = os.path.join(OUT, "fig3_phi_sweep.png")
    shutil.copy2(src, dst)
    print(f"Figure 3: Copied phi sweep from {src}")


# ============================================================================
# FIGURE 4: Copy validated GOE->GUE spacing distributions
# ============================================================================
def fig4_goe_gue():
    src = os.path.join(PEIERLS_DIR, "fig2_spacing_distributions.png")
    dst = os.path.join(OUT, "fig4_goe_gue.png")
    shutil.copy2(src, dst)
    print(f"Figure 4: Copied GOE->GUE from {src}")


# ============================================================================
# FIGURE 5: Dedekind comparison - residuals
# ============================================================================
def fig5_dedekind():
    print("Generating Figure 5: Dedekind comparison at L=30...")
    r = np.load(os.path.join(MONT_DIR, "r_vals.npy"))
    g_M = np.load(os.path.join(DED_DIR, "g_M_L30_A27.npy"))
    g_riem = np.load(os.path.join(DED_DIR, "g_riem_A27.npy"))
    g_ded = np.load(os.path.join(DED_DIR, "g_ded_A27.npy"))

    res_riem = g_M - g_riem
    res_ded = g_M - g_ded

    rms_riem = np.sqrt(np.mean(res_riem**2))
    rms_ded = np.sqrt(np.mean(res_ded**2))
    improvement = (rms_riem - rms_ded) / rms_riem * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(r, res_riem, '-', color=C_RIEM, lw=2.0,
            label=f'M $-$ Riemann (RMS = {rms_riem:.4f})')
    ax.plot(r, res_ded, '--', color=C_DED, lw=2.0,
            label=f'M $-$ Dedekind (RMS = {rms_ded:.4f})')
    ax.axhline(0, color='grey', ls='-', lw=0.5, alpha=0.5)

    ax.set_xlabel(r'$r$ (units of mean spacing)')
    ax.set_ylabel(r'$\Delta g(r)$  residual')
    ax.set_xlim(0, 4.0)
    ax.legend(loc='upper right', framealpha=0.9)

    ax.text(0.03, 0.03, f'Dedekind {improvement:.1f}% closer to M',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.savefig(os.path.join(OUT, "fig5_dedekind.png"))
    plt.close(fig)
    print(f"  Saved. RMS: Riemann={rms_riem:.4f}, Dedekind={rms_ded:.4f}, improvement={improvement:.1f}%")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  PAPER 6 -- FIGURE GENERATION")
    print("=" * 60)

    fig1_pair_correlation()
    fig2_convergence()
    fig3_phi_sweep()
    fig4_goe_gue()
    fig5_dedekind()

    print("\n" + "=" * 60)
    print("  ALL 5 FIGURES GENERATED")
    print(f"  Output: {OUT}")
    print("=" * 60)
