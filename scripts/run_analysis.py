#!/usr/bin/env python3
"""Run the full accretion-disk parameter study and write figures and tables.

Usage (from the repository root):

    python scripts/run_analysis.py

Outputs go to ``figures/`` (PNG) and ``results/`` (CSV + Markdown tables).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accretion_disk import ThinDisk, constants as C, eddington_luminosity, planck_nu  # noqa: E402

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Representative systems
# ---------------------------------------------------------------------------
STELLAR = dict(name="Stellar-mass BH", mass=10.0, distance=8.0 * C.kpc,
               dist_label="8 kpc", inclination=60.0)
SMBH = dict(name="Supermassive BH", mass=1.0e8, distance=100.0 * C.Mpc,
            dist_label="100 Mpc", inclination=30.0)
EDD_RATIO = 0.1  # fiducial L / L_Edd

# ---------------------------------------------------------------------------
# Plot styling (light surface, recessive axes, fixed categorical order)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#f6b892", "#f19a6a", "#eb6834", "#c4521f", "#9a3f14", "#6f2c0b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "font.size": 10,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

BANDS = [  # (label, nu_min, nu_max) in Hz
    ("radio/mm", 1e8, 3e11),
    ("IR", 3e11, 3.9e14),
    ("opt", 3.9e14, 7.9e14),
    ("UV", 7.9e14, 3e16),
    ("X-ray", 3e16, 3e19),
]


def shade_bands(ax, ymax_frac=0.97):
    """Draw recessive alternating background bands with labels."""
    lo, hi = ax.get_xlim()
    for j, (label, a, b) in enumerate(BANDS):
        a, b = max(a, lo), min(b, hi)
        if a >= b:
            continue
        if j % 2:
            ax.axvspan(a, b, color="#f0efec", zorder=0, lw=0)
        ax.text(np.sqrt(a * b), ymax_frac, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=INK_2)


def add_energy_axis(ax):
    """Top axis showing photon energy in keV (same quantity, different unit)."""
    k = C.h / C.keV
    sec = ax.secondary_xaxis("top", functions=(lambda nu: nu * k, lambda E: E / k))
    sec.set_xlabel("Photon energy [keV]", color=INK_2)
    sec.tick_params(colors=INK_2)
    return sec


def fmt_mass(m):
    if m < 1e3:
        return f"{m:g} M$_\\odot$"
    e = int(np.floor(np.log10(m)))
    mant = m / 10 ** e
    return (f"10$^{{{e}}}$ M$_\\odot$" if np.isclose(mant, 1)
            else f"{mant:g}$\\times$10$^{{{e}}}$ M$_\\odot$")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_temperature_profiles():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, sysd in zip(CAT, (STELLAR, SMBH)):
        d = ThinDisk.from_eddington(sysd["mass"], EDD_RATIO)
        x = np.geomspace(6.0, 1e5, 2000)
        T = d.temperature(x * d.r_g)
        ax.loglog(x, T, color=color, label=f"{sysd['name']} ({fmt_mass(sysd['mass'])})")
        ax.plot(d.r_T_max / d.r_g, d.T_max, "o", ms=6, color=color,
                mec=SURFACE, mew=1.5, zorder=5)
        e = int(np.floor(np.log10(d.T_max)))
        ax.annotate(f"T$_{{max}}$ = {d.T_max / 10**e:.2f}$\\times$10$^{{{e}}}$ K",
                    (d.r_T_max / d.r_g, d.T_max),
                    xytext=(10, 4), textcoords="offset points", color=INK, fontsize=9)
        # pure power law for reference
        ax.loglog(x, d.T_star * (6.0 / x) ** 0.75, ls=":", lw=1.0, color=color)
    ax.text(1.2e3, 3e5, r"dotted: $T\propto r^{-3/4}$ (no inner torque factor)",
            fontsize=8.5, color=INK_2)
    ax.axvline(6.0, color=INK_2, lw=0.8, ls="--")
    ax.text(6.4, 1.5e2, "r$_{in}$ = ISCO (6 GM/c$^2$)", fontsize=8.5, color=INK_2)
    ax.set_xlabel(r"Radius $r$ [$GM/c^2$]")
    ax.set_ylabel("Effective temperature T(r) [K]")
    ax.set_title(f"Radial temperature profile (L/L$_{{Edd}}$ = {EDD_RATIO})", loc="left")
    ax.set_ylim(1e2, 2e7)
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "fig1_temperature_profiles.png")
    plt.close(fig)


def fig_spectra_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    nu = np.geomspace(1e10, 1e19, 3000)
    for color, sysd in zip(CAT, (STELLAR, SMBH)):
        d = ThinDisk.from_eddington(sysd["mass"], EDD_RATIO)
        nuL = nu * d.luminosity_nu(nu)
        ax.loglog(nu, nuL, color=color, label=f"{sysd['name']} ({fmt_mass(sysd['mass'])})")
        nu_p = d.peak_frequency()
        nuL_p = nu_p * d.luminosity_nu(nu_p)[0]
        ax.plot(nu_p, nuL_p, "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=5)
        ax.annotate(f"$\\nu_{{peak}}$ = {nu_p:.2e} Hz\n({C.h * nu_p / C.keV:.3g} keV)",
                    (nu_p, nuL_p), xytext=(12, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlim(nu[0], nu[-1])
    ax.set_ylim(1e30, 1e48)
    shade_bands(ax)
    add_energy_axis(ax)
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
    ax.set_title("Same physics, different characteristic frequency", loc="left", pad=28)
    ax.legend(loc="center left")
    fig.savefig(FIG_DIR / "fig2_spectra_stellar_vs_smbh.png")
    plt.close(fig)


def fig_vary_mass():
    masses = [10.0, 1e3, 1e5, 1e7, 1e9]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    nu = np.geomspace(1e11, 1e19, 3000)
    for color, m in zip(BLUE_RAMP[1:], masses):
        d = ThinDisk.from_eddington(m, EDD_RATIO)
        ax.loglog(nu, nu * d.luminosity_nu(nu), color=color, label=fmt_mass(m))
        nu_p = d.peak_frequency()
        ax.plot(nu_p, nu_p * d.luminosity_nu(nu_p)[0], "o", ms=5, color=color,
                mec=SURFACE, mew=1.2)
    # trace of the peaks: nu_peak ~ M^-1/4, nuL_peak ~ M
    mm = np.geomspace(3, 3e9, 50)
    peaks = [ThinDisk.from_eddington(m, EDD_RATIO, n_rings=400) for m in mm]
    pnu = np.array([p.peak_frequency() for p in peaks])
    pL = np.array([f * p.luminosity_nu(f)[0] for f, p in zip(pnu, peaks)])
    ax.loglog(pnu, pL, ls="--", lw=1.0, color=INK_2)
    ax.text(pnu[-8] * 1.3, pL[-8], r"peak locus: $\nu_{peak}\propto M^{-1/4}$",
            fontsize=8.5, color=INK_2)
    ax.set_xlim(nu[0], nu[-1])
    ax.set_ylim(1e33, 3e47)
    shade_bands(ax)
    add_energy_axis(ax)
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
    ax.set_title(f"Effect of black hole mass (fixed L/L$_{{Edd}}$ = {EDD_RATIO})",
                 loc="left", pad=28)
    ax.legend(title="Mass", loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fig.savefig(FIG_DIR / "fig3_vary_mass.png")
    plt.close(fig)


def fig_vary_accretion_rate():
    ratios = [0.01, 0.03, 0.1, 0.3, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, sysd, ramp in zip(axes, (STELLAR, SMBH), (BLUE_RAMP, ORANGE_RAMP)):
        d0 = ThinDisk.from_eddington(sysd["mass"], 1.0)
        nu = d0.frequency_grid(2000, low=1e-4, high=15)
        for color, lam in zip(ramp[1:], ratios):
            d = ThinDisk.from_eddington(sysd["mass"], lam)
            mdot_yr = d.mdot * C.yr / C.M_sun
            ax.loglog(nu, nu * d.luminosity_nu(nu), color=color,
                      label=f"{lam:g}  ({mdot_yr:.1e} M$_\\odot$/yr)")
        ax.set_xlim(nu[0], nu[-1])
        top = (nu * d0.luminosity_nu(nu)).max()
        ax.set_ylim(top * 1e-6, top * 5)
        ax.set_xlabel(r"Frequency $\nu$ [Hz]")
        ax.set_title(f"{sysd['name']}, M = {fmt_mass(sysd['mass'])}", loc="left")
        ax.legend(title=r"L/L$_{Edd}$  ($\dot M$)", loc="lower center", fontsize=8)
    axes[0].set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
    fig.suptitle(r"Effect of accretion rate: $L\propto\dot M$, "
                 r"$\nu_{peak}\propto\dot M^{1/4}$", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig4_vary_accretion_rate.png")
    plt.close(fig)


def fig_scalings():
    masses = np.geomspace(3, 3e9, 40)
    ratios = [0.01, 0.1, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    for color, lam in zip(CAT, ratios):
        nup = [ThinDisk.from_eddington(m, lam, n_rings=400).peak_frequency() for m in masses]
        ax.loglog(masses, nup, color=color, label=f"L/L$_{{Edd}}$ = {lam:g}")
    ax.set_xlabel(r"Black hole mass [M$_\odot$]")
    ax.set_ylabel(r"$\nu_{peak}$ of $\nu L_\nu$ [Hz]")
    ax.set_title(r"(a) $\nu_{peak}\propto M^{-1/4}$ at fixed L/L$_{Edd}$", loc="left")
    ax.legend()

    ax = axes[1]
    lam = np.geomspace(1e-3, 1, 30)
    for color, m in zip(CAT, (10.0, 1e5, 1e8)):
        nup = [ThinDisk.from_eddington(m, l, n_rings=400).peak_frequency() for l in lam]
        ax.loglog(lam, nup, color=color, label=fmt_mass(m))
    ax.set_xlabel(r"Eddington ratio L/L$_{Edd}$")
    ax.set_ylabel(r"$\nu_{peak}$ [Hz]")
    ax.set_title(r"(b) $\nu_{peak}\propto\dot M^{1/4}$ at fixed M", loc="left")
    ax.legend()

    ax = axes[2]
    mdot = np.geomspace(1e14, 1e27, 50)
    ax.loglog(mdot, mdot * C.c ** 2 / 12.0, color=INK_2, lw=1.0, ls="--",
              label=r"$L=\dot M c^2/12$")
    for color, m in zip(CAT, (10.0, 1e5, 1e8)):
        md = np.geomspace(1e-3, 1, 6) * ThinDisk.from_eddington(m, 1.0).mdot
        L = [ThinDisk(mass=m * C.M_sun, mdot=x, n_rings=400).bolometric_luminosity() for x in md]
        ax.loglog(md, L, "o", ms=6, color=color, mec=SURFACE, mew=1.2,
                  label=f"{fmt_mass(m)} (up to L$_{{Edd}}$)")
    ax.set_xlabel(r"Accretion rate $\dot M$ [g s$^{-1}$]")
    ax.set_ylabel(r"Bolometric luminosity [erg s$^{-1}$]")
    ax.set_title("(c) L depends on $\\dot M$ only", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_scaling_relations.png")
    plt.close(fig)


def fig_observed_flux():
    incs = [0.0, 30.0, 60.0, 85.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sysd in zip(axes, (STELLAR, SMBH)):
        d = ThinDisk.from_eddington(sysd["mass"], EDD_RATIO)
        nu = d.frequency_grid(2000, low=1e-4, high=15)
        for color, inc in zip(BLUE_RAMP[2:][::-1], incs):
            F = d.flux_nu(nu, sysd["distance"], inc)
            lw = 2.4 if inc == sysd["inclination"] else 1.5
            ax.loglog(nu, nu * F, color=color, lw=lw, label=f"i = {inc:g}°"
                      + ("  (fiducial)" if inc == sysd["inclination"] else ""))
        ax.set_xlim(nu[0], nu[-1])
        top = (nu * d.flux_nu(nu, sysd["distance"], 0.0)).max()
        ax.set_ylim(top * 1e-5, top * 3)
        ax.set_xlabel(r"Frequency $\nu$ [Hz]")
        ax.set_title(f"{sysd['name']} ({fmt_mass(sysd['mass'])}) at {sysd['dist_label']}",
                     loc="left")
        ax.legend(loc="lower center", fontsize=8)
    axes[0].set_ylabel(r"$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]")
    fig.suptitle(r"Observed flux: $F_\nu = (\cos i / D^2)\,\sum_k B_\nu(T_k)A_k$"
                 f"   (L/L$_{{Edd}}$ = {EDD_RATIO})", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig6_observed_flux.png")
    plt.close(fig)


def fig_convergence():
    n_list = np.unique(np.geomspace(5, 5000, 25).astype(int))
    err_L, err_nu = [], []
    ref = ThinDisk.from_eddington(10.0, EDD_RATIO, n_rings=20000)
    nu_ref = ref.peak_frequency()
    for n in n_list:
        d = ThinDisk.from_eddington(10.0, EDD_RATIO, n_rings=int(n))
        err_L.append(abs(d.bolometric_luminosity() / d.analytic_luminosity() - 1))
        err_nu.append(abs(d.peak_frequency() / nu_ref - 1))
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.loglog(n_list, err_L, "o-", ms=4, color=CAT[0], label="bolometric luminosity vs analytic")
    ax.loglog(n_list, err_nu, "s-", ms=4, color=CAT[1], label="peak frequency vs 20 000 rings")
    ax.loglog(n_list, 3 * n_list ** -2.0, ls=":", color=INK_2, lw=1.0, label=r"$\propto N^{-2}$")
    ax.set_xlabel("Number of rings N (log-spaced, 6 to 10$^5$ GM/c$^2$)")
    ax.set_ylabel("Fractional error")
    ax.set_title("Numerical convergence of the ring sum", loc="left")
    ax.legend()
    fig.savefig(FIG_DIR / "fig7_convergence.png")
    plt.close(fig)


def fig_ring_decomposition():
    """Show how individual blackbody zones add up to the disk spectrum."""
    d = ThinDisk.from_eddington(STELLAR["mass"], EDD_RATIO)
    nu = np.geomspace(1e13, 3e18, 1500)
    radii, T, A = d.ring_radii, d.ring_temperatures, d.ring_areas
    zones = np.array_split(np.arange(d.n_rings), 8)
    colors = plt.cm.Blues_r(np.linspace(0.05, 0.7, len(zones)))  # dark = hot inner zone
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (idx, color) in enumerate(zip(zones, colors)):
        Lz = 2 * np.pi * planck_nu(nu[:, None], T[None, idx]) @ A[idx]
        label = (f"{radii[idx[0]] / d.r_g:,.0f}–{radii[idx[-1]] / d.r_g:,.0f} GM/c$^2$"
                 if j in (0, 3, 7) else None)
        ax.loglog(nu, nu * Lz, lw=1.2, color=color, label=label)
    ax.loglog(nu, nu * d.luminosity_nu(nu), color=INK, lw=2.4, label="sum over all rings")
    ax.set_ylim(1e32, 3e38)
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
    ax.set_title(f"Ring decomposition, {fmt_mass(STELLAR['mass'])}, "
                 f"L/L$_{{Edd}}$ = {EDD_RATIO}", loc="left")
    ax.legend(title="ring zone (8 zones of 250 rings)", loc="upper left", fontsize=8.5)
    fig.savefig(FIG_DIR / "fig8_ring_decomposition.png")
    plt.close(fig)


def fig_agnpy_crosscheck():
    """Compare the ring sum with agnpy's own SSDisk SED integration."""
    m = STELLAR["mass"]
    r_g = C.G * m * C.M_sun / C.c ** 2
    D, inc = STELLAR["distance"], STELLAR["inclination"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for color, rout, rlab in zip((CAT[0], CAT[1]), (50.0, 1e5), ("50", "10$^5$")):
        d = ThinDisk.from_eddington(m, EDD_RATIO, r_out=rout * r_g)
        nu = np.geomspace(1e13, 3e18, 600)
        ax.loglog(nu, nu * d.flux_nu(nu, D, inc), color=color,
                  label=f"ring sum, r$_{{out}}$ = {rlab} GM/c$^2$")
        ax.loglog(nu, np.maximum(d.agnpy_nuFnu(nu, D, inc), 1e-300), color=color, ls="--",
                  lw=1.4, label=f"agnpy SSDisk, r$_{{out}}$ = {rlab} GM/c$^2$")
    ax.set_ylim(1e-14, 3e-8)
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_title("(a) Spectra: ring sum vs agnpy's built-in integral", loc="left")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for color, rout in zip(BLUE_RAMP[2:], (20.0, 50.0, 200.0, 1000.0)):
        d = ThinDisk.from_eddington(m, EDD_RATIO, r_out=rout * r_g)
        nu = d.frequency_grid(500, low=1e-3, high=10)
        ours = nu * d.flux_nu(nu, D, inc)
        sel = ours > 1e-3 * ours.max()
        x = C.h * nu[sel] / (C.k_B * d.T_max)
        ax.semilogx(x, d.agnpy_nuFnu(nu, D, inc)[sel] / ours[sel], color=color,
                    label=f"r$_{{out}}$ = {rout:g} GM/c$^2$")
    ax.axhline(1.0, color=INK_2, lw=0.8, ls=":")
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel(r"$h\nu / kT_{max}$")
    ax.set_ylabel("agnpy / ring sum")
    ax.set_title("(b) Ratio: agrees to 0.1% for compact disks", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_agnpy_crosscheck.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_rows():
    rows = []
    for sysd in (STELLAR, SMBH):
        for lam in (0.01, 0.1, 1.0):
            d = ThinDisk.from_eddington(sysd["mass"], lam)
            nu_p = d.peak_frequency()
            nu = d.frequency_grid(4000)
            F = d.flux_nu(nu, sysd["distance"], sysd["inclination"])
            rows.append({
                "system": sysd["name"],
                "M_Msun": sysd["mass"],
                "L_over_LEdd": lam,
                "Mdot_g_s": d.mdot,
                "Mdot_Msun_yr": d.mdot * C.yr / C.M_sun,
                "L_bol_erg_s": d.bolometric_luminosity(),
                "L_Edd_erg_s": eddington_luminosity(d.mass),
                "T_max_K": d.T_max,
                "nu_peak_Hz": nu_p,
                "E_peak_keV": C.h * nu_p / C.keV,
                "lambda_peak_A": C.c / nu_p / C.angstrom,
                "distance_Mpc": sysd["distance"] / C.Mpc,
                "inclination_deg": sysd["inclination"],
                "F_bol_erg_s_cm2": np.trapezoid(F, nu),
                "nuFnu_peak_erg_s_cm2": float(np.max(nu * F)),
            })
    return rows


def write_tables(rows):
    with open(RES_DIR / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})

    hdr = ("| System | M [M☉] | L/L_Edd | Ṁ [M☉/yr] | L_bol [erg/s] | T_max [K] "
           "| ν_peak [Hz] | E_peak [keV] | λ_peak [Å] | D, i | F_bol [erg/s/cm²] |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        dist = (f"{r['distance_Mpc'] * 1e3:g} kpc" if r["distance_Mpc"] < 1
                else f"{r['distance_Mpc']:g} Mpc")
        lines.append(
            f"| {r['system']} | {r['M_Msun']:.0e} | {r['L_over_LEdd']:g} | {r['Mdot_Msun_yr']:.2e} "
            f"| {r['L_bol_erg_s']:.2e} | {r['T_max_K']:.2e} | {r['nu_peak_Hz']:.2e} "
            f"| {r['E_peak_keV']:.3g} | {r['lambda_peak_A']:.3g} | {dist}, {r['inclination_deg']:g}° "
            f"| {r['F_bol_erg_s_cm2']:.2e} |")
    (RES_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    fig_temperature_profiles()
    fig_spectra_comparison()
    fig_vary_mass()
    fig_vary_accretion_rate()
    fig_scalings()
    fig_observed_flux()
    fig_convergence()
    fig_ring_decomposition()
    fig_agnpy_crosscheck()
    print(write_tables(summary_rows()))
    print(f"\nFigures written to {FIG_DIR}\nTables written to {RES_DIR}")


if __name__ == "__main__":
    main()
