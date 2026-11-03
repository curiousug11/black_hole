#!/usr/bin/env python3
"""Compare the thin-disk model with real observations.

Run ``python scripts/fetch_observations.py`` first, then (from the repository root)

    python scripts/compare_observations.py

Three comparisons are made (see data/observations/README.md for the data):

1. The SDSS composite quasar spectrum (Vanden Berk et al. 2001) against the
   model disk spectrum for a typical SDSS quasar.
2. The local continuum slopes of ~10^5 SDSS DR7 quasars (Shen et al. 2011)
   against the slope the model predicts for each quasar's own black hole mass
   and Eddington ratio.
3. RXTE/PCA X-ray spectra of the black hole binary LMC X-3, fitted with the
   model (folded through the instrument response) using the published mass,
   inclination and distance of the system.

Figures go to figures/obs*.png and tables to results/observations_*.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.ticker import NullFormatter
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from accretion_disk import ThinDisk, constants as C, eddington_luminosity, planck_nu  # noqa: E402
import run_analysis as style  # noqa: E402  (shared plot styling; sets rcParams)

plt = style.plt
DATA = ROOT / "data" / "observations"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Published system parameters
# ---------------------------------------------------------------------------
LMCX3 = dict(
    mass=6.98,          # Msun, Orosz et al. 2014, ApJ 794, 154
    mass_err=0.56,
    inclination=69.24,  # deg, Orosz et al. 2014
    inclination_err=0.72,
    distance=48.1,      # kpc, Orosz et al. 2009 (as adopted by Steiner et al. 2014)
    distance_err=2.2,
)
PCA_BAND = (3.0, 20.0)     # keV; Steiner et al. 2014 used 2.8-25 keV
PCA_SYSTEMATIC = 0.01      # 1% systematic per channel, as in Steiner et al. 2014
F_COL_STANDARD = 1.7       # spectral hardening factor (Shimura & Takahara 1995)

# Continuum windows of Shen et al. (2011), rest frame, Angstrom
SHEN_WINDOWS = {
    "CIV": [(1445.0, 1465.0), (1700.0, 1705.0)],
    "MGII": [(2200.0, 2700.0), (2900.0, 3090.0)],
    "HB": [(4435.0, 4700.0), (5100.0, 5535.0)],
}
LINE_LABELS = {"CIV": "C IV (1450–1705 Å)", "MGII": "Mg II (2200–3090 Å)",
               "HB": r"H$\beta$ (4435–5535 Å)"}
# Windows used by Vanden Berk et al. (2001) for the continuum power law
VDB_WINDOWS = [(1350.0, 1365.0), (4200.0, 4230.0)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window_wavelengths(windows, step=2.0):
    return np.concatenate([np.arange(lo, hi + 1e-9, step) for lo, hi in windows])


def model_alpha_lambda(disk, windows):
    """Power-law slope alpha_lambda (f_lambda ~ lambda^alpha) of the model in ``windows``.

    Fitted by least squares in log-log space over the window wavelengths, as
    done for the data by Shen et al. (2011).
    """
    lam = window_wavelengths(windows)
    nu = C.c / (lam * C.angstrom)
    L_lam = nu ** 2 * disk.luminosity_nu(nu)        # L_lambda = L_nu c / lambda^2
    return float(np.polyfit(np.log(lam), np.log(L_lam), 1)[0])


def tmax_for(log_mass, log_edd):
    """Model T_max [K] for log10(M/Msun) and log10(L/L_Edd); T_max ~ (lambda/M)^(1/4)."""
    ref = ThinDisk.from_eddington(1e8, 0.1, n_rings=50)
    return ref.T_max * (10 ** log_edd / 0.1) ** 0.25 * (10 ** log_mass / 1e8) ** -0.25


def alpha_grid(n=90):
    """Model alpha_lambda in each Shen window on a grid of T_max.

    For a fixed r_out / r_g the disk spectrum has a universal shape set only by
    T_max, so the grid is built at fixed mass by varying the Eddington ratio.
    """
    lams = np.geomspace(1e-7, 1e2, n)
    logT = np.empty(n)
    alphas = {k: np.empty(n) for k in SHEN_WINDOWS}
    for i, lam in enumerate(lams):
        d = ThinDisk.from_eddington(1e8, lam, n_rings=800)
        logT[i] = np.log10(d.T_max)
        for k, w in SHEN_WINDOWS.items():
            alphas[k][i] = model_alpha_lambda(d, w)
    # sanity check of the universality claim: a different (M, lambda) pair with the same T_max
    chk = ThinDisk.from_eddington(1e6, 1e-2 * 1e-2, n_rings=800)
    ref = ThinDisk.from_eddington(1e8, 1e-2, n_rings=800)
    assert abs(chk.T_max / ref.T_max - 1) < 1e-9
    assert abs(model_alpha_lambda(chk, SHEN_WINDOWS["MGII"])
               - model_alpha_lambda(ref, SHEN_WINDOWS["MGII"])) < 1e-6
    return logT, alphas


# ---------------------------------------------------------------------------
# 1. SDSS composite quasar spectrum
# ---------------------------------------------------------------------------

def composite_comparison(shen):
    lam, flam, err = np.loadtxt(DATA / "vandenberk2001_composite.txt", unpack=True)

    def window_median(windows):
        return np.array([np.median(flam[(lam >= lo) & (lam <= hi)]) for lo, hi in windows])

    # observed continuum slope through the Vanden Berk et al. windows
    f1, f2 = window_median(VDB_WINDOWS)
    l1, l2 = [np.mean(w) for w in VDB_WINDOWS]
    alpha_lam_obs = np.log(f2 / f1) / np.log(l2 / l1)
    alpha_nu_obs = -(alpha_lam_obs + 2.0)

    # representative SDSS quasar: median mass and Eddington ratio of the DR7 sample
    good = (shen.BAL_FLAG == 0) & np.isfinite(shen.LOGEDD_RATIO)
    logM = float(np.median(shen.LOGBH[good]))
    logE = float(np.median(shen.LOGEDD_RATIO[good]))
    lo_T, hi_T = np.percentile(tmax_for(shen.LOGBH[good], shen.LOGEDD_RATIO[good]), [16, 84])

    lam_model = np.geomspace(800, 9000, 600)
    nu_model = C.c / (lam_model * C.angstrom)
    ref_window = [VDB_WINDOWS[0]]

    def model_curve(disk):
        L_lam = nu_model ** 2 * disk.luminosity_nu(nu_model)
        nu_ref = C.c / (window_wavelengths(ref_window) * C.angstrom)
        norm = np.median(nu_ref ** 2 * disk.luminosity_nu(nu_ref))
        return L_lam / norm * window_median(ref_window)[0]

    fid = ThinDisk.from_eddington(10 ** logM, 10 ** logE)
    alpha_nu_model = -(np.log(model_curve(fid)[np.argmin(abs(lam_model - l2))]
                              / model_curve(fid)[np.argmin(abs(lam_model - l1))])
                       / np.log(l2 / l1) + 2.0)
    # disks spanning the 16-84% range of T_max in the DR7 sample
    lam_lo = 10 ** logE * (lo_T / fid.T_max) ** 4
    lam_hi = 10 ** logE * (hi_T / fid.T_max) ** 4
    band = [model_curve(ThinDisk.from_eddington(10 ** logM, x)) for x in (lam_lo, lam_hi)]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(lam_model, np.minimum(*band) * lam_model, np.maximum(*band) * lam_model,
                    color=style.CAT[1], alpha=0.18, lw=0,
                    label="thin disk, 16–84% range of SDSS $T_{max}$")
    ax.loglog(lam_model, model_curve(fid) * lam_model, color=style.CAT[1], lw=2,
              label=(f"thin disk, M = 10$^{{{logM:.1f}}}$ M$_\\odot$, "
                     f"L/L$_{{Edd}}$ = {10 ** logE:.2f}"))
    ax.loglog(lam, flam * lam, color=style.CAT[0], lw=0.9,
              label="SDSS composite (Vanden Berk et al. 2001)")
    pl = f1 * (lam_model / l1) ** alpha_lam_obs * lam_model
    ax.loglog(lam_model, pl, color=style.INK_2, ls="--", lw=1.0,
              label=f"composite continuum, $\\alpha_\\nu$ = {alpha_nu_obs:.2f}")
    for lo, hi in VDB_WINDOWS:
        ax.axvspan(lo, hi, color=style.INK_2, alpha=0.25, lw=0)
    ax.set_xlim(900, 8600)
    ax.set_ylim(np.nanmin(flam[lam > 1000] * lam[lam > 1000]) * 0.5, np.nanmax(flam * lam) * 6)
    ax.set_xlabel("Rest wavelength [Å]")
    ax.set_ylabel(r"$\lambda f_\lambda$ (normalised at 1350 Å)")
    ax.set_title("Composite quasar spectrum vs thin-disk model", loc="left")
    ax.legend(loc="upper center", fontsize=8)
    fig.savefig(FIG_DIR / "obs1_composite_quasar.png")
    plt.close(fig)
    return dict(alpha_nu_obs=alpha_nu_obs, alpha_nu_model=alpha_nu_model, logM=logM,
                logEdd=logE, T_max=fid.T_max)


# ---------------------------------------------------------------------------
# 2. Continuum slopes of SDSS DR7 quasars
# ---------------------------------------------------------------------------

def quasar_slope_comparison(shen):
    logT_grid, alpha_model = alpha_grid()
    base = (shen.BAL_FLAG == 0) & np.isfinite(shen.LOGEDD_RATIO) & (shen.LOGBH > 0)
    logT_q = np.log10(tmax_for(shen.LOGBH, shen.LOGEDD_RATIO))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    summary = []
    for j, line in enumerate(("CIV", "MGII", "HB")):
        a, e, sn = shen[f"ALPHA_{line}"], shen[f"ALPHA_{line}_ERR"], shen[f"LINE_MED_SN_{line}"]
        sel = base & (a != 0) & (e > 0) & (e < 0.3) & (sn >= 5)
        obs = a[sel].to_numpy()
        pred = np.interp(logT_q[sel], logT_grid, alpha_model[line])
        diff = obs - pred
        summary.append(dict(line=line, n=int(sel.sum()),
                            obs_median=float(np.median(obs)),
                            obs_p16=float(np.percentile(obs, 16)),
                            obs_p84=float(np.percentile(obs, 84)),
                            model_median=float(np.median(pred)),
                            model_p16=float(np.percentile(pred, 16)),
                            model_p84=float(np.percentile(pred, 84)),
                            diff_median=float(np.median(diff)),
                            frac_obs_bluer_than_model=float(np.mean(diff < 0))))

        ax = axes[0, j]
        bins = np.linspace(-4, 1.5, 56)
        ax.hist(obs, bins=bins, color=style.CAT[0], alpha=0.75, label="observed (Shen+2011)")
        ax.hist(pred, bins=bins, histtype="step", color=style.CAT[1], lw=2,
                label="thin-disk prediction")
        ax.axvline(-7 / 3, color=style.INK_2, ls=":", lw=1)
        ax.text(-7 / 3 - 0.05, ax.get_ylim()[1] * 0.45, r"$\nu^{1/3}$ limit", rotation=90,
                ha="right", va="top", fontsize=8, color=style.INK_2)
        ax.set_title(f"{LINE_LABELS[line]}   (N = {sel.sum():,})", loc="left", fontsize=11)
        ax.set_xlabel(r"continuum slope $\alpha_\lambda$  ($f_\lambda\propto\lambda^{\alpha_\lambda}$)")
        if j == 0:
            ax.set_ylabel("number of quasars")
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[1, j]
        edges = np.percentile(logT_q[sel], np.linspace(2, 98, 11))
        mid, med, q16, q84 = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (logT_q[sel] >= lo) & (logT_q[sel] < hi)
            if m.sum() < 30:
                continue
            mid.append(np.median(logT_q[sel][m]))
            med.append(np.median(obs[m]))
            q16.append(np.percentile(obs[m], 16))
            q84.append(np.percentile(obs[m], 84))
        mid, med = np.array(mid), np.array(med)
        ax.fill_between(mid, q16, q84, color=style.CAT[0], alpha=0.2, lw=0)
        ax.plot(mid, med, "o-", color=style.CAT[0], ms=5, label="observed median (16–84%)")
        tt = np.linspace(logT_q[sel].min(), logT_q[sel].max(), 200)
        ax.plot(tt, np.interp(tt, logT_grid, alpha_model[line]), color=style.CAT[1], lw=2,
                label="thin-disk model")
        ax.set_xlabel(r"model $\log T_{max}$ [K] from $M_{BH}$ and $L/L_{Edd}$")
        ax.set_ylim(-3.2, 0.5)
        if j == 0:
            ax.set_ylabel(r"$\alpha_\lambda$")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("SDSS DR7 quasars: observed continuum slopes vs thin-disk prediction for "
                 "each quasar's mass and accretion rate", x=0.02, ha="left", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "obs2_quasar_slopes.png")
    plt.close(fig)

    with open(RES_DIR / "observations_quasar_slopes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return summary


# ---------------------------------------------------------------------------
# 3. RXTE spectra of LMC X-3
# ---------------------------------------------------------------------------

@dataclass
class PCASpectrum:
    obsid: str
    date: str
    exposure: float
    e_lo: np.ndarray        # model energy grid [keV]
    e_hi: np.ndarray
    resp: np.ndarray        # (n_energy, n_channel) response incl. effective area [cm^2]
    ch_lo: np.ndarray       # channel energy bounds [keV]
    ch_hi: np.ndarray
    rate: np.ndarray        # background-subtracted count rate [counts/s]
    rate_err: np.ndarray
    use: np.ndarray         # channels inside PCA_BAND


def load_pca(obsid, date):
    tag = obsid.replace("-", "")
    d = DATA / "rxte_lmcx3"
    with fits.open(d / f"xp{tag}_s2.pha.gz") as s, fits.open(d / f"xp{tag}_b2.pha.gz") as b, \
            fits.open(d / f"xp{tag}.rsp.gz") as r:
        src, bkg = s["SPECTRUM"], b["SPECTRUM"]
        texp = float(src.header["EXPOSURE"])
        if abs(float(bkg.header["EXPOSURE"]) - texp) > 1e-3 * texp:
            raise ValueError(f"{obsid}: source and background exposures differ")
        rate = (src.data["COUNTS"] - bkg.data["COUNTS"]) / texp
        err = np.hypot(src.data["STAT_ERR"], bkg.data["STAT_ERR"]) / texp
        m = r["SPECRESP MATRIX"]
        offset = int(m.header.get("TLMIN4", 1))
        nchan = int(m.header["DETCHANS"])
        rows = m.data
        resp = np.zeros((len(rows), nchan))
        for i, row in enumerate(rows):
            vals, pos = np.asarray(row["MATRIX"]), 0
            fch, nch = np.atleast_1d(row["F_CHAN"]), np.atleast_1d(row["N_CHAN"])
            for g in range(int(row["N_GRP"])):
                c0, n = int(fch[g]) - offset, int(nch[g])
                resp[i, c0:c0 + n] = vals[pos:pos + n]
                pos += n
        eb = r["EBOUNDS"].data
        ch_lo, ch_hi = np.asarray(eb["E_MIN"], float), np.asarray(eb["E_MAX"], float)
        e_lo, e_hi = np.asarray(rows["ENERG_LO"], float), np.asarray(rows["ENERG_HI"], float)
    use = (ch_lo >= PCA_BAND[0]) & (ch_hi <= PCA_BAND[1]) & (err > 0)
    return PCASpectrum(obsid, date, texp, e_lo, e_hi, resp, ch_lo, ch_hi, rate, err, use)


class DiskPowerLawModel:
    """Colour-corrected thin disk (fixed M, i, D) plus a power law, in photons/cm^2/s per bin.

    Each ring radiates a diluted blackbody f^-4 B_nu(f T): the spectral
    hardening factor f shifts the colour temperature without changing the
    bolometric flux.  Because B_nu(sT) = s^3 B_{nu/s}(T), the disk spectrum for
    any (Mdot, f) is a rescaling of one precomputed template.
    """

    def __init__(self, e_lo, e_hi, mass_msun, inclination, distance_kpc, n_sub=4):
        self.mass = mass_msun * C.M_sun
        self.cos_i = np.cos(np.radians(inclination))
        self.D = distance_kpc * C.kpc
        self.mdot0 = ThinDisk.from_eddington(mass_msun, 0.1).mdot
        disk = ThinDisk(mass=self.mass, mdot=self.mdot0, n_rings=1500,
                        r_out=1e4 * C.G * self.mass / C.c ** 2)
        self.disk0 = disk
        # template S0(nu) = sum_k B_nu(T_k) A_k on a wide log grid
        self.nu_t = np.geomspace(1e14, 1e20, 3000)
        self.logS0 = np.log(np.maximum(disk._ring_sum(self.nu_t), 1e-300))
        # sub-sampled energy grid for bin integration (midpoint rule)
        w = (e_hi - e_lo)[:, None]
        self.E = (e_lo[:, None] + w * (np.arange(n_sub) + 0.5)[None, :] / n_sub)  # keV
        self.dE = w / n_sub
        self.e_lo, self.e_hi = e_lo, e_hi
        self.l_edd = eddington_luminosity(self.mass)

    def disk_photons(self, mdot, f_col):
        s = f_col * (mdot / self.mdot0) ** 0.25          # temperature scale factor
        nu = self.E * C.keV / C.h
        S = s ** 3 * np.exp(np.interp(np.log(nu / s), np.log(self.nu_t), self.logS0))
        F_nu = self.cos_i / self.D ** 2 * f_col ** -4 * S   # erg/s/cm^2/Hz
        F_E = F_nu * C.keV / C.h                           # erg/s/cm^2/keV
        return np.sum(F_E / (self.E * C.keV) * self.dE, axis=1)

    def powerlaw_photons(self, norm, gamma):
        if abs(gamma - 1.0) < 1e-6:
            return norm * np.log(self.e_hi / self.e_lo)
        return norm * (self.e_hi ** (1 - gamma) - self.e_lo ** (1 - gamma)) / (1 - gamma)

    def eddington_ratio(self, mdot):
        return mdot * C.c ** 2 / 12.0 / self.l_edd

    def T_max_keV(self, mdot):
        return self.disk0.T_max * (mdot / self.mdot0) ** 0.25 * C.k_B / C.keV

    def disk_bolometric_flux(self, mdot):
        """Observed bolometric disk flux [erg/s/cm^2] for a flat disk at inclination i."""
        return mdot * C.c ** 2 / 12.0 * self.cos_i / (2 * np.pi * self.D ** 2)


def fit_pca(spec, model, f_col=None):
    """Fit disk (+ power law) to one PCA spectrum.  ``f_col`` fixed if given."""
    sig = np.hypot(spec.rate_err, PCA_SYSTEMATIC * np.abs(spec.rate))[spec.use]
    data = spec.rate[spec.use]
    free_f = f_col is None

    def unpack(p):
        mdot = model.mdot0 * 10 ** p[0]
        f = p[1] if free_f else f_col
        norm, gamma = 10 ** p[-2], p[-1]
        return mdot, f, norm, gamma

    def counts(p, parts=False):
        mdot, f, norm, gamma = unpack(p)
        d = model.disk_photons(mdot, f) @ spec.resp
        pl = model.powerlaw_photons(norm, gamma) @ spec.resp
        return (d, pl) if parts else d + pl

    def resid(p):
        return (counts(p)[spec.use] - data) / sig

    p0 = [0.0] + ([1.7] if free_f else []) + [-2.0, 2.5]
    lo = [-3.0] + ([1.0] if free_f else []) + [-8.0, 1.5]
    hi = [2.0] + ([3.5] if free_f else []) + [1.0, 4.0]
    best = None
    for start_mdot in (-1.0, 0.0):          # two starts guard against local minima
        p0[0] = start_mdot
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    chi2 = 2 * best.cost
    dof = int(spec.use.sum()) - len(best.x)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * max(1.0, chi2 / dof)
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        perr = np.full(len(best.x), np.nan)
    mdot, f, norm, gamma = unpack(best.x)
    disk_c, pl_c = counts(best.x, parts=True)
    band = spec.use
    return dict(
        obsid=spec.obsid, date=spec.date, exposure_s=spec.exposure,
        eddington_ratio=model.eddington_ratio(mdot),
        eddington_ratio_err=model.eddington_ratio(mdot) * np.log(10) * perr[0],
        f_col=f, f_col_err=perr[1] if free_f else 0.0,
        T_col_keV=f * model.T_max_keV(mdot),
        disk_flux_cgs=model.disk_bolometric_flux(mdot),
        gamma=gamma, disk_count_fraction=float(disk_c[band].sum() / (disk_c + pl_c)[band].sum()),
        chi2=chi2, dof=dof, net_rate=float(data.sum()),
        _p=best.x, _counts=counts(best.x), _parts=(disk_c, pl_c),
    )


def lmcx3_comparison():
    index = [l.split(",") for l in (DATA / "rxte_lmcx3" / "observations.csv").read_text().splitlines()
             if l and not l.startswith("#")]
    index = [(o, t) for o, t, _, status in index if status == "ok"]
    specs = [load_pca(o, t) for o, t in index]
    model = DiskPowerLawModel(specs[0].e_lo, specs[0].e_hi, LMCX3["mass"],
                              LMCX3["inclination"], LMCX3["distance"])
    fits_free, fits_fixed = [], []
    for spec in specs:
        if not np.allclose(spec.e_lo, model.e_lo):
            m = DiskPowerLawModel(spec.e_lo, spec.e_hi, LMCX3["mass"],
                                  LMCX3["inclination"], LMCX3["distance"])
        else:
            m = model
        fits_free.append(fit_pca(spec, m))
        fits_fixed.append(fit_pca(spec, m, f_col=F_COL_STANDARD))
    # thermal-state selection, following Steiner et al. (2014): disk-dominated, good fit
    thermal = np.array([(r["disk_count_fraction"] > 0.75) and (r["chi2"] / r["dof"] < 2)
                        for r in fits_free])

    cols = ["obsid", "date", "exposure_s", "eddington_ratio", "eddington_ratio_err", "f_col",
            "f_col_err", "T_col_keV", "disk_flux_cgs", "gamma", "disk_count_fraction", "chi2",
            "dof", "chi2_fcol_1.7", "thermal_state"]
    with open(RES_DIR / "observations_lmcx3_fits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r, rf, th in zip(fits_free, fits_fixed, thermal):
            w.writerow([r["obsid"], r["date"], f"{r['exposure_s']:.0f}"]
                       + [f"{r[c]:.5g}" for c in cols[3:13]] + [f"{rf['chi2']:.2f}", int(th)])

    # ---- figure: example spectrum ------------------------------------------------
    th_idx = np.where(thermal)[0]
    ex = th_idx[np.argmin([abs(fits_free[i]["eddington_ratio"]
                               - np.median([fits_free[k]["eddington_ratio"] for k in th_idx]))
                           for i in th_idx])]
    spec, r, rf = specs[ex], fits_free[ex], fits_fixed[ex]
    u = spec.use
    emid = 0.5 * (spec.ch_lo + spec.ch_hi)[u]
    ewid = (spec.ch_hi - spec.ch_lo)[u]
    sig = np.hypot(spec.rate_err, PCA_SYSTEMATIC * np.abs(spec.rate))[u]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
    ax.errorbar(emid, spec.rate[u] / ewid, yerr=sig / ewid, xerr=ewid / 2, fmt="none",
                ecolor=style.INK, elinewidth=1, label="RXTE/PCA data (background subtracted)")
    ax.step(emid, r["_counts"][u] / ewid, where="mid", color=style.CAT[0], lw=1.8,
            label=f"thin disk + power law, $f_{{col}}$ = {r['f_col']:.2f} (free)")
    ax.step(emid, r["_parts"][0][u] / ewid, where="mid", color=style.CAT[0], lw=1, ls="--",
            label="disk component")
    ax.step(emid, rf["_counts"][u] / ewid, where="mid", color=style.CAT[1], lw=1.4,
            label=f"$f_{{col}}$ = {F_COL_STANDARD} fixed")
    ax.set_yscale("log")
    ax.set_ylabel("counts s$^{-1}$ keV$^{-1}$")
    ax.set_title(f"LMC X-3, RXTE obsid {spec.obsid} ({spec.date[:10]}), "
                 f"L/L$_{{Edd}}$ = {r['eddington_ratio']:.2f}", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    axr.axhline(0, color=style.INK_2, lw=0.8)
    axr.plot(emid, (spec.rate[u] - r["_counts"][u]) / sig, "o", ms=3.5, color=style.CAT[0],
             label=f"free $f_{{col}}$: $\\chi^2$/dof = {r['chi2']:.0f}/{r['dof']}")
    axr.plot(emid, (spec.rate[u] - rf["_counts"][u]) / sig, "s", ms=3, color=style.CAT[1],
             label=f"$f_{{col}}$ = 1.7: $\\chi^2$/dof = {rf['chi2']:.0f}/{rf['dof']}")
    axr.set_xscale("log")
    axr.xaxis.set_minor_formatter(NullFormatter())
    axr.set_xticks([3, 5, 10, 20])
    axr.set_xticklabels(["3", "5", "10", "20"])
    axr.set_ylim(-6.5, 6.5)
    axr.set_xlabel("Energy [keV]")
    axr.set_ylabel(r"(data$-$model)/$\sigma$")
    axr.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.savefig(FIG_DIR / "obs3_lmcx3_spectrum.png")
    plt.close(fig)

    # ---- figure: population ------------------------------------------------------
    lam = np.array([r["eddington_ratio"] for r in fits_free])
    lam_e = np.array([r["eddington_ratio_err"] for r in fits_free])
    fcol = np.array([r["f_col"] for r in fits_free])
    fcol_e = np.array([r["f_col_err"] for r in fits_free])
    Tcol = np.array([r["T_col_keV"] for r in fits_free])
    Fd = np.array([r["disk_flux_cgs"] for r in fits_free])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.errorbar(lam[thermal], fcol[thermal], xerr=lam_e[thermal], yerr=fcol_e[thermal], fmt="o",
                ms=5, color=style.CAT[0], ecolor=style.CAT[0], elinewidth=0.8,
                mec=style.SURFACE, label="thermal state")
    ax.plot(lam[~thermal], fcol[~thermal], "o", ms=5, color=style.CAT[1], mec=style.SURFACE,
            label="other (power law > 25% or poor fit)")
    med = np.median(fcol[thermal])
    ax.axhline(med, color=style.CAT[0], lw=1, ls="--")
    ax.axhspan(1.6, 1.8, color=style.INK_2, alpha=0.12, lw=0)
    ax.text(lam.min() * 1.05, 1.62, "theory: $f_{col}$ ≈ 1.6–1.8", fontsize=8, color=style.INK_2)
    ax.set_ylim(0.9, 2.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"fitted L/L$_{Edd}$ (model, $\eta$ = 1/12)")
    ax.set_ylabel(r"fitted hardening factor $f_{col}$")
    ax.set_title(f"(a) $f_{{col}}$ needed by the model (median {med:.2f})", loc="left")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for mask, color in [(thermal, style.CAT[0]), (~thermal, style.CAT[1])]:
        ax.loglog(Tcol[mask], Fd[mask], "o", ms=5, color=color, mec=style.SURFACE)
    tt = np.geomspace(Tcol.min() * 0.9, Tcol.max() * 1.1, 50)
    k = np.median(Fd[thermal] / Tcol[thermal] ** 4)
    ax.loglog(tt, k * tt ** 4, color=style.INK_2, ls="--", lw=1, label=r"$F\propto T^4$ (constant inner radius)")
    slope = np.polyfit(np.log(Tcol[thermal]), np.log(Fd[thermal]), 1)[0]
    ax.set_xlabel(r"colour temperature $f_{col}\,kT_{max}$ [keV]")
    ax.set_ylabel(r"bolometric disk flux [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_title(f"(b) Flux–temperature relation (fitted slope {slope:.2f})", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "obs4_lmcx3_population.png")
    plt.close(fig)

    # ---- systematic uncertainty of f_col from the adopted M, D and i -------------
    syst = []
    for label, key, sign in [("M -1 sigma", "mass", -1), ("M +1 sigma", "mass", +1),
                             ("D -1 sigma", "distance", -1), ("D +1 sigma", "distance", +1),
                             ("i -1 sigma", "inclination", -1), ("i +1 sigma", "inclination", +1)]:
        par = dict(LMCX3)
        par[key] = LMCX3[key] + sign * LMCX3[f"{key}_err"]
        fs = []
        for spec, th in zip(specs, thermal):
            if th:
                mv = DiskPowerLawModel(spec.e_lo, spec.e_hi, par["mass"], par["inclination"],
                                       par["distance"])
                fs.append(fit_pca(spec, mv)["f_col"])
        syst.append((label, f"{par[key]:.4g}", float(np.median(fs))))
    with open(RES_DIR / "observations_lmcx3_fcol_systematics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variation", "value", "median_f_col_thermal"])
        w.writerow(["fiducial", "", f"{np.median(fcol[thermal]):.4f}"])
        for row in syst:
            w.writerow([row[0], row[1], f"{row[2]:.4f}"])
    dev = np.array([row[2] for row in syst]) - np.median(fcol[thermal])
    syst_err = float(np.sqrt(np.sum([max(abs(dev[2 * k]), abs(dev[2 * k + 1])) ** 2
                                     for k in range(3)])))

    # pure (undiluted) blackbody rings, f_col = 1, for the thermal-state spectra
    chi_bb = np.array([(lambda r: r["chi2"] / r["dof"])(fit_pca(spec, model, f_col=1.0))
                       for spec, th in zip(specs, thermal) if th])

    chi_free = np.array([r["chi2"] / r["dof"] for r in fits_free])
    chi_fix = np.array([rf["chi2"] / rf["dof"] for rf in fits_fixed])
    return dict(n_obs=len(specs), n_thermal=int(thermal.sum()),
                fcol_median=float(med),
                fcol_p16=float(np.percentile(fcol[thermal], 16)),
                fcol_p84=float(np.percentile(fcol[thermal], 84)),
                lam_min=float(lam[thermal].min()), lam_max=float(lam[thermal].max()),
                FT_slope=float(slope), fcol_syst=syst_err, fcol_syst_rows=syst,
                chi2nu_free_median=float(np.median(chi_free[thermal])),
                chi2nu_fixed_median=float(np.median(chi_fix[thermal])),
                chi2nu_bb_median=float(np.median(chi_bb)),
                example_obsid=specs[ex].obsid)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    shen = pd.read_csv(DATA / "shen2011_dr7_quasars.csv.gz", comment="#")
    comp = composite_comparison(shen)
    slopes = quasar_slope_comparison(shen)
    xrb = lmcx3_comparison()

    lines = ["# Comparison with observations: summary", ""]
    lines += ["## 1. SDSS composite quasar spectrum (Vanden Berk et al. 2001)", "",
              f"* Composite continuum slope (1350–1365 Å to 4200–4230 Å): alpha_nu = {comp['alpha_nu_obs']:.3f}",
              f"* Thin-disk model, median DR7 quasar (log M = {comp['logM']:.2f}, "
              f"log L/L_Edd = {comp['logEdd']:.2f}, T_max = {comp['T_max']:.3g} K): "
              f"alpha_nu = {comp['alpha_nu_model']:.3f}", ""]
    lines += ["## 2. SDSS DR7 quasar continuum slopes (Shen et al. 2011)", "",
              "| Window | N | observed alpha_lambda median (16–84%) | model median (16–84%) | median obs − model | fraction redder than model |",
              "|---|---|---|---|---|---|"]
    for s in slopes:
        lines.append(f"| {s['line']} | {s['n']:,} | {s['obs_median']:.2f} ({s['obs_p16']:.2f}, {s['obs_p84']:.2f}) "
                     f"| {s['model_median']:.2f} ({s['model_p16']:.2f}, {s['model_p84']:.2f}) "
                     f"| {s['diff_median']:+.2f} | {1 - s['frac_obs_bluer_than_model']:.3f} |")
    lines += ["", "## 3. LMC X-3, RXTE/PCA (1996–1999)", "",
              f"* Observations fitted: {xrb['n_obs']}; thermal-state (disk > 75% of 3–20 keV counts, chi2/dof < 2): {xrb['n_thermal']}",
              f"* Fitted L/L_Edd range (thermal state): {xrb['lam_min']:.3f} – {xrb['lam_max']:.3f}",
              f"* Fitted f_col: median {xrb['fcol_median']:.3f} (16–84%: {xrb['fcol_p16']:.3f} – {xrb['fcol_p84']:.3f})",
              f"* Systematic error on f_col from M, D, i (each +/-1 sigma, added in quadrature): +/-{xrb['fcol_syst']:.3f}",
              "  " + "; ".join(f"{a}: {c:.3f}" for a, _, c in xrb["fcol_syst_rows"]),
              f"* Disk flux vs colour temperature: F ~ T^{xrb['FT_slope']:.2f} (constant inner radius predicts T^4)",
              f"* Median chi2/dof: f_col free {xrb['chi2nu_free_median']:.2f}; f_col = 1.7 fixed {xrb['chi2nu_fixed_median']:.2f}; "
              f"f_col = 1 (pure blackbody) {xrb['chi2nu_bb_median']:.1f}",
              f"* Example spectrum (Fig. obs3): obsid {xrb['example_obsid']}"]
    text = "\n".join(lines) + "\n"
    (RES_DIR / "observations_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
