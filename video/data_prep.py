#!/usr/bin/env python3
"""Compute every curve shown in the video from the project's model and data.

Writes video/data/video_data.npz (and a small JSON of scalar facts), so the
Manim scenes only load arrays and never run the physics themselves.

    python video/data_prep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from accretion_disk import ThinDisk, constants as C, planck_nu  # noqa: E402
import compare_observations as co  # noqa: E402

OUT = HERE / "data"


def main():
    OUT.mkdir(exist_ok=True)
    d = {}
    facts = {}

    # ---- S2: temperature profile of the 10 Msun disk (lambda = 0.1) ----------------
    st = ThinDisk.from_eddington(10.0, 0.1)
    x = np.geomspace(6.0, 1e5, 600)
    T = st.temperature(x * st.r_g)
    d["T_x"], d["T_T"] = x, T
    d["T_pow"] = st.T_star * (6.0 / x) ** 0.75
    facts["T_max"] = st.T_max
    facts["r_Tmax_rg"] = st.r_T_max / st.r_g

    # ---- S3: ring decomposition of the spectrum (L_nu) ------------------------------
    nu = np.geomspace(1e12, 3e18, 700)
    zones = np.array_split(np.arange(st.n_rings), 8)
    TT, A = st.ring_temperatures, st.ring_areas
    zone_L = np.array([2 * np.pi * planck_nu(nu[:, None], TT[None, z]) @ A[z] for z in zones])
    d["sp_nu"], d["sp_zone_L"], d["sp_total_L"] = nu, zone_L, st.luminosity_nu(nu)
    d["sp_zone_r"] = np.array([[st.ring_radii[z[0]] / st.r_g, st.ring_radii[z[-1]] / st.r_g]
                               for z in zones])
    d["sp_zone_T"] = np.array([np.median(TT[z]) for z in zones])

    # ---- S4: universal nuL_nu shape at 10 Msun, lambda = 0.1 -------------------------
    nu4 = np.geomspace(1e9, 1e20, 1500)
    d["sc_nu"], d["sc_nuL"] = nu4, nu4 * st.luminosity_nu(nu4)
    smbh = ThinDisk.from_eddington(1e8, 0.1)
    for name, disk in (("stellar", st), ("smbh", smbh)):
        nup = disk.peak_frequency()
        facts[f"{name}_nu_peak"] = nup
        facts[f"{name}_E_keV"] = C.h * nup / C.keV
        facts[f"{name}_lambda_A"] = C.c / nup / C.angstrom
        facts[f"{name}_L"] = disk.bolometric_luminosity()
    facts["peak_ratio"] = facts["stellar_nu_peak"] / facts["smbh_nu_peak"]

    # ---- S6: convergence of the ring sum ----------------------------------------------
    N = np.unique(np.geomspace(5, 5000, 18).astype(int))
    err = [abs(ThinDisk.from_eddington(10, 0.1, n_rings=int(n)).bolometric_luminosity()
               / st.analytic_luminosity() - 1) for n in N]
    d["cv_N"], d["cv_err"] = N.astype(float), np.array(err)
    facts["L_err_2000"] = abs(st.bolometric_luminosity() / st.analytic_luminosity() - 1)

    # ---- S7a: SDSS composite vs median-quasar disk -----------------------------------
    import pandas as pd
    shen = pd.read_csv(co.DATA / "shen2011_dr7_quasars.csv.gz", comment="#")
    lam, flam, _ = np.loadtxt(co.DATA / "vandenberk2001_composite.txt", unpack=True)
    keep = (lam >= 1000) & (lam <= 8500)
    lam, flam = lam[keep], flam[keep]
    # smooth lightly (5 A median) and thin out for drawing
    k = 5
    lam_s = lam[: len(lam) // k * k].reshape(-1, k).mean(1)
    flam_s = np.median(flam[: len(flam) // k * k].reshape(-1, k), axis=1)
    d["cp_lam"], d["cp_lflam"] = lam_s, lam_s * flam_s
    good = (shen.BAL_FLAG == 0) & np.isfinite(shen.LOGEDD_RATIO)
    logM = float(np.median(shen.LOGBH[good]))
    logE = float(np.median(shen.LOGEDD_RATIO[good]))
    q = ThinDisk.from_eddington(10 ** logM, 10 ** logE)
    lm = np.geomspace(1000, 8500, 400)
    nu_m = C.c / (lm * C.angstrom)
    Llam = nu_m ** 2 * q.luminosity_nu(nu_m)
    ref = (lam >= 1350) & (lam <= 1365)
    nu_ref = C.c / (np.arange(1350, 1366, 1.0) * C.angstrom)
    norm = np.median(flam[ref]) / np.median(nu_ref ** 2 * q.luminosity_nu(nu_ref))
    d["cp_mlam"], d["cp_mlflam"] = lm, lm * Llam * norm
    facts["quasar_logM"], facts["quasar_logEdd"] = logM, logE

    # ---- S7b: continuum-slope histograms ----------------------------------------------
    logT_grid, alpha_model = co.alpha_grid(n=70)
    base = (shen.BAL_FLAG == 0) & np.isfinite(shen.LOGEDD_RATIO) & (shen.LOGBH > 0)
    logT_q = np.log10(co.tmax_for(shen.LOGBH, shen.LOGEDD_RATIO))
    bins = np.linspace(-3.5, 1.0, 46)
    d["hist_bins"] = bins
    for line in ("CIV", "MGII", "HB"):
        a, e, sn = shen[f"ALPHA_{line}"], shen[f"ALPHA_{line}_ERR"], shen[f"LINE_MED_SN_{line}"]
        sel = base & (a != 0) & (e > 0) & (e < 0.3) & (sn >= 5)
        obs = a[sel].to_numpy()
        pred = np.interp(logT_q[sel], logT_grid, alpha_model[line])
        d[f"hist_obs_{line}"] = np.histogram(obs, bins)[0].astype(float)
        d[f"hist_mod_{line}"] = np.histogram(pred, bins)[0].astype(float)
        facts[f"{line}_n"] = int(sel.sum())
        facts[f"{line}_obs_med"] = float(np.median(obs))
        facts[f"{line}_mod_med"] = float(np.median(pred))
        facts[f"{line}_redder"] = float(np.mean(obs > pred))

    # ---- S7c: LMC X-3 example spectrum and population ---------------------------------
    import csv
    rows = list(csv.DictReader(open(ROOT / "results" / "observations_lmcx3_fits.csv")))
    th = [r for r in rows if r["thermal_state"] == "1"]
    d["lx_lam"] = np.array([float(r["eddington_ratio"]) for r in th])
    d["lx_fcol"] = np.array([float(r["f_col"]) for r in th])
    obsid = "20188-02-21-00"
    date = next(r["date"] for r in rows if r["obsid"] == obsid)
    spec = co.load_pca(obsid, date)
    model = co.DiskPowerLawModel(spec.e_lo, spec.e_hi, co.LMCX3["mass"],
                                 co.LMCX3["inclination"], co.LMCX3["distance"])
    free = co.fit_pca(spec, model)
    fixed = co.fit_pca(spec, model, f_col=1.7)
    bb = co.fit_pca(spec, model, f_col=1.0)
    u = spec.use
    ew = (spec.ch_hi - spec.ch_lo)[u]
    sig = np.hypot(spec.rate_err, co.PCA_SYSTEMATIC * np.abs(spec.rate))[u]
    d["lx_E"] = 0.5 * (spec.ch_lo + spec.ch_hi)[u]
    d["lx_rate"], d["lx_err"] = spec.rate[u] / ew, sig / ew
    d["lx_free"] = free["_counts"][u] / ew
    d["lx_fixed"] = fixed["_counts"][u] / ew
    d["lx_bb"] = bb["_counts"][u] / ew
    facts.update(lx_obsid=obsid, lx_date=date[:10], lx_fcol=free["f_col"],
                 lx_chi_free=f"{free['chi2']:.0f}/{free['dof']}",
                 lx_chi_fixed=f"{fixed['chi2']:.0f}/{fixed['dof']}",
                 lx_chi_bb=f"{bb['chi2']:.0f}/{bb['dof']}",
                 lx_fcol_median=float(np.median(d["lx_fcol"])), lx_n_thermal=len(th))

    np.savez_compressed(OUT / "video_data.npz", **d)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main()
