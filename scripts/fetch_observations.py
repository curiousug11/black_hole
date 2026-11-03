#!/usr/bin/env python3
"""Download the observational data used in scripts/compare_observations.py.

Everything is written to ``data/observations/``; see the README there for the
provenance of each dataset.  Run from the repository root:

    python scripts/fetch_observations.py

Datasets
--------
1. SDSS median composite quasar spectrum (Vanden Berk et al. 2001, AJ 122, 549),
   Table 1 of the paper, taken from the arXiv source of astro-ph/0105231.
2. SDSS DR7 quasar properties catalogue (Shen et al. 2011, ApJS 194, 45),
   version dr7_bh_Nov19_2013; a subset of columns is saved as a compressed CSV.
3. RXTE/PCA standard-product spectra of the black hole binary LMC X-3
   (HEASARC archive): source spectrum, background spectrum and response
   matrix for every 1996-1999 pointing with at least 3 ks of exposure.
"""

from __future__ import annotations

import gzip
import io
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "observations"

VANDENBERK_SRC = "https://arxiv.org/src/astro-ph/0105231"
SHEN_URL = "http://quasar.astro.illinois.edu/BH_mass/data/catalogs/dr7_bh_Nov19_2013.fits.gz"
XAMIN = "https://heasarc.gsfc.nasa.gov/xamin/query"
RXTE_ARCHIVE = "https://heasarc.gsfc.nasa.gov/FTP/xte/data/archive"

SHEN_COLUMNS = [
    "SDSS_NAME", "RA", "DEC", "REDSHIFT", "BAL_FLAG",
    "LOGLBOL", "LOGL5100", "LOGL3000", "LOGL1350",
    "LOGBH", "LOGBH_ERR", "LOGEDD_RATIO",
    "ALPHA_HB", "ALPHA_HB_ERR", "LINE_MED_SN_HB",
    "ALPHA_MGII", "ALPHA_MGII_ERR", "LINE_MED_SN_MGII",
    "ALPHA_CIV", "ALPHA_CIV_ERR", "LINE_MED_SN_CIV",
]


def download(url, dest, retries=4):
    """Download ``url`` to ``dest`` with simple exponential-backoff retries."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh)
            return dest
        except Exception as exc:  # noqa: BLE001 - report and retry any network error
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry {attempt + 1} for {url} in {wait}s ({exc})")
            time.sleep(wait)


def fetch_vandenberk():
    dest = OUT / "vandenberk2001_composite.txt"
    if dest.exists():
        print(f"exists: {dest.relative_to(ROOT)}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tarball = download(VANDENBERK_SRC, Path(tmp) / "src.tar.gz")
        with tarfile.open(tarball) as tar:
            member = tar.getmember("VandenBerk.tab1.dat")
            raw = tar.extractfile(member).read().decode("ascii")
    rows = []
    for line in raw.splitlines():   # skip the byte-by-byte format header
        parts = line.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    data = np.array([r for r in rows if len(r) == 3])
    header = ("SDSS median composite quasar spectrum (Vanden Berk et al. 2001, AJ 122, 549),\n"
              "Table 1, from the arXiv source of astro-ph/0105231 (file VandenBerk.tab1.dat).\n"
              "Columns: rest wavelength [Angstrom], f_lambda [arbitrary units],\n"
              "68% semi-interquartile range of f_lambda [arbitrary units]")
    np.savetxt(dest, data, fmt=["%.1f", "%.4f", "%.4f"], header=header)
    print(f"wrote {dest.relative_to(ROOT)} ({len(data)} rows)")


def fetch_shen():
    dest = OUT / "shen2011_dr7_quasars.csv.gz"
    if dest.exists():
        print(f"exists: {dest.relative_to(ROOT)}")
        return
    from astropy.io import fits
    with tempfile.TemporaryDirectory() as tmp:
        path = download(SHEN_URL, Path(tmp) / "dr7_bh.fits.gz")
        with fits.open(path) as hdul:
            d = hdul[1].data
            keep = d["LOGBH"] > 0
            cols = {c: d[c][keep] for c in SHEN_COLUMNS}
    n = int(keep.sum())
    with gzip.open(dest, "wt") as fh:
        fh.write("# SDSS DR7 quasar properties, Shen et al. 2011 (ApJS 194, 45), catalogue file\n")
        fh.write(f"# dr7_bh_Nov19_2013.fits from {SHEN_URL}\n")
        fh.write("# Subset of columns; only rows with a fiducial virial mass (LOGBH > 0).\n")
        fh.write("# ALPHA_* are local continuum slopes alpha_lambda (f_lambda ~ lambda^alpha).\n")
        fh.write(",".join(SHEN_COLUMNS) + "\n")
        for i in range(n):
            vals = []
            for c in SHEN_COLUMNS:
                v = cols[c][i]
                vals.append(v.strip() if isinstance(v, str) else f"{v:.6g}")
            fh.write(",".join(vals) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({n} quasars)")


def rxte_observation_list(min_exposure=3000.0, years=("1996", "1997", "1998", "1999")):
    query = urllib.parse.urlencode({
        "table": "xtemaster", "position": "LMC X-3", "radius": "5", "format": "text",
        "fields": "obsid,time,exposure,status", "resultmax": "5000",
    })
    with urllib.request.urlopen(f"{XAMIN}?{query}", timeout=300) as r:
        lines = r.read().decode().splitlines()[1:]
    obs = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or parts[3] != "archived":
            continue
        try:
            exposure = float(parts[2])
        except ValueError:
            continue
        if exposure >= min_exposure and parts[1][:4] in years:
            obs.append((parts[0], parts[1], exposure))
    return sorted(obs, key=lambda o: o[1])


def fetch_rxte():
    outdir = OUT / "rxte_lmcx3"
    outdir.mkdir(parents=True, exist_ok=True)
    obs = rxte_observation_list()
    print(f"RXTE: {len(obs)} LMC X-3 pointings (1996-1999, >= 3 ks)")
    index = ["# obsid,start_time_utc,exposure_s,status"]
    for obsid, start, exposure in obs:
        tag = obsid.replace("-", "")
        base = f"{RXTE_ARCHIVE}/AO{obsid[0]}/P{obsid.split('-')[0]}/{obsid}/stdprod"
        files = [f"xp{tag}_s2.pha.gz", f"xp{tag}_b2.pha.gz", f"xp{tag}.rsp.gz"]
        status = "ok"
        for f in files:
            dest = outdir / f
            if dest.exists():
                continue
            try:
                download(f"{base}/{f}", dest, retries=2)
            except Exception as exc:  # noqa: BLE001
                status = f"missing ({f})"
                print(f"  {obsid}: {f} unavailable ({exc})")
                dest.unlink(missing_ok=True)
                break
        index.append(f"{obsid},{start},{exposure:.0f},{status}")
    (outdir / "observations.csv").write_text("\n".join(index) + "\n")
    n_ok = sum(1 for line in index[1:] if line.endswith(",ok"))
    print(f"wrote {outdir.relative_to(ROOT)} ({n_ok} complete observations)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    steps = {"vandenberk": fetch_vandenberk, "shen": fetch_shen, "rxte": fetch_rxte}
    for name in (sys.argv[1:] or steps):
        steps[name]()


if __name__ == "__main__":
    main()
