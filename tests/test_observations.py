"""Checks of the observational-comparison code (skipped if the data are absent)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "observations"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not (DATA / "rxte_lmcx3" / "observations.csv").exists(),
                                reason="run scripts/fetch_observations.py first")


@pytest.fixture(scope="module")
def co():
    import compare_observations
    return compare_observations


def test_composite_slope_matches_published_value(co):
    # Vanden Berk et al. (2001) quote alpha_nu = -0.44 for the composite continuum
    lam, flam, _ = np.loadtxt(DATA / "vandenberk2001_composite.txt", unpack=True)
    med = [np.median(flam[(lam >= lo) & (lam <= hi)]) for lo, hi in co.VDB_WINDOWS]
    l1, l2 = [np.mean(w) for w in co.VDB_WINDOWS]
    alpha_nu = -(np.log(med[1] / med[0]) / np.log(l2 / l1) + 2)
    assert alpha_nu == pytest.approx(-0.44, abs=0.05)


def test_pca_response_is_summed_over_five_pcus(co):
    obsid, date = (DATA / "rxte_lmcx3" / "observations.csv").read_text().splitlines()[1].split(",")[:2]
    spec = co.load_pca(obsid, date)
    area = spec.resp.sum(axis=1)
    e = 0.5 * (spec.e_lo + spec.e_hi)
    assert 5500 < area[np.argmin(abs(e - 8.0))] < 7500     # ~1300 cm^2 per PCU
    assert spec.use.sum() > 30 and np.all(spec.ch_lo[spec.use] >= co.PCA_BAND[0])


def test_scaled_disk_template_matches_direct_calculation(co):
    from accretion_disk import ThinDisk, constants as C, planck_nu
    e_lo = np.geomspace(2.0, 30.0, 41)[:-1]
    e_hi = np.geomspace(2.0, 30.0, 41)[1:]
    m = co.DiskPowerLawModel(e_lo, e_hi, 7.0, 60.0, 50.0, n_sub=1)
    mdot, f = 2.3 * m.mdot0, 1.7
    fast = m.disk_photons(mdot, f)
    d = ThinDisk(mass=m.mass, mdot=mdot, n_rings=1500, r_out=1e4 * C.G * m.mass / C.c ** 2)
    nu = m.E[:, 0] * C.keV / C.h
    F_nu = m.cos_i / m.D ** 2 * f ** -4 * (planck_nu(nu[:, None], f * d.ring_temperatures[None, :])
                                           @ d.ring_areas)
    direct = F_nu * C.keV / C.h / (m.E[:, 0] * C.keV) * m.dE[:, 0]
    np.testing.assert_allclose(fast, direct, rtol=2e-3)


def test_power_law_bin_integral(co):
    e_lo, e_hi = np.array([3.0, 10.0]), np.array([4.0, 20.0])
    m = co.DiskPowerLawModel(e_lo, e_hi, 7.0, 60.0, 50.0)
    np.testing.assert_allclose(m.powerlaw_photons(2.0, 2.0), 2.0 * (1 / e_lo - 1 / e_hi))
