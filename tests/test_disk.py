"""Physical and numerical checks of the thin-disk model."""

import numpy as np
import pytest

from accretion_disk import ThinDisk, constants as C, eddington_luminosity, planck_nu


@pytest.fixture(params=[10.0, 1.0e8])
def disk(request):
    return ThinDisk.from_eddington(request.param, 0.1)


def test_planck_integrates_to_stefan_boltzmann():
    T = 1.0e4
    nu = np.geomspace(1e10, 1e17, 20000)
    total = np.pi * np.trapezoid(planck_nu(nu, T), nu)
    assert total == pytest.approx(C.sigma_SB * T ** 4, rel=1e-4)


def test_planck_is_safe_in_wien_tail_and_for_zero_temperature():
    assert planck_nu(1e20, 10.0) == 0.0
    assert planck_nu(1e15, 0.0) == 0.0
    assert np.all(np.isfinite(planck_nu(np.geomspace(1, 1e25, 50), 1e5)))


def test_temperature_profile_shape(disk):
    # zero at the torque-free inner edge, maximum at 49/36 r_in, T ~ r^-3/4 far out
    assert disk.temperature(disk.r_in) == 0.0
    r = np.linspace(1.01, 3.0, 4001) * disk.r_in
    assert r[np.argmax(disk.temperature(r))] == pytest.approx(disk.r_T_max, rel=1e-3)
    assert disk.T_max == pytest.approx(0.488 * disk.T_star, rel=1e-3)
    r1, r2 = 1e3 * disk.r_in, 1e4 * disk.r_in
    slope = np.log(disk.temperature(r2) / disk.temperature(r1)) / np.log(r2 / r1)
    assert slope == pytest.approx(-0.75, abs=0.01)


def test_ring_sum_matches_analytic_luminosity(disk):
    assert disk.bolometric_luminosity() == pytest.approx(disk.analytic_luminosity(), rel=1e-4)


def test_spectrum_integrates_to_bolometric_luminosity(disk):
    nu = disk.frequency_grid(n=4000, low=1e-7)
    L = np.trapezoid(disk.luminosity_nu(nu), nu)
    assert L == pytest.approx(disk.bolometric_luminosity(), rel=1e-3)


def test_infinite_disk_efficiency_and_eddington_ratio():
    d = ThinDisk.from_eddington(10.0, 0.3)
    assert d.efficiency == pytest.approx(1.0 / 12.0)
    assert d.eddington_ratio == pytest.approx(0.3)
    L_inf = d.analytic_luminosity(infinite=True)
    assert L_inf == pytest.approx(C.G * d.mass * d.mdot / (2 * d.r_in))
    assert L_inf == pytest.approx(0.3 * eddington_luminosity(d.mass))


def test_flux_inclination_and_distance_scaling(disk):
    nu = disk.frequency_grid(50)
    D = 10 * C.kpc
    F0 = disk.flux_nu(nu, D, 0.0)
    np.testing.assert_allclose(disk.flux_nu(nu, D, 60.0), 0.5 * F0, rtol=1e-12)
    np.testing.assert_allclose(disk.flux_nu(nu, 2 * D, 0.0), 0.25 * F0, rtol=1e-12)
    # face-on flux corresponds to L_nu / (2 pi D^2)
    np.testing.assert_allclose(F0, disk.luminosity_nu(nu) / (2 * np.pi * D ** 2), rtol=1e-12)


def test_intermediate_spectral_slope_is_one_third():
    # need a wide disk so the nu^(1/3) power law is well developed
    d = ThinDisk.from_eddington(10.0, 0.1, r_out=1e8 * C.G * 10 * C.M_sun / C.c ** 2,
                                n_rings=4000)
    nu0 = C.k_B * d.T_max / C.h
    nu = np.array([3e-4, 3e-3]) * nu0
    L = d.luminosity_nu(nu)
    slope = np.log(L[1] / L[0]) / np.log(nu[1] / nu[0])
    assert slope == pytest.approx(1.0 / 3.0, abs=0.02)


def test_rayleigh_jeans_slope_below_outer_edge(disk):
    T_out = disk.temperature(disk.r_out)
    nu = np.array([1e-4, 1e-3]) * C.k_B * T_out / C.h
    L = disk.luminosity_nu(nu)
    assert np.log(L[1] / L[0]) / np.log(10.0) == pytest.approx(2.0, abs=0.02)


def test_mass_and_accretion_rate_scalings():
    # T_max ~ (Mdot / M^2)^(1/4): at fixed Eddington ratio T ~ M^-1/4
    a = ThinDisk.from_eddington(10.0, 0.1)
    b = ThinDisk.from_eddington(1.0e8, 0.1)
    assert b.T_max / a.T_max == pytest.approx((1e8 / 10.0) ** -0.25, rel=1e-10)
    assert b.peak_frequency() / a.peak_frequency() == pytest.approx(1e7 ** -0.25, rel=1e-3)
    # at fixed mass, L ~ Mdot and T ~ Mdot^(1/4)
    c = ThinDisk.from_eddington(10.0, 1.0)
    assert c.bolometric_luminosity() / a.bolometric_luminosity() == pytest.approx(10.0)
    assert c.T_max / a.T_max == pytest.approx(10 ** 0.25)


def test_peak_location_in_units_of_T_max(disk):
    x = C.h * disk.peak_frequency() / (C.k_B * disk.T_max)
    assert 2.0 < x < 3.0


def test_invalid_input():
    with pytest.raises(ValueError):
        ThinDisk(mass=-1.0, mdot=1.0)
    with pytest.raises(ValueError):
        ThinDisk(mass=C.M_sun, mdot=1e18, r_in=1e8, r_out=1e7)
    d = ThinDisk.from_eddington(10.0, 0.1)
    with pytest.raises(ValueError):
        d.peak_frequency(kind="bogus")


# ---------------------------------------------------------------------------
# Cross-checks against the published agnpy implementation (Nigro et al. 2022)
# ---------------------------------------------------------------------------

def test_temperature_comes_from_agnpy_ssdisk(disk):
    import astropy.units as u
    r = np.geomspace(1.01, 1e4, 50) * disk.r_in
    T_agnpy = disk.ssdisk.T(r * u.cm).to_value(u.K)
    np.testing.assert_allclose(disk.temperature(r), T_agnpy, rtol=1e-12)
    # ... and agrees with the closed-form Shakura-Sunyaev profile
    x = disk.r_in / r
    np.testing.assert_allclose(T_agnpy, disk.T_star * (x ** 3 * (1 - np.sqrt(x))) ** 0.25, rtol=1e-10)
    assert disk.ssdisk.m_dot.to_value("g s-1") == pytest.approx(disk.mdot, rel=1e-12)
    assert disk.temperature(0.5 * disk.r_in) == 0.0


@pytest.mark.parametrize("mass", [10.0, 1.0e8])
def test_ring_sum_matches_agnpy_sed_for_compact_disk(mass):
    # agnpy integrates over 100 linear radii, adequate only when r_out ~ r_in
    r_g = C.G * mass * C.M_sun / C.c ** 2
    d = ThinDisk.from_eddington(mass, 0.1, r_out=50 * r_g)
    nu = d.frequency_grid(300, low=1e-3, high=10)
    D, inc = 10 * C.kpc, 30.0
    ours = nu * d.flux_nu(nu, D, inc)
    theirs = d.agnpy_nuFnu(nu, D, inc)
    sel = ours > 1e-3 * ours.max()
    np.testing.assert_allclose(theirs[sel], ours[sel], rtol=3e-3)
