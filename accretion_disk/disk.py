"""Geometrically thin, optically thick (Shakura & Sunyaev 1973) accretion disk.

The physical model is the standard steady, Newtonian thin disk with a
zero-torque inner boundary, as implemented in the published ``agnpy`` package
(Nigro et al. 2022, A&A 660, A18; class :class:`agnpy.targets.SSDisk`, which
follows Dermer & Menon 2009):

    sigma T(r)^4 = (3 G M Mdot) / (8 pi r^3) * [1 - sqrt(r_in / r)]

The building blocks come from established packages:

* local temperature T(r)      -> ``agnpy.targets.SSDisk.T``
* Planck function B_nu(T)     -> ``astropy.modeling.physical_models.BlackBody``
* physical constants          -> ``astropy.constants`` (via ``constants.py``)

What this module adds is the numerical integration over the disk.  The disk is
divided into logarithmically spaced concentric rings, each radiating as a
blackbody at its local temperature, and their emission is summed:

    L_nu = sum_k  2 * pi * B_nu(T_k) * A_k          (both faces, A_k one face)
    F_nu = cos(i) / D^2 * sum_k B_nu(T_k) * A_k     (observer at D, inclination i)

The flux expression is the same as agnpy's ``SSDisk.evaluate_multi_T_bb_sed``
(with mu_s = cos i), but agnpy integrates over 100 *linearly* spaced radii,
which cannot resolve the hot inner disk when r_out >> r_in.  The log-spaced
ring sum here converges for any disk size; ``agnpy_nuFnu`` exposes agnpy's own
calculation so the two can be compared (see tests/test_disk.py).

All quantities are in CGS units unless stated otherwise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import astropy.units as u
from astropy.modeling.physical_models import BlackBody
from agnpy.targets import SSDisk

from . import constants as const

__all__ = [
    "planck_nu",
    "eddington_luminosity",
    "eddington_accretion_rate",
    "ThinDisk",
]

_BLACKBODY = BlackBody()
_B_UNIT = u.Unit("erg s-1 cm-2 Hz-1 sr-1")


# ---------------------------------------------------------------------------
# Elementary physics
# ---------------------------------------------------------------------------

def planck_nu(nu, T):
    """Planck specific intensity B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].

    Evaluated with astropy's ``BlackBody`` model.  ``nu`` [Hz] and ``T`` [K]
    broadcast against each other.  Elements with T <= 0, or deep in the Wien
    tail (h nu / k T > 600, where B_nu < 1e-255 of its peak), are returned as
    exactly zero instead of being passed to astropy.
    """
    nu = np.asarray(nu, dtype=float)
    T = np.asarray(T, dtype=float)
    nu_b, T_b = np.broadcast_arrays(nu, T)
    out = np.zeros(nu_b.shape)
    good = T_b > 0
    x = np.full(nu_b.shape, np.inf)
    x[good] = const.h * nu_b[good] / (const.k_B * T_b[good])
    good &= x < 600.0  # astropy overflows c^2 (e^x - 1) in SI near x ~ 700
    if np.any(good):
        out[good] = _BLACKBODY.evaluate(nu_b[good] * u.Hz, T_b[good] * u.K,
                                        scale=1).to_value(_B_UNIT)
    return out


def eddington_luminosity(mass):
    """Eddington luminosity [erg/s] for fully ionised hydrogen; ``mass`` in g."""
    return 4.0 * np.pi * const.G * mass * const.m_p * const.c / const.sigma_T


def eddington_accretion_rate(mass, efficiency):
    """Accretion rate [g/s] that yields L = L_Edd for radiative ``efficiency``."""
    return eddington_luminosity(mass) / (efficiency * const.c ** 2)


# ---------------------------------------------------------------------------
# Disk model
# ---------------------------------------------------------------------------

@dataclass
class ThinDisk:
    """A Shakura-Sunyaev disk (agnpy ``SSDisk``) integrated ring by ring.

    Parameters
    ----------
    mass : float
        Black hole mass [g].
    mdot : float
        Mass accretion rate [g/s].
    r_in : float, optional
        Inner disk radius [cm].  Defaults to the Schwarzschild ISCO, 6 GM/c^2.
    r_out : float, optional
        Outer disk radius [cm].  Defaults to 1e5 GM/c^2.
    n_rings : int
        Number of logarithmically spaced annuli used in the integration.
    """

    mass: float
    mdot: float
    r_in: float | None = None
    r_out: float | None = None
    n_rings: int = 2000
    _edges: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.mass <= 0 or self.mdot <= 0:
            raise ValueError("mass and mdot must be positive")
        if self.r_in is None:
            self.r_in = 6.0 * self.r_g
        if self.r_out is None:
            self.r_out = 1.0e5 * self.r_g
        if not self.r_out > self.r_in:
            raise ValueError("r_out must exceed r_in")
        if self.n_rings < 1:
            raise ValueError("n_rings must be >= 1")
        self._edges = np.geomspace(self.r_in, self.r_out, self.n_rings + 1)

    # -- convenience constructor ------------------------------------------
    @classmethod
    def from_eddington(cls, mass_msun, eddington_ratio, **kwargs):
        """Build a disk from a mass in solar masses and an Eddington ratio.

        The Eddington ratio is defined self-consistently with this model,
        i.e. ``L_disk / L_Edd`` where ``L_disk = G M Mdot / (2 r_in)`` is the
        luminosity of an infinite Newtonian disk (efficiency 1/12 for
        r_in = 6 GM/c^2).
        """
        mass = mass_msun * const.M_sun
        r_g = const.G * mass / const.c ** 2
        r_in = kwargs.get("r_in") or 6.0 * r_g
        efficiency = r_g / (2.0 * r_in)
        mdot = eddington_ratio * eddington_accretion_rate(mass, efficiency)
        return cls(mass=mass, mdot=mdot, **kwargs)

    # -- the published model ------------------------------------------------
    @cached_property
    def ssdisk(self):
        """The equivalent :class:`agnpy.targets.SSDisk` instance.

        agnpy parametrises the disk by (M, L_disk, eta); choosing
        L_disk = eta Mdot c^2 with eta = r_g / (2 r_in) makes agnpy's accretion
        rate equal to ``self.mdot``.
        """
        return SSDisk(
            M_BH=self.mass * u.g,
            L_disk=self.efficiency * self.mdot * const.c ** 2 * u.Unit("erg s-1"),
            eta=self.efficiency,
            R_in=self.r_in * u.cm,
            R_out=self.r_out * u.cm,
        )

    # -- characteristic scales --------------------------------------------
    @property
    def mass_msun(self):
        return self.mass / const.M_sun

    @property
    def r_g(self):
        """Gravitational radius GM/c^2 [cm]."""
        return const.G * self.mass / const.c ** 2

    @property
    def efficiency(self):
        """Radiative efficiency of an infinite Newtonian disk, r_g / (2 r_in)."""
        return self.r_g / (2.0 * self.r_in)

    @property
    def eddington_ratio(self):
        return self.analytic_luminosity(infinite=True) / eddington_luminosity(self.mass)

    @property
    def T_star(self):
        """Temperature scale (3 G M Mdot / 8 pi sigma r_in^3)^(1/4) [K]."""
        return (3.0 * const.G * self.mass * self.mdot
                / (8.0 * np.pi * const.sigma_SB * self.r_in ** 3)) ** 0.25

    @property
    def r_T_max(self):
        """Radius of maximum temperature, (49/36) r_in."""
        return (49.0 / 36.0) * self.r_in

    @property
    def T_max(self):
        """Maximum disk temperature [K]."""
        return float(self.temperature(self.r_T_max))

    # -- radial structure --------------------------------------------------
    def temperature(self, r):
        """Effective temperature T(r) [K] from agnpy's ``SSDisk.T``; zero inside r_in."""
        r = np.asarray(r, dtype=float)
        T = np.zeros(r.shape)
        inside = r >= self.r_in
        if np.any(inside):
            T[inside] = self.ssdisk.T(r[inside] * u.cm).to_value(u.K)
        return T

    @property
    def ring_edges(self):
        return self._edges

    @property
    def ring_radii(self):
        """Representative (geometric-mean) radius of each ring [cm]."""
        return np.sqrt(self._edges[:-1] * self._edges[1:])

    @property
    def ring_areas(self):
        """Area of one face of each ring [cm^2]."""
        return np.pi * (self._edges[1:] ** 2 - self._edges[:-1] ** 2)

    @cached_property
    def ring_temperatures(self):
        return self.temperature(self.ring_radii)

    # -- emission ----------------------------------------------------------
    def _ring_sum(self, nu):
        """sum_k B_nu(T_k) A_k  [erg s^-1 Hz^-1 sr^-1] for each frequency."""
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        T, A = self.ring_temperatures, self.ring_areas
        out = np.empty(nu.size)
        chunk = max(1, int(2_000_000 // T.size))   # bound memory use
        for s in range(0, nu.size, chunk):
            out[s:s + chunk] = planck_nu(nu[s:s + chunk, None], T[None, :]) @ A
        return out

    def luminosity_nu(self, nu):
        """Specific luminosity L_nu [erg s^-1 Hz^-1] from both disk faces."""
        return 2.0 * np.pi * self._ring_sum(nu)

    def flux_nu(self, nu, distance, inclination_deg):
        """Observed flux density F_nu [erg s^-1 cm^-2 Hz^-1].

        A flat, Lambertian disk at ``distance`` [cm] viewed at inclination
        ``inclination_deg`` (0 = face-on) subtends projected area A cos(i).
        No limb darkening, relativistic beaming, light bending or absorption.
        """
        cos_i = np.cos(np.radians(inclination_deg))
        if cos_i < 0:
            raise ValueError("inclination must be between 0 and 90 degrees")
        return cos_i / distance ** 2 * self._ring_sum(nu)

    def agnpy_nuFnu(self, nu, distance, inclination_deg):
        """nu F_nu [erg s^-1 cm^-2] computed by agnpy's own integration.

        Calls ``SSDisk.evaluate_multi_T_bb_sed`` (z = 0), which integrates
        over 100 linearly spaced radii.  Accurate only for compact disks;
        provided for cross-checking the ring sum.
        """
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # np.trapz deprecation, Wien-tail underflow
            sed = SSDisk.evaluate_multi_T_bb_sed(
                nu * u.Hz, 0.0, self.mass * u.g, self.mdot * u.Unit("g s-1"),
                self.r_in * u.cm, self.r_out * u.cm, distance * u.cm,
                mu_s=np.cos(np.radians(inclination_deg)),
            )
        return sed.to_value(u.Unit("erg cm-2 s-1"))

    def bolometric_luminosity(self):
        """Total luminosity [erg/s] from summing sigma T^4 over all rings."""
        return 2.0 * const.sigma_SB * np.sum(self.ring_temperatures ** 4 * self.ring_areas)

    def analytic_luminosity(self, infinite=False):
        """Exact Newtonian luminosity of the disk between r_in and r_out.

        With X = r_out / r_in:
            L = (3 G M Mdot / 2 r_in) [ (1 - 1/X) - (2/3)(1 - X^-3/2) ]
        which tends to G M Mdot / (2 r_in) as X -> infinity.
        """
        pref = 3.0 * const.G * self.mass * self.mdot / (2.0 * self.r_in)
        inv_X = 0.0 if infinite else self.r_in / self.r_out
        return pref * ((1.0 - inv_X) - (2.0 / 3.0) * (1.0 - inv_X ** 1.5))

    def frequency_grid(self, n=2000, low=1e-5, high=40.0):
        """Log-spaced frequencies spanning the disk emission [Hz].

        Limits are expressed as multiples of k T_max / h so the same relative
        coverage is obtained for any mass and accretion rate.
        """
        nu0 = const.k_B * self.T_max / const.h
        return np.geomspace(low * nu0, high * nu0, n)

    def peak_frequency(self, kind="nuLnu", nu=None):
        """Frequency [Hz] at which ``nu L_nu`` (default) or ``L_nu`` peaks.

        The maximum on a dense log grid is refined with a parabola in
        log-log space for sub-grid accuracy.
        """
        if kind not in ("nuLnu", "Lnu"):
            raise ValueError("kind must be 'nuLnu' or 'Lnu'")
        if nu is None:
            nu = self.frequency_grid(n=4000)
        L = self.luminosity_nu(nu)
        y = nu * L if kind == "nuLnu" else L
        i = int(np.argmax(y))
        if 0 < i < len(nu) - 1:
            lx = np.log(nu[i - 1:i + 2])
            ly = np.log(y[i - 1:i + 2])
            a, b, _ = np.polyfit(lx, ly, 2)
            if a < 0:
                return float(np.exp(-b / (2.0 * a)))
        return float(nu[i])
