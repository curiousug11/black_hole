"""Geometrically thin, optically thick (Shakura-Sunyaev) accretion disk model.

The disk is divided into concentric annuli ("rings").  Each ring is assumed
to radiate as a blackbody at the local effective temperature given by the
steady-state, Newtonian thin-disk solution with a zero-torque inner boundary:

    sigma T(r)^4 = (3 G M Mdot) / (8 pi r^3) * [1 - sqrt(r_in / r)]

The disk spectrum is obtained by summing the Planck emission of every ring,
weighted by its surface area:

    L_nu = sum_k  2 * pi * B_nu(T_k) * A_k          (both faces, A_k one face)
    F_nu = cos(i) / D^2 * sum_k B_nu(T_k) * A_k     (observer at D, inclination i)

All quantities are in CGS units unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as const

__all__ = [
    "planck_nu",
    "eddington_luminosity",
    "eddington_accretion_rate",
    "ThinDisk",
]


# ---------------------------------------------------------------------------
# Elementary physics
# ---------------------------------------------------------------------------

def planck_nu(nu, T):
    """Planck specific intensity B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].

    ``nu`` and ``T`` broadcast against each other.  Overflow in the Wien tail
    is handled by returning exactly zero where h nu / k T is very large, and
    T <= 0 also returns zero.
    """
    nu = np.asarray(nu, dtype=float)
    T = np.asarray(T, dtype=float)
    nu_b, T_b = np.broadcast_arrays(nu, T)
    out = np.zeros(nu_b.shape)
    good = T_b > 0
    x = np.zeros(nu_b.shape)
    x[good] = const.h * nu_b[good] / (const.k_B * T_b[good])
    good &= x < 700.0  # exp(700) ~ 1e304; beyond this B_nu is negligible
    out[good] = (2.0 * const.h * nu_b[good] ** 3 / const.c ** 2) / np.expm1(x[good])
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
    """A steady, Newtonian, geometrically thin and optically thick disk.

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
    torque_free_inner_edge : bool
        If True (default) include the [1 - sqrt(r_in/r)] zero-torque factor;
        if False use the simple T ~ r^-3/4 power law everywhere.
    """

    mass: float
    mdot: float
    r_in: float | None = None
    r_out: float | None = None
    n_rings: int = 2000
    torque_free_inner_edge: bool = True
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
        """Radius of maximum temperature, (49/36) r_in (torque-free case)."""
        return (49.0 / 36.0) * self.r_in if self.torque_free_inner_edge else self.r_in

    @property
    def T_max(self):
        """Maximum disk temperature [K]."""
        return float(self.temperature(self.r_T_max))

    # -- radial structure --------------------------------------------------
    def temperature(self, r):
        """Effective temperature T(r) [K]; zero inside r_in."""
        r = np.asarray(r, dtype=float)
        x = self.r_in / r
        f = 1.0 - np.sqrt(np.clip(x, 0.0, 1.0)) if self.torque_free_inner_edge else 1.0
        T = self.T_star * (x ** 3 * f) ** 0.25
        return np.where(r >= self.r_in, T, 0.0)

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

    @property
    def ring_temperatures(self):
        return self.temperature(self.ring_radii)

    # -- emission ----------------------------------------------------------
    def _ring_sum(self, nu):
        """sum_k B_nu(T_k) A_k  [erg s^-1 Hz^-1 sr^-1] for each frequency."""
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        B = planck_nu(nu[:, None], self.ring_temperatures[None, :])
        return B @ self.ring_areas

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

    def bolometric_luminosity(self):
        """Total luminosity [erg/s] from summing sigma T^4 over all rings."""
        return 2.0 * const.sigma_SB * np.sum(self.ring_temperatures ** 4 * self.ring_areas)

    def analytic_luminosity(self, infinite=False):
        """Exact Newtonian luminosity of the disk between r_in and r_out.

        With X = r_out / r_in and the torque-free boundary:
            L = (3 G M Mdot / 2 r_in) [ (1 - 1/X) - (2/3)(1 - X^-3/2) ]
        which tends to G M Mdot / (2 r_in) as X -> infinity.
        """
        pref = 3.0 * const.G * self.mass * self.mdot / (2.0 * self.r_in)
        inv_X = 0.0 if infinite else self.r_in / self.r_out
        if self.torque_free_inner_edge:
            return pref * ((1.0 - inv_X) - (2.0 / 3.0) * (1.0 - inv_X ** 1.5))
        return pref * (1.0 - inv_X)

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
