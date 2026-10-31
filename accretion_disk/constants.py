"""Physical constants in CGS units, taken from :mod:`astropy.constants`.

Astropy provides CODATA 2018 physical constants and IAU 2015 astronomical
constants; they are converted once to plain floats (CGS) here so the rest of
the code can do fast NumPy arithmetic.
"""

import astropy.constants as _ac
import astropy.units as _u

G = _ac.G.cgs.value                 # gravitational constant [cm^3 g^-1 s^-2]
c = _ac.c.cgs.value                 # speed of light [cm s^-1]
h = _ac.h.cgs.value                 # Planck constant [erg s]
k_B = _ac.k_B.cgs.value             # Boltzmann constant [erg K^-1]
sigma_SB = _ac.sigma_sb.cgs.value   # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
m_p = _ac.m_p.cgs.value             # proton mass [g]
sigma_T = _ac.sigma_T.cgs.value     # Thomson cross-section [cm^2]

M_sun = _ac.M_sun.cgs.value         # solar mass [g] (IAU 2015 nominal)
yr = (1 * _u.yr).to_value(_u.s)     # Julian year [s]
pc = _ac.pc.cgs.value               # parsec [cm]
kpc = 1.0e3 * pc
Mpc = 1.0e6 * pc

eV = (1 * _u.eV).to_value(_u.erg)   # electron-volt [erg]
keV = 1.0e3 * eV
angstrom = (1 * _u.AA).to_value(_u.cm)
