# Thermal Emission from Thin Accretion Disks

How do black hole mass and accretion rate shape the thermal spectrum of an
accretion disk? This project models a geometrically thin, optically thick
disk as a set of concentric rings. Each ring radiates as a blackbody at its
local temperature, and the spectrum is the sum of the emission from all rings.
The model is then used to compare stellar-mass and supermassive black holes.

```
accretion_disk/      Python package: the disk model
  constants.py       CGS physical constants and unit conversions
  disk.py            ThinDisk class, Planck function, Eddington helpers
scripts/
  run_analysis.py    Parameter study: writes every figure and table
notebooks/
  accretion_disk_walkthrough.ipynb   Step-by-step walkthrough of the analysis
tests/test_disk.py   Physics and numerical checks (pytest)
figures/             Generated figures (PNG)
results/             Generated summary tables (CSV and Markdown)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest              # 18 tests, about 3 s
python scripts/run_analysis.py  # regenerates figures/ and results/, about 1 min
jupyter lab notebooks/accretion_disk_walkthrough.ipynb   # guided walkthrough
```

The notebook is committed with its outputs, so it can be read on GitHub
without running anything. It covers the temperature profile, how individual
rings combine into the spectrum, checks of the numerical integration, the
mass and accretion-rate sweeps, observed flux versus inclination, and how to
infer the black hole mass from $L$ and $\nu_{peak}$. It ends with an
`explore()` helper for trying any combination of parameters.

Using the model directly:

```python
from accretion_disk import ThinDisk, constants as C

disk = ThinDisk.from_eddington(mass_msun=10, eddington_ratio=0.1)  # or ThinDisk(mass=..., mdot=...)
nu = disk.frequency_grid()                       # Hz
L_nu = disk.luminosity_nu(nu)                    # erg/s/Hz
F_nu = disk.flux_nu(nu, distance=8 * C.kpc, inclination_deg=60)  # erg/s/cm^2/Hz
print(disk.T_max, disk.peak_frequency(), disk.bolometric_luminosity())
```

## Physical model

**Temperature profile.** For a steady, Newtonian, thin disk with no torque at
the inner edge, the local energy balance is

$$\sigma T^4(r) = \frac{3GM\dot M}{8\pi r^3}\left[1-\sqrt{\frac{r_{in}}{r}}\right],$$

with $r_{in} = 6GM/c^2$, the Schwarzschild innermost stable circular orbit
(ISCO). By default $r_{out} = 10^5\,GM/c^2$. Far from the hole $T\propto r^{-3/4}$.
The temperature peaks at $r = (49/36)\,r_{in}$, where
$T_{max} = 0.488\,T_*$ and $T_* = (3GM\dot M / 8\pi\sigma r_{in}^3)^{1/4}$.

**Ring integration.** The disk is divided into $N$ (default 2000)
logarithmically spaced annuli. Ring $k$ has temperature $T_k$, evaluated at
the geometric-mean radius, and one-face area $A_k = \pi(r_{k+1}^2 - r_k^2)$.
Then

$$L_\nu = 2\pi\sum_k B_\nu(T_k)\,A_k \quad\text{(both faces)},\qquad
F_\nu = \frac{\cos i}{D^2}\sum_k B_\nu(T_k)\,A_k .$$

The flux formula treats the disk as a flat Lambertian emitter seen at
inclination $i$ and distance $D$, so its projected area is $A\cos i$.

**Accretion rate.** Rates are given as an Eddington ratio
$\lambda = L/L_{Edd}$, with $L_{Edd} = 4\pi GMm_pc/\sigma_T$. For consistency
with the model, $L = GM\dot M/2r_{in} = \dot M c^2/12$, so the radiative
efficiency is $\eta = 1/12$.

### Validation (tests/test_disk.py)

| Check | Result |
|---|---|
| $\pi\int B_\nu\,d\nu = \sigma T^4$ | agrees to < 1e-4 |
| Ring sum vs. exact $L(r_{in}, r_{out})$ | agrees to < 1e-4 (error scales as $N^{-2}$, Fig. 7) |
| $\int L_\nu\,d\nu$ vs. ring sum of $\sigma T^4$ | agrees to < 1e-3 |
| $T_{max}$ location and value | $r = 49/36\,r_{in}$, $T_{max} = 0.488\,T_*$ |
| Spectral slopes | $L_\nu\propto\nu^{2}$ (Rayleigh–Jeans) and $\nu^{1/3}$ (intermediate) |
| Scalings | $T_{max}\propto(\dot M/M^2)^{1/4}$; $L\propto\dot M$ |
| Observer geometry | $F_\nu\propto\cos i/D^2$; face-on $F_\nu = L_\nu/2\pi D^2$ |

## Results

Fiducial systems at $\lambda = 0.1$:

* **Stellar-mass BH:** $10\,M_\odot$ at 8 kpc, $i = 60°$, like a Galactic X-ray binary.
* **Supermassive BH:** $10^8\,M_\odot$ at 100 Mpc, $i = 30°$, like a Seyfert or quasar nucleus.

| System | M [M☉] | L/L_Edd | Ṁ [M☉/yr] | L_bol [erg/s] | T_max [K] | ν_peak [Hz] | E_peak [keV] | λ_peak [Å] | F_bol [erg/s/cm²] |
|---|---|---|---|---|---|---|---|---|---|
| Stellar | 10 | 0.01 | 2.7e-9 | 1.3e37 | 2.5e6 | 1.3e17 | 0.53 | 24 | 1.6e-9 |
| Stellar | 10 | 0.1 | 2.7e-8 | 1.3e38 | 4.4e6 | 2.3e17 | 0.94 | 13 | 1.6e-8 |
| Stellar | 10 | 1 | 2.7e-7 | 1.3e39 | 7.9e6 | 4.0e17 | 1.67 | 7.4 | 1.6e-7 |
| SMBH | 1e8 | 0.01 | 0.027 | 1.3e44 | 4.4e4 | 2.3e15 | 0.0094 | 1320 | 1.8e-10 |
| SMBH | 1e8 | 0.1 | 0.27 | 1.3e45 | 7.9e4 | 4.0e15 | 0.017 | 742 | 1.8e-9 |
| SMBH | 1e8 | 1 | 2.7 | 1.3e46 | 1.4e5 | 7.2e15 | 0.030 | 417 | 1.8e-8 |

The full-precision table is in `results/summary.csv`. $\nu_{peak}$ is the
peak of $\nu L_\nu$.

### Key findings

1. **Temperature structure (Fig. 1).** Both disks have the same shape in units
   of $GM/c^2$. Only the normalisation differs: at a fixed Eddington ratio,
   $\dot M\propto M$, so $T\propto M^{-1/4}$. The inner region of the
   $10\,M_\odot$ disk reaches $\sim4\times10^6$ K. The $10^8\,M_\odot$ disk
   reaches only $\sim8\times10^4$ K, even though it radiates $10^7$ times more
   power. Its emitting area is $\propto M^2$, so a lower temperature suffices.

2. **Characteristic frequency (Figs. 2, 3, 5a).** The same model puts the
   stellar-mass disk in the **soft X-ray** band ($\sim$1 keV) and the
   supermassive disk in the **far/extreme UV** ($\sim$700 Å). The second is
   the "big blue bump" seen in quasars. Their peak frequencies differ by
   exactly $(10^7)^{1/4} = 56$. The $\nu L_\nu$ peak always sits at
   $h\nu_{peak}\approx 2.47\,kT_{max}$, and the $L_\nu$ peak at
   $\approx 0.80\,kT_{max}$.

3. **Accretion rate (Figs. 4, 5b, 5c).** At fixed mass, raising $\dot M$
   multiplies the luminosity in proportion ($L = \dot M c^2/12$). It moves the
   peak only weakly, as $\dot M^{1/4}$: a factor of 100 in $\dot M$ shifts
   $\nu_{peak}$ by only about 3. Bolometric luminosity alone therefore measures
   $\dot M$ and says nothing about $M$. Combining $L$ with the peak
   temperature, $T_{max}^4\propto \dot M/M^2$, constrains both.

4. **Spectral shape.** Every spectrum shows three regimes. A Rayleigh–Jeans
   $\nu^2$ tail comes from the cool outer edge. In the middle, overlapping
   blackbodies give the characteristic $L_\nu\propto\nu^{1/3}$ (so
   $\nu L_\nu\propto\nu^{4/3}$). An exponential Wien cutoff is set by
   $T_{max}$. The low-frequency break depends on the assumed $r_{out}$:
   $T(r_{out})\approx 6000$ K for the stellar disk and $\approx 110$ K for the
   SMBH disk.

5. **Observed flux (Fig. 6).** The flux scales as $\cos i/D^2$, but the
   spectral shape is unchanged in this non-relativistic model. At
   $\lambda=0.1$ the X-ray binary gives $F_{bol}\approx1.6\times10^{-8}$
   erg s⁻¹ cm⁻², a bright X-ray source. The AGN at 100 Mpc gives
   $\approx1.8\times10^{-9}$ erg s⁻¹ cm⁻². An edge-on view ($i=85°$) reduces
   the flux about 10× relative to face-on. An observer who assumes isotropic
   emission would infer $L_{iso} = 2\cos i\,L$, which overestimates $L$ by up
   to 2× for face-on disks.

6. **Numerics (Fig. 7).** The log-spaced ring sum converges as $N^{-2}$. About
   100 rings give 0.2% accuracy in luminosity. The default 2000 rings give
   < 1e-5.

## Limitations

The model shows the main physics, but it leaves out several effects that
matter when comparing to real data:

* **Relativistic disk structure.** The Newtonian $T(r)$ with a Schwarzschild
  ISCO should be replaced by the Novikov–Thorne solution. Black hole spin
  moves the ISCO from $6\,GM/c^2$ (spin 0) down to $1\,GM/c^2$ (maximal
  prograde spin). That raises the efficiency from about 6% to as much as 42%,
  makes the inner disk hotter, and shifts $\nu_{peak}$ upward. Our $\eta=1/12$
  is only a Newtonian approximation.
* **Relativistic light propagation.** The model ignores Doppler boosting from
  orbital speeds of about $0.4c$ near the ISCO, gravitational redshift, and
  light bending. All of these depend on inclination and change the spectral
  shape as well as the normalisation. Here, inclination only scales the
  flux by $\cos i$.
* **Non-blackbody radiative transfer.** In hot inner disks, electron
  scattering dominates the opacity, which hardens the spectrum. This is
  usually described by a color-correction factor $f_{col}\approx1.7$ for
  X-ray binaries. Atomic edges and limb darkening also distort the local
  spectra, especially in AGN disks.
* **Additional emission components.** The model omits the hot Comptonizing
  corona (power-law X-rays), the reflection spectrum and Fe Kα line, jet
  synchrotron emission, irradiation of the outer disk (important in X-ray
  binaries), emission from the companion star or host galaxy, the AGN
  broad-line region, and dust torus re-emission.
* **Validity of the thin-disk assumption.** Near and above $L_{Edd}$ the disk
  thickens (slim disk, advection). At very low $\dot M$ it may become a
  radiatively inefficient flow. AGN disks also become self-gravitating beyond
  about $10^3$–$10^4\,GM/c^2$, so the default $r_{out}$ affects their IR
  emission.
* **Observational effects.** The model includes no interstellar or intrinsic
  absorption (which removes most of the EUV), no cosmological redshift (fine
  at 100 Mpc, $z\approx0.02$), and no time variability.

## References

* Shakura, N. I., & Sunyaev, R. A. (1973), *A&A*, 24, 337 — the standard thin-disk model.
* Novikov, I. D., & Thorne, K. S. (1973), in *Black Holes (Les Astres Occlus)* — relativistic thin disks.
* Frank, J., King, A., & Raine, D. (2002), *Accretion Power in Astrophysics*, 3rd ed., Cambridge University Press — ch. 5 derives $T(r)$ and the multi-temperature blackbody spectrum.
* Shimura, T., & Takahara, F. (1995), *ApJ*, 445, 780 — spectral hardening (color correction) in disk atmospheres.
