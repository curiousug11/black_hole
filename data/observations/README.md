# Observational data

Everything here was downloaded by `scripts/fetch_observations.py` on 2026-11-03.
Re-run that script to reproduce it (existing files are not re-downloaded).
`scripts/compare_observations.py` uses these files.

| File | Content | Source |
|---|---|---|
| `vandenberk2001_composite.txt` | SDSS median composite quasar spectrum: rest wavelength (Å), f_λ and its 68% semi-interquartile range (arbitrary units), 800–8555 Å | Vanden Berk et al. 2001, AJ 122, 549 ([doi:10.1086/321167](https://doi.org/10.1086/321167)), Table 1, from the arXiv source of [astro-ph/0105231](https://arxiv.org/abs/astro-ph/0105231) |
| `shen2011_dr7_quasars.csv.gz` | 104,746 SDSS DR7 quasars with a virial black hole mass: redshift, BAL flag, bolometric and monochromatic luminosities, fiducial log M_BH, Eddington ratio, and the local continuum slopes α_λ (f_λ ∝ λ^α_λ) fitted around Hβ, Mg II and C IV, with errors and S/N | Shen et al. 2011, ApJS 194, 45 ([doi:10.1088/0067-0049/194/2/45](https://doi.org/10.1088/0067-0049/194/2/45)); catalogue file `dr7_bh_Nov19_2013.fits` from <http://quasar.astro.illinois.edu/BH_mass/dr7.htm>; only the columns listed in the file header are kept |
| `rxte_lmcx3/` | RXTE/PCA standard-product spectra of LMC X-3: for each pointing, the source spectrum (`xp*_s2.pha.gz`), background spectrum (`xp*_b2.pha.gz`) and response matrix (`xp*.rsp.gz`), summed over all PCA detectors and layers. 65 pointings from 1996–1999 with ≥ 3 ks exposure; `observations.csv` lists them | NASA HEASARC RXTE archive, <https://heasarc.gsfc.nasa.gov/FTP/xte/data/archive/>; observation list from the HEASARC `xtemaster` table |

## Definitions used in the comparison

* **Continuum windows (rest frame).** Shen et al. (2011) fit the local
  power law f_λ = A λ^α_λ, together with an iron template, in these windows:
  Hβ [4435, 4700] + [5100, 5535] Å; Mg II [2200, 2700] + [2900, 3090] Å;
  C IV [1445, 1465] + [1700, 1705] Å.
  The model slope is measured in the same windows.
* **Eddington ratio.** In the catalogue, L_bol comes from fixed bolometric
  corrections (BC₅₁₀₀ = 9.26, BC₃₀₀₀ = 5.15, BC₁₃₅₀ = 3.81; Richards et al. 2006)
  and L_Edd = 1.26×10³⁸ (M/M☉) erg s⁻¹.
* **LMC X-3 system parameters.**
  * M = 6.98 ± 0.56 M☉ and i = 69.24 ± 0.72° (Orosz et al. 2014, ApJ 794, 154).
  * D = 48.1 ± 2.2 kpc (Orosz et al. 2009, as adopted by Steiner et al. 2014, ApJL 793, L29).
  * N_H = 4×10²⁰ cm⁻². At this column the absorption is below 1% above 3 keV, so it is neglected in the 3–20 keV fits.

## Caveats

* The quasar catalogue is downloaded over plain HTTP (the server offers no
  HTTPS). The file is a public data product, but its integrity is not
  cryptographically verified.
* RXTE standard products are quick-look products. They are not corrected for
  detector dead time, and they do not include the Crab-based flux recalibration
  (`crabcor`) used by Steiner et al. (2014). That recalibration changes the
  normalisation by ~10%, and hence the fitted f_col by ~2–3%.
