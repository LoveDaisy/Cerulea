# Reference data sources

Everything in `validation/data/` was produced from, or checked against, the
sources below.  No reference data was invented; where a number is derived
from a published formula rather than transcribed from a table this is stated
explicitly.

## check1 — Rayleigh optical depth

* **B. A. Bodhaine, N. B. Wood, E. G. Dutton and J. R. Slusser (1999)**,
  *On Rayleigh optical depth calculations*, J. Atmos. Oceanic Technol. **16**,
  1854–1861.  DOI: [10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2](https://doi.org/10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2)
  * Table 3: scattering cross section, Rayleigh optical depth at sea level
    (45°N, 1013.25 mb) and at Mauna Loa, and the King factor, for 250–1000 nm.
  * Eq. (30): five-parameter fit to the same data,
    `τ_R = 0.0021520 (1.0455996 − 341.29061 λ⁻² − 0.90230850 λ²) /
    (1 + 0.0027059889 λ⁻² − 85.968563 λ²)`, λ in µm.
    The paper quotes an accuracy better than 0.01 % over 250–850 nm.
  * Full text consulted via a public transcript:
    <https://studyres.com/doc/15079508/on-rayleigh-optical-depth-calculations>
    (the journal version is paywalled).  Tabulated spot values were
    cross-checked against eq. (30) to better than 2×10⁻⁴ relative; the
    transcript's σ column has an OCR row offset, so only the τ column is
    used.

`data/bodhaine1999_table3.csv` columns: wavelength, τ from eq. (30),
τ from the transcribed Table 3 (interpolated onto the 1 nm grid), and the
model τ.

## check2 — clear-sky illuminance / luminance scale

* **CIE 110-1994**, *Spatial distribution of daylight — Luminance
  distributions of various reference skies*, Commission Internationale de
  l'Éclairage, Vienna.  The clear-sky relative luminance distribution and the
  empirical zenith-luminance relation
  `Yz = (1.376 T − 1.81) tan h + 0.38` kcd m⁻² (T = turbidity, h = solar
  elevation) are used.  The standard itself is paywalled; the equations are
  reproduced, with a citation to CIE 110-1994, in:
  * **G. Zotti (2007)**, *Computer graphics in historical and modern sky
    observations*, PhD thesis, TU Wien, eq. (4.31) and §4.5,
    <https://repositum.tuwien.at/bitstream/20.500.12708/13984/2/Zotti%20Georg%20-%202007%20-%20Computer%20graphics%20in%20historical%20and%20modern%20sky%20observations.pdf>
* **ASTM G173-03**, *Standard Tables for Reference Solar Spectral
  Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface*.
  The direct-normal column is used as an independent spectral reference at
  air mass 1.5 (solar elevation 48.19°).  The file is already in the
  repository at `data_raw/ASTMG173.csv` (see `skycolor/data/SOURCES.md`);
  no new download was needed.

## check3 — CIE clear-sky luminance distribution

Same CIE 110-1994 equations as check2 (relative distribution is independent
of the absolute scale).  CIE 110-1994 is the predecessor of the general-sky
system of ISO 15469:2004 / CIE S 011; the *standard clear sky* is the same
distribution (in the 15-type system it is type 12, parameters a = −1,
b = −0.32, c = 10, d = −3, e = 0.45 — but we use the closed form above
directly, which is unambiguous).

## check4 — sky chromaticity

* **P. Balakrishnan, A. Diakite-Kortlever, D. Dumortier, J. Hernández-Andrés,
  P. Kenny, M. Maskarenj, C. Pierson, A. Thorseth, P. Xue and M. Knoop
  (2023)**, *SKYSPECTRA: an opensource data package for worldwide spectral
  daylight*, Proceedings of the 30th CIE Session, Ljubljana, CIE x050:2023
  OP026.  DOI: [10.25039/x50.2023.OP026](https://doi.org/10.25039/x50.2023.OP026).
  Open data: Zenodo record
  [10.5281/zenodo.8147546](https://doi.org/10.5281/zenodo.8147546).
  * `spectral_patch_radiance.csv`: measured 10°-aperture spectral radiance of
    sky patches, 360–800 nm at 1 nm.
  * `meta_sun_positions.csv`, `meta_weather.csv`: sun geometry and
    `sky_condition` / `sky_type_cie` flags.
  * `meta_location.csv`: Berlin (DE-BLN), 52.51°N, 13.33°E, 35 m.
  * We used the DE-BLN block (first 53 MB of the 1.42 GB
    `spectral_patch_radiance.csv`; the full file was not downloaded).  All
    distributed Berlin patches are at the 6° almucantar and azimuths
    180–240°.
  * Extracted subset: `data/skyspectra_berlin_horizon_xy.csv` (76 patch
    spectra from 13 clear-sky timestamps with solar elevation 30–50°) and
    `data/skyspectra_berlin_horizon_mean_spectrum.csv` (mean spectrum per
    patch).  The selection and extraction are documented in
    `validation/check4_chromaticity.py`.

## check5 — twilight

* **F. Patat, O. S. Ugolnikov and O. V. Postylyakov (2006)**, *UBVRI twilight
  sky brightness at ESO-Paranal*, Astronomy & Astrophysics **454**, 229–235.
  DOI: [10.1051/0004-6361:20064986](https://doi.org/10.1051/0004-6361:20064986);
  arXiv: [astro-ph/0604128](https://arxiv.org/abs/astro-ph/0604128).
  Table 1, V band: `m_V(ζ) = 11.84 + 1.518(ζ−95) − 0.057(ζ−95)²`
  mag arcsec⁻², fitted over 95° ≤ ζ ≤ 105°.
* **V surface brightness → luminance.**  Standard Johnson V relation
  `L = 1.08×10⁵ × 10^(−0.4 m_V)` cd m⁻², i.e. 1 cd m⁻² ≈ 12.6 V
  mag arcsec⁻².  A 0.1 mag error is a 10 % luminance error.
* Derived values: `data/twilight_patat2006.csv`.  The −3° and −18° entries are
  extrapolations beyond the fit range; the −18° value is the natural night-sky
  level (Patat et al. find it is reached at ζ ≈ 105–106°).

## check5 — twilight

*(to be completed; see REPORT.md for what was actually used)*

## Model data (not references)

* `skycolor/data/` — solar spectrum (ASTM G173 AM0), CIE 1931 CMFs, O₃ and
  NO₂ cross sections.  Provenance: `skycolor/data/SOURCES.md`.
* All model atmospheres are built from `skycolor/presets.py`
  (`clear`: 300 DU O₃, AOD₅₅₀ = 0.05, exponential aerosol with H = 1.5 km,
  Hinterland aerosol; `pristine`: Rayleigh + O₃ only).

## check6 — a second, independent twilight dataset

* **M. J. Koomen, C. Lock, D. M. Packer, R. Scolnik, R. Tousey and E. O.
  Hulburt (1952)**, *Measurements of the Brightness of the Twilight Sky*,
  Journal of the Optical Society of America **42**, 353–359.  DOI:
  [10.1364/JOSA.42.000353](https://doi.org/10.1364/JOSA.42.000353).
  * A recording photometer of **photopic** sensitivity (no filter), at two
    stations: **Maryland, altitude 30 m**, and **Sacramento Peak, New Mexico,
    altitude 2800 m**; solar altitude +5° to −15°, clear air, no clouds.
  * The Maryland **zenith** surface brightness in cd ft⁻², and the conversion
    `m = −2.5 log₁₀ I[cd ft⁻²] + 9.9846` mag arcsec⁻², are taken from Table 5
    of **S. Nawar, A. B. Morcos, R. M. El Agmy, G. M. A. Gad and S.
    Elgohary (2020)**, *Sky twilight brightness at zenith expressed in
    magnitudes*, NRIAG J. Astron. Geophys. **9**, 63–70, DOI
    [10.1080/20909977.2020.1714176](https://doi.org/10.1080/20909977.2020.1714176)
    (CC-BY).  The 1952 article itself is paywalled; its abstract was read from
    the Optica archive and confirms the instrument, both sites, their
    altitudes and the Maryland/Sacramento Peak ratio.
  * Conversion applied here: `L = 10.7639 I[cd ft⁻²]` cd m⁻².  The source's
    magnitude column is kept separately; it agrees with the conversion to
    0.0006 mag.  The values are quoted to 1–2 significant figures.
  * Derived values: `data/twilight_koomen1952.csv`.

* **S. Nawar et al. (2020)**, same paper, Tables 1–3: zenith twilight
  brightness at **Helwan** (115 m), **Daraw** (250 m) and **Abu Simbel**
  (210 m), 0–15° solar depression, in mag arcsec⁻², for broad glass filters at
  4410 Å (B), 5500 Å (yellow) and 7900 Å (R).  Converted by the authors from
  S₁₀(λ) units with `m = −2.5 log₁₀ S₁₀ + 27.78`.  These are **not** Johnson
  bands; the yellow (5500 Å) column is used as a V proxy only.  Luminance via
  `L = 1.08×10⁵ × 10^(−0.4 m)` cd m⁻², the same relation used by check5.
  * Derived values: `data/twilight_nawar2020.csv` (112 rows, all three bands).
  * Every row was checked against the S₁₀→magnitude relation; maximum residual
    0.055 mag (a single Abu Simbel blue row that the source rounds to 9.8).

* **Patat et al. (2006)**, CrAO vs Paranal pressure comparison used in
  `SECOND_DATASET.md` §4.2: §4 of the paper (arXiv:astro-ph/0604128) gives
  ΔV = 0.27 ± 0.03 mag between the CrAO site (600 m, 961 hPa) and Paranal
  (2400 m, 743 hPa), against a pressure prediction of 0.28 mag.
