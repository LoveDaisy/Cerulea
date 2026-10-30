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

*(to be completed; see REPORT.md for what was actually used)*

## check5 — twilight

*(to be completed; see REPORT.md for what was actually used)*

## Model data (not references)

* `skycolor/data/` — solar spectrum (ASTM G173 AM0), CIE 1931 CMFs, O₃ and
  NO₂ cross sections.  Provenance: `skycolor/data/SOURCES.md`.
* All model atmospheres are built from `skycolor/presets.py`
  (`clear`: 300 DU O₃, AOD₅₅₀ = 0.05, exponential aerosol with H = 1.5 km,
  Hinterland aerosol; `pristine`: Rayleigh + O₃ only).
