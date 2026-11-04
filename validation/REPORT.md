# Validation report — is the model right?

This is the **original single-scattering baseline**, recorded on
`feat/validation` with Python 3.11 / numpy 2.4.6 / scipy 1.17.1. Multiple
scattering was still a stub. The five-check table and interpretation below
retain that historical context; they are not the current harness output.

The latest committed six-check [results.json](results.json) has **3 PASS /
3 FAIL**; [STATUS.md](../docs/STATUS.md) explains the present limits.
[ADDENDUM.md](ADDENDUM.md) records subsequent changes, and
[the documentation guide](../docs/README.md) maps the later independent audits.
Reference sources are documented in [SOURCES.md](data/SOURCES.md).

From the repository root, run the **current** checks with:

```bash
uv run python -m validation.run_all
```

This writes `validation/results.json` and figures; it is a new calculation,
not a replay of the historical tables below.

## Historical summary

| # | Check | Reference | Verdict |
|---|---|---|---|
| 1 | Rayleigh optical depth, 300–700 nm | Bodhaine et al. (1999), Table 3 / eq. (30) | **PASS** |
| 2 | Direct + diffuse clear-sky illuminance | CIE 110-1994, ASTM G173-03, Bird & Hulstrom | **PASS** |
| 3 | Clear-sky relative luminance distribution | CIE 110-1994 | **FAIL** (horizon only) |
| 4 | Sky chromaticity (horizon) | SKYSPECTRA measured spectra | **PASS** (marginal) |
| 5 | Twilight zenith luminance | Patat et al. (2006), V band | **FAIL** (−9° and below) |

---

## Check 1 — Rayleigh optical depth vs Bodhaine et al. (1999)

**Reference.** Bodhaine, Wood, Dutton & Slusser (1999), *On Rayleigh optical
depth calculations*, J. Atmos. Oceanic Technol. 16, 1854–1861.  We compare the
model's vertical Rayleigh optical depth with the paper's recommended values
for dry air with 360 ppm CO₂ at sea level, 1013.25 mb, 45°N (Table 3), using
the paper's eq. (30) fit for the dense 1 nm comparison and the tabulated
values as an independent spot check.  The tabulated values reproduce eq. (30)
to better than 2×10⁻⁴ relative, which validates our transcription.

**What this tests.** The model uses the same *formula family*, so this is not
an independent derivation of Rayleigh physics.  It is an independent test of
the implementation: refractive index of air, King (depolarisation) factor,
vertical air column, units.

**Numbers.**

| λ (nm) | model τ_R | Bodhaine eq. (30) | deviation |
|---:|---:|---:|---:|
| 300 | 1.203842 | 1.216406 | −1.03 % |
| 320 | 0.913867 | 0.922028 | −0.89 % |
| 340 | 0.706987 | 0.712476 | −0.77 % |
| 350 | 0.625745 | 0.630300 | −0.72 % |
| 400 | 0.358240 | 0.360213 | −0.55 % |
| 450 | 0.220136 | 0.221106 | −0.44 % |
| 500 | 0.142830 | 0.143353 | −0.36 % |
| 550 | 0.096762 | 0.097065 | −0.31 % |
| 600 | 0.067901 | 0.068087 | −0.27 % |
| 650 | 0.049064 | 0.049184 | −0.24 % |
| 700 | 0.036340 | 0.036421 | −0.22 % |

RMS deviation over 300–700 nm: **0.48 %**; worst case **1.03 %** at 300 nm.

The model's molecular cross section at 550 nm is 4.507×10⁻³¹ m²; Bodhaine's
recommended calculation gives 4.511×10⁻³¹ m² (−0.1 %).  The small τ deficit
at short wavelengths is the expected consequence of using Edlén's refractive
index and a constant depolarisation factor (0.0279) instead of Bodhaine's
recommended Peck & Reeder index and wavelength-dependent Bates King factor.

**Tolerance.** 2 % at every wavelength (1 % over 400–700 nm).  The model
passes with a large margin.

**Verdict: PASS.**  The molecular scattering backbone is numerically sound.

**Documentation defect (not a physics bug).** The docstring of
`skycolor/spectra/rayleigh_cross_section` says it returns 1.135×10⁻³¹ m² at
550 nm.  The code actually returns 4.507×10⁻³¹ m², which is the correct
value (and the one consistent with τ_R(550) ≈ 0.097).  The docstring number
is wrong by a factor of four.  No code was changed.

---

## Check 2 — clear-sky direct and diffuse illuminance

**References.**  No single published table covers direct and diffuse
illuminance at both requested elevations, so three independent references are
used:

1. **ASTM G173-03** direct-normal spectrum at air mass 1.5 (solar elevation
   48.19°) — the standard reference spectrum.  Its photopic integral is
   **97.14 klx** and its 400–700 nm integral is **374.8 W m⁻²** (computed from
   the committed `data_raw/ASTMG173.csv`).
2. **CIE 110-1994** clear-sky diffuse illuminance at 30° and 60°, obtained by
   integrating the absolute luminance distribution with the standard's
   zenith-luminance relation, at turbidity **T = 2.2** (a typical clean
   continental value for AOD₅₅₀ ≈ 0.05).  At T = 2.2 the CIE zenith
   luminance agrees with the model to within 3 %, which is itself a useful
   consistency check.
3. **Bird & Hulstrom (1981)** clear-sky broadband model (as implemented in
   pvlib), with AOD₅₀₀ = 0.057, precipitable water 1.5 cm and ozone
   0.3 atm-cm, converted to illuminance with the G173 direct-beam luminous
   efficacy **107.9 lm/W**.  This supplies an independent direct-normal
   estimate at 30° and 60°.

**Numbers** (model / reference):

| quantity | model | reference | deviation | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| DNI illuminance, h = 30° (Bird) | 94.8 klx | 91.2 klx | +4.0 % | 15 % | PASS |
| DNI illuminance, h = 60° (Bird) | 109.5 klx | 103.6 klx | +5.7 % | 15 % | PASS |
| DNI illuminance, AM1.5 (G173) | 101.9 klx | 97.1 klx | +4.9 % | 10 % | PASS |
| DNI 400–700 nm, AM1.5 | 394.2 W m⁻² | 374.8 W m⁻² | +5.2 % | 10 % | PASS |
| DNI spectral RMS, 400–700 nm | 6.3 % | 0 | 6.3 % | 10 % | PASS |
| DHI illuminance, h = 30° (CIE 110, T=2.2) | 7.31 klx | 6.92 klx | +5.6 % | 25 % | PASS |
| DHI illuminance, h = 60° (CIE 110, T=2.2) | 8.75 klx | 7.94 klx | +10.2 % | 25 % | PASS |

Figure: `figures/check2_illuminance.png`.

**Verdict: PASS** on all metrics, with two honest caveats.

* The direct beam is systematically **4–6 % high**.  The model has no water
  vapour, O₂ or CO₂ absorption.  In the visible this produces the 45 %
  spike at the O₂ A-band (688 nm) visible in the spectral panel; integrated
  over 400–700 nm it is only +5.2 %, and the illuminance error is +4.9 %.
* The diffuse check is sensitive to the assumed CIE turbidity.  At **T = 3**
  (a hazier sky) the same model values are **33 % and 38 % low** at h = 30°
  and 60° respectively.  This is the expected signature of the missing
  multiple scattering.  At T = 2.2, which matches the model's clean
  continental aerosol load, the integrated diffuse illuminance agrees within
  10 % — the horizon-distribution error seen in check 3 largely averages out
  in the horizontal integral.

---

## Check 3 — CIE 110-1994 clear-sky relative luminance

**Reference.** CIE 110-1994, *Spatial distribution of daylight — Luminance
distributions of various reference skies*.  The standard clear-sky
distribution is

```
L(θ,γ)/Lz = (0.91 + 10 e^(−3γ) + 0.45 cos²γ)(1 − e^(−0.32/cosθ))
          / [(0.91 + 10 e^(−3θs) + 0.45 cos²θs)(1 − e^(−0.32))]
```

with θ the viewing zenith angle, γ the angular distance from the sun and θs
the solar zenith angle (reproduced as eq. (4.31) of Zotti 2007, citing CIE
110-1994).  This is a purely geometric test: both sides are normalised to
their own zenith luminance, so no absolute radiometry or solar spectrum is
involved.

**Method.**  Model photopic luminance is the 683 lm/W integral of the
single-scattered spectral radiance (`presets.clear`, AOD₅₅₀ = 0.05,
exponential aerosol H = 1.5 km) on a 37 × 72 sky grid, divided by its own
zenith value.  The CIE map is evaluated on the same grid.

**Numbers** (model / CIE, sun-side horizon means the horizon point in the
sun's azimuth; anti-solar is 180° away):

| solar elev. | horizon sun-side / zenith | horizon anti-solar / zenith | RMS θ≤60° | RMS 75–90° | RMS whole sky | max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 30° | **19.86 / 8.35 (2.38×)** | 3.05 / 3.14 (−2.9 %) | 3.0 % | 43.0 % | 19.4 % | 138 % |
| 45° | **7.36 / 3.65 (2.01×)** | 2.09 / 2.00 (+4.1 %) | 3.8 % | 40.3 % | 18.4 % | 101 % |
| 60° | **2.76 / 1.60 (1.73×)** | 1.25 / 1.14 (+9.4 %) | 3.8 % | 36.5 % | 16.8 % | 73 % |

Figures: `figures/check3_cie_luminance.png` (maps and model/CIE ratio) and
`figures/check3_cie_cuts.png` (cuts through the sun meridian at h = 45°).

**Verdict: FAIL**, but the failure is localised and interpretable:

* For view zenith angles below 60° (the upper sky) the model is excellent:
  RMS 3–4 %, maximum deviation below 20 %.
* The anti-solar horizon agrees with CIE to within 4–9 %.
* The sun-side horizon is **about 2× too bright** at h = 45° (2.4× at
  h = 30°, 1.7× at h = 60°), and the whole-sky RMS (18.4 %) is dominated by
  the 75–90° band (40.3 %).

**The model puts the maximum luminance in the wrong place.**  At h = 45° the
CIE sky is brightest at the sun (7.25 at θ = 45°, γ = 0) and falls to 3.65 at
the sun-side horizon.  The model is the other way round: 5.82 at the sun and
7.36 at the horizon.  The model's aureole is therefore **too weak relative to
the horizon**, not too strong.

**Attribution.**  We re-ran h = 45° with two variants:

| atmosphere | zenith L [cd m⁻²] | horizon sun / zenith | horizon anti / zenith |
|---|---:|---:|---:|
| pristine (Rayleigh + O₃ only) | 977 | 9.78 | 9.74 |
| clear (AOD 0.05, H = 1.5 km, as used above) | 1571 | 7.36 | 2.09 |
| clear, well-mixed aerosol (H = 20 km) | 1578 | 6.30 | 4.84 |
| CIE 110 | — | 3.65 | 2.00 |

* The 1.5 km exponential aerosol profile is **not** the cause of the
  sun-side excess: removing the aerosol makes it *worse* (9.78), and a
  well-mixed profile only brings it down to 6.30.  What the exponential
  profile does do is suppress the anti-solar horizon from 9.74 (pristine)
  to 2.09, which is why the anti-solar side happens to agree with CIE.
* The likely dominant cause is the **absence of multiple scattering**.
  At 550 nm the horizon path has τ ≈ 3.7, so the limb is optically thick and
  single scattering is a poor approximation there, while the zenith is
  optically thin (τ ≈ 0.13).  Multiple scattering adds a diffuse component
  that raises the zenith much more (relatively) than the already bright
  limb, which is exactly the direction needed to reduce the ratio from 7.4
  to 3.6.  This is consistent with the absolute scale: the model's zenith
  luminance at h = 45° is 1.57 kcd m⁻², while CIE 110 with T = 3 gives
  2.70 kcd m⁻² — multiple scattering would supply much of the missing
  zenith light.
* The residual forward aureole is a secondary effect.  Its shape is wrong
  (too weak at the sun, too strong at the limb), which is what one expects
  when the limb's single-scattered radiance is not redistributed by higher
  orders.

### Pristine / uniform-H evidence

The table above already records the key attribution experiment: the limb
excess appears with and without aerosol (pristine 9.78, exponential 7.36,
well-mixed 6.30, CIE 3.65).  The aerosol profile changes *which* side of the
horizon is wrong, but not the existence of a ~2x limb error.  That is the
signature of the missing scattering orders, not of a bad aerosol model.

We did **not** change the model.

## Check 4 — horizon chromaticity vs measured spectra

**Reference.** SKYSPECTRA (Balakrishnan et al. 2023, CIE x050:2023 OP026;
open data, Zenodo DOI [10.5281/zenodo.8147546](https://doi.org/10.5281/zenodo.8147546)).
The `spectral_patch_radiance` table contains measured 10°-aperture spectral
radiance of sky patches.  For Berlin the distributed patches are on the
horizon ring (6° almucantar, azimuths 180/192/204/216/228/240°).

**Selection.** All DE-BLN setup-1 timestamps with `sky_condition = clear` and
solar elevation 30–50° (13 timestamps, 76 patch spectra, mostly 2015-03-08
plus a few June/September days).  The model is evaluated at the same sun
position and the same viewing geometry, with `presets.clear` (AOD₅₅₀ = 0.05).
The extracted reference data are in
`data/skyspectra_berlin_horizon_xy.csv` and
`data/skyspectra_berlin_horizon_mean_spectrum.csv`.

**Numbers** (CIE 1931 xy):

| quantity | model | measured | deviation | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| mean x, horizon | 0.2989 | 0.2913 | +0.0076 | 0.02 | PASS |
| mean y, horizon | 0.3242 | 0.3128 | +0.0115 | 0.02 | PASS |
| RMS Δxy | 0.0193 | 0 | 0.0193 | 0.02 | PASS (marginal) |
| RMS Δu′v′ | 0.0110 | 0 | 0.0110 | 0.02 | PASS |

Figure: `figures/check4_chromaticity.png`.

**Verdict: PASS, but marginal.**  The model reproduces the measured horizon
chromaticity to about 0.019 in xy (0.011 in u′v′), which is a few JNDs.  The
residual is systematic: the model horizon is slightly **too yellow** (both x
and y are high).  The result depends on the assumed aerosol load:

| AOD₅₅₀ | RMS Δxy |
|---:|---:|
| 0.02 | 0.0154 |
| 0.05 | 0.0193 |
| 0.10 | 0.0360 |

A cleaner atmosphere (AOD 0.02) fits Berlin better; a hazy one (AOD 0.10) is
clearly too yellow.  This is consistent with the check-3 finding that the
model's aerosol/horizon treatment is not calibrated.

**What this does *not* test.**  All distributed Berlin patches are at 6°
elevation, so this is a horizon check only.  No open measured **zenith**
spectrum was found in the timebox.  For information, the model gives
zenith (x, y) = (0.2602, 0.2734), (0.2644, 0.2771) and (0.2731, 0.2864) at
h = 30°, 45°, 60°.  At h = 30° the model's y is just below the 0.28–0.32
range quoted in `docs/ARCHITECTURE.md`; adding multiple scattering would
whiten the zenith and raise y, so the missing MS is a plausible explanation,
but this is *not* established by a measurement here.

## Check 5 — twilight zenith luminance vs Patat et al. (2006)

**Reference.** F. Patat, O. S. Ugolnikov & O. V. Postylyakov (2006), *UBVRI
twilight sky brightness at ESO-Paranal*, A&A 454, 229,
[arXiv:astro-ph/0604128](https://arxiv.org/abs/astro-ph/0604128).  Table 1
gives a second-order fit to the measured V-band zenith surface brightness for
sun zenith distance 95–105°:

```
m_V(ζ) = 11.84 + 1.518 (ζ − 95) − 0.057 (ζ − 95)²   [mag arcsec⁻²]
```

converted to luminance with `L = 1.08e5 × 10^(−0.4 m_V)` cd m⁻² (1 cd m⁻²
≈ 12.6 V mag arcsec⁻²).  The values used are in
`data/twilight_patat2006.csv`.

**Numbers** (model / reference, zenith):

| solar elev. | model L [cd m⁻²] | Patat L [cd m⁻²] | model/ref | deviation | verdict |
|---:|---:|---:|---:|---:|---|
| −3° | 28.4 | 40.1 | 0.71 | −0.15 dex | PASS (ref. extrapolated) |
| −6° | 0.567 | 0.516 | 1.10 | +0.04 dex | PASS |
| −9° | 7.24×10⁻⁴ | 1.71×10⁻² | 0.042 | −1.37 dex | **FAIL** |
| −12° | 0 (exactly) | 1.46×10⁻³ | 0 | < −9 dex | **FAIL** |
| −18° | 0 (exactly) | 1.81×10⁻⁴ | 0 | < −8 dex | **FAIL** |

Figure: `figures/check5_twilight.png`.

**Verdict: FAIL**, exactly where the architecture predicts it should fail.
The single-scattering model is remarkably good at −6° (10 % high), but it
collapses at −9° (24× too dark) and returns **exactly zero** at −12° and
below.  The reason is geometric, not numerical: at −12° the zenith line of
sight lies entirely inside the Earth's shadow, so there is no first-order
scattering along it.  All the light seen at the zenith in real deep twilight
arrives by multiple scattering from the still-sunlit layers near the horizon.
At −18° the reference is the natural night sky (Patat et al. state that the
night-sky level is reached at ζ ≈ 105–106°); the model contains no airglow
and no multiple scattering, so zero is the expected answer.

Caveats: the −3° and −18° reference values are extrapolations of the fit
beyond its stated 95–105° validity range.  The −3° comparison should be read
as indicative only; the −12° and −9° comparisons are within the fitted range
and are the meaningful ones.

---

## Overall judgement

**What is trustworthy.**

* The molecular optical depth is correct to ~1 % (check 1).
* Absolute clear-sky *photometry* is good: direct normal illuminance within
  5–6 % of ASTM G173 and Bird, diffuse horizontal illuminance within 10 % of
  CIE 110 at a matching turbidity (check 2).
* For the upper sky (view zenith ≤ 60°) the relative luminance distribution
  matches CIE 110 to 3–4 % RMS (check 3).
* The daytime horizon *hue* matches measured Berlin spectra to Δxy ≈ 0.019
  (check 4), with a small yellow bias.

**What is not trustworthy yet.**

1. **Deep twilight.**  Below −9° the model is a factor 24 too dark; below
   −12° it is identically zero (check 5).  Any twilight or "blue hour"
   colour is therefore a single-scattering approximation, not a prediction.
2. **The bright limb.**  The sun-side horizon is ~2× too bright relative to
   the zenith at h = 45°, and the model puts the luminance maximum at the
   horizon instead of at the sun (check 3).  Absolute horizon luminance and
   the shape of the aureole should not be trusted.
3. **The forward aureole / circumsolar region.**  It is too weak at the sun
   and too strong at the limb (check 3); this is the same missing-MS problem.
4. **Gas absorption.**  The model has no H₂O, O₂ or CO₂; direct beam is 4–6 %
   high in the visible and up to 45 % high at the O₂ A-band (check 2).
5. **Zenith saturation.**  The model zenith is at the blue edge of the
   expected chromaticity range (y ≈ 0.273 at h = 30°); multiple scattering
   would whiten it.  Not measured here, so this is a hypothesis, not a
   verdict.

**Bottom line.**  The model's *colour* predictions are trustworthy for the
daytime upper sky and qualitatively right for the horizon and sunset, at the
level of "is this sky blue/white/orange and roughly how saturated".  They are
**not** trustworthy for absolute luminance near the horizon or for any
twilight scene deeper than about −6°, until multiple scattering is
implemented.  The failures are consistent, understood, and all point at the
same missing physics rather than at independent bugs.

## Running current checks

From the repository root:

```bash
uv run python -m validation.run_all
uv run python -m validation.check3_cie_luminance
```

`run_all.py` exits 0 if every check that ran passed (skipped checks do not
fail the run), 1 if any check failed. Exact reproduction of the tables in
this historical report requires its original model and environment.
