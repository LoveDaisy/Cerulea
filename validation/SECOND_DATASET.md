# A second twilight dataset: Koomen et al. (1952)

*The second dataset asked for by open question #2 of `docs/ROADMAP.md`.*

**Short answer.** A second, independent, absolute twilight measurement exists
and it does **not** show the model 1.5 × high at −6°.  Koomen et al. (1952)
measured the zenith twilight sky at Maryland (30 m above sea level) with a
photopic photometer — the same photopic weighting the model computes.  Against
it the sea-level model is **1.14 ×** high at −6°, not 1.5 ×.  Koomen's
sea-level sky is itself **1.25 ×** brighter than the Patat et al. (2006) fit,
and that 0.24 mag is the altitude difference between Maryland (30 m) and
Paranal (2400 m), which Patat et al. measure independently (ΔV = 0.27 mag
between 743 and 961 hPa) and which Koomen's own second station confirms
(Sacramento Peak, 2800 m, is 2/3 to 1/2 as bright as Maryland).

So the 1.5 × at −6° is mostly a **site mismatch**, not a wrong reference and
not a wrong sea-level calibration: 1.14 (model vs sea level) × 1.25 (sea level
vs Paranal) = 1.43, which is the model/Patat ratio.  What *is* a model problem
is that the model has **no altitude response at all** (1.00 × over the same
pressure ratio where the data give 1.25–1.39 ×).  The model is right at sea
level and cannot represent a 2400 m site.

---

## 1. Why a second dataset was needed

`validation/check5_twilight.py` compares the model with a single reference, the
V-band fit of Patat, Ugolnikov & Postylyakov (2006, A&A 455, 385), obtained
with FORS1 on the VLT at **ESO-Paranal, altitude 2400 m**.  The model
atmosphere is a **sea-level** standard atmosphere (1013 hPa, 300 DU O₃,
AOD₅₅₀ = 0.05 in `presets.clear`).  The comparison therefore mixes a model
error with a site difference.  A second measurement at a *different altitude*
is the only way to separate them.

## 2. The datasets

### 2.1 Koomen et al. (1952) — the primary second dataset

M. J. Koomen, C. Lock, D. M. Packer, R. Scolnik, R. Tousey and E. O. Hulburt,
*Measurements of the Brightness of the Twilight Sky*, J. Opt. Soc. Am. **42**,
353–359 (1952), DOI [10.1364/JOSA.42.000353](https://doi.org/10.1364/JOSA.42.000353).

From the published abstract (the article itself is paywalled):

* "a recording photometer of **photopic** sensitivity" — no filter, so the
  measured quantity is photopic luminance, exactly what
  `validation.common.luminance` computes from the model spectrum.  This is a
  closer match to the model than a Johnson V magnitude;
* two stations: **Maryland, altitude 30 m**, and **Sacramento Peak, New
  Mexico, altitude 2800 m**;
* solar altitude **H = +5° to −15°**, clear air, no clouds;
* "except at the horizon the Sacramento Peak sky was about 2/3 to 1/2 as
  bright as the Maryland sky because of clearer air".

The zenith surface brightness at Maryland, in cd ft⁻², is quoted in **Table 5
of Nawar et al. (2020)** (see below), which is CC-BY and also gives the
conversion to mag arcsec⁻².  Values at D = 0, 3, 6, 9, 12, 15°:

| solar depression | I [cd ft⁻²] | *m* [mag arcsec⁻²] | *L* [cd m⁻²] |
|---:|---:|---:|---:|
| 0° | 15 | 7.044 | 161.5 |
| 3° | 2 | 9.232 | 21.53 |
| 6° | 0.06 | 13.039 | 0.6458 |
| 9° | 0.0015 | 17.044 | 0.01615 |
| 12° | 0.00012 | 19.786 | 1.292 × 10⁻³ |
| 15° | 0.00004 | 20.979 | 4.31 × 10⁻⁴ |

Caveats, stated up front:

* the numbers are quoted to **one or two significant figures** (0.06, 0.0015,
  0.00012, 0.00004), so the −6° point carries a ±0.05 mag rounding and the
  −12/−15° points ±0.02–0.04 mag.  The *systematic* uncertainty of a 1952
  absolute calibration is not stated;
* only every third degree is available from the reproduction (the original
  paper has denser tables);
* the 0° and 3° rows are outside the range where the Patat fit is defined and
  are discussed separately in §4.4.

### 2.2 Nawar et al. (2020) — secondary, used only as a cross-check

S. Nawar, A. B. Morcos, R. M. El Agmy, G. M. A. Gad and S. Elgohary,
*Sky twilight brightness at zenith expressed in magnitudes*, NRIAG J. Astron.
Geophys. **9**, 63–70 (2020), DOI
[10.1080/20909977.2020.1714176](https://doi.org/10.1080/20909977.2020.1714176).
CC-BY.

Tables 1–3 give the zenith twilight brightness at **Helwan** (115 m),
**Daraw** (250 m) and **Abu Simbel** (210 m), 0–15° solar depression, in
mag arcsec⁻², for broad glass filters at 4410 Å (B), 5500 Å (yellow) and
7900 Å (R).  They are converted by the authors from the S₁₀(λ) units of the
original 1977/1995 Egyptian measurements with

```
m = −2.5 log₁₀ S₁₀(λ) + 27.78  [mag arcsec⁻²].
```

The yellow (5500 Å) column is the closest thing to V and is the only one used
here.  These are **not** Johnson bands, the conversion is second-hand, and the
site-to-site spread is 1.5 mag at −6° — an order of magnitude larger than the
effect being tested.  They are therefore reported but not used for the verdict.

## 3. Digitisation and conversions

* **Koomen.**  Table 5 of Nawar et al. (2020) was read from the CC-BY PDF at
  250 dpi and transcribed cell by cell; the text layer of that PDF is OCR'd
  and was not trusted.  The cd ft⁻² column was converted with
  `L = 10.7639 × I[cd ft⁻²]` cd m⁻².  The source's own magnitude column was
  checked against `m = −2.5 log₁₀ I + 9.9846`: maximum residual **0.0006 mag**
  over the six rows.  The primary paper could not be read (paywalled), so the
  numbers rest on the CC-BY reproduction; the abstract was used to confirm the
  instrument, both sites, their altitudes and the Maryland/Sacramento Peak
  ratio.
* **Nawar.**  Tables 1–3 were read the same way.  Every row was checked against
  `m = −2.5 log₁₀ S₁₀ + 27.78`: maximum residual **0.055 mag** (one Abu Simbel
  blue row, where the source itself rounds 9.855 to 9.8).  Luminance from
  `L = 1.08 × 10⁵ × 10^(−0.4 m)` cd m⁻², the same Johnson-V relation used by
  `check5`, so the two references share one zero point.

Both datasets are in `validation/data/twilight_koomen1952.csv` and
`validation/data/twilight_nawar2020.csv`; `validation/check6_second_dataset.py`
reproduces every number below.

## 4. Comparison

### 4.1 The headline table

Zenith luminance [cd m⁻²].  Model: sea level, 300 DU O₃; *clear* =
AOD₅₅₀ 0.05, *pristine* = Rayleigh + O₃ only.  Deterministic
(`sky_radiance_total`) at −3° and −6°, spherical engine
(`twilight_radiance`, `quality="low"`) at −9° and −12°.

| h | Koomen 1952<br>(Maryland, 30 m) | Patat 2006<br>(Paranal, 2400 m) | model<br>clear | model<br>pristine | **model / Koomen** | model / Patat |
|---:|---:|---:|---:|---:|---:|---:|
| 0° | 161.5 | — | 149 | 154 | **0.92** | — |
| −3° | 21.53 | 40.09 † | 37.53 | 41.15 | **1.74** | 0.94 |
| −6° | 0.6458 | 0.5164 | 0.7362 | 0.7849 | **1.14** | 1.43 |
| −9° | 0.01615 | 0.01712 | 0.0210 ± 0.0056 | — | **1.30** [0.95–1.64] | 1.22 |
| −12° | 0.001292 | 0.001460 | 0.00430 ± 0.00255 | — | **3.33** [1.35–5.31] | 2.95 |
| −15° | 0.000431 | 0.000320 | 1.5 × 10⁻⁵ ‡ | — | 0.04 | 0.05 |

† extrapolation of the Patat fit outside its stated 95° ≤ ζ ≤ 105° range.
‡ airglow-dominated; the model has no emission component.  Neither the −3° nor
the −15° row is used for the verdict.
Brackets are the 1σ range propagated from the engine's own statistical error.

**At −6° the model is 1.14 × the second dataset and 1.43 × Patat.**  The
1.5–1.6 × excess reported in `docs/ROADMAP.md` is reproduced only against
Patat, and only in its pristine-atmosphere form (1.43 with `clear`, 1.52 with
`pristine`).

### 4.2 The two references differ by 0.24 mag — and that is the altitude

At −6°:

* Koomen (sea level): 0.6458 cd m⁻² = 13.04 mag arcsec⁻²;
* Patat (2400 m): 0.5164 cd m⁻² = 13.30 mag arcsec⁻²;
* ratio 1.251, i.e. Patat is **0.243 mag** fainter (0.26 mag if one
  compares the source magnitude columns instead of the luminances; the 0.02 mag
  is zero-point bookkeeping, not sky).

This is not a discrepancy between two measurements of the same sky; it is the
altitude effect, and both papers measure it themselves:

| pair | pressure ratio | measured ΔV | pressure prediction<br>−2.5 log₁₀(p₁/p₂) |
|---|---:|---:|---:|
| CrAO 600 m (961 hPa) → Paranal 2400 m (743 hPa) *(Patat et al. §4)* | 1.293 | **0.27 ± 0.03 mag** | 0.28 mag |
| Maryland 30 m (~1009 hPa) → Sacramento Peak 2800 m (~719 hPa) *(Koomen et al., abstract)* | 1.40 | **0.44–0.75 mag** | 0.37 mag |
| Maryland 30 m → Paranal 2400 m (this work) | 1.36 | **0.24 mag** | 0.34 mag |

The Maryland–Paranal difference (0.24 mag) is therefore within 0.1 mag of the
0.34 mag that the pressure ratio predicts, and of the 0.27 mag that Patat et
al. measured between their own two sites.  **The reference is not wrong; it is a 2400 m reference.**

Applying that measured site factor to the model's sea-level value gives
0.7362 / 1.251 = 0.588 cd m⁻², i.e. 1.14 × Patat — exactly the same residual as
the direct sea-level comparison.  The 1.43 decomposes as
**1.14 (model vs sea level) × 1.25 (sea level vs Paranal)**.

### 4.3 The deep twilight

At −9° the engine gives 0.0210 ± 0.0056 against Koomen's 0.01615 (1.30 ×,
1σ 0.95–1.64) and Patat's 0.01712 (1.22 ×).  The model is high by 0.1–0.3 mag,
but the engine's `low` budget is 27 % per point.  At −12° the `low` run gives
0.0043 ± 0.0026, i.e. 3.3 × Koomen with a 1σ lower bound of 1.35 ×; the
higher-budget production values quoted in `docs/TWILIGHT.md` §2.2
(0.0238 ± 0.0043 at −9°, 0.00212 ± 0.00128 at −12°) give 1.47 × and 1.64 ×
against Koomen.  **The second dataset is consistent with a 1.4–1.6 × excess
below −9°, but at `quality="low"` it cannot confirm it at more than ~1σ.**

### 4.4 The two points that do not fit

**−3°.**  Koomen gives 21.53 cd m⁻², Patat's fit 40.09 (extrapolated) and the
model 37.53.  The model is 1.74 × Koomen and 0.94 × Patat: the two references
disagree by 0.66 mag here, and the model sits with Patat.  Koomen's own
`0° → −3°` slope (0.97 mag deg⁻¹) is much shallower than its
`−3° → −6°` slope (1.27 mag deg⁻¹), and the 3° value is quoted to one
significant figure, so it may simply be a weak point.  It does not affect the
−6° question, but it is the one place where the second dataset and the model
disagree by more than 0.5 mag, and it is not resolved here.

**0°.**  The model gives 149 cd m⁻² against Koomen's 161.5 (0.92 ×).  The
Patat fit is not defined there (its extrapolation, 8 × 10³ cd m⁻², is
meaningless), so only Koomen can be used.

### 4.5 The Egyptian data (secondary)

Yellow filter (5500 Å), luminance [cd m⁻²]:

| h | Helwan (115 m) | Daraw (250 m) | Abu Simbel (210 m) | model | model/Daraw |
|---:|---:|---:|---:|---:|---:|
| 0° | 105.1 | 114.8 | — | 149 | 1.29 |
| −3° | 16.65 | 13.23 | 5.94 | 37.53 | 2.84 |
| −6° | 0.761 | 0.551 | 0.179 | 0.736 | 1.34 |
| −9° | 0.0167 | 0.0145 | 0.00727 | 0.0210 | 1.44 |
| −12° | 0.00115 | 0.00096 | 0.000985 | 0.00430 | 4.49 |

The three Egyptian sites span 1.5 mag at −6° (Helwan 0.76, Daraw 0.55,
Abu Simbel 0.18 cd m⁻²).  Helwan (a suburb of Cairo) and Daraw bracket the
model and Patat, Abu Simbel is 3 × fainter than the model.  A spread this
large, in data converted from S₁₀ units through a non-Johnson filter, cannot
settle a 0.15 mag question; it is shown here as an honest measure of how large
the systematic uncertainty of the older absolute twilight photometry is
(≈0.5 mag), not as evidence for or against the model.

## 5. The model's altitude response — the real remaining problem

Running the same model at Paranal's altitude
(`atmosphere("clear").at_altitude(2.4)`, surface pressure 756 hPa) gives
**0.734 cd m⁻² at −6°, i.e. a 0.3 % change** (the sea-level value is 1.003 ×
the 2.4 km value).  The measurements require 1.25–1.39 × for the same
pressure ratio.

So the model reproduces the sea-level sky but not the *altitude dependence* of
the twilight.  Physically this is expected: at −6° the zenith line of sight is
already in shadow below ~35 km, so the first-order term comes from the
stratosphere and is almost independent of what happens below 2.4 km; the
observed pressure dependence is carried by multiple scattering in the low
atmosphere, which the plane-parallel solver under-represents.  This is the same
physics as open question #2's "shape of the discrepancy", and it is the part of
the problem the second dataset does *not* solve.

## 6. Verdict

**The 1.5 × excess at −6° is largely a Patat-specific (site) effect, not a
model calibration error.**

* Like-for-like, at sea level: model / Koomen = **1.14** (0.06 mag).  The
  model's absolute twilight level at −6° is good to ~15 %.
* Patat's Paranal sky is **0.24 mag darker** than Koomen's Maryland sky, as
  expected from the 1013 → 743 hPa pressure ratio; the model/Patat ratio is
  therefore 1.14 × 1.25 = **1.43** (`clear`) or 1.52 (`pristine`), reproducing
  the number in the roadmap.
* The second dataset does **not** show the model 1.5 × high; it sits between
  the model and Patat, and its position is quantitatively explained by the
  altitude difference between the two sites.
* What the second dataset *does* expose is a genuine model deficiency: the
  model has essentially **no altitude response** (0.3 %) where the data give
  1.25–1.39 ×.  The model is right at sea level and wrong at 2400 m, for a
  reason unrelated to the absolute calibration.
* Below −9° the second dataset is consistent with the previously reported
  1.4–1.6 × excess, but at `quality="low"` it cannot confirm it beyond ~1σ.

The practical consequence for the project: `check5` should compare the model
with Patat at **Paranal's pressure**, or state explicitly that the comparison
is sea-level-model vs 2400 m-site; the ~0.3 mag offset is not a model error at
−6°.  The open question that remains is why the model's multiple scattering
does not scale with the low-atmosphere column.

## 7. What would settle the rest

* a **measured** twilight profile at a high-altitude site with a stated
  pressure (Paranal, Mauna Kea, Dome C) so that the model can be run at the
  same pressure rather than corrected by a scaling law;
* a re-analysis of the model's plane-parallel multiple scattering against a
  spherical Monte Carlo at −6° with the *same* atmosphere at two pressures —
  the engine exists and would say directly whether the 1.003 × is physical or
  a defect of the plane-parallel solver;
* the original Koomen et al. tables (all degrees, both stations), which would
  give a two-pressure dataset from a single instrument — the cleanest possible
  test of the altitude response.
