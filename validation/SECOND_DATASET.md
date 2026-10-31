# A second twilight dataset: Koomen et al. (1952)

*The second dataset asked for by open question #2 of `docs/ROADMAP.md`.*

**Short answer.** The −6° level is no longer a factor-of-two problem.  After
the M10 chord fix the sea-level model is **0.69×** Koomen et al. (1952)
(0.44 vs 0.646 cd m⁻²) and **0.86×** Patat et al. (2006); the spherical
engine gives **0.85×** Koomen and **1.07×** Patat.  The old 1.14× (geometric)
and 2.20× (refracted) numbers were artifacts of a solar path that returned
only one perigee → space traversal, dropping the near-side perigee → point
segment of every grazing twilight ray.  The remaining 15–30 % is inside the
1950s absolute calibration and the plane-parallel multiple-scattering
approximation.

What *is* still a model problem is the altitude response, and the chord fix
changed its sign: under the consistent (truncated) `at_altitude` construction
the model is **1.21× brighter** at 2.4 km (deterministic; 1.01× engine),
while Koomen's Maryland (30 m) and Patat's Paranal (2400 m) skies, and
Patat's own 961 vs 743 hPa pair, are **0.80× darker**.  The old construction
(double pressure scaling, multiple-scattering ground left at sea level) was
replaced in the architecture refactor (Step 3c); the remaining wrong sign is
a physics gap, not a construction artifact.  See §5 below and
`validation/ADDENDUM.md` (sixth addendum).

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
reproduces the deterministic and `quality="low"` engine numbers below.  The
medium-quality engine values come from `tools/twilight_benchmark.py`.

## 4. Comparison

### 4.1 The headline table

Zenith luminance [cd m⁻²].  Model: sea level, 300 DU O₃; *clear* =
AOD₅₅₀ 0.05, *pristine* = Rayleigh + O₃ only.  Deterministic
(`sky_radiance_total`) at −3° and −6°, spherical engine
(`twilight_radiance`, `quality="medium"`) at −9° and −12°; the faster
`quality="low"` check6 run gives 0.0124 ± 0.0043 and 0.0030 ± 0.0026.

| h | Koomen 1952<br>(Maryland, 30 m) | Patat 2006<br>(Paranal, 2400 m) | model<br>clear | model<br>pristine | **model / Koomen** | model / Patat |
|---:|---:|---:|---:|---:|---:|---:|
| 0° | 161.5 | — | 157 | — | **0.97** | — |
| −3° | 21.53 | 40.09 † | 20.1 | 22.3 | **0.93** | 0.50 |
| −6° | 0.6458 | 0.5164 | 0.444 | 0.498 | **0.69** | 0.86 |
| −9° | 0.01615 | 0.01712 | 9.31 × 10⁻³ ± 1.91 × 10⁻³ | — | **0.58** [0.46–0.70] | 0.54 |
| −12° | 0.001292 | 0.001460 | 5.46 × 10⁻⁴ ± 3.16 × 10⁻⁴ | — | **0.42** [0.18–0.67] | 0.37 |
| −15° | 0.000431 | 0.000320 | 1.5 × 10⁻⁵ ‡ | — | 0.04 | 0.05 |

† extrapolation of the Patat fit outside its stated 95° ≤ ζ ≤ 105° range.
‡ airglow-dominated; the model has no emission component.  Neither the −3° nor
the −15° row is used for the verdict.
Brackets are the 1σ range propagated from the engine's own statistical error.

**At −6° the model is 0.69× Koomen (deterministic) or 0.85× (spherical
engine, low budget; 0.87× at medium), and 0.86× Patat (deterministic) or
1.07× (engine; 1.09× at medium).**  The old 1.14× and 2.20× were the
one-branch chord (M10); the residual 15–30 % is inside the 1950s absolute
calibration and the plane-parallel multiple-scattering approximation.

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
0.444 / 1.251 = 0.355 cd m⁻², i.e. 0.69 × Patat — the same residual as the
direct sea-level comparison.  The raw model/Patat ratio 0.86 decomposes as
**0.69 (model vs sea level) × 1.25 (sea level vs Paranal)**.  This is a
diagnostic, not a prediction: the model's own altitude response has the wrong
sign (see §5), so the scaled value is not what the model would compute at
2400 m.

### 4.3 The deep twilight

At −9° the medium-quality benchmark gives 9.31 × 10⁻³ ± 1.91 × 10⁻³ against
Koomen's 0.01615 (0.58 ×, 1σ 0.46–0.70) and Patat's 0.01712 (0.54 ×, 1σ
0.43–0.66).  The low-quality `check6` run gives 0.0124 ± 0.0043 (0.77 ×),
consistent within 1σ.  At −12° the medium run gives 5.46 × 10⁻⁴ ± 3.16 × 10⁻⁴,
0.42 × Koomen (1σ 0.18–0.67) and 0.37 × Patat; the 58 % error makes this
non-conclusive.  **The chord fix removed the old 1.4–1.6 × excess and revealed
a deficit that grows with depth; at −9° it is ~2σ.**

### 4.4 The two points that do not fit

**−3°.**  Koomen gives 21.53 cd m⁻², Patat's fit 40.09 (extrapolated) and the
model 20.1.  The model is 0.93 × Koomen and 0.50 × Patat: the two references
disagree by 0.66 mag here, and the patched model now sits with Koomen.
Koomen's own `0° → −3°` slope (0.97 mag deg⁻¹) is much shallower than its
`−3° → −6°` slope (1.27 mag deg⁻¹), and the 3° value is quoted to one
significant figure, so it may simply be a weak point.  It does not affect the
−6° question, but it is the one place where the second dataset and the model
disagree by more than 0.5 mag, and it is not resolved here.

**0°.**  The model gives 157 cd m⁻² against Koomen's 161.5 (0.97 ×).  The
Patat fit is not defined there (its extrapolation, 8 × 10³ cd m⁻², is
meaningless), so only Koomen can be used.

### 4.5 The Egyptian data (secondary)

Yellow filter (5500 Å), luminance [cd m⁻²]:

| h | Helwan (115 m) | Daraw (250 m) | Abu Simbel (210 m) | model | model/Daraw |
|---:|---:|---:|---:|---:|---:|
| 0° | 105.1 | 114.8 | — | 157 | 1.37 |
| −3° | 16.65 | 13.23 | 5.94 | 20.1 | 1.52 |
| −6° | 0.761 | 0.551 | 0.179 | 0.444 | 0.81 |
| −9° | 0.0167 | 0.0145 | 0.00727 | 0.0124 | 0.86 |
| −12° | 0.00115 | 0.00096 | 0.000985 | 0.00303 | 3.16 |

The three Egyptian sites span 1.5 mag at −6° (Helwan 0.76, Daraw 0.55,
Abu Simbel 0.18 cd m⁻²).  Helwan (a suburb of Cairo) and Daraw bracket the
model and Patat, Abu Simbel is 3 × fainter than the model.  A spread this
large, in data converted from S₁₀ units through a non-Johnson filter, cannot
settle a 0.15 mag question; it is shown here as an honest measure of how large
the systematic uncertainty of the older absolute twilight photometry is
(≈0.5 mag), not as evidence for or against the model.

## 5. The altitude response — still the real remaining problem

After the chord fix the sea-level value at −6° is 0.444 cd m⁻².  With the
consistent (truncated) `at_altitude` construction — the layer grid starts at
the site, the ground sits there, nothing is renormalised (architecture
refactor Step 3c) — `at_altitude(2.4)` gives 0.537, i.e. the deterministic
model is **1.21× brighter** at altitude (spherical engine **1.01×**).  The
data are the other way round: Koomen's Maryland (30 m) sky is 1.25× brighter
than Patat's Paranal (2400 m), and Patat's own 961 → 743 hPa pair gives
1.28×.  The model's altitude response therefore still has the **wrong sign**.

Two things have been separated.  The old numbers (1.16× deterministic,
0.98× engine) came from a double-scaled, ground-inconsistent construction
(`p(z0)·p(z0)/P0`, observer raised but the multiple-scattering ground still
at sea level); that construction is gone.  What remains is the same
low-atmosphere multiple scattering: at sea level the engine adds 0.11 cd m⁻²
of diffuse light that the plane-parallel solver misses, and at 2.4 km that
low layer is gone.  This is the physical gap the second dataset exposes; the
chord fix does not close it, and with the construction now consistent there
is no remaining audit to do — the altitude comparison still should not be
used as a calibration constraint.  Open item: the refraction shadow
thresholds still assume sea-level grazing rays, which matters for
horizon/low-altitude viewing but not for the zenith points here.

## 6. Verdict

**The −6° level is no longer a factor-of-two problem; the altitude response
is.**

* At sea level: model / Koomen = **0.69** (deterministic) or **0.85**
  (spherical engine).  The model is 15–30 % low, inside the 1950s calibration.
* Against Patat: 0.86 (deterministic) or 1.07 (engine).  The old 1.14/2.20
  were the one-branch chord.
* At −9° the engine is 0.58 × Koomen (0.54 × Patat, medium; 1σ ~0.46–0.70);
  the old 1.4–1.6 × excess is gone, replaced by a ~1.7 × deficit at ~2σ.
* The model's altitude response has the wrong sign: under the consistent
  (truncated) `at_altitude` construction it is 1.21× / 1.01× (deterministic /
  engine) brighter at 2.4 km where the data are 0.80× darker.  The pressure
  double-scaling and the ground mismatch are fixed (Step 3c); the
  low-atmosphere multiple scattering is the remaining physical candidate.
* The practical consequence: `check5` is single-scattering only and now reads
  0.70× Patat at −6°; compare the full model/engine, not the single-scattering
  harness, when discussing the absolute level.

## 7. What would settle the rest

* an **audit of `at_altitude`**: fix the `p(z0)/P0` scaling and re-run the
  two-site comparison; then compare the engine (spherical MS) at 0 and 2.4 km
  against Koomen/Patat;
* a **measured** twilight profile at a high-altitude site with a stated
  pressure (Paranal, Mauna Kea, Dome C) so that the model can be run at the
  same pressure rather than corrected by a scaling law;
* the original Koomen et al. tables (all degrees, both stations), which would
  give a two-pressure dataset from a single instrument — the cleanest possible
  test of the altitude response.
