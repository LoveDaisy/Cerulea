# A second twilight dataset: Koomen et al. (1952)

> Historical production comparisons below predate the AOD550 anchor-optics correction. Current source/transport experiments and primary-reference contracts are in [CURVED_TRANSPORT_AUDIT.md](../docs/CURVED_TRANSPORT_AUDIT.md) and [TWILIGHT_REFERENCE_AUDIT.md](../docs/TWILIGHT_REFERENCE_AUDIT.md).

*The second dataset asked for by open question #2 of `docs/ROADMAP.md`.*

**Short answer.** The −6° level is no longer a factor-of-two problem.  After
the M10 chord fix the sea-level model is **0.69×** Koomen et al. (1952)
(0.44 vs 0.646 cd m⁻²) and **0.86×** Patat et al. (2006); the spherical
engine gives **0.85×** Koomen and **1.07×** Patat.  The old 1.14× (geometric)
and 2.20× (refracted) numbers were artifacts of a solar path that returned
only one perigee → space traversal, dropping the near-side perigee → point
segment of every grazing twilight ray.  The remaining difference is not causally identified; the paper does not
provide the absolute error budget needed to assign it to calibration.

The altitude interpretation was revised by the subsequent
[audit](../docs/TWILIGHT_ALTITUDE_AUDIT.md): Step 3c fixed truncation, but
solar visibility still let rays cross the raised opaque ground.  Correcting
that changes the deterministic -6° high/low ratio **1.208 → 1.027**;
full-spectrum direct MC gives dimming instead.  The cross-study ratio 0.80
is not a controlled pure-altitude target.  Koomen's original two-station
table is now available and gives 0.367 at -6°, with different seasons and
measured atmospheric transparency.  See §5 for what is and is not resolved.

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
353–356 (1952), DOI [10.1364/JOSA.42.000353](https://doi.org/10.1364/JOSA.42.000353).

The original [article scan](https://static1.squarespace.com/static/54694fa6e4b0eaec4530f99d/t/5e876886cdfbaa441f538327/1585932427823/Measurements%2Bof%2Bthe%2BBrightness%2Bof%2Bthe%2BTwilight%2BSky%2B1952.pdf)
is now available (pp. 353–354 were inspected in the altitude audit):

* a **green-filtered photomultiplier** intended to approximate photopic
  response.  Photopic luminance is a closer comparison with
  `validation.common.luminance` than a Johnson V magnitude, but an exact
  instrumental response match is not established;
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
  over the six rows.  Those six existing reference rows still retain their
  original Nawar reproduction provenance.  The later original-paper audit
  independently confirms the Maryland -6° zenith cell and adds selected
  two-station zenith pairs in `data/twilight_altitude_audit/`; it does not
  silently replace this validation baseline.
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
one-branch chord (M10); the residual in those runs is not independently assigned to calibration
or to missing scattering.

### 4.2 The two references differ by 0.24 mag; the cause is not isolated

At -6° the Maryland photopic value is 0.6458 cd m⁻² and the Paranal
V-derived value is 0.5164 cd m⁻², a ratio of 0.800 or 0.243 mag.
This mixes site, epoch, atmospheric transparency and spectral response.
It cannot establish an altitude-only transfer function.

[Patat et al. §4–5](https://arxiv.org/html/astro-ph/0604128) compare CrAO
(600 m, 961 hPa, three December 2002 twilights) with Paranal (743 hPa,
a different instrument and sample).  Their mean difference of 0.27±0.03 mag
at solar depression 5.5–11° corresponds to a flux ratio of 0.780±0.022.
Its agreement with the pressure ratio 743/961=0.773 motivates a density
hypothesis; it does not exclude aerosol or other station differences.

Koomen's original Tables I/II provide the same-instrument zenith values:
Sacramento Peak 2800 m / Maryland 30 m = 0.022/0.060 = **0.367** at -6°.
The high and low stations were observed in May–June and January–March 1951,
respectively; reported vertical solar transmission was 85–90% versus 75–85%.
The station signal is real in these tables; its cause remains mixed.
Machine-readable cells, source coordinates and conditions are in
`data/twilight_altitude_audit/koomen1952_zenith_pairs.*`.

A common photometric zero point cancels within a same-band ratio.  A
photopic-to-Johnson-V conversion need not cancel when spectra differ.
No numerical correction for that difference is inferred here.

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

## 5. The altitude audit: a fixed shadow defect and a residual comparison

The architecture refactor removed double pressure scaling and made the
transport bottom agree across single/SOS/MC.  Its reported 1.208 deterministic
and 1.007 low-budget engine altitude ratios were still affected by a separate
solar-visibility defect: rays with perigees between sea level and the raised
bottom were treated as lit, even though they intersected the opaque ground.
The refracted shadow table had the same sea-level boundary error.

Correcting all solar masks and the refracted grazing threshold changes the
-6° deterministic high-site result from 0.536728 to **0.456350 cd m⁻²**;
sea level remains **0.444279**, so the ratio is now **1.027**.  The single
term changes 0.411475→0.391830 and the SOS term 0.125253→0.064520 at altitude.
Thus the old conclusion that this was necessarily a physical deficit, and
that the shadow boundary could not affect zenith twilight, was incorrect.

Eight independent-seed, full 81-wavelength direct MC runs bypass the engine's
smoothing and spectral interpolation.  Their refracted-single + geometric-MS
high/low ratio is **0.957±0.007** (empirical standard error of the mean ratio);
the fully geometric split gives **0.920±0.009**.  The higher-order term itself
falls to **0.744** of its sea-level value.  These results show that the
spherical calculation does dim under this particular truncation, whereas
the SOS approximation retains a small opposite net response.

This is not a validation against a universal 0.80 law.  Truncation raises a
global opaque sphere; an actual mountain with surrounding lowland and a
cross-site change in atmospheric profiles are different interventions.
Nor is the engine fully refracted: its higher orders and solar next-event
estimates remain geometric.  The full evidence, raw runs, uncertainty
calculation and remaining hypotheses are in
[the audit](../docs/TWILIGHT_ALTITUDE_AUDIT.md).

## 6. Verdict

- M10's two-branch chord correction stands.  Sea-level -6° ratios remain
  0.69 to Koomen (deterministic), about 0.85–0.87 with the engine, and
  0.86 / about 1.07–1.09 to Patat.  These alone do not isolate the absolute
  calibration or site contribution to the residual.
- The raised-ground solar-occultation defect is fixed independently of any
  observational fit.  The old 1.208 altitude ratio is superseded by 1.027;
  independent spherical direct MC shows dimming in this standard scene.
- The 0.80 cross-study ratio, 0.78 Patat station ratio and 0.367 Koomen station
  ratio represent different comparisons.  None is a controlled altitude-only
  target for the same US76 medium.
- The historical medium-quality deep-twilight deficits remain observations
  of a heavy-tailed estimator, not proof of a missing specific component.
- `check5` is single-scattering only; use the full solver/engine results when
  discussing absolute twilight levels.

## 7. What would settle the rest

- Match a **specific station**, its pressure, aerosol/profile information and
  photometric response, rather than applying an empirical altitude factor.
- Quantify photopic versus Johnson-V synthetic photometry from the same spectra.
- Isolate refracted solar illumination from diffuse-path bending in the MC
  before attributing residuals to a missing scattering mechanism.
- For deeper twilight, use independent seeds and direct angular/spectral
  samples to distinguish sampling variance from interpolation bias.
