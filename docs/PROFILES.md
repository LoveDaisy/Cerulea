# Vertical structure and the twilight excess at −6°

> **Superseded by the M10 chord fix.**  The "factor 0.66" this document tries
> to explain was the one-branch solar chord, not a real excess.  After the fix
> the model is 0.69× Koomen (deterministic) / 0.85× (spherical engine) at −6°,
> so there is no 0.66 to fill.  The sensitivity results below are retained as a
> record: they correctly show that vertical structure is not a lever, and the
> statement "the −6° zenith is lit by 35–60 km and dimmed by the shadowed
> troposphere" is still the right physical picture.

**Verdict: no.**  No plausible change to the vertical structure of the
atmosphere reduces the −6° zenith luminance by the factor **0.66** that would
be needed to reach Patat et al. (2006).  The ozone profile *shape* moves it by
at most 6 % — and a higher ozone layer makes it **brighter**, the wrong
direction.  The ground albedo moves it by 2 %, the phase-function truncation by
exactly zero.  The only lever that reaches 0.66 is the temperature above 20 km,
and it needs a **−26 K** offset over the whole 20–50 km layer: a stratopause at
244 K instead of 271 K, with the density at 35–60 km cut by 17–43 %.  No
standard atmosphere and no observed mid-latitude or tropical sounding looks
like that.  In the spherical engine — the production model at −6° — not even
that works: a third of the signal is multiple scattering that a stratospheric
temperature offset does not touch, so a −26 K offset leaves the engine at
1.5 × Patat (§4.2).

The discrepancy is therefore not a vertical-structure problem.  What the sweep
*does* show is where the sensitivity really lives (§5): the −6° zenith is lit
by air at **35–60 km** and dimmed by the **shadowed troposphere below 20 km**;
because those two contributions have opposite signs, changing the *column*
does almost nothing while changing the *profile* can move the radiance by tens
of per cent.  The vertical structure is a strong lever — it is just not set to
the value that would fix the discrepancy.

This document reports `experiments/12_profile_sensitivity.py`
(`figures/12_profile_sensitivity.png`,
`experiments/out/12_profile_sensitivity.json`).  The variants are implemented
in `experiments/profile_variants.py`; **nothing in `skycolor/` was modified.**

---

## 1. What was varied, and what was held fixed

Every case is the roadmap's baseline atmosphere —
`Atmosphere(wl, ozone_du=300, aerosols=[])`, sea level, ground albedo 0.1,
geometric shadow (`refraction=False`) — with exactly one vertical profile
replaced.  The ozone column is renormalised to 300 DU in every variant, and the
temperature variants keep the surface pressure anchored, so the *total* air
column above 20 km is conserved (it is fixed by `p(20)/g`; the Rayleigh optical
depth at 550 nm is 0.09682 in every temperature case, to 5 digits).

| # | lever | values | implementation |
|---|---|---|---|
| 1a | ozone peak altitude | 20, 22, 25, 28, 32 km | rigid height shift of the climatological table (`shifted_shape`) |
| 1b | ozone width | Gaussian σ = 3, 6, 9 km, centred 25 km | `gaussian_shape` |
| 1c | tropospheric ozone | 0, 30, 60 DU below 10 km | `tropospheric_shape` (remainder above 10 km) |
| 2 | temperature above 20 km | ΔT = −30 … +30 K (requested: ±10, ±20) | `OffsetTemperatureAtmosphere`: `T → T + ΔT` above 20 km, pressure re-integrated hydrostatically |
| 3a | ground albedo | 0.0, 0.1, 0.3 | `ground_albedo` |
| 3b | phase-function truncation | `n_moments` = 16, 32, 48 | SOS solver option |

The reference profile peaks at 22 km (the table is flat 22–25 km; the median
height is 24.6 km) and carries 15.5 DU below 10 km.  The Gaussian variants have
no tropospheric ozone at σ = 3 km, 1.8 DU at σ = 6 and 13.5 DU at σ = 9.  The
"peak altitude" shifts are chosen so that the realised maximum lands on the
nominal value (`peak − 22` km); the table in
`experiments/out/12_profile_sensitivity.json` records the realised peak, median
and tropospheric column for every case.

All luminances are photopic integrals (`683 ∫ L_λ ȳ dλ`) on the 5 nm grid,
matching the numbers quoted in `docs/ROADMAP.md`.  The same cases solved on the
1 nm grid differ by 0.50 % (reference) and 0.48 % (ΔT = −20 K), so the 5 nm
grid does not distort any *ratio* in this document.

## 2. The reference point

| quantity | value |
|---|---|
| L(−3°) | 41.15 cd m⁻² (Patat 40.09, ×1.03 — but the −3° reference is an extrapolation) |
| L(−4°) | 13.90 cd m⁻² (Patat fit 8.46, ×1.64 — the reference is an extrapolation of the fit) |
| L(−6°) | 0.7849 cd m⁻² (Patat 0.516, **×1.52**) |
| L(−8°) | 0.01725 cd m⁻² (Patat fit 0.0480, ×0.36 — **the deterministic model is not valid here**, see below) |
| single-scattering share at −6° | 84 % in the deterministic model; **67 % in the spherical engine** (0.659 of 0.985) |

The engine's total at −6° is `0.659 (single) + 0.326 (multiple) = 0.985` at
`quality="low"`; `quality="medium"` gives 1.035, i.e. 5 % higher, which is the
size of the documented smoothing bias.  Take the engine's −6° value as
**1.0 ± 0.05 cd m⁻²**, or **1.9–2.0 × Patat**.

The last row matters for everything that follows.  The deterministic model's
plane-parallel multiple scattering is only 16 % of the total at −6° and is
known to be ~30 % low there (`docs/TWILIGHT.md` §2.1); the spherical engine
gives 0.985 cd m⁻² at −6° (×1.91 of Patat) with 33 % multiple scattering.  The
sweep below uses the deterministic solver because that is the model the
roadmap's 0.785 refers to and because it is fast (~0.3 s per case).  The one
temperature case that matters was re-run with the spherical engine (§4.2),
which required a run-time patch — the engine's Monte Carlo silently discards
subclass profile overrides (§4.2, item ii).

## 3. Results

Zenith luminance at −6° (the requested table; −3/−4/−8 in the JSON):

| lever | variant | L(−6°) [cd m⁻²] | × reference | × Patat |
|---|---|---:|---:|---:|
| — | **reference** | **0.78492** | **1.000** | **1.521** |
| 1a | peak 20 km | 0.78082 | 0.995 | 1.513 |
| 1a | peak 22 km (= reference) | 0.78492 | 1.000 | 1.521 |
| 1a | peak 25 km | 0.79546 | 1.013 | 1.540 |
| 1a | peak 28 km | 0.80941 | 1.031 | 1.567 |
| 1a | peak 32 km | 0.83154 | **1.059** | 1.610 |
| 1b | σ = 3 km | 0.77151 | 0.983 | 1.494 |
| 1b | σ = 6 km | 0.77108 | **0.982** | 1.493 |
| 1b | σ = 9 km | 0.78163 | 0.996 | 1.513 |
| 1c | 0 DU below 10 km | 0.77918 | 0.993 | 1.509 |
| 1c | 30 DU below 10 km | 0.79070 | 1.007 | 1.531 |
| 1c | 60 DU below 10 km | 0.80382 | 1.024 | 1.556 |
| 2 | ΔT = −30 K | 0.48181 | **0.614** | 0.933 |
| 2 | ΔT = −20 K | 0.57477 | 0.732 | 1.113 |
| 2 | ΔT = −10 K | 0.67591 | 0.861 | 1.309 |
| 2 | ΔT = 0 | 0.78492 | 1.000 | 1.521 |
| 2 | ΔT = +10 K | 0.90148 | 1.148 | 1.746 |
| 2 | ΔT = +20 K | 1.02522 | 1.306 | 1.985 |
| 2 | ΔT = +30 K | 1.15577 | 1.472 | 2.238 |
| 3a | albedo 0.0 | 0.77762 | 0.991 | 1.506 |
| 3a | albedo 0.3 | 0.79994 | 1.019 | 1.549 |
| 3b | `n_moments` = 16 | 0.78492 | 1.000 | 1.521 |
| 3b | `n_moments` = 48 | 0.78492 | 1.000 | 1.521 |

The required factor is **0.516 / 0.785 = 0.657**.

| lever | range swept | ×L(−6°) | reaches 0.657? |
|---|---|---|---|
| ozone peak altitude | 20–32 km | 0.995 – 1.059 | no (wrong sign) |
| ozone width | σ 3–9 km | 0.982 – 0.996 | no |
| tropospheric ozone | 0–60 DU | 0.993 – 1.024 | no |
| temperature, plausible | ±10 K | 0.861 – 1.148 | no |
| temperature, extreme | ±20 K | 0.732 – 1.306 | no |
| ground albedo | 0.0–0.3 | 0.991 – 1.019 | no |
| phase moments | 16–48 | 1.000 | no |
| **temperature, required** | **−26.3 K** | **0.657** | **yes — and implausible (§4.2)** |

## 4. The levers one by one

### 4.1 Ozone: the shape is irrelevant at the 5 % level

Every ozone variant stays within −1.8 % / +5.9 % of the reference, and the
sign is systematically the wrong one: raising the layer **brightens** the
twilight.  This is not an accident of the particular profiles.

At −6° the zenith line of sight is lit by air at 35–60 km, i.e. *above* the
ozone layer.  The vertical view path therefore crosses essentially the whole
300 DU whatever the shape, and the only shape dependence is (a) the part of the
column above the scattering altitude, and (b) the slant path of the solar beam,
which reaches the scattering point after grazing down to ≈5 km (impact
parameter `r sin 96°`) and therefore crosses the layer twice.  Those two terms
partly cancel and leave a few per cent.  Since the *column* is fixed, no
redistribution of it can produce a 34 % dimming: that would need *more* ozone,
not the same amount moved.  The column itself was already ruled out by the
roadmap (with the spherical engine: 250 DU → 2.15×, 400 DU → 1.78×); with the
deterministic baseline used here the same sweep gives 0.842 cd m⁻² (1.63×) at
250 DU and 0.684 (1.33×) at 400 DU, so +100 DU buys only 0.19 in the ratio.
The column is a real but far too weak lever; the shape is a weaker one still.

The Gaussian-width sweep is the cleanest demonstration: σ = 3 km (a
monochromatic layer with no tropospheric ozone at all) and σ = 9 km (13.5 DU
below 10 km) differ by 1.4 %.  **Ozone vertical structure is dead as an
explanation.**

### 4.2 Temperature above 20 km: strong, right shape — and implausible

This is the only lever with real authority, and the only one whose *elevation
dependence* has the right shape:

| ΔT above 20 km | L(−3°) ×ref | L(−4°) ×ref | L(−6°) ×ref |
|---|---:|---:|---:|
| −20 K | 0.984 | 0.949 | **0.732** |
| −10 K | 0.992 | 0.974 | 0.861 |
| +10 K | 1.008 | 1.025 | 1.148 |
| +20 K | 1.015 | 1.050 | 1.306 |

A perturbation that moves −3° by 1.5 % moves −6° by 30 %.  That is exactly the
kind of elevation dependence the discrepancy needs: the model is right at −3°
and 1.5× high at −6°.

The mechanism is the stratospheric scale height.  The offset is applied above
20 km with the pressure re-integrated hydrostatically, so the column above
20 km is exactly conserved (∫ρ dz = p(20)/g, independent of T).  The
temperature therefore redistributes the same air: for ΔT = −26.3 K the density
is unchanged at 20 km, ×0.92 at 30 km, ×0.83 at 35 km, ×0.76 at 40 km, ×0.65 at
50 km and ×0.57 at 60 km.  Since the −6° radiance is emitted at 35–60 km, it
falls.

**What it would take.**  Interpolating the sweep and refining gives
**ΔT = −26.3 K** (L(−6°) = 0.5150, i.e. 0.998 × Patat).  At that offset the
stratopause (US 1976: 270.65 K at 47 km) sits at 244 K and the 35–60 km density
is 17–43 % below the standard atmosphere.  This is not a site mismatch or a
seasonal variation: the mid-latitude stratopause varies by roughly ±10 K, and
−26 K is 2.6 times that envelope — below anything a tropical or mid-latitude
climatology produces.  Paranal (24° S) in particular sits under the tropical
stratopause, among the most stable in the atmosphere.

**The temperature lever can produce the number, but only with an atmosphere
that does not exist.**

**Robustness.**  (i) The step at 20 km creates a density discontinuity of
ΔT/T ≈ 8 %; a smooth linear ramp from 20 to 30 km (which keeps the density
continuous) still gives 0.794 at ΔT = −20 K instead of 0.732, so ~60 % of the
effect survives — it is the scale height, not the jump.

(ii) **The spherical engine cannot be fixed this way at all.**  *(Historical
account — the defect described here is fixed; see the "engine, current" note
at the end of this item.)*  The engine's Monte Carlo term did not see the
variants: `skycolor.rt.twilight._sub_atmosphere` used to rebuild a *plain*
`Atmosphere` from `ozone_du` / `surface_pressure` before each
single-wavelength run, silently discarding any custom profile.  The evidence
was unambiguous — `twilight_radiance(..., single=False)` returned
**bit-identical** multiple scattering, 0.3262779066, for ΔT = 0 and
ΔT = −20 K — and the defect mattered beyond this experiment: *any* profile
study run through the engine was a hybrid of a corrected first-order term and
an uncorrected multiple-scattering term.  `skycolor/` was read-only here, so
`experiments/profile_variants.py` installed a run-time patch
(`install_twilight_profile_support`) that routed `_sub_atmosphere` through the
variant's `_restrict` method; the upstream fix is the same one line.
With the patch, ΔT = −20 K takes the engine's multiple scattering from 0.326
to 0.337 (within its 5 % statistical error, nominally *up*, because the thinner
upper atmosphere attenuates the twilight arch less), while the first-order
term falls from 0.659 to 0.483.  The engine total goes 0.985 → 0.820 (×0.83),
and the −26 K offset that fixes the deterministic model leaves the engine at
≈0.77 cd m⁻² (first order 0.433 + the same 0.34 of multiple scattering), still
**1.5 × Patat**.  Since two thirds of the engine's −6°
signal is first order, driving the engine to the Patat level by temperature
alone would need far more than −40 K.  The deterministic −26 K is a *lower bound*
on what the engine would need, and the engine never gets there.

**Engine, current (architecture refactor Step 3b).**  The rebuild fallback and
the run-time patch are both retired: custom profiles are now injected as data
(``Atmosphere(ozone_profile=..., air_profile=...)``) and the engine reads them
through its native `_restrict` wavelength subsets — no patching, no hybrid
term (`docs/ARCHITECTURE.md` migration map).  `experiments/12` runs the
engine check patch-free: reference L(−6°) 0.554 vs ΔT = −20 K 0.458, the
multiple-scattering terms distinct (0.153 vs 0.165).

(iii) A −26 K offset does not even fit the *shape* of the Patat fit across the
measured range: it fixes −6° (and, coincidentally, the extrapolated −3°), but
at −5° it is still 1.46 × Patat and at −7° it is 0.53 ×.  It crosses the
reference at one point; it does not reproduce it.

### 4.3 Ground albedo: 2 %

Albedo 0 → 0.1 → 0.3 changes L(−6°) from 0.7776 to 0.7849 to 0.7999.  At −6°
the ground is in the Earth's shadow, so it emits nothing itself; it only
reflects the (still sunlit) sky above it, and that reflected light is a small
part of the diffuse field at the zenith.  Even a snow-white ground (0.6) moves the
value by only +5 %.  Not a candidate.

### 4.4 Phase-function truncation: exactly zero

`n_moments` = 16, 32 and 48 give **bit-identical** radiances (maximum relative
difference over the spectrum: 0.0).  This is not a numerical coincidence: with
no aerosol the only nonzero Legendre moments are Rayleigh's `l = 0, 1, 2`, so
`delta_m_moments` returns `f = β_N = 0` and the same three moments for every
`N ≥ 3`.  The 32-moment truncation is exact for a pristine atmosphere.  The
roadmap's suspicion is confirmed, and no further work on this is warranted
unless a stratospheric aerosol layer is added.

## 5. Why the profile matters at all: the contribution function

The logarithmic response `d ln L / d ln n_air` of the −6° zenith to a density
change in each altitude band (central difference, ±5 %; `BandDensityAtmosphere`
in `experiments/profile_variants.py`):

| band [km] | d ln L / d ln n | band [km] | d ln L / d ln n |
|---|---:|---|---:|
| 0–10 | **−0.425** | 35–40 | +0.154 |
| 10–20 | **−0.325** | 40–45 | +0.183 |
| 20–25 | −0.051 | 45–50 | +0.219 |
| 25–30 | −0.024 | 50–60 | **+0.277** |
| 30–35 | −0.006 | 60–80 | +0.152 |
| | | 80–120 | +0.009 |

The two signs are the whole story.  Below 20 km the air is **in shadow**: it
contributes nothing but extinction, so more of it means less light (the sum
over 0–20 km is −0.75).  Above 35 km the air is **sunlit** and is the source
(the sum over 35–80 km is +0.99).  A *uniform* change of the density — which is
what changing the surface pressure, the site altitude or the total column does
— moves both terms together and almost cancels: a 1 % uniform increase changes
L by only +0.16 %.  This is why the column levers in the roadmap were all weak,
and it is *not* evidence that vertical structure is weak: a change that acts on
one side only (like the temperature above 20 km, or a scene that removes the
troposphere) is a strong lever.

The ozone result has the same origin, though its sign is a residual of two
competing slant-path terms rather than a single one: the beam's path through a
layer that sits higher is longer, the vertical view path through it is shorter,
and at −6° the second wins by a few per cent.  The magnitude — never more than
6 % — is what matters; the sign is not worth building an argument on.

## 6. What would discriminate the remaining candidates

After this sweep the candidates are:

1. **A colder stratosphere** — the only vertical-structure change that can
   move −6° by 1.5×; requires −26 K (deterministic) or more (engine);
2. **Something grey in the comparison** — the V-band-to-photopic conversion
   (`docs/ROADMAP.md` open question 4), or the zero point of the Patat fit /
   the extrapolated −3° point.

The single measurement that discriminates them is a **calibration-free relative
twilight decay**, i.e. the ratio **L(−3°)/L(−6°)** measured with one instrument
on one night:

| | L(−3°)/L(−6°) |
|---|---:|
| model, US76 | 52.4 |
| model, ΔT = −26.3 K | 78.2 |
| Patat et al. (2006) fit | 77.6 |

No absolute calibration and no V-band conversion enters a ratio, so every grey
error cancels; a temperature change moves it by up to 50 % (ozone by less than
6 %).  If the measured ratio is ≈52 the model's *decay* is right and the 1.5×
is a level error (candidate 2).  If it is ≈78 the decay is wrong — and no
vertical profile fixes it: the temperature lever crosses the reference at −6°
but misses −5° and −7° (§4.2), and the spherical engine cannot be moved that
far at all.  In that case the decay error points at the radiative transfer or
the reference, not at the atmosphere's vertical structure.

An absolute spectrum at the same elevation would then split the remaining
level error into its two pieces: photopic integral versus Johnson-V surface
brightness, which is exactly the 10–20 % systematic of `docs/ROADMAP.md` open
question 4 and is currently unquantified.

In practice candidate 1 can be checked against existing data *now*: the
temperature at 20–50 km is not an unknown — it is measured globally several
times a day by satellite sounders (MLS, SABER) and by radiosondes reaching
10 hPa, and no mid-latitude or tropical climatology puts the stratopause 26 K
below US 1976.  Candidate 1 is already excluded; the open question is
candidate 2, and it needs the ratio, not a new atmospheric model.

## 7. Caveats

* The sweep uses the deterministic solver at −4° and −8°, where it is not
  reliable.  −4° is an extrapolation of the Patat fit (valid −5° to −15°) and
  −8° is 2.8× too dark because the plane-parallel multiple scattering has
  collapsed (`docs/TWILIGHT.md` §2.1).  Only the −6° column is a like-for-like
  comparison; the other elevations are there to show the *shape* of each lever.
* The −3° reference (40.09) is itself an extrapolation of the Patat fit.  The
  model's 3 % agreement there is therefore not evidence that the model is
  right at −3°; it is the reason §6 uses the ratio rather than the level.
* The temperature variants perturb the whole 20–50 km layer uniformly.  A
  perturbation localised where the contribution function is largest (35–60 km)
  would need a comparable magnitude, not a smaller one, because the uniform
  offset is the most efficient way to change the density over the whole
  emitting layer.
* The spherical-engine temperature check needs the run-time patch described in
  §4.2(ii).  Without it the engine's multiple scattering is that of US 1976
  whatever profile is passed in, so engine-based profile studies are invalid
  until `skycolor/rt/twilight.py::_sub_atmosphere` is fixed upstream.
* The ozone Gaussian variants are *not* realistic profiles (a σ = 9 km
  Gaussian puts 13.5 DU below 10 km and has no sharp tropopause).  They are
  there to bracket the shape, not to model a sounding.

## 8. Reproduction

```bash
uv sync
uv run python experiments/12_profile_sensitivity.py            # ~45 s
uv run python experiments/12_profile_sensitivity.py --engine   # + ~1.5 min
```

The script prints the tables above, writes
`experiments/out/12_profile_sensitivity.json` (all cases, all elevations,
profile diagnostics, the contribution function and the ramp/grid checks), and
regenerates `figures/12_profile_sensitivity.png`.  `--engine` adds the
spherical-engine spot check of §4.2(ii) (patched, `quality="low"`, two runs).  The JSON is
under the repository's ignored `experiments/out/` directory, like every other
experiment output; the figure and this document are the committed record.
