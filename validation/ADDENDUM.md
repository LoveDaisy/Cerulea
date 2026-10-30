# Addendum — the numbers after multiple scattering was integrated

The main report (`REPORT.md`) was written while `skycolor.rt.multiple` was still
a stub, so every number in it is **single scattering only**.  Multiple
scattering has since been merged.  The harness now defaults to the full model;
set `SKYCOLOR_MULTIPLE=0` to reproduce the single-scattering column.

| check | single scattering only | full model (single + multiple) |
|---|---|---|
| 1 Rayleigh optical depth | PASS (−0.31 %) | PASS (unchanged) |
| 2 DNI illuminance | PASS (+4.0 / +4.9 / +5.7 %) | PASS (unchanged: the direct beam does not involve multiple scattering) |
| 2 DHI illuminance, h = 30 / 60° | +5.6 / +10.2 % | **+37.6 / +42.3 %** → outside the 25 % tolerance |
| 3 CIE relative luminance, RMS θ ≤ 60° | 3.8 % | 14.5 % |
| 3 CIE horizon sun-side / zenith, h = 45° | 7.36 vs 3.65 | 8.43 vs 3.65 |
| 3 CIE horizon anti-solar / zenith, h = 45° | 2.09 vs 2.00 | 3.47 vs 2.00 |
| 4 chromaticity vs measured Berlin spectra | PASS (RMS Δxy = 0.019) | PASS (unchanged — the horizon chromaticity moves by < 0.003) |
| 5 twilight zenith, h = −6° | 0.567 vs 0.516 cd m⁻² (+10 %) | 0.567 vs 0.516 (+10 %) |
| 5 twilight zenith, h = −9° | 7.2e-4 vs 1.7e-2 (24× low) | 7.3e-4 vs 1.7e-2 (23× low) |
| 5 twilight zenith, h = −12° | 0 | 7.5e-7 vs 1.5e-3 |

## What the multiple scattering changed, and whether it is an improvement

**Absolute luminance: yes.**  The clear-sky zenith luminance rises by 25 %
(1.57 → 1.97 kcd m⁻² at h = 45°), which is the direction and the magnitude that
the CIE comparison asked for.  The independent Monte Carlo solver
(`skycolor/rt/mc.py`) confirms MS/SS = 31 % at the zenith for this atmosphere,
so the added light is not an artefact.

**Diffuse horizontal illuminance: the reference is the weaker link.**  The
CIE 110 T = 2.2 diffuse reference is itself a smooth model output, not a
measurement.  Converted at 110 lm W⁻¹ the model's 9.5 klx at h = 30° is
86 W m⁻², which is a normal measured clear-sky diffuse irradiance for
AOD ≈ 0.05.  The single-scattering value matched the reference by being low in
the same way.  We therefore do not treat the +38 % as a model failure, but the
tolerance should not be read as a validation of the diffuse level either.

**The limb: worse, and for an understood reason.**  The multiple-scattering
solver transports the diffuse field in a *plane-parallel* atmosphere.  At a
viewing zenith angle of 88° that geometry implies an airmass of 28.6 while the
true spherical value is about 19, so the limb's diffuse radiance is
over-estimated by roughly 50 %.  This is the documented limitation of the
SOS solver, and it is the reason the horizon/zenith ratio moves *away* from
the CIE value rather than towards it.  The upper sky (θ ≤ 60°) is affected
much less; its 14.5 % RMS is dominated by the single point at the sun's
position, where the model's Henyey-Greenstein aureole has no forward peak.

**Deep twilight: unchanged, and still the largest failure.**  Multiple
scattering cannot fix it, because the solver is horizontally homogeneous: the
light that lights the zenith at −9° has to travel several hundred kilometres
horizontally from the sunlit twilight arch, and a plane-parallel model has no
mechanism for that.  A spherical multiple-scattering solver would be needed.
Civil twilight down to about −6° is reproduced to 10 %; below −8° the model is
not trustworthy.

## Corrected defects

* `skycolor/spectra.rayleigh_cross_section` docstring said σ(550) = 1.135e-31
  m²; the code returns 4.507e-31 m² (correct).  Fixed in the docstring.
* The hemispherical irradiance integrator in `skycolor/render.py` was missing
  the cos θ factor (2.4× too large).  Found while reconciling check 2 with the
  analytic single-scattering estimate, fixed, and pinned by
  `tests/test_render.py`.
* The perspective camera's pitch had the wrong sign (`skycolor/render.py`).

---

# Second addendum — after the aerosol library, refraction and the twilight engine

> **Pre-M10.**  The engine values in this section predate the two-branch
> chord fix; the corrected ones are in the sixth addendum.  The 1.39/1.45
> rows in the summary table are historical.

Three further pieces have landed since the first addendum.  The validation
harness numbers move slightly but the verdicts do not:

| check | before this round | now |
|---|---|---|
| 1 Rayleigh | PASS | PASS (unchanged) |
| 2 DNI | PASS | PASS (+4.9 / +5.4 %) |
| 2 DHI | +38 % | +41 / +46 % (same reference caveat) |
| 3 CIE relative luminance, θ ≤ 60° | 14.5 % | 10.6 % RMS |
| 3 horizon sun-side / zenith | 8.43 | 8.83 (vs CIE 3.65) |
| 4 measured horizon chromaticity | Δxy = 0.0193 | **Δxy = 0.0184** |
| 5 twilight, −6° | 0.567 vs 0.516 | 0.568 vs 0.516 |
| 5 twilight, −9° | 7.3e-4 (24× low) | 7.4e-4 in the deterministic model, **0.0238 with the twilight engine (1.39× of Patat)** |
| 5 twilight, −12° | 7.5e-7 | **2.1e-3 with the engine (1.45× of Patat)** |

## The deep twilight is fixed

> **Pre-M10 values.**  The engine numbers in this section predate the
> two-branch chord fix; the corrected ones are in the sixth addendum
> (0.563 cd m⁻² at −6°, 9.3 × 10⁻³ at −9°, 5.5 × 10⁻⁴ at −12°).

`skycolor/rt/twilight.py` (spherical Monte Carlo, split estimator: exact
deterministic first order + Monte Carlo for orders ≥ 2) closes the largest gap
in the model.  Against Patat et al. (2006):

| solar elevation | deterministic | engine | Patat | engine / Patat |
|---|---|---|---|---|
| −3° | 37.6 | 37.8 | 40.1 | 0.94 |
| −6° | 0.737 | 0.978 | 0.516 | 1.89 |
| −9° | 9.6e-4 | 0.0238 | 0.0171 | 1.39 |
| −12° | 9.8e-7 | 2.1e-3 | 0.00146 | 1.45 |

The −6° overshoot is not a transport error: the direct Monte Carlo gives the
same value, and it is a model-atmosphere/site mismatch (sea-level standard
atmosphere versus the 2400 m Paranal site).  The check-5 harness still uses
the deterministic model, so it still reports a failure at −9° and below; that
is a deliberate choice to keep the harness comparing like with like, and the
engine's numbers are in `docs/TWILIGHT.md` §2.2.

## The aerosol library

`skycolor/aerosol.py` now has eight Mie-backed members with analytic Legendre
moments (no angular quadrature) and delta-M scaling.  The moments are exact to
2e-8 against an independent quadrature, `beta_1` matches the Bohren–Huffman
`g` to 1e-10, and the phase-function normalisation is within 2.2e-4 for every
member.  This fixed a real bug: the old fixed angular grid gave `g = 3.57` for
an 8 µm droplet (the physical value is 0.861).

| member | α₅₅₀ | SSA₅₅₀ | g₅₅₀ | τ(450)/τ(650) | delta-M |
|---|---|---|---|---|---|
| rural | 1.30 | 0.982 | 0.714 | 1.62 | no |
| urban | 1.25 | 0.901 | 0.704 | 1.59 | no |
| maritime | 0.55 | 1.000 | 0.813 | 1.23 | recommended |
| desert_dust | −0.04 | 0.876 | 0.816 | 0.984 | required |
| biomass_burning | 1.84 | 0.865 | 0.586 | 1.97 | no |
| volcanic_stratospheric | −0.31 | 1.000 | 0.710 | 0.892 | optional |
| cirrus | −0.04 | 0.9998 | 0.868 | 0.986 | required |
| water_cloud | −0.04 | 1.000 | 0.858 | 0.985 | required |

## Clouds

`skycolor/cloud.py` + `skycolor/rt/cloud.py` add a plane-parallel, homogeneous
cloud deck solved with delta-Eddington, validated against an independent Monte
Carlo: albedo within 0.1–0.7 % for a conservative `g = 0` layer and within 1 %
for `g = 0.85` at τ ≥ 5; `R + T + direct = 1` to 1e-12; the overcast
horizon/zenith ratio is exactly 1/3; τ → 0 reduces bit-for-bit to the clear
sky.  The brief's "albedo 0.7 at τ = 10" holds only for `g = 0.6`; for real
droplets (`g = 0.859`) it needs τ = 30 (`docs/CLOUD.md`).

---

# Third addendum — full refraction and delta-M

## Refraction now bends the optical path

`refraction.py` traces rays through the refractive atmosphere, and the
radiative transfer now uses the full geometry: the apparent solar elevation at
every scattering point is looked up from `data/refraction.csv`, the Bouguer
invariant `a = n r sin(theta_app)` is formed, and the per-shell path length is
`sqrt(r^2 - (a/n_k)^2)` differences.  Checking that closed form against
numerical path integration: **1659 km vs 1663 km** for a horizontal ray at the
surface (0.1 %).

Measured effect: twilight becomes **2–5× brighter**, because the terminator
rises by 0.5–1.0°.

| solar elev. | geometric shadow | full refraction | Patat 2006 |
|---|---|---|---|
| −3° | 41.2 | 57.6 | 40.1 |
| −6° | 0.79 | 1.69 | 0.516 |
| −9° | 1.1e-3 | 5.3e-3 | 0.0171 |

The geometric model matches the reference at −3° (1.03×); the refracted one is
1.4× high there.  `Atmosphere(refraction=True)` is therefore available but not
the default, and the **solar-altitude convention of the reference** (true or
apparent) is now the dominant open question — see `docs/ROADMAP.md` #1.

## delta-M is applied to the multiple-scattering solver

`atmosphere.delta_m_properties()` returns optical depth and moments rescaled by
the Wiscombe transformation, and `rt/multiple.py` now uses it by default.
Effect at 550 nm, zenith view, sun at 60°:

| atmosphere | MS without delta-M | MS with delta-M |
|---|---|---|
| clear, AOD 0.1 (fine) | 0.00654 | 0.00654 (f = 0, unchanged) |
| dusty, AOD 1.0 | 0.06003 | **0.02218** |

The dusty case moves by a factor 2.7 because 35 % of the scattered energy sits
in the forward peak that the 32-moment truncation was discarding.  Fine
aerosols are unaffected, which is the expected sanity check.

A performance caveat: a dusty atmosphere now takes ~100 s for one
multiple-scattering solve.  That is documented in the roadmap as the
performance item.

---

# Fourth addendum — the twilight excess was a site effect

## A second dataset

`validation/check6_second_dataset.py` and `validation/SECOND_DATASET.md` add
**Koomen et al. (1952)**, JOSA 42, 353: a *photopic* recording photometer at
Maryland, altitude 30 m, zenith, solar altitude +5° to −15°.  Photopic is the
same weighting the model computes, so this comparison avoids the V-band
conversion entirely.

| h | Koomen (30 m) | Patat (2400 m) | model | model/Koomen | model/Patat |
|---|---|---|---|---|---|
| −3° | 21.53 | 40.09 † | 37.53 | 1.74 | 0.94 |
| −6° | 0.6458 | 0.5164 | 0.7362 | **1.14** | 1.43 |
| −9° | 0.01615 | 0.01712 | 0.0210 ± 0.0056 | 1.30 [0.95–1.64] | 1.22 |
| −12° | 0.001292 | 0.001460 | 0.00430 ± 0.00255 | 3.33 [1.35–5.31] | 2.95 |

† outside the fitted range of the Patat fit.  Model: `clear` preset (300 DU,
AOD 0.05, sea level); `pristine` gives 1.52 vs Patat at −6°.

## What it settles

At −6° the model is **1.14×** a sea-level measurement, not 1.5×.  Koomen is
1.25× (0.24 mag) brighter than Patat, and that is the altitude of Paranal:
Patat et al. measure ΔV = 0.27 ± 0.03 mag between 961 and 743 hPa themselves,
and Koomen's second station at 2800 m shows the same effect.  So

```
1.14 (model vs sea level)  ×  1.25 (sea level vs Paranal)  =  1.43
```

which is exactly the old model/Patat ratio.  The "1.5× twilight excess" was a
comparison of a sea-level model with a 2400 m reference.

The −3° row is the one genuine tension: the two references disagree by
0.66 mag there, and Patat's −3° is an extrapolation.  It is excluded from the
verdict.

## The defect the exercise found

Run at 2.4 km the model changes by **0.3 %**, while the data demand
1.25–1.39× for the same pressure ratio.  The model is right at sea level and
wrong at altitude: its twilight does not scale with the low-atmosphere column.
That implies a non-negligible part of the twilight light is scattered *in the
shadowed troposphere*, lit from above by the diffuse field, and the model
produces too little of it.  This is now a milestone-level item in
`docs/ROADMAP.md`.

## Vertical structure is excluded

`docs/PROFILES.md` sweeps the ozone profile, the temperature above 20 km, the
ground albedo and the phase-function truncation.  None of them can supply the
factor 0.66 that the old comparison asked for; the only lever that reaches it
is a −26 K stratopause, which no observed climatology supports.  A useful
side result: the −6° zenith is lit by 35–60 km and attenuated by the shadowed
0–20 km, so *column* changes cancel to ~0.2 % per 1 % while *profile* changes
do not — which is why the earlier column-level tests were so unresponsive.

---

# Fifth addendum — refraction is now the default

`Atmosphere(refraction=True)` is the default from this commit on: the ray
bending is real, the paths are integrated along the bent chord, and the
terminator sits where it belongs.

This changes the twilight numbers quoted earlier in this file.  At 550 nm,
`clear` preset, zenith:

| solar elev. | geometric | with refraction | Koomen 1952 (sea level) | Patat 2006 (2400 m) |
|---|---|---|---|---|
| −3° | 37.5 | 49.4 | 21.5 | 40.1 † |
| −6° | 0.736 | 1.42 | 0.646 | 0.516 |

So the model/Koomen ratio at −6° goes from 1.14 to **2.20**, and model/Patat
from 1.43 to **2.75**.  † Patat's −3° is an extrapolation.

Read this the honest way: a real sky *does* include refraction, so the
refracted comparison is the physical one, and it says the model's twilight is
1.5–2.5× too bright.  The older "1.14×, therefore a site effect" conclusion
was partly a cancellation — a terminator that sat too low compensating a
twilight that was too bright.  Three things could carry the excess and none is
excluded yet:

1. the optical depth of the grazing solar ray at 30–35 km (the newly lit band
   at −6° has a transmittance of only ~0.03, so the answer is sensitive to
   how that chord is integrated);
2. the multiple-scattering level in a shadowed troposphere;
3. the absolute calibration of 1950s photometry (±0.2–0.4 mag is plausible).

The geometric-shadow numbers remain reachable with `refraction=False`, which
is why they are kept in the tables above.

> **Superseded by the sixth addendum.**  The 2.20× / 2.75× above is the
> one-branch chord, not a physical excess.

---

# Sixth addendum — the solar chord was one branch short

The −6° "2.20× too bright" was not a physical excess.  For a scattering point
whose sun is below the local horizontal, the ray to the sun crosses its
perigee: the light traverses the far branch (perigee → space) and the near
branch (perigee → point).  `path_lengths_sun` / `sun_path_lengths` returned
only one perigee → space traversal, stopping at the perigee; the near-side
perigee → point segment was missing.  Every grazing twilight ray was
under-attenuated by half of its dense lower-atmosphere column.

The fix is `up + 2*down` instead of `up + down` for `c < 0` in
`geometry.path_lengths_sun`, `geometry.path_lengths_sun_array` and
`refraction.sun_path_lengths`.  It is verified against a numerical line
integral (geometric: 1.99 before, 1.00 after) and an RK4 ray trace (refracted:
0.50 before, 0.995/1.010/0.984 after).

| h | before (refracted) | after deterministic | after engine | Koomen | Patat |
|---|---|---|---|---|---|
| −3° | 49.4 (2.29 K) | 20.1 (0.93 K) | — | 21.5 | 40.1 |
| −6° | 1.421 (2.20 K / 2.75 P) | 0.444 (0.69 K / 0.86 P) | 0.552 (0.85 K / 1.07 P) | 0.646 | 0.516 |
| −9° | 0.021 (1.30 K) | — | 0.0124 (0.77 K) | 0.0162 | 0.0171 |

The engine column is `quality="low"`; the medium benchmark gives 0.563 at
−6°, 9.31 × 10⁻³ at −9° and 5.46 × 10⁻⁴ at −12° (0.87 / 1.09 at −6°,
0.58 / 0.54 at −9°, 0.42 / 0.37 at −12°, against Koomen / Patat).

The old geometric 1.14× was a partial cancellation: the chord bug made
twilight ~2.5× too bright, the absence of refraction ~1.5× too dark, and the
two errors left the geometric model accidentally close to Koomen.  The
refraction convention is no longer degenerate: H1 (true axis) gives 0.69×
Koomen at −6°, H2 (apparent axis) 0.29×, so the references almost certainly
use the true solar altitude.

What this does *not* fix is the altitude response.  The model is a few per cent
brighter at 2.4 km where the data are 0.80× darker; `at_altitude` also scales
the US1976 profile by `p(z0)/P0`, making the density at the site 0.746× too
low.  That is the next audit.  The lesson: the Monte Carlo and the engine share
`path_lengths_sun`, so their 1 % agreement could not catch this; the
independent check has to be a line integral or a ray trace.
