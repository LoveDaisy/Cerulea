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
