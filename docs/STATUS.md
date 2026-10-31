# Status

*One page, current as of the last commit.  Everything else in `docs/` is the
argument; this is the state.*

## What works

| | |
|---|---|
| Spectral sky radiance, 378–780 nm | Rayleigh + Mie + O₃ + NO₂, spherical geometry |
| Single scattering | exact, including the curved (refracted) solar ray |
| Multiple scattering | successive orders, delta-M scaled, plane-parallel diffuse field |
| Deep twilight (below −8°) | spherical Monte Carlo engine, split estimator |
| Solar path | two branches (perigee→point and perigee→space), checked against a line integral and a ray trace |
| Aerosols | 8-member Mie library, analytic Legendre moments, RH growth |
| Refraction | ray-traced, refracted optical paths, **on by default** |
| Clouds | one plane-parallel homogeneous deck, delta-Eddington |
| Horizontal heterogeneity | radial aerosol field, first order only |
| Colour | CIE 1931 → Bradford → ACES, tone-mapping options |
| Cameras | panorama, fisheye, perspective; ground; sun disk |

Test suite: `uv run pytest -q` (see that command's output for the current
count; docs do not track absolute numbers).  Environment pinned by `uv.lock`
(Python 3.12, numpy 2.5, scipy 1.18).

## Architecture (post-refactor)

The physical medium is a frozen `Medium` (`skycolor/medium.py`), the observing
scenario a frozen `Scene` (`skycolor/scene.py`); `Atmosphere` is the facade.
Species listing and column algebra are single-sourced in `skycolor/optics.py`
(no backend imports another's privates; the MC's transport/sampling/estimator
stay deliberately independent).  Custom gas profiles are injected callables
(`ozone_profile=` / `air_profile=`); `at_altitude` truncates the model at the
site.  `rt.solver.solve` dispatches explicitly to clear / twilight / cloud and
returns a traceable `SkyResult`; `render` consumes any spectral field through
a `radiance_field` closure (one display path for clear and cloudy skies).
`MieAerosol` / `CloudLayer` are frozen (shared instances read-only by
structure); refraction tables cache on the medium's construction identity,
covering its density profile and opaque bottom radius.  Contracts:
`docs/ARCHITECTURE.md`.

## How well it is validated

| test | result |
|---|---|
| Rayleigh optical depth vs Bodhaine (1999) | RMS 0.48 % |
| Direct normal illuminance vs ASTM G173 / Bird | +4 to +6 % |
| CIE 110 relative luminance, upper sky | 10.6 % RMS |
| Horizon chromaticity vs measured spectra | Δxy = 0.018 |
| Twilight vs sea-level photometry (Koomen 1952), −6° | **0.69×** deterministic, **0.85×** engine (low; 0.87× medium) |
| Sea-level model / Patat (2006, 2400 m), −6° | **0.86×** deterministic, **1.07×** engine (low; 1.09× medium); different site conditions |
| Deep twilight vs Patat (2006), −9° | **0.54×** engine (medium, 1σ 0.43–0.66) |
| Multiple scattering vs independent Monte Carlo | 1 % |
| Cloud albedo vs independent Monte Carlo | ≤1 % |
| Refraction at the horizon | 33′ (Sæmundsson 29′, classical 34′) |

## What is known to be wrong, and by how much

1. **The −6° twilight level is 15–30 % low, not 1.5–2.5× high.**  The old
   2.2× was a chord-integration bug: the solar ray was followed only from the
   perigee out to space, not from the perigee back to the scattering point as
   well, so every grazing twilight ray was under-attenuated by half of its
   dense lower-atmosphere column.  Fixed in `geometry.path_lengths_sun` and
   `refraction.sun_path_lengths`; the −6° value moves from 1.42 to 0.44 cd m⁻²
   (deterministic) and 0.55 cd m⁻² (spherical engine).  The residual grows with
   solar depression in the recorded medium-budget run (0.58× Koomen at −9°,
   0.42× at −12°); deep-twilight MC statistics are heavy-tailed, so neither
   the exact deficit nor its physical cause is established by one run.
2. **The altitude comparison is unresolved; a raised-ground shadow defect is
   fixed.**  M12's site truncation removed double pressure scaling, but solar
   visibility still used the sea-level opaque radius.  M13 fixes geometric
   masks, refracted grazing tests and shadow tables to use the model bottom.
   For clear air, 81 wavelengths, refraction on, sun −6° and zenith view, the
   deterministic high-site/sea-level ratio changes from the historical
   **1.208× to 1.027×** (0.456350/0.444279 cd m⁻²).  The old engine 1.01× is
   also a pre-fix estimate; updated multi-seed results are in
   `docs/TWILIGHT_ALTITUDE_AUDIT.md`.  The often quoted 0.80× combines different
   studies/sites/observables; Koomen's same-instrument stations give about
   0.37× at −6°, with different seasons and transparency.  Neither is a
   controlled altitude-only target.  Missing low-altitude multiple scattering
   is not an established explanation of the remaining difference.
3. **The limb is ≈ 2× brighter than the CIE 110 fit** (which may be a
   reference problem — CIE 110 is a daylighting standard, not physics).
4. **The solar aureole is under-resolved in multiple scattering** (32-moment
   truncation without delta-fit; delta-M is applied now, delta-fit is not).
5. **No polarisation** — the Rayleigh sky is up to 70 % polarised; the scalar
   model cannot show it.
6. **Clouds are one homogeneous deck** — no cumulus, no silver linings, no
   halos (ice = equivalent spheres).
7. **Performance**: the bottleneck is Mie pre-processing, not the
   multiple-scattering solve.  The per-wavelength cache now holds the whole
   81-point working grid (capacity 8 → 256) and the preset aerosols are
   shared across atmospheres, so warm and repeat calls are ≈ 0 (warm single
   clear 1 dir 0.35 → 0.003 s; second `delta_m_properties(32)` 6.8 →
   0.002 s; a second `presets.dusty` build ~41 s → < 1 ms; values
   bit-identical).  Cold dusty Mie pre-processing (~40 s) still needs
   vectorised kernels or a lookup table.  Numbers: `tools/perf_bench.py`.

## Where to look

**Solar-source follow-up (M14):** controlled experiments find a 29% change
in sea-level MC higher orders from solar attenuation/shadow alone.  Independent
angle mapping and beam-angular-amplitude checks also change the first-order
source substantially.  These experimental variants retain straight diffuse
paths and do not replace production defaults; the absolute-level residual
cannot yet be assigned to missing multiple scattering.  See
[SOLAR_SOURCE_AUDIT.md](SOLAR_SOURCE_AUDIT.md).

`README.md` — what it is and how to run it.
`docs/REPORT.md` — the physics and the results, with the figures.
`docs/ROADMAP.md` — completed milestones, open questions, future directions.
`validation/` — the checks, and the addenda recording what each round changed.
`docs/ARCHITECTURE.md` — the module contracts.
`docs/TWILIGHT_ALTITUDE_AUDIT.md` — raised-ground correction and altitude evidence.
