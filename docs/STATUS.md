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

252 tests.  Environment pinned by `uv.lock` (Python 3.12, numpy 2.5, scipy 1.18).

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
structure); refraction tables cache on the medium's density-definition
identity.  Contracts: `docs/ARCHITECTURE.md`.

## How well it is validated

| test | result |
|---|---|
| Rayleigh optical depth vs Bodhaine (1999) | RMS 0.48 % |
| Direct normal illuminance vs ASTM G173 / Bird | +4 to +6 % |
| CIE 110 relative luminance, upper sky | 10.6 % RMS |
| Horizon chromaticity vs measured spectra | Δxy = 0.018 |
| Twilight vs sea-level photometry (Koomen 1952), −6° | **0.69×** deterministic, **0.85×** engine (low; 0.87× medium) |
| Twilight vs Patat (2006, 2400 m), −6° | **0.86×** deterministic, **1.07×** engine (low; 1.09× medium) |
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
   solar depression (0.58× Koomen at −9°, 0.42× at −12°), which is the same
   low-atmosphere multiple-scattering deficit; the −6° point itself is inside
   the 1950s calibration and the plane-parallel approximation.
2. **The altitude response has the wrong sign.**  `at_altitude` now truncates
   the model at the site (Step 3c): the layer grid starts at z0, the ground sits
   at z0 and nothing is renormalised, so single/SOS/MC share one ground for the
   first time.  Under that consistent construction the zenith luminance ratio
   at −6° is **1.21×** deterministic / **1.01×** engine while the data
   (Koomen's two stations, Patat's two pressures) are **0.80×** darker — the
   remaining gap is the same low-atmosphere multiple-scattering deficit, not a
   construction artifact.  (The old 1.16×/0.98× came from the double-scaled,
   ground-inconsistent construction.)  Open item: the refraction shadow
   thresholds still assume sea-level grazing rays.
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

`README.md` — what it is and how to run it.
`docs/REPORT.md` — the physics and the results, with the figures.
`docs/ROADMAP.md` — milestones M1–M10, open questions, future directions.
`validation/` — the checks, and the addenda recording what each round changed.
`docs/ARCHITECTURE.md` — the module contracts.
