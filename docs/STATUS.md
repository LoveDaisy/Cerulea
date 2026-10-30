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

196 tests.  Environment pinned by `uv.lock` (Python 3.12, numpy 2.5, scipy 1.18).

## How well it is validated

| test | result |
|---|---|
| Rayleigh optical depth vs Bodhaine (1999) | RMS 0.48 % |
| Direct normal illuminance vs ASTM G173 / Bird | +4 to +6 % |
| CIE 110 relative luminance, upper sky | 10.6 % RMS |
| Horizon chromaticity vs measured spectra | Δxy = 0.018 |
| Twilight vs sea-level photometry (Koomen 1952), −6° | **0.69×** deterministic, **0.85×** spherical engine |
| Twilight vs Patat (2006, 2400 m), −6° | **0.86×** deterministic, **1.07×** spherical engine |
| Deep twilight vs Patat (2006), −9° | **0.54×** engine (medium, 1σ 0.43–0.66) |
| Multiple scattering vs independent Monte Carlo | 1 % |
| Cloud albedo vs independent Monte Carlo | ≤1 % |
| Refraction at the horizon | 33′ (Sæmundsson 29′, classical 34′) |

## What is known to be wrong, and by how much

1. **The absolute twilight level is ~15–30 % low, not 1.5–2.5× high.**  The old
   2.2× was a chord-integration bug: the solar ray was followed only from the
   perigee out to space, not from the perigee back to the scattering point as
   well, so every grazing twilight ray was under-attenuated by half of its
   dense lower-atmosphere column.  Fixed in `geometry.path_lengths_sun` and
   `refraction.sun_path_lengths`; the −6° value moves from 1.42 to 0.44 cd m⁻²
   (deterministic) and 0.55 cd m⁻² (spherical engine).  The remaining
   difference is inside the 1950s absolute calibration and the plane-parallel
   multiple-scattering approximation.
2. **The altitude response has the wrong sign.**  At 2.4 km the model is a few
   per cent *brighter*, while the data (Koomen's two stations, Patat's two
   pressures) are 0.80× darker.  The `at_altitude` helper also scales the
   US1976 density profile by `p(z0)/P0`, which makes the density at the site
   0.746× too low at 2.4 km; the altitude comparison needs its own audit
   before it is used as a constraint.
3. **The limb is ≈ 2× brighter than the CIE 110 fit** (which may be a
   reference problem — CIE 110 is a daylighting standard, not physics).
4. **The solar aureole is under-resolved in multiple scattering** (32-moment
   truncation without delta-fit; delta-M is applied now, delta-fit is not).
5. **No polarisation** — the Rayleigh sky is up to 70 % polarised; the scalar
   model cannot show it.
6. **Clouds are one homogeneous deck** — no cumulus, no silver linings, no
   halos (ice = equivalent spheres).
7. **Performance**: a dusty atmosphere takes ~100 s for one
   multiple-scattering solve.

## Where to look

`README.md` — what it is and how to run it.
`docs/REPORT.md` — the physics and the results, with the figures.
`docs/ROADMAP.md` — milestones M1–M10, open questions, future directions.
`validation/` — the checks, and the addenda recording what each round changed.
`docs/ARCHITECTURE.md` — the module contracts.
