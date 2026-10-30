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
| Aerosols | 8-member Mie library, analytic Legendre moments, RH growth |
| Refraction | ray-traced, refracted optical paths, **on by default** |
| Clouds | one plane-parallel homogeneous deck, delta-Eddington |
| Horizontal heterogeneity | radial aerosol field, first order only |
| Colour | CIE 1931 → Bradford → ACES, tone-mapping options |
| Cameras | panorama, fisheye, perspective; ground; sun disk |

193 tests.  Environment pinned by `uv.lock` (Python 3.12, numpy 2.5, scipy 1.18).

## How well it is validated

| test | result |
|---|---|
| Rayleigh optical depth vs Bodhaine (1999) | RMS 0.48 % |
| Direct normal illuminance vs ASTM G173 / Bird | +4 to +6 % |
| CIE 110 relative luminance, upper sky | 10.6 % RMS |
| Horizon chromaticity vs measured spectra | Δxy = 0.018 |
| Twilight vs sea-level photometry (Koomen 1952), geometric | 1.14× at −6° |
| **Twilight vs the same, refraction on (the default)** | **2.20× at −6°** |
| Deep twilight vs Patat (2006) | 1.39× at −9°, 1.45× at −12° |
| Multiple scattering vs independent Monte Carlo | 1 % |
| Cloud albedo vs independent Monte Carlo | ≤1 % |
| Refraction at the horizon | 33′ (Sændersson 29′, classical 34′) |

## What is known to be wrong, and by how much

1. **The absolute twilight level is 1.5–2.5× too bright** once refraction is
   included.  The earlier "site effect" explanation was partly a cancellation.
   Candidates: the optical depth of the grazing solar ray at 30–35 km, the
   multiple scattering in a shadowed troposphere, the 1950s photometry.
2. **The model has almost no altitude response** — 0.3 % for 2400 m where the
   data demand 25–39 %.
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
`docs/ROADMAP.md` — milestones M1–M9, open questions, future directions.
`validation/` — the checks, and the addenda recording what each round changed.
`docs/ARCHITECTURE.md` — the module contracts.
