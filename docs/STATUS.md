# Status

*One page, current as of the last commit.  Everything else in `docs/` is the
argument; this is the state.*

## What works

| | |
|---|---|
| Spectral sky radiance, 378–780 nm | Rayleigh + Mie + O₃ + NO₂, spherical geometry |
| Single scattering | spherical first order with refracted solar paths; source conventions under audit |
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
| Twilight vs sea-level photometry (Koomen 1952), −6° | **0.69×** production deterministic; unified-source experiment below |
| Sea-level model / Patat (2006, 2400 m), −6° | **0.86×** production deterministic; different sites and observables |
| Deep twilight vs Koomen, −9° / −12° | **1.12× / 0.59×** production engine, one low-budget seed; not converged predictions |
| Multiple scattering vs independent Monte Carlo | 1 % |
| Cloud albedo vs independent Monte Carlo | ≤1 % |
| Refraction at the horizon | 33′ (Sæmundsson 29′, classical 34′) |

## What is known to be wrong, and by how much

1. **Absolute twilight brightness remains unresolved.**  The old
   2.2× was a chord-integration bug: the solar ray was followed only from the
   perigee out to space, not from the perigee back to the scattering point as
   well, so every grazing twilight ray was under-attenuated by half of its
   dense lower-atmosphere column.  Fixed in `geometry.path_lengths_sun` and
   `refraction.sun_path_lengths`; the −6° value moves from 1.42 to 0.44 cd m⁻²
   (deterministic).  The old 0.55 cd m⁻² engine figure predates the AOD550
   slicing correction below.  Historical medium-budget deep-twilight numbers
   likewise used the old anchor optics.  Current low-budget check6 values
   differ substantially; heavy-tailed MC and differing source prescriptions
   prevent assigning a physical cause from a single run or declaring a stable
   deficit.  The −6° unified-source experiment is a separate reference.
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

**Unified-source reference:** direction, continuous solar columns, angular
gain and a finite 2-D disk now feed the same deterministic and MC source.
At −6°, full-spectrum eight-seed SS+MS is **0.49323±0.00308** at sea level and
**0.45977±0.00347 cd m⁻²** at 2.4 km; high/sea **0.93215±0.00748**.  The first
orders agree within 0.23 seed-SE.  Diffuse paths in this source-only experiment
are straight; it has not replaced production defaults.  See
[UNIFIED_SOLAR_SOURCE_AUDIT.md](UNIFIED_SOLAR_SOURCE_AUDIT.md).

**AOD550 slicing defect fixed:** monochromatic Mie queries formerly renormalized
the medium at the query wavelength.  The reference is now always actual 550 nm;
eight-member full81/full401 optics stay bit-identical, while twilight's anchor
optics and one golden change.  Full regression covers 314 passing cases via a
313-pass run plus the independently attributed fixture rerun.  After two source
binding regressions were added, the final full run passes all 316 tests; six
physical checks remain 3 PASS / 3 FAIL.  See
[AEROSOL_GRID_AUDIT.md](AEROSOL_GRID_AUDIT.md).

**Continuous diffuse refraction:** an experimental arc-length/RK backend with
independent Hamiltonian and oblique first-order checks gives sea/high
**0.489±0.008 / 0.467±0.011 cd m⁻²**, high/sea **0.956±0.036** at −6°.
Eight 10k-path seeds do not determine the sign of its correction relative to
the 100k-path straight control.  The source/flight chain is consistent within
its stated checks; station residuals remain unassigned.  A thin-shell Snell
variant was rejected for artificial ducts.  See
[CURVED_TRANSPORT_AUDIT.md](CURVED_TRANSPORT_AUDIT.md).

**Observation boundaries:** Patat's V fit contains night background and is not
identical to photopic luminance.  A stated filter-only synthetic-V calculation
changes the present −6° spectra by only 0.5–0.7%; it is not a full FORS response.
Station ratios do not identify altitude alone, and a better fit cannot determine
the solar-angle convention.  See [TWILIGHT_REFERENCE_AUDIT.md](TWILIGHT_REFERENCE_AUDIT.md).

`README.md` — what it is and how to run it.
`docs/REPORT.md` — the physics and the results, with the figures.
`docs/ROADMAP.md` — completed milestones, open questions, future directions.
`validation/` — the checks, and the addenda recording what each round changed.
`docs/ARCHITECTURE.md` — the module contracts.
`docs/TWILIGHT_ALTITUDE_AUDIT.md` — raised-ground correction and altitude evidence.
