# The solar source: one authoritative definition

Every radiative-transfer backend consumes the same sun. This document is the
contract for :mod:`skycolor.solar_source` -- the module that owns the solar
direction, the continuous column attenuation, the ground occultation, the
finite solar disk and the angular-domain gain -- and for what each backend
still approximates on top of it. Unifying the source does **not** unify the
diffuse-transport approximation of each backend; those remain and are
declared in each module's ``CAPABILITIES``.

## The five layers

1. **`SolarRayField`** -- the refractive index profile `n(r)` of one density
   definition (Gladstone-Dale of the medium's air density; identically 1 for
   the `refraction=False` contract) and the radial integration machinery.
   Rays are integrated with Bouguer's invariant `a = n(r0) r0 cos(e_app)` in
   radius; the integration segments coincide with the *spline knots* of the
   tabulated profile, so no segment ever crosses a knot. Rays that hit the
   opaque ground (`r_bounds[0]`) return NaN -- the two-branch `up + 2*down`
   chord physics of the M10 repair is preserved, now with continuous
   densities instead of shell chords. Monotone `n·r` only: ducting media
   raise `ValueError` at construction.
2. **`SolarSpeciesColumns`** -- continuous air/O3/NO2/aerosol columns along
   the same rays (no shell chords, no shell averages). Wavelength
   independent: extinction is applied by the caller from its own cross
   sections, so wavelength slices and reorders never change the geometry.
3. **`PointSourceTable`** -- a shadow-aligned PCHIP table of (apparent
   direction, species columns) versus `(height, true elevation)`. Rows are
   stacked in `sqrt(height - ground)`; inside a row the knots are refined
   until every interval carries angle, slope and optical-depth certificates.
   The **beam gain is the derivative of the same direction interpolant** --
   a second interpolated gain field is never built. The slope certificate
   compares the interpolant derivative against the field's *analytic*
   derivative (the same quadrature, differentiated with the moving-limit
   terms); near the horizon kink (|apparent| < 0.05 deg) the slope
   certificate is exempt -- the analytic derivative's endpoint singularity
   is unresolvable there, and the gain is carried by the interpolant, whose
   horizon accuracy the experimental audit validated at 0.5 deg spacing.
   Built lazily once per medium construction (~10 s at the default density)
   and shared by every backend.
4. **`SolarGeometry`** -- the token-cached bundle of 1-3. Identity is the
   medium's construction token (the same rule as the refraction tables):
   wavelength subsets share one geometry; `at_altitude` and any other
   redefinition build their own. The cache holds no strong reference to the
   medium, so dead media do not pin their geometry.
5. **`SolarSource`** -- binds a geometry to an observing scenario and one
   true sun direction, adds the finite disk (uniform cap, radius 16', disk
   weights with `sum(w cos rho) = 1` under the centre-normal-plane `F_inf`
   normalisation) and exposes the consumer kernels: `vertex` (per-point
   direction/columns/T/gain), `scatter` (the angular integral
   `sum_beams T*G*p_mix` with 4-pi-normalised mixtures and the single
   `n^2` observer transfer already applied), `ground` (horizontal
   incidence), `direct_irradiance` (flux on an arbitrary plane -- the DNI
   seam) and `central_transmittance` (pure centre-beam attenuation for the
   legacy transmittance APIs, direct evaluation, no table build).

## Physical bookkeeping

* `G_local = (n_point/n_inf)^2 * dOmega_app/dOmega_true` concentrates the
  true-solid-angle disk irradiance `F_inf` into local flux; partial
  occultation removes beams without renormalising the rest.
* Along the retained straight diffuse paths the reduced radiance `L/n^2` is
  what is conserved; scattering and ground reflection happen at one local
  `n`, so the source-to-detector conversion collapses to the single factor
  contained in `G_local * (n_observer/n_point)^2`. Kernels apply it once;
  nothing in the chain multiplies a second `n^2`. A constant-index medium
  recovers `G = 1` exactly.
* `T = exp(-tau)` is always the pure attenuation of the physical columns,
  reported separately from gain; `T*G` is never named transmittance and
  never clipped.

## What each backend consumes, and what it still approximates

| backend | solar side | retained approximation |
|---|---|---|
| `rt.single` | per-scattering-point `scatter` kernel; phase from the *apparent* beam direction | straight viewing rays; view-path optical depth from the shell quadrature with the aerosol `HorizontalField` applied to the view-side coefficients only (solar columns stay radial) |
| `rt.multiple` (SOS) | per-disk-beam tables on the pseudo-spherical column; beams projected onto the Fourier source with their local azimuth factors (the disk spans only ~16'/sin(z) of local azimuth, so `cos(m d_az) ~ 1`); sun `T` is the raw physical extinction -- delta-M scales only the diffuse moments | plane-parallel diffuse transport; ground fixed at `z_bounds[0]` |
| `rt.mc`, `rt.twilight` | deferred batched `scatter`/`ground` evaluation; the RNG stream, free paths and event classification are bit-identical to a run with the solar values blanked | straight-line photon transport; horizontally uniform atmosphere |
| `rt.cloud` | per-beam classification (top entry / base entry / mid-deck perigee / blocked), per-beam two-stream responses, `geometry="mixed"` when the disk straddles the boundary | plane-parallel homogeneous deck; the per-beam generalisation of the historical midpoint closure for perigee-inside-deck beams |
| `render.ground_radiance` | `direct_irradiance` on the horizontal at the observer's ground | display-only; the retired `direct_transmittance` override still works but warns and cannot be combined with the new override |

The Monte Carlo signability note (plan 6.3): the deterministic-vs-MC
first-order closure passes `|mean-SS| <= 4SE + 1%|SS|` at all validation
points; deep-twilight points (-6 deg and below) need more photons than the
plan's budget cap for the `SE/|SS| <= 5%` *signing* threshold, and are
reported as statistically not signable at that budget rather than passed
silently -- see `validation/production_solar_source.py` (suite ``closure``).

## Validation and reproducibility

```bash
uv run python -m validation.production_solar_source --suite source  --out scratchpad/task-production-solar-source/source.json
uv run python -m validation.production_solar_source --suite closure --out scratchpad/task-production-solar-source/closure.json
uv run python -m tools.benchmark_solar_source --out scratchpad/task-production-solar-source/perf-final.json
```

Both suites exit non-zero on a missed threshold and write a JSON record with
the git head and per-check numbers. Source-level evidence:

* constant-index limits: direction unbent to 1e-10 deg, `G = 1` to 1e-10;
* continuous columns against the archived Cartesian-RK anchors
  (`validation/data/solar_source_audit/geometry-probe.json`): max |dT/T|
  1.3e-4 (budget 1e-3), max angle error 2.7e-4 deg (budget 2e-4; the worst
  single anchor is documented at 2.7e-4);
* table holdout: shadow disagreements 0, |T| error <= 5e-4 (budget 2e-3),
  |T*G| <= 0.9% (budget 1%);
* wavelength slices share one geometry; a reordered grid is bit-identical;
  a no-550 nm grid changes only the value-free column-conditioning scale
  (columns agree to ~1e-4, within the tau certificates).

## Known deviations and open items

* The plan's 2e-4 deg direction budget is met at the RK anchors to 2.7e-4
  and by the table holdout to 1.4e-3 at one point class (mid-height,
  mid-elevation); the physical impact is negligible and the full acceptance
  matrix re-measures it.
* Rays whose turning point lands within ~100 m of a profile-formula
  discontinuity (e.g. the 11 km US76 lapse break) have a local slope
  sensitivity larger than the smooth-field budget; the DOP853 ray-tube
  arbitration of plan 6.2 is the referee for those points and is part of
  the follow-up acceptance matrix.
* Availability gate (plan 6.5) measured on the reference machine: source
  cold build ~9.7 s (budget 10 s), warm single 3wl single-direction
  0.008 s (budget 1 s), MC 3wl x 20k 3.0x the legacy run (budget 4x),
  peak RSS +0.11 GiB (budget +0.5 GiB); the standard clear dome runs at
  ~6x the legacy wall time (budget 3x) because the per-beam contract costs
  `n_beams` table queries per scattering point -- see the escalation
  candidates in the task record.

Historical evidence for the unified source's physics (independent direction,
columns, etendue and first-order closure experiments) lives in
`docs/twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md` and
`docs/twilight/SOLAR_SOURCE_AUDIT.md`; those experiments predate
productionisation and are kept unchanged.
