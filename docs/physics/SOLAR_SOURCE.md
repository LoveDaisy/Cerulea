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
   medium, so dead media do not pin their geometry. The shared geometry
   serves exactly one table configuration (the `PointSourceTable`
   production defaults) -- `ensure_table` takes no configuration arguments,
   and custom-density probes build standalone tables so a probe can never
   re-tune later production queries on the same medium (gen3-R1).
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
* Reference-frame authority (which interface returns which quantity):
  the `single`, `mc`, `twilight` and `render` kernels return
  detector-referred quantities carrying the single
  `(n_observer/n_point)^2` transfer above.  Two documented LOCAL
  exceptions keep plain radiance/flux with no detector transfer anywhere:
  the SOS backend (`rt.multiple`) -- its diffuse paths, ground and TOA
  diagnostics included -- because along the straight-transport
  approximation plain radiance -- not `L/n^2` -- is the invariant, and
  reduced bookkeeping there broke energy conservation in check; and the
  cloud deck (`illuminate_cloud` beam incident flux, the two-stream layer
  responses and `cloud_radiance`), which is a local two-stream operator
  driven by local beam flux (`weight * F_inf * T_entry * G_local`).  When
  local cloud radiances are combined with detector-referred single/MC
  radiances, that reference-frame difference stays part of the declared
  propagation approximation; it must not be "reconciled" by adding a
  transfer to one side only.  The ~1e-4-level convention difference
  between SOS and single-scattering radiances is part of the declared
  SOS-vs-SS propagation error (plan 6.3).

## Disk quadrature conventions

* Production scatter disk: a 2x8 centre-normal product disk
  (`sum(w*cos(rho)) = 1`); per-beam lit flags drop occulted beams without
  renormalising the rest.
* Contact band (`|sun elevation - critical_true|` under one disk radius at
  the point): the shadow boundary crosses the cap as an azimuth-dependent
  curve, so flux and scatter kernels switch to the occultation-adaptive
  disk: all azimuth nodes concentrate on the lit arc, each azimuth's radial
  Gauss-Legendre rule covers exactly its lit interval (boundary by
  bisection), and every node carries its true azimuthal measure.  Flux
  kernels use 16x384 (evaluated once per point), scatter 8x64 (hot path);
  measured against 24x768 references across the band, the scatter kernel
  holds <= 0.06% and the horizontal-flux kernel <= 0.003% (plan 6.2 budget
  0.5%).  Outside the band every beam shares one lit state and the coarse
  disk's per-beam lit drop is exact.
* Receiver-plane boundary (flux kernels only, gen3-R1): `ground` and
  `direct_irradiance` share one kernel that also resolves the plane's own
  projection cut -- when the apparent disk straddles the receiving plane
  (`|apparent centre . normal| < sin alpha`), the coarse per-beam
  `max(., 0)` truncation mis-integrates the partially incident disk, so
  those points enter the exact per-point grid with their lit interval
  intersected against the plane's `apparent beam . normal = 0` root,
  bisected with the same apparent map the beam integrand reads.
  This boundary is orthogonal to the earth-shadow band: a point can need
  it while fully clear of occultation (raised observers, low sun over any
  tilted plane).  Every flux consumer routes through the seam -- the SOS
  BOA and TOA boundary diagnostics included (gen3-R2): a back-end that
  sums the coarse disk itself re-introduces the truncation error and the
  near-horizon false zero.
* Position domain (gen3-R2): every Cartesian position kernel
  (`scatter`, `ground`, `direct_irradiance`, and the band classification
  under them) shares one domain check -- finite coordinates, radius
  between the opaque ground and the model top, with only a +-1 m
  rounding band snapping to the boundary (MC ground intersections).
  Beyond it the query raises, matching the table's own rejection;
  silent clipping used to reinterpret underground points as surface
  scenes (plan 6.1: out-of-domain queries fail loudly).
* Selection is per point -- the point's own geometry decides.  Batch
  composition, chunking and call-site grouping change execution cost only,
  never a value (review round 4, Major batch invariance).

## What each backend consumes, and what it still approximates

| backend | solar side | retained approximation |
|---|---|---|
| `rt.single` | per-scattering-point `scatter` kernel; phase from the *apparent* beam direction | straight viewing rays; view-path optical depth from the shell quadrature with the aerosol `HorizontalField` applied to the view-side coefficients only (solar columns stay radial) |
| `rt.multiple` (SOS) | per-LAYER disk quadrature: layers inside the contact band get the source's adaptive disk at their mid-point height, layers outside keep the production coarse disk; beams projected onto the Fourier source with their local azimuth factors (the disk spans only ~16'/sin(z) of local azimuth, so `cos(m d_az) ~ 1`); BOA direct and TOA incident flux through the unified plane-flux seams (`ground` / `direct_irradiance`, both LOCAL at the boundary point); sun `T` is the raw physical extinction -- delta-M scales only the diffuse moments | plane-parallel diffuse transport; ground fixed at `z_bounds[0]`; layer sources project per layer (adaptive layers have their own beam counts) |
| `rt.mc`, `rt.twilight` | deferred batched `scatter`/`ground` evaluation; the RNG stream, free paths and event classification are bit-identical to a run with the solar values blanked | straight-line photon transport; horizontally uniform atmosphere |
| `rt.cloud` | per-beam classification (top entry / base entry / mid-deck column-shadow entry / blocked): the entry height is the vertical column's shadow root for that beam direction (`SolarSource.shadow_height`, exactly `R/cos(e)-R` in vacuum), the beam set is the adaptive disk at deck top, per-beam two-stream responses, `geometry="mixed"` when the disk straddles the boundary; partially lit decks evaluate the representative T/G/mu0 at the sunlit-segment MIDPOINT (plan 3.3 closure) while tau splits at the shadow root | plane-parallel homogeneous deck |
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

## Known deviations and accepted limits (owner adjudication, 2026-11-06)

The unified finite-disk source is the production default for every backend.
The deviations below are recorded conditions of that admission: each carries
its measured value, the plan budget it misses, and the adjudicated
disposition. They are settled for this task's review loop; funding any of
them further is a separate user decision.

1. **Direction anchor 2.673e-4 deg vs the 2e-4 deg plan budget.** The
   Cartesian-RK anchors measure a max direction error of 2.673e-4 deg
   (about 0.96 arcsec, ~0.05% of the solar disk diameter). This is the
   representation limit of the monotone-PCHIP knot grid, not a table
   density choice: rows whose turning point sits near the 150-200 km
   near-caustic fold are not representable by any monotone interpolant
   (stable across review rounds R4-R7; a 0.3 km refinement experiment
   reproduced the fold and was reverted). Accepted as a representation
   limit with this support-domain caveat. The table holdout adds 1.4e-3 deg
   at one mid-height/mid-elevation point class, with negligible radiative
   effect.
2. **Deep-twilight first-order closure is budget-limited-unresolved.** At
   the plan's budget cap (both grounds, sun -9/-12 deg, 8 seeds x 500k
   photons per seed) the MC-vs-SS first-order closure passes the
   |mean-SS| bound where the signal is detectable, but the signability
   gate SE/|SS| <= 5% fails at -9 deg (both grounds) and -12 deg
   g = 2.4 km, which also misses |mean-SS| <= 4SE + 1%. Recorded as the
   terminal "not signable at the funded budget" state: two independent
   chains (this suite and the archived deep-twilight-reference task,
   ROADMAP M19) measure the same statistical shape, which is the answer
   rather than a pending item. `closure.json` reports these honestly as
   failures; raising the budget or deleting the deep points is out of
   scope.
3. **Availability gate: dome query cost.** Reference-machine measurement
   after the round-6 rewrite, the gen2-R1 fixes and the task-dome-kernel
   vectorization: source cold build 9.3-10.4 s (budget 10 s), warm single
   81wl single-direction 0.011-0.015 s (budget 1 s), MC 3wl x 20k 2.9-3.5x
   legacy (budget 4x), SOS 0.03x, cold peak RSS 0.34-0.38 GiB (budget
   +0.5 GiB).  The standing breach is the standard clear dome: 5.5x the
   legacy wall time (budget 3x), i.e. 6.1 s for the 46x91 3-wavelength
   warm dome vs the legacy 1.12 s (6.7x / 7.45 s on the branch point in
   the same idle window; evidence pair in
   `validation/data/production_solar_source/perf-dome-kernel.json`).  The
   per-height query kernel itself is no longer the bottleneck: its
   row-group Python loop was replaced by a bitwise-equal vectorized core
   (argsort row groups with C `searchsorted` segment probes, cache-blocked
   flat coefficient gathers).  What remains is contractual volume, not
   dispatch overhead: (a) the per-point head/tail work of the broadcast
   (height x beam) grid -- height dedup, grazing interpolation, angular
   clipping, and the tail cos/expm1 field transforms -- which the plan's
   cProfile probe understated because the profiler inflated the old
   small-call kernel about 2x, and (b) the same gather+Horner arithmetic
   the row-group implementation performed point for point.  The second
   tier (aerosol phase kernel, `_geometry_arrays` fusion) was evaluated
   and rejected for this task: even fully removed it lands near 4x, and
   it is a different root cause.  Accepted as a recorded deviation; for
   interactive use lower
   `skycolor.render.sky_dome(n_zenith, n_azimuth)` (default 46/91) -- the
   dome grid, not the image size, dominates this cost.
4. **Ray-tube referee not built.** Rays whose turning point lands within
   ~100 m of a profile-formula discontinuity (e.g. the 11 km US76 lapse
   break) have a local slope sensitivity above the smooth-field budget.
   The DOP853 ray-tube arbitration of plan 6.2 remains unbuilt. Together
   with the per-height vectorized dome query kernel (item 3) it is
   recorded as a backlog candidate awaiting a separate user decision.

Historical evidence for the unified source's physics (independent direction,
columns, etendue and first-order closure experiments) lives in
`docs/twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md` and
`docs/twilight/SOLAR_SOURCE_AUDIT.md`; those experiments predate
productionisation and are kept unchanged.
