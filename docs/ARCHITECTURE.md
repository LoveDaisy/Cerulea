# Architecture and interfaces

This document is the contract between the modules.  If you change an interface,
update this file in the same commit.  It describes the architecture after the
`task-architecture-refactor` split (frozen `Medium` / `Scene`, species column
algebra in `optics`, `SkyResult`, `radiance_field` display seam).

## Conventions

* Wavelengths are **nanometres** everywhere in the public API.
* Radiance is **W m⁻² sr⁻¹ nm⁻¹**.
* Coefficients are **m⁻¹**, number densities **m⁻³**, cross sections **m²**.
* Heights are **km** in the atmosphere API and **m** in the geometry/RT layer.
* Angles are degrees on the outside, radians on the inside.
* Azimuths are measured clockwise from **north**; zenith angle 0 is up.

## Module boundaries

```
spectra/*            solar irradiance, CMF, cross sections (tabulated)
    |
medium.Medium        FROZEN physical medium: composition, profiles, z_bounds,
    |                cross sections; layer_properties / phase_moments /
    |                delta_m_* / restrict.  Pure function of its definitions.
scene.Scene          FROZEN observing scenario: observer elevation, spectral
    |                ground albedo, refraction flag.
    |
atmosphere.Atmosphere   facade = Medium + Scene; every public name of the
    |                pre-split class still resolves (read-only properties for
    |                medium state, setter-backed properties for the scenario).
    |
optics.py            shared species listing + column algebra: the species
    |                order (air, ozone, NO2, one entry per aerosol layer),
    |                shell_species_weights / species_extinction_weights,
    |                shell_species_columns / tau_from_columns.  All RT
    |                backends consume this; none imports another's privates.
    |
rt.single            exact first-order scattering, spherical geometry
rt.multiple          orders >= 2, plane-parallel discrete ordinates (SOS)
rt.twilight + rt.mc  deep-twilight spherical engine (split estimator: exact
    |                first order + Monte Carlo higher orders);
    |                twilight_radiance is a thin format adapter
rt.cloud             delta-Eddington cloud deck; apply_cloud_overlay is the
    |                local composition operator (takes clear components,
    |                never solves for them)
    |
rt.solver.solve      the one authoritative composition: four finite
    |                approximations (spherical-single / plane-sos /
    |                hybrid-sos / spherical-mc) resolved from one request
    |                with capability/support checks, then each needed local
    |                kernel once -> SkyResult (approximation, options,
    |                stderr, components, diagnostics, effective);
    |                sky_radiance_total / twilight_radiance /
    |                sky_radiance_with_cloud are thin forwarders
    |
color / render       spectral field -> XYZ -> adapted -> tone-mapped sRGB;
                     solve_dome prepares one result for display (grid, ground
                     lighting, apparent sun); render_result/chart_table are
                     pure consumers; sky_dome/render_scene keep a
                     ``radiance_field`` closure so ANY spectral field (cloudy,
                     experimental) uses the same display path;
                     scene.resolve_sun is the one solar input contract
                     (time or numbers -> pre-refraction direction)
```

### What is shared vs deliberately independent

Shared across backends (one authoritative implementation each):
geometry helpers (`geometry.py`), the species listing and column algebra
(`optics.py`), the medium definition (`medium.py`), refraction tables
(`refraction.py`, cached per density-definition identity).

Deliberately independent: the Monte Carlo's *transport, sampling and
estimator* (`rt/mc.py`) are written from scratch on purpose so the
MC-vs-deterministic cross-check stays meaningful.  **Known blind spot**: the
shared parts (geometry, species definitions, column algebra) are common
failure modes that the cross-check cannot cover — their independent evidence
lives in `validation/` against external references, not in the comparison.
The two Mie kernels (`mie.py` single-particle analytic moments vs
`cloud.py` batched numerical moments) are separate regimes on purpose and
anchor each other for small size parameters
(`tests/test_extension_scenarios.py`).

### Backing approximations (CAPABILITIES)

Each backend declares its standing approximations as a module-level
`CAPABILITIES` dict; `rt.solver.solve` warns on the one degenerate request
combination (twilight + non-uniform `HorizontalField`, which reaches the
first-order term only) and stays silent on the default paths.

| Backend | Standing approximations |
|---|---|
| `rt.single` | spherical quadrature; unified finite-disk source at every scattering point; viewing rays remain geometric; horizontal field applied to view-side coefficients |
| `rt.multiple` | plane-parallel diffuse field; ground fixed at `z_bounds[0]` (`elevation_km` not read); per-disk-beam Fourier source with raw-extinction sun `T` (delta-M scales only the diffuse moments); horizontal field not applied |
| `rt.twilight` | unified-source first order; MC assumes a horizontally uniform atmosphere; MC transport has no refraction; horizontal field in the first-order term only |
| `rt.cloud` | homogeneous plane-parallel deck (delta-Eddington); per-beam top/base/mixed illumination from the unified source |

### The unified solar source

One module owns the sun: `skycolor/solar_source.py` holds the refractive
field, the continuous species columns, the lazily built point-source table
(cached per medium-construction token), the finite disk and the consumer
kernels; its contract, per-backend consumption table, validation entry
points and measured budgets are in
[physics/SOLAR_SOURCE.md](physics/SOLAR_SOURCE.md).  The historical
limitations recorded by the source audits -- the sparse packaged inverse
angle table and the missing disk angular Jacobian -- are what this module
replaces; the audit documents remain unchanged as historical evidence
([SOLAR_SOURCE_AUDIT.md](twilight/SOLAR_SOURCE_AUDIT.md),
[unified-source](twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md)).  Agreement between
two backends using the same source cannot validate it: the independent
chain lives in `validation/production_solar_source.py` (RK anchors, analytic
limits, table holdout and the deterministic-vs-MC first-order closure).

### delta-M quantities vs raw physics (E4)

`phase_moments(n)` returns the *raw* Legendre moments of the layer-averaged
phase function (`p(cos t) = Σ (2l+1) β_l P_l`, ∫p dΩ = 4π).  `delta_m_moments`
returns the Wiscombe-scaled moments with the forward fraction
`f = β_n` (`β*_l = (β_l − f)/(1 − f)`), and `delta_m_properties` additionally
rescales `τ* = τ(1 − ωf)`, `ω* = ω(1 − f)/(1 − ωf)`.  The relation to raw
physics: a truncated discrete-ordinates code that used the raw moments would
silently drop the forward peak (~37 % of the scattered energy for dust and
cloud droplets); the delta-M set is the representation the SOS solver
consumes, and `delta_m_f` records exactly what was scaled away.

## Scenario semantics

`at_altitude(z0)` **truncates** the model at the site: `z_bounds` starts at
z0, the ground (and every backend's bottom boundary) sits there, and nothing
is renormalised — single scattering (integrating from the observer), SOS and
MC agree on where the ground is.  `elevation_km=0` is a no-op.  The
balloon scenario (observer raised, sea-level domain kept) is the plain
constructor with `elevation_km=z0`; the two are distinct constructions
(`tests/test_extension_scenarios.py`).

The opaque radius is **`r_bounds[0]` for both transport and solar occlusion**.
Geometric solar masks, refracted grazing tests and shadow tables must use that
radius; otherwise a solar ray can pass through the raised solid ground while
its atmospheric column silently excludes the underground segment.  M12's
truncation initially left this inconsistency in place.  M13 fixes it across
single scattering, SOS, MC and cloud illumination without changing M10's
two-branch solar-column integral (`tests/test_raised_ground.py`).  It affects
zenith twilight as well as horizon views.

For the recorded clear-air, refraction-on, −6° zenith comparison, the
deterministic 2.4 km/sea-level ratio changes from the historical 1.208× to
1.027×.  The historical low-budget engine 1.01× also predates the fix;
multi-seed results and the distinction between domain truncation and an
observed station comparison are in
[TWILIGHT_ALTITUDE_AUDIT.md](twilight/TWILIGHT_ALTITUDE_AUDIT.md).  These ratios do not
establish a missing low-altitude multiple-scattering component.

## Caching and identity

* `Medium` / `Scene` are frozen value objects; redefinition replaces them
  wholesale (facade setters), never patches.  Two separately constructed
  media never share caches.  "Frozen" guarantees the *binding*, not the
  content: the numpy arrays and nested objects held by a frozen object are
  still technically mutable in place — treat everything reachable from a
  frozen object as read-only (a51: the mutation surface is not fenced; the
  contract is documented here rather than enforced).
* Species-column caches per backend key on (wavelength, shell) values —
  per-instance, value-semantics keys.
* `MieAerosol` / `CloudLayer` are frozen; `shared()` returns read-only
  *by structure* (assignment raises `FrozenInstanceError`); redefinition is
  `replace()`, returning a fresh unshared object with rebuilt caches
  (`CloudLayer.replace` explicitly drops the carried-over optics cache).
* Refraction tables depend on the density profile **and opaque bottom radius**
  and cache on the medium's `refraction_token`: inherited by
  `restrict()` (a wavelength subset never changes densities), fresh per
  construction ("per-construction, don't share" — one ~2.3 s table build per
  genuinely distinct medium), stored in a `WeakKeyDictionary` so entries
  die with their medium (batch runs do not accumulate tables).  The token
  is the medium's construction identity: redefining *any* medium field
  through the facade (aerosols included) rotates it and triggers one
  rebuild, even when the air-density profile the tables actually read is
  untouched — never wrong, occasionally one rebuild more than necessary.

## Key interfaces

### `skycolor.atmosphere.Atmosphere`

```python
Atmosphere(wl_nm, ozone_du=300, no2_column=0, aerosols=[...],
           ground_albedo=0.1, surface_pressure=P0, z_bounds_km=None,
           elevation_km=0, refraction=True,
           ozone_profile=None, no2_profile=None, air_profile=None)
```

* `ozone_profile` / `no2_profile`: vertical *shape* callables, column-
  normalised to `ozone_du` / `no2_column` exactly like the built-in tables
  — the extension point for custom gas profiles.
* `air_profile`: complete `n_air(z)` callable replacing the US76 pressure /
  temperature construction.
* Frozen medium state surfaces as read-only properties (`wl`, `z_bounds`,
  `_sigma_*`, ...); the scenario (`elevation_km`, `ground_albedo`,
  `refraction`) is setter-backed, and `aerosols` returns an immutable
  tuple (redefine through the setter).
* `at_altitude(z0)`: truncation semantics above.

### `skycolor.rt.solver`

```python
solve(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
      sun_azimuth_deg=0.0, *,
      method=None,                    # legacy alias: "clear"|"twilight"|"cloud"
      approximation=None,             # "spherical-single"|"plane-sos"|
                                      # "hybrid-sos" (default)|"spherical-mc"
      cloud_layer=None, clear_radiance=None, ground_albedo=None,
      include_clear=True, include_cloud=True,
      single=None, multiple=None,     # order selection, default both on
      n_gauss=None, chunk=None, quality=None, seed=None, grid=None,
      single_config=None, sos_config=None, mc_config=None) -> SkyResult

sky_radiance_total(...)      # thin forwarder of the hybrid-sos branch
twilight_radiance(...)       # thin format adapter of spherical-mc
sky_radiance_with_cloud(...) # thin forwarder: solve + apply_cloud_overlay
```

`SkyResult` = `radiance (n_view, n_wl)`, `wl`, `method`, `options`,
`stderr` (MC/smoother composite estimate including the 3 % floor — not a
pure sample SE, not total physical error), `components` (the selected
`single`/`multiple` arrays, strictly closing the total; cloud keeps
`clear`/`cloud`/`transmittance`/`illumination`/`optics`), `diagnostics`
(SOS `orders`/`converged`, MC `undetected_wl` + grid, ...) plus
`approximation` + `approximations` (per-component kernel capabilities
actually run) + `effective` (clamped discretisation/settings as returned
by the kernels).  Statistical, convergence and physical-approximation
information stay separate; there is no single "confidence" number.

```python
from skycolor.rt.solver import solve, MCConfig, SOSConfig

res = solve(atmo, sun_z, vza, va, sun_az)              # hybrid-sos default
res.components["single"] + res.components["multiple"]  # closes res.radiance
res.effective["sos"]      # discretisation the kernel actually ran with
sph = solve(atmo, sun_z, vza, va, sun_az, approximation="spherical-single")
pln = solve(atmo, sun_z, vza, va, sun_az, approximation="plane-sos")
mc  = solve(atmo, sun_z, vza, va, sun_az, approximation="spherical-mc",
            mc_config=MCConfig(seed=7))
cloudy = solve(atmo, sun_z, vza, va, sun_az, cloud_layer=deck)  # overlay
pre = solve(atmo, sun_z, vza, va, sun_az, cloud_layer=deck,
            clear_radiance=cached_clear)   # external clear: not recomputed
```

### `skycolor.render`

```python
resolve_sun(sun_elevation=None, sun_azimuth=None, time=..., ...)  # scene.py
pres = solve_dome(atmo, sun_elev, sun_az, approximation=..., cloud_layer=...)
render_result(pres, camera=..., ground=..., sun_disk=None)   # pure display
chart_table(pres)                          # nearest-node rows + actual angles
format_result_summary(pres)                # provenance/limits, result fields only
presentation_from_result(res, zen, az_rel, sun_elev, sun_az)  # bind a ready result
presentation_from_field(field, atmo, ...)  # the external-field seam
sky_dome(atmo, sun_zenith_deg, ..., radiance_field=None)     # thin: dict form
render_scene(..., radiance_field=None, ground_direct_transmittance=None,
             approximation=..., cloud_layer=...)             # thin: ndarray form
ground_radiance(..., direct_transmittance=None)              # legacy ground seam
```

The production flow is one-directional: `resolve_sun -> one solve
(`solve_dome`) -> `SkyPresentation` -> outputs`.  Preparation binds the grid
layout, the ground albedo/irradiance snapshots and the adopted source's
apparent sun geometry; after that, switching cameras, tone maps or output
formats re-enters nothing (spied in `tests/test_presentation.py`).
`SkyResult` stays the only authority on solver facts — the carrier copies no
diagnostics and holds no solver callback.

`radiance_field(zen_flat, az_rel_flat) -> (n_view, n_wl)`: the display layer
consumes any spectral field without knowing the medium; bind medium and
solar geometry in the caller's closure.  With a field given, `atmo` is used
only for `wl` metadata (and the ground albedo default).  An external field
records provenance `"external"`: a ground display then requires a matching
`ground_direct_irradiance` (or the retired `ground_direct_transmittance`)
instead of silently pairing the custom sky with the clear beam.

## Entry-point migration map

| Old | Now |
|---|---|
| `rt.solver.sky_radiance_total` | kept: thin forwarder; the one composition lives in `solve` (hybrid-sos) |
| `rt.twilight.twilight_radiance` (2-tuple) | kept: thin format adapter over `solve(approximation="spherical-mc")`; `return_info=True` adds the MC info dict |
| `rt.cloud.sky_radiance_with_cloud` | kept: thin forwarder (clear via `solve`, overlay via `cloud.apply_cloud_overlay`); structured via `solve(cloud_layer=...)` |
| `experiments/profile_variants.py` subclasses | thin constructors over `ozone_profile=` / `air_profile=` |
| `install_twilight_profile_support` | removed — `_restrict` is the engine's native path |
| `cloud._Distribution` | `cloud.SizeDistribution` (alias kept) |
| `refraction._Profile` / `_trace` in data pipeline | `refractive_profile` / `trace_ray(..., profile=...)` |
| mutating a shared aerosol | `aerosol.replace(**kw)` (shared instances are frozen) |
| `at_altitude` (double-scaled rescaling) | truncation at the site (Step 3c) |
| `experiments/clouds/clouds.py` display helpers | `render.sky_dome` / `render_scene` with `radiance_field`; the deck ground-beam helper lives in `rt.cloud.ground_direct_irradiance` (experiments forward to it) |
| CLI `--elevation` | unchanged semantics (site altitude); values follow the truncation fix |
| CLI `--time` (chart ignored it; `refraction=True` ephemeris angle) | both subcommands consume `--time` through `scene.resolve_sun`: naive means UTC, explicit offsets convert before the ephemeris reads clock fields, and the resolved direction is pre-refraction (the source applies refraction once) |

### Semantics tightened by the presentation consolidation

Display-level corrections (task-presentation-flow); solve numbers inside the
supported domains are unchanged:

1. Time/numeric solar inputs now describe the same pre-refraction direction
   on every consumer; the old CLI path double-refracted (geometric
   correction at the ephemeris, then the source's traced bending).
   Explicit angles together with `--time` are an error; numeric defaults
   fill only when no time is given.
2. A cloudy display bound through `radiance_field` without a matching
   `ground_direct_irradiance`: silently paired with the clear beam →
   `ValueError` (`ground=False` skips the ground display).
3. Ground display for an observer off the model bottom: evaluated at the
   wrong height → `ValueError` (the sky keeps working with
   `ground=False`).
4. The decorative sun disk consumes the adopted source's apparent position
   and occultation-aware visibility (was: true elevation with a −0.9°
   cutoff) and defaults to off on cloud scenes (was: drawn on any request);
   `--no-sun` still disables it entirely.  The marker stays an annotation:
   it never enters charts or ground energy.
5. Convenience ndarray entries (`sky_dome`, `sky_radiance_grid`,
   `render_scene`) warn visibly on SOS non-convergence, MC undetected
   wavelengths and the statistical nature of MC stderr; CLI products print
   `format_result_summary` instead (result fields only, unknowns stay
   unknown).  An MC result that ran but came back all-undetected is
   reported as "ran, statistics not determinable from zeros" -- never as
   "no MC term ran".
6. An external `radiance_field` and a solver choice (`approximation`,
   `cloud_layer`, grouped configs, explicit order selection) are mutually
   exclusive: silently ignored → `ValueError` at every public entry
   (`solve_dome` / `render_scene` / `sky_radiance_grid`); the legacy
   `sky_dome` field forwarding injects no solver settings and is not
   misflagged.  `render_scene`'s retired `ground_direct_transmittance`
   converts once at the wrapper boundary when given alone (same deprecated
   formula, one warning) and errors when given together with
   `ground_direct_irradiance`.

### Semantics tightened by the entry consolidation

Previously silently ignored or silently miscomputed; numbers inside the
supported domains are unchanged (bitwise frozen-baseline evidence in the
task log).  Applicability of every setting and support check now follows
the request's *execution plan* -- which kernels actually run:

1. Elevated observer + a plane-parallel term (``plane-sos``; ``hybrid-sos``
   with the SOS higher orders enabled): the plane field was solved at the
   domain bottom regardless of the observer → `ValueError`.  Use
   `spherical-single`, `spherical-mc` (both evaluate the observer at its
   actual elevation), ``hybrid-sos`` with ``multiple=False``, or
   `at_altitude(z0)` for a raised station.
2. Cloud request with `single=False`: silently ignored → `ValueError`
   (cloud-only totals use `include_clear=False`).
3. Observer inside/above the cloud deck: unchecked on the external-clear
   path → `ValueError` on every cloud entry point.
4. `plane-sos` with a heterogeneous horizontal field: silently unused
   input → `ValueError`; likewise any request whose execution plan runs
   no spherical single term (higher orders only, external clear,
   cloud-only) — the field would apply to no output component.
5. Unknown `quality`/`grid` keys, explicit settings for a kernel the
   request will not run (e.g. `n_gauss`/`SingleConfig` on `plane-sos`,
   whose first order comes from the SOS run; single/SOS settings with an
   external clear field or a cloud-only request), disagreeing
   `method`/`approximation`, the same setting given through both the
   legacy and grouped channels: silently ignored or undefined →
   `ValueError` before any kernel runs.
6. View zenith outside `[0, 90]` degrees (or non-finite): the single
   kernel returned broken geometry, SOS clipped such rays onto the
   horizon → `ValueError` (the horizon at exactly 90° keeps the existing
   clipping approximation).
7. Cloud-only requests (`include_clear=False`, no external field): the
   formal entry used to run the whole clear solve and discard it → no
   clear kernel runs (the historical wrapper semantics), the excluded
   term is recorded as zeros.
8. Result records: `options` keeps the *request* values (`sos` kwargs,
   single `n_gauss`/`chunk` including on the MC route, `refraction`,
   `quality`/`seed`/`grid`); `effective` keeps what the kernels *ran*
   with (clamped SOS discretisation split into transport `n_moments` vs
   optics `n_moments_optics`, MC `n_anchor_run`/`n_photons_run` after
   anchor dedup and photon rounding); per-component provenance records
   the actual `source_refraction` switch instead of assuming it.
9. The legacy `sky_radiance_total` wrapper keeps its lenient behaviour
   (`single=False` silently drops `n_gauss`/`chunk`); the formal `solve`
   is strict.  The cloud wrapper now forwards the cloud request itself,
   so checks 2/3/4/6 hold identically on both entry points.
10. spherical-mc routing runs the single term with the wrapper default
   `chunk=512` (the old twilight path used the kernel default 128);
   arrays are bitwise unchanged, only the memory blocking differs.
11. `spherical-single` with explicit `multiple=True` → `ValueError`.

## Validation targets

| Quantity | Expected | Reference |
|---|---|---|
| Rayleigh optical depth at 550 nm | 0.0976 | Bodhaine et al. 1999 |
| Direct normal illuminance, sun 60° elev. | ~100 klx | clear-sky measurements |
| Clear-sky zenith luminance, sun 30° elev. | 2–5 kcd m⁻² | CIE / measured skies |
| Diffuse / global horizontal, clear sky | 0.10–0.20 | standard radiation data |
| Zenith chromaticity, clear sky | x≈0.25–0.28, y≈0.28–0.32 | measured spectra |
| Twilight (sun −6°) zenith luminance | ~1–10 cd m⁻² | measured |
