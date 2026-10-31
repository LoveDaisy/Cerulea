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
    |                first order + Monte Carlo higher orders)
rt.cloud             delta-Eddington cloud deck composed on the clear solver
    |
rt.solver.solve      explicit backend dispatch -> SkyResult (method, options,
    |                stderr, components, diagnostics); sky_radiance_total is
    |                the legacy ndarray wrapper of the "clear" branch
    |
color / render       spectral field -> XYZ -> adapted -> tone-mapped sRGB;
                     sky_dome/render_scene accept a ``radiance_field`` closure
                     so ANY spectral field (cloudy, experimental) uses the
                     same display path
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
| `rt.single` | spherical quadrature; solar attenuation/shadow can refract, viewing rays and phase angles remain geometric; horizontal field applied per scattering point |
| `rt.multiple` | plane-parallel diffuse field; ground fixed at `z_bounds[0]` (`elevation_km` not read); refracted shadow threshold but geometric solar optical depth; horizontal field not applied |
| `rt.twilight` | MC assumes a horizontally uniform atmosphere; MC transport has no refraction; horizontal field in the first-order term only |
| `rt.cloud` | homogeneous plane-parallel deck (delta-Eddington); inherits the clear solver's geometry |

### Solar-source approximations

The solar-source audit additionally records two limitations of the current
refraction prescription: the packaged inverse angle table is sparse, and the
source amplitude does not include the refracted solar disk's angular
Jacobian.  Independent reference tools quantify these without changing the
library defaults; see [SOLAR_SOURCE_AUDIT.md](SOLAR_SOURCE_AUDIT.md).  Agreement
between two backends using the same source prescription cannot validate it.

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
[TWILIGHT_ALTITUDE_AUDIT.md](TWILIGHT_ALTITUDE_AUDIT.md).  These ratios do not
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
      sun_azimuth_deg=0.0, *, method="clear"|"twilight"|"cloud",
      cloud_layer=None, single=True, multiple=True,
      quality="medium", seed=0, grid=None) -> SkyResult

sky_radiance_total(...)  # legacy ndarray entry, wraps the "clear" branch
```

`SkyResult` = `radiance (n_view, n_wl)`, `wl`, `method`, `options`,
`stderr` (twilight MC 1σ only), `components` (cloud decomposition),
`diagnostics` (SOS `orders`/`converged`, twilight `undetected_wl` + grid,
...).  Statistical, convergence and physical-approximation information stay
separate; there is no single "confidence" number.

### `skycolor.render`

```python
sky_dome(atmo, sun_zenith_deg, ..., radiance_field=None)
render_scene(..., radiance_field=None, ground_direct_transmittance=None)
ground_radiance(..., direct_transmittance=None)
```

`radiance_field(zen_flat, az_rel_flat) -> (n_view, n_wl)`: the display layer
consumes any spectral field without knowing the medium; bind medium and
solar geometry in the caller's closure.  With a field given, `atmo` is used
only for `wl` metadata (and the ground).

## Entry-point migration map

| Old | Now |
|---|---|
| `rt.solver.sky_radiance_total` | kept (compat); structured access via `rt.solver.solve` |
| `rt.twilight.twilight_radiance` (2-tuple) | kept; `return_info=True` adds the MC info dict |
| `rt.cloud.sky_radiance_with_cloud` | kept; structured via `solve(method="cloud")` |
| `experiments/profile_variants.py` subclasses | thin constructors over `ozone_profile=` / `air_profile=` |
| `install_twilight_profile_support` | removed — `_restrict` is the engine's native path |
| `cloud._Distribution` | `cloud.SizeDistribution` (alias kept) |
| `refraction._Profile` / `_trace` in data pipeline | `refractive_profile` / `trace_ray(..., profile=...)` |
| mutating a shared aerosol | `aerosol.replace(**kw)` (shared instances are frozen) |
| `at_altitude` (double-scaled rescaling) | truncation at the site (Step 3c) |
| `experiments/clouds.py` display helpers | `render.sky_dome` / `render_scene` with `radiance_field` |
| CLI `--elevation` | unchanged semantics (site altitude); values follow the truncation fix |

## Validation targets

| Quantity | Expected | Reference |
|---|---|---|
| Rayleigh optical depth at 550 nm | 0.0976 | Bodhaine et al. 1999 |
| Direct normal illuminance, sun 60° elev. | ~100 klx | clear-sky measurements |
| Clear-sky zenith luminance, sun 30° elev. | 2–5 kcd m⁻² | CIE / measured skies |
| Diffuse / global horizontal, clear sky | 0.10–0.20 | standard radiation data |
| Zenith chromaticity, clear sky | x≈0.25–0.28, y≈0.28–0.32 | measured spectra |
| Twilight (sun −6°) zenith luminance | ~1–10 cd m⁻² | measured |
