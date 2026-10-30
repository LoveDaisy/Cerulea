# Architecture and interfaces

This document is the contract between the modules.  If you change an interface,
update this file in the same commit.

## Conventions

* Wavelengths are **nanometres** everywhere in the public API.
* Radiance is **W m⁻² sr⁻¹ nm⁻¹**.
* Coefficients are **m⁻¹**, number densities **m⁻³**, cross sections **m²**.
* Heights are **km** in the atmosphere API and **m** in the geometry/RT layer.
* Angles are degrees on the outside, radians on the inside.
* Azimuths are measured clockwise from **north**; zenith angle 0 is up.

## Data flow

```
spectra/*            solar irradiance, CMF, cross sections (tabulated)
    |
atmosphere.Atmosphere   1-D spherical shells: T, p, n_air, n_O3, n_NO2, aerosol
    |                   -> layer_properties(), phase_moments(), r_bounds
    |
rt.single.sky_radiance  exact first-order scattering in spherical geometry
    |
rt.multiple.*           higher orders (diffuse field)
    |
rt.solver.total_radiance    L_1 + L_ms
    |
color.render_rgb        XYZ -> adapted -> tone mapped sRGB
    |
render.py               cameras, panoramas, fisheye, PNG
```

## Key interfaces

### `skycolor.atmosphere.Atmosphere`

```python
Atmosphere(wl_nm, ozone_du=300, no2_column=0, aerosols=[...],
           ground_albedo=0.1, elevation_km=0)
```

Attributes: `wl`, `z_bounds` (km, n+1), `z_mid` (km, n), `r_bounds` (m, n+1),
`aerosols` (list of `AerosolLayer`), `elevation_km`.

Methods:
* `n_air(z)`, `n_o3(z)`, `n_no2(z)` — vectorised, `z` in km.
* `layer_properties()` → dict of arrays `(n_layer, n_wl)`:
  `beta_ray`, `beta_as`, `beta_ae`, `beta_o3`, `beta_no2`, `beta_ext`, `beta_sca`.
* `phase_moments(n)` → `(n_layer, n, n_wl)` Legendre coefficients of the
  layer-averaged phase function, normalised as
  `p(cos t) = Σ_l (2l+1) β_l P_l(cos t)`, `∫p dΩ = 4π`.
* `_sigma_ray`, `_sigma_o3`, `_sigma_no2` — cached spectral cross sections.

### `skycolor.rt.single.sky_radiance`

```python
sky_radiance(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
             sun_azimuth_deg=0.0, n_gauss=3, chunk=512)
    -> ndarray (n_view, n_wl)
```

Exact single scattering; handles sun zenith angles beyond 90° (twilight).

### `skycolor.rt.multiple` (to be implemented)

```python
multiple_scattering(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                    sun_azimuth_deg=0.0, **options) -> ndarray (n_view, n_wl)
```

Returns the radiance from scattering orders ≥ 2.  It must:

* be zero when the atmosphere is empty;
* be smooth in the viewing direction (it is a diffuse field);
* reproduce the known magnitude: for a clear mid-latitude atmosphere at
  550 nm and the sun at 60° zenith, the multiple-scattering contribution to the
  zenith radiance is of order 10–20 % of the single-scattering value, and it
  grows towards the horizon and towards twilight.

### `skycolor.rt.solver`

```python
sky_radiance_total(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                   sun_azimuth_deg=0.0, multiple=True, **kw)
    -> ndarray (n_view, n_wl)
```

### `skycolor.color.render_rgb`

```python
render_rgb(radiance, wl_nm, exposure=None, tonemap="aces",
           adaptation="bradford", white_xy=(0.3127, 0.3290),
           reference_percentile=99.0, preserve_chromaticity=True) -> (..., 3)
```

## Validation targets

| Quantity | Expected | Reference |
|---|---|---|
| Rayleigh optical depth at 550 nm | 0.0976 | Bodhaine et al. 1999 |
| Direct normal illuminance, sun 60° elev. | ~100 klx | clear-sky measurements |
| Clear-sky zenith luminance, sun 30° elev. | 2–5 kcd m⁻² | CIE / measured skies |
| Diffuse / global horizontal, clear sky | 0.10–0.20 | standard radiation data |
| Zenith chromaticity, clear sky | x≈0.25–0.28, y≈0.28–0.32 | measured spectra |
| Twilight (sun −6°) zenith luminance | ~1–10 cd m⁻² | measured |

## Working agreement for parallel agents

* Work **only** inside your own worktree; never edit the main checkout.
* New code goes in **new files** wherever possible; touch shared files only if
  the task explicitly requires it, and keep the diff minimal.
* Every module gets a `tests/test_*.py` with at least one numerical check
  against an independent reference.
* Commit early and often with descriptive messages; do not merge or rebase.
* Use `/Users/zhangjiajie/Codes/Sky Color Simulation/.venv/bin/python` and set
  `PYTHONPATH` to your worktree (or just run from the worktree root, which
  works because every script starts with `sys.path.insert(0, ".")`).
