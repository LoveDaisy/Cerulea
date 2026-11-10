# The layer story, actual production calls

Production version: `286486e3a380bd810ac347aa1b53cc3963b504d8`

Configuration: `{'day_grid': {'n_zenith': 25, 'n_azimuth': 37}, 'twilight_grid': {'n_zenith': 11, 'n_azimuth': 23}, 'cloud_grid': {'n_zenith': 11, 'n_azimuth': 21}, 'mc_grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000.0}, 'no2_column': 4e+19, 'deck': 'cloud.stratus(): tau550=20, base 0.5, top 1.5 km, r_eff=8 um, water'}`

Media: `{'l1': 'pristine: Rayleigh only, ozone_du=0, no2_column=0, aerosols=(), albedo=0.1, sea-level observer, refraction=True', 'l2': 'clear family minus ozone: HinterlandAerosol AOD550=0.05, exponential H=1.5 km, ozone_du=0, no2_column=0, albedo=0.1', 'l3': 'HinterlandAerosol AOD550=0.05 (H=1.5 km), ozone_du=300, no2_column=4.000e+19 m^-2 (4e15 cm^-2), albedo=0.1'}`

Display groups: `{'day': {'camera': 'perspective', 'size': (640, 320), 'hfov_deg': 100.0, 'pitch_deg': 14.0, 'exposure': 0.03, 'tonemap': 'aces', 'adaptation': None, 'sun_disk': False, 'ground': False}, 'twilight': {'camera': 'perspective', 'size': (640, 320), 'hfov_deg': 100.0, 'pitch_deg': 14.0, 'exposure': 10.0, 'tonemap': 'aces', 'adaptation': None, 'sun_disk': False, 'ground': False}, 'cloud': {'camera': 'perspective', 'size': (640, 320), 'hfov_deg': 100.0, 'pitch_deg': 14.0, 'exposure': 0.03, 'tonemap': 'aces', 'adaptation': None, 'sun_disk': False, 'ground': True}}`

Order: L1, L2, L3 bridge+state, L4 bridge+full, L5 single/MC/plane, L6 clear+cloud.  First call per medium is cold; later calls share warmed optics.  No fair speed ranking.  MC: qualitative / budget-limited, single seed.
Closure: per-solve component additivity only; cross-ingredient additivity is not attempted (nonlinear in the optical depths).

## l1_rayleigh_30  (L1 · Rayleigh only)

Wall time: 17.155166 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l2_aerosol_30  (L2 · +Mie aerosol)

Wall time: 28.418614 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l3_bridge_3  (L3 · bridge (no O3))

Wall time: 0.509068 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 87.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 87.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l3_o3_3  (L3 · +O3/NO2)

Wall time: 13.906610 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 87.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 87.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l4_bridge_30  (L4 · bridge (single only))

Wall time: 0.832261 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l4_multiple_30  (L4 · +multiple scattering)

Wall time: 0.844918 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l5_single_-4  (L5 · spherical single)

Wall time: 6.666312 s

```text
approximation: spherical-single (method 'clear')
sun: zenith 94.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}
```

options: `{'sun_zenith_deg': 94.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'spherical-single', 'n_gauss': 3, 'chunk': 512}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}}`

diagnostics: `{}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}}`

## l5_mc_-4  (L5 · +MC higher orders)

Wall time: 30.842447 s

```text
approximation: spherical-mc (method 'twilight')
sun: zenith 94.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.twilight.multiple_scattered (source_refraction=True)
    scattering: split estimator: exact spherical first order + MC higher orders
    solar_source: unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token
    horizontal_field: first-order term only
    ms_geometry: Monte Carlo assumes a horizontally uniform atmosphere
    mc_refraction: none (straight-line photon transport)
    ss_refraction: unified refracted solar source in the first-order term
effective: single {'n_gauss': 3, 'chunk': 512}; mc {'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000.0}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}
mc: photons_run=200004 anchors_run=5 coarse grid 3x4 seed 7
mc stderr (W m-2 sr-1 nm-1): median 3.544e-06, max 1.527e-05 -- composite estimate: the MC standard error propagated through the angular smoother and wavelength interpolation plus a ~3% smoother resolution-bias floor (in quadrature); not the total physical error of the model, and the photon budget is not tail coverage
```

options: `{'sun_zenith_deg': 94.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'spherical-mc', 'n_gauss': 3, 'chunk': 512, 'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000.0}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'mc': {'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000.0}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}}`

diagnostics: `{'undetected_wl': [], 'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000.0, 'anchor_wl': [380.0, 480.0, 580.0, 680.0, 780.0], 'bandwidth': 0.001, 'n_anchor_run': 5, 'n_photons_run': 200004, 'n_zen_run': 3, 'n_az_run': 4, 'coarse_zenith': [0.0, 0.0, 0.0, 0.0, 83.0, 83.0, 83.0, 83.0, 86.0, 86.0, 86.0, 86.0], 'coarse_azimuth': [0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0], 'coarse_values': [[4.7132685371771884e-05, 7.277480731537443e-05, 5.63808509497054e-05, 5.988304848708432e-05, 0.00015698756776854575, 0.00013958660768065867, 0.00010111651786676614, 8.546780854139955e-05, 0.0001688546588502568, 0.00015349180684110365, 8.013114717610806e-05, 9.705672822790188e-05], [4.143299815976613e-05, 4.880572392691678e-05, 6.520707971559272e-05, 6.166746677659267e-05, 0.0005090073400568105, 0.0003644257427388529, 0.0001806265005499695, 0.00013926027243604138, 0.0006039443104444072, 0.00033581216074045003, 0.00018372565722911646, 0.00017100475992027825], [1.0807710438594442e-05, 1.714119037332259e-05, 1.5429794873648995e-05, 1.3840641750106828e-05, 0.00024599809771152034, 0.00012829811262673565, 7.349796268276318e-05, 6.683644550716186e-05, 0.0003123934545111469, 0.0001691561469688263, 6.901217621439697e-05, 6.696833969787268e-05], [1.4849311964396113e-05, 1.8275022316828297e-05, 1.3698278340971841e-05, 1.9852306557340516e-05, 0.00030664451222391125, 0.0001872624215642405, 8.184455621921317e-05, 7.337259646733463e-05, 0.0005365809932464801, 0.0002785448716212895, 0.00010913332645037528, 0.00010901386366060246], [8.525869560444296e-06, 1.4161088927100938e-05, 5.9031772309647e-06, 1.2225987506339657e-05, 0.0002209502465685009, 0.00010076051070784696, 5.7439686669751384e-05, 5.179678910307407e-05, 0.00037460651717180667, 0.0001954828657052586, 8.06685688205839e-05, 5.470561956357352e-05]], 'coarse_errors': [[9.82902021818834e-06, 1.5111837306063275e-05, 1.1569070240193357e-05, 1.248526371545369e-05, 1.9619199899337548e-05, 1.671070684300112e-05, 1.5143951802482853e-05, 1.4752306181568201e-05, 2.127067607155934e-05, 2.111954604517866e-05, 1.2883377763412643e-05, 1.5989235636624175e-05], [8.165939363195575e-06, 9.900704953285572e-06, 1.1755251360708266e-05, 1.3790555398898332e-05, 4.1347976266086455e-05, 3.157786470826323e-05, 2.0410152614628274e-05, 1.9025076107257034e-05, 4.341726705007058e-05, 2.8613844554913656e-05, 1.9312824337046737e-05, 2.2247871504834345e-05], [1.6237716551678942e-06, 2.894329401385495e-06, 2.3061387005189415e-06, 2.214038132751469e-06, 1.6839344065385282e-05, 6.905968514989231e-06, 5.990034755295816e-06, 5.459091753347899e-06, 1.36656904267763e-05, 8.553787631342808e-06, 4.2757873014874516e-06, 4.999367158809502e-06], [3.884136102021517e-06, 5.026458244020767e-06, 3.7736016923204838e-06, 4.022609265518012e-06, 2.494124863587521e-05, 1.5182996691383958e-05, 8.881059184977869e-06, 8.14969440651664e-06, 3.24532931551362e-05, 2.2456948064433633e-05, 1.0080847377353939e-05, 1.0755359760914486e-05], [2.7084746743363697e-06, 4.5558790162850825e-06, 2.7536869749701674e-06, 4.637459209884934e-06, 2.4669752257077628e-05, 1.4267017715503932e-05, 8.473883614325841e-06, 8.684939302308799e-06, 3.503223523724595e-05, 2.0875587201935092e-05, 1.1039881984247637e-05, 8.907668561142677e-06]]}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.twilight.multiple_scattered', 'approximations': {'scattering': 'split estimator: exact spherical first order + MC higher orders', 'solar_source': 'unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token', 'horizontal_field': 'first-order term only', 'ms_geometry': 'Monte Carlo assumes a horizontally uniform atmosphere', 'mc_refraction': 'none (straight-line photon transport)', 'ss_refraction': 'unified refracted solar source in the first-order term'}, 'source_refraction': True}}`

## l5_plane_-4  (L5 · plane-parallel world)

Wall time: 1.124458 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 94.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 94.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l6_clear_30  (L6 · clear (same contract))

Wall time: 3.106066 s

```text
approximation: hybrid-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.multiple.solve (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}; sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'hybrid-sos', 'n_gauss': 3, 'chunk': 512, 'sos': {}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.multiple.solve', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## l6_cloud_30  (L6 · +stratus deck)

Wall time: 3.687073 s

```text
approximation: hybrid-sos (method 'cloud')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.multiple.solve (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  cloud: rt.cloud.apply_cloud_overlay (source_refraction=True)
    geometry: plane-parallel homogeneous deck, delta-Eddington two-stream
    solar_source: unified finite disk, per-beam top/base/mixed classification with per-beam two-stream responses
    horizontal_field: inherited from the clear-sky solver for the clear part
    refraction: inherited from the unified solar source
effective: single {'n_gauss': 3, 'chunk': 512}; sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'hybrid-sos', 'cloud': 'water deck 0.5-1.5 km, tau550=20, r_eff=8 um, LWC=0.106 g/m3, n=200 cm^-3', 'include_clear': True, 'include_cloud': True, 'n_gauss': 3, 'chunk': 512, 'sos': {}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.multiple.solve', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'cloud': {'kernel': 'rt.cloud.apply_cloud_overlay', 'approximations': {'geometry': 'plane-parallel homogeneous deck, delta-Eddington two-stream', 'solar_source': 'unified finite disk, per-beam top/base/mixed classification with per-beam two-stream responses', 'horizontal_field': 'inherited from the clear-sky solver for the clear part', 'refraction': 'inherited from the unified solar source'}, 'source_refraction': True}}`
