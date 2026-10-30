# Multiple scattering

`skycolor.rt.multiple` adds all scattering orders `n >= 2` on top of the exact
first-order term of `skycolor.rt.single`.  This note describes the method, the
approximations, and the validation numbers obtained on branch `feat/ms`.

## Method

The solver is a **successive orders of scattering (SOS)** solver:

* **Vertical grid.**  The 54 layers of `Atmosphere.z_bounds` (ground to 120 km).
  Layer optical depths are `beta_ext * dz`; the extinction includes Rayleigh,
  aerosol, ozone and NO2.  The diffuse field is **plane-parallel**.
* **Angular grid.**  Gauss-Legendre nodes in `mu = cos(zenith)` (default 16,
  configurable) and a Fourier expansion in azimuth:

  ```
  L(tau, mu, phi) = sum_m L_m(tau, mu) cos(m phi)
  p(cos Theta)    = sum_m (2 - delta_m0) p_m(mu, mu') cos(m (phi - phi'))
  p_m             = sum_{l>=m} (2l+1) beta_l (l-m)!/(l+m)! P_l^m(mu) P_l^m(mu')
  ```

  The Legendre moments `beta_l` are the layer-averaged moments from
  `Atmosphere.phase_moments`, so Rayleigh and Mie are treated consistently.
  The azimuthal expansion is exact up to the mode/moment truncation; there is
  no `m = 0` approximation.  Twelve modes (default) are enough for the phase
  functions in this package.
* **Pseudo-spherical direct beam.**  For the point of the vertical column at
  radius `r` the solar ray has impact parameter `p = r sin(sza)` and direction
  cosine `mu_s = cos(sza)` (towards the sun).  Its transmittance is integrated
  along the true spherical ray with `geometry.path_lengths_sun` and
  `rt.single.species_columns`, and rays whose perigee falls inside the Earth
  are masked to zero.  This reproduces the twilight shadow boundary: at
  `sza = 96 deg` only the atmosphere above about 35 km is lit.
* **Beam source.**  The direct beam is a delta function in angle with radiance
  amplitude `F_sun * T(z)`.  Its contribution to the Fourier source is
  `(omega / 4 pi) F_sun T (2 - delta_m0) p_m(mu, mu_s)`.
* **Transfer.**  Each layer is integrated analytically assuming the source is
  constant across the layer.  The field is stored at layer mid-points; the
  midpoint value is obtained from the exact constant-source solution using
  `exp(-dtau / (2 |mu|))`.  Downward sweeps start at zero at the top of the
  atmosphere, upward sweeps at the Lambertian ground
  `L_g = (albedo/pi)(E_direct + E_diffuse_down)`.  Only the `m = 0` mode carries
  a flux; the ground therefore only feeds `m = 0`.
* **Orders.**  Order 1 is the scattering of the direct beam (the upward
  boundary also contains the ground reflection of the direct beam, so
  `beam -> ground -> scatter` paths are included).  Order `n` is the scattering
  of order `n-1` plus the ground reflection of the order `n-1` downward field.
  The returned multiple-scattering radiance is
  `view_integral(scattering(L_total))`, which is exactly `sum_{n>=2} L^(n)`.
  Iteration stops when the per-wavelength relative field change is below
  `tol = 1e-3` (default) or after 40 orders.
* **View integration.**  Instead of interpolating the discrete-ordinate field
  to the requested view angle, the scattering source is evaluated directly at
  the view direction with the Legendre expansion and integrated along the view
  ray.  This gives a smooth result in view angle and avoids interpolation
  ringing near the horizon.

## Approximations and limitations

* **Plane-parallel diffuse transport.**  The direct beam is spherical, the
  diffuse field is plane-parallel.  The first-order term from `rt.single`
  remains exactly spherical.  Consequences:
  - near the horizon the diffuse path is longer than the true spherical path
    (airmass 28.6 at 88 deg instead of about 19), but the radiance there is
    saturated (optically thick), so the error in the multiple-scattering
    contribution is modest;
  - for oblique sun the 1-D column energy budget differs from the true 3-D
    budget because the model cannot represent the horizontal divergence of the
    beam.  The energy closure test is therefore run with the sun near the
    zenith, where the pseudo-spherical beam reduces to the plane-parallel one.
* **Phase-function truncation.**  Default 32 Legendre moments.  For the
  aerosol models here (`g <= 0.75`) the truncation error of the phase function
  is below about 1e-4.  `n_moments` is capped at `2 * n_mu` so that the
  Gauss quadrature integrates `int p dmu = 2` exactly.
* **No polarisation, Lambertian ground, horizontal homogeneity** for the
  diffuse field (the `HorizontalField` only affects first-order scattering).
* **Order truncation.**  For the clear atmosphere the series converges in
  8-10 orders.  For a conservative atmosphere with albedo 1 and
  `tau ~ 0.4` about 80 orders are needed at `tol = 1e-10`; the default
  `max_orders = 40` is ample for real atmospheres but not for that stress
  test.

## Validation numbers

Unless stated otherwise: `presets.clear`, `n_mu = 32`, `n_modes = 12`,
`n_moments = 32`, `tol = 1e-3`, ground albedo 0.1.

| Check | Value |
|---|---|
| Empty atmosphere (no Rayleigh, no aerosol) | exactly 0 (early return) |
| `MS/SS` at zenith, 550 nm, `sza = 60` | 0.330 |
| `MS/SS` at zenith, 550 nm, `sza = 45` | 0.254 |
| `MS/SS` at `vza = 88 deg`, sun side (az 0), 550 nm, `sza = 45` | 0.415 |
| `MS/SS` at `vza = 88 deg`, az 90, 550 nm, `sza = 45` | 0.925 |
| Twilight `sza = 96`: zenith MS / SS / ratio at 550 nm | `2.42e-6` / `8.18e-6` / 0.296 |
| Energy closure, conservative atmosphere, albedo 0, `sza = 0` | 3.7e-4 |
| Energy closure, conservative atmosphere, albedo 1, `sza = 0`, `n_mu = 32` | 1.15e-3 |
| Energy closure, conservative atmosphere, albedo 1, `sza = 0`, `n_mu = 48` | 6.5e-4 |
| First-order check against `rt.single.sky_radiance` (vza <= 60, sza 60) | within 0.7 % |
| Runtime, 1000 views x 76 wavelengths | 0.58 s (`n_mu = 32`) |

The `MS/SS` ratio at the horizon on the sun side (0.42) is the headline number
for comparison with CIE 110 / turbidity-3 reference skies.  Note that
`presets.clear` has AOD(550) = 0.05, i.e. much cleaner than a turbidity-3
atmosphere; a hazier preset gives a larger ratio.

## Remaining work

* Test suite in `tests/test_multiple.py`.
* Optional: spherical view integration for a better horizon at very large
  viewing zenith angles.
