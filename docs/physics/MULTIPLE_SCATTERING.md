# Multiple scattering

`skycolor.rt.multiple` adds all scattering orders `n >= 2` on top of the exact
first-order term of `skycolor.rt.single`.  This note describes the method, the
approximations, and the validation numbers obtained on branch `feat/ms`.

The plane first-order component is formally reachable through
`multiple.solve(..., single_view=True)` (`SOSResult.radiance_single`);
`rt.solver.solve(approximation="plane-sos")` returns both plane terms from
one SOS run.  `SOSResult` reports the effective discretisation actually used
(`n_mu` / `n_modes` / `n_moments_eff`, `tol`, `max_orders`, `delta_m`); the
defaults quoted below are the *request* values — effective values may be
clamped (even `n_mu`, truncated modes/moments) and are the authoritative
record.

## Method

The solver is a **successive orders of scattering (SOS)** solver:

* **Vertical grid.** `Atmosphere.z_bounds` (default ground to 200 km;
  site truncation starts at `z_bounds[0]`).
  Layer optical depths are `beta_ext * dz`; the extinction includes Rayleigh,
  aerosol, ozone and NO2.  The diffuse field is **plane-parallel**.
* **Angular grid.**  Gauss-Legendre nodes in `mu = cos(zenith)` (default 32,
  configurable) and a Fourier expansion in azimuth:

  ```
  L(tau, mu, phi) = sum_m L_m(tau, mu) cos(m phi)
  p(cos Theta)    = sum_m (2 - delta_m0) p_m(mu, mu') cos(m (phi - phi'))
  p_m             = sum_{l>=m} (2l+1) beta_l (l-m)!/(l+m)! P_l^m(mu) P_l^m(mu')
  ```

  With default `delta_m=True`, optical depths, albedos and Legendre moments
  use `Atmosphere.delta_m_properties(n_moments)`. With `delta_m=False`,
  `Atmosphere.phase_moments` supplies the raw layer-averaged moments.
  See [ARCHITECTURE.md](../ARCHITECTURE.md#delta-m-quantities-vs-raw-physics-e4).
  The azimuthal expansion is exact up to the mode/moment truncation; there is
  no `m = 0` approximation.  Twelve modes (default) are enough for the phase
  functions in this package (the result changes by < 0.1 % between 8 and 20
  modes).
* **Pseudo-spherical direct beam.**  For the point of the vertical column at
  radius `r` the solar ray has impact parameter `p = r sin(sza)` and direction
  cosine `mu_s = cos(sza)` (towards the sun).  Its transmittance is integrated
  along the true spherical ray with `geometry.path_lengths_sun` and
  shared `optics.shell_species_columns`. With refraction on, the shadow
  threshold refracts but this beam optical depth remains geometric. Occlusion
  uses the opaque model bottom, `r_bounds[0]`. In the geometric control, at
  `sza = 96 deg` only the atmosphere above about 35 km is lit.
* **Beam source.**  The direct beam is a delta function in angle with radiance
  amplitude `F_sun * T(z)` (the physical beam radiance).  Its contribution to
  the Fourier source is
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
  `tol = 1e-3` (default) or after 40 orders.  The clear and hazy presets
  converge in 6-13 orders; the conservative `albedo = 1` stress test needs
  about 60-80 orders at `tol = 1e-10`.
* **View integration.**  Instead of interpolating the discrete-ordinate field
  to the requested view angle, the scattering source is evaluated directly at
  the view direction with the Legendre expansion and integrated along the view
  ray.  This gives a smooth result in view angle and avoids interpolation
  ringing near the horizon.  A scalar zenith angle may be combined with an
  array of azimuths.

## Approximations and limitations

* **Plane-parallel diffuse transport.**  The direct beam is spherical, the
  diffuse field is plane-parallel.  The first-order term from `rt.single`
  remains exactly spherical.  Consequences:
  - near the horizon the diffuse path is longer than the true spherical path
    (airmass 28.6 at 88 deg instead of about 19), but the radiance there is
    saturated (optically thick), so the error in the multiple-scattering
    contribution is only a few per cent;
  - for oblique sun the 1-D column energy budget differs from the true 3-D
    budget because the model cannot represent the horizontal divergence of the
    beam.  The energy closure is 5.3e-4 at `sza = 0` and 8.7e-4 at
    `sza = 30` (48 streams), but degrades to about 4e-3 at `sza = 60`.  The
    energy test is therefore run with the sun near the zenith, where the
    pseudo-spherical beam reduces to the plane-parallel one.
* **Phase-function truncation.**  Default 32 Legendre moments.  For the
  aerosol models here (`g <= 0.75`) the truncation error is of order 1e-4 in
  the phase function (the forward peak is the most sensitive part).
  `n_moments` is capped at `2 * n_mu` so that the Gauss quadrature integrates
  `int p dmu = 2` exactly; this is what makes the `m = 0` operator energy
  conserving.  Convergence with moments is complete by 16 moments for the
  presets in this package.
* **No polarisation, Lambertian ground, horizontal homogeneity** for the
  diffuse field (the `HorizontalField` only affects first-order scattering).
* **Angular quadrature.**  The `albedo = 1` conservative stress test improves
  from 3.8e-3 at 16 streams to 1.2e-3 at 32 and 5.3e-4 at 48 streams.  The
  production radiance is much less sensitive: MS changes by 0.5 % between 32
  and 48 streams.

## Validation numbers

Unless stated otherwise: `presets.clear`, monochromatic 550 nm, default
`n_mu = 32`, `n_modes = 12`, `n_moments = 32`, `tol = 1e-3`, ground albedo 0.1.

| Check | Value |
|---|---|
| Empty atmosphere (no Rayleigh, no aerosol) | exactly 0 (early return) |
| `MS/SS` at zenith, `sza = 45` | 0.241 |
| `MS/SS` at zenith, `sza = 60` | 0.314 |
| `MS/SS` at `vza = 88`, sun side (az 0), `sza = 45` | **0.400** |
| `MS/SS` at `vza = 88`, az 90, `sza = 45` | 0.895 |
| `MS/SS` at `vza = 88`, sun side (az 0), `sza = 60` | 0.245 |
| Twilight `sza = 96`: zenith MS / SS / ratio | `2.19e-6` / `7.94e-6` / 0.276 |
| Energy closure, conservative, albedo 0, `sza = 0`, `n_mu = 48` | 2.3e-4 |
| Energy closure, conservative, albedo 1, `sza = 0`, `n_mu = 48` | 5.3e-4 |
| Energy closure, conservative, albedo 1, `sza = 0`, `n_mu = 32` | 1.2e-3 |
| First-order check against `rt.single.sky_radiance` (`sza = 60`, `vza <= 60`) | within 0.7 % |
| Runtime, 1000 views x 76 wavelengths | 0.44 s |
| Runtime, 5000 views x 76 wavelengths | 1.03 s |

The **headline number** requested for comparison with CIE 110 /
turbidity-3 reference skies is the horizon value: for `presets.clear` at
550 nm with `sza = 45`, looking at `vza = 88` on the sun side,
`MS/SS = 0.400`, i.e. multiple scattering adds 40 % to the single-scattered
radiance at the limb.

The ratio depends strongly on the aerosol load.  At `sza = 45`, zenith:

| atmosphere | AOD(550) | MS/SS |
|---|---|---|
| `presets.clear` | 0.05 | 0.241 |
| `presets.clear(aod=0.2)` | 0.20 | 0.335 |
| `presets.hazy` | 0.30 | 0.448 |

A turbidity-3 atmosphere is closer to AOD 0.2 than to the very clean default
`clear` preset, which explains why the `clear` ratio is at the lower edge of
the 0.25-0.45 target band while the AOD 0.2 case sits in the middle of it.

On the sun side the `MS/SS` ratio is non-monotonic in viewing zenith angle:
single scattering has a strong forward peak there (0.08 at `vza = 60`,
`sza = 60`), so the ratio first falls and then rises again towards the limb.
Away from the sun (azimuth 90 deg) it increases monotonically from 0.31 at the
zenith to 0.86 at `vza = 88`.

## Tests

```
.venv/bin/python -m pytest tests/test_multiple.py -q
```

12 tests cover the five acceptance criteria plus: the first-order regression
against `rt.single`, an optically thick horizon convergence check, and the
scalar-zenith / array-azimuth API.
