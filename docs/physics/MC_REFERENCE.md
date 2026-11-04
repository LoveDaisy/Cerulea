# Monte Carlo reference solver

`skycolor/rt/mc.py` is an **independent** Monte Carlo solver for the *total*
(all orders) spectral sky radiance of a spherically stratified atmosphere.  It
exists to validate the deterministic radiative-transfer code
(`skycolor.rt.single` + `skycolor.rt.multiple`): it shares no physics code with
it, only the pure geometry helpers `skycolor.geometry.path_lengths_sun`,
`is_sunlit` and `shell_lengths_upward`.

The design goal is *trust*, not speed: the estimator is a plain analog
backward path tracer with one variance-reduction technique (next-event
estimation towards the sun).  Everything else — absorption, phase-function
sampling, ground reflection — is sampled exactly from its physical
distribution.

```
mc_radiance(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=0.0, n_photons=200_000, seed=0,
            max_scatter=50) -> (radiance, stderr)
```

Both returned arrays have shape `(n_view, n_wl)` and units
W m⁻² sr⁻¹ nm⁻¹.

---

## 1. Estimator

For one wavelength, the radiance arriving at the camera along a view ray is
the sum over all scattering vertices `x_k` on the backward path of a
*next-event* contribution towards the sun:

```
L(Ω_v) = Σ_k  w_k · ω(x_k) · p_mix(Ω_v·Ω_s; x_k) · T_sun(x_k) · S/(4π)
```

with

| symbol | meaning |
|---|---|
| `w_k` | path throughput (1 at the first vertex, multiplied by the ground albedo at every ground bounce) |
| `ω = b_sca/b_ext` | single-scattering albedo at the vertex |
| `p_mix` | local phase function: Rayleigh + aerosol mixture, normalised to `∫p dΩ = 4π` |
| `T_sun = exp(-τ_sun)` | direct solar transmittance from the vertex to space (zero in the Earth's shadow) |
| `S(λ)` | extraterrestrial solar spectral irradiance [W m⁻² nm⁻¹] |

The free-flight distance is sampled from the physical extinction, so no
explicit `exp(-τ)` factor for the path to the camera appears: sampling a
collision at distance `t` with density `b_ext(t) exp(-τ(t))` and weighting the
collision by `b_sca/b_ext` is exactly the single-scattering integrand, and
repeating it after each scattering generates all higher orders.

At a collision the photon either scatters or is absorbed:

* the next-event estimate is evaluated **always**, with the deterministic
  albedo factor `ω` (this is a variance reduction, not an approximation);
* the path continues with probability `ω` (analog absorption), otherwise it
  is terminated;
* if it continues, a scattering species is chosen with probability
  `b_s/b_sca` and a new direction is drawn from that species' phase function.

This makes the estimator unbiased for any atmosphere consisting of
Rayleigh, aerosol and absorbing-gas layers.

### Ground

The surface is a Lambertian sphere of albedo `ρ(λ)` at `r = R_earth`.  In
backward tracing a path that reaches the ground contributes

1. a next-event term `w · (ρ/π) · S · μ_sun · T_sun` for the directly
   illuminated surface (`μ_sun = Ω_s·n > 0`), and
2. a continuation with weight `w·ρ` and a new direction sampled from the
   cosine-weighted hemisphere around the local normal.

Both are needed: (1) is the "sun → ground → camera without further
scattering" path, (2) is the diffuse reflection.  For a white Lambertian
surface the combination conserves energy exactly (see §5.2).

### Russian roulette and truncation

* Weight roulette: if `w < 0.05`, the path is killed with probability
  `1 - w/0.05`, otherwise `w` is reset to 0.05.  This only matters after
  several ground bounces.
* Path-length roulette: after 10 scattering events a path survives with
  probability 0.75 per additional event and its weight is divided by 0.75.
* `max_scatter` (default 50) is a hard cap on the number of atmospheric
  scattering events.  The next-event estimate at the last vertex is still
  included, so the result contains orders up to `max_scatter`; the omitted
  orders are negligible in the visible but are a small negative bias in very
  thick atmospheres.

---

## 2. Geometry and free-flight sampling

Rays are straight lines in three dimensions (no refraction) through
concentric spherical shells.  The free flight uses **delta tracking**:

* the atmosphere is divided into a few coarse layers (boundaries at
  offsets 1, 4, 12, 40 and 120 km above the model bottom, clipped to the
  domain and bracketed by its bottom and top);
* inside a layer a majorant `b_max` bounds the extinction;
* a step of length `-ln(ξ)/b_max` is taken.  If it crosses a layer boundary
  the photon moves to the boundary and continues with the new majorant; if
  not, a real collision is accepted with probability `b_ext(x)/b_max`
  (otherwise it is a null collision and the photon continues).

The extinction used by the sampler is tabulated on a fine height grid
(701 quadratically spaced points plus model/sampling boundaries) and
log-linearly interpolated.  Log-linear interpolation cannot overshoot between
two grid points, so the maximum of the tabulated field inside a layer is a
true majorant of the field actually being sampled.  The sampler is therefore
unbiased for the tabulated medium; the difference from the continuous model
is the interpolation error of the table (below 10⁻³ in the column, and much
smaller for the smooth profiles used here).

The sun transmittance at a vertex uses the exact spherical-shell chord
lengths from `path_lengths_sun`; the density of each shell is evaluated at the
geometric mean radius.  This is the same quadrature that `rt.single` uses, so
the two solvers share this (small, ~0.1 %) approximation.

### Phase-function sampling

* Rayleigh: the depolarised phase function is inverted analytically with
  Newton iterations on its CDF.
* Aerosol: the model's own `phase(mu, wl)` is tabulated on 2049 cosine
  points at the wavelength and sampled by inverse transform.  This works for
  both the analytic Henyey–Greenstein aerosols and the tabulated Mie ones.

The mixture is handled by first picking a species with probability
`b_s/b_sca` and then sampling that species' phase function, which is the
correct sampling distribution for the mixture.

---

## 3. Spectral treatment

**Wavelengths are completely independent.**  For every wavelength the solver
builds its own extinction table, layer majorants, single-scattering albedo,
phase tables and ground albedo.  There is no reference wavelength and no
spectral weighting.

Each wavelength gets an independent random stream derived from the user seed
with `numpy.random.SeedSequence(...).spawn(n_wl)`, so

* results are reproducible for a fixed seed and ordered query grid,
* child streams are assigned by wavelength **index**; slicing/reordering a grid
  can assign a different stream to the same physical wavelength,
* the different wavelengths are statistically independent.

The medium optics do remain invariant under wavelength restriction; this is a
physical contract, distinct from equality of individual MC samples. See
[AEROSOL_GRID_AUDIT.md](../twilight/AEROSOL_GRID_AUDIT.md).

---

## 4. Statistical error

`stderr` is the sample standard deviation of the per-path radiance estimator
divided by `sqrt(N_dir)`, where `N_dir` is the number of paths for that view
direction.  It is a **1-sigma statistical** error only; it does not include
systematic errors (solar-beam shell quadrature, table interpolation,
`max_scatter` truncation), which are discussed in §6.

`n_photons` is the **total** number of backward paths **per wavelength**; it
is split as evenly as possible over the requested view directions.  With one
direction all photons go to that direction; with 100 directions each gets
`n_photons/100`.  The error therefore scales as

```
stderr ∝ 1 / sqrt( n_photons / n_view )
```

Measured scaling (clear sky, 550 nm, sun 60°, zenith view, seed 5):

| `n_photons` | radiance | stderr | relative |
|---|---|---|---|
| 20 000 | 0.019383 | 0.000545 | 2.81 % |
| 80 000 | 0.019656 | 0.000271 | 1.38 % |
| 320 000 | 0.019762 | 0.000135 | 0.68 % |
| 1 280 000 | 0.019533 | 0.000067 | 0.35 % |

The ratios are 2.04, 2.02, 1.98 — the expected `1/sqrt(N)` behaviour.

`mc_radiance_split` returns the same total split into `single` (exactly one
atmospheric scattering, no ground bounce — the quantity computed by
`rt.single.sky_radiance`) and `multiple` (orders ≥ 2 plus any path that has
touched the ground).

`mc_flux` integrates the radiance over the upper or lower hemisphere at the
observer (`F = ∫ L μ dΩ`) with Gauss–Legendre quadrature in `μ` and a uniform
azimuth grid.  It is used by the energy-conservation test.

---

## 5. Validation

All tests live in `tests/test_mc.py` (`pytest tests/test_mc.py -s`, ~16 s).

### 5.1 Thin atmosphere vs the analytic single-scattering solver

`presets.pristine`, all optical depths scaled by 10⁻², black ground (the
deterministic solver has no ground reflection), sun zenith 60°,
`n_photons = 500 000`, seed 11.  Multiple scattering and the ground term are
both below 1 % of the signal here.

| view zenith | λ | MC total | `rt.single` | MC / analytic | z-score | MC rel. error |
|---|---|---|---|---|---|---|
| 0° | 400 nm | 4.706e-4 | 4.503e-4 | 1.045 | +1.06 | 4.1 % |
| 0° | 500 nm | 1.849e-4 | 2.044e-4 | 0.905 | −1.58 | 6.7 % |
| 0° | 550 nm | 1.244e-4 | 1.347e-4 | 0.924 | −1.06 | 7.8 % |
| 60° | 400 nm | 1.343e-3 | 1.410e-3 | 0.952 | −1.67 | 3.0 % |
| 60° | 500 nm | 5.963e-4 | 6.405e-4 | 0.931 | −1.59 | 4.7 % |
| 60° | 550 nm | 4.086e-4 | 4.222e-4 | 0.968 | −0.62 | 5.4 % |
| 85° | 400 nm | 6.534e-3 | 6.494e-3 | 1.006 | +0.48 | 1.3 % |
| 85° | 500 nm | 3.062e-3 | 2.977e-3 | 1.029 | +1.41 | 2.0 % |
| 85° | 550 nm | 1.999e-3 | 1.966e-3 | 1.017 | +0.70 | 2.3 % |

All nine points agree within 1.7σ; the largest deviations are the noisiest
points (the optical depth — and hence the number of collisions — is smallest
at the zenith, so the relative error is largest there).  Multiple scattering
is below 0.7 % of the signal in all cases, so the comparison is a genuine test
of the first-order radiometry.

The *first-order part* of the estimator was also compared with
`rt.single` in a **normal** (not thin) atmosphere — `presets.clear`, black
ground, `n_photons = 10⁶`, seed 21.  The reference here uses the converged
quadrature `n_gauss=24` because the default `n_gauss=3` is itself low by
0.2–0.3 % at these angles (see §6):

| λ | view zenith | MC single | `rt.single` (24) | ratio |
|---|---|---|---|---|
| 450 nm | 0° | 0.02790 | 0.027998 | 0.996 |
| 550 nm | 0° | 0.01497 | 0.014919 | 1.003 |
| 450 nm | 60° | 0.21612 | 0.216069 | 1.000 |
| 550 nm | 60° | 0.16544 | 0.165296 | 1.001 |

The agreement is 0.4 % or better, i.e. within the statistical error of these
runs.  Against the default `n_gauss=3` the ratios are 0.2–0.6 % higher, all
of which is the deterministic solver's quadrature error.

### 5.2 Energy conservation

A Rayleigh-only, ozone-free atmosphere is conservative.  The *globally
integrated* power budget must satisfy

```
P_up(TOA) + P_absorbed(surface) = S · π R_earth²
```

(energy conservation is a statement about the whole sphere; the local flux at
a single point is not conserved in spherical geometry because of horizontal
transport — see §6).  The integral is over the solar zenith angle with
Gauss–Legendre quadrature (12 points), each hemisphere integrated with
`mc_flux`, and the direct solar beam is added analytically.  550 nm,
`n_photons = 60 000` per solar position:

| ground albedo | P_up / P_inc | P_abs / P_inc | sum | 
|---|---|---|---|
| 0 (black) | 0.0780 | 0.9205 | **0.9985** |
| 1 (white) | 0.9913 | 0.0 | **0.9913** |

Both close to 1 within 1 %.  The residual is the angular quadrature of the
sharp radiance peak at the horizon (a pure vacuum + white sphere gives
0.98–1.01 depending on the quadrature), not a bias of the transport.

### 5.3 Magnitude of multiple scattering

`presets.clear`, 550 nm, sun zenith 60°, zenith view, `n_photons = 10⁶`:

```
total     0.01952  +/- 0.00005  W m^-2 sr^-1 nm^-1
single    0.01489  +/- 0.00004     (deterministic single scattering: 0.01488)
multiple  0.00462  +/- 0.00003
MS / SS = 31.0 %      MS / total = 23.7 %
```

With a black ground the ground-reflected contribution disappears and the
ratio drops to 22.9 % (total 0.01837, single 0.01495, multiple 0.00342).
That is at the upper end of (but consistent with) the 10–20 % figure in
`docs/ARCHITECTURE.md`; the extra 8 percentage points in the default case
come from the ground albedo of 0.1.
The exact number depends on the aerosol model and on the ground albedo, both
of which are inputs to the model rather than to the solver.

Zenith-angle dependence of the same case (albedo 0.1):

| view zenith | total | single | multiple | MS / SS |
|---|---|---|---|---|
| 0° | 0.01954 | 0.01490 | 0.00464 | 31.2 % |
| 30° | 0.04399 | 0.03777 | 0.00623 | 16.5 % |
| 60° | 0.17926 | 0.16561 | 0.01365 | 8.2 % |
| 80° | 0.25172 | 0.21619 | 0.03553 | 16.4 % |
| 88° | 0.36715 | 0.29829 | 0.06886 | 23.1 % |

Multiple scattering is smallest where the single-scattered radiance peaks
(around 60° for this geometry) and grows towards the horizon, as expected.

### 5.4 Runtime

Measured on a laptop (single core, NumPy 2.4), `presets.clear`:

| call | time |
|---|---|
| 1 direction × 1 wavelength × 10⁶ photons | ~1.9 s |
| 3 directions × 3 wavelengths × 5·10⁵ photons | ~1.0 s |
| 100 directions × 80 wavelengths × 2·10⁵ photons (default) | ~33 s |

The runtime is linear in `n_photons` and in the number of wavelengths, and
grows with the optical depth of the view direction: horizon directions
scatter many times and cost several times more than vertical ones.  For a
large viewing grid, lower `n_photons` (e.g. 5·10⁴ gives ~8 s for the
100 × 80 case).

---

## 6. Systematic issues and limitations

1. **Horizon quadrature of the deterministic code.**  The MC single-scattering
   part agrees with `rt.single` at 88° only after increasing its `n_gauss`
   parameter: at 88° `sky_radiance(n_gauss=3)` gives 0.28817, `n_gauss=24`
   gives 0.29642 and `n_gauss=48` gives 0.29744, while the MC gives 0.29844.
   The default 3-point Gauss quadrature of the deterministic solver is
   therefore ~3 % low at grazing angles, and part of any "MC vs analytic"
   difference near the horizon is a property of the deterministic reference,
   not of the Monte Carlo.

2. **Monte Carlo noise at the horizon.**  Horizon rays traverse a long path
   and scatter several times, so they are both the most expensive and the
   noisiest directions.  The statistical error bar already reflects this; use
   more photons if a precise horizon value is needed.

3. **Solar-beam quadrature.**  The direct solar transmittance uses one
   density value per atmospheric shell (geometric mean radius).  This is the
   same approximation as `rt.single`; its error is of order 0.1 % for the
   default 55 shells and is systematic (it does not average out with more
   photons).

4. **Truncation at `max_scatter`.**  Orders beyond `max_scatter` are omitted.
   For a clear sky at 550 nm the omitted fraction is far below 10⁻⁶, but in a
   thick or strongly multiple-scattering atmosphere (optical depth ≫ 1, or
   twilight) it can become visible.  `max_scatter=50` is conservative.

5. **Interpolated transport medium.**  The delta-tracking sampler transports
   photons through the log-linearly interpolated extinction table, while the
   single-scattering albedo and phase functions are evaluated from the same
   table.  The table is fine enough (700 points, quadratic spacing) that the
   column optical depth differs from the continuous model by <10⁻³ relative.

6. **Energy conservation is global, not local.**  In a spherical atmosphere
   the upward flux at a single point is not equal to the incident solar flux
   even for a conservative atmosphere and a white surface: the radiance field
   is non-local, and a vacuum sphere with a white surface already gives
   `F_up/S = 0.963` at the sub-solar point (the limb of the visible cap is
   dimmer).  The test therefore integrates over the whole sphere.  The
   statement "upward flux at TOA equals the absorbed solar flux" only holds
   in that globally integrated form (and with `P_abs` being the power
   absorbed by the surface).

7. **Ground reflection.**  `rt.single.sky_radiance` does not include the
   ground at all.  Comparisons with it must either use a black ground or the
   `single` split from `mc_radiance_split`; otherwise the (physically real)
   ground-reflected light biases the comparison high by several percent.

8. **Refraction, polarisation and thermal emission** are not modelled, in
   common with the rest of the package.
