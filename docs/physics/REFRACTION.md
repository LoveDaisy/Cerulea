# Atmospheric refraction

`skycolor/refraction.py` traces real rays through the refractive atmosphere.
It replaces the usual empirical fits (Saemundsson, Bennett) for two reasons:
those fits are only defined for objects *above* the horizon, and we need the
geometry of rays that graze or pass below it; and a fit cannot give the
altitude dependence, which is what controls the twilight terminator.

## Method

Rays obey the ray equation for a spherically symmetric medium

```
d/ds (n u) = grad n,        n = n(r),   grad n = n'(r) rhat
```

integrated with fixed-step RK4 in a 2-D Cartesian frame.  The refractive index
follows Gladstone-Dale, `n - 1 = (n0 - 1) rho / rho0`, with
`n0 - 1 = 2.78e-4` at 15 °C and 1013.25 hPa (Edlén, 550 nm).

Two numerical details matter:

* the density profile is interpolated with a **cubic spline**, not linearly.
  A piecewise-linear `dn/dr` has kinks on a 300 m scale and destroys the RK4
  convergence; with the spline the answer is stable to four digits from
  `ds = 4000 m` down to `ds = 500 m`.
* the shadow threshold is found from **Bouguer's invariant**,
  `n r sin(theta) = const`, which gives the grazing angle in closed form,
  `sin(theta) = n(R_b) R_b / (n(r0) r0)`, where `R_b = r_bounds[0]` is
  the opaque model bottom (taking the obtuse root, because the ray
  points down toward the horizon).  No bisection, no noise.

## Validation

| apparent elevation | integrated bending | Sæmundsson (1974) |
|---|---|---|
| 0° | 33.0′ | 29.0′ |
| 1° | 23.6′ | 21.7′ |
| 5° | 9.67′ | 9.67′ |
| 10° | 5.22′ | 5.41′ |
| 30° | 1.65′ | 1.75′ |
| 45° | 0.95′ | 1.01′ |
| 80° | 0.17′ | 0.18′ |

The agreement is within a few per cent above 5°, and at the horizon we get 33′
against the classical 34′.  (Sæmundsson's fit gives 29′ there; the 34′ figure
is a mean over temperature profiles, which range from 28′ to 40′.)

The shadow threshold, i.e. the extra solar depression for which a point at
altitude `z` is still sunlit, comes out as

| z (km) | 0 | 2 | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|---|
| extra depression | 0.55° | 0.69° | 0.82° | 0.90° | 0.95° | 0.98° | 1.01° |

## What is used where

* **Solar position** (`geometry.sun_position(..., refraction=True)`) and the
  horizon dip.
* **The radiative transfer can use the full refractive geometry**
  (`Atmosphere(refraction=True)`): the apparent solar elevation at every
  scattering point is looked up in `data/refraction.csv`, the Bouguer invariant
  `a = n r sin(theta_app)` is formed from it, and the per-shell path length is
  evaluated with the impact parameter `b_k = a / n_k`.

  That last step is the whole trick.  Inside a homogeneous shell a refracted
  ray is a straight chord, so

  ```
  straight:  L_k = sqrt(r_{k+1}^2 - p^2) - sqrt(r_k^2 - p^2),   p = r sin(theta)
  refracted: L_k = sqrt(r_{k+1}^2 - b_k^2) - sqrt(r_k^2 - b_k^2), b_k = a / n_k
  ```

  so bending the *whole optical path* costs one division per shell rather than
  a ray trace per photon.  The formula was checked against numerical path
  integration: a horizontal ray at the surface gives 1659 km analytically and
  1663 km numerically (0.1 %).

  Because `b_k -> p` wherever the ray does not graze, the correction is
  self-disabling: no region switching, no threshold, no wasted work.  (The
  earlier worry that it should only be applied near the horizon was right
  physically, but the closed form removes the need to encode it.)

## The measured effect, and the two-branch chord

Bending the paths raises the terminator by 0.5–1.0 degrees and makes twilight
brighter.  With the corrected two-branch chord (see below) the model gives,
for the `clear` preset at sea level:

| solar elevation | geometric | refracted solar-path model | Koomen 1952 | Patat 2006 |
|---|---|---|---|---|
| −3° | 16.0 (0.75 K) | 20.1 (0.93 K) | 21.5 | 40.1 † |
| −6° | 0.295 (0.46 K) | 0.444 (0.69 K) | 0.646 | 0.516 |

† Patat's −3° is an extrapolation.

`sun_path_lengths` follows the *whole* solar ray: for a point whose sun is
below the local horizontal the light goes through the perigee, so the optical
depth is the far branch (perigee → space) plus the near branch (perigee →
point).  An earlier implementation returned only one perigee → space traversal
(the "one-branch chord"); it under-attenuated every grazing twilight ray and
was the real source of the old 2.20× excess.  See the sixth addendum of
`validation/ADDENDUM.md`.

### The solar-altitude convention

If the reference's "solar elevation" is the **apparent** altitude, a data point
at −6° apparent corresponds to −6.55° true, and the model must be evaluated
there:

| reading | −3° | −6° |
|---|---|---|
| H1: reference axis is *true*, refraction on | 0.50× | 0.86× |
| H2: reference axis is *apparent*, refraction on | — | 0.36× |
| geometric model (refraction off, true axis) | 0.40× | 0.57× |

These are historical model sensitivities (ratios to Patat), not evidence of
which convention the observations used. The primary papers do not explicitly
settle true/apparent altitude or disk centre/limb, and better agreement cannot
establish that contract. Current experiments assume true solar-centre altitude
without fitting an axis shift. See [TWILIGHT_REFERENCE_AUDIT.md](../twilight/TWILIGHT_REFERENCE_AUDIT.md).
Refraction stays on by default; `refraction=False` reproduces geometric controls.
