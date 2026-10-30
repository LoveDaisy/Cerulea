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
  `sin(theta) = n(R) R / (n(r0) r0)` (taking the obtuse root, because the ray
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
  horizon dip are unambiguous and are used as described.
* **The radiative-transfer shadow mask** (`Atmosphere(refraction=True)`) is
  available but **off by default**, for a reason worth recording.

## Why the shadow mask is off by default

Turning it on raises the terminator, which is physically right, but it lets the
single-scattering integrator admit rays that geometrically graze the surface.
Those rays must be traced along their *bent* path, and their optical depth is
then enormous (two passes through the whole boundary layer, τ ≈ 7).  Keeping
straight rays while shifting only the mask computes the optical depth of a path
that passes *below* the surface, which is meaningless, and the result is that
twilight comes out 3–6× too bright:

| solar elevation | geometric | refraction mask on | Patat et al. (2006) |
|---|---|---|---|
| −3° | 41.2 cd m⁻² | 58.4 | 40.1 |
| −6° | 0.79 | 3.02 | 0.516 |

So: refraction needs the **paths** bent, not just the mask moved.  That is the
honest next step (it means integrating the sun-ray optical depth along the
refracted chord), and `Atmosphere(refraction=True)` is left in place as the
hook for it.  Until then the geometric shadow, which reproduces the Patat fit
to +52 % at −6°, is the better approximation.
