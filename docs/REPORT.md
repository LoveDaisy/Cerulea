# The colour of the sky, from first principles

*This is the main write-up of the project.  Numbers quoted here are produced by
the scripts in `experiments/` and `validation/`; the figures live in
`figures/`.*

## 1. The question

Ignore clouds.  What decides the colour of the sky?

Four things, and only four:

1. **The sun's spectrum and where it is.**  A solar spectrum that is roughly
   black at 5000 K, attenuated by an amount that depends on the path length
   through the air — which is why sunset is red.
2. **What the air is made of.**  Nitrogen and oxygen molecules scatter
   (Rayleigh), ozone absorbs in the yellow-red, aerosol particles scatter and
   absorb (Mie).
3. **The geometry.**  The Earth is round, and so is the atmosphere.  This is
   not a detail: without it there is no twilight and no earth's shadow, and
   the horizon luminance comes out wrong by a large factor.
4. **The number of times light scatters.**  Once gives the blue sky.  Twice and
   more gives the twilight glow, the brightness of a hazy horizon, and the fact
   that the sky is brighter than a first-order calculation says.

The model implements all four, and then turns spectral radiance into colour
through an explicit, auditable chain.

## 2. The model

### 2.1 Atmosphere

A spherically stratified atmosphere between 0 and 120 km, with 54 layers.
Temperature and pressure follow the US Standard Atmosphere 1976; the air
number density follows from the ideal gas law.  Ozone uses a mid-latitude
climatological profile normalised to a column in Dobson units (default
300 DU).  NO₂ is available but off by default.  Aerosol is described by one or
more layers, each carrying an optical depth at 550 nm, a normalised vertical
profile (exponential or Gaussian), and an optional horizontal field.

### 2.2 Scattering

* **Rayleigh**: the Bodhaine et al. (1999) formulation with Edlén's refractive
  index and a depolarisation factor of 0.0279.  This gives
  σ(550 nm) = 4.507 × 10⁻³¹ m² and a sea-level vertical optical depth of
  0.0968.
* **Ozone**: JPL-2006 absorption cross sections (Chappuis band peak
  5.1 × 10⁻²¹ cm² at 600 nm).
* **Aerosol**: two families.  An analytic one (Ångström spectral dependence +
  Henyey-Greenstein phase function) for sweeps, and a physical one: log-normal
  size distributions integrated with Mie theory (Bohren & Huffman), which
  produces Q_ext, Q_sca, g and the phase function from a refractive index.
  The physical model is what makes the difference between haze (fine,
  α ≈ 1.5), dust (coarse, α ≈ 0.2) and smoke (absorbing, α ≈ 1.9) emerge
  rather than being asserted.

### 2.3 Radiative transfer

**First order — exact, spherical.**  For each viewing direction the model
integrates

```
L1(lambda) = (F0(lambda)/4pi) * integral beta_sca(s) p(Theta) T_sun(s) T_view(s) ds
```

along the true straight ray through the spherical shells.  Both transmittances
are computed on the true curved paths: `T_sun` is the transmittance from the
scattering point to the sun along a ray that may graze or dip below the Earth's
limb.  This single detail is what produces

* the **twilight arch** (the geometric shadow boundary is a curve on the sky,
  not a straight line),
* the fact that at solar elevation −6° the zenith is still lit by air at
  20–40 km while the ground is dark,
* the **belt of Venus** geometry.

The implementation uses a change of variables to the ray's *perigee distance*,
which makes every per-shell path length exact (no sampling error in the
geometry) and the twilight shadow boundary appear at the right place.

**Higher orders — successive orders of scattering.**  Orders ≥ 2 are computed
with a successive-orders solver: a plane-parallel diffuse field discretised
with Gauss-Legendre nodes in μ and a Fourier expansion in azimuth (the DISORT
structure), a pseudo-spherical direct beam (so the shadow is handled layer by
layer), a truncated Legendre expansion of the phase function, and a Lambertian
ground.  Orders are summed until the relative change falls below 10⁻³.

The split is deliberate: **first order is exact and spherical** (which is where
the sharp geometry lives), while **higher orders are plane-parallel** (they are
smooth and diffuse, so the plane-parallel approximation is much less costly
there).

### 2.4 Refraction

`skycolor/refraction.py` traces real rays through the refractive atmosphere
instead of using an empirical fit, because the fits are only defined above the
horizon and we need the geometry of rays that graze it.  The ray equation
`d/ds (n u) = grad n` is integrated with RK4; the density gradient comes from a
cubic spline (a piecewise-linear gradient destroys the convergence), and the
grazing angle is obtained in closed form from Bouguer's invariant
`n r sin(theta) = const`.

The integrated refraction is 33 arcminutes at the horizon and matches
Saemundsson's fit to a few per cent above 5 degrees.  It is used for the solar
position and for the dip of the horizon.

It is **not** used for the shadow mask in the radiative transfer, and that is a
deliberate, documented choice: raising the terminator while keeping straight
rays admits surface-grazing rays whose optical depth should be computed along
their *bent* path, and the result over-brightens twilight by 3–6×
(see `docs/REFRACTION.md`).  Bending the paths is the next step.

### 2.5 From spectrum to pixels

1. **Colorimetry**: integrate against the CIE 1931 2° colour matching
   functions.
2. **Adaptation** (optional): Bradford chromatic adaptation to a chosen white
   point.  This matters because the eye adapts, and because a camera's white
   balance is a choice — a sunset photographed with daylight balance looks
   orange, photographed with auto white balance it looks cyan.
3. **Tone mapping**: the sky spans four orders of magnitude in luminance and a
   display spans two.  The default is an ACES filmic curve with an automatic
   exposure; the chromaticity is preserved while the luminance is compressed.

## 3. Does it work?

See `validation/REPORT.md` for the full table.  Headlines:

| Test | Result |
|---|---|
| Rayleigh optical depth vs Bodhaine (1999), 300–700 nm | RMS 0.48 %, worst 1.0 % → **pass** |
| Direct normal illuminance vs Bird & Hulstrom / ASTM G173 | +4 to +6 % → **pass** |
| Diffuse horizontal illuminance vs CIE 110 (T = 2.2) | +38 %, but 9.5 klx = 86 W m⁻² is a normal measured clear-sky DHI → **reference is the weak link** |
| CIE 110 clear-sky relative luminance, θ ≤ 60° | 3.8 % RMS single-scattering only; 14.5 % with multiple scattering (dominated by one point, the solar aureole) |
| CIE 110 relative luminance, last 5° above the horizon | model ≈ 2× too bright → **known deviation** |
| CIE 110 relative luminance at the sun's position | model too weak (the HG aureole has no forward peak) → **known deviation** |
| Single scattering vs the analytic plane-parallel Rayleigh solution | 1.5–4 % → **pass** |
| Single scattering vs Monte Carlo (thin atmosphere, 9 points) | within 1.7σ → **pass** |
| Diffuse / direct ratio, pure Rayleigh atmosphere | 0.056 vs 0.061 analytic → **pass** |
| SOS energy conservation (conservative atmosphere) | 2.3 × 10⁻⁴ → **pass** |
| MC global energy conservation | 0.9985 → **pass** |
| MS/SS at the zenith, sun at 60°, clear sky | 0.31 (SOS) vs 0.31 (MC) → **pass** |
| Twilight zenith luminance at −6°, h = −6° | 0.567 vs 0.516 cd m⁻² (Patat et al. 2006) → +10 % → **pass** |
| Twilight zenith luminance at −9° and below | 20×–2000× too dark → **known failure**, see below |
| Horizon chromaticity vs measured Berlin spectra | Δxy = 0.018 (tolerance 0.02) → **marginal pass** |
| Twilight zenith luminance at −9°, spherical engine | 0.0238 vs 0.0171 cd m⁻² (1.39×) → **pass** |
| Twilight zenith luminance at −12°, spherical engine | 2.1e-3 vs 1.46e-3 cd m⁻² (1.45×) → **pass** |
| Mie phase moments vs quadrature | 2e-8, `beta_1` = g to 1e-10 → **pass** |
| Cloud albedo vs Monte Carlo | 0.1–0.7 % (`g=0`), ≤1 % (`g=0.85`, τ≥5) → **pass** |
| Cloud `R + T + direct` | 1 to 1e-12 → **pass** |
| Refraction at the horizon | 33′ (Sæmundsson 29′, classical 34′) → **pass** |

The two deviations are understood and documented rather than papered over:
the last few degrees above the horizon are where the plane-parallel diffuse
approximation and the aerosol vertical profile matter most, and the Henyey-
Greenstein phase function cannot reproduce the sharp forward diffraction peak
that makes the solar aureole.

### 3.1 The deep twilight needs a spherical solver

The single biggest failure of the deterministic model is twilight below −8°.
The reason is geometric: at −9° the zenith line of sight is entirely inside
the Earth's shadow, and the light that reaches it has travelled several
hundred kilometres sideways from the sunlit twilight arch.  A horizontally
homogeneous plane-parallel solver cannot represent that; a spherical
three-dimensional one can.

`skycolor/rt/twilight.py` runs the existing Monte Carlo as a production
engine with a split estimator (the first order is taken from the exact
deterministic solver, only orders ≥ 2 from the Monte Carlo), a coarse angular
grid with a spherical local-linear regression, and wavelength interpolation.
Against Patat et al. (2006):

| solar elevation | deterministic | engine | Patat | engine / Patat |
|---|---|---|---|---|
| −3° | 37.6 | 37.8 | 40.1 | 0.94 |
| −6° | 0.74 | 0.98 | 0.516 | 1.89 |
| −9° | 9.6 × 10⁻⁴ | 0.024 | 0.0171 | 1.39 |
| −12° | 9.8 × 10⁻⁷ | 2.1 × 10⁻³ | 0.00146 | 1.45 |

So the deepest twilight goes from being 24× and 1500× too dark to within a
factor of 1.5, at the cost of a few minutes of Monte Carlo per sky.

## 4. What the model says

### 4.1 A day

As the sun rises the sky brightens by four orders of magnitude and its
chromaticity traces a path from deep blue through white to orange and back.
The zenith stays blue all day but becomes *less* saturated near noon: at low
sun the light reaching the zenith has travelled a long slanted path and lost
much of its blue to Rayleigh extinction, while the remaining blue is
concentrated near the zenith and the horizon turns white or warm.

### 4.2 Sunset

At sunset the direct beam travels 10–40 air masses.  Blue light is removed by
Rayleigh scattering (σ ∝ λ⁻⁴·¹) and, at the red end, ozone's Chappuis band
removes the yellow.  The result is the characteristic orange-red.

### 4.3 Twilight and the blue hour

Three separate mechanisms produce the twilight sky:

1. **The earth's shadow** rises through the atmosphere.  The boundary of the
   shadow is a curve on the sky — high in the anti-solar direction, low in the
   sun's direction — and it is visible as the dark blue-grey edge below the
   belt of Venus.
2. **Ozone** continues to absorb the 500–700 nm light along the long slant
   path through the stratosphere.  With 300 DU the twilight zenith sits at
   (x, y) ≈ (0.25, 0.26); with no ozone it would be (0.33, 0.35), a neutral
   grey.  The blue hour is, quite literally, an ozone effect.
3. **Multiple scattering** keeps the shadowed part of the sky from going
   black: light scattered in the still-sunlit upper atmosphere diffuses down
   into the shadow.

The last of these is where the model stops working.  At −6° the zenith
luminance is within 10 % of the empirical fit of Patat et al. (2006), but at
−9° it is 20× too dark and at −12° it is essentially zero.  The reason is
geometric rather than radiative: the zenith line of sight at −9° is entirely
inside the Earth's shadow, and the light that actually reaches it has travelled
several hundred kilometres *horizontally* from the sunlit twilight arch.  A
horizontally homogeneous, plane-parallel multiple-scattering solver has no
mechanism for that transport.  Reproducing deep twilight needs a spherical
three-dimensional multiple-scattering model.

### 4.4 Air

The aerosol optical depth, the Ångström exponent and the single-scattering
albedo each move the colour in a different direction:

* increasing AOD **whitens the horizon** and desaturates the zenith;
* a low Ångström exponent (dust) makes the extinction nearly grey, so the sky
  keeps its brightness but loses contrast;
* an absorbing aerosol (smoke) darkens the sky and shifts it towards brown;
* humid haze (high RH, large particles) is nearly conservative and produces
  the classic milky white horizon with a blue zenith.

### 4.5 Near weather and far weather

A horizontally homogeneous model cannot produce the most recognisable summer
sky: deep blue overhead, milky white at the horizon.  Because every scattering
point knows its horizontal distance from the observer, the model can put a
clean air mass over the observer and a hazy one 100 km away (or the reverse).
The difference in the horizon chromaticity is comparable to changing the AOD
by a factor of ten, while the zenith stays clean.

### 4.6 Where a red afterglow comes from

"Red sky filling the whole sky" is not produced by haze near the ground.  A
tropospheric layer is in shadow as soon as the sun sets, so it can only
*subtract* light: at solar elevation −3°, a boundary-layer aerosol of AOD 0.2
drops the luminance at 3° elevation from 605 to 56 cd m⁻² and moves the
chromaticity from (0.427) to (0.319), i.e. dimmer and whiter.

What produces the effect is a scattering layer **above the earth's shadow**.
Sweeping the altitude of a 0.15 AOD layer at fixed solar elevation:

| layer altitude | chromaticity x at 8° elevation | luminance [cd m⁻²] |
|---|---|---|
| none | 0.34 | 44 |
| 3 km | 0.35 | 47 |
| 9 km (cirrus) | 0.36 | 55 |
| 15 km | 0.40 | 80 |
| 22 km | 0.52 | 250 |
| 30 km | 0.55 | 320 |

A 22 km layer with AOD 0.35 keeps the whole western sky glowing deep red down
to −7°, when the same sky without it is already dark (`08_volcanic_twilight.png`).
That is exactly the mechanism behind the vivid twilights that followed
Krakatoa (1883) and Pinatubo (1991): a stratospheric sulphate layer, still
sunlit long after the ground is in shadow.

The same layer produces Bishop's ring, because the ring is the diffraction
aureole of the same particles.

### 4.7 The blue sun

A blue sun is real physics, not a white-balance artefact.  It needs an
**extinction spectrum that is not monotonic**: more extinction in the red than
in the blue.  That happens when the particle radius is close to half the
wavelength and the size distribution is narrow enough for the Mie ripple to
survive.

Sweeping radius and width at AOD 1.0 (`09_blue_sun.png`):

| r_g [µm] | σ_g | τ(450)/τ(650) | beam chromaticity |
|---|---|---|---|
| 0.45 | 1.05 | **0.71** | (0.280, 0.300) — blue |
| 0.45 | 2.00 | 1.03 | (0.335, 0.347) — neutral |
| 0.25 | 1.10 | 1.53 | (0.369, 0.365) — red |
| Ångström α = 1.3 | — | 1.9 (by construction) | (0.378, 0.380) — red |
| no aerosol | — | 1.0 | (0.335, 0.347) |

An Ångström power law is monotonic by construction and can **never** make the
beam bluer; only Mie theory can.  For desert dust the effect is much weaker
because the particles are large and broadly distributed, so the extinction is
nearly grey: the blue sun of a dust storm is mostly a contrast effect (the
surrounding sky is butterscotch, so a merely less-orange sun reads as blue —
this is what happens on Mars), while the blue sun of a smoke or volcanic
aerosol is the real anomalous-extinction effect.

### 4.8 Clouds

A uniform cloud deck is grey and conservative, so its colour is borrowed from
the illuminant: white under a high sun, orange at sunset.  What the model adds
is the geometry of *which part of the deck is still lit*.  The shadow boundary
is at `z = R(1/sin θ − 1)`, i.e. 0.97 km at −1°, 3.9 km at −2°, 8.7 km at −3°.
So a stratus deck stops glowing at −1.2°, an altostratus at −2.3° and a cirrus
at −3.2°.  The altostratus base goes from (0.37, 0.38) at +5° to (0.60, 0.39)
at −2.2°, with the luminance falling by five orders of magnitude over the same
range.

### 4.9 Altitude

From 3 km the zenith gets darker and deeper blue (less air above), the horizon
is depressed, and the whole sky gains contrast.

## 5. Limitations

* **No clouds.**  By construction.
* **No polarisation.**  The Rayleigh phase function is used in its scalar form;
  the polarised sky (and the neutral points, Babinet and Brewster) would need a
  Stokes-vector treatment.
* **Plane-parallel multiple scattering.**  Higher orders ignore the Earth's
  curvature.  At a viewing zenith angle of 88° the plane-parallel airmass is
  28.6 against a true spherical value of about 19, so the limb's diffuse
  radiance is over-estimated by roughly 50 %.  Away from the horizon the error
  is small.
* **Deep twilight in the deterministic model.**  Below about −8° the failure
  is geometric: the light reaching the zenith has to travel horizontally from
  the sunlit twilight arch, which a plane-parallel multiple-scattering solver
  cannot represent.  `skycolor/rt/twilight.py` fixes it with a spherical
  Monte Carlo (see §3.1) at the cost of running the diffuse field as a Monte
  Carlo rather than a deterministic solve.
* **Refraction is only half-implemented.**  The solar position and the horizon
  dip use traced rays, but the radiative-transfer shadow mask does not: bending
  the terminator while keeping straight rays over-brightens twilight by 3–6×.
  Doing it properly means integrating the sun-ray optical depth along the
  refracted chord.
* **Henyey-Greenstein aureole.**  The solar aureole is too smooth.
* **Temperature-independent Chappuis band** (the JPL table has no temperature
  dependence in the visible).
* **Lambertian ground.**

## 6. What is interesting

* The colour of the sky is an *integral* over a spectral distribution that is
  itself shaped by two different absorption bands and one power law.  Small
  changes in the ozone column move the twilight colour more than small changes
  in the aerosol loading.
* The horizon is where every approximation hurts.  In a clear sky the optical
  depth along the line of sight goes from 0.1 at the zenith to almost 4 at the
  horizon: the same sky is optically thin and optically thick at the same time.
* The fact that the sky is blue and not violet is *not* simply because of the
  λ⁻⁴ law — the sun's spectrum, ozone absorption and the eye's sensitivity all
  contribute.  This model lets you turn each of them off in turn.
