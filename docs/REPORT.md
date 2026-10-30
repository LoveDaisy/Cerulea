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

### 2.4 From spectrum to pixels

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
| Diffuse horizontal illuminance vs CIE 110 | +6 to +10 % → **pass** |
| CIE 110 clear-sky relative luminance, θ ≤ 60° | RMS 3–4 % → **pass** |
| CIE 110 relative luminance, last 5° above the horizon | model ≈ 2× too bright → **known deviation** |
| CIE 110 relative luminance at the sun's position | model too weak (aureole under-resolved) → **known deviation** |
| Single scattering vs the analytic plane-parallel Rayleigh solution | 1.5–4 % → **pass** |
| Diffuse / direct ratio, pure Rayleigh atmosphere | 0.056 vs 0.061 analytic → **pass** |
| SOS energy conservation (conservative atmosphere) | 2.3 × 10⁻⁴ → **pass** |
| Monte Carlo reference vs the deterministic model | see `docs/MC_REFERENCE.md` |

The two deviations are understood and documented rather than papered over:
the last few degrees above the horizon are where the plane-parallel diffuse
approximation and the aerosol vertical profile matter most, and the Henyey-
Greenstein phase function cannot reproduce the sharp forward diffraction peak
that makes the solar aureole.

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

### 4.6 Altitude

From 3 km the zenith gets darker and deeper blue (less air above), the horizon
is depressed, and the whole sky gains contrast.

## 5. Limitations

* **No clouds.**  By construction.
* **No polarisation.**  The Rayleigh phase function is used in its scalar form;
  the polarised sky (and the neutral points, Babinet and Brewster) would need a
  Stokes-vector treatment.
* **Plane-parallel multiple scattering.**  Higher orders ignore the Earth's
  curvature; this is a few per cent except near the horizon.
* **120 km top.**  Astronomical twilight (sun below −10°) is dominated by
  scattering above 120 km; the model therefore goes dark too fast there.
* **No refraction.**  The sun's apparent position is not raised by ~0.5° at the
  horizon, so the twilight timing is off by a couple of minutes.
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
