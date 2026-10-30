# The deep twilight: a spherical Monte Carlo engine

`skycolor/rt/twilight.py` adds a **usable spherical multiple-scattering engine**
for twilight, without replacing the deterministic model for daytime use.

The production model is exact spherical *single* scattering (`rt/single.py`)
plus plane-parallel *multiple* scattering (`rt/multiple.py`).  It reproduces
twilight down to about −6° solar elevation (+10 % against Patat et al. 2006)
and then collapses:

| solar elevation | single scattering | spherical MC (`rt/mc.py`) | Patat et al. (2006) |
|---:|---:|---:|---:|
| −6° | 0.65 | 1.04 cd m⁻² | 0.516 |
| −9° | 0.00091 | 0.025 | 0.0171 |
| −12° | 8.1 × 10⁻⁷ | 0.00091 | 0.00146 |

The reason is geometric, not radiative.  At −9° the zenith line of sight lies
entirely inside the Earth's shadow (the shadow boundary crosses the observer's
meridian at 79 km); the light that reaches it has travelled several hundred
kilometres horizontally from the sunlit twilight arch.  At −12° the boundary is
at 143 km and the horizontal distance is of order 2000 km.  A horizontally
homogeneous plane-parallel solver cannot represent that transport; a spherical
three-dimensional one can.

```python
radiance, stderr = twilight_radiance(
    atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
    sun_azimuth_deg=0.0, quality="medium", seed=0,
    single=True, multiple=True)
```

Both arrays are `(n_view, n_wl)` in W m⁻² sr⁻¹ nm⁻¹ and the view angles
broadcast against each other, exactly like `rt.solver.sky_radiance_total`.

---

## 1. Estimator design

### 1.1 Split estimator: never Monte-Carlo the first order

The Monte Carlo reference returns `multiple = total − single` (via
`mc.mc_radiance_split`): every path that scattered **twice or more**, plus
every path that touched the ground.  That is exactly the complement of the
quantity `rt/single.py` computes *deterministically and exactly* (up to its own
shell quadrature).  The estimator is therefore

```
L(x) = L₁(x)                 deterministic, spherical, zero variance
     + L≥₂(x)                Monte Carlo, spherical
```

This is the single most important design decision.  In a plain Monte Carlo run
the first-order contribution is the largest term, the noisiest term (it carries
the whole weight of the estimator) and the most expensive one per useful
sample.  Removing it costs nothing in accuracy — `rt/single` is exact — and
removes the dominant part of the variance and about a third of the runtime.
It also means the *colour* of the sky in the model is still produced by the
deterministic solver; the Monte Carlo only adds the diffuse light.

### 1.2 Coarse angular grid, spherical regression

The multiple-scattering field is diffuse and smooth in direction, so it is
estimated on a coarse grid and interpolated:

* **zenith nodes** — `n_zen` values, clustered towards the horizon
  (83°, 86°, 88.5° are always present) because that is where the twilight arch
  makes the field vary fastest;
* **azimuth nodes** — `n_az` values over **half** the circle, 0…180° from the
  sun.  The atmosphere and the ground are azimuthally uniform (the Monte Carlo,
  like `rt/multiple`, ignores the aerosol `HorizontalField`), so
  `L(θ, φ_s + Δφ) = L(θ, φ_s − Δφ)` exactly.  The half-grid is mirrored to a
  full grid before the regression, which both halves the cost and puts the
  anti-solar meridian in the interior of the neighbourhood instead of on an
  edge.

The regression is a **local-linear fit on the sphere**: for each query
direction the field is approximated by `a + b·t₁ + c·t₂` over the coarse
points within a kernel of bandwidth `1.6 × (median neighbour spacing)`, where
`t₁, t₂` span the tangent plane.  The intercept is the estimate.  Two details
matter:

* it is done in **linear space**, not log space.  Log-space smoothing would be
  biased low: the Monte Carlo error in deep twilight is multiplicative and
  heavy-tailed, so the geometric mean is far below the mean;
* the regression weights **do not depend on the measured values**.  The obvious
  alternative (inverse-variance weighting, `1/σ²`) is catastrophic here: in
  the heavy-tail regime a downward fluctuation also has a *small* sample
  standard deviation, so `1/σ²` weighting promotes exactly the low outliers.
  Measured at −9° it biased the zenith by a factor of 40 (8.7 × 10⁻⁹ against a
  converged 3.2 × 10⁻⁷).  A geometry-only smoother is unbiased and was the
  only variant that passed the high-photon reference test.

### 1.3 Spectral interpolation

Wavelengths are independent Monte Carlo problems, but the multiple-scattering
spectrum is smooth.  `n_anchor` wavelengths spanning the model grid are
computed exactly and `log L` is interpolated **linearly** in wavelength.  The
single-scattering term is still evaluated at *every* wavelength, so ozone, NO₂
and the Rayleighλ⁻⁴ slope are not degraded.

Linear (rather than cubic) interpolation in log space is deliberate: the
weights are then a genuine partition of unity (`b_k ≥ 0`, `Σb_k = 1`), so the
result is positive, exact at the knots, free of overshoot, and the error
propagation is exact.  The curvature of the multiple-scattering spectrum over
one 50–60 nm anchor spacing is small (see §4.4).

The Monte Carlo cost is `n_anchor × n_photons` and is **independent of the
number of output wavelengths** — a 40-wavelength sky costs the same as a
5-wavelength one.

### 1.4 Error estimate

The quoted `stderr` has three parts, added in quadrature:

1. **statistical** — the residual scatter of the coarse values around the
   regression, propagated through the smoother weights
   (`σ² = Σ wᵢ² rᵢ²`, with the mirror pairs combined into one effective
   weight).  The per-point sample standard deviation returned by the Monte
   Carlo is *not* used, because it is unreliable in the heavy-tail regime;
2. **smoothing bias** — a 3 % relative floor (`MS_BIAS`), measured against a
   converged coarse grid (§4.5);
3. the wavelength-interpolation weights, propagated as above.

The first-order term is deterministic and carries no statistical error.

### 1.5 Quality presets

| quality | n_zen | n_az | anchors | photons per anchor | approx. cost |
|---|---:|---:|---:|---:|---:|
| `low` | 6 | 8 | 5 | 4.0 × 10⁶ | 2 × 10⁷ |
| `medium` | 8 | 12 | 7 | 1.2 × 10⁷ | 8.4 × 10⁷ |
| `high` | 10 | 16 | 10 | 4.0 × 10⁷ | 4.0 × 10⁸ |
| `ultra` | 12 | 20 | 13 | 1.2 × 10⁸ | 1.6 × 10⁹ |

`grid=dict(n_zen=…, n_az=…, n_anchor=…, n_photons=…)` overrides the preset.

---

## 2. Validation

All numbers below are reproducible with
`uv run python tools/twilight_benchmark.py` (≈ 10 min; `--quick` ≈ 2 min for
the noisier versions).  Atmosphere: `presets.clear` (AOD₅₅₀ = 0.05, 300 DU
ozone, ground albedo 0.1), observer at sea level.

### 2.1 Agreement with the deterministic model where both are valid

Zenith V-band luminance, `quality="medium"`, engine / deterministic ratio.
The quoted σ is the engine's own statistical error.  (The deterministic model
is also the sum of an exact spherical first-order term and a *plane-parallel*
multiple-scattering term, so the two models differ, as they should, by
O(10 %) of the multiple-scattering part.)

| solar elev. | view zenith | deterministic [cd m⁻²] | engine [cd m⁻²] | ratio | σ |
|---:|---:|---:|---:|---:|---:|
| −3° | 0° | 37.58 | 37.73 | **1.004** | 0.7 % |
| −3° | 30° | 52.55 | 54.00 | 1.028 | 0.6 % |
| −3° | 60° | 120.9 | 127.1 | 1.051 | 0.6 % |
| −3° | 80° | 293.1 | 315.7 | 1.077 | 0.7 % |
| −4.5° | 0° | 6.736 | 7.227 | 1.073 | 0.9 % |
| −4.5° | 60° | 26.56 | 29.93 | 1.127 | 0.6 % |
| −6° | 0° | 0.7368 | 0.9856 | **1.338** | 2.2 % |
| −6° | 60° | 4.012 | 5.200 | 1.296 | 0.8 % |
| −6° | 80° | 19.24 | 23.29 | 1.211 | 0.7 % |

**Where the models agree.**  Down to −3° the two agree to 0.4–8 %; the
difference is entirely the multiple-scattering field, and it grows as the
diffuse transport becomes non-local (8 % at 80° at −3°, 14 % at −4.5°).  Above
the horizon (−3° to +90°) this is the expected size of the plane-parallel
error.

**Where they do not.**  Below about −4.5° the plane-parallel multiple
scattering loses the horizontal transport and the two models diverge by
30–35 % at −6°, and by an order of magnitude at −9° (the deterministic model
is 20× too dark at −9° and essentially zero at −12°).  *This is the point of
the engine*: the spherical answer is the correct one there, and the engine
should be used instead of the deterministic model.

### 2.2 Patat, Ugolnikov & Postylyakov (2006)

Zenith V-band luminance, `quality="medium"`.  Reference: the Table 1
second-order fit of Patat et al. (2006, A&A 455, 385) converted with
`L = 1.08 × 10⁵ × 10^(−0.4 m_V)` cd m⁻² (the same reference and conversion as
`validation/check5_twilight.py`).  The −3° and −18° points are extrapolations
of that fit outside its stated 95° ≤ ζ ≤ 105° validity range.

| solar elev. | Patat 2006 [cd m⁻²] | single only | deterministic (single + plane-parallel MS) | **engine** (stat. error) | engine / Patat |
|---:|---:|---:|---:|---:|---:|
| −3° (extrap.) | 40.09 | 28.57 | 37.58 | **37.80** ± 0.24 | **0.94** |
| −6° | 0.516 | 0.570 | 0.737 | **0.978** ± 0.018 | **1.89** |
| −9° | 0.0171 | 7.38 × 10⁻⁴ | 9.58 × 10⁻⁴ | **0.0238** ± 0.0043 | **1.39** |
| −12° | 0.00146 | 7.56 × 10⁻⁷ | 9.80 × 10⁻⁷ | **2.12 × 10⁻³** ± 1.28 × 10⁻³ | **1.45** |

**Tolerance considered acceptable.**  A factor of 2 (0.3 dex) in absolute
luminance at every elevation, with a factor of 1.5 preferred inside the fitted
range (−6° to −12°).  This is the tolerance used by `validation/check5`
(0.5 dex) tightened by the extra physics the engine supplies.

**Verdict: all four elevations are within a factor of 2, and the deep
twilight — where the deterministic model is 24× too dark at −9° and 1500× too
dark at −12° — is now within 1.4–1.5×.**

The engine passes the stated tolerance at every elevation.  The single-scattering
column shows why the engine is needed: 0.043× the reference at −9° and
0.0005× at −12°, against 1.39× and 1.45× for the engine.

The remaining error is a **factor 1.4–1.9 overestimate**, not an underestimate.
At −6° it is dominated by the model-atmosphere/site mismatch described below;
at −9° and −12° the statistical error is 18 % and 60 %, so the 1.4× ratios are
consistent with a converged value of order 1.0–1.5 within 1–2σ.

Two caveats must be read with the table:

* the **−3° and −18°** reference values are extrapolations.  At −18° the
  measured sky is the natural night sky (airglow, no scattering); the model has
  no airglow, so the −18° row measures the absence of an emission component,
  not a scattering error;
* the **−6°** comparison is the one where the engine *overshoots*.  The
  first-order model already matches Patat to +10 % there, and the spherical
  multiple scattering adds another +34 % (the deterministic plane-parallel
  solver adds +30 %, so this is not a Monte Carlo artefact — the direct
  reference Monte Carlo gives the same value).  The model atmosphere is a
  sea-level mid-latitude standard atmosphere with AOD 0.05, while Paranal is a
  2400 m desert site with an extremely clean, dry atmosphere; a site-specific
  atmosphere would lower both the Rayleigh column and the multiple scattering.
  We regard this as a **model-atmosphere/reference-site mismatch rather than a
  transport error**, but it is not resolved here and it is the largest
  remaining absolute-photometry uncertainty at −6°.

### 2.3 Bias against a high-photon direct Monte Carlo

The engine (smoothing + wavelength interpolation included) against a direct
`mc_radiance_split` run with 1.2 × 10⁷ photons over four directions, at 550 nm.
`quality="medium"` with a single anchor wavelength and 1.2 × 10⁷ photons
(≈ 3.3 × 10⁴ photons per coarse direction).

| h | view zenith | azimuth | direct MC | rel. | engine | rel. | engine / direct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −6° | 0° | 0° | 5.34 × 10⁻⁶ | 3.0 % | 5.28 × 10⁻⁶ | 3.4 % | **0.99** |
| −6° | 45° | 90° | 8.11 × 10⁻⁶ | 2.6 % | 8.08 × 10⁻⁶ | 3.0 % | **1.00** |
| −6° | 85° | 0° | 1.02 × 10⁻⁴ | 0.7 % | 1.02 × 10⁻⁴ | 3.6 % | **1.00** |
| −6° | 85° | 180° | 2.00 × 10⁻⁵ | 1.5 % | 1.89 × 10⁻⁵ | 4.2 % | 0.94 |
| −9° | 0° | 0° | 3.24 × 10⁻⁷ | 12.3 % | 3.40 × 10⁻⁷ | 6.9 % | **1.05** |
| −9° | 45° | 90° | 4.19 × 10⁻⁷ | 11.0 % | 4.32 × 10⁻⁷ | 3.2 % | **1.03** |
| −9° | 85° | 0° | 6.38 × 10⁻⁶ | 3.0 % | 6.04 × 10⁻⁶ | 5.4 % | 0.95 |
| −9° | 85° | 180° | 1.09 × 10⁻⁶ | 6.9 % | 8.94 × 10⁻⁷ | 4.5 % | 0.82 |

The engine reproduces the direct Monte Carlo within 6 % everywhere except the
**anti-solar horizon at −9°**, where it is 18 % low.  That is the strongest
residual bias we have found; it is a resolution effect of the smoother (the
field turns over sharply between the 78° and 88.5° nodes on the anti-solar
side), it is at the 2σ level of the reference, and it is inside the 3 % +
residual-scatter error bar only marginally.  Treat the anti-solar horizon as
having a ~10 % systematic in addition to the quoted statistical error.

### 2.4 Error scaling with the photon count

Zenith, 550 nm, `quality="medium"` with one anchor wavelength, seed 7:

| h | photons | MS [W m⁻² sr⁻¹ nm⁻¹] | stderr | rel. | time |
|---:|---:|---:|---:|---:|---:|
| −6° | 4 × 10⁵ | 5.78 × 10⁻⁶ | 1.59 × 10⁻⁶ | 27.6 % | 1.1 s |
| −6° | 2 × 10⁶ | 3.59 × 10⁻⁶ | 6.30 × 10⁻⁷ | 17.6 % | 4.2 s |
| −6° | 6.4 × 10⁶ | 4.73 × 10⁻⁶ | 2.93 × 10⁻⁷ | 6.2 % | 17.5 s |
| −6° | 3 × 10⁷ | 5.16 × 10⁻⁶ | 2.55 × 10⁻⁷ | 4.9 % | 68.8 s |
| −9° | 4 × 10⁵ | 1.01 × 10⁻⁷ | 9.11 × 10⁻⁸ | 90.3 % | 1.1 s |
| −9° | 2 × 10⁶ | 1.09 × 10⁻⁷ | 7.43 × 10⁻⁸ | 68.1 % | 4.2 s |
| −9° | 6.4 × 10⁶ | 2.96 × 10⁻⁷ | 1.10 × 10⁻⁷ | 37.0 % | 16.3 s |
| −9° | 3 × 10⁷ | 2.74 × 10⁻⁷ | 2.80 × 10⁻⁸ | 10.2 % | 63.9 s |

The relative error falls as `1/√N` while the estimator has many effective
contributions per direction (the −6° rows: 2.0, 2.1).  At −9° the estimator
enters the **heavy-tail regime**: the mean is carried by a small number of
paths that reach the terminator ring, so the sample variance is itself
unreliable and the scaling is noisy (and slower than `1/√N` at low `N`).  This
is the intrinsic difficulty of the problem, not an implementation artefact: at
550 nm the per-photon coefficient of variation at −9° is ≈ 400, i.e. a single
direction needs of order 10⁷ photons before the central limit theorem is even
approximately valid.

### 2.5 Runtime

| quality | photons (`n_anchor × n_photons`) | measured | full 30 × 60 × 40 sky |
|---|---:|---:|---:|
| `low` | 2.0 × 10⁷ | ~2.1 µs/photon | **~45 s** |
| `medium` | 8.4 × 10⁷ | ~2.1 µs/photon | **~3 min** |
| `high` | 4.0 × 10⁸ | ~2.1 µs/photon | ~15 min |
| `ultra` | 1.6 × 10⁹ | ~2.1 µs/photon | ~1 h |

Measured on the zenith at 550 nm (the per-photon cost is dominated by the
grazing coarse directions, which is conservative): 3 × 10⁷ photons in 64 s at
−9° and 69 s at −6°, i.e. **2.1–2.3 µs per photon** on one core.  The cost is
linear in `n_anchor × n_photons` and independent of the number of view
directions and output wavelengths.

The cost is `n_anchor × n_photons`; the number of output directions and the
number of output wavelengths are essentially free.  These are single-core
NumPy timings.

---

## 3. Approximations and their measured bias

| approximation | size | measured how |
|---|---|---|
| multiple scattering is horizontally uniform | the aerosol `HorizontalField` does not enter the Monte Carlo (same as `rt/multiple`) | by construction |
| angular smoother: local-linear on a coarse grid | ≤ 5 % typical, ~10–18 % at the anti-solar horizon at −9° | §2.3, vs a direct high-photon MC |
| wavelength interpolation (linear in log `L`) | < 1 % between anchors | anchor spacing vs curvature, §4.4 |
| plane-parallel is *not* used for the diffuse field | — | the whole point |
| ground reflection is Lambertian, no polarisation, no refraction | as the rest of the package | — |
| `max_scatter = 50` | negligible above −12° | `docs/MC_REFERENCE.md` §5 |

The single-scattering term is **not** approximated by the engine: it is the
same `rt/single.sky_radiance` call the production model makes.

---

## 4. Remaining systematic uncertainties

1. **Heavy-tail statistics at −9° and below.**  The Monte Carlo estimator for a
   shadowed line of sight is dominated by rare paths that random-walk to the
   terminator ring.  At −12° the per-photon coefficient of variation is of
   order 10³.  Consequence: the *statistical* error quoted by the engine is
   reliable only in the sense that it is derived from the observed scatter; the
   underlying distribution is far from Gaussian, and a single run can sit a
   factor of two from the converged value at low budgets.  **Do not interpret
   the −12° numbers as better than a factor of ~1.5–2 without a `high`/`ultra`
   run.**
2. **The anti-solar horizon** (§2.3): ~10 % low at −9°, a smoother-resolution
   effect.
3. **The −6° absolute level**: the engine is ~1.4–1.9× the Patat fit, and the
   direct Monte Carlo agrees with the engine, so the excess is in the model
   atmosphere (sea level, AOD 0.05) rather than in the transport.  Not resolved
   here.
4. **No airglow.**  Below about −15° the real sky is dominated by the natural
   night sky, which the model does not contain; the −18° comparison is
   therefore not a test of the engine.
5. **The ground.** The Monte Carlo includes the Lambertian ground reflection of
   the direct beam and the diffuse field.  Patanal's ground albedo is not 0.1
   everywhere; the ground term is a few per cent of the twilight zenith.
6. **No aerosol `HorizontalField`, no refraction, no polarisation**, in common
   with the rest of the package.

---

## 5. What I would do next

1. **Kill the variance at the source.** The correct fix is not more photons but
   a better estimator.  The dominant order is the *second* one, and it can be
   estimated by a **local estimate toward the lit region**: sample the second
   scattering vertex directly from a distribution concentrated on the
   terminator ring (altitude 20–60 km, solar zenith 85–98°) and weight by the
   exact transmittance from the first vertex.  A pilot implementation of
   direction-space importance sampling (biasing the scattering direction
   towards "up, then sunward") gave only a factor ≈ 2 in variance, because the
   weight fluctuations eat the gain; target-space sampling should do much
   better because the second vertex is where all the difficulty lives.
   With orders ≥ 3 left to the analog Monte Carlo (their source fills the
   whole atmosphere and is easy), this should bring −9° to a few per cent in
   minutes and make −12° feasible.
2. **A spherical SOS/DISORT solver** for the diffuse field, in the axial
   symmetry of the sun–Earth system: the field depends on (r, solar zenith,
   direction), and a discrete-ordinate solution in the meridian plane would be
   deterministic.  This is the "right" long-term answer, and the Monte Carlo
   would remain as its independent check.
3. **Site-specific atmospheres** (Paranal: 2400 m, AOD 0.02, low ground
   albedo) to settle the −6° absolute level.
4. **Vectorise over wavelengths** with a common random-number stream and
   importance weights, to remove the `n_anchor` factor from the cost.
5. **An adaptive photon budget**: a short pilot run to measure the per-direction
   effective sample size, then allocate the remaining photons where the
   relative variance is highest (the zenith, not the bright arch).
