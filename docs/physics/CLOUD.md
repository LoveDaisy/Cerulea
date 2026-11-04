# Clouds

*What does a uniform cloud deck do to the colour of the sky?*

This document describes `skycolor/cloud.py` (cloud optics) and
`skycolor/rt/cloud.py` (radiative transfer), the numbers that validate them,
and -- at least as important -- what they cannot do.

---

## 1. The question, and the shape of the answer

Almost all of the colour of the clear sky comes from two things: the spectrum
of sunlight after it has crossed a long slanted path through the atmosphere,
and the λ⁻⁴ Rayleigh cross-section that produced that path dependence. A cloud
deck sits in front of that sky and does something qualitatively different: it
is **grey** (its extinction is nearly independent of wavelength) and
**conservative** (a water droplet absorbs essentially nothing in the visible).
A grey, conservative, optically thick layer therefore destroys the spectral
information that the Rayleigh sky encodes, and it must re-emit light whose
colour is *borrowed from the illuminant*.

That single sentence predicts everything in this document:

* under a high sun the deck is white-grey, slightly warm because sunlight is
  slightly warm;
* the sky seen through a hole or through a thin deck keeps the colour of the
  light source behind it;
* at sunset the deck is still lit after the ground is dark, and it glows with
  the colour of the *reddened sunset beam* -- orange, then red, then it goes
  out. The red underlit cloud is not a property of the cloud; it is a property
  of the light falling on it.

---

## 2. Cloud optics

### 2.1 The layer

```python
CloudLayer(base_km=0.5, top_km=1.5, tau550=20.0, r_eff_um=8.0,
           n_droplets_cm3=200.0, phase="water")
```

`tau550` is the vertical extinction optical depth at 550 nm -- the quantity a
satellite retrieval or a textbook table quotes -- and it is the primary knob.
`base_km`/`top_km` matter for two reasons: they set the extinction coefficient
`beta = tau / H`, hence the liquid water content, and they decide **when the
deck stops being sunlit** after sunset (§5).

| preset | base--top [km] | τ(550) | r_eff [µm] | g | δ-M `f` | g' | τ'/τ | LWC [g m⁻³] |
|---|---|---|---|---|---|---|---|---|
| `stratus` | 0.5--1.5 | 20 | 8 | 0.859 | 0.789 | 0.331 | 0.211 | 0.10 |
| `altostratus` | 3--5 | 5 | 6 | 0.853 | 0.782 | 0.323 | 0.218 | 0.009 |
| `cirrus` (ice) | 8--10 | 0.5 | 40 | 0.886 | 0.818 | 0.375 | 0.182 | 0.006 |

The liquid water contents are what the optical depths imply for those
geometries; 0.1 g m⁻³ is a textbook stratus. The altostratus is thin because
τ = 5 spread over 2 km is a tenuous deck.

### 2.2 Mie theory

The droplets are a log-normal distribution specified by the **effective
radius** `r_eff = <r³>/<r²>`, with a geometric standard deviation of 1.35
(water) or 1.45 (ice), i.e. an effective variance of 0.09--0.13, which is what
in-situ probes measure in stratus. The Mie coefficients `a_n`, `b_n` are
computed with the standard Bohren--Huffman recurrences, vectorised over the
size distribution; `Q_ext`, `Q_sca` and `g` follow from the coefficient sums,
and the phase function is evaluated from the scattering amplitudes `S₁`, `S₂`.

The implementation lives entirely in `cloud.py`. It is a rewrite, not an
import, of the aerosol Mie code: a cloud model must not silently change when
an aerosol model is refactored. For a single sphere the two implementations agree on
`Q_ext` and `g` to better than 10$^{-5}$ over `x = 11 ... 2000`.

Two numerical details matter and are worth recording:

* **Per-size truncation.** The Riccati--Bessel recurrences grow like
  `(2n+1)/x` per step. A broad size distribution spans `x` from ~30 to ~1000,
  so a single `n_max` overflows for the small end. Each column is truncated at
  its own `n_max(x) = x + 4x^(1/3) + 2` and frozen after that.
* **Forward-peak quadrature.** The diffraction peak has an angular width of
  order `1/x` (≲ 10⁻³ rad for a 40 µm crystal). The Legendre moments are
  integrated on a composite Gauss rule: `[0, 12/x_max]` with 64 nodes plus
  `[12/x_max, π]` with 384. Convergence checks: with 128 size quadrature
  points, doubling either count changes `g` by less than 0.1 % and `β_l` for
  `l ≤ 30` by less than 0.5 %.

The moments are *derived* quantities with a built-in cross-check: `β₁` from the
angular integral and `g` from the Mie coefficient sum are computed by different
code paths and agree to 10⁻⁴, and `β₀ = 1` holds to 3 × 10⁻⁴.

### 2.3 The optical depth is grey

Across 380--780 nm the stratus optical depth varies by **2.4 %** (19.78 to
20.25) and the cirrus by **1 %**. A cloud is grey because the extinction
efficiency of a large sphere is `Q_ext → 2` regardless of wavelength, and a
size distribution washes out the Mie resonances. That is the quantitative
statement behind "clouds are white".

The single-scattering albedo is exactly 1 in this model. Real water droplets
have `1 - ω₀ ~ 10⁻⁵` in the visible (`k ~ 10⁻⁸`, Hale & Querry 1973); ice is
similar. Absorption is not what makes a cloud grey, and this model does not
pretend otherwise.

---

## 3. Delta-M scaling

A Legendre series truncated at `L` moments cannot represent a phase function
that is half delta function. For a water droplet with `r_eff = 8 µm`,
`g = 0.859`, and the forward diffraction spike carries roughly half the
scattered energy; a truncated series smears that energy over the whole sphere
and gets both the reflection and the transmission wrong.

**Delta-M** (Wiscombe 1977) removes the spike and keeps the moments:

```
f      = β_L                              (the moment at the truncation order)
β'_l   = (β_l - f)/(1 - f)                (l < L,  and β'_L = 0)
g'     = (g - f)/(1 - f)
tau'   = (1 - ω f) tau
ω'     = ω (1 - f) / (1 - ω f)
```

The truncation order is set by the solver. A two-stream uses two moments
(`2N = 2`), so `f = β₂`. For the stratus that gives `f = 0.789` and therefore

```
tau' = 0.211 tau = 4.23      (at tau = 20)
g'   = 0.331
```

Two things are worth noting.

* `f = β₂ ≈ 0.79` is *larger* than the ~50 % of energy in the diffraction
  spike. That is correct: `β₂` also absorbs part of the broad forward lobe,
  because the residual phase function is required to have zero second moment.
* For a Henyey--Greenstein phase function `β_l = g^l`, so `f = g²` and
  `g' = g/(1+g)`: delta-M at `2N = 2` **is** the classical delta-Eddington
  scaling of Joseph, Wiscombe & Weinman (1976). The code is tested for this
  identity.

Delta-M preserves the similarity parameter `(1-g) tau` of a conservative
layer exactly, which is the reason it works: the stratus has
`(1 - 0.859)·20 = 2.827` before scaling and `(1 - 0.331)·4.23 = 2.827` after.

---

## 4. The two-stream flux solver

### 4.1 Equations

With `τ` increasing downwards, `F↓`/`F↑` the diffuse irradiances on a
horizontal surface and `B = E e^{-τ/μ₀}` the direct beam:

```
dF↓/dτ = -γ₁ F↓ + γ₂ F↑ + b₃ B
dF↑/dτ = -γ₂ F↓ + γ₁ F↑ - b₄ B
```

```
γ₁ = (7 - ω(4+3g))/4          γ₂ = -(1 - ω(4-3g))/4
b₃ = (ω/2)(1 + 3gμ₀/2)        b₄ = (ω/2)(1 - 3gμ₀/2)
```

The propagator matrix satisfies `M² = λ² I` with `λ² = γ₁² - γ₂²`, so the
solution is a `cosh`/`sinh` pair plus a particular solution proportional to
`e^{-τ/μ₀}`. Both boundary conditions are imposed in closed form:
`F↓(0) = 0` at the top and `F↑(τ*) = A_g (F↓(τ*) + μ₀B(τ*))` at the bottom.

The implementation deliberately **does not** diagonalise. For conservative
scattering (`ω = 1`) `γ₁ = γ₂`, so `λ = 0`: the eigenvector solution divides
by zero and the matrix is not diagonalisable. Clouds *are* conservative, so
this is the normal case, not a corner case. The `cosh`/`sinh` form has the
`λ → 0` limit built in analytically (`sinh(λτ)/λ → τ`,
`(1-e^{-aτ})/a → τ`, ...), and it is exercised by every cloud test.

Two further numerical points:

* the growing exponential is dropped (semi-infinite solution) once
  `λτ > 18`, beyond which the transmitted flux is below 10⁻⁷ and round-off
  would otherwise destroy it;
* energy is conserved to 10⁻¹² for conservative scattering
  (`R + T = 1`), which is a strong test that the coefficients are right.

### 4.2 Illuminating a deck from below

The interesting case is the sun below the horizon. A point at height `z` is
sunlit when its solar ray misses the Earth: `(R+z) sin θ > R`. The terminator
is therefore at

```
z_shadow = R (1/sin θ - 1)
```

which is 0.97 km at −1°, 3.9 km at −2°, 8.7 km at −3°. This is the whole story
of "which clouds glow at sunset":

* a **stratus** (top 1.5 km) is in shadow below −1.2°;
* an **altostratus** (top 5 km) stays lit to −2.3°;
* a **cirrus** (top 10 km) stays lit to −3.2°.

There are three illumination geometries:

| geometry | condition | what happens |
|---|---|---|
| `top` | `θ ≤ 90°` | beam enters through the cloud top; the transmitted diffuse light is seen from below |
| `base` | `θ > 90°`, `z_shadow < top` | the beam enters the *bottom of the sunlit part*; the light crosses the shadowed part below before reaching the observer |
| `none` | `z_shadow ≥ top` | the whole deck is in the Earth's shadow |

For the partially lit case (`base < z_shadow < top`) the layer is split at the
terminator: the optical depth of the lit part and of the shadowed part below it
follow from homogeneity, the two-stream is solved for the lit part, and the
result is attenuated by `exp(-τ_below/μ_view)`. The illumination is evaluated
at the middle of the lit part -- a one-dimensional closure, because in reality
a beam grazing *through* a deck enters from the side, and its transmittance
varies by a factor of ~50 across the lit slab.

### 4.3 Validation

All albedos below are for a conservative layer over a black surface with the
sun overhead. **MC** is an independent plane-parallel Monte Carlo (200 000--
400 000 photons; 1σ ≤ 0.001) with the same phase function, written from
scratch for this validation.

| layer | τ | MC | delta-Eddington | error |
|---|---|---|---|---|
| g = 0 (isotropic) | 1 | 0.3406 | 0.3383 | −0.7 % |
| | 2 | 0.5165 | 0.5135 | −0.6 % |
| | 5 | 0.7372 | 0.7372 | 0.0 % |
| | 10 | 0.8523 | 0.8529 | +0.1 % |
| | 20 | 0.9206 | 0.9219 | +0.1 % |
| g = 0.85 (forward) | 1 | 0.0424 | 0.0467 | +10 % |
| | 2 | 0.0913 | 0.0967 | +6.0 % |
| | 5 | 0.2372 | 0.2400 | +1.2 % |
| | 10 | 0.4224 | 0.4191 | −0.8 % |
| | 20 | 0.6176 | 0.6157 | −0.3 % |
| stratus Mie (g = 0.859) | 10 | 0.4025 | 0.4079 | +1.3 % |
| g = 0.85, μ₀ = 0.6 | 10 | 0.5651 | 0.5527 | −2.2 % |
| ω = 0.9, g = 0.8 | 10 | 0.1355 | 0.1441 | +6.3 % |

The two-stream error is the expected one: a few per cent for `τ ≳ 5`, growing
to 6--10 % at `τ ~ 1`, where single scattering dominates and the angular
distribution matters. Energy conservation is exact, not approximate.

The model reproduces the classical limits:

* `R → 1` as `τ → ∞` for conservative scattering (0.944 at τ = 200,
  0.988 at τ = 1000);
* `R + T = 1` for a conservative layer, to 10⁻¹²;
* the reflected fraction for a water cloud at `τ = 10` is **0.41**; it reaches
  **0.70 at τ = 30**.

> **On the "≈ 0.7 at τ = 10" reference.** The albedo of a conservative slab
> depends on `g` as well as on `τ`. At `τ = 10` the delta-Eddington solution
> gives 0.85 for `g = 0`, 0.74 for `g = 0.5`, **0.69 for `g = 0.6`** and 0.40
> for the `g = 0.86` of a real water cloud; 0.7 is reached at `τ = 30` for
> `g = 0.86`. All of these sit on the published two-stream curves, and the
> Monte-Carlo points in `figures/10_cloud_albedo_and_sunset.png` lie on the
> `g = 0` and `g = 0.85` curves. The "0.7 at τ = 10" figure therefore implies
> a moderately forward-scattering phase function (`g ≈ 0.6`), not the `g ≈
> 0.86` of a droplet distribution; the model's own numbers are quoted above
> rather than adjusted to match.

---

## 5. From fluxes to an angular radiance

The solver returns hemispherical fluxes. The sky needs a radiance. The model
uses two *assumed* angular distributions, both documented:

**Transmitted light (sun above the horizon).** The CIE overcast law

```
L(θ) = L_zenith (1 + 2 cos θ)/3
```

normalised so that `∫ L μ dΩ` equals the transmitted diffuse flux:
`L_zenith = 9 F↓ /(7π)`. It peaks at the zenith, its horizon/zenith ratio is
exactly 1/3, and the model reproduces it to the last digit because it *is* the
shape function. This is the "correct overcast-sky luminance distribution" the
brief asks for. (The model cannot derive it; a two-stream has no angular
information. It is an empirical law that a real multiple-scattering solution
happens to follow closely for a thick conservative deck.)

**Light from an underlit base (sun below the horizon).** Isotropic over the
upward hemisphere, `L = F↓,base/π`. A grazing beam illuminating a slab from
below is a three-dimensional geometry; the two-stream can say what the *flux*
is and nothing about its angular distribution. Isotropic is the honest
placeholder, not a prediction.

**Cosines to fluxes.** The clear sky seen through the deck is the same clear
sky, attenuated along the view path: `exp(-τ/μ_v)` with the *physical*
(unscaled) `τ`. The total is

```
L_total(λ) = L_clear(λ) exp(-τ_cloud(λ)/μ_v) + L_cloud(λ, μ_v)
```

so `τ_cloud = 0` gives the clear sky **exactly** (bit-for-bit; there is a test
for it), and a thick deck at the horizon hides the clear sky completely
(`exp(-τ/μ_v) → 0`).

---

## 6. What the model says

### 6.1 Overcast

The deck removes the Rayleigh blue and replaces it with a flat, slightly warm
grey whose luminance peaks at the zenith. `figures/10_cloud_clear_vs_overcast.png`
puts the clear sky and the two water decks side by side at five sun elevations
with a single exposure, so the panels show absolute brightness as well as
colour. `figures/10_cloud_overcast_fisheye.png` shows the fisheye and the
overcast luminance profile.

### 6.2 Sunset

`figures/10_cloud_sunset_sequence.png` is the sequence. Cloud-base
chromaticity and luminance (zenith view, altostratus 3--5 km, τ = 5):

| sun elevation | illuminated | (x, y) | luminance |
|---|---|---|---|
| +5° | top | (0.367, 0.379) | 834 cd m⁻² |
| +1° | top | (0.448, 0.424) | 59 |
| −0.5° | base | (0.547, 0.421) | 9.5 |
| −1.5° | base | (0.596, 0.393) | 3.7 |
| −2.2° | base (partial) | (0.615, 0.378) | 0.02 |
| −3° | shadow | -- | 0 |

and the cirrus, which stays lit a degree longer:

| sun elevation | illuminated | (x, y) | luminance |
|---|---|---|---|
| +5° | top | (0.334, 0.348) | 989 cd m⁻² |
| −1.5° | base | (0.485, 0.430) | 65 |
| −2.2° | base | (0.535, 0.425) | 38 |
| −3° | base (partial) | (0.602, 0.388) | 2.1 |

The colour path is the one you see: white → warm white → orange → red. Two
things make it work, and both are geometric rather than optical:

1. the beam reaching the deck has crossed a long grazing path and is
   reddened *before* it hits the cloud (the cloud itself has no colour);
2. the deck is above the terminator, so it is still lit when the ground is
   dark. `z_shadow` at −1.5° is 2.2 km, so an altostratus at 3 km is lit while
   a stratus at 0.5 km is not.

The model also says when the glow *stops*: at −3° the altostratus has no sunlit
part left and turns black (the panel still shows the twilight sky seen through
it). A real deck would still be faintly lit by the twilight sky and by
horizontal transport from the sunlit part of the atmosphere; this model has
neither.

### 6.3 Why clouds are white and sunsets are red

`figures/10_cloud_albedo_and_sunset.png` makes both statements quantitative:
the albedo curves with Monte-Carlo points, the deck's zenith luminance through
sunset against the clear-sky reference, and the cloud-base spectrum at four sun
elevations. The last panel is the clearest summary: the spectrum at noon is
flat (grey cloud, white light); at −2° it is a red illuminant times the same
grey albedo.

---

## 7. What this cannot do

These are not bugs to be fixed later; they are consequences of the model's
structure. Each one is a picture you have seen.

* **Puffy cumulus.** The deck is infinite and horizontally homogeneous. There
  are no cloud sides, no finite edges, no shadowing of one cloud by another,
  no "fair-weather" sky at all. A broken deck cannot be represented -- there
  are no gaps for the blue sky or the sun to come through.
* **Silver linings.** The bright rim on the edge of a backlit cumulus is
  light forward-scattered through a *finite* cloud edge. A homogeneous slab
  has no edge, and a two-stream has no angular phase information.
* **Crepuscular rays.** These need a broken deck to cast the beams. An
  infinite uniform deck casts no beams.
* **The 22° halo, sundogs, and any ice-crystal optics.** Cirrus uses
  equivalent spheres with a real refractive index, which reproduces the
  *extinction and the integrated scattering* of ice but none of the
  crystallographic optics: no halos, no arcs, no specular reflection from
  oriented plates. The cirrus phase function here is smooth and featureless.
* **The solar aureole and the sun seen through a thin deck.** The direct
  transmitted beam is a delta function in angle and is not part of the sky
  radiance at all. A thin cirrus will therefore look like a uniform veil; in
  reality the sun is still a blindingly bright disk with a bright halo around
  it.
* **The angular distribution of anything.** Both angular shapes are assumed.
  A multi-stream (DISORT-like) solver would give the real field: the
  brightening towards the sun through a thin deck, the limb darkening of a
  thick one, the anisotropic glow of an underlit base.
* **A finite deck, and the horizon.** The plane-parallel diffuse field runs to
  infinity, so at the horizon the deck is seen through an infinite optical
  depth. A real deck ends somewhere; the horizon of a cloudy sky is not this.
* **Light generated below the deck.** The clear-sky field is computed for the
  cloud-free atmosphere and then attenuated. Light that the ground and the
  boundary layer would scatter *upwards into* the cloud base is represented
  only through the Lambertian ground albedo in the two-stream boundary
  condition, not through the full below-cloud radiation field.
* **Deep twilight.** As in the clear-sky model, once the deck is inside the
  Earth's shadow the model has nothing to light it with. Real clouds at
  −6° to −10° are lit by the twilight sky and by horizontal transport
  from the still-sunlit atmosphere; a plane-parallel model has neither. This
  is the same geometric failure documented in `docs/REPORT.md` §5.
* **Absorption, polarisation, refraction, thermal emission.** `ω₀ = 1`
  exactly (real clouds: 1 − 10⁻⁵); no polarisation, so no rainbow
  polarisation or neutral points; no refraction inside droplets; no thermal
  emission (irrelevant in the visible). Aerosol absorption *inside* the cloud
  (a dirty or smoky deck) is supported through `omega`, but the presets are
  pure water/ice.
* **The two-stream closure itself.** 5--10 % errors in the fluxes at
  `τ ~ 1`, a few per cent at `τ ≳ 5`; the angular distribution is worse than
  the fluxes.

---

## 8. Reproducing

```bash
uv sync
uv run pytest tests/test_cloud.py -q        # 25 tests, ~6 s
uv run python experiments/clouds.py         # figures 09/10, ~2 min
```

Timings on the reference machine (81 wavelengths, 1960 view directions):

| step | time |
|---|---|
| Mie optics for one preset (first call, cached) | 0.4 s (water), 1.8 s (cirrus) |
| cloud radiance + attenuation for 1960 views × 81 λ | 5 ms |
| clear-sky single + multiple scattering for the same grid | 2.4 s |

The cloud layer is essentially free once its optics are computed; the cost of
a cloudy sky is the cost of the clear-sky solver it attenuates.

### Files

| file | contents |
|---|---|
| `skycolor/cloud.py` | `CloudLayer`, Mie core, moments, delta-M, presets |
| `skycolor/rt/cloud.py` | two-stream solver, illumination geometry, angular model, `sky_radiance_with_cloud` |
| `tests/test_cloud.py` | 25 acceptance tests |
| `experiments/clouds.py` | figures 09/10 |
| `figures/10_cloud_clear_vs_overcast.png` | clear vs overcast, five sun elevations |
| `figures/10_cloud_overcast_fisheye.png` | overcast fisheye and the (1+2cos θ)/3 law |
| `figures/10_cloud_sunset_sequence.png` | altostratus and cirrus through sunset |
| `figures/10_cloud_albedo_and_sunset.png` | albedo validation, when the deck stays lit, cloud-base spectra |

### Interface

`sky_radiance_with_cloud` has the same `(n_view, n_wl)` contract as
`rt.solver.sky_radiance_total`, so a renderer only needs to swap the call:

```python
from skycolor.rt.cloud import sky_radiance_with_cloud

rad = sky_radiance_with_cloud(atmo, cloud_layer, sun_zenith_deg,
                              view_zenith_deg, view_azimuth_deg,
                              sun_azimuth_deg=..., clear_radiance=clear)
```

Passing `clear_radiance` reuses a clear-sky field that has already been
computed (the multiple-scattering solve is the expensive part).
`experiments/clouds.py` shows how to build a full sky dome with it and feed
`render`'s cameras and colour pipeline without touching `render.py`.
