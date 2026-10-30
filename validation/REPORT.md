# Validation report — is the model right?

**Branch:** `feat/validation` · **Model under test:** `skycolor/` as committed
on this branch (single-scattering core; multiple scattering is still a stub).
**Environment:** Python 3.11, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2.

Run everything with:

```bash
cd "/Users/zhangjiajie/Codes/skycolor-wt/validation"
"/Users/zhangjiajie/Codes/Sky Color Simulation/.venv/bin/python" validation/run_all.py
```

`run_all.py` prints the table below, writes `validation/results.json`, and
exits non-zero only if a check that actually ran fails its tolerance.
Reference sources are documented in `validation/data/SOURCES.md`.

## Summary

| # | Check | Reference | Verdict |
|---|---|---|---|
| 1 | Rayleigh optical depth, 300–700 nm | Bodhaine et al. (1999), Table 3 / eq. (30) | **PASS** |
| 2 | Direct + diffuse clear-sky illuminance | CIE 110-1994, ASTM G173-03 | *pending* |
| 3 | Clear-sky relative luminance distribution | CIE 110-1994 | **FAIL** (horizon only) |
| 4 | Sky chromaticity | measured spectra | *pending* |
| 5 | Twilight zenith luminance | measured twilight tables | *pending* |

---

## Check 1 — Rayleigh optical depth vs Bodhaine et al. (1999)

**Reference.** Bodhaine, Wood, Dutton & Slusser (1999), *On Rayleigh optical
depth calculations*, J. Atmos. Oceanic Technol. 16, 1854–1861.  We compare the
model's vertical Rayleigh optical depth with the paper's recommended values
for dry air with 360 ppm CO₂ at sea level, 1013.25 mb, 45°N (Table 3), using
the paper's eq. (30) fit for the dense 1 nm comparison and the tabulated
values as an independent spot check.  The tabulated values reproduce eq. (30)
to better than 2×10⁻⁴ relative, which validates our transcription.

**What this tests.** The model uses the same *formula family*, so this is not
an independent derivation of Rayleigh physics.  It is an independent test of
the implementation: refractive index of air, King (depolarisation) factor,
vertical air column, units.

**Numbers.**

| λ (nm) | model τ_R | Bodhaine eq. (30) | deviation |
|---:|---:|---:|---:|
| 300 | 1.203842 | 1.216406 | −1.03 % |
| 320 | 0.913867 | 0.922028 | −0.89 % |
| 340 | 0.706987 | 0.712476 | −0.77 % |
| 350 | 0.625745 | 0.630300 | −0.72 % |
| 400 | 0.358240 | 0.360213 | −0.55 % |
| 450 | 0.220136 | 0.221106 | −0.44 % |
| 500 | 0.142830 | 0.143353 | −0.36 % |
| 550 | 0.096762 | 0.097065 | −0.31 % |
| 600 | 0.067901 | 0.068087 | −0.27 % |
| 650 | 0.049064 | 0.049184 | −0.24 % |
| 700 | 0.036340 | 0.036421 | −0.22 % |

RMS deviation over 300–700 nm: **0.48 %**; worst case **1.03 %** at 300 nm.

The model's molecular cross section at 550 nm is 4.507×10⁻³¹ m²; Bodhaine's
recommended calculation gives 4.511×10⁻³¹ m² (−0.1 %).  The small τ deficit
at short wavelengths is the expected consequence of using Edlén's refractive
index and a constant depolarisation factor (0.0279) instead of Bodhaine's
recommended Peck & Reeder index and wavelength-dependent Bates King factor.

**Tolerance.** 2 % at every wavelength (1 % over 400–700 nm).  The model
passes with a large margin.

**Verdict: PASS.**  The molecular scattering backbone is numerically sound.

**Documentation defect (not a physics bug).** The docstring of
`skycolor/spectra/rayleigh_cross_section` says it returns 1.135×10⁻³¹ m² at
550 nm.  The code actually returns 4.507×10⁻³¹ m², which is the correct
value (and the one consistent with τ_R(550) ≈ 0.097).  The docstring number
is wrong by a factor of four.  No code was changed.

---

## Check 3 — CIE 110-1994 clear-sky relative luminance

**Reference.** CIE 110-1994, *Spatial distribution of daylight — Luminance
distributions of various reference skies*.  The standard clear-sky
distribution is

```
L(θ,γ)/Lz = (0.91 + 10 e^(−3γ) + 0.45 cos²γ)(1 − e^(−0.32/cosθ))
          / [(0.91 + 10 e^(−3θs) + 0.45 cos²θs)(1 − e^(−0.32))]
```

with θ the viewing zenith angle, γ the angular distance from the sun and θs
the solar zenith angle (reproduced as eq. (4.31) of Zotti 2007, citing CIE
110-1994).  This is a purely geometric test: both sides are normalised to
their own zenith luminance, so no absolute radiometry or solar spectrum is
involved.

**Method.**  Model photopic luminance is the 683 lm/W integral of the
single-scattered spectral radiance (`presets.clear`, AOD₅₅₀ = 0.05,
exponential aerosol H = 1.5 km) on a 37 × 72 sky grid, divided by its own
zenith value.  The CIE map is evaluated on the same grid.

**Numbers** (model / CIE, sun-side horizon means the horizon point in the
sun's azimuth; anti-solar is 180° away):

| solar elev. | horizon sun-side / zenith | horizon anti-solar / zenith | RMS θ≤60° | RMS 75–90° | RMS whole sky | max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 30° | **19.86 / 8.35 (2.38×)** | 3.05 / 3.14 (−2.9 %) | 3.0 % | 43.0 % | 19.4 % | 138 % |
| 45° | **7.36 / 3.65 (2.01×)** | 2.09 / 2.00 (+4.1 %) | 3.8 % | 40.3 % | 18.4 % | 101 % |
| 60° | **2.76 / 1.60 (1.73×)** | 1.25 / 1.14 (+9.4 %) | 3.8 % | 36.5 % | 16.8 % | 73 % |

Figures: `figures/check3_cie_luminance.png` (maps and model/CIE ratio) and
`figures/check3_cie_cuts.png` (cuts through the sun meridian at h = 45°).

**Verdict: FAIL**, but the failure is localised and interpretable:

* Above 60° zenith angle the model is excellent: RMS 3–4 %, maximum
  deviation below 20 %.
* The anti-solar horizon agrees with CIE to within 4–9 %.
* The sun-side horizon is **about 2× too bright** at h = 45° (2.4× at
  h = 30°, 1.7× at h = 60°), and the whole-sky RMS (18.4 %) is dominated by
  the 75–90° band (40.3 %).

**The model puts the maximum luminance in the wrong place.**  At h = 45° the
CIE sky is brightest at the sun (7.25 at θ = 45°, γ = 0) and falls to 3.65 at
the sun-side horizon.  The model is the other way round: 5.82 at the sun and
7.36 at the horizon.  The model's aureole is therefore **too weak relative to
the horizon**, not too strong.

**Attribution.**  We re-ran h = 45° with two variants:

| atmosphere | zenith L [cd m⁻²] | horizon sun / zenith | horizon anti / zenith |
|---|---:|---:|---:|
| pristine (Rayleigh + O₃ only) | 977 | 9.78 | 9.74 |
| clear (AOD 0.05, H = 1.5 km, as used above) | 1571 | 7.36 | 2.09 |
| clear, well-mixed aerosol (H = 20 km) | 1578 | 6.30 | 4.84 |
| CIE 110 | — | 3.65 | 2.00 |

* The 1.5 km exponential aerosol profile is **not** the cause of the
  sun-side excess: removing the aerosol makes it *worse* (9.78), and a
  well-mixed profile only brings it down to 6.30.  What the exponential
  profile does do is suppress the anti-solar horizon from 9.74 (pristine)
  to 2.09, which is why the anti-solar side happens to agree with CIE.
* The likely dominant cause is the **absence of multiple scattering**.
  At 550 nm the horizon path has τ ≈ 3.7, so the limb is optically thick and
  single scattering is a poor approximation there, while the zenith is
  optically thin (τ ≈ 0.13).  Multiple scattering adds a diffuse component
  that raises the zenith much more (relatively) than the already bright
  limb, which is exactly the direction needed to reduce the ratio from 7.4
  to 3.6.  This is consistent with the absolute scale: the model's zenith
  luminance at h = 45° is 1.57 kcd m⁻², while CIE 110 with T = 3 gives
  2.70 kcd m⁻² — multiple scattering would supply much of the missing
  zenith light.
* The residual forward aureole is a secondary effect.  Its shape is wrong
  (too weak at the sun, too strong at the limb), which is what one expects
  when the limb's single-scattered radiance is not redistributed by higher
  orders.

We did **not** change the model.

## Checks 2, 4, 5

Check 2 (illuminance) is in progress.  Checks 4 and 5 are not yet
implemented; a check that cannot run is reported as `SKIP` and does not make
`run_all.py` fail.
