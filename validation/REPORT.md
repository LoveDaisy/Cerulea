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
| 3 | Clear-sky relative luminance distribution | CIE 110-1994 | *pending* |
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

## Check 3 — CIE 110-1994 clear-sky relative luminance *(in progress)*

Reference distribution:

```
L(θ,γ)/Lz = (0.91 + 10 e^(−3γ) + 0.45 cos²γ)(1 − e^(−0.32/cosθ))
          / [(0.91 + 10 e^(−3θs) + 0.45 cos²θs)(1 − e^(−0.32))]
```

with θ the viewing zenith angle, γ the angle from the sun, θs the solar zenith
angle (CIE 110-1994, reproduced as eq. (4.31) in Zotti 2007).  Preliminary
numbers at solar zenith 45° (`presets.clear`, AOD₅₅₀ = 0.05):

| quantity | model | CIE 110 |
|---|---:|---:|
| horizon, sun side / zenith | 7.4 | 3.65 |
| horizon, anti-solar / zenith | 2.1 | 2.00 |
| zenith | 1.00 | 1.00 |

The anti-solar horizon agrees well; the sun-side horizon is about **2× too
bright**.  A pristine (Rayleigh-only) atmosphere gives 9.8 at the sun-side
horizon, so the excess is *not* removed by turning the aerosol off — the
dominant cause is the absence of multiple scattering (which makes the real
sky more uniform), with the aerosol forward-scattering aureole adding to the
sun-side brightness at lower elevations.  A full map, RMS/max deviation and
figure are being produced.

## Checks 2, 4, 5

Not yet complete.  A check that cannot run is reported as `SKIP` and does not
make `run_all.py` fail.
