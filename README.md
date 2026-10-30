# Sky Colour Simulation

A physically based, spectral model of the colour of the sky.

The question this project tries to answer: *ignoring clouds, can we predict the
colour of the sky as a function of the sun's position, the composition of the
air, and where you are looking?* And more interestingly: what happens to that
colour when the air changes — more water vapour, more haze, a dust storm, or
clean air near you and haze on the horizon?

Everything is computed from first principles: a solar spectrum, molecular
(Rayleigh) and aerosol (Mie) scattering, ozone absorption, and a spherical
radiative-transfer solver.  The output is spectral radiance in
W m⁻² sr⁻¹ nm⁻¹, which is then converted to colour with a physically explicit
chain (CIE colorimetry → chromatic adaptation → tone mapping).

## What the model does

| Effect | Mechanism |
|---|---|
| Blue daytime sky | Rayleigh scattering, σ ∝ λ⁻⁴·¹ |
| Deep blue zenith, blue hour | O₃ Chappuis band absorption (500–700 nm) |
| Red/orange sunset | Long low-altitude path through the atmosphere |
| Twilight glow, earth's shadow | Spherical geometry: the shadow boundary is a curve, not a plane |
| Belt of Venus | Sunlit high atmosphere above the shadow |
| Haze whitening the horizon | Mie scattering with a forward-peaked phase function |
| Dust storms | Large particles, weak spectral dependence, absorbing |
| Near-vs-far weather | Horizontally varying aerosol optical depth along the line of sight |

## Layout

```
skycolor/
  spectra/        solar spectrum, CIE CMFs, O3/NO2 cross sections, Rayleigh
  atmosphere.py   vertical structure (US1976), ozone, aerosol layers
  aerosol.py      analytic Angström and physical Mie aerosol models
  geometry.py     solar position, spherical-shell ray tracing
  rt/single.py    exact single scattering in spherical geometry
  rt/multiple.py  multiple-scattering correction
  color.py        spectrum -> XYZ -> sRGB, adaptation, tone mapping
  render.py       sky dome, cameras, images
  presets.py      named scenarios
tools/build_data.py   rebuilds the packaged data tables from data_raw/
experiments/          scripts that produce the figures
validation/           comparisons against published results and photographs
```

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .[plot]
.venv/bin/python -m skycolor.cli render --sun-elevation -3 --out sunset.png
```

## Data provenance

See `skycolor/data/SOURCES.md`.  All data tables are committed so the code runs
offline; `tools/build_data.py` regenerates them from `data_raw/`.
