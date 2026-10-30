# Sky Colour Simulation

A physically based, spectral model of the colour of the sky.

*Ignore clouds: can we predict the colour of the sky from the sun's position,
the composition of the air, and the direction you are looking?  And what
happens to that colour when the air changes — more water vapour, more haze, a
dust storm, clean air near you and haze on the horizon?*

Everything is computed from first principles: a solar spectrum, molecular
(Rayleigh) and aerosol (Mie) scattering, ozone absorption, and a spherical
radiative-transfer solver.  The output is spectral radiance
(W m⁻² sr⁻¹ nm⁻¹), converted to colour through an explicit chain
(CIE colorimetry → chromatic adaptation → tone mapping).

![A sunset](figures/06_hero_sunset.png)

## What the model reproduces

| Effect | Mechanism |
|---|---|
| Blue daytime sky | Rayleigh scattering, σ ∝ λ⁻⁴·¹ |
| Deep blue zenith, blue hour | O₃ Chappuis band absorption (500–700 nm) |
| Red/orange sunset | Long low-altitude path, Rayleigh + ozone extinction |
| Twilight glow, earth's shadow | Spherical geometry: the shadow boundary is a curve |
| Belt of Venus | Sunlit high atmosphere above the shadow |
| Hazy horizon | Mie scattering with a forward-peaked phase function |
| Dust storms | Large particles, weak spectral dependence, absorbing |
| Near-vs-far weather | Horizontally varying aerosol optical depth along the line of sight |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .[plot]

# a 3-degree sun, looking at it, 100-degree field of view
.venv/bin/python -m skycolor.cli render --sun-elevation 3 --camera perspective \
    --size 1200x520 --pitch 14 --hfov 100 --preset clear --aod 0.05 \
    --yaw 180 --out sunset.png

# colour coordinates of the zenith / horizon
.venv/bin/python -m skycolor.cli chart --sun-elevation 20 --preset hazy --aod 0.3
```

```python
from skycolor.presets import clear
from skycolor import render
from skycolor.constants import WL_COARSE

atmo = clear(WL_COARSE, aod=0.05)          # clean mid-latitude air
img = render.render_scene(atmo, sun_elevation_deg=5.0,
                          camera="perspective", size=(900, 500))
```

## What is in the box

```
skycolor/
  spectra/        solar spectrum, CIE CMFs, O3/NO2 cross sections, Rayleigh
  atmosphere.py   US1976 + thermospheric tail, ozone, aerosol layers, horizontal fields
  aerosol.py      analytic Angstrom and physical Mie aerosol models
  geometry.py     solar position, spherical-shell ray tracing
  rt/single.py    exact single scattering in spherical geometry
  rt/multiple.py  successive-orders multiple scattering (Fourier + Gauss-Legendre)
  rt/mc.py        independent Monte Carlo reference solver
  rt/solver.py    single + multiple
  color.py        spectrum -> XYZ -> sRGB, adaptation, tone mapping
  render.py       sky dome, cameras (panorama / fisheye / perspective), images
  presets.py      named scenarios
tools/build_data.py   rebuilds the packaged data tables from data_raw/
experiments/          scripts that produce the figures (01..06)
validation/           five independent checks against published references
tests/                52 tests
docs/                 architecture, method notes, the write-up
```

## How good is it?

`validation/REPORT.md` and `validation/ADDENDUM.md` have the details.  In short:

* Rayleigh optical depth vs Bodhaine et al. (1999): **0.5 % RMS**
* Direct normal illuminance vs ASTM G173 / Bird & Hulstrom: **+4 to +6 %**
* CIE 110 relative luminance, upper sky: **3.8 % RMS** (single scattering),
  14.5 % with multiple scattering (dominated by the solar aureole)
* Horizon chromaticity vs measured Berlin spectra: **Δxy = 0.019**
* Twilight zenith luminance at −6°: **+10 %** vs Patat et al. (2006)
* Multiple scattering vs an independent Monte Carlo: **31 % of single
  scattering at the zenith, both codes agree to 1 %**
* Known failures, documented rather than hidden: the limb (≈2× too bright,
  plane-parallel diffuse transport), the solar aureole (Henyey-Greenstein has
  no forward peak), and deep twilight below −8° (needs horizontal transport
  from the twilight arch, which a plane-parallel model cannot do).

## Data provenance

See `skycolor/data/SOURCES.md`.  All tables are committed so the code runs
offline; `tools/build_data.py` regenerates them from `data_raw/`.
