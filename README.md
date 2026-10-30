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
| **Red afterglow filling the sky** | A scattering layer *above the shadow* — volcanic stratospheric sulphate or cirrus |
| **Blue sun** | Anomalous (non-monotonic) Mie extinction from a narrow size distribution |
| **Refraction, flattened sunset sun** | Traced rays through the refractive atmosphere (on by default) |
| **Overcast / underlit cloud decks** | Delta-Eddington two-stream on a plane-parallel layer |

## Quick start

The environment is managed by [uv](https://docs.astral.sh/uv/) and the exact
dependency versions are pinned in `uv.lock` (Python 3.12).  Everything below
assumes `uv run`, which uses that locked environment.

```bash
uv sync          # creates .venv from uv.lock, exactly reproducible
uv run pytest -q
```

To refresh the lock file on purpose: `uv lock --upgrade`, then re-run the
validation suite (`uv run python -m validation.run_all`) before committing it.

```bash

# a 3-degree sun, looking at it, 100-degree field of view
uv run python -m skycolor.cli render --sun-elevation 3 --camera perspective \
    --size 1200x520 --pitch 14 --hfov 100 --preset clear --aod 0.05 \
    --yaw 180 --out sunset.png

# colour coordinates of the zenith / horizon
uv run python -m skycolor.cli chart --sun-elevation 20 --preset hazy --aod 0.3
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
  mie.py          Mie coefficients, analytic Legendre moments, adaptive phase grids
  aerosol.py      eight-member aerosol library (rural … volcanic … cloud), delta-M
  geometry.py     solar position, spherical-shell ray tracing
  rt/single.py    exact single scattering in spherical geometry
  rt/multiple.py  successive-orders multiple scattering (Fourier + Gauss-Legendre)
  rt/mc.py        independent Monte Carlo reference solver
  rt/twilight.py  spherical multiple scattering for deep twilight
  rt/cloud.py     delta-Eddington cloud layer
  cloud.py        cloud optics (Mie droplets, delta-M, presets)
  refraction.py   traced refraction, apparent solar position, refracted optical paths
  color.py        spectrum -> XYZ -> sRGB, adaptation, tone mapping
  render.py       sky dome, cameras (panorama / fisheye / perspective), images
  presets.py      named scenarios
tools/build_data.py   rebuilds the packaged data tables from data_raw/
experiments/          scripts that produce the figures (01..10)
validation/           five independent checks against published references
tests/                196 tests
docs/                 architecture, method notes, the write-up, the roadmap
```

## Where it stands

`docs/STATUS.md` is the one-page summary; `docs/ROADMAP.md` has the milestones
and the open questions.  Short version: the daytime sky is validated to a few
per cent, twilight timing and shape are right, and the **twilight level** is
now within ~15–30 % after the solar-chord fix (M10).  The remaining twilight
problem is the **altitude response**, which has the wrong sign.

## How good is it?

`validation/REPORT.md` and `validation/ADDENDUM.md` have the details.  In short:

* Rayleigh optical depth vs Bodhaine et al. (1999): **0.5 % RMS**
* Direct normal illuminance vs ASTM G173 / Bird & Hulstrom: **+4 to +6 %**
* CIE 110 relative luminance, upper sky: **10.6 % RMS** (multiple scattering;
  dominated by the solar aureole)
* Horizon chromaticity vs measured Berlin spectra: **Δxy = 0.018**
* Twilight zenith luminance at −6°: **0.69× Koomen (1952)** deterministic,
  **0.85×** spherical engine; 0.86× / 1.07× Patat et al. (2006)
* Multiple scattering vs an independent Monte Carlo: **31 % of single
  scattering at the zenith, both codes agree to 1 %**
* Known failures, documented rather than hidden: the limb (≈2× too bright,
  plane-parallel diffuse transport), the solar aureole (32-moment truncation
  without delta-fit), and the twilight **altitude response** (the model is a
  few per cent brighter at 2.4 km where the data are 0.80× darker; the
  `at_altitude` pressure scaling also needs an audit).

## Data provenance

See `skycolor/data/SOURCES.md`.  All tables are committed so the code runs
offline; `tools/build_data.py` regenerates them from `data_raw/`.
