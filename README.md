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

# colour coordinates of the zenith / horizon (time input: naive = UTC,
# explicit offsets honoured; the resolved direction is pre-refraction)
uv run python -m skycolor.cli chart --time 2026-06-01T08:00:00Z \
    --latitude 39.9 --longitude 116.4

# pick an approximation and a uniform cloud deck
uv run python -m skycolor.cli render --sun-elevation 10 --cloud stratus \
    --approximation hybrid-sos --out overcast.png
```

```python
from skycolor.presets import clear
from skycolor import render
from skycolor.constants import WL_COARSE

atmo = clear(WL_COARSE, aod=0.05)          # clean mid-latitude air
pres = render.solve_dome(atmo, sun_elevation_deg=5.0)   # one solve, prepared
img = render.render_result(pres, camera="perspective", size=(900, 500))
# after preparation, tables and every camera are pure consumers of `pres`;
# render.format_result_summary(pres) states what ran and its limits
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
experiments/          phenomenon themes (thin production-entry scripts; cssp & retired topics)
validation/           six independent checks against published references
tests/                regression and mechanism tests
docs/                 project overview, architecture and development guide
  physics/            general physical and numerical method notes
  twilight/           twilight engine, audit sequence and historical sensitivities
```

## Where it stands

[STATUS.md](docs/STATUS.md) records current behavior, the physical-check
snapshot and validation limits;
[the documentation guide](docs/README.md) maps methods and historical evidence.
The six-check suite currently exits non-zero for known physical deviations
(their scope is in STATUS.md, which is the single place the counts live).

Recent work fixed raised-ground solar occlusion and AOD550 wavelength slicing.
Twilight source/transport experiments now have independent closure checks, but
have not replaced production defaults. Absolute twilight brightness, rare-path
MC uncertainty and comparisons between observing sites remain unresolved.
See [ROADMAP.md](docs/ROADMAP.md) for M11–M18 and proposed follow-up work.

## Data provenance

See `skycolor/data/SOURCES.md`.  All tables are committed so the code runs
offline; `tools/build_data.py` regenerates them from `data_raw/`.
