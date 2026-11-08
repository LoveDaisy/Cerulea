# Production figures

This is the **selected current-production set**, not the old candidate pool.
All four images were actually generated and opened on 2026-11-08 through
`render.solve_dome` → `rt.solver.solve` → saved spectra → `render.render_result`.
Production version **`6627c00`** (same `skycolor/` tree as `7d7ce15`); the
example and numerical archive are committed with this catalogue. No RT kernel,
solar-source prescription, dependency or physical baseline was changed to make these images.

| Figure | Purpose / what to compare | Numeric source |
|---|---|---|
| [sky_30.png](sky_30.png) | Daylight, sun 30°; default hybrid-sos | `e30_hybrid-sos/total` |
| [sky_03.png](sky_03.png) | Low sun 3°; README's current scene, **not** a solar-disk image | `e3_hybrid-sos/total` |
| [approximations.png](approximations.png) | Two scene rows × spherical-single / plane-sos total / hybrid-sos total; same display parameters | the corresponding saved total arrays |
| [spectra.png](spectra.png) | Untonemapped spectra: plane single / spherical single / hybrid total, zenith and near-horizon sun-side; units on axes | same arrays, exact sampled zenith 0° / 86.25° |

## Reproduce or redraw

From the repository root, with its locked environment:

```bash
# Real production solve, ordinary output stays inside the theme
uv run python -m experiments.approximations.compare
# Publish those results: no new solve, no changed physical settings
uv run python -m experiments.approximations.compare \
  --from-npz experiments/approximations/out/representative.npz --publish-figures
# Or reproduce the committed images from their saved source spectra
uv run python -m experiments.approximations.compare \
  --from-npz experiments/approximations/data/representative.npz --publish-figures
# The low-sun scene via CLI, numerically and pixel-equivalent
uv run python -m skycolor.cli render --sun-elevation 3 --sun-azimuth 0 \
  --preset clear --aod 0.05 --n-zenith 25 --n-azimuth 37 \
  --approximation hybrid-sos --camera perspective --size 640x320 \
  --yaw 0 --pitch 14 --hfov 100 --exposure 0.03 --tonemap aces \
  --no-sun --no-ground --out sunset.png
```

Change `--sun-elevation 3` to `30` for the daylight scene. Both use the same
complete settings: **clear**, AOD550=.05, O₃=300 DU, NO₂=0, default US1976
air and ozone profiles, continental Mie aerosol with exponential H=1.5 km,
albedo=.1, homogeneous/cloud-free medium, sea-level observer at the opaque
model bottom, **refraction on**. Solar elevation/azimuth are **pre-refraction**.
Wavelengths 380…780 nm / 5 nm; half dome 25 zenith × 37 relative-azimuth
nodes (0…90° × 0…180°). SOS retains 32 moments / 32 μ / 12 Fourier modes,
40-order cap and 1e-3 tolerance; spherical single uses n_gauss=3, chunk=512.
Full effective settings and actual directions are in the archive, not inferred
from names.

All images: perspective **640×320**, yaw0°/pitch14°/hfov100°, **exposure=.03**,
ACES, adaptation=None, preserve_chromaticity=True, no sun marker, ground=False.
The first trial 1e-4 was almost black; .03 was selected once from its actual
range and fixed for every image. No per-panel auto-exposure or white balance.
Below-horizon pixels follow the renderer's no-ground display convention and
must not be interpreted as a physically rendered surface. Plot axes use physical
spectra, not exposure. Actual pixel dimensions of montage/plot are larger
because they include labels.

[NPZ, CSV and full call record](../experiments/approximations/README.md) preserve
spectra/components, XYZ/xy/Lv, effective settings, diagnostics and wall times.
Nine production calls took **121.73 s** in total; MC three seeds took **68.62 s**.
First plane call includes cold startup; later calls share warm caches, so this
is not a fair speed ranking. MC is **qualitative / budget-limited**, not plotted
as a converged reference. Interpretation and known confounders belong to the
[approximation story](../docs/physics/APPROXIMATIONS.md); physical validation
limits belong to [STATUS](../docs/STATUS.md).

## Older figures: preserved, not re-certified

All 26 old candidates were inspected and moved **byte unchanged**; none were
silently overwritten with a lookalike. Old full generation SHAs/environments
are not reliably known, so source scripts are *locators*, not claims of exact
modern reproduction. Generated theme outputs never overwrite these `historical_`
files. Root now contains only the four images above.

| Old group | Destination | Historical meaning / why not a current representative |
|---|---|---|
| 01 day cycle (2) | [retired/figures](../experiments/retired/figures/) | −18°…70° default hybrid-SOS sweep; deep points unsupported, panorama individually adapted |
| 02 twilight (3) | [retired/figures](../experiments/retired/figures/) | pre-M10 first-order / deep-twilight levels; not current accuracy evidence |
| 03 aerosol (3) | [aerosol/figures](../experiments/aerosol/figures/) (`historical_03_*`) | loading/types/haze; original version uncertain and images individually adapted |
| 04 heterogeneity (2) | [heterogeneity/figures](../experiments/heterogeneity/figures/) (`historical_04_*`) | horizontal field in first order only; not 3D multiple transport |
| 05 altitude (3) | [retired/figures](../experiments/retired/figures/) | altitude truncation/solar blocking semantics changed since; not current altitude response |
| 06 gallery (4) + 07 GIF | [retired/figures](../experiments/retired/figures/) | frozen composition/automatic adaptation/annotation; code locator `afaae17` |
| 08 afterglow (3) | [retired/figures](../experiments/retired/figures/) | pre-M10 layers/volcanic twilight; conditions not current |
| 09 blue sun (2) | [bluesun/figures](../experiments/bluesun/figures/) (`historical_09_*`) | useful Mie-vs-power-law optics illustration; no certified generation version |
| 10 clouds (3) | [clouds/figures](../experiments/clouds/figures/) (`historical_10_*`) | pre-M10 uniform-deck theme with old display/illumination, not new scene output |

Original theme script locations and fixed historical reproduction environments:
[experiments](../experiments/README.md) / [retired index](../experiments/retired/README.md).
The earlier six audit/intermediate images remain under retired; check6's
`13_twilight_datasets.png` remains under `validation/figures/`, not here.
