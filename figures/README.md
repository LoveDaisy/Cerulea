# Production figures

This is the **selected current-production set**, not the old candidate pool.
Just want to look at the figures? Read them with their captions in the
reader-level walkthrough [docs/physics/MAINLINE.zh.md](../docs/physics/MAINLINE.zh.md)
(Chinese); everything below the guide line is provenance — versions, commands,
archives — for reproduction and audits.
The first four images were actually generated and opened on 2026-11-08 through
`render.solve_dome` → `rt.solver.solve` → saved spectra → `render.render_result`.
Production version **`6627c00`** (same `skycolor/` tree as `7d7ce15`); the
example and numerical archive are committed with this catalogue. No RT kernel,
solar-source prescription, dependency or physical baseline was changed to make these images.
The [layer story](#layer-story-figures-v3-2026-11-12) figures (2026-11-12)
carry their own provenance block below; the numeric archive is the 11-run
ladder (solve version `286486e`, `skycolor/` tree identical to the four
above, zero diff through the theme commits) plus one pre-registered
2026-11-12 augmentation solve (`l3_o3_dome_3`, its own recorded compute-time
HEAD). Retired v1/v2 layer-story figure bytes live in the tracked
[layerstory historical/](../experiments/layerstory/historical/README.md)
registry.

A reader-level guide to these figures (Chinese): [docs/physics/MAINLINE.zh.md](../docs/physics/MAINLINE.zh.md).

| Figure | Purpose / what to compare | Numeric source |
|---|---|---|
| [sky_30.png](sky_30.png) | Daylight, sun 30°; default hybrid-sos | `e30_hybrid-sos/total` |
| [sky_03.png](sky_03.png) | Low sun 3°; README's current scene, **not** a solar-disk image | `e3_hybrid-sos/total` |
| [approximations.png](approximations.png) | Two scene rows × spherical-single / plane-sos total / hybrid-sos total; same display parameters | the corresponding saved total arrays |
| [spectra.png](spectra.png) | Untonemapped spectra: plane single / spherical single / hybrid total, zenith and near-horizon sun-side; units on axes | same arrays, exact sampled zenith 0° / 86.25° |
| [layer_story_overview_v3.png](layer_story_overview_v3.png) | The ladder in 2×4 **eight renders, no curve panels**: L1 / L2 / L3 dome (pre-registered augmentation) / L4 anti-solar (day row), L5 single/MC/plane-world + L6 cloud (twilight/cloud row); locked exposure per group | `l*/total` arrays of `layer_story_v3.npz` |

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

## Layer story ladder archive (2026-11-10; roots retired 2026-11-12)

The numeric authority is the [layerstory theme](../experiments/layerstory/README.md)
ladder: **11 production solves** (10 deterministic + 1 single-seed MC) through
the same production entry, committed as `data/layer_story.npz` with solve
version **`286486e`** (the compute-time HEAD; theme files committed as
`43f0881` with `skycolor/` zero diff throughout). Narration single source:
[LAYER_STORY.md](../docs/physics/LAYER_STORY.md). The two v1 root montages
(`layer_story_overview.png`, `layer_story_spectra.png`) and the 2026-11-11
v2 per-position figures were retired from root on 2026-11-12 when the v3 set
replaced them; their bytes are preserved in the tracked
[historical/ hash registry](../experiments/layerstory/historical/README.md).

```bash
uv run python -m experiments.layerstory.layer_story
```

Display: three locked groups, no per-panel auto exposure — day (L1/L2/L4) and
cloud (L6) at **exposure=.03** (the approximations value), twilight (L5) at
**exposure=10.0**, locked once from the first real −4° result (twilight zenith
Lv ≈ 4.4 cd m⁻² — the day group's zenith spans 824–1357 cd m⁻², so 187–308×
darker depending on the panel). All: perspective 640×320,
pitch14°/hfov100°, ACES, adaptation=None, preserve_chromaticity=True,
no sun marker; yaw 0° except the L4 anti-solar panel (yaw 180°). Day/twilight
`ground=False` — for those panels, below-horizon pixels follow the renderer's
no-ground display convention and are not a physically rendered surface; the
cloud panel `ground=True` renders the solve-bound deck-lit floor (saved with
the NPZ so redraws can rebind it), which is part of the L6 story. L3 is a
zenith-spectrum panel (no
render); L5 MC is qualitative, single seed, budget-limited. Media, grids and
per-call records are single-sourced in the theme README and `data/layers.md`.

## Layer story figures v3 (2026-11-12)

Redrawn from the **augmented archive**
[`data/layer_story_v3.npz`](../experiments/layerstory/README.md) by
[refresh_figures.py](../experiments/layerstory/refresh_figures.py) with zero
new solves. The v3 archive keeps the 11 ladder runs (solve version `286486e`)
bit-for-bit and adds the single pre-registered L3 dome solve
`l3_o3_dome_3` (compute-time HEAD `3b9d19c`, run record
`data/l3_dome_run.json`, preregistration `data/l3_dome_preregister.json`) —
a **mixed-source archive**: per-run solve versions are stored in the NPZ and
distinguished in the audit, manifest and every caption below; it is not a
12-solve single-version batch. Each narrative position in
[LAYER_STORY.md](../docs/physics/LAYER_STORY.md) and
[MAINLINE.zh.md](../docs/physics/MAINLINE.zh.md) carries its own figure.
Per-figure provenance (runs, exact directions, display parameters, per-run
solve versions, consumers, input hash) is exported to
`experiments/layerstory/out/refresh_manifest.json` on every run.

| Figure | Serves | Content |
|---|---|---|
| [layer_story_overview_v3.png](layer_story_overview_v3.png) | MAINLINE 图1 · LAYER_STORY 总览 | 2×4 eight renders, **no curve panels**: L1/L2/L3 dome (the pre-registered augmentation run)/L4 anti-solar + L5 single/MC/plane-world/L6 cloud, locked exposure per group |
| [layer_story_rayleigh_v3.png](layer_story_rayleigh_v3.png) | MAINLINE 图2 · LAYER_STORY L1 | L1 log-log spectra (zenith / 86.25° sun-side) + archived perspective + equidistant 180° fisheye |
| [layer_story_aerosol_v3.png](layer_story_aerosol_v3.png) | MAINLINE 图3 · LAYER_STORY L2 | near-horizon (86.25° sun-side) same-scene spectra, L1 vs L2; sky views show the L2 state only |
| [layer_story_ozone_v3.png](layer_story_ozone_v3.png) | MAINLINE 图4 · LAYER_STORY L3 | zenith spectra @ sun 3° (archived ladder rows, bridge vs +O3/NO2); sky views from the pre-registered L3 dome run (separate provenance, same medium/scene) |
| [layer_story_multiple_v3.png](layer_story_multiple_v3.png) | MAINLINE 图5 · LAYER_STORY L4 | 45° anti-solar (relative azimuth 180°, not a 90° scattering angle) bridge vs +multiple; views face the anti-solar side (yaw 180°) |
| [layer_story_twilight_v3.png](layer_story_twilight_v3.png) | MAINLINE 图6 · LAYER_STORY L5 | the three −4° states, each as perspective + equidistant fisheye pair, locked exposure 10.0; MC panel single-seed / qualitative |
| [layer_story_cloud_v3.png](layer_story_cloud_v3.png) | MAINLINE 图7 · LAYER_STORY L6 | contract-matched clear/cloud pair, both with the solve-bound floor, locked exposure 0.03 |
| [layer_story_pipeline_spectrum_v3.png](layer_story_pipeline_spectrum_v3.png) | MAINLINE 图8 | the two hybrid-sos input spectra (zenith / 81° sun-side) of the same run imaged beside |
| [layer_story_pipeline_view_v3.png](layer_story_pipeline_view_v3.png) | MAINLINE 图9 | that run's annotated perspective view (same single-source pair; no ground) |

Display: the ladder groups' locked values, unchanged — day/cloud exposure
.03, twilight 10.0, ACES, adaptation None, no sun disk, perspective 640×320
pitch14°/hfov100°.  fisheye panels are square axis boxes (no pixel ticks;
rim degree labels carry direction), perspective panels carry **camera-relative
angle axes α/β** derived honestly from the archived camera parameters through
the production pixel-centre pinhole mapping (uneven tick spacing is the
pinhole nonlinearity, not relabelled linear pixels; horizon/sun annotations
stay absolute directions).  Figure notes are generated per run: each run's
own dome grid and — only where the archive actually holds them — its
horizon-row zero cells with their relative-azimuth spans and whether they
fall inside the rect frame's azimuth window.  Measured fact (owner probe,
2026-11-12): the `l5_single_-4` truncation zeros sit at rel-az 147–180°
(anti-solar), **outside** the yaw-0° frame's ±50° sun-side window — the
perspective view does not face that side, it does not compress the structure
away; both cameras sample the sun side at comparable azimuth resolution
(0.156 vs 0.179 °/px).  Annotations (horizon, rim ticks, sun marker) are
**computed from the same camera/direction grids** the pixels came from; the
sun marker is the pre-refraction *input* direction, never a rendered disk
(a below-horizon sun gets a horizon indicator, an out-of-view sun a text
note).  The fisheye is a display re-sampling of the same dome through
`render.fisheye_grid`'s equidistant branch + `render.sample_dome` +
`color.render_rgb` — no new physics, no extra information.

```bash
# coverage audit (read-only; writes the theme's out/audit_coverage.{md,json})
uv run python -m experiments.layerstory.refresh_figures \
  --from-npz experiments/layerstory/data/layer_story_v3.npz --audit-only
# redraw all nine (plain run writes theme outputs only)
uv run python -m experiments.layerstory.refresh_figures \
  --from-npz experiments/layerstory/data/layer_story_v3.npz --figure all
# publish the named v3 files to this directory
uv run python -m experiments.layerstory.refresh_figures \
  --from-npz experiments/layerstory/data/layer_story_v3.npz --figure all --publish-figures
```

## Older figures: preserved, not re-certified

All 26 old candidates were inspected and moved **byte unchanged**; none were
silently overwritten with a lookalike. Old full generation SHAs/environments
are not reliably known, so source scripts are *locators*, not claims of exact
modern reproduction. Generated theme outputs never overwrite these `historical_`
files. Root contains only the current-production images above.

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
