# Historical layer-story figure snapshots (hash registry)

This directory is the durable record of retired layer-story figure bytes.
The theme's `figures/` and `out/` directories are gitignored run outputs, so
these *tracked* snapshots are the only copy of the retired pixels; the table
below is the single source for their hashes (pinned by
`test_historical_snapshots_match_recorded_hashes`).

- **v1** (2026-11-10, task-figures-refresh): the two grid montages generated
  by `layer_story.draw` from the committed archive (solve version `286486e`).
- **v2** (2026-11-11, task-figures-refresh): the eight per-position figures,
  redrawn from the same archive by `refresh_figures.py` with zero solves.
- Retired on 2026-11-12 (task-figures-polish): the v3 set replaced both
  generations in root `figures/`; the retired root copies were byte-copied
  here first, and the v1 root files (kept as locators through the v2 era)
  were removed at the same time.

| snapshot | sha256 (first 16) | original |
|---|---|---|
| `historical_layer_story_overview_v1.png` | `23dd53a8edf06fd5` | root `figures/layer_story_overview.png` (v1 eight-panel montage: 4 day renders + L3 zenith spectrum + L5 triple) |
| `historical_layer_story_spectra_v1.png` | `55baf486146e5473` | root `figures/layer_story_spectra.png` (v1 five-spectra montage + chromaticities; compatibility locator through the v2 era) |
| `historical_layer_story_rayleigh_v2.png` | `2afabe1789b23891` | root `figures/layer_story_rayleigh_v2.png` (v2 per-position set) |
| `historical_layer_story_aerosol_v2.png` | `1ba1797834506048` | root `figures/layer_story_aerosol_v2.png` (spectrum-only in v2) |
| `historical_layer_story_ozone_v2.png` | `6e4d0ac047b8afc3` | root `figures/layer_story_ozone_v2.png` (spectrum-only in v2; no L3 dome existed yet) |
| `historical_layer_story_multiple_v2.png` | `1324b573f3a4a2ab` | root `figures/layer_story_multiple_v2.png` (spectrum-only in v2) |
| `historical_layer_story_twilight_v2.png` | `26482b1bfea0a328` | root `figures/layer_story_twilight_v2.png` |
| `historical_layer_story_cloud_v2.png` | `0ac9d8aa0a6df89d` | root `figures/layer_story_cloud_v2.png` |
| `historical_layer_story_pipeline_spectrum_v2.png` | `bc039587167e7564` | root `figures/layer_story_pipeline_spectrum_v2.png` |
| `historical_layer_story_pipeline_view_v2.png` | `e5db66a8acf28788` | root `figures/layer_story_pipeline_view_v2.png` |

Version semantics: every generation here was produced from the *same*
committed numeric archive (`data/layer_story.npz`, solve version `286486e`)
-- the v1 generation by `layer_story.draw`, v2 and the current v3 set by
`refresh_figures.py` (zero solve, drawing-code version recorded separately
in the audit and manifest).  A plain run of the drawing entries writes only
theme-local outputs and never touches these snapshots; publishing is
explicit (`--publish-figures`) and writes only the registry's own current
names (`*_v3.png` since 2026-11-12).
