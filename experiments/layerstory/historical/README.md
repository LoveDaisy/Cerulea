# Historical v1 snapshots (byte-identical)

These files are byte-identical copies of the two v1 layer-story figures that
were selected in root `figures/` before the per-position v2 refresh
(task `figures-refresh`, 2026-11-11).  They are kept *tracked* on purpose:
the theme's `figures/` and `out/` directories are gitignored run outputs, so
this directory is the durable record of the v1 pixels, and the committed v2
figures can never silently overwrite them.

| snapshot | sha256 (first 16) | original |
|---|---|---|
| `historical_layer_story_overview_v1.png` | `23dd53a8edf06fd5` | root `figures/layer_story_overview.png` (kept in place; still the current shared overview entry) |
| `historical_layer_story_spectra_v1.png` | `55baf486146e5473` | root `figures/layer_story_spectra.png` (root copy kept as a compatibility locator; the current per-position figures are the v2 set) |

Version semantics: both v1 figures were generated from the same committed
archive (`data/layer_story.npz`, solve version `286486e`); the v2 figures in
root `figures/` are redrawn from that same archive by
`experiments/layerstory/refresh_figures.py` with zero new solves.  A plain
run of that entry writes only theme-local outputs and never touches these
snapshots or the root figures; publishing is explicit (`--publish-figures`)
and writes only the registry's own `*_v2.png` names.
