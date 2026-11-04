# CSSP observation inputs

Frozen validation input for the public CSSP twilight hemispheric-irradiance
dataset (Spitschan et al. 2016, doi:10.1038/srep26756). Scope: source identity,
raw-acquisition contract, independent night screening and the minimal
observation operator. This packet does not claim an absolute-calibration
validation of the sky model; scientific interpretation lives in
[CSSP_OBSERVATION_INPUTS](../../../docs/twilight/CSSP_OBSERVATION_INPUTS.md)
and [MATCHED_OBSERVATIONS §4](../../../docs/twilight/MATCHED_OBSERVATIONS.md).

## Contents

- `sources.json`: identities for whole files, the raw archive (single
  sequential read), vendored members and verbatim public responses; distinct
  `kind` per claim level. `external_refs` pins the older frozen packets this
  packet builds on (relative paths + SHA).
- `records.json` + `spectra.csv`: per selected column - verbatim timestamp
  label and published (wrong-site) axes, corrected CSSP axes recomputed from
  the vendored upstream MICA tables, the raw `.mat` acquisition contract
  (serial, integration time, repeats, temperatures), the nominal exposure
  window with explicit unknowns (clock offset, setup latency, gaps, driver
  caching: null + reason), and per-record quality/band masks. Spectral cells
  are byte-identical excerpts of the official CSV.
- `raw/mat/`: unmodified archive members for 42 records
  (serials ['USB2+H14462']; repeats [1, 5, 10]; integration times [512000, 1000000, 2000000, 4000000, 8000000, 32000000, 60000000] us).
- `ephemeris/`: full upstream MICA solar/lunar tables (UT1 as UTC, 5-minute
  cadence) - the corrected-axis source for every night in the dataset span.
- `raw/weather/`: verbatim IEM/NOAA/AERONET responses for candidate-night
  windows. Failed probes are kept as evidence.
- `selection.json`: the full session inventory of all 3192 published columns
  plus frozen selection decisions and reasons.
- `conditions.json`: moon geometry, per-side (dusk/dawn) weather windows and
  independent JPL axis checks for each selected night (what was queried,
  what returned, and what stays missing), with data-referenced verdicts.
- `operator.json`: the mathematical observation operator with evidence
  levels; `diagnostics.json`: timing/response/background scenario tables.
- `calibration-extract.json`: reduced subset of the pinned generated
  calibration struct (meter 'a' dark library nodes and temperature rows the
  records reference, the full correction factor and wavelength shift), tied
  to the whole-file identity of its 21.8 MB source.
- `replay.json`: per-record port of the upstream Preprocess.m +
  RawToAbsolute.m chain. It reproduces the published columns to ~5e-11
  median relative difference (CSV print precision) - proof that the raw
  members and the published spectra are the same data through the same
  pipeline, not an independent absolute calibration.

## Commands

From the worktree root (offline paths never touch the network):

```bash
uv run python -I tools/cssp_observation_inputs.py verify \
  --packet validation/data/cssp_observation_inputs
uv run python -I tools/cssp_observation_inputs.py replay \
  --packet validation/data/cssp_observation_inputs
uv run python -I tools/cssp_observation_inputs.py sensitivity \
  --packet validation/data/cssp_observation_inputs
```

`replay`/`sensitivity` are read-only acceptance by default and compare against
the frozen tables plus an independent quantisation-derived tolerance;
``--write`` is the explicit re-freeze entry (build only re-freezes as part of
construction).

Acquisition (writes only to the cache directory, never to the packet). The
fetch entry covers every input build requires - official CSV, upstream MICA
tables, the meter-'a' calibration struct, IEM/NOAA/AERONET responses for all
session windows, and the two JPL spot checks; known-failing probes are
recorded as evidence and build preflight rejects an incomplete cache:

```bash
uv run python -I tools/cssp_observation_inputs.py fetch \
  --cache-dir <new-empty-dir> --raw-archive
uv run python -I tools/cssp_observation_inputs.py build \
  --source-dir <cache-dir> --out <new-output-dir>
```

Optional full-source check (complete official CSV + byte-level excerpt
comparison):

```bash
uv run python -I tools/cssp_observation_inputs.py verify \
  --packet validation/data/cssp_observation_inputs --source-dir <dir with CSSP_spectra.csv>
```

## Boundaries

- The nominal exposure window is a scenario; no measured exposure midpoint
  exists (see records.json `timing.unknown`).
- Published axes describe DRL (upstream index bug) and are retained verbatim
  for provenance; corrected axes come from the vendored MICA tables.
- Weather responses are neighbour stations; distances/elevation gaps are
  recorded and no in-situ column quantities are derived from them.
- Scenario ranges in `diagnostics.json` are demonstrations of dependence,
  not confidence intervals.
- `replay.json` documents reproduction of the upstream pipeline; the
  absolute radiometric scale rests on the upstream lab calibration and is
  not independently certified here.

## Attribution

Spitschan et al. (2016), Scientific Reports 6:26756,
[doi:10.1038/srep26756](https://doi.org/10.1038/srep26756): published spectra
and ephemeris tables CC BY 4.0; author code snapshots MIT (retained in the
matched-observation and feasibility packets). The raw archive and the
generated calibration structure are the author's public Figshare/GitHub
releases, referenced by the identities in `sources.json`. Scientific
interpretation and the use-verdict matrix live in
[docs/twilight/CSSP_OBSERVATION_INPUTS.md](../../../docs/twilight/CSSP_OBSERVATION_INPUTS.md).
