# Matched twilight observations: frozen availability and metadata probes

2026-11-04, model baseline `96b021e`. Scientific conclusions and comparison
limits live in [MATCHED_OBSERVATIONS.md](../../../docs/twilight/MATCHED_OBSERVATIONS.md).
This packet contains observation evidence, without new sky-model predictions or
changes to the six-check reference targets.

## Contents and scope

- `spectral/`: 35 selected original CSSP spectral columns, their unmodified
  timestamp/metadata fields, exact MICA interpolation knots for both sites,
  source-code snapshots and the full-audit diagnostic summary. Selection includes
  the original wrong-axis counterexample, the reconstructed-axis nearest spectra,
  and both interpolation brackets. Column indices are zero-based.
- `fors/`: actual downloaded FITS header text, raw-catalogue query results,
  an official remote 1999 header, the 2005-08-02 gain configurations, old QE and
  QC/provenance. Full FITS images are publicly retrievable but not vendored here.
- `photometric/`: actual Galicia/TESS raw-sequence extracts and source/range
  provenance. A range or decoded-prefix SHA does not validate the complete
  upstream archive/member. The selected nights have not been certified clear.
- `atmosphere/`: actual contemporaneous ESO METEO CSV responses and request URLs.
  Surface pressure/temperature are available; a matched whole atmosphere is not.
- `independent/`: the separate observer-position probe and original JPL Horizons
  AIRLESS response for one CSSP sample. Recorded scratchpad paths are historical
  provenance strings, not runtime dependencies.
- `packet-provenance.json`: original full-CSV identity, selection and omissions.
  `inputs.sha256.json` freezes every other file in this packet.

The 3192-column official Table S2 and the pinned GitHub CSV are byte-identical:
SHA256 `e0e3e895b3a66fc83d9f63bd02e92899a11a44b1d1707f8a6d7a05279cbe6bab`,
25,481,368 bytes. The archived summary's all-column ephemeris residuals describe
the completed full-source audit. Default offline verification independently
replays the archived selected spectra/knots only; it does not pretend to rerun
the omitted 3192-column data.

## Recheck

From the repository root, without network access or any sky-model calculation:

```bash
uv run python validation/data/matched_twilight_observations/verify.py
```

This checks hashes, spectral integrals, original DRL-axis reproduction,
reconstructed CSSP coordinates, interpolation brackets, the independent JPL
sample, downloaded FITS heights and the actual raw minute-record count.
It writes no files.

For the optional full 3192-column replay, download the official Table S2 as
`CSSP_spectra.csv` and the four full `CSSP/DRL_Sun/MoonLocalZenith.txt` files
into a separate directory. Exact URLs/hashes are in `spectral/` provenance;
GitHub sources are pinned to `1d2ecc69cb15885b147d2831cae89fbd15d471fa`.
The extracted-knot metadata gives full upstream MICA hashes and line numbers.

```bash
uv run python validation/data/matched_twilight_observations/verify.py \
  --full-source-folder /path/to/downloaded-sources
```

The optional mode verifies the complete CSV SHA, re-extracts all archived
spectral columns, and repeats all 3192 solar/lunar metadata comparisons using
the full original tables. Do not overwrite the frozen packet to rerun queries.

## Attribution and interpretation

Spitschan et al. (2016), Scientific Reports 6:26756,
[doi:10.1038/srep26756](https://doi.org/10.1038/srep26756), published spectra
and selected derivatives: CC BY 4.0; author code snapshot: MIT, retained in
`spectral/LICENSE-MIT.txt`. Data provenance preserves both the official
supplement and the pinned author repository. The separate derived metadata
uses the stated station and timestamps; it is not a brightness-fitted axis.

Bará (2016), [doi:10.1098/rsos.160541](https://doi.org/10.1098/rsos.160541),
Galicia source [doi:10.5061/dryad.kv072](https://doi.org/10.5061/dryad.kv072).
TESS source [doi:10.5281/zenodo.4265155](https://doi.org/10.5281/zenodo.4265155).
ESO raw records, QC, detector response and METEO use their original archive
URLs in the respective provenance files.

CSSP spectra measure hemispherical irradiance, FORS measures directional sky
through an aperture/slit, and SQM/TESS integrate their angular/spectral response.
None is interchangeable with a photopic zenith point. Cloud, moon, background,
instrument response and atmospheric matching remain separate checks.
