# Twilight methods and audits

Current production behavior is summarised in [STATUS.md](../STATUS.md); completed
work and proposed next scopes are in [ROADMAP.md](../ROADMAP.md). The experimental
unified source and continuous transport have not replaced production defaults.

## Engine and historical sensitivity study

| Document | Scope |
|---|---|
| [TWILIGHT.md](TWILIGHT.md) | Production split estimator; engine benchmarks retain their historical optics |
| [PROFILES.md](PROFILES.md) | Pre-M10 profile sweeps; the old excess and its causal interpretations are superseded |

## Audit sequence (M13–M18)

Each report preserves a different intervention and its conditions. Their numerical
results are not interchangeable. Scripts and frozen evidence remain in the
repository's `tools/` and `validation/data/` trees, linked from each report.

| Document | Question and outcome |
|---|---|
| [TWILIGHT_ALTITUDE_AUDIT.md](TWILIGHT_ALTITUDE_AUDIT.md) | Raised-ground occlusion fixed; domain truncation distinguished from observed station changes |
| [SOLAR_SOURCE_AUDIT.md](SOLAR_SOURCE_AUDIT.md) | Paired source ablations; inverse-angle and angular-gain limitations isolated |
| [UNIFIED_SOLAR_SOURCE_AUDIT.md](UNIFIED_SOLAR_SOURCE_AUDIT.md) | Continuous columns and finite solar disk; deterministic/MC source closure |
| [AEROSOL_GRID_AUDIT.md](AEROSOL_GRID_AUDIT.md) | AOD550 slicing defect fixed; golden and physical-check changes independently attributed |
| [TWILIGHT_REFERENCE_AUDIT.md](TWILIGHT_REFERENCE_AUDIT.md) | Primary observations, solar-axis uncertainty and limited synthetic V response |
| [CURVED_TRANSPORT_AUDIT.md](CURVED_TRANSPORT_AUDIT.md) | Continuous diffuse refraction, independent checks and final exploration limits |
| [DEEP_TWILIGHT_REFERENCE.md](DEEP_TWILIGHT_REFERENCE.md) | −9°/−12° zenith reference: pre-registered budgets/gates, paired finite-disk correction, two-scatter candidate and tail diagnostics |

## Matched-observation follow-up

[MATCHED_OBSERVATIONS.md](MATCHED_OBSERVATIONS.md) records the 2026-11-04
raw-data checks: recoverable FORS metadata and same-twilight candidates, an
independently identified station-ephemeris mismatch in published CSSP spectra,
and original photometric time series. These are candidate validation inputs;
instrument, background and atmospheric matching are still incomplete.

General scattering/refraction methods are in [physics/](../physics/README.md).
Reference transcription is in [SECOND_DATASET.md](../../validation/SECOND_DATASET.md).
[Documentation guide](../README.md).
