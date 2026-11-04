# Twilight reference and filter-only bandpass audit

Companion interpretation: [`docs/TWILIGHT_REFERENCE_AUDIT.md`](../../../docs/twilight/TWILIGHT_REFERENCE_AUDIT.md).
The audit does not change validation targets or identify an altitude causal effect.
The −12° night-floor calculation is a historical-mean illustration, not a matched
measurement or an instruction to subtract that value from the observations.

Files:

- `references/fors-bess-v.txt`: original ESO filter transmission, 2037 bytes.
- `references/alpha_lyr_stis_011.fits`: original fixed CALSPEC Vega spectrum, 288000 bytes.
- `source_provenance.json`: primary URLs, page/section references, download hashes and conventions.
- `inputs.sha256.json`: repository-relative identities of all five frozen calculation inputs.
- `reference-calculations.json`: arithmetic from published V coefficients/historical night mean.
- `bandpass-audit.json`: all seed metrics, means and paired ratios, conventions and computed checks.
- `bandpass-audit.csv`: compact field-level summary. Magnitudes are means of seed magnitudes;
  the correction column explicitly uses the ratio of mean linear signals.

The raw spectra are already archived in `../unified_solar_source_audit/raw/` and
are not duplicated here. No scratchpad files are required. From the repository root:

```bash
uv run python -m tools.summarize_bandpass_audit
```

This rewrites only the derived JSON/CSV, takes below one second on the audit host,
and performs no Monte Carlo, network download, or reference-data fitting.
Defaults are resolved relative to the script, so execution from another working
directory also works when the repository package is on the Python import path.
The archived input hashes are verified before recomputation.

The filter-only photon response is `lambda*T(lambda)` for energy spectral radiance.
This uses QE=1 inside 380–780nm and zero outside. It is not the old FORS1 total
instrument response or an exact Johnson-Cousins standard-system reconstruction.
AB is fixed at 3631 Jy. A separate conditional V approximation adopts CALSPEC011
and Vega V=0.031, without any fit to Patat or to model residuals.
