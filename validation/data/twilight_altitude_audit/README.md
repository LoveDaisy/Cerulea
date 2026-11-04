# Twilight altitude audit records

Report: [TWILIGHT_ALTITUDE_AUDIT.md](../../../docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md).
Baseline commit: `a15555b`. All runs are zenith, true solar elevation -6°,
clear aerosol AOD550 0.05, ozone 300 DU, Lambertian albedo 0.1, unless named
geometric. Radiance arrays are W m⁻² sr⁻¹ nm⁻¹; photopic integrals are cd m⁻²
over each record's stated wavelength range.

- `baseline-deterministic.jsonl`: before the raised-ground occultation fix;
  full 380–780 nm / 5 nm grid, both altitudes.
- `corrected-deterministic.jsonl`: the same grid after correction.
- `baseline-direct.jsonl`, `corrected-direct.jsonl`: matched-seed occultation
  ablation, five wavelengths 450/500/550/600/650 nm, 1 million paths per
  wavelength, seeds 0–3. Their restricted-band integrals are **not** the
  full-spectrum headline numbers.
- `geometric-deterministic.jsonl`: refraction off, as an approximation audit;
  refraction remains on by default in production.
- `full-direct.jsonl`: 81 wavelengths directly computed with no angular or
  spectral interpolation; 100,000 paths per wavelength and seeds 100–107 at
  each altitude. Full geometric MC, geometric split and hybrid split are
  recorded separately. Their shared higher orders remain geometric.
- `engine-low.jsonl`: production engine cross-check, seeds 0–3 at each
  altitude, 5 spectral anchors with 4 million paths each on 6×8 directions.
  Results agree with the direct estimate within replicate scatter; this does
  not certify interpolation accuracy in deeper twilight.
- `single_quadrature.json`: single-scattering quadrature sensitivity;
  these are probes, not replacements for any golden baseline.
- `summary.json`: reproducibly derived means, replicate scatter and paired
  ratio standard errors / empirical t intervals. Random errors exclude
  transport, scenario and photometric-response errors.
- `koomen1952_zenith_pairs.csv`, `.meta.json`: selected original-paper zenith
  cells, manually read from Tables I/II, p354 (not OCR); source URL, units,
  observation conditions and limitations are in the metadata.

`head` and `source_diff_sha256` identify the baseline and local source diff
at run start. The correction was not yet committed during the runs; later
docstring/capability-label edits change source hashes without changing the
numerical experiment. Local absolute worktree paths have been removed.
`source_provenance.json` identifies the final numerical implementation in
commit `c45a03b` and hashes its library diff from `a15555b`. A temporary patch
snapshot was verified byte-for-byte against that commit and is superseded by
the committed source itself. Later docstring/capability corrections mean that
this final diff does not reproduce the historical pre-docstring diff hashes.
Those original dirty patches were not saved; exact verification of their
hashes is unavailable. Both baseline and final source are recoverable by Git.
An older generic `photons` field was normalised to `photons_per_wavelength`
for direct MC; it was unused in deterministic/engine mode. Engine runs use
their declared quality configuration, not that unused CLI default.

Reproduce current calculations with `tools.twilight_altitude_audit`; the
pre-fix records are historical measurements from the baseline source.
Rebuild summary/figure with `uv run python -m tools.summarize_twilight_altitude`.
The original article and local screenshots are not redistributed; the public
source link and transcription coordinates suffice to inspect the source.
