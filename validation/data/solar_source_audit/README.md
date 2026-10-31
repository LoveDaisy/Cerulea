# Solar-source audit evidence

Report: [SOLAR_SOURCE_AUDIT.md](../../../docs/SOLAR_SOURCE_AUDIT.md).
Baseline library: `2d0294c`. Experiments are standard clear air, AOD550 0.05,
ozone 300 DU, albedo 0.1, solar elevation -6°, zenith viewing, and globally
spherical opaque ground at 0/2.4 km. No production library or prior validation
baseline was changed in this phase.

- `pilot.jsonl`: five wavelengths 450–650 nm, 500,000 paths per wavelength,
  four independent seeds, geometric vs packaged attenuation/shadow only.
  Its wavelength-limited integrals are not the full-spectrum headline.
- `full-paired.jsonl`: 81 wavelengths 380–780 nm at 5 nm, 100,000 paths per
  wavelength, seeds 100–107, both heights and two solar prescriptions.
  All 16 mode pairs share their full sequence of NEE-position fingerprints.
  The 16 geometric baselines reproduce phase-one spectra bit-for-bit.
- `geometry-probe.json`: independent Cartesian RK shooting, continuously
  integrated columns, source-table nodes and angular derivatives.
- `geometry-refinement-pilot.json`: held-out local true-angle/height grid
  refinement. This is not certification of a new global LUT.
- `geometry-step-check.json`: RK path-length-step refinement.
- `radial-mapping-validation.json`: independent radial curvature integration
  against RK, quadrature refinement, constant-index limits and performance.
- `single-radial.jsonl`: 12 view Gauss nodes / 8 radial Gauss nodes, source
  angle/amplitude ablations and 5/9-node vertical solar-disk marginal.
- `single-radial-refined.jsonl`: 24/16 Gauss refinement and 17 disk nodes;
  retains geometric view, phase and production shell optical depths.
- `summary.json`: means, paired differences and empirical ratio standard
  errors, derived by `tools.summarize_solar_source`.

Spectra are W m^-2 sr^-1 nm^-1; photopic integrals are cd m^-2 over the stated
wavelength range. Probe angles are degrees; heights km; path steps metres.
Errors are empirical across-seed standard errors, not a complete physical
error budget. Finite-disk records are a one-dimensional vertical marginal
approximation, not a complete 2-D disk or all-path refracted RT solution.

The recorded `head` anchors the production library. `tool_sha256` identifies
the running entry script, not its entire dependency graph. Final tool hashes
and source commit identity are listed in `source_provenance.json`; the radial
reference's numerical implementation was frozen before the refined run.
This distinction matters because the new tools were untracked during initial
experiments. No single script hash is claimed to identify all runtime code.

Use fresh JSONL output files for a rerun, then summarise matching archived
records. No published reference values or prior golden fixtures are replaced
by these exploratory results.
