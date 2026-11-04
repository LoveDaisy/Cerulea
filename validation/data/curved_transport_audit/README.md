# Frozen continuous diffuse-transport audit

The conclusion and its scope are in
[`docs/CURVED_TRANSPORT_AUDIT.md`](../../../docs/twilight/CURVED_TRANSPORT_AUDIT.md).
This directory is an experimental reference, not the production solver default.

## Main inputs and reduction

`raw/` preserves complete spectra and run metadata for:

- `curved-ground0.jsonl` and `curved-ground2p4.jsonl`: eight seeds310–317,
  81 wavelengths,10k paths/wavelength/seed,2km RK step,2×8 solar disk.
- `global-straight-high-stat.jsonl`: same scene, seeds and disk,100k paths,
  straight flights with the same global-majorant construction.
- `global-straight.jsonl`: original10k-path control, retained after increasing
  the cheaper control budget.
- `native-new-seeds-ground2p4.jsonl`:100k-path original coarse-majorant control
  with the same new seeds. The source-binding precondition was added before
  this run; the numerical source is unchanged.
- `step-refinement-ground0.jsonl`: seed310 at1km instead of2km.
- `curved-pilot-ground0.jsonl`: initial2k-path probe, excluded from main means;
  its large fluctuation is retained as sampling evidence.

`inputs.sha256.json` freezes those input bytes. The reducer checks the manifest,
recorded source/table hashes, scene/disk/spectral parameters and seed coverage,
then reintegrates the raw spectra. It does not substitute recorded aggregates
for that integration. Table bytes and their medium/label contracts are checked
by the preceding [unified-source archive](../unified_solar_source_audit/README.md).

```bash
uv run python -m tools.summarize_curved_transport \
  --directory validation/data/curved_transport_audit/raw \
  --straight-file global-straight-high-stat.jsonl \
  --out validation/data/curved_transport_audit/summary.json \
  --figure validation/data/curved_transport_audit/summary.png
```

`summary.json` preserves seed replicates, empirical SEM, within-path variance
diagnostics, half-batch means, covariance-aware ratios and conditional Student
intervals. Eight batches do not prove unknown rare-path tails have converged.
The unequal curved/control budgets are explicit. The independently checked new
native/global comparison is in `majorant-control-summary.json`.

## Independent and negative evidence

- `validation/`: rejected Snell-layer artificial ducts; continuous trajectory,
  first-boundary and constant-index checks; composition review; native-exit
  audit. A thin-shell result is not accepted as a continuous-atmosphere result.
- `independent-single/`: DOP853 oblique one-scatter integrals and three raw MC
  batches. Both the initially low70° batch and the additional independent
  high-count batch remain available.
- `disk-convergence/`: same-path16/64 solar-node comparisons, eight straight
  seeds and one curved seed at each height; no claim about unsampled vertices.
- `final-review/`: independent Hamiltonian flights, phase moments, the actual
  wrong-medium table counterexample, its verified fix, historical source files
  and a separate numeric closeout review. Initial pending statements are not
  overwritten by the later review.
- `majorant-control/`: optical-field/upper-bound audit and the additional
  same-seed native/global control interpretation.
- `logs/`: completed main runs, the interrupted attempt, its faulthandler
  continuation, and the final316-pass full pytest log.

The high-site initial process exited139 after writing seeds310–312, with no
Python traceback or corresponding crash report. Its cause remains unknown.
Seeds313–317 were rerun successfully in a fresh faulthandler-enabled process;
the frozen combined file contains each required seed exactly once. A completed
continuation is not a diagnosis of the original native failure.

## Runtime-source provenance

Experimental flight/source composition is checkpointed in commit`2756b2f`.
The long runs used the source before its six-line binding guard. Applying the
reverse of `source-binding-guard.patch` to the checkpoint source reconstructs
the exact runtime source SHA recorded in the JSONL;
`runtime-source-provenance.json` independently verifies both hashes. The guard
adds construction checks only. Existing loaders had already validated the
correct table for each medium.

Runs started before and after committing the aerosol fix have different HEAD /
dirty-diff metadata but the same actual physical library. The exact aerosol
patch and before/after source hashes are in the
[grid-invariance archive](../aerosol_grid_audit); disk-convergence also retains
an independent dirty-to-committed-library check. This does not grant arbitrary
hash changes an exemption: these concrete differences were read and checked.

All main paths use the corrected full81 medium, same finite solar source,
scalar unpolarized transfer, Lambertian ground and the original MC's50-order
limit/Russian roulette. These inherited approximations and rare-tail sampling
are outside the error bars. No atmosphere, solar-axis shift or observational
zeropoint was fitted.

## New experiments

Use the source table manifest in `validation/data/unified_solar_source_audit/`
with `tools.run_unified_solar_audit --transport continuous --flight-step 2000`.
Select the complete81-point grid, matching height, disk and explicit photon
budget. Use a **new** `--out` JSONL; the driver appends, so never point it at a
frozen input. Curved full ensembles can take tens of minutes.

Independent probes use `tools.check_curved_single`,
`tools.check_curved_hamiltonian`, `tools.check_solar_disk_mc` and
`tools.check_majorant_equivalence`; their own CLIs specify inputs/output.
Use a temporary output location when checking an immutable historical record.
