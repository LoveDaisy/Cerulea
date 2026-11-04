# Deep-twilight reference data (−9°/−12°, sea-level zenith)

Frozen inputs, append-only raw records and recomputed summaries for
`task-deep-twilight-reference` (see `docs/twilight/DEEP_TWILIGHT_REFERENCE.md`
for the narrative and `scratchpad` task dir for the process ledger).

## Layout

- `protocol.json` — the pre-registered contract: scenario/medium/source,
  budgets, seeds, interval methods, precision gates, wall-clock caps,
  verdict policy. `schema: deep-twilight-reference/1`. Nothing here may be
  edited after data exists; changes mean a new protocol generation.
- `probe/` — the three planning-phase probe batches (byte-preserved, frozen
  hash manifest `inputs.sha256.json`), including the 3-wavelength planning
  solar table reused by the candidate/sensitivity experiments.
- `raw/direct/` — the formal 16-seed × {400k, 1.6M} × {−9°, −12°} disk-source
  baseline (`s<sun>b<budget>/attempt-NN.jsonl`).
- `raw/source-pairs/` — point-source pairing at 1.6 M with the same seeds
  (`s<sun>b1-point/`).
- `raw/candidate/` — exactly-two-scattering candidate experiment
  (`s<sun>t<tier>/`, frozen 3-wavelength planning table, seeds 80000+).
- `raw/sensitivity/` — discretisation/refinement batches (seeds 110000+).
- `state.json` — cumulative per-stage wall-clock against the protocol caps.
- `manifest.json`, `summary.json` — written only by
  `tools/summarize_deep_twilight_reference.py`.

## Reproduce

```bash
# formal stages (append-only; reruns resume missing seeds only)
uv run python -m tools.deep_twilight_reference --stage direct
uv run python -m tools.deep_twilight_reference --stage source-pairs
uv run python -m tools.deep_twilight_reference --stage candidate
uv run python -m tools.deep_twilight_reference --stage sensitivity
# recompute everything from raw and verify the manifest
uv run python -m tools.summarize_deep_twilight_reference \
    --directory validation/data/deep_twilight_reference
uv run python -m tools.summarize_deep_twilight_reference \
    --directory validation/data/deep_twilight_reference --verify-only
```

## Invariants the tooling enforces

- one *valid complete attempt* per batch; duplicated recomputation is
  refused, never merged as extra seeds; corrupt trailing lines invalidate
  an attempt and force a fresh one;
- record-level identities (`total = single + multiple`,
  `split_total = deterministic single + multiple`) are re-checked at
  summary time; disk/point pairs must share the source fingerprint
  (same native paths);
- an all-zero multiple spectrum is a photon-budget detection limit
  ("no-tail-evidence") and fails the precision gate by construction;
- `--verify-only` exits non-zero on any hash mismatch, missing batch or
  summary/recomputation divergence.

## Provenance

Hardware, thread environment, code and source-table SHA256s are frozen in
each batch `meta.json` and in the summary. The solar source is the archived
unified-source audit table (`phase3-clear-81-ground0-refined`, hash in
`protocol.json`); it is loaded by explicit label and never rebuilt.
