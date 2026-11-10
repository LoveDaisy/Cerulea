# production_solar_source — tracked acceptance records

Current-version acceptance evidence for the unified solar source (plan §6.4).
Regenerate from the repository root; the suites return non-zero when any
gate or threshold fails (exit 1 = honest failures present, see the JSON
`failed`/`deviations` lists):

```bash
uv run python -m validation.production_solar_source --suite source --out /tmp/source.json
uv run python -m validation.production_solar_source --suite closure --out /tmp/closure.json
uv run python -m tools.benchmark_solar_source --out /tmp/perf-final.json
```

* `source.json` — source-level thresholds (RK direction anchors, holdout
  direction/TG, column identities, disk quadrature). Current known failures:
  the RK direction anchors (2.673e-4 deg vs the 2e-4 deg plan budget),
  adjudicated 2026-11-06 as the monotone-PCHIP representation limit
  (near-caustic fold rows); see `docs/physics/SOLAR_SOURCE.md`.
* `closure.json` — deterministic-vs-MC first-order closure and signability.
  Current known failures: the five deep-twilight items (-9/-12 deg, both
  grounds, at the plan's 8x500k budget cap), adjudicated 2026-11-06 as the
  terminal budget-limited-unresolved state (corroborated by the archived
  deep-twilight-reference task, ROADMAP M19); see `docs/physics/SOLAR_SOURCE.md`.
* `perf-final.json` — availability gate (plan §6.5). Current known breach:
  clear dome 5.9-6.3x legacy (budget 3x), structural to the per-beam
  contract, accepted as a recorded deviation with a documented resolution
  knob; all other rows pass; see `docs/physics/SOLAR_SOURCE.md`.
* `perf-dome-kernel.json` — dome query-kernel vectorization evidence
  (task-dome-kernel, same measurement contract and legacy anchors).  The
  row-group Python loop is gone (bitwise-equal vectorized kernel); the
  standing breach is clear dome 5.5x legacy (budget 3x), measured 6.7x on
  the branch point in the same idle window (7.45 -> 6.11 s for the 46x91
  3-wavelength warm dome).  The JSON carries a `provenance` block with the
  before/after pair, the loaded-window disclaimer and the baseline record;
  residual cost attribution in `docs/physics/SOLAR_SOURCE.md` item 3.
