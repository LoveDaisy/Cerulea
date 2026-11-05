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
  the RK direction anchors (2.67e-4 deg vs the 2e-4 deg plan budget) — an
  owner-level tabulation-design adjudication (near-caustic rows), recorded
  in the task dispute record.
* `closure.json` — deterministic-vs-MC first-order closure and signability.
  Current known failures: deep-twilight (-9/-12 deg) `SE/|SS|` signability at
  the plan's budget cap — budget-limited, owner adjudication.
* `perf-final.json` — availability gate (plan §6.5). Current known breach:
  clear dome ~7.5x legacy (budget 3x), structural to the per-beam contract;
  owner adjudication.
