# Fine-grid solve cost

**2026-11-16 — COST_GATE_FAILED. No production optimization retained.**

This experiment does **not** authorize the demo's full scan or certify its M1a
milestone. The measured production-dome route does not support the requested
1848-state generation within 8 hours, including 20% overall headroom. This is a
failure of the present cost admission evidence, **not a proof that every possible
single-source implementation is too slow**. Two bounded optimization candidates
were rejected; production was restored to the original byte-identical baseline.

The authoritative numeric ledger is
[`fine_grid_solve_cost.json`](fine_grid_solve_cost.json).
Its `decision.projections`, `runs`, `references`, `replay`, `candidates`, and
`budget_ledger` retain the operands, signatures, requests and resource decisions.
The tables below round those values, not a second cost model.

## Identity and scope

- Apple M2 Max, 12 logical CPUs, 32 GiB, macOS 14.7; CPython 3.12.11,
  NumPy 2.5.4 / system Accelerate. No thread-limit environment overrides.
- Production baseline: `ffe0132e343b02e37512851d2711f508c6c3e5e1`.
  Production-tree SHA256:
  `c827acf82fb21bf2939fc233b9f7169d89b52e2a7b2401d6c78496a7bb18d34d`.
  Lock SHA256: `7fc3e9da983cdf715ac9ac69b8819261f774fe4b2599c5585454c1a3e16f2a99`.
- Final `skycolor/`, `tests/`, `uv.lock`, and `validation/results.json` are unchanged
  from that baseline. Results SHA256:
  `53adf2662c632a04d2ded0d4b9ac8271df9c37838fb6825cbbe389e35a207d5d`.
- Each call uses production `render.solve_dome`: 91 zenith nodes from 0–90°,
  181 relative-azimuth nodes from 0–180°, `WL_COARSE` (81 wavelengths), sea level,
  `hybrid-sos`, refraction enabled. Single defaults are Gauss 3 / chunk 512;
  SOS and finite-disk defaults are untouched. The display-bound direct-solar /
  ground preparation is included, not merely the radiance kernel.
- The target comprises 24 media (two presets × six AOD × two DU), 77 solar heights,
  **1848 states**, with S/M giving **48 component volumes**. Do not double the
  state count for the components. No formal 1848-state scan was run.

## Fixed work and resource protocol

Every worker count ran the same 16 requests in a fresh `spawn` pool. Each worker
received at least one of eight identical difficult requests
(hazy / AOD .30 / DU 0 / solar elevation −6°). The other eight requests cover
30°, 3°, .25°, −.125°, −3°, −6°, 90°; both clear/hazy; AOD 0, .05, .10, .30;
and DU 0/300. The exact joint combinations are in `runs[].records[].request`.
This is a cost sample, not exhaustive coverage of every medium and solar angle.

Worker bundles are static; each process owns its medium cache and output files.
The four-, six- and eight-worker pools run **sequentially**, never concurrently.
State timing includes constructor/lazy preparation, full solve, compressed NPZ,
readback, hash, finite/shape/grid checks and exact S+M closure. Throughput uses
all 16 completions and the entire pool wall, including tail and startup.
Neither cProfile nor tracemalloc is enabled in throughput runs.

A separate sampler records the simultaneously resident process-tree RSS, system
VM counters, swap usage and load every five seconds. RSS is a residency
measurement, **not live NumPy allocation**. Existing swap does not disqualify a
run by itself: increases relative to the initial sample or new swapouts do.
Load remains a separately reported uncertainty, not a silently corrected time.
Other sessions were not stopped or modified.

| Run | Pool wall (s) | States/s | Process-tree peak (GiB) | New swapouts (pages) | 1-minute load min–max |
|---|---:|---:|---:|---:|---:|
| initial / 4 | 1195.27 | .013386 | 5.70 | 0 | 2.80–22.74 |
| initial / 6 | 1323.25 | .012091 | 7.52 | 74060 | 9.09–163.97 |
| initial / 8 | 1161.51 | .013775 | 9.24 | 93136 | 9.85–305.06 |
| sole retake / 4 | 1130.89 | .014148 | 6.86 | 0 | 3.34–6.57 |
| sole retake / 6 | 998.50 | .016024 | 9.59 | 0 | 2.12–7.83 |
| sole retake / 8 | 828.49 | .019312 | 10.09 | 0 | 2.27–91.34 |

The initial six/eight-worker records are memory-ineligible, not discarded.
One retake was predeclared after observing that pressure. It was the **only**
retake. The low-load four/six-worker windows still do not show linear scaling;
the eight-worker window remains load-contaminated. These data do not identify
memory bandwidth, BLAS threading, or external load as the unique root cause.

Separate tracemalloc on the difficult request measured **1,099,406,061 bytes**
of tracked peak allocation. Its instrumented solve took **403.963 s**, excluded
from throughput. This does not measure all native-library allocations. Its
output signatures match the non-instrumented difficult request.

## Cost ruler and admission result

The executable authority is `tools.fine_grid_cost.projection`:

```text
empirical = 1848 * pool_wall / 16
C = maximum measured complete-state time (including cold states, conservatively)
B = pool-startup upper bound + 24 * maximum measured complete cold-state time
O_tail = maximum measured worker-bundle duration
T = max(empirical, B + ceil(1848 / p) * C + O_tail) / 0.8
```

The intended full-scan schedule assigns each complete 77-height medium to just
one worker: 24 medium jobs, not 24 builds per worker. All measured worker-local
first-use costs remain in the measured states. B additionally charges 24 full
cold states **serially**, on top of the 1848 complete states. This overcounts
cold work instead of assuming unmeasured perfect preparation parallelism.
`startup_bound_s` derives from the budget start to the last worker's first
started record; the original raw `startup_s` only measured submission latency.
The published `runs[].startup_s` uses the corrected bound.

State-level compression and numerical verification are already included in C
and empirical throughput; `O_compress_extra_s = O_verify_extra_s = 0` means
**no additional pass in this concrete lossless strategy**, not that compression
or verification is free. `O_tail` deliberately overcounts a whole observed
bundle. Pool wall also includes tail. These are conservative accounting choices,
not calibrated predictions of their separate underlying physical costs.

| Run | C (s) | B (s) | Empirical incl. margin (h) | Adopted incl. margin (h) | Capacity worker-hours¹ |
|---|---:|---:|---:|---:|---:|
| initial / 4 | 353.83 | 8492.15 | 47.94 | **60.12** | 240.50 |
| initial / 6 | 487.20 | 11469.03 | 53.07 | **56.54** | 339.27 |
| initial / 8 | 659.08 | 15818.40 | 46.58 | **58.76** | 470.08 |
| sole retake / 4 | 324.93 | 7798.54 | 45.35 | **55.22** | 220.90 |
| sole retake / 6 | 357.18 | 8572.53 | 40.04 | **41.52** | 249.13 |
| sole retake / 8 | 415.67 | 9976.23 | 33.23 | **37.09** | 296.73 |

¹ `p*T`, including headroom and idle capacity, **not consumed CPU time**.
Actual per-state process CPU, worker offset and pool wall are retained for
separate CPU/allocated-worker-hour accounting.

**No worker count is adoptable.** All cost projections exceed 8h, even before
considering any new demo compression or shader-validation pass. No positive
remaining budget can be allocated to those unmeasured stages. Sample maxima
are not mathematical full-domain upper bounds: .025/.20 AOD, most heights,
other contact-band patterns and long-run memory retention were not swept.
Fine-grid shared-kernel assembly, future PCA, local-detail tables, shader and
browser validation were not certified. Coarse 31×46 historical timings do not
enter this ruler. Cold solar-table/Mie preparation is bounded inside full cold
states; its individual contributions were **not independently isolated**.

## Bounded candidates and rollback

1. **Prepare phase theta/u once across wavelengths.** Both the original-phase
   micro probe and two complete production requests were byte-identical.
   Complete hard/day times were 294.815→235.804 s (**1.250×**) and
   157.616→143.846 s (**1.096×**). The windows had different load (peaks
   57.54→25.45), so those ratios are observations, not clean causal speedups.
   They neither certify parallel improvement nor support an 8h admission.
   The candidate was fully reverted.
2. **Shared-grid vector interpolation.** A 10404×81 micro probe differed in
   **3854 elements**, maximum absolute difference `1.0658141036401503e-14`.
   The precise instruction-level source of those differences was not resolved;
   the byte gate alone suffices to reject it. It never entered production.

The two-candidate boundary is exhausted. No new global cache, changed quadrature,
phase approximation, chunk default, tolerance, golden or physical result was
retained. Spending the remaining compute allowance on further candidates would
exceed the agreed revision scope, even though time remains.

## Integrity, software verification and budget

- All original eight records (seven distinct requests) match every originally
  archived array's dtype, shape and element bytes, including S/M/total and axes;
  the first record also includes apparent-sun/visibility fields. Their
  `effective` and convergence diagnostics match. Original NPZ SHA256s are
  rechecked, not replaced by newly written archives. Newly captured ground
  fields have cross-worker checks; no historical ground field is invented.
- All eight repeats in each of six pools have equal signatures, including the
  newly archived display fields. Every state is finite and SOS-converged;
  S+M is exactly total. Completion files, started worker/request, archive hash,
  grid and production identity are checked before exporting evidence.
- Initial full test run: **1 failed, 587 passed / 570.91 s**. The new tool test
  incorrectly lived in the production suite and violated its existing ban on
  importing `tools`. It was moved to `tools/tests`, not hidden with a dynamic
  import or a weakened guard. Default collection is unchanged: **389 fast +
  193 slow = 582**. No scheduling-table or entry-point count changes remain.
- Final explicit tool contracts: **5 passed**. Final default fast run:
  **389 passed / 63.78 s**. Final full run: **582 passed / 432.65 s**;
  three warnings (one deprecation, two SciPy overflow warnings) are retained in the log. This
  single full run is **not a new formal certification pair**. No final numerical
  production change remains, so the six physical checks were not rerun and
  their existing 3 PASS / 3 FAIL snapshot was not modified.
- Ledger compute: **8252.27 s** plus a conservative **60 s** allowance for the
  initial audit, small unjournaled inspection and export bookkeeping; **2.309 h
  total charged/bounded**, below the 4h ceiling. Tests are separately logged,
  not charged as performance probes. There were **103 new dome calls**:
  96 calibration + 2 missing-reference replay + 1 instrumented + 4 paired-probe
  calls. The original eight references were read-only.

## Reproduction and evidence limits

From a fresh checkout, validate the **recorded** arithmetic and signatures,
without pretending to rerun the physical measurement:

```bash
uv sync --frozen
uv run python -m tools.fine_grid_cost recheck-report \
  --report docs/performance/fine_grid_solve_cost.json
uv run python -m pytest -q -n 0 tools/tests/test_fine_grid_cost.py
```

For exact operands / named fields behind the rounded tables:

```bash
uv run python - <<'PY'
from tools.fine_grid_cost import load, projection
report = load('docs/performance/fine_grid_solve_cost.json')
for run, row in zip(report['runs'], report['decision']['projections']):
    print(row['run'], projection(run), row['resources'])
print('compute_charged_s', report['compute_charged_s'])
print('audit_and_export_allowance_s', report['audit_and_export_allowance_s'])
print('memory_probe', report['memory_probe'])
print('candidates', {k: v for k, v in report['candidates'].items()
                     if k.endswith('micro')})
PY
```

Raw spectra, source-environment details, per-worker JSON/started files,
`resources.jsonl`, candidate patch/scripts and test logs live in local ignored
`scratchpad/task-fine-grid-solve-cost/`. The JSON's relative archive paths,
container SHA256s and per-array signatures index those outputs. Original
references live in the unchanged local `task-web-demo` archives; they are not
bundled into Git. Fresh-clone recheck is therefore an audit of recorded evidence,
**not an independent numerical reproduction**.

For a new actual calibration, retain old evidence and use a newly authorized
task ledger. Do not erase an existing budget or use `--label` to reset it:

```bash
uv run python -m tools.fine_grid_cost audit-baseline \
  --task-dir <new-task-dir> --reference-dir <local-original-reference-dir>
uv run python -m tools.fine_grid_cost calibrate --task-dir <new-task-dir> --workers 4 6 8
# Optional sole predeclared retake, only after diagnosing the first window:
uv run python -m tools.fine_grid_cost calibrate --task-dir <new-task-dir> \
  --workers 4 6 8 --label retake
uv run python -m tools.fine_grid_cost replay --task-dir <new-task-dir>
uv run python -m tools.fine_grid_cost verify --task-dir <new-task-dir>
```

Full `verify` needs raw data. Exit **0** means evidence and cost admission pass;
**2** denotes missing reference coverage; **3** denotes verified evidence but
failed cost admission; invalid/missing files or mismatches fail nonzero. The
present experiment returns **3**, explicitly `target_supported=false`.

## Owner handoff — unresolved, not demo re-entry

The implementation phase must remain **BLOCKED**, not DONE. The owner can
commission a separately bounded performance direction with a new explicit
acceptance boundary, or revisit the demo's cost/grid/generation contract; neither
is authorized by this report. The demo's own ≤2-worker / 16-worker-hour limit
is unchanged. Testing 4/6/8 workers here does not amend that authorization.
There is no resume sidecar, demo-state update, full scan, merge, push or release.
