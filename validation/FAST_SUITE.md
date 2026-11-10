# Fast validation suite — measured baseline, design and budgets

Status: implemented 2026-11-08 (task `fast-validation-suite`, branch
`task/fast-validation-suite` off main `6bdb1f8`).  Measured on the reference
development machine (macOS arm64, 12 cores, 32 GiB, Python 3.12.11,
NumPy 2.5.4; `environment-pytest.json`).  Revised later the same day after
code review: the scheduling hook initially never engaged in real xdist
workers (xdist nulls `numprocesses` there) and crashed default
`--collect-only`; fixed, covered by real-subprocess integration tests, and
re-accepted — the second formal pair below is the authoritative one.

## Goal and acceptance criteria

User target (2026-11-07): full `uv run pytest -q` and
`uv run python -m validation.run_all` must each finish in **≤ 300 s wall
clock on the same machine, twice in a row**, without losing regression
detection capability and without any numerical change in `skycolor/`
production behaviour (golden bit-identical; same seeds, grids, budgets,
tolerances).

**Final measured acceptance** (2026-11-08, sequential runs):

| entry point | runs (wall s) | verdict |
|---|---|---|
| `uv run pytest -q` | **formal pair, schedule engaged (post-fix): 257.41 ✓ / 256.22 ✓ outer wall** (`formal2-pytest-run{1,2}.log`; pytest-internal 257.26 / 256.01) | **达标**: both ≤ 300 s strictly, exit=0, 436 passed |
| `uv run python -m validation.run_all` | 240.02 probe; **241.13 / 239.98** formal pair (`formal-runall-run{1,2}.log`) | **达标**: both ≤ 300 s, exit=1, 2 PASS/4 FAIL/0 SKIP, `results.json` bitwise identical to the committed baseline across all runs |

The first pytest pair (293.70 ✓ / 352.15 ✗, `formal-pytest-run{1,2}.log`) is
kept as historical evidence of the **ungrouped** path: it predates the
scheduling fix, and code review established that those runs executed plain
per-test load distribution — the LPT bins never reached the controller
scheduler (see below).  Its numbers do not describe the current entry point.

pytest variance findings on the pre-fix (ungrouped) path — historical,
documented because they motivated the owner adjudication:

* Fast and slow runs burned the **same total CPU** (~2330 s user+sys) — the
  +59 s of the slow regime was idle stall, not compute.
* Ruled out and fixed en route: a scheduler bug (module weight counted once
  per collected test), serial-measured weights (per-test cost inflates
  1.3–2.5× under 12-way contention; the weights table is regenerated from a
  full `--durations=0` listing of a parallel run), xdist worker startup
  (measured 1.2 s with 12 workers) and per-test channel overhead (≤10 ms).
* Ruled out as the slow regime's cause: external CPU load (slow runs also
  happen at load average 1.2), swap growth during the run (constant in a
  sampled 294 s run), worker count (n=8 → 368 s, n=16 → 324 s, both worse).
* Independently: one heavy MC test peaks at 1.89 GB RSS; the machine hosts
  concurrent agent sessions.

After the scheduling fix the engaged path measured **257.41 / 256.22 s
outer wall** (startup, collection, scheduling and report writing included;
pytest-internal 257.26 / 256.01) —
strictly under the 300 s line twice in a row, so the windowed acceptance the
owner granted (see the adjudication note at the bottom of this file) is no
longer needed for the current pair; the adjudication's regression clause
(any suite stably exceeding 6 min is escalated) remains the standing guard.

## Later tree: architecture consolidation (2026-11-08)

This does **not** replace or retroactively extend the 436-test formal pair above.
On source tree `6c8d9ed` (same production/default-test tree as `7d7ce15`),
`task-approximation-story` ran the default entries sequentially in the designated
macOS worktree with its own uv environment:

| Entry | Actual result / outer wall | Interpretation |
|---|---|---|
| `uv run pytest --collect-only -q` | 526 collected; no experiments/tools | Default production boundary retained |
| `uv run pytest -q` | **526 passed**, exit 0, **364.54 s** (pytest internal 364.32 s) | Functional regression green; **not performance acceptance**: exceeds 300 s target and 360 s guard |
| `uv run python -m validation.run_all` | exit 1, **242.10 s**, 2P/4F/0 skip (2/3/5/6) | Existing physical failures; all validation output files including results.json byte-identical to before |

The preceding research-boundary closeout measured 526 tests / 357.39 s on
`7d7ce15`; the current single run crosses 360 s and is explicitly escalated to
the owner under the standing guard. No production/default-test/settings or
budget change was made by the representation task, so it does not claim a new
physical or functional regression, does not re-bin tests, and does not reduce
moments/photons/tolerances to mask the cost. Whether to fund a suite-performance
follow-up belongs to the owner, not this example/documentation task. These are
one-run current-tree observations, **not** a new two-run ≤300 s certification.

The isolated research-free export of `6c8d9ed` subsequently ran the same default
suite: **526 passed, zero skip, exit 0, 355.72 s outer / 355.54 s internal**.
It has its own locked environment and removes all experiments/research data;
only `tools/make_scheduling_weights.py` remains as a byte-identical maintenance
input to `test_validation_harness` (all other tools absent). An initial export
that removed this maintenance consumer too produced 502 passes + worker
collection errors, not a valid green run. Correcting the export restored all
526 tests; no test or source implementation was copied/changed/skipped. This
supports research independence but does not erase the full-tree guard breach
or certify the ≤300 s target.

## What was slow, and why

1. **pytest default collection was broken.**  Without `testpaths`, pytest
   collected `scratchpad/explore-project-value-audit/snapshots/` (65 test
   modules re-used as refactor snapshots) → duplicate basenames → 65
   collection errors, exit code 2, **no tests ran**
   (`pytest-default-collect-failure.log`).  The suite therefore had no
   working default entry point at all.
2. **Serial runtime.**  The real suite is 413 tests (not the historical
   "355"; task `cssp-observation-inputs` added 58): **1570.23 s** serial,
   all passing (`pytest-serial-runs.json`, `pytest-serial.log`).
   Hot spots: `test_twilight.py` 429 s, `test_mc.py` 159 s,
   `test_solar_source.py` 145 s, `test_multiple.py` 134 s
   (`pytest-files.json`).
3. **`validation.run_all` serial runtime.**  Per check with shared-atmosphere
   probes: check1 0.96 s, check2 5332 s (dominated by two unused diffuse
   solves, see below), check3 148 s, check4 25 s, check5 51 s, check6 432 s
   (`reuse-checks-part1/2.json`).  Two solar-table builds alone cost 20.6 s
   (`reuse-builds.json`).
4. **check2's ASTM case computed data nobody consumed.**  The AM1.5 case
   (`model_case(48.19, wl=400..700/1nm)`) solved a 73×144 diffuse sky, but
   `run()` only consumes its DNI/`e_dni`.  The h=30/60 DHI/GHI outputs are
   real and kept.
5. **check6's four spherical-engine points (−6/−9/−12/−15°) ran inside the
   serial check**, although they are independent Monte-Carlo jobs.

## Design (no budget cuts anywhere)

### pytest: fixed collection boundary + measured LPT scheduling

* `pyproject.toml` sets `testpaths = ["tests"]` — the default `pytest -q`
  now collects exactly the 413 controlled tests (`pytest-nodeids.json`).
* `pytest-xdist` (dev dependency, uv-locked) runs the default entry point
  with `-n 12 --dist=loadgroup`.
* `tests/conftest.py` reproduces the measured LPT binning: one scheduling
  group per test module; modules whose parallel-measured cost exceeds
  ~100 s (and `test_twilight.py` always) are split into per-nodeid groups,
  because one indivisible multi-hundred-second module would set the
  wall-clock floor of the whole run (`tests/data/scheduling_weights.json`,
  schema `{modules, nodes}`).  Tests absent from the table get a nonzero
  default weight and are still scheduled (collection is the single source
  of truth — no persisted whitelist of nodeids).
* The hook attaches the bin marks **tryfirst** inside each xdist worker.
  xdist's own worker-side hook then encodes each bin into the reported
  collection as a `nodeid@binN` suffix — the *only* channel the
  controller's `LoadGroupScheduling` sees, since distributed runs block
  controller-side collection entirely.  The pre-fix hook read
  `config.option.numprocesses`, which xdist nulls in workers, so nothing
  was ever marked and `--dist=loadgroup` silently degraded to per-test
  load distribution; it now reads `config.workerinput["workercount"]` in
  workers and the option only on controller/serial paths (`-n 0` marks
  nothing).
* Real-subprocess integration tests (`tests/test_validation_harness.py`)
  copy the real hook files into a synthetic project and assert on what
  actually executed: worker-side marking, one module = one group on one
  worker, hot nodes split across bins, every test executed exactly once,
  serial runs carry no groups, and default `--collect-only` exits 0 (it
  used to crash with a `KeyError` on the pre-fix hook).
* `-n 0` runs serially for equivalence diagnostics; it is **not** an
  alternative acceptance path.

Probe evidence: subprocess shards with the same grouping measured 254.56 s
(8 processes) / 225.92 s (12 processes) with all 413 tests passing
(`pytest-shards-*-summary.json`).

### validation: shared atmospheres, direct-only ASTM case, one dispatch pool

* `validation/common.atmosphere()` keeps its signature and gains a
  process-local, bounded (16-entry) reuse keyed by
  `(kind, wavelength bytes, aod, ozone_du)`.  Returned objects are
  **read-only shared**: checks must not mutate them (`clear_atmosphere_cache()`
  is the test hook).  Equivalence evidence: independently built vs reused
  atmospheres give exactly equal luminance
  (`check5-atmosphere-reuse-equivalence.json`), and all reuse-probe metrics
  are identical to the serial baseline (`reuse-checks-part1/2.json`).
* `validation/check2_illuminance.py`: the direct-beam solve is extracted into
  `direct_case()`; the ASTM branch calls it directly, so the unused diffuse
  sky is genuinely never computed (no stub returning fake arrays).  h=30/60
  keep the full 73×144 grid and DHI/GHI outputs.
* `validation/check6_second_dataset.py`: the spherical-engine point is one
  authoritative helper `engine_point(h)` (same clear preset, `WL_COARSE`,
  zenith, `quality="low"`, `seed=0`).  `model_luminance` and the −6°
  supplemental print consume it.  `run(engine_points=None)` computes engine
  points on the fly (standalone); `run_all` injects this round's results —
  no second implementation, no cross-run precomputation.
* `validation/run_all.py`: one `ProcessPoolExecutor(max_workers=8)` (spawn)
  created in the entry point.  The four engine points are dispatched first,
  then check2/3/4/5/1; the parent runs check6's deterministic/figure work and
  consumes the engine futures.  Results are assembled in the original CHECKS
  order; only the parent writes `validation/results.json`; worker exceptions
  propagate (never disguised as a physical FAIL or a "successful" partial
  run).  `checkN.run()` and standalone commands are unchanged entry points.
* check1–5 write mutually distinct figure/CSV paths; check6's
  `validation/figures/13_twilight_datasets.png` is written only by the parent.

Candidate wall clock (probe adapter, same budgets/seeds/grids): **239.17 s**
with every metric's name/model/reference/tolerance/dev/dev_kind/unit
identical to the committed baseline and failure set still
{check2, check3, check5, check6} (`validation-candidate-summary.json`,
independently re-read in `probe-comparison.json`).

## What did NOT change (regression-detection budgets)

No wavelength/angle grid, photon budget, MC seed, tolerance, assertion or
scoring rule changed.  In particular:

* check5: single-scattering twilight points on `WL` and the 120/200-point
  figure curves on `WL_COARSE` — unchanged.
* check6: `quality="low"`, `seed=0`, split at −8°, factor-1.5 tolerance plus
  the engine's own 1σ relaxation — unchanged; the four engine depths are the
  same calls, executed once each.
* check2: DNI tolerance 10% (G173) / 15% (Bird), DHI 25% (CIE 110), spectral
  RMS 10% — unchanged; only an unconsumed computation was removed.
* The FAILing checks keep failing on the same metrics — the failure set is a
  regression guard: any deviation beyond the committed baseline must be
  explained, not silenced.

## Verification protocol

```bash
uv run pytest --collect-only -q          # 413 + harness additions, no scratchpad
uv run pytest -n 0 -q tests/test_validation_harness.py   # structural tests
/usr/bin/time -p uv run pytest -q        # ×2, sequential, ≤300 s each
/usr/bin/time -p uv run python -m validation.run_all     # ×2, exit=1, 2 PASS/4 FAIL
uv run python -m validation.check2_illuminance   # standalone vs pool comparison
uv run python -m validation.check6_second_dataset # standalone vs pool comparison
git diff -- skycolor tests/data/golden*  # production tree must be empty
```

Tracked evidence in `validation/data/fast_validation_suite/`: the two
formal acceptance pairs and their memory/load samples
(`formal*-pytest-run*.log`, `formal-runall-run*.log`,
`memory-sample-during-pytest-formal2-run*.log`), the executed scheduling
plan (`scheduling-assignment-final.json`: 436 nodeids → 12 bins,
predicted 218.3–218.6 s per bin), and the baseline/probe data this design
rests on.  Session-local scratchpad copies under
`scratchpad/task-fast-validation-suite/probes/formal/` are not tracked.

Owner adjudication (2026-11-08, `issue.md` §裁决记录), applying to the
pre-fix pair: windowed evidence was accepted on the strength that both
regimes burned the same total CPU (~2330 s) and the slow regime was read
as kernel-side wait — an attribution made on the **ungrouped** path and
not carried over as a settled claim about the current schedule.  The
standing clauses: windowed acceptance may be revoked by the user, and any
suite stably exceeding 6 min counts as a regression and is escalated.
The post-fix pair (257.41 / 256.22 s outer wall) passes strictly, so the
windowed
interpretation is not exercised by the current acceptance evidence.

## 526-test recert (2026-11-10, task `suite-recert`)

This does **not** replace or retroactively extend any earlier pair. On source
tree `0dd68a6` (branch `task/suite-recert`, same production/default-test tree
as `286486e`), `task-suite-recert` regenerated the LPT weights table from a
fresh 12-worker `--durations=0` listing of the current 526-test population
(`recert-durations-526.{txt,json}`: 319 listed nodeids — pytest hides <5 ms
phases — totalling 4490.5 s; the table now holds 7 module groups + 213
individually scheduled nodes, regenerated via
`tools/make_scheduling_weights.py`, no `@bin` keys). Offline preview
(`recert-scheduling-assignment.json`): 12 bins at 380.46–380.47 s predicted
load, max/min ratio 1.00003; `tests/test_validation_harness.py` structural
tests 24 passed; `--collect-only` still 526, exit 0.

**Formal pairs — four sequential runs, all 526 passed / exit 0, all over the
360 s guard: certification not achieved, escalated under the standing
clause.**

| pair | runs (outer wall s) | window load (1-min start → end) | logs |
|---|---|---|---|
| recert | **425.20 ✗ / 446.86 ✗** | 5.91 → 47.23 (15-min avg 21 → 41) | `recert-pytest-run{1,2}.log` |
| recert2 (pre-registered single retake) | **376.54 ✗ / 397.04 ✗** | 5.05 → 78.81 (sampler peak 129.3) | `recert2-pytest-run{1,2}.log`, `memory-sample-during-pytest-recert2.log` |

Measurement-window attribution: the machine hosts concurrent agent sessions
and every measurement window overlapped a sustained load wave (sampled
1-minute averages peaked at 129, ≈10.8× the 12-core capacity). pytest's own
CPU accounting (recert2 run1: user+sys 3392.5 s ⇒ ~283 s across 12 cores,
plus ~40 s startup/collection/report overhead ≈ 320–330 s) suggests an
idle-machine run might land near or below the line, but no uncontended
certification window could be obtained on this machine. Per the
pre-registered protocol the retake was taken once; both pairs are retained
without selection and no windowed interpretation is applied. The disposition
(accept / split the default suite / adjust the target) belongs to the owner;
the durations distribution and binning preview above are delivered as the
evidence package for that decision.

`uv run python -m validation.run_all` formal pair on the same tree,
sequential: **285.14 / 353.50 s**, exit=1 both, **2 PASS / 4 FAIL / 0 SKIP**,
failure set exactly {check2, check3, check5, check6}
(`recert-runall-run{1,2}.log`); tracked validation outputs byte-identical to
the committed baseline (`git status --porcelain validation/` lists only the
new evidence logs).

Freeze tags created (local only; pushing is an owner action):
`freeze/research-boundary-afaae17` → afaae17,
`freeze/cssp-inputs-5ac0c74` → 5ac0c74,
`freeze/deep-twilight-dda8673` → dda8673,
`freeze/task7-exec-2756b2f` → 2756b2f.
