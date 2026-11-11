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
tests 24 passed; `--collect-only` still 526, exit 0.  Both the listing and
the table are a **single** parallel sample: entry-level contention noise
swings by orders of magnitude between samples (e.g. `test_underlit_base_turns_red`
0.30 s in the 2026-11-08 table vs 18.14 s here; `test_output_shape_matches_solver`
15.60 → 0.59 s), so the 1.00003 ratio is paper balance — real scheduling
balance is dominated by this noise, and a median-of-K listing is the natural
upgrade for the next recertification.  Top modules: `test_twilight.py` 882.4 s
and `test_simulation_entry.py` 822.6 s, ~38% of the listed total — the
headline targets for any future split decision.

**Formal pairs — four sequential runs, all 526 passed / exit 0, all over the
360 s guard: certification not achieved, escalated under the standing
clause.**

| pair | runs (outer wall s) | window load (1-min start → end) | logs |
|---|---|---|---|
| recert | **425.20 ✗ / 446.86 ✗** | 5.91 → 47.23 (15-min avg 21 → 41) | `recert-pytest-run{1,2}.log` |
| recert2 (pre-registered single retake) | **376.54 ✗ / 397.04 ✗** | 5.05 → 78.81 (sampler peak 151.4) | `recert2-pytest-run{1,2}.log`, `memory-sample-during-pytest-recert2.log` |

Measurement-window attribution: the machine hosts concurrent agent sessions
and every measurement window overlapped a sustained load wave (sampled
1-minute averages peaked at 151.4, ≈12.6× the 12-core capacity). pytest's own
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
## Later tree: cold-optics batch kernels (2026-11-10)

Task-cold-optics added `tests/test_mie_batch.py` (anchor/binary-equivalence
for the batched Mie kernels; 11 tests after the review-fix rounds) and
regenerated the scheduling weights from a parallel `--durations=0` run on the
final tree (`scratchpad/task-cold-optics/durations_parallel5.log`, 4300.0 s
total over 8 module groups + 200 individual nodes).  Test-count chain,
mechanically exported (`uv run pytest --collect-only -q`): the 526-test tree
above is the 2026-10-30 measurement; this task's base tree (286486e)
collects 527; final tree collects **538** (= 527 + the 11 batch-anchor
tests).  Formal pairs on the green tree:

| entry | result | verdict |
|---|---|---|
| `uv run pytest --collect-only -q` (final tree) | 538 collected | count chain above; the earlier "533 collected / +6 vs 526" row was arithmetically wrong and is superseded here |
| `uv run pytest -q` (pre-fix-tree pair, 2026-11-10 21:25/21:31) | 533 passed, 368.76 / 370.86 s (`formal_run{1,2}.log`) | **superseded evidence**: measured before the review-fix commits (d4e9c03/e1d9c59 never saw a full run until the 206ba7b pair) |
| `uv run pytest -q` (206ba7b pair) | 537 passed, 399.84 / 391.90 s (`formal_run{3,4}.log`) | **superseded by the review-rework tree**: the round-2 Minor rework added one k=1 branch test (538) |
| `uv run pytest -q` (final-tree formal pair) | **538 passed, 395.81 / 382.86 s** (`formal_run{5,6}.log`) | green, **over the 360 s guard**: pre-task tree already 364.54 s (escalated); the added batch-anchor tests carry their own cost under 12-worker memory-bandwidth contention (largest single item: the production-single consumption test, 266 s parallel-measured). Standing escalation clause applies, owned together with the suite-recert task |
| `uv run python -m validation.run_all` (final tree) | 2 PASS / 4 FAIL / 0 SKIP, exit 1 (`run_all_final{,2}.log`) | same FAIL set as the committed baseline; `validation/results.json` bit-identical after both runs (zero numeric drift, D2 declared-tolerance criterion satisfied) |

Performance context for the same task: cold dusty `delta_m(32)` 39.4 -> 3.1 s
(`scratchpad/task-cold-optics/bench_{baseline,after}.log`).

## Two-tier suite (2026-11-11, task `suite-tier`)

The suite is two-tier from this tree on: the default `uv run pytest -q` is
the **fast dev-loop tier** (373 tests at close; 358 at first
finalization), the acceptance gate is the **full
tier** (`SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup`,
565 tests, zero skips expected).  Tiers are declared in code (`slow` markers
with `kind:`/`cost:` reasons, filter + yield rules single-sourced in
`tests/scheduling.py`, applied in `tests/conftest.py`); the placement rules
live in docs/DEVELOPMENT.md "Test tiers".  This section records the
measured evidence and the provenance of every number the task rewrote.

**Count chain** (mechanically exported; `SKYCOLOR_FULL_SUITE=1 uv run pytest --collect-only -q`):
issue baseline 542 + 23 tests added by this task − 1 removed (the
audit-gate unit test, deleted together with its gate by the option-2
adjudication) + 1 added (the marker-change-window pinning test, gen 4) =
**565** = **373 fast + 192 slow** (identity guarded by
`test_tier_accounting_full_equals_fast_union_slow`, no count literals;
the 23 additions were set-diff verified against a baseline-tree collection
at the r4 export, 23 added / 0 removed).

**Merge integration** (2026-11-11, owner): task-mc-restrict-segv landed on
main before this branch (+15 tests, one of them the ~45 s stress anchor).
Per the tier cost principle the anchor is `slow`-marked at module level
(the manual-marker protocol's first live use), and the authority literals
were re-derived on the merged tree: **580 = 387 fast + 193 slow**
(identity re-verified by `test_tier_accounting_full_equals_fast_union_slow`).
The task-close numbers above are historical records of this task's own
tree and are not rewritten.
The 23: 8 in Step 1 (7 mini tier-filter behavior tests + 1 policy-helper
unit test), 12 weights-tool unit tests, 2 tree guards (accounting + mirror),
1 review-round addition (mini scenario (g), xdist worker-side `-m` yield).
The mirror equality (`test_tier_slow_mirror_matches_live_markers`) is live
since the first regen below; it fired once for real when scenario (g) moved
the slow population 201 → 202, and the mandated marker-change cycle
restored it; the gen-4 correction below then moved the slow population
202 → 192 and the final regen re-derived the mirror from a fresh
collection.  Note that the owner-side rebase onto main will fold in
further test files (mc-restrict-segv), requiring one more export at merge
time.

**Slow list provenance:** mechanical draft = the cold-optics 12-worker
durations listing (`scratchpad/task-cold-optics/durations_parallel5.log`,
>2 s = 144 readings, supersedes the issue's weights-table-derived "about
130"; the table cannot see 14 heavy tests inside light module groups),
expanded to 201 collected items by (a) kind-dimension module marking
(`test_mc.py` 6, `test_golden_equivalence.py` 7), (b) parametrized
functions marked at function level, (c) 20 upgrade-adjudication marks
(below; 10 of them corrected back to the fast tier by the batch protocol
after the option-2 adjudication).  Kind census (decorator-level): `kind:grid` 73, `kind:render` 24,
`kind:mc` 10, `kind:golden` 5, `cost:` 45.  The kind additions review found
**zero** kind-exclusive tests among the unlisted fast candidates: the only
MC-named candidate (`test_mc_solar_beam_respects_the_opaque_bottom`) is a
vacuum ray/intersection contract check.

**Form measurement** (3 interleaved samples per form, `--durations=0`,
window load 2.2-6.8, logs `scratchpad/task-suite-tier/step5/`):
**12-worker xdist wins, median 50.51 s (50.49-51.03) vs serial 97.00 s
(95.66-99.73)** — the pre-registered rule (median wall, tie → serial)
decides without mid-tier samples; `pyproject.toml` addopts stay
`-n 12 --dist=loadgroup`.  The issue's serial-favouring estimate (fast-tier
compute ~11 s) measured as ~87 s serial CPU for the marked-down population;
the >2 s cutoff retains 144 genuinely costly tests' worth of mid-weight
items the >10 s estimate excluded.

**Monitoring-surface adjudication (risk 1):** the xdist run summary does
**not** aggregate worker-side deselected counts (serial listings show "N
deselected"; xdist summaries omit the count entirely — verified on all
samples).  Per the pre-registered fallback, full-tier runs are judged by
the formal要件 "deselected = 0 (absent) and full-population passed", the
collect-only probes carry the tier counts, and no fast-tier run claims
completeness from its summary line.

**Upgrade adjudication and the startup transient.** The winning-form
samples' K=3 per-test medians flagged 20 unmarked tests at 6.2-14.6 s.
A deselect probe proved these readings are a **positional xdist startup
transient**, not intrinsic cost: deselecting the 20 moved the 12-17 s
readings wholesale onto the new bin-head tests — including a test whose
intrinsic cost is 0.02 s reading 16.90 s.  Every xdist run pays this
transient once per scheduling bin (~12-17 s absorbed by each bin's first
tests); it is already inside the 50.5 s wall and excluded from
`fast_predicted_wall_s` by construction (true while the transient-carrying
bin heads stayed slow-marked; after the gen-4 correction the standing fast
listings' then-current bin heads contribute their transient again — see
the gen-4 regen paragraph).  The 20 were nevertheless marked
`slow` (`cost:` reasons, commit `ab6812b`) per the pre-registered rule —
the safe, reversible direction — with the documented consequence that the
transient follows bin position: a post-review re-measurement flagged the
new bin heads (below), and the ratchet had no mechanical exit.  The
follow-up was adjudicated by the user (2026-11-11 "option 2"): the
automated fast→slow gate is removed from `tools/make_scheduling_weights.py`
(markers are human-owned declarations; cost verification is carried by the
weights-regen rule), the transition freeze is lifted, and the 20 marks
were re-measured via the pinned transitional batch protocol and corrected
by hand (outcome below).  The once-registered transient-aware-gate backlog
entry is superseded — the batch protocol is the standing measurement
de-pollution method.

**Correction of the 20 marks (2026-11-11, option-2 adjudication).**  The 20
candidates were re-measured via the pinned transitional batch protocol:
three `-m slow` full-tier samples over the 202-test slow tier
(`scratchpad/task-suite-tier/step10/slow_batch_{1,2,3}.log`, 202 passed ×3,
~357 s wall each; window load 1.8 rising to 22-23 across samples — recorded;
the split is bimodal with no candidate near the 2 s boundary, so the load
anomaly cannot flip any decision).  K=3 medians: **10 candidates read
0.05-0.97 s** — the original 6-14 s fast-listing readings were pure
bin-head transients (e.g. `test_azimuth_symmetry` 14.36 s → 0.06 s) — and
their markers were deleted, restoring them to the fast tier; **10
candidates read 11.03-20.53 s, stable across samples** — genuinely heavy
(grid/render/SOS-class dome integration) — and keep their `slow` markers
with reasons refreshed to cite the batch medians.  No candidate was
unresolved, so no serial attribution run was needed.  Identity after the
correction: 564 = 372 fast + 192 slow (565 = 373 + 192 once the gen-4
window-red pinning test below is counted); marker-line census at the corrected
head: `cost:` 36, `kind:grid` 73, `kind:render` 24, `kind:mc` 9,
`kind:golden` 5 (module-level `pytestmark` on `test_mc.py` /
`test_golden_equivalence.py` expands to the collected counts).

**Weights regen (authoritative, 2026-11-11):** full listing
`SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup
--durations=0 --durations-min=0` (465 s window, load rising to 18.5 —
recorded; scheduling weights target the 12-worker form, contention
inflation is their accepted semantics) + the three fast samples →
`tests/data/scheduling_weights.json`: 12 module groups + 271 individual
nodes (5109.2 s on that listing), `slow` mirror 201, **`fast_predicted_wall_s`
26.4**
(sum of unmarked fast-tier medians; informational only, no assertion
consumes it; residuals: excludes worker startup/imbalance, contains
contention-inflated per-test readings, sub-listing-cutoff tests contribute
zero).  `--durations-min=0` is mandatory for the full listing: the first
regen attempt with the bare `--durations=0` failed the reuse gate on 8
tests whose intrinsic cost sits at the 5 ms listing cutoff (listed in
fast-tier samples, hidden in the full listing) — the full listing must be
exhaustive for the reuse gate to be exact (suite-recert learning applied).

**Regen input-health checks (historical).**  Audit gate green after the
adjudication marks (zero unmarked >2 s); reuse gate green; slow mirror
equality green (live since this regen); accounting identity green.  A
second authoritative regen
(2026-11-11 12:15, full listing re-measured for the mirror delta after
review-round additions) refreshed the table: 12 module groups + 271 nodes,
4651.5 s total, **mirror 202**, `fast_predicted_wall_s` 26.4 — audit and
reuse gates green on its inputs (the standing fast-tier authority samples;
the fast population had not changed, only the slow tier grew by one).
**The three fast-tier tool-unit tests added in the same review round fall in
the accepted inter-regen blind spot (plan 待补全项 ①): unmeasured by the
authority samples, invisible to the gate and the predicted sum.  The
marker-policy audit gate referenced above was removed by the option-2
adjudication (commit `87ca9c3`): marker placement is human-owned, and the
identity / structural-consistency / reuse / mirror-equality checks remain
as input-health validation, never marker policy.

**Regen after the correction (gen 4, 2026-11-11).**  One flip-window full
listing + three fresh fast-tier listings →
`tests/data/scheduling_weights.json`: 12 module groups + 271 individual
nodes, 4149.3 s total, **slow mirror 192** (re-derived from a fresh
collection — current), **`fast_predicted_wall_s` 249.7**.  Input-health
checks green (identity / structural consistency / reuse / mirror equality).
The full listing was measured in the marker-change window: its sole failure
is the mirror-equality guard (table mirror 202 vs live 192 before this
regen) — the designed transition state, accepted by the full slot's named
exception (commit `fc8f002`); any other failure still rejects.  The
predicted sum now structurally includes the listing's own bin-head
transient readings (~12-17 s each on the then-current scheduling heads):
the previous 26.4 excluded them only because the transient-carrying heads
were slow-marked by construction — with the correction those heads are
honest fast-tier members again.  The metric stays informational (no
assertion consumes it); the same tree's actual fast wall measured
39.1-39.9 s (final verification below).  Window load 10-19 (full listing)
/ 19-21 (fast samples) — recorded.

**The ratchet fired live (2026-11-11 12:16-12:18).**  A post-review
re-measurement of the fast tier (three samples over the 361-test population,
`scratchpad/task-suite-tier/step8/fast2_*.log`) hit the audit gate on
exactly 12 tests at medians 7.16-12.71 s — the NEW scheduling-bin heads
predicted by the deselect probe, including a 0.02 s-intrinsic test reading
12.71 s.  Marking them slow would start the removal spiral (~12 tests per
regen cycle, no convergence); the table was kept and the set recorded as
transient evidence.  This live firing is the recorded proof that the gate
family measured bin position, not cost — the evidence base for the
option-2 adjudication that removed the gate.  In the no-gate world the
same readings reappear as inflated weights on the then-current bin heads
inside fresh fast listings: expected artifacts of the transient, absorbed
by the packing, never a marker instruction.

**Caller inventory (default-entry semantics):** `validation/` and `tools/`
contain no pytest invocation (`ruler_audit_recheck.py` mentions pytest only
in a docstring); no CI configuration exists in the repo; the only in-repo
pytest subprocess caller is the harness mini-runner itself (explicit args,
explicit env); `validation.run_all` dispatches the six checks via
`ProcessPoolExecutor` without pytest.  The tier split therefore has no
silent victim among programmatic callers.

**Formal pair protocol (unchanged semantics, explicit full tier):** the
acceptance pair is taken on the full-tier entry —
`SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup`, full
population passed ×2, zero skips, real xdist default parameters — recorded
here when next taken; this task did not take a certification pair, and the
quiet-window certification remains the separate backlog item.  The 360 s
guard line and escalation clause continue to apply to full-tier pairs.

**Issue-prose rewrites carried by this section** (all sourced): "exactly
542 items" → identity + N_new chain (this task's own tests falsify the
literal); "about 130 items" → 144 draft readings; "≈412 magnitude" → 358
fast; "12-worker parallel measured ≤2 s" → target-form (xdist) caliber per
the parallel-evidence learning; "budget assertion computed from the weights
table" → authority-list medians (the ≤90 s aggregate assertion was dropped
by user adjudication).  User adjudications of 2026-11-11 embodied here:
drop the automated downgrade mechanism (reclassification is manual), drop
the 90 s aggregate budget assertion (per-test ≤2 s enforced by the regen
audit gate, aggregate stays informational), pin the explicit-nodeid yield
semantics.  The follow-up **"option 2" adjudication** (2026-11-11, after
the implement code-review budget fused at four rounds) is embodied in this
section's current state: the automated fast→slow audit gate is removed
(markers are human-owned declarations; cost verification = the
weights-regen rule), the transition freeze is lifted, the 20 upgrade marks
are corrected via the transitional batch protocol, and the
transient-aware-gate backlog entry is superseded.

**Final verification (this task, 2026-11-11):** fast tier 358 passed ×2
(outer wall 62.50 / 56.70 s, window load 8-24), harness serial diagnostics
41 passed (30.67 s); post-review-round verification on the refreshed table:
fast tier 361 passed ×3 (pre-r3/r4-gate-tree) (outer wall 38.74 / 38.52 / 37.94 s — the
tier-updated packing distributes startup cost better than the pre-tier
table), full tier **563 passed, zero skips**, exit 0 (393.70 s); r3/r4
gate-hardening commits added tests after those runs, so the
**delivered-HEAD verification** re-ran the chain: harness serial 47
passed, fast tier 363 passed (38.48 s wall), full tier **565 passed,
zero skips**, exit 0 (390.74 s, window load 2.6→10).  Attributions, per run: the 62.50 / 56.70 s pair sat
above the 50.5 s form-measurement median under a loaded window (1-min load
8→24) — recorded attribution per the measurement-window protocol,
informational only (no aggregate assertion exists by adjudication); the
38.74 / 38.52 / 37.94 s triple ran on the refreshed (tier-updated) packing;
the 393.70 s full-tier run compares against the 465 s authoritative
full-suite measurement, whose extra time over 393 s is the exhaustive
`--durations-min=0` listing overhead plus its own climbing window load.

**Final verification (gen 4 correction, 2026-11-11, HEAD `fc8f002`):**
harness serial diagnostics 47 passed (24.06 s); fast tier 373 passed ×2
(50.22 / 51.50 s wall, window load ~12 — the AGENTS.md wall band spans
this task's loaded and quiet windows); full tier **565 passed, zero
skips, exit 0** (358.66 s, window load 16-20).  `git diff main...HEAD --
skycolor/` empty; the `validation/` diff contains only this file;
`validation/results.json` untouched; run_all not re-run (the caller
inventory above shows it does not go through pytest).

## Quiet-window certification (2026-11-11, post two-tier)

First certification of the full tier on the final two-tier population
(580 = 387 fast + 193 slow), taken in a genuinely quiet window (no
concurrent agent sessions; the two full-tier runs differ by 3.3 s).
Sequential pair, `SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12
--dist=loadgroup`:

| entry | result | wall |
|---|---|---|
| full tier run 1 | **580 passed, zero skips, exit 0** | **399.78 s** |
| full tier run 2 | **580 passed, zero skips, exit 0** | **403.12 s** |
| `validation.run_all` run 1 | 3 PASS / 3 FAIL / 0 SKIP, failure set {check3, check5, check6} | 237.14 s |
| `validation.run_all` run 2 | 3 PASS / 3 FAIL / 0 SKIP, failure set {check3, check5, check6} | 235.56 s |

(logs at `validation/data/fast_validation_suite/quiet-cert-*.log`; the
run_all failure set is the post-ruler-fix baseline.  These runs also close
the task-mc-restrict-segv acceptance-4 exception on the tree: the routed
`contiguous_transpose` code regenerates `validation/results.json` with the
check6 zenith −9° derived-tolerance field at its confirmed 1-ulp value
(model fields bitwise unchanged; owner-confirmed 2026-11-11), so the
committed guardrail file now matches what the current code produces.)  The 6-min standing line is **breached in a quiet window, with
tight variance** — this is no longer measurement pollution.  Attribution:
the population grew 542 -> 580 (+38 tests, mc-restrict-segv and suite-tier)
at unchanged per-test cost (~0.69 s/test); per-test degradation is ruled
out.  The disposition of the line (re-scale for the acceptance-gate tier /
attach the wall line to the fast dev-loop tier / keep 360 s as aspiration)
is escalated to the owner; the fast dev-loop tier measures 37.5 s on the
same quiet machine, an order of magnitude inside any historical line.
