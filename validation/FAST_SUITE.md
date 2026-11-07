# Fast validation suite — measured baseline, design and budgets

Status: implemented 2026-11-08 (task `fast-validation-suite`, branch
`task/fast-validation-suite` off main `6bdb1f8`).  Measured on the reference
development machine (macOS arm64, 12 cores, 32 GiB, Python 3.12.11,
NumPy 2.5.4; `environment-pytest.json`).

## Goal and acceptance criteria

User target (2026-11-07): full `uv run pytest -q` and
`uv run python -m validation.run_all` must each finish in **≤ 300 s wall
clock on the same machine, twice in a row**, without losing regression
detection capability and without any numerical change in `skycolor/`
production behaviour (golden bit-identical; same seeds, grids, budgets,
tolerances).

**Final measured acceptance (Step 4)**:

| entry point | run 1 | run 2 | exit | tests / failure set |
|---|---|---|---|---|
| `uv run pytest -q` | TBD | TBD | TBD | TBD |
| `uv run python -m validation.run_all` | TBD | TBD | TBD | TBD |

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
  group per test module, except `test_twilight.py` where each nodeid is its
  own group (its 15 tests span 20–90 s each).  Measured wall-second weights
  live in `tests/data/scheduling_weights.json`; tests absent from the table
  get a nonzero default weight and are still scheduled (collection is the
  single source of truth — no persisted whitelist of nodeids).
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
  `figures/13_twilight_datasets.png` is written only by the parent.

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

Run results of the two double acceptances are recorded at the top of this
file and in `scratchpad/task-fast-validation-suite/probes/` (session-local).
