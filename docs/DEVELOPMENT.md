# Development and verification

[AGENTS.md](../AGENTS.md) is the project entry point. This guide describes the
branch workflow and how to verify a change. Environment/dependency management
is maintained in [RUNNING.md](../RUNNING.md); the current physical-check baseline
is in [STATUS.md](STATUS.md).

## Start a change

Use a topic branch based on the current local `main` for code, documentation and
configuration changes. Follow the existing prefixes (`task/`, `chore/`, `fix/`,
`perf/`, `explore/`), choosing a name that describes the work:

```bash
git switch main
git switch -c chore/documentation-layout
```

Manage the work in local `scratchpad/` according to `scratchpad/common.md` and the
applicable skills. For simultaneous independent tasks, use separate worktrees
and their own environments, sharing scratchpad through the established workflow.
Before running package code/tests in a worktree, confirm the actual import path:

```bash
uv run python -c 'import skycolor; print(skycolor.__file__)'
```

## Verify the change

| Change | Verification |
|---|---|
| Documentation and document-path text | Check local links/heading anchors, stale paths and `git diff --check`; numerical reruns are unnecessary |
| Code and interfaces | Run relevant tests; expand coverage for changed behavior and follow the task's review requirements |
| Physics or numerical methods | Full-tier `pytest` run (below) plus affected `validation/check*.py`; independently explain any physical-result change |
| Dependencies | Use RUNNING's lock-file workflow and rerun regression/physical checks |

The full physical suite runs from the repository root:

```bash
uv run python -m validation.run_all
```

It writes `validation/results.json` and figures. Compare against the committed
baseline and distinguish existing physical failures from regressions. Passing
software tests and passing physical checks are different claims. Updating golden
files or physical results requires independent attribution, not just better fit.

### Test tiers

The suite is two-tier (task `suite-tier`, 2026-11-11 user adjudication): the
default entry is the **fast tier** (dev loop), the **full tier** is the
acceptance gate.  Tiers are declared in code with the `slow` marker — never
derived from a measured threshold at run time — and reordering tiers never
touches budgets, seeds, tolerances or assertions (zero detection-power
reduction).

```bash
uv run pytest -q                                    # fast tier (default entry)
SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup   # full tier
SKYCOLOR_FULL_SUITE=1 uv run pytest -n 0 -q tests/test_validation_harness.py  # serial full diagnostics
```

**Two-dimension placement rule** — a test belongs in the fast tier only when
BOTH hold:

1. **Species** (`kind:`): its guard object is logic / contract / boundary /
   small-scale numeric anchor.  MC statistical convergence, large-grid /
   full-dome integration, and golden-fixture full verification are full-tier
   exclusive regardless of cost.
2. **Cost** (`cost:`): its fast-tier K=3 median under the target form
   (12-worker xdist) is ≤2 s.  The `tools/make_scheduling_weights.py`
   regen audit gate enforces this mechanically: an unmarked fast-tier test
   above the budget fails the regen.  Lawful outcomes are marking `slow` or
   opening an optimization task — there is no "leave unmarked" outcome.
   Transient boundary (until the registered backlog task lands): when the
   gate output carries the xdist bin-head signature — ~12 positional
   7-17 s medians including intrinsically sub-0.1 s tests, reproducing
   across samples — the adjudicated outcome is the third one documented in
   validation/FAST_SUITE.md "The ratchet fired live": keep the
   authoritative table, record the flagged set as evidence, mark nothing.
   **Transition freeze:** because any fresh fast listing deterministically
   trips the bin-head gate while the transient exists, marker flips in
   BOTH directions on existing tests (including successful batch-protocol
   reclassification candidates) do not land during the transition —
   candidates measuring ≤2 s are recorded as *reclassification-ready*
   and released in one batch once the transient-aware gate task lands;
   the authoritative table stays valid throughout (no marker change → no
   mirror lag).

**New tests are tiered by their author at write time** (species dimension from
the rule above; cost dimension from an honest estimate, marker reason prefixed
`cost:` with the estimate basis).  The cost dimension is mechanically
re-checked by the next weights regen.

**Optimization-task semantics:** a test over the budget whose species would
allow the fast tier is marked `slow` *and* an optimization task is opened
(the gate stays green; the full tier carries the test meanwhile).  After the
optimization lands, re-measure its median via the authority list; ≤2 s, the
marker is removed by hand through the reclassification rule below.

**Reclassification (slow → fast) is a manual operation**, reviewed in
code-review: evidence is a target-form K=3 median ≤2 s, then delete the
marker, then re-measure the listings required by the marker change and
regen.  Readings from other calibers (durations tables, full-suite
listings) are not reclassification evidence.  **Transitional measurement
protocol** (until the xdist startup-transient mitigation lands — registered
in scratchpad/backlog.md): the default `-n 12` form concentrates a ~12-17 s
startup transient on each scheduling bin's first tests (evidence:
validation/FAST_SUITE.md "suite-tier"), so a small `<nodeids>` selection
would read its every member as a bin head.  Measure candidates in a batch
instead — `SKYCOLOR_FULL_SUITE=1 uv run pytest -q -m slow --durations=0`
(three samples): the genuinely heavy slow tests head the bins and absorb
the transient, so candidates read near-intrinsic; their K=3 medians are
the evidence.  A candidate that still reads >2 s may be attributed with a
serial reference (`uv run pytest -n 0 <nodeids> --durations=0` — an
attribution aid, never the pass/fail evidence itself); unresolved
candidates keep their marker until the transient task lands.  During the
transition freeze, even successful candidates do NOT have their markers
deleted (their release would need a fresh fast regen, which the bin-head
gate deterministically trips) — they are recorded reclassification-ready
and released in one batch after the transient task lands.

**Marker reasons are machine-readable** — `kind:mc` / `kind:golden` /
`kind:grid` / `kind:render` for species-exclusive tests, `cost: <median> s
(<source>)` for cost-only placements; greppable by species for audits.

**Filter yield semantics:** the default fast-tier filter stands down for an
explicit selection — `-m`, `-k`, or a positional argument containing `::`
(nodeid) — so `pytest -m slow` and single-test debug runs work as typed.
Bare paths/directories are not test selections: running a whole file keeps
the fast filter (its slow members leave via the deselected count, visible in
serial and collect-only output; the xdist run summary does not aggregate
worker-side deselected counts).  Running an all-slow module by bare path
(e.g. `pytest tests/test_mc.py`) therefore deselects everything and exits 5
with no deselected count on the xdist summary — that is the filter working
as pinned, not a mechanism failure; use `-m slow`, the env switch, or
explicit nodeids.  `--lf`/`--ff` cache selections are likewise NOT yield
triggers today: re-running a previously failed slow test via bare `--lf`
(serial or xdist) deselects it and exits 5; the working paths are
`SKYCOLOR_FULL_SUITE=1` or an explicit selection (`-m`/nodeid).  `SKYCOLOR_FULL_SUITE=1`
overrides to the full population.  Priority: explicit
`-m`/`-k`/`::` nodeid > env switch > default fast tier.

**Marker-change cost:** the table's `slow` mirror must equal the live
markers (harness equality test).  After adding/removing markers, the only
path back to a green full suite is re-measuring the listings the change
requires + regen.  Three scenarios: a fast-tier population change
(new/removed fast tests) requires a fresh full-suite listing; a
slow-tier-only population change (new `slow`-marked tests, fast population
untouched) likewise requires only a fresh full-suite listing + regen; in
both cases the standing fast-tier authority samples are reused (the
verified minimal closure) and tests added since the authority measurement
stay in the accepted inter-regen blind spot until the transient task's
regen.  A pure marker FLIP with unchanged population would re-measure the
three fast listings + regen — but during the transition freeze flips do
not land at all (see the transient boundary above: fresh fast listings
deterministically trip the bin-head gate), so scenario three is unusable
until the transient task releases the freeze.

### Test-suite execution profile

The default `uv run pytest -q` runs the fast tier on 12 xdist workers
scheduled by measured-cost bins, and `validation.run_all` runs the six
checks on one 8-worker pool (run_all does not go through pytest and is
unaffected by the tier split).  Measured numbers, the tier design, the
frozen regression budgets and the current acceptance evidence live in
[validation/FAST_SUITE.md](../validation/FAST_SUITE.md) — read it before
changing test structure, photon budgets or validation dispatch.  Diagnostics:

```bash
uv run pytest -n 0 -q ...      # serial equivalence run (no scheduling)
uv run pytest -q --durations=0 # per-test wall times under the real schedule
# refresh tests/data/scheduling_weights.json after material cost shifts
# (one FULL listing + three fast-tier listings, all exhaustive):
SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup --durations=0 --durations-min=0 > /tmp/full.txt
uv run pytest -q --durations=0 > /tmp/fast1.txt   # + fast2.txt, fast3.txt
uv run python tools/make_scheduling_weights.py /tmp/full.txt /tmp/fast1.txt /tmp/fast2.txt /tmp/fast3.txt
```

**Maintenance rule (two-tier version, superseding the single-suite rule from
the pre-tier suite):** any task that adds or removes tests from the suite
must, before closing, regenerate `tests/data/scheduling_weights.json` per
the flow above (a population change requires re-measuring the full-suite
listing) and record the table, tier counts and `fast_predicted_wall_s` in
[validation/FAST_SUITE.md](../validation/FAST_SUITE.md).  The formal
wall-clock pair for acceptance is taken on the **full tier** entry
(`SKYCOLOR_FULL_SUITE=1 …`, full population ×2, zero skips); quiet-window
certification remains a separate backlog item.  The 360 s guard line and
its escalation clause live in FAST_SUITE.md.

## Commit and merge locally

Commit the task's relevant files on its topic branch after verification. Before
merging, review the branch changes from the merge base and bring the branch up
to the current `main`. If main has advanced, reverify the affected changes after
rebasing:

```bash
# On the topic branch (in its worktree, when applicable)
git diff main...HEAD
git rebase main

# In the main checkout
git switch main
git merge --no-ff chore/documentation-layout
```

The branch name above is an example. Use explicit `--no-ff` even when a
fast-forward is possible, preserving the project's task boundary in history.
The repository's configured Git hooks enforce the clean merge policy.

Confirm that the merge tree matches the verified topic-branch tree and that the
main working tree is clean. Record the verification and merge in the task's
local closeout. Remove a temporary worktree with `git worktree remove` and delete
a fully merged topic branch with `git branch -d` when it is no longer needed.
Push to a remote when that action has been requested.
