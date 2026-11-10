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
| Physics or numerical methods | Full `uv run pytest -q` plus affected `validation/check*.py`; independently explain any physical-result change |
| Dependencies | Use RUNNING's lock-file workflow and rerun regression/physical checks |

The full physical suite runs from the repository root:

```bash
uv run python -m validation.run_all
```

It writes `validation/results.json` and figures. Compare against the committed
baseline and distinguish existing physical failures from regressions. Passing
software tests and passing physical checks are different claims. Updating golden
files or physical results requires independent attribution, not just better fit.

### Test-suite execution profile

The default `uv run pytest -q` runs on 12 xdist workers scheduled by
measured-cost bins, and `validation.run_all` runs the six checks on one
8-worker pool. Measured numbers, the scheduling design, the frozen
regression budgets and the current acceptance evidence live in
[validation/FAST_SUITE.md](../validation/FAST_SUITE.md) — read it before
changing test structure, photon budgets or validation dispatch. Diagnostics:

```bash
uv run pytest -n 0 -q ...      # serial equivalence run (no scheduling)
uv run pytest -q --durations=0 # per-test wall times under the real schedule
# refresh tests/data/scheduling_weights.json after material cost shifts:
uv run python tools/make_scheduling_weights.py <durations-log>
```

**Maintenance rule:** any task that adds or removes tests from the default
collection must, before closing, regenerate `tests/data/scheduling_weights.json`
from a fresh parallel `--durations=0` listing (commands above), re-take the
sequential formal wall-clock pair for the default suite, and record the table
and pair evidence in [validation/FAST_SUITE.md](../validation/FAST_SUITE.md) —
the 360 s guard line and its escalation clause live there.

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
