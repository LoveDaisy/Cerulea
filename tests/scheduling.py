"""Measured-cost scheduling groups for pytest-xdist (``--dist=loadgroup``).

The default ``pytest -q`` entry point runs the suite on 12 xdist workers.
To keep module fixtures together, tests of a small module form one
scheduling group; hot modules — those whose measured cost exceeds the
threshold applied by ``tools/make_scheduling_weights.py`` — are split into
per-nodeid groups, because one indivisible multi-hundred-second module
would set the wall-clock floor of the whole run.  Groups are packed
largest-first into the currently lightest of N bins (LPT bin packing, the
policy measured fastest in the planning probes), N being the xdist worker
count, so every worker receives a balanced, mutually exclusive shard.

Weights come from ``data/scheduling_weights.json`` (wall seconds measured
under the parallel configuration itself — per-test cost inflates ~1.3-2.5x
under 12-way contention, so serial measurements underbalance the bins).
Collection is the single source of truth: tests absent from the table get a
nonzero default weight and are still packed and executed exactly once — no
persisted whitelist of nodeids gates execution.

The module also owns the two-tier suite vocabulary (task-suite-tier): the
``FULL_SUITE_ENV`` switch and the explicit-selection yield rules consumed by
``tests/conftest.py``.  The tier decision must agree in the controller and in
every xdist worker; the environment variable is the one channel guaranteed to
cross the process boundary (xdist workers rebuild their config from forwarded
CLI args and do not receive custom options any other way).
"""
from __future__ import annotations

import json
import os
import pathlib

#: Environment switch selecting the full (acceptance) suite population.
#: Unset, empty, or ``"0"`` runs the fast dev-loop tier — tests marked
#: ``slow`` are deselected at collection; any other value keeps every
#: collected test.  Naming follows the ``SKYCOLOR_SCHEDULING_WEIGHTS``
#: precedent.
FULL_SUITE_ENV = "SKYCOLOR_FULL_SUITE"


def full_suite_selected() -> bool:
    """Whether this invocation asked for the full acceptance population."""
    return os.environ.get(FULL_SUITE_ENV, "") not in ("", "0")


def slow_filter_yields(config) -> bool:
    """Whether the default slow-tier filter must stand down for this call.

    An explicit selection expresses the intent to run the matched tests,
    slow or not — deselecting them would turn ``pytest -m slow`` or a single
    ``path::test`` debug run into a misleading "no tests ran" (exit 5).
    Bare paths and directories are *not* test selections: running a whole
    file is still a fast-tier request whose slow members leave via the
    reported ``N deselected`` summary line.
    """
    if config.option.markexpr or config.option.keyword:
        return True
    return any("::" in str(arg) for arg in getattr(config, "args", ()))


def fast_tier_active(config) -> bool:
    """True when this invocation must deselect ``slow``-marked tests.

    Priority: explicit ``-m`` / ``-k`` / ``::`` nodeid selection > full-suite
    environment switch > default fast tier.
    """
    return not slow_filter_yields(config) and not full_suite_selected()

#: Weights table produced by ``tools/make_scheduling_weights.py``.
WEIGHTS_PATH = pathlib.Path(__file__).parent / "data" / "scheduling_weights.json"

#: Environment override for the weights table.  Used by the subprocess
#: integration tests in ``test_validation_harness.py`` so a synthetic project
#: can exercise the real hook without touching the tracked table.
WEIGHTS_ENV = "SKYCOLOR_SCHEDULING_WEIGHTS"

#: Estimated cost [s] for tests missing from the measured table.  Any
#: nonzero value schedules them; it only perturbs the packing, never runs.
DEFAULT_WEIGHT_S = 5.0


def load_weights(path=None):
    """Return ``(modules, nodes)`` weight tables.

    ``modules`` maps a test file path (as in a nodeid, e.g.
    ``tests/test_mc.py``) to its measured wall seconds; ``nodes`` maps
    individually scheduled nodeids (hot modules and ``test_twilight.py``)
    to theirs.  ``path`` defaults to :data:`WEIGHTS_PATH`, or the
    :data:`WEIGHTS_ENV` environment variable when set.
    """
    if path is None:
        path = os.environ.get(WEIGHTS_ENV) or WEIGHTS_PATH
    data = json.loads(pathlib.Path(path).read_text())
    return dict(data["modules"]), dict(data["nodes"])


def group_weights(nodeids, modules, nodes):
    """Map each scheduling group key to its estimated cost [s].

    A group key is a module path, or a full nodeid for individually
    scheduled tests.  Unknown nodeids count as :data:`DEFAULT_WEIGHT_S` so
    new tests are always scheduled.  A module's weight is its measured
    *total* and enters once per module regardless of how many tests were
    collected in it.
    """
    groups: dict[str, float] = {}
    for nodeid in nodeids:
        module = nodeid.split("::")[0]
        if nodeid in nodes:
            groups[nodeid] = groups.get(nodeid, 0.0) + float(nodes[nodeid])
        else:
            # Module totals must not accumulate per collected test.
            groups[module] = float(modules.get(module, DEFAULT_WEIGHT_S))
    return groups


def pack_lpt(groups, n_bins):
    """Pack groups largest-first into the lightest bin; return ``key -> bin``.

    Ties break by key name so the packing is deterministic across workers
    and runs.
    """
    n_bins = max(int(n_bins), 1)
    assignment: dict[str, int] = {}
    loads = [0.0] * n_bins
    for key, weight in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])):
        k = min(range(n_bins), key=lambda i: loads[i])
        assignment[key] = k
        loads[k] += weight
    return assignment


def assign_bins(nodeids, n_bins, weights_path=None):
    """Map every nodeid to its scheduling bin index ``0..n_bins-1``.

    Total coverage is exact: every collected nodeid lands in exactly one
    bin, whether or not it appears in the weights table.  The result is
    keyed by full nodeid — consumers must not re-derive module keys, or
    the two key spaces drift apart (the conftest hook consumes this
    mapping verbatim for exactly that reason).
    """
    modules, nodes = load_weights(weights_path)
    bin_of_group = pack_lpt(group_weights(nodeids, modules, nodes), n_bins)
    out = {}
    for nodeid in nodeids:
        key = nodeid if nodeid in nodes else nodeid.split("::")[0]
        out[nodeid] = bin_of_group[key]
    return out
