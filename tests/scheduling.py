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
"""
from __future__ import annotations

import json
import os
import pathlib

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
