"""Rebuild ``tests/data/scheduling_weights.json`` from a measured durations log.

The weights table holds measured wall seconds per scheduling group: light
modules as one entry, hot modules (measured total >=
``NODE_SPLIT_THRESHOLD_S``) and ``test_twilight.py`` as per-nodeid entries.
``tests/conftest.py`` uses it to balance xdist workers.  Refresh it when the
suite's cost profile has materially shifted (new heavy tests, moved
budgets) — ideally from a run under the parallel configuration itself, since
per-test cost inflates under 12-way contention:

    uv run pytest -q --durations=0 > /tmp/durations.txt
    uv run python tools/make_scheduling_weights.py /tmp/durations.txt

Parsing follows pytest's ``--durations`` listing: one line per
setup/call/teardown phase, whose seconds are summed per nodeid.  Under
``--dist=loadgroup`` the reported nodeid carries xdist's ``@binN``
scheduling suffix; it is stripped here so the regenerated table is keyed
exactly like the next run's collection (see ``_raw_nodeid``).  Phases
below 5 ms are hidden by pytest (use ``-vv`` to include them); their total
weight is negligible.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "tests" / "data" / "scheduling_weights.json"

_LINE = re.compile(r"^\s*([0-9.]+)s\s+(?:call|setup|teardown)\s+(\S+)")
_HEADER = re.compile(r"=+ slowest(?: (\d+))? durations =+")
_SCHED_SUFFIX = re.compile(r"@bin\d+$")


def _raw_nodeid(nodeid):
    """Strip xdist's loadgroup scheduling suffix (``nodeid@binN``).

    The duration report and the next run's collection use different
    identities for the same test: under ``--dist=loadgroup`` xdist appends
    the bin to the reported nodeid, while collection looks weights up by
    the raw nodeid.  Keying the table on suffixed ids would silently
    unschedule every hot node on the next run.  A parametrized nodeid
    always ends in ``]``, so a trailing ``@bin<digits>`` can only be the
    scheduling suffix; ``@`` characters inside parameter values and
    unsuffixed (serial) reports pass through untouched.
    """
    return _SCHED_SUFFIX.sub("", nodeid, count=1)

#: Modules whose measured total exceeds this many seconds are scheduled as
#: individual nodeids (``nodes`` in the weights table); anything lighter
#: stays one module group.  One indivisible hot module would set the wall
#: floor of the whole parallel run.  ``test_twilight.py`` is always split.
NODE_SPLIT_THRESHOLD_S = 100.0
ALWAYS_SPLIT = {"tests/test_twilight.py"}

#: Self-description written into the table's ``comment`` field.  Owned here
#: (not inherited from the previous table) so the described policy cannot
#: drift from the policy this tool actually implements.
TABLE_COMMENT = (
    "Measured wall seconds (sum of setup+call+teardown). One scheduling group "
    "per light module (listed in `modules`); hot modules (measured total >= "
    "100 s) and tests/test_twilight.py are split into per-nodeid groups "
    "(listed in `nodes`). Collected tests absent from this table attach to "
    "their module group; a module without any measured entry gets the 5.0 s "
    "default weight. Regenerate with tools/make_scheduling_weights.py from a "
    "FULL parallel --durations=0 listing."
)


def parse_durations(text):
    """``--durations=0`` listing -> {nodeid: total seconds}.

    Refuses a partial listing: when pytest's header advertises N durations
    but fewer lines follow, the table would silently lose most weights and
    unbalance the schedule.
    """
    totals = {}
    header_n = None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            if m.group(1) is not None:
                header_n = int(m.group(1))
            continue
        m = _LINE.match(line)
        if m:
            seconds, nodeid = float(m.group(1)), _raw_nodeid(m.group(2))
            totals[nodeid] = totals.get(nodeid, 0.0) + seconds
    if header_n is not None and len(totals) < header_n:
        raise SystemExit(
            f"partial durations listing: header announces {header_n} entries, "
            f"parsed {len(totals)}.  Re-run with --durations=0.")
    return totals


def build(totals):
    """Split the totals into module weights and individually scheduled nodes.

    Tests of hot modules (measured total >= ``NODE_SPLIT_THRESHOLD_S``) and
    of ``ALWAYS_SPLIT`` modules each become their own ``nodes`` entry; the
    remaining modules keep their summed weight as one group.
    """
    per_module: dict[str, dict[str, float]] = {}
    for nodeid, seconds in totals.items():
        module = nodeid.split("::")[0]
        per_module.setdefault(module, {})[nodeid] = seconds
    modules, nodes = {}, {}
    for module, tests in per_module.items():
        total = sum(tests.values())
        if module in ALWAYS_SPLIT or total >= NODE_SPLIT_THRESHOLD_S:
            nodes.update({n: round(s, 2) for n, s in tests.items()})
        else:
            modules[module] = round(total, 2)
    return modules, nodes


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    totals = parse_durations(pathlib.Path(sys.argv[1]).read_text())
    if not totals:
        raise SystemExit("no durations lines found in the input")
    modules, nodes = build(totals)
    payload = {
        "comment": TABLE_COMMENT,
        "measured": (
            f"{datetime.date.today().isoformat()} from {pathlib.Path(sys.argv[1]).name}: "
            f"per-test wall seconds under the parallel configuration"
        ),
        "modules": modules,
        "nodes": nodes,
    }
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    total = sum(modules.values()) + sum(nodes.values())
    print(f"wrote {WEIGHTS_PATH}: {len(modules)} module groups, "
          f"{len(nodes)} individual nodes, {total:.1f} s total")


if __name__ == "__main__":
    main()
