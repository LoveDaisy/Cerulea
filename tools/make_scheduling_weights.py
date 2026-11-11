"""Rebuild ``tests/data/scheduling_weights.json`` from measured durations logs.

Two-tier suite (task-suite-tier): the tool consumes four listings and
regenerates the whole table in one pass.

* one FULL-suite listing (``SKYCOLOR_FULL_SUITE=1`` with the constant
  12-worker form) feeds the scheduling weights (``modules``/``nodes``) —
  the packing must describe the full population it schedules;
* three FAST-tier listings (the default entry point, winning form) are the
  upgrade authority: per-test K=3 medians drive the audit gate — any unmarked
  test whose fast-tier median exceeds ``FAST_BUDGET_S`` fails the run — and
  the informational ``fast_predicted_wall_s`` (sum of unmarked medians;
  recorded in validation/FAST_SUITE.md, consumed by no assertion);
* the ``slow`` mirror is derived from a fresh ``pytest -m slow`` collection
  subprocess, so the table's mirror can never lag the live markers at regen
  time (the harness equality test then forbids lagging afterwards).

    SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup --durations=0 > /tmp/full.txt
    uv run pytest -q --durations=0 > /tmp/fast1.txt   # + fast2.txt, fast3.txt
    uv run python tools/make_scheduling_weights.py /tmp/full.txt /tmp/fast1.txt /tmp/fast2.txt /tmp/fast3.txt

Parsing follows pytest's ``--durations`` listing: one line per
setup/call/teardown phase, whose seconds are summed per nodeid.  Under
``--dist=loadgroup`` the reported nodeid carries xdist's ``@binN``
scheduling suffix; it is stripped here so the regenerated table is keyed
exactly like the next run's collection (see ``_raw_nodeid``).  Phases below
5 ms are hidden by pytest (use ``-vv`` to include them); their total weight
is negligible.

Tolerance semantics: a test missing from some fast listings contributes no
reading to those samples (the <5 ms listing cutoff makes presence drift at
the boundary); only structural mismatches — whole modules absent from a
listing, disagreeing passed counts — are rejected.  The full-listing reuse
gate is the sharp counterpart: a test with a real fast-tier reading that is
missing from the full listing means the population changed and the full
suite must be re-measured (otherwise the test would silently fall back to
the 5.0 s default weight).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "tests" / "data" / "scheduling_weights.json"

_LINE = re.compile(r"^\s*([0-9.]+)s\s+(?:call|setup|teardown)\s+(\S+)")
_HEADER = re.compile(r"=+ slowest(?: (\d+))? durations =+")
_SCHED_SUFFIX = re.compile(r"@bin\d+$")
_PASSED = re.compile(r"(\d+) passed")
_DESELECTED = re.compile(r"(\d+) deselected")

#: Fast-tier per-test budget [s] (K=3 median, target-form measurement).
#: The gate fails the regen when an unmarked test exceeds it; the lawful
#: outcomes are marking ``slow`` or opening an optimization task — never a
#: silent pass-through.
FAST_BUDGET_S = 2.0

#: Durations listings hide phases below ~5 ms, so presence near the cutoff
#: drifts between samples.  Readings at or below this value count as the
#: boundary band: tolerated when missing from the full listing (the 5.0 s
#: default weight is the accepted fallback), while anything heavier missing
#: from the full listing means the population changed.
BOUNDARY_BAND_S = 0.01


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
    "default weight. `slow` mirrors the live pytest markers at regeneration "
    "time (harness equality test forbids lagging); `fast_predicted_wall_s` "
    "sums unmarked fast-tier K=3 medians (informational). Regenerate with "
    "tools/make_scheduling_weights.py from one FULL listing plus three "
    "fast-tier listings, all with --durations=0."
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


def parse_summary(text):
    """Return ``(passed, deselected)`` from a pytest run's summary line."""
    passed = _PASSED.search(text)
    deselected = _DESELECTED.search(text)
    if passed is None:
        raise SystemExit("no 'N passed' summary line found in the listing")
    return int(passed.group(1)), (int(deselected.group(1)) if deselected else 0)


def validate_list_identity(label, text):
    """Reject listings that do not match their slot in the CLI.

    A fast-tier listing must show a nonzero ``deselected`` count (the slow
    filter engaged); the full-suite listing must show none.  Identity is
    judged by content, not filename, so argument-order mistakes fail loudly
    here instead of quietly poisoning the medians or the weights.
    """
    passed, deselected = parse_summary(text)
    is_full = label.startswith("full")
    if is_full and deselected:
        raise SystemExit(
            f"{label}: shows {deselected} deselected — not a full-suite "
            f"listing.  Re-measure with SKYCOLOR_FULL_SUITE=1 and no -m/-k.")
    if not is_full and not deselected:
        raise SystemExit(
            f"{label}: shows no deselected summary — not a fast-tier "
            f"listing.  Re-measure with the default entry point (no "
            f"-m/-k/nodeid arguments).")
    return passed, deselected


def check_population_consistency(labels, passed_counts, nodeid_lists):
    """Structural consistency across the four listings.

    Module-level sets must agree between the three fast listings (they run
    the same population), the full listing must cover them all, and the
    passed counts of the fast listings must be equal — a truncated or
    partial run must not silently enter the medians.  Single missing tests
    are *not* structural: the <5 ms listing cutoff makes presence drift at
    the boundary (see ``reuse_gate``).
    """
    full_label, full_ids = labels[0], nodeid_lists[0]
    fast_labels, fast_ids = labels[1:], nodeid_lists[1:]
    full_modules = {n.split("::")[0] for n in full_ids}
    fast_modules = [ {n.split("::")[0] for n in ids} for ids in fast_ids ]
    for label, modules in zip(fast_labels, fast_modules):
        missing = fast_modules[0] - modules
        extra = modules - fast_modules[0]
        if missing or extra:
            raise SystemExit(
                f"{label}: module population diverges from {fast_labels[0]} "
                f"(missing {sorted(missing)}, extra {sorted(extra)}).  "
                f"The fast listings must cover the same population.")
    uncovered = fast_modules[0] - full_modules
    if uncovered:
        raise SystemExit(
            f"{full_label}: modules absent from the full listing: "
            f"{sorted(uncovered)}.  Re-measure the full suite.")
    if len(set(passed_counts[1:])) != 1:
        raise SystemExit(
            f"fast listings disagree on passed counts {passed_counts[1:]} — "
            f"a partial or failing run is unusable for medians.")


def fast_medians(fast_totals):
    """``[3 listings' totals]`` -> {nodeid: median of available readings}.

    A test missing from a listing contributes no sample for it (boundary
    drift); with a single reading available that reading is the median.
    """
    ids = set().union(*fast_totals)
    return {n: statistics.median([t[n] for t in fast_totals if n in t])
            for n in ids}


def upgrade_gate_offenders(medians, slow, budget_s=FAST_BUDGET_S):
    """Unmarked tests whose fast-tier median exceeds the budget.

    Full-suite readings never enter here by construction: only fast-tier
    medians are passed in, so contention-inflated full-listing costs cannot
    fail a genuinely fast test (caliber isolation).
    """
    return {n: m for n, m in sorted(medians.items())
            if m > budget_s and n not in slow}


def reuse_gate(full_totals, medians, band_s=BOUNDARY_BAND_S):
    """Fail when the full listing misses tests with real fast-tier readings.

    A test measured in the fast tier but absent from the full listing would
    silently fall back to the 5.0 s default scheduling weight; that is only
    acceptable inside the <5 ms boundary band (or for tests never observed
    at all, which have no better weight available either way).
    """
    missing = sorted(
        n for n, m in medians.items() if m > band_s and n not in full_totals)
    if missing:
        raise SystemExit(
            "full listing does not cover tests with fast-tier readings: "
            f"{missing}.  The population changed since the full measurement — "
            "re-run the full-suite listing (SKYCOLOR_FULL_SUITE=1, 12 workers, "
            "--durations=0) and pass the fresh file.")


def predicted_wall_s(medians, slow):
    """Sum of unmarked fast-tier medians (informational, no assertion).

    Unmarked tests are constructed to all have fast-tier readings; tests
    without any reading are sub-listing-cutoff and contribute zero by
    convention.  Slow-marked tests never enter, marked or not.
    """
    return sum(m for n, m in medians.items() if n not in slow)


def collect_slow_nodeids():
    """Live slow-tier population via a fresh ``pytest -m slow`` collection.

    ``-m slow`` makes the default filter yield (explicit selection), so the
    subprocess reports exactly the slow-marked tests of the whole tree,
    regardless of the ambient ``SKYCOLOR_FULL_SUITE`` value.  The mirror is
    regenerated here — never inherited from the previous table — so it
    cannot lag the markers the moment the table is written.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--co", "-q", "-m", "slow",
         "-n", "0", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(
            "slow-tier collection subprocess failed:\n"
            + (proc.stdout + proc.stderr)[-2000:])
    return sorted(
        line.strip() for line in proc.stdout.splitlines()
        if "::" in line and not line.startswith("="))


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


def build_payload(full_totals, medians, slow):
    """Pure synthesis of the new table from validated inputs."""
    modules, nodes = build(full_totals)
    return {
        "comment": TABLE_COMMENT,
        "measured": (
            f"{datetime.date.today().isoformat()}: scheduling weights from the "
            f"full-suite listing; slow mirror from a fresh -m slow collection; "
            f"fast medians from three fast-tier listings"
        ),
        "modules": modules,
        "nodes": nodes,
        "slow": sorted(slow),
        "fast_predicted_wall_s": round(predicted_wall_s(medians, slow), 2),
    }


def main():
    args = sys.argv[1:]
    if len(args) != 4:
        raise SystemExit(
            "usage: make_scheduling_weights.py FULL_LOG FAST1 FAST2 FAST3\n"
            "FULL_LOG: SKYCOLOR_FULL_SUITE=1 ... -n 12 --dist=loadgroup --durations=0\n"
            "FAST{1,2,3}: default fast-tier entry, --durations=0, three samples")
    paths = [pathlib.Path(a) for a in args]
    texts = [p.read_text() for p in paths]
    for label, path in zip(("full (arg 1)", "fast1 (arg 2)", "fast2 (arg 3)",
                            "fast3 (arg 4)"), paths):
        if not path.exists():
            raise SystemExit(f"{label}: file not found: {path}")
    labels = ["full (arg 1)", "fast1 (arg 2)", "fast2 (arg 3)", "fast3 (arg 4)"]
    passed = []
    for label, text in zip(labels, texts):
        p, _d = validate_list_identity(label, text)
        passed.append(p)
    totals = [parse_durations(text) for text in texts]
    if not all(totals):
        raise SystemExit("no durations lines found in an input")
    check_population_consistency(
        labels, passed, [set(t) for t in totals])
    medians = fast_medians(totals[1:])
    slow = set(collect_slow_nodeids())
    reuse_gate(totals[0], medians)
    offenders = upgrade_gate_offenders(medians, slow)
    if offenders:
        lines = "\n".join(
            f"  {n}  median {m:.2f} s" for n, m in offenders.items())
        raise SystemExit(
            f"fast-tier audit gate: {len(offenders)} unmarked test(s) above "
            f"the {FAST_BUDGET_S} s budget:\n{lines}\n"
            "Lawful outcomes: mark them slow (kind:/cost: reason), or open an "
            "optimization task and mark them slow meanwhile.  There is no "
            "'leave unmarked' outcome.")
    payload = build_payload(totals[0], medians, slow)
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    total = sum(payload["modules"].values()) + sum(payload["nodes"].values())
    print(f"wrote {WEIGHTS_PATH}: {len(payload['modules'])} module groups, "
          f"{len(payload['nodes'])} individual nodes, {total:.1f} s total, "
          f"slow mirror {len(payload['slow'])} tests, "
          f"fast_predicted_wall_s {payload['fast_predicted_wall_s']}")


if __name__ == "__main__":
    main()
