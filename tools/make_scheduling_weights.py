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

    SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup --durations=0 --durations-min=0 > /tmp/full.txt
    uv run pytest -q --durations=0 > /tmp/fast1.txt   # + fast2.txt, fast3.txt
    uv run python tools/make_scheduling_weights.py /tmp/full.txt /tmp/fast1.txt /tmp/fast2.txt /tmp/fast3.txt

``--durations-min=0`` on the full listing is mandatory: the bare ``=0``
output hides <5 ms phases, which both starves the reuse gate (tests with
real fast readings missing from the full listing) and starves the table
(the 5.0 s default silently covering cheap tests).

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
#: xdist loadgroup evidence on a durations line: a real listed nodeid closes
#: with the scheduling suffix.  Required in the full slot — the full-suite
#: listing is measured under the constant 12-worker loadgroup form, so its
#: absence means a serial (or otherwise mis-formed) listing was passed.
_SUFFIXED_DURATION_LINE = re.compile(
    r"^\s*[0-9.]+s\s+(?:call|setup|teardown)\s+\S+@bin\d+\s*$", re.MULTILINE)

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
    "tools/make_scheduling_weights.py from one FULL listing (SKYCOLOR_"
    "FULL_SUITE=1, 12 workers, --durations=0 --durations-min=0) plus three "
    "fast-tier listings."
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
    """Return ``(passed, deselected)`` from the run's final summary line.

    Anchored to the last non-empty line on purpose: the durations listing
    precedes the summary, so a whole-text search could match "N passed"
    inside a test *name* on a durations line and misparse the counts.
    """
    tail = text.rstrip().splitlines()[-1] if text.strip() else ""
    passed = _PASSED.search(tail)
    deselected = _DESELECTED.search(tail)
    if passed is None:
        raise SystemExit(
            "no 'N passed' summary line found at the end of the listing")
    return int(passed.group(1)), (int(deselected.group(1)) if deselected else 0)


def validate_list_identity(label, text):
    """Reject listings whose content contradicts their slot in the CLI.

    A full-suite listing never deselects (``SKYCOLOR_FULL_SUITE=1``, no
    ``-m``/``-k``) and must carry xdist loadgroup evidence (``@binN``
    scheduling suffixes on durations lines) — the full listing is measured
    under the constant 12-worker form, so a serial listing passed here
    would silently produce uncontended scheduling weights.  A fast-tier
    listing either reports the deselected count (serial form) or — under
    xdist, which does not aggregate worker-side deselected counts into the
    summary — reports nothing, so absence is accepted and mis-slotted
    listings are caught structurally instead (population consistency and
    the reuse gate fire on swaps and on all-fast inputs).  Identity is
    judged by content, not filename.
    """
    passed, deselected = parse_summary(text)
    is_full = label.startswith("full")
    if is_full:
        if deselected:
            raise SystemExit(
                f"{label}: shows {deselected} deselected — not a full-suite "
                f"listing.  Re-measure with SKYCOLOR_FULL_SUITE=1 and no -m/-k.")
        if not _SUFFIXED_DURATION_LINE.search(text):
            raise SystemExit(
                f"{label}: no xdist loadgroup scheduling suffixes on any "
                f"durations line — not a 12-worker loadgroup listing.  "
                f"Re-measure with SKYCOLOR_FULL_SUITE=1 uv run pytest -q "
                f"-n 12 --dist=loadgroup --durations=0 --durations-min=0.")
    elif deselected == 0 and "deselected" in text:
        raise SystemExit(
            f"{label}: reports a zero deselected count — not a fast-tier "
            f"listing.  Re-measure with the default entry point (no "
            f"-m/-k/nodeid arguments).")
    return passed, deselected


def check_population_consistency(labels, passed_counts, totals_list, slow=()):
    """Structural consistency across the four listings.

    Module-level comparison is substance-gated: a module is compared only
    when its best reading across the fast samples exceeds the listing-
    cutoff band, so modules whose entire content flickers around the ~5 ms
    report cutoff (presence drifting between samples under contention) are
    exempt from the structural check, while a genuinely missing module
    (operator error, truncated run — a heavy module has multi-second
    readings) still fails loudly.  The full listing must cover the whole
    slow tier: a fast listing passed in the full slot — or four fast
    listings, whose xdist summaries are shape-identical — fails here
    instead of silently regenerating a table without the slow weights.
    The passed counts of the fast listings must be equal with the full
    count above them.  Single missing tests are *not* structural: the
    <5 ms cutoff makes per-test presence drift at the boundary (see
    ``reuse_gate``).
    """
    full_label, full_totals = labels[0], totals_list[0]
    fast_labels, fast_totals = labels[1:], totals_list[1:]
    missing_slow = sorted(set(slow) - set(full_totals))
    if slow and missing_slow:
        raise SystemExit(
            f"{full_label}: does not cover the slow tier ({len(missing_slow)} "
            f"missing, e.g. {missing_slow[:3]}).  A full-suite listing must run "
            f"the complete population — re-measure with SKYCOLOR_FULL_SUITE=1 "
            f"-n 12 --dist=loadgroup --durations=0 --durations-min=0.")
    # Best-evidence substance per module across the fast samples.
    ref: dict[str, float] = {}
    for totals in fast_totals:
        for nodeid, seconds in totals.items():
            module = nodeid.split("::")[0]
            ref[module] = max(ref.get(module, 0.0), seconds)
    comparable = {m for m, s in ref.items() if s > BOUNDARY_BAND_S}
    present = [{n.split("::")[0] for n in t} for t in fast_totals]
    for label, mods in zip(fast_labels, present):
        missing = {m for m in comparable if m in present[0]} - mods
        extra = {m for m in comparable if m in mods} - set(present[0])
        if missing or extra:
            raise SystemExit(
                f"{label}: module population diverges from {fast_labels[0]} "
                f"(missing {sorted(missing)}, extra {sorted(extra)}).  "
                f"The fast listings must cover the same population.")
    uncovered = comparable - {n.split("::")[0] for n in full_totals}
    if uncovered:
        raise SystemExit(
            f"{full_label}: modules absent from the full listing: "
            f"{sorted(uncovered)}.  Re-measure the full suite.")
    if len(set(passed_counts[1:])) != 1:
        raise SystemExit(
            f"fast listings disagree on passed counts {passed_counts[1:]} — "
            f"a partial or failing run is unusable for medians.")
    if passed_counts[0] < max(passed_counts[1:]):
        raise SystemExit(
            f"{full_label}: passed count {passed_counts[0]} is below the fast "
            f"listings' {max(passed_counts[1:])} — the full listing must run "
            f"a superset of the fast population.")


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
            "re-run the full-suite listing (SKYCOLOR_FULL_SUITE=1, "
            "-n 12 --dist=loadgroup, --durations=0 --durations-min=0) and "
            "pass the fresh file.")


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
            "FULL_LOG: SKYCOLOR_FULL_SUITE=1 ... -n 12 --dist=loadgroup "
            "--durations=0 --durations-min=0\n"
            "FAST{1,2,3}: default fast-tier entry, --durations=0, three samples")
    paths = [pathlib.Path(a) for a in args]
    labels = ["full (arg 1)", "fast1 (arg 2)", "fast2 (arg 3)", "fast3 (arg 4)"]
    # existence before read: a missing file must produce the labeled
    # SystemExit, not an unlabeled FileNotFoundError traceback
    for label, path in zip(labels, paths):
        if not path.exists():
            raise SystemExit(f"{label}: file not found: {path}")
    texts = [p.read_text() for p in paths]
    passed = []
    for label, text in zip(labels, texts):
        p, _d = validate_list_identity(label, text)
        passed.append(p)
    totals = [parse_durations(text) for text in texts]
    if not all(totals):
        raise SystemExit("no durations lines found in an input")
    medians = fast_medians(totals[1:])
    slow = set(collect_slow_nodeids())
    check_population_consistency(
        labels, passed, totals, slow=slow)
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
