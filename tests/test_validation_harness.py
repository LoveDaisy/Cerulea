"""Structural tests for the fast validation harness.

Two families, both structural only — the numeric equivalence of the physical
results is verified by real runs against the committed baseline (see
``validation/FAST_SUITE.md``), not here:

* pytest scheduling: the LPT packing used by ``tests/conftest.py`` must be
  deterministic, cover every collected test exactly once, tolerate unknown
  tests, and work for any worker count — and the collection hook must
  actually attach the groups in real runs: in xdist workers, during
  controller-side ``--collect-only`` collection, and nowhere when serial.
* validation restructuring: atmosphere reuse is keyed and read-only,
  check2's ASTM path genuinely skips the diffuse solve while h=30/60 keep
  it, and check6's engine-point injection consumes exactly the dispatched
  Monte-Carlo depths.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import scheduling
from validation import check6_second_dataset as c6
from validation import common as C

TESTS_DIR = pathlib.Path(__file__).parent


def _load_weights_tool():
    """Import ``tools/make_scheduling_weights.py`` by path (no package)."""
    spec = importlib.util.spec_from_file_location(
        "make_scheduling_weights",
        TESTS_DIR.parent / "tools" / "make_scheduling_weights.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WEIGHTS_TOOL = _load_weights_tool()


# ---------------------------------------------------------------------------
# scheduling (Step 2)
# ---------------------------------------------------------------------------
def test_weights_table_references_existing_modules():
    modules, nodes = scheduling.load_weights()
    assert modules or nodes, "weights table must not be empty"
    for module in modules:
        assert (TESTS_DIR / pathlib.Path(module).name).exists(), module
    for nodeid in nodes:
        module = nodeid.split("::")[0]
        assert (TESTS_DIR / pathlib.Path(module).name).exists(), nodeid


def test_group_weights_unknown_nodeid_gets_default():
    modules, nodes = scheduling.load_weights()
    groups = scheduling.group_weights(
        ["tests/some_new_module.py::test_new"], modules, nodes)
    assert groups == {"tests/some_new_module.py": scheduling.DEFAULT_WEIGHT_S}
    assert scheduling.DEFAULT_WEIGHT_S > 0


def test_group_weights_individually_scheduled_nodes_are_their_own_groups():
    modules, nodes = scheduling.load_weights()
    some = sorted(nodes)[:2]
    groups = scheduling.group_weights(some, modules, nodes)
    assert set(groups) == set(some)  # one group per individually scheduled nodeid


def test_assign_bins_covers_every_nodeid_exactly_once():
    nodeids = [
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two[param1]",
        "tests/test_a.py::test_two[param2]",
        "tests/test_b.py::test_x",
        "tests/test_twilight.py::test_deep",
        "tests/test_twilight.py::test_unmeasured_new",
    ]
    for n in (1, 2, 3, 8, 12):
        bins = scheduling.assign_bins(nodeids, n)
        assert set(bins) == set(nodeids), f"coverage broken at n={n}"
        assert all(0 <= b < n for b in bins.values()), f"bin out of range at n={n}"


def test_assign_bins_is_deterministic():
    nodeids = [f"tests/test_c.py::test_{i}" for i in range(6)]
    nodeids += ["tests/test_twilight.py::test_deep"]
    for n in (2, 12):
        assert scheduling.assign_bins(nodeids, n) == scheduling.assign_bins(nodeids, n)


def test_group_weights_counts_module_weight_once():
    # A module's measured total must enter the packing once, not once per
    # collected test in it (a regression here silently unbalances the bins).
    modules, nodes = scheduling.load_weights()
    module = sorted(modules)[0]  # any module-scheduled module
    nodeids = [f"{module}::test_{i}" for i in range(6)]
    groups = scheduling.group_weights(nodeids, modules, nodes)
    assert groups == {module: modules[module]}


def test_assign_bins_separates_the_two_heaviest_groups():
    # The two costliest groups must not share a bin when one is available:
    # a regression that merges hot groups silently sets the wall floor.
    modules, nodes = scheduling.load_weights()
    heaviest = sorted(
        [(s, m) for m, s in modules.items()] + [(s, n) for n, s in nodes.items()],
        key=lambda kv: -kv[0],
    )[:2]
    assert len(heaviest) == 2
    # Keep node-group keys intact (they only carry weight as full nodeids);
    # module-group keys get a synthetic test appended.
    nodeids = [path if path in nodes else f"{path}::test_a" for _s, path in heaviest]
    bins = scheduling.assign_bins(nodeids, 2)
    assert bins[nodeids[0]] != bins[nodeids[1]]


# ---------------------------------------------------------------------------
# scheduling hook integration: the hook must engage in real runs
# ---------------------------------------------------------------------------
# A pure-function test of ``assign_bins`` cannot prove the conftest hook
# consumes it: past defects lived exactly in that interface (a module key
# lookup that crashed collect-only, and a worker-detection path that silenced
# the whole schedule in real xdist workers).  These tests build a tiny
# project that *copies the real conftest.py and scheduling.py*, schedule it
# against a synthetic weights table (``scheduling.WEIGHTS_ENV``), and assert
# on what actually executed in pytest subprocesses.
_MINI_WEIGHTS = {
    "modules": {"test_mini_a.py": 5.0},
    "nodes": {
        "test_mini_hot.py::test_heavy_one": 100.0,
        "test_mini_hot.py::test_heavy_two": 90.0,
    },
}
# test_mini_b.py is deliberately absent from the table: unknown modules and
# parametrized nodeids must still be scheduled (DEFAULT_WEIGHT_S path).

# Slow-marker reason used by the tier-filter mini fixtures.  ``kind:`` is the
# machine-readable species prefix (see docs/DEVELOPMENT.md "Test tiers");
# these fixtures are marked for *behavior*, not measured cost.
_MINI_SLOW_REASON = "kind: mini-project tier-filter fixture (behavior, not cost)"

_RECORDER = '''\
def _record(request):
    marker = request.node.get_closest_marker("xdist_group")
    worker = getattr(request.config, "workerinput", {}).get("workerid", "serial")
    group = marker.args[0] if marker else "-"
    # xdist rewrites item._nodeid with a "@binN" scheduling suffix under
    # loadgroup; record the real nodeid, the suffix is asserted separately.
    nodeid = request.node.nodeid.rsplit("@", 1)[0]
    with (request.config.rootpath / "results.tsv").open("a") as fh:
        fh.write(f"{nodeid}\\t{worker}\\t{group}\\n")
'''


def _mini_project(root: pathlib.Path, *, slow: bool = False) -> pathlib.Path:
    """Materialise the synthetic project wired to the real hook code.

    ``slow=True`` additionally adds slow-marked members for the tier-filter
    behavior tests: one slow test inside the otherwise-fast ``test_mini_a.py``
    (the bare-path boundary of the yield rule needs a mixed file) and the
    all-slow module ``test_mini_slow.py``.  The default keeps the original
    all-fast population the scheduling tests above assert on.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "conftest.py").write_text((TESTS_DIR / "conftest.py").read_text())
    (root / "scheduling.py").write_text((TESTS_DIR / "scheduling.py").read_text())
    weights = root / "mini_weights.json"
    weights.write_text(json.dumps(_MINI_WEIGHTS))
    header = "import pytest\n\n\n" if slow else ""
    alpha_tests = "\n\n".join(
        f"def test_{name}(request):\n    _record(request)"
        for name in ("alpha", "beta", "gamma")
    )
    if slow:
        alpha_tests += (
            "\n\n\n"
            + f'@pytest.mark.slow(reason="{_MINI_SLOW_REASON}")\n'
            + "def test_slow_member(request):\n    _record(request)"
        )
    (root / "test_mini_a.py").write_text(header + _RECORDER + "\n\n" + alpha_tests + "\n")
    (root / "test_mini_b.py").write_text(
        "import pytest\n\n\n"
        + _RECORDER
        + "\n\n"
        + '@pytest.mark.parametrize("case", ["x", "y"])\n'
        + "def test_case(request, case):\n    _record(request)\n"
    )
    (root / "test_mini_hot.py").write_text(
        _RECORDER
        + "\n\n"
        + "\n\n".join(
            f"def test_{name}(request):\n    _record(request)"
            for name in ("heavy_one", "heavy_two")
        )
        + "\n"
    )
    if slow:
        (root / "test_mini_slow.py").write_text(
            "import pytest\n\n\n"
            + _RECORDER
            + "\n\n"
            + "\n\n".join(
                f'@pytest.mark.slow(reason="{_MINI_SLOW_REASON}")\n'
                f"def test_slow_{name}(request):\n    _record(request)"
                for name in ("one", "two")
            )
            + "\n"
        )
    return weights


def _run_mini(root: pathlib.Path, weights: pathlib.Path, *args: str, full_suite=None):
    """Run pytest in the mini project with an explicitly constructed tier env.

    ``SKYCOLOR_FULL_SUITE`` is always set or deleted here, never inherited:
    the host session may carry it (e.g. under the full-suite entry point) and
    inheriting it would silently flip the scenario under test.
    """
    env = {k: v for k, v in os.environ.items() if k != scheduling.FULL_SUITE_ENV}
    env[scheduling.WEIGHTS_ENV] = str(weights)
    if full_suite is not None:
        env[scheduling.FULL_SUITE_ENV] = full_suite
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=180,
    )


def _read_results(root: pathlib.Path):
    out = {}
    for line in (root / "results.tsv").read_text().splitlines():
        nodeid, worker, group = line.split("\t")
        assert nodeid not in out, f"executed twice: {nodeid}"
        out[nodeid] = (worker, group)
    return out


_MINI_NODEIDS = {
    "test_mini_a.py::test_alpha",
    "test_mini_a.py::test_beta",
    "test_mini_a.py::test_gamma",
    "test_mini_b.py::test_case[x]",
    "test_mini_b.py::test_case[y]",
    "test_mini_hot.py::test_heavy_one",
    "test_mini_hot.py::test_heavy_two",
}


def test_hook_groups_are_attached_in_real_xdist_workers(tmp_path):
    weights = _mini_project(tmp_path)
    proc = _run_mini(tmp_path, weights, "-n", "2", "--dist=loadgroup")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _MINI_NODEIDS, "every test executed exactly once"
    # The hook engaged inside the workers: every executed test carries a bin.
    assert all(group != "-" for _worker, group in records.values())
    # Worker-side grouping matches the packing function bit for bit.
    expected = scheduling.assign_bins(sorted(records), 2, weights_path=weights)
    for nodeid, (_worker, group) in records.items():
        assert group == f"bin{expected[nodeid]}", nodeid
    hot = {n: g for n, (_w, g) in records.items() if "test_mini_hot" in n}
    assert len(set(hot.values())) == 2, "hot nodes must land in separate bins"
    for prefix in ("test_mini_a", "test_mini_b"):
        groups = {g for n, (_w, g) in records.items() if n.startswith(prefix)}
        workers = {w for n, (w, _g) in records.items() if n.startswith(prefix)}
        assert len(groups) == 1 and len(workers) == 1, prefix  # module = 1 group
    assert len({w for w, _g in records.values()}) >= 2, "both workers used"


def test_hook_attaches_no_groups_when_serial(tmp_path):
    weights = _mini_project(tmp_path)
    proc = _run_mini(tmp_path, weights)  # no -n: plain serial invocation
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _MINI_NODEIDS
    assert all(group == "-" for _worker, group in records.values())
    assert all(worker == "serial" for worker, _group in records.values())


def test_collect_only_with_default_parallel_addopts_exits_zero(tmp_path):
    # Regression: the hook once looked module keys up in the nodeid-keyed
    # bin map, crashing exactly this controller-side collect-only path.
    weights = _mini_project(tmp_path)
    proc = _run_mini(tmp_path, weights, "-n", "2", "--collect-only")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "INTERNALERROR" not in proc.stderr


def test_weights_refresh_round_trips_loadgroup_report_ids(tmp_path):
    # Regression: the durations report under --dist=loadgroup carries xdist's
    # "@binN" suffix on every nodeid, while the next collection looks the
    # table up by raw nodeids — parsing the report verbatim silently
    # unscheduled every hot node on the first refresh.
    suffixed = (
        "======= slowest durations =======\n"
        "1.00s setup     tests/test_twilight.py::test_deep@bin0\n"
        "120.00s call    tests/test_twilight.py::test_deep@bin0\n"
        "0.50s teardown  tests/test_twilight.py::test_deep@bin0\n"
        "110.00s call    tests/test_twilight.py::test_deep2@bin1\n"
        "3.00s call      tests/test_light.py::test_a@bin1\n"
        "2.00s call      tests/test_light.py::test_b@bin1\n"
        "1.00s call      tests/test_params.py::test_case[x@y]@bin0\n"
    )
    totals = WEIGHTS_TOOL.parse_durations(suffixed)
    # phases of one suffixed id aggregate onto the raw nodeid
    assert totals["tests/test_twilight.py::test_deep"] == pytest.approx(121.5)
    assert "tests/test_twilight.py::test_deep@bin0" not in totals
    # a legitimate '@' inside a parameter value survives, the suffix goes
    assert "tests/test_params.py::test_case[x@y]" in totals
    assert "tests/test_params.py::test_case[x@y]@bin0" not in totals
    # unsuffixed (serial) input parses to the same identity
    serial = WEIGHTS_TOOL.parse_durations(
        suffixed.replace("@bin0", "").replace("@bin1", ""))
    assert serial == totals

    # round trip: the regenerated table must serve the *next* collection,
    # whose nodeids carry no suffix
    modules, nodes = WEIGHTS_TOOL.build(totals)
    table = tmp_path / "regenerated.json"
    table.write_text(json.dumps({"modules": modules, "nodes": nodes}))
    bins = scheduling.assign_bins(sorted(totals), 2, weights_path=table)
    assert set(bins) == set(totals)
    hot_bins = {
        bins["tests/test_twilight.py::test_deep"],
        bins["tests/test_twilight.py::test_deep2"],
    }
    assert len(hot_bins) == 2, "hot nodes must stay individually scheduled"
    assert nodes["tests/test_twilight.py::test_deep"] == pytest.approx(121.5)

    # a parameter value that itself ends in "@bin<digits>" is not a suffix:
    # a real nodeid always closes the bracket, so the id must pass through
    tricky = WEIGHTS_TOOL.parse_durations(
        "5.00s call      tests/test_t2.py::test_case[v@bin0]\n")
    assert "tests/test_t2.py::test_case[v@bin0]" in tricky


# ---------------------------------------------------------------------------
# two-tier suite: slow filter policy (task-suite-tier Step 1)
# ---------------------------------------------------------------------------
def test_tier_policy_helpers_priority(monkeypatch):
    # Priority: explicit -m / -k / :: nodeid > full-suite env > default fast.
    import types

    def cfg(markexpr="", keyword="", args=()):
        return types.SimpleNamespace(
            option=types.SimpleNamespace(markexpr=markexpr, keyword=keyword),
            args=list(args),
        )

    assert scheduling.full_suite_selected() is (os.environ.get(
        scheduling.FULL_SUITE_ENV, "") not in ("", "0"))
    # The scenarios below need the env out of the way; the fixture restores
    # it afterwards.  The subprocess-level scenarios construct their own env.
    monkeypatch.delenv(scheduling.FULL_SUITE_ENV, raising=False)
    assert scheduling.fast_tier_active(cfg())
    for yielded in (
        cfg(markexpr="slow"),
        cfg(keyword="slow"),
        cfg(args=["tests/x.py::test_t"]),
    ):
        assert scheduling.slow_filter_yields(yielded)
        assert not scheduling.fast_tier_active(yielded)
    # bare paths / directories are not explicit test selections
    assert not scheduling.slow_filter_yields(cfg(args=["tests/x.py", "tests/"]))
    assert scheduling.fast_tier_active(cfg(args=["tests/x.py"]))
    monkeypatch.setenv(scheduling.FULL_SUITE_ENV, "1")
    assert scheduling.full_suite_selected()
    assert not scheduling.fast_tier_active(cfg())
    for value in ("", "0"):
        monkeypatch.setenv(scheduling.FULL_SUITE_ENV, value)
        assert not scheduling.full_suite_selected()
        assert scheduling.fast_tier_active(cfg())


_FAST_MINI_IDS = {
    "test_mini_a.py::test_alpha",
    "test_mini_a.py::test_beta",
    "test_mini_a.py::test_gamma",
    "test_mini_b.py::test_case[x]",
    "test_mini_b.py::test_case[y]",
    "test_mini_hot.py::test_heavy_one",
    "test_mini_hot.py::test_heavy_two",
}
_SLOW_MINI_IDS = {
    "test_mini_a.py::test_slow_member",
    "test_mini_slow.py::test_slow_one",
    "test_mini_slow.py::test_slow_two",
}


# (a) env unset, plain invocation: slow members deselected, fast members run.
@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_default_fast_tier_deselects_slow(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _FAST_MINI_IDS
    assert "3 deselected" in proc.stdout


# (b) env=1: every collected test runs, none deselected.
@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_full_suite_env_runs_everything(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, full_suite="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _FAST_MINI_IDS | _SLOW_MINI_IDS
    assert "deselected" not in proc.stdout


# (c) env unset + -m slow: the filter yields; only slow-marked tests run.
@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_markexpr_slow_yields(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, "-m", "slow")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _SLOW_MINI_IDS


# (d) env unset + xdist: workers inherit the env, deselect on their side, and
# pack the surviving (fast) population into coherent bins.
@pytest.mark.slow(reason="cost: mini-project xdist subprocess scenario (~4-6 s); estimated, pre-tier marking")
def test_tier_filter_xdist_workers_deselect_and_pack_survivors(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, "-n", "2", "--dist=loadgroup")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _FAST_MINI_IDS, "slow members must never execute"
    expected = scheduling.assign_bins(sorted(records), 2, weights_path=weights)
    for nodeid, (_worker, group) in records.items():
        assert group == f"bin{expected[nodeid]}", nodeid
    assert len({w for w, _g in records.values()}) >= 2, "both workers used"


# (e) env unset + -k matching slow members: same yield family as -m.
@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_keyword_expression_yields(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, "-k", "slow")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == _SLOW_MINI_IDS


# (f) nodeid yield boundary, both sides: an explicit :: nodeid runs even when
# slow; a bare path keeps the fast filter (its slow member is deselected and
# the fact is visible on the summary line).
@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_explicit_nodeid_yields(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, "test_mini_slow.py::test_slow_one")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == {"test_mini_slow.py::test_slow_one"}


@pytest.mark.slow(reason="cost: mini-project pytest subprocess scenario (~2-4 s each); estimated, pre-tier marking")
def test_tier_filter_bare_path_keeps_fast_filter(tmp_path):
    weights = _mini_project(tmp_path, slow=True)
    proc = _run_mini(tmp_path, weights, "test_mini_a.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = _read_results(tmp_path)
    assert set(records) == {
        "test_mini_a.py::test_alpha",
        "test_mini_a.py::test_beta",
        "test_mini_a.py::test_gamma",
    }
    assert "1 deselected" in proc.stdout


# ---------------------------------------------------------------------------
# validation atmosphere reuse (Step 3)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_atmosphere_cache():
    C.clear_atmosphere_cache()
    yield
    C.clear_atmosphere_cache()


def test_atmosphere_cache_same_key_returns_same_object():
    a1 = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    a2 = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    assert a1 is a2


def test_atmosphere_cache_key_isolates_every_parameter():
    base = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    assert C.atmosphere("clear", C.WL_COARSE, aod=0.07, ozone_du=300.0) is not base
    assert C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=343.8) is not base
    assert C.atmosphere("pristine", C.WL_COARSE, aod=0.05, ozone_du=300.0) is not base
    other_wl = C.WL_COARSE[:40]
    assert C.atmosphere("clear", other_wl, aod=0.05, ozone_du=300.0) is not base


def test_atmosphere_cache_clear_rebuilds_equivalent():
    a1 = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    C.clear_atmosphere_cache()
    a2 = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    assert a2 is not a1
    # Same construction inputs: the rebuilt atmosphere carries the same
    # defining state, so cached and fresh objects are interchangeable.
    assert np.array_equal(a1.wl, a2.wl)
    n1 = 0 if a1.aerosols is None else len(a1.aerosols)
    n2 = 0 if a2.aerosols is None else len(a2.aerosols)
    assert n1 == n2


def test_at_altitude_returns_independent_truncation():
    atmo = C.atmosphere("clear", C.WL_COARSE, aod=0.05, ozone_du=300.0)
    ground = float(atmo.r_bounds[0])
    raised = atmo.at_altitude(2.4)
    assert raised is not atmo
    assert float(raised.r_bounds[0]) == pytest.approx(ground + 2400.0)
    # The shared cached object itself is untouched by the truncation.
    assert float(atmo.r_bounds[0]) == ground


@pytest.mark.slow(reason="cost: 12-worker parallel 17.0 s (cold-optics listing); fast-tier budget 2 s")
def test_direct_case_never_calls_the_diffuse_solver(monkeypatch):
    import validation.check2_illuminance as c2

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("direct_case must not solve the diffuse sky")

    monkeypatch.setattr(C, "sky_radiance", _forbidden)
    small_wl = C.WL_COARSE[:9]
    d = c2.direct_case(48.19, wl=small_wl)
    assert set(d) == {"dni", "e_dni", "e_horiz", "atmo"}


@pytest.mark.slow(reason="cost: 12-worker parallel 17.1 s (cold-optics listing); fast-tier budget 2 s")
def test_model_case_calls_the_diffuse_solver_exactly_once(monkeypatch):
    import validation.check2_illuminance as c2

    calls = []

    def _record(atmo, sun_zen, z, a):
        calls.append((len(z), len(a)))
        return np.zeros((np.asarray(z).size, len(atmo.wl)))

    monkeypatch.setattr(C, "sky_radiance", _record)
    small_wl = C.WL_COARSE[:9]
    m = c2.model_case(30.0, wl=small_wl, n_zen=5, n_az=4)
    assert calls == [(20, 20)]  # one 5x4 diffuse solve for the full case
    assert set(m) == {"dni", "dhi", "ghi", "e_dni", "atmo"}


# ---------------------------------------------------------------------------
# check6 engine-point plumbing (Step 3)
# ---------------------------------------------------------------------------
@pytest.fixture
def _fast_check6(monkeypatch, tmp_path):
    """Make ``c6.run()`` cheap and side-effect free for plumbing tests.

    The deterministic solver is stubbed (what is under test is the engine
    point flow, not the physics) and the figure write is redirected out of
    the tracked tree.  Reference data (``C.DATA``) stays real.
    """

    def _zero(atmo, sun_zen, z, a):
        return np.zeros((np.asarray(z).size, len(atmo.wl)))

    monkeypatch.setattr(c6, "sky_radiance_total", _zero)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    # check6 writes through validation.common.FIG (research-boundary Step 4
    # moved it off C.ROOT): patch the module attribute the write actually
    # reads, so the stub-physics run can never touch the tracked figure
    monkeypatch.setattr(C, "FIG", fig_dir)
    return fig_dir


def _fake_engine_arrays():
    """Fake ``engine_point`` output: (n_dir, n_wl) like ``twilight_radiance``."""
    rad = np.full((1, len(C.WL_COARSE)), 1e-3)
    err = np.full((1, len(C.WL_COARSE)), 1e-4)
    return rad, err


def test_engine_depths_match_dispatch_contract():
    depths = c6.engine_depths()
    assert len(depths) == len(set(depths)), "duplicate dispatch depths"
    assert depths == sorted(depths)
    # -6 is the supplemental engine print; the rest are below the split.
    assert -6.0 in depths
    assert all(h < c6.SPLIT or h == -6.0 for h in depths)


def test_run_with_injection_never_computes_engine_points(_fast_check6, monkeypatch):
    rad, err = _fake_engine_arrays()
    injected = {h: (rad.copy(), err.copy()) for h in c6.engine_depths()}

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("injected engine points must not be recomputed")

    monkeypatch.setattr(c6, "engine_point", _forbidden)
    result = c6.run(engine_points=injected)
    assert result.details["model"]["-9"] == float(C.luminance(rad, C.WL_COARSE)[0])


def test_run_without_injection_uses_engine_point_per_depth(_fast_check6, monkeypatch):
    rad, err = _fake_engine_arrays()
    requested = []

    def _stub(h, kind="clear", wl=None):
        requested.append((float(h), kind, None if wl is None else np.asarray(wl)))
        return rad.copy(), err.copy()

    monkeypatch.setattr(c6, "engine_point", _stub)
    c6.run()
    depths = sorted({h for h, _kind, _wl in requested})
    assert depths == sorted(c6.engine_depths())
    # every engine call uses the canonical engine parameters
    for h, kind, wl in requested:
        assert kind == "clear"
        assert wl is not None and np.array_equal(wl, C.WL_COARSE)


def test_run_engine_results_flow_into_metrics(_fast_check6):
    rad, err = _fake_engine_arrays()
    injected = {h: (rad.copy(), err.copy()) for h in c6.engine_depths()}
    result = c6.run(engine_points=injected)
    # deep engine points consumed the injected arrays: model values equal the
    # photopic integral of the fake radiance
    expected = float(C.luminance(rad, C.WL_COARSE)[0])
    expected_err = float(C.luminance(err, C.WL_COARSE)[0])
    for h in ("-9", "-12", "-15"):
        assert result.details["model"][h] == expected
        assert result.details["engine_error"][h] == expected_err
    assert result.details["engine_l6"] == expected
    assert result.details["engine_l6_err"] == expected_err


# ---------------------------------------------------------------------------
# run_all dispatch structure (Step 3)
# ---------------------------------------------------------------------------
def test_run_all_worker_functions_are_top_level():
    # spawn workers unpickle these by qualified name: they must be module
    # level functions of validation.run_all (not closures or lambdas).
    import validation.run_all as ra

    for name in ("_run_check", "_engine_point_worker"):
        fn = getattr(ra, name)
        assert fn.__qualname__ == name, name
        assert fn.__module__.endswith("validation.run_all")


def test_run_check_propagates_worker_errors(monkeypatch):
    import validation.run_all as ra

    def _boom(name):
        raise ValueError("boom")

    monkeypatch.setattr(ra.importlib, "import_module", _boom)
    with pytest.raises(ValueError):
        ra._run_check("validation.check_whatever", "checkX", "t")


def test_run_check_reports_missing_module_as_skip():
    import validation.run_all as ra

    result = ra._run_check("validation.check999_absent", "check999", "missing")
    assert result.verdict == "SKIP"
