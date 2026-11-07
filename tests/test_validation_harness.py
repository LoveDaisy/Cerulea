"""Structural tests for the fast validation harness.

Two families, both structural only — the numeric equivalence of the physical
results is verified by real runs against the committed baseline (see
``validation/FAST_SUITE.md``), not here:

* pytest scheduling: the LPT packing used by ``tests/conftest.py`` must be
  deterministic, cover every collected test exactly once, tolerate unknown
  tests, and work for any worker count.
* validation restructuring: atmosphere reuse is keyed and read-only,
  check2's ASTM path genuinely skips the diffuse solve while h=30/60 keep
  it, and check6's engine-point injection consumes exactly the dispatched
  Monte-Carlo depths.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

import scheduling
from validation import check6_second_dataset as c6
from validation import common as C

TESTS_DIR = pathlib.Path(__file__).parent


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


def test_direct_case_never_calls_the_diffuse_solver(monkeypatch):
    import validation.check2_illuminance as c2

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("direct_case must not solve the diffuse sky")

    monkeypatch.setattr(C, "sky_radiance", _forbidden)
    small_wl = C.WL_COARSE[:9]
    d = c2.direct_case(48.19, wl=small_wl)
    assert set(d) == {"dni", "e_dni", "e_horiz", "atmo"}


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
    monkeypatch.setattr(C, "ROOT", tmp_path)
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
