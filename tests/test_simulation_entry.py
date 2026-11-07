"""Tests for the consolidated simulation entry (``rt.solver.solve``).

Contract under test (plan task-simulation-entry, acceptance A1-A5):

1. the four finite approximations are reachable, each component comes from
   exactly one kernel call (spied), and totals are exact component sums;
2. ``plane-sos`` serves both orders from one SOS run (``single_view``) and
   equals a direct kernel call; ``hybrid-sos`` is bit-identical to the
   legacy ndarray wrapper;
3. invalid requests fail *before* any kernel runs (unknown names, settings
   given twice, settings for kernels the request does not run, unknown
   quality/grid keys, both orders off);
4. support-domain boundaries: plane terms require the model-bottom observer
   (``at_altitude`` stations qualify), ``plane-sos`` rejects a non-uniform
   ``HorizontalField`` (no component applies it) while ``hybrid-sos`` keeps
   the partial support with one warning;
5. the result records the *effective* kernel settings (post-clamp), not a
   copy of the request.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor import presets  # noqa: E402
from skycolor.rt import multiple, single, twilight  # noqa: E402
from skycolor.rt.solver import (  # noqa: E402
    MCConfig, SOSConfig, SingleConfig, solve, sky_radiance_total,
)

WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
VZA = np.array([0.0, 30.0, 60.0, 85.0])
ZEN_TW = np.array([0.0, 30.0, 70.0])
AZ_TW = np.array([0.0, 180.0, 90.0])
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)


@pytest.fixture(scope="module")
def atmo7():
    return presets.clear(WL7)


# ---------------------------------------------------------------------------
# 1. the four approximations are reachable and closed
# ---------------------------------------------------------------------------
def test_spherical_single_reachable_and_closed(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0, approximation="spherical-single")
    ref = single.sky_radiance(atmo7, 60.0, VZA, 90.0)
    assert np.array_equal(res.radiance, ref)
    assert res.approximation == "spherical-single"
    assert res.method == "clear"
    assert np.array_equal(res.components["single"], ref)
    assert res.components["multiple"] is None
    assert res.diagnostics == {}          # no SOS ran, no orders claimed
    assert res.stderr is None


def test_spherical_single_rejects_explicit_multiple(atmo7):
    with pytest.raises(ValueError, match="spherical-single"):
        solve(atmo7, 60.0, 30.0, 0.0, approximation="spherical-single",
              multiple=True)


def test_plane_sos_matches_direct_kernel_call(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos")
    sos = multiple.solve(atmo7, 60.0, VZA, 90.0, single_view=True)
    assert np.array_equal(res.radiance, sos.radiance_single + sos.radiance)
    comps = res.components
    assert np.array_equal(comps["single"], sos.radiance_single)
    assert np.array_equal(comps["multiple"], sos.radiance)
    assert np.array_equal(res.radiance, comps["single"] + comps["multiple"])
    assert res.diagnostics["orders"] == sos.orders
    assert res.approximation == "plane-sos"
    assert res.method == "clear"


def test_plane_sos_order_selection(atmo7):
    sos = multiple.solve(atmo7, 60.0, VZA, 90.0, single_view=True)
    res_ss = solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos",
                   multiple=False)
    assert np.array_equal(res_ss.radiance, sos.radiance_single)
    assert res_ss.components["multiple"] is None
    res_ms = solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos",
                   single=False)
    assert np.array_equal(res_ms.radiance, sos.radiance)
    assert res_ms.components["single"] is None


def test_plane_sos_single_only_still_one_sos_run(atmo7, monkeypatch):
    """single-only plane-sos takes the explicit existing-kernel cost: the
    full SOS runs once with single_view and the first order is extracted."""
    import skycolor.rt.solver as solver

    calls = {"sos": 0, "single": 0}
    real_solve = multiple.solve

    def spy_solve(*a, **k):
        calls["sos"] += 1
        assert k.get("single_view") is True
        return real_solve(*a, **k)

    monkeypatch.setattr(solver._multiple, "solve", spy_solve)
    monkeypatch.setattr(solver, "_single",
                        lambda *a, **k: calls.__setitem__(
                            "single", calls["single"] + 1))
    solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos", multiple=False)
    assert calls == {"sos": 1, "single": 0}


def test_sos_config_reaches_kernel(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos",
                sos_config=SOSConfig(n_mu=8, max_orders=3, tol=0.5))
    sos = multiple.solve(atmo7, 60.0, VZA, 90.0, single_view=True,
                         n_mu=8, max_orders=3, tol=0.5)
    assert np.array_equal(res.radiance, sos.radiance_single + sos.radiance)


def test_spherical_mc_reachable(atmo7):
    res = solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
                mc_config=MCConfig(quality="low", seed=0, grid=TINY))
    ref, ref_err = twilight.twilight_radiance(
        atmo7, 96.0, ZEN_TW, AZ_TW, quality="low", seed=0, grid=TINY)
    assert np.array_equal(res.radiance, ref)
    assert np.array_equal(res.stderr, ref_err)
    assert res.approximation == "spherical-mc"
    assert res.method == "twilight"
    assert res.effective["mc"]["seed"] == 0


def test_mc_seed_reaches_the_stream(atmo7):
    kw = dict(approximation="spherical-mc",
              mc_config=MCConfig(quality="low", grid=TINY))
    r0 = solve(atmo7, 96.0, ZEN_TW, AZ_TW, seed=0, **kw)
    r1 = solve(atmo7, 96.0, ZEN_TW, AZ_TW, seed=1, **kw)
    assert not np.array_equal(r0.stderr, r1.stderr)
    assert not np.array_equal(r0.radiance, r1.radiance)


# ---------------------------------------------------------------------------
# 2. hybrid-sos: one single call + one SOS call, legacy parity
# ---------------------------------------------------------------------------
def test_hybrid_matches_legacy_wrapper_bitwise(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0)
    ref = sky_radiance_total(atmo7, 60.0, VZA, 90.0)
    assert np.array_equal(res.radiance, ref)
    assert res.approximation == "hybrid-sos"
    assert np.array_equal(res.radiance,
                          res.components["single"] + res.components["multiple"])


def test_hybrid_calls_each_kernel_exactly_once(atmo7, monkeypatch):
    import skycolor.rt.solver as solver

    real_solve = multiple.solve
    calls = {"sos": 0, "single": 0}

    def spy_solve(*a, **k):
        calls["sos"] += 1
        return real_solve(*a, **k)

    def spy_single(*a, **k):
        calls["single"] += 1
        return single.sky_radiance(*a, **k)

    monkeypatch.setattr(solver._multiple, "solve", spy_solve)
    monkeypatch.setattr(solver, "_single", spy_single)
    solve(atmo7, 60.0, VZA, 90.0)
    assert calls == {"sos": 1, "single": 1}
    # disabled terms are not called
    calls.clear(); calls.update(sos=0, single=0)
    solve(atmo7, 60.0, VZA, 90.0, multiple=False)
    assert calls == {"sos": 0, "single": 1}
    calls.clear(); calls.update(sos=0, single=0)
    solve(atmo7, 60.0, VZA, 90.0, single=False)
    assert calls == {"sos": 1, "single": 0}


# ---------------------------------------------------------------------------
# 3. request validation happens before any kernel runs
# ---------------------------------------------------------------------------
def test_method_aliases_and_conflicts(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0, method="clear",
                approximation="hybrid-sos")
    assert res.approximation == "hybrid-sos"
    res = solve(atmo7, 96.0, ZEN_TW, AZ_TW, method="twilight",
                approximation="spherical-mc",
                mc_config=MCConfig(quality="low", grid=TINY))
    assert res.approximation == "spherical-mc"
    with pytest.raises(ValueError, match="disagree"):
        solve(atmo7, 60.0, 30.0, 0.0, method="clear",
              approximation="plane-sos")
    with pytest.raises(ValueError, match="unknown method"):
        solve(atmo7, 60.0, 30.0, 0.0, method="auto")
    with pytest.raises(ValueError, match="unknown approximation"):
        solve(atmo7, 60.0, 30.0, 0.0, approximation="auto")


def test_invalid_requests_fail_before_kernels(atmo7, monkeypatch):
    import skycolor.rt.solver as solver

    def boom(*a, **k):
        raise AssertionError("kernel must not run on an invalid request")

    monkeypatch.setattr(solver, "_single", boom)
    monkeypatch.setattr(solver._multiple, "solve", boom)
    with pytest.raises(ValueError, match="twice"):
        solve(atmo7, 60.0, 30.0, 0.0, n_gauss=4,
              single_config=SingleConfig(n_gauss=5))
    with pytest.raises(ValueError, match="Monte Carlo"):
        solve(atmo7, 60.0, 30.0, 0.0, quality="low")
    with pytest.raises(ValueError, match="SOS kernel"):
        solve(atmo7, 60.0, 30.0, 0.0, approximation="spherical-single",
              sos_config=SOSConfig(n_mu=48))
    with pytest.raises(ValueError, match="single-scattering kernel"):
        solve(atmo7, 60.0, 30.0, 0.0, single=False, n_gauss=4)
    with pytest.raises(ValueError, match="unknown quality"):
        solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
              quality="extreme")
    with pytest.raises(ValueError, match="grid keys"):
        solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
              mc_config=MCConfig(quality="low", grid={"bad_key": 1}))
    with pytest.raises(ValueError, match="computes nothing"):
        solve(atmo7, 60.0, 30.0, 0.0, single=False, multiple=False)
    with pytest.raises(ValueError, match="requires cloud_layer"):
        solve(atmo7, 60.0, 30.0, 0.0, method="cloud")


# ---------------------------------------------------------------------------
# 4. support-domain boundaries
# ---------------------------------------------------------------------------
def test_plane_terms_require_bottom_observer(atmo7):
    balloon = presets.clear(WL7)
    balloon.elevation_km = 2.0
    with pytest.raises(ValueError, match="model bottom"):
        solve(balloon, 60.0, VZA, 90.0)                 # hybrid SOS
    with pytest.raises(ValueError, match="model bottom"):
        solve(balloon, 60.0, VZA, 90.0, approximation="plane-sos")
    # spherical single has no plane term: allowed off the ground
    res = solve(balloon, 60.0, VZA, 90.0, approximation="spherical-single")
    assert res.radiance.shape == (VZA.size, WL7.size)
    # a truncated station qualifies by construction
    station = presets.clear(WL7).at_altitude(2.0)
    res = solve(station, 60.0, VZA, 90.0, approximation="plane-sos")
    assert res.components["multiple"] is not None


def test_horizontal_field_support_boundaries(atmo7):
    from skycolor.aerosol import HinterlandAerosol, shared
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile, \
        RadialSplitField

    het = presets.clear(WL7)
    het.aerosols = [AerosolLayer(shared(HinterlandAerosol), aod550=0.05,
                                 profile=ExponentialProfile(1.5),
                                 horizontal=RadialSplitField(radius_km=20.0))]
    # plane-sos applies the field in no component: reject
    with pytest.raises(ValueError, match="HorizontalField"):
        solve(het, 60.0, VZA, 90.0, approximation="plane-sos")
    # hybrid keeps the historical partial support: warn once
    with pytest.warns(RuntimeWarning, match="HorizontalField"):
        solve(het, 60.0, VZA, 90.0)
    # single-only hybrid applies the field everywhere: silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve(het, 60.0, VZA, 90.0, multiple=False)


# ---------------------------------------------------------------------------
# 5. effective settings are kernel truth, request values stay in options
# ---------------------------------------------------------------------------
def test_effective_settings_match_kernel_clamps(atmo7):
    res = solve(atmo7, 60.0, VZA, 90.0,
                sos_config=SOSConfig(n_mu=33, n_moments=64, tol=1e-2))
    sos_eff = res.effective["sos"]
    assert sos_eff["n_mu"] == 34              # odd node counts are evened
    assert sos_eff["n_moments"] == 64         # under the 2*n_mu = 68 cap
    assert sos_eff["tol"] == pytest.approx(1e-2)
    assert res.effective["single"] == {"n_gauss": 3, "chunk": 512}
    assert res.options["n_gauss"] == 3 and res.options["chunk"] == 512
    assert res.options["approximation"] == "hybrid-sos"


def test_default_options_bookkeeping_superset(atmo7):
    res = solve(atmo7, 80.0, 60.0, 120.0)
    assert {"sun_zenith_deg", "sun_azimuth_deg", "single", "multiple",
            "n_gauss", "chunk"} <= set(res.options)
