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

from skycolor import cloud, presets  # noqa: E402
from skycolor.rt import multiple, single, twilight  # noqa: E402
from skycolor.rt.cloud import apply_cloud_overlay, sky_radiance_with_cloud  # noqa: E402
from skycolor.rt.solver import (  # noqa: E402
    MCConfig, SOSConfig, SingleConfig, solve, sky_radiance_total,
)

WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
VZA = np.array([0.0, 30.0, 60.0, 85.0])
ZEN_C = np.array([0.0, 30.0, 60.0])
ZEN_TW = np.array([0.0, 30.0, 70.0])
AZ_TW = np.array([0.0, 180.0, 90.0])
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)


@pytest.fixture(scope="module")
def atmo7():
    return presets.clear(WL7)


@pytest.fixture(scope="module")
def stratus():
    return cloud.get_preset("stratus")


# ---------------------------------------------------------------------------
# 1. the four approximations are reachable and closed
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
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
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_default_options_bookkeeping_superset(atmo7):
    res = solve(atmo7, 80.0, 60.0, 120.0)
    assert {"sun_zenith_deg", "sun_azimuth_deg", "single", "multiple",
            "n_gauss", "chunk"} <= set(res.options)


# ---------------------------------------------------------------------------
# 6. MC consolidation: the solver assembles single + multiple_scattered
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
def test_mc_route_matches_direct_composition(atmo7):
    res = solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
                mc_config=MCConfig(quality="low", seed=0, grid=TINY))
    ss = single.sky_radiance(atmo7, 96.0, ZEN_TW, AZ_TW)
    ms, err, _ = twilight.multiple_scattered(atmo7, 96.0, ZEN_TW, AZ_TW,
                                             quality="low", seed=0, grid=TINY)
    assert np.array_equal(res.components["single"], ss)
    assert np.array_equal(res.components["multiple"], ms)
    assert np.array_equal(res.radiance, ss + ms)
    assert np.array_equal(res.stderr, err)


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
def test_mc_route_calls_kernels_exactly_once(atmo7, monkeypatch):
    import skycolor.rt.solver as solver

    calls = {"single": 0, "mc": 0}
    real_ms = twilight.multiple_scattered

    def spy_ms(*a, **k):
        calls["mc"] += 1
        return real_ms(*a, **k)

    def spy_single(*a, **k):
        calls["single"] += 1
        return single.sky_radiance(*a, **k)

    monkeypatch.setattr(solver._twilight, "multiple_scattered", spy_ms)
    monkeypatch.setattr(solver, "_single", spy_single)
    # the full twilight wrapper must NOT be re-entered by the solver
    monkeypatch.setattr(solver._twilight, "twilight_radiance",
                        lambda *a, **k: pytest.fail(
                            "solver must compose the MC route itself"))
    solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
          mc_config=MCConfig(quality="low", seed=0, grid=TINY))
    assert calls == {"single": 1, "mc": 1}


def test_mc_single_only_keeps_legacy_zero_error(atmo7):
    res = solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
                multiple=False)
    ref = single.sky_radiance(atmo7, 96.0, ZEN_TW, AZ_TW)
    assert np.array_equal(res.radiance, ref)
    assert np.all(res.stderr == 0.0)
    assert res.options["mc_run"] is False
    assert res.components["multiple"] is None


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_twilight_wrapper_scalar_array_broadcast(atmo7):
    """scalar zenith + array azimuth keeps the historical wrapper contract
    and is bitwise identical to the explicit array request."""
    r0, e0 = twilight.twilight_radiance(atmo7, 96.0, 30.0,
                                        np.array([0.0, 180.0, 90.0]),
                                        quality="low", grid=TINY, seed=0)
    r1, e1 = twilight.twilight_radiance(
        atmo7, 96.0, np.array([30.0, 30.0, 30.0]),
        np.array([0.0, 180.0, 90.0]), quality="low", grid=TINY, seed=0)
    assert np.array_equal(r0, r1)
    assert np.array_equal(e0, e1)


# ---------------------------------------------------------------------------
# 7. cloud consolidation: local operator + solver overlay
# ---------------------------------------------------------------------------
def test_solve_overlay_matches_wrapper_bitwise(atmo7, stratus):
    res = solve(atmo7, 45.0, ZEN_C, 135.0, cloud_layer=stratus)
    ref = sky_radiance_with_cloud(atmo7, stratus, 45.0, ZEN_C, 135.0)
    assert np.array_equal(res.radiance, ref)
    comps = res.components
    assert np.array_equal(res.radiance, comps["clear"] + comps["cloud"])
    # raw un-occluded clear sub-components are recorded, not summed
    ss = single.sky_radiance(atmo7, 45.0, ZEN_C, 135.0)
    ms = multiple.solve(atmo7, 45.0, ZEN_C, 135.0).radiance
    assert np.array_equal(comps["single"], ss)
    assert np.array_equal(comps["multiple"], ms)


def test_apply_cloud_overlay_never_computes_clear(atmo7, stratus):
    """The local operator composes a caller-supplied field; hand it a field
    of NaN sentinels and verify they flow through untouched (no kernel)."""
    nan_field = np.full((ZEN_C.size, WL7.size), np.nan)
    total, comps = apply_cloud_overlay(atmo7, stratus, 45.0, ZEN_C, nan_field,
                                       include_clear=True)
    assert np.all(np.isnan(comps["clear"]))
    # trans/cloud parts are finite: the NaNs only ride the clear column
    assert np.all(np.isfinite(comps["cloud"]))
    assert np.all(np.isfinite(comps["transmittance"]))
    # the NaN sentinels propagate into the total exactly where the clear
    # field enters it (clear*trans), nowhere else
    assert np.all(np.isnan(total) == np.broadcast_to(
        np.isnan(nan_field * comps["transmittance"]), total.shape))


def test_external_clear_is_not_recomputed(atmo7, stratus, monkeypatch):
    import skycolor.rt.solver as solver

    def boom(*a, **k):
        raise AssertionError("no clear kernel may run for an external field")

    pre = solve(atmo7, 45.0, ZEN_C, 135.0, cloud_layer=stratus)
    ext = pre.components["single"] + pre.components["multiple"]
    monkeypatch.setattr(solver, "_single", boom)
    monkeypatch.setattr(solver._multiple, "solve", boom)
    res = solve(atmo7, 45.0, ZEN_C, 135.0, cloud_layer=stratus,
                clear_radiance=ext)
    assert np.array_equal(res.radiance, pre.radiance)
    assert res.options["clear_radiance"] == "external"
    # unknown provenance: no scattering decomposition claimed
    assert "single" not in res.approximations
    assert "multiple" not in res.approximations
    assert "single" not in res.components
    assert "multiple" not in res.components


def test_solver_overlay_does_not_route_through_the_wrapper(atmo7, stratus,
                                                           monkeypatch):
    """No solve -> legacy cloud wrapper -> solve cycle: patching the wrapper
    to fail must not affect the solver's overlay route."""
    import skycolor.rt.cloud as rtcloud

    monkeypatch.setattr(rtcloud, "sky_radiance_with_cloud",
                        lambda *a, **k: pytest.fail(
                            "solver overlay must not call the wrapper"))
    solve(atmo7, 45.0, ZEN_C, 135.0, cloud_layer=stratus)


def test_cloud_observer_domain(atmo7, stratus):
    above = presets.clear(WL7)
    above.elevation_km = 2.0        # above the stratus base (0.5 km)
    with pytest.raises(ValueError, match="cloud base"):
        solve(above, 45.0, 30.0, 0.0, cloud_layer=stratus)
    with pytest.raises(ValueError, match="cloud base"):
        solve(above, 45.0, 30.0, 0.0, cloud_layer=stratus,
              clear_radiance=np.zeros((1, WL7.size)))


def test_overlay_request_validation(atmo7, stratus):
    with pytest.raises(ValueError, match="contradicts"):
        solve(atmo7, 45.0, 30.0, 0.0, method="clear", cloud_layer=stratus)
    with pytest.raises(ValueError, match="contradicts"):
        solve(atmo7, 45.0, 30.0, 0.0, method="twilight", cloud_layer=stratus)
    with pytest.raises(ValueError, match="composes with"):
        solve(atmo7, 45.0, 30.0, 0.0, approximation="spherical-mc",
              cloud_layer=stratus)
    with pytest.raises(ValueError, match="pass cloud_layer"):
        solve(atmo7, 45.0, 30.0, 0.0, ground_albedo=0.3)
    with pytest.raises(ValueError, match="pass cloud_layer"):
        solve(atmo7, 45.0, 30.0, 0.0, include_cloud=False)


def test_include_switches_and_ground_override(atmo7, stratus, monkeypatch):
    res_c = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus)
    res_nc = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                   include_cloud=False)
    assert np.array_equal(res_nc.radiance, res_c.components["clear"])
    # cloud-only request: no clear-sky kernel runs at all (the historical
    # wrapper skipped the clear solve; the formal entry now shares that
    # execution plan), the excluded clear term is recorded as zeros
    import skycolor.rt.solver as solver
    calls = {"sos": 0, "single": 0}
    real_solve = multiple.solve

    def spy_solve(*a, **k):
        calls["sos"] += 1
        return real_solve(*a, **k)

    monkeypatch.setattr(solver._multiple, "solve", spy_solve)
    monkeypatch.setattr(solver, "_single",
                        lambda *a, **k: calls.__setitem__(
                            "single", calls["single"] + 1))
    res_ncl = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                    include_clear=False)
    assert calls == {"sos": 0, "single": 0}
    assert np.array_equal(res_ncl.radiance, res_c.components["cloud"])
    assert np.all(res_ncl.components["clear"] == 0.0)
    assert res_ncl.effective == {} and "sos" not in res_ncl.approximations
    # the wrapper forwards to the same cloud request: identical numbers and
    # identical zero clear-kernel work
    tot_w, comp_w = sky_radiance_with_cloud(
        atmo7, stratus, 45.0, ZEN_C, 90.0, include_clear=False,
        return_components=True)
    assert np.array_equal(tot_w, res_ncl.radiance)
    assert np.all(comp_w["clear"] == 0.0)
    res_g = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                  ground_albedo=0.3)
    assert not np.array_equal(res_g.radiance, res_c.radiance)
    assert res_g.options["ground_albedo_override"] is True


def test_tau0_deck_reduces_to_clear(atmo7):
    zero = cloud.get_preset("stratus", tau550=0.0)
    res = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=zero)
    ref = solve(atmo7, 45.0, ZEN_C, 90.0)      # hybrid-sos clear
    assert np.all(res.components["transmittance"] == 1.0)
    assert np.array_equal(res.radiance, ref.radiance)


def test_overlay_honors_sos_config(atmo7, stratus):
    res = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                sos_config=SOSConfig(n_mu=8, max_orders=3, tol=0.5))
    assert res.effective["sos"]["n_mu"] == 8
    assert res.effective["sos"]["max_orders"] == 3


# ---------------------------------------------------------------------------
# 8. H3 invariants: media/caching/restriction/render seam through the entry
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_same_medium_reentry_is_stable(atmo7):
    """Repeated solve() calls on one medium are bitwise identical (no
    per-call drift from shared caches), for a deterministic and an MC
    route."""
    r1 = solve(atmo7, 65.0, VZA, 90.0)
    r2 = solve(atmo7, 65.0, VZA, 90.0)
    assert np.array_equal(r1.radiance, r2.radiance)
    m1 = solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
               mc_config=MCConfig(quality="low", seed=0, grid=TINY))
    m2 = solve(atmo7, 96.0, ZEN_TW, AZ_TW, approximation="spherical-mc",
               mc_config=MCConfig(quality="low", seed=0, grid=TINY))
    assert np.array_equal(m1.radiance, m2.radiance)
    assert np.array_equal(m1.stderr, m2.stderr)


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_independent_construction_same_result():
    """Two independently built (equal) media give bitwise equal results."""
    a1 = presets.clear(WL7, aod=0.05)
    a2 = presets.clear(WL7, aod=0.05)
    r1 = solve(a1, 65.0, VZA, 90.0)
    r2 = solve(a2, 65.0, VZA, 90.0)
    assert np.array_equal(r1.radiance, r2.radiance)
    assert r1.diagnostics["orders"] == r2.diagnostics["orders"]


def test_restrict_column_consistency(atmo7):
    """Solving on a wavelength-restricted medium reproduces the restricted
    columns of the full-grid solve bit-for-bit (no re-normalisation on the
    restriction path).  Deterministic routes only: the MC anchor grid is a
    function of the wavelength count, so a restricted MC run is a different
    problem by design."""
    idx = [0, 2, 4, 6]
    full = solve(atmo7, 65.0, VZA, 90.0).radiance
    sub = solve(atmo7._restrict(idx), 65.0, VZA, 90.0).radiance
    assert np.array_equal(sub, full[:, idx])
    full_ss = solve(atmo7, 65.0, VZA, 90.0, multiple=False).radiance
    sub_ss = solve(atmo7._restrict(idx), 65.0, VZA, 90.0, multiple=False).radiance
    assert np.array_equal(sub_ss, full_ss[:, idx])


def test_render_default_path_reaches_the_entry(atmo7):
    """render.sky_radiance_grid (no radiance_field) is bit-identical to the
    authoritative entry -- the display layer rides the thin wrapper."""
    from skycolor import render

    zen = np.array([0.0, 30.0, 60.0, 90.0])
    az = np.array([0.0, 45.0, 90.0, 135.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    grid = render.sky_radiance_grid(atmo7, 65.0, Z, A)
    direct = solve(atmo7, 65.0, Z.ravel(), A.ravel()).radiance
    assert np.array_equal(grid.reshape(-1, WL7.size), direct)


# ---------------------------------------------------------------------------
# 9. review round 1: single execution plan, honest records, uniform domains
# ---------------------------------------------------------------------------
def test_wrapper_routes_every_cloud_request_through_solve(atmo7, stratus,
                                                          monkeypatch):
    """The legacy cloud wrapper forwards the *cloud request* itself, so the
    support domain cannot differ between the two entry points."""
    import skycolor.rt.solver as solver

    # normal request: identical numbers through both entries
    res = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus)
    tot = sky_radiance_with_cloud(atmo7, stratus, 45.0, ZEN_C, 90.0)
    assert np.array_equal(tot, res.radiance)
    # precomputed clear: also one solve call (overlay only), same numbers
    res_ext = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                    clear_radiance=np.zeros((ZEN_C.size, WL7.size)))
    tot_ext = sky_radiance_with_cloud(
        atmo7, stratus, 45.0, ZEN_C, 90.0,
        clear_radiance=np.zeros((ZEN_C.size, WL7.size)))
    assert np.array_equal(tot_ext, res_ext.radiance)
    assert res_ext.options["clear_radiance"] == "external"
    # an observer inside the deck is rejected on the wrapper too -- the old
    # external-clear path used to bypass the cloud-base check entirely
    above = presets.clear(WL7)
    above.elevation_km = 2.0
    with pytest.raises(ValueError, match="cloud base"):
        sky_radiance_with_cloud(above, stratus, 45.0, 30.0, 0.0,
                                clear_radiance=np.zeros((1, WL7.size)))


def test_settings_for_kernels_that_will_not_run_are_rejected(atmo7, stratus):
    """Applicability follows the execution plan, not a switch: plane-sos
    evaluates its first order inside the SOS run, and an external-clear /
    cloud-only request runs no clear kernel at all."""
    with pytest.raises(ValueError, match="single-scattering kernel"):
        solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos", n_gauss=9)
    with pytest.raises(ValueError, match="single-scattering kernel"):
        solve(atmo7, 60.0, VZA, 90.0, approximation="plane-sos",
              single_config=SingleConfig(chunk=17))
    with pytest.raises(ValueError, match="single-scattering kernel"):
        solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
              clear_radiance=np.zeros((ZEN_C.size, WL7.size)),
              single_config=SingleConfig(n_gauss=9))
    with pytest.raises(ValueError, match="SOS kernel"):
        solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
              clear_radiance=np.zeros((ZEN_C.size, WL7.size)),
              sos_config=SOSConfig(n_mu=8))
    with pytest.raises(ValueError, match="SOS kernel"):
        solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
              include_clear=False, sos_config=SOSConfig(n_mu=8))


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
def test_mc_records_actual_single_settings(atmo7):
    """spherical-mc with a non-default single config: the request values
    land in options AND the effective record."""
    res = solve(atmo7, 60.0, VZA, 90.0, approximation="spherical-mc",
                single_config=SingleConfig(n_gauss=2, chunk=64),
                mc_config=MCConfig(grid=dict(TINY)))
    assert res.options["n_gauss"] == 2 and res.options["chunk"] == 64
    assert res.effective["single"] == {"n_gauss": 2, "chunk": 64}


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
def test_mc_effective_reports_what_ran_not_what_was_asked():
    """Anchor dedup and per-direction photon rounding are engine facts:
    a single-wavelength model collapses n_anchor=5 to one anchor, and a
    2x2 coarse grid rounds n_photons=1 up to one path per direction."""
    wl1 = np.array([550.0])
    tiny2 = dict(n_zen=2, n_az=2, n_anchor=5, n_photons=1.0)
    res = solve(presets.clear(wl1), 60.0, VZA, 90.0,
                approximation="spherical-mc", mc_config=MCConfig(grid=tiny2))
    mc_eff = res.effective["mc"]
    assert mc_eff["n_anchor_run"] == 1          # unique indices <= n_wl
    assert mc_eff["n_photons_run"] == 2 * 2 * 1  # per_dir floor of 1 x n_dir
    # the *requested* preset values stay visible in diagnostics/info
    assert res.diagnostics["n_anchor"] == 5
    assert res.diagnostics["n_photons"] == 1.0


def test_sos_records_optics_and_transport_moment_counts(atmo7):
    """delta-M optics are built from the requested moments before the
    transport truncation (2*n_mu cap); both numbers affect the result and
    are recorded separately, and the raw request stays in options."""
    res = solve(atmo7, 60.0, VZA, 90.0,
                sos_config=SOSConfig(n_mu=4, n_moments=32))
    assert res.options["sos"] == {"n_mu": 4, "n_moments": 32}
    sos_eff = res.effective["sos"]
    assert sos_eff["n_moments_optics"] == 32    # optics built with 32
    assert sos_eff["n_moments"] == 8            # transport truncated to 2*n_mu
    assert sos_eff["n_mu"] == 4


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_refraction_state_recorded_not_assumed(atmo7):
    """The per-component record carries the actual solar-side refraction
    switch; a result must not claim a refraction treatment that did not
    happen."""
    on = solve(atmo7, 60.0, VZA, 90.0)
    assert on.options["refraction"] is True
    assert on.approximations["single"]["source_refraction"] is True
    atmo_nr = presets.clear(WL7)
    atmo_nr.refraction = False
    off = solve(atmo_nr, 60.0, VZA, 90.0)
    assert off.options["refraction"] is False
    assert off.approximations["single"]["source_refraction"] is False
    assert off.approximations["multiple"]["source_refraction"] is False


def test_horizontal_field_unused_by_the_plan_is_rejected(atmo7, monkeypatch):
    """hybrid/mc with the higher orders only (single term off) applies the
    field in no component: reject before any kernel runs."""
    from skycolor.aerosol import HinterlandAerosol, shared
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile, \
        RadialSplitField

    het = presets.clear(WL7)
    het.aerosols = [AerosolLayer(shared(HinterlandAerosol), aod550=0.05,
                                 profile=ExponentialProfile(1.5),
                                 horizontal=RadialSplitField(radius_km=20.0))]
    import skycolor.rt.solver as solver
    calls = {"n": 0}
    real_single = solver._single

    def spy_single(*a, **k):
        calls["n"] += 1
        return real_single(*a, **k)

    monkeypatch.setattr(solver, "_single", spy_single)
    with pytest.raises(ValueError, match="no spherical single term"):
        solve(het, 60.0, VZA, 90.0, single=False)          # hybrid, SOS only
    with pytest.raises(ValueError, match="no spherical single term"):
        solve(het, 60.0, VZA, 90.0, approximation="spherical-mc",
              single=False)
    assert calls["n"] == 0                                 # zero kernel work
    # an external clear field consumes the field in no component either
    with pytest.raises(ValueError, match="no spherical single term"):
        solve(het, 45.0, ZEN_C, 90.0,
              cloud_layer=cloud.get_preset("stratus"),
              clear_radiance=np.zeros((ZEN_C.size, WL7.size)))


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_view_domain_is_the_public_sky_domain(atmo7, monkeypatch):
    """[0, 90] deg exactly: the horizon keeps the kernel clipping, anything
    below it fails before any kernel runs."""
    import skycolor.rt.solver as solver
    calls = {"n": 0}
    real_single = solver._single

    def spy_single(*a, **k):
        calls["n"] += 1
        return real_single(*a, **k)

    monkeypatch.setattr(solver, "_single", spy_single)
    ok = solve(atmo7, 60.0, np.array([0.0, 45.0, 90.0]), 90.0)
    assert ok.radiance.shape == (3, WL7.size)
    base = calls["n"]                       # the legal request ran once
    for bad in (np.array([90.0, 120.0]), np.array([-5.0]),
                np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError, match="view_zenith_deg"):
            solve(atmo7, 60.0, bad, 90.0)
    assert calls["n"] == base               # rejections ran zero kernels


# ---------------------------------------------------------------------------
# 10. review round 2: one single-term record, kernel-run grid sizes
# ---------------------------------------------------------------------------
def test_mc_single_only_records_its_settings(atmo7, monkeypatch):
    """spherical-mc with the higher orders off: the single term still runs,
    so its settings are recorded (options + effective) and the kernel
    receives exactly them; the zero-stderr compatibility stays."""
    import skycolor.rt.solver as solver
    seen = {}
    real_single = solver._single

    def spy_single(*a, **k):
        seen.update(n_gauss=k.get("n_gauss"), chunk=k.get("chunk"))
        return real_single(*a, **k)

    monkeypatch.setattr(solver, "_single", spy_single)
    res = solve(atmo7, 60.0, VZA, 90.0, approximation="spherical-mc",
                multiple=False, single_config=SingleConfig(n_gauss=2, chunk=64))
    assert seen == {"n_gauss": 2, "chunk": 64}
    assert res.options["n_gauss"] == 2 and res.options["chunk"] == 64
    assert res.effective["single"] == {"n_gauss": 2, "chunk": 64}
    assert res.options["mc_run"] is False
    assert np.all(res.stderr == 0.0)


@pytest.mark.slow(reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")
def test_mc_effective_grid_sizes_are_what_ran():
    """_coarse_grid clamps the angular sizes (min 2 each): effective
    reports the grid the engine built, consistent with coarse_zenith."""
    tiny = dict(n_zen=1, n_az=1, n_anchor=1, n_photons=1.0)
    res = solve(presets.clear(WL7), 60.0, VZA, 90.0,
                approximation="spherical-mc", mc_config=MCConfig(grid=tiny))
    mc_eff = res.effective["mc"]
    assert mc_eff["n_zen_run"] == 2 and mc_eff["n_az_run"] == 2
    assert mc_eff["n_zen_run"] * mc_eff["n_az_run"] == \
        len(res.diagnostics["coarse_zenith"])
    assert len(res.diagnostics["coarse_azimuth"]) == \
        len(res.diagnostics["coarse_zenith"])
    # requested values stay in info/diagnostics
    assert res.diagnostics["n_zen"] == 1 and res.diagnostics["n_az"] == 1


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_cloud_overlay_warns_for_field_even_without_higher_orders(atmo7):
    """The cloud two-stream is another field-blind component: an overlay
    over a single-only clear base still emits the one partial-support
    warning, while a plain single-only clear request stays silent."""
    from skycolor.aerosol import HinterlandAerosol, shared
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile, \
        RadialSplitField

    het = presets.clear(WL7)
    het.aerosols = [AerosolLayer(shared(HinterlandAerosol), aod550=0.05,
                                 profile=ExponentialProfile(1.5),
                                 horizontal=RadialSplitField(radius_km=20.0))]
    with pytest.warns(RuntimeWarning, match="cloud two-stream"):
        solve(het, 45.0, ZEN_C, 90.0,
              cloud_layer=cloud.get_preset("stratus"), multiple=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve(het, 60.0, VZA, 90.0, multiple=False)   # pure single: silent


# ---------------------------------------------------------------------------
# 11. review round 3: cloud observer at the model bottom, albedo record
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_cloud_requests_require_the_model_bottom_observer(atmo7, stratus):
    """Plan 3.4: the deck composition is a view from the model bottom,
    below the cloud base -- on every cloud path, whatever clear kernels
    the execution plan runs.  at_altitude stations still qualify."""
    raised = presets.clear(WL7)
    raised.elevation_km = 0.2          # below the 0.5 km base, off the bottom
    zeros = np.zeros((ZEN_C.size, WL7.size))
    for kwargs in (dict(multiple=False),                  # no SOS runs
                   dict(clear_radiance=zeros),            # external clear
                   dict(include_clear=False)):            # cloud-only
        with pytest.raises(ValueError, match="model bottom"):
            solve(raised, 45.0, ZEN_C, 90.0, cloud_layer=stratus, **kwargs)
        with pytest.raises(ValueError, match="model bottom"):
            sky_radiance_with_cloud(raised, stratus, 45.0, ZEN_C, 90.0,
                                    **({"clear_radiance": zeros}
                                       if "clear_radiance" in kwargs else {}))
    station = presets.clear(WL7).at_altitude(0.2)   # truncated == bottom
    res = solve(station, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                include_clear=False)
    assert np.all(res.components["clear"] == 0.0)


def test_external_clear_still_records_ground_albedo_override(atmo7, stratus):
    """The override record follows the deck kernel's inputs, not the clear
    field's provenance."""
    res = solve(atmo7, 45.0, ZEN_C, 90.0, cloud_layer=stratus,
                clear_radiance=np.zeros((ZEN_C.size, WL7.size)),
                ground_albedo=0.8)
    assert res.options["ground_albedo_override"] is True
    assert res.options["clear_radiance"] == "external"
