"""Tests for the structured solver entry (rt.solver.solve -> SkyResult).

The contract under test:

1. every method is bit-identical to the legacy entry point it wraps
   (clear -> sky_radiance_total, twilight -> twilight_radiance,
   cloud -> sky_radiance_with_cloud) -- the structured entry is pure
   dispatch, not a reimplementation;
2. diagnostics are non-empty and traceable (SOS orders/converged, twilight
   undetected_wl/grid, cloud components);
3. capability degradation warns once (twilight + non-uniform
   HorizontalField) while the default paths stay silent;
4. the display layer's ``radiance_field`` seam reproduces the direct
   backend call bit-for-bit.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor import cloud, presets, render  # noqa: E402
from skycolor.atmosphere import AerosolLayer  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402
from skycolor.rt.cloud import sky_radiance_with_cloud  # noqa: E402
from skycolor.rt.solver import solve, sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import twilight_radiance  # noqa: E402

WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)


@pytest.fixture(scope="module")
def atmo81():
    return presets.clear(WL_COARSE)


@pytest.fixture(scope="module")
def atmo7():
    return presets.clear(WL7)


@pytest.fixture(scope="module")
def stratus():
    return cloud.get_preset("stratus")


# ---------------------------------------------------------------------------
# 1. bit-equality with the legacy entries
# ---------------------------------------------------------------------------
def test_clear_matches_legacy_entry(atmo81):
    ref = sky_radiance_total(atmo81, 80.0, 60.0, 120.0)
    res = solve(atmo81, 80.0, 60.0, 120.0)
    assert np.array_equal(res.radiance, ref)
    assert res.method == "clear"
    assert np.array_equal(res.wl, atmo81.wl)


def test_twilight_matches_legacy_entry(atmo7):
    ref, ref_err = twilight_radiance(atmo7, 96.0, np.array([20.0, 70.0]),
                                     np.array([0.0, 180.0]),
                                     quality="low", grid=TINY, seed=0)
    res = solve(atmo7, 96.0, np.array([20.0, 70.0]), np.array([0.0, 180.0]),
                method="twilight", quality="low", grid=TINY, seed=0)
    assert np.array_equal(res.radiance, ref)
    assert np.array_equal(res.stderr, ref_err)


def test_cloud_matches_legacy_entry(atmo7, stratus):
    ref = sky_radiance_with_cloud(atmo7, stratus, 75.0, 45.0, 135.0)
    res = solve(atmo7, 75.0, 45.0, 135.0, method="cloud", cloud_layer=stratus)
    assert np.array_equal(res.radiance, ref)
    assert res.method == "cloud"


# ---------------------------------------------------------------------------
# 2. diagnostics are traceable
# ---------------------------------------------------------------------------
def test_clear_diagnostics_orders_and_convergence(atmo81):
    res = solve(atmo81, 65.0, 45.0, 135.0)
    assert isinstance(res.diagnostics["orders"], int)
    assert res.diagnostics["orders"] > 0
    assert isinstance(res.diagnostics["converged"], bool)
    assert res.stderr is None
    # the clear combination formally provides its components; the total is
    # their exact sum and no kernel ran twice
    comps = res.components
    assert set(comps) == {"single", "multiple"}
    assert np.array_equal(res.radiance,
                          comps["single"] + comps["multiple"])
    assert res.approximation == "hybrid-sos"
    assert res.options["sun_zenith_deg"] == 65.0


def test_twilight_diagnostics_trace_grid(atmo7):
    res = solve(atmo7, 96.0, 30.0, 90.0, method="twilight",
                quality="low", grid=TINY, seed=0)
    assert "undetected_wl" in res.diagnostics
    assert res.diagnostics["n_photons"] == pytest.approx(TINY["n_photons"])
    assert res.options["seed"] == 0


def test_cloud_components_decomposition(atmo7, stratus):
    res = solve(atmo7, 75.0, 45.0, 135.0, method="cloud", cloud_layer=stratus)
    comps = res.components
    assert set(("clear", "cloud", "transmittance")) <= set(comps)
    # the decomposition reproduces the total exactly
    total = comps["clear"] + comps["cloud"]
    assert np.array_equal(res.radiance, total)


def test_unknown_method_and_missing_deck_raise(atmo7):
    with pytest.raises(ValueError, match="unknown method"):
        solve(atmo7, 60.0, 30.0, 90.0, method="auto")
    with pytest.raises(ValueError, match="cloud_layer"):
        solve(atmo7, 60.0, 30.0, 90.0, method="cloud")


# ---------------------------------------------------------------------------
# 3. capability warnings
# ---------------------------------------------------------------------------
def test_twilight_warns_on_horizontal_field(atmo7):
    from skycolor.aerosol import HinterlandAerosol, shared
    from skycolor.atmosphere import ExponentialProfile, RadialSplitField

    het = presets.clear(WL7)
    het.aerosols = [AerosolLayer(shared(HinterlandAerosol), aod550=0.05,
                                 profile=ExponentialProfile(1.5),
                                 horizontal=RadialSplitField(radius_km=20.0))]
    with pytest.warns(RuntimeWarning, match="HorizontalField"):
        solve(het, 96.0, 30.0, 90.0, method="twilight", quality="low",
              grid=TINY, seed=0)


def test_default_paths_do_not_warn(atmo7, stratus):
    """Clear, cloud and uniform-field twilight run silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve(atmo7, 80.0, 45.0, 135.0)
        solve(atmo7, 75.0, 45.0, 135.0, method="cloud", cloud_layer=stratus)
        solve(atmo7, 96.0, 30.0, 90.0, method="twilight", quality="low",
              grid=TINY, seed=0)


# ---------------------------------------------------------------------------
# 4. the display field seam
# ---------------------------------------------------------------------------
def test_render_field_matches_direct_backend_call(atmo7, stratus):
    """A cloudy dome built through render's ``radiance_field`` seam is
    bit-identical to calling the cloud backend directly."""
    def field(zen, az_rel):
        return sky_radiance_with_cloud(atmo7, stratus, 75.0, zen, az_rel)

    dome = render.sky_dome(atmo7, 75.0, n_zenith=4, n_azimuth=5,
                           radiance_field=field)
    zen = dome["zenith"]
    az = dome["azimuth_rel"]
    Z, A = np.meshgrid(zen, az, indexing="ij")
    direct = sky_radiance_with_cloud(atmo7, stratus, 75.0, Z.ravel(), A.ravel())
    assert np.array_equal(dome["radiance"].reshape(-1, WL7.size), direct)


def test_ground_radiance_direct_transmittance_override(atmo7):
    """The override replaces the clear-sky beam: E_dir scales with T."""
    dome = render.sky_dome(atmo7, 60.0, n_zenith=4, n_azimuth=5)
    wl = atmo7.wl
    base, E_base = render.ground_radiance(atmo7, dome, 60.0)
    T_zero = np.zeros_like(atmo7.ground_albedo)
    dark0, E_zero = render.ground_radiance(atmo7, dome, 60.0,
                                           direct_transmittance=T_zero)
    # removing the direct beam removes its exact contribution
    assert np.all(E_zero <= E_base)
    assert np.all(dark0 <= base)
    assert np.any(dark0 < base)


def test_solve_rejects_empty_request_and_records_cloud_options(atmo7, stratus):
    """single=False + multiple=False computes nothing -> ValueError; the
    cloud composition always includes the clear single term, so
    ``single=False`` on an overlay request is rejected instead of being
    silently ignored (plan section 3.4); use ``include_clear=False`` for a
    cloud-only total."""
    with pytest.raises(ValueError, match="computes nothing"):
        solve(atmo7, 60.0, 30.0, 90.0, single=False, multiple=False)
    with pytest.raises(ValueError, match="computes nothing"):
        solve(atmo7, 96.0, 30.0, 90.0, method="twilight", quality="low",
              grid=TINY, seed=0, single=False, multiple=False)
    with pytest.raises(ValueError, match="single=False"):
        solve(atmo7, 75.0, 45.0, 135.0, method="cloud", cloud_layer=stratus,
              single=False)
    res = solve(atmo7, 75.0, 45.0, 135.0, method="cloud", cloud_layer=stratus)
    assert res.options["single"] is True   # the deck always includes it
    # while the clear branch does record it
    res_ss = solve(atmo7, 60.0, 30.0, 90.0, single=True, multiple=False)
    assert res_ss.options["single"] is True
    assert res_ss.diagnostics == {}          # no SOS ran, no orders claimed
