"""Golden bit-exact regression fixtures for the architecture refactor.

Every structural migration step of task-architecture-refactor must reproduce
these arrays with ``np.array_equal`` (equivalence criterion A of the task
plan: pure structural wrapping must not change any bit).  The fixtures were
generated from the pre-refactor baseline by
``tests/data/make_golden_fixtures.py``; regenerate only with an explicit,
recorded reason in the task progress log -- never to make a failing
comparison pass.

All scenarios are pinned at ``elevation_km=0`` so the structural criterion
stays decoupled from the altitude-semantics change (plan Step 3c), which is
verified separately against the HANDOFF z0=2.4 km numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skycolor import cloud, presets, render
from skycolor.atmosphere import Atmosphere
from skycolor.constants import WL_COARSE
from skycolor.rt import multiple
from skycolor.rt.cloud import sky_radiance_with_cloud
from skycolor.rt.mc import mc_radiance_split
from skycolor.rt.solver import sky_radiance_total
from skycolor.rt.twilight import twilight_radiance

DATA = Path(__file__).parent / "data"
WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)


def _load(name: str) -> dict:
    path = DATA / f"golden_{name}.npz"
    if not path.exists():
        pytest.fail(
            f"golden fixture {path.name} is missing; regenerate with "
            "`uv run python tests/data/make_golden_fixtures.py` -- but only "
            "with an explicit, recorded reason (see the module docstring)")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def clear81():
    return presets.clear(WL_COARSE)


@pytest.fixture(scope="module")
def clear7():
    return presets.clear(WL7)


@pytest.fixture(scope="module")
def stratus():
    return cloud.get_preset("stratus")


def test_clear_direction(clear81):
    golden = _load("clear_direction")
    rad = sky_radiance_total(clear81, 80.0, 60.0, 120.0)
    assert rad.shape == golden["radiance"].shape
    assert np.array_equal(rad, golden["radiance"])
    assert np.array_equal(clear81.wl, golden["wl"])


def test_sos_diagnostics(clear81):
    """Pin the full SOSResult, including the diagnostics that SkyResult
    (plan Step 4) must surface without loss."""
    golden = _load("sos")
    res = multiple.solve(clear81, 65.0, np.array([0.0, 45.0, 85.0]),
                         np.array([0.0, 90.0, 180.0]), n_mu=6,
                         single_view=True)
    for key in ("radiance", "radiance_single", "flux_up_toa", "flux_down_boa",
                "flux_up_boa", "direct_down_boa", "incident_flux"):
        assert np.array_equal(getattr(res, key), golden[key]), key
    assert res.orders == int(golden["orders"])
    assert res.converged == bool(golden["converged"])
    assert np.array_equal(clear81.wl, golden["wl"])


def test_dome(clear7):
    golden = _load("dome")
    dome = render.sky_dome(clear7, 78.0, n_zenith=4, n_azimuth=5)
    assert np.array_equal(dome["radiance"], golden["radiance"])
    assert np.array_equal(dome["zenith"], golden["zenith"])
    assert np.array_equal(dome["azimuth_rel"], golden["azimuth_rel"])
    assert np.array_equal(clear7.wl, golden["wl"])


def test_cloud(clear7, stratus):
    golden = _load("cloud")
    atmo = Atmosphere(WL7, ozone_du=300.0, aerosols=[])
    total, comps = sky_radiance_with_cloud(atmo, stratus, 75.0, 45.0, 135.0,
                                           return_components=True)
    assert np.array_equal(total, golden["radiance"])
    assert np.array_equal(comps["clear"], golden["clear"])
    assert np.array_equal(comps["cloud"], golden["cloud"])
    assert np.array_equal(comps["transmittance"], golden["transmittance"])
    assert np.array_equal(atmo.wl, golden["wl"])


def test_twilight(clear7):
    golden = _load("twilight")
    rad, err = twilight_radiance(clear7, 96.0, np.array([0.0, 30.0, 70.0]),
                                 np.array([0.0, 90.0, 180.0]),
                                 quality="low", grid=TINY, seed=0)
    assert np.array_equal(rad, golden["radiance"])
    assert np.array_equal(err, golden["stderr"])
    assert np.array_equal(clear7.wl, golden["wl"])


def test_mc_split(clear7):
    golden = _load("mc")
    res = mc_radiance_split(clear7, 85.0, np.array([30.0, 70.0]),
                            np.array([90.0, 180.0]), n_photons=600_000, seed=0)
    for key in ("total", "single", "multiple", "total_err"):
        assert np.array_equal(res[key], golden[key]), key
    assert np.array_equal(clear7.wl, golden["wl"])


def test_altitude_truncated_baseline(clear7):
    """Step 3c truncation semantics, pinned as a NEW-semantics baseline:
    generated after the at_altitude fix and never compared with the
    elevation=0 fixtures.  The observer radius must coincide with the first
    shell boundary (single integrates from it, SOS/MC ground is z_bounds[0])."""
    golden = _load("altitude")
    atmo = clear7.at_altitude(2.4)
    rad = sky_radiance_total(atmo, 85.0, 45.0, 135.0)
    assert atmo.z_bounds[0] == pytest.approx(2.4)
    np.testing.assert_array_equal(atmo.z_bounds, golden["z_bounds"])
    assert np.array_equal(rad, golden["radiance"])
    assert np.array_equal(atmo.wl, golden["wl"])
