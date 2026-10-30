import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere
from skycolor.constants import R_EARTH, WL_COARSE
from skycolor.refraction import RefractionTables, trace_ray


@pytest.fixture(scope="module")
def atmo():
    return Atmosphere(WL_COARSE, aerosols=[])


def saemundsson(elev_deg):
    h = elev_deg
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))


def test_horizon_refraction(atmo):
    """The integrated bending at the horizon is about half a degree."""
    r = trace_ray(atmo, R_EARTH * (1 + 1e-9), np.pi / 2, ds=1000.0)
    minutes = np.degrees(r["bending"]) * 60
    assert 28.0 < minutes < 36.0, minutes
    assert not r["hit_earth"]


@pytest.mark.parametrize("elev", [2.0, 5.0, 10.0, 30.0, 45.0, 80.0])
def test_refraction_matches_saemundsson(atmo, elev):
    r = trace_ray(atmo, R_EARTH * (1 + 1e-9), np.pi / 2 - np.radians(elev), ds=1000.0)
    model = np.degrees(r["bending"]) * 60
    ref = saemundsson(elev)
    assert abs(model - ref) / ref < 0.10, (elev, model, ref)


def test_apparent_elevation_monotonic_and_signed(atmo):
    t = RefractionTables(atmo)
    true = np.array([0.0, 0.5, 1.0, 5.0, 20.0, 45.0, 90.0])
    app = t.apparent_elevation(true)
    assert np.all(np.diff(app) > 0)
    assert np.all(app >= true - 1e-9)
    assert 0.3 < app[0] < 0.9          # horizon: ~0.5 deg
    assert app[-1] == pytest.approx(90.0, abs=1e-6)


def test_shadow_threshold(atmo):
    t = RefractionTables(atmo)
    thr = t.shadow_zenith_true(np.array([0.0, 10.0, 30.0, 60.0]))
    geometric = 90.0 + t.horizon_dip(np.array([0.0, 10.0, 30.0, 60.0]))
    ext = thr - geometric
    assert 0.3 < ext[0] < 0.8                 # surface: ~0.55 deg
    assert np.all(ext > 0)
    assert np.all(ext < 1.5)
    assert np.all(np.diff(thr) > 0)


def test_horizon_dip():
    t = RefractionTables(Atmosphere(WL_COARSE, aerosols=[]))
    assert t.horizon_dip(np.array([0.0]))[0] == pytest.approx(0.0)
    # 1 km -> 1.02 deg, 10 km -> 3.2 deg
    assert t.horizon_dip(np.array([1.0]))[0] == pytest.approx(1.02, abs=0.02)
    assert t.horizon_dip(np.array([10.0]))[0] == pytest.approx(3.21, abs=0.05)
