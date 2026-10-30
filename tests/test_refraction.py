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


def test_refracted_path_length_horizontal(atmo):
    """A horizontal ray at the surface traverses ~1660 km of atmosphere."""
    from skycolor.constants import R_EARTH
    from skycolor.refraction import refractivity_of, sun_path_lengths

    rb = atmo.r_bounds
    n_layer = 1.0 + refractivity_of(atmo, atmo.z_mid)
    a = (1.0 + refractivity_of(atmo, 0.0)) * R_EARTH
    L = sun_path_lengths(np.array([a]), np.array([0.0]), np.array([R_EARTH]),
                         rb, n_layer)
    total = float(L.sum())
    assert 1.55e6 < total < 1.75e6, total


def test_refraction_geometry_lookup():
    from skycolor.refraction import geometry

    g = geometry()
    app = g.apparent_elevation(np.array([0.0, 1.0, 5.0, 45.0]),
                               np.zeros(4))
    assert 0.3 < app[0] < 0.75          # horizon: ~0.5 deg
    assert np.all(np.diff(app) > 0)
    assert app[3] == pytest.approx(45.0, abs=0.1)


def test_delta_m_identity_for_small_particles():
    from skycolor.atmosphere import Atmosphere
    from skycolor.constants import WL_COARSE

    at = Atmosphere(WL_COARSE, aerosols=[])       # Rayleigh only
    beta, f = at.delta_m_moments(16)
    assert np.all(f < 1e-6)
    raw = at.phase_moments(16)
    assert np.allclose(beta, raw, atol=1e-6)


def test_delta_m_scales_coarse_aerosol():
    from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile
    from skycolor.aerosol import DesertDustAerosol
    from skycolor.constants import WL_COARSE

    at = Atmosphere(WL_COARSE, aerosols=[AerosolLayer(DesertDustAerosol(), 1.0,
                                                      ExponentialProfile(2.0))])
    props = at.delta_m_properties(32)
    f = props["delta_m_f"]
    assert f.max() > 0.2                     # dust loses a lot of forward peak
    assert np.all(props["beta_ext"] <= at.layer_properties()["beta_ext"] + 1e-12)
    assert np.allclose(props["moments"][:, 0, :], 1.0)


def test_restrict_preserves_subclass_overrides():
    """`_restrict` must keep subclass profiles: rebuilding silently drops them."""
    from skycolor.atmosphere import Atmosphere
    from skycolor.constants import WL_COARSE

    class Cold(Atmosphere):
        def n_air(self, z):
            zz = np.asarray(z, dtype=float)
            return super().n_air(z) * np.exp(-np.maximum(zz - 20.0, 0.0) * (1 / 6.0 - 1 / 6.5))

    at = Cold(WL_COARSE, ozone_du=300.0, aerosols=[])
    sub = at._restrict([0, 1, 2])
    assert isinstance(sub, Cold)
    assert sub.wl.size == 3
    assert sub.n_air(np.array([40.0]))[0] != at.n_air(np.array([40.0]))[0] * 0 + \
        Atmosphere(WL_COARSE[:3], ozone_du=300.0, aerosols=[]).n_air(np.array([40.0]))[0]
