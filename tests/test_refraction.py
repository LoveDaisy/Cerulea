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


@pytest.mark.slow(reason="cost: 12-worker parallel 11.7 s (cold-optics listing); fast-tier budget 2 s")
def test_apparent_elevation_monotonic_and_signed(atmo):
    t = RefractionTables(atmo)
    true = np.array([0.0, 0.5, 1.0, 5.0, 20.0, 45.0, 90.0])
    app = t.apparent_elevation(true)
    assert np.all(np.diff(app) > 0)
    assert np.all(app >= true - 1e-9)
    assert 0.3 < app[0] < 0.9          # horizon: ~0.5 deg
    assert app[-1] == pytest.approx(90.0, abs=1e-6)


@pytest.mark.slow(reason="cost: 12-worker parallel 5.3 s (cold-optics listing); fast-tier budget 2 s")
def test_shadow_threshold(atmo):
    t = RefractionTables(atmo)
    thr = t.shadow_zenith_true(np.array([0.0, 10.0, 30.0, 60.0]))
    geometric = 90.0 + t.horizon_dip(np.array([0.0, 10.0, 30.0, 60.0]))
    ext = thr - geometric
    assert 0.3 < ext[0] < 0.8                 # surface: ~0.55 deg
    assert np.all(ext > 0)
    assert np.all(ext < 1.5)
    assert np.all(np.diff(thr) > 0)


@pytest.mark.slow(reason="cost: 12-worker parallel 5.8 s (cold-optics listing); fast-tier budget 2 s")
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


def test_refracted_path_below_horizon_includes_both_branches():
    """The n != 1 helper must include the near-side perigee -> point segment."""
    from skycolor.constants import R_EARTH
    from skycolor.refraction import sun_path_lengths

    rb = np.array([R_EARTH, R_EARTH + 2000, R_EARTH + 5000,
                   R_EARTH + 20000, R_EARTH + 60000, R_EARTH + 120000.0])
    n_layer = np.array([1.0003, 1.0003, 1.0002, 1.0001, 1.0])
    r = R_EARTH + 40000.0
    elev = np.radians(-6.0)
    c = np.sin(elev)                       # mu_app = cos(apparent zenith)
    a = n_layer[3] * r * np.cos(elev)      # point is in shell 3
    L = sun_path_lengths(np.array([a]), np.array([c]), np.array([r]),
                         rb, n_layer)[:, 0]

    b = a / n_layer
    s_lo = np.sqrt(np.maximum(rb[:-1]**2 - b**2, 0.0))
    s_hi = np.sqrt(np.maximum(rb[1:]**2 - b**2, 0.0))
    u = r * abs(c)
    expected = (s_hi - s_lo) + (np.minimum(s_hi, u) - np.minimum(s_lo, u))
    assert np.allclose(L, expected, rtol=1e-12)


@pytest.mark.slow(reason="cost: 12-worker parallel 8.9 s (cold-optics listing); fast-tier budget 2 s")
def test_refracted_path_matches_ray_trace():
    """Independent RK4 column integral along the refracted ray."""
    from skycolor import presets
    from skycolor.constants import R_EARTH
    from skycolor.refraction import (
        _Profile, geometry, refractivity_of, sun_path_lengths,
    )

    atmo = presets.clear([550.0])
    prof = _Profile(atmo)
    rb = atmo.r_bounds
    zmid = (np.sqrt(rb[:-1] * rb[1:]) - R_EARTH) / 1e3
    n_layer = 1.0 + refractivity_of(atmo, zmid)

    def column(r0, theta0, ds=1000.0):
        y = np.array([0.0, r0])
        u = np.array([np.sin(theta0), np.cos(theta0)])
        u /= np.linalg.norm(u)
        col = 0.0

        def accel(pos, d):
            r = np.linalg.norm(pos)
            n = prof.n_of(r)
            g = prof.dndr_of(r) / n * (pos / r)
            return d, g - d * np.dot(d, g)

        for _ in range(400000):
            if np.linalg.norm(y) >= R_EARTH + 1.2e6:
                break
            k1y, k1u = accel(y, u)
            k2y, k2u = accel(y + 0.5 * ds * k1y, u + 0.5 * ds * k1u)
            k3y, k3u = accel(y + 0.5 * ds * k2y, u + 0.5 * ds * k2u)
            k4y, k4u = accel(y + ds * k3y, u + ds * k3u)
            y = y + ds / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            un = u + ds / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            un /= np.linalg.norm(un)
            z1 = (np.linalg.norm(y) - R_EARTH) / 1e3
            z2 = (np.linalg.norm(y - ds * u) - R_EARTH) / 1e3
            col += 0.5 * (atmo.n_air(np.array([z1]))[0]
                          + atmo.n_air(np.array([z2]))[0]) * ds
            u = un
        return col

    for z in (38.0, 47.0, 60.0):
        r0 = R_EARTH + z * 1e3
        e_app = float(geometry().apparent_elevation(
            np.array([-6.0]), np.array([z]))[0])
        ref = column(r0, np.pi / 2 - np.radians(e_app))
        a = (1.0 + refractivity_of(atmo, z)) * r0 * np.cos(np.radians(e_app))
        mu = np.sin(np.radians(e_app))
        L = sun_path_lengths(np.array([a]), np.array([mu]), np.array([r0]),
                             rb, n_layer)[:, 0]
        code = float(np.sum(L * atmo.n_air(zmid)))
        assert code == pytest.approx(ref, rel=0.03), (z, code, ref)


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


@pytest.mark.slow(reason="cost: 12-worker parallel 4.7 s (cold-optics listing); fast-tier budget 2 s")
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


def test_restrict_preserves_custom_air_profile():
    """A custom air profile injected as data must survive the wavelength
    subset the twilight engine takes (historically: subclass overrides,
    which a rebuild silently dropped)."""
    from skycolor.atmosphere import Atmosphere
    from skycolor.constants import WL_COARSE

    base = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[])

    def cold_air(z):
        zz = np.asarray(z, dtype=float)
        return base.n_air(z) * np.exp(-np.maximum(zz - 20.0, 0.0) * (1 / 6.0 - 1 / 6.5))

    at = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[], air_profile=cold_air)
    sub = at._restrict([0, 1, 2])
    assert sub.wl.size == 3
    # the injected profile survives the subset, unchanged
    assert np.array_equal(sub.n_air(np.array([40.0])), at.n_air(np.array([40.0])))
    # ... and still differs from the US76 default
    plain = Atmosphere(WL_COARSE[:3], ozone_du=300.0, aerosols=[])
    assert not np.array_equal(sub.n_air(np.array([40.0])), plain.n_air(np.array([40.0])))


@pytest.mark.slow(reason="cost: 12-worker parallel 15.8 s (cold-optics listing); fast-tier budget 2 s")
def test_refraction_cache_identity():
    """Cache identity is the medium's density definition, not a proxy.

    (a) media with different custom profiles but the same surface pressure /
        observer altitude must NOT share tables (the old two-number proxy
        key silently conflated them);
    (b) a wavelength-restricted medium shares its source's tables (the
        density definition is untouched by a spectral subset), while two
        separately constructed identical media each own their tables.
    """
    from skycolor.atmosphere import Atmosphere
    from skycolor.constants import WL_COARSE
    from skycolor.refraction import get_tables

    base = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[])

    def profile_a(z):
        return base.n_air(z) * 1.0

    def profile_b(z):
        return base.n_air(z) * (1.0 + 1e-3)

    a = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[],
                   surface_pressure=101325.0, elevation_km=0.0,
                   air_profile=profile_a)
    b = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[],
                   surface_pressure=101325.0, elevation_km=0.0,
                   air_profile=profile_b)
    # (a) different density definitions, same proxy numbers -> no sharing
    assert get_tables(a) is not get_tables(b)
    # (b1) restrict inherits the source token
    assert get_tables(a._restrict([0, 1, 2])) is get_tables(a)
    # (b2) same definition, separately constructed -> own tables by design
    a2 = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[],
                    surface_pressure=101325.0, elevation_km=0.0,
                    air_profile=profile_a)
    assert get_tables(a2) is not get_tables(a)
    # ... and the rebuilt tables agree with the first ones
    assert np.array_equal(get_tables(a2).shadow_zenith_true(5.0),
                          get_tables(a).shadow_zenith_true(5.0))
