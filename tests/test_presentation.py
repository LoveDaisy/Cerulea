"""The unified presentation layer over one solver result (task plan A1-A7).

Contract under test, by milestone:

1. the one solar input contract: ``resolve_sun`` returns the
   pre-refraction direction, converts aware times to naive UTC before the
   ephemeris reads them, and rejects conflicting/invalid inputs;
2. one solve per request: ``solve_dome`` -> ``SkyPresentation`` binds the
   grid, the solar display geometry and the matching ground illumination at
   preparation time; ``presentation_from_result`` only reshapes/binds a
   ready result; after preparation every output (chart tables, panorama,
   fisheye, perspective) is a pure function of the bound arrays -- solves
   and solar-source queries are forbidden and spied to be absent;
3. scene-consistent lighting: clear/cloud ground direct irradiance bound at
   preparation, the cloud helper living in exactly one place, external
   fields refusing unmatched ground lighting, albedo and wavelength
   snapshots immune to later facade mutation, and the decorative sun disk
   consuming the adopted source's apparent geometry;
4. (CLI-level: see ``tests/test_cli.py``).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor.scene import resolve_sun  # noqa: E402

BEIJING = (39.9, 116.4)


# ---------------------------------------------------------------------------
# 1. the one solar input contract (M1)
# ---------------------------------------------------------------------------
def test_naive_utc_z_and_offset_are_the_same_instant():
    """Naive means UTC; 'Z' and an explicit offset convert before the
    ephemeris reads the clock fields -- the same physical instant gives
    bit-identical angles."""
    naive = dt.datetime(2026, 11, 5, 0, 0)
    z = resolve_sun(time="2026-11-05T00:00:00Z", latitude=BEIJING[0],
                    longitude=BEIJING[1])
    offset = resolve_sun(time="2026-11-05T08:00:00+08:00",
                         latitude=BEIJING[0], longitude=BEIJING[1])
    aware = resolve_sun(time=naive.replace(tzinfo=dt.timezone.utc),
                        latitude=BEIJING[0], longitude=BEIJING[1])
    plain = resolve_sun(time=naive, latitude=BEIJING[0], longitude=BEIJING[1])
    assert z == offset == aware == plain


def test_time_resolution_is_pre_refraction():
    """The resolved direction is the ``refraction=False`` ephemeris angle;
    the retired ``refraction=True`` path visibly double-refracted."""
    from skycolor.geometry import sun_position

    naive = dt.datetime(2026, 11, 5, 0, 0)
    az, el = resolve_sun(time=naive, latitude=BEIJING[0], longitude=BEIJING[1])
    assert az == sun_position(BEIJING[0], BEIJING[1], naive, refraction=False)[0]
    assert el == sun_position(BEIJING[0], BEIJING[1], naive, refraction=False)[1]
    _, el_refracted = sun_position(BEIJING[0], BEIJING[1], naive, refraction=True)
    assert abs(el_refracted - el) > 0.01   # the semantic gap is real, not a rewording


def test_ephemeris_is_called_without_refraction():
    """Spy: resolve_sun must request the geometric (pre-refraction) angle."""
    import skycolor.scene as scene

    seen = {}
    real = scene.sun_position

    def spy(lat, lon, when, refraction=False):
        seen["refraction"] = refraction
        seen["when"] = when
        return real(lat, lon, when, refraction=refraction)

    seen_ref = scene.sun_position
    scene.sun_position = spy
    try:
        resolve_sun(time="2026-11-05T12:00:00+00:00", latitude=0.0,
                    longitude=0.0)
    finally:
        scene.sun_position = seen_ref
    assert seen["refraction"] is False
    assert seen["when"].tzinfo is None           # naive UTC reaches the ephemeris


def test_time_and_numbers_conflict():
    with pytest.raises(ValueError, match="conflict"):
        resolve_sun(sun_elevation=10.0, time="2026-11-05T00:00:00Z",
                    latitude=0.0, longitude=0.0)
    with pytest.raises(ValueError, match="conflict"):
        resolve_sun(sun_azimuth=137.0, time="2026-11-05T00:00:00Z",
                    latitude=0.0, longitude=0.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="ISO 8601"):
        resolve_sun(time="not-a-time", latitude=0.0, longitude=0.0)
    with pytest.raises(ValueError, match="latitude and longitude"):
        resolve_sun(time="2026-11-05T00:00:00Z")
    with pytest.raises(ValueError, match="latitude"):
        resolve_sun(time="2026-11-05T00:00:00Z", latitude=91.0, longitude=0.0)
    with pytest.raises(ValueError, match="longitude"):
        resolve_sun(time="2026-11-05T00:00:00Z", latitude=0.0, longitude=181.0)
    with pytest.raises(ValueError, match="finite"):
        resolve_sun(sun_elevation=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        resolve_sun(sun_azimuth=float("inf"))
    with pytest.raises(ValueError, match="sun_elevation"):
        resolve_sun(sun_elevation=90.5)


def test_numeric_defaults_and_azimuth_wrap():
    """Without time the historical CLI defaults fill in -- once, here, not
    in argparse defaults that would masquerade as explicit conflicts."""
    assert resolve_sun() == (180.0, 10.0)
    assert resolve_sun(sun_elevation=0.0) == (180.0, 0.0)
    az, el = resolve_sun(sun_elevation=-12.0, sun_azimuth=-180.0)
    assert az == 180.0 and el == -12.0
    az, _ = resolve_sun(sun_azimuth=450.0)
    assert az == 90.0


# ---------------------------------------------------------------------------
# 2. one solve, then pure display (M2)
# ---------------------------------------------------------------------------
WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
N_ZEN, N_AZ = 5, 7


@pytest.fixture(scope="module")
def atmo7():
    from skycolor import presets

    return presets.clear(WL7)


def _forbid_solve_and_source(monkeypatch):
    """Replace every solve path and the solar source with raisers: after
    preparation, displays must not re-enter any of them."""
    import skycolor.rt.solver as solver
    import skycolor.rt.multiple as multiple
    import skycolor.rt.twilight as twilight
    import skycolor.solar_source as solar_source

    def boom(name):
        def _boom(*a, **k):
            raise AssertionError(f"{name} must not run after preparation")
        return _boom

    monkeypatch.setattr(solver, "solve", boom("solve"))
    monkeypatch.setattr(solver, "_single", boom("rt.single"))
    monkeypatch.setattr(multiple, "solve", boom("rt.multiple"))
    monkeypatch.setattr(twilight, "multiple_scattered", boom("rt.twilight"))
    monkeypatch.setattr(solar_source, "SolarSource", boom("SolarSource"))


def test_prepare_once_then_display_without_solve_or_source(atmo7, monkeypatch):
    """A4: after ``solve_dome`` preparation, chart tables and all three
    cameras render repeatedly from the bound arrays alone."""
    from skycolor import render

    pres = render.solve_dome(atmo7, 20.0, 30.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    _forbid_solve_and_source(monkeypatch)
    for _ in range(2):
        rows = render.chart_table(pres)
        assert len(rows) == len(render.DEFAULT_CHART_ROWS)
        assert render.render_result(pres, camera="panorama", size=(24, 12),
                                    ground=False).shape == (12, 24, 3)
        assert render.render_result(pres, camera="fisheye", size=(20,),
                                    ground=False).shape == (20, 20, 3)
        assert render.render_result(pres, camera="perspective", size=(24, 16),
                                    ground=True).shape == (16, 24, 3)


def test_solve_dome_sends_absolute_azimuths_to_one_solve(atmo7, monkeypatch):
    """The relative grid plus the same solar azimuth goes to exactly one
    solve call."""
    import skycolor.rt.solver as solver
    from skycolor import render

    seen = {}
    real = solver.solve

    def spy(atmo, sun_zen, vzen, vaz, **k):
        seen["sun_zen"] = sun_zen
        seen["vzen"] = np.array(vzen)
        seen["vaz"] = np.array(vaz)
        seen["sun_az"] = k.get("sun_azimuth_deg")
        return real(atmo, sun_zen, vzen, vaz, **k)

    monkeypatch.setattr(solver, "solve", spy)
    pres = render.solve_dome(atmo7, 20.0, 137.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    zen = np.linspace(0.0, 90.0, N_ZEN)
    az = np.linspace(0.0, 180.0, N_AZ)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    assert seen["sun_zen"] == 70.0
    assert seen["sun_az"] == 137.0
    np.testing.assert_array_equal(seen["vzen"], Z.ravel())
    np.testing.assert_allclose(seen["vaz"], A.ravel() + 137.0, rtol=0, atol=0)
    assert pres.radiance.shape == (N_ZEN, N_AZ, WL7.size)


def test_joint_rotation_leaves_the_sky_invariant(atmo7):
    """Rotating the sun and the grid together rotates the physics, not the
    numbers.  Tolerance source: the joint rotation changes only the order of
    trigonometric evaluations in the kernels (last-bit rounding), so a pure
    relative tolerance at 1e-10 covers rounding, not physics (atol=0)."""
    from skycolor import render

    p0 = render.solve_dome(atmo7, 20.0, 0.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    p1 = render.solve_dome(atmo7, 20.0, 137.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    np.testing.assert_allclose(p1.radiance, p0.radiance, rtol=1e-10, atol=0)


def test_panorama_brightness_follows_absolute_sun_azimuth(atmo7):
    """Camera rotation with a non-zero sun azimuth: the brightest horizon
    column sits at the sun's absolute azimuth, not at the grid origin."""
    from skycolor import render

    pres = render.solve_dome(atmo7, 20.0, 137.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    img = render.render_result(pres, camera="panorama", size=(72, 8),
                               ground=False, sun_disk=False)
    row = img[-1].astype(float)
    col = int(np.argmax(row.mean(axis=1)))
    expected = 137.0 / 360.0 * 72.0                    # 27.4 -> pixel 27/28
    assert abs(col - expected) <= 1.5


def test_presentation_from_result_is_pure_binding(atmo7, monkeypatch):
    """An explicit result + grid binds without any solve or source query."""
    from skycolor import render
    from skycolor.rt.solver import solve

    zen = np.linspace(0.0, 90.0, N_ZEN)
    az = np.linspace(0.0, 180.0, N_AZ)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel(), sun_azimuth_deg=50.0)
    _forbid_solve_and_source(monkeypatch)
    pres = render.presentation_from_result(result, zen, az, 20.0, 50.0,
                                           ground_albedo=0.1,
                                           ground_direct_irradiance=np.zeros(WL7.size))
    assert render.render_result(pres, camera="panorama", size=(16, 8),
                                ground=True).shape == (8, 16, 3)


def test_presentation_from_result_rejects_mismatch(atmo7):
    """Grid rows, wavelength count and the recorded solar direction must
    match the result; nothing is silently reshaped or re-solved."""
    from skycolor import render
    from skycolor.rt.solver import solve

    zen = np.linspace(0.0, 90.0, N_ZEN)
    az = np.linspace(0.0, 180.0, N_AZ)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel(), sun_azimuth_deg=50.0)
    with pytest.raises(ValueError, match="view rows"):
        render.presentation_from_result(result, zen[:-1], az, 20.0, 50.0)
    with pytest.raises(ValueError, match="grid"):
        render.presentation_from_result(result, zen, az[:-1], 20.0, 50.0)
    with pytest.raises(ValueError, match="zenith"):
        # solved at sun zenith 70 (elevation 20); binding 21 misrepresents it
        render.presentation_from_result(result, zen, az, 21.0, 50.0)
    with pytest.raises(ValueError, match="azimuth"):
        render.presentation_from_result(result, zen, az, 20.0, 51.0)


def test_external_field_presentation_and_ground_rules(atmo7):
    """The seam keeps its convention (relative azimuths, one sampling); the
    ground display requires matching lighting and refuses both overrides."""
    from skycolor import render

    def field(zen, az_rel):
        return np.zeros((np.asarray(zen).size, WL7.size))

    zen = np.linspace(0.0, 90.0, N_ZEN)
    az = np.linspace(0.0, 180.0, N_AZ)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    pres = render.presentation_from_field(field, atmo7, 20.0, 0.0,
                                          n_zenith=N_ZEN, n_azimuth=N_AZ)
    assert pres.provenance == "external"
    assert pres.radiance.shape == (N_ZEN, N_AZ, WL7.size)
    assert np.array_equal(pres.radiance.reshape(-1, WL7.size), field(Z.ravel(), A.ravel()))
    # ground without lighting: refused, not silently defaulted
    with pytest.raises(ValueError, match="ground_direct_irradiance"):
        render.render_result(pres, camera="panorama", size=(8, 4), ground=True)
    # both overrides at once: refused
    pres_lit = render.presentation_from_field(
        field, atmo7, 20.0, 0.0, n_zenith=N_ZEN, n_azimuth=N_AZ,
        ground_direct_irradiance=np.zeros(WL7.size))
    with pytest.raises(ValueError, match="not both"):
        render.render_result(pres_lit, size=(8, 4), ground=True,
                             ground_direct_transmittance=np.zeros(WL7.size))
    # lit or sky-only both render
    assert render.render_result(pres_lit, size=(8, 4), ground=True).shape == (4, 8, 3)
    assert render.render_result(pres, size=(8, 4), ground=False).shape == (4, 8, 3)


def test_sky_dome_thin_forward_is_bit_identical(atmo7):
    """The dict path rides the same solve: bit-identical to the historical
    ndarray wrapper's composition."""
    from skycolor import render
    from skycolor.rt.solver import sky_radiance_total

    zen = np.linspace(0.0, 90.0, N_ZEN)
    az = np.linspace(0.0, 180.0, N_AZ)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    direct = sky_radiance_total(atmo7, 70.0, Z.ravel(), A.ravel()).reshape(
        N_ZEN, N_AZ, WL7.size)
    dome = render.sky_dome(atmo7, 70.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    assert set(dome) == {"zenith", "azimuth_rel", "radiance", "wl",
                         "sun_zenith_deg"}
    assert np.array_equal(dome["radiance"], direct)
    np.testing.assert_array_equal(dome["zenith"], zen)
    np.testing.assert_array_equal(dome["azimuth_rel"], az)
    assert dome["sun_zenith_deg"] == 70.0


def test_sky_radiance_grid_passes_new_options_through(atmo7):
    """The keyword-only solver options reach the entry; the default request
    stays the historical one."""
    from skycolor import render
    from skycolor.rt.single import sky_radiance as single_kernel
    from skycolor.rt.solver import sky_radiance_total

    zen = np.array([0.0, 45.0, 90.0])
    az = np.array([0.0, 90.0, 180.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    grid = render.sky_radiance_grid(atmo7, 60.0, Z, A)
    direct = sky_radiance_total(atmo7, 60.0, Z.ravel(), A.ravel()).reshape(
        3, 3, WL7.size)
    assert np.array_equal(grid, direct)
    single_only = render.sky_radiance_grid(atmo7, 60.0, Z, A,
                                           approximation="spherical-single")
    ref = single_kernel(atmo7, 60.0, Z.ravel(), A.ravel()).reshape(3, 3, WL7.size)
    assert np.array_equal(single_only, ref)


def test_render_scene_is_prepare_then_render_result(atmo7):
    """render_scene keeps its ndarray contract and adds no second display
    implementation: identical pixels to solve_dome + render_result."""
    from skycolor import render

    img = render.render_scene(atmo7, 20.0, 30.0, camera="perspective",
                              size=(24, 16), sun_disk=False, ground=True,
                              n_zenith=N_ZEN, n_azimuth=N_AZ)
    pres = render.solve_dome(atmo7, 20.0, 30.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    ref = render.render_result(pres, camera="perspective", size=(24, 16),
                               sun_disk=False, ground=True)
    assert img.shape == (16, 24, 3)
    assert np.array_equal(img, ref)


# ---------------------------------------------------------------------------
# 3. scene-consistent lighting and the one cloud helper (M3)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stratus():
    from skycolor import cloud

    return cloud.get_preset("stratus")


@pytest.fixture(scope="module")
def cirrus():
    from skycolor import cloud

    return cloud.get_preset("cirrus")


def _reference_deck_ground(atmo, layer, sun_zenith_deg):
    """Verbatim pre-move implementation of the adopted prescription (the
    array-equality anchor for the helper relocation)."""
    from skycolor import spectra
    from skycolor.constants import R_EARTH
    from skycolor.solar_source import SolarSource

    source = SolarSource(atmo, float(sun_zenith_deg), 0.0)
    z_ground = float(atmo.z_bounds[0])
    if z_ground > 0.0 and layer is not None and layer.base_km < z_ground:
        return np.zeros(len(atmo.wl))
    f_inf = spectra.solar_irradiance(atmo.wl)
    if layer is None:
        return source.ground(np.array([[0.0, 0.0, R_EARTH + z_ground * 1e3]]),
                             iw=None)[0] * f_inf
    directions, weights = source.disk_for_column(z_ground)
    n_beam = directions.shape[0]
    up = np.array([0.0, 0.0, 1.0])
    true_e = np.degrees(np.arcsin(np.clip(directions @ up, -1.0, 1.0)))
    app_g, t_g, gain_g, lit_g = source.vertex(np.full(n_beam, z_ground), true_e)
    optics = layer.optics(atmo.wl) if layer is not None else None
    if optics is not None:
        app_t, _, _, _ = source.vertex(np.full(n_beam, float(layer.top_km)),
                                       true_e)
        mu_entry = np.maximum(np.abs(np.sin(np.radians(app_t))), 1e-3)
        deck_trans = np.exp(-optics.tau[None, :] / mu_entry[:, None])
    else:
        deck_trans = np.ones((n_beam, 1))
    sin_app = np.maximum(np.sin(np.radians(app_g)), 0.0)
    per_beam = (lit_g[:, None] * t_g * gain_g[:, None] * sin_app[:, None]
                * deck_trans)
    return (weights[:, None] * per_beam).sum(axis=0) * f_inf


def test_cloud_ground_helper_matches_pre_move_reference(atmo7, stratus, cirrus):
    """The helper moved, nothing else: bit-identical to the verbatim
    pre-move beam sum for decks, thin decks and no deck."""
    from skycolor.rt.cloud import ground_direct_irradiance

    for layer, el in ((stratus, 20.0), (cirrus, 5.0), (None, 20.0)):
        moved = ground_direct_irradiance(atmo7, layer, 90.0 - el)
        ref = _reference_deck_ground(atmo7, layer, 90.0 - el)
        assert np.array_equal(moved, ref), f"drift for {layer} at el={el}"


def test_zero_tau_deck_ground_equals_plain_beam_sum(atmo7):
    """tau=0 must reduce the deck beam sum to the no-deck beam sum: same
    contact-band disk beams, same weights, same per-beam products.  Tolerance
    source: the two branches evaluate the identical sum through different
    code paths inside the source (``ground`` vs the explicit beam loop), so
    only last-bit summation-order rounding differs -- pure relative
    tolerance, atol=0."""
    from skycolor import cloud
    from skycolor.rt.cloud import ground_direct_irradiance

    zero = cloud.get_preset("stratus", tau550=0.0)
    moved = ground_direct_irradiance(atmo7, zero, 70.0)
    plain = ground_direct_irradiance(atmo7, None, 70.0)
    np.testing.assert_allclose(moved, plain, rtol=1e-12, atol=0)


def test_thick_deck_and_shadow_darken_the_ground_beam(atmo7, stratus):
    """A thick deck attenuates the ground beam; a below-horizon sun (all
    ground beams unlit) and a deck under a raised opaque ground leave it
    dark."""
    from skycolor import presets
    from skycolor.rt.cloud import ground_direct_irradiance

    plain = ground_direct_irradiance(atmo7, None, 70.0)
    decked = ground_direct_irradiance(atmo7, stratus, 70.0)
    assert np.all(decked < plain)
    assert np.any(decked > 0.0)
    # sun 3 deg below the horizon: every ground beam is unlit
    assert np.all(ground_direct_irradiance(atmo7, stratus, 93.0) == 0.0)
    # deck base 0.5 km below a 2 km truncated ground: no direct light
    station = presets.clear(WL7).at_altitude(2.0)
    assert np.all(ground_direct_irradiance(station, stratus, 70.0) == 0.0)


def test_solve_dome_binds_deck_lighting_and_cloud_marker_off(atmo7, stratus):
    """A cloud request binds the deck's ground irradiance at preparation
    (the one library helper) and defaults to no decorative sun disk."""
    from skycolor import render
    from skycolor.rt.cloud import ground_direct_irradiance

    pres = render.solve_dome(atmo7, 20.0, 0.0, n_zenith=3, n_azimuth=3,
                             cloud_layer=stratus)
    assert pres.cloud is True
    assert pres.ground_direct_irradiance is not None
    np.testing.assert_array_equal(pres.ground_direct_irradiance,
                                  ground_direct_irradiance(atmo7, stratus, 70.0))
    auto = render.render_result(pres, size=(8, 4), ground=True)
    off = render.render_result(pres, size=(8, 4), ground=True, sun_disk=False)
    assert np.array_equal(auto, off)          # no unattenuated disk on a deck


def test_binding_snapshots_immune_to_facade_mutation():
    """After preparation, mutating the facade cannot change the bound
    display (and a fresh preparation does see the change).  Local medium:
    the assertions deliberately mutate the facade, so nothing is shared."""
    from skycolor import presets, render

    atmo = presets.clear(WL7)
    pres = render.solve_dome(atmo, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    before = render.render_result(pres, size=(8, 4), ground=True,
                                  sun_disk=False)
    atmo.ground_albedo = 0.9                    # replaces the frozen scene
    after = render.render_result(pres, size=(8, 4), ground=True,
                                 sun_disk=False)
    assert np.array_equal(before, after)
    fresh = render.solve_dome(atmo, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    assert np.all(fresh.ground_albedo == 0.9)


def test_albedo_bound_on_carrier_drives_display(atmo7):
    """The bound albedo (not the facade's current value) lights the display
    ground."""
    from skycolor import presets, render
    from skycolor.rt.solver import solve

    atmo = presets.clear(WL7)
    zen = np.linspace(0.0, 90.0, 3)
    az = np.linspace(0.0, 180.0, 3)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo, 70.0, Z.ravel(), A.ravel())
    e_dir = np.full(WL7.size, 1.0)
    dark = render.render_result(
        render.presentation_from_result(result, zen, az, 20.0, 0.0,
                                        ground_albedo=0.1,
                                        ground_direct_irradiance=e_dir),
        size=(8, 4), ground=True, sun_disk=False)
    bright = render.render_result(
        render.presentation_from_result(result, zen, az, 20.0, 0.0,
                                        ground_albedo=0.9,
                                        ground_direct_irradiance=e_dir),
        size=(8, 4), ground=True, sun_disk=False)
    assert np.any(bright > dark)
    atmo.ground_albedo = 0.5                    # facade value must be irrelevant
    again = render.render_result(
        render.presentation_from_result(result, zen, az, 20.0, 0.0,
                                        ground_albedo=0.1,
                                        ground_direct_irradiance=e_dir),
        size=(8, 4), ground=True, sun_disk=False)
    assert np.array_equal(again, dark)


def test_raised_observer_rejects_ground_display(atmo7):
    """Off-the-bottom observers keep their sky (solver permitting) but have
    no ground display: refused, not evaluated at another height."""
    from skycolor import presets, render

    balloon = presets.clear(WL7)
    balloon.elevation_km = 2.0
    pres = render.solve_dome(balloon, 30.0, 0.0, n_zenith=3, n_azimuth=3,
                             approximation="spherical-single")
    assert pres.ground_supported is False
    with pytest.raises(ValueError, match="model bottom"):
        render.render_result(pres, size=(8, 4), ground=True, sun_disk=False)
    img = render.render_result(pres, size=(8, 4), ground=False, sun_disk=False)
    assert img.shape == (4, 8, 3)


def test_sun_marker_consumes_apparent_source_geometry():
    """The decorative marker binds the adopted source's apparent geometry at
    preparation; without atmospheric refraction apparent == true.  Local
    media: isolated from other tests' facades."""
    from skycolor import presets, render

    atmo = presets.clear(WL7)
    pres = render.solve_dome(atmo, 0.5, 0.0, n_zenith=3, n_azimuth=3)
    assert pres.sun_apparent_elevation_deg > 0.5   # refraction lifts the disk
    assert pres.sun_visible is True
    center = render._sun_marker_geometry(pres, None)
    assert center == (pres.sun_azimuth_deg, pres.sun_apparent_elevation_deg)
    plane = presets.clear(WL7)
    plane.refraction = False
    pres_nr = render.solve_dome(plane, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    assert pres_nr.sun_apparent_elevation_deg == 20.0


def test_sun_marker_visibility_and_explicit_rules(atmo7, stratus, monkeypatch):
    """Auto marker: only a solver-prepared, sunlit clear sky draws it.  An
    explicit True needs bound geometry; False always disables; the marker
    never touches numeric outputs."""
    from skycolor import render

    # below the horizon: unlit source, no marker even on the clear path
    pres_tw = render.solve_dome(atmo7, -3.0, 0.0, n_zenith=3, n_azimuth=3,
                                approximation="spherical-single")
    assert pres_tw.sun_visible is False
    assert render._sun_marker_geometry(pres_tw, None) is None
    # external field has no bound geometry: auto skips, explicit True raises
    pres_ext = render.presentation_from_field(
        lambda zen, azr: np.zeros((np.asarray(zen).size, WL7.size)),
        atmo7, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    assert render._sun_marker_geometry(pres_ext, None) is None
    with pytest.raises(ValueError, match="apparent solar"):
        render._sun_marker_geometry(pres_ext, True)
    # explicit False disables everywhere
    assert render._sun_marker_geometry(pres_ext, False) is None
