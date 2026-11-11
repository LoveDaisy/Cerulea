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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
def test_joint_rotation_leaves_the_sky_invariant(atmo7):
    """Rotating the sun and the grid together rotates the physics, not the
    numbers.  Tolerance source: the joint rotation changes only the order of
    trigonometric evaluations in the kernels (last-bit rounding), so a pure
    relative tolerance at 1e-10 covers rounding, not physics (atol=0)."""
    from skycolor import render

    p0 = render.solve_dome(atmo7, 20.0, 0.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    p1 = render.solve_dome(atmo7, 20.0, 137.0, n_zenith=N_ZEN, n_azimuth=N_AZ)
    np.testing.assert_allclose(p1.radiance, p0.radiance, rtol=1e-10, atol=0)


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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
                                           ground_direct_irradiance=np.zeros(WL7.size),
                                           ground_supported=True)
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
        render.presentation_from_result(result, np.linspace(0.0, 90.0, 4),
                                        az, 20.0, 50.0)
    with pytest.raises(ValueError, match="grid"):
        render.presentation_from_result(result, zen,
                                        np.linspace(0.0, 180.0, 6), 20.0, 50.0)
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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
                                        ground_direct_irradiance=e_dir,
                                        ground_supported=True),
        size=(8, 4), ground=True, sun_disk=False)
    bright = render.render_result(
        render.presentation_from_result(result, zen, az, 20.0, 0.0,
                                        ground_albedo=0.9,
                                        ground_direct_irradiance=e_dir,
                                        ground_supported=True),
        size=(8, 4), ground=True, sun_disk=False)
    assert np.any(bright > dark)
    atmo.ground_albedo = 0.5                    # facade value must be irrelevant
    again = render.render_result(
        render.presentation_from_result(result, zen, az, 20.0, 0.0,
                                        ground_albedo=0.1,
                                        ground_direct_irradiance=e_dir,
                                        ground_supported=True),
        size=(8, 4), ground=True, sun_disk=False)
    assert np.array_equal(again, dark)


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
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


# ---------------------------------------------------------------------------
# 4. diagnostics through to the consumer (M4)
# ---------------------------------------------------------------------------
TINY_MC = dict(n_zen=3, n_az=4, n_anchor=3, n_photons=2.0e5)


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
def test_summary_reports_hybrid_provenance(atmo7):
    """A converged deterministic run states what ran and its numerics."""
    from skycolor import render
    from skycolor.rt.solver import solve

    res = solve(atmo7, 70.0, np.array([0.0, 45.0, 90.0]), 0.0)
    text = render.format_result_summary(res)
    assert "approximation: hybrid-sos (method 'clear')" in text
    assert "converged=yes" in text
    assert "n_gauss" in text and "n_mu" in text     # request + effective numerics
    assert "source-side refraction: on" in text
    assert "input direction pre-refraction" in text


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
def test_summary_reports_mc_budget_and_composite_stderr(atmo7):
    """A real low-budget MC run: actual photon/grid counts, the composite
    stderr statement and the budget-is-not-tail-coverage caveat."""
    from skycolor import render
    from skycolor.rt.solver import solve

    res = solve(atmo7, 96.0, np.array([0.0, 45.0]), np.array([0.0, 180.0]),
                approximation="spherical-mc", quality="low", seed=0,
                grid=TINY_MC)
    text = render.format_result_summary(res)
    assert "approximation: spherical-mc (method 'twilight')" in text
    assert "photons_run=" in text and "anchors_run=" in text
    assert "coarse grid 3x4" in text
    assert "composite estimate" in text
    assert "not tail coverage" in text
    assert "converged" not in text          # no SOS ran: no convergence claim


def test_summary_reports_single_term_without_convergence_claim(atmo7):
    from skycolor import render
    from skycolor.rt.solver import solve

    res = solve(atmo7, 70.0, np.array([45.0]), 0.0,
                approximation="spherical-single")
    text = render.format_result_summary(res)
    assert "approximation: spherical-single" in text
    assert "converged" not in text
    assert "effective: single" in text


def test_summary_unknowns_stay_unknown():
    """Missing diagnostics are reported as unknown, never filled in."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    bare = SkyResult(radiance=np.zeros((1, 3)), wl=np.array([400.0, 500.0, 600.0]),
                     method="clear", approximation="hybrid-sos")
    text = render.format_result_summary(bare)
    assert "components: unknown" in text
    assert "effective: unknown" in text


def test_summary_flags_non_convergence_and_undetected():
    """Synthetic failure states (hard to reproduce physically) surface in
    the summary as limitations, not as normal output."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    bad = SkyResult(radiance=np.zeros((1, 3)), wl=np.array([400.0, 500.0, 600.0]),
                    method="clear", approximation="hybrid-sos",
                    diagnostics={"orders": 40, "converged": False})
    text = render.format_result_summary(bad)
    assert "converged=NO" in text and "truncated" in text

    und = SkyResult(radiance=np.zeros((1, 3)), wl=np.array([400.0, 500.0, 600.0]),
                    method="twilight", approximation="spherical-mc",
                    diagnostics={"n_photons": 1.0e5,
                                 "undetected_wl": np.array([500.0, 600.0])},
                    stderr=np.array([0.0, 1e-12, 2e-12]))
    text = render.format_result_summary(und)
    assert "undetected wavelengths: 2" in text and "unresolved" in text


def test_convenience_entries_warn_on_result_limits(atmo7, monkeypatch):
    """The ndarray conveniences surface non-convergence / MC statistics as
    visible warnings instead of hiding them in the result dict."""
    import warnings as _warnings

    from skycolor import render
    from skycolor.rt.result import SkyResult
    import skycolor.rt.solver as solver

    zen = np.linspace(0.0, 90.0, 3)
    az = np.linspace(0.0, 180.0, 3)
    Z, A = np.meshgrid(zen, az, indexing="ij")

    def fake_solve(atmo, sun_zen, vzen, vaz, sun_azimuth_deg=0.0, **k):
        return SkyResult(radiance=np.zeros((vzen.size, atmo.wl.size)),
                         wl=atmo.wl,
                         method="clear", options={"sun_zenith_deg": float(sun_zen),
                                                  "sun_azimuth_deg": float(sun_azimuth_deg),
                                                  "refraction": True},
                         approximation="hybrid-sos",
                         diagnostics={"orders": 40, "converged": False})

    monkeypatch.setattr(solver, "solve", fake_solve)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        render.sky_dome(atmo7, 70.0, n_zenith=3, n_azimuth=3)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        render.sky_radiance_grid(atmo7, 70.0, Z, A)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        render.render_scene(atmo7, 20.0, 0.0, size=(8, 4),
                            n_zenith=3, n_azimuth=3)


def test_carrier_path_stays_silent_and_offers_summary(atmo7):
    """The carrier path reports through the summary function, not through
    blanket warnings: a converged deterministic preparation is silent."""
    import warnings as _warnings

    from skycolor import render

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        pres = render.solve_dome(atmo7, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    assert "converged=yes" in render.format_result_summary(pres)


# ---------------------------------------------------------------------------
# 5. review round 1: wrapper-boundary semantics (R1 fixes)
# ---------------------------------------------------------------------------
def test_summary_partial_mc_diagnostics_do_not_crash():
    """MC diagnostics present but undetected_wl missing/None: report the
    undetected state as unknown instead of crashing on a list attribute."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    wl = np.array([400.0, 500.0, 600.0])
    missing = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="twilight",
                        approximation="spherical-mc",
                        diagnostics={"n_photons": 1.0e5})
    text = render.format_result_summary(missing)
    assert "mc undetected wavelengths: unknown" in text
    explicit_none = SkyResult(radiance=np.zeros((1, 3)), wl=wl,
                              method="twilight", approximation="spherical-mc",
                              diagnostics={"n_photons": 1.0e5,
                                           "undetected_wl": None})
    assert "mc undetected wavelengths: unknown" in \
        render.format_result_summary(explicit_none)


def test_summary_mc_ran_but_all_undetected_is_not_reported_as_not_run():
    """A real engine state (rt.twilight returns zero stderr when every
    anchor is undetected): the summary must not claim 'no MC term ran'."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    res = SkyResult(radiance=np.zeros((2, 3)), wl=np.array([400.0, 500.0, 600.0]),
                    method="twilight", approximation="spherical-mc",
                    diagnostics={"n_photons": 1.0e4,
                                 "undetected_wl": np.array([400.0, 500.0, 600.0])},
                    stderr=np.zeros((2, 3)))
    text = render.format_result_summary(res)
    assert "although the Monte Carlo term ran" in text
    assert "not determinable from zeros" in text
    assert "no Monte Carlo term ran" not in text
    assert "unresolved" in text


def test_summary_mc_not_run_and_non_finite_states():
    from skycolor import render
    from skycolor.rt.result import SkyResult

    wl = np.array([400.0, 500.0, 600.0])
    not_run = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="twilight",
                        approximation="spherical-mc",
                        options={"mc_run": False}, stderr=np.zeros((1, 3)))
    assert "no Monte Carlo term ran" in render.format_result_summary(not_run)
    broken = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="twilight",
                       approximation="spherical-mc",
                       stderr=np.full((1, 3), np.nan))
    assert "no finite values" in render.format_result_summary(broken)


def test_convenience_warns_on_mc_even_with_zero_stderr(monkeypatch):
    """The MC statistics caveat fires whenever the MC term ran -- a zero
    stderr is absence of evidence, not a tight estimate."""
    import skycolor.rt.result as result_mod
    from skycolor import render

    res = result_mod.SkyResult(radiance=np.zeros((1, 3)),
                               wl=np.array([400.0, 500.0, 600.0]),
                               method="twilight", approximation="spherical-mc",
                               options={"mc_run": True}, stderr=np.zeros((1, 3)))
    with pytest.warns(RuntimeWarning, match="Monte Carlo statistical"):
        render._warn_result_limits(res)


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
def test_render_scene_default_sun_disk_is_auto():
    """The convenience entry shares render_result's single marker policy:
    unseen suns and external fields never force a ValueError, thick clouds
    default to no disk."""
    from skycolor import cloud, presets, render

    atmo = presets.clear(WL7)
    # twilight, source below the horizon: renders without any argument
    img = render.render_scene(atmo, -3.0, approximation="spherical-single",
                              ground=False, size=(8, 4), n_zenith=3,
                              n_azimuth=3)
    assert img.shape == (4, 8, 3)
    # external field without bound solar geometry: renders
    field = lambda zen, azr: np.zeros((np.asarray(zen).size, WL7.size))  # noqa: E731
    img = render.render_scene(atmo, 20.0, ground=False, size=(8, 4),
                              n_zenith=3, n_azimuth=3, radiance_field=field)
    assert img.shape == (4, 8, 3)
    # thick deck: the auto default equals an explicit off
    deck = cloud.get_preset("stratus")
    auto = render.render_scene(atmo, 20.0, size=(8, 4), n_zenith=3,
                               n_azimuth=3, cloud_layer=deck)
    off = render.render_scene(atmo, 20.0, size=(8, 4), n_zenith=3,
                              n_azimuth=3, cloud_layer=deck, sun_disk=False)
    assert np.array_equal(auto, off)


def test_external_field_with_explicit_true_marker_still_raises():
    from skycolor import presets, render

    atmo = presets.clear(WL7)
    field = lambda zen, azr: np.zeros((np.asarray(zen).size, WL7.size))  # noqa: E731
    with pytest.raises(ValueError, match="apparent solar"):
        render.render_scene(atmo, 20.0, ground=False, size=(8, 4),
                            n_zenith=3, n_azimuth=3, radiance_field=field,
                            sun_disk=True)


def test_legacy_transmittance_override_survives_on_production_path(atmo7):
    """The retired override works on the plain (no radiance_field) scene:
    converted once at the wrapper boundary to the bound irradiance, one
    deprecation warning, both-given still rejected."""
    from skycolor import render
    from skycolor.spectra import solar_irradiance

    T = np.full(WL7.size, 0.5)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        img = render.render_scene(atmo7, 20.0, size=(8, 4), n_zenith=3,
                                  n_azimuth=3, sun_disk=False,
                                  ground_direct_transmittance=T)
    # the conversion equals binding the historical formula directly
    pres = render.solve_dome(atmo7, 20.0, 0.0, n_zenith=3, n_azimuth=3)
    cosz = max(np.cos(np.radians(pres.sun_zenith_deg)), 0.0)
    pres = render._dc_replace(pres, ground_direct_irradiance=(
        solar_irradiance(pres.wl) * T * cosz))
    ref = render.render_result(pres, size=(8, 4), sun_disk=False)
    assert np.array_equal(img, ref)
    with pytest.raises(ValueError, match="not both"):
        render.render_scene(atmo7, 20.0, size=(8, 4), n_zenith=3,
                            n_azimuth=3, ground_direct_transmittance=T,
                            ground_direct_irradiance=np.zeros(WL7.size))


def test_external_field_rejects_solver_options(atmo7, monkeypatch):
    """Plan 3.2 enforced: an external field and solver choices are mutually
    exclusive at every public entry -- rejected before any sampling, and the
    legacy sky_dome forwarding is not misflagged by its internal defaults."""
    from skycolor import render

    calls = {"n": 0}

    def field(zen, azr):
        calls["n"] += 1
        return np.zeros((np.asarray(zen).size, WL7.size))

    with pytest.raises(ValueError, match="mutually exclusive"):
        render.solve_dome(atmo7, 20.0, radiance_field=field,
                          approximation="spherical-mc")
    from skycolor import cloud as _cloud

    with pytest.raises(ValueError, match="mutually exclusive"):
        render.render_scene(atmo7, 20.0, size=(8, 4), n_zenith=3,
                            n_azimuth=3, radiance_field=field,
                            cloud_layer=_cloud.get_preset("stratus"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        zen = np.linspace(0.0, 90.0, 3)
        az = np.linspace(0.0, 180.0, 3)
        Z, A = np.meshgrid(zen, az, indexing="ij")
        render.sky_radiance_grid(atmo7, 70.0, Z, A, radiance_field=field,
                                 approximation="spherical-single")
    with pytest.raises(ValueError, match="mutually exclusive"):
        zen = np.linspace(0.0, 90.0, 3)
        az = np.linspace(0.0, 180.0, 3)
        Z, A = np.meshgrid(zen, az, indexing="ij")
        render.sky_radiance_grid(atmo7, 70.0, Z, A, radiance_field=field,
                                 multiple=False)
    assert calls["n"] == 0        # rejected before the field is sampled
    # the legacy sky_dome field forwarding still works (nothing injected)
    dome = render.sky_dome(atmo7, 70.0, n_zenith=3, n_azimuth=3,
                           radiance_field=field)
    assert dome["radiance"].shape == (3, 3, WL7.size)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 6. review round 2: forwarding fidelity, ground capability, unknown facts,
#    grid sanity (R2 fixes)
# ---------------------------------------------------------------------------
def test_sky_radiance_grid_forwards_explicit_legacy_settings(atmo7, monkeypatch):
    """Explicit n_gauss/chunk reach the authoritative entry even in
    new-options mode; unset defaults are not injected (no phantom
    rejection); inapplicable or conflicting settings are rejected by the
    solver, not silently dropped by the wrapper."""
    import skycolor.rt.solver as solver
    from skycolor import render
    from skycolor.rt.solver import SingleConfig

    seen = {}
    real = solver.solve

    def spy(atmo, sun_zen, vzen, vaz, sun_azimuth_deg=0.0, **k):
        seen.update(k)
        return real(atmo, sun_zen, vzen, vaz, sun_azimuth_deg=sun_azimuth_deg,
                    **k)

    monkeypatch.setattr(solver, "solve", spy)
    zen = np.array([0.0, 60.0, 90.0])
    az = np.array([0.0, 120.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=9, chunk=17,
                             approximation="spherical-single")
    assert seen.get("n_gauss") == 9 and seen.get("chunk") == 17
    seen.clear()
    render.sky_radiance_grid(atmo7, 60.0, Z, A,
                             approximation="spherical-single")
    assert "n_gauss" not in seen and "chunk" not in seen
    assert seen.get("single_config") is None
    seen.clear()
    with pytest.raises(ValueError, match="single-scattering kernel"):
        render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=9,
                                 approximation="plane-sos")
    with pytest.raises(ValueError, match="twice"):
        render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=9,
                                 approximation="spherical-single",
                                 single_config=SingleConfig(n_gauss=5))


def test_ground_capability_is_checked_before_material(atmo7):
    """The model-bottom rule is a scene fact, not implied by bound lighting:
    a fully-lit but off-bottom presentation (solve-bound or external field)
    refuses ground=True and keeps ground=False."""
    from skycolor import presets, render
    from skycolor.rt.solver import solve

    zen = np.array([0.0, 45.0, 90.0])
    az = np.array([0.0, 90.0, 180.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel())
    pres = render.presentation_from_result(
        result, zen, az, 20.0, 0.0, ground_albedo=0.1,
        ground_direct_irradiance=np.zeros(WL7.size), ground_supported=False)
    with pytest.raises(ValueError, match="model bottom"):
        render.render_result(pres, size=(8, 4), ground=True, sun_disk=False)
    assert render.render_result(pres, size=(8, 4), ground=False,
                                sun_disk=False).shape == (4, 8, 3)
    # external field on a raised facade: no unconditional ground claim
    balloon = presets.clear(WL7)
    balloon.elevation_km = 2.0
    field = lambda z, a: np.zeros((np.asarray(z).size, WL7.size))  # noqa: E731
    pres_ext = render.presentation_from_field(
        field, balloon, 20.0, 0.0, n_zenith=3, n_azimuth=3,
        ground_direct_irradiance=np.zeros(WL7.size))
    assert pres_ext.ground_supported is False
    with pytest.raises(ValueError, match="model bottom"):
        render.render_result(pres_ext, size=(8, 4), ground=True,
                             sun_disk=False)
    # sea-level facade keeps the ground display
    pres_sea = render.presentation_from_field(
        field, presets.clear(WL7), 20.0, 0.0, n_zenith=3, n_azimuth=3,
        ground_direct_irradiance=np.zeros(WL7.size))
    assert pres_sea.ground_supported is True


def test_summary_keeps_missing_metadata_unknown():
    """Missing refraction/azimuth stay unknown: no fabricated off/0 values."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    wl = np.array([400.0, 500.0, 600.0])
    bare = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="clear",
                     approximation="hybrid-sos")
    text = render.format_result_summary(bare)
    assert "source-side refraction: unknown" in text
    assert "refraction: off" not in text
    partial = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="clear",
                        approximation="hybrid-sos",
                        options={"sun_zenith_deg": 70.0})
    text = render.format_result_summary(partial)
    assert "azimuth unknown" in text
    assert "refraction False" not in text
    off = SkyResult(radiance=np.zeros((1, 3)), wl=wl, method="clear",
                    approximation="hybrid-sos",
                    options={"sun_zenith_deg": 70.0, "sun_azimuth_deg": 5.0,
                             "refraction": False})
    assert "source-side refraction: off" in render.format_result_summary(off)


def test_summary_external_presentation_is_summarised_as_external():
    from skycolor import presets, render

    atmo = presets.clear(WL7)
    field = lambda z, a: np.zeros((np.asarray(z).size, WL7.size))  # noqa: E731
    pres = render.presentation_from_field(field, atmo, 20.0, 137.0,
                                          n_zenith=3, n_azimuth=3)
    text = render.format_result_summary(pres)
    assert "external field" in text
    assert "not applicable" in text
    assert "refraction: off" not in text          # no fabricated solver facts
    assert "137.000" in text                       # caller-declared direction shown as such


@pytest.mark.slow(reason="kind:render - full-image / display integration is full-tier exclusive")
def test_presentation_rejects_non_monotonic_grid(atmo7):
    """sample_dome clips and bisects on the grid axes: descending,
    duplicated or single-node axes are rejected at the binding boundary
    instead of silently producing a wrong sky."""
    from skycolor import render
    from skycolor.rt.solver import solve

    zen = np.array([0.0, 45.0, 90.0])
    az = np.array([0.0, 90.0, 180.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel())
    with pytest.raises(ValueError, match="strictly increasing"):
        render.presentation_from_result(result, zen[::-1], az, 20.0, 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        render.presentation_from_result(result, zen, az * 0.0, 20.0, 0.0)
    with pytest.raises(ValueError, match="at least two"):
        render.presentation_from_result(
            result, np.array([45.0]), az, 20.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        render.presentation_from_result(
            result, np.array([0.0, np.nan, 90.0]), az, 20.0, 0.0)


# ---------------------------------------------------------------------------
# 7. review round 3: per-parameter sentinel, shared grid invariant,
#    half-dome domain, run-state honesty (R3 fixes)
# ---------------------------------------------------------------------------
def test_grid_settings_forward_per_parameter_with_true_sentinel(atmo7,
                                                                monkeypatch):
    """Mixed flat/grouped settings merge at the entry (no phantom conflict
    from injected defaults); an explicit default value still conflicts with
    a grouped one (no swallowed conflict)."""
    import skycolor.rt.solver as solver
    from skycolor import render
    from skycolor.rt.solver import SingleConfig

    seen = {}
    real = solver.solve

    def spy(atmo, sun_zen, vzen, vaz, sun_azimuth_deg=0.0, **k):
        seen.update(k)
        return real(atmo, sun_zen, vzen, vaz, sun_azimuth_deg=sun_azimuth_deg,
                    **k)

    monkeypatch.setattr(solver, "solve", spy)
    zen = np.array([0.0, 60.0, 90.0])
    az = np.array([0.0, 120.0])
    Z, A = np.meshgrid(zen, az, indexing="ij")
    # legal mix: flat n_gauss + grouped chunk -> both reach the entry
    res = render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=9,
                                   approximation="spherical-single",
                                   single_config=SingleConfig(chunk=17))
    assert seen.get("n_gauss") == 9
    assert seen.get("single_config") == SingleConfig(chunk=17)
    assert res.shape == (3, 2, WL7.size)
    # explicit default value still conflicts with the grouped channel
    with pytest.raises(ValueError, match="twice"):
        render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=3,
                                 approximation="spherical-single",
                                 single_config=SingleConfig(n_gauss=9))
    with pytest.raises(ValueError, match="twice"):
        render.sky_radiance_grid(atmo7, 60.0, Z, A, chunk=512,
                                 approximation="spherical-single",
                                 single_config=SingleConfig(chunk=13))
    # legacy explicit values keep flowing in the legacy path
    seen.clear()
    render.sky_radiance_grid(atmo7, 60.0, Z, A, n_gauss=9, chunk=17)
    assert seen.get("n_gauss") == 9 and seen.get("chunk") == 17


def test_external_field_binding_shares_the_grid_invariant(atmo7, monkeypatch):
    """The external-field entry enforces the same axes invariant *before*
    sampling: single-node and constant axes are rejected, the field is
    never called."""
    from skycolor import render

    calls = {"n": 0}

    def field(zen, azr):
        calls["n"] += 1
        return np.zeros((np.asarray(zen).size, WL7.size))

    with pytest.raises(ValueError, match="at least two"):
        render.presentation_from_field(field, atmo7, 20.0, 0.0,
                                       n_zenith=1, n_azimuth=3)
    with pytest.raises(ValueError, match="strictly increasing"):
        render.presentation_from_field(field, atmo7, 20.0, 0.0, n_zenith=3,
                                       n_azimuth=3, min_zenith_deg=90.0)
    assert calls["n"] == 0


def test_binding_enforces_the_half_dome_domain(atmo7):
    """The display layer's supported input is the regular half dome: a
    full-circle azimuth layout is refused at binding instead of doubling
    the ground diffuse integral downstream."""
    from skycolor import render
    from skycolor.rt.solver import solve

    zen = np.linspace(0.0, 90.0, 3)
    az_full = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    Z, A = np.meshgrid(zen, az_full, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel())
    with pytest.raises(ValueError, match="half-dome"):
        render.presentation_from_result(result, zen, az_full, 20.0, 0.0)
    with pytest.raises(ValueError, match="half-dome"):
        render.presentation_from_result(result, zen - 10.0,
                                        az_full[:3], 20.0, 0.0)
    # the isotropic anchor on an accepted half dome: E = pi L exactly
    import dataclasses

    az = np.linspace(0.0, 180.0, 7)
    Z2, A2 = np.meshgrid(zen, az, indexing="ij")
    result2 = solve(atmo7, 70.0, Z2.ravel(), A2.ravel())
    ones = np.ones_like(result2.radiance)
    pres = render.presentation_from_result(
        dataclasses.replace(result2, radiance=ones), zen, az, 20.0, 0.0,
        ground_albedo=1.0, ground_direct_irradiance=np.zeros(WL7.size),
        ground_supported=True)
    e_diff = render.hemispherical_irradiance(pres.radiance, pres.zenith,
                                             pres.azimuth_rel)
    np.testing.assert_allclose(e_diff, np.pi, rtol=1e-3)


def test_summary_zero_stderr_without_run_metadata():
    """A zero stderr with no execution record must not claim the term ran."""
    from skycolor import render
    from skycolor.rt.result import SkyResult

    bare = SkyResult(radiance=np.zeros((1, 3)), wl=np.array([400.0, 500.0, 600.0]),
                     method="twilight", approximation="spherical-mc",
                     stderr=np.zeros((1, 3)))
    text = render.format_result_summary(bare)
    assert "run state not recorded" in text
    assert "although the Monte Carlo term ran" not in text
    # with an execution record the "ran" statement is warranted
    ran = SkyResult(radiance=np.zeros((1, 3)), wl=np.array([400.0, 500.0, 600.0]),
                    method="twilight", approximation="spherical-mc",
                    effective={"mc": {"n_photons_run": 100}},
                    stderr=np.zeros((1, 3)))
    assert "although the Monte Carlo term ran" in render.format_result_summary(ran)


# ---------------------------------------------------------------------------
# 8. review round 4: full-span half dome (R4 fix)
# ---------------------------------------------------------------------------
def test_binding_requires_full_span_half_dome(atmo7, monkeypatch):
    """A cropped layout lying inside the domain but not covering 0..180 /
    0..90 is refused: cameras would clamp the missing sky onto the edge and
    the ground integral would halve.  The isotropic anchor (L=1, albedo=1,
    no direct light -> ground spectrum exactly 1) sits at the *actual
    consumer*: the ground pixels render_result hands to color.render_rgb,
    spy-captured before tone mapping."""
    import dataclasses

    from skycolor import render
    from skycolor.rt.solver import solve

    zen = np.linspace(0.0, 90.0, 3)
    az = np.linspace(0.0, 180.0, 5)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo7, 70.0, Z.ravel(), A.ravel())
    ones = np.ones_like(result.radiance)
    with pytest.raises(ValueError, match="full half dome"):
        render.presentation_from_result(result, zen, az[:3], 20.0, 0.0)
    with pytest.raises(ValueError, match="full half dome"):
        render.presentation_from_result(result, zen, az[1:], 20.0, 0.0)
    with pytest.raises(ValueError, match="full half dome"):
        render.presentation_from_result(result, zen[1:], az, 20.0, 0.0)
    # external entry: same invariant
    field = lambda z, a: np.ones((np.asarray(z).size, WL7.size))  # noqa: E731
    with pytest.raises(ValueError, match="full half dome"):
        render.presentation_from_field(field, atmo7, 20.0, 0.0,
                                       n_zenith=3, n_azimuth=5,
                                       min_zenith_deg=30.0)
    # accepted shape: the analytic anchor sits at the *actual consumer* --
    # render_result's ground pixels, spy-captured from the spectrum handed
    # to color.render_rgb before tone mapping.  With L=1, albedo=1 and zero
    # direct light, Lg = albedo/pi * (0 + pi*L) = 1 per wavelength, so a
    # path that drops, double-counts or bypasses the diffuse ground light
    # cannot pass here; the helper-level integral is cross-checked above.
    full = render.presentation_from_result(
        dataclasses.replace(result, radiance=ones), zen, az, 20.0, 0.0,
        ground_albedo=1.0, ground_direct_irradiance=np.zeros(WL7.size),
        ground_supported=True)
    e_diff = render.hemispherical_irradiance(full.radiance, full.zenith,
                                             full.azimuth_rel)
    np.testing.assert_allclose(e_diff, np.pi, rtol=1e-3)
    captured = {}

    def _spy_render_rgb(rad, wl, **kwargs):
        captured["rad"] = np.asarray(rad, dtype=float)
        return np.zeros(np.asarray(rad).shape[:2] + (3,))

    monkeypatch.setattr(render.color, "render_rgb", _spy_render_rgb)
    render.render_result(full, camera="panorama", size=(8, 8),
                         ground=True, sun_disk=False)
    _, el = render.panorama_grid(8, 8)
    ground_px = captured["rad"][el < 0.0]
    assert ground_px.size > 0
    np.testing.assert_allclose(ground_px, 1.0, rtol=1e-3)
