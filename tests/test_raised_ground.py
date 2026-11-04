"""Raised-ground solar occultation, checked against independent geometry.

A globally raised opaque surface and a balloon above sea level have the same
observer radius but different solar shadows.  Zero-density media separate that
boundary condition from attenuation and Monte Carlo sampling noise.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere
from skycolor.constants import N_STD_AIR, R_EARTH
from skycolor.refraction import RefractionTables
from skycolor.rt import cloud, mc, multiple, single


WL = np.array([550.0])


def _vacuum(*, ground_km=0.0, observer_km=2.4, refraction=False):
    return Atmosphere(
        WL, aerosols=[], ozone_du=0.0, no2_column=0.0,
        air_profile=lambda z: np.zeros_like(np.asarray(z, dtype=float)),
        z_bounds_km=np.array([ground_km, 36.0, 40.0, 80.0]),
        elevation_km=observer_km, refraction=refraction,
    )


def _solar_direction(zenith_deg):
    angle = np.radians(zenith_deg)
    return np.array([np.sin(angle), 0.0, np.cos(angle)])


def _ray_hits_solid(x, direction, ground_radius):
    """Minimise distance along the forward ray, without impact parameters."""
    distance = np.maximum(-(x @ direction), 0.0)
    closest = x + distance[:, None] * direction
    return np.einsum("ij,ij->i", closest, closest) < ground_radius**2 * (1 - 1e-12)


@pytest.mark.parametrize("ground_km", [0.0, 2.4])
@pytest.mark.parametrize("sun_zenith", [30.0, 96.0])
def test_mc_solar_beam_respects_the_opaque_bottom(ground_km, sun_zenith):
    """In vacuum, each beam is exactly one or zero by ray/sphere intersection."""
    atmo = _vacuum(ground_km=ground_km)
    points = np.zeros((2, 3))
    points[:, 2] = R_EARTH + np.array([36.0, 40.0]) * 1e3
    direction = _solar_direction(sun_zenith)
    blocked = _ray_hits_solid(points, direction, R_EARTH + ground_km * 1e3)
    rel = mc._aerosol_spectral(atmo)
    transmittance = mc._sun_transmittance(
        atmo, points, direction, mc._shell_species_density(atmo),
        mc._species_tau_weights(atmo, 0, rel),
    )
    np.testing.assert_array_equal(transmittance, (~blocked).astype(float))


@pytest.mark.parametrize("ground_km, expected", [(0.0, 1.0), (2.4, 0.0)])
def test_direct_beam_distinguishes_a_balloon_from_a_raised_surface(ground_km, expected):
    """A -1 degree solar ray escapes a balloon but enters the high plateau."""
    atmo = _vacuum(ground_km=ground_km)
    result = single.direct_sun_transmittance(atmo, 91.0)
    np.testing.assert_array_equal(result, [expected])


@pytest.mark.parametrize("refraction", [False, True])
def test_single_scattering_layer_and_the_plateau_shadow(refraction):
    """Sea level is brighter than the plateau, and both go exactly dark deep.

    The finite disk blurs the sharp sea/plateau contrast: at 36.5 km height
    the two shadow boundaries sit only ~0.2 deg apart (sea -6.12 deg,
    plateau -5.91 deg), less than the 0.53 deg disk, so at -6 deg both decks
    are partially lit by the upper limb (sea more than plateau).  A whole
    -disk-shadowed sun (-7 deg) leaves both exactly dark.  The original
    centre-beam contrast survives in ``central_transmittance``, whose
    contract is the centre ray only.
    """

    def layer_density(z):
        z = np.asarray(z, dtype=float)
        return np.where((z >= 36.0) & (z <= 37.0), 1e18, 0.0)

    def build(ground_km):
        return Atmosphere(
            WL, aerosols=[], ozone_du=0.0, air_profile=layer_density,
            z_bounds_km=np.array([ground_km, 35.9, 36.0, 36.5, 37.0, 37.1, 80.0]),
            elevation_km=2.4, refraction=refraction,
        )

    sea6 = single.sky_radiance(build(0.0), 96.0, [0.0], [0.0])[0, 0]
    plateau6 = single.sky_radiance(build(2.4), 96.0, [0.0], [0.0])[0, 0]
    assert sea6 > plateau6 > 0.0
    assert single.sky_radiance(build(0.0), 97.0, [0.0], [0.0])[0, 0] == 0.0
    assert single.sky_radiance(build(2.4), 97.0, [0.0], [0.0])[0, 0] == 0.0
    # the centre-beam diagnostic keeps the sharp boundary *at the layer
    # height* (the observer sits at 2.4 km in both scenes and is shadowed at
    # -6 deg regardless of the ground below)
    from skycolor.solar_source import SolarSource
    sea_atmo = build(0.0)          # keep references: the source holds the
    plateau_atmo = build(2.4)      # atmosphere weakly
    t_sea = SolarSource(sea_atmo, 96.0, 0.0).central_transmittance(36.5)[0]
    t_plateau = SolarSource(plateau_atmo, 96.0, 0.0).central_transmittance(36.5)[0]
    assert t_sea > 0.0
    assert t_plateau == 0.0


@pytest.mark.parametrize("refraction", [False, True])
def test_upper_limb_partially_lights_the_plateau_deck(refraction):
    """At -6 deg the disk's upper limb crosses the plateau boundary.

    The centre beam is blocked (the old whole-source semantics) but the
    finite disk keeps a thin lit sliver, so the deck radiance is strictly
    positive yet far below the sea-level value.
    """

    def layer_density(z):
        z = np.asarray(z, dtype=float)
        return np.where((z >= 36.0) & (z <= 37.0), 1e18, 0.0)

    results = []
    for ground_km in [0.0, 2.4]:
        atmo = Atmosphere(
            WL, aerosols=[], ozone_du=0.0, air_profile=layer_density,
            z_bounds_km=np.array([ground_km, 35.9, 36.0, 36.5, 37.0, 37.1, 80.0]),
            elevation_km=2.4, refraction=refraction,
        )
        results.append(single.sky_radiance(atmo, 96.0, [0.0], [0.0])[0, 0])
    assert results[0] > 0.0
    assert 0.0 < results[1] < results[0]


@pytest.mark.parametrize("air_density", [0.0, N_STD_AIR])
def test_constant_index_shadow_matches_analytic_tangent(air_density):
    """Any spatially constant n has zero bending and the same tangent."""
    atmo = Atmosphere(
        WL, aerosols=[], ozone_du=0.0,
        air_profile=lambda z: np.full_like(np.asarray(z, dtype=float), air_density),
        z_bounds_km=np.array([2.4, 36.0, 40.0, 80.0]),
        elevation_km=2.4, refraction=True,
    )
    heights = np.array([2.4, 36.0, 40.0, 80.0])
    tables = RefractionTables(atmo, alt_km=heights)
    # Pythagoras in the tangent right triangle; independent of boundary_theta.
    radius = R_EARTH + heights * 1e3
    tangent_distance = np.sqrt(radius**2 - (R_EARTH + 2400.0)**2)
    dip = np.degrees(np.arctan2(tangent_distance, R_EARTH + 2400.0))
    np.testing.assert_allclose(tables.shadow_zenith_true(heights), 90.0 + dip, atol=1e-7)
    np.testing.assert_allclose(tables.shadow_zenith_apparent(heights), 90.0 + dip,
                               atol=1e-10)
    np.testing.assert_allclose(tables.horizon_dip(heights), dip, atol=1e-10)
    np.testing.assert_allclose(tables.refraction_extension(heights), 0.0, atol=1e-7)


@pytest.mark.parametrize("refraction", [False, True])
def test_sos_solar_source_has_the_same_shadow(refraction):
    """The unified source's per-beam tables obey the analytic opaque sphere.

    In this vacuum the refractive index is exactly 1, so the source's
    occultation boundary is the geometric ray/sphere intersection; a whole
    -disk-shadowed sun (zenith 97) then gives transmittance exactly 1 or 0
    per boundary node, equal to the independent ray/sphere test.
    """
    from skycolor.solar_source import SolarSource

    atmo = _vacuum(ground_km=2.4, refraction=refraction)
    source = SolarSource(atmo, 97.0, 0.0)
    geom = source.geometry
    # the constant-index branch keeps the shadow exact (no interpolation)
    assert geom.field.n.std() == 0.0
    heights = np.asarray(atmo.z_bounds, dtype=float)
    _, t_all, _, lit = source.vertex(heights, np.full(heights.shape, -7.0))
    points = np.zeros((heights.size, 3))
    points[:, 2] = atmo.r_bounds
    blocked = _ray_hits_solid(points, _solar_direction(97.0), R_EARTH + 2400.0)
    np.testing.assert_array_equal((~blocked).astype(float), lit.astype(float))
    np.testing.assert_array_equal(t_all[:, 0], (~blocked).astype(float))


def test_cloud_point_and_deck_share_the_raised_ground_shadow():
    """A cloud wholly inside the high plateau's shadow gets no direct beam."""
    atmo = _vacuum(ground_km=2.4)
    transmittance, sunlit = cloud.sun_transmittance_to_height(atmo, 36.0, 96.0)
    assert not sunlit
    np.testing.assert_array_equal(transmittance, [0.0])

    # Optical closure is inexpensive and independent of the Mie droplet
    # calculation; the geometric classification is the assertion here.
    deck = SimpleNamespace(
        base_km=36.0, top_km=37.0,
        optics=lambda wl: SimpleNamespace(
            tau=np.ones_like(wl), omega=np.full_like(wl, 0.9),
            g=np.full_like(wl, 0.5), f=np.full_like(wl, 0.25)),
    )
    # the finite disk keeps the deck's upper limb lit at -6 deg (the
    # refracted boundary at 36.5 km is -7.17 deg true): partial light is
    # physical, and the whole-disk shadow needs a deeper sun
    high = cloud.illuminate_cloud(atmo, deck, 96.0)
    assert high.geometry in ("base", "mixed")
    high_deep = cloud.illuminate_cloud(atmo, deck, 98.5)
    assert high_deep.geometry == "none"
    low = cloud.illuminate_cloud(_vacuum(ground_km=0.0), deck, 96.0)
    assert low.geometry == "base"


def test_balloon_occultation_uses_ground_refractivity():
    """A raised observer does not replace n_ground in the Bouguer boundary."""
    from skycolor import refraction as refr
    from skycolor.solar_source import SolarSource

    def density(z):
        z = np.asarray(z, dtype=float)
        # A smooth real-scale-height refractive profile plus the weak test
        # layer: the observer sits above the refractive lower atmosphere and
        # the only scattering visible along the upward ray is the thin layer.
        # (Discontinuous test densities are genuine ducts/near-caustics for
        # any refractive spline: n*r must keep increasing, which needs a
        # scale height well above (n-1)*R ~ 1.8 km, and derivative jumps can
        # fold the apparent<->true map.)
        # the 3 km tail keeps n*r monotone (needs H > (n-1)*R ~ 1.8 km)
        # while collapsing the upper-atmosphere density fast enough that the
        # disk's upper limb cannot light anything above the test layer
        return (N_STD_AIR * np.exp(-z / 3.0)
                + np.where((z >= 36.4) & (z <= 36.6), 1e18, 0.0))

    z_bounds = np.array([0.0, 0.5, 1.0, 2.4, 36.4, 36.6, 80.0])
    atmo = Atmosphere(
        WL, aerosols=[], ozone_du=0.0, air_profile=density,
        z_bounds_km=z_bounds,
        elevation_km=2.4, refraction=True,
    )
    # The centre beam's invariant at the layer lies between R and
    # n_ground*R: only the *sea-level* refractivity makes it opaque.
    invariant = (R_EARTH + 36.5e3) * np.cos(np.radians(6.0))
    assert R_EARTH < invariant < (1 + refr.REFRACTIVITY_STD) * R_EARTH
    source = SolarSource(atmo, 96.0, 0.0)
    geom = source.geometry
    assert geom.field.q_ground == pytest.approx(
        (1 + refr.REFRACTIVITY_STD * float(atmo.n_air(np.array([0.0]))[0])
         / N_STD_AIR) * R_EARTH, rel=1e-6)
    # the shadow boundary at the layer height is exactly the ray whose
    # invariant equals q_ground (anchored at sea level, not the observer)
    lower = geom.field.grazing_apparent_elevation(np.array([36.5]))
    r_layer = R_EARTH + 36.5e3
    invariant = (float(geom.field.refractive_index(np.array([36.5]))[0])
                 * r_layer * np.cos(np.radians(float(lower[0]) + 1e-6)))
    assert invariant == pytest.approx(geom.field.q_ground, rel=1e-6)
    # a whole-disk-shadowed sun leaves the deck exactly dark
    atmo_deep = Atmosphere(
        WL, aerosols=[], ozone_du=0.0, air_profile=density,
        z_bounds_km=z_bounds,
        elevation_km=2.4, refraction=True,
    )
    deep = single.sky_radiance(atmo_deep, 98.0, [0.0], [0.0])[0, 0]
    # whole-disk-shadowed: exactly dark up to the table's numerical floor
    # (the disk edge straddles the boundary within one height cell)
    assert deep < 1e-9
