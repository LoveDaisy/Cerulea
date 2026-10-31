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
def test_single_scattering_layer_inside_the_shadow_is_dark(refraction, monkeypatch):
    """A weak layer at 36-37 km is sunlit over sea level, occulted over 2.4 km."""
    # The density is tiny so attenuation/refraction are negligible.  Fix the
    # apparent direction to the geometric one to isolate the occultation in
    # both code paths without the packaged standard-atmosphere lookup table.
    if refraction:
        from skycolor import refraction as refr
        monkeypatch.setattr(
            refr, "geometry", lambda: SimpleNamespace(
                apparent_elevation=lambda true_elev, z: np.asarray(true_elev)),
        )

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
    assert results[1] == 0.0


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
def test_sos_solar_source_has_the_same_shadow(refraction, monkeypatch):
    """The source sampled by SOS obeys the same analytic opaque sphere."""
    atmo = _vacuum(ground_km=2.4, refraction=refraction)
    if refraction:
        from skycolor import refraction as refr
        # Build only the requested boundary nodes; interpolation cannot blur
        # the geometric shadow in this zero-density limiting case.
        tables = RefractionTables(atmo, alt_km=atmo.z_bounds)
        monkeypatch.setattr(refr, "get_tables", lambda _: tables)
    points = np.zeros((len(atmo.z_bounds), 3))
    points[:, 2] = atmo.r_bounds
    blocked = _ray_hits_solid(points, _solar_direction(96.0), R_EARTH + 2400.0)
    result = multiple._sun_transmittance_boundaries(atmo, 96.0)
    np.testing.assert_array_equal(result[:, 0], (~blocked).astype(float))


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
    high = cloud.illuminate_cloud(atmo, deck, 96.0)
    assert high.geometry == "none"
    low = cloud.illuminate_cloud(_vacuum(ground_km=0.0), deck, 96.0)
    assert low.geometry == "base"


def test_balloon_occultation_uses_ground_refractivity(monkeypatch):
    """A raised observer does not replace n_ground in the Bouguer boundary."""
    from skycolor import refraction as refr

    monkeypatch.setattr(
        refr, "geometry", lambda: SimpleNamespace(
            apparent_elevation=lambda true_elev, z: np.asarray(true_elev)),
    )

    def density(z):
        z = np.asarray(z, dtype=float)
        # The observer is above the refractive lower layer; the only
        # scattering visible along the upward ray is the weak test layer.
        return np.where(z < 1.0, N_STD_AIR,
                        np.where((z >= 36.4) & (z <= 36.6), 1e18, 0.0))

    atmo = Atmosphere(
        WL, aerosols=[], ozone_du=0.0, air_profile=density,
        z_bounds_km=np.array([0.0, 0.5, 1.0, 2.4, 36.4, 36.6, 80.0]),
        elevation_km=2.4, refraction=True,
    )
    # Given the supplied apparent direction, the invariant lies between
    # n_observer*R and n_ground*R.  Only the latter is the opaque boundary.
    invariant = (R_EARTH + 36.5e3) * np.cos(np.radians(6.0))
    assert R_EARTH < invariant < (1 + refr.REFRACTIVITY_STD) * R_EARTH
    np.testing.assert_array_equal(single.sky_radiance(atmo, 96.0, [0.0], [0.0]), [[0.0]])
