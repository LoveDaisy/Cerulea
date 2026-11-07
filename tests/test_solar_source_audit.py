"""Independent limiting cases and RK anchors for experimental solar sources.

These tests constrain the audit tools without changing production transport.
The RK roots/gains below come from Cartesian ray stepping and differencing,
not from the radial-integral implementation being tested.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere
from skycolor.constants import N_STD_AIR, R_EARTH
from skycolor import refraction

# Audit tools are repository modules, not part of the installed library wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_single_solar_source import InverseSolarMapping
from tools.radial_solar_mapping import RadialSolarMapping


WL = np.array([550.0])


def _constant_index_atmosphere(density=0.0, ground_km=0.0):
    return Atmosphere(
        WL, aerosols=[], ozone_du=0.0, no2_column=0.0,
        air_profile=lambda z: np.full_like(np.asarray(z, dtype=float), density),
        z_bounds_km=np.array([ground_km, 30.0, 36.0, 40.0, 120.0]),
        elevation_km=ground_km, refraction=True,
    )


@pytest.mark.parametrize("density", [0.0, N_STD_AIR])
def test_constant_index_has_no_bending_or_angular_gain(density):
    """A uniform index, including its far field, has gain exactly one."""
    atmo = _constant_index_atmosphere(density)
    mapping = RadialSolarMapping(atmo, n_quad=8)
    heights = np.array([0.0, 20.0, 36.0, 80.0])
    apparent = np.array([1.0, 10.0, -6.0, -8.0])
    np.testing.assert_allclose(mapping.forward(heights, apparent), apparent, atol=1e-12)
    np.testing.assert_allclose(mapping.beam_gain(heights, apparent), 1.0, atol=1e-9)


@pytest.mark.parametrize("ground_km", [0.0, 2.4])
def test_constant_index_shadow_matches_forward_ray_minimum(ground_km):
    """Minimising distance along a Cartesian ray gives an independent mask."""
    atmo = _constant_index_atmosphere(ground_km=ground_km)
    mapping = RadialSolarMapping(atmo, n_quad=8)
    heights = np.array([30.0, 36.0, 40.0])
    points = np.column_stack([np.zeros(3), R_EARTH + heights*1e3])
    ray = np.array([np.cos(np.radians(-6.0)), np.sin(np.radians(-6.0))])
    closest = points + np.maximum(-(points @ ray), 0.0)[:, None] * ray
    blocked = np.linalg.norm(closest, axis=1) < R_EARTH + ground_km*1e3
    result = mapping.forward(heights, np.full(3, -6.0))
    np.testing.assert_array_equal(np.isnan(result), blocked)
    np.testing.assert_allclose(result[~blocked], -6.0, atol=1e-12)


@pytest.mark.parametrize("ground_km", [0.0, 2.4])
def test_inverse_does_not_extrapolate_a_solar_image_into_shadow(ground_km):
    """An unlit inverse output is paired with zero source amplitude."""
    atmo = _constant_index_atmosphere(ground_km=ground_km)
    inverse = InverseSolarMapping(atmo, n_quad=8)
    height = np.array([30.0, 36.0, 40.0])
    true = np.full(3, -6.0)
    app = inverse.apparent_elevation(true, height)
    amplitude = inverse.amplitude(true, height, app)
    # The geometric terminator is h=R(sec(6°)-1) above sea level;
    # replacing R by the raised radius gives h≈37.51 km at 2.4 km ground.
    critical_height = ((R_EARTH + ground_km*1e3)/np.cos(np.radians(6.0))-R_EARTH)/1e3
    lit = height > critical_height
    np.testing.assert_allclose(app[lit], true[lit], atol=1e-8)
    np.testing.assert_array_equal(amplitude[~lit], np.zeros((~lit).sum()))
    np.testing.assert_allclose(amplitude[lit], 1.0, atol=1e-8)


@pytest.fixture(scope="module")
def standard_air():
    return Atmosphere(WL, aerosols=[], ozone_du=0.0, refraction=True)


@pytest.mark.parametrize("height, apparent, rk_gain", [
    (38.0, -5.487342769121213, 0.6167382499741723),
    (40.0, -5.55511684323352, 0.6513858154491060),
    (47.0, -5.7677034867361385, 0.6931055480628819),
])
def test_radial_mapping_matches_cartesian_rk_roots_and_gains(
        standard_air, height, apparent, rk_gain):
    """RK step/difference-width scatter sets the 5e-5 gain tolerance."""
    mapping = RadialSolarMapping(standard_air, n_quad=16)
    assert float(mapping.forward(height, apparent)) == pytest.approx(-6.0, abs=2e-6)
    assert float(mapping.beam_gain(height, apparent)) == pytest.approx(rk_gain, abs=5e-5)

    inverse = InverseSolarMapping(standard_air, n_quad=8)
    result = inverse.apparent_elevation(np.array([-6.0]), np.array([height]))
    assert result[0] == pytest.approx(apparent, abs=2e-6)
    gain = inverse.amplitude(np.array([-6.0]), np.array([height]), result)
    assert gain[0] == pytest.approx(rk_gain, abs=5e-5)


# The inspect/exec ablation drivers below this line were retired with the
# unified solar source: they anchored production internals that the unified
# source replaced.  Historical evidence lives in
# docs/twilight/SOLAR_SOURCE_AUDIT.md and
# docs/twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md.
