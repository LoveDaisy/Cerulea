"""Independent disk moments and transfer algebra for the unified solar source.

Toy point tables intentionally remove the expensive interpolation/reference
implementation: expected values come from spherical-cap integrals and local
energy transfer, not by repeating the disk-node summation under test.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from skycolor.constants import DEPOLARIZATION, R_EARTH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.unified_solar_transport import FiniteSolarSource, solar_disk
from tools.radial_solar_mapping import RadialSolarMapping
from skycolor.atmosphere import Atmosphere


class IsotropicAerosol:
    @staticmethod
    def phase(mu, wavelengths):
        return np.ones((np.size(mu), len(wavelengths)))


class ToyTable:
    """An analytic transparent identity map, optionally behind a half mask."""

    def __init__(self, atmo, *, n=1.0, half=False, transmission=(1.0, 1.0),
                 vertex_n=None, gain_local=None, app_shift_deg=0.):
        self.atmo, self.half = atmo, half
        self.wl = np.asarray(atmo.wl).copy()
        self.transmission = np.asarray(transmission, float)
        self.vertex_n = vertex_n if vertex_n is not None else lambda z: np.full_like(
            np.asarray(z, float), n)
        self.mapping = SimpleNamespace(refractive_index=self.vertex_n)
        self.gain_local = gain_local
        self.app_shift_deg = app_shift_deg

    def query(self, height, true, iw=None):
        height, true = np.broadcast_arrays(height, true)
        lit = true > 0 if self.half else np.ones(true.shape, bool)
        gain = np.ones(true.shape) if self.gain_local is None else self.gain_local(height)
        if iw is None:
            trans = lit[..., None] * self.transmission
        else:
            trans = lit * self.transmission[iw]
        return true + self.app_shift_deg, trans, gain, lit


def atmosphere(*, aerosol=False, observer_km=0.0):
    layers = [SimpleNamespace(aerosol=IsotropicAerosol())] if aerosol else []
    return SimpleNamespace(wl=np.array([500., 600.]), elevation_km=observer_km,
                           z_bounds=np.array([0., 120.]), aerosols=layers)


def rayleigh_cap_average(sun, backward, radius_deg):
    """Exact cap second moment, including center-normal irradiance convention."""
    c = np.cos(np.radians(radius_deg))
    longitudinal = (1 + c + c*c) / 3
    transverse = (1 - longitudinal) / 2
    second_moment = transverse + (longitudinal - transverse) * (backward @ sun)**2
    gamma = DEPOLARIZATION / (2 - DEPOLARIZATION)
    constant = 3 * (1 + 3*gamma) / (4 * (1 + 2*gamma))
    quadratic = 3 * (1 - gamma) / (4 * (1 + 2*gamma))
    return (constant + quadratic * second_moment) * 2 / (1 + c)


@pytest.mark.parametrize("radius_deg", [0.0, 16/60, 5.0])
def test_uniform_disk_rayleigh_matches_exact_spherical_cap_moment(radius_deg):
    atmo = atmosphere()
    table = ToyTable(atmo, transmission=(1., .4))
    sun = np.array([.8, 0., .6])
    source = FiniteSolarSource(atmo, table, n_radial=3, n_azimuth=8, radius_deg=radius_deg)
    radius = R_EARTH + 40_000
    positions = radius * np.array([[0., 0., 1.], [0., 1/np.sqrt(2), 1/np.sqrt(2)],
                                   [.6, 0., .8]])
    backward = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    expected = rayleigh_cap_average(sun, backward, radius_deg)
    full = source.scatter(positions, backward, sun, None,
                          np.ones((3, 2)), np.empty((3, 0, 2)))
    np.testing.assert_allclose(full, expected[:, None]*[1., .4], rtol=2e-13, atol=2e-14)
    scalar = source.scatter(positions, backward, sun, 1, np.ones(3), np.empty((3, 0)))
    np.testing.assert_allclose(scalar, expected*.4, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("n", [1., 1.000278, 1.3])
def test_constant_index_full_disk_preserves_normal_irradiance(n):
    atmo = atmosphere()
    source = FiniteSolarSource(atmo, ToyTable(atmo, n=n), n_radial=3, n_azimuth=8)
    sun = np.array([.8, 0., .6])
    normals = np.array([[0., 0., 1.], [.6, 0., .8]])
    positions = normals * (R_EARTH + 10_000)
    result = source.ground(positions, sun, 0)
    # Every cap direction is above each plane. The integrated incident vector
    # of a uniform cap is precisely F_inf * sun, not F_inf * mean_one * sun.
    np.testing.assert_allclose(result, normals @ sun, atol=3e-14)


def test_half_occulted_isotropic_disk_loses_half_its_scalar_source():
    atmo = atmosphere(aerosol=True)
    half = FiniteSolarSource(atmo, ToyTable(atmo, half=True), n_radial=3, n_azimuth=8)
    full = FiniteSolarSource(atmo, ToyTable(atmo), n_radial=3, n_azimuth=8)
    position = np.array([[0., 0., R_EARTH]])
    sun = np.array([1., 0., 0.])
    backward = np.array([[0., 1., 0.]])
    a = half.scatter(position, backward, sun, 0, np.zeros(1), np.ones((1, 1)))
    b = full.scatter(position, backward, sun, 0, np.zeros(1), np.ones((1, 1)))
    np.testing.assert_allclose(a, .5*b, rtol=2e-14)


def test_horizon_ground_flux_matches_analytic_uniform_disk_integral():
    atmo = atmosphere()
    radius_deg = 5.0
    source = FiniteSolarSource(atmo, ToyTable(atmo), n_radial=16, n_azimuth=128,
                               radius_deg=radius_deg)
    result = source.ground(np.array([[0., 0., R_EARTH]]), np.array([1., 0., 0.]), 0)
    alpha = np.radians(radius_deg)
    expected = (alpha - .5*np.sin(2*alpha)) / (np.pi*np.sin(alpha)**2)
    np.testing.assert_allclose(result, [expected], rtol=3e-4)


def test_detector_reduced_radiance_factor_cancels_local_n_squared_once():
    """Algebraic connector oracle; this toy n profile is not a ray solution."""
    atmo = atmosphere(aerosol=True, observer_km=2.0)
    n_of_z = lambda z: 1 + .01*np.asarray(z)
    table = ToyTable(atmo, vertex_n=n_of_z, gain_local=lambda z: n_of_z(z)**2)
    source = FiniteSolarSource(atmo, table, radius_deg=0)
    points = np.column_stack([np.zeros((2, 2)), R_EARTH+np.array([10., 40.])*1000])
    result = source.scatter(points, np.tile([0., 0., 1.], (2, 1)),
                            np.array([.8, 0., .6]), 0, np.zeros(2), np.ones((2, 1)))
    np.testing.assert_allclose(result, n_of_z(2.0)**2, rtol=2e-14)


def test_spectral_aerosol_mixture_is_linear_in_local_phase_components():
    atmo = atmosphere(aerosol=True)
    source = FiniteSolarSource(atmo, ToyTable(atmo), n_radial=3, n_azimuth=8)
    sun, backward = np.array([.8, 0., .6]), np.array([[0., 0., 1.]])
    position = np.array([[0., 0., R_EARTH+40_000]])
    ray = np.array([[.3, .7]])
    aerosol = (1-ray)[:, None, :]
    result = source.scatter(position, backward, sun, None, ray, aerosol)
    cap_normalizer = 2 / (1 + np.cos(np.radians(16/60)))
    expected = rayleigh_cap_average(sun, backward, 16/60)[:, None]*ray + cap_normalizer*(1-ray)
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-14)


def test_local_apparent_direction_controls_signed_phase_and_ground_cosine():
    """A linear phase distinguishes incoming-ray sign as well as true/app angle."""
    class LinearAerosol:
        @staticmethod
        def phase(mu, wavelengths):
            return np.broadcast_to((1+.5*np.asarray(mu).ravel())[:, None],
                                   (np.size(mu), len(wavelengths)))

    atmo = atmosphere(aerosol=True)
    atmo.aerosols[0].aerosol = LinearAerosol()
    source = FiniteSolarSource(atmo, ToyTable(atmo, app_shift_deg=20.), radius_deg=0)
    sun = np.array([np.cos(np.radians(30.)), 0., np.sin(np.radians(30.))])
    point = np.array([[0., 0., R_EARTH+40_000]])
    result = source.scatter(point, np.array([[0., 0., 1.]]), sun, 0,
                            np.zeros(1), np.ones((1, 1)))
    local_mu = np.sin(np.radians(50.))
    np.testing.assert_allclose(result, [1+.5*local_mu], atol=2e-14)
    np.testing.assert_allclose(source.ground(point, sun, 0), [local_mu], atol=2e-14)


def test_explicit_source_rejects_a_mismatched_table_wavelength_grid():
    atmo = atmosphere()
    table = ToyTable(atmo)
    table.wl = table.wl + 1
    with pytest.raises(ValueError, match="wavelength"):
        FiniteSolarSource(atmo, table)


def test_transport_rejects_an_anchor_subset_or_modified_scene():
    atmo = atmosphere()
    source = FiniteSolarSource(atmo, ToyTable(atmo))
    source.validate_transport(atmo)
    anchor = atmosphere()
    anchor.wl = np.array([600.])
    with pytest.raises(ValueError, match="unchanged construction"):
        source.validate_transport(anchor)
    # A replaced frozen scene must be rejected even when the facade survives.
    atmo._scene = object()
    with pytest.raises(ValueError, match="unchanged construction"):
        source.validate_transport(atmo)


@pytest.mark.parametrize("n_quad", [8, 16, 32])
def test_radial_derivative_resolves_profile_spline_knots(n_quad):
    """Cartesian DOP853 anchors constrain a previously aliased tangent ray."""
    atmo = Atmosphere([550.], aerosols=[], ozone_du=0)
    mapping = RadialSolarMapping(atmo, n_quad=n_quad)
    height, app = 100., -9.48538509285
    # Independent DOP853 direction and its +/-1e-4 degree difference. The
    # wider +/-0.001 difference is physically different here, not an oracle.
    assert float(mapping.forward(height, app)) == pytest.approx(
        -9.869439304834236, abs=2e-7)
    derivative = float(mapping.jacobian(height, app, step_deg=1e-5))
    assert derivative == pytest.approx(1.3377073390596905, abs=5e-5)
    gain = float(mapping.beam_gain(height, app, step_deg=1e-5))
    assert gain == pytest.approx(0.748402640628298, abs=5e-5)
