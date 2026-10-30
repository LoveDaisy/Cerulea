import numpy as np
import pytest

from skycolor import render


def _dome(radiance):
    zen = np.linspace(0, 90, 91)
    az = np.linspace(0, 180, 91)
    return radiance[:, None, :] * np.ones((1, len(az), 1))


def test_irradiance_isotropic_field():
    """An isotropic radiance L gives E = pi L."""
    zen = np.linspace(0, 90, 91)
    az = np.linspace(0, 180, 91)
    L = np.ones((len(zen), len(az), 2))
    E = render.hemispherical_irradiance(L, zen, az)
    assert np.allclose(E, np.pi, rtol=1e-3)


def test_irradiance_cosine_field():
    """L = cos(theta) gives E = 2 pi / 3."""
    zen = np.linspace(0, 90, 91)
    az = np.linspace(0, 180, 91)
    L = np.cos(np.radians(zen))[:, None, None] * np.ones((1, len(az), 2))
    E = render.hemispherical_irradiance(L, zen, az)
    assert np.allclose(E, 2 * np.pi / 3, rtol=5e-3)


def test_irradiance_zero_cosine():
    """A pure cos^2 field integrates to pi/2."""
    zen = np.linspace(0, 90, 181)
    az = np.linspace(0, 180, 91)
    L = np.cos(np.radians(zen))[:, None, None] ** 2 * np.ones((1, len(az), 1))
    E = render.hemispherical_irradiance(L, zen, az)
    assert np.allclose(E, np.pi / 2, rtol=1e-2)
