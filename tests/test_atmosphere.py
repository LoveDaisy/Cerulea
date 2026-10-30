import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere, us76_pressure, us76_temperature
from skycolor.constants import DU, WL_COARSE


def test_us76_reference_values():
    z = np.array([0.0, 5.0, 11.0, 20.0, 32.0, 50.0, 71.0])
    p = us76_pressure(z)
    expected = np.array([101325, 54020, 22632, 5475, 868, 76.0, 3.96])
    assert np.allclose(p / expected, 1.0, rtol=0.02)


def test_temperature_layers():
    assert us76_temperature(np.array([0.0]))[0] == pytest.approx(288.15)
    assert us76_temperature(np.array([15.0]))[0] == pytest.approx(216.65)
    assert us76_temperature(np.array([25.0]))[0] == pytest.approx(221.65)


def test_ozone_column_matches_du():
    from skycolor.constants import Z_TOA

    at = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[])
    z = np.linspace(0, Z_TOA / 1e3, 20000)
    col = np.trapezoid(at.n_o3(z), z * 1e3)
    assert col / DU == pytest.approx(300.0, rel=1e-6)


def test_rayleigh_tau():
    at = Atmosphere(WL_COARSE, aerosols=[])
    z = np.linspace(0, 120, 8000)
    tau = np.trapezoid(at.rayleigh_coefficient(z), z * 1e3, axis=0)
    i = np.argmin(abs(at.wl - 550))
    assert 0.095 < tau[i] < 0.099


def test_aerosol_column_normalisation():
    from skycolor.aerosol import HinterlandAerosol
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile

    at = Atmosphere(WL_COARSE, aerosols=[AerosolLayer(HinterlandAerosol(), 0.25,
                                                      ExponentialProfile(1.2))])
    beta = at.aerosol_coefficients(at.z_mid)[0]
    tau = np.sum(beta * np.diff(at.z_bounds)[:, None] * 1e3, axis=0)
    i = np.argmin(abs(at.wl - 550))
    assert tau[i] == pytest.approx(0.25, rel=0.02)


def test_phase_moments_normalised():
    at = Atmosphere.standard(WL_COARSE)
    m = at.phase_moments(8)
    assert np.allclose(m[:, 0, :], 1.0)


def test_elevation_rescales_aerosol_column():
    at = Atmosphere.standard(WL_COARSE)
    high = at.at_altitude(3.0)
    beta = high.aerosol_coefficients(high.z_mid)[0]
    tau = np.sum(beta * np.diff(high.z_bounds)[:, None] * 1e3, axis=0)
    i = np.argmin(abs(high.wl - 550))
    assert 0 < tau[i] < 0.05
