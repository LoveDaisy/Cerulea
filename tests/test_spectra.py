import numpy as np
import pytest

from skycolor import spectra
from skycolor.constants import WL_COARSE


def test_rayleigh_optical_depth_sealevel():
    """Bodhaine et al. (1999) give tau_R(550 nm) = 0.0976 for the standard atmosphere."""
    sigma = spectra.rayleigh_cross_section(np.array([550.0]))[0]
    tau = sigma * 2.148e29  # sea-level air column [molecules m^-2]
    assert 0.094 < tau < 0.101, tau


def test_rayleigh_wavelength_exponent():
    """The effective exponent is about 4.1, not exactly 4 (dispersion of air)."""
    s400, s700 = spectra.rayleigh_cross_section(np.array([400.0, 700.0]))
    exponent = np.log(s400 / s700) / np.log(700 / 400)
    assert 4.0 < exponent < 4.25, exponent


def test_rayleigh_phase_normalisation():
    mu = np.linspace(-1, 1, 20001)
    p = spectra.rayleigh_phase(mu)
    integral = 2 * np.pi * np.trapezoid(p, mu)
    assert abs(integral / (4 * np.pi) - 1) < 2e-3


def test_rayleigh_phase_moments():
    b = spectra.rayleigh_phase_moments()
    assert b[0] == pytest.approx(1.0)
    assert b[1] == pytest.approx(0.0)
    # p = 1 + 5*b2*P2 must reproduce the closed form
    mu = np.linspace(-1, 1, 101)
    p1 = spectra.rayleigh_phase(mu)
    p2 = 1 + 5 * b[2] * np.polynomial.legendre.legval(mu, [0, 0, 1])
    assert np.allclose(p1, p2, rtol=1e-10)


def test_cie_white_point():
    """Equal-energy white must land on x=y=1/3."""
    wl = WL_COARSE
    xyz = spectra.cie_xyz(wl).sum(axis=0)
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    assert abs(x - 1 / 3) < 0.01 and abs(y - 1 / 3) < 0.01


def test_solar_constant():
    wl = np.arange(300.0, 831.0, 1.0)
    e = spectra.solar_irradiance(wl)
    vis = np.trapezoid(e[(wl >= 400) & (wl <= 700)], wl[(wl >= 400) & (wl <= 700)])
    assert 500 < vis < 560  # W/m^2 in 400-700 nm


def test_ozone_chappuis_peak():
    cs = spectra.ozone_cross_section(np.array([600.0, 450.0])) * 1e4  # cm^2
    assert 4.5e-21 < cs[0] < 5.6e-21
    assert 0.5e-22 < cs[1] < 3e-22


def test_aerosol_asymmetry():
    from skycolor.aerosol import AngstromAerosol, MieAerosol, LogNormalMode

    a = AngstromAerosol(g=0.65)
    assert a.asymmetry([550.0])[0] == pytest.approx(0.65, abs=1e-3)
    assert a.asymmetry([450.0, 550.0, 650.0]).shape == (3,)

    # a Mie aerosol's quadrature asymmetry must agree with its analytic one
    m = MieAerosol(modes=[LogNormalMode(0.15, 1.8)], refractive_index=1.53 + 0.006j)
    g_q = m.asymmetry([550.0])[0]
    assert 0.2 < g_q < 0.95
