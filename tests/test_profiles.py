"""Data-path custom profiles: one physical definition across all backends.

Custom gas profiles enter as callables (``ozone_profile`` / ``air_profile``)
injected into the frozen ``Medium``.  These tests pin the contract that
motivated them: the same physical definition must reach the single-scattering
solver, the SOS multiple-scattering solver and the twilight engine's
wavelength subsets, with the requested column normalisation -- without any
subclassing or runtime patching (the inheritance protocol these replaced is
retired).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor.atmosphere import Atmosphere  # noqa: E402
from skycolor.constants import DU, Z_TOA  # noqa: E402
from skycolor.rt import multiple  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import twilight_radiance  # noqa: E402

WL = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])


def _gaussian_ozone(sigma_km=6.0, center_km=25.0):
    c, s = float(center_km), float(sigma_km)
    return lambda z: np.exp(-0.5 * ((np.asarray(z, dtype=float) - c) / s) ** 2)


@pytest.fixture
def custom():
    return Atmosphere(WL, ozone_du=250.0, ozone_profile=_gaussian_ozone(),
                      aerosols=[])


@pytest.fixture
def default():
    return Atmosphere(WL, ozone_du=250.0, aerosols=[])


def test_column_normalisation_matches_du(custom):
    """A callable shape is column-normalised to the requested ozone_du."""
    z = np.linspace(0, Z_TOA / 1e3, 20000)
    col = np.trapezoid(custom.n_o3(z), z * 1e3)
    assert col / DU == pytest.approx(250.0, rel=1e-6)


def test_profile_reaches_single_and_sos(custom, default):
    """single and SOS see the injected shape, not the climatological one."""
    z_mid = custom.z_mid
    # SOS layer optics are built from the injected density
    beta_o3 = custom.ozone_coefficient(z_mid)
    expected = np.multiply.outer(custom.n_o3(z_mid), custom._sigma_o3)
    assert np.array_equal(beta_o3, expected)
    # and the two atmospheres are genuinely different in every backend
    assert not np.array_equal(custom.n_o3(z_mid), default.n_o3(z_mid))
    assert not np.array_equal(
        sky_radiance_total(custom, 60.0, 30.0, 120.0),
        sky_radiance_total(default, 60.0, 30.0, 120.0))


def test_twilight_single_uses_same_profile(custom):
    """The twilight engine's deterministic term (its wavelength subsets go
    through ``_restrict``) is bit-identical to the single-scattering solver
    on the same injected profile."""
    zen = np.array([0.0, 45.0, 85.0])
    az = np.array([0.0, 60.0, 180.0])
    rad, err = twilight_radiance(custom, 96.0, zen, az, single=True,
                                 multiple=False)
    ref = sky_radiance_total(custom, 96.0, zen, az, multiple=False)
    assert np.array_equal(rad, ref)
    assert np.all(err == 0.0)


def test_twilight_ms_sees_injected_profile(custom, default):
    """The Monte Carlo multiple-scattering term reflects the injected ozone
    (historically this needed a runtime patch of the engine)."""
    # low-budget but seeded: deterministic for a given seed
    rad_custom, _ = twilight_radiance(custom, 96.0, np.array([0.0, 70.0]),
                                      np.array([0.0, 180.0]), quality="low",
                                      grid=dict(n_zen=6, n_az=8, n_anchor=3,
                                                n_photons=6.0e5), seed=0)
    rad_default, _ = twilight_radiance(default, 96.0, np.array([0.0, 70.0]),
                                       np.array([0.0, 180.0]), quality="low",
                                       grid=dict(n_zen=6, n_az=8, n_anchor=3,
                                                 n_photons=6.0e5), seed=0)
    assert not np.array_equal(rad_custom, rad_default)
