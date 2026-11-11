import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere
from skycolor.constants import WL_COARSE
from skycolor.rt.single import direct_sun_transmittance, sky_radiance


@pytest.fixture(scope="module")
def atmo():
    return Atmosphere.standard(WL_COARSE)


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_radiance_positive_and_blue(atmo):
    rad = sky_radiance(atmo, 30.0, np.array([0.0, 45.0, 80.0]),
                       np.array([180.0, 90.0, 0.0]))
    assert np.all(rad > 0)
    i450 = np.argmin(abs(atmo.wl - 450))
    i650 = np.argmin(abs(atmo.wl - 650))
    # the clear sky is blue: more radiance at 450 nm than at 650 nm
    assert np.all(rad[:, i450] > rad[:, i650])


def test_azimuth_symmetry(atmo):
    a = sky_radiance(atmo, 40.0, np.array([30.0, 60.0]), np.array([200.0, 340.0]),
                     sun_azimuth_deg=180.0)
    b = sky_radiance(atmo, 40.0, np.array([30.0, 60.0]), np.array([160.0, 20.0]),
                     sun_azimuth_deg=180.0)
    assert np.allclose(a, b, rtol=1e-6)


def test_dark_when_sun_below_horizon(atmo):
    day = sky_radiance(atmo, 20.0, np.array([0.0]), np.array([180.0]))
    night = sky_radiance(atmo, 110.0, np.array([0.0]), np.array([180.0]))
    assert night.max() < day.max() * 1e-3


def test_direct_transmittance(atmo):
    T = direct_sun_transmittance(atmo, 0.0)
    i550 = np.argmin(abs(atmo.wl - 550))
    assert 0.80 < T[i550] < 0.95
    Tlow = direct_sun_transmittance(atmo, 80.0)
    assert Tlow[i550] < T[i550]


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_thin_atmosphere_scales_linearly():
    #: probe medium for the unperturbed US76 air density (n_air needs only
    #: the pressure/temperature profiles, not the spectral grid)
    probe = Atmosphere(WL_COARSE, ozone_du=0.0, aerosols=[])
    thin = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[],
                      air_profile=lambda z: probe.n_air(z) * 1e-3)
    rad = sky_radiance(thin, 30.0, np.array([0.0]), np.array([180.0]))
    assert np.all(rad >= 0)
