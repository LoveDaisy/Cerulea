import datetime as dt

import numpy as np
import pytest

from skycolor.constants import R_EARTH
from skycolor.geometry import (
    path_lengths_sun,
    shell_lengths_upward,
    sun_position,
    is_sunlit,
)


def test_zenith_path_lengths_are_shell_thicknesses():
    rb = np.array([R_EARTH, R_EARTH + 1000, R_EARTH + 3000, R_EARTH + 120000.0])
    L = shell_lengths_upward(0.0, rb, R_EARTH, R_EARTH + 120000.0).ravel()
    assert np.allclose(L, [1000, 2000, 117000])


def test_vertical_sun_path_decreases_with_height():
    rb = np.linspace(R_EARTH, R_EARTH + 120000, 121)
    z = np.array([0.0, 10.0, 40.0])
    r = R_EARTH + z * 1e3
    L = path_lengths_sun(np.zeros(3), np.ones(3), r, rb)
    total = L.sum(axis=0)
    assert total[0] > total[1] > total[2] > 0


def test_sun_path_below_horizon_includes_both_branches():
    """A downward sun ray crosses the perigee: far branch + near branch."""
    rb = np.array([R_EARTH, R_EARTH + 2000, R_EARTH + 5000,
                   R_EARTH + 20000, R_EARTH + 60000, R_EARTH + 120000.0])
    r = R_EARTH + 40000.0
    chi = np.radians(96.0)
    p = r * np.sin(chi)
    c = np.cos(chi)
    L = path_lengths_sun(np.array([p]), np.array([c]), np.array([r]), rb)[:, 0]

    s = np.sqrt(np.maximum(rb**2 - p**2, 0.0))
    u = r * abs(c)
    # far branch: a full chord in every shell; near branch: perigee -> point
    expected = (s[1:] - s[:-1]) + (np.minimum(s[1:], u) - np.minimum(s[:-1], u))
    assert np.allclose(L, expected, rtol=1e-12)
    # one branch alone (the old behaviour) is measurably shorter
    assert expected.sum() > 1.4 * (s[-1] - s[0])


def test_sun_path_above_horizon_is_single_branch():
    """A ray that starts upwards never reaches the perigee; no extra branch."""
    rb = np.array([R_EARTH, R_EARTH + 2000, R_EARTH + 5000,
                   R_EARTH + 20000, R_EARTH + 60000, R_EARTH + 120000.0])
    r = R_EARTH + 40000.0
    chi = np.radians(80.0)
    p = r * np.sin(chi)
    c = np.cos(chi)
    L = path_lengths_sun(np.array([p]), np.array([c]), np.array([r]), rb)[:, 0]

    s = np.sqrt(np.maximum(rb**2 - p**2, 0.0))
    u = r * abs(c)
    expected = np.maximum(s[1:] - np.maximum(s[:-1], u), 0.0)
    assert np.allclose(L, expected, rtol=1e-12)


def test_sunlit_criterion():
    # looking up -> always lit, even with a tiny impact parameter
    assert is_sunlit(np.array([1e5]), np.array([0.5]))
    # a downward ray with p < R hits the Earth
    assert not is_sunlit(np.array([R_EARTH - 1.0]), np.array([-0.1]))
    # a downward ray that grazes above the surface is lit
    assert is_sunlit(np.array([R_EARTH + 100.0]), np.array([-0.1]))


def test_sun_position_epoch():
    """Solar noon at Beijing on the June solstice: elevation ~73.3 deg."""
    az, el = sun_position(39.9, 116.4, dt.datetime(2024, 6, 21, 4, 0))
    assert el == pytest.approx(73.3, abs=0.6)
    assert az == pytest.approx(180.0, abs=15.0)


def test_sun_position_solstice_winter():
    az, el = sun_position(39.9, 116.4, dt.datetime(2024, 12, 21, 4, 0))
    assert el == pytest.approx(90 - 39.9 - 23.44, abs=0.8)


def test_sun_position_polar_day():
    """At the north pole in June the sun never sets."""
    for hour in range(24):
        _, el = sun_position(89.9, 0.0, dt.datetime(2024, 6, 21, hour))
        assert el > 0
