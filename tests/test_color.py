import numpy as np
import pytest

from skycolor import color, spectra
from skycolor.constants import WL_COARSE


def test_d65_maps_to_white():
    """The D65 white point must map to linear sRGB (1, 1, 1)."""
    xyz = color.xy_to_xyz((0.31271, 0.32902))
    rgb = color.xyz_to_linear_srgb(xyz)
    assert np.allclose(rgb, 1.0, rtol=1e-3)


def test_equal_energy_white_is_neutral():
    """The CIE CMFs integrate to x = y = 1/3 for an equal-energy spectrum."""
    xyz = spectra.cie_xyz(WL_COARSE).sum(axis=0)
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    assert abs(x - 1 / 3) < 0.01 and abs(y - 1 / 3) < 0.01


def test_spectrum_to_xyz_shapes():
    rad = np.ones((4, 5, len(WL_COARSE)))
    out = color.spectrum_to_xyz(rad, WL_COARSE)
    assert out.shape == (4, 5, 3)


def test_gamma_roundtrip():
    x = np.linspace(0, 1, 100)
    y = color.srgb_encode(x)
    assert y[0] == 0.0 and abs(y[-1] - 1.0) < 1e-9
    assert np.all(np.diff(y) >= 0)


def test_render_rgb_range():
    wl = WL_COARSE
    rad = np.ones((3, len(wl))) * 0.01 * spectra.solar_irradiance(wl)
    rgb = color.render_rgb(rad, wl)
    assert rgb.shape == (3, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_adaptation_preserves_neutral():
    xyz = np.array([0.95047, 1.0, 1.08883])
    out = color.bradford_adapt(xyz[None, :], [0.95047, 1.0, 1.08883])
    assert np.allclose(out, xyz, rtol=1e-6)
