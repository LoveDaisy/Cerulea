"""From spectral radiance to pixels.

Three separate operations are involved and it is worth keeping them apart,
because the choice you make changes what you see:

1. **Colorimetry** -- integrate the spectrum against the CIE 1931 colour
   matching functions to get XYZ.  This is objective.
2. **Adaptation** -- the eye (and a camera's auto white balance) adapts to the
   prevailing illumination.  A sunset sky is physically orange; it looks orange
   mostly because everything else is orange too.  We offer von Kries
   adaptation to a chosen white, or none at all for absolute colorimetry.
3. **Tone mapping** -- the sky spans 4-5 orders of magnitude in luminance, a
   display spans about two.  A tone curve decides which part of the range you
   get to see.

Every function here takes radiance in W m^-2 sr^-1 nm^-1 and wavelengths in nm.
"""
from __future__ import annotations

import numpy as np

from . import spectra
from .optics import contiguous_transpose as _cT


# ---------------------------------------------------------------------------
# colorimetry
# ---------------------------------------------------------------------------
def spectrum_to_xyz(radiance: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    """Integrate spectra against the CIE 1931 colour matching functions.

    ``radiance`` may have shape ``(..., n_wl)``; the output is ``(..., 3)``.
    """
    cmf = spectra.cie_xyz(wl_nm)  # (n_wl, 3)
    return np.trapezoid(np.asarray(radiance)[..., None] * cmf, wl_nm, axis=-2)


def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    return xyz @ _cT(m)


def linear_srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return rgb @ _cT(m)


def luminance(xyz) -> np.ndarray:
    return xyz[..., 1]


def chromaticity(xyz) -> np.ndarray:
    xyz = np.asarray(xyz)
    s = xyz.sum(axis=-1, keepdims=True)
    return xyz[..., :2] / np.maximum(s, 1e-30)


# ---------------------------------------------------------------------------
# white point / adaptation
# ---------------------------------------------------------------------------
_D65 = np.array([0.95047, 1.0, 1.08883])
_D65_xy = np.array([0.31271, 0.32902])
#: the D65 white point in chromaticity coordinates -- the public spelling of
#: ``_D65_xy`` for callers outside this module (render, experiments)
D65_XY = _D65_xy


def xy_to_xyz(xy, Y: float = 1.0) -> np.ndarray:
    x, y = xy
    return np.array([x / y * Y, Y, (1.0 - x - y) / y * Y])


def von_kries_adapt(xyz: np.ndarray, src_white, dst_white=_D65) -> np.ndarray:
    """von Kries (diagonal) chromatic adaptation from ``src_white`` to ``dst_white``."""
    src = np.asarray(src_white, dtype=float)
    dst = np.asarray(dst_white, dtype=float)
    d = src / np.maximum(dst, 1e-12)
    return xyz * d


def _rgb_to_lms(rgb):
    m = np.array([[0.38971, 0.68898, -0.07868],
                  [-0.22981, 1.18340, 0.04641],
                  [0.0, 0.0, 1.0]])
    return rgb @ _cT(m)


def _lms_to_rgb(lms):
    m = np.linalg.inv(np.array([[0.38971, 0.68898, -0.07868],
                               [-0.22981, 1.18340, 0.04641],
                               [0.0, 0.0, 1.0]]))
    return lms @ _cT(m)


def bradford_adapt(xyz: np.ndarray, src_white, dst_white=_D65) -> np.ndarray:
    """Bradford chromatic adaptation (closer to the human eye than von Kries)."""
    m = np.array([[0.8951, 0.2664, -0.1614],
                  [-0.7502, 1.7135, 0.0367],
                  [0.0389, -0.0685, 1.0296]])
    mi = np.linalg.inv(m)
    s = m @ np.asarray(src_white, dtype=float)
    d = m @ np.asarray(dst_white, dtype=float)
    return (xyz @ _cT(m) * (d / np.maximum(s, 1e-12))) @ _cT(mi)


# ---------------------------------------------------------------------------
# tone mapping
# ---------------------------------------------------------------------------
def tonemap_linear(x):
    return np.asarray(x)


def tonemap_reinhard(x, key: float = 1.0):
    x = np.maximum(np.asarray(x), 0.0)
    return x / (1.0 + x / key)


def tonemap_reinhard_extended(x, white: float = 4.0):
    x = np.maximum(np.asarray(x), 0.0)
    return x * (1.0 + x / white**2) / (1.0 + x)


def tonemap_aces(x):
    """ACES filmic approximation (Narkowicz 2015)."""
    x = np.maximum(np.asarray(x), 0.0) * 0.6
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap_hable(x):
    x = np.maximum(np.asarray(x), 0.0)
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    curve = lambda v: ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F  # noqa: E731
    return curve(x) / curve(np.array(11.2))


TONEMAPS = {
    "linear": tonemap_linear,
    "reinhard": tonemap_reinhard,
    "reinhard_ext": tonemap_reinhard_extended,
    "aces": tonemap_aces,
    "hable": tonemap_hable,
}


def srgb_encode(linear_rgb: np.ndarray) -> np.ndarray:
    """sRGB transfer function (linear -> display)."""
    x = np.clip(linear_rgb, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# the full pipeline
# ---------------------------------------------------------------------------
def render_rgb(
    radiance: np.ndarray,
    wl_nm: np.ndarray,
    exposure: float | None = None,
    tonemap: str = "aces",
    adaptation: str | None = "bradford",
    white_xy=_D65_xy,
    reference_percentile: float = 99.0,
    preserve_chromaticity: bool = True,
) -> np.ndarray:
    """Turn spectral radiance into an 8-bit-displayable RGB image.

    Parameters
    ----------
    radiance : ``(..., n_wl)``
    exposure : float, optional
         Multiplies the radiance before tone mapping.  If ``None`` an exposure
        is chosen automatically so that the ``reference_percentile`` of the
        luminance maps to 1.0.
    tonemap : {"aces", "reinhard", "reinhard_ext", "hable", "linear"}
    adaptation : {None, "von_kries", "bradford"}
        Chromatic adaptation applied so that the *white point of the scene*
        maps to the display white.  ``None`` gives absolute colorimetry.
    white_xy : (x, y)
        Scene white point used for the adaptation.  Set this to the chromaticity
        of the direct sunlight (or of a sunlit white surface) to emulate a
        camera's white balance.
    preserve_chromaticity : bool
        If True the tone curve is applied to luminance only and the chroma is
        preserved (good for scientific work).  If False each RGB channel is
        tone-mapped independently (what a camera's film curve does; clipping
        towards white).

    Returns
    -------
    ndarray of shape ``(..., 3)`` in display sRGB, values in [0, 1].
    """
    radiance = np.asarray(radiance, dtype=float)
    xyz = spectrum_to_xyz(radiance, wl_nm)
    rgb = xyz_to_linear_srgb(xyz)

    if adaptation is not None:
        src = xy_to_xyz(white_xy, Y=1.0)
        if adaptation == "von_kries":
            xyz = von_kries_adapt(xyz, src)
        else:
            xyz = bradford_adapt(xyz, src)
        rgb = xyz_to_linear_srgb(xyz)

    if exposure is None:
        lum = np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 0.0)
        pos = lum[lum > 0]
        ref = np.percentile(pos, reference_percentile) if pos.size else 1.0
        exposure = 1.0 / max(ref, 1e-12)
    rgb = rgb * exposure

    if preserve_chromaticity:
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        curve = TONEMAPS[tonemap]
        lum_out = curve(lum)
        scale = np.where(lum > 1e-12, lum_out / np.maximum(lum, 1e-12), 0.0)
        rgb = rgb * scale[..., None] if rgb.ndim > 1 else rgb * scale
    else:
        rgb = TONEMAPS[tonemap](rgb)
    return np.clip(srgb_encode(rgb), 0.0, 1.0)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
