"""Rendering: build a sky dome, map it onto cameras and write images."""
from __future__ import annotations

import numpy as np

from . import color
from .constants import WL_COARSE


def sky_dome(atmo, sun_zenith_deg: float, n_zenith: int = 46, n_azimuth: int = 91,
             n_gauss: int = 3, chunk: int = 512, min_zenith_deg: float = 0.0):
    """Compute the radiance of the whole sky on a (zenith, relative azimuth) grid.

    The atmosphere is horizontally homogeneous, so the radiance only depends on
    the angle between the viewing direction and the sun.  The grid covers
    relative azimuths 0..180 degrees.

    Returns a dict with ``zenith`` (deg), ``azimuth_rel`` (deg), ``radiance``
    ``(n_zenith, n_azimuth, n_wl)`` and ``wl``.
    """
    zen = np.linspace(min_zenith_deg, 90.0, n_zenith)
    az = np.linspace(0.0, 180.0, n_azimuth)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    rad = sky_radiance_grid(atmo, sun_zenith_deg, Z, A, n_gauss=n_gauss, chunk=chunk)
    return dict(zenith=zen, azimuth_rel=az, radiance=rad, wl=atmo.wl,
                sun_zenith_deg=sun_zenith_deg)


def sky_radiance_grid(atmo, sun_zenith_deg, Z, A, n_gauss=3, chunk=512, sun_azimuth_deg=0.0):
    from .rt.single import sky_radiance

    shape = np.shape(Z)
    rad = sky_radiance(atmo, sun_zenith_deg, np.ravel(Z), np.ravel(A) + sun_azimuth_deg,
                       sun_azimuth_deg=sun_azimuth_deg, n_gauss=n_gauss, chunk=chunk)
    return rad.reshape(shape + (rad.shape[-1],))


def sample_dome(dome: dict, view_zenith, view_azimuth_rel) -> np.ndarray:
    """Bilinear interpolation of a sky dome onto new directions."""
    zen = dome["zenith"]
    az = dome["azimuth_rel"]
    rad = dome["radiance"]
    zv = np.clip(view_zenith, zen[0], zen[-1])
    av = np.clip(np.abs(view_azimuth_rel), az[0], az[-1])
    iz = np.clip(np.searchsorted(zen, zv) - 1, 0, len(zen) - 2)
    ia = np.clip(np.searchsorted(az, av) - 1, 0, len(az) - 2)
    fz = ((zv - zen[iz]) / (zen[iz + 1] - zen[iz]))[..., None]
    fa = ((av - az[ia]) / (az[ia + 1] - az[ia]))[..., None]
    r00 = rad[iz, ia]
    r01 = rad[iz, ia + 1]
    r10 = rad[iz + 1, ia]
    r11 = rad[iz + 1, ia + 1]
    return (r00 * (1 - fz) * (1 - fa) + r01 * (1 - fz) * fa
            + r10 * fz * (1 - fa) + r11 * fz * fa)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def panorama_grid(width: int, height: int, min_elevation: float = -10.0):
    """Equirectangular panorama: x -> azimuth, y -> elevation."""
    azimuth = np.linspace(0.0, 360.0, width, endpoint=False)
    elevation = np.linspace(90.0, min_elevation, height)
    A, E = np.meshgrid(azimuth, elevation, indexing="xy")
    return A, E


def fisheye_grid(size: int, max_zenith: float = 90.0, projection: str = "equal_area"):
    """Circular fisheye image.

    ``equal_area`` maps solid angle linearly onto the disk (a "hemispherical
    fisheye"), ``equidistant`` maps zenith angle linearly onto the radius.
    Returns ``(azimuth, elevation, mask)``.
    """
    y, x = np.mgrid[0:size, 0:size].astype(float)
    cx = cy = (size - 1) / 2.0
    r = np.hypot(x - cx, y - cy) / cx
    mask = r <= 1.0
    if projection == "equal_area":
        zen = np.degrees(np.arcsin(np.clip(r, 0, 1)))
    else:
        zen = r * max_zenith
    azimuth = (np.degrees(np.arctan2(x - cx, -(y - cy)))) % 360.0
    return azimuth, 90.0 - zen, mask


def render_from_dome(dome, sun_azimuth_deg, azimuth, elevation, mask=None,
                     exposure=None, tonemap="aces", adaptation="bradford",
                     white_xy=None, reference_percentile=99.0, sun_disk=False):
    """Render an image from a pre-computed dome."""
    rel = (azimuth - sun_azimuth_deg + 180.0) % 360.0 - 180.0
    rad = sample_dome(dome, 90.0 - elevation, rel)
    if white_xy is None:
        white_xy = color._D65_xy
    rgb = color.render_rgb(rad, dome["wl"], exposure=exposure, tonemap=tonemap,
                           adaptation=adaptation, white_xy=white_xy,
                           reference_percentile=reference_percentile)
    if sun_disk:
        sun_zen = dome.get("sun_zenith_deg", 90.0)
        sun_elev = 90.0 - sun_zen
        ang = np.hypot(((azimuth - sun_azimuth_deg + 180) % 360 - 180) * np.cos(np.radians(elevation)),
                       elevation - sun_elev)
        disk = ang < 0.266
        rgb[disk] = 1.0
    if mask is not None:
        rgb = rgb * mask[..., None]
    return rgb


def save_png(rgb: np.ndarray, path, background=None):
    from PIL import Image

    arr = color.to_uint8(rgb)
    if background is not None and arr.shape[-1] == 3:
        alpha = (np.abs(arr).sum(axis=-1) > 0).astype(np.uint8) * 255
    im = Image.fromarray(arr)
    im.save(path)
    return path
