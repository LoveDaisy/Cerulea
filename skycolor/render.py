"""Rendering: build a sky dome, map it onto cameras and write images."""
from __future__ import annotations

import numpy as np

from . import color


# ---------------------------------------------------------------------------
# sky dome
# ---------------------------------------------------------------------------
def sky_dome(atmo, sun_zenith_deg: float, n_zenith: int = 46, n_azimuth: int = 91,
             n_gauss: int = 3, chunk: int = 512, min_zenith_deg: float = 0.0,
             multiple: bool = True, radiance_field=None):
    """Radiance of the whole sky on a (zenith, relative azimuth) grid.

    The atmosphere is horizontally homogeneous, so the radiance only depends on
    the angle between the viewing direction and the sun; the grid covers
    relative azimuths 0..180 degrees.  Returns a dict with ``zenith``,
    ``azimuth_rel``, ``radiance`` (n_zenith, n_azimuth, n_wl), ``wl`` and
    ``sun_zenith_deg``.

    ``radiance_field`` decouples the display from the solver: a callable
    ``(zen_flat, az_rel_flat) -> (n_view, n_wl)`` consulted instead of the
    built-in clear-sky solver.  The medium and the solar geometry are bound
    in the caller's closure, so the display layer needs no notion of what
    produced the field (a cloudy sky, an experiment, a validation harness).
    With a field given, ``atmo`` is used only for the ``wl`` metadata.
    """
    zen = np.linspace(min_zenith_deg, 90.0, n_zenith)
    az = np.linspace(0.0, 180.0, n_azimuth)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    rad = sky_radiance_grid(atmo, sun_zenith_deg, Z, A, n_gauss=n_gauss,
                            chunk=chunk, multiple=multiple,
                            radiance_field=radiance_field)
    return dict(zenith=zen, azimuth_rel=az, radiance=rad, wl=atmo.wl,
                sun_zenith_deg=sun_zenith_deg)


def sky_radiance_grid(atmo, sun_zenith_deg, Z, A, n_gauss=3, chunk=512,
                      sun_azimuth_deg=0.0, multiple=True, radiance_field=None):
    if radiance_field is not None:
        rad = np.asarray(radiance_field(np.ravel(Z), np.ravel(A)), dtype=float)
        return rad.reshape(np.shape(Z) + (rad.shape[-1],))

    from .rt.solver import sky_radiance_total

    shape = np.shape(Z)
    rad = sky_radiance_total(atmo, sun_zenith_deg, np.ravel(Z),
                             np.ravel(A) + sun_azimuth_deg,
                             sun_azimuth_deg=sun_azimuth_deg, multiple=multiple,
                             n_gauss=n_gauss, chunk=chunk)
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


def hemispherical_irradiance(radiance: np.ndarray, zenith_deg, azimuth_rel_deg):
    """Diffuse irradiance on a horizontal surface from a sky dome.

    ``E = int L cos(theta) dOmega``.  With ``mu = cos(theta)`` and
    ``dOmega = -dmu dphi`` this is ``int int L mu dmu dphi``.  The ``mu`` factor
    is easy to forget and doubles the answer if missing -- see
    ``tests/test_render.py``.
    """
    mu = np.cos(np.radians(np.asarray(zenith_deg, dtype=float)))
    phi = np.radians(np.asarray(azimuth_rel_deg, dtype=float))
    # integrate over mu first (mu is decreasing, hence the sign)
    over_mu = np.abs(np.trapezoid(np.asarray(radiance) * mu[:, None, None], mu, axis=0))
    return np.trapezoid(over_mu, phi, axis=0) * 2.0  # 0..pi -> full circle


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def panorama_grid(width: int, height: int, min_elevation: float = -8.0,
                  max_elevation: float = 90.0):
    azimuth = np.linspace(0.0, 360.0, width, endpoint=False)
    elevation = np.linspace(max_elevation, min_elevation, height)
    A, E = np.meshgrid(azimuth, elevation, indexing="xy")
    return A, E


def fisheye_grid(size: int, max_zenith: float = 90.0, projection: str = "equal_area",
                 center=None):
    y, x = np.mgrid[0:size, 0:size].astype(float)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    if center is not None:
        cx, cy = center
    r = np.hypot(x - cx, y - cy) / ((size - 1) / 2.0)
    mask = r <= 1.0
    if projection == "equal_area":
        zen = np.degrees(np.arcsin(np.clip(r, 0, 1)))
    else:
        zen = r * max_zenith
    azimuth = (np.degrees(np.arctan2(x - cx, -(y - cy)))) % 360.0
    return azimuth, 90.0 - zen, mask


def perspective_grid(width: int, height: int, hfov_deg: float = 90.0,
                     pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """Pinhole camera looking towards ``yaw`` at ``pitch`` (degrees)."""
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    x, y = np.meshgrid(np.arange(width) + 0.5 - width / 2.0,
                       height / 2.0 - (np.arange(height) + 0.5), indexing="xy")
    dx, dy, dz = x / f, y / f, np.ones_like(x)
    n = np.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx / n, dy / n, dz / n
    p = np.radians(pitch_deg)
    v = np.radians(yaw_deg)
    dy2 = dy * np.cos(p) + dz * np.sin(p)   # +pitch looks up
    dz2 = -dy * np.sin(p) + dz * np.cos(p)
    dx3 = dx * np.cos(v) + dz2 * np.sin(v)
    dz3 = -dx * np.sin(v) + dz2 * np.cos(v)
    elevation = np.degrees(np.arcsin(np.clip(dy2, -1, 1)))
    azimuth = (np.degrees(np.arctan2(dx3, dz3))) % 360.0
    return azimuth, elevation


# ---------------------------------------------------------------------------
# high level renderer
# ---------------------------------------------------------------------------
def ground_radiance(atmo, dome, sun_zenith_deg, ground_albedo=None,
                    direct_transmittance=None, *, direct_irradiance=None):
    """Spectral radiance of a Lambertian ground illuminated by sun + sky.

    The default direct term is the unified source's horizontal direct
    irradiance (finite disk, refracted, occultation-aware) at the observer's
    ground.  ``direct_irradiance`` overrides it wholesale -- e.g. a cloud
    deck's beam-attenuated irradiance, so the display of a cloudy scene keeps
    its physically darkened ground without reaching into the solver.

    ``direct_transmittance`` is the retired centre-beam override: it still
    works for explicit old callers (multiplied by F*cos(z) exactly as before)
    but is deprecated; passing it together with ``direct_irradiance`` is an
    error, and nothing defaults to it.
    """
    import warnings

    from .rt.single import direct_sun_transmittance
    from .solar_source import SolarSource
    from .spectra import solar_irradiance

    if direct_irradiance is not None and direct_transmittance is not None:
        raise ValueError(
            "pass either direct_irradiance or the deprecated "
            "direct_transmittance, not both")
    albedo = (atmo.ground_albedo if ground_albedo is None
              else np.full(len(atmo.wl), ground_albedo))
    E_diff = hemispherical_irradiance(dome["radiance"], dome["zenith"], dome["azimuth_rel"])
    if direct_irradiance is not None:
        E_dir = np.broadcast_to(np.asarray(direct_irradiance, dtype=float),
                                (len(atmo.wl),)).copy()
    elif direct_transmittance is not None:
        warnings.warn(
            "direct_transmittance is deprecated by the unified solar source; "
            "pass direct_irradiance instead (it carries the disk geometry)",
            DeprecationWarning, stacklevel=2)
        T = np.asarray(direct_transmittance, dtype=float)
        cosz = max(np.cos(np.radians(sun_zenith_deg)), 0.0)
        E_dir = solar_irradiance(atmo.wl) * T * cosz
    else:
        source = SolarSource(atmo, float(sun_zenith_deg), 0.0)
        x_ground = np.array([[0.0, 0.0, atmo.r_bounds[0] + (atmo.elevation_km - atmo.z_bounds[0]) * 1e3]])
        E_dir = np.asarray(source.direct_irradiance(x_ground, np.array([0.0, 0.0, 1.0]))[0],
                           dtype=float)
    return np.asarray(albedo) / np.pi * (E_dir + E_diff), E_dir + E_diff


def render_scene(atmo, sun_elevation_deg: float, sun_azimuth_deg: float = 0.0,
                 camera: str = "panorama", size=(1000, 500), exposure=None,
                 tonemap: str = "aces", adaptation=None,
                 reference_percentile: float = 92.0, sun_disk: bool = True,
                 ground: bool = True, n_zenith: int = 50, n_azimuth: int = 91,
                 hfov_deg: float = 90.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 white_xy=None, preserve_chromaticity: bool = True,
                 radiance_field=None, ground_direct_transmittance=None,
                 *, ground_direct_irradiance=None):
    """Render a complete scene (sky + ground + sun disk) to an RGB image.

    ``radiance_field`` / ``ground_direct_irradiance`` have the same meaning
    as in :func:`sky_dome` / :func:`ground_radiance`: they let a cloudy or
    otherwise non-default spectral field flow through the same display path
    instead of a duplicated one.  The pair is coupled: a non-default
    ``radiance_field`` usually needs a matching ``ground_direct_irradiance``
    (a cloudy sky also attenuates the direct beam that lights the ground) --
    leaving the ground default pairs a custom sky with a clear-sky beam.
    ``ground_direct_transmittance`` is the retired override; see
    :func:`ground_radiance` for its deprecation semantics.
    """
    sun_zen = 90.0 - sun_elevation_deg
    dome = sky_dome(atmo, sun_zen, n_zenith=n_zenith, n_azimuth=n_azimuth,
                    radiance_field=radiance_field)

    if camera == "panorama":
        az, el = panorama_grid(size[0], size[1])
        mask = None
    elif camera == "fisheye":
        az, el, mask = fisheye_grid(size[0])
    elif camera == "perspective":
        az, el = perspective_grid(size[0], size[1], hfov_deg, pitch_deg, yaw_deg)
        mask = None
    else:
        raise ValueError(f"unknown camera {camera}")

    rel = (az - sun_azimuth_deg + 180.0) % 360.0 - 180.0
    above = el >= 0.0
    rad = sample_dome(dome, 90.0 - np.clip(el, 0.0, 90.0), rel)
    if ground:
        Lg, _ = ground_radiance(atmo, dome, sun_zen,
                                direct_transmittance=ground_direct_transmittance,
                                direct_irradiance=ground_direct_irradiance)
        rad = np.where(above[..., None], rad, Lg[None, None, :])

    if white_xy is None:
        white_xy = color.D65_XY
    if adaptation is None:
        white_xy = color.D65_XY
    rgb = color.render_rgb(rad, dome["wl"], exposure=exposure, tonemap=tonemap,
                           adaptation=adaptation, white_xy=white_xy,
                           reference_percentile=reference_percentile,
                           preserve_chromaticity=preserve_chromaticity)

    if sun_disk and sun_elevation_deg > -0.9:
        ang = np.hypot(((az - sun_azimuth_deg + 180) % 360 - 180) * np.cos(np.radians(el)),
                       el - sun_elevation_deg)
        rgb[ang < 0.30] = 1.0
    if mask is not None:
        rgb = rgb * mask[..., None]
    return rgb


def save_png(rgb: np.ndarray, path):
    from PIL import Image

    Image.fromarray(color.to_uint8(rgb)).save(path)
    return path


def save_hdr(radiance: np.ndarray, wl, path):
    np.savez_compressed(path, radiance=radiance, wl=wl)
    return path
