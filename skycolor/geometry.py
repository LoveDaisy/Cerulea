"""Geometry: solar position, coordinate frames and spherical-shell ray tracing.

The atmosphere is modelled as a set of concentric shells.  For a straight ray
in a spherically symmetric atmosphere every quantity of interest can be
expressed in terms of the ray's *impact parameter* ``p`` and the distance ``u``
measured **from the perigee** of the ray (the point of closest approach to the
Earth's centre).  That parametrisation makes the per-shell path lengths exact
and cheap, and it treats the "sun ray that grazes the Earth" correctly, which
is what twilight is all about.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np

from .constants import R_EARTH


# ===========================================================================
# Sun position
# ===========================================================================
def _julian_day(when: _dt.datetime) -> float:
    """Julian day (UT) from a timezone-naive UTC datetime."""
    y, m = when.year, when.month
    d = (when.day + when.hour / 24.0 + when.minute / 1440.0
         + when.second / 86400.0 + when.microsecond / 86400e6)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def sun_position(latitude_deg: float, longitude_deg: float,
                 when: _dt.datetime, refraction: bool = False) -> tuple[float, float]:
    """Solar azimuth and elevation in degrees.

    Implements the NOAA solar-position algorithm (Meeus, *Astronomical
    Algorithms*), accurate to about 0.01 degrees for dates between 1900 and
    2100.  ``when`` must be UTC.  Azimuth is measured clockwise from north.
    """
    jd = _julian_day(when)
    T = (jd - 2451545.0) / 36525.0

    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)

    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    seconds = 21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))
    eps0 = 23.0 + (26.0 + seconds / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))

    decl = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(app_long))))

    y = np.tan(np.radians(eps / 2.0)) ** 2
    eot = 4.0 * np.degrees(
        y * np.sin(2 * np.radians(L0))
        - 2 * e * np.sin(Mr)
        + 4 * e * y * np.sin(Mr) * np.cos(2 * np.radians(L0))
        - 0.5 * y * y * np.sin(4 * np.radians(L0))
        - 1.25 * e * e * np.sin(2 * Mr)
    )

    minutes = when.hour * 60 + when.minute + when.second / 60.0
    true_solar_time = (minutes + eot + 4.0 * longitude_deg) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0

    lat = np.radians(latitude_deg)
    dec = np.radians(decl)
    ha = np.radians(hour_angle)
    cos_zen = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    zenith = np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))

    # NOAA azimuth, clockwise from north
    denom = np.cos(lat) * np.sin(np.radians(zenith))
    if abs(denom) < 1e-12:
        az = 0.0
    else:
        cos_az = ((np.sin(dec) - np.sin(lat) * np.cos(np.radians(zenith)))
                  / denom)
        az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
        if hour_angle > 0:
            az = 360.0 - az
    elevation = 90.0 - zenith
    if refraction:
        elevation = elevation + refraction_correction(elevation)
    return float(az % 360.0), float(elevation)


def refraction_correction(elev_deg: float) -> float:
    """Atmospheric refraction [degrees] (Saemundsson 1986), for elevation > -1.5 deg."""
    h = float(elev_deg)
    if h < -1.5:
        return 0.0
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0


def sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector towards the sun in a local east/north/up frame."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


# ===========================================================================
# Spherical shell ray geometry
# ===========================================================================
def shell_lengths_upward(p: float, r_bounds: np.ndarray, u_from, u_to) -> np.ndarray:
    """Per-shell path lengths along a ray, between two distances from the perigee.

    Both ``u_from`` and ``u_to`` are measured **from the perigee** along the
    outgoing branch of the ray, so ``u_to > u_from`` means the segment goes
    away from the Earth.  The returned array has shape ``(n_shell, n_points)``.
    """
    rb = np.asarray(r_bounds, dtype=float)
    s = np.sqrt(np.maximum(rb[:, None] ** 2 - np.atleast_1d(p) ** 2, 0.0))  # (ns+1, np)
    a = np.maximum(np.minimum(u_from, u_to), s[:-1])
    b = np.minimum(np.maximum(u_from, u_to), s[1:])
    return np.maximum(b - a, 0.0)


def path_lengths_sun(p_sun, cos_zenith_point, r_point, r_bounds) -> np.ndarray:
    """Per-shell path lengths of the sun ray from a point out to space.

    When the sun is below the local horizontal (``cos_zenith_point < 0``) the
    ray that reaches the point first descends to its perigee and then climbs
    out on the far side.  The light traverses two branches:

    * the far branch, perigee -> space, and
    * the near branch, perigee -> the point.

    Both are part of the optical depth.  ``up`` is the part of the near branch
    above the point, ``down`` the part below it; their sum is one complete
    perigee -> space traversal.  The far branch is the mirror image of that
    traversal about the perigee, so it contributes the same length shell by
    shell.  The full sun ray is therefore ``(up + down) + down``.  Counting
    only ``up + down`` stops at the perigee and halves the optical depth of
    every grazing twilight ray.

    Consumption note: in production this fix (M-registry ``M10``,
    ``docs/ROADMAP.md``) is exercised by the MC retired branch
    (``rt/mc.py``) and the tests; the deterministic backends consume the
    equivalent columns through ``solar_source``.  The fix itself is pinned --
    do not re-derive it.

    Returns ``(n_shell, n_points)``.
    """
    rb = np.asarray(r_bounds, dtype=float)
    p = np.atleast_1d(np.asarray(p_sun, dtype=float))
    c = np.atleast_1d(np.asarray(cos_zenith_point, dtype=float))
    r = np.atleast_1d(np.asarray(r_point, dtype=float))
    u_point = (r * np.abs(c))[None, :]
    s = np.sqrt(np.maximum(rb[:, None] ** 2 - p[None, :] ** 2, 0.0))  # (ns+1, np)
    s0, s1 = s[:-1], s[1:]
    up = np.maximum(s1 - np.maximum(s0, u_point), 0.0)
    down = np.maximum(np.minimum(s1, u_point) - np.minimum(s0, u_point), 0.0)
    return up + np.where(c[None, :] < 0.0, 2.0 * down, 0.0)


def is_sunlit(p_sun, cos_zenith_point, r_earth: float = R_EARTH) -> np.ndarray:
    """True where the geometric ray towards the sun does not hit the Earth.

    A ray that starts upwards (``cos_zenith >= 0``) never intersects the
    sphere, whatever its impact parameter.  A downward ray is blocked when its
    perigee radius ``p`` is smaller than the Earth radius.
    """
    p = np.asarray(p_sun, dtype=float)
    c = np.asarray(cos_zenith_point, dtype=float)
    return (c >= 0.0) | (p > r_earth)


def path_lengths_sun_array(p, cos_zenith_point, r_point, r_bounds) -> np.ndarray:
    """Vectorised :func:`path_lengths_sun` for arrays of arbitrary shape.

    The two-branch geometry of :func:`path_lengths_sun` applies unchanged:
    when the sun is below the local horizontal the near-side
    ``perigee -> point`` segment is included, hence the factor two on
    ``down``.

    Returns an array of shape ``(n_shell,) + np.shape(p)``.
    """
    rb = np.asarray(r_bounds, dtype=float)
    p = np.asarray(p, dtype=float)
    c = np.asarray(cos_zenith_point, dtype=float)
    r = np.asarray(r_point, dtype=float)
    u_point = r * np.abs(c)
    shape = (len(rb) - 1,) + p.shape
    s = np.sqrt(np.maximum(rb.reshape((-1,) + (1,) * p.ndim) ** 2 - p[None] ** 2, 0.0))
    s0, s1 = s[:-1], s[1:]
    up = np.maximum(s1 - np.maximum(s0, u_point[None]), 0.0)
    down = np.maximum(np.minimum(s1, u_point[None]) - np.minimum(s0, u_point[None]), 0.0)
    return up + np.where(c[None] < 0.0, 2.0 * down, 0.0)
