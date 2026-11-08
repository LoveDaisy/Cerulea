"""The observing scenario: where the observer stands and what the ground does.

A ``Scene`` bundles the three quantities that describe *where the model is
being evaluated from* rather than what the atmosphere is made of:

* ``observer_elevation_km`` -- the radius the RT backends integrate from;
* ``ground_albedo`` -- the spectral Lambertian albedo below the observer;
* ``refraction`` -- whether the direct-beam geometry bends rays.

It is frozen: a scene is replaced wholesale, never patched.  The spectral
grid lives in the :class:`~skycolor.medium.Medium`; ``ground_albedo`` is
stored as an array already broadcast to the medium's wavelength grid by the
caller (:class:`skycolor.atmosphere.Atmosphere` does this in its
constructor), so the two can only desynchronise through the facade's
public setters, which re-broadcast as needed.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .geometry import sun_position


@dataclass(frozen=True)
class Scene:
    """The observing scenario (see the module docstring)."""

    observer_elevation_km: float = 0.0
    ground_albedo: np.ndarray | float = 0.1
    refraction: bool = True


# ---------------------------------------------------------------------------
# the one solar-input resolver (CLI and library convenience callers share it)
# ---------------------------------------------------------------------------
#: the historical CLI defaults for the numeric solar input.  They are filled
#: only when no ``time`` is given, so an argparse default can never masquerade
#: as an explicitly conflicting setting.
DEFAULT_SUN_ELEVATION = 10.0
DEFAULT_SUN_AZIMUTH = 180.0


def _check_finite(name: str, value, lo: float, hi: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if v < lo or v > hi:
        raise ValueError(f"{name} must be within [{lo}, {hi}] degrees, got {v}")
    return v


def _as_utc_naive(when) -> dt.datetime:
    """Normalise an ISO string or datetime to a naive UTC datetime.

    The ephemeris reads UTC clock fields, so a timezone-aware input is
    converted to UTC *before* the tzinfo is dropped: an explicit offset
    names a different physical instant than reading the clock fields
    literally.  A naive input is already interpreted as UTC.
    """
    if isinstance(when, str):
        try:
            when = dt.datetime.fromisoformat(when)
        except ValueError:
            raise ValueError(
                f"time must be an ISO 8601 string or a datetime, got {when!r}"
            ) from None
    if not isinstance(when, dt.datetime):
        raise ValueError(
            f"time must be an ISO 8601 string or a datetime, "
            f"got {type(when).__name__}")
    if when.tzinfo is not None:
        when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return when


def resolve_sun(sun_elevation: float | None = None,
                sun_azimuth: float | None = None, *,
                time=None,
                latitude: float | None = None,
                longitude: float | None = None) -> tuple[float, float]:
    """Resolve the solar direction from numbers or a time, pre-refraction.

    The one input contract for every production consumer (CLI and library):
    the returned ``(azimuth_deg, elevation_deg)`` pair describes the solar
    direction **before atmospheric refraction**, which is what
    ``SolarSource`` applies its own refraction treatment to.  Callers who
    used to ask the ephemeris for the ``refraction=True`` angle got a
    double-refracted scene (the geometric ``refraction_correction`` here,
    then the source's ray-traced bending); passing this function's output
    to the solver keeps refraction applied exactly once.

    Either a ``time`` (ISO 8601 string or :class:`datetime.datetime`) or an
    explicit numeric direction is given, never both:

    * ``time``: converted to naive UTC first (aware inputs via an explicit
      UTC conversion, so ``Z`` and ``+08:00`` name their own instants;
      naive means UTC), then evaluated with ``sun_position(...,
      refraction=False)`` at ``latitude``/``longitude`` (both required).
    * numbers: validated finite; elevation within [-90, 90], azimuth
      wrapped to [0, 360).  Missing numbers fall back to the historical
      CLI defaults (10 deg elevation, 180 deg azimuth) -- defaults are
      filled here rather than in argparse so a parser default can never
      look like an explicit conflict with ``time``.

    Raises
    ------
    ValueError
        Explicit angles together with ``time``; a malformed time; missing
        or out-of-range latitude/longitude with ``time``; non-finite or
        out-of-range angles.
    """
    if time is not None:
        if sun_elevation is not None or sun_azimuth is not None:
            raise ValueError(
                "explicit sun angles conflict with time: the solar direction "
                "comes either from the clock or from numbers, not both")
        if latitude is None or longitude is None:
            raise ValueError("latitude and longitude are required with time")
        lat = _check_finite("latitude", latitude, -90.0, 90.0)
        lon = _check_finite("longitude", longitude, -180.0, 180.0)
        return sun_position(lat, lon, _as_utc_naive(time), refraction=False)

    el = (DEFAULT_SUN_ELEVATION if sun_elevation is None
          else _check_finite("sun_elevation", sun_elevation, -90.0, 90.0))
    az = (DEFAULT_SUN_AZIMUTH if sun_azimuth is None
          else _check_finite("sun_azimuth", sun_azimuth, -1.0e9, 1.0e9) % 360.0)
    return float(az), float(el)
