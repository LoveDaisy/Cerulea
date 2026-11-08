"""The unified presentation layer over one solver result (task plan A1-A7).

Contract under test, by milestone:

1. the one solar input contract: ``resolve_sun`` returns the
   pre-refraction direction, converts aware times to naive UTC before the
   ephemeris reads them, and rejects conflicting/invalid inputs;
2. one solve per request: ``solve_dome`` -> ``SkyPresentation`` binds the
   grid, the solar display geometry and the matching ground illumination at
   preparation time; ``presentation_from_result`` only reshapes/binds a
   ready result; after preparation every output (chart tables, panorama,
   fisheye, perspective) is a pure function of the bound arrays -- solves
   and solar-source queries are forbidden and spied to be absent;
3. scene-consistent lighting: clear/cloud ground direct irradiance bound at
   preparation, the cloud helper living in exactly one place, external
   fields refusing unmatched ground lighting, albedo and wavelength
   snapshots immune to later facade mutation, and the decorative sun disk
   consuming the adopted source's apparent geometry;
4. (CLI-level: see ``tests/test_cli.py``).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor.scene import resolve_sun  # noqa: E402

BEIJING = (39.9, 116.4)


# ---------------------------------------------------------------------------
# 1. the one solar input contract (M1)
# ---------------------------------------------------------------------------
def test_naive_utc_z_and_offset_are_the_same_instant():
    """Naive means UTC; 'Z' and an explicit offset convert before the
    ephemeris reads the clock fields -- the same physical instant gives
    bit-identical angles."""
    naive = dt.datetime(2026, 11, 5, 0, 0)
    z = resolve_sun(time="2026-11-05T00:00:00Z", latitude=BEIJING[0],
                    longitude=BEIJING[1])
    offset = resolve_sun(time="2026-11-05T08:00:00+08:00",
                         latitude=BEIJING[0], longitude=BEIJING[1])
    aware = resolve_sun(time=naive.replace(tzinfo=dt.timezone.utc),
                        latitude=BEIJING[0], longitude=BEIJING[1])
    plain = resolve_sun(time=naive, latitude=BEIJING[0], longitude=BEIJING[1])
    assert z == offset == aware == plain


def test_time_resolution_is_pre_refraction():
    """The resolved direction is the ``refraction=False`` ephemeris angle;
    the retired ``refraction=True`` path visibly double-refracted."""
    from skycolor.geometry import sun_position

    naive = dt.datetime(2026, 11, 5, 0, 0)
    az, el = resolve_sun(time=naive, latitude=BEIJING[0], longitude=BEIJING[1])
    assert az == sun_position(BEIJING[0], BEIJING[1], naive, refraction=False)[0]
    assert el == sun_position(BEIJING[0], BEIJING[1], naive, refraction=False)[1]
    _, el_refracted = sun_position(BEIJING[0], BEIJING[1], naive, refraction=True)
    assert abs(el_refracted - el) > 0.01   # the semantic gap is real, not a rewording


def test_ephemeris_is_called_without_refraction():
    """Spy: resolve_sun must request the geometric (pre-refraction) angle."""
    import skycolor.scene as scene

    seen = {}
    real = scene.sun_position

    def spy(lat, lon, when, refraction=False):
        seen["refraction"] = refraction
        seen["when"] = when
        return real(lat, lon, when, refraction=refraction)

    seen_ref = scene.sun_position
    scene.sun_position = spy
    try:
        resolve_sun(time="2026-11-05T12:00:00+00:00", latitude=0.0,
                    longitude=0.0)
    finally:
        scene.sun_position = seen_ref
    assert seen["refraction"] is False
    assert seen["when"].tzinfo is None           # naive UTC reaches the ephemeris


def test_time_and_numbers_conflict():
    with pytest.raises(ValueError, match="conflict"):
        resolve_sun(sun_elevation=10.0, time="2026-11-05T00:00:00Z",
                    latitude=0.0, longitude=0.0)
    with pytest.raises(ValueError, match="conflict"):
        resolve_sun(sun_azimuth=137.0, time="2026-11-05T00:00:00Z",
                    latitude=0.0, longitude=0.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="ISO 8601"):
        resolve_sun(time="not-a-time", latitude=0.0, longitude=0.0)
    with pytest.raises(ValueError, match="latitude and longitude"):
        resolve_sun(time="2026-11-05T00:00:00Z")
    with pytest.raises(ValueError, match="latitude"):
        resolve_sun(time="2026-11-05T00:00:00Z", latitude=91.0, longitude=0.0)
    with pytest.raises(ValueError, match="longitude"):
        resolve_sun(time="2026-11-05T00:00:00Z", latitude=0.0, longitude=181.0)
    with pytest.raises(ValueError, match="finite"):
        resolve_sun(sun_elevation=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        resolve_sun(sun_azimuth=float("inf"))
    with pytest.raises(ValueError, match="sun_elevation"):
        resolve_sun(sun_elevation=90.5)


def test_numeric_defaults_and_azimuth_wrap():
    """Without time the historical CLI defaults fill in -- once, here, not
    in argparse defaults that would masquerade as explicit conflicts."""
    assert resolve_sun() == (180.0, 10.0)
    assert resolve_sun(sun_elevation=0.0) == (180.0, 0.0)
    az, el = resolve_sun(sun_elevation=-12.0, sun_azimuth=-180.0)
    assert az == 180.0 and el == -12.0
    az, _ = resolve_sun(sun_azimuth=450.0)
    assert az == 90.0
