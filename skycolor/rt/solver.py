"""High-level radiative transfer entry point."""
from __future__ import annotations

import numpy as np

from . import multiple as _multiple
from .single import sky_radiance as _single


def sky_radiance_total(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                       sun_azimuth_deg=0.0, multiple=True, single=True,
                       n_gauss=3, chunk=512):
    """Total spectral sky radiance (single + multiple scattering)."""
    out = None
    if single:
        out = _single(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                      sun_azimuth_deg=sun_azimuth_deg, n_gauss=n_gauss, chunk=chunk)
    if multiple:
        ms = _multiple.multiple_scattering(atmo, sun_zenith_deg, view_zenith_deg,
                                           view_azimuth_deg, sun_azimuth_deg=sun_azimuth_deg)
        out = ms if out is None else out + ms
    return out
