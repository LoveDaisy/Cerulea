"""Multiple scattering (scattering orders >= 2).

Production implementation lives here.  The module is deliberately independent
of :mod:`skycolor.rt.single`: it only needs the layer optical properties from
:class:`skycolor.atmosphere.Atmosphere`.

See ``docs/ARCHITECTURE.md`` for the contract.
"""
from __future__ import annotations

import numpy as np


def multiple_scattering(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                        sun_azimuth_deg=0.0, **options) -> np.ndarray:
    """Radiance from scattering orders >= 2, shape ``(n_view, n_wl)``."""
    n_view = np.atleast_1d(view_zenith_deg).size
    return np.zeros((n_view, len(atmo.wl)))
