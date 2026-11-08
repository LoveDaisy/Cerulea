"""Sky colour simulation: spectral radiative transfer of the atmosphere.

Quick start::

    from skycolor.presets import clear
    from skycolor import render
    from skycolor.constants import WL_COARSE

    atmo = clear(WL_COARSE)                      # clean mid-latitude air
    pres = render.solve_dome(atmo, sun_elevation_deg=5.0)
    img = render.render_result(pres, camera="perspective", size=(900, 500))

One request, one solve: ``solve_dome`` prepares the result for display
(grid, ground lighting, apparent sun geometry); ``render_result``,
``chart_table`` and ``format_result_summary`` are pure consumers of the
prepared presentation.
"""
from .atmosphere import AerosolLayer, Atmosphere, ExponentialProfile, GaussianProfile
from .constants import R_EARTH, WL_COARSE, WL_DEFAULT
from . import color, presets, render, spectra  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Atmosphere", "AerosolLayer", "ExponentialProfile", "GaussianProfile",
           "WL_DEFAULT", "WL_COARSE", "R_EARTH", "color", "presets", "render", "spectra"]
