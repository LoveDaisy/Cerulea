"""Experiment 06 -- the gallery.

A contact sheet of skies: the same landscape camera looking at the sun through
a day, and at a fixed sun position through a range of atmospheres.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from common import FIG, WL, row, strip
from skycolor import presets, render
from skycolor.aerosol import DustAerosol, HazeAerosol, MaritimeAerosol, SmokeAerosol
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile

W, H = 360, 240
CAM = dict(camera="perspective", size=(W, H), hfov_deg=100, pitch_deg=16,
           reference_percentile=93, n_zenith=44, n_azimuth=61, sun_disk=True)


def shot(atmo, sun_el, sun_az=0.0, **kw):
    return render.render_scene(atmo, sun_el, sun_az, **kw)


def main():
    clear = presets.clear(WL, aod=0.06)

    # ---- a day ------------------------------------------------------------
    day = []
    for el in [55, 25, 8, 2]:
        day.append(shot(clear, el))
    sunset = []
    for el in [0.5, -1.5, -4, -7]:
        sunset.append(shot(clear, el))
    render.save_png(strip([row(day), row(sunset)], gap=5), FIG / "06_gallery_day.png")
    print("  -> 06_gallery_day.png")

    # ---- atmospheres ------------------------------------------------------
    sky = []
    for aer, aod in [(None, 0.0), (HazeAerosol(0.4), 0.08), (HazeAerosol(0.8), 0.35),
                     (DustAerosol(), 0.8)]:
        if aer is None:
            atmo = Atmosphere(WL, aerosols=[])
        else:
            atmo = Atmosphere(WL, aerosols=[AerosolLayer(aer, aod, ExponentialProfile(1.3))])
        sky.append(shot(atmo, 12.0))
    sky2 = []
    for aer, aod in [(SmokeAerosol(), 0.5), (MaritimeAerosol(), 0.15),
                     (DustAerosol(), 2.0), (HazeAerosol(0.9), 1.5)]:
        atmo = Atmosphere(WL, aerosols=[AerosolLayer(aer, aod, ExponentialProfile(1.3))])
        sky2.append(shot(atmo, 12.0))
    render.save_png(strip([row(sky), row(sky2)], gap=5), FIG / "06_gallery_air.png")
    print("  -> 06_gallery_air.png")

    # ---- a big hero image --------------------------------------------------
    hero = render.render_scene(presets.clear(WL, aod=0.05), 3.0, 0.0,
                               camera="perspective", size=(1400, 560), hfov_deg=100,
                               pitch_deg=14, reference_percentile=94,
                               n_zenith=56, n_azimuth=91)
    render.save_png(hero, FIG / "06_hero_sunset.png")
    print("  -> 06_hero_sunset.png")

    hero2 = render.render_scene(presets.hazy(WL, aod=0.25, rh=0.7), 22.0, 0.0,
                                camera="perspective", size=(1400, 560), hfov_deg=100,
                                pitch_deg=14, reference_percentile=94,
                                n_zenith=56, n_azimuth=91)
    render.save_png(hero2, FIG / "06_hero_hazy.png")
    print("  -> 06_hero_hazy.png")


if __name__ == "__main__":
    main()
