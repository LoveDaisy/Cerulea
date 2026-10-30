"""Experiment 07 -- a sunset time lapse.

Renders a sequence of frames at a fixed exposure (so the darkening is real and
not compensated by the automatic exposure) and writes an animated GIF plus a
contact sheet.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from common import FIG, WL
from skycolor import presets, render

W, H = 420, 260
FRAMES = 26
SUN_FROM, SUN_TO = 8.0, -8.0
#: the eye (and a camera's auto exposure) dark-adapts; without this the last
#: frames would be black.  The model itself has no such adaptation.
ADAPT_DECADES = 1.9


def main():
    atmo = presets.clear(WL, aod=0.05)
    # one exposure for the whole sequence: the sky at 5 degrees above the
    # horizon defines the reference
    ref = render.render_scene(atmo, 5.0, 0.0, camera="perspective", size=(W, H),
                              hfov_deg=110, pitch_deg=18, sun_disk=True,
                              reference_percentile=95, n_zenith=50, n_azimuth=71)
    from skycolor import color
    # recover an absolute exposure by rendering once with exposure=1 and
    # taking the same percentile
    dome = render.sky_dome(atmo, 85.0, n_zenith=50, n_azimuth=71)
    rel = (render.perspective_grid(W, H, 110, 18, 0)[0])
    el = render.perspective_grid(W, H, 110, 18, 0)[1]
    rad = render.sample_dome(dome, 90.0 - np.clip(el, 0, 90), rel)
    xyz = color.spectrum_to_xyz(rad, atmo.wl)
    rgb = color.xyz_to_linear_srgb(xyz)
    lum = np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 0)
    exposure = 1.0 / np.percentile(lum[lum > 0], 95)
    print(f"  fixed exposure = {exposure:.3g}")

    frames = []
    for se in np.linspace(SUN_FROM, SUN_TO, FRAMES):
        ramp = 10.0 ** (ADAPT_DECADES * max(-float(se), 0.0) / abs(SUN_TO))
        img = render.render_scene(atmo, float(se), 0.0, camera="perspective",
                                  size=(W, H), hfov_deg=110, pitch_deg=18,
                                  sun_disk=se > -0.9, exposure=exposure * ramp,
                                  n_zenith=50, n_azimuth=71)
        frames.append(img)
    render.save_png(np.concatenate([frames[0], frames[-1]], axis=1),
                    FIG / "07_sunset_ends.png")

    from PIL import Image

    ims = [Image.fromarray(color.to_uint8(f)) for f in frames]
    ims[0].save(FIG / "07_sunset.gif", save_all=True, append_images=ims[1:],
                duration=120, loop=0)
    print("  -> figures/07_sunset.gif")


if __name__ == "__main__":
    main()
