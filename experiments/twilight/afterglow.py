"""Experiment 08 -- where the red afterglow comes from.

A red twilight that fills a large part of the sky is not produced by haze near
the ground.  It needs a scattering layer that is still *sunlit* after the
ground is in shadow, and the height of that layer is the controlling variable.

The figure sweeps the altitude of a thin scattering layer at fixed optical
depth, at fixed solar elevation, and shows (a) the sky's chromaticity and
luminance along the sun's meridian and (b) what it looks like.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiments.common import WL, chromaticity_of, row, save

THEME = Path(__file__).resolve().parent          # experiments/<theme>
FIG = THEME / "figures"      # theme-owned figure output (not the root candidates)
OUT = THEME / "out"          # theme-owned numeric output (gitignored)
from skycolor import presets, render
from skycolor.aerosol import AngstromAerosol
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile, GaussianProfile

LAYER_AOD = 0.15
ALTS = [0.0, 3.0, 9.0, 15.0, 22.0, 30.0]


def scene(alt_km, aod=LAYER_AOD):
    if alt_km <= 0:
        lay = AerosolLayer(AngstromAerosol(alpha=0.6, ssa=0.98, g=0.65), aod,
                           ExponentialProfile(1.5))
    else:
        lay = AerosolLayer(AngstromAerosol(alpha=0.6, ssa=0.98, g=0.65), aod,
                           GaussianProfile(alt_km, max(1.0, alt_km * 0.18)))
    return Atmosphere(WL, aerosols=[lay])


def main():
    sun_elev = -4.0
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    imgs = []
    for alt in ALTS:
        atmo = scene(alt)
        dome = render.sky_dome(atmo, 90.0 - sun_elev, n_zenith=44, n_azimuth=61)
        elev = 90.0 - dome["zenith"]
        xy, Lv = chromaticity_of(dome["radiance"])
        axes[0].plot(elev, xy[:, 0].mean(axis=1), label=f"layer at {alt:.0f} km")
        axes[1].semilogy(elev, Lv.mean(axis=1), label=f"layer at {alt:.0f} km")
        img = render.render_scene(atmo, sun_elev, 0.0, camera="perspective",
                                  size=(300, 200), hfov_deg=110, pitch_deg=12,
                                  reference_percentile=96, n_zenith=44, n_azimuth=61)
        imgs.append(img)
    axes[0].set_xlabel("elevation above the horizon [deg]")
    axes[0].set_ylabel("chromaticity x (azimuth average)")
    axes[0].set_title(f"colour of a {LAYER_AOD:.2f} AOD layer vs its altitude (sun {sun_elev:+.0f}°)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("elevation above the horizon [deg]")
    axes[1].set_ylabel("luminance [cd m$^{-2}$]")
    axes[1].set_title("brightness")
    axes[1].grid(alpha=0.3)
    save(fig, FIG / "08_afterglow_curve.png")

    grid = np.concatenate([row(imgs[:3]), row(imgs[3:])], axis=0)
    render.save_png(grid, FIG / "08_afterglow_gallery.png")
    print(f"  figure -> {FIG / '08_afterglow_gallery.png'}")

    # a volcano: a strong stratospheric layer at 22 km
    volc = scene(22.0, aod=0.35)
    clean = Atmosphere(WL, aerosols=[])
    imgs = []
    for se in [-1, -3, -5, -7]:
        imgs.append(render.render_scene(clean, se, 0.0, camera="perspective",
                                        size=(320, 220), hfov_deg=110, pitch_deg=12,
                                        reference_percentile=96, n_zenith=44, n_azimuth=61))
        imgs.append(render.render_scene(volc, se, 0.0, camera="perspective",
                                        size=(320, 220), hfov_deg=110, pitch_deg=12,
                                        reference_percentile=96, n_zenith=44, n_azimuth=61))
    render.save_png(np.concatenate([row(imgs[:4]), row(imgs[4:])], axis=0),
                    FIG / "08_volcanic_twilight.png")
    print(f"  figure -> {FIG / '08_volcanic_twilight.png'}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "08_afterglow.npz", alts=ALTS, aod=LAYER_AOD, sun_elev=sun_elev)
    print("done")


if __name__ == "__main__":
    main()
