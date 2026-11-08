"""Experiment 03 -- what the air does to the colour.

Sweeps the aerosol optical depth and the aerosol type at a fixed sun position,
and shows where on the chromaticity diagram the sky moves.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiments.common import WL, chromaticity_of, save, strip

THEME = Path(__file__).resolve().parent          # experiments/<theme>
FIG = THEME / "figures"      # theme-owned figure output (not the root candidates)
OUT = THEME / "out"          # theme-owned numeric output (gitignored)
from skycolor import color, presets, render
from skycolor.aerosol import (
    AngstromAerosol,
    DustAerosol,
    HazeAerosol,
    HinterlandAerosol,
    MaritimeAerosol,
    SmokeAerosol,
)
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile

AODS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6]
TYPES = {
    "continental (α=1.3)": HinterlandAerosol(),
    "humid haze (α=1.1)": HazeAerosol(0.8),
    "maritime (α=0.8)": MaritimeAerosol(),
    "desert dust (α=0.25)": DustAerosol(),
    "smoke (α=1.9)": SmokeAerosol(),
}


def main():
    sun_el = 25.0
    wl = WL

    # ---- AOD sweep, continental aerosol ------------------------------------
    zen_xy, hor_xy, hor_sun_xy, lum = [], [], [], []
    for aod in AODS:
        atmo = Atmosphere(wl, aerosols=[AerosolLayer(HinterlandAerosol(), aod,
                                                     ExponentialProfile(1.5))])
        dome = render.sky_dome(atmo, 90.0 - sun_el, n_zenith=40, n_azimuth=37)
        xy0, L0 = chromaticity_of(dome["radiance"][0, 0])
        xyh, Lh = chromaticity_of(dome["radiance"][-1, -1])      # anti-solar horizon
        xys, Ls = chromaticity_of(dome["radiance"][-1, 0])        # sun-side horizon
        zen_xy.append(xy0)
        hor_xy.append(xyh)
        hor_sun_xy.append(xys)
        lum.append((L0, Lh, Ls))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for arr, name in [(zen_xy, "zenith"), (hor_xy, "horizon, anti-solar"),
                      (hor_sun_xy, "horizon, sun side")]:
        a = np.array(arr)
        sc = axes[0].scatter(a[:, 0], a[:, 1], c=AODS, cmap="viridis", s=45,
                             edgecolor="k", linewidth=0.3)
        axes[0].annotate(name, a[-1], fontsize=8)
    plt.colorbar(sc, ax=axes[0], label="AOD at 550 nm")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"aerosol loading vs colour (sun {sun_el:.0f}° up)")
    axes[0].grid(alpha=0.3)
    lum = np.array(lum)
    axes[1].plot(AODS, lum[:, 0], "o-", label="zenith")
    axes[1].plot(AODS, lum[:, 1], "s-", label="horizon, anti-solar")
    axes[1].plot(AODS, lum[:, 2], "^-", label="horizon, sun side")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("AOD at 550 nm")
    axes[1].set_ylabel("luminance [cd m$^{-2}$]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_title("brightness vs aerosol loading")
    save(fig, FIG / "03_aerosol_loading.png")
    plt.close(fig)

    # ---- aerosol type comparison ------------------------------------------
    imgs = []
    for name, aer in TYPES.items():
        atmo = Atmosphere(wl, aerosols=[AerosolLayer(aer, 0.4, ExponentialProfile(1.4))])
        rgb = render.render_scene(atmo, sun_el, 0.0, camera="panorama", size=(620, 230),
                                  reference_percentile=92, n_zenith=40, n_azimuth=61)
        imgs.append(rgb)
        xy, L = chromaticity_of(render.sky_dome(atmo, 90 - sun_el, n_zenith=30,
                                                n_azimuth=25)["radiance"][0, 0])
        print(f"  {name:22s} zenith xy = ({xy[0]:.3f}, {xy[1]:.3f})  Lv={L:.0f} cd/m2")
    render.save_png(strip(imgs), FIG / "03_aerosol_types.png")
    print(f"  figure -> {FIG / '03_aerosol_types.png'}")

    # ---- pictures of increasing haze --------------------------------------
    imgs = []
    for aod in [0.02, 0.1, 0.3, 0.6, 1.0, 1.6]:
        atmo = Atmosphere(wl, aerosols=[AerosolLayer(HazeAerosol(0.75), aod,
                                                     ExponentialProfile(1.2))])
        imgs.append(render.render_scene(atmo, 8.0, 0.0, camera="perspective",
                                        size=(300, 320), hfov_deg=100, pitch_deg=18,
                                        reference_percentile=92, n_zenith=40,
                                        n_azimuth=61))
    render.save_png(np.concatenate(imgs, axis=1), FIG / "03_haze_progression.png")
    print(f"  figure -> {FIG / '03_haze_progression.png'}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "03_aerosol.npz", aod=AODS, zen_xy=np.array(zen_xy),
             hor_xy=np.array(hor_xy), hor_sun_xy=np.array(hor_sun_xy), lum=lum)
    print("done")


if __name__ == "__main__":
    main()
