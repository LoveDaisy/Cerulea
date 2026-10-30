"""Experiment 02 -- twilight and the blue hour.

The sun goes below the horizon; the sky does not switch off, it changes
character in stages.  This script traces the zenith brightness through
twilight, maps where the earth's shadow falls on the sky, and renders the
sequence as fisheye views.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG, OUT, WL, chromaticity_of, save, strip
from skycolor import color, presets, render

#: Clear-sky zenith luminance in the V band, from the empirical fit of
#: Patat, Ugolnikov & Postylyakov (2006), A&A 455, 385 (the reference used in
#: ``validation/check5_twilight.py``).  A factor-of-two band is drawn around it.
PATAT_2006 = {-3.0: 0.0, -6.0: 0.516, -9.0: 0.0171, -12.0: 0.00146, -18.0: 0.000181}
LIT = [(k, v / 2, v * 2) for k, v in PATAT_2006.items() if v > 0]


def main():
    atmo = presets.clear(WL, aod=0.08)
    depressions = np.arange(0.0, 18.01, 0.25)
    elevations = -depressions
    zen_lum, zen_xy = [], []
    hor_lum = {0: [], 90: [], 180: []}
    for se in elevations:
        dome = render.sky_dome(atmo, 90.0 - se, n_zenith=36, n_azimuth=25)
        xy, Lv = chromaticity_of(dome["radiance"][0, 0])
        zen_lum.append(Lv)
        zen_xy.append(xy)
        for az in hor_lum:
            j = np.argmin(abs(dome["azimuth_rel"] - az))
            k = np.argmin(abs(dome["zenith"] - 88.0))
            hor_lum[az].append(683 * color.spectrum_to_xyz(
                dome["radiance"][k, j], WL)[..., 1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].semilogy(depressions, zen_lum, "k-", lw=2, label="model, zenith")
    for e, lo, hi in LIT:
        axes[0].plot([-e, -e], [lo, hi], "o-", color="tab:red", ms=6, lw=2,
                     label="Patat et al. 2006 (x2 band)" if e == -6 else None)
    for name, e in [("civil", -6), ("nautical", -12), ("astronomical", -18)]:
        axes[0].axvline(-e, color="0.7", ls="--", lw=0.8)
        axes[0].text(-e + 0.15, 1e3, name, rotation=90, va="top", fontsize=7, color="0.4")
    axes[0].set_xlabel("solar depression below the horizon [deg]")
    axes[0].set_ylabel("zenith luminance [cd m$^{-2}$]")
    axes[0].set_xlim(-0.2, 18.2)
    axes[0].set_ylim(1e-4, 3e3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("how bright is twilight?")
    axes[0].grid(True, alpha=0.25)

    for az, label in [(0, "towards the sun"), (90, "perpendicular"), (180, "anti-solar")]:
        axes[1].semilogy(depressions, hor_lum[az], label=label)
    axes[1].set_xlabel("solar depression below the horizon [deg]")
    axes[1].set_xlim(-0.2, 18.2)
    axes[1].set_ylabel("horizon luminance [cd m$^{-2}$]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("horizon brightness through twilight")
    axes[1].grid(True, alpha=0.25)
    save(fig, "02_twilight_luminance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    xy = np.array(zen_xy)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=depressions, cmap="turbo", s=12)
    plt.colorbar(sc, label="solar depression [deg]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("zenith colour through twilight")
    ax.grid(alpha=0.3)
    save(fig, "02_twilight_zenith_xy.png")
    plt.close(fig)

    imgs = []
    for se in [-1, -2, -3, -4, -5, -6, -7, -8, -10]:
        rgb = render.render_scene(atmo, se, 0.0, camera="fisheye", size=(330,),
                                  reference_percentile=99.5, n_zenith=40,
                                  n_azimuth=61, sun_disk=False)
        imgs.append(rgb)
    n = len(imgs)
    half = (n + 1) // 2
    rows = [np.concatenate(imgs[:half], axis=1)]
    second = imgs[half:]
    if len(second) < half:
        second = second + [np.zeros_like(imgs[0])] * (half - len(second))
    rows.append(np.concatenate(second, axis=1))
    render.save_png(strip(rows, gap=6), FIG / "02_twilight_fisheye.png")
    print("  figure -> figures/02_twilight_fisheye.png")

    np.savez(OUT / "02_twilight.npz", depressions=depressions, zen_lum=zen_lum,
             zen_xy=np.array(zen_xy),
             **{f"hor_{k}": np.array(v) for k, v in hor_lum.items()})
    print("done")


if __name__ == "__main__":
    main()
