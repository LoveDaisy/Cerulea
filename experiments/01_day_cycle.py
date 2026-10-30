"""Experiment 01 -- one day, the colour of the sky.

Plots the colour of three sky directions (zenith, horizon on the sun side,
horizon on the anti-solar side) as the sun goes from -18 to +70 degrees,
on a CIE 1931 chromaticity diagram, plus the luminance curve and a panorama
strip showing what it actually looks like.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG, OUT, WL, chromaticity_of, save, strip
from skycolor import color, presets, render

# ---------------------------------------------------------------------------
# CIE diagram background
# ---------------------------------------------------------------------------
def draw_cie(ax):
    from skycolor import spectra

    x, y = np.meshgrid(np.linspace(0, 0.8, 400), np.linspace(0, 0.9, 450))
    with np.errstate(divide="ignore", invalid="ignore"):
        XYZ = np.stack([x / y, np.ones_like(x), (1 - x - y) / y], axis=-1)
        rgb = color.xyz_to_linear_srgb(XYZ)
    inside = (rgb.min(axis=-1) > -0.02) & (x + y < 1)
    loc = spectra.cie_xyz(np.arange(380, 781, 1.0))
    wl = np.arange(380, 781, 1.0)
    locus = loc / loc.sum(axis=1, keepdims=True)
    ax.fill(locus[:, 0], locus[:, 1], color="none")
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    path = Path(np.vstack([locus[::4, :2], [locus[0, 0], locus[0, 1]]]))
    patch = PathPatch(path, facecolor="none", edgecolor="0.4", lw=0.8)
    ax.add_patch(patch)
    clip = patch
    with np.errstate(invalid="ignore"):
        tint = np.clip(np.nan_to_num(rgb) ** (1 / 2.2), 0, 1)
    im = ax.imshow(tint, extent=[0, 0.8, 0, 0.9], origin="lower", alpha=0.35)
    im.set_clip_path(clip)
    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def main():
    atmo = presets.clear(WL, aod=0.08)
    elevations = np.arange(-18.0, 71.0, 1.0)
    dirs = {"zenith": (1.0, 0.0), "horizon, sun side": (88.0, 0.0),
            "horizon, anti-solar": (88.0, 180.0)}
    traj = {k: [] for k in dirs}
    lum = {k: [] for k in dirs}
    for se in elevations:
        dome = render.sky_dome(atmo, 90.0 - se, n_zenith=40, n_azimuth=37)
        for name, (zen, az) in dirs.items():
            i = np.argmin(abs(dome["zenith"] - zen))
            j = np.argmin(abs(dome["azimuth_rel"] - az))
            xy, Lv = chromaticity_of(dome["radiance"][i, j])
            traj[name].append(xy)
            lum[name].append(Lv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    draw_cie(axes[0])
    for name, (zen, az) in dirs.items():
        xy = np.array(traj[name])
        m = elevations > -18
        sc = axes[0].scatter(xy[m, 0], xy[m, 1], c=elevations[m], cmap="twilight_shifted",
                             s=8, vmin=-18, vmax=70)
        axes[0].annotate(name, xy[-1], fontsize=8)
    plt.colorbar(sc, ax=axes[0], label="solar elevation [deg]")
    axes[0].set_title("colour of the sky through the day")

    for name in dirs:
        axes[1].semilogy(elevations, lum[name], label=name)
    axes[1].axvspan(-18, -6, color="navy", alpha=0.08, label="astronomical / nautical")
    axes[1].axvspan(-6, 0, color="orange", alpha=0.10, label="civil twilight")
    axes[1].set_xlabel("solar elevation [deg]")
    axes[1].set_ylabel("luminance [cd m$^{-2}$]")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("sky luminance")
    save(fig, "01_day_cycle_chromaticity.png")
    plt.close(fig)

    # visual strip
    imgs = []
    for se in [60, 40, 20, 10, 5, 2, 0, -2, -4, -6, -10]:
        rgb = render.render_scene(atmo, se, 0.0, camera="panorama", size=(620, 210),
                                  reference_percentile=92, sun_disk=True,
                                  n_zenith=40, n_azimuth=61)
        imgs.append(rgb)
    render.save_png(strip(imgs), FIG / "01_day_cycle_panorama.png")
    print("  figure -> figures/01_day_cycle_panorama.png")

    np.savez(OUT / "01_day_cycle.npz", elevations=elevations,
             **{f"xy_{k.replace(' ', '_')}": np.array(v) for k, v in traj.items()},
             **{f"lum_{k.replace(' ', '_')}": np.array(v) for k, v in lum.items()})
    print("done")


if __name__ == "__main__":
    main()
