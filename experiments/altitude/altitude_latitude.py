"""Experiment 05 -- altitude and latitude.

Two things change when you climb a mountain: there is less air above you (so
the zenith gets darker and deeper blue) and more sky below the horizon (a
depressed horizon).  Latitude acts differently: it changes the *angle* at which
the sun crosses the horizon, and therefore how long twilight lasts.

The first half of the figure is radiative transfer, the second half is pure
geometry from the solar-position routine.
"""
from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiments.common import WL, chromaticity_of, save, strip

THEME = Path(__file__).resolve().parent          # experiments/<theme>
FIG = THEME / "figures"      # theme-owned figure output (not the root candidates)
from skycolor import color, render
from skycolor.color import K_M
from skycolor.geometry import sun_position
from skycolor.presets import clear


def altitude_panel():
    alts = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5]
    zens, lums, hor = [], [], []
    imgs = []
    for km in alts:
        atmo = clear(WL, aod=0.05).at_altitude(km)
        dome = render.sky_dome(atmo, 30.0, n_zenith=40, n_azimuth=37)
        xy, Lv = chromaticity_of(dome["radiance"][0, 0])
        zens.append(xy)
        lums.append(Lv)
        hor.append(K_M * color.spectrum_to_xyz(dome["radiance"][-1, -1], WL)[..., 1])
        imgs.append(render.render_scene(atmo, 4.0, 0.0, camera="perspective",
                                        size=(300, 240), hfov_deg=110, pitch_deg=-6,
                                        reference_percentile=92, n_zenith=40,
                                        n_azimuth=61))
    zens = np.array(zens)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(alts, zens[:, 1], "o-", label="zenith y (blue-ness)")
    ax.set_xlabel("observer altitude [km]")
    ax.set_ylabel("chromaticity y")
    ax2 = ax.twinx()
    ax2.semilogy(alts, lums, "s--", color="tab:red", label="zenith luminance")
    ax2.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.set_title("the sky from a mountain (sun 60° up)")
    save(fig, FIG / "05_altitude.png")
    render.save_png(np.concatenate(imgs, axis=1), FIG / "05_altitude_views.png")
    print(f"  figure -> {FIG / '05_altitude*.png'}")


def twilight_duration(lat, date, lo=-6.0, hi=0.0):
    """Longest continuous time the sun spends in ``[lo, hi)`` degrees, in hours."""
    t0 = dt.datetime(date.year, date.month, date.day, 0, 0)
    elev = np.array([sun_position(lat, 0.0, t0 + dt.timedelta(minutes=i))[1]
                     for i in range(1441)])
    mask = (elev >= lo) & (elev < hi)
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best / 60.0


def latitude_panel():
    lats = np.arange(-60, 80.1, 5.0)
    dates = {"equinox (Mar 20)": dt.date(2024, 3, 20),
             "June solstice": dt.date(2024, 6, 21),
             "December solstice": dt.date(2024, 12, 21)}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, d in dates.items():
        civil = [twilight_duration(lat, d) for lat in lats]
        ax.plot(lats, civil, "o-", ms=3, label=name)
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel("civil twilight duration [h]")
    ax.set_title("why twilight lasts longer near the poles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, FIG / "05_twilight_duration.png")


def main():
    altitude_panel()
    latitude_panel()
    print("done")


if __name__ == "__main__":
    main()
