"""Experiment 04 -- the weather is not the same everywhere.

Convective weather makes the aerosol field horizontally inhomogeneous.  A thin
haze layer two hundred kilometres away brightens and whitens the horizon while
the zenith stays a clean deep blue.  That contrast is one of the most
recognisable features of a real summer sky, and a one-dimensional radiative
transfer model simply cannot produce it.

The model attaches a radial weight to the aerosol amount
(:class:`skycolor.atmosphere.HorizontalField`), which is applied at every
scattering point, so this is treated with the full spherical single-scattering
integral.

**Caveat.**  Only the first-order term knows about the horizontal field: the
multiple-scattering solver is horizontally homogeneous and uses the layer's
nominal optical depth.  The multiple-scattering contribution therefore dilutes
the contrast shown here.  The true contrast is somewhere between the
single-scattering-only images (very strong) and what is plotted.
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
from skycolor import color, render
from skycolor.aerosol import HazeAerosol
from skycolor.atmosphere import (
    AerosolLayer,
    Atmosphere,
    ExponentialProfile,
    RadialSplitField,
    UniformField,
)

CASES = {
    "uniform clean (AOD 0.05)": (0.05, None),
    "uniform hazy (AOD 0.35)": (0.35, None),
    "clean here, hazy far": (0.35, RadialSplitField(90.0, near=0.05, far=2.2, transition_km=40.0)),
    "hazy here, clean far": (0.35, RadialSplitField(90.0, near=2.2, far=0.05, transition_km=40.0)),
}


def build(wl, aod, field):
    lay = AerosolLayer(HazeAerosol(0.75), aod, ExponentialProfile(1.3),
                       horizontal=field or UniformField())
    return Atmosphere(wl, aerosols=[lay])


def main(quick: bool = False):
    """Horizon colour vs haze distribution; ``--quick`` coarsens geometry."""
    sun_el = 12.0
    imgs = []
    cases = CASES if not quick else dict(list(CASES.items())[:2])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, (aod, field) in cases.items():
        atmo = build(WL, aod, field)
        dome = render.sky_dome(atmo, 90.0 - sun_el,
                               n_zenith=20 if quick else 44,
                               n_azimuth=25 if quick else 61)
        elev = 90.0 - dome["zenith"]
        xy, Lv = chromaticity_of(dome["radiance"])
        # average over azimuth
        ax.plot(elev, xy[..., 0].mean(axis=1), label=f"{name}  (x)")
        img = render.render_scene(atmo, sun_el, 0.0, camera="panorama",
                                  size=(300, 120) if quick else (620, 250),
                                  reference_percentile=92,
                                  n_zenith=20 if quick else 44,
                                  n_azimuth=25 if quick else 61)
        imgs.append(img)
        print(f"  {name:26s} zenith x={xy[0, 0, 0]:.3f}  horizon(88deg) x={xy[-1].mean(axis=0)[0]:.3f}")
    ax.set_xlabel("elevation [deg]")
    ax.set_ylabel("chromaticity x (azimuth average)")
    ax.set_title("how the horizon colour depends on where the haze is")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, FIG / "04_heterogeneity_curve.png")
    render.save_png(strip(imgs), FIG / "04_heterogeneity_panorama.png")
    print(f"  figure -> {FIG / '04_heterogeneity_panorama.png'}")
    print("done")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="coarser geometry and two cases for a bounded smoke run")
    main(quick=ap.parse_args().quick)
