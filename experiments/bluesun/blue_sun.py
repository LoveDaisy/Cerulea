"""Experiment 09 -- the blue sun.

A sun that looks blue is real physics, not a white-balance trick.  It needs an
extinction spectrum that is *non-monotonic* in wavelength: more extinction in
the red than in the blue.  That happens for particles whose radius is close to
half the wavelength and whose size distribution is narrow enough for the Mie
ripple to survive.

An Angstrom power law, ``tau ~ lambda**-alpha``, can never do it: it is
monotonic, so the direct beam always reddens.  This script makes the point by
comparing the two.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiments.common import save

THEME = Path(__file__).resolve().parent          # experiments/<theme>
FIG = THEME / "figures"      # theme-owned figure output (not the root candidates)
OUT = THEME / "out"          # theme-owned numeric output (gitignored)
from skycolor import color
from skycolor.aerosol import AngstromAerosol, LogNormalMode, MieAerosol
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile
from skycolor.rt.single import direct_sun_transmittance
from skycolor.spectra import solar_irradiance

WL = np.arange(380.0, 781.0, 3.0)
i450, i550, i650 = [np.argmin(abs(WL - w)) for w in (450, 550, 650)]


def beam(atmo, sza=40.0):
    T = direct_sun_transmittance(atmo, sza)
    spec = solar_irradiance(WL) * T
    xyz = color.spectrum_to_xyz(spec, WL)
    return color.chromaticity(xyz), xyz[1], T


def main():
    radii = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0])
    sigmas = np.array([1.05, 1.10, 1.20, 1.35, 1.60, 2.00])
    ratio = np.full((len(radii), len(sigmas)), np.nan)
    xys = {}
    for i, r in enumerate(radii):
        for j, sg in enumerate(sigmas):
            aer = MieAerosol(name=f"r{r}g{sg}", modes=[LogNormalMode(r, sg)],
                             refractive_index=1.53 + 0.008j, r_max_um=6.0,
                             n_radii=40, n_angles=361)
            atmo = Atmosphere(WL, aerosols=[AerosolLayer(aer, 1.0, ExponentialProfile(1.5))])
            T = direct_sun_transmittance(atmo, 40.0)
            tau = -np.log(np.maximum(T, 1e-30))
            ratio[i, j] = tau[i450] / tau[i650]
            xys[(r, sg)] = color.chromaticity(color.spectrum_to_xyz(
                solar_irradiance(WL) * T, WL))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    im = axes[0].imshow(ratio, origin="lower", aspect="auto", cmap="RdBu_r",
                        vmin=0.6, vmax=1.6,
                        extent=[0, len(sigmas), 0, len(radii)])
    axes[0].set_xticks(np.arange(len(sigmas)) + 0.5)
    axes[0].set_xticklabels([f"{s:.2f}" for s in sigmas])
    axes[0].set_yticks(np.arange(len(radii)) + 0.5)
    axes[0].set_yticklabels([f"{r:.2f}" for r in radii])
    axes[0].set_xlabel("geometric standard deviation")
    axes[0].set_ylabel("geometric mean radius [µm]")
    axes[0].set_title("τ(450 nm) / τ(650 nm)\nblue = the beam gets bluer")
    plt.colorbar(im, ax=axes[0], label="ratio")
    for i, r in enumerate(radii):
        for j, sg in enumerate(sigmas):
            if ratio[i, j] < 0.80:
                axes[0].text(j + 0.5, i + 0.5, "×", ha="center", va="center", fontsize=9)

    # beam colour along a slice and the Angstrom comparison
    for sg, style in [(1.05, "-o"), (2.00, "-s")]:
        pts = np.array([xys[(r, sg)] for r in radii])
        axes[1].plot(pts[:, 0], pts[:, 1], style, ms=4, label=f"Mie, σg={sg:.2f}")
    for alpha, style in [(1.3, "--"), (0.2, ":")]:
        atmo = Atmosphere(WL, aerosols=[AerosolLayer(AngstromAerosol(alpha=alpha), 1.0,
                                                     ExponentialProfile(1.5))])
        xy, _, _ = beam(atmo)
        axes[1].plot([xy[0]], [xy[1]], style, marker="x", ms=10,
                     label=f"Ångström α={alpha}")
    clean = Atmosphere(WL, aerosols=[])
    xy, _, _ = beam(clean)
    axes[1].plot([xy[0]], [xy[1]], "k*", ms=14, label="no aerosol")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("colour of the direct beam (sun 50° up, AOD 1.0)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    save(fig, FIG / "09_blue_sun.png")

    # spectra for the most anomalous case
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r, sg, lab in [(0.45, 1.05, "r=0.45 µm, σg=1.05"),
                       (0.45, 2.00, "r=0.45 µm, σg=2.00 (broad)"),
                       (1.00, 1.10, "r=1.00 µm, σg=1.10")]:
        aer = MieAerosol(name=lab, modes=[LogNormalMode(r, sg)],
                         refractive_index=1.53 + 0.008j, r_max_um=6.0, n_radii=40,
                         n_angles=361)
        atmo = Atmosphere(WL, aerosols=[AerosolLayer(aer, 1.0, ExponentialProfile(1.5))])
        T = direct_sun_transmittance(atmo, 40.0)
        tau = -np.log(np.maximum(T, 1e-30))
        ax.plot(WL, tau / tau[i550], label=lab)
    atmo = Atmosphere(WL, aerosols=[AerosolLayer(AngstromAerosol(alpha=1.3), 1.0,
                                                 ExponentialProfile(1.5))])
    T = direct_sun_transmittance(atmo, 40.0)
    tau = -np.log(np.maximum(T, 1e-30))
    ax.plot(WL, tau / tau[i550], "k--", label="Ångström α=1.3")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("τ(λ) / τ(550 nm)")
    ax.set_title("extinction spectra: Mie ripples vs a power law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, FIG / "09_blue_sun_spectra.png")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "09_blue_sun.npz", radii=radii, sigmas=sigmas, ratio=ratio)
    best = np.unravel_index(np.nanargmin(ratio), ratio.shape)
    print(f"most anomalous: r={radii[best[0]]:.2f} µm σg={sigmas[best[1]]:.2f} "
          f"ratio={ratio[best]:.3f}")
    print("done")


if __name__ == "__main__":
    main()
