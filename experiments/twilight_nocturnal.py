"""Experiment 08 -- the deep twilight: down to -18 degrees.

The deterministic production model (exact spherical single scattering plus a
*plane-parallel* multiple-scattering field) reproduces twilight down to about
-6 degrees solar elevation and then collapses: at -9 degrees the zenith line of
sight lies entirely inside the Earth's shadow and the light that reaches it has
travelled several hundred kilometres horizontally from the sunlit twilight
arch.  The plane-parallel solver has no mechanism for that transport.

This experiment shows the spherical Monte Carlo engine
(:func:`skycolor.rt.twilight.twilight_radiance`) through the same range, and
renders the sky at -9 and -12 degrees.

Run::

    uv run python experiments/twilight_nocturnal.py
    uv run python experiments/twilight_nocturnal.py --quick      # ~2 min

Output: ``figures/08_twilight_deep.png`` and
``experiments/out/08_twilight_deep.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import FIG, OUT, WL, chromaticity_of, save  # noqa: E402
from skycolor import color, presets, render, spectra  # noqa: E402
from skycolor.rt.single import sky_radiance as single_radiance  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import twilight_radiance  # noqa: E402

#: Patat, Ugolnikov & Postylyakov (2006), A&A 455, 385: V-band zenith
#: surface-brightness fit, converted to luminance (the same reference used in
#: ``validation/check5_twilight.py``).
PATAT = {-3.0: 40.1, -6.0: 0.516, -9.0: 0.0171, -12.0: 0.00146, -18.0: 1.81e-4}


def patat_fit(h):
    zeta = 90.0 - h
    m_v = 11.84 + 1.518 * (zeta - 95.0) - 0.057 * (zeta - 95.0) ** 2
    return 1.08e5 * 10.0 ** (-0.4 * m_v)


def luminance(rad, wl=WL):
    v = spectra.cie_xyz(np.asarray(wl, float))[:, 1]
    return 683.0 * np.trapezoid(np.asarray(rad) * v, np.asarray(wl, float), axis=-1)


def fisheye(atmo, sun_elevation_deg, size, quality, seed=0):
    """Spectral radiance and RGB of an equal-area fisheye view."""
    az, el, mask = render.fisheye_grid(size)
    zen = 90.0 - el
    rad, err = twilight_radiance(atmo, 90.0 - sun_elevation_deg,
                                 zen.ravel(), az.ravel(), quality=quality,
                                 seed=seed)
    shape = zen.shape + (rad.shape[-1],)
    rad = np.where(mask[..., None], rad.reshape(shape), 0.0)
    err = np.where(mask[..., None], err.reshape(shape), 0.0)
    lum = luminance(rad)
    rgb = color.render_rgb(rad, atmo.wl, tonemap="aces",
                           reference_percentile=99.0,
                           preserve_chromaticity=True)
    if mask is not None:
        rgb = rgb * mask[..., None]
    return dict(radiance=rad, error=err, luminance=lum, rgb=rgb, mask=mask)


def deterministic_fisheye(atmo, sun_elevation_deg, size):
    az, el, mask = render.fisheye_grid(size)
    zen = 90.0 - el
    rad = sky_radiance_total(atmo, 90.0 - sun_elevation_deg,
                             zen.ravel(), az.ravel())
    shape = zen.shape + (rad.shape[-1],)
    rad = np.where(mask[..., None], rad.reshape(shape), 0.0)
    rgb = color.render_rgb(rad, atmo.wl, tonemap="aces",
                           reference_percentile=99.0,
                           preserve_chromaticity=True)
    return dict(radiance=rad, rgb=rgb * mask[..., None])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--size", type=int, default=220)
    args = ap.parse_args(argv)

    quality = "low" if args.quick else "medium"
    base = (dict(n_anchor=2, n_photons=2.5e6) if args.quick
            else dict(n_anchor=4, n_photons=1.2e7))

    def grid_for(h):
        """More photons where the shadow makes the estimator rare-event limited."""
        scale = min(10.0, 10.0 ** max(0.0, (-h - 9.0) / 3.0))
        return dict(base, n_photons=base["n_photons"] * scale)
    atmo = presets.clear(WL)

    # --- 1. zenith luminance through twilight ------------------------------
    elevations = np.array([-4.0, -5.0, -6.0, -7.0, -8.0, -9.0,
                           -10.5, -12.0, -14.0, -18.0])
    t0 = time.time()
    det, eng, eng_err, ss = [], [], [], []
    for h in elevations:
        d = sky_radiance_total(atmo, 90.0 - h, [0.0], [0.0])
        s = single_radiance(atmo, 90.0 - h, [0.0], [0.0])
        tw, err = twilight_radiance(atmo, 90.0 - h, [0.0], [0.0],
                                    quality=quality, seed=3, grid=grid_for(h))
        det.append(float(luminance(d[0])))
        ss.append(float(luminance(s[0])))
        eng.append(float(luminance(tw[0])))
        eng_err.append(float(luminance(err[0])))
        print(f"  h={h:6.1f}  single={ss[-1]:.3e}  det={det[-1]:.3e}  "
              f"engine={eng[-1]:.3e} +/- {eng_err[-1]:.2e}  ({time.time()-t0:.0f}s)")
    det, eng, eng_err, ss = map(np.asarray, (det, eng, eng_err, ss))

    # --- 2. fisheye renders -------------------------------------------------
    size = args.size
    renders = {}
    for h in (-9.0, -12.0):
        t = time.time()
        renders[h] = fisheye(atmo, h, size, quality, seed=5)
        print(f"  fisheye h={h}: {time.time()-t:.0f}s  "
              f"max L={np.nanmax(renders[h]['luminance']):.2e} cd/m2")
    det12 = deterministic_fisheye(atmo, -12.0, size)

    # --- 3. figure ----------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.28,
                          wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    hh = np.linspace(-19.0, -2.0, 200)
    ax.semilogy(hh, [patat_fit(h) for h in hh], "k--", lw=1.6,
                label="Patat et al. (2006), V band")
    ax.semilogy(-np.array([3, 6, 9, 12, 18]),
                [PATAT[-3.0], PATAT[-6.0], PATAT[-9.0], PATAT[-12.0], PATAT[-18.0]],
                "ko", ms=6, mfc="none", label="reference values")
    ax.semilogy(-elevations, det, "C0.-", lw=1.6, ms=4,
                label="deterministic (single + plane-parallel MS)")
    ax.semilogy(-elevations, ss, "C2:", lw=1.2, label="single scattering only")
    ax.errorbar(-elevations, eng, yerr=eng_err, fmt="C3o-", lw=1.6, ms=5,
                capsize=2, label="MC twilight engine")
    for name, e in [("civil", -6), ("nautical", -12)]:
        ax.axvline(-e, color="0.85", ls="--", lw=0.8)
        ax.text(-e + 0.2, 2e-5, name, rotation=90, fontsize=7, color="0.5")
    ax.set_xlim(-0.2, 18.5)
    ax.set_ylim(1e-8, 1e3)
    ax.set_xlabel("solar depression below the horizon [deg]")
    ax.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.set_title("zenith brightness through twilight")
    ax.grid(alpha=0.25, which="both")
    ax.annotate("model has no airglow:\nthe -18$^\\circ$ reference is the "
                "night sky", xy=(18.0, 3e-7), xytext=(10.5, 3e-6),
                fontsize=6.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    ax.legend(fontsize=7, loc="lower left")

    ax2 = fig.add_subplot(gs[0, 1])
    ratio = np.where(det > 0, eng / np.maximum(det, 1e-30), np.nan)
    ax2.semilogy(-elevations, ratio, "C3o-", lw=1.6, ms=4,
                 label="engine / deterministic")
    ax2.axhline(1.0, color="0.6", lw=0.8)
    ax2.set_xlim(3.0, 18.5)
    ax2.set_ylim(0.5, 1e4)
    ax2.set_xlabel("solar depression below the horizon [deg]")
    ax2.set_ylabel("ratio of zenith luminance")
    ax2.set_title("how much the plane-parallel model misses")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=7, loc="upper left")

    ax3 = fig.add_subplot(gs[0, 2])
    for h, c in ((-9.0, "C0"), (-12.0, "C3")):
        i = int(np.argmin(np.abs(elevations - h)))
        # spectral radiance relative to its own value at 550 nm
        r = renders[h]["radiance"][size // 2, size // 2]
        ax3.semilogy(WL, r / np.nanmax(r), c + "-", lw=1.4,
                     label=f"sky at {h:.0f} deg (zenith)")
    ax3.set_xlabel("wavelength [nm]")
    ax3.set_ylabel("radiance / max")
    ax3.set_title("colour of the deep twilight")
    ax3.grid(alpha=0.25, which="both")
    ax3.legend(fontsize=7)

    for col, (h, data) in enumerate(((-9.0, renders[-9.0]),
                                     (-12.0, renders[-12.0]))):
        axf = fig.add_subplot(gs[1, col])
        im = data["rgb"]
        # stretch each panel independently: the two skies differ by four
        # orders of magnitude in absolute luminance
        lo, hi = np.percentile(im[data["mask"]], [2, 99.8])
        im = np.clip((im - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        axf.imshow(im, origin="upper")
        axf.set_xticks([])
        axf.set_yticks([])
        L = data["luminance"]
        rel = np.median(data["error"][data["mask"]]
                        / np.maximum(data["radiance"][data["mask"]], 1e-30))
        axf.set_title(
            f"MC engine, sun {h:.0f}$^\\circ$:  L = {np.nanmax(L):.2e} cd m$^{{-2}}$ "
            f"(auto-exposed)", fontsize=9)

    axf = fig.add_subplot(gs[1, 2])
    im = det12["rgb"]
    vals = im[np.any(im > 0, axis=2)]
    if vals.size:
        lo, hi = np.percentile(vals, [2.0, 99.8])
        im = np.clip((im - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    axf.imshow(im, origin="upper")
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_title("deterministic model, sun -12$^\\circ$\n"
                  "(single scattering only: the sky is black)",
                  fontsize=9)

    fig.suptitle("The deep twilight: a spherical Monte Carlo for the light "
                 "that travels sideways", fontsize=12)
    save(fig, "08_twilight_deep.png")
    plt.close(fig)

    out = dict(
        elevations=elevations.tolist(),
        single=ss.tolist(), deterministic=det.tolist(),
        engine=eng.tolist(), engine_err=eng_err.tolist(),
        patat={str(k): v for k, v in PATAT.items()},
        quality=quality, panel_size=size,
        runtime_s=time.time() - t0,
    )
    (OUT / "08_twilight_deep.json").write_text(json.dumps(out, indent=2))
    print(f"  data -> {OUT / '08_twilight_deep.json'}")
    return out


if __name__ == "__main__":
    main()
