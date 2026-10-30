"""Check 3 -- sky luminance distribution vs CIE 110-1994 clear sky.

This is the strongest purely geometric test in the harness: the CIE standard
clear sky prescribes the *relative* luminance L(theta, gamma)/Lz as a
function of the sun position, with no dependence on the absolute solar
spectrum or on a measured radiometric scale.

Reference
---------
CIE 110-1994, *Spatial distribution of daylight -- Luminance distributions of
various reference skies*.  The standard clear-sky distribution is

    L(theta,gamma)/Lz =
        (0.91 + 10 exp(-3 gamma) + 0.45 cos^2 gamma)
        (1 - exp(-0.32 / cos theta))
      / [(0.91 + 10 exp(-3 theta_s) + 0.45 cos^2 theta_s)
         (1 - exp(-0.32))]

with theta the viewing zenith angle, gamma the angular distance from the sun,
theta_s the solar zenith angle (CIE 110-1994; reproduced as eq. (4.31) of
Zotti 2007, which cites the CIE standard).

The model's luminance is the photopic integral of its spectral radiance,
normalised by the model's own zenith luminance, so the test is scale-free.

What the model is missing
-------------------------
The production model currently has no multiple scattering (the module is a
stub returning zero).  Multiple scattering makes the real sky more uniform:
it brightens the zenith and the anti-solar sky relative to the single-scatter
horizon.  We therefore expect the model to be too dark at the zenith
*relative to the horizon* -- i.e. the horizon/zenith ratio to be too large.
We also run pristine (Rayleigh-only) and a well-mixed aerosol profile to
separate the aerosol contribution from the multiple-scattering contribution.

Tolerances (chosen before looking at the final numbers)
-------------------------------------------------------
* horizon/zenith ratio: 40 % (sun side), 25 % (anti-solar),
* RMS relative deviation: 20 % for theta <= 60 deg, 25 % for 60-75 deg,
  30 % for the 75-90 deg horizon band, 15 % for the whole sky.

Run:  python validation/check3_cie_luminance.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from validation import common as C  # noqa: E402

ELEVATIONS = (30.0, 45.0, 60.0)
N_ZEN, N_AZ = 37, 72


def model_relative_luminance(elevation, kind="clear", wl=C.WL_COARSE):
    """Model relative luminance map and the absolute zenith luminance."""
    z, a = C.sky_grid(N_ZEN, N_AZ)
    atmo = C.atmosphere(kind, wl)
    rad = C.sky_radiance(atmo, 90.0 - elevation, z.ravel(), a.ravel())
    lum = C.luminance(rad.reshape(z.shape + (len(wl),)), wl)
    lz = float(lum[0, 0])
    return z, a, lum / lz, lz


def _ratio(model, ref):
    return float(model), float(ref), 100.0 * (float(model) / float(ref) - 1.0)


def run():
    z, a = C.sky_grid(N_ZEN, N_AZ)
    maps = {}
    metrics = []
    table = []

    for h in ELEVATIONS:
        _, _, rel, lz = model_relative_luminance(h)
        cie = C.cie110_relative_luminance(z, a, 90.0 - h)
        dev = C.rel_dev(rel, cie)
        maps[h] = (rel, cie, dev, lz)
        upper = z <= 60.0
        mid = (z > 60.0) & (z <= 75.0)
        horizon = z >= 75.0
        rms_upper = float(np.sqrt(np.mean(dev[upper] ** 2)))
        rms_mid = float(np.sqrt(np.mean(dev[mid] ** 2)))
        rms_hor = float(np.sqrt(np.mean(dev[horizon] ** 2)))
        rms_all = float(np.sqrt(np.mean(dev**2)))
        table.append(
            dict(
                elevation=h,
                lz_model=lz,
                horizon_sun_model=float(rel[-1, 0]),
                horizon_sun_cie=float(cie[-1, 0]),
                horizon_anti_model=float(rel[-1, N_AZ // 2]),
                horizon_anti_cie=float(cie[-1, N_AZ // 2]),
                rms_upper=rms_upper,
                rms_mid=rms_mid,
                rms_horizon=rms_hor,
                rms_all=rms_all,
                max_abs=float(np.max(np.abs(dev))),
            )
        )

    # -- metrics at the reference (45 deg) elevation ------------------------
    t45 = next(t for t in table if t["elevation"] == 45.0)
    m, r, d = _ratio(t45["horizon_sun_model"], t45["horizon_sun_cie"])
    metrics.append(C.Metric(
        "horizon sun-side / zenith, h=45deg", m, r, unit="",
        tolerance=40.0, dev=d,
    ))
    m, r, d = _ratio(t45["horizon_anti_model"], t45["horizon_anti_cie"])
    metrics.append(C.Metric(
        "horizon anti-solar / zenith, h=45deg", m, r, unit="",
        tolerance=25.0, dev=d,
    ))
    metrics.append(C.Metric(
        "RMS dev, theta<=60deg, h=45deg", t45["rms_upper"], 0.0, unit="%",
        tolerance=20.0, dev=t45["rms_upper"], dev_kind="abs",
    ))
    metrics.append(C.Metric(
        "RMS dev, 60-75deg, h=45deg", t45["rms_mid"], 0.0, unit="%",
        tolerance=25.0, dev=t45["rms_mid"], dev_kind="abs",
    ))
    metrics.append(C.Metric(
        "RMS dev, 75-90deg horizon, h=45deg", t45["rms_horizon"], 0.0, unit="%",
        tolerance=30.0, dev=t45["rms_horizon"], dev_kind="abs",
    ))
    metrics.append(C.Metric(
        "RMS dev, whole sky, h=45deg", t45["rms_all"], 0.0, unit="%",
        tolerance=15.0, dev=t45["rms_all"], dev_kind="abs",
    ))

    # -- the same discrepancy at the other elevations -----------------------
    for h in (30.0, 60.0):
        t = next(t for t in table if t["elevation"] == h)
        m, r, d = _ratio(t["horizon_sun_model"], t["horizon_sun_cie"])
        metrics.append(C.Metric(
            f"horizon sun-side / zenith, h={h:.0f}deg", m, r, unit="",
            tolerance=50.0, dev=d,
        ))

    # -- attribution: pristine and well-mixed aerosol at h=45 ---------------
    attribution = {}
    for kind in ("pristine", "clear_uniform"):
        try:
            _, _, rel, lz = model_relative_luminance(45.0, kind=kind)
            attribution[kind] = dict(
                zenith_luminance=lz,
                horizon_sun=float(rel[-1, 0]),
                horizon_anti=float(rel[-1, N_AZ // 2]),
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            attribution[kind] = {"error": str(exc)}

    # -- figure ---------------------------------------------------------------
    plt = C.plt()
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(
        len(ELEVATIONS), 3, figsize=(13.2, 3.6 * len(ELEVATIONS)), constrained_layout=True
    )
    levels = np.logspace(np.log10(0.3), np.log10(12.0), 60)
    extent = (0.0, 360.0, 90.0, 0.0)
    images = []
    for i, h in enumerate(ELEVATIONS):
        rel, cie, dev, lz = maps[h]
        for j, (arr, title, cmap, norm, label) in enumerate(
            [
                (rel, f"model, sun elev {h:.0f}deg", "viridis", LogNorm(0.3, 12.0),
                 "L/L$_{zenith}$"),
                (cie, f"CIE 110-1994, sun elev {h:.0f}deg", "viridis",
                 LogNorm(0.3, 12.0), "L/L$_{zenith}$"),
                (rel / np.maximum(cie, 1e-9), f"model / CIE, sun elev {h:.0f}deg",
                 "coolwarm", None, "ratio"),
            ]
        ):
            ax = axes[i, j]
            if norm is None:
                im = ax.imshow(arr.T, origin="upper", extent=extent, aspect="auto",
                               cmap=cmap, vmin=0.0, vmax=2.0)
            else:
                im = ax.imshow(arr.T, origin="upper", extent=extent, aspect="auto",
                               cmap=cmap, norm=norm)
            ax.plot([0, 360], [90 - h, 90 - h], "k-", lw=0.6, alpha=0.5)
            ax.plot([0, 360], [90 - h, 90 - h], "y*", ms=9, markevery=[0, 1])
            ax.set_xlabel("azimuth from sun [deg]")
            ax.set_ylabel("view zenith [deg]")
            ax.set_title(f"{title}", fontsize=9)
            images.append(im)
    for i, h in enumerate(ELEVATIONS):
        fig.colorbar(images[3 * i + 2], ax=axes[i, 2], label="model / CIE")
        fig.colorbar(images[3 * i], ax=axes[i, :2], label="relative luminance")
    fig.suptitle("Check 3: sky relative luminance vs CIE 110-1994 clear sky")
    fig.savefig(C.FIG / "check3_cie_luminance.png", dpi=145)
    plt.close(fig)

    # cuts through the sun meridian -- the clearest view of the discrepancy
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    rel, cie, _, _ = maps[45.0]
    zz = z[:, 0]
    for ax, j, side in [(axes[0], 0, "towards the sun"), (axes[1], N_AZ // 2, "anti-solar")]:
        ax.plot(zz, rel[:, j], "C0-", label="model")
        ax.plot(zz, cie[:, j], "k--", label="CIE 110-1994")
        ax.set_yscale("log")
        ax.set_xlabel("view zenith angle [deg]")
        ax.set_ylabel("L / L$_{zenith}$")
        ax.set_title(f"h = 45deg, {side} azimuth")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Check 3: luminance cuts at solar elevation 45deg")
    fig.savefig(C.FIG / "check3_cie_cuts.png", dpi=150)
    plt.close(fig)

    # -- data ----------------------------------------------------------------
    C.save_csv(
        C.DATA / "check3_cie110_luminance.csv",
        ["solar_elevation_deg", "view_zenith_deg", "model_rel_sun_az",
         "cie_rel_sun_az", "model_rel_anti_az", "cie_rel_anti_az"],
        [
            np.repeat([t["elevation"] for t in table], len(zz)),
            np.tile(zz, len(ELEVATIONS)),
            np.concatenate([maps[h][0][:, 0] for h in ELEVATIONS]),
            np.concatenate([maps[h][1][:, 0] for h in ELEVATIONS]),
            np.concatenate([maps[h][0][:, N_AZ // 2] for h in ELEVATIONS]),
            np.concatenate([maps[h][1][:, N_AZ // 2] for h in ELEVATIONS]),
        ],
    )

    d45 = 100.0 * (t45["horizon_sun_model"] / t45["horizon_sun_cie"] - 1.0)
    summary = (
        f"h=45deg: sun-side horizon/zenith {t45['horizon_sun_model']:.2f} vs CIE "
        f"{t45['horizon_sun_cie']:.2f} ({d45:+.0f}%); anti-solar "
        f"{t45['horizon_anti_model']:.2f} vs {t45['horizon_anti_cie']:.2f}; "
        f"RMS {t45['rms_upper']:.1f}% (theta<=60deg), "
        f"{t45['rms_horizon']:.1f}% (horizon band)."
    )
    details = dict(
        table=table,
        attribution=attribution,
        note=(
            "The sun-side horizon is about 2x the CIE value at h=45deg. "
            "Turning the aerosol off (pristine) makes it larger, not smaller, "
            "so the dominant cause is the missing multiple scattering, not the "
            "aerosol profile; the forward-scattering aerosol aureole adds a "
            "secondary excess at low elevations."
        ),
    )
    return C.CheckResult(
        key="check3", title="CIE 110-1994 clear-sky luminance distribution",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    lines = ["", "  elevation  horizon_sun(model/CIE)  horizon_anti  RMS<=60   RMS_horizon  RMS_all  max|dev|"]
    for t in result.details["table"]:
        lines.append(
            f"  {t['elevation']:8.0f}deg  {t['horizon_sun_model']/t['horizon_sun_cie']:8.2f}x"
            f" ({t['horizon_sun_model']:.2f}/{t['horizon_sun_cie']:.2f})"
            f"  {t['horizon_anti_model']:.2f}/{t['horizon_anti_cie']:.2f}"
            f"  {t['rms_upper']:7.1f}%  {t['rms_horizon']:9.1f}%  {t['rms_all']:6.1f}%"
            f"  {t['max_abs']:7.1f}%"
        )
    a = result.details["attribution"]
    lines.append("")
    lines.append("  attribution at h=45deg (horizon sun / zenith, horizon anti / zenith):")
    for k, v in a.items():
        if "horizon_sun" in v:
            lines.append(
                f"    {k:14s} zenith={v['zenith_luminance']:7.1f} cd/m2  "
                f"sun-side={v['horizon_sun']:.2f}  anti-solar={v['horizon_anti']:.2f}"
            )
    return C.run_standalone(result, lines)


if __name__ == "__main__":
    raise SystemExit(main())
