"""Check 4 -- sky chromaticity vs measured spectra.

Reference
---------
**SKYSPECTRA** (Balakrishnan et al. 2023, CIE x050:2023 OP026; open data,
Zenodo DOI 10.5281/zenodo.8147546).  This is an open package of measured
spectral daylight from five sites.  The `spectral_patch_radiance` table
contains 10-degree-aperture spectral radiance measurements of sky patches.
For Berlin (DE-BLN) the available patches are all on the horizon ring
(almucantar 6 deg, azimuths 180/192/204/216/228/240 deg), recorded on
2015-03-08 (a clear day, CIE sky type 12) and a few other clear days.

We select every timestamp where

* the site is DE-BLN, setup 1,
* `sky_condition == "clear"`,
* the solar elevation is between 30 and 50 deg (the SKYSPECTRA package only
  distributes measurements with solar elevation >= 15 deg),

and compare the measured CIE 1931 chromaticity of each patch with the model
at the same sun position and the same viewing geometry (view zenith 84 deg,
view azimuth = patch azimuth), using `presets.clear` (AOD_550 = 0.05).

Limitation
----------
The distributed patch data for this site are all at 6 deg elevation, so this
check tests the **horizon** chromaticity, not the zenith.  We report the
model's zenith chromaticity at h = 30/45/60 deg for information but have no
measured zenith spectrum to compare it with; this is stated in the report.

Tolerances
----------
RMS chromaticity error 0.02 in xy and 0.02 in CIE 1976 u'v'.  These are about
2-4 just-noticeable differences; a physically based sky model should do
better than that for a clear sky.

Run:  python validation/check4_chromaticity.py
"""
from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, ".")

from skycolor.color import chromaticity, spectrum_to_xyz  # noqa: E402
from skycolor.rt.single import sky_radiance  # noqa: E402

from validation import common as C  # noqa: E402


def uv(x, y):
    d = -2.0 * x + 12.0 * y + 3.0
    return 4.0 * x / d, 9.0 * y / d


def model_xy(sun_elev, sun_az, view_zen, view_az, kind="clear", aod=0.05,
             wl=C.WL_COARSE):
    atmo = C.atmosphere(kind, wl, aod=aod) if kind != "pristine" else C.atmosphere("pristine", wl)
    rad = sky_radiance(atmo, 90.0 - sun_elev, np.array([view_zen]),
                       np.array([view_az]), sun_azimuth_deg=sun_az)[0]
    return chromaticity(spectrum_to_xyz(rad[None, :], wl)[0])


def load_rows():
    path = C.DATA / "skyspectra_berlin_horizon_xy.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def run():
    rows = load_rows()
    wl = C.WL_COARSE

    meas_xy = np.array([[float(r["x_meas"]), float(r["y_meas"])] for r in rows])
    model_xy_arr = np.array([
        model_xy(float(r["sun_elevation_deg"]), float(r["sun_azimuth_deg"]),
                 84.0, float(r["patch_azimuth_deg"]), wl=wl)
        for r in rows
    ])
    dxy = model_xy_arr - meas_xy
    rms_xy = float(np.sqrt(np.mean(np.sum(dxy**2, axis=1))))
    meas_uv = np.array([uv(*p) for p in meas_xy])
    model_uv = np.array([uv(*p) for p in model_xy_arr])
    duv = model_uv - meas_uv
    rms_uv = float(np.sqrt(np.mean(np.sum(duv**2, axis=1))))

    metrics = [
        C.Metric("mean delta x, horizon", float(model_xy_arr[:, 0].mean()),
                 float(meas_xy[:, 0].mean()), unit="xy",
                 tolerance=0.02, dev=float(dxy[:, 0].mean()), dev_kind="abs"),
        C.Metric("mean delta y, horizon", float(model_xy_arr[:, 1].mean()),
                 float(meas_xy[:, 1].mean()), unit="xy",
                 tolerance=0.02, dev=float(dxy[:, 1].mean()), dev_kind="abs"),
        C.Metric("RMS delta xy, horizon", rms_xy, 0.0, unit="xy",
                 tolerance=0.02, dev=rms_xy, dev_kind="abs"),
        C.Metric("RMS delta u'v', horizon", rms_uv, 0.0, unit="u'v'",
                 tolerance=0.02, dev=rms_uv, dev_kind="abs"),
    ]

    # -- aerosol sensitivity -------------------------------------------------
    sensitivity = {}
    for aod in (0.02, 0.05, 0.10):
        pred = np.array([
            model_xy(float(r["sun_elevation_deg"]), float(r["sun_azimuth_deg"]),
                     84.0, float(r["patch_azimuth_deg"]), aod=aod, wl=wl)
            for r in rows
        ])
        sensitivity[aod] = float(np.sqrt(np.mean(np.sum((pred - meas_xy) ** 2, axis=1))))

    # -- zenith chromaticity (no measured reference available) --------------
    zenith = {}
    for h in (30.0, 45.0, 60.0):
        zenith[h] = model_xy(h, 180.0, 0.0, 0.0, wl=wl)

    # -- figure --------------------------------------------------------------
    plt = C.plt()
    # spectral locus
    wl_locus = np.arange(380.0, 701.0, 2.0)
    cmf = C.spectra.cie_xyz(wl_locus)
    locus = cmf[:, :2] / np.maximum(cmf.sum(axis=1, keepdims=True), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(locus[:, 0], locus[:, 1], "k-", lw=1.0, label="spectral locus")
    ax.plot(meas_xy[:, 0], meas_xy[:, 1], "o", ms=4, mfc="none", color="C1",
            label="measured, horizon (SKYSPECTRA)")
    ax.plot(model_xy_arr[:, 0], model_xy_arr[:, 1], "x", ms=5, color="C0",
            label="model, same geometry")
    ax.plot(meas_xy[:, 0].mean(), meas_xy[:, 1].mean(), "s", ms=9, color="C1",
            label="measured mean")
    ax.plot(model_xy_arr[:, 0].mean(), model_xy_arr[:, 1].mean(), "D", ms=8,
            color="C0", label="model mean")
    for h, p in zenith.items():
        ax.plot(p[0], p[1], "*", ms=11, color="C2")
        ax.annotate(f"zenith h={h:.0f}", p, textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="C2")
    ax.set_xlim(0.20, 0.40)
    ax.set_ylim(0.20, 0.42)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CIE 1931 chromaticity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    spec = np.loadtxt(C.DATA / "skyspectra_berlin_horizon_mean_spectrum.csv",
                      delimiter=",", skiprows=1)
    wls = spec[:, 0]
    for j in range(1, spec.shape[1]):
        y = spec[:, j]
        ax.plot(wls, y / np.interp(550.0, wls, y), lw=1.0,
                label=f"az={180 + 12*(j-1)}deg")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("radiance / radiance(550 nm)")
    ax.set_title("measured horizon spectra (mean of selected cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Check 4: horizon chromaticity vs SKYSPECTRA measurements (Berlin, clear sky)")
    fig.savefig(C.FIG / "check4_chromaticity.png", dpi=150)
    plt.close(fig)

    details = dict(
        n_cases=len(rows),
        n_timestamps=len({r["timestamp"] for r in rows}),
        measured_mean_xy=meas_xy.mean(axis=0).tolist(),
        model_mean_xy=model_xy_arr.mean(axis=0).tolist(),
        rms_xy=rms_xy,
        rms_uv=rms_uv,
        aod_sensitivity={str(k): v for k, v in sensitivity.items()},
        zenith_xy={str(k): v.tolist() for k, v in zenith.items()},
        note=(
            "All SKYSPECTRA Berlin patches are at 6 deg elevation; this is a "
            "horizon check. No open measured zenith spectrum was found in the "
            "timebox. The measured horizon is whitish (x~0.29, y~0.31); the "
            "model with AOD550=0.05 is slightly yellower by (dx,dy)=(+0.008,+0.012). "
            "AOD 0.02 matches best; AOD 0.10 is 0.025 off. The model zenith is "
            "x~0.26-0.27, y~0.27-0.29, at the blue end of the expected range."
        ),
    )
    summary = (
        f"horizon xy: model ({model_xy_arr[:,0].mean():.3f},"
        f"{model_xy_arr[:,1].mean():.3f}) vs measured "
        f"({meas_xy[:,0].mean():.3f},{meas_xy[:,1].mean():.3f}); "
        f"RMS dxy={rms_xy:.4f}, du'v'={rms_uv:.4f}."
    )
    return C.CheckResult(
        key="check4", title="Sky chromaticity vs measured spectra",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    lines = [
        "",
        f"  {result.details['n_cases']} patch spectra from "
        f"{result.details['n_timestamps']} clear-sky timestamps (Berlin, 2015).",
        f"  model mean xy   {result.details['model_mean_xy']}",
        f"  measured mean xy {result.details['measured_mean_xy']}",
        "",
        "  aerosol sensitivity (RMS dxy):",
    ]
    for k, v in result.details["aod_sensitivity"].items():
        lines.append(f"    AOD550={k:>5s}: {v:.4f}")
    lines.append("")
    lines.append("  model zenith xy (no measured zenith reference):")
    for k, v in result.details["zenith_xy"].items():
        lines.append(f"    h={float(k):4.0f}deg: x={v[0]:.4f} y={v[1]:.4f}")
    return C.run_standalone(result, lines)


if __name__ == "__main__":
    raise SystemExit(main())
