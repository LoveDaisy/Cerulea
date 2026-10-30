"""Check 5 -- twilight zenith luminance.

Reference
---------
F. Patat, O. S. Ugolnikov and O. V. Postylyakov (2006), *UBVRI twilight sky
brightness at ESO-Paranal*, Astronomy & Astrophysics 454, 229-235,
arXiv:astro-ph/0604128.

This is an absolute photometric measurement of the twilight sky in the
Johnson-Cousins system, based on more than 2000 FORS1 images (sun zenith
distance 94-112 deg).  Table 1 gives a second-order fit to the V-band
surface brightness, valid for 95 <= zeta <= 105 deg (solar depression
5-15 deg):

    m_V(zeta) = 11.84 + 1.518 (zeta - 95) - 0.057 (zeta - 95)^2
        [mag arcsec^-2],  zeta in degrees.

The V surface brightness is converted to luminance with the standard
Johnson V relation ``L = 1.08e5 * 10^(-0.4 m_V)`` cd m^-2, i.e.
1 cd m^-2 = 12.6 mag arcsec^-2.  The values used here are stored in
``data/twilight_patat2006.csv``.

Caveats
-------
* The fit is only valid for solar elevations -5 to -15 deg.  The -3 and -18
  deg rows are extrapolations; Patat et al. state that the night-sky level is
  reached at zeta = 105-106 deg, so the -18 deg reference is really the
  natural night sky (airglow), not scattered twilight.  The model contains
  no airglow, so the -18 deg comparison measures the total absence of both
  multiple scattering and night-sky emission.
* The model is a single-scattering code.  At -12 deg and below the zenith
  line of sight lies entirely inside the Earth's shadow, so the model returns
  exactly zero.  This is not a numerical failure; it is the physical
  consequence of having no multiple scattering.  We quantify the deficit
  rather than hiding it.

Tolerances
----------
0.5 dex (a factor of ~3) in luminance at each elevation.  The model is
expected to fail at -9 deg and below; the failure is the point of the check.

Run:  python validation/check5_twilight.py
"""
from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, ".")

from skycolor.rt.single import sky_radiance  # noqa: E402

from validation import common as C  # noqa: E402

FLOOR = 1e-12  # cd/m^2, used only for logarithms when the model is exactly zero


def load_reference():
    path = C.DATA / "twilight_patat2006.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def model_zenith_luminance(sun_elevation_deg, kind="clear", wl=C.WL):
    atmo = C.atmosphere(kind, wl)
    rad = sky_radiance(atmo, 90.0 - sun_elevation_deg,
                       np.array([0.0]), np.array([0.0]))[0]
    return float(C.luminance(rad[None, :], wl)[0])


def run():
    ref = load_reference()
    elevations = [float(r["sun_elevation_deg"]) for r in ref]
    ref_lum = [float(r["zenith_luminance_cd_m2"]) for r in ref]
    ref_mag = [float(r["V_mag_arcsec2"]) for r in ref]

    model = {h: model_zenith_luminance(h) for h in elevations}
    model_pristine = {h: model_zenith_luminance(h, "pristine") for h in elevations}

    metrics = []
    for h, rl in zip(elevations, ref_lum):
        ml = model[h]
        dev = float(np.log10(max(ml, FLOOR) / rl))
        tol = 0.5
        name = f"zenith luminance, h={h:+.0f}deg"
        if h not in (-6.0, -9.0, -12.0, -18.0):
            name += " (extrapolated ref)"
        metrics.append(C.Metric(
            name, ml, rl, unit="cd/m2",
            tolerance=tol, dev=dev, dev_kind="abs",
        ))

    # -- figure --------------------------------------------------------------
    plt = C.plt()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    hs = np.linspace(-18.0, -5.0, 120)
    m_fit = np.array([11.84 + 1.518 * (90 - h - 95) - 0.057 * (90 - h - 95) ** 2
                      for h in hs])
    ax.plot(hs, 1.08e5 * 10 ** (-0.4 * m_fit), "k--", lw=1.5,
            label="Patat et al. (2006), V band fit")
    ax.plot(elevations, ref_lum, "ko", ms=6, mfc="none", label="reference points used")
    hh = np.linspace(-18.0, -2.0, 200)
    model_curve = [model_zenith_luminance(float(h), wl=C.WL_COARSE) for h in hh]
    ax.plot(hh, np.maximum(model_curve, FLOOR), "C0-", lw=1.8, label="model (clear)")
    hp = np.linspace(-18.0, -2.0, 120)
    pristine_curve = [model_zenith_luminance(float(h), "pristine", wl=C.WL_COARSE) for h in hp]
    ax.plot(hp, np.maximum(pristine_curve, FLOOR), "C2:", lw=1.2, label="model (pristine)")

    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlim(-19, -1)
    ax.set_xlabel("solar elevation [deg]")
    ax.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.set_title("Check 5: twilight zenith luminance")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.axhline(FLOOR, color="C3", lw=0.8, ls=":")
    ax.text(-18.5, FLOOR * 2.0, "model returns exactly 0 below -12 deg (shown at floor)",
            fontsize=7, color="C3")
    fig.savefig(C.FIG / "check5_twilight.png", dpi=150)
    plt.close(fig)

    details = dict(
        model={f"{h:+.0f}": model[h] for h in elevations},
        model_pristine={f"{h:+.0f}": model_pristine[h] for h in elevations},
        reference={f"{h:+.0f}": r for h, r in zip(elevations, ref_lum)},
        reference_mag_arcsec2={f"{h:+.0f}": m for h, m in zip(elevations, ref_mag)},
        floor=FLOOR,
        note=(
            "At -6 deg the single-scattering model is within 10% of the "
            "measured Paranal value.  At -9 deg it is ~24x too dark, and at "
            "-12/-18 deg it is exactly zero because the zenith line of sight "
            "is entirely in the Earth's shadow.  The missing multiple "
            "scattering is the cause; at -18 deg the reference is the natural "
            "night sky, which the model does not include at all."
        ),
    )
    summary = (
        f"zenith L: -6deg {model[-6.0]:.3g} vs {dict(zip(elevations,ref_lum))[-6.0]:.3g}; "
        f"-9deg {model[-9.0]:.2g} vs {dict(zip(elevations,ref_lum))[-9.0]:.2g}; "
        f"-12deg {model[-12.0]:.1g} vs {dict(zip(elevations,ref_lum))[-12.0]:.2g} cd/m2."
    )
    return C.CheckResult(
        key="check5", title="Twilight zenith luminance (Patat et al. 2006)",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    lines = ["", "  solar elev   model L      Patat L       ratio      reference"]
    for h in [-3.0, -6.0, -9.0, -12.0, -18.0]:
        m = result.details["model"][f"{h:+.0f}"]
        r = result.details["reference"][f"{h:+.0f}"]
        ratio = "--" if m <= 0 else f"{m/r:.3g}"
        note = "extrapolated fit" if h in (-3.0, -18.0) else "fit, 95<=zeta<=105"
        lines.append(f"  {h:8.0f} deg {m:11.3e} {r:11.3e} {ratio:>10s}   {note}")
    lines.append("")
    lines.append("  The model has no multiple scattering.  At -12 deg and below the")
    lines.append("  zenith line of sight is entirely in shadow, so the single-scattering")
    lines.append("  radiance is exactly zero.  At -18 deg the reference is the natural")
    lines.append("  night sky (airglow), which the model does not include either.")
    return C.run_standalone(result, lines)


if __name__ == "__main__":
    raise SystemExit(main())
