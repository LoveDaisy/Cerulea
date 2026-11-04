"""Check 6 -- twilight against a second, independent dataset.

Reference
---------
M. J. Koomen, C. Lock, D. M. Packer, R. Scolnik, R. Tousey and E. O. Hulburt
(1952), *Measurements of the Brightness of the Twilight Sky*, Journal of the
Optical Society of America **42**, 353-359.  DOI: 10.1364/JOSA.42.000353.

A recording photometer of **photopic** sensitivity was used at two stations,
Maryland (altitude 30 m) and Sacramento Peak, New Mexico (2800 m), for solar
altitudes H = +5 to -15 deg.  The zenith surface brightness at Maryland, in
cd ft^-2, is reproduced in Table 5 of Nawar et al. (2020, NRIAG J. Astron.
Geophys. 9, 63), which also gives the conversion used here:

    m [mag arcsec^-2] = -2.5 log10 I[cd ft^-2] + 9.9846.

There is no filter in the optical train: the photometer response *is* the
photopic curve, so the measured quantity is the same photopic luminance that
``validation.common.luminance`` computes from the model spectrum.  This makes
Koomen et al. the closest like-for-like check available: a photopic, absolute,
sea-level twilight measurement, independent of Patat et al. (2006).

Second reference (secondary, used only as a cross-check)
--------------------------------------------------------
S. Nawar, A. B. Morcos, R. M. El Agmy, G. M. A. Gad and S. Elgohary (2020),
*Sky twilight brightness at zenith expressed in magnitudes*, NRIAG Journal of
Astronomy and Geophysics **9**, 63-70.  DOI: 10.1080/20909977.2020.1714176.
Tables 1-3 give zenith twilight brightness at Helwan (115 m), Daraw (250 m)
and Abu Simbel (210 m) in mag arcsec^-2, converted by the authors from the
S10(lambda) units of the original 1977/1995 measurements.  The passbands are
broad glass filters at 4410 (B), 5500 (yellow, treated here as a V proxy) and
7900 A (R); they are *not* Johnson bands and carry a corresponding systematic.

Model
-----
``skycolor.rt.solver.sky_radiance_total`` for h >= -8 deg, and
``skycolor.rt.twilight.twilight_radiance`` (``quality="low"``) below that.
Atmosphere: ``presets.pristine`` and ``presets.clear`` (300 DU O3,
AOD_550 = 0.05, sea level); the absolute level depends on this choice.

The check is a comparison, not a gate: it reports model/reference at each
point and scores only h = -6, -9, -12 (the fitted range of the primary
reference).  h = 0 is the sun on the horizon, h = -15 is airglow.

The solar path uses the two-branch chord (M10): for a point whose sun is below
the local horizontal the optical depth is the far branch (perigee -> space)
plus the near branch (perigee -> point).  The one-branch version returned
1.14x / 2.20x at -6 deg; the two-branch values are 0.69x / 0.86x.

The altitude response is diagnostic only: cross-site measurements do not
isolate a pure height intervention.  Aligning solar occultation with the
raised ground changed the deterministic -6 deg ratio from 1.208 to 1.027;
see docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md for independent MC and reference audits.

Runtime is a few minutes, almost all of it in the two engine points.

Run:  python validation/check6_second_dataset.py
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, ".")

from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import twilight_radiance  # noqa: E402

from validation import common as C  # noqa: E402

FLOOR = 1e-12
#: Below this solar elevation the deterministic plane-parallel multiple
#: scattering is replaced by the spherical engine.
SPLIT = -8.0
#: Factor-1.5 tolerance, in dex.
TOL = math.log10(1.5)
#: 20 nm grid, used only for the smooth curve in the figure (the table uses
#: the 5 nm ``WL_COARSE``).
WL_FIG = np.arange(380.0, 781.0, 20.0)


def load_koomen():
    path = C.DATA / "twilight_koomen1952.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_nawar():
    path = C.DATA / "twilight_nawar2020.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def patat_fit(h):
    """Patat et al. (2006) V-band fit, converted to luminance [cd m^-2]."""
    zeta = 90.0 - h
    m_v = 11.84 + 1.518 * (zeta - 95.0) - 0.057 * (zeta - 95.0) ** 2
    return 1.08e5 * 10.0 ** (-0.4 * m_v)


def model_luminance(h, kind="clear", wl=None):
    """Zenith luminance and (for the engine) its 1-sigma statistical error."""
    wl = C.WL_COARSE if wl is None else np.asarray(wl, dtype=float)
    atmo = C.atmosphere(kind, wl)
    if h >= SPLIT:
        rad = sky_radiance_total(atmo, 90.0 - float(h),
                                 np.array([0.0]), np.array([0.0]))
        return float(C.luminance(rad[0][None, :], wl)[0]), None
    rad, err = twilight_radiance(atmo, 90.0 - float(h),
                                 np.array([0.0]), np.array([0.0]),
                                 quality="low", seed=0)
    return (float(C.luminance(rad[0][None, :], wl)[0]),
            float(C.luminance(err[0][None, :], wl)[0]))


def ratio_text(model, ref, err=None):
    if model <= 0 or ref <= 0:
        return "--"
    r = model / ref
    if err is None:
        return f"{r:.2f}"
    lo = max(model - err, FLOOR) / ref
    return f"{r:.2f} [{lo:.1f}-{(model + err) / ref:.1f}]"


def run():
    koomen = load_koomen()
    hs = [float(r["sun_elevation_deg"]) for r in koomen]
    ref = [float(r["zenith_luminance_cd_m2"]) for r in koomen]

    model = {}
    model_pristine = {}
    engine_err = {}
    for h in hs:
        model[h], engine_err[h] = model_luminance(h, "clear")
        if h >= SPLIT:
            model_pristine[h], _ = model_luminance(h, "pristine")
        else:
            # The engine costs a minute per point; the pristine variant adds no
            # information that the clear variant does not already give.
            model_pristine[h] = None

    #: Only the fitted range of the reference and the range where the
    #: deterministic model is valid are scored.  h=0 is the sun on the
    #: horizon, h=-15 is airglow, and h=-3 is discussed in the report.
    scored = (-6.0, -9.0, -12.0)
    metrics = []
    for h, rl in zip(hs, ref):
        if h not in scored:
            continue
        ml = model[h]
        dev = math.log10(max(ml, FLOOR) / rl)
        # A point is accepted if it is within a factor 1.5, or if the model's
        # own 1-sigma range reaches that factor (the engine is noise-limited
        # in deep twilight).
        tol = TOL
        if engine_err[h]:
            tol += abs(math.log10(max(1.0 - engine_err[h] / ml, 1e-3)))
        metrics.append(C.Metric(
            name=f"zenith luminance, h={h:+.0f}deg",
            model=ml, reference=rl, unit="cd/m2",
            tolerance=tol, dev=dev, dev_kind="abs",
            note="engine" if h < SPLIT else "deterministic",
        ))

    # -- figure --------------------------------------------------------------
    plt = C.plt()
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.8, 8.2), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.0]),
        constrained_layout=True,
    )

    hh = np.arange(-18.0, -4.4, 0.5)
    ax.plot(hh, [patat_fit(h) for h in hh], "k--", lw=1.5,
            label="Patat et al. (2006), V fit (Paranal, 2400 m)")

    ax.plot(hs, ref, "s-", color="C3", ms=5, lw=1.2,
            label="Koomen et al. (1952), photopic (Maryland, 30 m)")

    # Nawar et al. (2020): yellow filter, the V proxy.  Three Egyptian sites.
    nawar = [r for r in load_nawar() if r["band"].startswith("V")]
    for site, style in (("Helwan", ":"), ("Daraw", "-"), ("Abu Simbel", "-.")):
        rows = sorted([r for r in nawar if r["site"] == site],
                      key=lambda r: float(r["sun_elevation_deg"]))
        ax.plot([float(r["sun_elevation_deg"]) for r in rows],
                [float(r["zenith_luminance_cd_m2"]) for r in rows],
                style, color="0.6", lw=0.9, zorder=1,
                label=("Nawar et al. (2020), yellow filter (Egypt)"
                       if site == "Daraw" else None))

    hcurve = np.arange(-8.0, 0.1, 1.0)
    curve = np.array([model_luminance(float(h), "clear", WL_FIG)[0] for h in hcurve])
    ax.plot(hcurve[hcurve >= -6], curve[hcurve >= -6], "-", color="C0", lw=2.2,
            label="model, deterministic (sea level)")
    ax.plot(hcurve[hcurve <= -6], curve[hcurve <= -6], ":", color="C0", lw=2.0,
            label="deterministic, $-6$ to $-8$ deg (degrades)")
    deep = sorted([h for h in hs if h < SPLIT], reverse=True)
    ax.errorbar(deep, [model[h] for h in deep], yerr=[engine_err[h] for h in deep],
                fmt="o", color="C0", ms=6, capsize=3,
                label="model, spherical engine (quality=low)")

    ax.axvline(-6.0, color="0.55", lw=0.7, ls=":")
    r6 = model[-6.0] / ref[hs.index(-6.0)]
    p6 = model[-6.0] / patat_fit(-6.0)
    ax.annotate(f"-6 deg:\n{r6:.2f}x Koomen\n{p6:.2f}x Patat",
                xy=(-6.0, model[-6.0]), xytext=(-8.2, 1.2), fontsize=7.5,
                color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax.set_yscale("log")
    ax.set_xlim(0.5, -18.5)
    ax.set_ylim(3e-5, 4e2)
    ax.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Check 6: twilight against a second, independent dataset")

    # -- ratio panel ---------------------------------------------------------
    scored = [h for h in hs if h >= -12.0]
    axr.plot(scored, [model[h] / ref[hs.index(h)] for h in scored], "s",
             color="C3", ms=5, label="model / Koomen")
    hp = np.arange(-6.0, -4.4, 0.5)
    axr.plot(hp, [model_luminance(float(h), "clear", WL_FIG)[0] / patat_fit(h)
                  for h in hp], "k--", lw=1.3, label="model / Patat (deterministic)")
    axr.errorbar(deep,
                 [model[h] / patat_fit(h) for h in deep],
                 yerr=[engine_err[h] / patat_fit(h) for h in deep],
                 fmt="o", color="C0", ms=5, capsize=3, label="engine / Patat")
    axr.axvline(-6.0, color="0.55", lw=0.7, ls=":")
    axr.axhline(1.0, color="0.3", lw=0.8)
    axr.axhline(1.5, color="C1", lw=0.9, ls=":")
    axr.text(-18.3, 1.53, "1.5x", color="C1", fontsize=7, va="bottom")
    axr.set_ylim(0.4, 3.8)
    axr.set_xlim(0.5, -18.5)
    axr.set_xlabel("solar elevation [deg]")
    axr.set_ylabel("model / reference")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=7.5, loc="upper center", ncol=3)

    fig.savefig(C.ROOT / "figures" / "13_twilight_datasets.png", dpi=150)
    plt.close(fig)

    # -- the model's own altitude response, for the interpretation ------------
    atmo_alt = C.atmosphere("clear", C.WL_COARSE).at_altitude(2.4)
    rad_alt = sky_radiance_total(atmo_alt, 96.0, np.array([0.0]), np.array([0.0]))
    model_at_2p4km = float(C.luminance(rad_alt[0][None, :], C.WL_COARSE)[0])

    # the engine at -6 deg, so the printed engine ratios are computed, not
    # copied from another run
    atmo_l6 = C.atmosphere("clear", C.WL_COARSE)
    rad_e6, err_e6 = twilight_radiance(
        atmo_l6, 96.0, np.array([0.0]), np.array([0.0]),
        quality="low", seed=0)
    engine_l6 = float(C.luminance(rad_e6[0][None, :], C.WL_COARSE)[0])
    engine_l6_err = float(C.luminance(err_e6[0][None, :], C.WL_COARSE)[0])

    details = dict(
        model_at_2p4km=model_at_2p4km,
        engine_l6=engine_l6,
        engine_l6_err=engine_l6_err,
        model={f"{h:+.0f}": model[h] for h in hs},
        model_pristine={f"{h:+.0f}": model_pristine[h] for h in hs},
        engine_error={f"{h:+.0f}": engine_err[h] for h in hs},
        koomen={f"{h:+.0f}": r for h, r in zip(hs, ref)},
        patat={f"{h:+.0f}": patat_fit(h) for h in hs},
        split=SPLIT,
    )
    summary = (
        f"model/Koomen: -6deg {model[-6.0] / ref[hs.index(-6.0)]:.2f}, "
        f"-9deg {model[-9.0] / ref[hs.index(-9.0)]:.2f}, "
        f"-12deg {model[-12.0] / ref[hs.index(-12.0)]:.2f}; "
        f"model/Patat at -6deg {model[-6.0] / patat_fit(-6.0):.2f}."
    )
    return C.CheckResult(
        key="check6",
        title="Twilight against Koomen et al. (1952)",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    d = result.details
    print("\ncheck6: twilight against a second, independent dataset")
    print("-" * 92)
    print("  Koomen et al. (1952): zenith, photopic (V), Maryland, altitude 30 m.")
    print("  Patat et al. (2006) : zenith, V, Paranal, altitude 2400 m (AOD 0.05, 300 DU).")
    print("  Model: sea level, 300 DU O3; 'clear' has AOD_550 = 0.05, 'pristine' has none.")
    print()
    print(f"  {'h':>5s} {'Koomen':>11s} {'Patat':>11s} {'model(clr)':>11s} "
          f"{'prst':>11s} {'M/Koomen':>16s} {'M/Patat':>8s}")
    for h in [-3.0, -6.0, -9.0, -12.0]:
        k = f"{h:+.0f}"
        ml, mp = d["model"][k], d["model_pristine"][k]
        rk, rp = d["koomen"][k], d["patat"][k]
        err = d["engine_error"][k]
        mps = "--" if mp is None else f"{mp:11.3g}"
        print(f"  {h:5.0f} {rk:11.3g} {rp:11.3g} {ml:11.3g} {mps:>11s} "
              f"{ratio_text(ml, rk, err):>16s} "
              f"{('--' if ml <= 0 else f'{ml / rp:.2f}'):>8s}")
    print()
    print("  Not scored: h=0 (sun on the horizon), where the model gives")
    print(f"  {d['model']['+0']:.3g} cd/m2 against Koomen's {d['koomen']['+0']:.3g} "
          f"({d['model']['+0'] / d['koomen']['+0']:.2f}x); and h=-15, where the measured")
    print("  sky is airglow, which the model does not contain.")
    e6 = d["engine_l6"]
    print(f"  Spherical engine at -6 deg (low budget): {e6:.3f} cd m-2 "
          f"({e6 / d['koomen']['-6']:.2f}x Koomen, "
          f"{e6 / d['patat']['-6']:.2f}x Patat).")
    print()
    print("  Model/reference in brackets is the 1-sigma range from the engine's")
    print("  own statistical error (multiple scattering only; the first order is")
    print("  deterministic).")
    print()
    print("  Secondary check -- Nawar et al. (2020), yellow filter (V proxy):")
    nawar = [r for r in load_nawar() if r["band"].startswith("V")]
    print(f"  {'h':>5s} {'Helwan':>10s} {'Daraw':>10s} {'Abu Simbel':>12s} "
          f"{'model':>10s} {'M/Daraw':>8s}")
    for h in [0.0, -3.0, -6.0, -9.0, -12.0]:
        vals = {}
        for r in nawar:
            if abs(float(r["sun_elevation_deg"]) - h) < 1e-6:
                vals[r["site"]] = float(r["zenith_luminance_cd_m2"])
        m = d["model"][f"{h:+.0f}"]
        print(f"  {h:5.0f} {vals.get('Helwan', float('nan')):10.3g} "
              f"{vals.get('Daraw', float('nan')):10.3g} "
              f"{vals.get('Abu Simbel', float('nan')):12.3g} {m:10.3g} "
              f"{m / vals['Daraw']:8.2f}")
    print()
    print("  The Egyptian yellow values are converted from S10(lambda) units and")
    print("  are not Johnson V; their site-to-site spread (1.5 mag at h=-6) is a")
    print("  measure of the systematic uncertainty of that conversion, not of the sky.")
    print()
    print(f"  Model altitude response at -6 deg: {d['model']['-6']:.3f} cd/m2 at sea level")
    print(f"  -> {d['model_at_2p4km']:.3f} cd/m2 at 2.4 km (756 hPa), i.e. "
          f"{d['model_at_2p4km'] / d['model']['-6']:.2f}x brighter.")
    print("  The measurements are the other way round: Koomen's Maryland sky is")
    print("  1.25x brighter than Patat's Paranal sky, and Patat's own 961->743 hPa")
    print("  pair gives 1.28x.  These are cross-site comparisons, not a controlled")
    print("  intervention on altitude alone.  at_altitude raises a global opaque")
    print("  sphere, including the solar shadow, with no column renormalisation.")
    print("  See docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md for the residual and its limits.")
    print()
    print(f"  verdict: {result.verdict}")
    print()
    d6k = d["model"]["-6"] / d["koomen"]["-6"]
    d6p = d["model"]["-6"] / d["patat"]["-6"]
    e6k = d["engine_l6"] / d["koomen"]["-6"]
    e6p = d["engine_l6"] / d["patat"]["-6"]
    print("  Read the -6 deg row first.  After the two-branch solar-chord fix (M10)")
    print(f"  the model is {d6k:.2f}x Koomen (deterministic) or {e6k:.2f}x (engine),")
    print(f"  and {d6p:.2f}x / {e6p:.2f}x Patat.  The old 1.14x / 2.20x were the one-branch")
    print("  chord.  Remaining residuals do not independently identify calibration,")
    print("  site conditions or missing scattering; see docs/twilight/TWILIGHT_REFERENCE_AUDIT.md.")
    print("  The old 1.208x altitude ratio included solar rays crossing the raised")
    print("  opaque ground.  That defect is fixed; the current deterministic ratio")
    print("  is printed above.  A cross-study ratio of 0.80 is not a calibration target.")
    return 0 if result.verdict in ("PASS", "SKIP") else 1


if __name__ == "__main__":
    raise SystemExit(main())
