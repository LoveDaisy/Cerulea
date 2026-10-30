"""Experiment 12 -- can the vertical structure explain the twilight excess?

The model reproduces the zenith twilight luminance at -3 deg solar elevation
but is 1.5-1.6x too bright at -6 deg (and 1.4-1.5x too bright at -9/-12 deg in
the spherical engine).  The column amounts -- ozone 300 DU, AOD, surface
pressure -- have already been ruled out (``docs/ROADMAP.md``, open question 2).
This experiment asks the next question: **is the vertical *shape* of the
atmosphere the missing input?**

Three levers are swept, all at a fixed 300 DU ozone column and sea level:

1. **ozone profile shape** -- peak altitude 20/22/25/28/32 km (a rigid height
   shift of the climatological table), Gaussian width sigma = 3/6/9 km
   (centred at 25 km), and the tropospheric partition 0/30/60 DU below 10 km;
2. **temperature above 20 km** -- a constant additive offset of -30..+30 K,
   with the pressure re-integrated hydrostatically so that the surface stays
   anchored and the column above 20 km is conserved (a pure redistribution of
   the same air);
3. **ground albedo** 0.0/0.1/0.3 and the **phase-function truncation**
   ``n_moments`` = 16/32/48.

Plus a **contribution function**: the logarithmic response of the -6 deg
zenith radiance to a density change in each altitude band, which shows *why*
each lever does what it does.

The variants live in ``experiments/profile_variants.py``; nothing in
``skycolor/`` is modified.

Run::

    uv run python experiments/12_profile_sensitivity.py

Output: ``figures/12_profile_sensitivity.png`` and
``experiments/out/12_profile_sensitivity.json``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import OUT, WL, save  # noqa: E402
from profile_variants import (  # noqa: E402
    BandDensityAtmosphere,
    OffsetTemperatureAtmosphere,
    OzoneShapeAtmosphere,
    air_diagnostics,
    gaussian_shape,
    install_twilight_profile_support,
    ozone_diagnostics,
    shifted_shape,
    tropospheric_shape,
)
from skycolor import spectra  # noqa: E402
from skycolor.atmosphere import Atmosphere  # noqa: E402
from skycolor.constants import WL_DEFAULT  # noqa: E402
from skycolor.rt.multiple import multiple_scattering  # noqa: E402
from skycolor.rt.single import sky_radiance as single_radiance  # noqa: E402

# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------
#: Patat, Ugolnikov & Postylyakov (2006), A&A 455, 385: V-band zenith
#: luminance converted with ``L = 1.08e5 * 10^(-0.4 m_V)`` cd m^-2.  The
#: -3 deg point is an extrapolation of the second-order fit (valid
#: 95 <= zeta <= 105 deg, i.e. -5 .. -15 deg).
PATAT = {-3.0: 40.09, -6.0: 0.516, -9.0: 0.0171, -12.0: 0.00146}

#: the elevations the sweep is run at.  -3 and -6 are the two reference
#: points quoted in the roadmap; -4 and -8 bracket -6 and are reported for
#: shape, but the deterministic model is not trustworthy at -8 (it is 2.8x
#: too dark there; see docs/TWILIGHT.md, section 2.1).
ELEVATIONS = (-3.0, -4.0, -6.0, -8.0)

ALBEDO = 0.1
OZONE_DU = 300.0


def patat_fit(h_deg: float) -> float:
    """The Patat et al. (2006) second-order fit, converted to cd m^-2."""
    zeta = 90.0 - h_deg
    m_v = 11.84 + 1.518 * (zeta - 95.0) - 0.057 * (zeta - 95.0) ** 2
    return 1.08e5 * 10.0 ** (-0.4 * m_v)


V_PHOT = spectra.cie_xyz(WL)[:, 1]


def luminance(radiance) -> float:
    """Photopic luminance [cd m^-2] of a spectrum on the 5 nm grid."""
    return float(683.0 * np.trapezoid(np.asarray(radiance) * V_PHOT, WL))


def solve_luminance(atmo, h_deg: float, n_moments: int = 32):
    """(single, total) zenith luminance at solar elevation ``h_deg``."""
    sz = 90.0 - h_deg
    s = single_radiance(atmo, sz, np.array([0.0]), np.array([0.0]),
                        n_gauss=3, chunk=512)[0]
    m = multiple_scattering(atmo, sz, np.array([0.0]), np.array([0.0]),
                            n_moments=n_moments)[0]
    return luminance(s), luminance(s + m)


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------
def base_kw(**over):
    kw = dict(ozone_du=OZONE_DU, aerosols=[], ground_albedo=ALBEDO)
    kw.update(over)
    return kw


def build_cases():
    """Every swept case: ``(lever, label, x, x_unit, factory, n_moments)``."""
    cases = []

    def add(lever, label, x, x_unit, factory, n_moments=32):
        cases.append(dict(lever=lever, label=label, x=float(x), x_unit=x_unit,
                          factory=factory, n_moments=n_moments))

    # -- 1a. ozone peak altitude: a rigid shift of the climatological table.
    # The table's maximum sits at 22 km, so the shift is ``peak - 22``.
    for peak in (20.0, 22.0, 25.0, 28.0, 32.0):
        add("ozone_peak", f"peak {peak:.0f} km", peak, "km",
            (lambda peak=peak: OzoneShapeAtmosphere(
                WL, shifted_shape(peak - 22.0), **base_kw())))

    # -- 1b. ozone width: a Gaussian centred on the climatological peak.
    for sigma in (3.0, 6.0, 9.0):
        add("ozone_sigma", f"sigma {sigma:.0f} km", sigma, "km",
            (lambda sigma=sigma: OzoneShapeAtmosphere(
                WL, gaussian_shape(sigma, center_km=25.0), **base_kw())))

    # -- 1c. tropospheric ozone: split the 300 DU at 10 km.
    for trop in (0.0, 30.0, 60.0):
        add("ozone_trop", f"trop {trop:.0f} DU", trop, "DU",
            (lambda trop=trop: OzoneShapeAtmosphere(
                WL, tropospheric_shape(trop, split_km=10.0,
                                       total_du=OZONE_DU), **base_kw())))

    # -- 2. temperature above 20 km (step offset; +/-30 K is context only).
    for dT in (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0):
        add("temperature", f"dT {dT:+.0f} K", dT, "K",
            (lambda dT=dT: OffsetTemperatureAtmosphere(
                WL, dT, z_anchor_km=20.0, **base_kw())))

    # -- 3a. ground albedo.
    for alb in (0.0, 0.1, 0.3):
        add("albedo", f"albedo {alb:.1f}", alb, "-",
            (lambda alb=alb: Atmosphere(WL, **base_kw(ground_albedo=alb))))

    # -- 3b. phase-function truncation (same atmosphere, different moments).
    for nm in (16, 32, 48):
        add("moments", f"n_moments {nm}", nm, "-",
            (lambda: Atmosphere(WL, **base_kw())), n_moments=nm)

    return cases


# ---------------------------------------------------------------------------
# contribution function
# ---------------------------------------------------------------------------
CONTRIB_BANDS = [(0, 10), (10, 20), (20, 25), (25, 30), (30, 35), (35, 40),
                 (40, 45), (45, 50), (50, 60), (60, 80), (80, 120)]
CONTRIB_EPS = 0.05


def contribution_function(h_deg: float = -6.0):
    """``d ln L / d ln n_air`` in each altitude band, by central difference."""
    out = []
    for z0, z1 in CONTRIB_BANDS:
        vals = []
        for sign in (+1.0, -1.0):
            atmo = BandDensityAtmosphere(
                WL, (z0, z1), 1.0 + sign * CONTRIB_EPS, **base_kw())
            _, total = solve_luminance(atmo, h_deg)
            vals.append(total)
        slope = (np.log(vals[0]) - np.log(vals[1])) / (2.0 * CONTRIB_EPS)
        out.append(dict(band_km=[z0, z1], dlnL_dlnn=float(slope)))
    return out


# ---------------------------------------------------------------------------
# spherical-engine spot check
# ---------------------------------------------------------------------------
def engine_check():
    """Engine totals at -6 deg for the reference and a -20 K stratosphere.

    The engine's Monte Carlo term reconstructs a plain ``Atmosphere`` before
    each single-wavelength run and therefore needs the run-time patch from
    ``profile_variants.install_twilight_profile_support`` to see a subclass
    profile.  Costs ~1.5 min at ``quality="low"``.
    """
    from skycolor.rt.twilight import twilight_radiance

    install_twilight_profile_support()
    out = {}
    for label, dT in (("reference", 0.0), ("dT=-20 K", -20.0)):
        atmo = (Atmosphere(WL, **base_kw()) if dT == 0.0 else
                OffsetTemperatureAtmosphere(WL, dT, z_anchor_km=20.0, **base_kw()))
        rad, err = twilight_radiance(atmo, 96.0, np.array([0.0]), np.array([0.0]),
                                     quality="low", seed=0)
        total = luminance(rad[0])
        out[label] = dict(luminance_6=total,
                          stderr=float(luminance(err[0])),
                          ratio_to_patat=total / PATAT[-6.0])
        print(f"  engine {label:10s}: L(-6)={total:.4f} +- "
              f"{luminance(err[0]):.4f}  (x{total / PATAT[-6.0]:.2f} Patat)")
    # the first-order term is deterministic, so the engine's MS follows exactly
    single = {}
    for label, dT in (("reference", 0.0), ("dT=-20 K", -20.0)):
        atmo = (Atmosphere(WL, **base_kw()) if dT == 0.0 else
                OffsetTemperatureAtmosphere(WL, dT, z_anchor_km=20.0, **base_kw()))
        single[label] = solve_luminance(atmo, -6.0)[0]
        out[label]["single_6"] = single[label]
        out[label]["multiple_6"] = out[label]["luminance_6"] - single[label]
    out["ratio_dT_minus20_over_reference"] = (
        out["dT=-20 K"]["luminance_6"] / out["reference"]["luminance_6"])
    out["note"] = ("unpatched, the engine returns bit-identical multiple "
                   "scattering for both profiles; install_twilight_profile_support "
                   "is required for these numbers")
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def main(run_engine: bool = False):
    t_start = time.time()
    print("Experiment 12 -- vertical-structure sensitivity of the twilight zenith")
    print(f"ozone {OZONE_DU:.0f} DU, no aerosol, ground albedo {ALBEDO}, "
          f"{len(WL)} wavelengths, solar zenith angles "
          f"{', '.join(f'{-h:.0f}' for h in ELEVATIONS)}\n")

    ref = Atmosphere(WL, **base_kw())
    ref_lum = {}
    ref_single = {}
    for h in ELEVATIONS:
        s, t = solve_luminance(ref, h)
        ref_single[h] = s
        ref_lum[h] = t
    ref_diag = dict(ozone=ozone_diagnostics(ref), air=air_diagnostics(ref))
    print("reference atmosphere")
    print(f"  L(-3)={ref_lum[-3.0]:.4f}  L(-4)={ref_lum[-4.0]:.4f}  "
          f"L(-6)={ref_lum[-6.0]:.5f}  L(-8)={ref_lum[-8.0]:.5f} cd m^-2")
    print(f"  single-scattering share at -6: "
          f"{ref_single[-6.0] / ref_lum[-6.0] * 100:.0f}%   "
          f"ozone column {ref_diag['ozone']['ozone_column_du']:.1f} DU "
          f"({ref_diag['ozone']['tropospheric_du']:.1f} DU below 10 km), "
          f"Rayleigh tau(550) {ref_diag['air']['rayleigh_tau_550']:.4f}\n")

    cases = build_cases()
    rows = []
    for case in cases:
        atmo = case["factory"]()
        lum = {}
        for h in ELEVATIONS:
            _, t = solve_luminance(atmo, h, n_moments=case["n_moments"])
            lum[h] = t
        rows.append(dict(
            lever=case["lever"], label=case["label"], x=case["x"],
            x_unit=case["x_unit"], n_moments=case["n_moments"],
            luminance={f"{h:+.0f}": lum[h] for h in ELEVATIONS},
            ratio_to_base={f"{h:+.0f}": lum[h] / ref_lum[h] for h in ELEVATIONS},
            ratio_to_patat={f"{h:+.0f}": lum[h] / patat_fit(h) for h in ELEVATIONS},
            ozone=ozone_diagnostics(atmo),
            air=air_diagnostics(atmo),
        ))
        print(f"  {case['lever']:12s} {case['label']:14s} "
              f"L(-6)={lum[-6.0]:.5f}  x{ lum[-6.0] / ref_lum[-6.0]:.3f} "
              f"  ({time.time() - t_start:.0f}s)")

    # -- phase-function truncation: exact or not? ---------------------------
    mom_check = {}
    for nm in (16, 48):
        atmo = Atmosphere(WL, **base_kw())
        r16 = multiple_scattering(atmo, 96.0, np.array([0.0]), np.array([0.0]),
                                  n_moments=nm)[0]
        r32 = multiple_scattering(atmo, 96.0, np.array([0.0]), np.array([0.0]),
                                  n_moments=32)[0]
        mom_check[f"n_moments={nm}"] = float(np.max(np.abs(r16 - r32) / r32))
    print(f"\n  n_moments 16 vs 32: max relative radiance difference "
          f"{mom_check['n_moments=16']:.2e} (Rayleigh has only l<=2)")

    # -- contribution function ---------------------------------------------
    print("\ncontribution function at -6 deg (d ln L / d ln n_air):")
    contrib = contribution_function(-6.0)
    for c in contrib:
        print(f"  {c['band_km'][0]:3d}-{c['band_km'][1]:3d} km  "
              f"{c['dlnL_dlnn']:+.3f}")

    # -- how cold would the stratosphere have to be? ------------------------
    t_rows = sorted((r for r in rows if r["lever"] == "temperature"),
                    key=lambda r: r["x"])
    dts = np.array([r["x"] for r in t_rows])
    l6 = np.array([r["luminance"]["-6"] for r in t_rows])
    target = PATAT[-6.0]
    dT_required = float(np.interp(target, l6, dts))
    # one refinement solve at the interpolated offset
    atmo = OffsetTemperatureAtmosphere(WL, dT_required, z_anchor_km=20.0,
                                       **base_kw())
    _, l6_refined = solve_luminance(atmo, -6.0)
    print(f"\n  a -6 deg luminance of {target:.4f} cd m^-2 requires "
          f"dT = {dT_required:+.1f} K above 20 km "
          f"(gives {l6_refined:.4f}, i.e. {l6_refined / target:.3f} x Patat)")

    # -- ramp robustness: is the step's density jump doing the work? --------
    ramp = {}
    for dT in (-20.0, 20.0):
        atmo = OffsetTemperatureAtmosphere(WL, dT, z_anchor_km=20.0,
                                           ramp_km=10.0, **base_kw())
        _, t = solve_luminance(atmo, -6.0)
        ramp[f"dT={dT:+.0f} K ramped 20->30 km"] = dict(
            luminance_6=t, ratio_to_base=t / ref_lum[-6.0])
        step = next(r["ratio_to_base"]["-6"] for r in t_rows if r["x"] == dT)
        print(f"  robustness: dT={dT:+.0f} K ramped over 20-30 km -> "
              f"L(-6)={t:.5f} (x{t / ref_lum[-6.0]:.3f}); step gives x{step:.3f}")

    # -- spectral-grid check ------------------------------------------------
    grid_check = {}
    v1 = spectra.cie_xyz(WL_DEFAULT)[:, 1]
    for name, factory in (
        ("reference", lambda w: Atmosphere(w, **base_kw())),
        ("dT=-20 K", lambda w: OffsetTemperatureAtmosphere(
            w, -20.0, z_anchor_km=20.0, **base_kw())),
    ):
        # the same physical case solved on the 5 nm grid and the 1 nm grid
        pair = {}
        for tag, grid in (("l_5nm", WL), ("l_1nm", WL_DEFAULT)):
            atmo = factory(grid)
            vv = spectra.cie_xyz(grid)[:, 1]
            s1 = single_radiance(atmo, 96.0, np.array([0.0]), np.array([0.0]))[0]
            m1 = multiple_scattering(atmo, 96.0, np.array([0.0]), np.array([0.0]))[0]
            pair[tag] = float(683.0 * np.trapezoid((s1 + m1) * vv, grid))
        grid_check[name] = pair
        print(f"  grid check {name:10s}: 5 nm {pair['l_5nm']:.5f}  "
              f"1 nm {pair['l_1nm']:.5f}  ratio {pair['l_5nm'] / pair['l_1nm']:.4f}")

    # -----------------------------------------------------------------------
    # table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{'lever':12s} {'variant':14s} {'x':>6s} "
          f"{'L(-3)':>9s} {'L(-4)':>9s} {'L(-6)':>9s} {'L(-8)':>9s} "
          f"{'x base -6':>9s} {'/Patat -6':>9s}")
    print("-" * 96)
    for r in rows:
        l = r["luminance"]
        print(f"{r['lever']:12s} {r['label']:14s} {r['x']:6.1f} "
              f"{l['-3']:9.4f} {l['-4']:9.4f} {l['-6']:9.5f} {l['-8']:9.5f} "
              f"{r['ratio_to_base']['-6']:9.3f} {r['ratio_to_patat']['-6']:9.3f}")
    print("-" * 96)
    print(f"{'reference':12s} {'-':14s} {'-':>6s} "
          f"{ref_lum[-3.0]:9.4f} {ref_lum[-4.0]:9.4f} {ref_lum[-6.0]:9.5f} "
          f"{ref_lum[-8.0]:9.5f} {1.0:9.3f} "
          f"{ref_lum[-6.0] / PATAT[-6.0]:9.3f}")
    print(f"{'Patat 2006':12s} {'-':14s} {'-':>6s} "
          f"{PATAT[-3.0]:9.4f} {patat_fit(-4.0):9.4f} {PATAT[-6.0]:9.5f} "
          f"{patat_fit(-8.0):9.5f} {'-':>9s} {'1.000':>9s}")
    print("=" * 96)

    # -- optional spherical-engine spot check ------------------------------
    engine = None
    if run_engine:
        print("\nspherical engine spot check at -6 deg (patched; ~1.5 min):")
        engine = engine_check()
        print(f"  engine x(dT=-20 K)/x(reference) = "
              f"{engine['ratio_dT_minus20_over_reference']:.3f}")

    # -----------------------------------------------------------------------
    # figure
    # -----------------------------------------------------------------------
    plot_figure(rows, ref_lum, contrib, dT_required)

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------
    payload = dict(
        experiment="12_profile_sensitivity",
        description="Vertical-structure sensitivity of the twilight zenith "
                    "luminance; see docs/PROFILES.md.",
        atmosphere=dict(ozone_du=OZONE_DU, aerosols=[], ground_albedo=ALBEDO,
                        surface_pressure_Pa=101325.0, elevation_km=0.0),
        wavelengths_nm=[float(WL[0]), float(WL[-1]), float(WL[1] - WL[0])],
        elevations_deg=list(ELEVATIONS),
        patat_2006={f"{h:+.0f}": PATAT[h] for h in PATAT},
        patat_fit={f"{h:+.0f}": patat_fit(h) for h in ELEVATIONS},
        reference=dict(luminance={f"{h:+.0f}": ref_lum[h] for h in ELEVATIONS},
                       single={f"{h:+.0f}": ref_single[h] for h in ELEVATIONS},
                       diagnostics=ref_diag),
        cases=rows,
        phase_moment_check=mom_check,
        contribution={"-6": contrib},
        temperature_required=dict(
            target_cd_m2=target, dT_K=dT_required,
            luminance_6=l6_refined),
        robustness=dict(ramp=ramp, spectral_grid=grid_check),
        engine=engine,
        verdict=dict(
            max_reduction_at_minus6=min(r["ratio_to_base"]["-6"] for r in rows
                                        if r["lever"] != "temperature"),
            temperature_only_reduction=min(r["ratio_to_base"]["-6"] for r in t_rows),
            required_reduction=target / ref_lum[-6.0],
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "12_profile_sensitivity.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"\n  json -> {path.relative_to(path.parents[2])}")
    print(f"  done in {time.time() - t_start:.0f}s")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def plot_figure(rows, ref_lum, contrib, dT_required):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8))
    axes = axes.ravel()

    col6, col4 = "C0", "C1"
    lw6, lw4 = 2.2, 1.1

    def decorate(ax, title, xlabel):
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(0.25, 60.0)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, which="both")
        # Patat et al. (2006) at the two plotted elevations
        ax.axhline(PATAT[-6.0], color="0.35", ls="--", lw=1.0, zorder=1)
        ax.axhline(patat_fit(-4.0), color="0.35", ls=":", lw=1.0, zorder=1)

    # (a) ozone peak altitude
    ax = axes[0]
    sub = sorted((r for r in rows if r["lever"] == "ozone_peak"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    ax.plot([22], [ref_lum[-6.0]], "o", mfc="none", mec="k", ms=9, mew=1.0)
    decorate(ax, "(a) ozone peak altitude", "peak altitude [km]")
    ax.text(20.2, 2.6, "climatological peak = 22 km\n(open circle = reference)",
            fontsize=7.2, color="0.3", va="center")

    # (b) ozone width
    ax = axes[1]
    sub = sorted((r for r in rows if r["lever"] == "ozone_sigma"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    decorate(ax, "(b) ozone width (Gaussian, 25 km)", "sigma [km]")
    ax.text(3.2, 2.6, "the reference profile is not\nGaussian (peak 22, median 24.6)",
            fontsize=7.2, color="0.3", va="center")

    # (c) tropospheric ozone
    ax = axes[2]
    sub = sorted((r for r in rows if r["lever"] == "ozone_trop"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    ax.plot([15.5], [ref_lum[-6.0]], "o", mfc="none", mec="k", ms=9, mew=1.0)
    decorate(ax, "(c) tropospheric ozone (< 10 km)", "ozone below 10 km [DU]")
    ax.text(1.5, 2.6, "reference = 15.5 DU below 10 km", fontsize=7.2,
            color="0.3", va="center")

    # (d) temperature above 20 km
    ax = axes[3]
    sub = sorted((r for r in rows if r["lever"] == "temperature"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    ax.axvline(dT_required, color="C3", ls="--", lw=1.2)
    ax.text(-32.0, 45.0, f"needs {dT_required:+.0f} K",
            fontsize=7.5, color="C3", va="top")
    ax.plot([0], [ref_lum[-6.0]], "o", mfc="none", mec="k", ms=9, mew=1.0)
    decorate(ax, "(d) temperature above 20 km", "dT [K] (surface anchored)")
    ax.set_xlim(-33, 33)

    # (e) ground albedo
    ax = axes[4]
    sub = sorted((r for r in rows if r["lever"] == "albedo"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    decorate(ax, "(e) ground albedo", "Lambertian albedo")
    ax.text(0.02, 2.6, "the ground is in shadow at -6$^\\circ$;\n"
                       "only the sky it reflects matters",
            fontsize=7.2, color="0.3", va="center")

    # (f) phase-function moments
    ax = axes[5]
    sub = sorted((r for r in rows if r["lever"] == "moments"), key=lambda r: r["x"])
    x = [r["x"] for r in sub]
    ax.plot(x, [r["luminance"]["-6"] for r in sub], "o-", color=col6, lw=lw6, ms=5)
    ax.plot(x, [r["luminance"]["-4"] for r in sub], "o-", color=col4, lw=lw4, ms=3.5)
    decorate(ax, "(f) phase-function truncation", "Legendre moments")
    ax.text(17.0, 2.6, "Rayleigh has only $l \\leq 2$:\nall three are bit-identical",
            fontsize=7.2, color="0.3", va="center")

    # (g) contribution function
    ax = axes[6]
    labels = [f"{c['band_km'][0]}-{c['band_km'][1]}" for c in contrib]
    vals = [c["dlnL_dlnn"] for c in contrib]
    colours = ["C3" if v < 0 else "C0" for v in vals]
    ax.bar(range(len(vals)), vals, color=colours)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, fontsize=6.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("(g) where the -6$^\\circ$ light comes from", fontsize=9.5)
    ax.set_xlabel("altitude band [km]", fontsize=8.5)
    ax.set_ylabel(r"$d\ln L / d\ln n_{\rm air}$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")
    ax.text(0.03, 0.97, "red: shadowed absorber\n(more air = dimmer)",
            transform=ax.transAxes, fontsize=7, color="C3", va="top")

    # (h) summary
    ax = axes[7]
    series = [
        ("ozone peak 20-32 km", "ozone_peak"),
        ("ozone width $\\sigma$ 3-9", "ozone_sigma"),
        ("tropospheric O$_3$ 0-60 DU", "ozone_trop"),
        ("$\\Delta T$ $\\pm$10 K", "temperature_10"),
        ("$\\Delta T$ $\\pm$20 K", "temperature_20"),
        ("albedo 0.0-0.3", "albedo"),
        ("moments 16-48", "moments"),
    ]
    ys, lo, hi, txt = [], [], [], []
    for name, key in series:
        if key.startswith("temperature_"):
            lim = 10.0 if key.endswith("10") else 20.0
            sub = [r for r in rows if r["lever"] == "temperature"
                   and abs(r["x"]) <= lim + 1e-9]
        else:
            sub = [r for r in rows if r["lever"] == key]
        ratios = [r["ratio_to_base"]["-6"] for r in sub]
        ys.append(name)
        lo.append(min(ratios))
        hi.append(max(ratios))
        txt.append(f"{min(ratios):.2f}-{max(ratios):.2f}")
    ypos = np.arange(len(ys))
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.plot([a, b], [i, i], "-", color="C0", lw=5, solid_capstyle="butt")
        if b - a < 1e-6:
            ax.plot([a], [i], "s", color="C0", ms=6)
    ax.axvline(1.0, color="k", lw=0.9)
    ax.axvline(PATAT[-6.0] / ref_lum[-6.0], color="C3", ls="--", lw=1.3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ys, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.35)
    ax.set_ylim(len(ys) - 0.4, -0.7)
    ax.set_xlabel("zenith luminance at -6$^\\circ$ / reference "
                  f"(dashed: needed {PATAT[-6.0] / ref_lum[-6.0]:.2f}x)", fontsize=8.2)
    ax.set_title("(h) range of each lever", fontsize=9.5)
    ax.grid(alpha=0.25, axis="x")
    for i, t in enumerate(txt):
        ax.text(hi[i] + 0.015, i, t, fontsize=6.8, va="center", color="0.25")

    # legend for the two elevations and reference lines, on panel (a)
    a = axes[0]
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=col6, lw=lw6, marker="o", ms=4,
               label="zenith at -6$^\\circ$"),
        Line2D([], [], color=col4, lw=lw4, marker="o", ms=3,
               label="zenith at -4$^\\circ$"),
        Line2D([], [], color="0.35", ls="--", lw=1.0,
               label="Patat et al. (2006), -6$^\\circ$"),
        Line2D([], [], color="0.35", ls=":", lw=1.0,
               label="Patat fit, -4$^\\circ$ (extrap.)"),
    ]


    fig.suptitle("Vertical structure vs the 1.5x twilight excess at -6$^\\circ$ "
                 "(300 DU ozone, no aerosol, sea level)", fontsize=11, y=0.995)
    fig.legend(handles=handles, fontsize=7.5, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "12_profile_sensitivity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", action="store_true",
                    help="also run the spherical-engine spot check (~1.5 min)")
    main(run_engine=ap.parse_args().engine)
