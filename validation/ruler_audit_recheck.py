"""Ruler audit recheck -- independent recomputations for RULER_AUDIT.md.

This entry point belongs to the read-only对照合同 audit of check2 / check3 /
check6 (task-ruler-audit).  It recomputes the numbers quoted in
``validation/RULER_AUDIT.md`` from the committed model and reference data, so
every figure in the report can be regenerated with one command::

    uv run python -m validation.ruler_audit_recheck            # all sections
    uv run python -m validation.ruler_audit_recheck --only check2

It is deliberately NOT part of ``run_all.CHECKS`` and NOT collected by pytest:
the audit adds no runtime to the validation harness.  It reads the model only
through the same public helpers the checks use (``validation.common``,
production solver entry points) and never writes anything except stdout.
Physical numbers of ``validation/results.json`` are inputs to this audit, not
outputs; the recheck never touches them.

Evidence-grade legend used in the report:
* 原文前提核对 -- premise read from the original publication (or its archived
  full-text audit, ``docs/twilight/TWILIGHT_REFERENCE_AUDIT.md``),
* 复算       -- computed here from committed data/models,
* 复现（二级） -- premise cross-checked against a named public reproduction of
  a paywalled original (pvlib 0.11 source; Zotti 2007 thesis),
* 推断       -- inference, explicitly labelled as such.

Compute budget
--------------
The expensive items are full-solver sky maps.  The check-level metrics run on
a 73x144 grid (~200 s per solve); the audit's factors and scans are computed
on the 37x72 grid (~60-70 s per solve) and validated against the committed
check3 baseline point, with the CIE-side discretisation difference quantified
analytically.  Independent solves run in a small process pool so the whole
audit stays in the ~5 min class (plan §7 risk 2).  Engine points are never
recomputed: check6's deep-twilight numbers are read from results.json.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from validation import common as C  # noqa: E402

SECTIONS = ("check2", "check3", "check6")

# Solar-elevation refraction offset (Saemundsson's formula, as reproduced in
# Meeus, "Astronomical Algorithms"; R = 1.02/tan(h + 10.3/(h+5.11)) arcmin).
# This is the true-vs-apparent elevation difference the model input convention
# could absorb; values: h=30 -> 1.75', h=45 -> 1.01', h=60 -> 0.59'.
def saemundsson_refraction_arcmin(h_deg: float) -> float:
    h = float(h_deg)
    return 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11)))


def _hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# shared sky-solve job machinery (executed in the process pool)
# ---------------------------------------------------------------------------
def _atmo_for(kind: str, aod: float, ozone_du: float, wl: np.ndarray,
              ground_albedo: float | None):
    """Fresh Atmosphere with the check-level scene parameters.

    Built directly from the preset (not through ``C.atmosphere``) so the
    albedo probe can vary ``ground_albedo`` on an instance owned by this
    process; the shared aerosol instances stay read-only.
    """
    from skycolor import presets

    if kind == "pristine":
        atmo = presets.pristine(wl, ozone_du=ozone_du)
    else:
        atmo = presets.clear(wl, aod=aod, ozone_du=ozone_du)
    if ground_albedo is not None:
        atmo.ground_albedo = float(ground_albedo)
    return atmo


def _sky_job(spec: dict) -> dict:
    """One full-solver sky case on the audit grid.

    Returns zenith luminance, DHI, and the check3-style relative-distribution
    statistics (model vs CIE 110 on the same grid).  Runs inside a pool
    worker; every physical call is the same ``sky_radiance_total`` the checks
    use.
    """
    from skycolor.rt.solver import sky_radiance_total

    wl = C.WL_COARSE
    n_zen, n_az = spec.get("grid", (37, 72))
    atmo = _atmo_for(spec.get("kind", "clear"), spec.get("aod", 0.05),
                     spec.get("ozone_du", 300.0), wl, spec.get("albedo"))
    sun_elev = float(spec["sun_elev"])
    z, a = C.sky_grid(n_zen, n_az)
    rad = sky_radiance_total(atmo, 90.0 - sun_elev, z.ravel(), a.ravel())
    rad = rad.reshape(z.shape + (len(wl),))
    lum = C.luminance(rad, wl)
    lz = float(lum[0, 0])
    rel = lum / lz
    dhi = C.horizontal_illuminance(rad, z, a, wl)
    cie = C.cie110_relative_luminance(z, a, 90.0 - sun_elev)
    dev = C.rel_dev(rel, cie)
    upper = z <= 60.0
    mid = (z > 60.0) & (z <= 75.0)
    horizon = z >= 75.0
    return dict(
        name=spec["name"],
        lz=lz,
        dhi=dhi,
        horizon_sun=float(rel[-1, 0]),
        horizon_anti=float(rel[-1, n_az // 2]),
        cie_horizon_sun=float(cie[-1, 0]),
        cie_horizon_anti=float(cie[-1, n_az // 2]),
        rms_upper=float(np.sqrt(np.mean(dev[upper] ** 2))),
        rms_mid=float(np.sqrt(np.mean(dev[mid] ** 2))),
        rms_horizon=float(np.sqrt(np.mean(dev[horizon] ** 2))),
        rms_all=float(np.sqrt(np.mean(dev ** 2))),
    )


def _cie_integral_factor(sun_zen_deg: float, n_zen: int, n_az: int) -> float:
    """Hemisphere integral factor  ∫rel·cosθ dΩ  of the CIE 110 sky."""
    z, a = C.sky_grid(n_zen, n_az)
    rel = C.cie110_relative_luminance(z, a, sun_zen_deg)
    return C.horizontal_illuminance(rel, z, a)  # Lz = 1 units


def run_sky_jobs(jobs: list[dict], workers: int = 6) -> dict[str, dict]:
    """Run independent sky solves in a small spawn pool; return by name."""
    t0 = time.time()
    ctx = mp.get_context("spawn")
    out: dict[str, dict] = {}
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(_sky_job, jobs):
            out[res["name"]] = res
            print(f"  [sky] {res['name']:<16s} lz={res['lz']:9.2f} "
                  f"dhi={res['dhi']/1000:7.3f} klx "
                  f"hsun={res['horizon_sun']:6.2f} "
                  f"({time.time()-t0:5.1f}s)")
    print(f"  [sky] {len(jobs)} solves in {time.time()-t0:.1f}s wall")
    return out


# ---------------------------------------------------------------------------
# check2 -- CIE 110 DHI / G173 / Bird reference premises
# ---------------------------------------------------------------------------
def _g173_columns():
    data = np.loadtxt(C.ROOT / "data_raw" / "ASTMG173.csv",
                      delimiter=",", skiprows=2)
    return data[:, 0], data[:, 1], data[:, 3]  # wl, ET, DNI


def _hinterland_aod(wl: np.ndarray, aod550: float) -> np.ndarray:
    """Spectral AOD of the Hinterland aerosol at the preset loading.

    The preset scales the layer so that the *column* extinction at 550 nm
    equals ``aod550``; the spectral shape comes from the aerosol's own
    extinction cross section (Ångström-like, RH 50 % rural model).
    """
    from skycolor.aerosol import HinterlandAerosol, shared

    a = shared(HinterlandAerosol)
    return a.extinction(wl) / a.extinction(np.array([550.0])) * aod550


def recheck_check2(workers: int) -> None:
    _hdr("check2 -- illuminance reference contract")
    from skycolor import spectra
    from skycolor.rt.single import direct_sun_transmittance

    # -- 2a: G173 premises -------------------------------------------------
    wl_g, et_g, dni_g = _g173_columns()
    ybar_g = spectra.cie_xyz(wl_g)[:, 1]
    vis = (wl_g >= 380) & (wl_g <= 780)
    g173_klx = 683.0 * np.trapezoid((dni_g * ybar_g)[vis], wl_g[vis]) / 1000.0
    g173_w = float(np.trapezoid(dni_g, wl_g))
    print(f"2a  G173 DNI photopic illuminance 380-780 = {g173_klx:.3f} klx; "
          f"broadband 280-4000 = {g173_w:.2f} W/m2; "
          f"efficacy = {g173_klx*1000/g173_w:.2f} lm/W")

    # G173 vs model vertical transmission, per wavelength: continuum = aerosol
    # correspondence, dips = model-missing absorption bands (water 720-740,
    # O2 A 760).  The model scene: clear preset, AOD550=0.0734, O3=343.8 DU.
    wl_model = np.arange(380.0, 781.0, 1.0)
    atmo_g173 = _atmo_for("clear", 0.0734, 343.8, wl_model, None)
    t_model = direct_sun_transmittance(atmo_g173, 48.19)
    dni_m = np.interp(wl_model, wl_g, dni_g)
    et_m = np.interp(wl_model, wl_g, et_g)
    ybar_m = spectra.cie_xyz(wl_model)[:, 1]
    t_g173 = dni_m / et_m
    rho = t_g173 / t_model
    cont = (rho > 0.98) & (rho < 1.06)
    if not cont.any():
        raise RuntimeError("2a continuum mask is empty -- rho range shifted, "
                           "the 0.98/1.06 window no longer brackets the continuum")
    print(f"2a  t_G173/t_model continuum (380-780, excl. dips): "
          f"median={np.median(rho[cont]):.4f}  p10={np.percentile(rho[cont],10):.4f} "
          f"p90={np.percentile(rho[cont],90):.4f}")
    for lo, hi, tag in ((715, 745, "H2O 720-band"), (755, 770, "O2 A-band")):
        m = (wl_model >= lo) & (wl_model <= hi)
        dip = 1.0 - float(np.min(rho[m]))
        # photopic-weighted upper bound of the missing absorption on DNI-lux
        w = dni_m * ybar_m
        frac = float(np.trapezoid(w[m], wl_model[m]) / np.trapezoid(w, wl_model))
        print(f"2a  {tag:<12s} max t-gap={dip*100:5.2f}%  "
              f"photopic-weighted band share={frac*100:.3f}% of DNI-lux")
    aod_model = _hinterland_aod(np.array([380.0, 500.0, 550.0, 700.0]), 0.0734)
    print(f"2a  model AOD(G173 case, Hinterland): 380={aod_model[0]:.4f} "
          f"500={aod_model[1]:.4f} 550={aod_model[2]:.4f} 700={aod_model[3]:.4f}")

    # -- 2b: Bird transcription (pvlib 0.11 constant-level audit) ----------
    # done by source comparison at report time; here quantify the two
    # documented deviations: dni_extra 1361 vs 1364, Kasten66 vs KY89 airmass
    from validation.check2_illuminance import bird_clear_sky
    # aod380/aod500 mirror check2_illuminance's own Bird inputs (its lines
    # "aod380 = 0.05 * (380/550)**-1.3"), NOT the Hinterland spectral shape
    # (alpha ~ 1.01) -- the Bird reference is compared on the same loading
    # convention the check used.
    aod380 = 0.05 * (380.0 / 550.0) ** -1.3
    aod500 = 0.05 * (500.0 / 550.0) ** -1.3
    for h in (30.0, 60.0):
        d0 = bird_clear_sky(90.0 - h, aod380, aod500)
        d1 = bird_clear_sky(90.0 - h, aod380, aod500, dni_extra=1364.0)
        # KY1989 airmass sensitivity
        z = 90.0 - h
        am66 = 1.0 / (math.cos(math.radians(z)) + 0.15 * (93.885 - z) ** -1.253)
        am89 = 1.0 / (math.sin(math.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)
        print(f"2b  h={h:.0f}: DNI(1361)={d0[0]:.2f} DNI(1364)={d1[0]:.2f} "
              f"({100*(d1[0]/d0[0]-1):+.2f}%);  am66={am66:.5f} am89={am89:.5f} "
              f"({100*(am89/am66-1):+.3f}%)")

    # -- 2c: CIE 110 DHI contract ------------------------------------------
    # sky jobs on the audit grid (37x72); the check metrics (73x144) are read
    # from results.json and the grid transfer is quantified
    import json
    res = json.loads((C.VAL / "results.json").read_text())
    m2 = {m["name"]: m for m in
          next(c for c in res["checks"] if c["key"] == "check2")["metrics"]}
    dhi30 = m2["DHI illuminance, h=30deg (CIE 110, T=2.2)"]
    dhi60 = m2["DHI illuminance, h=60deg (CIE 110, T=2.2)"]
    print(f"2c  results.json: model DHI h=30 = {dhi30['model']:.4f} klx, "
          f"CIE = {dhi30['reference']:.4f} klx (dev {dhi30['dev']:+.1f}%)")
    print(f"2c  results.json: model DHI h=60 = {dhi60['model']:.4f} klx, "
          f"CIE = {dhi60['reference']:.4f} klx (dev {dhi60['dev']:+.1f}%)")

    r_of = saemundsson_refraction_arcmin
    jobs2 = [
        dict(name="c2_h30_base", sun_elev=30.0),
        dict(name="c2_h60_base", sun_elev=60.0),
        dict(name="c2_h30_soff", sun_elev=30.0 + r_of(30.0) / 60.0),
        dict(name="c2_h60_soff", sun_elev=60.0 + r_of(60.0) / 60.0),
        dict(name="c2_h30_a020", sun_elev=30.0, aod=0.02),
        dict(name="c2_h30_a035", sun_elev=30.0, aod=0.035),
        dict(name="c2_h30_a080", sun_elev=30.0, aod=0.08),
        dict(name="c2_h30_a020b0", sun_elev=30.0, aod=0.02, albedo=0.0),
        dict(name="c2_h30_b000", sun_elev=30.0, albedo=0.0),
        dict(name="c2_h30_b020", sun_elev=30.0, albedo=0.2),
        dict(name="c2_h60_b000", sun_elev=60.0, albedo=0.0),
        dict(name="c2_h60_b020", sun_elev=60.0, albedo=0.2),
    ]
    sky = run_sky_jobs(jobs2, workers=workers)

    for h, base, metric in ((30.0, "c2_h30_base", dhi30), (60.0, "c2_h60_base", dhi60)):
        b = sky[base]
        yz_cie = C.cie110_zenith_luminance(h, 2.2)
        print(f"2c  h={h:.0f}: Lz_model(37x72)={b['lz']:.1f} vs Yz(CIE,T=2.2)={yz_cie:.1f} "
              f"({100*(b['lz']/yz_cie-1):+.1f}%)  [docstring claims 3%]")
        # two-factor decomposition on the audit grid
        ifac_model = b["dhi"] / b["lz"]
        ifac_cie = _cie_integral_factor(90.0 - h, 37, 72)
        ifac_cie_hi = _cie_integral_factor(90.0 - h, 73, 144)
        print(f"2c  h={h:.0f}: DHI_model(37x72)={b['dhi']/1000:.4f} klx vs "
              f"metric(73x144)={metric['model']:.4f} klx "
              f"({100*(b['dhi']/1000/metric['model']-1):+.2f}% grid)")
        print(f"2c  h={h:.0f}: factors  Lz_model/Lz_CIE="
              f"{b['lz']/yz_cie:.3f}  I_model/I_CIE={ifac_model/ifac_cie:.3f} "
              f"(CIE grid 73x144 vs 37x72: {100*(ifac_cie_hi/ifac_cie-1):+.2f}%)")
        # effective Linke turbidity of the model scene in the CIE Yz metric:
        # invert Yz = (1.376 T - 1.81) tanh + 0.38 kcd/m2 for the model's own
        # zenith luminance
        t_eff = ((b['lz'] / 1000.0 - 0.38) / math.tan(math.radians(h)) + 1.81) / 1.376
        dhi_cie_eff = _cie_integral_factor(90.0 - h, 37, 72) * \
            C.cie110_zenith_luminance(h, t_eff)
        print(f"2c  h={h:.0f}: model scene effective TL (CIE Yz metric) = {t_eff:.3f}; "
              f"DHI dev at scene-matched T: "
              f"{100*(b['dhi']/dhi_cie_eff-1):+.1f}%")

    # AOD scan: what turbidity would the CIE DHI correspond to?
    for h, base in ((30.0, "c2_h30_base"),):
        print("2c  DHI vs AOD scan (h=30, 37x72, albedo 0.1):")
        for nm, aod in (("c2_h30_a020", 0.02), ("c2_h30_a035", 0.035),
                        (base, 0.05), ("c2_h30_a080", 0.08)):
            s = sky[nm]
            dhi_cie = C.cie110_zenith_luminance(
                h, 2.2) * _cie_integral_factor(90.0 - h, 37, 72)
            print(f"    AOD550={aod:.3f}: DHI={s['dhi']/1000:7.3f} klx  "
                  f"vs CIE(T=2.2)={dhi_cie/1000:.3f} klx  "
                  f"ratio={s['dhi']/dhi_cie:.3f}")
        s = sky["c2_h30_a020b0"]
        dhi_cie = C.cie110_zenith_luminance(
            h, 2.2) * _cie_integral_factor(90.0 - h, 37, 72)
        print(f"    AOD550=0.020 + albedo 0.0 (both premises 'pure'): "
              f"DHI={s['dhi']/1000:7.3f} klx  ratio={s['dhi']/dhi_cie:.3f}"
              "  [measured combination point]")

    # model effective Linke turbidity: independent transcription of the
    # Kasten-96 pyrheliometric formula -- constants checked verbatim against
    # the pvlib 0.11 source (its kasten96_lt; refs Ineichen 2008 /
    # Molineaux 1998).  pvlib itself is NOT installed here, so this inline
    # form is the reproducible chain and the report labels it as such.
    # TL = (9.4 + 0.9 am)(d_clean + d_water + d_aod), d_clean/d_water from
    # the Molineaux/Ineichen fits quoted in that source.
    am = 2.0
    d_clean = -0.101 + 0.235 * am ** -0.16
    aod_model550 = _hinterland_aod(np.array([380.0, 500.0, 700.0]), 0.05)
    aod_bb_model = 0.27583 * aod_model550[0] + 0.35 * aod_model550[1]
    tl_no_water = (9.4 + 0.9 * am) * (d_clean + 0.0 + aod_bb_model)
    tl_water142 = (9.4 + 0.9 * am) * (
        d_clean + 0.112 * am ** -0.55 * 1.42 ** 0.34 + aod_bb_model)
    print(f"2c  model scene (clear, AOD550=0.05): AOD380={aod_model550[0]:.4f} "
          f"AOD500={aod_model550[1]:.4f} AOD700={aod_model550[2]:.4f} "
          f"AOD_bb={aod_bb_model:.4f}")
    print(f"2c  effective TL (Kasten-96 inline, source-verified vs pvlib 0.11, am=2): "
          f"pw=0 -> {tl_no_water:.2f}, pw=1.42cm -> {tl_water142:.2f}   "
          f"[CIE side uses T=2.2]")
    # invert: AOD_bb that gives TL=2.2 at pw=1.42
    aod_bb_22 = 2.2 / (9.4 + 0.9 * am) - d_clean - 0.112 * am ** -0.55 * 1.42 ** 0.34
    print(f"2c  AOD_bb implied by T=2.2 (pw=1.42, am=2): {aod_bb_22:.4f}")

    # -- 2d: ground albedo premises ----------------------------------------
    # Bird's reference side carries its own albedo term: DHI is enhanced by
    # 1/(1 - albedo*rs); quantify that enhancement for the audit comparison.
    for h in (30.0, 60.0):
        z = 90.0 - h
        am = 1.0 / (math.cos(math.radians(z)) + 0.15 * (93.885 - z) ** -1.253)
        aod_bb = 0.27583 * aod380 + 0.35 * aod500
        t_aer = math.exp(-(aod_bb ** 0.873) * (1.0 + aod_bb - aod_bb ** 0.7088)
                         * am ** 0.9108)
        taa = 1.0 - 0.1 * (1.0 - am + am ** 1.06) * (1.0 - t_aer)
        rs = 0.0685 + (1.0 - 0.85) * (1.0 - t_aer / taa)
        print(f"2d  Bird reference albedo term (h={h:.0f}): rs={rs:.4f} -> "
              f"DHI enhanced by {100*(1/(1-0.2*rs)-1):+.2f}% (albedo=0.2)")
    for h, alb_base, alb0, alb2 in ((30.0, "c2_h30_base", "c2_h30_b000", "c2_h30_b020"),
                                    (60.0, "c2_h60_base", "c2_h60_b000", "c2_h60_b020")):
        b, z0, z2 = sky[alb_base], sky[alb0], sky[alb2]
        print(f"2d  h={h:.0f}: DHI(albedo) 0.0={z0['dhi']/1000:.4f} "
              f"0.1={b['dhi']/1000:.4f} 0.2={z2['dhi']/1000:.4f} klx; "
              f"0.2 vs 0.1 = {100*(z2['dhi']/b['dhi']-1):+.2f}%, "
              f"0.0 vs 0.1 = {100*(z0['dhi']/b['dhi']-1):+.2f}%")


# ---------------------------------------------------------------------------
# check3 -- CIE 110 relative distribution premises
# ---------------------------------------------------------------------------
def recheck_check3(workers: int) -> None:
    _hdr("check3 -- relative distribution reference contract")

    # -- 3a: formula fidelity ----------------------------------------------
    z, a = C.sky_grid(181, 144)
    sun_zen = 45.0
    rel_closed = C.cie110_relative_luminance(z, a, sun_zen)
    # CIE S 011 / ISO 15469 general-sky form (Zotti 2007 eq. 5.7):
    #   F = (1 + A e^{B/cosZ}) (1 + C (e^{D g} - e^{D pi/2}) + E cos^2 g)
    # type 12 parameters (repo SOURCES.md quote; Darula & Kittler 2002 chain):
    A_, B_, C_, D_, E_ = -1.0, -0.32, 10.0, -3.0, 0.45
    g = C.angular_distance(z, a, sun_zen)
    zs = math.radians(sun_zen)
    grad = 1.0 + A_ * np.exp(B_ / np.cos(np.radians(z)))
    ind = 1.0 + C_ * (np.exp(D_ * np.radians(g)) - np.exp(D_ * math.pi / 2)) \
        + E_ * np.cos(np.radians(g)) ** 2
    rel_s011 = (grad * ind) / (grad[0, 0] * ind[0, 0])
    dev = np.max(np.abs(rel_s011 / rel_closed - 1.0))
    print(f"3a  closed form vs S011 type-12 (Zotti eq 5.7, repo params): "
          f"max |rel diff| over grid = {dev:.2e}")
    # indicatrix constant identity
    const = 1.0 + C_ * (0.0 - np.exp(D_ * math.pi / 2))
    print(f"3a  S011 indicatrix constant 1 - 10*e^(-3*pi/2) = {const:.5f}  "
          f"(CIE 110 closed form uses 0.91)")

    # -- 3c + 3b + 3e: sky jobs --------------------------------------------
    r45 = saemundsson_refraction_arcmin(45.0) / 60.0
    jobs3 = [
        dict(name="c3_h45_base", sun_elev=45.0),
        dict(name="c3_h45_a000", sun_elev=45.0, kind="pristine"),
        dict(name="c3_h45_clr000", sun_elev=45.0, aod=0.0),
        dict(name="c3_h45_a025", sun_elev=45.0, aod=0.025),
        dict(name="c3_h45_a100", sun_elev=45.0, aod=0.10),
        dict(name="c3_h45_a200", sun_elev=45.0, aod=0.20),
        dict(name="c3_h45_soff", sun_elev=45.0 + r45),
        dict(name="c3_h45_b000", sun_elev=45.0, albedo=0.0),
        dict(name="c3_h45_g073", sun_elev=45.0, grid=(73, 72)),
    ]
    sky = run_sky_jobs(jobs3, workers=workers)

    b = sky["c3_h45_base"]
    clr0 = sky["c3_h45_clr000"]
    p0 = sky["c3_h45_a000"]
    print(f"3c  pristine vs clear(aod=0) equivalence (same O3/pressure/grid): "
          f"lz {p0['lz']:.2f} vs {clr0['lz']:.2f} "
          f"({100*(clr0['lz']/p0['lz']-1):+.3f}%), "
          f"hsun/zen {p0['horizon_sun']:.4f} vs {clr0['horizon_sun']:.4f} "
          f"({100*(clr0['horizon_sun']/p0['horizon_sun']-1):+.3f}%)")
    print(f"3c  baseline (AOD .05, 37x72): hsun/zen={b['horizon_sun']:.3f} "
          f"(CIE {b['cie_horizon_sun']:.3f}), anti={b['horizon_anti']:.3f} "
          f"(CIE {b['cie_horizon_anti']:.3f}), rms_all={b['rms_all']:.1f}%, "
          f"rms_hor={b['rms_horizon']:.1f}%")
    import json
    res = json.loads((C.VAL / "results.json").read_text())
    m3 = {m["name"]: m for m in
          next(c for c in res["checks"] if c["key"] == "check3")["metrics"]}
    print(f"3c  results.json hsun/zen={m3['horizon sun-side / zenith, h=45deg']['model']:.3f} "
          f"-> baseline match: {abs(b['horizon_sun']/b['cie_horizon_sun']-1-m3['horizon sun-side / zenith, h=45deg']['dev']/100):.2e}")
    print("3c  AOD scan (h=45, 37x72): hsun/zen and rms_all response")
    for nm, aod in (("c3_h45_a000", 0.0), ("c3_h45_a025", 0.025),
                    ("c3_h45_base", 0.05), ("c3_h45_a100", 0.10),
                    ("c3_h45_a200", 0.20)):
        s = sky[nm]
        print(f"    AOD550={aod:.3f}: hsun/zen={s['horizon_sun']:7.3f} "
              f"({100*(s['horizon_sun']/s['cie_horizon_sun']-1):+6.1f}% vs CIE)  "
              f"anti={s['horizon_anti']:6.3f} ({100*(s['horizon_anti']/s['cie_horizon_anti']-1):+6.1f}%)  "
              f"rms_all={s['rms_all']:5.1f}%")
    g = sky["c3_h45_g073"]
    print(f"3c  grid 37->73 (zenith): hsun/zen {b['horizon_sun']:.3f} -> "
          f"{g['horizon_sun']:.3f} ({100*(g['horizon_sun']/b['horizon_sun']-1):+.2f}%), "
          f"rms_all {b['rms_all']:.1f}% -> {g['rms_all']:.1f}%")
    so = sky["c3_h45_soff"]
    print(f"3b  sun-axis: elevation +{r45*60:.2f}' (Saemundsson at 45deg): "
          f"hsun/zen {b['horizon_sun']:.4f} -> {so['horizon_sun']:.4f} "
          f"({100*(so['horizon_sun']/b['horizon_sun']-1):+.2f}%), "
          f"rms_all {b['rms_all']:.2f}% -> {so['rms_all']:.2f}%")
    z0 = sky["c3_h45_b000"]
    print(f"3e  albedo 0.0 vs 0.1 (h=45): hsun/zen {z0['horizon_sun']:.4f} vs "
          f"{b['horizon_sun']:.4f} ({100*(z0['horizon_sun']/b['horizon_sun']-1):+.2f}%), "
          f"anti {z0['horizon_anti']:.4f} vs {b['horizon_anti']:.4f} "
          f"({100*(z0['horizon_anti']/b['horizon_anti']-1):+.2f}%), "
          f"rms_all {z0['rms_all']:.2f}% vs {b['rms_all']:.2f}%")


# ---------------------------------------------------------------------------
# check6 -- Koomen (1952) observation contract
# ---------------------------------------------------------------------------
#: Maryland zenith column (Z=0, P=90) of Koomen 1952 Table II, p354, read
#: from the archived original scan (sha256-verified against
#: validation/data/twilight_reference_audit/source_provenance.json).  The
#: scan's deep rows use the repeated-zero subscript notation
#: (0.0_2 15 = 0.0015).  +5/+3 deg are context rows from the scan (the CSV
#: starts at 0); the check's reference depths are the non-positive ones.
KOOMEN_SCAN_MARYLAND_ZENITH = {
    5.0: 43.0, 3.0: 35.0, 0.0: 15.0, -3.0: 2.0, -6.0: 0.06,
    -9.0: 0.0015, -12.0: 0.00012, -15.0: 0.00004,
}
#: Depths the check's reference CSV must cover (its scored/fitted range).
KOOMEN_CSV_REQUIRED_DEPTHS = (0.0, -3.0, -6.0, -9.0, -12.0, -15.0)


def recheck_check6(workers: int) -> None:
    _hdr("check6 -- Koomen observation contract")
    from skycolor.rt.single import direct_sun_transmittance
    from skycolor.rt.solver import sky_radiance_total

    # -- 4a: transcription table -------------------------------------------
    with (C.DATA / "twilight_koomen1952.csv").open() as fh:
        rows = {float(r["sun_elevation_deg"]): r for r in csv.DictReader(fh)}
    # coverage gate: every scanned depth must exist in the CSV -- a missing
    # row must fail loudly, never silently shrink the "full-depth" claim
    # (learnings: grid-domain checks must assert endpoint coverage).
    missing = sorted(set(KOOMEN_CSV_REQUIRED_DEPTHS) - set(rows))
    if missing:
        raise RuntimeError(f"4a: twilight_koomen1952.csv missing depth(s) "
                           f"{missing} -- transcription coverage broken")
    print("4a  CSV vs original scan (Maryland zenith, cd/ft^2):")
    ok = True
    n_checked = 0
    for h, scan in sorted(KOOMEN_SCAN_MARYLAND_ZENITH.items(), reverse=True):
        if h not in rows:
            continue
        csvv = float(rows[h]["I_cd_ft2"])
        match = abs(csvv - scan) <= 1e-12 + 1e-9 * abs(scan)
        ok &= match
        n_checked += 1
        print(f"    h={h:+5.1f}: scan={scan:<10g} csv={csvv:<10g} "
              f"{'OK' if match else 'MISMATCH'}")
    cd_ft2 = 1.0 / 0.09290304
    print(f"4a  unit chain: 1 cd/ft2 = {cd_ft2:.5f} cd/m2 "
          f"(=1/0.09290304); scan -6 deg 0.06 -> {0.06*cd_ft2:.6f} "
          f"(CSV 0.645835)")
    for h, csvv in ((-6.0, 0.06), (-9.0, 0.0015)):
        mag = -2.5 * math.log10(csvv) + 9.9846
        print(f"4a  Nawar mag conversion h={h:+.0f}: m={mag:.4f} "
              f"(CSV V_mag_arcsec2_source={rows[h]['V_mag_arcsec2_source']})")
    print(f"4a  full-depth transcription vs original scan: "
          f"{'ALL MATCH' if ok else 'DIFFERENCES FOUND'} "
          f"(checked {n_checked}/{len(KOOMEN_SCAN_MARYLAND_ZENITH)} depths)")

    # -- 4c: site vertical transmission ------------------------------------
    wl = C.WL
    et = np.interp(wl, *_g173_columns()[:2])
    v = C.photopic(wl)
    for kind, aod in (("clear", 0.05), ("pristine", 0.0)):
        atmo = _atmo_for(kind, aod, 300.0, wl, None)
        t = direct_sun_transmittance(atmo, 0.0)
        t_phot = float(np.trapezoid(t * et * v, wl) / np.trapezoid(et * v, wl))
        t_bb = float(np.trapezoid(t * et, wl) / np.trapezoid(et, wl))
        print(f"4c  {kind:<8s} vertical transmission (photometric V-weighted) "
              f"= {t_phot*100:.1f}%   (flat-weighted 380-780 = {t_bb*100:.1f}%)")
    print("4c  Koomen p353 original: Maryland 75-85%, Sacramento Peak 85-90%, "
          "pure Rayleigh 90%")

    # -- 4f: albedo probe at the deterministic -6 deg point -----------------
    base = _atmo_for("clear", 0.05, 300.0, C.WL_COARSE, None)
    alb0 = _atmo_for("clear", 0.05, 300.0, C.WL_COARSE, 0.0)
    lums = {}
    for tag, atmo in (("alb0.1", base), ("alb0.0", alb0)):
        rad = sky_radiance_total(atmo, 96.0, np.array([0.0]), np.array([0.0]))
        lums[tag] = float(C.luminance(rad[0][None, :], C.WL_COARSE)[0])
    print(f"4f  -6 deg deterministic zenith luminance: albedo 0.1 = "
          f"{lums['alb0.1']:.4f}, 0.0 = {lums['alb0.0']:.4f} cd/m2 "
          f"({100*(lums['alb0.0']/lums['alb0.1']-1):+.2f}%)")

    # -- 4e: branch distinction from results.json ---------------------------
    import json
    res = json.loads((C.VAL / "results.json").read_text())
    m6 = {m["name"]: m for m in
          next(c for c in res["checks"] if c["key"] == "check6")["metrics"]}
    tol_15 = math.log10(1.5)
    for name, branch in (("zenith luminance, h=-6deg", "deterministic"),
                         ("zenith luminance, h=-9deg", "engine"),
                         ("zenith luminance, h=-12deg", "engine")):
        m = m6[name]
        extra = m["tolerance"] - tol_15
        h = name.split("h=")[-1]
        print(f"4e  h={h}: ratio={10**m['dev']:.3f} tol(dex)={m['tolerance']:.4f} "
              f"[{tol_15:.3f} base + {extra:.3f} engine 1-sigma]  branch={branch}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=SECTIONS, default=None,
                    help="run a single section (default: all)")
    ap.add_argument("--workers", type=int, default=6,
                    help="process-pool workers for the sky solves")
    args = ap.parse_args()
    which = SECTIONS if args.only is None else (args.only,)
    for name in which:
        {"check2": recheck_check2, "check3": recheck_check3,
         "check6": recheck_check6}[name](args.workers)
    print("\nall requested sections completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
