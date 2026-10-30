"""Benchmark and validation numbers for the spherical twilight engine.

This script produces every number quoted in ``docs/TWILIGHT.md``:

1. the ratio to the deterministic model where both are valid (-3 to -6 deg),
2. the comparison with Patat, Ugolnikov & Postylyakov (2006) at -3, -6, -9
   and -12 degrees,
3. the statistical-error scaling with the photon count,
4. runtime for a full 30 x 60 direction, 40 wavelength twilight sky,
5. the bias of the engine against a high-photon direct Monte Carlo reference.

Run from the worktree root::

    uv run python tools/twilight_benchmark.py            # ~10 min
    uv run python tools/twilight_benchmark.py --quick    # ~2 min

The ``--quick`` mode reduces the photon budgets; the numbers are then noisier.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets, spectra  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402
from skycolor.rt import mc as MC  # noqa: E402
from skycolor.rt.single import sky_radiance as single_radiance  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import multiple_scattered, twilight_radiance  # noqa: E402

K_M = 683.0
WL = WL_COARSE


# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------
def patat_luminance(sun_elevation_deg):
    """Patat et al. (2006) Table 1 fit, V-band zenith luminance [cd m^-2]."""
    zeta = 90.0 - sun_elevation_deg
    m_v = 11.84 + 1.518 * (zeta - 95.0) - 0.057 * (zeta - 95.0) ** 2
    return 1.08e5 * 10.0 ** (-0.4 * m_v)


PATAT = {h: patat_luminance(h) for h in (-3.0, -6.0, -9.0, -12.0, -18.0)}


def luminance(radiance, wl=WL):
    v = spectra.cie_xyz(np.asarray(wl, float))[:, 1]
    return K_M * np.trapezoid(np.asarray(radiance) * v, np.asarray(wl, float), axis=-1)


# ---------------------------------------------------------------------------
# 1. agreement with the deterministic model
# ---------------------------------------------------------------------------
def check_deterministic(quality):
    print("\n== 1. twilight engine vs deterministic model ==")
    atmo = presets.clear(WL)
    zen = np.array([0.0, 30.0, 60.0, 80.0])
    az = np.zeros_like(zen)
    print(f"{'h':>6} {'vza':>5} {'deterministic':>14} {'engine':>14} "
          f"{'ratio':>7} {'sigma':>7}")
    rows = []
    for h in (-3.0, -4.5, -6.0):
        det = sky_radiance_total(atmo, 90.0 - h, zen, az)
        tw, err = twilight_radiance(atmo, 90.0 - h, zen, az, quality=quality)
        for i in range(zen.size):
            num = float(luminance(tw[i]))
            den = float(luminance(det[i]))
            sig = float(luminance(err[i]))
            rows.append((h, zen[i], den, num, num / den, sig / num))
            print(f"{h:6.1f} {zen[i]:5.1f} {den:14.4e} {num:14.4e} "
                  f"{num / den:7.3f} {sig / num * 100:6.1f}%")
    return rows


# ---------------------------------------------------------------------------
# 2. Patat et al. (2006)
# ---------------------------------------------------------------------------
def check_patat(quality):
    print("\n== 2. zenith V-band luminance vs Patat et al. (2006) ==")
    atmo = presets.clear(WL)
    atmo_p = presets.pristine(WL)
    print(f"{'h':>6} {'Patat':>10} {'single':>10} {'det(1+ms)':>10} "
          f"{'engine':>10} {'+/-(stat)':>9} {'eng/Patat':>9} {'MC/Patat':>8}")
    rows = []
    for h in (-3.0, -6.0, -9.0, -12.0):
        one = single_radiance(atmo, 90.0 - h, [0.0], [0.0])
        detr = sky_radiance_total(atmo, 90.0 - h, [0.0], [0.0])
        tw, err = twilight_radiance(atmo, 90.0 - h, [0.0], [0.0], quality=quality)
        ref = PATAT[h]
        l_one = float(luminance(one[0]))
        l_det = float(luminance(detr[0]))
        l_tw = float(luminance(tw[0]))
        l_err = float(luminance(err[0]))
        # the parent's reference run: pristine atmosphere, 21 wavelengths
        rows.append(dict(h=h, ref=ref, single=l_one, det=l_det, engine=l_tw,
                         err=l_err, ratio=l_tw / ref))
        print(f"{h:6.1f} {ref:10.3e} {l_one:10.3e} {l_det:10.3e} "
              f"{l_tw:10.3e} {l_err:9.2e} {l_tw / ref:9.3f} "
              f"{(l_one + 0.0) / ref:8.3f}")
    return rows


# ---------------------------------------------------------------------------
# 3. error scaling with the photon count
# ---------------------------------------------------------------------------
def check_error_scaling():
    print("\n== 3. statistical error vs photon count (zenith, 550 nm) ==")
    atmo = presets.clear([550.0])
    print(f"{'h':>6} {'photons':>10} {'MS':>12} {'stderr':>10} {'rel':>7} "
          f"{'time':>7}")
    out = []
    for h in (-6.0, -9.0):
        prev = None
        for n in (1e6, 4e6, 1.6e7, 6.4e7):
            t = time.time()
            ms, err, _ = multiple_scattered(
                atmo, 90.0 - h, [0.0], [0.0], quality="medium",
                grid=dict(n_anchor=1, n_photons=n), seed=7)
            dt = time.time() - t
            rel = float(err[0, 0] / ms[0, 0])
            out.append((h, n, float(ms[0, 0]), float(err[0, 0]), rel, dt))
            print(f"{h:6.1f} {n:10.0e} {ms[0, 0]:12.4e} {err[0, 0]:10.3e} "
                  f"{rel * 100:6.1f}% {dt:6.1f}s")
            prev = rel
    # the ratio of successive relative errors should be ~sqrt(4) = 2
    return out


# ---------------------------------------------------------------------------
# 4. runtime of a full sky
# ---------------------------------------------------------------------------
def check_runtime(quality):
    print("\n== 4. runtime, full twilight sky ==")
    wl = np.arange(380.0, 780.0 + 1e-9, 10.0)          # 40 wavelengths
    atmo = presets.clear(wl)
    zz, aa = np.meshgrid(np.linspace(0.0, 88.0, 30),
                         np.linspace(0.0, 360.0, 60, endpoint=False), indexing="ij")
    zen, az = zz.ravel(), aa.ravel()
    print(f"grid: {zen.size} directions x {wl.size} wavelengths")
    for q in (["low", "medium"] if quality == "medium" else [quality]):
        t = time.time()
        rad, err = twilight_radiance(atmo, 99.0, zen, az, quality=q, seed=1)
        dt = time.time() - t
        lum = luminance(rad, wl)
        elum = luminance(err, wl)
        bright = lum > 0.01 * lum.max()
        print(f"  quality={q:7s} {dt:7.1f}s   median rel. error (bright pixels) "
              f"{np.median(elum[bright] / lum[bright]) * 100:5.1f}%   "
              f"90th pct {np.percentile(elum[bright] / lum[bright], 90) * 100:5.1f}%")
    return dt


# ---------------------------------------------------------------------------
# 5. bias against a direct Monte Carlo reference
# ---------------------------------------------------------------------------
def check_bias(quality):
    print("\n== 5. engine vs high-photon direct Monte Carlo ==")
    atmo = presets.clear([550.0])
    zen = np.array([0.0, 45.0, 85.0, 85.0])
    az = np.array([0.0, 90.0, 0.0, 180.0])
    print(f"{'h':>6} {'vza':>5} {'az':>5} {'direct MC':>12} {'rel':>6} "
          f"{'engine':>12} {'rel':>6} {'ratio':>6}")
    rows = []
    n_ref = 1.2e7
    for h in (-6.0, -9.0):
        t = time.time()
        ref = MC.mc_radiance_split(atmo, 90.0 - h, zen, az, n_photons=int(n_ref),
                                   seed=42)
        print(f"   (reference: {n_ref:.1e} photons, {time.time() - t:.0f}s)")
        ms, err, _ = multiple_scattered(
            atmo, 90.0 - h, zen, az, quality="medium",
            grid=dict(n_anchor=1, n_photons=1.2e7), seed=11)
        for i in range(zen.size):
            r = ref["multiple"][i, 0]
            re = ref["multiple_err"][i, 0]
            e = ms[i, 0]
            ee = err[i, 0]
            rows.append((h, zen[i], az[i], r, re / r, e, ee / e, e / r))
            print(f"{h:6.1f} {zen[i]:5.1f} {az[i]:5.0f} {r:12.4e} "
                  f"{re / r * 100:5.1f}% {e:12.4e} {ee / e * 100:5.1f}% "
                  f"{e / r:6.3f}")
    return rows


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="", help="comma separated list of checks")
    args = ap.parse_args(argv)
    quality = "low" if args.quick else "medium"
    skip = set(x for x in args.skip.split(",") if x)
    t0 = time.time()
    if "1" not in skip:
        check_deterministic(quality)
    if "2" not in skip:
        check_patat(quality)
    if "3" not in skip:
        check_error_scaling()
    if "4" not in skip:
        check_runtime(quality)
    if "5" not in skip:
        check_bias(quality)
    print(f"\ntotal wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
