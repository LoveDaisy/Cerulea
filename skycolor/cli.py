"""Command line interface: ``python -m skycolor``."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np

from . import color, presets, render
from .constants import WL_COARSE, WL_DEFAULT
from .geometry import sun_position


def _build_atmo(args):
    wl = np.arange(args.wl_min, args.wl_max + 1e-9, args.wl_step)
    p = presets.PRESETS[args.preset]
    try:
        atmo = p(wl, aod=args.aod)
    except TypeError:
        atmo = p(wl)
    if args.elevation:
        atmo = atmo.at_altitude(args.elevation)
    return atmo, wl


def _sun_from_time(args):
    when = dt.datetime.fromisoformat(args.time)
    az, el = sun_position(args.latitude, args.longitude, when, refraction=True)
    return az, el


def cmd_render(args):
    atmo, wl = _build_atmo(args)
    if args.time:
        sun_az, sun_el = _sun_from_time(args)
    else:
        sun_el = args.sun_elevation
        sun_az = args.sun_azimuth
    size = tuple(int(v) for v in args.size.split("x"))
    rgb = render.render_scene(
        atmo, sun_el, sun_az, camera=args.camera, size=size,
        exposure=args.exposure, tonemap=args.tonemap, adaptation=args.adaptation,
        reference_percentile=args.percentile, sun_disk=not args.no_sun,
        ground=not args.no_ground, hfov_deg=args.hfov, yaw_deg=args.yaw,
        pitch_deg=args.pitch, n_zenith=args.n_zenith, n_azimuth=args.n_azimuth,
    )
    render.save_png(rgb, args.out)
    print(f"wrote {args.out}  (sun elevation {sun_el:.1f} deg, azimuth {sun_az:.1f} deg, "
          f"{args.preset}, AOD {args.aod})")


def cmd_chart(args):
    """Print the chromaticity and luminance of a few sky directions."""
    atmo, wl = _build_atmo(args)
    dome = render.sky_dome(atmo, 90.0 - args.sun_elevation,
                           n_zenith=args.n_zenith, n_azimuth=args.n_azimuth)
    rows = [("zenith", 0, 0), ("45deg sun-side", 45, 0), ("45deg anti-solar", 45, 180),
            ("horizon sun-side", 88, 0), ("horizon anti-solar", 88, 180)]
    print(f"{'direction':22s} {'x':>6s} {'y':>6s} {'Lv [cd/m2]':>12s}")
    for name, zen, az in rows:
        i = np.argmin(abs(dome["zenith"] - zen))
        j = np.argmin(abs(dome["azimuth_rel"] - az))
        xyz = color.spectrum_to_xyz(dome["radiance"][i, j], wl)
        xy = color.chromaticity(xyz)
        print(f"{name:22s} {xy[0]:6.3f} {xy[1]:6.3f} {683 * xyz[1]:12.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skycolor", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--preset", default="clear", choices=sorted(presets.PRESETS))
        p.add_argument("--aod", type=float, default=0.1, help="aerosol optical depth at 550 nm")
        p.add_argument("--elevation", type=float, default=0.0, help="observer altitude [km]")
        p.add_argument("--sun-elevation", type=float, default=10.0)
        p.add_argument("--sun-azimuth", type=float, default=180.0)
        p.add_argument("--time", default=None, help="ISO UTC time; overrides sun position")
        p.add_argument("--latitude", type=float, default=39.9)
        p.add_argument("--longitude", type=float, default=116.4)
        p.add_argument("--wl-min", type=float, default=380.0)
        p.add_argument("--wl-max", type=float, default=780.0)
        p.add_argument("--wl-step", type=float, default=5.0)
        p.add_argument("--n-zenith", type=int, default=50)
        p.add_argument("--n-azimuth", type=int, default=91)

    r = sub.add_parser("render", help="render a sky image")
    common(r)
    r.add_argument("--camera", default="panorama", choices=["panorama", "fisheye", "perspective"])
    r.add_argument("--size", default="1000x500")
    r.add_argument("--hfov", type=float, default=90.0)
    r.add_argument("--yaw", type=float, default=0.0)
    r.add_argument("--pitch", type=float, default=0.0)
    r.add_argument("--exposure", type=float, default=None)
    r.add_argument("--tonemap", default="aces", choices=sorted(color.TONEMAPS))
    r.add_argument("--adaptation", default=None, choices=[None, "bradford", "von_kries"])
    r.add_argument("--percentile", type=float, default=92.0)
    r.add_argument("--no-sun", action="store_true")
    r.add_argument("--no-ground", action="store_true")
    r.add_argument("--out", default="sky.png")
    r.set_defaults(func=cmd_render)

    c = sub.add_parser("chart", help="print colour coordinates of a few directions")
    common(c)
    c.set_defaults(func=cmd_chart)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
