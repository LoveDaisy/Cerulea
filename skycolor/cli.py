"""Command line interface: ``python -m skycolor``.

Both subcommands share one preparation and one entry: the solar input is
resolved by :func:`skycolor.scene.resolve_sun` (time or numbers, always the
pre-refraction direction), one :func:`skycolor.render.solve_dome` request
runs the authoritative solver entry with the requested approximation/cloud
combination, and the output formats (PNG image, chart table) are pure
consumers of the prepared result.  The result's provenance and statistical
limits are printed to stderr with every product.  Request errors exit
non-zero with the solver's own message instead of falling back to a default
path.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import color, presets, render
from .scene import resolve_sun


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


def _prepare(args):
    """The shared scene/sun/config preparation of both subcommands.

    Numeric MC settings stay ``None`` unless explicitly given (the parser
    has no phantom defaults), so the solver's own applicability checks see
    exactly what the caller asked for.
    """
    from .rt.solver import MCConfig

    atmo, wl = _build_atmo(args)
    sun_az, sun_el = resolve_sun(args.sun_elevation, args.sun_azimuth,
                                 time=args.time, latitude=args.latitude,
                                 longitude=args.longitude)
    grid = {}
    if args.mc_photons is not None:
        grid["n_photons"] = args.mc_photons
    if args.mc_n_zen is not None:
        grid["n_zen"] = args.mc_n_zen
    if args.mc_n_az is not None:
        grid["n_az"] = args.mc_n_az
    if args.mc_n_anchor is not None:
        grid["n_anchor"] = args.mc_n_anchor
    mc_config = MCConfig(quality=args.quality, seed=args.seed,
                         grid=grid or None)
    cloud_layer = None
    if args.cloud:
        from .cloud import get_preset

        cloud_layer = get_preset(args.cloud)
    return atmo, wl, sun_az, sun_el, mc_config, cloud_layer


def cmd_render(args):
    atmo, wl, sun_az, sun_el, mc_config, cloud_layer = _prepare(args)
    pres = render.solve_dome(atmo, sun_el, sun_az,
                             n_zenith=args.n_zenith, n_azimuth=args.n_azimuth,
                             approximation=args.approximation,
                             cloud_layer=cloud_layer, mc_config=mc_config)
    size = tuple(int(v) for v in args.size.split("x"))
    rgb = render.render_result(
        pres, camera=args.camera, size=size, exposure=args.exposure,
        tonemap=args.tonemap, adaptation=args.adaptation,
        reference_percentile=args.percentile,
        sun_disk=False if args.no_sun else None,
        ground=not args.no_ground, hfov_deg=args.hfov, yaw_deg=args.yaw,
        pitch_deg=args.pitch)
    render.save_png(rgb, args.out)
    print(f"wrote {args.out}  (sun elevation {sun_el:.1f} deg, azimuth {sun_az:.1f} deg, "
          f"{args.preset}, AOD {args.aod})")
    print(render.format_result_summary(pres), file=sys.stderr)


def cmd_chart(args):
    """Print the chromaticity and luminance of a few sky directions."""
    atmo, wl, sun_az, sun_el, mc_config, cloud_layer = _prepare(args)
    pres = render.solve_dome(atmo, sun_el, sun_az,
                             n_zenith=args.n_zenith, n_azimuth=args.n_azimuth,
                             approximation=args.approximation,
                             cloud_layer=cloud_layer, mc_config=mc_config)
    rows = render.chart_table(pres)
    print(f"{'direction':22s} {'zen':>6s} {'az':>6s} {'x':>6s} {'y':>6s} {'Lv [cd/m2]':>12s}")
    for name, zen, az_rel, x, y, lv in rows:
        print(f"{name:22s} {zen:6.1f} {az_rel:6.1f} {x:6.3f} {y:6.3f} {lv:12.2f}")
    print(render.format_result_summary(pres), file=sys.stderr)


def main(argv=None):
    from .cloud import PRESETS as CLOUD_PRESETS
    from .rt.solver import (HYBRID_SOS, PLANE_SOS, SPHERICAL_MC,
                            SPHERICAL_SINGLE)

    ap = argparse.ArgumentParser(prog="skycolor", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--preset", default="clear", choices=sorted(presets.PRESETS))
        p.add_argument("--aod", type=float, default=0.1, help="aerosol optical depth at 550 nm")
        p.add_argument("--elevation", type=float, default=0.0, help="observer altitude [km]")
        p.add_argument("--sun-elevation", type=float, default=None,
                       help="sun elevation [deg] (default 10 without --time)")
        p.add_argument("--sun-azimuth", type=float, default=None,
                       help="sun azimuth [deg, from north] (default 180 without --time)")
        p.add_argument("--time", default=None,
                       help="ISO time (naive = UTC, offsets honoured); "
                            "overrides numeric sun position")
        p.add_argument("--latitude", type=float, default=39.9)
        p.add_argument("--longitude", type=float, default=116.4)
        p.add_argument("--wl-min", type=float, default=380.0)
        p.add_argument("--wl-max", type=float, default=780.0)
        p.add_argument("--wl-step", type=float, default=5.0)
        p.add_argument("--n-zenith", type=int, default=50)
        p.add_argument("--n-azimuth", type=int, default=91)
        p.add_argument("--approximation", default=None,
                       choices=[SPHERICAL_SINGLE, PLANE_SOS, HYBRID_SOS,
                                SPHERICAL_MC],
                       help="radiative-transfer approximation (default "
                            "hybrid-sos)")
        p.add_argument("--cloud", default=None, choices=sorted(CLOUD_PRESETS),
                       help="add a uniform cloud deck preset")
        p.add_argument("--quality", default=None,
                       choices=["low", "medium", "high", "ultra"],
                       help="Monte Carlo quality preset (spherical-mc only)")
        p.add_argument("--seed", type=int, default=None,
                       help="Monte Carlo seed (spherical-mc only)")
        p.add_argument("--mc-photons", type=float, default=None,
                       help="explicit MC photon budget override")
        p.add_argument("--mc-n-zen", type=int, default=None,
                       help="explicit MC coarse-grid zenith override")
        p.add_argument("--mc-n-az", type=int, default=None,
                       help="explicit MC coarse-grid azimuth override")
        p.add_argument("--mc-n-anchor", type=int, default=None,
                       help="explicit MC anchor-wavelength override")

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
    try:
        code = args.func(args)
    except ValueError as exc:
        # a rejected request is a normal outcome, not a traceback: exit
        # non-zero with the solver's own explanation
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if code is None else code


if __name__ == "__main__":
    sys.exit(main())
