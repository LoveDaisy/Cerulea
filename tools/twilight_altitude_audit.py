"""Reproduce the altitude-response audit, writing one JSONL record per run.

Run from the repository root with ``uv run python -m tools.twilight_altitude_audit``.
Direct MC bypasses the twilight engine's angular smoother and spectral interpolation.
Every seed is an independent replicate; matching seeds across altitudes do not
guarantee correlated paths because collision/termination histories diverge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from skycolor import presets, spectra
from skycolor.rt.single import sky_radiance
from skycolor.rt.solver import sky_radiance_total
from skycolor.rt.mc import mc_radiance_split
from skycolor.rt.twilight import QUALITY, twilight_radiance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["deterministic", "direct", "engine"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elevations", type=float, nargs="+", default=[0.0, 2.4])
    parser.add_argument("--sun", type=float, default=-6.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--photons", type=int, default=1_000_000,
                        help="Direct MC paths per wavelength; engine uses --quality")
    parser.add_argument("--wavelengths", type=float, nargs="+")
    parser.add_argument("--albedo", type=float, default=0.1)
    parser.add_argument("--aod", type=float, default=0.05)
    parser.add_argument("--max-scatter", type=int, default=50)
    parser.add_argument("--geometric", action="store_true")
    parser.add_argument("--quality", default="low")
    args = parser.parse_args()
    if args.mode == "engine" and args.max_scatter != 50:
        parser.error("engine fixes max_scatter=50; use direct mode for an order-cap audit")
    wl = np.array(args.wavelengths if args.wavelengths else
                  (np.arange(380.0, 781.0, 5.0) if args.mode != "direct" else
                   [450.0, 500.0, 550.0, 600.0, 650.0]))
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "skycolor"], cwd=root,
                                   text=True)
    import hashlib
    meta = dict(head=head, root=str(root), source_diff_sha256=hashlib.sha256(
        diff.encode()).hexdigest(), mode=args.mode, wavelengths=wl.tolist(),
        sun_elevation=args.sun, albedo=args.albedo, aod=args.aod,
        refraction=not args.geometric,
        photons_per_wavelength=args.photons if args.mode == "direct" else None,
        engine_config=dict(QUALITY[args.quality]) if args.mode == "engine" else None,
        max_scatter=args.max_scatter, quality=args.quality)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base = presets.clear(wl)
    if args.aod != base.aerosols[0].aod550:
        from dataclasses import replace
        base.aerosols = [replace(base.aerosols[0], aod550=args.aod)]
    base.ground_albedo = args.albedo
    base.refraction = not args.geometric
    for elevation in args.elevations:
        atmo = base if elevation == 0 else base.at_altitude(elevation)
        single = sky_radiance(atmo, 90.0 - args.sun, [0.0], [0.0])[0]
        atmo.refraction = False
        single_geometric = sky_radiance(atmo, 90.0 - args.sun, [0.0], [0.0])[0]
        atmo.refraction = not args.geometric
        seeds = [None] if args.mode == "deterministic" else args.seeds
        for seed in seeds:
            started = time.perf_counter()
            rec = dict(meta, elevation=elevation, seed=seed,
                       deterministic_single=single.tolist(),
                       deterministic_single_geometric=single_geometric.tolist())
            if args.mode == "deterministic":
                total = sky_radiance_total(atmo, 90.0 - args.sun, [0.0], [0.0])[0]
                fields = dict(single=single, multiple=total - single, total=total)
            elif args.mode == "direct":
                result = mc_radiance_split(
                    atmo, 90.0 - args.sun, [0.0], [0.0], seed=seed,
                    n_photons=args.photons, max_scatter=args.max_scatter)
                fields = {key: value[0] for key, value in result.items()}
                fields["hybrid_total"] = single + fields["multiple"]
                fields["geometric_split_total"] = single_geometric + fields["multiple"]
            else:
                total, error, info = twilight_radiance(
                    atmo, 90.0 - args.sun, [0.0], [0.0], seed=seed,
                    quality=args.quality, return_info=True)
                fields = dict(total=total[0], total_err=error[0], single=single,
                              multiple=total[0] - single)
                rec["coarse"] = {key: np.asarray(info[key]).tolist() for key in
                    ["anchor_wl", "coarse_zenith", "coarse_azimuth",
                     "coarse_values", "coarse_errors"]}
            rec["fields"] = {key: value.tolist() for key, value in fields.items()}
            if wl.size > 1:
                # This is the integral over the recorded wavelength range only.
                v = spectra.cie_xyz(wl)[:, 1]
                rec["photopic_integrals"] = {key: float(683.0 * np.trapezoid(
                    value * v, wl)) for key, value in fields.items()
                    if not key.endswith("_err")}
            rec["seconds"] = time.perf_counter() - started
            with args.out.open("a") as stream:
                stream.write(json.dumps(rec, allow_nan=False) + "\n")
            print(json.dumps({key: rec[key] for key in
                ["mode", "elevation", "seed", "seconds"]} |
                dict(photopic=rec.get("photopic_integrals"))), flush=True)


if __name__ == "__main__":
    main()
