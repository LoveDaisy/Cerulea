"""Run paired geometric/finite-refracted source experiments on one medium.

Tables are explicitly labelled frozen experiment artifacts. Reuse is requested
by --table-dir, with the medium, wavelength grid and build parameters included
in the label. Use a fresh --out JSONL for each run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from skycolor import presets, spectra
from skycolor.rt import mc
from skycolor.rt.single import sky_radiance
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source, single_zenith


def table_for(atmo, folder, refracted, height_count, derivative_tolerance):
    definition = dict(wl=atmo.wl.tolist(), height_count=height_count,
        ground=float(atmo.z_bounds[0]), refracted=refracted,
        derivative_tolerance=derivative_tolerance,
        standard_scene="clear-aod0.05-o3_300du-surface_P0")
    label = json.dumps(definition, sort_keys=True)
    name = hashlib.sha256(label.encode()).hexdigest()[:16]
    path = folder / f"point-sun-{name}.npz"
    if path.exists():
        result = PointSolarTable.load(path, atmo, expected_label=label)
    else:
        result = PointSolarTable(atmo, n_height=height_count, refracted=refracted,
            derivative_tolerance=derivative_tolerance, label=label)
        result.save(path)
    return result, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--tables-manifest", type=Path,
                        help="Explicit saved table entries override build controls")
    parser.add_argument("--modes", nargs="+", choices=["legacy", "geometric", "unified"],
                        default=["geometric", "unified"])
    parser.add_argument("--elevations", nargs="+", type=float, default=[0., 2.4])
    parser.add_argument("--wavelengths", nargs="+", type=float, default=[450., 550., 650.])
    parser.add_argument("--seeds", nargs="+", type=int, default=[200, 201, 202, 203])
    parser.add_argument("--photons", type=int, default=10000)
    parser.add_argument("--sun", type=float, default=-6)
    parser.add_argument("--n-height", type=int, default=33)
    parser.add_argument("--derivative-tolerance", type=float, default=.005)
    parser.add_argument("--disk-radial", type=int, default=2)
    parser.add_argument("--disk-azimuth", type=int, default=8)
    parser.add_argument("--disk-radius", type=float, default=16/60)
    parser.add_argument("--ss-gauss", type=int, default=8)
    parser.add_argument("--ss-segment", type=float, default=.5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    library_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "skycolor"], cwd=root)
    tools = ["point_solar_table.py", "continuous_solar_columns.py", "radial_solar_mapping.py",
             "unified_solar_transport.py", "run_unified_solar_audit.py"]
    source_hashes = {name: hashlib.sha256((root/"tools"/name).read_bytes()).hexdigest()
                     for name in tools}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    wl = np.array(args.wavelengths)
    base = presets.clear(wl)
    v = spectra.cie_xyz(wl)[:, 1]
    source_options = dict(n_radial=args.disk_radial, n_azimuth=args.disk_azimuth,
                          radius_deg=args.disk_radius)
    config = vars(args).copy()
    config["out"], config["table_dir"] = str(args.out), str(args.table_dir)
    config["tables_manifest"] = str(args.tables_manifest) if args.tables_manifest else None
    manifest = json.loads(args.tables_manifest.read_text()) if args.tables_manifest else {}
    for height in args.elevations:
        atmo = base if height == 0 else base.at_altitude(height)
        prepared = {}
        for mode in args.modes:
            started = time.perf_counter()
            entry = manifest.get(mode, {}).get(str(float(height)))
            if entry:
                path = Path(entry["path"])
                table = PointSolarTable.load(path, atmo, expected_label=entry["label"])
            else:
                table, path = table_for(atmo, args.table_dir, mode == "unified",
                                        args.n_height, args.derivative_tolerance)
            source = FiniteSolarSource(atmo, table, **source_options)
            if mode == "legacy":
                single = sky_radiance(atmo, 90-args.sun, [0], [0])[0]
                ss_info = dict(legacy=True)
            else:
                single, ss_info = single_zenith(atmo, source, 90-args.sun,
                    n_gauss=args.ss_gauss, segment_km=args.ss_segment)
            prepared[mode] = (table, path, single, ss_info)
            print(json.dumps(dict(stage="table_and_single", mode=mode, elevation=height,
                single_photopic=float(683*np.trapezoid(single*v, wl)),
                table_build_seconds=table.build_seconds, seconds=time.perf_counter()-started)), flush=True)
        for seed in args.seeds:
            fingerprint = None
            for mode in args.modes:
                table, path, single, ss_info = prepared[mode]
                source = FiniteSolarSource(atmo, table, **source_options)
                started = time.perf_counter()
                with mc_solar_source(source, legacy=mode == "legacy"):
                    result = mc.mc_radiance_split(atmo, 90-args.sun, [0], [0],
                        n_photons=args.photons, seed=seed)
                fp = source.fingerprint.hexdigest()
                if fingerprint is None:
                    fingerprint = fp
                else:
                    assert fingerprint == fp, "Source-only variants changed transport paths"
                fields = {key: value[0] for key, value in result.items()}
                fields["split_total"] = single + fields["multiple"]
                integrals = {k: float(683*np.trapezoid(value*v, wl)) for k, value in
                             fields.items() if not k.endswith("_err")}
                record = dict(head=head, library_diff_sha256=hashlib.sha256(library_diff).hexdigest(),
                    source_hashes=source_hashes, mode=mode, elevation=height, seed=seed,
                    wavelengths=wl.tolist(), sun_elevation=args.sun, config=config,
                    deterministic_single=single.tolist(), single_info=ss_info,
                    point_table_file=path.name,
                    point_table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    point_table_parameters=table.parameters,
                    path_fingerprint=fp, n_nee_calls=source.n_calls, n_nee_points=source.n_points,
                    fields={k: val.tolist() for k, val in fields.items()},
                    photopic_integrals=integrals, seconds=time.perf_counter()-started)
                with args.out.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
                print(json.dumps({key: record[key] for key in
                    ["mode", "elevation", "seed", "photopic_integrals", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
