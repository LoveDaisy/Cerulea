"""Paired 16-to-64 solar-disk quadrature check on identical geometric MC paths.

Use --run to create a fresh 64-point candidate with the public experiment driver;
otherwise only reduce existing JSONL. The archived raw records suffice to rerun
this reduction without scratchpad or an expensive new MC run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from skycolor import spectra


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT/"scratchpad/explore-twilight-altitude/phase-04-curved-diffuse"


def moments(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1)/np.sqrt(len(values))),
                replicates=values.tolist())


def reduce_pair(reference: Path, candidate: Path) -> dict:
    """Reintegrate spectra and use seed-level paired differences/covariances."""
    groups = []
    wavelengths = np.arange(380., 781., 5.)
    photopic_y = spectra.cie_xyz(wavelengths)[:, 1]

    def luminance(spectrum):
        return float(683*np.trapezoid(np.asarray(spectrum)*photopic_y, wavelengths))

    for path, resolution in [(reference, (2, 8)), (candidate, (4, 16))]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keyed = {(r["elevation"], r["seed"]): r for r in rows}
        if len(keyed) != 16 or len(rows) != 16:
            raise ValueError("Require 16 unique height/seed records for each disk resolution")
        if set(keyed) != {(h, seed) for h in (0., 2.4) for seed in range(310, 318)}:
            raise ValueError("Unexpected seed or height coverage")
        for row in rows:
            cfg = row["config"]
            if (row["mode"] != "unified" or row["sun_elevation"] != -6
                    or cfg["transport"] != "global-straight" or cfg["photons"] != 10000
                    or (cfg["disk_radial"], cfg["disk_azimuth"]) != resolution):
                raise ValueError("Unexpected source/transport/disk configuration")
            np.testing.assert_array_equal(row["wavelengths"], wavelengths)
            for field, stored in row["photopic_integrals"].items():
                np.testing.assert_allclose(luminance(row["fields"][field]), stored,
                                           atol=1e-15, rtol=2e-14)
        groups.append(keyed)
    low, high = groups
    pairs = []
    for key, row in low.items():
        other = high[key]
        for name in ["path_fingerprint", "n_nee_calls", "n_nee_points", "point_table_sha256"]:
            if row[name] != other[name]:
                raise ValueError(f"Disk refinement changed paired transport/table: {key}, {name}")
        if row["flight_stats"] != other["flight_stats"]:
            raise ValueError(f"Disk refinement changed flight events: {key}")
        if row["source_hashes"] != other["source_hashes"]:
            raise ValueError(f"Disk refinement used different implementation files: {key}")
        pairs.append(dict(height_km=key[0], seed=key[1], fingerprint=row["path_fingerprint"],
                          n_nee_calls=row["n_nee_calls"], n_nee_points=row["n_nee_points"]))
    result = dict(method="identical-path paired seed differences; eight independent seeds",
        disk_points=[16, 64], paths_per_seed_per_wavelength=10000,
        wavelength_count=81, sun_elevation_deg=-6, transport="global-straight",
        checks=dict(paired_fingerprints=pairs, flight_stats_identical=True), heights={},
        provenance={label: dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    heads=sorted({r["head"] for r in group.values()}),
                    library_diff_sha256=sorted({r["library_diff_sha256"] for r in group.values()}),
                    source_hashes=next(iter(group.values()))["source_hashes"])
                    for label,path,group in zip(["disk16", "disk64"],
                                               [reference,candidate],groups)})
    totals = {16: {}, 64: {}}
    for height in (0., 2.4):
        fields = {}
        for field in ["single", "multiple", "total", "split_total"]:
            a, b = [np.array([luminance(group[height,seed]["fields"][field])
                             for seed in range(310, 318)]) for group in groups]
            ratio = b.mean()/a.mean()
            influence = (b-ratio*a)/a.mean()
            fields[field] = dict(disk16=moments(a), disk64=moments(b),
                paired_difference=moments(b-a), fractional_change=dict(
                    estimate=float(ratio-1), se=float(influence.std(ddof=1)/np.sqrt(8))))
            if field == "split_total":
                totals[16][height], totals[64][height] = a, b
        deterministic = {}
        for points, group in zip([16,64],groups):
            spectra_ss = [group[height,seed]["deterministic_single"] for seed in range(310,318)]
            for item in spectra_ss[1:]:
                np.testing.assert_array_equal(item, spectra_ss[0])
            deterministic[str(points)] = luminance(spectra_ss[0])
        result["heights"][str(height)] = dict(fields=fields, deterministic_single=deterministic)
    ratios, influences = {}, {}
    for points in [16,64]:
        a,b = totals[points][0.],totals[points][2.4]
        ratio = b.mean()/a.mean()
        influence = (b-ratio*a)/a.mean()
        ratios[str(points)] = dict(estimate=float(ratio), se=float(influence.std(ddof=1)/np.sqrt(8)))
        influences[points] = influence
    ratios["difference"] = dict(estimate=ratios["64"]["estimate"]-ratios["16"]["estimate"],
        se=float((influences[64]-influences[16]).std(ddof=1)/np.sqrt(8)))
    result["height_ratios"] = ratios
    result["limitations"] = ["Source quadrature check on the sampled straight-path vertices only",
        "No certification of unsampled rare vertices or curved-path source quadrature",
        "Seed SE excludes source-table, optical-grid, view-integral and MC-tail systematic errors"]
    return result


def reduce_curved_seed(reference: Path, candidate: Path) -> dict:
    """Check one paired curved trajectory realization, without inventing a seed SE."""
    records = []
    for path in [reference,candidate]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("Curved probe requires one frozen record in each input")
        records.append(rows[0])
    a,b = records
    for row,resolution in zip(records,[(2,8),(4,16)]):
        cfg = row["config"]
        if (row["seed"] != 310 or row["mode"] != "unified"
                or cfg["transport"] != "continuous" or cfg["flight_step"] != 2000.
                or cfg["photons"] != 10000 or row["sun_elevation"] != -6
                or (cfg["disk_radial"],cfg["disk_azimuth"]) != resolution):
            raise ValueError("Unexpected curved probe definition")
        np.testing.assert_array_equal(row["wavelengths"],np.arange(380.,781.,5.))
    for key in ["elevation","path_fingerprint","n_nee_calls","n_nee_points", "flight_stats",
                "point_table_sha256","source_hashes"]:
        if a[key] != b[key]:
            raise ValueError(f"Curved disk refinement changed paired source/transport: {key}")
    v = spectra.cie_xyz(a["wavelengths"])[:,1]
    photopic = lambda spectrum: float(683*np.trapezoid(np.asarray(spectrum)*v,a["wavelengths"]))
    fields = {}
    for field in ["single","multiple","total","split_total"]:
        x,y = [photopic(row["fields"][field]) for row in records]
        for row,value in zip(records,[x,y]):
            np.testing.assert_allclose(value,row["photopic_integrals"][field],rtol=2e-14,atol=1e-15)
        fields[field] = dict(disk16=x,disk64=y,difference=y-x,fractional_change=y/x-1,
                             seed_standard_error=None)
    return dict(transport="continuous",max_step_m=2000.,seed=310,height_km=a["elevation"],
        disk_points=[16,64],path_fingerprint=a["path_fingerprint"],
        checks=dict(positions_calls_points_flight_stats_and_source_hashes_identical=True),
        fields=fields,deterministic_single={str(points): photopic(row["deterministic_single"])
            for points,row in zip([16,64],records)},
        inputs={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [reference,candidate]},
        limitation="One sampled curved-path realization; no independent-seed uncertainty estimate "
                   "and no bound on unvisited rare source vertices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=PHASE/"global-straight.jsonl")
    parser.add_argument("--candidate", type=Path, default=PHASE/"disk64-global-straight.jsonl")
    parser.add_argument("--out", type=Path, default=PHASE/"disk-convergence.json")
    parser.add_argument("--run", action="store_true", help="Run the 64-point candidate first")
    parser.add_argument("--curved",action="store_true",help="Reduce one pre-run curved seed pair")
    args = parser.parse_args()
    if args.run:
        if args.curved:
            parser.error("Run curved cases with tools.run_unified_solar_audit, then reduce snapshots")
        if args.candidate.exists():
            raise FileExistsError("Use a fresh candidate path; JSONL driver appends")
        args.candidate.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable,"-m","tools.run_unified_solar_audit",
            "--out",str(args.candidate),"--table-dir",str(args.candidate.parent/"tables"),
            "--tables-manifest",str(ROOT/"validation/data/unified_solar_source_audit/tables-manifest.json"),
            "--modes","unified","--elevations","0","2.4","--seeds",
            *map(str,range(310,318)),"--photons","10000","--transport","global-straight",
            "--disk-radial","4","--disk-azimuth","16","--wavelengths",
            *map(str,range(380,781,5))]
        with args.candidate.with_suffix(".log").open("w") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = (reduce_curved_seed(args.reference,args.candidate) if args.curved
              else reduce_pair(args.reference,args.candidate))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    if args.curved:
        print(json.dumps(result,indent=2))
    else:
        print(json.dumps(dict(heights={h: {f: v["paired_difference"] for f,v in x["fields"].items()
                    if f in ["multiple","split_total"]} for h,x in result["heights"].items()},
                    height_ratios=result["height_ratios"]),indent=2))


if __name__ == "__main__":
    main()
