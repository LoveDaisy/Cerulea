"""Verify frozen observation samples without running or fitting a sky solver.

The default check covers only the archived samples and exact MICA knots.
--full-source-folder additionally replays all 3192 published metadata columns.
Inputs are never modified; output is a JSON summary on stdout.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np


HERE = Path(__file__).resolve().parent


def read_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="latin1").splitlines():
        if not line.startswith("2014 "):
            continue
        fields = line.split()
        date = datetime.strptime(" ".join(fields[:4]), "%Y %b %d %H:%M:%S.%f")
        rows.append((date.replace(tzinfo=timezone.utc).timestamp(), 90 - float(fields[4])))
    times, elevation = np.asarray(rows).T
    assert np.all(np.diff(times) > 0)
    return times, elevation


def local_unix(label: str) -> float:
    return datetime.strptime(label, "%d-%b-%Y %H:%M:%S").replace(
        tzinfo=ZoneInfo("America/New_York")).timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-source-folder", type=Path)
    args = parser.parse_args()
    hashes = json.loads((HERE / "inputs.sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name

    packet = json.loads((HERE / "packet-provenance.json").read_text())
    audit = json.loads((HERE / "spectral/CSSP-sample-probe.json").read_text())
    with (HERE / "spectral/selected-metadata.csv").open() as stream:
        metadata = {int(row["source_column_zero_based"]): row for row in csv.DictReader(stream)}
    with (HERE / "spectral/selected-spectra.csv").open() as stream:
        reader = csv.reader(stream)
        columns = [int(name.removeprefix("col")) for name in next(reader)[1:]]
        matrix = np.asarray(list(reader), dtype=float)
    assert columns == packet["selected_source_columns_zero_based"] == sorted(metadata)
    wl = matrix[:, 0]
    np.testing.assert_array_equal(wl, np.arange(280, 841))
    visible = (wl >= 380) & (wl <= 780)
    energy = dict(zip(columns, np.trapezoid(matrix[visible, 1:], wl[visible], axis=0)))
    tables = {}
    for site in ("CSSP", "DRL"):
        for body in ("Sun", "Moon"):
            tables[f"{site}_{body}"] = read_table(
                HERE / f"spectral/{site}_{body}LocalZenith-knots.txt")
    reconstructed = {}
    for col, row in metadata.items():
        timestamp = local_unix(row["timestamp_EDT"])
        reconstructed[col] = {}
        for key, (times, angles) in tables.items():
            assert times[0] <= timestamp <= times[-1]
            reconstructed[col][key] = float(np.interp(timestamp, times, angles))
        np.testing.assert_allclose(reconstructed[col]["DRL_Sun"],
                                   float(row["published_sun_deg"]), atol=5e-8, rtol=0)
        np.testing.assert_allclose(reconstructed[col]["DRL_Moon"],
                                   float(row["published_moon_deg"]), atol=5e-8, rtol=0)
    for row in audit["selected_original_columns"]:
        col = row["source_column_zero_based"]
        np.testing.assert_allclose(energy[col], row["integrated_380_780_W_m2"], rtol=2e-13)
        for key, value in reconstructed[col].items():
            np.testing.assert_allclose(value, row[key + "_reinterpolated_deg"], atol=1e-11)
    for row in audit["samples_selected_on_reconstructed_CSSP_axis"]:
        col = row["source_column_zero_based"]
        np.testing.assert_allclose(energy[col], row["nearest_spectrum_integrated_380_780_W_m2"],
                                   rtol=2e-13)
        bracket = row["interpolation_bracket_source_columns_zero_based"]
        angles = [reconstructed[i]["CSSP_Sun"] for i in bracket]
        target = row["selection_target_reconstructed_CSSP_solar_elevation_deg"]
        assert angles[0] < target < angles[1]
        value = float(np.exp(np.interp(target, angles, np.log([energy[i] for i in bracket]))))
        np.testing.assert_allclose(value,
                                   row["target_integrated_380_780_W_m2_log_linear_interpolation"],
                                   rtol=2e-13)

    # Horizons is an independent airless topocentric sample, not a fitted axis shift.
    horizons = json.loads((HERE / "independent/horizons-cssp-sample.json").read_text())
    first_line = horizons["result"].split("$$SOE")[1].strip().splitlines()[0]
    jpl_elevation = float(first_line.split(",")[-2])
    np.testing.assert_allclose(jpl_elevation, reconstructed[132]["CSSP_Sun"], atol=1e-4)
    heights = []
    for name in ("FORS1.2001-09-21T10-06-58.303.header.txt",
                 "FORS2.2003-11-29T09-02-25.596.header.txt"):
        text = (HERE / "fors" / name).read_text()
        match = re.search(r"^HIERARCH ESO TEL GEOELEV\s*=\s*([\d.]+)", text, re.M)
        heights.append(float(match.group(1)))
    assert heights == [2648, 2648]
    meteo = list(csv.DictReader(io.StringIO(
        (HERE / "atmosphere/meteo-2003-11-29.csv").read_text().lstrip())))
    assert any(row["Date time"] == "2003-11-29T09:02:40" for row in meteo)
    labrada = (HERE / "photometric/labrada-20150120-21.txt").read_text().splitlines()
    assert len([line for line in labrada if line.strip()]) == 2880

    full_result = None
    if args.full_source_folder:
        source = args.full_source_folder / "CSSP_spectra.csv"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == packet["full_spectrum_sha256"]
        with source.open() as stream:
            rows = list(csv.reader(stream))
        assert rows[0][-1] == ""
        labels = rows[0][:-1]
        values = np.asarray(rows[1:], dtype=float)
        assert values.shape == (565, 3192)
        np.testing.assert_array_equal(values[4:, columns], matrix[:, 1:])
        unix = np.asarray([local_unix(label) for label in labels])
        full_result = {}
        for site in ("CSSP", "DRL"):
            for body, field in (("Sun", 1), ("Moon", 2)):
                times, angles = read_table(args.full_source_folder / f"{site}_{body}LocalZenith.txt")
                identity = next(row for row in json.loads(
                    (HERE / "spectral/knot-extractions.json").read_text())
                    if row["upstream_file"] == f"{site}_{body}LocalZenith.txt")
                assert hashlib.sha256((args.full_source_folder / identity["upstream_file"]).read_bytes()
                                      ).hexdigest() == identity["upstream_sha256"]
                assert times[0] <= unix.min() and unix.max() <= times[-1]
                diff = np.interp(unix, times, angles) - values[field]
                key = f"{site}_{body}"
                full_result[key] = float(np.max(abs(diff)))
                np.testing.assert_allclose(full_result[key],
                    audit["ephemeris_independent_reinterpolation"][key]["max_abs_diff_from_csv_deg"],
                    atol=1e-11)
    print(json.dumps({"frozen_files_verified": len(hashes), "selected_spectral_columns": len(columns),
                      "jpl_elevation_deg": jpl_elevation, "fits_heights_m": heights,
                      "labrada_minutes": 2880, "full_3192_metadata_replay": full_result}, indent=2))


if __name__ == "__main__":
    main()
