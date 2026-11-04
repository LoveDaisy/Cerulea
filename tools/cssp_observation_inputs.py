"""CSSP observation-input packet: sources, acquisition contract and operator.

This tool turns the public CSSP twilight-irradiance dataset (Spitschan et al.
2016, doi:10.1038/srep26756) into an auditable validation input: it pins
source identities, links published spectral columns to the original raw
``.mat`` acquisition records, recovers the per-record instrument/timing
contract, screens candidate nights with independent (non-model) conditions,
and defines the minimal mathematical observation operator a sky model must
integrate before any comparison.

Design bounds (plan task-cssp-observation-inputs):

- offline ``verify`` is the default acceptance path and never writes;
- ``fetch`` performs the authorised online acquisition (including one
  sequential read of the public raw archive) into a cache outside Git;
- ``build`` constructs the packet from the cache and refuses to overwrite an
  existing packet; Git receives selected small members plus derived records;
- ``sensitivity`` documents how derived quantities depend on unmeasured
  timing/response parameters. Scenario ranges demonstrate dependence; they
  are not confidence intervals or error bars.
- ``replay`` documents the raw->published processing chain on the selected
  samples; agreement with the published column shows reproduction of the
  upstream pipeline, not an independent absolute calibration.

Nothing here runs a sky solver or touches ``validation/results.json``.
Run from the worktree root, e.g.::

    uv run python -I tools/cssp_observation_inputs.py verify \
        --packet validation/data/cssp_observation_inputs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PACKET = ROOT / "validation/data/cssp_observation_inputs"
OLD_MATCHED = ROOT / "validation/data/matched_twilight_observations"
OLD_FEASIBILITY = ROOT / "validation/data/cssp_observation_feasibility"

# Observer site; coordinates are the ones validated against the independent
# JPL sample archived in the matched-observation packet (verify.py there).
CSSP = {
    "name_full": "Cherry Springs State Park",
    "name_short": "CSSP",
    "lat_deg": 41.6646,
    "lon_deg": -77.8125,
    "elevation_m": 710.0,
}
# The published solar/lunar axes used DRL (the upstream index bug); kept so
# verify can assert the two sites are never conflated.
DRL = {
    "name_full": "David Rittenhouse Laboratory",
    "name_short": "DRL",
    "lat_deg": 39.9522,
    "lon_deg": -75.1932,
    "elevation_m": 20.0,
}
# Neighbour weather stations characterised in the feasibility packet.
KN38 = {"name": "Wellsboro/KN38 (AWOS, via IEM)", "lat_deg": 41.7279,
        "lon_deg": -77.3965, "elevation_m": 572.88}
KBFD = {"name": "Bradford/KBFD (NOAA Global Hourly)", "lat_deg": 41.79835,
        "lon_deg": -78.6355, "elevation_m": 652.9}

AUTHOR_REPO_COMMIT = "1d2ecc69cb15885b147d2831cae89fbd15d471fa"
FULL_CSV_SHA256 = "e0e3e895b3a66fc83d9f63bd02e92899a11a44b1d1707f8a6d7a05279cbe6bab"
FULL_CSV_BYTES = 25481368
FULL_CSV_URL = ("https://static-content.springer.com/esm/"
                "art%3A10.1038%2Fsrep26756/MediaObjects/"
                "41598_2016_BFsrep26756_MOESM104_ESM.csv")
WAVELENGTHS_URL = ("https://static-content.springer.com/esm/"
                   "art%3A10.1038%2Fsrep26756/MediaObjects/"
                   "41598_2016_BFsrep26756_MOESM103_ESM.csv")
EPHEMERIS_URL = ("https://raw.githubusercontent.com/spitschan/"
                 "IlluminationSpectraDataset/" + AUTHOR_REPO_COMMIT +
                 "/analysis/data/ephemeris/{}.txt")
RAW_ARCHIVE_URL = "https://ndownloader.figshare.com/files/4862119"
RAW_ARCHIVE_FINAL_URL = ("https://s3-eu-west-1.amazonaws.com/"
                         "pfigshare-u-files/4862119/dataraw.tar.gz")
RAW_ARCHIVE_TOTAL_BYTES = 701893601
# Byte identity of the complete raw archive, from the single authorised full
# sequential read (2026-11-04). The historical 1 MiB prefix hash below stays
# prefix-only evidence and must never be presented as whole-archive identity.
RAW_ARCHIVE_FULL_SHA256 = (
    "e6f7a5c716a12a9f4dfb935c383321fe2270958df909d38edd0970d359c711ec")
RAW_ARCHIVE_PREFIX_1MIB_SHA256 = (
    "9bd6202b80a3c4612121a701519fae774d250a78a73bf6a26d578a617f0a0ea9")
PUBLISHED_WAVELENGTHS_SHA256 = (
    "479dad2965c373b73052b518d33621385b80dead1d64964e19b610b363a8200e")
PUBLISHED_WAVELENGTHS_BYTES = 2244

# Full-CSV layout: 5 metadata rows then 561 spectrum rows, all x 3192 columns.
# Row 0 is the authors' sequence (acquisition-run) index; rows 1/2 are the
# published solar/lunar elevations (deg, DRL site - known wrong for CSSP);
# row 3 is the published lunar illumination fraction; spectra follow on the
# published 1 nm grid which the CSV itself does not store.
CSV_SEQUENCE_ROW, CSV_SUN_ROW, CSV_MOON_ROW, CSV_MOONFRAC_ROW, CSV_FIRST_SPECTRUM_ROW = 0, 1, 2, 3, 4

# Nominal cadence inside a session is ~60 s; sessions are hours apart.
SESSION_GAP_SECONDS = 900.0

EDT = ZoneInfo("America/New_York")

EVIDENCE_RAW = "raw-measured"
EVIDENCE_DERIVED = "derived-documented"
EVIDENCE_PUBLISHED = "published-upstream"
EVIDENCE_SCENARIO = "scenario"
EVIDENCE_UNKNOWN = "unknown"

IEM_STATION = {"station": "N38", "network": "PA_ASOS"}
NOAA_STATION = "72526604751"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km; longitude in degrees east-positive."""
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def parse_edt_label(label: str) -> datetime:
    """Parse one official CSV timestamp label (local wall clock) into UTC.

    Labels are ``%d-%b-%Y %H:%M:%S`` in the acquisition machine's local zone.
    ZoneInfo resolves ``America/New_York`` including the EDT (UTC-4) period
    the dataset spans. Unparseable labels raise: callers never silently
    re-sort or drop records.
    """
    try:
        naive = datetime.strptime(label, "%d-%b-%Y %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp label {label!r}: {exc}") from exc
    return naive.replace(tzinfo=EDT)


def label_to_utc_iso(label: str) -> str:
    return parse_edt_label(label).astimezone(timezone.utc).isoformat()


def label_to_invoke_time(label: str) -> str:
    """Official label -> raw ``describe.invokeTime`` form ``20140703T042449``.

    Used to join raw archive members to CSV columns. Equality of the derived
    name is necessary but not sufficient; sequence/site/wavelength checks are
    applied separately in :func:`map_columns_to_members`.
    """
    return parse_edt_label(label).strftime("%Y%m%dT%H%M%S")


def invoke_time_to_utc_iso(invoke_time: str) -> str:
    moment = datetime.strptime(invoke_time, "%Y%m%dT%H%M%S")
    return moment.replace(tzinfo=EDT).astimezone(timezone.utc).isoformat()


def unix_of(utc_iso: str) -> float:
    return datetime.fromisoformat(utc_iso).timestamp()


# ---------------------------------------------------------------------------
# Full CSV access
# ---------------------------------------------------------------------------

def parse_full_csv_rows(rows: list[list[str]]) -> tuple[list[str], np.ndarray,
                                                        dict[int, list[str]]]:
    """Parse official-CSV text rows without touching any field.

    Returns ``(labels, values, raw_columns)`` where ``raw_columns`` keeps the
    verbatim cell strings of every column (spectra excerpts later stay
    byte-identical to upstream instead of round-tripping through floats).
    The trailing empty header cell (the wavelength corner) is checked.
    """
    if rows[0][-1] != "":
        raise ValueError("full CSV header must end with the empty corner cell")
    labels = rows[0][:-1]
    values = np.asarray(rows[1:], dtype=float)
    if values.shape[1] != len(labels) or values.shape[0] != 565:
        raise ValueError(f"unexpected full CSV shape {values.shape}")
    return labels, values, {c: [rows[r][c] for r in range(1, len(rows))]
                            for c in range(len(labels))}


def load_full_csv(path: Path) -> tuple[list[str], np.ndarray, dict[int, list[str]]]:
    with path.open(newline="") as stream:
        return parse_full_csv_rows(list(csv.reader(stream)))


def sessions_of(labels: list[str], gap_seconds: float = SESSION_GAP_SECONDS) -> list[dict]:
    """Split the columns into acquisition sessions; frozen rule, no model input.

    A session is a maximal run of columns whose consecutive labels differ by
    at most ``gap_seconds``. Sessions are the audit unit for night selection.
    """
    times = [parse_edt_label(label) for label in labels]
    edges = [0]
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]).total_seconds() > gap_seconds:
            edges.append(i)
    edges.append(len(times))
    out = []
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        span = times[stop - 1] - times[start]
        crosses = any(times[k].date() != times[start].date() for k in range(start, stop))
        out.append({
            "session_id": f"s{index:02d}",
            "columns_zero_based": [start, stop - 1],
            "n_columns": stop - start,
            "start_label_edt": labels[start],
            "end_label_edt": labels[stop - 1],
            "start_utc": label_to_utc_iso(labels[start]),
            "end_utc": label_to_utc_iso(labels[stop - 1]),
            "span_minutes": round(span.total_seconds() / 60.0, 3),
            "crosses_local_midnight": crosses,
            "local_date_of_first_sample": times[start].date().isoformat(),
        })
    return out


def session_signal_stats(values: np.ndarray, start: int, stop: int) -> dict:
    """Signal-level screening descriptors (never used for fitting).

    ``no_uv_columns`` counts records whose published 280-359 nm rows are all
    NaN; per-record band masks live in records.json. The median-positive
    screen separates twilight/night sessions (1e-9..1e-6 W m-2 nm-1) from
    daytime ones (~1e-1).
    """
    spectra = values[CSV_FIRST_SPECTRUM_ROW:, start:stop + 1]
    nan_mask = np.isnan(spectra)
    uv_all_nan = np.isnan(spectra[:80, :]).all(axis=0)
    positive = spectra[~nan_mask & (spectra > 0)]
    return {
        "nan_fraction": round(float(nan_mask.mean()), 6),
        "no_uv_columns": int(uv_all_nan.sum()),
        "zero_cells": int((spectra == 0).sum()),
        "negative_cells": int((spectra < 0).sum()),
        "median_positive_w_m2_nm": float(np.median(positive)) if positive.size else None,
    }


def build_session_inventory(labels: list[str], values: np.ndarray) -> list[dict]:
    inventory = []
    for session in sessions_of(labels):
        start, stop = session["columns_zero_based"]
        stats = session_signal_stats(values, start, stop)
        median = stats["median_positive_w_m2_nm"]
        session.update(stats)
        session["sequence_values"] = sorted({int(v) for v in np.unique(
            values[CSV_SEQUENCE_ROW, start:stop + 1])})
        session["signal_kind"] = (
            "twilight-or-night" if median is not None and median < 1e-5
            else "daytime-level" if median is not None else "empty")
        inventory.append(session)
    return inventory


# ---------------------------------------------------------------------------
# Ephemeris: vendored upstream MICA tables (UT1, 5-minute cadence)
# ---------------------------------------------------------------------------

def read_mica_table(path: Path) -> dict:
    """Read a full upstream MICA ``*_LocalZenith.txt`` table.

    Data lines are ``YYYY Mon DD HH:MM:SS.f <zenith_deg> <azimuth_deg> <au>``;
    times are UT1 (documented upstream header), treated as UTC here - the
    UT1-UTC difference in 2014 was < 0.4 s, far below the 5-minute bracket
    interpolation error, and recorded in the packet contract. Elevation is
    90 - zenith. Times must be strictly increasing (asserted, not re-sorted).
    """
    rows = []
    for line in path.read_text(encoding="latin1").splitlines():
        if not line.startswith("2014 "):
            continue
        fields = line.split()
        date = datetime.strptime(" ".join(fields[:4]), "%Y %b %d %H:%M:%S.%f")
        rows.append((date.replace(tzinfo=timezone.utc).timestamp(),
                     90.0 - float(fields[4]), float(fields[5])))
    if not rows:
        raise ValueError(f"no data lines parsed from {path.name}")
    times, elevation, azimuth = np.asarray(rows).T
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"MICA times not strictly increasing in {path.name}")
    return {"times": times, "elevation_deg": elevation, "azimuth_deg": azimuth}


def mica_elevation(table: dict, unix_seconds: float) -> dict:
    """Interpolate one axis; refuse to extrapolate beyond the table.

    Returns the elevation plus the exact bracket used, so the packet can show
    the interpolation is inside the 5-minute cadence (no table-extrapolated
    ephemeris for nominal exposure nodes).
    """
    times = table["times"]
    if not times[0] <= unix_seconds <= times[-1]:
        raise ValueError("requested time outside MICA table span")
    i = int(np.searchsorted(times, unix_seconds)) - 1
    i = min(max(i, 0), len(times) - 2)
    if not times[i] <= unix_seconds <= times[i + 1]:
        raise ValueError("bracket search failed")
    h0, h1 = table["elevation_deg"][i], table["elevation_deg"][i + 1]
    value = float(h0 + (h1 - h0) * (unix_seconds - times[i]) / (times[i + 1] - times[i]))
    return {"elevation_deg": value,
            "bracket_utc": [datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                            datetime.fromtimestamp(times[i + 1], tz=timezone.utc).isoformat()],
            "bracket_elevation_deg": [float(h0), float(h1)],
            "max_bracket_minutes": float((times[i + 1] - times[i]) / 60.0)}


# ---------------------------------------------------------------------------
# Raw archive: index, extraction, MAT contract
# ---------------------------------------------------------------------------

def scan_raw_index(tar_path: Path) -> dict:
    """Sequentially scan the gzip tar (no random access exists) for members."""
    import tarfile
    members = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            members.append({"name": m.name, "size": m.size, "mtime": m.mtime,
                            "type": m.type.decode("ascii", "replace")})
    return {"archive_sha256": sha256_file(tar_path),
            "total_bytes": tar_path.stat().st_size, "members": members}


def extract_raw_members(tar_path: Path, wanted_names: list[str],
                        out_dir: Path) -> dict[str, dict]:
    """One sequential pass extracting only the named regular members.

    The upstream names are preserved as file names inside ``out_dir``; the
    result records byte identity so the packet can prove members were not
    rewritten. Traversal is bounded to exact-name matches (no glob, no
    symlink following).
    """
    import tarfile
    wanted = set(wanted_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    got: dict[str, dict] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if m.name not in wanted or not m.isreg():
                continue
            target = out_dir / Path(m.name).name
            with tar.extractfile(m) as src, target.open("wb") as dst:
                data = src.read()
                dst.write(data)
            got[m.name] = {"packet_path": f"raw/mat/{target.name}",
                           "bytes": len(data), "sha256": sha256_bytes(data)}
            if len(got) == len(wanted):
                break
    missing = wanted - set(got)
    if missing:
        raise ValueError(f"members not found in archive: {sorted(missing)}")
    return got


def read_mat_contract(mat_path: Path) -> dict:
    """Extract the acquisition contract fields from one raw ``.mat`` record.

    Every field records where it came from; a field absent from the file
    becomes ``{"value": None, "reason": "absent in raw"}`` rather than a
    fabricated number. Repeats are derived from the actual ``spd`` matrix
    shape (the source code proves columns are successive spectra; the
    hardware never writes per-exposure start/end times).
    """
    from scipy.io import loadmat
    meas = loadmat(mat_path, simplify_cells=True)["meas"]
    describe = meas.get("describe", {}) or {}
    spec_info = describe.get("spectrometerInfo", {}) or {}
    position = describe.get("position", {}) or {}
    pos_name = position.get("name", {}) or {}
    spec_name = spec_info.get("name", {}) or {}
    wls = np.asarray(meas["wls"], dtype=float).ravel()
    spd = np.asarray(meas["spd"], dtype=float)
    if spd.ndim == 1:
        # simplify_cells collapses a single repeat to 1-D; columns are repeats.
        spd = spd.reshape(-1, 1)
    temps = meas.get("boardTempInCelsius")
    temps = [float(t) for t in np.atleast_1d(temps)] if temps is not None else []
    finite = spd[np.isfinite(spd)]

    def field(value):
        return {"value": value,
                "evidence": EVIDENCE_RAW if value is not None else EVIDENCE_UNKNOWN,
                **({} if value is not None else {"reason": "absent in raw"})}

    return {
        "invoke_time": field(str(describe.get("invokeTime"))),
        "site_abbrev": field(str(pos_name.get("abbrev"))),
        "sample_id": field(str(describe.get("sampleID"))),
        "serial_number": field(str(spec_info.get("serialNumber"))),
        "spectrometer_abbrev": field(str(spec_name.get("abbrev"))),
        "model_type": field(str(spec_info.get("modelType"))),
        "script_file_name": field(str(describe.get("scriptFileName"))),
        "dark_flag": field(int(meas["darkFlag"]) if meas.get("darkFlag") is not None else None),
        "nonlinearity_correction_flag": field(
            int(meas["setCorrectForDetectorNonlinearityFlag"])
            if meas.get("setCorrectForDetectorNonlinearityFlag") is not None else None),
        "electrical_dark_correction_flag": field(
            int(meas["setCorrectForElectricalDarkFlag"])
            if meas.get("setCorrectForElectricalDarkFlag") is not None else None),
        "integration_time_us": field(int(meas["integrationTime"])
                                     if meas.get("integrationTime") is not None else None),
        "integration_time_set_us": field(int(meas["integrationTimeSet"])
                                         if meas.get("integrationTimeSet") is not None else None),
        "n_exposures": field(int(spd.shape[1])),
        "n_pixels": field(int(spd.shape[0])),
        "board_temp_c": field(temps),
        "wls_first_nm": field(float(wls[0])),
        "wls_last_nm": field(float(wls[-1])),
        "wls_sha256": sha256_bytes(np.ascontiguousarray(wls, dtype="<f8").tobytes()),
        "spd_shape": [int(spd.shape[0]), int(spd.shape[1])],
        "spd_max_value": field(float(finite.max()) if finite.size else None),
    }


def nominal_exposure_window(contract: dict, invoke_utc_iso: str) -> dict:
    """Nominal exposure windows per plan §3.2, with unknowns kept unknown.

    With recorded per-exposure duration T and N repeats, the scenario window
    assumes zero clock offset, zero setup latency, zero inter-exposure gap
    and no driver caching - none of which is measured. ``N*T`` is the total
    photon integration time, not the wall-clock span. A midpoint exists only
    inside this scenario and must never be quoted as a measured exposure
    midpoint (learned lesson: invoke timestamp != exposure window).
    """
    inv = datetime.fromisoformat(invoke_utc_iso)
    t_us = contract["integration_time_us"]["value"]
    n = contract["n_exposures"]["value"]
    if t_us is None or n is None or n < 1:
        return {"scenario": "zero-extra-latency back-to-back",
                "evidence": EVIDENCE_UNKNOWN,
                "reason": "integration time or repeat count not recoverable",
                "exposure_seconds": None, "window_start_utc": None,
                "window_end_utc": None, "midpoint_utc_scenario": None,
                "nominal_total_exposure_s": None,
                "unknown": _timing_unknowns()}
    t_s = t_us / 1e6
    start = inv
    end = inv + timedelta(seconds=n * t_s)
    return {
        "scenario": "zero clock offset, zero setup latency, back-to-back exposures, no driver caching",
        "evidence": EVIDENCE_SCENARIO,
        "exposure_seconds": [t_s] * n,
        "window_start_utc_scenario": start.isoformat(),
        "window_end_utc_scenario": end.isoformat(),
        "midpoint_utc_scenario": (start + (end - start) / 2).isoformat(),
        "nominal_total_exposure_s": n * t_s,
        "note": "N*T is total photon integration, not the measured wall-clock span",
        "unknown": _timing_unknowns(),
    }


def _timing_unknowns() -> dict:
    return {
        "clock_offset_s": {"value": None, "evidence": EVIDENCE_UNKNOWN,
                           "reason": "no hardware clock or sync log is archived"},
        "setup_latency_s": {"value": None, "evidence": EVIDENCE_UNKNOWN,
                            "reason": "OO_MeasureAndSave.m records invokeTime before "
                                      "setting integration time; delay not recorded"},
        "inter_exposure_gap_s": {"value": None, "evidence": EVIDENCE_UNKNOWN,
                                 "reason": "no per-exposure start/end times exist in raw"},
        "driver_caching": {"value": None, "evidence": EVIDENCE_UNKNOWN,
                           "reason": "acquisition source proves sequential calls, "
                                     "not absence of SDK buffering"},
    }


# ---------------------------------------------------------------------------
# Column <-> raw member mapping
# ---------------------------------------------------------------------------

def map_columns_to_members(labels: list[str], index: dict,
                           columns: list[int]) -> dict[str, dict]:
    """Map CSV columns to raw archive members by derived member name.

    A match requires the exact upstream name built from the label; sequence
    and site/wavelength consistency are verified later per record from the
    MAT itself. Zero matches raise (never silently drop); duplicate matches
    raise for explicit review (plan: one-to-many must fail loudly).
    """
    by_name: dict[str, list[dict]] = {}
    for m in index["members"]:
        if m["type"] == "0" and m["size"] > 0 and m["name"].startswith("dataraw/CSSP/"):
            by_name.setdefault(Path(m["name"]).name, []).append(m)
    mapping = {}
    for col in columns:
        stamp = label_to_invoke_time(labels[col])
        matches = [name for name in by_name
                   if name.startswith(f"{stamp}_sOO_") and name.endswith("_pCSSP.mat")]
        if len(matches) > 1:
            raise ValueError(f"column {col}: ambiguous raw members for {stamp}: {matches}")
        if not matches:
            raise ValueError(f"column {col}: no raw member for invoke time {stamp}")
        entries = by_name[matches[0]]
        if len(entries) > 1:
            raise ValueError(f"column {col}: ambiguous duplicate raw members "
                             f"for {matches[0]}")
        mapping[col] = {"upstream_member": entries[0]["name"],
                        "upstream_name": matches[0],
                        "size": entries[0]["size"]}
    return mapping


# ---------------------------------------------------------------------------
# Sources and packet file helpers
# ---------------------------------------------------------------------------

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def external_ref(path_from_root: str) -> dict:
    """Reference an existing Git-tracked file by repo-relative path + SHA."""
    p = ROOT / path_from_root
    return {"path": path_from_root, "sha256": sha256_file(p), "bytes": p.stat().st_size}


def check_manifest(packet: Path) -> dict:
    manifest = read_json(packet / "inputs.sha256.json")
    failures = []
    for name, expected in manifest.items():
        path = packet / name
        if not path.is_file():
            failures.append({"file": name, "problem": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"file": name, "problem": "hash mismatch",
                             "expected": expected, "actual": actual})
    return {"n_files": len(manifest), "failures": failures}


def check_external_refs(packet: Path) -> dict:
    sources = read_json(packet / "sources.json")
    failures = []
    n_checked = 0
    for entry in sources.get("external_refs", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append({"path": entry["path"], "problem": "missing"})
            continue
        if sha256_file(path) != entry["sha256"]:
            failures.append({"path": entry["path"], "problem": "hash mismatch"})
        n_checked += 1
    return {"n_refs": n_checked, "failures": failures}


# ---------------------------------------------------------------------------
# Packet data loading
# ---------------------------------------------------------------------------

def load_packet_spectra(packet: Path) -> dict:
    """Load ``spectra.csv``: published 1 nm grid + verbatim spectral excerpts.

    Cells are the upstream strings (``NaN`` preserved), so the excerpt is
    byte-comparable with the official CSV column.
    """
    with (packet / "spectra.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    if header[0] != "wl_nm" or not all(name.startswith("col") for name in header[1:]):
        raise ValueError("unexpected spectra.csv header")
    columns = [int(name.removeprefix("col")) for name in header[1:]]
    data = np.asarray([[float(v) for v in row[1:]] for row in rows[1:]], dtype=float)
    wl = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    if not np.array_equal(wl, np.arange(280.0, 841.0)):
        raise ValueError("spectra.csv wavelength grid is not the published 280..840 grid")
    if data.shape[1] != len(columns):
        raise ValueError("spectra.csv column count mismatch")
    return {"wavelength_nm": wl, "columns": columns,
            "record_ids": [f"c{col}" for col in columns], "data": data}


# ---------------------------------------------------------------------------
# verify (offline acceptance; never writes)
# ---------------------------------------------------------------------------

def verify_packet(packet: Path, source_dir: Path | None = None) -> dict:
    """Full offline verification of the frozen packet.

    Recomputes every derived value the packet claims (UTC derivation, MICA
    axes, band masks, raw contracts) from the vendored originals and checks
    identity of every frozen and referenced file. With ``source_dir``, also
    checks the complete official CSV and byte-compares every excerpt column
    against it.
    """
    report: dict = {"packet": str(packet)}
    problems: list[str] = []

    manifest = check_manifest(packet)
    report["frozen_files"] = manifest
    refs = check_external_refs(packet)
    report["external_refs"] = refs
    problems += [f"manifest: {f}" for f in manifest["failures"]]
    problems += [f"external ref: {f}" for f in refs["failures"]]

    records = read_json(packet / "records.json")
    selection = read_json(packet / "selection.json")
    conditions_doc = read_json(packet / "conditions.json")
    conditions = conditions_doc["conditions"]
    operator = read_json(packet / "operator.json")
    sources = read_json(packet / "sources.json")
    report["n_records"] = len(records["records"])

    if sources.get("site", {}).get("lat_deg") != CSSP["lat_deg"] or \
            sources.get("site", {}).get("lon_deg") != CSSP["lon_deg"]:
        problems.append("packet site is not CSSP")
    if operator.get("formula_name") != "hemispheric-spectral-exposure":
        problems.append("operator contract missing or renamed")

    sun_tab = read_mica_table(packet / "ephemeris/CSSP_SunLocalZenith.txt")
    moon_tab = read_mica_table(packet / "ephemeris/CSSP_MoonLocalZenith.txt")

    spectra = load_packet_spectra(packet)
    by_id = {record["record_id"]: record for record in records["records"]}
    if spectra["columns"] != [by_id[r]["source_column_zero_based"] for r in spectra["record_ids"]]:
        problems.append("spectra.csv columns do not match records.json")

    for record_id, col in zip(spectra["record_ids"], spectra["columns"]):
        record = by_id[record_id]
        if record["source_column_zero_based"] != col:
            problems.append(f"{record_id}: column mismatch between files")
            continue
        # UTC derivation must round-trip the verbatim label.
        if label_to_utc_iso(record["timestamp_label_edt"]) != record["invoke_time_utc"]:
            problems.append(f"{record_id}: UTC derivation does not round-trip label")
        # Corrected axes must recompute from the vendored MICA tables.
        unix = unix_of(record["invoke_time_utc"])
        for body, tab, key in (("sun", sun_tab, "sun_elevation_deg"),
                               ("moon", moon_tab, "moon_elevation_deg")):
            try:
                got = mica_elevation(tab, unix)["elevation_deg"]
            except ValueError:
                got = None
            claimed = record["corrected_axis_cssp"].get(key)
            if claimed is None:
                if got is not None:
                    problems.append(f"{record_id}: corrected {body} axis is null "
                                    "but table covers the instant")
            elif got is None or abs(got - claimed) > 1e-9:
                problems.append(f"{record_id}: corrected {body} axis mismatch")
        # Band mask must recompute from the excerpt.
        column = spectra["data"][:, spectra["record_ids"].index(record_id)]
        nan_mask = np.isnan(column)
        expected_mask = []
        if nan_mask.any():
            edges = np.flatnonzero(np.diff(np.concatenate(([0], nan_mask.view(np.int8),
                                                            [0]))))
            for a, b in zip(edges[::2], edges[1::2]):
                expected_mask.append([int(spectra["wavelength_nm"][a]),
                                      int(spectra["wavelength_nm"][b - 1])])
        if record["quality"]["available_band_mask_nm"] != expected_mask:
            problems.append(f"{record_id}: band mask does not recompute")
        # Raw contract must re-read identically from the vendored MAT.
        raw_ref = record.get("raw")
        if raw_ref is not None:
            member = packet / raw_ref["packet_path"]
            if not member.is_file():
                problems.append(f"{record_id}: raw member missing")
            elif sha256_file(member) != raw_ref["sha256"]:
                problems.append(f"{record_id}: raw member hash mismatch")
            else:
                contract = read_mat_contract(member)
                for key, claimed in record["instrument"].items():
                    got = contract.get(key)
                    got_value = got.get("value") if isinstance(got, dict) else got
                    claimed_value = claimed.get("value") if isinstance(claimed, dict) else claimed
                    if key == "board_temp_c" and got_value and claimed_value:
                        if not np.allclose(got_value, claimed_value, atol=1e-9):
                            problems.append(f"{record_id}: board temp mismatch")
                        continue
                    if got_value != claimed_value:
                        problems.append(f"{record_id}: instrument field {key} mismatch "
                                        f"({got_value!r} vs {claimed_value!r})")
        else:
            problems.append(f"{record_id}: no raw member bound (AC1 requires one)")

    # Independent axis checks must recompute from the vendored JPL responses.
    for check in conditions_doc.get("independent_axis_checks", []):
        path = packet / check["response_path"]
        if not path.is_file():
            problems.append(f"jpl check {check['record_id']}: response missing")
            continue
        jpl = read_jpl_elevation(path)
        mica = mica_elevation(sun_tab, unix_of(check["instant_utc"]))["elevation_deg"]
        if abs(jpl - check["jpl_elevation_deg"]) > 1e-9 or \
                abs(mica - check["mica_elevation_deg"]) > 1e-9:
            problems.append(f"jpl check {check['record_id']}: stored values differ from "
                            "a fresh recomputation")
        if abs(jpl - mica) >= check["agreement_bound_deg"]:
            problems.append(f"jpl check {check['record_id']}: agreement bound exceeded")

    # Selection and conditions coherence.
    inventory_ids = {s["session_id"] for s in selection["sessions"]}
    for decision in selection["decisions"]:
        if decision["session_id"] not in inventory_ids:
            problems.append(f"selection references unknown {decision['session_id']}")
        if not decision.get("reason"):
            problems.append(f"{decision['session_id']}: decision without reason")
        if decision["decision"] == "selected" and decision["session_id"] not in conditions:
            problems.append(f"{decision['session_id']}: selected but no conditions entry")

    # Selected-session coherence: every record's session_id must be the
    # session that actually contains its column.
    for record in records["records"]:
        col = record["source_column_zero_based"]
        owning = next((s["session_id"] for s in selection["sessions"]
                       if s["columns_zero_based"][0] <= col <= s["columns_zero_based"][1]),
                      None)
        if owning != record["session_id"]:
            problems.append(f"{record['record_id']}: session_id does not match inventory")

    report["problems"] = problems

    if source_dir is not None:
        csv_path = source_dir / "CSSP_spectra.csv"
        digest = sha256_file(csv_path)
        report["full_csv"] = {"sha256": digest, "bytes": csv_path.stat().st_size,
                              "match": digest == FULL_CSV_SHA256
                              and csv_path.stat().st_size == FULL_CSV_BYTES}
        if report["full_csv"]["match"]:
            with csv_path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            _, _, raw_cols = parse_full_csv_rows(rows)
            with (packet / "spectra.csv").open(newline="") as stream:
                excerpt = list(csv.reader(stream))
            for j, col in enumerate(spectra["columns"]):
                for i in range(1, len(excerpt)):
                    if excerpt[i][j + 1] != raw_cols[col][CSV_FIRST_SPECTRUM_ROW + i - 1]:
                        problems.append(f"column {col}: excerpt row {i} differs "
                                        "from official CSV")
                        break
        report["problems"] = problems

    report["ok"] = not problems and not manifest["failures"] and not refs["failures"]
    return report


# ---------------------------------------------------------------------------
# fetch (online; explicit subcommand, cache outside Git)
# ---------------------------------------------------------------------------

def fetch_sources(cache_dir: Path, with_raw_archive: bool) -> dict:
    """Download public sources into ``cache_dir``; skip items already cached.

    Every item records URL, bytes and sha256 in ``fetch-provenance.json``.
    Identity is checked after download; a wrong identity leaves the failure
    in the provenance and raises. Failed probes (e.g. an endpoint returning
    an error page) are kept as evidence, never turned into empty success.
    """
    import urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = [
        {"id": "full-csv", "url": FULL_CSV_URL, "path": "CSSP_spectra.csv",
         "sha256": FULL_CSV_SHA256, "bytes": FULL_CSV_BYTES},
        {"id": "published-wavelengths", "url": WAVELENGTHS_URL,
         "path": "published-wavelengths.csv", "sha256": PUBLISHED_WAVELENGTHS_SHA256,
         "bytes": PUBLISHED_WAVELENGTHS_BYTES},
        {"id": "mica-cssp-sun", "url": EPHEMERIS_URL.format("CSSP_SunLocalZenith"),
         "path": "CSSP_SunLocalZenith.txt"},
        {"id": "mica-cssp-moon", "url": EPHEMERIS_URL.format("CSSP_MoonLocalZenith"),
         "path": "CSSP_MoonLocalZenith.txt"},
    ]
    if with_raw_archive:
        items.append({"id": "raw-archive", "url": RAW_ARCHIVE_URL,
                      "final_url": RAW_ARCHIVE_FINAL_URL, "path": "dataraw.tar.gz",
                      "bytes": RAW_ARCHIVE_TOTAL_BYTES,
                      "sha256": RAW_ARCHIVE_FULL_SHA256})
    provenance = []
    for item in items:
        target = cache_dir / item["path"]
        entry = dict(item)
        entry.pop("final_url", None)
        entry["cached"] = target.is_file()
        if not target.is_file():
            request = urllib.request.Request(item["url"],
                                             headers={"User-Agent": "sky-color-sim/1.0"})
            with urllib.request.urlopen(request, timeout=600) as resp, \
                    target.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            entry["cached"] = True
        entry["bytes"] = target.stat().st_size
        entry["sha256"] = sha256_file(target)
        if "sha256" in item and entry["sha256"] != item["sha256"]:
            entry["identity"] = "MISMATCH"
            provenance.append(entry)
            write_json(cache_dir / "fetch-provenance.json", provenance)
            raise ValueError(f"{item['id']}: sha mismatch after download")
        if "bytes" in item and entry["bytes"] != item["bytes"]:
            entry["identity"] = "BYTE_COUNT_MISMATCH"
            provenance.append(entry)
            write_json(cache_dir / "fetch-provenance.json", provenance)
            raise ValueError(f"{item['id']}: byte count mismatch after download")
        entry["identity"] = "verified"
        provenance.append(entry)
    write_json(cache_dir / "fetch-provenance.json", provenance)
    return {"items": [{k: v for k, v in e.items() if k != "members"}
                      for e in provenance]}


# ---------------------------------------------------------------------------
# build: packet construction
# ---------------------------------------------------------------------------

# Frozen sample-selection rules for the packet (no model residuals involved).
TARGET_ELEVATIONS_DEG = (-12.0, -9.0, -6.0)
DEEP_NIGHT_LOCAL_HOURS = (2, 3, 4)
# Session ids come from the frozen 900 s gap split of the 3192 columns; the
# 7/18 dusk -> 7/19 dawn night is ONE session (the sequence 8->9 restart
# inside it has no time gap), so the secondary night has a single session id.
PRIMARY_SESSION = "s04"     # 2014-07-02 dusk -> 2014-07-03 dawn
SECONDARY_SESSION = "s11"   # 2014-07-18 dusk  -> 2014-07-19 dawn


def corrected_axes_for_columns(labels: list[str], columns: list[int],
                               sun_tab: dict, moon_tab: dict) -> dict[int, dict]:
    axes = {}
    for col in columns:
        unix = unix_of(label_to_utc_iso(labels[col]))
        entry = {}
        for body, tab in (("sun_elevation_deg", sun_tab),
                          ("moon_elevation_deg", moon_tab)):
            try:
                hit = mica_elevation(tab, unix)
                entry[body] = hit["elevation_deg"]
                entry[f"{body.split('_')[0]}_bracket"] = hit["bracket_utc"]
            except ValueError:
                entry[body] = None
                entry[f"{body.split('_')[0]}_reason"] = "outside vendored MICA table span"
        axes[col] = entry
    return axes


def choose_session_samples(labels: list[str], values: np.ndarray,
                           inventory: list[dict], sun_tab: dict) -> dict[str, list[int]]:
    """Apply the frozen selection rules to the two selected nights.

    Dusk/dawn twilight targets: published grid columns closest to the target
    corrected solar elevations, separately on each side of the night. Deep
    night: columns closest to fixed local hours. Neighbours: +-1 (always)
    and +-2 (primary night) columns for adjacent-spectrum change rates.
    """
    picks: dict[str, list[int]] = {PRIMARY_SESSION: [], SECONDARY_SESSION: []}
    sun_by_col: dict[int, float] = {}

    def sun(col: int) -> float:
        if col not in sun_by_col:
            unix = unix_of(label_to_utc_iso(labels[col]))
            sun_by_col[col] = mica_elevation(sun_tab, unix)["elevation_deg"]
        return sun_by_col[col]

    def session_cols(sid: str) -> list[int]:
        sess = next(s for s in inventory if s["session_id"] == sid)
        return list(range(sess["columns_zero_based"][0], sess["columns_zero_based"][1] + 1))

    for sid, radius in ((PRIMARY_SESSION, 2), (SECONDARY_SESSION, 1)):
        cols = session_cols(sid)
        for target_deg in TARGET_ELEVATIONS_DEG:
            dawn_cols = [c for c in cols if parse_edt_label(labels[c]).hour < 12]
            target = min(dawn_cols, key=lambda c: abs(sun(c) - target_deg))
            picks[sid] += _with_neighbors(cols, [target], radius)
        dusk_cols = [c for c in cols if parse_edt_label(labels[c]).hour >= 12]
        target = min(dusk_cols, key=lambda c: abs(sun(c) - (-6.0)))
        picks[sid] += _with_neighbors(cols, [target], radius)
        # Sequence boundary pair: the acquisition-run restart INSIDE the night
        # (both columns must belong to this session; the session-start
        # transition to the previous night's last sample is not it).
        boundary = next(c for c in cols[1:]
                        if values[CSV_SEQUENCE_ROW, c] != values[CSV_SEQUENCE_ROW, c - 1])
        picks[sid] += [boundary - 1, boundary]
        for hour in DEEP_NIGHT_LOCAL_HOURS:
            best = min(cols, key=lambda c: abs(
                (parse_edt_label(labels[c]).hour * 60
                 + parse_edt_label(labels[c]).minute) - hour * 60))
            picks[sid].append(best)
    out = {}
    for sid in picks:
        seen, uniq = set(), []
        for c in sorted(picks[sid]):
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        out[sid] = uniq
    return out


def _with_neighbors(all_cols: list[int], targets: list[int], radius: int) -> list[int]:
    out = []
    valid = set(all_cols)
    for t in targets:
        for d in range(-radius, radius + 1):
            if t + d in valid:
                out.append(t + d)
    return out


def quality_of_column(raw_cells: list[str], n_metadata_rows: int = 4) -> dict:
    """Per-record quality marks derived from the verbatim published column."""
    values = np.asarray([float(v) if v != "NaN" else np.nan
                         for v in raw_cells[n_metadata_rows:]], dtype=float)
    wl = np.arange(280.0, 841.0)
    nan_mask = np.isnan(values)
    mask = []
    if nan_mask.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], nan_mask.view(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            mask.append([int(wl[a]), int(wl[b - 1])])
    finite = values[~nan_mask]
    return {
        "available_band_mask_nm": mask,
        "nan_cells": int(nan_mask.sum()),
        "zero_cells": int((values == 0).sum()),
        "negative_cells": int((values < 0).sum()),
        "min_value_w_m2_nm": float(finite.min()) if finite.size else None,
        "max_value_w_m2_nm": float(finite.max()) if finite.size else None,
        "notes": ["published values are already dark-corrected and calibrated upstream; "
                  "no re-scaling applied here",
                  "saturation is assessed from the raw spd maximum in records.json, not "
                  "from calibrated published values"],
    }


def build_packet(cache_dir: Path, out: Path, force: bool) -> dict:
    """Construct the packet from the verified cache. Refuses to overwrite."""
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"refusing to overwrite existing packet {out}; "
                         "pass --force only for a deliberate rebuild")
    if force:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tar_path = cache_dir / "dataraw.tar.gz"
    labels, values, raw_cols = load_full_csv(cache_dir / "CSSP_spectra.csv")
    if sha256_file(cache_dir / "CSSP_spectra.csv") != FULL_CSV_SHA256:
        raise ValueError("cached full CSV identity mismatch")
    sun_tab = read_mica_table(cache_dir / "CSSP_SunLocalZenith.txt")
    moon_tab = read_mica_table(cache_dir / "CSSP_MoonLocalZenith.txt")

    inventory = build_session_inventory(labels, values)
    picks = choose_session_samples(labels, values, inventory, sun_tab)
    columns = sorted(c for cols in picks.values() for c in cols)
    mapping = map_columns_to_members(labels, scan_raw_index(tar_path), columns)

    (out / "raw/mat").mkdir(parents=True, exist_ok=True)
    (out / "ephemeris").mkdir(parents=True, exist_ok=True)
    (out / "raw/weather").mkdir(parents=True, exist_ok=True)

    # Vendored upstream ephemeris tables (byte-identical to pinned upstream;
    # the same identities the matched-observation packet anchors by line).
    vendored_identity = {
        "CSSP_SunLocalZenith.txt":
            "12b3f2c55c936d9d5e275765b5b8c7e99d24c19e5ae0ae9a324440bb38becea6",
        "CSSP_MoonLocalZenith.txt":
            "4fdf4c2b2792fab0042aefb1f855fc3e92d4a89be8151ce3603f6c647ebd6578",
    }
    for name, expected in vendored_identity.items():
        src = cache_dir / name
        if sha256_file(src) != expected:
            raise ValueError(f"{name}: upstream identity mismatch")
        (out / "ephemeris" / name).write_bytes(src.read_bytes())

    # Weather responses for candidate-night windows (verbatim).
    for name in sorted((cache_dir / "weather").iterdir()):
        (out / "raw/weather" / name.name).write_bytes(name.read_bytes())
    # The primary night's frozen feasibility window rows join the response
    # set (byte-identical copies; the original stays referenced in sources).
    for src_rel, vendored_name in (
            ("validation/data/cssp_observation_feasibility/atmosphere/"
             "noaa-BFD-window.json", "frozen-noaa-BFD-window.json"),
            ("validation/data/cssp_observation_feasibility/atmosphere/"
             "iem-N38-window.json", "frozen-iem-N38-window.json")):
        src = ROOT / src_rel
        (out / "raw/weather" / vendored_name).write_bytes(src.read_bytes())
        if sha256_file(out / "raw/weather" / vendored_name) != sha256_file(src):
            raise ValueError(f"{vendored_name}: vendored copy differs from source")
    # Independent JPL AIRLESS spot checks for the final windows.
    (out / "raw/jpl").mkdir(parents=True, exist_ok=True)
    jpl_checks = []
    for name, record_id, label in (
            ("horizons-cssp-0703.json", "c1743", "03-Jul-2014 04:24:49"),
            ("horizons-cssp-0719.json", "c3074", "19-Jul-2014 04:39:17")):
        src = cache_dir / "jpl" / name
        if not src.is_file():
            continue
        (out / "raw/jpl" / name).write_bytes(src.read_bytes())
        jpl_elev = read_jpl_elevation(out / "raw/jpl" / name)
        mica_elev = mica_elevation(sun_tab, unix_of(label_to_utc_iso(label)))["elevation_deg"]
        jpl_checks.append({
            "record_id": record_id,
            "instant_utc": label_to_utc_iso(label),
            "jpl_elevation_deg": jpl_elev,
            "mica_elevation_deg": mica_elev,
            "diff_deg": round(jpl_elev - mica_elev, 6),
            "response_path": f"raw/jpl/{name}",
            "url": ("https://ssd.jpl.nasa.gov/api/horizons.api (COMMAND=10, "
                    "APPARENT=AIRLESS, geodetic CSSP coordinates)"),
            "agreement_bound_deg": 0.01,
            "note": "independent ephemeris source vs the vendored MICA table; the bound "
                    "is a cross-source agreement statement, not a fitted tolerance",
        })
    jpl_ok = all(abs(c["diff_deg"]) < c["agreement_bound_deg"] for c in jpl_checks)
    if jpl_checks and not jpl_ok:
        raise ValueError("JPL/MICA axis disagreement exceeds the stated bound")

    # Raw members: one sequential extraction pass.
    wanted = [mapping[c]["upstream_member"] for c in columns]
    extracted = extract_raw_members(tar_path, wanted, out / "raw/mat")

    records = []
    for col in columns:
        label = labels[col]
        axes = corrected_axes_for_columns(labels, [col], sun_tab, moon_tab)[col]
        member = extracted[mapping[col]["upstream_member"]]
        contract = read_mat_contract(out / member["packet_path"])
        invoke_utc = invoke_time_to_utc_iso(contract["invoke_time"]["value"])
        derived_label = parse_edt_label(label)
        if label_to_invoke_time(label) != contract["invoke_time"]["value"]:
            raise ValueError(f"column {col}: invokeTime/label mismatch")
        if contract["site_abbrev"]["value"] != "CSSP":
            raise ValueError(f"column {col}: raw site is {contract['site_abbrev']['value']}")
        n_exp = contract["n_exposures"]["value"]
        t_us = contract["integration_time_us"]["value"]
        set_us = contract["integration_time_set_us"]["value"]
        records.append({
            "record_id": f"c{col}",
            "source_column_zero_based": col,
            "session_id": next(s["session_id"] for s in inventory
                               if s["columns_zero_based"][0] <= col
                               <= s["columns_zero_based"][1]),
            "sequence_published": int(values[CSV_SEQUENCE_ROW, col]),
            "timestamp_label_edt": label,
            "invoke_time_utc": invoke_utc,
            "local_edt_hour": derived_label.hour,
            "published_axis_drl_wrong_site": {
                "sun_elevation_deg": float(values[CSV_SUN_ROW, col]),
                "moon_elevation_deg": float(values[CSV_MOON_ROW, col]),
                "moon_illumination_fraction": float(values[CSV_MOONFRAC_ROW, col]),
                "evidence": EVIDENCE_PUBLISHED,
                "note": "published axes describe DRL (upstream index bug); kept verbatim, "
                        "never used as a CSSP axis",
            },
            "corrected_axis_cssp": {
                **axes,
                "source": "vendored upstream MICA tables (UT1-as-UTC, 5-min brackets)",
                "evidence": EVIDENCE_DERIVED,
                "note": "never a brightness-fitted axis; independent JPL spot checks live "
                        "in the matched-observation packet",
            },
            "instrument": {
                key: contract[key] for key in (
                    "invoke_time", "site_abbrev", "sample_id", "serial_number",
                    "spectrometer_abbrev", "model_type", "script_file_name",
                    "dark_flag", "nonlinearity_correction_flag",
                    "electrical_dark_correction_flag", "integration_time_us",
                    "integration_time_set_us", "n_exposures", "n_pixels",
                    "board_temp_c", "wls_first_nm", "wls_last_nm", "wls_sha256",
                    "spd_shape", "spd_max_value")},
            "saturation": {
                "suspect": bool(contract["spd_max_value"]["value"] is not None
                                and contract["spd_max_value"]["value"] >= 62258.0),
                "assumed_ceiling_counts": 65535,
                "suspect_threshold_counts": 62258,
                "spd_max_over_assumed_ceiling": (
                    None if contract["spd_max_value"]["value"] is None
                    else round(contract["spd_max_value"]["value"] / 65535.0, 4)),
                "evidence": EVIDENCE_SCENARIO,
                "note": "USB2000+ is a 16-bit detector; the 65535 ceiling is an "
                        "assumption (not a published specification) so the flag is "
                        "scenario-level. The raw-measured spd_max_over_assumed_ceiling "
                        "is the dependable quantity; values within a few percent of "
                        "ceiling need exposure context before band ratios are trusted"},
            "integration_set_readback_consistent": (
                None if (t_us is None or set_us is None)
                else bool(t_us == set_us)),
            "timing": nominal_exposure_window(contract, invoke_utc),
            "raw": {"upstream_member": mapping[col]["upstream_member"],
                    **member},
            "quality": quality_of_column(raw_cols[col]),
        })

    write_json(out / "records.json", {
        "note": "derived records; original fields verbatim, unknowns are null+reason",
        "records": records})

    # spectra.csv: wavelength grid from the published wavelengths file, spectral
    # cells byte-identical to the official CSV.
    wl_lines = (cache_dir / "published-wavelengths.csv").read_text().split()
    if len(wl_lines) != 561:
        raise ValueError("unexpected published wavelength count")
    with (out / "spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wl_nm"] + [f"col{c}" for c in columns])
        for i, wl in enumerate(wl_lines):
            writer.writerow([wl] + [raw_cols[c][CSV_FIRST_SPECTRUM_ROW + i]
                                    for c in columns])

    # Selection: full inventory + frozen decisions (verdict text lands with
    # conditions; the decisions here reference the screening rules only).
    decisions = []
    for sess in inventory:
        sid = sess["session_id"]
        if sid == PRIMARY_SESSION:
            decisions.append({"session_id": sid, "decision": "selected",
                              "role": "primary",
                              "reason": "complete dusk-to-dawn sampling and the only "
                                        "night with all three target twilight depths "
                                        "(-6/-9/-12 deg); moon below horizon at the dawn "
                                        "targets, while the dawn window shows valley fog "
                                        "at KN38 and BR/BKN at KBFD (blocked for absolute "
                                        "use, kept for spectral/relative use)"})
        elif sid == SECONDARY_SESSION:
            decisions.append({"session_id": sid, "decision": "selected",
                              "role": "secondary",
                              "reason": "its dusk -6 deg window is the cleanest in the "
                                        "dataset (both neighbour stations CLR, moon below "
                                        "horizon); its dawn targets are moonlit (+43..+50 "
                                        "deg) and retained only as a documented moonlit "
                                        "sample"})
        elif sess["signal_kind"] == "twilight-or-night":
            decisions.append({"session_id": sid, "decision": "rejected",
                              "reason": "twilight-level session kept out of the packet to "
                                        "bound cost; inventory retained for audit"})
        else:
            decisions.append({"session_id": sid, "decision": "rejected",
                              "reason": "daytime-level signal; not a twilight comparison "
                                        "candidate"})
    write_json(out / "selection.json", {
        "rule": "sessions are gap-split (900 s); candidates need twilight-level signal and "
                "independently obtainable weather/moon evidence; no sky-model residual is "
                "read at any point",
        "sessions": inventory,
        "decisions": decisions})
    write_json(out / "conditions.json",
               build_conditions(out, labels, values, inventory, decisions, sun_tab,
                                moon_tab, selected_by_session=picks,
                                independent_axis_checks=jpl_checks))

    write_json(out / "operator.json", operator_contract())
    diagnostics = compute_diagnostics(out, labels, values, raw_cols)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "sources.json", build_sources(out, cache_dir, columns))

    # README and manifest last.
    write_readme(out, columns)
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "inputs.sha256.json":
            manifest[str(path.relative_to(out))] = sha256_file(path)
    write_json(out / "inputs.sha256.json", manifest)

    report = verify_packet(out)
    if not report["ok"]:
        raise ValueError(f"freshly built packet failed verification: "
                         f"{report['problems'][:10]}")
    return {"records": len(records), "sessions": len(inventory),
            "raw_members": len(columns), "verify": report}


# ---------------------------------------------------------------------------
# Operator contract
# ---------------------------------------------------------------------------

REFERENCE_BANDS_NM = {
    "violet": [380.0, 440.0],
    "green": [500.0, 570.0],
    "red": [600.0, 700.0],
    "visible_all": [380.0, 780.0],
}


def operator_contract() -> dict:
    """The minimal mathematical observation operator (plan §3.3), as data.

    ``y_ip`` integrates time, wavelength and the upper hemisphere with the
    white-board/horizon response; ``P`` is the (verified minimal) publishing
    pipeline. Every element carries an evidence level; idealisations are
    marked as approximations, never as measured responses.
    """
    return {
        "formula_name": "hemispheric-spectral-exposure",
        "formula": ("y_ip = P_p{ integral dt w_i(t) integral dlambda R_p(lambda) "
                    "integral_Omega+ dOmega A(lambda,Omega) H(Omega) cos(theta) "
                    "[L_tw(lambda,Omega,t) + L_bg(lambda,Omega,t)] } + r_dark_ip"),
        "quantity": "downwelling hemispheric spectral irradiance via a horizontal "
                    "diffuser (white board), NOT zenith radiance",
        "definitions": {
            "w_i": {"text": "exposure-weighting boxcars over the record's exposure window; "
                            "equal-length repeats carry a_ij = 1/N",
                    "evidence": EVIDENCE_DERIVED,
                    "unknown": "per-exposure gaps/latencies; see records.json timing"},
            "R_p": {"text": "spectral channel kernel; published 1 nm grid suggests but "
                            "does not prove a ~1 nm triangular response; the slit function "
                            "was not measured",
                    "evidence": EVIDENCE_UNKNOWN,
                    "reason": "line-spread function not published; band integrals on the "
                              "published grid assume unit-transmission channels"},
            "A": {"text": "white-board angular/horizontal correction relative to an ideal "
                          "cosine receiver",
                  "evidence": EVIDENCE_UNKNOWN,
                  "reason": "no measured angular-response curve is public; A=1 is the "
                            "idealisation, not a measurement"},
            "H": {"text": "horizon visibility (terrain/obstruction mask above the board)",
                  "evidence": EVIDENCE_UNKNOWN,
                  "reason": "site horizon not characterised; H=1 is the idealisation"},
            "P": {"text": "upstream publishing steps (dark subtraction, wavelength mapping, "
                          "sensitivity calibration, resampling to 1 nm grid, exposure "
                          "normalisation); replayed minimally in the replay command; no "
                          "re-application of calibration factors to published spectra",
                  "evidence": EVIDENCE_DERIVED},
            "L_bg": {"text": "astronomical + artificial night background; distinct from "
                             "instrument dark residual r_dark",
                     "evidence": EVIDENCE_DERIVED},
            "r_dark": {"text": "residual instrument dark signal after upstream dark "
                               "subtraction",
                       "evidence": EVIDENCE_UNKNOWN,
                       "reason": "dark handling is upstream (GetDarkSpectrum by "
                                 "integration time + board temperature); residual not "
                                 "independently measured here"},
        },
        "units": {
            "published_spectra": "W m-2 nm-1 (spectral density, as published)",
            "band_integrals": "W m-2 (per band, trapezoidal on the published grid)",
            "radiance_terms": "W m-2 sr-1 nm-1",
            "cos_theta": "applied once, inside the hemisphere integral",
        },
        "reference_bands_nm": REFERENCE_BANDS_NM,
        "forbidden_comparisons": [
            "hemispheric irradiance vs zenith radiance (different quantity)",
            "published spectra vs model spectra without integrating model radiance over "
            "the hemisphere and the exposure window",
            "treating night-spectrum medians as a measured background subtraction",
        ],
        "scenario_policy": "response/horizon/background scenarios demonstrate dependence; "
                           "they are not confidence intervals",
    }


def read_jpl_elevation(path: Path) -> float:
    """Extract the solar elevation from one archived Horizons AIRLESS CSV
    response (first ephemeris row, second-to-last CSV field, as verify.py in
    the matched-observation packet does)."""
    body = read_json(path)["result"]
    line = body.split("$$SOE")[1].strip().splitlines()[0]
    return float([f.strip() for f in line.split(",")][-2])


# ---------------------------------------------------------------------------
# Conditions: independent (non-model) evidence per selected night
# ---------------------------------------------------------------------------

def utc_dates_between(start_iso: str, end_iso: str) -> list[str]:
    d0 = datetime.fromisoformat(start_iso).date()
    d1 = datetime.fromisoformat(end_iso).date()
    days = (d1 - d0).days
    return [(d0 + timedelta(days=k)).isoformat() for k in range(days + 1)]


def _parse_response_stamp(stamp: str) -> float | None:
    """Parse IEM (``2014-07-03T08:15Z``), NOAA (``2014-07-03T08:53:00``) and
    frozen-window (``2014-07-03 08:15``) stamps as UTC."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def weather_window_rows(response_path: Path, start_iso: str, end_iso: str) -> list[dict]:
    """Extract the observation rows inside the UTC window from raw responses.

    Handles the three shapes actually archived: IEM obhistory
    (``{"data": [...]}`` with ``utc_valid``/``raw``), NOAA Global Hourly
    (list with ``DATE``/``REM``) and the frozen feasibility window subset
    (list with ``valid``/``metar``). Rows whose stamp cannot be parsed are
    kept with a null utc so missing times stay visible instead of being
    silently dropped.
    """
    payload = read_json(response_path)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    lo = unix_of(start_iso)
    hi = unix_of(end_iso)
    out = []
    for row in rows or []:
        stamp = row.get("utc_valid") or row.get("DATE") or row.get("valid") or ""
        unix = _parse_response_stamp(stamp) if stamp.startswith("2014") else None
        if unix is not None and not (lo <= unix <= hi):
            continue
        out.append({
            "utc": stamp if stamp.startswith("2014") else None,
            "visibility_mi_or_m": row.get("vsby", row.get("VIS")),
            "cloud_cover": (row.get("skyc1") or "").strip() or None,
            "cloud_height_ft": row.get("skyl1"),
            "wx_codes": row.get("wxcodes", row.get("REM") or row.get("metar")),
            "raw": (row.get("raw") or row.get("metar") or "")[:120] or None,
        })
    return out


def summarize_weather_rows(rows: list[dict], source: str) -> dict:
    """Classify window rows into explicit, data-referenced categories.

    IEM rows carry structured vsby/skyc1/wxcodes; NOAA rows carry the METAR
    text in ``wx_codes`` (the REM field). Everything stays verbatim-quoted so
    the verdict can be traced back to the archived responses.
    """
    fog_rows, low_opaque_rows, clear_rows, other_rows = [], [], [], []
    cloud_re = None
    if source == "KBFD":
        import re
        cloud_re = re.compile(r"\b(CLR|FEW\d+|SCT\d+|BKN\d+|OVC\d+|VV\d*)\b")
    for row in rows:
        text = str(row.get("wx_codes") or "")
        cloud = row.get("cloud_cover")
        vis = row.get("visibility_mi_or_m")
        try:
            vis_value = float(vis) if vis is not None and str(vis).replace(".", "").isdigit() else None
        except ValueError:
            vis_value = None
        if cloud_re is not None:
            tokens = cloud_re.findall(text)
            cloud = tokens[0] if tokens else None
            vis_value = None
            if "VIS" in row and isinstance(row.get("VIS"), str):
                head = row["VIS"].split(",")[0]
                vis_value = float(head) / 1609.34 if head.isdigit() else None
        label = str(row.get("utc") or "?")
        entry = {"utc": label, "visibility": vis, "cloud": cloud, "wx": text[:40]}
        if "FG" in text or (cloud or "").upper() == "FG":
            fog_rows.append(entry)
        elif (cloud or "").upper().startswith("VV") or (
                cloud in ("OVC", "BKN") and isinstance(row.get("cloud_height_ft"),
                                                       (int, float))
                and row["cloud_height_ft"] <= 200):
            low_opaque_rows.append(entry)
        elif cloud == "CLR" and "BR" not in text:
            clear_rows.append(entry)
        else:
            other_rows.append(entry)
    return {"n_rows": len(rows), "fog": fog_rows, "low_opaque": low_opaque_rows,
            "clear": clear_rows, "other": other_rows}


def weather_rows_for_window(response_paths: list[Path], start_iso: str,
                            end_iso: str) -> list[dict]:
    """Merge window rows across daily response files (IEM days are not
    UTC-midnight aligned - the archived 7/2 file runs 04:15Z..03:55Z), dedupe
    by stamp and keep chronological order."""
    merged: dict[str, dict] = {}
    for path in response_paths:
        if not path.is_file():
            continue
        for row in weather_window_rows(path, start_iso, end_iso):
            key = str(row.get("utc") or id(row))
            merged.setdefault(key, row)
    rows = [r for r in merged.values() if r.get("utc")]
    rows.sort(key=lambda r: _parse_response_stamp(r["utc"]) or 0.0)
    return rows


def station_state(summary: dict, context_summary: dict) -> str:
    """Classify one station's window, falling back to the +-60 min context
    rows when no observation falls inside the tight window (hourly stations).
    The label names which rows were used so the verdict stays traceable."""
    effective, from_context = summary, False
    if not summary["n_rows"] and context_summary["n_rows"]:
        effective, from_context = context_summary, True
    if not effective["n_rows"]:
        return "no observation archived in window"
    if effective["fog"] or effective["low_opaque"]:
        state = "fog/low-cloud"
    elif effective["clear"] and not effective["other"]:
        state = "CLR"
    else:
        state = "partial/high-cloud"
    return f"{state} (context rows)" if from_context else state


def session_verdict(side_windows: dict) -> str:
    """Compose the per-session verdict text from the per-side windows."""
    parts = []
    for side in ("dusk", "dawn"):
        window = side_windows.get(side)
        if window is None:
            continue
        moon_state = "moon-free" if window["moon_free"] else "moonlit"
        states = [f"{key}: {station_state(s['summary'], s['context_summary'])}"
                  for key, s in window["weather"].items()]
        parts.append(f"{side} [{moon_state}; {', '.join(states)}]")
    return "; ".join(parts)


def build_conditions(out: Path, labels: list[str], values: np.ndarray,
                     inventory: list[dict], decisions: list[dict],
                     sun_tab: dict, moon_tab: dict,
                     selected_by_session: dict[str, list[int]] | None = None,
                     independent_axis_checks: list[dict] | None = None) -> dict:
    """Independent conditions per selected session; verdicts stay explicit.

    Each side (dusk/dawn) of a selected night gets its own weather window,
    moon geometry at the actual target instants and a data-referenced
    verdict. No model quantity is read here; neighbour-station rows are kept
    verbatim so every verdict clause traces back to an archived response.
    """
    conditions: dict[str, dict] = {}
    weather_files = sorted(p.name for p in (out / "raw/weather").iterdir())
    for decision in decisions:
        if decision["decision"] != "selected":
            continue
        sess = next(s for s in inventory if s["session_id"] == decision["session_id"])
        start, end = sess["start_utc"], sess["end_utc"]
        a, b = sess["columns_zero_based"]
        cols = list(range(a, b + 1))
        if selected_by_session and decision["session_id"] in selected_by_session:
            cols = sorted(selected_by_session[decision["session_id"]])
        moon = [mica_elevation(moon_tab, unix_of(label_to_utc_iso(labels[c])))
                ["elevation_deg"] for c in cols]
        fractions = values[CSV_MOONFRAC_ROW, a:b + 1]
        dates = utc_dates_between(start, end)
        date_keys = [d.replace("-", "") for d in dates]
        weather_paths = {
            "KN38": [out / "raw/weather" / f for f in weather_files
                     if (f.startswith("iem-N38-obhistory") or f.startswith("frozen-iem"))
                     and (any(k in f for k in date_keys) or f.startswith("frozen"))],
            "KBFD": [out / "raw/weather" / f for f in weather_files
                     if (f.startswith("noaa-BFD") or f.startswith("frozen-noaa"))
                     and (any(k in f for k in date_keys) or f.startswith("frozen"))],
        }
        station_meta = {
            "KN38": {"station": KN38,
                     "distance_km": round(haversine_km(CSSP["lat_deg"], CSSP["lon_deg"],
                                                       KN38["lat_deg"], KN38["lon_deg"]), 2),
                     "elevation_diff_m": round(KN38["elevation_m"] - CSSP["elevation_m"], 2)},
            "KBFD": {"station": KBFD,
                     "distance_km": round(haversine_km(CSSP["lat_deg"], CSSP["lon_deg"],
                                                       KBFD["lat_deg"], KBFD["lon_deg"]), 2),
                     "elevation_diff_m": round(KBFD["elevation_m"] - CSSP["elevation_m"], 2)},
        }
        # Per-side target windows: the selected records split into dusk/dawn
        # halves; the weather window covers the twilight-depth records only
        # (sun above -13 deg) so deep-night samples do not dilute it, and a
        # +-60 min context window keeps hourly stations (KBFD) represented
        # when no observation falls inside the tight window.
        side_windows = {}
        for side, hour_test in (("dusk", lambda h: h >= 12), ("dawn", lambda h: h < 12)):
            side_cols = [c for c in cols if hour_test(parse_edt_label(labels[c]).hour)]
            if not side_cols:
                continue
            twilight_cols = [c for c in side_cols
                             if mica_elevation(sun_tab, unix_of(label_to_utc_iso(labels[c])))
                             ["elevation_deg"] > -13.0]
            window_cols = twilight_cols or side_cols
            times = [parse_edt_label(labels[c]) for c in window_cols]
            w_start = (min(times) - timedelta(minutes=5)).astimezone(timezone.utc)
            w_end = (max(times) + timedelta(minutes=10)).astimezone(timezone.utc)
            start_iso, end_iso = w_start.isoformat(), w_end.isoformat()
            ctx_start = (min(times) - timedelta(minutes=60)).astimezone(timezone.utc)
            ctx_end = (max(times) + timedelta(minutes=60)).astimezone(timezone.utc)
            ctx_iso = [ctx_start.isoformat(), ctx_end.isoformat()]
            moon_at = [mica_elevation(moon_tab, unix_of(label_to_utc_iso(labels[c])))
                       ["elevation_deg"] for c in window_cols]
            summaries = {}
            for station_key, paths in weather_paths.items():
                rows = weather_rows_for_window(paths, start_iso, end_iso)
                context_rows = weather_rows_for_window(paths, ctx_iso[0], ctx_iso[1])
                summaries[station_key] = {
                    **station_meta[station_key],
                    "window_utc": [start_iso, end_iso],
                    "summary": summarize_weather_rows(rows, station_key),
                    "context_window_utc": ctx_iso,
                    "context_summary": summarize_weather_rows(context_rows, station_key),
                }
            moon_free = all(m < 0.0 for m in moon_at)
            any_fog = any(s["summary"]["fog"] or s["summary"]["low_opaque"]
                          or s["summary"]["other"] and not s["summary"]["clear"]
                          for s in summaries.values())
            all_clear = all(s["summary"]["clear"] and not s["summary"]["fog"]
                            and not s["summary"]["low_opaque"]
                            and not s["summary"]["other"]
                            for s in summaries.values())
            if moon_free and all_clear:
                verdict = ("cleanest-available: moon below horizon at all target instants "
                           "and both neighbour stations report CLR through the window")
            elif moon_free and any_fog:
                verdict = ("conditional: moon below horizon, but fog or low cloud is "
                           "reported at neighbour stations inside the window; CSSP itself "
                           "(710 m plateau) has no in-situ evidence, so absolute use is "
                           "blocked and spectral/relative use carries this caveat")
            elif not moon_free:
                verdict = ("moonlit: moon above the horizon at target instants "
                           f"(range {min(moon_at):.1f}..{max(moon_at):.1f} deg); "
                           "hemispheric irradiance includes moonlight, so clean solar "
                           "twilight comparisons are blocked; retained as a moonlit-sky "
                           "sample")
            else:
                verdict = "mixed: neighbour reports split between clear and cloud/fog rows"
            side_windows[side] = {
                "columns_zero_based": [min(side_cols), max(side_cols)],
                "window_utc": [start_iso, end_iso],
                "moon_elevation_deg_at_targets": [round(m, 3) for m in moon_at],
                "moon_free": moon_free,
                "weather": summaries,
                "verdict": verdict,
            }
        conditions[decision["session_id"]] = {
            "utc_window": {"start": start, "end": end},
            "local_dates_utc_covered": dates,
            "moon": {
                "elevation_deg_min": float(np.min(moon)),
                "elevation_deg_max": float(np.max(moon)),
                "published_illumination_fraction_min": float(np.min(fractions)),
                "published_illumination_fraction_max": float(np.max(fractions)),
                "axis_source": "vendored upstream MICA lunar table",
            },
            "target_windows": side_windows,
            "aod": {
                "queried": [f for f in weather_files if f.startswith("aeronet")
                            and any(k in f for k in date_keys)],
                "note": "AERONET direct-sun products are daytime; any return is a far-site "
                        "column, not an in-situ twilight value",
            },
            "verdict": session_verdict(side_windows),
            "verdict_note": "derived only from these archived responses and the MICA lunar "
                            "table; no sky-model residual is read anywhere",
        }
    # The primary night's dawn window additionally carries frozen feasibility
    # evidence (its original bounded probes); referenced, not duplicated.
    if PRIMARY_SESSION in conditions:
        conditions[PRIMARY_SESSION]["frozen_feasibility_refs"] = [
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "noaa-BFD-window.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "iem-N38-window.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "aeronet-requests.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "nearest-sites.json"),
        ]
    return {
        "note": "independent conditions only; geometry from vendored MICA, weather verbatim "
                "public responses, AOD availability as returned",
        "independent_axis_checks": independent_axis_checks or [],
        "conditions": conditions,
    }


def refresh_diagnostics(packet: Path) -> dict:
    """Recompute diagnostics.json from the frozen packet and refresh manifest."""
    records = read_json(packet / "records.json")["records"]
    labels = {r["source_column_zero_based"]: r["timestamp_label_edt"]
              for r in records}
    diagnostics = compute_diagnostics(packet, labels, None, None)
    write_json(packet / "diagnostics.json", diagnostics)
    manifest = read_json(packet / "inputs.sha256.json")
    manifest["diagnostics.json"] = sha256_file(packet / "diagnostics.json")
    write_json(packet / "inputs.sha256.json", manifest)
    return {"ok": True, "diagnostics_files": ["diagnostics.json"],
            "scenario_not_error_bar": True}


# ---------------------------------------------------------------------------
# Sensitivity diagnostics (scenario tables, not error bars)
# ---------------------------------------------------------------------------

def band_energy(wl: np.ndarray, spectrum: np.ndarray, band: tuple[float, float]) -> float:
    """Trapezoidal band integral of one published spectrum; NaN-aware.

    Zero or negative values are not silently logged or clipped: the band
    integral integrates the published values as they are (zeros included),
    which is why callers must treat near-floor bands as flagged, not clean.
    """
    lo, hi = band
    sel = (wl >= lo) & (wl <= hi) & ~np.isnan(spectrum)
    if not sel.any():
        return float("nan")
    return float(np.trapezoid(spectrum[sel], wl[sel]))


def compute_diagnostics(packet: Path, labels: list[str], values: np.ndarray,
                        raw_cols: dict[int, list[str]]) -> dict:
    """Scenario diagnostics over the frozen records (plan §3.4).

    Everything here is a scenario: measured adjacent-spectrum change rates
    bound what a timing offset could do; horizon/background scenarios show
    the magnitude of unknown responses. No scenario is fitted back into any
    parameter, and none is presented as an error bar.
    """
    wl = np.arange(280.0, 841.0)
    records = read_json(packet / "records.json")["records"]
    by_col = {r["source_column_zero_based"]: r for r in records}
    sun_tab = read_mica_table(packet / "ephemeris/CSSP_SunLocalZenith.txt")

    spectra = load_packet_spectra(packet)
    data = spectra["data"]
    col_index = {c: i for i, c in enumerate(spectra["columns"])}

    timing_offsets_s = [1, 5, 10, 30, 60]
    adjacent = {}
    for record in records:
        col = record["source_column_zero_based"]
        if record["timing"]["nominal_total_exposure_s"] is None:
            continue
        neighbours = [c for c in (col - 1, col + 1) if c in col_index]
        if not neighbours:
            continue
        rows = []
        for nb in sorted(neighbours):
            per_band = {name: band_energy(wl, data[:, col_index[col]], tuple(band))
                        for name, band in REFERENCE_BANDS_NM.items()}
            nb_band = {name: band_energy(wl, data[:, col_index[nb]], tuple(band))
                       for name, band in REFERENCE_BANDS_NM.items()}
            gap_minutes = abs(unix_of(label_to_utc_iso(labels[nb]))
                              - unix_of(label_to_utc_iso(labels[col]))) / 60.0
            rows.append({
                "neighbour_column": nb,
                "gap_minutes": round(gap_minutes, 3),
                "relative_change_per_band": {
                    name: (nb_band[name] - per_band[name]) / per_band[name]
                    if per_band[name] and not np.isnan(per_band[name]) else None
                    for name in REFERENCE_BANDS_NM},
            })
        sun = mica_elevation(sun_tab, unix_of(record["invoke_time_utc"]))["elevation_deg"] \
            if record["corrected_axis_cssp"]["sun_elevation_deg"] is not None else None
        adjacent[record["record_id"]] = {
            "corrected_sun_elevation_deg": sun,
            "nominal_total_exposure_s": record["timing"]["nominal_total_exposure_s"],
            "neighbours": rows,
            "timing_offset_scenario_per_band": {
                str(offset): {
                    name: (rows[0]["relative_change_per_band"][name]
                           * offset / 60.0 / max(rows[0]["gap_minutes"], 1e-9))
                    if rows[0]["relative_change_per_band"][name] is not None else None
                    for name in REFERENCE_BANDS_NM}
                for offset in timing_offsets_s},
        }

    # Night-floor context: deep-night columns vs twilight targets, same night,
    # same device (per-band ratio; common gain may cancel, nothing else assumed).
    night_floor = {}
    for session_id in sorted({r["session_id"] for r in records}):
        sess_records = [r for r in records if r["session_id"] == session_id]
        deep = [r for r in sess_records
                if r["corrected_axis_cssp"]["sun_elevation_deg"] is not None
                and r["corrected_axis_cssp"]["sun_elevation_deg"] < -15.0]
        twilight = [r for r in sess_records
                    if r["corrected_axis_cssp"]["sun_elevation_deg"] is not None
                    and r["corrected_axis_cssp"]["sun_elevation_deg"] > -13.0]
        if not deep or not twilight:
            continue
        deep_col = min(deep, key=lambda r: r["source_column_zero_based"])
        tw_col = max(twilight, key=lambda r: r["corrected_axis_cssp"]["sun_elevation_deg"])
        ratios = {}
        for name, band in REFERENCE_BANDS_NM.items():
            e_deep = band_energy(wl, data[:, col_index[deep_col["source_column_zero_based"]]],
                                 tuple(band))
            e_tw = band_energy(wl, data[:, col_index[tw_col["source_column_zero_based"]]],
                               tuple(band))
            ratios[name] = {"deep_night_column": deep_col["record_id"],
                            "twilight_column": tw_col["record_id"],
                            "deep_over_twilight": (e_deep / e_tw
                                                   if e_tw and not np.isnan(e_deep) else None)}
        night_floor[session_id] = ratios

    horizon = {
        "scenario": "isotropic radiance L, horizon depression angle h: "
                    "E(h) = pi L cos^2(h) against E(0) = pi L",
        "evidence": EVIDENCE_SCENARIO,
        "depression_deg": [0.0, 1.0, 2.0, 5.0, 10.0],
        "E_over_E0": [float(np.cos(np.radians(h)) ** 2)
                      for h in (0.0, 1.0, 2.0, 5.0, 10.0)],
        "note": "flat-horizon geometric factor only; the real site horizon and white-board "
                "angular response are unmeasured, so this bounds geometry, not response",
    }
    return {
        "generated_by": "tools/cssp_observation_inputs.py compute_diagnostics",
        "scenario_not_error_bar": True,
        "timing_offsets_s": timing_offsets_s,
        "adjacent_change": adjacent,
        "night_floor_ratios": night_floor,
        "horizon_scenarios": horizon,
        "notes": [
            "timing offsets are evaluated with the measured adjacent-spectrum change rate "
            "over the actual sampling gap; linear extrapolation, scenario only",
            "night-floor ratios involve common device gain that may cancel; background "
            "colour structure and time variation do not cancel",
            "no background subtraction is applied anywhere in this packet",
        ],
    }


# ---------------------------------------------------------------------------
# sources.json and README
# ---------------------------------------------------------------------------

def build_sources(out: Path, cache_dir: Path, columns: list[int]) -> dict:
    """Source identities: whole-file, member and derived claims stay distinct."""
    fetched = read_json(cache_dir / "fetch-provenance.json") if \
        (cache_dir / "fetch-provenance.json").is_file() else []
    raw_members = []
    for name in sorted((out / "raw/mat").iterdir()):
        raw_members.append({"packet_path": f"raw/mat/{name.name}",
                            "bytes": name.stat().st_size,
                            "sha256": sha256_file(name)})
    weather_files = []
    for name in sorted((out / "raw/weather").iterdir()):
        weather_files.append({"packet_path": f"raw/weather/{name.name}",
                              "bytes": name.stat().st_size,
                              "sha256": sha256_file(name)})
    return {
        "site": {**CSSP,
                 "coordinate_validation": "independent JPL AIRLESS sample and full MICA "
                                          "replay live in validation/data/"
                                          "matched_twilight_observations (see external_refs)"},
        "author_repo_commit": AUTHOR_REPO_COMMIT,
        "sources": [
            {"id": "full-csv", "kind": "whole-file", "role": "published spectra + "
             "published (wrong-site) axes + sequence + moon fraction",
             "url": FULL_CSV_URL,
             "sha256": FULL_CSV_SHA256, "bytes": FULL_CSV_BYTES,
             "evidence": EVIDENCE_PUBLISHED},
            {"id": "published-wavelengths", "kind": "whole-file",
             "role": "published 1 nm wavelength grid", "url": WAVELENGTHS_URL,
             "sha256": sha256_file(cache_dir / "published-wavelengths.csv"),
             "bytes": (cache_dir / "published-wavelengths.csv").stat().st_size,
             "evidence": EVIDENCE_PUBLISHED},
            {"id": "mica-cssp-sun", "kind": "whole-file", "role": "corrected CSSP solar axis",
             "url": EPHEMERIS_URL.format("CSSP_SunLocalZenith"),
             "sha256": sha256_file(cache_dir / "CSSP_SunLocalZenith.txt"),
             "bytes": (cache_dir / "CSSP_SunLocalZenith.txt").stat().st_size,
             "evidence": EVIDENCE_PUBLISHED},
            {"id": "mica-cssp-moon", "kind": "whole-file", "role": "corrected CSSP lunar axis",
             "url": EPHEMERIS_URL.format("CSSP_MoonLocalZenith"),
             "sha256": sha256_file(cache_dir / "CSSP_MoonLocalZenith.txt"),
             "bytes": (cache_dir / "CSSP_MoonLocalZenith.txt").stat().st_size,
             "evidence": EVIDENCE_PUBLISHED},
            {"id": "raw-archive", "kind": "whole-archive",
             "url": RAW_ARCHIVE_URL, "final_url": RAW_ARCHIVE_FINAL_URL,
             "total_bytes": RAW_ARCHIVE_TOTAL_BYTES,
             "sha256_full_archive": RAW_ARCHIVE_FULL_SHA256,
             "note": "single sequential read; full-archive sha256 recorded once, from the "
                     "complete file. The historical 1 MiB prefix hash "
                     f"({RAW_ARCHIVE_PREFIX_1MIB_SHA256}) in the feasibility packet remains "
                     "prefix-only evidence",
             "evidence": EVIDENCE_PUBLISHED},
        ],
        "vendored_raw_members": {"kind": "member", "count": len(raw_members),
                                 "items": raw_members,
                                 "note": "byte-identical members of the raw archive; "
                                         "member hashes do not certify the whole archive"},
        "vendored_weather": {"kind": "response", "items": weather_files,
                             "note": "verbatim public responses (IEM representation of "
                                     "GHCNh, NOAA Global Hourly, AERONET); failed probes "
                                     "kept as evidence"},
        "external_refs": [
            external_ref("validation/data/matched_twilight_observations/verify.py"),
            external_ref("validation/data/matched_twilight_observations/spectral/"
                         "CSSP-sample-probe.json"),
            external_ref("validation/data/matched_twilight_observations/spectral/"
                         "knot-extractions.json"),
            external_ref("validation/data/matched_twilight_observations/spectral/"
                         "selected-spectra.csv"),
            external_ref("validation/data/cssp_observation_feasibility/timing/"
                         "OO_MeasureAndSave.m"),
            external_ref("validation/data/cssp_observation_feasibility/timing/"
                         "OO_IntervalMeasurements.m"),
            external_ref("validation/data/cssp_observation_feasibility/timing/"
                         "control-raw-metadata.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "noaa-BFD-window.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "iem-N38-window.json"),
            external_ref("validation/data/cssp_observation_feasibility/atmosphere/"
                         "nearest-sites.json"),
            external_ref("docs/twilight/MATCHED_OBSERVATIONS.md"),
        ],
        "columns_in_packet": columns,
        "fetch_provenance": fetched,
    }


def write_readme(out: Path, columns: list[int]) -> None:
    from_records = read_json(out / "records.json")["records"]
    serials = sorted({r["instrument"]["serial_number"]["value"] for r in from_records})
    exposures = sorted({r["instrument"]["n_exposures"]["value"] for r in from_records})
    integration = sorted({r["instrument"]["integration_time_us"]["value"]
                          for r in from_records})
    readme = f"""# CSSP observation inputs

Frozen validation input for the public CSSP twilight hemispheric-irradiance
dataset (Spitschan et al. 2016, doi:10.1038/srep26756). Scope: source identity,
raw-acquisition contract, independent night screening and the minimal
observation operator. This packet does not claim an absolute-calibration
validation of the sky model; scientific interpretation lives in
[CSSP_OBSERVATION_INPUTS](../../../docs/twilight/CSSP_OBSERVATION_INPUTS.md)
and [MATCHED_OBSERVATIONS §4](../../../docs/twilight/MATCHED_OBSERVATIONS.md).

## Contents

- `sources.json`: identities for whole files, the raw archive (single
  sequential read), vendored members and verbatim public responses; distinct
  `kind` per claim level. `external_refs` pins the older frozen packets this
  packet builds on (relative paths + SHA).
- `records.json` + `spectra.csv`: per selected column - verbatim timestamp
  label and published (wrong-site) axes, corrected CSSP axes recomputed from
  the vendored upstream MICA tables, the raw `.mat` acquisition contract
  (serial, integration time, repeats, temperatures), the nominal exposure
  window with explicit unknowns (clock offset, setup latency, gaps, driver
  caching: null + reason), and per-record quality/band masks. Spectral cells
  are byte-identical excerpts of the official CSV.
- `raw/mat/`: unmodified archive members for {len(columns)} records
  (serials {serials}; repeats {exposures}; integration times {integration} us).
- `ephemeris/`: full upstream MICA solar/lunar tables (UT1 as UTC, 5-minute
  cadence) - the corrected-axis source for every night in the dataset span.
- `raw/weather/`: verbatim IEM/NOAA/AERONET responses for candidate-night
  windows. Failed probes are kept as evidence.
- `selection.json`: the full session inventory of all 3192 published columns
  plus frozen selection decisions and reasons.
- `conditions.json`: moon geometry and independent weather/AOD coverage per
  selected night (what was queried, what returned, and what stays missing).
- `operator.json`: the mathematical observation operator with evidence
  levels; `diagnostics.json`: timing/response/background scenario tables.

## Commands

From the worktree root (offline paths never touch the network):

```bash
uv run python -I tools/cssp_observation_inputs.py verify \\
  --packet validation/data/cssp_observation_inputs
uv run python -I tools/cssp_observation_inputs.py replay \\
  --packet validation/data/cssp_observation_inputs
uv run python -I tools/cssp_observation_inputs.py sensitivity \\
  --packet validation/data/cssp_observation_inputs
```

Acquisition (writes only to the cache directory, never to the packet):

```bash
uv run python -I tools/cssp_observation_inputs.py fetch \\
  --cache-dir <new-empty-dir> --raw-archive
uv run python -I tools/cssp_observation_inputs.py build \\
  --source-dir <cache-dir> --out <new-output-dir>
```

Optional full-source check (complete official CSV + byte-level excerpt
comparison):

```bash
uv run python -I tools/cssp_observation_inputs.py verify \\
  --packet validation/data/cssp_observation_inputs --source-dir <dir with CSSP_spectra.csv>
```

## Boundaries

- The nominal exposure window is a scenario; no measured exposure midpoint
  exists (see records.json `timing.unknown`).
- Published axes describe DRL (upstream index bug) and are retained verbatim
  for provenance; corrected axes come from the vendored MICA tables.
- Weather responses are neighbour stations; distances/elevation gaps are
  recorded and no in-situ column quantities are derived from them.
- Scenario ranges in `diagnostics.json` are demonstrations of dependence,
  not confidence intervals.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="offline packet verification (default)")
    p_verify.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    p_verify.add_argument("--source-dir", type=Path, default=None,
                          help="folder with the full official CSSP_spectra.csv")

    p_fetch = sub.add_parser("fetch", help="acquire public sources into a cache")
    p_fetch.add_argument("--cache-dir", type=Path, required=True)
    p_fetch.add_argument("--raw-archive", action="store_true",
                         help="also download the full raw archive (one sequential read)")

    p_build = sub.add_parser("build", help="construct the packet from cache")
    p_build.add_argument("--source-dir", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--force", action="store_true")

    p_replay = sub.add_parser("replay", help="raw->published processing replay")
    p_replay.add_argument("--packet", type=Path, default=DEFAULT_PACKET)

    p_sens = sub.add_parser("sensitivity", help="recompute scenario diagnostics")
    p_sens.add_argument("--packet", type=Path, default=DEFAULT_PACKET)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = verify_packet(args.packet, args.source_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1
    if args.command == "fetch":
        result = fetch_sources(args.cache_dir, args.raw_archive)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "build":
        result = build_packet(args.source_dir, args.out, args.force)
        print(json.dumps(result["verify"], indent=2, ensure_ascii=False))
        return 0 if result["verify"]["ok"] else 1
    if args.command == "sensitivity":
        report = refresh_diagnostics(args.packet)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    if args.command == "replay":
        raise NotImplementedError("replay lands with the calibration step (plan M3)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
