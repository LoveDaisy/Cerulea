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
import re
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
# Generated calibration structure for meter 'a' (CSSP records have no
# wavelength above 1000 nm, so GetWhichMeterFromWls returns 1 -> 'a');
# whole-file identity measured from the pinned-repo download.
CAL_STRUCT_URL = ("https://raw.githubusercontent.com/spitschan/"
                  "IlluminationSpectraDataset/" + AUTHOR_REPO_COMMIT +
                  "/calibration/cals/OO_USB2000%2Ba.mat")
CAL_STRUCT_SHA256 = (
    "d90270171e417b1ea606dd2b595f8b8482da6a2f7bdc2f7dca0b0ec30e2dca80")
CAL_STRUCT_BYTES = 21804126
REPLAY_FILTER_SPAN = 8          # Preprocess.m medianFilterSpan
REPLAY_CROP_NM = (280.0, 840.0)  # Preprocess.m wlRange
# Independent replay acceptance bound. The published CSV carries ~10
# significant digits, so faithful reproduction is quantisation-limited at
# ~1e-10 relative; the port performs the same arithmetic on the same bytes,
# and 1e-6 leaves four orders of magnitude of margin. This bound is derived
# from the publication precision, never fitted to observed replay output.
REPLAY_MEDIAN_REL_TOLERANCE = 1e-6

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
# UTC dates whose archived responses cover every twilight-level session's
# windows (IEM days are not UTC-midnight aligned, so both boundary dates of
# each night are included).
WEATHER_DATES = ["2014-06-30", "2014-07-01", "2014-07-02", "2014-07-03",
                 "2014-07-12", "2014-07-13", "2014-07-14", "2014-07-18",
                 "2014-07-19"]
AERONET_RANGES = [("2014-07-12", "2014-07-14"), ("2014-07-18", "2014-07-19")]
AERONET_SITES = ["Penn_State_Univ", "Toronto"]
JPL_SPOT_CHECKS = [
    {"name": "horizons-cssp-0703.json", "record_id": "c1743",
     "start": "2014-07-03 08:24:49", "stop": "2014-07-03 08:25:49"},
    {"name": "horizons-cssp-0719.json", "record_id": "c3074",
     "start": "2014-07-19 08:39:17", "stop": "2014-07-19 08:40:17"},
]


def iem_obhistory_url(date: str) -> str:
    return (f"https://mesonet.agron.iastate.edu/api/1/obhistory.json"
            f"?station={IEM_STATION['station']}&network={IEM_STATION['network']}"
            f"&date={date}")


def noaa_global_hourly_url(date: str) -> str:
    return ("https://www.ncei.noaa.gov/access/services/data/v1"
            f"?dataset=global-hourly&stations={NOAA_STATION}"
            f"&startDate={date}&endDate={date}&format=json&includeAttributes=true")


def aeronet_url(site: str, start: str, end: str) -> str:
    return ("https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
            f"?site={site}&year={start[:4]}&month={int(start[5:7])}"
            f"&day={int(start[8:10])}&year2={end[:4]}&month2={int(end[5:7])}"
            f"&day2={int(end[8:10])}&AOD20=1&AVG=10&if_no_html=1")


def jpl_horizons_url(start: str, stop: str) -> str:
    quote = lambda s: s.replace(" ", "+").replace(":", "%3A")
    return ("https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND=%2710%27"
            "&OBJ_DATA=%27NO%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27OBSERVER%27"
            "&CENTER=%27coord%40399%27&COORD_TYPE=%27GEODETIC%27"
            "&SITE_COORD=%27-77.8125%2C41.6646%2C0.710%27"
            f"&START_TIME=%27{quote(start)}%27&STOP_TIME=%27{quote(stop)}%27"
            "&STEP_SIZE=%271+m%27&QUANTITIES=%274%27&TIME_DIGITS=%27SECONDS%27"
            "&APPARENT=%27AIRLESS%27&CSV_FORMAT=%27YES%27")


def aeronet_file_range(filename: str) -> tuple[str, str] | None:
    """(start_date, end_date) as ISO dates from an archived AERONET file name.

    The archive convention is ``aeronet-<site>-<YYYYMMDD>-<D2>-AOD20.txt``
    where ``D2`` is the end day within the same month; parsing the name (not
    substring search) is what keeps acquisition, provenance and per-session
    AOD coverage tied to the same dates.
    """
    if not (filename.startswith("aeronet-") and filename.endswith("-AOD20.txt")):
        return None
    stem = filename[len("aeronet-"):-len("-AOD20.txt")]
    for site in AERONET_SITES:
        if stem.startswith(site + "-"):
            span = stem[len(site) + 1:]
            if re.fullmatch(r"(\d{8})-(\d{1,2})", span):
                c1, d2 = span.split("-")
                start = f"{c1[:4]}-{c1[4:6]}-{c1[6:]}"
                end = f"{c1[:4]}-{c1[4:6]}-{int(d2):02d}"
                return start, end
    return None


def aeronet_range_intersects(filename: str, dates: list[str]) -> bool:
    rng = aeronet_file_range(filename)
    if rng is None:
        return False
    lo = datetime.fromisoformat(rng[0]).date()
    hi = datetime.fromisoformat(rng[1]).date()
    return any(lo <= datetime.fromisoformat(d).date() <= hi for d in dates)


def weather_source_url(filename: str) -> str | None:
    """Canonical request URL for an archived weather/JPL response file name,
    so sources.json records the exact request even without fetch provenance."""
    if filename.startswith("iem-N38-obhistory-"):
        return iem_obhistory_url(f"{filename[18:22]}-{filename[22:24]}-{filename[24:26]}")
    if filename.startswith("noaa-BFD-") and filename.endswith(".json"):
        d = filename[9:17]
        return noaa_global_hourly_url(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    rng = aeronet_file_range(filename)
    if rng is not None:
        for site in AERONET_SITES:
            if Path(filename).name.startswith(f"aeronet-{site}-"):
                return aeronet_url(site, rng[0], rng[1])
    if filename.startswith("frozen-"):
        return None  # feasibility-packet subsets; see frozen_feasibility_refs
    return None


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
                # The timing contract is derived, raw-evidenced content: the
                # acceptance re-derives it through the build's single source
                # instead of trusting the frozen copy (r02 Major: a hand-edited
                # exposure list or a single-exposure midpoint used to pass all
                # three entries while internally contradicting the raw MAT).
                if invoke_time_to_utc_iso(contract["invoke_time"]["value"]) \
                        != record["invoke_time_utc"]:
                    problems.append(f"{record_id}: raw invokeTime does not bind "
                                    "to the record's UTC instant")
                expected_timing = nominal_exposure_window(
                    contract, record["invoke_time_utc"])
                if json.dumps(expected_timing, sort_keys=True) != \
                        json.dumps(record.get("timing"), sort_keys=True):
                    problems.append(f"{record_id}: timing block does not recompute "
                                    "from the raw contract via "
                                    "nominal_exposure_window")
                expected_saturation = saturation_block(
                    contract["spd_max_value"]["value"])
                if json.dumps(expected_saturation, sort_keys=True) != \
                        json.dumps(record.get("saturation"), sort_keys=True):
                    problems.append(f"{record_id}: saturation block does not "
                                    "recompute from the raw spd maximum")
        else:
            problems.append(f"{record_id}: no raw member bound (AC1 requires one)")

    # Replay acceptance is read-only and independent (M3/M4 revision): every
    # record is recomputed from the vendored extract + MAT + published excerpt
    # and judged against the quantisation-derived tolerance plus zero-cell and
    # NaN-mask correspondence - never against the generator's own stored copy.
    replay_eval = evaluate_replay(packet)
    report["replay_acceptance"] = {k: v for k, v in replay_eval.items()
                                   if k != "detail"}
    problems += replay_eval["problems"]
    if not (packet / "replay.json").is_file():
        problems.append("packet has no frozen replay.json")

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

    # Selection and conditions coherence, including the single authority for
    # 'cleanest' claims and full twilight-session coverage.
    inventory_ids = {s["session_id"] for s in selection["sessions"]}
    if selection.get("cleanest_screened_windows") != cleanest_windows(conditions):
        problems.append("selection cleanest-screened-window list disagrees with "
                        "the conditions verdicts")
    for sess in selection["sessions"]:
        sid = sess["session_id"]
        if sess.get("signal_kind") == "twilight-or-night" and sid not in conditions:
            problems.append(f"{sid}: twilight-level session lacks a conditions entry")
    for decision in selection["decisions"]:
        if decision["session_id"] not in inventory_ids:
            problems.append(f"selection references unknown {decision['session_id']}")
        if not decision.get("reason"):
            problems.append(f"{decision['session_id']}: decision without reason")
        if decision["decision"] == "selected" and decision["session_id"] not in conditions:
            problems.append(f"{decision['session_id']}: selected but no conditions entry")

    # Derived availability marks must recompute from the vendored evidence
    # (r03 Major, AC7): a hand-flipped saturation flag or a moonlit window
    # marked moon-free used to pass with a re-signed manifest - hashes prove
    # frozen identity, not that the derivation agrees with the evidence.
    station_meta = _weather_station_meta()
    weather_dir = packet / "raw/weather"
    weather_files = sorted(p.name for p in weather_dir.iterdir())
    paths_by_station = {
        "KN38": [weather_dir / f for f in weather_files
                 if f.startswith(("iem-N38-obhistory", "frozen-iem"))],
        "KBFD": [weather_dir / f for f in weather_files
                 if f.startswith(("noaa-BFD", "frozen-noaa"))],
    }
    for sid, cond in conditions.items():
        date_keys = [d.replace("-", "") for d in cond["local_dates_utc_covered"]]
        for side, window in cond.get("target_windows", {}).items():
            # Moon: moon_free must follow from the stored range, and the
            # range must sit inside the vendored MICA envelope over the
            # stated window (the window columns' own labels are not vendored
            # for unselected sessions, so the envelope bounds instead of
            # replaying the derivation).
            lo, hi = window["moon_elevation_deg_range"]
            if window["moon_free"] != (hi < 0.0):
                problems.append(f"{sid}/{side}: moon_free contradicts the stored "
                                "moon elevation range")
            w_start = datetime.fromisoformat(window["window_utc"][0])
            w_end = datetime.fromisoformat(window["window_utc"][1])
            instants = [(w_start + (w_end - w_start) * k / 24).isoformat()
                        for k in range(25)]
            envelope = [mica_elevation(moon_tab, unix_of(iso))["elevation_deg"]
                        for iso in instants]
            if lo < min(envelope) - 0.05 or hi > max(envelope) + 0.05:
                problems.append(f"{sid}/{side}: stored moon range is outside "
                                "the vendored MICA envelope for the window")
            # Weather: summaries must recompute from the archived verbatim
            # responses, and states/verdict must re-render through the same
            # single source the build used.
            summaries = {}
            for station_key, station_paths in paths_by_station.items():
                paths = [p for p in station_paths
                         if p.name.startswith("frozen")
                         or any(k in p.name for k in date_keys)]
                frozen_station = window["weather"][station_key]
                rows = weather_rows_for_window(paths, *window["window_utc"])
                context_rows = weather_rows_for_window(
                    paths, *frozen_station["context_window_utc"])
                summaries[station_key] = {
                    **station_meta[station_key],
                    "window_utc": window["window_utc"],
                    "summary": summarize_weather_rows(rows, station_key),
                    "context_window_utc": frozen_station["context_window_utc"],
                    "context_summary": summarize_weather_rows(context_rows, station_key),
                }
            if json.dumps(summaries, sort_keys=True) != \
                    json.dumps(window["weather"], sort_keys=True):
                problems.append(f"{sid}/{side}: weather summaries do not "
                                "recompute from the archived responses")
            states = window_states({"weather": summaries})
            if json.dumps(states, sort_keys=True) != \
                    json.dumps(window["weather_states"], sort_keys=True):
                problems.append(f"{sid}/{side}: weather_states do not recompute "
                                "from the archived responses")
            if render_window_verdict(window["moon_free"], states) != window["verdict"]:
                problems.append(f"{sid}/{side}: verdict does not re-render from "
                                "the recomputed states")

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
        size = csv_path.stat().st_size
        matched = digest == FULL_CSV_SHA256 and size == FULL_CSV_BYTES
        report["full_csv"] = {"sha256": digest, "bytes": size, "match": matched}
        if not matched:
            # A full-source check with the wrong source must fail loudly, not
            # silently skip the stronger comparison.
            problems.append(f"full-source CSV identity mismatch (sha {digest[:12]}…, "
                            f"{size} bytes; expected {FULL_CSV_SHA256[:12]}…, "
                            f"{FULL_CSV_BYTES} bytes)")
        else:
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

def acquisition_items(with_raw_archive: bool) -> list[dict]:
    """The full acquisition contract: every item build's preflight requires.

    Weather/JPL responses and the calibration struct are required inputs, so
    the explicit fetch entry covers them; known-failing probes stay in the
    contract as recorded evidence (optional items never fail the fetch).
    """
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
        {"id": "cal-struct-a", "url": CAL_STRUCT_URL,
         "path": "calibration/OO_USB2000+a.mat",
         "bytes": CAL_STRUCT_BYTES, "sha256": CAL_STRUCT_SHA256},
    ]
    if with_raw_archive:
        items.append({"id": "raw-archive", "url": RAW_ARCHIVE_URL,
                      "final_url": RAW_ARCHIVE_FINAL_URL, "path": "dataraw.tar.gz",
                      "bytes": RAW_ARCHIVE_TOTAL_BYTES,
                      "sha256": RAW_ARCHIVE_FULL_SHA256})
    for source_name in ("IlluminationSpectraDataset_SpecCal_GetDarkSpectrum.m",
                        "IlluminationSpectraDataset_SpecCal_DarkCalibration.m",
                        "IlluminationSpectraDataset_SpecCal_WavelengthCalibration.m",
                        "IlluminationSpectraDataset_SpecCal_RawToAbsolute.m"):
        items.append({"id": f"code-{source_name}",
                      "url": ("https://raw.githubusercontent.com/spitschan/"
                              "IlluminationSpectraDataset/" + AUTHOR_REPO_COMMIT +
                              "/calibration/" + source_name),
                      "path": f"calibration/{source_name}"})
    for date in WEATHER_DATES:
        compact = date.replace("-", "")
        items.append({"id": f"iem-{compact}",
                      "url": iem_obhistory_url(date),
                      "path": f"weather/iem-N38-obhistory-{compact}.json"})
        noaa_item = {"id": f"noaa-{compact}",
                     "url": noaa_global_hourly_url(date),
                     "path": f"weather/noaa-BFD-{compact}.json"}
        if noaa_item["path"] in WEATHER_SUBSTITUTES:
            noaa_item["optional"] = True
            noaa_item["substitute"] = WEATHER_SUBSTITUTES[noaa_item["path"]]
            noaa_item["note"] = ("full-day endpoint timed out repeatedly at acquisition; "
                                 "the frozen feasibility window subset is the documented "
                                 "substitute and is vendored by build")
        items.append(noaa_item)
    for start, end in AERONET_RANGES:
        for site in AERONET_SITES:
            # Archive convention: YYYYMMDD of the start, bare end day (same
            # month) - the names already frozen in the cache and packet.
            path_name = f"weather/aeronet-{site}-{start.replace('-', '')}-" \
                        f"{int(end[8:10])}-AOD20.txt"
            items.append({"id": f"aeronet-{site}-{start}-{end}",
                          "url": aeronet_url(site, start, end),
                          "path": path_name})
    for check in JPL_SPOT_CHECKS:
        items.append({"id": f"jpl-{check['name']}",
                      "url": jpl_horizons_url(check["start"], check["stop"]),
                      "path": f"jpl/{check['name']}"})
    for date in ("2014-07-12", "2014-07-14", "2014-07-18"):
        compact = date.replace("-", "")
        items.append({"id": f"iem-daily-{compact}",
                      "url": ("https://mesonet.agron.iastate.edu/api/1/"
                              "daily_history.json?station=N38&network=PA_ASOS"
                              f"&year={date[:4]}&month={date[5:7]}"),
                      "path": f"weather/iem-N38-daily-{compact}-month.json",
                      "optional": True,
                      "note": "endpoint returned 404 at acquisition time; kept as "
                              "a recorded failed probe"})
    return items


def fetch_sources(cache_dir: Path, with_raw_archive: bool,
                  items: list[dict] | None = None) -> dict:
    """Download the acquisition contract into ``cache_dir``; skip cached hits.

    Every item records URL, bytes and sha256 in ``fetch-provenance.json``.
    Identity is checked after download; a wrong identity leaves the failure
    in the provenance and raises. Failed optional probes are kept as
    evidence, never turned into empty success. Download targets create their
    parent directories (calibration/, weather/, jpl/ are part of the
    contract).
    """
    import urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    for item in (items if items is not None
                 else acquisition_items(with_raw_archive)):
        target = cache_dir / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(item)
        entry.pop("final_url", None)
        entry["cached"] = target.is_file()
        if not target.is_file():
            request = urllib.request.Request(item["url"],
                                             headers={"User-Agent": "sky-color-sim/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=600) as resp, \
                        target.open("wb") as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            except Exception as exc:   # failed probes are evidence, not errors
                target.unlink(missing_ok=True)
                if item.get("optional"):
                    entry["cached"] = False
                    entry["identity"] = "failed-probe"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    provenance.append(entry)
                    continue
                provenance.append(entry)
                write_json(cache_dir / "fetch-provenance.json", provenance)
                raise
            entry["cached"] = True
        elif item.get("optional"):
            entry["identity"] = "kept-probe"
            provenance.append(entry)
            continue
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


# Documented cache substitutions: a cache path whose full-day fetch has
# failed (recorded as a failed probe) may be substituted by an explicit
# frozen source file, which build vendors alongside the cache responses.
# The substitute must exist in the Git tree, so the packet still carries
# real evidence - absence is never upgraded into coverage.
WEATHER_SUBSTITUTES = {
    "weather/noaa-BFD-20140703.json": (
        "validation/data/cssp_observation_feasibility/atmosphere/"
        "noaa-BFD-window.json"),
}


def verify_archive_identity(index: dict) -> None:
    """Build attests the WHOLE raw-archive identity, not just member hits.

    A re-tarred archive containing exactly the selected members would map,
    extract and replay identically while being a different upstream object;
    the measured identity of the cached archive must equal the pinned
    whole-file identity before build may sign it into sources.json.
    """
    if index.get("archive_sha256") != RAW_ARCHIVE_FULL_SHA256 \
            or index.get("total_bytes") != RAW_ARCHIVE_TOTAL_BYTES:
        raise ValueError(
            "raw archive identity mismatch: measured "
            f"{index.get('archive_sha256', '?')[:12]}…/{index.get('total_bytes', '?')} B, "
            f"expected {RAW_ARCHIVE_FULL_SHA256[:12]}…/{RAW_ARCHIVE_TOTAL_BYTES} B; "
            "the cache archive is not the pinned upstream whole file")


def check_cache_inputs(cache_dir: Path, substitutes: dict | None = None,
                       substitute_root: Path | None = None) -> None:
    """Preflight: build refuses to start without every required input.

    The required set is derived from the same acquisition contract fetch uses
    (non-optional items except the documentation-only author code files), so
    a single weather file cannot stand in for the full response set and an
    empty directory cannot pass as evidence. Entries of WEATHER_SUBSTITUTES
    may stand in for their cache path only while the named frozen source
    file exists.
    """
    subs = WEATHER_SUBSTITUTES if substitutes is None else substitutes
    root = ROOT if substitute_root is None else substitute_root
    required_paths = [
        item["path"] for item in acquisition_items(with_raw_archive=True)
        if not item.get("optional") and not item["id"].startswith("code-")]
    missing, substituted = [], []
    for name in required_paths:
        if (cache_dir / name).is_file():
            continue
        if name in subs and (root / subs[name]).is_file():
            substituted.append(f"{name} <- {subs[name]}")
            continue
        missing.append(name)
    if missing:
        raise ValueError(f"cache is missing required inputs: {missing}; "
                         "run fetch --raw-archive to acquire the full set")


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
    """Construct the packet from the verified cache. Refuses to overwrite.

    Transactional: the build AND its full acceptance run inside a staging
    directory; the previous packet is only retired through a recoverable
    rename after the staged copy has passed verification, and is restored
    if the swap itself fails. Every failure mode - preflight, extraction,
    acceptance (a failed report or a raised error), swap - therefore leaves either the previous packet or a
    clean state behind, never a half-built or failed replacement (the
    plan's "failed attempts withdraw only their own incomplete artefacts"
    rule).
    """
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"refusing to overwrite existing packet {out}; "
                         "pass --force only for a deliberate rebuild")
    check_cache_inputs(cache_dir)
    import shutil
    import tempfile
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-staging-",
                                    dir=out.parent))
    try:
        _build_packet_into(staging, cache_dir)
        # Full acceptance happens INSIDE staging, before the previous packet
        # is touched: a build that cannot pass verification must never
        # replace a good packet (r05 Major: acceptance used to run after the
        # irreversible rmtree, so an acceptance failure destroyed the packet).
        report = verify_packet(staging)
        if not report["ok"]:
            raise ValueError(f"freshly built packet failed verification: "
                             f"{report['problems'][:10]}")
        backup = None
        if out.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{out.name}-previous-",
                                           dir=out.parent))
            backup.rmdir()
            out.rename(backup)   # reversible: the old packet stays on disk
        try:
            staging.rename(out)
        except Exception:
            if backup is not None:
                backup.rename(out)   # restore the previous packet in full
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        # This attempt owns its staging from build through acceptance to the
        # swap: whatever escapes above - build failure, acceptance returning
        # problems OR raising, swap failure - withdraws only the staging
        # (r02 Minor). After a successful rename the staging is already
        # gone, so this is a no-op.
        shutil.rmtree(staging, ignore_errors=True)
    records = read_json(out / "records.json")["records"]
    selection = read_json(out / "selection.json")
    return {"records": len(records), "sessions": len(selection["sessions"]),
            "raw_members": report["n_records"], "verify": report}


def _build_packet_into(out: Path, cache_dir: Path) -> None:
    """Build every packet file into ``out`` (already created, empty).

    Raises on any failure; the orchestrator stages this into a temporary
    directory so a failed attempt never destroys the previous packet.
    """
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
    archive_index = scan_raw_index(tar_path)
    verify_archive_identity(archive_index)
    mapping = map_columns_to_members(labels, archive_index, columns)

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
            "saturation": saturation_block(contract["spd_max_value"]["value"]),
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

    # Screening before selection: every twilight-level session gets its
    # independent conditions verdict first; the selection reasons then quote
    # those verdicts, so selection.json and conditions.json cannot diverge.
    conditions_doc = build_conditions(out, labels, values, inventory, sun_tab,
                                      moon_tab, selected_by_session=picks,
                                      independent_axis_checks=jpl_checks)
    decisions = derive_decisions(inventory, conditions_doc["conditions"],
                                 conditions_doc["not_conditions_evaluated"], picks)
    write_json(out / "selection.json", {
        "rule": "sessions are gap-split (900 s); twilight-level candidates are all "
                "independently screened (moon geometry + archived neighbour weather); "
                "raw sampling is bounded to the two best-supported nights and the "
                "decision reasons quote the conditions verdicts; no sky-model residual "
                "is read at any point",
        "cleanest_screened_windows": cleanest_windows(conditions_doc["conditions"]),
        "sessions": inventory,
        "decisions": decisions})
    write_json(out / "conditions.json", conditions_doc)

    write_json(out / "operator.json", operator_contract())

    # Reduced calibration extract + processing replay (plan M3). The extract
    # is checked against the measured whole-file hash of the source struct;
    # equivalence holds by construction (same nodes/rows are vendored).
    cal_cache = cache_dir / "calibration/OO_USB2000+a.mat"
    if not cal_cache.is_file():
        raise ValueError("calibration struct missing from cache; run fetch")
    if sha256_file(cal_cache) != CAL_STRUCT_SHA256:
        raise ValueError("cached calibration struct identity mismatch")
    cal_full = load_full_cal_struct(cal_cache)
    integration_times = [r["instrument"]["integration_time_us"]["value"]
                         for r in records]
    temps_needed = [float(np.mean(r["instrument"]["board_temp_c"]["value"]))
                    for r in records]
    build_calibration_extract(cal_full, integration_times, temps_needed,
                              out / "calibration-extract.json", CAL_STRUCT_SHA256)

    diagnostics = compute_diagnostics(out, labels, values, raw_cols)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "sources.json", build_sources(out, cache_dir, columns))
    run_replay(out, write=True)  # explicit re-freeze at construction time only

    # README and manifest last.
    write_readme(out, columns)
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "inputs.sha256.json":
            manifest[str(path.relative_to(out))] = sha256_file(path)
    write_json(out / "inputs.sha256.json", manifest)


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
                            "was not measured. Unit convention: kernels are normalised to "
                            "unit transmission so band integrals keep spectral-density "
                            "units (W m-2 nm-1); band-integral quantities use the "
                            "trapezoidal band integral explicitly",
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
            "P_publish": {"text": "publishing steps that apply to already-calibrated "
                                  "physical spectra (W m-2 nm-1): resampling onto the "
                                  "FULL calibration grid (150-1100 nm), the published "
                                  "median filter span 8 applied there in the "
                                  "detector-response domain - E -> D*median(E/D), "
                                  "D = correction_factor*pi from the vendored extract "
                                  "(filtering and wavelength-dependent scaling do not "
                                  "commute) - negative clamp, and only then the crop to "
                                  "the published 280-840 band. The edge channels "
                                  "(280-283, 838-840 nm) are not closable in physical "
                                  "units: the upstream smoothing context there is "
                                  "count-domain signal outside the calibrated band, for "
                                  "which no public correction factor exists, so "
                                  "publish_pipeline returns NaN for filter windows "
                                  "touching the uncalibrated region and band "
                                  "comparisons must use interior bands (all reference "
                                  "bands are interior)",
                          "evidence": EVIDENCE_DERIVED},
            "C_capture": {"text": "the raw-count-to-physical calibration chain (dark "
                                  "subtraction by nearest integration-time node and "
                                  "temperature interpolation, division by integration "
                                  "time in microseconds, sensitivity correction factor, "
                                  "pi factor). Defined ONLY on detector counts; it is "
                                  "replayed per record in replay.json to prove the raw/"
                                  "published binding. Applying C_capture to a model's "
                                  "physical irradiance would double-calibrate and must "
                                  "never happen",
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
            "operator_domain": "the formula integrates PHYSICAL radiance; only "
                               "P_publish applies to its output. C_capture lives on the "
                               "detector-count side of replay.json and is documentation "
                               "of the upstream pipeline, not a model-facing step",
        },
        "reference_bands_nm": REFERENCE_BANDS_NM,
        "forbidden_comparisons": [
            "hemispheric irradiance vs zenith radiance (different quantity)",
            "published spectra vs model spectra without integrating model radiance over "
            "the hemisphere and the exposure window",
            "treating night-spectrum medians as a measured background subtraction",
            "applying C_capture (dark counts, sensitivity correction factor, pi) to a "
            "model's physical irradiance - that chain is defined on detector counts only "
            "and would double-calibrate",
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


def window_states(side_window: dict) -> dict:
    """Single authority for one window's weather evidence (revision r02).

    Selects the effective evidence per station (tight window when an
    observation exists, else the +-60 min context rows, labelled as such)
    and derives the structured flags the window verdict, session summary,
    cleanest list and selection reasons all render from. No other code may
    apply the fallback independently - that duplication is what produced
    contradictory machine-readable verdicts in r01.
    """
    stations = {}
    for key, s in side_window["weather"].items():
        summary, context = s["summary"], s["context_summary"]
        effective, from_context = summary, False
        if not summary["n_rows"] and context["n_rows"]:
            effective, from_context = context, True
        if not effective["n_rows"]:
            state = "no-observation"
        elif effective["fog"] or effective["low_opaque"]:
            state = "fog/low-cloud"
        elif effective["clear"] and not effective["other"]:
            state = "CLR"
        else:
            state = "partial/high-cloud"
        stations[key] = {"state": state,
                         "evidence_scope": "context" if from_context else "window",
                         "n_rows_tight": summary["n_rows"],
                         "n_rows_context": context["n_rows"]}
    states = [v["state"] for v in stations.values()]
    return {
        "stations": stations,
        "all_clear": bool(states) and all(s == "CLR" for s in states),
        "any_fog": any(s == "fog/low-cloud" for s in states),
        "any_clear": any(s == "CLR" for s in states),
        "evidence_scope": ("window" if all(v["evidence_scope"] == "window"
                                           for v in stations.values())
                           else "mixed"
                           if any(v["evidence_scope"] == "window"
                                  for v in stations.values())
                           else "context"),
    }


def render_window_verdict(moon_free: bool, states: dict) -> str:
    """Render the human-readable verdict from window_states() output."""
    states_list = [v["state"] for v in states["stations"].values()]
    scope_note = ""
    if states["evidence_scope"] == "context":
        scope_note = (" (CLR established from +-60 min context rows; no in-window "
                      "observation at hourly stations)")
    if moon_free and states["all_clear"]:
        return ("cleanest-available: moon below horizon at all target instants "
                "and both neighbour stations report CLR through the window"
                + scope_note)
    if moon_free and states["any_fog"]:
        return ("conditional: moon below horizon, but fog or low cloud is "
                "reported at neighbour stations inside the window; CSSP itself "
                "(710 m plateau) has no in-situ evidence, so absolute use is "
                "blocked and spectral/relative use carries this caveat")
    if not moon_free:
        return "moonlit: moon above the horizon during the window"
    return "mixed: neighbour reports split between clear and cloud/fog rows"


def session_verdict(side_windows: dict) -> str:
    """Compose the per-session verdict text from the per-side windows."""
    parts = []
    for side in ("dusk", "dawn"):
        window = side_windows.get(side)
        if window is None:
            continue
        moon_state = "moon-free" if window["moon_free"] else "moonlit"
        states = [f"{key}: {v['state']}"
                  + (" (context rows)" if v["evidence_scope"] == "context" else "")
                  for key, v in window["weather_states"]["stations"].items()]
        parts.append(f"{side} [{moon_state}; {', '.join(states)}]")
    return "; ".join(parts)


def saturation_block(spd_max_value: float | None) -> dict:
    """Scenario-level saturation flag derived from the raw spd maximum.

    USB2000+ is a 16-bit detector; the 65535 ceiling is an assumption (not a
    published specification), so the flag is scenario-level. The raw-measured
    ratio is the dependable quantity; values within a few percent of ceiling
    need exposure context before band ratios are trusted. Single source for
    build and acceptance (r03 Major): a flipped frozen flag must fail verify.
    """
    return {
        "suspect": bool(spd_max_value is not None and spd_max_value >= 62258.0),
        "assumed_ceiling_counts": 65535,
        "suspect_threshold_counts": 62258,
        "spd_max_over_assumed_ceiling": (
            None if spd_max_value is None
            else round(spd_max_value / 65535.0, 4)),
        "evidence": EVIDENCE_SCENARIO,
        "note": "USB2000+ is a 16-bit detector; the 65535 ceiling is an "
                "assumption (not a published specification) so the flag is "
                "scenario-level. The raw-measured spd_max_over_assumed_ceiling "
                "is the dependable quantity; values within a few percent of "
                "ceiling need exposure context before band ratios are trusted",
    }


def _weather_station_meta() -> dict:
    """Neighbour-station geometry relative to CSSP (shared build+verify)."""
    return {
        "KN38": {"station": KN38,
                 "distance_km": round(haversine_km(CSSP["lat_deg"], CSSP["lon_deg"],
                                                   KN38["lat_deg"], KN38["lon_deg"]), 2),
                 "elevation_diff_m": round(KN38["elevation_m"] - CSSP["elevation_m"], 2)},
        "KBFD": {"station": KBFD,
                 "distance_km": round(haversine_km(CSSP["lat_deg"], CSSP["lon_deg"],
                                                   KBFD["lat_deg"], KBFD["lon_deg"]), 2),
                 "elevation_diff_m": round(KBFD["elevation_m"] - CSSP["elevation_m"], 2)},
    }


def build_conditions(out: Path, labels: list[str], values: np.ndarray,
                     inventory: list[dict], sun_tab: dict, moon_tab: dict,
                     selected_by_session: dict[str, list[int]] | None = None,
                     independent_axis_checks: list[dict] | None = None) -> dict:
    """Independent conditions for every twilight-level session.

    The plan's screening is over all existing candidates, so this evaluates
    moon geometry and weather windows for each twilight-or-night session,
    selected or not (raw extraction stays bounded to the selected nights;
    screening costs no raw downloads). Daytime-level sessions are recorded
    as rule-screened, not conditions-evaluated. No model quantity is read
    here; neighbour-station rows stay verbatim.
    """
    conditions: dict[str, dict] = {}
    not_evaluated: dict[str, dict] = {}
    weather_files = sorted(p.name for p in (out / "raw/weather").iterdir())
    weather_paths_all = {
        "KN38": [out / "raw/weather" / f for f in weather_files
                 if f.startswith(("iem-N38-obhistory", "frozen-iem"))],
        "KBFD": [out / "raw/weather" / f for f in weather_files
                 if f.startswith(("noaa-BFD", "frozen-noaa"))],
    }
    station_meta = _weather_station_meta()
    for sess in inventory:
        sid = sess["session_id"]
        if sess["signal_kind"] != "twilight-or-night":
            not_evaluated[sid] = {
                "reason": "daytime-level signal under the frozen screening rule "
                          "(median positive level >= 1e-5 W m-2 nm-1); not a twilight "
                          "comparison candidate",
                "signal_kind": sess["signal_kind"],
            }
            continue
        start, end = sess["start_utc"], sess["end_utc"]
        a, b = sess["columns_zero_based"]
        all_cols = list(range(a, b + 1))
        selected_here = bool(selected_by_session and sid in selected_by_session)
        fractions = values[CSV_MOONFRAC_ROW, a:b + 1]
        dates = utc_dates_between(start, end)
        date_keys = [d.replace("-", "") for d in dates]
        weather_paths = {
            key: [p for p in paths
                  if p.name.startswith("frozen")
                  or any(k in p.name for k in date_keys)]
            for key, paths in weather_paths_all.items()
        }
        # Uniform window basis for every session (revision r02): the columns
        # where the corrected sun sits inside the target-depth band
        # (-13, -5.5] deg, taken from the full session regardless of which
        # records were sampled raw. This makes the moon/weather verdicts
        # comparable across sessions instead of tracking the raw-sample set.
        side_windows = {}
        for side, hour_test in (("dusk", lambda h: h >= 12), ("dawn", lambda h: h < 12)):
            side_cols = [c for c in all_cols if hour_test(parse_edt_label(labels[c]).hour)]
            if not side_cols:
                continue
            elevations = {c: mica_elevation(sun_tab, unix_of(label_to_utc_iso(labels[c])))
                          ["elevation_deg"] for c in side_cols}
            window_cols = [c for c in side_cols
                           if -13.0 < elevations[c] <= -5.5] or side_cols
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
            # One authority: window_states applies the context fallback once;
            # the window verdict, session summary, cleanest list and selection
            # reasons all render from this result.
            states = window_states({"weather": summaries})
            verdict = render_window_verdict(moon_free, states)
            side_windows[side] = {
                "columns_zero_based": [min(window_cols), max(window_cols)],
                "window_utc": [start_iso, end_iso],
                "n_window_columns": len(window_cols),
                "moon_elevation_deg_range": [round(min(moon_at), 3),
                                             round(max(moon_at), 3)],
                "moon_free": moon_free,
                "weather": summaries,
                "weather_states": states,
                "verdict": verdict,
            }
        moon_ranges = [w["moon_elevation_deg_range"]
                       for w in side_windows.values() if w]
        aeronet_hits = [f for f in weather_files if f.startswith("aeronet")
                        and aeronet_range_intersects(f, dates)]
        conditions[sid] = {
            "utc_window": {"start": start, "end": end},
            "local_dates_utc_covered": dates,
            "selected_for_raw_sampling": selected_here,
            "moon": {
                "window_elevation_deg_min": float(min(r[0] for r in moon_ranges)),
                "window_elevation_deg_max": float(max(r[1] for r in moon_ranges)),
                "published_illumination_fraction_min": float(np.min(fractions)),
                "published_illumination_fraction_max": float(np.max(fractions)),
                "axis_source": "vendored upstream MICA lunar table",
            },
            "target_windows": side_windows,
            "aod": {
                "queried": aeronet_hits,
                "note": "AERONET direct-sun products are daytime; any return is a far-site "
                        "column, not an in-situ twilight value",
            },
            "verdict": session_verdict(side_windows),
            "verdict_note": "derived only from these archived responses and the MICA lunar "
                            "table; no sky-model residual is read anywhere",
        }
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
                "public responses, AOD availability as returned; every twilight-level "
                "session is evaluated, daytime-level sessions are rule-screened",
        "independent_axis_checks": independent_axis_checks or [],
        "conditions": conditions,
        "not_conditions_evaluated": not_evaluated,
    }


def cleanest_windows(conditions: dict) -> list[str]:
    """Moon-free + all-clear side windows among the screened sessions, as
    'session/side' keys - rendered from the stored weather_states, which are
    themselves the product of the single window_states() authority (context
    fallback already applied there, so these claims cannot contradict the
    per-window verdict strings)."""
    out = []
    for sid, entry in conditions.items():
        for side, window in entry.get("target_windows", {}).items():
            if window["moon_free"] and window["weather_states"]["all_clear"]:
                out.append(f"{sid}/{side}")
    return out


def derive_decisions(inventory: list[dict], conditions: dict, not_evaluated: dict,
                     selected_by_session: dict[str, list[int]]) -> list[dict]:
    """Selection decisions derived from the evaluated conditions.

    The raw-sample bound (two nights) is a plan-side cost bound; the reasons
    quote the same conditions verdicts recorded in conditions.json so the two
    artefacts cannot diverge. 'Cleanest' claims are restricted to the
    screened windows via cleanest_windows().
    """
    cleanest = cleanest_windows(conditions)
    decisions = []
    for sess in inventory:
        sid = sess["session_id"]
        if sid in not_evaluated:
            decisions.append({"session_id": sid, "decision": "rejected",
                              "reason": not_evaluated[sid]["reason"]})
            continue
        verdict = conditions[sid]["verdict"]
        if sid in selected_by_session:
            role = "primary" if sid == PRIMARY_SESSION else "secondary"
            reason = (f"selected for raw sampling ({role}); conditions verdict: "
                      f"{verdict}")
            if any(w.startswith(sid) for w in cleanest):
                reason += (f"; its window is among the dataset's cleanest screened "
                           f"windows ({', '.join(w for w in cleanest
                                                 if w.startswith(sid))})")
            if len(selected_by_session[sid]) and sid == PRIMARY_SESSION:
                reason += ("; the only night with all three target twilight depths "
                           "(-6/-9/-12 deg)")
            decisions.append({"session_id": sid, "decision": "selected",
                              "role": role, "reason": reason})
        else:
            decisions.append({"session_id": sid, "decision": "not-selected",
                              "reason": f"conditions verdict: {verdict}; raw sampling "
                                        f"bounded to the two best-supported nights "
                                        f"({', '.join(sorted(selected_by_session))}); "
                                        "inventory retained for audit"})
    return decisions


def refresh_diagnostics(packet: Path, write: bool = False) -> dict:
    """Recompute the scenario diagnostics.

    Default is read-only: compare the fresh computation with the frozen
    ``diagnostics.json`` (identity of every stored value). ``write=True`` is
    the explicit re-freeze used by build; standalone runs must not refresh
    the acceptance baseline as a side effect.
    """
    records = read_json(packet / "records.json")["records"]
    labels = {r["source_column_zero_based"]: r["timestamp_label_edt"]
              for r in records}
    diagnostics = compute_diagnostics(packet, labels, None, None)
    frozen_path = packet / "diagnostics.json"
    problems = []
    if write:
        write_json(frozen_path, diagnostics)
        manifest_path = packet / "inputs.sha256.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest["diagnostics.json"] = sha256_file(frozen_path)
            write_json(manifest_path, manifest)
    elif frozen_path.is_file():
        frozen = read_json(frozen_path)
        if json.dumps(frozen, sort_keys=True) != json.dumps(diagnostics, sort_keys=True):
            problems.append("recomputed diagnostics differ from the frozen table; "
                            "use sensitivity --write only as a deliberate re-freeze")
    else:
        problems.append("packet has no frozen diagnostics.json; run build")
    return {"ok": not problems, "problems": problems,
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

    timing_offsets_minutes = [1, 5, 10, 30, 60]
    adjacent = {}
    for record in records:
        col = record["source_column_zero_based"]
        if record["timing"]["nominal_total_exposure_s"] is None:
            continue
        neighbours = [c for c in (col - 1, col + 1) if c in col_index]
        if not neighbours:
            continue
        col_band = {name: band_energy(wl, data[:, col_index[col]], tuple(band))
                    for name, band in REFERENCE_BANDS_NM.items()}
        col_unix = unix_of(label_to_utc_iso(labels[col]))
        rows = []
        for nb in sorted(neighbours):
            nb_band = {name: band_energy(wl, data[:, col_index[nb]], tuple(band))
                       for name, band in REFERENCE_BANDS_NM.items()}
            nb_unix = unix_of(label_to_utc_iso(labels[nb]))
            gap_minutes = (nb_unix - col_unix) / 60.0   # signed: past is negative
            rows.append({
                "neighbour_column": nb,
                "signed_gap_minutes": round(gap_minutes, 3),
                "relative_change_per_band": {
                    name: (nb_band[name] - col_band[name]) / col_band[name]
                    if col_band[name] and not np.isnan(col_band[name]) else None
                    for name in REFERENCE_BANDS_NM},
            })
        sun = mica_elevation(sun_tab, unix_of(record["invoke_time_utc"]))["elevation_deg"] \
            if record["corrected_axis_cssp"]["sun_elevation_deg"] is not None else None
        # Signed band-rate estimate: central difference when both neighbours
        # exist, one-sided otherwise. rate = dE/E per minute along time, so a
        # POSITIVE offset (later instant) maps to rate * offset.
        rate_per_band = {}
        for name in REFERENCE_BANDS_NM:
            t_diff = [r["signed_gap_minutes"] for r in rows
                      if r["relative_change_per_band"][name] is not None]
            e_diff = [r["relative_change_per_band"][name] for r in rows
                      if r["relative_change_per_band"][name] is not None]
            if len(t_diff) >= 2 and (t_diff[1] - t_diff[0]) != 0:
                rate_per_band[name] = ((e_diff[1] - e_diff[0])
                                       / (t_diff[1] - t_diff[0]))
            elif len(t_diff) == 1 and t_diff[0] != 0:
                rate_per_band[name] = e_diff[0] / t_diff[0]
            else:
                rate_per_band[name] = None
        adjacent[record["record_id"]] = {
            "corrected_sun_elevation_deg": sun,
            "nominal_total_exposure_s": record["timing"]["nominal_total_exposure_s"],
            "neighbours": rows,
            "signed_rate_per_minute_per_band": {k: (round(v, 8) if v is not None else None)
                                                for k, v in rate_per_band.items()},
            "timing_offset_scenario_per_band": {
                f"{'+' if offset > 0 else ''}{offset}": {
                    name: (rate_per_band[name] * offset)
                    if rate_per_band[name] is not None else None
                    for name in REFERENCE_BANDS_NM}
                for offset in ([-m for m in timing_offsets_minutes]
                               + timing_offsets_minutes)},
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
        "scenario": "isotropic radiance L, horizon raised (terrain "
                    "obstruction) by h: E = pi L cos^2(h) against E(0) = pi L; "
                    "a horizon depressed below the true horizon (open site) "
                    "leaves E = pi L because directions below 0 deg elevation "
                    "never illuminate the upward board (Omega+)",
        "evidence": EVIDENCE_SCENARIO,
        "raised_horizon_deg": [0.0, 1.0, 2.0, 5.0, 10.0],
        "E_over_E0": [float(ideal_hemispheric_irradiance(1.0, -h) / np.pi)
                      for h in (0.0, 1.0, 2.0, 5.0, 10.0)],
        "note": "flat-horizon geometric factor only; the real site horizon and white-board "
                "angular response are unmeasured, so this bounds geometry, not response",
    }
    return {
        "generated_by": "tools/cssp_observation_inputs.py compute_diagnostics",
        "scenario_not_error_bar": True,
        "timing_offsets_minutes": timing_offsets_minutes,
        "adjacent_change": adjacent,
        "night_floor_ratios": night_floor,
        "horizon_scenarios": horizon,
        "notes": [
            "timing offsets use the signed central-difference band rate (per minute); "
            "a positive offset means later in time, and the scenario values carry the "
            "rate's sign rather than a magnitude"
            "night-floor ratios involve common device gain that may cancel; background "
            "colour structure and time variation do not cancel",
            "no background subtraction is applied anywhere in this packet",
        ],
    }


# ---------------------------------------------------------------------------
# Calibration replay: upstream RawToAbsolute port on the selected records
# ---------------------------------------------------------------------------

def load_full_cal_struct(path: Path) -> dict:
    """Load the generated calibration struct (``cals{1}`` = meter 'a')."""
    from scipy.io import loadmat
    raw = loadmat(path, simplify_cells=True)["cals"]
    cal = raw[0] if isinstance(raw, list) else raw
    if cal["whichMeter"] != "a":
        raise ValueError(f"expected meter 'a', got {cal['whichMeter']!r}")
    dark = cal["dark"]
    return {
        "wl_shift": float(cal["wlShift"]),
        "wls": np.asarray(cal["wls"], dtype=float),
        "correction_factor": np.asarray(cal["correctionFactor"], dtype=float),
        "effective_wl_range": [float(v) for v in cal["effectiveWlRange"]],
        "dark_integration_times_us": np.asarray(dark["integrationTimes"], dtype=float),
        "dark_temperatures": [np.asarray(t, dtype=float).ravel() for t in dark["darkTemperature"]],
        "dark_spds": [np.asarray(s, dtype=float) for s in dark["darkSpd"]],
    }


def dark_spectrum(cal: dict, integration_time_us: float,
                  temperature_c: float) -> np.ndarray:
    """Port of ``GetDarkSpectrum.m``: nearest integration-time node, linear
    temperature interpolation, NaN outside the measured temperature range."""
    idx = int(np.argmin(np.abs(cal["dark_integration_times_us"] - integration_time_us)))
    temps = cal["dark_temperatures"][idx]
    spds = cal["dark_spds"][idx]          # (2048, n_temps) or (n_temps, 2048)
    if spds.shape[0] == temps.size:
        values = spds                      # (n_temps, 2048)
    else:
        values = spds.T
    if not temps.min() <= temperature_c <= temps.max():
        raise ValueError(f"temperature {temperature_c} outside dark library "
                         f"[{temps.min()}, {temps.max()}] at node "
                         f"{cal['dark_integration_times_us'][idx]} us")
    return np.array([np.interp(temperature_c, temps, values[:, k])
                     for k in range(values.shape[1])])


def build_calibration_extract(cal: dict, integration_times_us: list[float],
                              temperatures_needed: list[float],
                              out_path: Path, source_sha256: str) -> None:
    """Vendor a reduced calibration extract that still replays every record.

    Keeps the full correction factor and wavelength grid, but only the dark
    library nodes actually referenced by the records (nearest-node rule, as
    GetDarkSpectrum.m resolves) and only the dark temperature rows that
    immediately bracket each needed temperature (the interpolation is linear,
    so those rows reproduce it exactly). The extract records the whole-file
    identity of its source; the 21.8 MB source itself stays in the cache and
    is not vendored.
    """
    node_indices = sorted({int(np.argmin(np.abs(cal["dark_integration_times_us"] - t)))
                           for t in integration_times_us})
    kept_nodes = []
    for i in node_indices:
        node_us = float(cal["dark_integration_times_us"][i])
        temps = cal["dark_temperatures"][i]
        spds = np.asarray(cal["dark_spds"][i], dtype=float)
        values = spds if spds.shape[0] == temps.size else spds.T   # (n_temps, 2048)
        rows = set()
        for temperature_c in temperatures_needed:
            if not temps.min() <= temperature_c <= temps.max():
                raise ValueError(f"record temperature {temperature_c} outside dark "
                                 f"library range at node {node_us} us")
            upper = int(np.searchsorted(temps, temperature_c, side="left"))
            lower = max(upper - 1, 0)
            rows.add(lower)
            if upper < temps.size and temps[upper] != temps[lower]:
                rows.add(upper)
            if temps[lower] == temperature_c:
                rows.add(lower)   # exact row; linear interp is degenerate
        sel = np.array(sorted(rows))
        kept_nodes.append({"integration_time_us": node_us,
                           "temperatures_c": temps[sel].tolist(),
                           "dark_spd_rows": values[sel].tolist()})
    extract = {
        "source_whole_file_sha256": source_sha256,
        "source_url": CAL_STRUCT_URL,
        "which_meter": "a",
        "wl_shift": cal["wl_shift"],
        "wls": cal["wls"].tolist(),
        "correction_factor": cal["correction_factor"].tolist(),
        "effective_wl_range": cal["effective_wl_range"],
        "dark_reduction": {
            "note": "only integration-time nodes referenced by the records (nearest-node "
                    "rule) and only the dark temperature rows that immediately bracket "
                    "each needed temperature (linear interpolation is exact on those "
                    "rows); interpolation semantics follow GetDarkSpectrum.m",
            "temperatures_needed_c": sorted(set(round(t, 4)
                                                for t in temperatures_needed)),
            "nodes": kept_nodes,
        },
    }
    write_json(out_path, extract)


def published_correction_factor_pi(packet: Path) -> np.ndarray:
    """D = correction_factor * pi cropped to the published 280-840 grid -
    the response-domain axis of the model-facing publish pipeline."""
    extract = load_calibration_extract(packet)
    wls = extract["wls"]
    crop = (wls >= REPLAY_CROP_NM[0]) & (wls <= REPLAY_CROP_NM[1])
    return extract["correction_factor"][crop] * np.pi


def load_calibration_extract(packet: Path) -> dict:
    raw = read_json(packet / "calibration-extract.json")
    nodes = raw["dark_reduction"]["nodes"]
    return {
        "wl_shift": raw["wl_shift"],
        "wls": np.asarray(raw["wls"], dtype=float),
        "correction_factor": np.asarray(raw["correction_factor"], dtype=float),
        "effective_wl_range": raw["effective_wl_range"],
        "dark_nodes": {n["integration_time_us"]:
                       (np.asarray(n["temperatures_c"], dtype=float),
                        np.asarray(n["dark_spd_rows"], dtype=float))
                       for n in nodes},
    }


def dark_spectrum_from_extract(cal: dict, integration_time_us: float,
                               temperature_c: float) -> np.ndarray:
    node_us = min(cal["dark_nodes"],
                  key=lambda u: abs(u - integration_time_us))
    temps, spds = cal["dark_nodes"][node_us]
    if not temps.min() <= temperature_c <= temps.max():
        raise ValueError(f"temperature {temperature_c} outside extract bracket "
                         f"[{temps.min()}, {temps.max()}] at node {node_us} us")
    if spds.shape[0] == temps.size:
        values = spds
    else:
        values = spds.T
    return np.array([np.interp(temperature_c, temps, values[:, k])
                     for k in range(values.shape[1])])


def replay_record(cal: dict, mat_path: Path, published: np.ndarray,
                  filter_span: int = REPLAY_FILTER_SPAN,
                  return_spectrum: bool = False) -> np.ndarray | dict:
    """Python port of the upstream Preprocess.m + RawToAbsolute.m chain.

    Stages are recorded individually so the residual against the published
    column can be attributed; agreement documents reproduction of the
    upstream pipeline, not an independent absolute calibration. Zero-padded
    sliding-window median (numpy median, which averages the two central
    order statistics for even spans) approximates MATLAB ``medfilt1``; any
    convention difference shows up in the filter stage diff, not silently.
    """
    from scipy.io import loadmat
    meas = loadmat(mat_path, simplify_cells=True)["meas"]
    wls_raw = np.asarray(meas["wls"], dtype=float).ravel()
    spd = np.asarray(meas["spd"], dtype=float)
    if spd.ndim == 1:
        spd = spd.reshape(-1, 1)
    t_us = float(meas["integrationTime"])
    temps = np.atleast_1d(np.asarray(meas["boardTempInCelsius"], dtype=float))
    temp = float(temps.mean())

    floor_applied = False
    if t_us < 1000:                      # Preprocess.m: <1 ms corrected to 1 ms
        t_us = 1000.0
        floor_applied = True

    mean_spd = spd.mean(axis=1)
    saturated_pixels = int((mean_spd == 65535).sum())
    dark = dark_spectrum_from_extract(cal, t_us, temp)
    # Upstream divides by the raw integration time in MICROSECONDS
    # (Preprocess.m stores tmp.meas.integrationTime verbatim), so the 1e6
    # scale is absorbed by the fitted correction factor. We follow that
    # convention exactly; "per_us" here is upstream's unit, not a physical
    # per-second quantity.
    per_us = (mean_spd - dark) / t_us

    corrected_wls = wls_raw - cal["wl_shift"]
    on_grid = np.interp(cal["wls"], corrected_wls, per_us)
    outside = (cal["wls"] < corrected_wls.min()) | (cal["wls"] > corrected_wls.max())
    on_grid[outside] = np.nan

    pad = filter_span // 2
    padded = np.pad(on_grid, pad, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, filter_span)
    import warnings
    with warnings.catch_warnings():
        # The all-NaN slices are the documented out-of-range UV boundary
        # (interp NaNs below the corrected wavelength range); the median of
        # such a window is legitimately NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered = np.nanmedian(windows, axis=1)
    filtered = filtered[:cal["wls"].size]

    absolute = filtered * cal["correction_factor"] * np.pi
    # Count the negatives BEFORE the stray clamp - after np.where the
    # counter was identically zero and the nonlinear step left no trace in
    # the staged evidence (r03 Minor).
    negative_clamped = int((absolute < 0).sum())
    absolute = np.where(absolute < 0, 0.0, absolute)   # Preprocess.m stray clamp

    crop = (cal["wls"] >= REPLAY_CROP_NM[0]) & (cal["wls"] <= REPLAY_CROP_NM[1])
    replay = absolute[crop]
    wl_crop = cal["wls"][crop]
    if replay.size != published.size:
        raise ValueError("replay/published grid size mismatch")
    if return_spectrum:
        return replay
    valid = ~np.isnan(replay) & ~np.isnan(published)
    diff = replay[valid] - published[valid]
    denom = np.abs(published[valid])
    nonzero = denom > 0
    rel = diff[nonzero] / denom[nonzero]
    return {
        "stages": {
            "exposure_mean": {"n_exposures": int(spd.shape[1])},
            "integration_time_us": t_us, "integration_floor_applied": floor_applied,
            "board_temp_c": temp,
            "dark": {"saturated_pixels_mean_equals_65535": saturated_pixels,
                     "node_us": float(min(cal["dark_nodes"],
                                          key=lambda u: abs(u - t_us))),
                     "max_dark_counts": float(np.nanmax(dark))},
            "per_us_max": float(np.nanmax(per_us)),
            "interp_nan_outside_count": int(outside.sum()),
            "filter_span": filter_span,
            "negative_clamped_pixels": negative_clamped,
        },
        "published_vs_replay": {
            "n_valid_wavelengths": int(valid.sum()),
            "max_abs_diff_w_m2_nm": float(np.max(np.abs(diff))),
            "median_abs_diff_w_m2_nm": float(np.median(np.abs(diff))),
            "median_abs_relative_diff": float(np.median(np.abs(rel))) if rel.size else None,
            "p95_abs_relative_diff": float(np.percentile(np.abs(rel), 95)) if rel.size else None,
            "published_zero_cells": int((published == 0).sum()),
            "replay_zero_after_clamp": int((replay[valid] == 0).sum()),
        },
    }


def evaluate_replay(packet: Path) -> dict:
    """Read-only replay acceptance against the frozen published excerpts.

    Recomputes every record from the vendored extract + MAT + published
    column and applies the independent quantisation-derived tolerance plus
    zero-cell and NaN-mask correspondence. This is the acceptance path; it
    writes nothing, so a processing regression can never re-sign itself as
    success by refreshing stored artefacts.
    """
    cal = load_calibration_extract(packet)
    records = read_json(packet / "records.json")["records"]
    spectra = load_packet_spectra(packet)
    col_index = {c: i for i, c in enumerate(spectra["columns"])}
    problems = []
    per_record = {}
    for record in records:
        rid = record["record_id"]
        col = record["source_column_zero_based"]
        member = packet / record["raw"]["packet_path"]
        published = spectra["data"][:, col_index[col]]
        replay = replay_record(cal, member, published, return_spectrum=True)
        # Position-wise acceptance: NaN masks must agree cell by cell, every
        # published zero must replay to an exact zero (the upstream clamp),
        # and every comparable channel must sit inside the quantisation-
        # derived tolerance. Medians are summaries, never the gate.
        record_problems = []
        nan_published = np.isnan(published)
        nan_replay = np.isnan(replay)
        if not np.array_equal(nan_published, nan_replay):
            record_problems.append(
                f"NaN masks differ at {int((nan_published ^ nan_replay).sum())} channels")
        comparable = ~nan_published & ~nan_replay
        # Infinities are corrupted excerpts, not physics: without this gate a
        # published +inf slipped through as inf/inf = NaN and NaN >= tolerance
        # is False (r02 Major). Non-finite channels are excluded from every
        # later comparison so they cannot poison the relative-error math.
        nonfinite = comparable & ~(np.isfinite(published) & np.isfinite(replay))
        if nonfinite.any():
            wl = np.arange(280.0, 841.0)[nonfinite][:5]
            record_problems.append(
                f"{int(nonfinite.sum())} published channels are non-finite "
                f"(+-inf is a corrupted excerpt, not physics; "
                f"first wl {wl.tolist()})")
        comparable = comparable & ~nonfinite
        negative = comparable & (published < 0)
        if negative.any():
            wl = np.arange(280.0, 841.0)[negative][:5]
            record_problems.append(
                f"{int(negative.sum())} published channels are negative "
                f"(upstream clamps to zero; first wl {wl.tolist()}) - "
                "corrupted excerpt, not physics")
        zero_cells = comparable & (published == 0)
        broken_zeros = zero_cells & (replay != 0)
        if broken_zeros.any():
            record_problems.append(
                f"{int(broken_zeros.sum())} published-zero channels replay non-zero")
        positive = comparable & (published > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(replay[positive] - published[positive]) / published[positive]
        if rel.size and not np.isfinite(rel).all():
            record_problems.append(
                f"{int((~np.isfinite(rel)).sum())} comparable channels produce "
                "a non-finite relative error (overflow or degenerate published "
                "value) - not a tolerance pass")
        finite_rel = rel[np.isfinite(rel)]
        breaches = np.flatnonzero(rel >= REPLAY_MEDIAN_REL_TOLERANCE)
        if breaches.size:
            wl = np.arange(280.0, 841.0)[positive][breaches[:5]]
            record_problems.append(
                f"{breaches.size} channels exceed the per-channel tolerance "
                f"{REPLAY_MEDIAN_REL_TOLERANCE:.0e} (worst rel "
                f"{rel[breaches].max():.3e}; first wl {wl.tolist()})")
        per_record[rid] = {
            "median_abs_relative_diff":
                float(np.median(finite_rel)) if finite_rel.size else None,
            "max_abs_relative_diff": float(finite_rel.max()) if finite_rel.size else None,
            "n_comparable_channels": int(positive.sum()),
            "n_zero_channels": int(zero_cells.sum()),
            "problems": record_problems,
        }
        problems += [f"{rid}: {msg}" for msg in record_problems]
    return {"ok": not problems, "problems": problems,
            "tolerance": REPLAY_MEDIAN_REL_TOLERANCE,
            "records_evaluated": len(per_record), "detail": per_record}


def run_replay(packet: Path, write: bool = False) -> dict:
    """Compute the replay table.

    Default (``write=False``) is read-only acceptance: compare against the
    frozen ``replay.json`` identity AND the independent tolerance, touching
    no packet byte. ``write=True`` is an explicit re-freeze (build only):
    it rewrites ``replay.json`` and its manifest entry and still reports any
    tolerance breach, which verify then enforces against the stored table.
    """
    # The verdict comes from the single position-wise authority; this method
    # only rebuilds the frozen-table detail (write path) and keeps summaries.
    acceptance = evaluate_replay(packet)
    cal = load_calibration_extract(packet)
    records = read_json(packet / "records.json")["records"]
    spectra = load_packet_spectra(packet)
    col_index = {c: i for i, c in enumerate(spectra["columns"])}
    per_record = {}
    for record in records:
        col = record["source_column_zero_based"]
        member = packet / record["raw"]["packet_path"]
        published = spectra["data"][:, col_index[col]]
        result = replay_record(cal, member, published)
        per_record[record["record_id"]] = {
            "corrected_sun_elevation_deg":
                record["corrected_axis_cssp"]["sun_elevation_deg"],
            "integration_time_us": record["instrument"]["integration_time_us"]["value"],
            "n_exposures": record["instrument"]["n_exposures"]["value"],
            **result,
        }
    rel = [v["published_vs_replay"]["median_abs_relative_diff"]
           for v in per_record.values()
           if v["published_vs_replay"]["median_abs_relative_diff"] is not None]
    summary = {
        "n_records": len(per_record),
        "median_abs_relative_diff_range": [float(np.min(rel)), float(np.median(rel)),
                                           float(np.max(rel))] if rel else None,
        "tolerance": REPLAY_MEDIAN_REL_TOLERANCE,
        "tolerance_basis": "published CSV carries ~10 significant digits; the bound is "
                           "quantisation-derived and never fitted to replay output",
        "tolerance_breaches": [p.split(":")[0] for p in acceptance["problems"]],
        "ok": acceptance["ok"],
        "note": "agreement documents reproduction of the upstream processing chain "
                "(Preprocess.m + RawToAbsolute.m port); it is not an independent "
                "absolute calibration and must not be quoted as one",
        "filter_convention": "zero-padded sliding median over the raw 1 nm grid; "
                             "convention differences vs MATLAB medfilt1 would appear "
                             "as stage-level diffs, which are recorded per record",
    }
    if write:
        payload = {"summary": summary, "records": per_record}
        write_json(packet / "replay.json", payload)
        manifest_path = packet / "inputs.sha256.json"
        if manifest_path.is_file():   # standalone runs refresh the manifest;
            manifest = read_json(manifest_path)   # inside build it is written last
            manifest["replay.json"] = sha256_file(packet / "replay.json")
            write_json(manifest_path, manifest)
    else:
        frozen_path = packet / "replay.json"
        if frozen_path.is_file():
            frozen = read_json(frozen_path)
            if set(frozen["records"]) != set(records_json_ids(records)):
                summary["ok"] = False
                summary.setdefault("problems", []).append(
                    "frozen replay.json record set differs from the packet records")
        else:
            summary["ok"] = False
            summary.setdefault("problems", []).append(
                "packet has no frozen replay.json; run build or replay --write")
    summary["problems"] = acceptance["problems"]
    if acceptance["problems"]:
        summary["ok"] = False
    return {"ok": summary["ok"], **summary}


def records_json_ids(records: list[dict]) -> set:
    return {r["record_id"] for r in records}


# ---------------------------------------------------------------------------
# Observation operator: analytic core (plan §3.3)
# ---------------------------------------------------------------------------

def ideal_hemispheric_irradiance(radiance: float,
                                 horizon_depression_deg: float = 0.0) -> float:
    """Analytic anchor for the operator's Omega+ direction integral.

    An isotropic radiance L on a one-sided upward cosine receiver gives
    E = pi L. A horizon raised (obstructed) by h degrees - a NEGATIVE
    ``horizon_depression_deg`` - removes the band 0..h from the upper
    hemisphere and gives E = pi L cos^2(h). A horizon DEPRESSED below the
    true horizon (positive argument, open site) still gives pi L: the
    directions it admits below 0 deg elevation never illuminate the
    upward face, so they carry no weight (the negative-flux failure mode
    this anchor guards against would show up here as pi L cos^2(h)).
    """
    raised = max(-horizon_depression_deg, 0.0)
    return np.pi * radiance * np.cos(np.radians(raised)) ** 2


def publish_pipeline(spectrum_full_grid: np.ndarray, extract: dict,
                     filter_span: int = REPLAY_FILTER_SPAN) -> np.ndarray:
    """Model-facing publish steps (P_publish), with honest edge handling.

    Input: spectral irradiance on the FULL calibration grid
    (``extract["wls"]``, 150-1100 nm). Output: the published 280-840 crop,
    W m-2 nm-1. Wherever the 8-point filter window lies entirely inside the
    calibrated band, the chain is exactly the replayed upstream order -
    resample onto the calibration grid (caller's step), smooth in the
    detector-response domain (``E -> D*median(E/D)``, ``D =
    correction_factor*pi``; median filtering and wavelength-dependent
    scaling do not commute), clamp negatives, crop last.

    The edge channels are NOT closable in physical units: the upstream
    filter ran on dark-subtracted counts that extend below 280 and above
    840 nm, and no public correction factor exists there to translate a
    physical spectrum into that count-domain context. Windows touching the
    uncalibrated region therefore produce NaN instead of a fabricated edge
    value; band comparisons must use interior bands (all reference bands
    are interior). Count-domain calibration itself (C_capture) remains
    absent - see operator.json.
    """
    spectrum = np.asarray(spectrum_full_grid, dtype=float)
    d_full = np.asarray(extract["correction_factor"], dtype=float) * np.pi
    if d_full.shape != spectrum.shape:
        raise ValueError("spectrum must live on the full calibration grid "
                         f"({d_full.size} channels)")
    with np.errstate(divide="ignore", invalid="ignore"):
        response_domain = spectrum / d_full
    pad = filter_span // 2
    padded = np.pad(response_domain, pad, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, filter_span)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered = np.nanmedian(windows, axis=1)[:spectrum.size]
        # A window is only reproducible when every one of its response-domain
        # samples is finite; nanmedian would otherwise silently fall back to
        # the crop-first behaviour the upstream chain never performed.
        fully_finite = np.isfinite(windows).all(axis=1)[:spectrum.size]
    published_full = np.where(d_full * filtered < 0, 0.0, d_full * filtered)
    published_full[~fully_finite] = np.nan
    crop = (extract["wls"] >= REPLAY_CROP_NM[0]) & (extract["wls"] <= REPLAY_CROP_NM[1])
    return published_full[crop]




def hemispheric_irradiance_from_field(radiance: np.ndarray,
                                      elevation_deg: np.ndarray,
                                      azimuth_deg: np.ndarray,
                                      horizon_depression_deg: float = 0.0,
                                      angular_response=None) -> np.ndarray:
    """Linear hemisphere integral of a directional radiance field (plan §3.3).

    ``radiance`` has shape (..., n_time, n_wl, n_elev, n_azim) in
    W m-2 sr-1 nm-1; ``elevation_deg``/``azimuth_deg`` are 1-D grids in
    degrees (azimuth east of north). Returns (..., n_time, n_wl) in
    W m-2 nm-1:

        E = trapz_trapz( L(λ,Ω,t) * A(Ω) * H(Ω) * cosθ )   over Omega+

    The integration domain is the upper hemisphere Omega+ (a one-sided
    upward cosine receiver): directions below the true horizon never
    illuminate the board, so their weight is zero - never a negative
    flux, whatever ``horizon_depression_deg`` admits into visibility.
    ``H`` is the horizon visibility 1[elevation > -horizon_depression_deg]:
    a positive depression models an open site that sees below the true
    horizon (the admitted directions still carry zero weight), a negative
    one raises the horizon (terrain obstruction) and masks that band
    inside Omega+. ``angular_response`` is the white-board correction
    A(elev_deg, azim_deg) and defaults to the documented A=1 idealisation
    (no measured response exists). This is the operator contract's
    direction integral as code, so consumers integrate their model's own
    arrays instead of re-deriving it.

    Azimuth contract: coordinates strictly increasing, spanning at most
    one full circle. A grid that already spans exactly 360 deg restates
    the seam direction in its last sample, whose value must equal the
    first sample (direction space is periodic); the restated sample is
    dropped and re-attached at azim[0] + 2*pi so the wrap sector keeps
    its trapezoid weight. A shorter grid is closed periodically, with
    the wrap sector running from the last sample back to the first
    direction. Anything else (non-monotonic, over one circle, aperiodic
    seam) is rejected: silently returning a partial-hemisphere flux that
    still looks like a sky integral is exactly the failure mode this
    contract exists to prevent.
    """
    radiance = np.asarray(radiance, dtype=float)
    elev = np.asarray(elevation_deg, dtype=float)
    azim = np.asarray(azimuth_deg, dtype=float)
    if radiance.shape[-2:] != (elev.size, azim.size):
        raise ValueError("radiance grid must match (n_elev, n_azim)")
    if elev.size < 2 or not (np.diff(elev) > 0).all() \
            or not np.isfinite(elev).all() \
            or elev[0] < -90.0 - 1e-9 or elev[-1] > 90.0 + 1e-9:
        raise ValueError("elevation grid must be finite, strictly increasing, "
                         "and within [-90, 90] degrees")
    if elev[0] > 1e-9 or elev[-1] < 90.0 - 1e-9:
        raise ValueError("elevation grid must cover the integrated upper "
                         "hemisphere (first <= 0 deg, last >= 90 deg); a "
                         "partial band would silently return a "
                         "partial-hemisphere flux")
    if azim.size < 2 or not (np.diff(azim) > 0).all():
        raise ValueError("azimuth grid must be strictly increasing with "
                         ">= 2 samples")
    span = float(azim[-1] - azim[0])
    if span > 360.0 + 1e-3:
        raise ValueError("azimuth grid spans more than one full circle")
    # cos(zenith) = sin(elevation); the solid-angle measure in elevation
    # representation is dOmega = cos(elevation) d(elevation) d(phi), so the
    # integrand carries sin(elev)*cos(elev) - omitting the Jacobian doubles
    # an isotropic sky (caught by the analytic anchor, not by inspection).
    # Clipping sin(elevation) at zero restricts the integral to Omega+.
    cos_theta = np.clip(np.sin(np.radians(elev)), 0.0, None)
    jacobian = np.cos(np.radians(elev))
    horizon = (elev > -horizon_depression_deg).astype(float)
    response = (angular_response(elev[:, None], azim[None, :])
                if angular_response is not None else np.ones((elev.size, azim.size)))
    weights = cos_theta[:, None] * jacobian[:, None] * horizon[:, None] * response
    values = radiance * weights   # broadcasts weights over time/wavelength
    if abs(span - 360.0) <= 1e-3:
        # Restated seam: drop the duplicate and re-attach the first sample
        # at azim[0] + 2*pi, so the wrap sector is integrated, not dropped.
        if azim.size < 3:
            raise ValueError("a closed azimuth ring needs >= 3 samples")
        # Scale-relative only (rtol, atol=0): an absolute tolerance would
        # pass seams that are off by an order of magnitude exactly at
        # deep-twilight radiance levels (~1e-9 W m-2 sr-1 nm-1).
        seam_same = np.isclose(values[..., -1], values[..., 0],
                               rtol=1e-5, atol=0.0, equal_nan=True)
        if not bool(np.all(seam_same)):
            raise ValueError("azimuth field is not periodic at the seam: "
                             "first and last samples share one direction")
        values = np.concatenate([values[..., :-1], values[..., :1]], axis=-1)
        phi = np.concatenate([np.radians(azim[:-1]),
                              [np.radians(azim[0]) + 2.0 * np.pi]])
    else:
        # Short of the seam: close the ring periodically - the wrap sector
        # runs from the last sample back to the first direction.
        values = np.concatenate([values, values[..., :1]], axis=-1)
        phi = np.concatenate([np.radians(azim),
                              [np.radians(azim[0]) + 2.0 * np.pi]])
    integrated = np.trapezoid(values, phi, axis=-1)
    return np.trapezoid(integrated, np.radians(elev), axis=-1)


def exposure_time_average(samples: np.ndarray, sample_times: np.ndarray,
                          window: tuple[float, float]) -> float:
    """Boxcar exposure average of a measured time series (plan §3.3 w_i(t)).

    Integrates the piecewise-linear interpolation of ``(sample_times,
    samples)`` over ``window`` and divides by the window length. Unmeasured
    stretches between samples are bridged linearly (the sky is continuous);
    deliberately excluded intervals - the unmeasured gaps between a record's
    repeated exposures - are handled by calling this once per exposure window
    and averaging the results weighted by each window's duration (see
    ``record_exposure_average``).
    """
    samples = np.asarray(samples, dtype=float)
    times = np.asarray(sample_times, dtype=float)
    if times.size != samples.size or times.size < 2:
        raise ValueError("need aligned sample times/values with >= 2 samples")
    # np.interp requires ascending coordinates; an unordered series would
    # otherwise change the integral silently with the array order (r03).
    if not (np.diff(times) > 0).all():
        raise ValueError("sample times must be strictly increasing")
    lo, hi = min(window), max(window)
    if hi <= lo:
        raise ValueError("window must have positive duration")
    if not (times.min() <= lo and hi <= times.max()):
        raise ValueError("window extends beyond the measured samples")
    grid = np.unique(np.concatenate(([lo, hi], times[(times > lo) & (times < hi)])))
    values = np.interp(grid, times, samples)
    return float(np.trapezoid(values, grid) / (hi - lo))


def record_exposure_average(samples: np.ndarray, sample_times: np.ndarray,
                            exposure_seconds: list[float],
                            gaps_seconds: list[float],
                            window_start: float) -> float:
    """w_i(t) for a multi-exposure record with unknown-length gaps as scenario.

    Builds the nominal back-to-back windows (start, start+T) separated by the
    scenario gaps and returns the duration-weighted mean of the per-window
    averages of the measured series. Gaps contribute no exposure time; the
    caller decides the gap scenario (records.json timing.unknown).
    """
    gaps = list(gaps_seconds)
    if len(gaps) == len(exposure_seconds) - 1:
        gaps.append(0.0)   # gaps are BETWEEN windows; no trailing gap
    if len(gaps) != len(exposure_seconds):
        raise ValueError("gaps_seconds needs len(exposures) or len(exposures)-1")
    if any(g < 0 for g in gaps):
        raise ValueError("gaps must be non-negative")
    averages, durations = [], []
    cursor = float(window_start)
    for duration, gap in zip(exposure_seconds, gaps):
        averages.append(exposure_time_average(samples, sample_times,
                                              (cursor, cursor + duration)))
        durations.append(duration)
        cursor += duration + gap
    total = sum(durations)
    return float(sum(a * d for a, d in zip(averages, durations)) / total)


def exposure_weighted_mean(values: np.ndarray, sample_seconds: list[float],
                           gaps_seconds: list[float] | None = None) -> float:
    """Boxcar-weighted mean over the nominal exposure window (scenario).

    ``values[k]`` is the sky value during sample k (e.g. a linear ramp),
    ``sample_seconds[k]`` its duration; ``gaps_seconds[k]`` the unmeasured
    pause after sample k (scenario: zero). This is the w_i(t) integral of
    the operator contract for equal-duration repeats and any weight rule
    reduces to it only under the same-duration assumption.
    """
    gaps = gaps_seconds if gaps_seconds is not None else [0.0] * len(values)
    if len(values) != len(sample_seconds) or len(values) != len(gaps):
        raise ValueError("values, sample_seconds and gaps_seconds must align "
                         "one-to-one; silent truncation would drop exposures")
    if any(d <= 0 for d in sample_seconds):
        raise ValueError("exposure durations must be positive")
    if any(g < 0 for g in gaps):
        raise ValueError("inter-exposure gaps must be non-negative")
    total = 0.0
    weight_sum = 0.0
    for value, duration, gap in zip(values, sample_seconds, gaps):
        total += value * duration
        weight_sum += duration
    if weight_sum <= 0:
        raise ValueError("exposure window has zero total duration")
    return total / weight_sum


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
            {"id": "cal-struct-a", "kind": "whole-file-not-vendored",
             "role": "generated calibration structure for meter 'a' (dark library, "
                     "correction factor, wavelength shift) used by the replay",
             "url": CAL_STRUCT_URL, "sha256": CAL_STRUCT_SHA256,
             "bytes": CAL_STRUCT_BYTES,
             "note": "too large to vendor whole; calibration-extract.json in this packet "
                     "carries the reduced subset the records reference, tied to this "
                     "identity. Whole-struct replay from the cache is the build path.",
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
        "vendored_weather": {"kind": "response",
                             "items": [{**w, "request_url": weather_source_url(
                                 Path(w["packet_path"]).name)} for w in weather_files],
                             "note": "verbatim public responses (IEM representation of "
                                     "GHCNh, NOAA Global Hourly, AERONET, JPL Horizons "
                                     "spot checks); request_url is the exact acquisition "
                                     "request; failed probes kept as evidence"},
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
- `conditions.json`: moon geometry, per-side (dusk/dawn) weather windows and
  independent JPL axis checks for each selected night (what was queried,
  what returned, and what stays missing), with data-referenced verdicts.
- `operator.json`: the mathematical observation operator with evidence
  levels; `diagnostics.json`: timing/response/background scenario tables.
- `calibration-extract.json`: reduced subset of the pinned generated
  calibration struct (meter 'a' dark library nodes and temperature rows the
  records reference, the full correction factor and wavelength shift), tied
  to the whole-file identity of its 21.8 MB source.
- `replay.json`: per-record port of the upstream Preprocess.m +
  RawToAbsolute.m chain. It reproduces the published columns to ~5e-11
  median relative difference (CSV print precision) - proof that the raw
  members and the published spectra are the same data through the same
  pipeline, not an independent absolute calibration.

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

`replay`/`sensitivity` are read-only acceptance by default and compare against
the frozen tables plus an independent quantisation-derived tolerance;
``--write`` is the explicit re-freeze entry (build only re-freezes as part of
construction).

Acquisition (writes only to the cache directory, never to the packet). The
fetch entry covers every input build requires - official CSV, upstream MICA
tables, the meter-'a' calibration struct, IEM/NOAA/AERONET responses for all
session windows, and the two JPL spot checks; known-failing probes are
recorded as evidence and build preflight rejects an incomplete cache:

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
- `replay.json` documents reproduction of the upstream pipeline; the
  absolute radiometric scale rests on the upstream lab calibration and is
  not independently certified here.

## Attribution

Spitschan et al. (2016), Scientific Reports 6:26756,
[doi:10.1038/srep26756](https://doi.org/10.1038/srep26756): published spectra
and ephemeris tables CC BY 4.0; author code snapshots MIT (retained in the
matched-observation and feasibility packets). The raw archive and the
generated calibration structure are the author's public Figshare/GitHub
releases, referenced by the identities in `sources.json`. Scientific
interpretation and the use-verdict matrix live in
[docs/twilight/CSSP_OBSERVATION_INPUTS.md](../../../docs/twilight/CSSP_OBSERVATION_INPUTS.md).
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
    p_replay.add_argument("--write", action="store_true",
                          help="explicit re-freeze of replay.json (build-only by "
                               "default; acceptance stays read-only)")

    p_sens = sub.add_parser("sensitivity", help="recompute scenario diagnostics")
    p_sens.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    p_sens.add_argument("--write", action="store_true",
                        help="explicit re-freeze of diagnostics.json")

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
        report = refresh_diagnostics(args.packet, write=args.write)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    if args.command == "replay":
        report = run_replay(args.packet, write=args.write)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
