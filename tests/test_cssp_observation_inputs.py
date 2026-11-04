"""Independent checks for the CSSP observation-input packet.

These tests pin the packet's semantic contracts (timestamps, session
splitting, MICA interpolation bounds, raw acquisition contract, tamper
detection) without a network and without running any sky solver. Expected
values come from the frozen evidence or from analytic construction - never
from the generator's own output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cssp_observation_inputs import (  # noqa: E402
    CSSP, DRL, KN38,
    REPLAY_MEDIAN_REL_TOLERANCE,
    acquisition_items,
    band_energy,
    check_cache_inputs,
    check_external_refs,
    check_manifest,
    dark_spectrum_from_extract,
    exposure_weighted_mean,
    external_ref,
    fetch_sources,
    haversine_km,
    ideal_hemispheric_irradiance,
    invoke_time_to_utc_iso,
    label_to_invoke_time,
    label_to_utc_iso,
    load_calibration_extract,
    load_packet_spectra,
    map_columns_to_members,
    mica_elevation,
    nominal_exposure_window,
    parse_edt_label,
    parse_full_csv_rows,
    publish_pipeline,
    published_correction_factor_pi,
    quality_of_column,
    read_jpl_elevation,
    read_json,
    read_mat_contract,
    read_mica_table,
    replay_record,
    run_replay,
    session_signal_stats,
    sessions_of,
    sha256_file,
    window_states,
    summarize_weather_rows,
    unix_of,
    verify_packet,
    weather_source_url,
    weather_window_rows,
    write_json,
)

PACKET = ROOT / "validation/data/cssp_observation_inputs"
OLD_FEASIBILITY = ROOT / "validation/data/cssp_observation_feasibility"
OLD_MATCHED = ROOT / "validation/data/matched_twilight_observations"


# ---------------------------------------------------------------------------
# Timestamps: EDT labels, UTC derivation, raw invoke times
# ---------------------------------------------------------------------------

def test_edt_label_to_utc_crosses_date_correctly():
    # Dawn labels roll the UTC date forward; dusk labels share the UTC date.
    assert label_to_utc_iso("03-Jul-2014 04:24:49") == "2014-07-03T08:24:49+00:00"
    assert label_to_utc_iso("30-Jun-2014 21:06:06") == "2014-07-01T01:06:06+00:00"
    assert label_to_utc_iso("02-Jul-2014 19:38:19") == "2014-07-02T23:38:19+00:00"


def test_invoke_time_forms_round_trip():
    assert label_to_invoke_time("03-Jul-2014 04:24:49") == "20140703T042449"
    assert invoke_time_to_utc_iso("20140703T042449") == "2014-07-03T08:24:49+00:00"


@pytest.mark.parametrize("label", ["", "not-a-time", "32-Jul-2014 04:24:49"])
def test_unparseable_labels_raise(label):
    with pytest.raises(ValueError):
        parse_edt_label(label)


# ---------------------------------------------------------------------------
# Session splitting (frozen rule) and screening statistics
# ---------------------------------------------------------------------------

def _labels_from_minutes(start: datetime, minutes: list[int]) -> list[str]:
    return [(start + timedelta(minutes=int(m))).strftime("%d-%b-%Y %H:%M:%S")
            for m in minutes]


def test_sessions_split_on_gaps_and_flag_midnight_crossing():
    start = datetime(2014, 7, 2, 19, 0)
    minutes = list(range(0, 10)) + [200, 201, 202]  # 60 s cadence, 190 min hole
    sessions = sessions_of(_labels_from_minutes(start, minutes))
    assert [s["n_columns"] for s in sessions] == [10, 3]
    assert sessions[0]["local_date_of_first_sample"] == "2014-07-02"
    # A run crossing local midnight keeps the first date and flags it.
    minutes2 = list(range(290, 312))  # 23:50 -> 00:12 next day, one session
    cross = sessions_of(_labels_from_minutes(start, minutes2))[0]
    assert cross["crosses_local_midnight"] is True


def test_session_signal_stats_screen_twilight_vs_daytime():
    values = np.zeros((565, 3))
    values[0, :] = 5
    values[4:, :] = 1e-8
    values[4:, 0] = 1e-8
    values[4:84, 1] = np.nan  # one column missing the UV band
    stats = session_signal_stats(values, 0, 2)
    assert stats["no_uv_columns"] == 1
    assert stats["median_positive_w_m2_nm"] == pytest.approx(1e-8)
    values[4:, :] = 0.2
    daytime = session_signal_stats(values, 0, 2)
    assert daytime["median_positive_w_m2_nm"] == pytest.approx(0.2)
    assert daytime["no_uv_columns"] == 0


# ---------------------------------------------------------------------------
# MICA tables: interpolation inside brackets, refusal outside
# ---------------------------------------------------------------------------

def _write_synthetic_mica(path: Path) -> None:
    lines = ["2014 Jul 02 08:00:00.0    100.0    300.0   1.016600000",
             "2014 Jul 02 08:05:00.0    101.0    301.0   1.016600000",
             "2014 Jul 02 08:10:00.0    102.0    302.0   1.016600000"]
    path.write_text("\n".join(lines) + "\n", encoding="latin1")


def test_mica_interpolation_is_linear_within_bracket_and_refuses_outside(tmp_path):
    table_path = tmp_path / "fake.txt"
    _write_synthetic_mica(table_path)
    table = read_mica_table(table_path)
    # Zenith 100..102 -> elevation -10..-12; halfway through the first 5-min
    # bracket the elevation must be -10.5.
    t0 = datetime(2014, 7, 2, 8, 2, 30, tzinfo=timezone.utc).timestamp()
    hit = mica_elevation(table, t0)
    assert hit["elevation_deg"] == pytest.approx(-10.5)
    assert hit["max_bracket_minutes"] == pytest.approx(5.0)
    outside = datetime(2014, 7, 2, 9, 0, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError):
        mica_elevation(table, outside)


def test_vendored_mica_tables_reproduce_known_axis_values():
    """Full tables must agree with the matched-observation packet anchors."""
    sun = read_mica_table(PACKET / "ephemeris/CSSP_SunLocalZenith.txt")
    probe = read_json(OLD_MATCHED / "spectral/CSSP-sample-probe.json")
    sample = probe["selected_original_columns"][0]
    unix = unix_of(label_to_utc_iso("30-Jun-2014 21:06:06"))
    got = mica_elevation(sun, unix)["elevation_deg"]
    assert got == pytest.approx(sample["CSSP_Sun_reinterpolated_deg"], abs=1e-9)


# ---------------------------------------------------------------------------
# Full CSV parsing
# ---------------------------------------------------------------------------

def _synthetic_full_csv(n_cols: int) -> list[list[str]]:
    rows = []
    rows.append([f"02-Jul-2014 19:{m:02d}:00" for m in range(n_cols)] + [""])
    rows.append(["5"] * n_cols)                       # sequence
    rows.append(["-6.0"] * n_cols)                    # published sun (DRL)
    rows.append(["-40.0"] * n_cols)                   # published moon (DRL)
    rows.append(["0.3"] * n_cols)                     # moon fraction
    for wl in range(280, 841):
        row = ["NaN"] * n_cols if wl < 360 else [f"{wl}.0"] * n_cols
        rows.append(row)
    return rows


def test_parse_full_csv_rows_keeps_verbatim_cells_and_shape():
    labels, values, raw_cols = parse_full_csv_rows(_synthetic_full_csv(4))
    assert len(labels) == 4
    assert values.shape == (565, 4)
    assert int(np.isnan(values[4:, 0]).sum()) == 80   # UV rows are NaN
    assert raw_cols[0][4] == "NaN" and raw_cols[0][84] == "360.0"


def test_parse_full_csv_rejects_missing_corner_cell():
    rows = _synthetic_full_csv(4)
    rows[0][-1] = "wl"
    with pytest.raises(ValueError):
        parse_full_csv_rows(rows)


# ---------------------------------------------------------------------------
# Raw member mapping and the acquisition contract
# ---------------------------------------------------------------------------

def test_column_mapping_fails_loudly_on_missing_and_ambiguous():
    labels = ["03-Jul-2014 04:24:49"]
    member = {"name": "dataraw/CSSP/20140702/20140703T042449_sOO_USB2000+_pCSSP.mat",
              "size": 100, "type": "0"}
    with pytest.raises(ValueError, match="042549"):
        map_columns_to_members(["03-Jul-2014 04:25:49"], {"members": [member]}, [0])
    with pytest.raises(ValueError, match="ambiguous"):
        map_columns_to_members(labels, {"members": [member, dict(member)]}, [0])
    got = map_columns_to_members(labels, {"members": [member]}, [0])
    assert got[0]["upstream_name"].endswith("20140703T042449_sOO_USB2000+_pCSSP.mat")


def test_control_mat_contract_regression():
    """The frozen DRL control record keeps its documented field values."""
    contract = read_mat_contract(
        OLD_FEASIBILITY / "timing/control-DRL-20140717T184659.mat")
    assert contract["integration_time_us"]["value"] == 32000
    assert contract["integration_time_set_us"]["value"] == 32000
    assert contract["n_exposures"]["value"] == 10
    assert contract["serial_number"]["value"] == "USB2+F00926"
    assert contract["site_abbrev"]["value"] == "DRL"
    assert contract["wls_first_nm"]["value"] == pytest.approx(339.2332689, abs=1e-6)
    assert contract["board_temp_c"]["value"] == pytest.approx([36.622656, 36.622656],
                                                              abs=1e-6)


def test_target_mat_contract_differs_from_control():
    """Real CSSP targets must not inherit the control sample's configuration."""
    records = read_json(PACKET / "records.json")["records"]
    target = next(r for r in records if r["source_column_zero_based"] == 1784)
    assert target["instrument"]["serial_number"]["value"] == "USB2+H14462"
    assert target["instrument"]["serial_number"]["value"] != "USB2+F00926"
    assert target["instrument"]["site_abbrev"]["value"] == "CSSP"
    assert target["instrument"]["integration_time_us"]["value"] == 2000000
    assert target["instrument"]["n_exposures"]["value"] == 10
    # The raw wavelength range differs from the control's grating as well.
    assert target["instrument"]["wls_first_nm"]["value"] == pytest.approx(169.23, abs=0.1)
    assert target["instrument"]["wls_last_nm"]["value"] == pytest.approx(877.86, abs=0.1)


def test_nominal_window_converts_us_and_keeps_unknowns_unknown():
    contract = {"integration_time_us": {"value": 2000000},
                "n_exposures": {"value": 10}}
    window = nominal_exposure_window(contract, "2014-07-03T09:08:12+00:00")
    assert window["exposure_seconds"] == [2.0] * 10
    assert window["window_end_utc_scenario"] == "2014-07-03T09:08:32+00:00"
    assert window["nominal_total_exposure_s"] == pytest.approx(20.0)
    # Midpoint exists only inside the labelled scenario.
    assert window["midpoint_utc_scenario"] == "2014-07-03T09:08:22+00:00"
    assert window["evidence"] == "scenario"
    for unknown in window["unknown"].values():
        assert unknown["value"] is None and unknown["reason"]


def test_nominal_window_refuses_to_invent_midpoint_without_fields():
    contract = {"integration_time_us": {"value": None}, "n_exposures": {"value": None}}
    window = nominal_exposure_window(contract, "2014-07-03T09:08:12+00:00")
    assert window["midpoint_utc_scenario"] is None
    assert window["nominal_total_exposure_s"] is None


def test_records_exposure_schemes_match_interval_branches():
    """Observed (integration, repeats) pairs must be the documented branches."""
    records = read_json(PACKET / "records.json")["records"]
    pairs = {(r["instrument"]["integration_time_us"]["value"],
              r["instrument"]["n_exposures"]["value"]) for r in records}
    assert pairs <= {(512000, 10), (1000000, 10), (2000000, 10), (4000000, 10),
                     (8000000, 5), (32000000, 1), (60000000, 1)}
    # Source-code semantics: <=4 s averages 10, 8 s averages 5, 32/60 s single.
    assert all(n == 1 for t, n in pairs if t >= 32000000)
    assert all(n == 5 for t, n in pairs if t == 8000000)
    assert all(n == 10 for t, n in pairs if t <= 4000000)


def test_hard_saturation_flag_is_present_for_the_pinned_record():
    """c3114's raw spd pins the ADC ceiling (65535) - the packet must say so."""
    records = read_json(PACKET / "records.json")["records"]
    saturated = next(r for r in records if r["source_column_zero_based"] == 3114)
    assert saturated["saturation"]["suspect"] is True
    assert saturated["instrument"]["spd_max_value"]["value"] == pytest.approx(65535.0)
    clean = next(r for r in records if r["source_column_zero_based"] == 1600)
    assert clean["saturation"]["suspect"] is False


# ---------------------------------------------------------------------------
# Quality masks and band integrals
# ---------------------------------------------------------------------------

def test_quality_mask_marks_published_nan_band():
    cells = ["5", "-6.0", "-40.0", "0.3"] + ["NaN"] * 80 + [f"{i}.0" for i in range(481)]
    quality = quality_of_column(cells)
    assert quality["available_band_mask_nm"] == [[280, 359]]
    assert quality["nan_cells"] == 80
    assert quality["min_value_w_m2_nm"] == 0.0


def test_band_energy_is_trapezoid_and_nan_aware():
    wl = np.arange(280.0, 841.0)
    spectrum = np.full(561, 2.0)
    assert band_energy(wl, spectrum, (500.0, 600.0)) == pytest.approx(200.0)
    spectrum[:100] = np.nan  # 280..379 nm missing
    value = band_energy(wl, spectrum, (280.0, 500.0))
    assert value == pytest.approx(240.0)  # 380..500 contributes 2 x 120


# ---------------------------------------------------------------------------
# Weather response parsing
# ---------------------------------------------------------------------------

def test_weather_window_rows_handles_both_response_shapes(tmp_path):
    iem = {"data": [{"utc_valid": "2014-07-03T08:15Z", "vsby": 0.25,
                     "skyc1": "VV", "skyl1": 0.0, "wxcodes": "FG", "raw": "KN38 ..."},
                    {"utc_valid": "2014-07-02T10:15Z", "vsby": 10.0,
                     "skyc1": "CLR", "skyl1": None, "wxcodes": None, "raw": ""}]}
    noaa = [{"DATE": "2014-07-03T08:53:00", "VIS": "016093,5,N,5",
             "REM": "BR BKN039 OVC090", "raw": None},
            {"DATE": "2014-07-01T08:53:00", "VIS": "10", "REM": "", "raw": None}]
    cases = [(iem, ["2014-07-03T08:00:00+00:00", "2014-07-03T09:00:00+00:00"],
              lambda rows: rows[0]["wx_codes"] == "FG"),
             (noaa, ["2014-07-03T08:00:00+00:00", "2014-07-03T09:00:00+00:00"],
              lambda rows: "BR BKN039 OVC090" in (rows[0]["wx_codes"] or ""))]
    for payload, window, check in cases:
        path = tmp_path / "response.json"
        path.write_text(json.dumps(payload))
        rows = weather_window_rows(path, *window)
        assert len(rows) == 1
        assert check(rows)


# ---------------------------------------------------------------------------
# Sites and distances
# ---------------------------------------------------------------------------

def test_site_identity_and_neighbour_distances():
    assert (CSSP["lat_deg"], CSSP["lon_deg"]) == pytest.approx((41.6646, -77.8125))
    assert (CSSP["lat_deg"], CSSP["lon_deg"]) != (DRL["lat_deg"], DRL["lon_deg"])
    assert haversine_km(CSSP["lat_deg"], CSSP["lon_deg"],
                        KN38["lat_deg"], KN38["lon_deg"]) == pytest.approx(35.25, abs=0.5)


# ---------------------------------------------------------------------------
# Packet-level identity: tamper detection must fire
# ---------------------------------------------------------------------------

def test_real_packet_passes_offline_verify():
    report = verify_packet(PACKET)
    assert report["ok"], report["problems"]


def test_tampered_copy_is_detected(tmp_path):
    copy = tmp_path / "packet"
    shutil.copytree(PACKET, copy)
    target = copy / "spectra.csv"
    data = bytearray(target.read_bytes())
    data[0] = data[0] ^ 0x01
    target.write_bytes(bytes(data))
    failures = check_manifest(copy)["failures"]
    assert any(f["file"] == "spectra.csv" for f in failures)
    with pytest.raises(ValueError):
        load_packet_spectra(copy)


def test_external_ref_mismatch_is_detected(tmp_path):
    ref = dict(external_ref("docs/STATUS.md"), sha256="0" * 64)
    tmp_packet = tmp_path / "packet"
    tmp_packet.mkdir()
    write_json(tmp_packet / "sources.json", {"external_refs": [ref]})
    result = check_external_refs(tmp_packet)
    assert result["failures"] and result["failures"][0]["problem"] == "hash mismatch"


# ---------------------------------------------------------------------------
# Conditions verdicts and independent axis checks (M2 semantics)
# ---------------------------------------------------------------------------

def test_conditions_verdicts_match_frozen_evidence():
    """The primary/secondary roles must follow the archived weather rows."""
    conditions = read_json(PACKET / "conditions.json")
    s04 = conditions["conditions"]["s04"]
    s11 = conditions["conditions"]["s11"]
    assert s04["target_windows"]["dawn"]["moon_free"] is True
    assert s04["target_windows"]["dusk"]["moon_free"] is False
    assert "fog/low-cloud" in s04["verdict"]
    assert s11["target_windows"]["dusk"]["moon_free"] is True
    assert s11["target_windows"]["dawn"]["moon_free"] is False
    # Single authority: the window verdict must agree with its own stored
    # weather_states (this exact pair was contradictory in review r01/r02).
    for entry in conditions["conditions"].values():
        for window in entry["target_windows"].values():
            states = window["weather_states"]
            if states["all_clear"] and window["moon_free"]:
                assert window["verdict"].startswith("cleanest-available")
            if not window["moon_free"]:
                assert window["verdict"].startswith("moonlit")


def test_independent_axis_checks_present_and_recomputable():
    checks = read_json(PACKET / "conditions.json")["independent_axis_checks"]
    assert len(checks) == 2
    sun = read_mica_table(PACKET / "ephemeris/CSSP_SunLocalZenith.txt")
    for check in checks:
        jpl = read_jpl_elevation(PACKET / check["response_path"])
        assert jpl == pytest.approx(check["jpl_elevation_deg"], abs=1e-9)
        mica = mica_elevation(sun, unix_of(check["instant_utc"]))["elevation_deg"]
        assert abs(jpl - mica) < check["agreement_bound_deg"]


def test_weather_classification_and_context_fallback():
    fog = summarize_weather_rows(
        [{"utc": "2014-07-03T08:15Z", "visibility_mi_or_m": 0.25, "cloud_cover": "VV",
          "cloud_height_ft": 0.0, "wx_codes": "FG"}], "KN38")
    assert fog["fog"] and not fog["clear"]
    clear = summarize_weather_rows(
        [{"utc": "2014-07-02T23:15Z", "visibility_mi_or_m": 10.0, "cloud_cover": "CLR",
          "cloud_height_ft": None, "wx_codes": None}], "KN38")
    assert clear["clear"] and not fog["clear"]
    haze = summarize_weather_rows(
        [{"utc": "2014-07-03T08:53:00", "visibility_mi_or_m": "016093,5,N,5",
          "cloud_cover": None, "cloud_height_ft": None,
          "wx_codes": "KBFD 030853Z 00000KT 5SM BR BKN039 OVC090 16/15 A3000"}], "KBFD")
    assert haze["other"] and not haze["fog"]
    empty = {"n_rows": 0, "fog": [], "low_opaque": [], "clear": [], "other": []}
    def summary_of(s):
        return {"summary": s, "context_summary": s}
    states = window_states({"weather": {
        "KN38": summary_of(clear), "KBFD": summary_of(empty)}})
    assert states["stations"]["KN38"]["state"] == "CLR"
    assert states["stations"]["KBFD"]["state"] == "no-observation"
    assert states["evidence_scope"] == "window"
    mixed = window_states({"weather": {
        "KN38": {"summary": empty, "context_summary": clear},
        "KBFD": summary_of(clear)}})
    assert mixed["stations"]["KN38"]["state"] == "CLR"
    assert mixed["stations"]["KN38"]["evidence_scope"] == "context"
    assert mixed["all_clear"] is True
    assert mixed["evidence_scope"] == "mixed"


# ---------------------------------------------------------------------------
# Observation operator analytic anchors and calibration replay (M3)
# ---------------------------------------------------------------------------

def test_operator_isotropic_sky_gives_pi_l():
    assert ideal_hemispheric_irradiance(2.0) == pytest.approx(np.pi * 2.0)


def test_operator_flat_horizon_gives_cos_squared():
    for h in (0.0, 1.0, 2.0, 5.0, 10.0):
        assert ideal_hemispheric_irradiance(1.0, h) == \
            pytest.approx(np.pi * np.cos(np.radians(h)) ** 2)


def test_exposure_weighted_mean_matches_analytic_ramp_with_gaps():
    # Linear ramp L(t) = 1 + t over samples of 2 s with a 3 s gap:
    # duration-weighted mean of the sample values (not a time-average across
    # the gap) is the scenario the operator contract defines.
    values = np.array([1.0, 3.0])
    got = exposure_weighted_mean(values, [2.0, 2.0], gaps_seconds=[3.0, 0.0])
    assert got == pytest.approx((1.0 * 2.0 + 3.0 * 2.0) / 4.0)
    with pytest.raises(ValueError):
        exposure_weighted_mean(values, [2.0, 2.0], gaps_seconds=[3.0])
    # Equal weights fall out when durations are equal and gaps vanish.
    assert exposure_weighted_mean(np.array([2.0, 6.0]), [1.0, 1.0]) == 4.0
    with pytest.raises(ValueError):
        exposure_weighted_mean(np.array([1.0]), [0.0])


def test_dark_extract_semantics_follow_getdarkspectrum():
    cal = {"dark_nodes": {
        2000000.0: (np.array([25.0, 30.0]), np.array([[10.0] * 4, [30.0] * 4])),
        60000000.0: (np.array([25.0, 30.0]), np.array([[20.0] * 4, [40.0] * 4])),
    }}
    # Nearest node wins: 32 s (32_000_000 us) resolves to the 60 s library row;
    # 27.5 C interpolates linearly between 25 C (20) and 30 C (40) -> 30.
    dark = dark_spectrum_from_extract(cal, 32000000.0, 27.5)
    assert np.allclose(dark, 30.0)
    with pytest.raises(ValueError):
        dark_spectrum_from_extract(cal, 2000000.0, 50.0)


def test_replay_reproduces_published_columns_to_print_precision():
    """The ported upstream chain must reproduce the published excerpts."""
    replay = read_json(PACKET / "replay.json")
    assert replay["summary"]["n_records"] == 42
    for record_id in ("c1743", "c1765", "c1784", "c3114"):
        entry = replay["records"][record_id]
        pv = entry["published_vs_replay"]
        assert pv["median_abs_relative_diff"] < 1e-6
        assert pv["replay_zero_after_clamp"] == pv["published_zero_cells"]
    # The hard-saturated record stays flagged even though upstream kept it.
    assert replay["records"]["c3114"]["stages"]["dark"][
        "saturated_pixels_mean_equals_65535"] == 0


def test_calibration_extract_covers_every_record_node_and_temperature():
    extract = load_calibration_extract(PACKET)
    assert extract["wl_shift"] == pytest.approx(-0.789232450342638, abs=1e-9)
    records = read_json(PACKET / "records.json")["records"]
    for record in records:
        t_us = record["instrument"]["integration_time_us"]["value"]
        node_us = min(extract["dark_nodes"], key=lambda u: abs(u - t_us))
        temps, _ = extract["dark_nodes"][node_us]
        temp = float(np.mean(record["instrument"]["board_temp_c"]["value"]))
        assert temps.min() <= temp <= temps.max()


# ---------------------------------------------------------------------------
# Revision round 01: acceptance/writer separation, fetch contract, screening
# coverage, operator domains, signed sensitivity
# ---------------------------------------------------------------------------

def test_full_source_mismatch_fails_and_correct_source_passes(tmp_path):
    report = verify_packet(PACKET)   # no source dir: full check skipped
    assert report["ok"]
    bad = tmp_path / "CSSP_spectra.csv"
    bad.write_text("not the official CSV")
    report = verify_packet(PACKET, bad.parent)
    assert not report["ok"]
    assert any("full-source CSV identity mismatch" in p for p in report["problems"])
    truncated = tmp_path / "CSSP_spectra.csv"
    truncated.write_bytes(b"x" * 1024)
    report = verify_packet(PACKET, truncated.parent)
    assert not report["ok"]


def test_replay_read_only_acceptance_never_writes_and_detects_scaling_errors(tmp_path):
    copy = tmp_path / "packet"
    shutil.copytree(PACKET, copy)
    before = {p.name: sha256_file(p) for p in copy.rglob("*") if p.is_file()}
    report = run_replay(copy, write=False)
    assert report["ok"]
    after = {p.name: sha256_file(p) for p in copy.rglob("*") if p.is_file()}
    assert before == after, "read-only replay must not touch any packet byte"
    # A calibration regression (correction factor x2) must breach the
    # independent tolerance and still not rewrite anything.
    extract = read_json(copy / "calibration-extract.json")
    extract["correction_factor"] = [v * 2.0 for v in extract["correction_factor"]]
    write_json(copy / "calibration-extract.json", extract)
    manifest = read_json(copy / "inputs.sha256.json")
    manifest["calibration-extract.json"] = hashlib.sha256(
        (copy / "calibration-extract.json").read_bytes()).hexdigest()
    write_json(copy / "inputs.sha256.json", manifest)
    tampered = {p.name: sha256_file(p) for p in copy.rglob("*") if p.is_file()}
    report = run_replay(copy, write=False)
    assert not report["ok"]
    assert report["tolerance_breaches"], "x2 scaling must breach the tolerance"
    after2 = {p.name: sha256_file(p) for p in copy.rglob("*") if p.is_file()}
    assert tampered == after2, "failing read-only replay must not re-sign anything"


def test_verify_enforces_independent_replay_tolerance(tmp_path):
    copy = tmp_path / "packet"
    shutil.copytree(PACKET, copy)
    extract = read_json(copy / "calibration-extract.json")
    extract["correction_factor"] = [v * 2.0 for v in extract["correction_factor"]]
    write_json(copy / "calibration-extract.json", extract)
    report = verify_packet(copy)
    assert not report["ok"]
    assert report["replay_acceptance"]["tolerance"] == REPLAY_MEDIAN_REL_TOLERANCE


def test_fetch_creates_parent_dirs_and_records_provenance(tmp_path, monkeypatch):
    """Directory creation and provenance mechanics with stubbed network."""
    import urllib.error
    import urllib.request
    import io

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        if "calibration/" in url:
            return FakeResponse(b"fake-cal")
        if "obhistory" in url:
            return FakeResponse(b'{"data": []}')
        if "horizons.api" in url:
            return FakeResponse(b'{"result": ""}')
        if "daily_history" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return FakeResponse(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cache = tmp_path / "cache"
    items = [
        {"id": "cal-struct-a", "url": "https://example.com/calibration/cal.mat",
         "path": "calibration/OO_USB2000+a.mat"},
        {"id": "iem-20140630",
         "url": "https://example.com/obhistory.json?station=N38",
         "path": "weather/iem-N38-obhistory-20140630.json"},
        {"id": "jpl-0703", "url": "https://example.com/horizons.api?x=1",
         "path": "jpl/horizons-cssp-0703.json"},
        {"id": "iem-daily-20140712",
         "url": "https://example.com/daily_history.json",
         "path": "weather/iem-N38-daily-20140712-month.json",
         "optional": True},
    ]
    result = fetch_sources(cache, with_raw_archive=False, items=items)
    assert (cache / "calibration").is_dir(), "parent dirs must be created"
    assert (cache / "weather").is_dir() and (cache / "jpl").is_dir()
    provenance = read_json(cache / "fetch-provenance.json")
    failed = [e for e in provenance if e.get("identity") == "failed-probe"]
    verified = [e for e in provenance if e.get("identity") == "verified"]
    assert len(failed) == 1 and len(verified) == 3
    assert all(e["url"] for e in verified)
    assert "Not Found" in failed[0]["error"]


def test_build_preflight_rejects_incomplete_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="missing required inputs"):
        check_cache_inputs(cache)   # completely empty: no directory trick passes
    items = [i for i in acquisition_items(with_raw_archive=True)
             if not i.get("optional") and not i["id"].startswith("code-")]
    for item in items[:-1]:   # satisfy all but the last required file
        target = cache / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    with pytest.raises(ValueError, match=items[-1]["path"].split("/")[-1]):
        check_cache_inputs(cache)
    target = cache / items[-1]["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    check_cache_inputs(cache)
    # A documented substitute covers exactly one cache path (the NOAA 7/3
    # full-day file, whose endpoint timed out); any other missing response
    # still fails - absence is never upgraded into coverage.
    partial = tmp_path / "cache2"
    partial.mkdir()
    for item in items:
        if item["path"] == "weather/noaa-BFD-20140703.json":
            continue   # substitute-eligible
        target = partial / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    check_cache_inputs(partial)   # substitute exists in the repo tree
    (partial / "weather/noaa-BFD-20140702.json").unlink()
    with pytest.raises(ValueError, match="noaa-BFD-20140702.json"):
        check_cache_inputs(partial)


def test_window_verdicts_independent_of_raw_sample_set(tmp_path):
    """Verdicts derive from the uniform depth-window rule, not the raw picks.

    Synthetic dusk session evaluated twice - once with a raw pick subset,
    once without - must produce identical window verdicts and moon flags.
    """
    from tools.cssp_observation_inputs import build_conditions
    # Synthetic MICA table: sun descending -5 deg per 5 min across the span.
    base = datetime(2014, 7, 12, 20, 0)
    lines = []
    for k in range(25):
        # Labels are EDT wall clock; the MICA table is UTC, so shift +4 h.
        moment = base + timedelta(minutes=5 * k) + timedelta(hours=4)
        zenith = 90.0 + 1.0 * k          # elevation -0 .. -24 deg
        lines.append(moment.strftime("%Y %b %d %H:%M:%S.0")
                     + f"    {zenith:.5f}    300.0   1.016600000")
    table_path = tmp_path / "synthetic-sun.txt"
    table_path.write_text("\n".join(lines) + "\n", encoding="latin1")
    sun_tab = read_mica_table(table_path)
    moon_tab = read_mica_table(table_path)   # same shape; moon flag check only
    out = tmp_path / "pkg"
    (out / "raw/weather").mkdir(parents=True)
    labels = [(base + timedelta(minutes=k)).strftime("%d-%b-%Y %H:%M:%S")
              for k in range(21)]
    values = np.zeros((565, 21))
    values[3, :] = 0.3
    inventory = [{
        "session_id": "sX", "columns_zero_based": [0, 20], "n_columns": 21,
        "start_label_edt": labels[0], "end_label_edt": labels[20],
        "start_utc": label_to_utc_iso(labels[0]),
        "end_utc": label_to_utc_iso(labels[20]),
        "span_minutes": 20.0, "crosses_local_midnight": False,
        "local_date_of_first_sample": "2014-07-12",
        "signal_kind": "twilight-or-night",
        "sequence_values": [1],
    }]
    with_picks = build_conditions(out, labels, values, inventory, sun_tab,
                                  moon_tab, selected_by_session={"sX": [10, 11]})
    without = build_conditions(out, labels, values, inventory, sun_tab,
                               moon_tab, selected_by_session={})
    w1 = with_picks["conditions"]["sX"]["target_windows"]["dusk"]
    w2 = without["conditions"]["sX"]["target_windows"]["dusk"]
    assert w1["verdict"] == w2["verdict"]
    assert w1["moon_free"] == w2["moon_free"]
    assert w1["columns_zero_based"] == w2["columns_zero_based"]


def test_all_twilight_sessions_screened_and_selection_derived():
    conditions = read_json(PACKET / "conditions.json")
    selection = read_json(PACKET / "selection.json")
    twilight = {s["session_id"] for s in selection["sessions"]
                if s["signal_kind"] == "twilight-or-night"}
    assert twilight <= set(conditions["conditions"]), \
        "every twilight-level session must carry a conditions verdict"
    assert set(conditions["not_conditions_evaluated"]) == \
        {s["session_id"] for s in selection["sessions"]
         if s["signal_kind"] != "twilight-or-night"}
    # Selection reasons quote the conditions verdicts (single authority).
    for decision in selection["decisions"]:
        sid = decision["session_id"]
        if sid in conditions["conditions"]:
            assert decision["reason"].startswith(
                tuple({"selected": ("selected for raw sampling",),
                       "not-selected": ("conditions verdict:",)}[decision["decision"]]))
            assert conditions["conditions"][sid]["verdict"] in decision["reason"]
    # 'Cleanest' claims come from the same verdicts, and s11/dusk is the
    # unique moon-free clear window among the screened set.
    assert selection["cleanest_screened_windows"] == ["s11/dusk"]


def test_signed_sensitivity_rates_and_forward_operator_domain():
    import numpy as np
    diagnostics = read_json(PACKET / "diagnostics.json")
    c1743 = diagnostics["adjacent_change"]["c1743"]
    rate = c1743["signed_rate_per_minute_per_band"]["visible_all"]
    assert rate > 0, "dawn brightening must carry a positive time-derivative sign"
    assert "+30" in c1743["timing_offset_scenario_per_band"], \
        "signed offsets (+/- minutes) must both be tabulated"
    assert c1743["timing_offset_scenario_per_band"]["+30"]["visible_all"] > 0
    assert c1743["timing_offset_scenario_per_band"]["-30"]["visible_all"] < 0
    # Operator contract separates the model-facing publish steps from the
    # count-domain calibration chain.
    operator = read_json(PACKET / "operator.json")
    assert "P_publish" in operator["definitions"]
    assert "C_capture" in operator["definitions"]
    assert "P_publish" not in operator["definitions"] or \
        "correction factor" not in operator["definitions"]["P_publish"]["text"]
    assert any("double-calibrate" in f for f in operator["forbidden_comparisons"])
    # P_publish smooths in the detector-response domain (E -> D*median(E/D)):
    # a constant spectrum passes unscaled; with a wavelength-dependent D the
    # result equals the response-domain reference and differs from a naive
    # physical-domain median filter (the two operations do not commute -
    # exactly the inequality the r02 review probed numerically).
    d = published_correction_factor_pi(PACKET)
    assert d.shape == (561,)
    # A constant physical sky does NOT publish to a constant spectrum: the
    # published smoothing runs in the response domain, so D*median(c/D)
    # carries a real (upstream-inherited) spectral distortion. The contract
    # is the mapping itself, verified against its reference below.
    flat = np.full(561, 2.5)
    padded_flat = np.pad(flat / d, (4, 4), mode="constant", constant_values=np.nan)
    assert publish_pipeline(flat, d) == pytest.approx(
        d * np.asarray([np.nanmedian(padded_flat[k:k + 8]) for k in range(561)]),
        nan_ok=True)
    ramp = np.linspace(0.5, 5.0, 561)
    padded_response = np.pad(ramp / d, (4, 4), mode="constant",
                             constant_values=np.nan)
    reference = d * np.asarray([np.nanmedian(padded_response[k:k + 8])
                                for k in range(561)])
    assert publish_pipeline(ramp, d) == pytest.approx(reference, nan_ok=True)
    padded_physical = np.pad(ramp, (4, 4), mode="constant",
                             constant_values=np.nan)
    naive = np.asarray([np.nanmedian(padded_physical[k:k + 8])
                        for k in range(561)])
    assert not np.allclose(publish_pipeline(ramp, d), naive)


def test_weather_source_urls_reconstruct_acquisition_requests():
    assert weather_source_url("iem-N38-obhistory-20140703.json") is not None
    assert "obhistory.json" in weather_source_url("iem-N38-obhistory-20140703.json")
    assert "global-hourly" in weather_source_url("noaa-BFD-20140703.json")
    assert "print_web_data_v3" in weather_source_url(
        "aeronet-Toronto-20140718-19-AOD20.txt")
    assert weather_source_url("frozen-noaa-BFD-window.json") is None


def test_spectra_excerpts_match_old_packet_overlap():
    """Overlap columns must numerically match the first-round frozen excerpts."""
    spectra = load_packet_spectra(PACKET)
    with (OLD_MATCHED / "spectral/selected-spectra.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    old_columns = [int(name.removeprefix("col")) for name in rows[0][1:]]
    old = {c: {int(float(row[0])): float(row[1 + i]) for row in rows[1:]}
           for i, c in enumerate(old_columns)}
    wl = spectra["wavelength_nm"]
    n_checked = 0
    for i, col in enumerate(spectra["columns"]):
        if col not in old:
            continue
        for k, wavelength in enumerate(wl):
            assert float(spectra["data"][k, i]) == pytest.approx(old[col][int(wavelength)],
                                                                 rel=1e-12, abs=0.0)
            n_checked += 1
    assert n_checked > 0
