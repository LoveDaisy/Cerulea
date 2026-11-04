"""Tests for the deep-twilight reference benchmark.

Three layers are covered separately:

* the opt-in Monte Carlo diagnostics sink (bit-identity, component closure,
  default-off payload absence) -- real physics on a small atmosphere;
* the protocol/batch harness (seed rule, resume/attempt semantics, record
  invariants) -- synthetic records where cost would buy no extra evidence;
* the CLI and summarizer end-to-end on a synthetic mini-dataset plus one
  real CLI smoke run and one real disk/point fingerprint pairing check.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets
from skycolor.rt import mc
from tools.deep_twilight_reference import (
    Batch, load_protocol, run_record, seed_for, DEFAULT_PROTOCOL)
from tools.summarize_deep_twilight_reference import (
    compute_summary, load_batch, _deep_equal, KNOWN_PROTOCOL_GENERATIONS)

KNOWN_SHA = next(iter(KNOWN_PROTOCOL_GENERATIONS))

WL3 = np.array([450.0, 550.0, 650.0])


# ---------------------------------------------------------------------------
# diagnostics sink: bit identity and closure
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def atmo3():
    return presets.clear(WL3)


def test_mc_diagnostics_do_not_change_fields_or_rng(atmo3):
    off = mc.mc_radiance_split(atmo3, 99.0, [0.0], [0.0], n_photons=20_000, seed=7)
    on = mc.mc_radiance_split(atmo3, 99.0, [0.0], [0.0], n_photons=20_000,
                              seed=7, diagnostics=True)
    per_path = mc.mc_radiance_split(atmo3, 99.0, [0.0], [0.0], n_photons=20_000,
                                    seed=7, diagnostics=dict(keep_per_path=True))
    assert set(off) == {"total", "single", "multiple", "total_err",
                        "single_err", "multiple_err"}
    for key in off:
        assert np.array_equal(off[key], on[key]), key
        assert np.array_equal(off[key], per_path[key]), key


def test_mc_diagnostics_component_closure(atmo3):
    result = mc.mc_radiance_split(atmo3, 99.0, [0.0], [0.0], n_photons=20_000,
                                  seed=7, diagnostics=dict(keep_per_path=True))
    n_paths = 20_000
    for iw in range(len(WL3)):
        d = result["diagnostics"][iw]
        pv = d["per_view"][0]
        c_mult = np.array(pv["per_path"]["multiple"])
        c2 = np.array(pv["per_path"]["second_no_touch"])
        rest = np.array(pv["per_path"]["rest"])
        # decomposition is exact, not floating-point-close
        assert np.array_equal(c2 + rest, c_mult)
        # order histograms recombine the multiple part bit-for-bit in sum
        hist = (np.array(d["nee_order_hist"]).sum()
                + np.array(d["ground_order_hist"]).sum())
        np.testing.assert_allclose(hist, result["multiple"][0, iw] * n_paths,
                                   rtol=1e-12)
        # moments agree with the per-path arrays; sums add up
        np.testing.assert_allclose(pv["s1"][0], c_mult.sum(), rtol=1e-12)
        np.testing.assert_allclose(pv["s1"][1] + pv["s1"][2], pv["s1"][0],
                                   rtol=1e-12)
        if pv["s2"] is not None:
            s2 = np.array(pv["s2"])
            for j, cj in enumerate((c_mult, c2, rest)):
                for k, ck in enumerate((c_mult, c2, rest)):
                    np.testing.assert_allclose(s2[j, k], float((cj * ck).sum()),
                                               rtol=1e-12)
            assert pv["top"] is not None
            assert np.isclose(max(pv["top"]["values"]), c_mult.max())
        else:
            # an all-zero wavelength is reported as "no tail evidence"
            assert result["multiple"][0, iw] == 0.0
            assert pv["top"] is None and pv["top001_fraction"] is None


def test_mc_diagnostics_black_ground_accounting():
    # albedo 0 (black ground): ground events absorb the path, so the kill
    # counter must record them, no ground continuation may survive, and the
    # two NEE histograms must still recombine the whole multiple part.
    from skycolor.atmosphere import Atmosphere
    atmo = presets.clear(WL3)
    black = Atmosphere(atmo.wl, ozone_du=300.0,
                       aerosols=list(atmo.aerosols), ground_albedo=0.0)
    result = mc.mc_radiance_split(black, 99.0, [0.0], [0.0], n_photons=10_000,
                                  seed=3, diagnostics=True)
    for iw in range(len(WL3)):
        d = result["diagnostics"][iw]
        hist = (np.array(d["nee_order_hist"]).sum()
                + np.array(d["ground_order_hist"]).sum())
        np.testing.assert_allclose(hist, result["multiple"][0, iw] * 10_000,
                                   rtol=1e-12)
    ground_absorbed = sum(d["kills"]["ground_absorbed"]
                          for d in result["diagnostics"])
    assert ground_absorbed > 0, "test scenario must actually touch the ground"


# ---------------------------------------------------------------------------
# protocol + seed rule
# ---------------------------------------------------------------------------
def test_seed_rule_matches_plan_and_is_unique():
    assert seed_for(0, 0, 0) == 40000
    assert seed_for(0, 1, 15) == 41015
    assert seed_for(1, 0, 15) == 50015
    assert seed_for(1, 1, 15) == 51015
    seeds = [seed_for(s, b, r) for s in (0, 1) for b in (0, 1)
             for r in range(16)]
    assert len(set(seeds)) == 64


def test_load_protocol_contract(tmp_path):
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    loaded = load_protocol(DEFAULT_PROTOCOL)  # the frozen file itself
    assert loaded["schema"] == "deep-twilight-reference/1"
    bad = dict(protocol, schema="other/1")
    p = tmp_path / "p.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema"):
        load_protocol(p)
    bad = json.loads(DEFAULT_PROTOCOL.read_text())
    bad["scenario"]["spectrum"] = {"grid": [400.0, 700.0, 5.0], "n_wavelengths": 61}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="contract"):
        load_protocol(p)
    bad = json.loads(DEFAULT_PROTOCOL.read_text())
    del bad["precision"]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing"):
        load_protocol(p)
    bad = json.loads(DEFAULT_PROTOCOL.read_text())
    bad["scenario"]["anchor_wavelengths_nm"] = [451.0, 550.0, 650.0]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="not on the spectral grid"):
        load_protocol(p)


# ---------------------------------------------------------------------------
# batch bookkeeping: resume, attempts, refusal semantics
# ---------------------------------------------------------------------------
def _mini_meta(tmp_path):
    return dict(stage="direct", created="t", code_sha256={}, hardware={})


def test_batch_resume_skips_present_seeds(tmp_path):
    batch = Batch(tmp_path, "b0", [1, 2, 3], _mini_meta(tmp_path))
    batch.record_seed(1, {"seed": 1})  # left open, as after a crash
    reopened = Batch(tmp_path, "b0", [1, 2, 3], _mini_meta(tmp_path))
    assert reopened.done == {1}
    reopened.record_seed(2, {"seed": 2})
    assert reopened.attempt == 0  # continued, not duplicated
    rows = [json.loads(line)
            for line in reopened.path.read_text().splitlines()]
    assert [r["seed"] for r in rows] == [1, 2]


def test_batch_completed_is_adopted_not_recomputed(tmp_path):
    batch = Batch(tmp_path, "b4", [1, 2], _mini_meta(tmp_path))
    batch.record_seed(1, {"seed": 1})
    batch.record_seed(2, {"seed": 2})
    batch.close("complete")
    reopened = Batch(tmp_path, "b4", [1, 2], _mini_meta(tmp_path))
    assert reopened.done == {1, 2}
    assert reopened.attempt == 0
    assert len(reopened.meta["attempts"]) == 1  # no second attempt spawned


def test_batch_corrupt_tail_opens_new_attempt(tmp_path):
    batch = Batch(tmp_path, "b1", [1, 2], _mini_meta(tmp_path))
    batch.record_seed(1, {"seed": 1})
    with batch.path.open("a") as stream:  # simulate a crash mid-write
        stream.write('{"seed": 2, "trunc')
    reopened = Batch(tmp_path, "b1", [1, 2], _mini_meta(tmp_path))
    assert reopened.attempt == 1
    assert reopened.done == set()
    assert reopened.meta["attempts"][0]["status"] == "invalid-corrupt"
    assert reopened.meta["attempts"][-1]["status"] == "open"
    # the runner now recomputes the *full* pre-registered seed set into the
    # new attempt (seed 1 is deterministic, so recomputation is exact, but it
    # is a new attempt, never a silently merged partial one)
    for seed in (1, 2):
        reopened.record_seed(seed, {"seed": seed})
    reopened.close("complete")
    records, meta = load_batch(tmp_path / "b1", [1, 2], stage="direct",
                               batch_id="b1")
    assert [int(r["seed"]) for r in records] == [1, 2]
    assert meta["attempts"][0]["status"] == "invalid-corrupt"


def test_summarizer_refuses_two_complete_attempts(tmp_path):
    batch = Batch(tmp_path, "b2", [1], _mini_meta(tmp_path))
    batch.record_seed(1, {"seed": 1})
    batch.close("complete")
    # a second complete attempt: duplicated recomputation must not merge
    second = tmp_path / "b2" / "attempt-01.jsonl"
    second.write_text(json.dumps({"seed": 1}) + "\n")
    meta_path = tmp_path / "b2" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["attempts"].append(dict(index=1, status="complete"))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="complete attempts"):
        load_batch(tmp_path / "b2", [1], stage="direct", batch_id="b2")


def test_summarizer_refuses_seed_deviation(tmp_path):
    batch = Batch(tmp_path, "b3", [1, 2], _mini_meta(tmp_path))
    batch.record_seed(1, {"seed": 1})
    batch.record_seed(3, {"seed": 3})  # 3 is not pre-registered
    batch.close("complete")
    with pytest.raises(ValueError, match="deviate"):
        load_batch(tmp_path / "b3", [1, 2], stage="direct", batch_id="b3")


# ---------------------------------------------------------------------------
# summarizer end-to-end on a synthetic mini-dataset
# ---------------------------------------------------------------------------
def _synthetic_record(seed, zero=False, rng=None):
    rng = rng or np.random.default_rng(seed)
    single = np.full(6, 1.0e-8)
    multiple = np.zeros(6) if zero else rng.uniform(0.0, 1.0e-9, 6)
    total = single + multiple
    tail = {"max_fraction": 0.004, "top001_fraction": 0.2,
            "effective_contributors": 50.0, "nonzero": 3, "capped_paths": 0,
            "escaped": 1, "ground_absorbed": 0, "absorbed": 0,
            "rr_weight": 0, "rr_length": 0, "order_cap": 0, "wl": 550.0}
    anchors = {}
    for a in (450.0, 550.0, 650.0):
        anchors[str(a)] = dict(wl=a, n_paths=100, s1=[1e-9, 4e-10, 6e-10],
                               s2=[[1e-18, 0, 0], [0, 1e-19, 0], [0, 0, 2e-19]],
                               top=None, max_fraction=0.004,
                               top001_fraction=0.2, effective_contributors=50.0,
                               nee_order_hist=[[0.0, 0.0]] * 8,
                               ground_order_hist=[[0.0, 0.0]] * 8,
                               kills={}, capped_paths=0)
    return dict(seed=seed, sun_elevation_deg=-9.0, photons_per_wavelength=400,
                mode="disk", seconds=0.1, max_rss_bytes=1,
                fields={k: v.tolist() for k, v in
                        (("total", total), ("total_err", total * 0.01),
                         ("single", single), ("single_err", single * 0.01),
                         ("multiple", multiple), ("multiple_err", multiple * 0.01))},
                deterministic_single=single.tolist(),
                split_total=(single + multiple).tolist(),
                fingerprint="fab", nee_calls=1, nee_points=1,
                diagnostics=dict(per_wl=[tail], anchors=anchors))


MINI_PROTOCOL = {
    "schema": "deep-twilight-reference/1",
    "scenario": {"preset": "clear", "anchor_wavelengths_nm": [450.0, 550.0, 650.0],
                 "sun_azimuth_deg": 0.0, "sun_elevations_deg": [-9.0],
                 "spectrum": {"grid": [380.0, 780.0, 5.0], "n_wavelengths": 81}},
    "source": {"full81_table": {"path": "x", "label": "x", "sha256": "t"},
               "point_pairing": {"radius_deg": 0.0}},
    "direct": {"budgets_photons": [400, 800], "repeats": 4,
               "analysis": {"student": {"confidence": 0.95, "df": 3},
                            "bootstrap": {"resamples": 200, "seed": 90000}}},
    "precision": {"anchor_scales": {"-9.0": [1e-3, 1e-3, 1e-3]},
                  "anchor_scale_order": [450.0, 550.0, 650.0],
                  "anchor_halfwidth_factor": 0.5,
                  "luminance_relative_halfwidth": 0.5,
                  "drift_factor": 0.5, "max_path_fraction": 0.01,
                  "top001_fraction": 0.5},
    "luminance": {"constant": 683.0, "method": "test"},
    "wall_clock_caps_hours": {"direct_and_source_pairs": 1.0, "candidate": 1.0,
                              "sensitivity": 1.0},
    "verdict_policy": {},
}


def _write_mini_dataset(tmp_path):
    """Mini data directory with the mini protocol (synthetic, 81-wide records)."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "protocol.json").write_text(json.dumps(MINI_PROTOCOL))
    return data


# The invariants check len == 81, so synthetic records are 81 wide.
def _synthetic_record81(seed, zero=False):
    row = _synthetic_record(seed, zero=zero)
    rng = np.random.default_rng(seed)
    single = np.full(81, 1.0e-8)
    multiple = np.zeros(81) if zero else rng.uniform(0.0, 1.0e-9, 81)
    total = single + multiple
    row = dict(row)
    row["fields"] = {k: v.tolist() for k, v in
                     (("total", total), ("total_err", total * 0.01),
                      ("single", single), ("single_err", single * 0.01),
                      ("multiple", multiple), ("multiple_err", multiple * 0.01))}
    row["deterministic_single"] = single.tolist()
    row["split_total"] = (single + multiple).tolist()
    return row


def _populate(data: Path, *, zero_one_record=False, point=True):
    from tools.deep_twilight_reference import contract_view
    meta = contract_view(dict(stage="direct", table_sha256="t",
                              protocol_sha256=KNOWN_SHA, mode="disk"))
    for sun_index, sun in enumerate([-9.0]):
        for budget_index, photons in enumerate([400, 800]):
            batch_id = f"s{sun_index}b{budget_index}"
            batch = Batch(data / "raw" / "direct", batch_id,
                          [40000 + 10000 * sun_index + 1000 * budget_index + r
                           for r in range(4)], dict(meta))
            for r in range(4):
                seed = 40000 + 10000 * sun_index + 1000 * budget_index + r
                zero = zero_one_record and seed == 40000
                row = _synthetic_record81(seed, zero=zero)
                row["table_sha256"] = "t"
                row["protocol_sha256"] = KNOWN_SHA
                row["mode"] = "disk"
                row["photons_per_wavelength"] = photons
                row["sun_elevation_deg"] = -9.0
                batch.record_seed(seed, row)
            batch.close("complete")
        if point:
            batch = Batch(data / "raw" / "source-pairs", "s0b1-point",
                          [41000 + r for r in range(4)],
                          dict(stage="source-pairs", table_sha256="t",
                               protocol_sha256=KNOWN_SHA, mode="point"))
            for r in range(4):
                seed = 41000 + r
                row = _synthetic_record81(seed)
                row["table_sha256"] = "t"
                row["protocol_sha256"] = KNOWN_SHA
                row["mode"] = "point"
                row["photons_per_wavelength"] = 800
                row["sun_elevation_deg"] = -9.0
                # point-source variant: half the higher-order radiance, same
                # identities (split_total = single + multiple) preserved
                multiple_p = 0.5 * np.asarray(row["fields"]["multiple"])
                row["fields"]["multiple"] = multiple_p.tolist()
                row["fields"]["total"] = (np.asarray(row["deterministic_single"])
                                          + multiple_p).tolist()
                row["fields"]["multiple_err"] = (0.01 * multiple_p).tolist()
                row["split_total"] = (np.asarray(row["deterministic_single"])
                                      + multiple_p).tolist()
                batch.record_seed(seed, row)
            batch.close("complete")


def test_summary_end_to_end_and_gates(tmp_path):
    data = _write_mini_dataset(tmp_path)
    _populate(data, zero_one_record=True)
    summary = compute_summary(data, MINI_PROTOCOL)
    depth = summary["direct"]["-9.0"]
    b400, b800 = depth["budgets"]["400"], depth["budgets"]["800"]
    # hand-checked statistics on the synthetic values (seed 40000 is the
    # planted all-zero record)
    rows = [_synthetic_record81(s, zero=(s == 40000))
            for s in (40000, 40001, 40002, 40003)]
    spectra = np.array([r["split_total"] for r in rows])
    np.testing.assert_allclose(b400["mean"], spectra.mean(axis=0), rtol=1e-12)
    # zero-record batch: gates must fail with the no-tail-evidence marker
    assert b400["gates"]["pass_"] is False
    assert any("no-tail-evidence" in f for f in b400["gates"]["student_failures"])
    assert b400["all_zero_record_seeds"] == [40000]
    # healthy batch with generous synthetic scales must pass both intervals
    assert b800["gates"]["pass_"] is True
    assert depth["verdict"] == "budget-limited-unresolved"  # 400-block failed
    # pairing: delta = -multiple (synthetic), interval must exclude zero
    pair = summary["source_pairs"]["-9.0"]
    assert pair["fingerprints_paired"] is True
    for anchor in pair["anchors"].values():
        assert anchor["verdict"] == "identified"


def test_verify_only_detects_tampering(tmp_path):
    from tools.summarize_deep_twilight_reference import (
        build_manifest, verify_manifest)
    data = _write_mini_dataset(tmp_path)
    _populate(data)
    manifest = build_manifest(data)
    (data / "manifest.json").write_text(json.dumps(manifest))
    assert verify_manifest(data, manifest) == []
    victim = sorted((data / "raw").rglob("*.jsonl"))[0]
    rows = victim.read_text().splitlines()
    row = json.loads(rows[0])
    row["split_total"][0] *= 2  # tamper one number
    rows[0] = json.dumps(row)
    victim.write_text("\n".join(rows) + "\n")
    problems = verify_manifest(data, manifest)
    assert problems and "hash mismatch" in problems[0]


def test_missing_batch_refuses(tmp_path):
    data = _write_mini_dataset(tmp_path)
    _populate(data, point=False)  # source-pairs missing
    with pytest.raises(ValueError, match="missing batch meta"):
        compute_summary(data, MINI_PROTOCOL)


def test_candidate_and_sensitivity_sections(tmp_path):
    """The Step-3/Step-4 analysis sections run on synthetic mini batches."""
    from tools.deep_twilight_reference import Batch
    from tools.summarize_deep_twilight_reference import _candidate_stats, \
        _sensitivity_stats
    data = _write_mini_dataset(tmp_path)
    proto = json.loads((data / "protocol.json").read_text())
    proto["candidate"] = {
        "table": {"path": "probe/x.npz", "label": "x", "sha256": "c"},
        "wavelengths_nm": [450.0, 550.0, 650.0], "repeats": 4,
        "seed_rule": "test", "tiers": [
            {"candidate_samples": 100, "rest_photons": 400},
            {"candidate_samples": 200, "rest_photons": 800}]}
    proto["sensitivity"] = {"repeats": 4, "photons_per_wavelength": 300,
                            "mc_items": ["baseline", "disk4x16"],
                            "not_run_here": ["curved"]}
    (data / "protocol.json").write_text(json.dumps(proto))
    rng = np.random.default_rng(0)
    wl_anchor = {450.0: 14, 550.0: 34, 650.0: 54}
    tiers = [(100, 400), (200, 800)]
    for sun_index in (0,):
        for tier_index, (cs, rp) in enumerate(tiers):
            batch = Batch(data / "raw" / "candidate", f"s{sun_index}t{tier_index}",
                          [80000 + 10000 * sun_index + 1000 * tier_index + r
                           for r in range(4)],
                          dict(stage="candidate", table_sha256="c",
                               protocol_sha256="p", mode="disk",
                               candidate_samples=cs, rest_photons=rp,
                               code_sha256={"x.py": "1"}))
            for r in range(4):
                seed = 80000 + 10000 * sun_index + 1000 * tier_index + r
                per_lambda = {}
                for a in wl_anchor:
                    # vary rest_mean per repeat; total_analog and
                    # total_cand follow the exact closure identities, so
                    # the two estimators share one variance sequence
                    rest = 2e-10 + 1e-11 * ((r % 2) * 2 - 1)
                    mc_mult = 5e-10 + rest
                    l2 = 1e-9 * (1 + tier_index)
                    single = 1e-8
                    per_lambda[str(a)] = dict(
                        L2=l2, L2_second_moment=2e-18,
                        L2_max_abs=1e-8, candidate_seconds=0.2,
                        candidate_samples=cs, c2nt_mean=5e-10,
                        rest_mean=rest,
                        rest_second_moment=(0.0 if r == 3 else 5e-12),
                        mc_multiple=mc_mult, deterministic_single=single,
                        total_cand=single + l2 + rest,
                        total_analog=single + mc_mult, max_fraction=0.004,
                        top001_fraction=0.2, effective_contributors=50.0)
                batch.record_seed(seed, dict(
                    seed=seed, sun_elevation_deg=-9.0, tier_index=tier_index,
                    candidate_samples=cs, rest_photons=rp, mode="disk",
                    per_lambda=per_lambda, mc_seconds=0.1,
                    seconds=0.3 + 0.1 * (1 + tier_index),
                    table_sha256="c", protocol_sha256="p",
                    code_sha256={"x.py": "1"}, batch_id="x", attempt=0))
            batch.close("complete")
    batch = Batch(data / "raw" / "sensitivity", "s0mc",
                  [110000 + r for r in range(4)],
                  dict(stage="sensitivity", table_sha256="c",
                       protocol_sha256="p", mode="disk",
                       photons_per_wavelength=300))
    for r in range(4):
        seed = 110000 + r
        items = {}
        for item in ("baseline", "disk4x16"):
            items[item] = dict(
                fingerprint="fp",
                per_lambda={str(a): dict(deterministic_single=1e-8,
                                         multiple=5e-10, total=1.05e-8)
                            for a in wl_anchor}, seconds=0.1)
        batch.record_seed(seed, dict(seed=seed, sun_elevation_deg=-9.0,
                                     photons_per_wavelength=300, mode="disk",
                                     table_sha256="c", items=items,
                                     batch_id="s0mc", attempt=0,
                                     code_sha256={"x.py": "1"}))
    batch.close("complete")
    (data / "raw" / "sensitivity" / "singles-refinement.json").write_text(
        json.dumps(dict(schema="s/1", items={"ss_gauss16": "d"},
                        by_sun={"-9.0": dict(
                            baseline=[1e-8] * 81,
                            ss_gauss16=[1e-8 * (1 + 1e-6)] * 81)})))

    wl = np.arange(380.0, 781.0, 5.0)
    cand = _candidate_stats(data, proto, wl, [14, 34, 54], {})
    tier0_550 = cand["-9.0"]["tiers"]["0"]["anchors"]["550.0"]
    tier1_550 = cand["-9.0"]["tiers"]["1"]["anchors"]["550.0"]
    assert tier1_550["candidate_mean"] > tier0_550["candidate_mean"]
    # zero rest moments are real zeros in the mean, not dropped samples;
    # zero_rest_records counts only genuinely missing (None) values
    assert tier0_550["zero_rest_records"] == 0
    assert tier1_550["zero_rest_records"] == 0
    moments = [0.0 if r == 3 else 5e-12 for r in range(4)]
    assert abs(tier0_550["rest_second_moment_mean"]
               - sum(moments) / 4) < 1e-30
    # a genuinely missing moment (None from a no-tail-evidence record)
    rows = json.loads(json.dumps([
        dict(seed=s, per_lambda={"550.0": dict(rest_second_moment=(
            None if s == 80003 else 5e-12))}) for s in range(80000, 80004)]))
    n_none = sum(1 for r in rows
                 if r["per_lambda"]["550.0"]["rest_second_moment"] is None)
    assert n_none == 1
    assert cand["-9.0"]["drift"]["550.0"]["abs_drift"] > 0.0
    eff = cand["-9.0"]["efficiency"]["550.0"]
    assert eff["verdict"] in ("candidate-more-efficient",
                              "candidate-not-more-efficient")
    sens = _sensitivity_stats(data, proto, wl, [14, 34, 54])
    entry = sens["by_sun"]["-9.0"]["mc"]["disk4x16"]["550.0"]
    assert entry["paired_delta_mean"] == 0.0 and entry["fingerprints_same"]
    assert sens["singles"]["-9.0"]["ss_gauss16"]["max_abs_relative_drift"] > 0.0
    assert sens["not_run_here"] == ["curved"]


def test_cli_smoke_real(tmp_path):
    from tools.deep_twilight_reference import main
    out = tmp_path / "smoke"
    rc = main(["--stage", "smoke", "--out", str(out)])
    assert rc == 0
    summary = json.loads((out / "smoke-summary.json").read_text())
    assert summary["status"] == "scratch-only"
    assert len(summary["rows"]) == 4


def test_disk_point_fingerprint_paired_real():
    from tools.point_solar_table import PointSolarTable
    from tools.unified_solar_transport import FiniteSolarSource
    from skycolor.constants import WL_COARSE
    protocol = load_protocol(DEFAULT_PROTOCOL)
    atmo = presets.clear(WL_COARSE)
    source = protocol["source"]["full81_table"]
    table = PointSolarTable.load(ROOT / source["path"], atmo,
                                 expected_label=source["label"])
    common = {}
    for mode in ("disk", "point"):
        record = run_record(atmo, table, -9.0, 12345, 500,
                            np.zeros(len(WL_COARSE)), mode=mode)
        common[mode] = record["fingerprint"]
    assert common["disk"] == common["point"], \
        "point/disk sources must not change the native MC paths"


# ---------------------------------------------------------------------------
# review-response: contract identity, units and fresh-run-root mechanics
# ---------------------------------------------------------------------------
def test_batch_contract_mismatch_refused(tmp_path):
    code_a = {"a.py": "1"}
    code_b = {"a.py": "2"}
    batch = Batch(tmp_path, "c0", [1], dict(stage="direct", mode="disk",
                                            photons_per_wavelength=400,
                                            protocol_sha256="p1",
                                            code_sha256=code_a))
    batch.record_seed(1, {"seed": 1})
    batch.close("complete")
    # implementation change must not append into the old identity
    with pytest.raises(ValueError, match="stored contract differs"):
        Batch(tmp_path, "c0", [1], dict(stage="direct", mode="disk",
                                        photons_per_wavelength=400,
                                        protocol_sha256="p1",
                                        code_sha256=code_b))
    # budget change likewise
    with pytest.raises(ValueError, match="stored contract differs"):
        Batch(tmp_path, "c0", [1], dict(stage="direct", mode="disk",
                                        photons_per_wavelength=1600000,
                                        protocol_sha256="p1",
                                        code_sha256=code_a))
    # unchanged contract resumes and adopts the completed attempt
    reopened = Batch(tmp_path, "c0", [1], dict(stage="direct", mode="disk",
                                               photons_per_wavelength=400,
                                               protocol_sha256="p1",
                                               code_sha256=code_a))
    assert reopened.done == {1} and reopened.attempt == 0


def test_check_batch_contract_rejects_tampered_record():
    from tools.summarize_deep_twilight_reference import check_batch_contract
    meta = dict(protocol_sha256=KNOWN_SHA, table_sha256="t")
    good = [dict(seed=1, sun_elevation_deg=-9.0, photons_per_wavelength=400,
                 mode="disk", protocol_sha256=KNOWN_SHA, table_sha256="t")]
    notes = check_batch_contract(good, meta,
                                 dict(sun_elevation_deg=-9.0,
                                      photons_per_wavelength=400, mode="disk",
                                      table_sha256="t"),
                                 stage="direct", batch_id="b",
                                 protocol_sha_now=KNOWN_SHA)
    assert notes == []
    tampered = [dict(good[0], photons_per_wavelength=1600000)]
    with pytest.raises(ValueError, match="contract field"):
        check_batch_contract(tampered, meta,
                             dict(sun_elevation_deg=-9.0,
                                  photons_per_wavelength=400, mode="disk",
                                  table_sha256="t"),
                             stage="direct", batch_id="b")
    # implementation identity: a record whose code hash differs from the
    # batch meta keeps its own (uniform) per-record identity as the
    # authority and the difference is reported as a note
    meta_code = dict(meta, code_sha256={"x.py": "1"})
    code_tampered = [dict(good[0], code_sha256={"x.py": "evil"})]
    notes = check_batch_contract(code_tampered, meta_code,
                                 dict(sun_elevation_deg=-9.0,
                                      photons_per_wavelength=400,
                                      mode="disk", table_sha256="t"),
                                 stage="direct", batch_id="b")
    assert any("predates the records" in n for n in notes)
    # ... and a batch whose records span two implementations is refused
    mixed = [dict(good[0], protocol_sha256=KNOWN_SHA,
                  code_sha256={"x.py": "1"}),
             dict(good[0], seed=2, protocol_sha256=KNOWN_SHA,
                  code_sha256={"x.py": "2"})]
    meta_known = dict(meta_code, protocol_sha256=KNOWN_SHA)
    with pytest.raises(ValueError, match="distinct.*implementation"):
        check_batch_contract(mixed, meta_known,
                             dict(sun_elevation_deg=-9.0,
                                  photons_per_wavelength=400, mode="disk",
                                  table_sha256="t"),
                             stage="direct", batch_id="b")
    # a known historical generation is accepted with an auditable note
    drifted = check_batch_contract(good, meta,
                                   dict(sun_elevation_deg=-9.0,
                                        photons_per_wavelength=400,
                                        mode="disk", table_sha256="t"),
                                   stage="direct", batch_id="b",
                                   protocol_sha_now="current")
    assert len(drifted) == 1 and "known protocol generation" in drifted[0]
    # an UNKNOWN generation must be rejected, not silently merged
    meta_unknown = dict(meta, protocol_sha256="p-unknown")
    records_unknown = [dict(good[0], protocol_sha256="p-unknown")]
    with pytest.raises(ValueError, match="unknown protocol generation"):
        check_batch_contract(records_unknown, meta_unknown,
                             dict(sun_elevation_deg=-9.0,
                                  photons_per_wavelength=400, mode="disk",
                                  table_sha256="t"),
                             stage="direct", batch_id="b",
                             protocol_sha_now="current")
    # a missing expect field with meta fallback validates against the meta
    no_sun = [dict(good[0])]
    del no_sun[0]["sun_elevation_deg"]
    meta_sun = dict(meta, sun_elevation_deg=-9.0)
    notes = check_batch_contract(no_sun, meta_sun,
                                 dict(sun_elevation_deg=-9.0,
                                      photons_per_wavelength=400, mode="disk",
                                      table_sha256="t"),
                                 stage="direct", batch_id="b")
    assert any("values taken from the batch meta" in n for n in notes)
    # ... but a meta that contradicts the protocol is refused
    meta_bad = dict(meta, sun_elevation_deg=90.0)
    with pytest.raises(ValueError, match="contradicts the protocol"):
        check_batch_contract(no_sun, meta_bad,
                             dict(sun_elevation_deg=-9.0,
                                  photons_per_wavelength=400, mode="disk",
                                  table_sha256="t"),
                             stage="direct", batch_id="b")
    # ... and a field neither record nor meta carries is refused
    with pytest.raises(ValueError, match="records lack"):
        check_batch_contract(no_sun, meta,
                             dict(sun_elevation_deg=-9.0,
                                  photons_per_wavelength=400, mode="disk",
                                  table_sha256="t"),
                             stage="direct", batch_id="b")


def test_peak_rss_units_per_platform():
    import resource
    import tools.deep_twilight_reference as dtr
    raw = 1_000_000
    from types import SimpleNamespace
    real_getrusage = dtr.resource.getrusage
    real_system = dtr.platform.system
    try:
        dtr.resource.getrusage = lambda _: SimpleNamespace(ru_maxrss=raw)
        dtr.platform.system = lambda: "Darwin"
        assert dtr.peak_rss_bytes() == 1_000_000       # already bytes
        dtr.platform.system = lambda: "Linux"
        assert dtr.peak_rss_bytes() == 1_024_000_000   # KiB -> bytes
    finally:
        dtr.resource.getrusage = real_getrusage
        dtr.platform.system = real_system


def test_cost_probe_cli_exit_path(tmp_path, capsys, monkeypatch):
    """main() must not assume keys a scratch summary does not carry."""
    import tools.deep_twilight_reference as dtr
    fake = dict(stage="cost-probe", status="scratch-only-scheduling-estimate",
                rows=[dict(sun=-9.0, seed=1, photons=1, seconds=0.5)],
                seconds_per_wavelength_path=1e-6)
    monkeypatch.setattr(dtr, "stage_cost_probe",
                        lambda *a, **k: fake)
    rc = dtr.main(["--stage", "cost-probe", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "seconds_per_wavelength_path" in out


def test_fresh_run_root_direct_resume(tmp_path):
    """--out on a fresh directory must work end to end (Major 7/9)."""
    from tools.deep_twilight_reference import main, load_protocol
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    protocol["direct"]["repeats"] = 1
    protocol["direct"]["budgets_photons"] = [400, 800]
    (input_dir / "protocol.json").write_text(json.dumps(protocol))
    run_root = tmp_path / "run"
    argv = ["--protocol", str(input_dir / "protocol.json"),
            "--stage", "direct", "--out", str(run_root),
            "--max-stage-hours", "0.02"]
    rc = main(argv)  # will stop on budget after setup + ~1 record
    assert rc == 0
    assert (run_root / "protocol.snapshot.json").exists()
    batch_dirs = list((run_root / "raw" / "direct").iterdir())
    assert batch_dirs, "a batch directory must exist"
    group_state = json.loads(
        (run_root / "state.direct_and_source_pairs.json").read_text())
    assert (group_state["stages"]["direct_and_source_pairs"]
            ["cumulative_seconds"] > 0)
    # resume under the same contract continues the same attempt lineage
    rc2 = main(argv + ["--max-stage-hours", "0.05"])
    assert rc2 == 0
    meta = json.loads((sorted(batch_dirs)[0] / "meta.json").read_text())
    assert meta["protocol_sha256"]  # pinned at execution time
    # a contract change against an existing run root is refused at the
    # protocol snapshot guard (the batch-level refusal is covered by
    # test_batch_contract_mismatch_refused)
    protocol["direct"]["budgets_photons"] = [400, 1600]
    (input_dir / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(SystemExit, match="snapshot differs"):
        main(["--protocol", str(input_dir / "protocol.json"),
              "--stage", "direct", "--out", str(run_root),
              "--max-stage-hours", "0.01"])


def test_efficiency_metric_direction(tmp_path):
    """Variance x time: a slower estimator must never score better."""
    from tools.summarize_deep_twilight_reference import _candidate_stats
    data = _write_mini_dataset(tmp_path)
    proto = json.loads((data / "protocol.json").read_text())
    proto["candidate"] = {
        "table": {"path": "probe/x.npz", "label": "x", "sha256": "c"},
        "wavelengths_nm": [450.0, 550.0, 650.0], "repeats": 4,
        "seed_rule": "t", "tiers": [{"candidate_samples": 1, "rest_photons": 4},
                                    {"candidate_samples": 2, "rest_photons": 8}]}
    (data / "protocol.json").write_text(json.dumps(proto))
    from tools.deep_twilight_reference import Batch
    wl_anchor = {450.0: 14, 550.0: 34, 650.0: 54}
    for tier_index, (cs, rp, t_cand, t_mc) in enumerate(
            [(1, 4, 2.0, 1.0), (2, 8, 2.0, 1.0)]):
        batch = Batch(data / "raw" / "candidate", f"s0t{tier_index}",
                      [80000 + 1000 * tier_index + r for r in range(4)],
                      dict(stage="candidate", table_sha256="c",
                           protocol_sha256="p", mode="disk",
                           candidate_samples=cs, rest_photons=rp,
                           code_sha256={"x.py": "1"}))
        for r in range(4):
            seed = 80000 + 1000 * tier_index + r
            # NON-identical repeats via the closure identities: rest_mean
            # varies per repeat, mc_multiple = c2nt + rest, so
            # total_analog = single + mc_multiple varies while
            # total_cand = single + L2 + rest carries the same variance
            # (both sequences identical after the fixed offset); the
            # candidate is SLOWER (t_cand + t_mc vs t_mc) -- a correct
            # variance-x-time metric must score it strictly worse with
            # ratio == repeat-time ratio
            rest = 2e-10 + 1e-11 * ((r % 2) * 2 - 1)
            mc_mult = 7e-10 + 1e-11 * ((r % 2) * 2 - 1)
            per_lambda = {str(a): dict(
                L2=1e-9, L2_second_moment=2e-18, L2_max_abs=1e-8,
                candidate_seconds=t_cand, candidate_samples=cs, c2nt_mean=5e-10,
                rest_mean=rest, rest_second_moment=5e-12,
                mc_multiple=mc_mult,
                deterministic_single=1e-8, total_cand=1e-8 + 1e-9 + rest,
                total_analog=1e-8 + mc_mult, max_fraction=0.004,
                top001_fraction=0.2, effective_contributors=50.0)
                for a in wl_anchor}
            batch.record_seed(seed, dict(
                seed=seed, sun_elevation_deg=-9.0, tier_index=tier_index,
                candidate_samples=cs, rest_photons=rp, mode="disk",
                per_lambda=per_lambda, mc_seconds=t_mc,
                seconds=t_cand + t_mc,
                table_sha256="c", protocol_sha256="p",
                code_sha256={"x.py": "1"}, batch_id="x", attempt=0))
        batch.close("complete")
    wl = np.arange(380.0, 781.0, 5.0)
    out = _candidate_stats(data, proto, wl, [14, 34, 54], {})
    eff = out["-9.0"]["efficiency"]["550.0"]
    assert eff["candidate_var_x_time"] > eff["analog_var_x_time"]
    assert eff["verdict"] == "candidate-not-more-efficient"
    # ratio must equal the repeat-time ratio exactly (equal variances)
    assert abs(eff["ratio"] - (2.0 + 1.0) / 1.0) < 1e-9
    # the real archived flip case: -9 deg tier-1 650 nm, candidate worse
    # (ratios from summary.json after the variance-x-time correction)
    assert out["-9.0"]["efficiency"]["650.0"]["ratio"] > 1.0


def test_fresh_run_root_full_chain(tmp_path):
    """fresh --out: direct + pairs + summary + verify-only, no manual copies."""
    from tools.deep_twilight_reference import main
    from tools.summarize_deep_twilight_reference import (resolve_protocol_path,
                                                         compute_summary)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    # n=2: the statistical minimum with a defined Student interval
    protocol["direct"]["repeats"] = 2
    protocol["direct"]["budgets_photons"] = [400, 800]
    (input_dir / "protocol.json").write_text(json.dumps(protocol))
    run_root = tmp_path / "run"
    base = ["--protocol", str(input_dir / "protocol.json"),
            "--out", str(run_root)]
    assert main(base + ["--stage", "direct", "--max-stage-hours", "0.05"]) == 0
    assert main(base + ["--stage", "source-pairs",
                        "--max-stage-hours", "0.03"]) == 0
    # summary CLI on the run root alone
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "tools.summarize_deep_twilight_reference",
         "--directory", str(run_root)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-1500:]
    resolved = resolve_protocol_path(run_root)
    assert resolved.name == "protocol.snapshot.json"
    summary = compute_summary(run_root, protocol)
    assert summary["protocol_file"] == "protocol.snapshot.json"
    assert summary["direct"]["-9.0"]["budgets"]["800"]["n"] == 2
    assert "source_pairs" in summary
    # verify-only on the run root
    proc = subprocess.run(
        [sys.executable, "-m", "tools.summarize_deep_twilight_reference",
         "--directory", str(run_root), "--verify-only"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-1500:]
    assert "VERIFY OK" in proc.stdout


def test_group_state_files_no_cross_group_loss(tmp_path):
    """Concurrent cap groups must not lose each other's accounting."""
    from tools.deep_twilight_reference import (
        load_state, note_stage_seconds, save_state, STAGE_CAP_GROUP)
    state = {"stages": {}}
    note_stage_seconds(tmp_path, state, "direct", 100.0, 64)
    save_state(tmp_path, state, stage="direct")
    # a different group saves its own file afterwards (its in-memory view
    # never contained the direct group)
    other = {"stages": {}}
    note_stage_seconds(tmp_path, other, "candidate", 200.0, 64)
    save_state(tmp_path, other, stage="candidate")
    merged = load_state(tmp_path)
    assert merged["stages"]["direct_and_source_pairs"]["cumulative_seconds"] == 100.0
    assert merged["stages"]["candidate"]["cumulative_seconds"] == 200.0
    # budget accounting for a stage reads its own group, not the merge
    from tools.deep_twilight_reference import stage_budget, DEFAULT_PROTOCOL
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    spent, left = stage_budget(tmp_path, "source-pairs", protocol, None)
    assert spent == 100.0
    assert left == protocol["wall_clock_caps_hours"]["direct_and_source_pairs"] * 3600.0 - 100.0
