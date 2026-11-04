"""Recompute the deep-twilight reference summary from raw records.

This is the analysis authority for ``validation/data/deep_twilight_reference/``:
it never trusts a previous summary.  Every number in ``summary.json`` is
recomputed from the append-only raw records (statistics, precision gates,
budget drift, bootstrap sensitivity, source-pair deltas, tail diagnostics,
cost), and a SHA256 manifest pins every input.  ``--verify-only`` repeats the
whole recomputation and compares against the stored summary; any mismatch,
missing batch, duplicate "valid" attempt, or altered hash exits non-zero
instead of being merged silently.

Statistical contract (from ``protocol.json``):

* the sampling unit is one independent complete spectrum; luminance
  ``683 * trapezoid(L * V, wl)`` integrates each spectrum *before* any
  statistics, so no cross-wavelength independence is assumed;
* per budget: mean, sample SD, Student-t 95 % conditional half-width
  (df = 15); the bootstrap (5000 resamples of complete repeats, analysis
  seed 90000) is a sensitivity check and must not flip the verdict;
* source correction: paired (disk - point) differences on identical seeds,
  analysed per repeat; "identified" requires the anchor interval to exclude
  zero -- a stable reference alone never implies a resolved correction;
* an all-zero multiple spectrum is a detection limit of the photon budget
  ("no-tail-evidence") and fails the precision gate by construction.

Run from the repository root::

    uv run python -m tools.summarize_deep_twilight_reference \\
        --directory validation/data/deep_twilight_reference            # write
    uv run python -m tools.summarize_deep_twilight_reference \\
        --directory validation/data/deep_twilight_reference --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor.constants import WL_COARSE  # noqa: E402  (contract anchor)

#: record-level recombination tolerance (float64 add/subtract exactness)
CLOSURE_RTOL = 2e-13


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# raw ingestion and validation
# ---------------------------------------------------------------------------
def read_attempt(path: Path):
    """Parse one attempt file; returns (records, corrupt_tail)."""
    records, corrupt = [], False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            corrupt = True
            break
    return records, corrupt


def load_batch(batch_dir: Path, expected_seeds, *, stage, batch_id):
    """Return the records of the unique valid complete attempt.

    Refuses to merge when: no attempt is complete, more than one attempt is
    complete (duplicated recomputation must never count as extra seeds), the
    chosen attempt's seeds deviate from the pre-registered set, or it has a
    corrupt trailing line or duplicate seeds.
    """
    meta_path = batch_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"{stage}/{batch_id}: missing batch meta")
    meta = json.loads(meta_path.read_text())
    candidates = []
    for attempt in meta.get("attempts", []):
        path = batch_dir / f"attempt-{attempt['index']:02d}.jsonl"
        if not path.exists() or attempt.get("status") != "complete":
            continue
        records, corrupt = read_attempt(path)
        if corrupt:
            raise ValueError(f"{stage}/{batch_id}: attempt "
                             f"{attempt['index']} is marked complete but has "
                             "a corrupt line")
        seeds = [int(r["seed"]) for r in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{stage}/{batch_id}: attempt "
                             f"{attempt['index']} repeats seeds -- duplicated "
                             "recomputation must not be merged")
        candidates.append((attempt["index"], records))
    if not candidates:
        raise ValueError(
            f"{stage}/{batch_id}: no valid complete attempt "
            f"(attempts: {[(a.get('index'), a.get('status')) for a in meta.get('attempts', [])]})")
    if len(candidates) > 1:
        raise ValueError(
            f"{stage}/{batch_id}: {len(candidates)} complete attempts "
            f"{[i for i, _ in candidates]} -- duplicated recomputation must not "
            "be merged; mark all but one attempt superseded in meta.json")
    index, records = candidates[0]
    seeds = [int(r["seed"]) for r in records]
    if sorted(seeds) != sorted(int(s) for s in expected_seeds):
        raise ValueError(f"{stage}/{batch_id}: seeds {sorted(seeds)} deviate "
                         f"from the pre-registered {sorted(expected_seeds)}")
    return records, meta


def check_record_invariants(records, *, stage, batch_id):
    """Physical/statistical invariants every record must satisfy."""
    for row in records:
        fields = row["fields"]
        total = np.asarray(fields["total"])
        single = np.asarray(fields["single"])
        multiple = np.asarray(fields["multiple"])
        # MC split identity: total = single + multiple
        if not np.allclose(total, single + multiple, rtol=CLOSURE_RTOL, atol=0):
            raise ValueError(f"{stage}/{batch_id} seed {row['seed']}: "
                             "MC split closure violated")
        # reference identity: split_total = deterministic single + MC multiple
        if not np.array_equal(np.asarray(row["split_total"]),
                              np.asarray(row["deterministic_single"]) + multiple):
            raise ValueError(f"{stage}/{batch_id} seed {row['seed']}: "
                             "split_total is not deterministic_single + multiple")
        if row["photons_per_wavelength"] <= 0 or row["seed"] < 0:
            raise ValueError(f"{stage}/{batch_id} seed {row['seed']}: "
                             "bad budget/seed fields")
        if len(np.asarray(row["split_total"])) != len(WL_COARSE):
            raise ValueError(f"{stage}/{batch_id} seed {row['seed']}: "
                             "spectrum does not span the 81-wavelength grid")


def check_batch_contract(records, meta, expect, *, stage, batch_id,
                         protocol_sha_now=None):
    """Records and batch meta must agree with the pre-registered contract.

    Checks the semantic identity (sun elevation, budget, mode, source and
    protocol hashes) of every record against ``expect`` and the batch meta;
    cross-batch protocol-hash drift is reported as an auditable note, not
    silently merged.
    """
    notes = []
    meta_protocol = meta.get("protocol_sha256")
    for row in records:
        for key, want in expect.items():
            got = row.get(key)
            if got != want:
                raise ValueError(
                    f"{stage}/{batch_id} seed {row['seed']}: contract field "
                    f"{key} is {got!r}, protocol requires {want!r}")
        for key in ("protocol_sha256", "table_sha256"):
            if meta.get(key) is not None and row.get(key) != meta[key]:
                raise ValueError(
                    f"{stage}/{batch_id} seed {row['seed']}: record {key} "
                    "differs from the batch meta provenance")
    if (protocol_sha_now is not None and meta_protocol is not None
            and meta_protocol != protocol_sha_now):
        notes.append(
            f"{stage}/{batch_id}: executed under protocol sha "
            f"{meta_protocol[:12]}... (current file {protocol_sha_now[:12]}...); "
            "stage-relevant sections validated semantically above")
    return notes


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def student_halfwidth(values: np.ndarray, confidence=0.95) -> np.ndarray:
    n = values.shape[0]
    se = values.std(axis=0, ddof=1) / math.sqrt(n)
    return student_t.ppf(0.5 + confidence / 2.0, n - 1) * se


def bootstrap_halfwidth(values: np.ndarray, resamples: int, seed: int):
    """Percentile half-width of resampled means (complete repeats only)."""
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = values[idx].mean(axis=1)              # (resamples, m)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return (hi - lo) / 2.0


def photopic(radiance_row: np.ndarray, v_curve: np.ndarray, wl: np.ndarray) -> float:
    return float(683.0 * np.trapezoid(radiance_row * v_curve, wl))


# ---------------------------------------------------------------------------
# summary computation
# ---------------------------------------------------------------------------
def compute_summary(data_dir: Path, protocol: dict) -> dict:
    scenario = protocol["scenario"]
    direct_cfg = protocol["direct"]
    precision = protocol["precision"]
    wl = WL_COARSE.copy()
    v_curve = _v_curve(wl)
    anchors = list(scenario["anchor_wavelengths_nm"])
    anchor_idx = [int(np.argmin(np.abs(wl - a))) for a in anchors]
    budgets = list(direct_cfg["budgets_photons"])
    repeats = int(direct_cfg["repeats"])
    suns = list(scenario["sun_elevations_deg"])

    raw_dir = data_dir / "raw"
    protocol_sha_now = sha256_file(data_dir / "protocol.json")
    contract_notes = []
    out = dict(schema="deep-twilight-reference/summary/1",
               generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               protocol_sha256=sha256_file(data_dir / "protocol.json"),
               grid=wl.tolist(), anchors_nm=anchors)

    # ---- direct batches --------------------------------------------------
    disk_records = {}
    disk_meta = {}
    for sun_index, sun in enumerate(suns):
        per_budget = {}
        for budget_index, photons in enumerate(budgets):
            batch_id = f"s{sun_index}b{budget_index}"
            records, meta = load_batch(
                raw_dir / "direct" / batch_id,
                [40000 + 10000 * sun_index + 1000 * budget_index + r
                 for r in range(repeats)],
                stage="direct", batch_id=batch_id)
            check_record_invariants(records, stage="direct", batch_id=batch_id)
            # NOTE: fingerprints identify the per-seed path SET, so they are
            # all distinct within a batch by design; the meaningful identity
            # is same-seed disk/point equality, checked in the pairs section
            contract_notes.extend(check_batch_contract(
                records, meta,
                dict(sun_elevation_deg=sun, photons_per_wavelength=photons,
                     mode="disk",
                     table_sha256=protocol["source"]["full81_table"]["sha256"]),
                stage="direct", batch_id=batch_id,
                protocol_sha_now=protocol_sha_now))
            disk_records[(sun, photons)] = records
            disk_meta[batch_id] = meta
            per_budget[str(photons)] = _budget_stats(
                records, wl, v_curve, anchor_idx, sun, str(photons), protocol,
                stage="direct")
        out.setdefault("direct", {})[str(sun)] = dict(
            budgets=per_budget, drift=_drift_stats(per_budget, anchor_idx,
                                                   wl, v_curve, sun, precision))
    # per-depth verdict: stable requires every budget gate (Student and
    # bootstrap agree) plus budget drift inside half of each target half-scale
    for sun_index, sun in enumerate(suns):
        stats = out["direct"][str(sun)]
        gates_ok = all(b["gates"]["pass_"] for b in stats["budgets"].values())
        drift_ok = stats["drift"]["anchors_pass"] and stats["drift"]["luminance_pass"]
        if gates_ok and drift_ok:
            stats["verdict"] = "sampling-stable-under-protocol"
            stats["verdict_failures"] = []
        else:
            stats["verdict"] = "budget-limited-unresolved"
            failures = [f"budget {b}: {f}"
                        for key, b in stats["budgets"].items()
                        for f in b["gates"]["student_failures"]
                        + b["gates"]["bootstrap_failures"]]
            if not stats["drift"]["anchors_pass"]:
                failures.append("anchor drift between budgets exceeds half of "
                                "the target half-scale")
            if not stats["drift"]["luminance_pass"]:
                failures.append("luminance drift between budgets exceeds a "
                                "quarter of the luminance gate")
            stats["verdict_failures"] = failures
    # ---- source pairs ----------------------------------------------------
    pair_cfg = protocol["source"]["point_pairing"]
    pairs = {}
    for sun_index, sun in enumerate(suns):
        top_budget_index = len(budgets) - 1
        photons = budgets[top_budget_index]
        batch_id = f"s{sun_index}b{top_budget_index}-point"
        records, meta = load_batch(
            raw_dir / "source-pairs" / batch_id,
            [40000 + 10000 * sun_index + 1000 * top_budget_index + r
             for r in range(repeats)],
            stage="source-pairs", batch_id=batch_id)
        check_record_invariants(records, stage="source-pairs", batch_id=batch_id)
        contract_notes.extend(check_batch_contract(
            records, meta,
            dict(sun_elevation_deg=sun, photons_per_wavelength=photons,
                 mode="point",
                 table_sha256=protocol["source"]["full81_table"]["sha256"]),
            stage="source-pairs", batch_id=batch_id,
            protocol_sha_now=protocol_sha_now))
        disk_rows = disk_records[(sun, photons)]
        pairs[str(sun)] = _pair_stats(disk_rows, records, wl, v_curve,
                                      anchor_idx, sun, protocol, pair_cfg)
    out["source_pairs"] = pairs
    out["contract_notes"] = contract_notes
    # ---- candidate experiment (plan Step 3; may be absent/incomplete) -----
    if (data_dir / "raw" / "candidate").exists():
        out["candidate"] = _candidate_stats(data_dir, protocol, wl, anchor_idx,
                                            out["direct"])
    # ---- sensitivity (plan Step 4; may be absent/incomplete) --------------
    if (data_dir / "raw" / "sensitivity").exists():
        out["sensitivity"] = _sensitivity_stats(data_dir, protocol, wl,
                                                anchor_idx)
    out["cost"] = _cost_stats(data_dir, disk_meta)
    out["unknowns"] = [
        "diffuse-transport curvature: straight diffuse lines are a fixed "
        "transport approximation of this reference (curved backends exist as "
        "experiments; their correction is not quantified here)",
        "continuous-wavelength integral: the 380-780 nm / 5 nm grid is a "
        "discrete contract; trapezoidal truncation/quadrature error of the "
        "continuum is not certified (angular smoothing and spectral anchors "
        "are not applicable by construction, not zero)",
        "solar-source discretisation: table quadrature and disk-node "
        "refinement sensitivities are Step-4 items and may remain unquantified",
        "tail coverage: conditional intervals do not certify that unsampled "
        "heavy tails are exhausted; all-zero records are detection limits",
    ]
    return out


def _v_curve(wl: np.ndarray) -> np.ndarray:
    from skycolor import spectra
    return spectra.cie_xyz(wl)[:, 1]


def _budget_stats(records, wl, v_curve, anchor_idx, sun, budget, protocol,
                  *, stage):
    """Per-budget statistics, gates and tail concentration for one batch."""
    precision = protocol["precision"]
    spectra_rows = np.array([r["split_total"] for r in records])  # (n, 81)
    multiple_rows = np.array([r["fields"]["multiple"] for r in records])
    n = spectra_rows.shape[0]
    mean = spectra_rows.mean(axis=0)
    half = student_halfwidth(spectra_rows,
                             protocol["direct"]["analysis"]["student"]["confidence"])
    luminance = np.array([photopic(row, v_curve, wl) for row in spectra_rows])
    y_mean = float(luminance.mean())
    y_half = float(student_halfwidth(luminance.reshape(-1, 1),
                                     protocol["direct"]["analysis"]["student"]["confidence"])[0])

    boot_cfg = protocol["direct"]["analysis"]["bootstrap"]
    boot_half_spec = bootstrap_halfwidth(spectra_rows, int(boot_cfg["resamples"]),
                                         int(boot_cfg["seed"]))
    boot_half_y = float(bootstrap_halfwidth(luminance.reshape(-1, 1),
                                            int(boot_cfg["resamples"]),
                                            int(boot_cfg["seed"]))[0])

    # all-zero record = detection limit ("no-tail-evidence")
    zero_records = [int(r["seed"]) for r, m in zip(records, multiple_rows)
                    if not np.any(m > 0.0)]
    anchor_all_zero = {str(float(wl[i])): bool(any(
        not (np.asarray(r["fields"]["multiple"])[i] > 0.0) for r in records))
        for i in anchor_idx}

    # tail concentration per anchor (max over records), plus global worst
    def _tail(key, index=None):
        values = []
        for r in records:
            if index is None:
                per = [d[key] for d in r["diagnostics"]["per_wl"]
                       if d[key] is not None]
            else:
                d = r["diagnostics"]["anchors"][str(float(wl[index]))]
                per = [d[key]] if d[key] is not None else []
            values.extend(per)
        return max(values) if values else None

    tail = {str(float(wl[i])): {"max_path_fraction": _tail("max_fraction", i),
                           "top001_fraction": _tail("top001_fraction", i),
                           "effective_contributors_min": min(
                               d["effective_contributors"]
                               for r in records
                               for d in [r["diagnostics"]["anchors"][str(float(wl[i]))]]
                               if d["effective_contributors"] is not None)
                           if any(r["diagnostics"]["anchors"][str(float(wl[i]))]["effective_contributors"]
                                  is not None for r in records) else None}
            for i in anchor_idx}
    tail["worst_any_wavelength"] = {"max_path_fraction": _tail("max_fraction"),
                                    "top001_fraction": _tail("top001_fraction")}

    # gates
    scale_key = str(sun)
    scales = precision["anchor_scales"][scale_key]
    anchor_hw = {str(float(wl[i])): float(half[i]) for i in anchor_idx}
    anchor_boot_hw = {str(float(wl[i])): float(boot_half_spec[i]) for i in anchor_idx}
    anchor_factor = float(precision["anchor_halfwidth_factor"])
    y_rel_half = y_half / y_mean if y_mean else math.inf
    y_rel_boot = boot_half_y / y_mean if y_mean else math.inf
    y_limit = float(precision["luminance_relative_halfwidth"])
    anchors = [float(a) for a in protocol["scenario"]["anchor_wavelengths_nm"]]

    def _verdict(hw_map, y_rel):
        failed = []
        if zero_records:
            failed.append(f"no-tail-evidence (all-zero records: {zero_records})")
        for a, s, i in zip(anchors, scales, anchor_idx):
            if anchor_all_zero[str(float(wl[i]))]:
                failed.append(f"anchor {a} nm has all-zero records")
            elif hw_map[str(float(wl[i]))] > float(s) * anchor_factor:
                failed.append(
                    f"anchor {a} nm half-width {hw_map[str(float(wl[i]))]:.4e} "
                    f"> {float(s) * anchor_factor:.4e}")
        if y_rel > y_limit:
            failed.append(f"luminance relative half-width {y_rel:.5f} > {y_limit}")
        tail_max = tail["worst_any_wavelength"]["max_path_fraction"]
        if tail_max is not None and tail_max > float(precision["max_path_fraction"]):
            failed.append(f"max path fraction {tail_max:.4f} > "
                          f"{precision['max_path_fraction']}")
        top01 = tail["worst_any_wavelength"]["top001_fraction"]
        if top01 is not None and top01 > float(precision["top001_fraction"]):
            failed.append(f"top-0.1% fraction {top01:.4f} > "
                          f"{precision['top001_fraction']}")
        return (len(failed) == 0), failed

    verdict_student, failures_student = _verdict(anchor_hw, y_rel_half)
    verdict_boot, failures_boot = _verdict(anchor_boot_hw, y_rel_boot)

    seconds = float(sum(r["seconds"] for r in records))
    return dict(
        n=n, seeds=sorted(int(r["seed"]) for r in records),
        mean=mean.tolist(), sd=spectra_rows.std(axis=0, ddof=1).tolist(),
        student_halfwidth=half.tolist(),
        anchors={str(float(wl[i])): {"mean": float(mean[i]),
                                "student_halfwidth": float(half[i]),
                                "bootstrap_halfwidth": float(boot_half_spec[i]),
                                "scale": float(s),
                                "gate_limit": float(s) * anchor_factor}
                 for s, i in zip(scales, anchor_idx)},
        luminance=dict(mean=y_mean, student_halfwidth=y_half,
                       relative_halfwidth=y_rel_half,
                       bootstrap_relative_halfwidth=y_rel_boot,
                       limit=y_limit),
        all_zero_record_seeds=zero_records,
        anchor_all_zero=anchor_all_zero,
        tail=tail,
        gates=dict(student_pass=verdict_student, student_failures=failures_student,
                   bootstrap_pass=verdict_boot, bootstrap_failures=failures_boot,
                   pass_=bool(verdict_student and verdict_boot)),
        seconds=seconds,
        stage=stage)


def _candidate_stats(data_dir: Path, protocol: dict, wl: np.ndarray,
                     anchor_idx, direct_section) -> dict:
    """Candidate vs analog comparison at identical work units.

    The candidate experiment is self-contained on the 3-wavelength planning
    table: for every (depth, tier, seed) one candidate sweep plus ONE rest
    MC on the same seed, so ``total_analog`` in the same records is the
    same-source, same-seed comparator and the repeat is the shared cost
    unit.  Reported per anchor: conditional intervals for both totals over
    the 16 repeats, variance-of-mean per second of true repeat wall-clock,
    tier drift, and the full81 direct baseline as a clearly separated
    *distribution reference* (different table/RNG streams -- never a cost
    comparison).  Zero rest moments are real zeros, not missing values.
    """
    cand_cfg = protocol["candidate"]
    anchors_nm = [float(wl[i]) for i in anchor_idx]
    confidence = protocol["direct"]["analysis"]["student"]["confidence"]
    repeats = int(cand_cfg["repeats"])
    suns = protocol["scenario"]["sun_elevations_deg"]
    raw_dir = data_dir / "raw" / "candidate"
    out = {}
    for sun_index, sun in enumerate(suns):
        tiers = {}
        for tier_index, tier in enumerate(cand_cfg["tiers"]):
            batch_id = f"s{sun_index}t{tier_index}"
            seeds = [80000 + 10000 * sun_index + 1000 * tier_index + r
                     for r in range(repeats)]
            records, _meta = load_batch(raw_dir / batch_id, seeds,
                                        stage="candidate", batch_id=batch_id)
            records = sorted(records, key=lambda r: int(r["seed"]))
            per_anchor = {}
            for a_nm in anchors_nm:
                cand_total = np.array([r["per_lambda"][str(a_nm)]["total_cand"]
                                       for r in records])
                analog_total = np.array([r["per_lambda"][str(a_nm)]["total_analog"]
                                         for r in records])
                rest_moments = np.array([
                    0.0 if r["per_lambda"][str(a_nm)]["rest_second_moment"] is None
                    else r["per_lambda"][str(a_nm)]["rest_second_moment"]
                    for r in records])
                n_zero_rest = sum(1 for r in records
                                  if r["per_lambda"][str(a_nm)]["rest_second_moment"]
                                  is None)
                repeat_seconds = np.array([float(r["seconds"]) for r in records])
                analog_seconds = np.array([float(r.get("mc_seconds")
                                                 or r["seconds"] * 0.0)
                                           for r in records]) \
                    if all("mc_seconds" in r for r in records) \
                    else np.array([float(r["seconds"]) for r in records])
                cand_var_mean = cand_total.var(axis=0, ddof=1) / len(records)
                analog_var_mean = analog_total.var(axis=0, ddof=1) / len(records)
                per_anchor[str(a_nm)] = dict(
                    candidate_mean=float(cand_total.mean()),
                    candidate_sd=float(cand_total.std(axis=0, ddof=1)),
                    candidate_t95_halfwidth=float(student_halfwidth(
                        cand_total.reshape(-1, 1), confidence)[0]),
                    analog_mean=float(analog_total.mean()),
                    analog_sd=float(analog_total.std(axis=0, ddof=1)),
                    analog_t95_halfwidth=float(student_halfwidth(
                        analog_total.reshape(-1, 1), confidence)[0]),
                    rest_second_moment_mean=float(rest_moments.mean()),
                    zero_rest_records=n_zero_rest,
                    # variance of the mean per second of true repeat cost:
                    # the same-work-unit efficiency metric
                    candidate_var_of_mean_per_second=float(
                        cand_var_mean / repeat_seconds.mean()),
                    analog_var_of_mean_per_second=float(
                        analog_var_mean / max(analog_seconds.mean(), 1e-9)),
                    candidate_seconds_per_repeat=float(repeat_seconds.mean()),
                    analog_mc_seconds_per_repeat=float(analog_seconds.mean()))
            tiers[str(tier_index)] = dict(
                rest_photons=tier["rest_photons"],
                candidate_samples=tier["candidate_samples"],
                anchors=per_anchor,
                seconds=float(sum(r["seconds"] for r in records)))
        lo_tier, hi_tier = sorted(tiers, key=int)
        drift = {}
        for a_nm in anchors_nm:
            lo = tiers[lo_tier]["anchors"][str(a_nm)]
            hi = tiers[hi_tier]["anchors"][str(a_nm)]
            drift[str(a_nm)] = dict(
                abs_drift=abs(hi["candidate_mean"] - lo["candidate_mean"]),
                low_mean=lo["candidate_mean"], high_mean=hi["candidate_mean"])
        efficiency = {}
        for a_nm in anchors_nm:
            hi_anchor = tiers[hi_tier]["anchors"][str(a_nm)]
            c = hi_anchor["candidate_var_of_mean_per_second"]
            a = hi_anchor["analog_var_of_mean_per_second"]
            efficiency[str(a_nm)] = dict(
                candidate_var_of_mean_per_second=c,
                analog_var_of_mean_per_second=a,
                verdict=("candidate-more-efficient" if c < a
                         else "candidate-not-more-efficient"))
        # full81 direct baseline: distribution reference only
        reference = {}
        direct_sun = direct_section.get(str(sun), {})
        for tier_index, tier in enumerate(cand_cfg["tiers"]):
            b_key = str(protocol["direct"]["budgets_photons"][tier_index])
            block = direct_sun.get("budgets", {}).get(b_key)
            if block is None:
                continue
            reference[b_key] = {
                str(a_nm): dict(direct_mean=float(block["mean"][i]),
                                direct_halfwidth=float(
                                    block["student_halfwidth"][i]),
                                note="distribution reference across the "
                                     "table/stream boundary, NOT a cost "
                                     "comparison")
                for i, a_nm in zip(anchor_idx, anchors_nm)}
        out[str(sun)] = dict(tiers=tiers, drift=drift,
                             efficiency=efficiency, reference=reference)
    return out


def _sensitivity_stats(data_dir: Path, protocol: dict, wl: np.ndarray,
                       anchor_idx) -> dict:
    """Paired same-seed drift of every refinement against its baseline."""
    sens = protocol.get("sensitivity", {})
    anchors_nm = [float(wl[i]) for i in anchor_idx]
    confidence = protocol["direct"]["analysis"]["student"]["confidence"]
    repeats = int(sens.get("repeats", 16))
    suns = protocol["scenario"]["sun_elevations_deg"]
    raw_dir = data_dir / "raw" / "sensitivity"
    out = dict(mc_items=list(sens.get("mc_items", [])), by_sun={})
    for sun_index, sun in enumerate(suns):
        batch_id = f"s{sun_index}mc"
        seeds = [110000 + 1000 * sun_index + r for r in range(repeats)]
        records, _meta = load_batch(raw_dir / batch_id, seeds,
                                    stage="sensitivity", batch_id=batch_id)
        per_item = {}
        items = [k for k in records[0]["items"] if k != "baseline"]
        for item in items:
            per_anchor = {}
            for a_nm in anchors_nm:
                base = np.array([r["items"]["baseline"]["per_lambda"][str(a_nm)]["total"]
                                 for r in records])
                variant = np.array([r["items"][item]["per_lambda"][str(a_nm)]["total"]
                                    for r in records])
                delta = variant - base
                per_anchor[str(a_nm)] = dict(
                    baseline_mean=float(base.mean()),
                    variant_mean=float(variant.mean()),
                    paired_delta_mean=float(delta.mean()),
                    paired_delta_t95_halfwidth=float(student_halfwidth(
                        delta.reshape(-1, 1), confidence)[0]),
                    fingerprints_same=all(
                        r["items"][item]["fingerprint"]
                        == r["items"]["baseline"]["fingerprint"] for r in records))
            per_item[item] = per_anchor
        out["by_sun"][str(sun)] = dict(mc=per_item)
    singles_path = raw_dir / "singles-refinement.json"
    if singles_path.exists():
        payload = json.loads(singles_path.read_text())
        singles = {}
        for sun, by in payload["by_sun"].items():
            base = np.array(by["baseline"])
            entries = {}
            for item, _desc in payload["items"].items():
                variant = np.array(by[_slug(item)])
                entries[item] = dict(
                    max_abs_relative_drift=float(
                        np.max(np.abs(variant - base)
                               / np.maximum(np.abs(base), 1e-300))),
                    variant=variant.tolist())
            singles[sun] = entries
        out["singles"] = singles
    out["not_run_here"] = sens.get("not_run_here", [])
    return out


def _slug(item: str) -> str:
    return {"ss_gauss16": "ss_gauss16",
            "ss_fine_density": "ss_density_step_0_0125"}[item]


def _drift_stats(per_budget, anchor_idx, wl, v_curve, sun, precision):
    hi_key, lo_key = sorted(per_budget, key=int, reverse=True)[:2]
    hi, lo = per_budget[hi_key], per_budget[lo_key]
    hi_mean = np.asarray(hi["mean"])
    lo_mean = np.asarray(lo["mean"])
    scales = precision["anchor_scales"][str(sun)]
    factor = float(precision["drift_factor"]) * float(precision["anchor_halfwidth_factor"])
    anchor_drift = {str(float(wl[i])): {"abs_drift": float(abs(hi_mean[i] - lo_mean[i])),
                                   "limit": float(s) * factor}
                    for s, i in zip(scales, anchor_idx)}
    anchors_ok = all(v["abs_drift"] <= v["limit"] for v in anchor_drift.values())
    y_hi = hi["luminance"]["mean"]
    y_lo = lo["luminance"]["mean"]
    y_limit = float(precision["luminance_relative_halfwidth"]) * float(precision["drift_factor"])
    y_drift = abs(y_hi - y_lo) / y_hi if y_hi else math.inf
    return dict(budgets=[lo_key, hi_key], anchors=anchor_drift,
                anchors_pass=anchors_ok,
                luminance=dict(relative_drift=float(y_drift), limit=y_limit),
                luminance_pass=bool(y_drift <= y_limit))


def _pair_stats(disk_rows, point_rows, wl, v_curve, anchor_idx, sun, protocol,
                pair_cfg):
    disk = sorted(disk_rows, key=lambda r: int(r["seed"]))
    point = sorted(point_rows, key=lambda r: int(r["seed"]))
    if [int(r["seed"]) for r in disk] != [int(r["seed"]) for r in point]:
        raise ValueError(f"source-pairs/{sun}: seed sets do not match the disk batch")
    mismatched = [int(d["seed"]) for d, p in zip(disk, point)
                  if d["fingerprint"] != p["fingerprint"]]
    if mismatched:
        raise ValueError(f"source-pairs/{sun}: fingerprints differ for seeds "
                         f"{mismatched} -- native paths changed between modes")
    delta = (np.array([r["split_total"] for r in disk])
             - np.array([r["split_total"] for r in point]))
    disk_mean = np.array([r["split_total"] for r in disk]).mean(axis=0)
    confidence = protocol["direct"]["analysis"]["student"]["confidence"]
    mean = delta.mean(axis=0)
    half = student_halfwidth(delta, confidence)
    anchors = {}
    for i in anchor_idx:
        lo, hi_ = mean[i] - half[i], mean[i] + half[i]
        anchors[str(float(wl[i]))] = dict(
            delta_mean=float(mean[i]), delta_halfwidth=float(half[i]),
            interval=[float(lo), float(hi_)],
            relative_to_disk_mean=float(mean[i] / disk_mean[i])
            if disk_mean[i] else None,
            verdict="identified" if (lo > 0 or hi_ < 0) else "contains-zero")
    return dict(n=len(delta),
                seeds=[int(r["seed"]) for r in disk],
                delta_mean=mean.tolist(),
                delta_student_halfwidth=half.tolist(),
                anchors=anchors,
                fingerprints_paired=len(mismatched) == 0,
                seconds=float(sum(r["seconds"] for r in point)))


#: the first formal run wave (2026-11-04, before the fix recorded in the
#: runner) multiplied Darwin's already-bytes ru_maxrss by 1024
RSS_ERRATUM = ("direct/source-pairs records executed on Darwin between "
               "2026-11-04 14:44 and 18:36 archived max_rss_bytes with a "
               "1024x unit error; divide archived values by 1024 for true "
               "bytes. Records are frozen; this note is the erratum.")


def _cost_stats(data_dir, disk_meta):
    state_path = data_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return dict(stage_state=state.get("stages", {}), rss_erratum=RSS_ERRATUM,
                batches={bid: {"created": meta.get("created"),
                               "photons": meta.get("photons_per_wavelength"),
                               "mode": meta.get("mode"),
                               "hardware": meta.get("hardware")}
                         for bid, meta in disk_meta.items()})


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def build_manifest(data_dir: Path) -> dict:
    files = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "summary.json"):
            files[str(path.relative_to(data_dir))] = sha256_file(path)
    return dict(schema="deep-twilight-reference/manifest/1",
                generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=files)


def verify_manifest(data_dir: Path, stored: dict) -> list:
    current = build_manifest(data_dir)
    problems = []
    for name, digest in current["files"].items():
        if name not in stored["files"]:
            problems.append(f"new file not in manifest: {name}")
        elif stored["files"][name] != digest:
            problems.append(f"hash mismatch: {name}")
    for name in stored["files"]:
        if name not in current["files"]:
            problems.append(f"manifest lists missing file: {name}")
    return problems


def _deep_equal(a, b, path="$", problems=None):
    if problems is None:
        problems = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            if key not in a or key not in b:
                problems.append(f"{path}.{key}: present in only one summary")
            else:
                _deep_equal(a[key], b[key], f"{path}.{key}", problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _deep_equal(x, y, f"{path}[{i}]", problems)
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0):
            problems.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        problems.append(f"{path}: {a!r} vs {b!r}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="deep_twilight_reference data directory")
    parser.add_argument("--verify-only", action="store_true",
                        help="recompute and compare against the stored "
                             "summary; write nothing")
    args = parser.parse_args(argv)
    data_dir = args.directory
    protocol_path = data_dir / "protocol.json"
    if not protocol_path.exists():
        protocol_path = data_dir / "protocol.snapshot.json"
    protocol = json.loads(protocol_path.read_text())

    summary = compute_summary(data_dir, protocol)

    if args.verify_only:
        summary_path = data_dir / "summary.json"
        if not summary_path.exists():
            print("VERIFY FAIL: no summary.json to verify against", flush=True)
            return 1
        stored = json.loads(summary_path.read_text())
        stored.pop("generated", None)
        current = dict(summary)
        current.pop("generated", None)
        problems = _deep_equal(current, stored)
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            problems.append("manifest.json missing")
        else:
            problems.extend(verify_manifest(
                data_dir, json.loads(manifest_path.read_text())))
        if problems:
            print(f"VERIFY FAIL: {len(problems)} problem(s):", flush=True)
            for p in problems[:50]:
                print(f"  - {p}", flush=True)
            return 1
        print("VERIFY OK: summary recomputed from raw matches; "
              "manifest hashes intact", flush=True)
        return 0

    manifest = build_manifest(data_dir)
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=1,
                                                      allow_nan=False) + "\n")
    print(json.dumps(dict(wrote=["manifest.json", "summary.json"],
                          verdicts={sun: stats.get("verdict", "n/a")
                                    for sun, stats in summary.get("direct", {}).items()},
                          pairs={sun: sorted({a["verdict"] for a in p["anchors"].values()})
                                 for sun, p in summary.get("source_pairs", {}).items()}),
                    indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
