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


#: verified compatibility mapping for protocol generations older than the
#: current file.  Each entry: executed protocol SHA256 -> human-verified
#: statement of what the generation changed relative to the current file.
#: A batch whose meta pins an unknown protocol generation is REJECTED, not
#: silently accepted; the mapping is the auditable witness for the archived
#: pre-candidate-section wave.
KNOWN_PROTOCOL_GENERATIONS = {
    "f6da541115dadeb612f7f9ec8eaf9449858f780b33c700fa0a039f1f37274f1e": {
        "statement":
            "direct/source-pairs wave (2026-11-04): identical scenario/"
            "source/direct/precision/luminance sections; candidate+"
            "sensitivity sections and the candidate wall-clock entry were "
            "appended later. Verified against the current contract for "
            "every stage-relevant field checked here.",
        # the historical protocol carried no candidate/sensitivity
        # sections, so this entry must never authorize those stages
        "stages": ("direct", "source-pairs"),
        # the exact target protocol whose stage-relevant sections this
        # compatibility was human-verified against; any other target
        # protocol must be re-verified, never auto-inherited
        "verified_target_sha256":
            "23b46bd6784b91181c34cd5ff05656ce86f9aeecfa2bbe29c0728f1b432c87a1",
    },
}
#: the PHYSICS anchor files whose hashes define the traced physics of
#: every comparison.  A difference in any of these refuses the comparison
#: unconditionally -- no compatibility entry may excuse it.
PHYSICS_ANCHOR_FILES = (
    "skycolor/rt/mc.py",
    "tools/point_solar_table.py",
    "tools/unified_solar_transport.py",
    "tools/twilight_second_order.py",
)
ORCHESTRATION_FILE = "tools/deep_twilight_reference.py"


def execution_fingerprint(code_sha256) -> str:
    """Canonical full execution identity: the complete per-file hash
    projection of the executing implementation (every recorded file,
    physics anchors and orchestration alike)."""
    return json.dumps(code_sha256, sort_keys=True)


def _physics_projection(fp: str) -> str:
    return json.dumps({k: json.loads(fp).get(k)
                       for k in PHYSICS_ANCHOR_FILES}, sort_keys=True)


def _archived_fp(orch_sha256, *, candidate_sha256=None) -> str:
    """Full fingerprint of one archived implementation generation.

    The archived waves all share these physics-anchor hashes (verified
    bit-identical across every batch in the data directory); only the
    orchestration file (and, for the candidate wave, the candidate
    estimator) varied.  Used solely to build the verified-compatibility
    keys below, so the map binds the exact archived fingerprint pairs.
    """
    files = {
        "skycolor/rt/mc.py":
            "b8cab675585fe72fd74340e57cd24b66c2f757c221398cd83fa9a74a858143df",
        "tools/deep_twilight_reference.py": orch_sha256,
        "tools/point_solar_table.py":
            "c189f8c043a2da0b234b57c36d3e1b5f5c6ed459a7c7a45e3b7048f4bf51d084",
        "tools/unified_solar_transport.py":
            "6ed1b03c3c49ad748540064d90ce8306b8855f08d2409ba9a1cc51e1a654dea2",
    }
    if candidate_sha256:
        files["tools/twilight_second_order.py"] = candidate_sha256
    return execution_fingerprint(files)


# archived orchestration-file generations (extracted from the frozen
# records, not invented): direct wave ran on two versions, source-pairs on
# two, candidate on one, sensitivity on one -- plus the stale hash the
# sensitivity batch meta pinned from before its run started.
_ORCH_DIRECT_A = "0326e2ebe70d1e4abeea77c8c808229d243e28b5f0146b499283d75c0db04c62"
_ORCH_DIRECT_B = "4be29c6c063dd6fbe361148a5476afab2dbb84e68d6c05aa6d6edf1a5277e856"
_ORCH_CANDIDATE = "9690e774b8cf8cd1767993fd2e00987558995591ce2c86092969dccbddc05489"
_ORCH_PAIRS_S0_SENS = "2b9d608f9623ec9d3292d25e1cd2a16fbe2781fcdf11089482fd578cb957a618"
_ORCH_PAIRS_S1 = "722942e6802ecb114532c898302aa1fcc73fb61a4ddedb0e1c5a2c1933378ebe"
_ORCH_SENS_META_STALE = "78a30731e803bb996dd51f009c526ed994eb569cf9b6a8b98dd165789645fca2"

#: verified implementation-compatibility records.  Key: (comparison wave,
#: fingerprint-low, fingerprint-high), the complete execution fingerprints
#: sorted.  A fingerprint difference inside a wave is REFUSED unless this
#: exact (wave, pair) binding exists -- a blanket "orchestration-only
#: differences are fine" rule was removed: the archived bookkeeping-only
#: audit covers these specific pairs, not any future unknown version.
VERIFIED_IMPLEMENTATION_COMPAT = {
    ("direct", _archived_fp(_ORCH_DIRECT_A), _archived_fp(_ORCH_DIRECT_B)):
        "orchestration-only diff (state merge + stage-budget accounting) "
        "within the direct wave; physics anchors bit-identical across all "
        "archived records; verified during code-review round 4",
    ("source-pairs",
     _archived_fp(_ORCH_DIRECT_A), _archived_fp(_ORCH_PAIRS_S0_SENS)):
        "orchestration-only diff between the direct wave and the "
        "source-pairs run; physics anchors bit-identical; verified during "
        "code-review round 4",
    ("source-pairs",
     _archived_fp(_ORCH_DIRECT_B), _archived_fp(_ORCH_PAIRS_S1)):
        "orchestration-only diff between the direct wave and the "
        "source-pairs run; physics anchors bit-identical; verified during "
        "code-review round 4",
    ("sensitivity",
     _archived_fp(_ORCH_PAIRS_S0_SENS), _archived_fp(_ORCH_SENS_META_STALE)):
        "batch meta pinned a stale orchestration hash from before the "
        "sensitivity run started; the records carry the executed truth; "
        "verified during code-review round 3",
}


def require_same_execution(wave: str, fp_a: str, fp_b: str, *,
                           context: str) -> str | None:
    """Single contract entry for implementation compatibility.

    Two batches (or a batch meta and its records) may enter one
    comparison only when their full execution fingerprints are equal, or
    when VERIFIED_IMPLEMENTATION_COMPAT binds this exact (wave, pair).
    A PHYSICS-anchor difference is refused unconditionally: those files
    define the traced physics, and no audited exception covers them.
    Callers must pass identities that already passed verified_identity()
    -- this entry certifies equality/compatibility, never content.
    Returns the verified statement when a mapped exception applies
    (the caller records it as an auditable note).
    """
    if fp_a == fp_b:
        return None
    if _physics_projection(fp_a) != _physics_projection(fp_b):
        raise ValueError(
            f"{context}: PHYSICS implementation differs across the "
            f"{wave!r} comparison; no compatibility entry may excuse this "
            "-- the batches are not comparable as independent repeats or "
            "as a budget/estimator pair")
    lo, hi = sorted((fp_a, fp_b))
    statement = VERIFIED_IMPLEMENTATION_COMPAT.get((wave, lo, hi))
    if statement is None:
        raise ValueError(
            f"{context}: execution fingerprints differ within the "
            f"{wave!r} comparison and no verified compatibility entry "
            f"binds this exact pair ({lo[:12]}... vs {hi[:12]}...) -- "
            "comparison refused; extend VERIFIED_IMPLEMENTATION_COMPAT "
            "only with a human-verified (wave, fingerprint-pair) binding")
    return statement


#: per-stage required implementation-identity CONTENT: the files whose
#: hashes every code_sha256 of that stage must carry.  The candidate
#: stage depends on the candidate estimator itself; the direct,
#: source-pairs and sensitivity waves predate that file and are
#: contractually exempt from it (their archived identities name exactly
#: the four base files).
_STAGE_BASE_IDENTITY_FILES = (
    "skycolor/rt/mc.py",
    "tools/deep_twilight_reference.py",
    "tools/point_solar_table.py",
    "tools/unified_solar_transport.py",
)
STAGE_REQUIRED_IDENTITY_FILES = {
    "direct": _STAGE_BASE_IDENTITY_FILES,
    "source-pairs": _STAGE_BASE_IDENTITY_FILES,
    "candidate": _STAGE_BASE_IDENTITY_FILES
                 + ("tools/twilight_second_order.py",),
    "sensitivity": _STAGE_BASE_IDENTITY_FILES,
}


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value.lower()))


def verified_identity(stage, code_sha256, context) -> str:
    """Validate the CONTENT of an implementation identity and return its
    canonical fingerprint.

    An identity that does not name the stage's physics (an empty or
    partial mapping, or one carrying non-sha256 values) must never be
    serialisable into a comparison identity: equality of two hollow
    identities proves nothing about which implementations ran.
    """
    if not isinstance(code_sha256, dict) or not code_sha256:
        raise ValueError(
            f"{context}: code_sha256 must be a non-empty file-hash "
            "mapping -- an absent implementation identity cannot be "
            "compared")
    required = STAGE_REQUIRED_IDENTITY_FILES[stage]
    missing = [f for f in required if not code_sha256.get(f)]
    if missing:
        raise ValueError(
            f"{context}: implementation identity lacks the stage-required "
            f"files {missing} -- it cannot certify that the stage's "
            "physics ran under one implementation")
    malformed = sorted(f for f, h in code_sha256.items()
                       if not _is_sha256(h))
    if malformed:
        raise ValueError(
            f"{context}: implementation identity carries non-sha256 "
            f"values for {malformed}")
    return execution_fingerprint(code_sha256)


def require_protocol_generation(meta_protocol, protocol_sha_now, context,
                                notes, *, stage=None) -> None:
    """The single protocol-generation gate: the executed protocol must
    equal the current one or be bound by a KNOWN_PROTOCOL_GENERATIONS
    entry whose verified STAGE set covers this stage; anything else
    (including an absent identity) refuses.  All inputs -- JSONL batches
    and JSON evidence files alike -- pass through this one rule, and a
    historical entry never authorizes a stage it was not verified
    against."""
    if protocol_sha_now is None:
        return  # no current-protocol context to compare against
    if meta_protocol is None:
        raise ValueError(
            f"{context}: no protocol_sha256 recorded -- the executed "
            "protocol identity is required")
    if meta_protocol == protocol_sha_now:
        return
    entry = KNOWN_PROTOCOL_GENERATIONS.get(meta_protocol)
    if entry is not None:
        allowed = entry["stages"]
        if stage is not None and stage not in allowed:
            raise ValueError(
                f"{context}: protocol generation {meta_protocol[:12]}... "
                f"is only verified for stages {list(allowed)}, not "
                f"{stage!r} -- a historical compatibility entry must not "
                "authorize a stage it was never verified against")
        target = entry.get("verified_target_sha256")
        if target is not None and protocol_sha_now != target:
            raise ValueError(
                f"{context}: protocol generation {meta_protocol[:12]}... "
                "was verified against target protocol "
                f"{target[:12]}..., not the current {protocol_sha_now[:12]}"
                "... -- a changed target physical/analysis contract must "
                "be re-verified, never auto-inherited from a known hash")
        notes.append(
            f"{context}: executed under known protocol generation "
            f"{meta_protocol[:12]}... ({entry['statement']})")
        return
    raise ValueError(
        f"{context}: executed under unknown protocol generation "
        f"{meta_protocol[:12]}...; no verified compatibility mapping "
        "exists -- extend KNOWN_PROTOCOL_GENERATIONS only after "
        "verifying the stage-relevant sections are equivalent")


def check_batch_contract(records, meta, expect, *, stage, batch_id,
                         protocol_sha_now=None):
    """Records and batch meta must agree with the pre-registered contract.

    Checks the semantic identity (sun elevation, budget, mode, source and
    protocol hashes) of every record against ``expect`` and the batch meta;
    cross-batch protocol-hash drift is reported as an auditable note, not
    silently merged.  Returns ``(verified_execution_fingerprint, notes)``:
    the fingerprint is the batch's audited implementation identity, which
    every inter-batch comparison must consume -- re-projecting identities
    downstream (e.g. normalizing a missing one to ``{}``) is exactly the
    fail-open path this contract exists to close.
    """
    notes = []
    meta_protocol = meta.get("protocol_sha256")
    missing_fields = set()
    filled_from_meta = set()
    for row in records:
        for key, want in expect.items():
            got = row.get(key)
            if got is None:
                # pre-fix instrumentation: the record lacks the field.  The
                # value is accepted only when the batch meta pins it AND it
                # matches the protocol expectation (checked below when the
                # meta pins an expect-listed field); otherwise rejected.
                if meta.get(key) is not None:
                    if meta[key] != want:
                        raise ValueError(
                            f"{stage}/{batch_id} seed {row['seed']}: batch "
                            f"meta {key}={meta[key]!r} contradicts the "
                            f"protocol requirement {want!r}")
                    filled_from_meta.add(key)
                else:
                    missing_fields.add(key)
            elif got != want:
                raise ValueError(
                    f"{stage}/{batch_id} seed {row['seed']}: contract field "
                    f"{key} is {got!r}, protocol requires {want!r}")
        # implementation identity is a REQUIRED contract field whose
        # content is verified below; a record without it is a pre-fix
        # wave this generation does not support
        if row.get("code_sha256") is None:
            missing_fields.add("code_sha256")
        # executed-protocol identity: a record that carries it must match
        # the batch meta (records predating the field fall through to the
        # meta fallback / known-generation gate)
        if (meta.get("protocol_sha256") is not None
                and row.get("protocol_sha256") is not None
                and row["protocol_sha256"] != meta["protocol_sha256"]):
            raise ValueError(
                f"{stage}/{batch_id} seed {row['seed']}: record "
                "protocol_sha256 differs from the batch meta provenance")
    # per-batch implementation identity through the single contract entry.
    # Identity CONTENT is verified before anything is serialised into a
    # fingerprint (round-7: an empty or partial mapping must not become a
    # "verified" comparison identity), and none of this is conditional on
    # the meta carrying the field: a batch without a pinned, well-formed
    # execution contract is refused outright (fail-closed)
    record_fps = set()
    for row in records:
        if row.get("code_sha256") is None:
            continue  # already recorded in missing_fields above
        record_fps.add(verified_identity(
            stage, row["code_sha256"],
            f"{stage}/{batch_id} seed {row['seed']}"))
    if len(record_fps) > 1:
        raise ValueError(
            f"{stage}/{batch_id}: records span multiple execution "
            "implementations -- mixed-implementation batches must not be "
            "merged as independent repeats")
    meta_code = meta.get("code_sha256")
    if meta_code is None:
        raise ValueError(
            f"{stage}/{batch_id}: batch meta lacks the execution contract "
            "(code_sha256) -- batches without a pinned implementation "
            "identity are refused, not silently compared")
    meta_fp = verified_identity(stage, meta_code,
                                f"{stage}/{batch_id} (batch meta)")
    verified_fp = meta_fp
    if record_fps:
        rec_fp = next(iter(record_fps))
        statement = require_same_execution(
            stage, meta_fp, rec_fp,
            context=f"{stage}/{batch_id} (batch meta vs records)")
        if statement:
            notes.append(
                f"{stage}/{batch_id}: meta and records differ under a "
                f"verified compatibility exception: {statement}")
        verified_fp = rec_fp  # the records are the executed truth
    if missing_fields:
        raise ValueError(
            f"{stage}/{batch_id}: records lack {sorted(missing_fields)} and "
            "the batch meta cannot substitute them; this generation is not "
            "a known pre-fix wave")
    if filled_from_meta:
        notes.append(f"{stage}/{batch_id}: records lack {sorted(filled_from_meta)}; "
                     "values taken from the batch meta provenance and "
                     "validated against the protocol")
    require_protocol_generation(meta_protocol, protocol_sha_now,
                                f"{stage}/{batch_id}", notes, stage=stage)
    return verified_fp, notes


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
def resolve_protocol_path(data_dir: Path) -> Path:
    """The run directory's protocol file: protocol.json, or the validated
    snapshot a fresh --out run root received.  If both exist they must
    agree byte-for-byte: a run root carrying two different protocol
    identities has no single authority (a later protocol.json must never
    silently shadow the frozen snapshot, else stages executed under
    different contracts each report complete while no one protocol can
    summarize the whole run root), so the conflict is refused outright.

    Raises FileNotFoundError when neither file exists; ValueError when
    the two identity files disagree.
    """
    path = data_dir / "protocol.json"
    snapshot = data_dir / "protocol.snapshot.json"
    if path.exists() and snapshot.exists():
        if path.read_bytes() != snapshot.read_bytes():
            raise ValueError(
                f"{data_dir}: protocol.json and protocol.snapshot.json "
                "disagree -- the run root carries two protocol identities; "
                "resolve the conflict explicitly before any further "
                "execution or summarization")
        return path
    if path.exists():
        return path
    if snapshot.exists():
        return snapshot
    raise FileNotFoundError(
        f"{data_dir}: neither protocol.json nor protocol.snapshot.json -- "
        "a run root must carry a protocol snapshot to be summarizable")


def compute_summary(data_dir: Path, protocol: dict, *,
                    protocol_path: Path | None = None,
                    allow_incomplete: bool = False) -> dict:
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
    resolved_protocol = protocol_path or resolve_protocol_path(data_dir)
    protocol_sha_now = sha256_file(resolved_protocol)
    contract_notes = []
    out = dict(schema="deep-twilight-reference/summary/1",
               generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               protocol_sha256=protocol_sha_now,
               protocol_file=resolved_protocol.name,
               grid=wl.tolist(), anchors_nm=anchors)

    # ---- direct batches --------------------------------------------------
    disk_records = {}
    disk_meta = {}
    batch_fp = {}
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
            batch_fp[batch_id], batch_notes = check_batch_contract(
                records, meta,
                dict(sun_elevation_deg=sun, photons_per_wavelength=photons,
                     mode="disk",
                     table_sha256=protocol["source"]["full81_table"]["sha256"]),
                stage="direct", batch_id=batch_id,
                protocol_sha_now=protocol_sha_now)
            contract_notes.extend(batch_notes)
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
    # cross-budget comparisons run through the single contract entry,
    # consuming the verified identities the batch checks returned (never a
    # re-projection): the four batches need one compatible execution
    # identity (verified orchestration-only exceptions pass as auditable
    # notes appended to the single contract_notes list)
    direct_fps = sorted(set(batch_fp.values()))
    if len(direct_fps) > 1:
        statements = [require_same_execution(
            "direct", direct_fps[0], other,
            context="direct wave (cross-budget comparison)")
            for other in direct_fps[1:]]
        unique = list(dict.fromkeys(s for s in statements if s))
        contract_notes.append(
            "direct wave spans multiple orchestration versions under "
            f"verified exceptions: {'; '.join(unique)}")
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
        pair_fp, batch_notes = check_batch_contract(
            records, meta,
            dict(sun_elevation_deg=sun, photons_per_wavelength=photons,
                 mode="point",
                 table_sha256=protocol["source"]["full81_table"]["sha256"]),
            stage="source-pairs", batch_id=batch_id,
            protocol_sha_now=protocol_sha_now)
        contract_notes.extend(batch_notes)
        disk_rows = disk_records[(sun, photons)]
        statement = require_same_execution(
            "source-pairs", pair_fp, batch_fp[f"s{sun_index}b{top_budget_index}"],
            context=f"source-pairs/{batch_id} vs the disk batch it is "
                    "paired against")
        if statement:
            contract_notes.append(
                f"source-pairs/{batch_id}: orchestration file version "
                f"differs from the direct wave under a verified "
                f"exception: {statement}")
        pairs[str(sun)] = _pair_stats(disk_rows, records, wl, v_curve,
                                      anchor_idx, sun, protocol, pair_cfg)
    out["source_pairs"] = pairs
    out["contract_notes"] = contract_notes
    # ---- candidate experiment (plan Step 3) -------------------------------
    # stage participation is decided by the protocol, not by what happens
    # to exist under raw/: a declared-but-missing stage either refuses the
    # complete summary or -- only with the explicit allow_incomplete
    # escape -- is recorded in incomplete_stages instead of silently
    # vanishing from the deliverable
    missing_stages = [s for s in ("candidate", "sensitivity")
                      if s in protocol and not (data_dir / "raw" / s).exists()]
    if missing_stages and not allow_incomplete:
        raise ValueError(
            f"protocol declares stages with no data under raw/: "
            f"{missing_stages} -- a complete summary requires every "
            "protocol-declared stage; pass --allow-incomplete to emit an "
            "explicit incompleteness record instead")
    if "candidate" in protocol and (data_dir / "raw" / "candidate").exists():
        out["candidate"] = _candidate_stats(data_dir, protocol, wl, anchor_idx,
                                            out["direct"],
                                            protocol_sha_now=protocol_sha_now,
                                            contract_notes=contract_notes)
    # ---- sensitivity (plan Step 4) ----------------------------------------
    if "sensitivity" in protocol and (data_dir / "raw" / "sensitivity").exists():
        out["sensitivity"] = _sensitivity_stats(data_dir, protocol, wl,
                                                anchor_idx,
                                                protocol_sha_now=protocol_sha_now,
                                                contract_notes=contract_notes)
    out["cost"] = _cost_stats(data_dir, disk_meta)
    if missing_stages:
        out["incomplete_stages"] = missing_stages
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
                     anchor_idx, direct_section, *, protocol_sha_now=None,
                     contract_notes=None) -> dict:
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
    if contract_notes is None:
        contract_notes = []
    out = {}
    for sun_index, sun in enumerate(suns):
        tiers = {}
        tiers_fp = {}
        for tier_index, tier in enumerate(cand_cfg["tiers"]):
            batch_id = f"s{sun_index}t{tier_index}"
            seeds = [80000 + 10000 * sun_index + 1000 * tier_index + r
                     for r in range(repeats)]
            records, meta = load_batch(raw_dir / batch_id, seeds,
                                       stage="candidate", batch_id=batch_id)
            records = sorted(records, key=lambda r: int(r["seed"]))
            tiers_fp[str(tier_index)], batch_notes = check_batch_contract(
                records, meta,
                dict(sun_elevation_deg=sun,
                     candidate_samples=tier["candidate_samples"],
                     rest_photons=tier["rest_photons"],
                     mode="disk",
                     table_sha256=cand_cfg["table"]["sha256"]),
                stage="candidate", batch_id=batch_id,
                protocol_sha_now=protocol_sha_now)
            contract_notes.extend(batch_notes)
            # component closure and wavelength coverage per record
            want_wl = {f"{float(a):.1f}" for a in cand_cfg["wavelengths_nm"]}
            for row in records:
                got_wl = set(row["per_lambda"])
                if got_wl != want_wl:
                    raise ValueError(
                        f"candidate/{batch_id} seed {row['seed']}: wavelength "
                        f"set {sorted(got_wl)} != protocol {sorted(want_wl)}")
                if row.get("candidate_samples") not in (None, tier["candidate_samples"]):
                    raise ValueError(
                        f"candidate/{batch_id} seed {row['seed']}: "
                        f"candidate_samples {row.get('candidate_samples')!r} "
                        f"!= tier budget {tier['candidate_samples']}")
                for a_nm, v in row["per_lambda"].items():
                    if v.get("candidate_samples") not in (None, tier["candidate_samples"]):
                        raise ValueError(
                            f"candidate/{batch_id} seed {row['seed']} {a_nm}: "
                            f"per-lambda candidate_samples {v.get('candidate_samples')!r} "
                            f"!= tier budget {tier['candidate_samples']}")
                    recomposed = (v["deterministic_single"] + v["L2"]
                                  + v["rest_mean"])
                    if not math.isclose(v["total_cand"], recomposed,
                                        rel_tol=1e-12, abs_tol=0.0):
                        raise ValueError(
                            f"candidate/{batch_id} seed {row['seed']} {a_nm}: "
                            "total_cand != single + L2 + rest")
                    if not math.isclose(v["total_analog"],
                                        v["deterministic_single"]
                                        + v["mc_multiple"],
                                        rel_tol=1e-12, abs_tol=0.0):
                        raise ValueError(
                            f"candidate/{batch_id} seed {row['seed']} {a_nm}: "
                            "total_analog != single + mc_multiple")
                    if not math.isclose(v["rest_mean"],
                                        v["mc_multiple"] - v["c2nt_mean"],
                                        rel_tol=1e-12, abs_tol=0.0):
                        raise ValueError(
                            f"candidate/{batch_id} seed {row['seed']} {a_nm}: "
                            "rest_mean != mc_multiple - c2nt_mean")
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
                cand_var = float(cand_total.var(axis=0, ddof=1))
                analog_var = float(analog_total.var(axis=0, ddof=1))
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
                    # fixed-wall-clock efficiency: with T seconds of budget
                    # the mean's variance is Var_repeat * t / T, so the
                    # comparable quantity is variance x time -- a slower
                    # estimator with the same variance must NOT score better
                    candidate_var_x_time=float(
                        cand_var * repeat_seconds.mean()),
                    analog_var_x_time=float(
                        analog_var * max(analog_seconds.mean(), 1e-9)),
                    candidate_seconds_per_repeat=float(repeat_seconds.mean()),
                    analog_mc_seconds_per_repeat=float(analog_seconds.mean()))
            tiers[str(tier_index)] = dict(
                rest_photons=tier["rest_photons"],
                candidate_samples=tier["candidate_samples"],
                anchors=per_anchor,
                seconds=float(sum(r["seconds"] for r in records)))
        lo_tier, hi_tier = sorted(tiers, key=int)
        # the two tiers enter one budget-drift comparison, so their
        # execution identities must be compatible first: consume the
        # verified identities the batch checks returned -- a missing
        # contract is already refused upstream, never normalized to {}
        tier_stmt = require_same_execution(
            "candidate", tiers_fp[lo_tier], tiers_fp[hi_tier],
            context=f"candidate {sun}: tier drift ({lo_tier} vs {hi_tier})")
        if tier_stmt:
            contract_notes.append(
                f"candidate {sun}: tiers differ under a verified "
                f"compatibility exception: {tier_stmt}")
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
            c = hi_anchor["candidate_var_x_time"]
            a = hi_anchor["analog_var_x_time"]
            efficiency[str(a_nm)] = dict(
                candidate_var_x_time=c,
                analog_var_x_time=a,
                ratio=c / a if a else None,
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


def _require_anchor_vector(name, values, n_anchor, context):
    """Anchor-grid contract for deterministic refinement evidence.

    Every protocol-required vector (baseline and each refinement variant)
    must be a finite 1-D array of exactly the anchor count -- NumPy
    broadcasting would otherwise turn missing wavelength evidence into a
    silently wrong physical comparison.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_anchor,):
        raise ValueError(
            f"{context}: {name} must be a 1-D vector of exactly "
            f"{n_anchor} anchor values, got shape {arr.shape} -- missing "
            "wavelength evidence must not be broadcast into a comparison")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{context}: {name} must be finite on every anchor")
    return arr


def validated_singles(payload, protocol, *, protocol_sha_now=None,
                      notes=None) -> dict:
    """THE singles-evidence contract, shared by the summarizer and the
    executor's recovery path (one authoritative implementation).

    Validates a singles-refinement payload against the protocol: the
    protocol generation identity, the exact protocol depth set, the exact
    protocol item metadata set, and the anchor-vector contract (exact
    anchor count, finite values, positive baseline) for the baseline and
    every protocol-required refinement.  Returns the validated by_sun
    mapping; raises ValueError with the refusing reason otherwise.

    'already present' must never be mistaken for 'complete': partial
    evidence (a dropped depth, a dropped item, a truncated vector) fails
    here, so both tools refuse it from the same rule.
    """
    notes = notes if notes is not None else []
    sens = protocol.get("sensitivity", {})
    proto_single_items = [d.split(" (")[0].strip()
                          for d in sens.get("single_items", [])]
    require_protocol_generation(payload.get("protocol_sha256"),
                                protocol_sha_now, "sensitivity singles",
                                notes, stage="sensitivity")
    got_items = set(payload.get("items", {}))
    if got_items != set(proto_single_items):
        raise ValueError(
            f"sensitivity singles: refinement-file items {sorted(got_items)} "
            f"!= protocol single_items {sorted(proto_single_items)} -- the "
            "payload metadata must match the protocol, not decide the "
            "reported comparison set")
    want_suns = {str(s) for s in protocol["scenario"]["sun_elevations_deg"]}
    want_single_items = {_slug(d) for d in proto_single_items}
    n_anchor = len(protocol["scenario"]["anchor_wavelengths_nm"])
    got_suns = set(payload.get("by_sun", {}))
    if got_suns != want_suns:
        raise ValueError(
            f"sensitivity singles: depths {sorted(got_suns)} != protocol "
            f"{sorted(want_suns)}")
    for sun, entries in payload["by_sun"].items():
        missing_items = want_single_items - set(entries)
        if missing_items:
            raise ValueError(
                f"sensitivity singles {sun}: missing protocol-required "
                f"items {sorted(missing_items)}")
        base = _require_anchor_vector("baseline", entries["baseline"],
                                      n_anchor, f"sensitivity singles {sun}")
        if not np.any(base > 0):
            raise ValueError(
                f"sensitivity singles {sun}: baseline must be positive on "
                f"the {n_anchor}-anchor grid")
        for item in proto_single_items:
            _require_anchor_vector(f"refinement {item!r}",
                                   entries[_slug(item)], n_anchor,
                                   f"sensitivity singles {sun}")
    return payload["by_sun"]


def _sensitivity_stats(data_dir: Path, protocol: dict, wl: np.ndarray,
                       anchor_idx, *, protocol_sha_now=None,
                       contract_notes=None) -> dict:
    """Paired same-seed drift of every refinement against its baseline."""
    sens = protocol.get("sensitivity", {})
    anchors_nm = [float(wl[i]) for i in anchor_idx]
    confidence = protocol["direct"]["analysis"]["student"]["confidence"]
    repeats = int(sens.get("repeats", 16))
    suns = protocol["scenario"]["sun_elevations_deg"]
    raw_dir = data_dir / "raw" / "sensitivity"
    if contract_notes is None:
        contract_notes = []
    # protocol entries are "key (description)"; the records carry the key.
    # The protocol-derived sets below are the SINGLE authority for both
    # validation and summary generation -- the refinement file's own
    # "items" metadata is checked against them, never consulted to decide
    # what gets computed and reported.
    required_items = [i.split(" (")[0].strip() for i in sens.get("mc_items", [])]
    proto_single_items = [d.split(" (")[0].strip()
                          for d in sens.get("single_items", [])]
    singles_path = raw_dir / "singles-refinement.json"
    if not singles_path.exists():
        raise ValueError(
            "sensitivity: singles-refinement.json missing -- the "
            "deterministic refinement evidence is required for a complete "
            "sensitivity summary")
    payload = json.loads(singles_path.read_text())
    # THE shared evidence contract (also consumed by the executor's
    # recovery path): protocol identity, depth/item sets, anchor vectors.
    # validated_singles is the only implementation of this contract -- the
    # duplicate checks that used to live here were removed so the two
    # entry points can never drift apart again.
    validated_by_sun = validated_singles(payload, protocol,
                                         protocol_sha_now=protocol_sha_now,
                                         notes=contract_notes)
    out = dict(mc_items=required_items, by_sun={})
    for sun_index, sun in enumerate(suns):
        batch_id = f"s{sun_index}mc"
        seeds = [110000 + 1000 * sun_index + r for r in range(repeats)]
        records, meta = load_batch(raw_dir / batch_id, seeds,
                                   stage="sensitivity", batch_id=batch_id)
        sens_budget = int(sens.get("photons_per_wavelength", 100_000))
        want_mc_items = set(required_items) | {"baseline"}
        for row in records:
            got_mc_items = set(row.get("items", {}))
            if got_mc_items != want_mc_items:
                raise ValueError(
                    f"sensitivity/{batch_id} seed {row['seed']}: items "
                    f"{sorted(got_mc_items)} != protocol set exactly "
                    f"(missing {sorted(want_mc_items - got_mc_items)}, "
                    f"unexpected {sorted(got_mc_items - want_mc_items)}) "
                    "-- the sensitivity summary must cover every "
                    "protocol-declared experiment and nothing else")
            # stage-result contract: every consumed quantity closes --
            # total is a derived value (as in the direct/candidate
            # checks) and a tampered total must not become a certified
            # solar-source discretisation sensitivity
            for item_name, item in row["items"].items():
                for a_key, v in item["per_lambda"].items():
                    recomposed = (v["deterministic_single"]
                                  + v["multiple"])
                    if not math.isclose(v["total"], recomposed,
                                        rel_tol=1e-12, abs_tol=0.0):
                        raise ValueError(
                            f"sensitivity/{batch_id} seed {row['seed']} "
                            f"{item_name} {a_key}: total != "
                            "deterministic_single + multiple -- the "
                            "sensitivity comparison consumes derived "
                            "quantities that must close")
        _sens_fp, batch_notes = check_batch_contract(
            records, meta,
            dict(sun_elevation_deg=sun,
                 photons_per_wavelength=sens_budget, mode="disk",
                 table_sha256=protocol["candidate"]["table"]["sha256"]),
            stage="sensitivity", batch_id=batch_id,
            protocol_sha_now=protocol_sha_now)
        contract_notes.extend(batch_notes)
        per_item = {}
        # "baseline" is part of the required record coverage but pairs
        # against itself, so it produces no drift entry
        for item in (i for i in required_items if i != "baseline"):
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
    singles = {}
    for sun, by in validated_by_sun.items():
        base = np.array(by["baseline"])
        entries = {}
        for item in proto_single_items:
            key = _slug(item)
            if key not in by:
                raise ValueError(
                    f"sensitivity singles {sun}: protocol item {item!r} "
                    f"has no refinement data under {key!r} -- the summary "
                    "must report every protocol-required comparison")
            variant = np.array(by[key])
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
    # per-group state files take precedence; the legacy combined
    # state.json (first formal wave) fills the groups without a file
    stages = {}
    legacy = data_dir / "state.json"
    if legacy.exists():
        stages.update(json.loads(legacy.read_text())["stages"])
    for path in sorted(data_dir.glob("state.*.json")):
        stages.update(json.loads(path.read_text())["stages"])
    return dict(stage_state=stages, rss_erratum=RSS_ERRATUM,
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
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="emit an explicit incomplete_stages record for "
                             "protocol-declared stages with no raw/ data, "
                             "instead of refusing the summary; generation "
                             "and --verify-only must use the same flag")
    args = parser.parse_args(argv)
    data_dir = args.directory
    protocol_path = resolve_protocol_path(data_dir)
    protocol = json.loads(protocol_path.read_text())

    summary = compute_summary(data_dir, protocol,
                              allow_incomplete=args.allow_incomplete)

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
            stored_manifest = json.loads(manifest_path.read_text())
            if stored_manifest["files"].get(protocol_path.name) is None:
                problems.append(
                    f"manifest does not pin {protocol_path.name}")
            problems.extend(verify_manifest(data_dir, stored_manifest))
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
