"""Deep-twilight spectral reference benchmark (-9/-12 deg, sea-level zenith).

This tool turns the single low-budget Monte Carlo numbers quoted in
``validation/results.json`` for -9 and -12 degrees solar elevation into a
defined, reproducible reference calculation: a fixed scenario/medium/source
contract, a pre-registered protocol (budgets, seeds, interval methods and
precision gates tied to the finite-disk source correction), and append-only
raw records whose summaries can be recomputed offline by
:mod:`tools.summarize_deep_twilight_reference`.

Contract (see ``validation/data/deep_twilight_reference/protocol.json``):

* Scenario: ``presets.clear`` (AOD550 0.05, 300 DU ozone, albedo 0.1,
  ground = observer = 0 km) on the full 81-wavelength 380-780 nm / 5 nm grid
  (``skycolor.constants.WL_COARSE``).
* Source: the archived full81 finite-solar-disk table (continuous refracted
  solar source), loaded by explicit label and SHA256 from the frozen
  unified-solar-source audit; the point-source limit (radius 0) is used for
  same-seed pairing.  No second solar-source implementation is written here.
* Transport: deterministic first order from :func:`single_zenith` (same
  finite source), Monte Carlo higher orders from
  :func:`skycolor.rt.mc.mc_radiance_split` on native straight diffuse paths
  with the default 50-order cap and Russian roulette; diagnostics of
  :mod:`skycolor.rt.mc` are recorded but never alter physics or RNG.
* Statistics: independent complete spectra are the sampling unit; per budget
  the mean, sample SD and Student-t 95 % conditional interval (df = 15) are
  reported, with a resampling-of-complete-repeats bootstrap as sensitivity
  analysis.  No seed selection, no outlier removal, and an all-zero batch is
  reported as a detection limit, never as evidence of a sampled-out tail.

Stages
------
``smoke``        tiny end-to-end run, scratch output only (``--out``).
``cost-probe``   full81 timing probe for scheduling, scratch output only.
``direct``       the formal 16-seed x {400k, 1.6M} x {-9, -12} baseline into
                 ``<data>/raw/direct/`` (shared wall-clock cap with pairs).
``source-pairs`` point-source pairing at the 1.6 M budget, same seeds, into
                 ``<data>/raw/source-pairs/``.
``candidate``    second-order candidate estimator experiment (Step 3).
``sensitivity``  discretisation/refinement sensitivity batches (Step 4).

Batch layout under the data directory::

    raw/<stage>/<batch-id>/attempt-NN.jsonl   one JSON record per completed run
    raw/<stage>/<batch-id>/meta.json          batch contract, attempts, cost
    state.json                                cumulative stage wall-clock

Records are appended and flushed one at a time; a rerun continues the newest
attempt and only computes seeds that are missing from it.  A batch with a
corrupt partial line is marked invalid in ``meta.json`` and any recomputation
goes to a fresh attempt; summaries must find exactly one valid complete
attempt per batch (duplicated recomputation never counts as an extra seed).

Run from the repository root::

    uv run python -m tools.deep_twilight_reference --protocol \\
        validation/data/deep_twilight_reference/protocol.json --stage smoke
"""
from __future__ import annotations

import argparse
import shutil
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402
from skycolor.rt import mc  # noqa: E402
from tools.point_solar_table import PointSolarTable  # noqa: E402
from tools.unified_solar_transport import (  # noqa: E402
    FiniteSolarSource, mc_solar_source, single_zenith)

#: code files whose SHA256 identifies the exact implementation of a record
CODE_FILES = [
    "skycolor/rt/mc.py",
    "tools/point_solar_table.py",
    "tools/unified_solar_transport.py",
    "tools/deep_twilight_reference.py",
    "tools/twilight_second_order.py",
]

#: stages that only write scratch output (never into the tracked data tree)
SCRATCH_STAGES = {"smoke", "cost-probe"}

#: wall-clock cap group per stage (protocol["wall_clock_caps_hours"] keys)
STAGE_CAP_GROUP = {
    "direct": "direct_and_source_pairs",
    "source-pairs": "direct_and_source_pairs",
    "candidate": "candidate",
    "sensitivity": "sensitivity",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def hardware_info() -> dict:
    """Platform, memory and thread-environment provenance for a batch."""
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):  # non-macOS
        brand = platform.processor() or "unknown"
    return dict(platform=platform.platform(), machine=platform.machine(),
                cpu_brand=brand, cpu_count=os.cpu_count(),
                thread_env={k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                python=sys.version.split()[0], numpy=np.__version__)


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far, in bytes.

    Darwin's ``ru_maxrss`` is already bytes; Linux reports KiB and needs the
    ×1024 conversion.  Records produced between 2026-11-04 14:44 and 18:36
    (all formal direct/pairs records) ran the erroneous both-platforms ×1024
    version, so their archived ``max_rss_bytes`` on Darwin are 1024× too
    large; the summarizer derives a corrected value instead of rewriting the
    frozen records.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(raw)            # already bytes
    return int(raw) * 1024         # Linux KiB -> bytes


# ---------------------------------------------------------------------------
# protocol / scenario / source contract
# ---------------------------------------------------------------------------
REQUIRED_PROTOCOL_KEYS = {
    "schema", "scenario", "source", "direct", "candidate", "precision",
    "luminance", "wall_clock_caps_hours",
}


def load_protocol(path: Path) -> dict:
    protocol = json.loads(Path(path).read_text())
    missing = REQUIRED_PROTOCOL_KEYS - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing keys: {sorted(missing)}")
    if protocol["schema"] != "deep-twilight-reference/1":
        raise ValueError(f"unknown protocol schema {protocol['schema']!r}")
    scenario = protocol["scenario"]
    if scenario["spectrum"] != {"grid": [380.0, 780.0, 5.0], "n_wavelengths": 81}:
        raise ValueError("this benchmark fixes the 380-780 nm / 5 nm contract")
    for anchor in scenario["anchor_wavelengths_nm"]:
        if not any(abs(anchor - wl) < 1e-9 for wl in WL_COARSE):
            raise ValueError(f"anchor {anchor} nm is not on the spectral grid")
    return protocol


def build_scenario(protocol: dict):
    """The frozen benchmark atmosphere, with contract assertions.

    The implementation realizes exactly ONE physical scenario; a protocol
    that declares anything else (different AOD, ozone, site heights, view
    geometry, sun azimuth or order cap) is refused BEFORE any computation
    instead of executing a different experiment under the same protocol
    hash.
    """
    scenario = protocol["scenario"]
    transport = scenario.get("transport", {})
    requirements = [
        ("preset", scenario.get("preset"), "clear"),
        ("aod550", float(scenario.get("aod550", -1)), 0.05),
        ("ozone_du", float(scenario.get("ozone_du", -1)), 300.0),
        ("ground_albedo", float(scenario.get("ground_albedo", -1)), 0.1),
        ("elevation_km", float(scenario.get("elevation_km", -1)), 0.0),
        ("ground_km", float(scenario.get("ground_km", -1)), 0.0),
        ("view_zenith_deg", float(scenario.get("view_zenith_deg", -1)), 0.0),
        ("view_azimuth_deg", float(scenario.get("view_azimuth_deg", -1)), 0.0),
        ("sun_azimuth_deg", float(scenario.get("sun_azimuth_deg", -1)), 0.0),
        ("max_scatter", int(transport.get("max_scatter", -1)), 50),
    ]
    for name, declared, implemented in requirements:
        if declared != implemented:
            raise ValueError(
                f"scenario contract violated: protocol declares {name}="
                f"{declared!r} but this tool implements {implemented!r}; "
                "a different physical scenario requires a different "
                "implementation, not a different protocol file")
    atmo = presets.clear(WL_COARSE)
    checks = [
        ("wavelength grid", np.array_equal(atmo.wl, WL_COARSE)),
        ("sea-level domain", float(atmo.z_bounds[0]) == 0.0
         and float(atmo.elevation_km) == 0.0),
        ("default albedo", float(atmo.ground_albedo[0]) == 0.1
         and len(np.unique(atmo.ground_albedo)) == 1),
    ]
    for name, ok in checks:
        if not ok:
            raise ValueError(f"scenario contract violated: {name}")
    return atmo


def load_source_table(protocol: dict, atmo):
    """Load the archived full81 table by explicit label + SHA256 contract."""
    source = protocol["source"]["full81_table"]
    path = ROOT / source["path"]
    digest = sha256_file(path)
    if digest != source["sha256"]:
        raise ValueError(
            f"archived solar table hash mismatch: {path} is {digest}, "
            f"protocol freezes {source['sha256']}; check the contract, "
            "never silently rebuild")
    table = PointSolarTable.load(path, atmo, expected_label=source["label"])
    if not np.array_equal(table.wl, atmo.wl):
        raise ValueError("loaded table does not span the benchmark grid")
    return table


def seed_for(sun_index: int, budget_index: int, repeat: int) -> int:
    """The pre-registered seed rule (plan section 3, fixed before any run)."""
    return 40000 + 10000 * sun_index + 1000 * budget_index + repeat


# ---------------------------------------------------------------------------
# one Monte Carlo record
# ---------------------------------------------------------------------------
def run_record(atmo, table, sun_elevation_deg, seed, photons_per_wavelength,
               deterministic_single, *, mode="disk", keep_per_path=False):
    """One independent complete-spectrum record (finite source, diagnostics on).

    ``deterministic_single`` is the (deterministic, cached by the caller)
    finite-source first-order spectrum for this sun/mode; ``split_total``
    is that spectrum + MC multiple under the *same* source contract.  The
    MC's own shell-based ``single`` field is kept as a cross-check only.
    """
    radius = 16.0 / 60.0 if mode == "disk" else 0.0
    source = FiniteSolarSource(atmo, table, radius_deg=radius)
    started = time.perf_counter()
    with mc_solar_source(source):
        result = mc.mc_radiance_split(
            atmo, 90.0 - sun_elevation_deg, [0.0], [0.0],
            n_photons=photons_per_wavelength, seed=seed, max_scatter=50,
            diagnostics=dict(keep_per_path=keep_per_path))
    seconds = time.perf_counter() - started
    split_total = deterministic_single + result["multiple"][0]
    return dict(
        sun_elevation_deg=sun_elevation_deg, seed=int(seed),
        photons_per_wavelength=int(photons_per_wavelength), mode=mode,
        seconds=seconds, max_rss_bytes=peak_rss_bytes(),
        fields={k: result[k][0].tolist() for k in
                ("total", "total_err", "single", "single_err",
                 "multiple", "multiple_err")},
        deterministic_single=deterministic_single.tolist(),
        split_total=split_total.tolist(),
        fingerprint=source.fingerprint.hexdigest(),
        nee_calls=source.n_calls, nee_points=source.n_points,
        diagnostics=_diagnostics_payload(result["diagnostics"], atmo.wl,
                                         keep_per_path=keep_per_path),
    )


def _diagnostics_payload(diag_list, wl, *, keep_per_path):
    """Compress per-wavelength diagnostics: scalars everywhere, detail at 450/550/650."""
    anchor_wl = (450.0, 550.0, 650.0)
    anchor_idx = [int(np.argmin(np.abs(wl - a))) for a in anchor_wl]
    per_wl = []
    for iw, d in enumerate(diag_list):
        pv = d["per_view"][0]
        per_wl.append(dict(
            wl=float(wl[iw]), nonzero=pv["nonzero"],
            max_fraction=pv["max_fraction"],
            top001_fraction=pv["top001_fraction"],
            effective_contributors=pv["effective_contributors"],
            capped_paths=d["capped_paths"], **d["kills"]))
    anchors = {}
    for iw in anchor_idx:
        d = diag_list[iw]
        pv = d["per_view"][0]
        entry = dict(wl=float(wl[iw]), n_paths=pv["n_paths"], s1=pv["s1"],
                     s2=pv["s2"], top=pv["top"],
                     max_fraction=pv["max_fraction"],
                     top001_fraction=pv["top001_fraction"],
                     effective_contributors=pv["effective_contributors"],
                     nee_order_hist=d["nee_order_hist"],
                     ground_order_hist=d["ground_order_hist"],
                     kills=d["kills"], capped_paths=d["capped_paths"])
        if keep_per_path:
            entry["per_path"] = pv["per_path"]
        anchors[str(float(wl[iw]))] = entry
    return dict(per_wl=per_wl, anchors=anchors)


# ---------------------------------------------------------------------------
# batch bookkeeping (append-only attempts)
# ---------------------------------------------------------------------------
#: fields that define a batch's computational contract, INCLUDING the
#: implementation identity (code_sha256): a resumed batch whose stored
#: contract differs -- budget, source, protocol OR implementation -- is
#: refused instead of silently mixing heterogeneous results under one
#: identity.  Compatible-with-history exceptions are explicit, auditable
#: meta edits, never silent appends.
CONTRACT_FIELDS = ("stage", "sun_index", "budget_index",
                   "photons_per_wavelength", "mode", "expected_seeds",
                   "table_sha256", "protocol_sha256", "code_sha256",
                   "candidate_samples", "rest_photons")


def contract_view(meta: dict) -> dict:
    """The batch-identity subset of a batch meta."""
    return {k: meta.get(k) for k in CONTRACT_FIELDS
            if meta.get(k) is not None}


class Batch:
    """One (stage, batch-id) group of records with resumable attempts.

    The newest attempt file is continued in place; seeds already present in
    it are never recomputed.  If its last line is corrupt (a crash during
    write), the attempt is marked invalid and ``next_attempt`` starts a new
    file -- a batch then has exactly one *valid* complete attempt.

    On reopen the stored computational contract (budget, source, protocol
    AND implementation hashes) is compared against the incoming
    ``meta_base``; a mismatch raises instead of appending records computed
    under a different identity into the old batch.  The only sanctioned way
    to run a different contract is a new run root (or an explicit, auditable
    meta edit that supersedes the old attempt before reopening) -- there is
    no in-place bypass.
    """

    def __init__(self, root: Path, batch_id: str, expected_seeds, meta_base: dict):
        self.dir = root / batch_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.batch_id = batch_id
        self.expected_seeds = [int(s) for s in expected_seeds]
        self.meta_path = self.dir / "meta.json"
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            incoming = contract_view(dict(meta_base,
                                          expected_seeds=self.expected_seeds))
            stored = contract_view(self.meta)
            if incoming != stored:
                raise ValueError(
                    f"{root.name}/{batch_id}: stored contract differs from "
                    f"this invocation; refusing to append. stored="
                    f"{json.dumps(stored, sort_keys=True)} incoming="
                    f"{json.dumps(incoming, sort_keys=True)}; use a new run "
                    "root for a different contract")
        else:
            self.meta = dict(meta_base, batch_id=batch_id, attempts=[],
                             expected_seeds=self.expected_seeds)
        self.attempt, self.path, self.done = self._open_attempt()

    def _open_attempt(self):
        attempts = self.meta["attempts"]
        # Adopt a complete attempt that already covers the pre-registered
        # seed set: a finished batch must never be recomputed into a second
        # complete attempt (duplicated recomputation is not extra evidence).
        for attempt in attempts:
            if attempt.get("status") != "complete":
                continue
            path = self.dir / f"attempt-{attempt['index']:02d}.jsonl"
            if not path.exists():
                continue
            seeds, corrupt = self._parse_lines(path)
            if not corrupt and seeds == set(self.expected_seeds):
                return attempt["index"], path, seeds
        if attempts and attempts[-1]["status"] == "open":
            index = attempts[-1]["index"]
            path = self.dir / f"attempt-{index:02d}.jsonl"
            done = self._read_valid_seeds(path)
            if done is None:  # corrupt trailing line: close as invalid
                attempts[-1]["status"] = "invalid-corrupt"
                attempts[-1]["seeds"] = sorted(
                    self._parse_lines(path)[0])
                path = self.dir / f"attempt-{index + 1:02d}.jsonl"
                index += 1
                attempts.append(dict(index=index, status="open", seeds=[]))
                done = set()
        else:
            index = attempts[-1]["index"] + 1 if attempts else 0
            path = self.dir / f"attempt-{index:02d}.jsonl"
            attempts.append(dict(index=index, status="open", seeds=[]))
            done = set()
        self._save_meta()
        return index, path, done

    @staticmethod
    def _parse_lines(path):
        """Return (valid seeds, corrupt_tail) for an attempt file."""
        seeds, corrupt = set(), False
        if not path.exists():
            return seeds, corrupt
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                seeds.add(int(row["seed"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                corrupt = True  # only the last line can be a partial write
                break
        return seeds, corrupt

    def _read_valid_seeds(self, path):
        seeds, corrupt = self._parse_lines(path)
        if corrupt:
            return None
        return seeds

    def record_seed(self, seed, payload: dict):
        line = json.dumps(payload, allow_nan=False)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.done.add(int(seed))

    def close(self, status):
        for attempt in self.meta["attempts"]:
            if attempt["status"] == "open":
                attempt["status"] = status
                attempt["path"] = f"attempt-{attempt['index']:02d}.jsonl"
        self._save_meta()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=1) + "\n")


# ---------------------------------------------------------------------------
# wall-clock budget state
# ---------------------------------------------------------------------------
def _group_state_path(data_dir: Path, group: str) -> Path:
    return data_dir / f"state.{group}.json"


def _read_group_state(data_dir: Path, group: str) -> dict | None:
    """This cap group's own state file (the only writer is its stage)."""
    path = _group_state_path(data_dir, group)
    if path.exists():
        return json.loads(path.read_text())["stages"][group]
    return None


def load_state(data_dir: Path, *, group: str | None = None) -> dict:
    """Stage view of the cost accounting.

    With ``group``, only that cap group is loaded (its own file, falling
    back to the legacy combined ``state.json`` archived by the first
    formal wave).  Without it, the merged view across all known sources
    (group files take precedence; legacy fills the gaps).
    """
    merged = {"stages": {}}
    legacy = data_dir / "state.json"
    if legacy.exists():
        merged["stages"].update(json.loads(legacy.read_text())["stages"])
    for path in sorted(data_dir.glob("state.*.json")):
        merged["stages"].update(json.loads(path.read_text())["stages"])
    if group is not None:
        own = _read_group_state(data_dir, group)
        if own is not None:
            merged["stages"][group] = own
    return merged


def save_state(data_dir: Path, state: dict, *, stage: str | None = None) -> None:
    """Persist cost accounting.

    With ``stage`` the write is scoped to the stage's own cap-group file
    via atomic replace: concurrent stages in *different* groups each own
    one file and cannot lose each other's updates; stages inside the same
    group are serial by protocol (single writer per file).  The legacy
    combined ``state.json`` is never written again.
    """
    if stage is None:
        raise ValueError("save_state requires a stage (group-scoped files)")
    group = STAGE_CAP_GROUP[stage]
    path = _group_state_path(data_dir, group)
    payload = {"stages": {group: state["stages"][group]}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)


def stage_budget(data_dir: Path, stage: str, protocol: dict, override_hours):
    """Cumulative seconds spent and the remaining budget for this stage.

    Reads this stage's cap-group accounting (own file first, legacy
    combined state.json as fallback) -- never the merged cross-group view.
    """
    group = STAGE_CAP_GROUP[stage]
    group_state = _read_group_state(data_dir, group)
    if group_state is None:
        group_state = load_state(data_dir)["stages"].get(group, {})
    spent = float(group_state.get("cumulative_seconds", 0.0))
    if override_hours is not None:
        cap = float(override_hours) * 3600.0
    else:
        cap = float(protocol["wall_clock_caps_hours"][group]) * 3600.0
    return spent, max(0.0, cap - spent)


def note_stage_seconds(data_dir: Path, state: dict, stage: str, seconds: float,
                       records_delta: int) -> None:
    group = STAGE_CAP_GROUP[stage]
    entry = state["stages"].setdefault(
        group, {"cumulative_seconds": 0.0, "records": 0, "stages": []})
    entry["cumulative_seconds"] += float(seconds)
    entry["records"] += int(records_delta)
    if stage not in entry["stages"]:
        entry["stages"].append(stage)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def deterministic_singles(protocol, atmo, table):
    """Finite-source first-order spectra, computed once per (sun, mode).

    ``single_zenith`` is deterministic; caching per (sun elevation, mode)
    keeps the stage wall-clock budget for the Monte Carlo.  A fresh
    ``FiniteSolarSource`` is used per call (one instance per solar geometry).
    """
    singles = {}
    for sun_elev in protocol["scenario"]["sun_elevations_deg"]:
        for mode, radius in (("disk", 16.0 / 60.0), ("point", 0.0)):
            started = time.perf_counter()
            value, _ = single_zenith(
                atmo, FiniteSolarSource(atmo, table, radius_deg=radius),
                90.0 - sun_elev)
            singles[(sun_elev, mode)] = value
            print(json.dumps(dict(single=sun_elev, mode=mode,
                                  seconds=time.perf_counter() - started)),
                  flush=True)
    return singles


def _common_meta(protocol_path: Path, table_sha: str, stage: str, sun_index,
                 budget_index, photons, mode) -> dict:
    return dict(stage=stage, schema="deep-twilight-reference/1",
                protocol_sha256=sha256_file(protocol_path),
                code_sha256=code_hashes(), table_sha256=table_sha,
                hardware=hardware_info(), sun_index=sun_index,
                budget_index=budget_index, photons_per_wavelength=photons,
                mode=mode, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def run_formal_stage(protocol, atmo, table, stage, data_dir, state, *,
                     protocol_path, table_sha, max_stage_hours=None,
                     batch_filter=None, new_attempt=False) -> int:
    """Run the pre-registered direct / source-pairs batches that are missing.

    Returns the number of records computed.  Honours the shared wall-clock
    cap (measured as true stage wall-clock, including setup): when the
    remaining budget cannot fit another record the stage is left
    ``incomplete-budget`` and the summarizer refuses full-budget conclusions
    from it.  Reruns resume the newest attempt and only compute missing
    seeds; cumulative cost accumulates in ``state.json``.
    """
    direct = protocol["direct"]
    budgets = direct["budgets_photons"]
    repeats = int(direct["repeats"])
    suns = protocol["scenario"]["sun_elevations_deg"]
    # the stage clock starts BEFORE any setup so the cap truly bounds the
    # whole wall-clock including table load and deterministic singles; the
    # budget is checked before that work, too
    started = time.perf_counter()
    spent, budget_left = stage_budget(data_dir, stage, protocol, max_stage_hours)
    singles = None
    if budget_left > 0.0:
        singles = deterministic_singles(protocol, atmo, table)
    billed = started  # seconds of this invocation already noted into state
    computed = 0
    stopped = None
    # execution-time provenance for every record written by THIS invocation;
    # never inherited from a stored batch meta (resume must not relabel new
    # computation with old hashes)
    provenance = dict(protocol_sha256=sha256_file(protocol_path),
                      code_sha256=code_hashes(), table_sha256=table_sha)

    def remaining():
        # budget_left already excludes prior invocations' cumulative cost
        return budget_left - (time.perf_counter() - started)

    def bill(records_done):
        nonlocal billed, billed_records
        now = time.perf_counter()
        note_stage_seconds(data_dir, state, stage, now - billed,
                           records_done - billed_records)
        billed = now
        billed_records = records_done
        save_state(data_dir, state, stage=stage)

    billed_records = 0
    bill(0)  # stage clock includes setup; singles run only within budget

    for sun_index, sun_elev in enumerate(suns):
        budget_indices = [len(budgets) - 1] if stage == "source-pairs" \
            else list(range(len(budgets)))
        for budget_index in budget_indices:
            if stage == "source-pairs":
                batch_id = f"s{sun_index}b{budget_index}-point"
                mode = "point"
            else:
                batch_id = f"s{sun_index}b{budget_index}"
                mode = "disk"
            if batch_filter and batch_filter not in batch_id:
                continue
            seeds = [seed_for(sun_index, budget_index, r) for r in range(repeats)]
            meta_base = _common_meta(protocol_path, table_sha, stage, sun_index,
                                     budget_index, budgets[budget_index], mode)
            batch = Batch(data_dir / "raw" / stage, batch_id, seeds, meta_base)
            missing = [s for s in seeds if s not in batch.done]
            if missing and singles is None:
                stopped = "budget-exhausted"
                break
            if not missing:
                batch.close("complete")
                continue
            if new_attempt and batch.meta["attempts"][-1]["status"] == "open":
                batch.meta["attempts"][-1]["status"] = "superseded"
                batch.attempt, batch.path, batch.done = batch._open_attempt()
                missing = [s for s in seeds if s not in batch.done]
            for seed in missing:
                if remaining() <= 0.0:
                    stopped = "budget-exhausted"
                    break
                record = run_record(atmo, table, sun_elev, seed,
                                    budgets[budget_index],
                                    deterministic_single=singles[(sun_elev, mode)],
                                    mode=mode)
                record["batch_id"] = batch_id
                record["attempt"] = batch.attempt
                record["protocol_sha256"] = provenance["protocol_sha256"]
                record["code_sha256"] = provenance["code_sha256"]
                record["table_sha256"] = provenance["table_sha256"]
                batch.record_seed(seed, record)
                computed += 1
                bill(computed)
                print(json.dumps(dict(
                    stage=stage, batch_id=batch_id, seed=seed,
                    seconds=record["seconds"],
                    cumulative_stage_seconds=spent_total(data_dir, stage))),
                  flush=True)
            still_missing = [s for s in seeds if s not in batch.done]
            batch.close("complete" if not still_missing else "open")
            if stopped:
                break
        if stopped:
            break
    bill(computed)
    print(json.dumps(dict(
        stage_status="complete" if not stopped else f"incomplete-{stopped}",
        stage=stage, computed=computed,
        stage_seconds_here=time.perf_counter() - started,
        cumulative_seconds=spent_total(data_dir, stage))), flush=True)
    return computed


def spent_total(data_dir: Path, stage: str) -> float:
    group = STAGE_CAP_GROUP[stage]
    own = _read_group_state(data_dir, group)
    if own is not None:
        return float(own.get("cumulative_seconds", 0.0))
    return float(load_state(data_dir)["stages"].get(group, {})
                 .get("cumulative_seconds", 0.0))


def stage_smoke(protocol, atmo, table, out: Path) -> dict:
    """End-to-end exercise on the real grid with a tiny budget (scratch)."""
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    singles = deterministic_singles(protocol, atmo, table)
    rows = []
    for sun_index, sun_elev in enumerate(protocol["scenario"]["sun_elevations_deg"]):
        for repeat in (0, 1):
            seed = 1000 + sun_index * 10 + repeat  # scratch seeds, not formal
            record = run_record(atmo, table, sun_elev, seed, 2000,
                                deterministic_single=singles[(sun_elev, "disk")],
                                keep_per_path=True)
            rows.append(dict(seed=seed, sun=sun_elev,
                             seconds=record["seconds"],
                             multiple_550=record["fields"]["multiple"][34],
                             split_total_550=record["split_total"][34],
                             max_fraction_550=record["diagnostics"]["anchors"]["550.0"]["max_fraction"]))
    summary = dict(stage="smoke", seconds=time.perf_counter() - started,
                   rows=rows, status="scratch-only")
    (out / "smoke-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def stage_cost_probe(protocol, atmo, table, out: Path) -> dict:
    """Full81 timing probe: 2 seeds x 2 depths x 100k photons (scratch).

    100k photons per wavelength keeps the per-wavelength fixed costs
    (optics tables, majorants, phase sampling tables) a small share of the
    measurement, so the extrapolation to the formal batches is meaningful.
    """
    out.mkdir(parents=True, exist_ok=True)
    singles = deterministic_singles(protocol, atmo, table)
    photons = 100_000
    rows = []
    for sun_index, sun_elev in enumerate(protocol["scenario"]["sun_elevations_deg"]):
        for seed in (901, 902):
            record = run_record(atmo, table, sun_elev, seed, photons,
                                deterministic_single=singles[(sun_elev, "disk")])
            rows.append(dict(sun=sun_elev, seed=seed, photons=photons,
                             seconds=record["seconds"]))
    # linear scheduling extrapolation to the formal batches (not a promise)
    n_wl = len(atmo.wl)
    per_wl_path = sum(r["seconds"] for r in rows) / (len(rows) * photons * n_wl)
    direct_paths = n_wl * 16 * sum(protocol["direct"]["budgets_photons"]) * 2
    pair_paths = n_wl * 16 * max(protocol["direct"]["budgets_photons"]) * 2
    summary = dict(stage="cost-probe", rows=rows,
                   seconds_per_wavelength_path=per_wl_path,
                   extrapolated_direct_hours=per_wl_path * direct_paths / 3600.0,
                   extrapolated_source_pairs_hours=per_wl_path * pair_paths / 3600.0,
                   status="scratch-only-scheduling-estimate")
    (out / "cost-probe-summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


def candidate_table_load(protocol: dict, input_root: Path, atmo):
    """Load the candidate experiment's frozen 3-wavelength table.

    ``input_root`` is the *protocol* directory (read-only frozen inputs),
    never the run output root.
    """
    from tools.twilight_second_order import estimate  # noqa: F401 (wiring)

    spec = protocol["candidate"]["table"]
    path = input_root / spec["path"]
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(f"candidate table hash mismatch: {path} is {digest}, "
                         f"protocol freezes {spec['sha256']}")
    return PointSolarTable.load(path, atmo, expected_label=spec["label"])


def stage_candidate(protocol, atmo, table, data_dir, state, *, protocol_path,
                    input_root, max_stage_hours=None) -> int:
    """Candidate estimator experiment (plan Step 3).

    For every (depth, tier, seed): candidate L2 at the three anchor
    wavelengths plus an MC rest run at the tier budget on the same frozen
    3-wavelength table, with per-path decomposition from the diagnostics
    sink so the comparison is per-path subtraction, never two independent
    means.  Records go to ``raw/candidate/``; the shared 2 h cap applies.
    """
    from tools.twilight_second_order import estimate, sun_hat_for

    cand = protocol["candidate"]
    anchors = [int(np.argmin(np.abs(WL_COARSE - a)))
               for a in cand["wavelengths_nm"]]
    wl3 = WL_COARSE[anchors]
    atmo3 = atmo._restrict(anchors)
    table3 = candidate_table_load(protocol, input_root, atmo3)
    singles3 = {}
    for sun_elev in protocol["scenario"]["sun_elevations_deg"]:
        started = time.perf_counter()
        value, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                 90.0 - sun_elev)
        singles3[sun_elev] = value
        print(json.dumps(dict(candidate_single=sun_elev,
                              seconds=time.perf_counter() - started)), flush=True)
    spent, budget_left = stage_budget(data_dir, "candidate", protocol, max_stage_hours)
    started = time.perf_counter()
    billed = started
    billed_records = 0
    computed = 0
    stopped = None

    def remaining():
        # budget_left already excludes prior invocations' cumulative cost
        return budget_left - (time.perf_counter() - started)

    def bill(records_done):
        nonlocal billed, billed_records
        now = time.perf_counter()
        note_stage_seconds(data_dir, state, "candidate",
                           now - billed, records_done - billed_records)
        billed = now
        billed_records = records_done
        save_state(data_dir, state, stage="candidate")

    bill(0)
    suns = protocol["scenario"]["sun_elevations_deg"]
    for sun_index, sun_elev in enumerate(suns):
        for tier_index, tier in enumerate(cand["tiers"]):
            batch_id = f"s{sun_index}t{tier_index}"
            seeds = [80000 + 10000 * sun_index + 1000 * tier_index + r
                     for r in range(int(cand["repeats"]))]
            meta_base = _common_meta(protocol_path, cand["table"]["sha256"],
                                     "candidate", sun_index, tier_index,
                                     tier["rest_photons"], "disk")
            meta_base["candidate_samples"] = tier["candidate_samples"]
            meta_base["rest_photons"] = tier["rest_photons"]
            batch = Batch(data_dir / "raw" / "candidate", batch_id, seeds,
                          meta_base)
            missing = [s for s in seeds if s not in batch.done]
            if not missing:
                batch.close("complete")
                continue
            for seed in missing:
                if remaining() <= 0.0:
                    stopped = "budget-exhausted"
                    break
                # ONE rest MC per (seed, tier) covering all three anchor
                # wavelengths; every lambda's candidate estimate then
                # reuses the same decomposition -- running the MC once per
                # lambda tripled the cost and broke the per-repeat cost unit
                source_mc = FiniteSolarSource(atmo3, table3)
                t_mc = time.perf_counter()
                with mc_solar_source(source_mc):
                    mc_res = mc.mc_radiance_split(
                        atmo3, 90.0 - sun_elev, [0.0], [0.0],
                        n_photons=int(tier["rest_photons"]), seed=seed,
                        max_scatter=50, diagnostics=True)
                mc_seconds = time.perf_counter() - t_mc
                per_lambda = {}
                for pos, iw in enumerate(anchors):
                    source = FiniteSolarSource(atmo3, table3)
                    cand_res = estimate(atmo3, pos, source, sun_elev,
                                        int(tier["candidate_samples"]),
                                        seed=seed)
                    d = mc_res["diagnostics"][pos]["per_view"][0]
                    n_paths = int(tier["rest_photons"])
                    c2nt_mean = d["s1"][1] / n_paths
                    rest_mean = float(mc_res["multiple"][0, pos]) - c2nt_mean
                    # per-path second moment of rest = mult - c2nt; an
                    # all-zero rest batch has s2 = None ("no tail evidence")
                    # and a genuine zero second moment, not a missing value
                    rest_second = 0.0 if d["s2"] is None else \
                        float((np.array(d["s2"]) / n_paths)[0][0]
                              + (np.array(d["s2"]) / n_paths)[1][1]
                              - 2.0 * (np.array(d["s2"]) / n_paths)[0][1])
                    deterministic = float(singles3[sun_elev][pos])
                    per_lambda[str(wl3[pos])] = dict(
                        L2=cand_res["L2"], L2_second_moment=cand_res["second_moment"],
                        L2_max_abs=cand_res["max_abs"],
                        candidate_seconds=cand_res["seconds"],
                        candidate_samples=int(cand_res["n_samples"]),
                        c2nt_mean=c2nt_mean, rest_mean=rest_mean,
                        rest_second_moment=rest_second,
                        mc_multiple=float(mc_res["multiple"][0, pos]),
                        deterministic_single=deterministic,
                        total_cand=deterministic + cand_res["L2"] + rest_mean,
                        total_analog=deterministic
                        + float(mc_res["multiple"][0, pos]),
                        max_fraction=mc_res["diagnostics"][pos]["per_view"][0]["max_fraction"],
                        top001_fraction=mc_res["diagnostics"][pos]["per_view"][0]["top001_fraction"],
                        effective_contributors=mc_res["diagnostics"][pos]["per_view"][0]["effective_contributors"])
                record = dict(seed=int(seed), sun_elevation_deg=sun_elev,
                              tier_index=tier_index, mode="disk",
                              candidate_samples=tier["candidate_samples"],
                              rest_photons=tier["rest_photons"],
                              per_lambda=per_lambda,
                              # one rest MC per repeat is the shared MC cost
                              # of all three anchors (never per-lambda)
                              mc_seconds=mc_seconds,
                              seconds=sum(v["candidate_seconds"]
                                          for v in per_lambda.values())
                              + mc_seconds,
                              table_sha256=cand["table"]["sha256"],
                              protocol_sha256=sha256_file(protocol_path),
                              code_sha256=code_hashes(),
                              batch_id=batch_id, attempt=batch.attempt)
                batch.record_seed(seed, record)
                computed += 1
                bill(computed)
                print(json.dumps(dict(stage="candidate", batch_id=batch_id,
                                      seed=seed,
                                      seconds=record["seconds"],
                                      cumulative_stage_seconds=spent_total(data_dir, "candidate"))),
                      flush=True)
            still_missing = [s for s in seeds if s not in batch.done]
            batch.close("complete" if not still_missing else "open")
            if stopped:
                break
        if stopped:
            break
    bill(computed)
    print(json.dumps(dict(
        stage_status="complete" if not stopped else f"incomplete-{stopped}",
        stage="candidate", computed=computed,
        stage_seconds_here=time.perf_counter() - started,
        cumulative_seconds=spent_total(data_dir, "candidate"))), flush=True)
    return computed


#: sensitivity items (plan Step 4): (key, description); item "baseline" must
#: come first so every other item is a paired same-seed comparison against it
SENSITIVITY_MC_ITEMS = (
    ("baseline", "default configuration"),
    ("disk4x16", "solar disk quadrature 2x8 -> 4x16 nodes"),
    ("max_scatter100", "scattering order cap 50 -> 100"),
    ("rr_conservative", "RR weight 0.05 -> 0.02, survival 0.75 -> 0.9"),
    ("fine_optical", "MC fine optical grid 700 -> 1400 levels"),
    ("fine_phase", "aerosol phase sampling table 2049 -> 4097 nodes"),
)

#: deterministic-single refinement items (no Monte Carlo involved)
SENSITIVITY_SINGLE_ITEMS = (
    ("ss_gauss16", "single_zenith n_gauss 8 -> 16"),
    ("ss_fine_density", "single_zenith density step 0.025 -> 0.0125 km"),
)

#: sensitivity seeds (disjoint from direct 40000+, candidate 80000+, probes)
SENSITIVITY_SEED_BASE = 110000


def stage_sensitivity(protocol, atmo, table, data_dir, state, *, protocol_path,
                      input_root, max_stage_hours=None) -> int:
    """Discretisation/refinement sensitivity (plan Step 4).

    Three anchor wavelengths, 16 seeds per depth at 100k photons, on the
    frozen 3-wavelength planning table; every MC item is compared against
    the same-seed baseline run, and the single-scattering refinements are
    deterministic and compared directly.  Curved-diffuse transport is NOT
    in this stage: the experiment backend exists but its inclusion is a
    documented unknown, not a run under this budget.
    """
    from tools.twilight_second_order import estimate  # noqa: F401 (parity)

    sens = protocol.get("sensitivity", {})
    anchors = [int(np.argmin(np.abs(WL_COARSE - a)))
               for a in protocol["scenario"]["anchor_wavelengths_nm"]]
    wl3 = WL_COARSE[anchors]
    atmo3 = atmo._restrict(anchors)
    table3 = candidate_table_load(protocol, input_root, atmo3)
    singles3 = {}
    for sun_elev in protocol["scenario"]["sun_elevations_deg"]:
        singles3[sun_elev] = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                           90.0 - sun_elev)[0]
    repeats = int(sens.get("repeats", 16))
    photons = int(sens.get("photons_per_wavelength", 100_000))
    suns = protocol["scenario"]["sun_elevations_deg"]
    spent, budget_left = stage_budget(data_dir, "sensitivity", protocol, max_stage_hours)
    started = time.perf_counter()
    billed = started
    billed_records = 0
    computed = 0
    stopped = None

    def remaining():
        # budget_left already excludes prior invocations' cumulative cost
        return budget_left - (time.perf_counter() - started)

    def bill(records_done):
        nonlocal billed, billed_records
        now = time.perf_counter()
        note_stage_seconds(data_dir, state, "sensitivity",
                           now - billed, records_done - billed_records)
        billed = now
        billed_records = records_done
        save_state(data_dir, state, stage="sensitivity")

    bill(0)

    def run_mc(item, sun_elev, seed):
        """One MC record under a sensitivity configuration.

        Every item runs inside the finite-source clone (``mc_solar_source``)
        so the solar NEE uses the SAME source contract as the formal stages;
        the disk-node item changes only the source angular quadrature, so
        the traced paths -- and the fingerprint -- stay identical to the
        baseline while the NEE values change.  Module-constant patches are
        applied *before* the clone is built so the cloned namespace sees
        them, and are restored afterwards.
        """
        source = FiniteSolarSource(
            atmo3, table3,
            n_radial=4 if item == "disk4x16" else 2,
            n_azimuth=16 if item == "disk4x16" else 8,
            radius_deg=16.0 / 60.0)
        patches = {}
        if item == "rr_conservative":
            patches = dict(_RR_WEIGHT=0.02, _RR_SURVIVE=0.9)
        elif item == "fine_optical":
            patches = dict(_N_FINE=1400)
        elif item == "fine_phase":
            patches = dict(_N_PHASE=4097)
        max_scatter = 100 if item == "max_scatter100" else 50
        saved = {k: getattr(mc, k) for k in patches}
        started = time.perf_counter()
        try:
            for k, v in patches.items():
                setattr(mc, k, v)
            with mc_solar_source(source):
                result = mc.mc_radiance_split(
                    atmo3, 90.0 - sun_elev, [0.0], [0.0],
                    n_photons=photons, seed=seed, max_scatter=max_scatter,
                    diagnostics=False)
        finally:
            for k, v in saved.items():
                setattr(mc, k, v)
        seconds = time.perf_counter() - started
        fingerprint = source.fingerprint.hexdigest()
        per_lambda = {}
        for pos, iw in enumerate(anchors):
            deterministic = float(singles3[sun_elev][pos])
            per_lambda[str(wl3[pos])] = dict(
                deterministic_single=deterministic,
                multiple=float(result["multiple"][0, pos]),
                total=deterministic + float(result["multiple"][0, pos]))
        return dict(per_lambda=per_lambda, seconds=seconds,
                    fingerprint=fingerprint)

    for sun_index, sun_elev in enumerate(suns):
        batch_id = f"s{sun_index}mc"
        seeds = [SENSITIVITY_SEED_BASE + 1000 * sun_index + r
                 for r in range(repeats)]
        meta_base = _common_meta(protocol_path, table3_hash(protocol, data_dir),
                                 "sensitivity", sun_index, 0, photons, "disk")
        batch = Batch(data_dir / "raw" / "sensitivity", batch_id, seeds,
                      meta_base)
        missing = [s for s in seeds if s not in batch.done]
        if missing:
            for seed in missing:
                if remaining() <= 0.0:
                    stopped = "budget-exhausted"
                    break
                items = {}
                for item, _desc in SENSITIVITY_MC_ITEMS:
                    items[item] = run_mc(item, sun_elev, seed)
                record = dict(seed=int(seed), sun_elevation_deg=sun_elev,
                              photons_per_wavelength=photons,
                              items=items, batch_id=batch_id,
                              attempt=batch.attempt,
                              code_sha256=code_hashes(),
                              table_sha256=meta_base["table_sha256"])
                batch.record_seed(seed, record)
                computed += 1
                bill(computed)
                print(json.dumps(dict(stage="sensitivity", batch_id=batch_id,
                                      seed=seed)), flush=True)
        still_missing = [s for s in seeds if s not in batch.done]
        batch.close("complete" if not still_missing else "open")
        if stopped:
            break
    bill(computed)
    # deterministic single-scattering refinements (no MC, no seeds) --
    # inside the billed window and skipped when the budget is gone
    single_ref = {}
    singles_path = data_dir / "raw" / "sensitivity" / "singles-refinement.json"
    existing_singles = None
    if singles_path.exists():
        try:
            payload = json.loads(singles_path.read_text())
        except json.JSONDecodeError:
            payload = None
        # a completed prior run's evidence: non-empty depths AND pinned to
        # the current protocol (anything else is not reusable evidence)
        if isinstance(payload, dict) and payload.get("by_sun") \
                and payload.get("protocol_sha256") \
                == sha256_file(protocol_path):
            existing_singles = payload
    singles_sourced = "missing"
    if budget_left > 0.0 and table3 is not None:
        single_ref = _sensitivity_singles(protocol, atmo3, table3, suns)
        # only a fresh, complete computation may replace the evidence
        # file -- a budget-exhausted recovery must never overwrite
        # finished evidence with an empty placeholder
        singles_path.write_text(json.dumps(dict(
            schema="deep-twilight-reference/sensitivity-singles/1",
            protocol_sha256=sha256_file(protocol_path),
            items={k: d for k, d in SENSITIVITY_SINGLE_ITEMS}, by_sun=single_ref),
            indent=1) + "\n")
        singles_sourced = "computed"
    elif existing_singles is not None:
        singles_sourced = "reused-existing"
    bill(computed)  # final bill includes the deterministic refinements
    singles_ok = bool(single_ref) or existing_singles is not None
    if not stopped and not singles_ok:
        stopped = "no-singles-evidence"
    print(json.dumps(dict(
        stage_status="complete" if not stopped else f"incomplete-{stopped}",
        stage="sensitivity", computed=computed,
        singles=singles_sourced,
        stage_seconds_here=time.perf_counter() - started,
        cumulative_seconds=spent_total(data_dir, "sensitivity"))), flush=True)
    return computed


def _sensitivity_singles(protocol, atmo3, table3, suns):
    single_ref = {}
    for sun_elev in suns:
        base, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                90.0 - sun_elev)
        g16, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                               90.0 - sun_elev, n_gauss=16)
        fine, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                90.0 - sun_elev, segment_km=0.25)
        step, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                90.0 - sun_elev, density_step_km=0.0125)
        single_ref[str(sun_elev)] = dict(
            baseline=base.tolist(), ss_gauss16=g16.tolist(),
            ss_segment_km_0_25=fine.tolist(),
            ss_density_step_0_0125=step.tolist())
    return single_ref


def table3_hash(protocol: dict, data_dir: Path) -> str:
    return protocol["candidate"]["table"]["sha256"]


STAGES = ("smoke", "cost-probe", "direct", "source-pairs", "candidate",
          "sensitivity")
DEFAULT_PROTOCOL = ROOT / "validation/data/deep_twilight_reference/protocol.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--out", type=Path, default=None,
                        help="output root; defaults to the protocol directory "
                             "(formal stages), required for scratch stages")
    parser.add_argument("--max-stage-hours", type=float, default=None,
                        help="override the protocol wall-clock cap")
    parser.add_argument("--batch", default=None,
                        help="only run batches whose id contains this string")
    parser.add_argument("--new-attempt", action="store_true",
                        help="supersede the open attempt instead of continuing it")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    atmo = build_scenario(protocol)
    table = load_source_table(protocol, atmo)

    # the protocol directory is the READ-ONLY input root (frozen tables,
    # frozen protocol); --out is the writable run root and defaults to the
    # input root for the default archive layout.  A fresh run root is
    # created and receives a validated protocol snapshot so the summarizer
    # can work offline against the run alone.
    input_root = args.protocol.resolve().parent
    run_root = args.out.resolve() if args.out is not None else input_root
    if args.stage in SCRATCH_STAGES:
        if args.out is None:
            parser.error(f"--stage {args.stage} requires --out (scratch only)")
        if args.stage == "smoke":
            summary = stage_smoke(protocol, atmo, table, args.out)
            print(json.dumps(dict(stage=args.stage, status=summary["status"],
                                  seconds=summary["seconds"])))
            return 0
        summary = stage_cost_probe(protocol, atmo, table, args.out)
        print(json.dumps(dict(stage=args.stage, status=summary["status"],
                              seconds=sum(r["seconds"] for r in summary["rows"]),
                              seconds_per_wavelength_path=(
                                  summary["seconds_per_wavelength_path"]))))
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    if run_root != input_root:
        snapshot = run_root / "protocol.snapshot.json"
        current = json.loads(args.protocol.read_text())
        if snapshot.exists():
            frozen = json.loads(snapshot.read_text())
            relevant = {k: current[k] for k in
                        ("scenario", "source", "direct", "precision",
                         "luminance") if k in current}
            frozen_relevant = {k: frozen[k] for k in
                               ("scenario", "source", "direct", "precision",
                                "luminance") if k in frozen}
            if relevant != frozen_relevant:
                raise SystemExit(
                    "run-root protocol snapshot differs from the input "
                    "protocol in contract-relevant sections; use a fresh "
                    "run root or consciously delete the snapshot")
        else:
            # verbatim byte copy: the snapshot must carry the SAME sha as
            # the executed protocol, or the summarizer's generation gate
            # would see a fabricated third generation
            import shutil
            shutil.copyfile(args.protocol, snapshot)
    state = load_state(run_root)
    if args.stage == "candidate":
        stage_candidate(protocol, atmo, table, run_root, state,
                        protocol_path=args.protocol.resolve(),
                        input_root=input_root,
                        max_stage_hours=args.max_stage_hours)
        return 0
    if args.stage == "sensitivity":
        stage_sensitivity(protocol, atmo, table, run_root, state,
                          protocol_path=args.protocol.resolve(),
                          input_root=input_root,
                          max_stage_hours=args.max_stage_hours)
        return 0

    run_formal_stage(protocol, atmo, table, args.stage, run_root, state,
                     protocol_path=args.protocol.resolve(),
                     table_sha=protocol["source"]["full81_table"]["sha256"],
                     max_stage_hours=args.max_stage_hours,
                     batch_filter=args.batch, new_attempt=args.new_attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
