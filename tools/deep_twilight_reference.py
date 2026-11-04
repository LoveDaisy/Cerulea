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
    """Peak resident set size of this process so far (platform-dependent unit)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw * 1024 if platform.system() == "Darwin" else raw * 1024


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
    """The frozen benchmark atmosphere, with contract assertions."""
    scenario = protocol["scenario"]
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
    if float(scenario["sun_azimuth_deg"]) != 0.0:
        raise ValueError("the reference fixes sun azimuth 0")
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
class Batch:
    """One (stage, batch-id) group of records with resumable attempts.

    The newest attempt file is continued in place; seeds already present in
    it are never recomputed.  If its last line is corrupt (a crash during
    write), the attempt is marked invalid and ``next_attempt`` starts a new
    file -- a batch then has exactly one *valid* complete attempt.
    """

    def __init__(self, root: Path, batch_id: str, expected_seeds, meta_base: dict):
        self.dir = root / batch_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.batch_id = batch_id
        self.expected_seeds = [int(s) for s in expected_seeds]
        self.meta_path = self.dir / "meta.json"
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
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
def load_state(data_dir: Path) -> dict:
    path = data_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"stages": {}}


def save_state(data_dir: Path, state: dict) -> None:
    (data_dir / "state.json").write_text(json.dumps(state, indent=1) + "\n")


def stage_budget(state: dict, stage: str, protocol: dict, override_hours):
    """Cumulative seconds spent and the remaining budget for this stage."""
    group = STAGE_CAP_GROUP[stage]
    spent = float(state["stages"].get(group, {}).get("cumulative_seconds", 0.0))
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
    singles = deterministic_singles(protocol, atmo, table)
    spent, _cap = stage_budget(state, stage, protocol, max_stage_hours)
    started = time.perf_counter()
    billed = started  # seconds of this invocation already noted into state
    computed = 0
    stopped = None

    def remaining():
        return _cap - (spent + (time.perf_counter() - started))

    def bill(records_done):
        nonlocal billed, billed_records
        now = time.perf_counter()
        note_stage_seconds(data_dir, state, stage, now - billed,
                           records_done - billed_records)
        billed = now
        billed_records = records_done
        save_state(data_dir, state)

    billed_records = 0
    bill(0)  # setup (table load, deterministic singles) counts against the cap

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
                record["protocol_sha256"] = batch.meta["protocol_sha256"]
                record["code_sha256"] = batch.meta["code_sha256"]
                record["table_sha256"] = batch.meta["table_sha256"]
                batch.record_seed(seed, record)
                computed += 1
                bill(computed)
                print(json.dumps(dict(
                    stage=stage, batch_id=batch_id, seed=seed,
                    seconds=record["seconds"],
                    cumulative_stage_seconds=spent_total(state, stage))),
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
        cumulative_seconds=spent_total(state, stage))), flush=True)
    return computed


def spent_total(state: dict, stage: str) -> float:
    group = STAGE_CAP_GROUP[stage]
    return float(state["stages"].get(group, {}).get("cumulative_seconds", 0.0))


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


def candidate_table_load(protocol: dict, data_dir: Path, atmo):
    """Load the candidate experiment's frozen 3-wavelength table."""
    from tools.twilight_second_order import estimate  # noqa: F401 (wiring)

    spec = protocol["candidate"]["table"]
    path = data_dir / spec["path"]
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(f"candidate table hash mismatch: {path} is {digest}, "
                         f"protocol freezes {spec['sha256']}")
    return PointSolarTable.load(path, atmo, expected_label=spec["label"])


def stage_candidate(protocol, atmo, table, data_dir, state, *, protocol_path,
                    max_stage_hours=None) -> int:
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
    table3 = candidate_table_load(protocol, data_dir, atmo3)
    singles3 = {}
    for sun_elev in protocol["scenario"]["sun_elevations_deg"]:
        started = time.perf_counter()
        value, _ = single_zenith(atmo3, FiniteSolarSource(atmo3, table3),
                                 90.0 - sun_elev)
        singles3[sun_elev] = value
        print(json.dumps(dict(candidate_single=sun_elev,
                              seconds=time.perf_counter() - started)), flush=True)
    spent, _cap = stage_budget(state, "candidate", protocol, max_stage_hours)
    started = time.perf_counter()
    billed = started
    billed_records = 0
    computed = 0
    stopped = None

    def remaining():
        return _cap - (spent + (time.perf_counter() - started))

    def bill(records_done):
        nonlocal billed, billed_records
        now = time.perf_counter()
        note_stage_seconds(data_dir, state, "candidate",
                           now - billed, records_done - billed_records)
        billed = now
        billed_records = records_done
        save_state(data_dir, state)

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
                per_lambda = {}
                for pos, iw in enumerate(anchors):
                    source = FiniteSolarSource(atmo3, table3)
                    cand_res = estimate(atmo3, pos, source, sun_elev,
                                        int(tier["candidate_samples"]),
                                        seed=seed)
                    # MC rest with per-path decomposition on the same seed
                    source = FiniteSolarSource(atmo3, table3)
                    t_mc = time.perf_counter()
                    with mc_solar_source(source):
                        mc_res = mc.mc_radiance_split(
                            atmo3, 90.0 - sun_elev, [0.0], [0.0],
                            n_photons=int(tier["rest_photons"]), seed=seed,
                            max_scatter=50, diagnostics=True)
                    mc_seconds = time.perf_counter() - t_mc
                    d = mc_res["diagnostics"][pos]["per_view"][0]
                    n_paths = int(tier["rest_photons"])
                    c2nt_mean = d["s1"][1] / n_paths
                    rest_mean = float(mc_res["multiple"][0, pos]) - c2nt_mean
                    # per-path second moment of rest = mult - c2nt; an
                    # all-zero rest batch has s2 = None ("no tail evidence")
                    rest_second = None
                    if d["s2"] is not None:
                        s2 = np.array(d["s2"]) / n_paths
                        rest_second = float(s2[0][0] + s2[1][1] - 2.0 * s2[0][1])
                    deterministic = float(singles3[sun_elev][pos])
                    per_lambda[str(wl3[pos])] = dict(
                        L2=cand_res["L2"], L2_second_moment=cand_res["second_moment"],
                        L2_max_abs=cand_res["max_abs"],
                        candidate_seconds=cand_res["seconds"],
                        candidate_samples=int(cand_res["n_samples"]),
                        c2nt_mean=c2nt_mean, rest_mean=rest_mean,
                        rest_second_moment=rest_second,
                        mc_multiple=float(mc_res["multiple"][0, pos]),
                        mc_seconds=mc_seconds,
                        deterministic_single=deterministic,
                        total_cand=deterministic + cand_res["L2"] + rest_mean,
                        total_analog=deterministic
                        + float(mc_res["multiple"][0, pos]),
                        max_fraction=mc_res["diagnostics"][pos]["per_view"][0]["max_fraction"],
                        top001_fraction=mc_res["diagnostics"][pos]["per_view"][0]["top001_fraction"],
                        effective_contributors=mc_res["diagnostics"][pos]["per_view"][0]["effective_contributors"])
                record = dict(seed=int(seed), sun_elevation_deg=sun_elev,
                              tier_index=tier_index,
                              candidate_samples=tier["candidate_samples"],
                              rest_photons=tier["rest_photons"],
                              per_lambda=per_lambda,
                              seconds=sum(v["candidate_seconds"] + v["mc_seconds"]
                                          for v in per_lambda.values()),
                              table_sha256=cand["table"]["sha256"],
                              code_sha256=code_hashes(),
                              batch_id=batch_id, attempt=batch.attempt)
                batch.record_seed(seed, record)
                computed += 1
                bill(computed)
                print(json.dumps(dict(stage="candidate", batch_id=batch_id,
                                      seed=seed,
                                      seconds=record["seconds"],
                                      cumulative_stage_seconds=spent_total(state, "candidate"))),
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
        cumulative_seconds=spent_total(state, "candidate"))), flush=True)
    return computed


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

    if args.stage in SCRATCH_STAGES:
        if args.out is None:
            parser.error(f"--stage {args.stage} requires --out (scratch only)")
        if args.stage == "smoke":
            summary = stage_smoke(protocol, atmo, table, args.out)
        else:
            summary = stage_cost_probe(protocol, atmo, table, args.out)
        print(json.dumps(dict(stage=args.stage, status=summary["status"],
                              seconds=summary["seconds"])))
        return 0

    if args.stage == "candidate":
        data_dir = args.out if args.out is not None else args.protocol.parent
        state = load_state(data_dir)
        stage_candidate(protocol, atmo, table, data_dir, state,
                        protocol_path=args.protocol.resolve(),
                        max_stage_hours=args.max_stage_hours)
        return 0
    if args.stage == "sensitivity":
        raise SystemExit(
            "stage 'sensitivity' is delivered by plan Step 4; not implemented yet")

    data_dir = args.out if args.out is not None else args.protocol.parent
    state = load_state(data_dir)
    run_formal_stage(protocol, atmo, table, args.stage, data_dir, state,
                     protocol_path=args.protocol.resolve(),
                     table_sha=protocol["source"]["full81_table"]["sha256"],
                     max_stage_hours=args.max_stage_hours,
                     batch_filter=args.batch, new_attempt=args.new_attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
