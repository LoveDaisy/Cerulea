"""Bounded fine-grid production-chain calibration (no physical implementation).

Run with ``uv run python -m tools.fine_grid_cost --help``. Large artifacts and
its cumulative, exclusive budget journal belong to one task directory. Timing
runs use spawn and unmodified production defaults; resource sampling is external.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGET_S = 14400
STATE_COUNT = 24 * 77


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_signature(a):
    a = np.asarray(a)
    return dict(dtype=a.dtype.str, shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())


def archive_signature(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: array_signature(data[name]) for name in sorted(data.files)}


def check_arrays(data):
    expected = (91 * 181, 81)
    for name in ("single", "multiple", "total"):
        a = data[name]
        if a.shape != expected or a.dtype != np.dtype("float64") or not np.isfinite(a).all():
            raise ValueError(f"invalid {name}: shape/dtype/finite")
    if not np.array_equal(data["single"] + data["multiple"], data["total"]):
        raise ValueError("component closure failed")
    for name, values in (("wl", np.linspace(380, 780, 81)),
                         ("zenith", np.linspace(0, 90, 91)),
                         ("azimuth_rel", np.linspace(0, 180, 181))):
        if not np.array_equal(data[name], values):
            raise ValueError(f"invalid {name} grid")


def request(preset, aod, du, height):
    return dict(preset=preset, aod=aod, ozone_du=du, sun_elevation_deg=height,
                n_zenith=91, n_azimuth=181, approximation="hybrid-sos", refraction=True)


def request_key(req):
    return tuple(req[k] for k in request("clear", 0, 0, 0))


def worklist():
    difficult = request("hazy", .3, 0., -6.)
    unique = [request("clear", 0., 0., 30.), request("hazy", .05, 300., 30.),
              request("clear", .05, 300., 3.), request("hazy", .3, 300., .25),
              request("clear", .3, 300., -.125), request("clear", .1, 0., -3.),
              difficult, request("hazy", .3, 300., 90.)]
    return [dict(id=f"repeat-{i:02d}", request=difficult) for i in range(8)] + [
        dict(id=f"sample-{i:02d}", request=r) for i, r in enumerate(unique)]


def identity():
    import skycolor
    stream = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(stream):
        np.show_config()
    digest = hashlib.sha256()
    for path in sorted((ROOT / "skycolor").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    if Path(skycolor.__file__).resolve().parent != ROOT / "skycolor":
        raise ValueError("editable install points outside this worktree")
    return dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True).strip(),
                production_sha256=digest.hexdigest(), uv_lock_sha256=sha(ROOT / "uv.lock"),
                results_sha256=sha(ROOT / "validation/results.json"),
                python=sys.version, numpy=np.__version__, import_path=skycolor.__file__,
                machine=platform.platform(), cpu_count=os.cpu_count(),
                hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string",
                                                   "hw.memsize"], text=True).strip(),
                blas=stream.getvalue(), thread_env={k: os.environ.get(k) for k in (
                    "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def audit(task, reference):
    task.mkdir(parents=True, exist_ok=True)
    first = load(reference / "m1-first-scene/report.json")
    ident = identity()
    for key in ("production_sha256", "uv_lock_sha256"):
        if ident[key] != first[key]:
            raise ValueError(f"baseline identity mismatch: {key}")
    scenes = load(reference / "m1-spectral/scenes.json")
    if len(scenes) != 8:
        raise ValueError("expected eight reference records")
    records = []
    for i, scene in enumerate(scenes):
        # Historical absolute worktree paths are provenance, never execution roots.
        path = (reference / "m1-first-scene/scene.npz" if i == 0 else
                reference / "m1-spectral" / Path(scene["archive"]).name)
        expected_hash = first["archive_sha256"] if i == 0 else scene["archive_sha256"]
        if sha(path) != expected_hash:
            raise ValueError(f"reference archive modified: {path}")
        with np.load(path, allow_pickle=False) as data:
            check_arrays(data)
        req = request(scene["preset"], scene["aod"], scene["ozone_du"],
                      scene["sun_elevation_deg"])
        records.append(dict(id=i, request=req, archive=str(path.resolve()),
                            archive_sha256=expected_hash, arrays=archive_signature(path),
                            effective=first["effective"] if i == 0 else scene["effective"],
                            diagnostics=first["diagnostics"] if i == 0 else scene["diagnostics"]))
    result = dict(identity=ident, references=records, worklist=worklist(), budget_s=BUDGET_S)
    target = task / "baseline.json"
    if target.exists():
        raise ValueError("baseline already frozen; do not overwrite it")
    dump(target, result)
    print(json.dumps(dict(status="BASELINE_OK", references=len(records), identity=ident)))


@contextmanager
def budget(task, label):
    """Charge interrupted sessions until recovery; never reset with a new run name."""
    with (task / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = task / "budget.json"
        ledger = load(path) if path.exists() else []
        if any(row["status"] == "RUNNING" for row in ledger):
            raise ValueError("interrupted budget entry; adjudicate elapsed time before resuming")
        used = sum(row["elapsed_s"] for row in ledger)
        if used >= BUDGET_S:
            raise ValueError("cumulative four-hour budget exhausted")
        row = dict(label=label, started_epoch=time.time(), status="RUNNING")
        ledger.append(row)
        dump(path, ledger)
        start = time.monotonic()
        try:
            yield BUDGET_S - used
            row["status"] = "DONE"
        except BaseException:
            row["status"] = "FAILED"
            raise
        finally:
            row["elapsed_s"] = time.monotonic() - start
            row["finished_epoch"] = time.time()
            dump(path, ledger)


def observe(pid):
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,%cpu="], text=True)
    processes = [line.split() for line in rows.splitlines()]
    family = {pid}
    for _ in range(len(processes)):
        children = {int(r[0]) for r in processes if int(r[1]) in family}
        if children <= family:
            break
        family |= children
    rss = sum(int(r[2]) * 1024 for r in processes if int(r[0]) in family)
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    vm = subprocess.check_output(["vm_stat"], text=True)
    return dict(epoch=time.time(), rss_tree_bytes=rss, load=os.getloadavg(),
                swap=swap, vm_stat=vm, pids=sorted(family))


class Sampler:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open("x") as stream:
            while not self.stop.is_set():
                try:
                    sample = observe(os.getpid())
                except Exception as exc:
                    sample = dict(epoch=time.time(), error=repr(exc))
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def worker(bundle, out, instrumented=False):
    from skycolor import presets
    from skycolor.constants import WL_COARSE
    from skycolor.render import solve_dome
    import tracemalloc
    out = Path(out)
    cache = {}
    records = []
    worker_start = time.perf_counter()
    for item in bundle:
        req = item["request"]
        tag = item["id"]
        dump(out / f"{tag}.started.json", dict(pid=os.getpid(), request=req, epoch=time.time()))
        if instrumented:
            tracemalloc.start()
        start = time.perf_counter()
        cpu_start = time.process_time()
        key = (req["preset"], req["aod"], req["ozone_du"])
        cold = key not in cache
        prep_start = time.perf_counter()
        if cold:
            cache[key] = getattr(presets, req["preset"])(
                WL_COARSE, aod=req["aod"], ozone_du=req["ozone_du"])
        constructor_s = time.perf_counter() - prep_start
        pres = solve_dome(cache[key], req["sun_elevation_deg"], n_zenith=91,
                          n_azimuth=181, approximation="hybrid-sos")
        solve_s = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if instrumented else None
        if instrumented:
            tracemalloc.stop()
        result = pres.result
        arrays = dict(single=result.components["single"], multiple=result.components["multiple"],
                      total=result.radiance, wl=pres.wl, zenith=pres.zenith,
                      azimuth_rel=pres.azimuth_rel,
                      ground_albedo=pres.ground_albedo,
                      ground_direct_irradiance=pres.ground_direct_irradiance,
                      sun_apparent_elevation_deg=pres.sun_apparent_elevation_deg,
                      sun_visible=pres.sun_visible)
        check_arrays(arrays)
        if result.diagnostics.get("converged") is not True:
            raise ValueError(f"SOS did not converge: {req}")
        archive = out / f"{tag}.npz"
        io_start = time.perf_counter()
        np.savez_compressed(archive, **arrays)
        compress_s = time.perf_counter() - io_start
        verify_start = time.perf_counter()
        with np.load(archive, allow_pickle=False) as data:
            check_arrays(data)
            signatures = {name: array_signature(data[name]) for name in data.files}
        archive_hash = sha(archive)
        verify_s = time.perf_counter() - verify_start
        row = dict(id=tag, pid=os.getpid(), request=req, cold_medium=cold,
                   constructor_s=constructor_s, solve_s=solve_s, compress_s=compress_s,
                   verify_s=verify_s, complete_s=time.perf_counter() - start,
                   cpu_s=time.process_time() - cpu_start,
                   worker_offset_s=time.perf_counter() - worker_start,
                   epoch=time.time(), archive=str(archive.resolve()), archive_sha256=archive_hash,
                   arrays=signatures, diagnostics=result.diagnostics, effective=result.effective,
                   options=result.options, instrumented=instrumented,
                   tracemalloc_peak_bytes=peak)
        dump(out / f"{tag}.json", row)
        records.append(row)
        print(json.dumps({k: row[k] for k in ("id", "pid", "solve_s", "complete_s")}), flush=True)
    return records


def validate_run(run):
    rows = run["records"]
    expected = {r["id"]: r["request"] for r in worklist()}
    if len(rows) != 16 or {r["id"] for r in rows} != set(expected):
        raise ValueError("missing/duplicate completion records")
    pids = {r["pid"] for r in rows}
    repeats = [r for r in rows if r["id"].startswith("repeat-")]
    if len(pids) != run["workers"] or {r["pid"] for r in repeats} != pids:
        raise ValueError("missing worker or cross-worker repeat")
    for row in rows:
        if row["request"] != expected[row["id"]] or row["instrumented"]:
            raise ValueError("changed request or instrumented throughput")
        if row["diagnostics"].get("converged") is not True:
            raise ValueError("unconverged record")
        if row["effective"]["single"] != dict(n_gauss=3, chunk=512):
            raise ValueError("changed single defaults")
    if any(r["arrays"] != repeats[0]["arrays"] for r in repeats):
        raise ValueError("cross-worker byte difference")


def projection(run):
    validate_run(run)
    rows = run["records"]
    p = run["workers"]
    # Static full-scan schedule: one complete 77-height medium per job, 24 jobs.
    # Charge 24 *serial* complete cold states, on top of all 1848 full states.
    # This intentionally overcounts cold, IO, verification and tail rather than
    # assuming an unmeasured perfectly parallel table construction.
    cold = max(r["complete_s"] for r in rows if r["cold_medium"])
    c = max(r["complete_s"] for r in rows)
    b = run["startup_s"] + 24 * cold
    tail = max(r["worker_offset_s"] for r in rows)
    empirical = STATE_COUNT * run["pool_wall_s"] / len(rows)
    conservative = b + math.ceil(STATE_COUNT / p) * c + tail
    adopted = max(empirical, conservative) / .8
    return dict(workers=p, states=STATE_COUNT, component_volumes=48,
                measured_pool_wall_s=run["pool_wall_s"], throughput_states_s=16/run["pool_wall_s"],
                C_s=c, B_s=b, O_tail_s=tail, O_compress_extra_s=0., O_verify_extra_s=0.,
                extra_zero_reason="compression/hash/readback/finite/closure included per state",
                cold_medium_builds=24, empirical_s=empirical,
                conservative_without_margin_s=conservative, adopted_s=adopted,
                adopted_hours=adopted/3600, projected_worker_hours=adopted*p/3600,
                budget_headroom_s=8*3600*.8-max(empirical, conservative),
                cost_eligible=adopted <= 8*3600)


def calibrate(task, workers, label):
    baseline = load(task / "baseline.json")
    current = identity()
    for key in ("uv_lock_sha256", "results_sha256"):
        if current[key] != baseline["identity"][key]:
            raise ValueError(f"changed identity: {key}")
    for p in workers:
        with budget(task, f"{label}-{p}") as remaining:
            out = task / f"{label}-{p}"
            out.mkdir(exist_ok=False)
            jobs = worklist()
            bundles = [jobs[i::p] for i in range(p)]
            start = time.perf_counter()
            with Sampler(out / "resources.jsonl"):
                pool = ProcessPoolExecutor(p, mp_context=multiprocessing.get_context("spawn"))
                try:
                    futures = [pool.submit(worker, bundle, str(out)) for bundle in bundles]
                    submitted_s = time.perf_counter() - start
                    rows = []
                    for future in futures:
                        rows.extend(future.result(timeout=max(1., remaining-(time.perf_counter()-start))))
                except BaseException:
                    for process in pool._processes.values():
                        process.terminate()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
            run = dict(workers=p, identity=current, records=rows, startup_s=submitted_s,
                       pool_wall_s=time.perf_counter()-start)
            validate_run(run)
            run["projection"] = projection(run)
            dump(out / "run.json", run)
            print(json.dumps(run["projection"]), flush=True)


def resource_summary(path):
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    good = [s for s in samples if "error" not in s]
    if not good:
        raise ValueError("no resource samples")
    swaps = [float(re.search(r"used = ([\d.]+)M", s["swap"])[1]) for s in good]
    pageouts = [int(re.search(r"Swapouts:\s+(\d+)", s["vm_stat"])[1]) for s in good]
    # Existing swap is not itself pressure during the measurement. An increase
    # or active swap-outs is: such a run cannot certify a usable configuration.
    pressure = max(swaps) - min(swaps) > 1. or max(pageouts) > min(pageouts)
    return dict(sample_count=len(good), errors=len(samples)-len(good),
                rss_tree_peak_bytes=max(s["rss_tree_bytes"] for s in good),
                swap_used_start_mb=swaps[0], swap_used_end_mb=swaps[-1],
                swap_growth_mb=max(swaps)-min(swaps),
                swapout_pages=max(pageouts)-min(pageouts),
                load_min=min(s["load"][0] for s in good),
                load_max=max(s["load"][0] for s in good),
                memory_eligible=not pressure and len(good) == len(samples))


def replay(task):
    baseline = load(task / "baseline.json")
    out = task / "reference-replay"
    with budget(task, "reference-replay"):
        out.mkdir(exist_ok=False)
        # Only requests not present in the fixed calibration worklist need a
        # new solve. Matching outputs from either run remain independent of the
        # old reference and are checked against every original array below.
        covered = {request_key(j["request"]) for j in worklist()}
        jobs = [dict(id=f"reference-{r['id']:02d}", request=r["request"])
                for r in baseline["references"] if request_key(r["request"]) not in covered]
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = pool.submit(worker, jobs, str(out)).result(timeout=1800)
        dump(out / "replay.json", rows)
    with budget(task, "tracemalloc-difficult"):
        out = task / "memory-probe"
        out.mkdir(exist_ok=False)
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = pool.submit(worker, [worklist()[0]], str(out), True).result(timeout=1800)
        dump(out / "memory.json", rows)


def verify(task):
    baseline = load(task / "baseline.json")
    runs = [load(path) for path in sorted(task.glob("*/run.json"))]
    candidates = [row for run in runs for row in run["records"]]
    replay_path = task / "reference-replay/replay.json"
    if replay_path.exists():
        candidates += load(replay_path)
    for row in candidates:
        if sha(row["archive"]) != row["archive_sha256"]:
            raise ValueError("changed output archive")
        if archive_signature(row["archive"]) != row["arrays"]:
            raise ValueError("changed output arrays")
    for run in runs:
        validate_run(run)
        for row in run["records"]:
            if sha(row["archive"]) != row["archive_sha256"]:
                raise ValueError("changed output archive")
            if archive_signature(row["archive"]) != row["arrays"]:
                raise ValueError("changed output arrays")
    missing = []
    for ref in baseline["references"]:
        matching = [r for r in candidates if request_key(r["request"]) == request_key(ref["request"])]
        if not matching:
            missing.append(ref["id"])
            continue
        for row in matching:
            if any(row["arrays"].get(k) != v for k, v in ref["arrays"].items()):
                raise ValueError(f"reference byte difference: {ref['id']}")
            if row["effective"] != ref["effective"] or row["diagnostics"] != ref["diagnostics"]:
                raise ValueError("reference metadata difference")
    costs = []
    for path in sorted(task.glob("*/run.json")):
        run = load(path)
        row = projection(run)
        row["run"] = path.parent.name
        row["resources"] = resource_summary(path.parent / "resources.jsonl")
        row["adoptable"] = row["cost_eligible"] and row["resources"]["memory_eligible"]
        costs.append(row)
    report = dict(status="VERIFIED" if not missing else "INCOMPLETE", missing_references=missing,
                  projections=costs, target_supported=any(r["adoptable"] for r in costs))
    dump(task / "cost-report.json", report)
    print(json.dumps(report, indent=2))
    return 2 if missing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit-baseline", "calibrate", "replay", "verify"))
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", choices=(4, 6, 8), default=[4, 6, 8])
    parser.add_argument("--label", default="parallel")
    args = parser.parse_args()
    task = args.task_dir.resolve()
    if args.action == "audit-baseline":
        if args.reference_dir is None:
            parser.error("audit-baseline requires --reference-dir")
        audit(task, args.reference_dir.resolve())
    elif args.action == "calibrate":
        calibrate(task, args.workers, args.label)
    elif args.action == "replay":
        replay(task)
    else:
        with budget(task, "verify"):
            return verify(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
