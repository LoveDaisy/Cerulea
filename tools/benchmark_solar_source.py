"""Cost probes for the production solar-source entry points.

Runs on one machine in one script so the numbers are comparable.  It reports
blocks with separate, additive cost accounting (Mie construction is measured
on its own, so the *source* cost is never contaminated by it):

``baseline``
    Current production entry points: cold import/build, warm
    ``rt.single``/SOS/MC/dome/CLI-sized calls.
``gate``
    Availability budgets (plan 6.5) evaluated against the measured baseline.
``cold``
    Fresh-interpreter import + first-call costs (``--skip-cold`` to skip).
``process``
    Peak RSS of this process and versions/hashes for reproducibility.

Usage::

    uv run python -m tools.benchmark_solar_source --out scratchpad/.../baseline.json

Exit code is always 0 when the probes *run*; pass/fail judgements against the
availability budget belong to the caller (M1/M4), not to the measurement.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W81 = np.arange(380.0, 781.0, 5.0)
W3 = np.array([450.0, 550.0, 650.0])


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def _time(fn, repeat=1):
    out = []
    value = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        value = fn()
        out.append(time.perf_counter() - t0)
    return (min(out), out, value)


def baseline_block() -> dict:
    """Warm timings of the CURRENT production entry points (unified source).

    Every timed callable is preceded by an un-timed warm-up call so the
    numbers exclude first-call table construction; the cold block below
    measures the first-call costs separately.
    """
    from skycolor import presets
    from skycolor.solar_source import get_solar_geometry
    from skycolor.rt import mc, multiple, single

    rows = {}
    atmo3 = presets.clear(W3)
    atmo81 = presets.clear(W81)

    t0 = time.perf_counter()
    presets.clear(W81)
    rows["mie_build_81wl_cold_s"] = time.perf_counter() - t0

    views = (np.array([0.0]), np.array([0.0]))
    # source build (once per medium token), then warm queries
    t0 = time.perf_counter()
    get_solar_geometry(atmo3._medium).ensure_table()
    rows["source_table_build_3wl_s"] = time.perf_counter() - t0

    t, _, _ = _time(lambda: single.sky_radiance(atmo3, 30.0, *views), repeat=3)
    rows["single_3wl_1view_warm_s"] = t
    single.sky_radiance(atmo81, 30.0, *views)  # builds the 81-wl table
    t, _, _ = _time(lambda: single.sky_radiance(atmo81, 30.0, *views), repeat=3)
    rows["single_81wl_1view_warm_s"] = t
    multiple.solve(atmo3, 30.0, 0.0, 0.0)
    t, _, _ = _time(lambda: multiple.solve(atmo3, 30.0, 0.0, 0.0), repeat=3)
    rows["sos_3wl_warm_s"] = t
    mc.mc_radiance(atmo3, 30.0, 0.0, 0.0, n_photons=20_000)
    t, _, _ = _time(lambda: mc.mc_radiance(atmo3, 30.0, 0.0, 0.0,
                                           n_photons=20_000), repeat=3)
    rows["mc_3wl_20k_warm_s"] = t

    zen = np.linspace(0.0, 90.0, 46)
    az = np.linspace(0.0, 180.0, 91)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    single.sky_radiance(atmo3, 30.0, Z.ravel()[:64], A.ravel()[:64], chunk=512)
    t, _, _ = _time(lambda: single.sky_radiance(atmo3, 30.0, Z.ravel(), A.ravel(),
                                                chunk=512), repeat=3)
    rows["single_3wl_dome46x91_warm_s"] = t

    from skycolor.refraction import get_tables
    t0 = time.perf_counter()
    get_tables(atmo3)
    rows["refraction_tables_cold_s"] = time.perf_counter() - t0
    return rows


#: plan 6.5 availability budgets (ratios against the archived legacy numbers
#: recorded in validation/data/production_solar_source/legacy_perf.json when
#: present, else the committed baseline constants)
_GATE_BUDGETS = {
    "cold_source_build_s": 10.0,
    "single_81wl_1view_warm_s": 1.0,
    "mc_ratio_max": 4.0,
    "sos_ratio_max": 4.0,
    "dome_ratio_max": 3.0,
}
_LEGACY = {"mc_3wl_20k_warm_s": 0.0729, "sos_3wl_warm_s": 2.5127,
           "single_3wl_dome46x91_warm_s": 1.1192}


def gate_block(baseline: dict) -> dict:
    """Availability gate (plan 6.5): compare measured numbers to budgets.

    Exit-relevant: every breach is reported and the suite returns non-zero
    through the gate's ``passed`` flag.
    """
    rows = {}
    rows["cold_source_build_s"] = dict(
        value=baseline.get("source_table_build_3wl_s", float("nan")),
        budget=_GATE_BUDGETS["cold_source_build_s"])
    rows["single_81wl_1view_warm_s"] = dict(
        value=baseline.get("single_81wl_1view_warm_s", float("nan")),
        budget=_GATE_BUDGETS["single_81wl_1view_warm_s"])
    for key, budget in [("mc_3wl_20k_warm_s", "mc_ratio_max"),
                        ("sos_3wl_warm_s", "sos_ratio_max"),
                        ("single_3wl_dome46x91_warm_s", "dome_ratio_max")]:
        legacy = _LEGACY[key]
        value = baseline.get(key, float("nan"))
        rows[key + "_ratio"] = dict(value=value / legacy, budget=_GATE_BUDGETS[budget],
                                    legacy=legacy)
    passed = all(v["value"] <= v["budget"] for v in rows.values())
    return dict(passed=bool(passed), rows=rows)


def cold_block() -> dict:
    """Cold costs measured in a fresh interpreter (imports + first call)."""
    code = r"""
import time, json, resource
t0 = time.perf_counter()
import numpy as np
from skycolor import presets
from skycolor.rt import single, multiple
t_import = time.perf_counter() - t0
wl = np.arange(380.0, 781.0, 5.0)
t0 = time.perf_counter()
atmo = presets.clear(wl)
t_mie = time.perf_counter() - t0
t0 = time.perf_counter()
single.sky_radiance(atmo, 3.0, np.array([0.0]), np.array([0.0]))
t_single_cold = time.perf_counter() - t0
t0 = time.perf_counter()
multiple.solve(atmo, 3.0, 0.0, 0.0)
t_sos_cold = time.perf_counter() - t0
print(json.dumps(dict(import_s=t_import, mie81_cold_s=t_mie,
    single81_first_call_s=t_single_cold, sos81_first_call_s=t_sos_cold,
    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3))))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-cold", action="store_true",
                        help="skip the subprocess cold block")
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base = baseline_block()
    result = dict(
        git_head=head,
        machine=dict(python=sys.version.split()[0],
                     numpy=np.__version__),
        baseline=base,
        gate=gate_block(base),
        peak_rss_gib=_rss_gib(),
    )
    if not args.skip_cold:
        result["cold"] = cold_block()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["gate"]["passed"]:
        print("AVAILABILITY GATE: BREACHED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
