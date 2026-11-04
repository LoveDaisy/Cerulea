"""Cost probes for the production solar-source task (baseline vs unified source).

Runs on one machine in one script so the numbers are comparable.  It reports
three blocks with separate, additive cost accounting (Mie construction is
measured on its own, so the *source* cost is never contaminated by it):

``baseline``
    Current production entry points (no unified source): cold import/build,
    warm ``rt.single``/SOS/MC/dome/CLI-sized calls.  This is the "old" side
    of every ratio in the availability gate.
``source_experimental``
    The verified experimental algorithms that the production source will
    migrate (``tools.radial_solar_mapping`` / ``continuous_solar_columns`` /
    ``point_solar_table``): profile build, mapping build, column tool build,
    table build at pilot and default density, warm queries at the sizes the
    backends actually need (single point, 1000 vertices, 46x91 dome, 81-wl
    single direction).
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
    """Warm timings of current production entry points."""
    from skycolor import presets
    from skycolor.rt import mc, multiple, single

    rows = {}
    atmo3 = presets.clear(W3)
    atmo81 = presets.clear(W81)

    # Mie construction is the dominant cold cost of an 81-wl medium; measure
    # it separately so source costs can be compared against a clean base.
    t0 = time.perf_counter()
    presets.clear(W81)
    rows["mie_build_81wl_cold_s"] = time.perf_counter() - t0

    views = (np.array([0.0]), np.array([0.0]))
    single.sky_radiance(atmo3, 30.0, *views)   # warm-up: builds the table
    t, _, _ = _time(lambda: single.sky_radiance(atmo3, 30.0, *views), repeat=3)
    rows["single_3wl_1view_warm_s"] = t
    t, _, _ = _time(lambda: single.sky_radiance(atmo81, 30.0, *views))
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
    single.sky_radiance(atmo3, 30.0, Z.ravel()[:8], A.ravel()[:8], chunk=512)
    t, _, _ = _time(lambda: single.sky_radiance(atmo3, 30.0, Z.ravel(), A.ravel(),
                                                chunk=512), repeat=3)
    rows["single_3wl_dome46x91_warm_s"] = t

    # refraction tables (shadow thresholds) are a separate cold cost shared by
    # every refracted backend; time the first call that builds them.
    from skycolor.refraction import get_tables
    t0 = time.perf_counter()
    get_tables(atmo3)
    rows["refraction_tables_cold_s"] = time.perf_counter() - t0
    return rows


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


def source_experimental_block() -> dict:
    """Build/query costs of the experimental algorithms to be migrated."""
    from skycolor import presets
    from tools.continuous_solar_columns import ContinuousSolarColumns
    from tools.point_solar_table import PointSolarTable
    from tools.radial_solar_mapping import RadialSolarMapping

    rows = {}
    atmo3 = presets.clear(W3)
    atmo81 = presets.clear(W81)

    t0 = time.perf_counter()
    prof3 = RadialSolarMapping(atmo3)
    rows["mapping_build_3wl_s"] = time.perf_counter() - t0
    # mapping construction itself is wavelength independent; the 81-wl number
    # isolates RefractiveProfile tabulation + spline setup
    t0 = time.perf_counter()
    RadialSolarMapping(atmo81)
    rows["mapping_build_81wl_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cols3 = ContinuousSolarColumns(atmo3, mapping=prof3)
    rows["columns_build_3wl_s"] = time.perf_counter() - t0

    h = np.full(1000, 40.0)
    app = np.linspace(-5.2, -5.8, 1000)
    t, _, _ = _time(lambda: cols3.columns(h[:2], app[:2]), repeat=3)
    rows["columns_2pts_warm_s"] = t
    t, _, v = _time(lambda: cols3.columns(h, app), repeat=3)
    rows["columns_1000pts_warm_s"] = t
    rows["columns_1000pts_finite"] = bool(np.isfinite(v).all())

    # pilot table (the density the availability gate assumes achievable)
    t0 = time.perf_counter()
    table3 = PointSolarTable(atmo3, n_height=17, label="cost-probe")
    rows["table_build_pilot17_3wl_s"] = time.perf_counter() - t0
    rows["table_pilot17_rows"] = int(len(table3.heights))
    rows["table_pilot17_knots"] = int(sum(len(r["t"]) for r in table3.rows))

    t, _, _ = _time(lambda: table3.query(np.array([40.0]), np.array([-6.0]), iw=0))
    rows["query_1pt_1wl_warm_s"] = t
    t, _, _ = _time(lambda: table3.query(np.array([40.0]), np.array([-6.0])))
    rows["query_1pt_allwl_warm_s"] = t
    rng = np.random.default_rng(7)
    zq = np.full(1000, 40.0)
    eq = rng.uniform(-15.0, 89.0, 1000)
    t, _, _ = _time(lambda: table3.query(zq, eq, iw=0))
    rows["query_1000pts_1wl_warm_s"] = t
    t, _, _ = _time(lambda: table3.query(zq, eq))
    rows["query_1000pts_allwl_warm_s"] = t
    zen = np.linspace(0.0, 90.0, 46)
    az = np.linspace(0.0, 180.0, 91)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    t, _, _ = _time(lambda: table3.query_fields(np.full(Z.size, 40.0),
                                                np.full(Z.size, 20.0)))
    rows["query_dome46x91_1wl_warm_s"] = t

    # full-density table at 3 wl: the default the backends would build
    t0 = time.perf_counter()
    table3f = PointSolarTable(atmo3, label="cost-probe-full")
    rows["table_build_default_3wl_s"] = time.perf_counter() - t0
    rows["table_default_rows"] = int(len(table3f.heights))
    rows["table_default_knots"] = int(sum(len(r["t"]) for r in table3f.rows))

    t0 = time.perf_counter()
    PointSolarTable(atmo81, label="cost-probe-81")
    rows["table_build_default_81wl_s"] = time.perf_counter() - t0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-cold", action="store_true",
                        help="skip the subprocess cold block")
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = dict(
        git_head=head,
        machine=dict(python=sys.version.split()[0],
                     numpy=np.__version__),
        baseline=baseline_block(),
        source_experimental=source_experimental_block(),
        peak_rss_gib=_rss_gib(),
    )
    if not args.skip_cold:
        result["cold"] = cold_block()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
