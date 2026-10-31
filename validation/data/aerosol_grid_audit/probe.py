"""Record grid invariance and exact full-grid regression before/after the fix."""
from pathlib import Path
import json
import subprocess
import sys
from time import perf_counter

import numpy as np

from skycolor.aerosol import AngstromAerosol, get_aerosol

HERE = Path(__file__).resolve().parent
NAMES = ["rural", "urban", "maritime", "desert_dust", "biomass_burning",
         "volcanic_stratospheric", "cirrus", "water_cloud"]


def main(stage):
    report = {"stage": stage, "head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "members": {}}
    arrays = {}
    for name in NAMES:
        a = get_aerosol(name)
        start = perf_counter()
        reference = np.array(a.relative_coefficients([450., 550., 650.]))
        cold3 = perf_counter() - start
        scalar = np.array([a.relative_coefficients([w]) for w in [450., 550., 650.]])[:, :, 0].T
        no550 = np.array(a.relative_coefficients([450., 650.]))
        one = {"three_nm": [450, 550, 650], "three": reference.tolist(),
               "scalar": scalar.tolist(), "without550": no550.tolist(),
               "cold_three_seconds": cold3,
               "scalar_rel_change_vs_three": (scalar / reference - 1).tolist()}
        for size, wl in [(81, np.arange(380., 781., 5.)),
                         (401, np.arange(380., 781.))]:
            start = perf_counter()
            full = np.array(a.relative_coefficients(wl))
            one[f"full{size}_seconds"] = perf_counter() - start
            arrays[f"{name}_{size}"] = full
            if size == 81:
                single = np.array([a.relative_coefficients([w]) for w in wl])[:, :, 0].T
                mask = wl != 550
                subset = np.array(a.relative_coefficients(wl[mask]))
                one["single81_max_rel_error"] = float(np.max(abs(single / full - 1)))
                one["without550_80_max_rel_error"] = float(np.max(abs(subset / full[:, mask] - 1)))
                one["full81_over_single81_minmax"] = [float(np.min(full / single)),
                                                       float(np.max(full / single))]
        if stage == "after":
            old = np.load(HERE / "baseline-full-grids.npz")
            one["full81_bitwise_unchanged"] = bool(np.array_equal(old[f"{name}_81"], arrays[f"{name}_81"]))
            one["full401_bitwise_unchanged"] = bool(np.array_equal(old[f"{name}_401"], arrays[f"{name}_401"]))
        # A fresh off-reference scalar query needs one extra reference state.
        cold = get_aerosol(name)
        start = perf_counter()
        cold.relative_coefficients([450.])
        one["cold_scalar_seconds"] = perf_counter() - start
        one["cold_scalar_cached_wavelengths"] = list(cold._states)
        start = perf_counter()
        cold.relative_coefficients([450.])
        one["warm_scalar_seconds"] = perf_counter() - start
        report["members"][name] = one
        print(name, "done", flush=True)
    a = AngstromAerosol(alpha=1.3, ssa=.94, ssa_slope=1e-4)
    report["angstrom"] = {"without550": np.array(a.relative_coefficients([450., 650.])).tolist(),
        "analytic_extinction": ((np.array([450., 650.]) / 550.) ** -1.3).tolist()}
    (HERE / f"probe-{stage}.json").write_text(json.dumps(report, indent=2) + "\n")
    if stage == "before":
        np.savez_compressed(HERE / "baseline-full-grids.npz", **arrays)
    else:
        (HERE / "probe.json").write_text(json.dumps({"before": json.loads(
            (HERE / "probe-before.json").read_text()), "after": report}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
