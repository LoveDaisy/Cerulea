"""Bisect the P3 segfault: tiny-budget MC across view/wavelength subsets.

The full probe crashed with SIGSEGV inside ``solve(approximation='spherical-mc')``
on the restricted 14-wl atmosphere and the 88-view meridian cut.  This script
re-runs that request at a tiny photon budget across view subsets and
atmosphere variants, twice each, to isolate the trigger.

Run: PYTHONPATH=<repo root> uv run python scratchpad/task-aureole-delta-fit/probe_segfault_bisect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation import common as C  # noqa: E402
from skycolor.rt.solver import (  # noqa: E402
    SPHERICAL_MC, MCConfig, solve,
)

SUN_ZEN = 45.0
TINY = dict(n_photons=3.0e5, n_anchor=3)

z_grid, a_grid = C.sky_grid(37, 72)
zv = z_grid[:, 0]
near = np.concatenate([45.0 - np.array([1, 1.5, 2, 3, 4, 6, 8]),
                       45.0 + np.array([1, 1.5, 2, 3, 4, 6, 8])])
SUBSETS = {
    "zen_horizon_2": (np.array([0.0, 90.0]), np.array([0.0, 0.0])),
    "az0_col_37": (zv, np.zeros(37)),
    "az180_col_37": (zv, np.full(37, 180.0)),
    "near_sun_14": (near, np.zeros(near.size)),
    "cut_88": (np.concatenate([zv, near, zv]),
               np.concatenate([np.zeros(37), np.zeros(14), np.full(37, 180.0)])),
}


def try_case(label, atmo, z, a, mc_cfg, repeat=2):
    for i in range(repeat):
        try:
            res = solve(atmo, SUN_ZEN, z, a, sun_azimuth_deg=0.0,
                        approximation=SPHERICAL_MC, mc_config=mc_cfg)
            rad = res.radiance
            ok = bool(np.all(np.isfinite(rad)))
            print(f"  [{label}] rep{i+1}: OK finite={ok} "
                  f"shape={rad.shape} sum={float(np.nansum(rad)):.4e}",
                  flush=True)
        except Exception as exc:
            print(f"  [{label}] rep{i+1}: EXCEPTION {type(exc).__name__}: {exc}",
                  flush=True)


def main() -> None:
    atmo_full = C.atmosphere("clear", C.WL_COARSE)
    sub = np.arange(0, C.WL_COARSE.size, 6)
    atmo_sub = atmo_full._restrict(sub)
    mc = MCConfig(quality="medium", seed=0, grid=dict(TINY))

    for name, (z, a) in SUBSETS.items():
        print(f"subset {name} (n={z.size}) restricted-atmo:", flush=True)
        try_case(name, atmo_sub, z, a, mc)

    print("subset cut_88 (n=88) FULL-spectrum atmo:", flush=True)
    z, a = SUBSETS["cut_88"]
    try_case("cut88_full_atmo", atmo_full, z, a, mc)

    print("subset cut_88 restricted, horizon clamped to 89.9:", flush=True)
    z2, a2 = SUBSETS["cut_88"]
    z2 = z2.copy()
    z2[z2 == 90.0] = 89.9
    try_case("cut88_no90", atmo_sub, z2, a2, mc)

    print("subset cut_88 restricted, near-sun dropped:", flush=True)
    z3 = np.concatenate([zv, zv])
    a3 = np.concatenate([np.zeros(37), np.full(37, 180.0)])
    try_case("cut74_no_near", atmo_sub, z3, a3, mc)

    print("bisect done", flush=True)


if __name__ == "__main__":
    main()
