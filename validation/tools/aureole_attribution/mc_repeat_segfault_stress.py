"""Stress the tiny-budget MC in one process to characterise the segfault.

30 repeats of the same solve() call (same seed, tiny budget) in a single
process: if the crash is cumulative-state-dependent, it appears after N
clean runs; the faulthandler traceback marks the site.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).absolute().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation import common as C  # noqa: E402
from skycolor.rt.solver import SPHERICAL_MC, MCConfig, solve  # noqa: E402

z_grid, a_grid = C.sky_grid(37, 72)
zv = z_grid[:, 0]
near = np.concatenate([45.0 - np.array([1, 1.5, 2, 3, 4, 6, 8]),
                       45.0 + np.array([1, 1.5, 2, 3, 4, 6, 8])])

sub = np.arange(0, C.WL_COARSE.size, 6)
atmo = C.atmosphere("clear", C.WL_COARSE)._restrict(sub)
mc = MCConfig(quality="medium", seed=0, grid=dict(n_photons=3.0e5, n_anchor=3))

for i in range(30):
    res = solve(atmo, 45.0, near, np.zeros(near.size), sun_azimuth_deg=0.0,
                approximation=SPHERICAL_MC, mc_config=mc)
    print(f"rep {i+1}: OK sum={float(np.sum(res.radiance)):.6e}", flush=True)
print("all 30 reps OK", flush=True)
