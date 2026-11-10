"""Repeatable performance benchmark for the optical pre-processing path.

For Mie aerosols the dominant cost of a sky-radiance evaluation is the
per-wavelength pre-processing -- Mie coefficients, analytic Legendre
moments and the adaptive phase tables -- not the single-scattering
transfer itself.  This script times exactly that boundary, so cache-related
changes can be evaluated in isolation.  It prints numbers only; the
acceptance thresholds live in the task notes (scratchpad/task-mie-cache,
handoff 2026-10-30 §3.6), not here.

Scenarios (81-wavelength working grid, ``skycolor.constants.WL_COARSE``):

1. warm ``single`` sky_radiance, clear preset, one viewing direction; the
   cold call is acceptance-relevant for the batched Mie kernels since
   task-cold-optics: the headline gate is cold dusty ``delta_m(32)`` at
   >= 10x over that task's Step-1 baseline (measured 39.4 -> 3.1 s, ~13x,
   as recorded in AGENTS.md and ``validation/FAST_SUITE.md``; the
   clear-scene cold call is dominated by the out-of-scope solar-source
   table construction);
2. the second ``Atmosphere.delta_m_properties(32)`` on the *same* clear
   atmosphere -- with per-wavelength states and cached moments this
   should be nearly free;
3. ``delta_m(32)`` of the aerosol of a *second* ``presets.dusty(...)``
   construction -- this is the cross-instance case: with shared preset
   instances it hits the caches warmed by the first build;
4. peak RSS from ``resource.getrusage``, the memory checkpoint of the
   enlarged wavelength cache.

Run from the repository root:

    uv run python tools/perf_bench.py
"""
from __future__ import annotations

import pathlib
import resource
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets  # noqa: E402
from skycolor.rt.single import sky_radiance as single  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<46s} {dt:9.3f} s")
    return out


def main() -> None:
    wl = WL_COARSE
    print(f"perf_bench: {wl.size} wavelengths "
          f"({wl[0]:.0f}-{wl[-1]:.0f} nm, 5 nm), 1 viewing direction\n")

    # 1. warm single-scattering radiance, clear, one direction
    atm = presets.clear(wl)
    _timed("cold single clear 1 dir (context only)",
           lambda: single(atm, 45.0, np.array([0.0]), np.array([0.0])))
    _timed("warm single clear 1 dir",
           lambda: single(atm, 45.0, np.array([0.0]), np.array([0.0])))

    # 2. second delta_m_properties on the same atmosphere
    _timed("first delta_m_properties(32) clear (context only)",
           lambda: atm.delta_m_properties(32))
    _timed("second delta_m_properties(32) clear",
           lambda: atm.delta_m_properties(32))

    # 3. second presets.dusty() build, aerosol-level delta_m
    dusty1 = presets.dusty(wl)
    _timed("cold dusty delta_m(32) (context only)",
           lambda: dusty1.aerosols[0].aerosol.delta_m(wl, 32))
    dusty2 = presets.dusty(wl)
    reused = dusty1.aerosols[0].aerosol is dusty2.aerosols[0].aerosol
    print(f"  {'preset dust aerosol reused (is)':<46s} {str(reused):>9s}")
    _timed("second presets.dusty delta_m(32)",
           lambda: dusty2.aerosols[0].aerosol.delta_m(wl, 32))

    # 4. memory checkpoint.  ru_maxrss is bytes on macOS, KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    print(f"\n  peak RSS (ru_maxrss): {mib:.0f} MiB")


if __name__ == "__main__":
    main()
