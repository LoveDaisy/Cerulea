"""Regression anchor: repeated spherical-mc solves on a restricted atmosphere.

Issue hard constraint (task-mc-restrict-segv): the reproduction script
(``validation/tools/aureole_attribution/mc_repeat_segfault_stress.py``, 30
repeats of the same tiny-budget solve in one process) must live in the
suite as an automated anchor.  The crash it reproduces was traced to an
Accelerate dgemm overread (F-contiguous second operand at k=3) reached
through the deterministic single-scattering pass of EVERY solve -- one
process repeating ``solve(approximation='spherical-mc')`` on a
``_restrict``-subset atmosphere crashed after 4-8 reps, heap-layout
dependent and intermittent (deterministic under libgmalloc).  See
``skycolor/optics.py`` (``contiguous_transpose``, the single authoritative
mitigation) for the mechanism and
``scratchpad/task-mc-restrict-segv/findings.md`` for the probe evidence.

The anchor keeps the stress SHAPE -- one process, one ``_restrict``-subset
atmosphere, repeated identical solves, the promoted stress script's exact
atmosphere construction (``validation.common.atmosphere`` + ``_restrict``)
-- at a reduced per-rep cost: 2 view directions (the bisect script's
minimal case) and a 3e4-photon budget instead of the original 14-view /
3e5-photon scene.  The overreading matmuls live in the deterministic pass
and do not depend on the photon budget, so the call-pattern coverage (30
reps, ~4x the historical crash window of 4-8) is preserved at a fraction
of the original wall clock: the whole anchor measured 47.7 s in the suite
(task progress, 2026-11-11; ~25 s of that is the one-time solar table
build, warm reps ~1.5 s each).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow  # tier: ~45 s serial stress anchor (task-mc-restrict-segv); cost principle <=2 s puts it in the slow tier

import numpy as np
import pytest

from skycolor.rt.solver import MCConfig, SPHERICAL_MC, solve
from validation import common as C

SUB = np.arange(0, C.WL_COARSE.size, 6)
MC = MCConfig(quality="medium", seed=0, grid=dict(n_photons=3.0e4, n_anchor=3))


@pytest.fixture(scope="module")
def restricted_atmo():
    return C.atmosphere("clear", C.WL_COARSE)._restrict(SUB)


def test_repeated_restricted_mc_solves(restricted_atmo):
    views = np.array([0.0, 89.9])
    reference = None
    for i in range(30):
        res = solve(restricted_atmo, 45.0, views, np.zeros(views.size),
                    sun_azimuth_deg=0.0, approximation=SPHERICAL_MC,
                    mc_config=MC)
        total = res.radiance
        assert np.all(np.isfinite(total))
        if reference is None:
            reference = total
        else:
            # same seed and inputs: every rep must reproduce the first
            # solve bitwise -- a drift would mean hidden mutable state
            # crossed calls, which is exactly the failure family this
            # anchor guards
            np.testing.assert_array_equal(total, reference)
    assert float(np.sum(reference)) > 0.0
