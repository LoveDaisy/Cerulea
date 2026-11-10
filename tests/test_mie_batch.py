"""Anchors for the batch-vectorised Mie kernels (task-cold-optics).

The batch kernels must reproduce the scalar production path:
``mie_coefficients_batch`` bitwise (probe route A3c -- both division sites of
the scalar path are reproduced exactly), while the batched moment and
phase-table kernels carry the task's D2 declared tolerances (the per-band
``np.dot`` reduction of the scalar path dispatches to BLAS on long rows and
its accumulation order cannot be replicated by any batched form; measured
drift is <= 9e-13 relative on the real pair population -- see
scratchpad/task-cold-optics/probe_equivalence.log).

The pinned arrays in ``tests/data/mie_anchor.npz`` were captured from the
scalar tree; regenerate only with an explicit recorded reason (see
``tools/make_mie_anchor_arrays.py``).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from skycolor import presets
from skycolor.aerosol import DustAerosol, HinterlandAerosol
from skycolor.constants import WL_COARSE
from skycolor.mie import mie_coefficients, mie_coefficients_batch

DATA = pathlib.Path(__file__).parent / "data" / "mie_anchor.npz"

X_GRID = np.array([1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 8.4, 51.5, 167.0, 352.1,
                   500.0])
M_GRID = np.array([
    1.53 + 0.006j,
    1.55 + 0.001j,
    1.324 + 3200.0 / 550.0**2 + 1e-8j,
    1.75 + 0.45j,
    1.43 + 1e-7j,
])

# D2 declared tolerances (per-array; rtol/atol pairs for cross-zero
# quantities, pure relative for positive definite ones).  Measured drifts on
# the full clear+dusty pair population: moments <= 8.98e-13 rel, phase tables
# <= 1 ulp (3.1e-16); tolerances carry >10x margin.
TOL_MOMENTS = dict(rtol=1e-11, atol=1e-12)
TOL_TABLE = dict(rtol=1e-13, atol=0.0)
TOL_SINGLE = dict(rtol=1e-10, atol=0.0)


@pytest.fixture(scope="module")
def anchor():
    if not DATA.exists():
        pytest.fail("mie_anchor.npz missing; run tools/make_mie_anchor_arrays.py")
    with np.load(DATA, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# kernel level: bitwise
# ---------------------------------------------------------------------------
def test_batch_coefficients_match_scalar_bitwise():
    xs = np.concatenate([np.full(M_GRID.size, xv) for xv in X_GRID])
    ms = np.concatenate([M_GRID for _ in X_GRID])
    bat = mie_coefficients_batch(xs, ms)
    for j in range(xs.size):
        ref = mie_coefficients(float(xs[j]), complex(ms[j]))
        nx = ref.nmax
        assert np.array_equal(bat.a[j, : nx + 1], ref.a)
        assert np.array_equal(bat.b[j, : nx + 1], ref.b)
        assert bat.qext[j] == ref.qext
        assert bat.qsca[j] == ref.qsca
        assert bat.n_max[j] == ref.nmax


def test_batch_coefficients_anchor_bitwise(anchor):
    xs = anchor["coef_x"]
    ms = anchor["coef_m"]
    bat = mie_coefficients_batch(xs, ms)
    assert np.array_equal(bat.a, anchor["coef_a"])
    assert np.array_equal(bat.b, anchor["coef_b"])
    assert np.array_equal(bat.qext, anchor["coef_qext"])
    assert np.array_equal(bat.qsca, anchor["coef_qsca"])


def test_batch_moments_match_scalar_within_declaration(anchor):
    """Kernel-level: batched U_l vs per-pair scalar U_l on the anchor grid."""
    from types import SimpleNamespace

    from skycolor.mie import mie_moment_numerators, mie_moment_numerators_batch

    for j in range(anchor["coef_x"].size):
        xv = float(anchor["coef_x"][j])
        nx = int(anchor["coef_nmax"][j])
        coeff = SimpleNamespace(
            x=xv, qsca=float(anchor["coef_qsca"][j]), nmax=nx,
            a=anchor["coef_a"][j, : nx + 1], b=anchor["coef_b"][j, : nx + 1])
        U_ref = mie_moment_numerators(coeff, 32)
        U_bat = mie_moment_numerators_batch(
            np.array([xv]), np.array([coeff.qsca]),
            anchor["coef_a"][j: j + 1, : nx + 1],
            anchor["coef_b"][j: j + 1, : nx + 1],
            np.array([nx]), 32)
        assert np.allclose(U_bat[0], U_ref, **TOL_MOMENTS)


def test_batch_phase_table_preset_anchor_declared_tolerance(anchor):
    """Phase tables and the production radiance vs pinned (scalar) arrays."""
    for name, factory in (("clear", presets.clear), ("dusty", presets.dusty)):
        aer = factory(WL_COARSE).aerosols[0].aerosol
        for w in (380.0, 550.0, 780.0):
            p = aer.phase_table(w).p
            assert np.allclose(p, anchor[f"{name}_table_p{int(w)}"],
                               **TOL_TABLE)
        assert np.allclose(
            np.array([aer.phase_table(w).p_peak for w in WL_COARSE]),
            anchor[f"{name}_p_peak"], **TOL_TABLE)
        assert np.allclose(
            np.array([aer.phase_table(w).normalisation() for w in WL_COARSE]),
            anchor[f"{name}_p_norm"], **TOL_TABLE)


def test_batch_moments_preset_anchor_declared_tolerance(anchor):
    """Production surface: moments through the aerosol path vs pinned arrays."""
    for name, factory in (("clear", presets.clear), ("dusty", presets.dusty)):
        aer = factory(WL_COARSE).aerosols[0].aerosol
        beta = aer.legendre_moments(WL_COARSE, 36)
        assert beta.shape == anchor[f"{name}_beta_direct"].shape
        assert np.allclose(beta, anchor[f"{name}_beta_direct"], **TOL_MOMENTS)
        res = aer.delta_m(WL_COARSE, 32)
        assert np.allclose(res.beta, anchor[f"{name}_delta_m_beta"],
                           **TOL_MOMENTS)
        assert np.allclose(res.f, anchor[f"{name}_delta_m_f"],
                           rtol=1e-11, atol=0.0)


def test_batch_coefficients_full_preset_population_bitwise():
    """Every (node, wavelength) pair of both presets, the production set."""
    for factory in (HinterlandAerosol, DustAerosol):
        aer = factory()
        xs_l, ms_l = [], []
        for w in WL_COARSE:
            xs, ms = [], []
            for r, _wgt, mi in aer._nodes:
                x = 2.0 * np.pi * r / (float(w) * 1e-3)
                if x < 1e-4:
                    continue
                xs.append(x)
                ms.append(aer._index(mi, w))
            xs_l.append(np.array(xs))
            ms_l.append(np.array(ms, dtype=complex))
        for xs, ms in zip(xs_l, ms_l):
            bat = mie_coefficients_batch(xs, ms)
            for j in range(xs.size):
                ref = mie_coefficients(float(xs[j]), complex(ms[j]))
                nx = ref.nmax
                assert np.array_equal(bat.a[j, : nx + 1], ref.a)
                assert np.array_equal(bat.b[j, : nx + 1], ref.b)
                assert bat.qext[j] == ref.qext
                assert bat.qsca[j] == ref.qsca
