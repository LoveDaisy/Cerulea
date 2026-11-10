"""Anchors for the batch-vectorised Mie kernels (task-cold-optics).

The batch kernels must reproduce the scalar production path:
``mie_coefficients_batch`` bitwise (probe route A3c -- both division sites of
the scalar path are reproduced exactly), while the batched moment and
phase-table kernels carry the task's D2 declared tolerances (the per-band
``np.dot`` reduction of the scalar path dispatches to BLAS on long rows and
its accumulation order cannot be replicated by any batched form; measured
drift is <= 9e-13 relative on the real pair population).

The pinned arrays in ``tests/data/mie_anchor.npz`` have two provenance
classes: the kernel-level ``coef_*`` grids are scalar-path truth (the scalar
kernel is unchanged by the task); the preset-level production surfaces were
regenerated from the batched tree on 2026-11-10 during the 9-wavelength
table-anchor slimming, with scalar equivalence at that point evidenced by the
dev-period ref_checkpoint (24/24 -- see the task progress log and the
``mie_anchor.meta.json`` sidecar).  Regenerate only with an explicit recorded
reason -- never to make a failing comparison pass (same rule as the golden
fixtures).

Platform binding: the bitwise class is anchored to this machine's toolchain
(CPython 3.12 Smith+FMA division contraction, this numpy/BLAS build -- see
``_fma`` / ``_smith_division_constants`` in ``skycolor/mie.py``).  On a
different toolchain a wholesale bitwise-anchor failure is the expected
scream, not a physics regression: attribute the platform difference first,
then re-baseline through the recorded-reason route.
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


def test_state_surface_arrays_vs_anchor(anchor):
    """Plan 3b/3d test points: rc/asym bitwise, production single within TOL.

    The anchor arrays pin the pre-change scalar tree; the coefficient kernel
    is bitwise (so rc/asym must match exactly) and the production radiance
    carries the declared table-kernel tolerance.
    """
    from skycolor.rt.single import sky_radiance as single

    for name, factory in (("clear", presets.clear), ("dusty", presets.dusty)):
        aer = factory(WL_COARSE).aerosols[0].aerosol
        ext, sca = aer.relative_coefficients(WL_COARSE)
        assert np.array_equal(ext, anchor[f"{name}_rc_ext"])
        assert np.array_equal(sca, anchor[f"{name}_rc_sca"])
        assert np.array_equal(aer.asymmetry(WL_COARSE), anchor[f"{name}_asym"])
        atm = factory(WL_COARSE)
        out = single(atm, 80.0,
                     np.array([0.0, 20.0, 45.0, 70.0]),
                     np.array([0.0, 90.0, 180.0, 270.0]))
        assert np.allclose(out, anchor[f"{name}_single"], **TOL_SINGLE)


def test_batch_coefficients_absorbing_index_branch2_bitwise():
    """Branch-2 Smith pairs (|Im m| > |Re m|) reproduce the scalar path.

    The library indices are all branch 1; the public kernel accepts any m,
    and the branch-2 column patch must genuinely write through.
    """
    xs = np.array([0.5, 2.0, 8.0, 40.0])
    for m in (1.0 + 10.0j, 0.5 + 3.0j, 0.2 + 5.0j):
        ms = np.full(xs.size, m)
        bat = mie_coefficients_batch(xs, ms)
        for j in range(xs.size):
            ref = mie_coefficients(float(xs[j]), complex(m))
            nx = ref.nmax
            assert np.array_equal(bat.a[j, : nx + 1], ref.a)
            assert np.array_equal(bat.b[j, : nx + 1], ref.b)
            assert bat.qext[j] == ref.qext
            assert bat.qsca[j] == ref.qsca


def test_batch_coefficients_mixed_branch_population_bitwise():
    """Mixed branch-1/branch-2 batches reproduce the scalar path.

    The branch-2 column patch must handle arbitrary branch composition:
    uniform branch-2 is covered by the absorbing-index test above, but a
    mixed population previously crashed on the full-width scratch
    (ValueError at the write-through assignment).  Two of the four pairs
    here are branch 2, so both the k=1 and 1<k<P shapes are exercised.
    """
    xs = np.array([0.5, 2.0, 8.0, 40.0])
    ms = np.array([1.5 + 0.1j, 0.2 + 5.0j, 1.75 + 0.45j, 0.2 + 5.0j])
    bat = mie_coefficients_batch(xs, ms)
    for j in range(xs.size):
        ref = mie_coefficients(float(xs[j]), complex(ms[j]))
        nx = ref.nmax
        assert np.array_equal(bat.a[j, : nx + 1], ref.a)
        assert np.array_equal(bat.b[j, : nx + 1], ref.b)
        assert bat.qext[j] == ref.qext
        assert bat.qsca[j] == ref.qsca


def test_empty_state_phase_falls_back_to_isotropic():
    """A state whose every node falls below the Rayleigh cutoff keeps the
    scalar path's isotropic fallback on the phase surface (no crash)."""
    from skycolor.aerosol import LogNormalMode, MieAerosol

    aer = MieAerosol(modes=[LogNormalMode(1e-8, 1.01)])
    assert len(aer._nodes) == 0
    t = aer.phase_table(550.0)
    assert t.p_peak == 1.0
    assert np.allclose(t.p, 1.0)
    mom = aer.legendre_moments(np.array([550.0]), 4)
    assert mom[0, 0] == 1.0
    assert np.allclose(mom[1:], 0.0)


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
        wl_sub = anchor[f"{name}_p_peak_wl"]
        assert np.allclose(
            np.array([aer.phase_table(w).p_peak for w in wl_sub]),
            anchor[f"{name}_p_peak"], **TOL_TABLE)
        assert np.allclose(
            np.array([aer.phase_table(w).normalisation() for w in wl_sub]),
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
