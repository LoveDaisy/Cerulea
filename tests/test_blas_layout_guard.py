"""Guards for the Accelerate dgemm F-contiguous-B overread mitigation.

macOS arm64 numpy wheels (2.5.4 verified) dispatch matmul to the system
Accelerate BLAS, whose dgemm OVERREADS past an operand buffer end when the
*second* operand is F-contiguous (a transposed view) and the inner
dimension is k=3 -- the Cartesian-vector shape of every geometry matmul in
skycolor.  The overread is read-only (no value corruption) but SIGSEGVs
whenever it crosses into an unmapped page, which is heap-layout dependent:
intermittent in production, immediate and deterministic under libgmalloc
(task-mc-restrict-segv; probe evidence and the no-skycolor minimal
reproduction in ``scratchpad/task-mc-restrict-segv/findings.md``).

The single authoritative mitigation is
``skycolor.optics.contiguous_transpose``: every k=3 transposed operand is
routed through its C-contiguous copy (``solar_source`` / ``color`` /
``rt.twilight`` import it as the module-local ``_cT`` alias).  Two
deterministic anchors live here:

* the source inventory: no ``x @ y.T`` matmul node (AST-level) remains in
  skycolor executable code -- a reintroduced site fails this test before
  it can reach the buggy kernel.  Coverage boundary: only ``@``-operator
  nodes with a ``.T`` in the right operand subtree are scanned;
  ``np.matmul(x, y.T)``, 2-D ``np.dot(M, N.T)`` and ``y.transpose()``
  spellings of the same hazard are NOT mechanically caught -- they are
  barred by the AGENTS.md DO-NOT convention and by review.  (The gate is
  deliberately stricter than k=3: any transposed right operand is
  flagged, which over-blocks harmless k>3 sites in the safe direction.)
* the bit-equality contract: for production row counts (m >= 64, where
  numpy selects the BLAS path) the transposed-view and C-contiguous
  kernels agree bitwise, which is what makes the mitigation a
  physical-zero-change fix (the golden fixtures pin the end-to-end
  property; below the BLAS threshold the two layouts differ at the ulp
  level because numpy's own inner loop accumulates along strides).

The ONLY passing rewrites of a flagged site: route the operand through
``contiguous_transpose`` (its ``.T`` lives inside the helper, outside any
matmul), or assign ``np.ascontiguousarray(y.T)`` to a variable first and
matmul against that variable.  Writing ``x @ np.ascontiguousarray(y.T)``
inline does NOT pass -- the scanner walks the whole right-operand subtree
and the ``.T`` inside the call argument is still flagged.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import skycolor.color as color
import skycolor.solar_source as solar_source
from skycolor.optics import contiguous_transpose

SKYCOLOR = Path(__file__).absolute().parents[1] / "skycolor"


def _matmul_transposed_right_operands(tree: ast.AST):
    """Yield the line numbers of MatMult nodes whose right operand uses ``.T``.

    ``a @ b.T`` reaches Accelerate's overreading kernel; ``a.T @ b`` (the
    F-contiguous FIRST operand) is a different kernel that measurably does
    not overread, so only right operands are flagged.  The scan is
    subtree-wide, which is why the inline form
    ``x @ np.ascontiguousarray(y.T)`` is also flagged -- see the module
    docstring for the passing rewrites.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            for sub in ast.walk(node.right):
                if isinstance(sub, ast.Attribute) and sub.attr == "T":
                    yield node.lineno
                    break


def test_no_transposed_right_operand_matmul_in_skycolor():
    offenders = []
    for path in sorted(SKYCOLOR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno in _matmul_transposed_right_operands(tree):
            offenders.append(f"{path.relative_to(SKYCOLOR.parent)}:{lineno}")
    assert not offenders, (
        "matmul with a transposed-view right operand (Accelerate dgemm "
        "overread at k=3): route the operand through "
        "skycolor.optics.contiguous_transpose, or assign "
        "np.ascontiguousarray(y.T) to a variable before the matmul "
        "(an inline call argument is still scanned) -- "
        + ", ".join(offenders))


def test_transpose_aliases_are_the_single_authority():
    """The module-local ``_cT`` names must BE ``optics.contiguous_transpose``.

    One authoritative implementation (AGENTS.md a56 discipline): the
    solar_source / color aliases exist only as import spellings.
    """
    assert solar_source._cT is contiguous_transpose
    assert color._cT is contiguous_transpose


def test_contiguous_transpose_helper_contract():
    a = np.arange(24.0).reshape(8, 3)
    out = contiguous_transpose(a)
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, a.T)
    assert not np.shares_memory(out, a)


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="the bitwise kernel-equality contract is a "
                           "property of the macOS Accelerate backend")
@pytest.mark.skipif("libgmalloc" in os.environ.get("DYLD_INSERT_LIBRARIES", ""),
                    reason="the transposed-view operand is computed here "
                           "deliberately, to pin the bitwise contract "
                           "against the fixed form; under guard malloc "
                           "that operand triggers exactly the overread "
                           "this suite guards against")
@pytest.mark.parametrize("m", [64, 2604, 65536])
@pytest.mark.parametrize("nb", [2, 16, 64])
def test_transposed_and_contiguous_kernels_agree_bitwise_at_production_rows(
        m, nb):
    """The mitigation must not move a bit at BLAS-path row counts (m >= 64).

    Below ~64 rows numpy's own inner loop (stride-dependent accumulation)
    differs between layouts at the ulp level; that regime is accepted and
    is exercised end-to-end by the golden fixtures instead.  The second
    operand's column count varies across production call sites (disk
    beams, view counts), so the contract is pinned across nb, not just at
    the disk's 16.
    """
    rng = np.random.default_rng(11)
    a = rng.standard_normal((m, 3))
    b = rng.standard_normal((nb, 3))
    r_trans = a @ b.T
    r_contig = a @ contiguous_transpose(b)
    assert np.array_equal(r_trans.view(np.int64), r_contig.view(np.int64))
