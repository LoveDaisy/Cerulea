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

The mitigation routes every k=3 transposed operand through a C-contiguous
copy (``solar_source._cT`` / ``color._cT`` / the inlined copy in
``rt/twilight._angular_distance``).  Two deterministic anchors live here:

* the source inventory: no ``x @ y.T`` matmul node (AST-level) remains in
  skycolor executable code -- a reintroduced site fails this test before
  it can reach the buggy kernel;
* the bit-equality contract: for production row counts (m >= 64, where
  numpy selects the BLAS path) the transposed-view and C-contiguous
  kernels agree bitwise, which is what makes the mitigation a
  physical-zero-change fix (the golden fixtures pin the end-to-end
  property; below the BLAS threshold the two layouts differ at the ulp
  level because numpy's own inner loop accumulates along strides).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from skycolor import color
from skycolor.solar_source import _cT

SKYCOLOR = Path(__file__).absolute().parents[1] / "skycolor"


def _matmul_transposed_right_operands(tree: ast.AST):
    """Yield (file, lineno) of MatMult nodes whose right operand uses ``.T``.

    ``a @ b.T`` reaches Accelerate's overreading kernel; the mitigation
    rewrites such sites as ``a @ _cT(b)`` (or ``np.ascontiguousarray(b.T)``),
    whose right operand is a Call and carries no ``.T``.  ``a.T @ b`` (the
    F-contiguous FIRST operand) is a different kernel that measurably does
    not overread, so only right operands are flagged.
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
        "overread at k=3): route through solar_source._cT / color._cT or "
        "np.ascontiguousarray -- " + ", ".join(offenders))


def test_contiguous_transpose_helper_contract():
    a = np.arange(24.0).reshape(8, 3)
    for cT in (_cT, color._cT):
        out = cT(a)
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
def test_transposed_and_contiguous_kernels_agree_bitwise_at_production_rows(m):
    """The mitigation must not move a bit at BLAS-path row counts (m >= 64).

    Below ~64 rows numpy's own inner loop (stride-dependent accumulation)
    differs between layouts at the ulp level; that regime is accepted and
    is exercised end-to-end by the golden fixtures instead.
    """
    rng = np.random.default_rng(11)
    a = rng.standard_normal((m, 3))
    b = rng.standard_normal((16, 3))
    r_trans = a @ b.T
    r_contig = a @ _cT(b)
    assert np.array_equal(r_trans.view(np.int64), r_contig.view(np.int64))
