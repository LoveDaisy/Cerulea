"""Standalone probe for the upstream Accelerate dgemm overread (macOS arm64).

macOS arm64 numpy wheels (2.5.4 verified) dispatch matmul to the system
Accelerate BLAS, whose dgemm overreads past an operand buffer end when the
SECOND operand is F-contiguous (a transposed view) and the inner dimension
is k=3 -- the Cartesian shape of every geometry matmul in skycolor.  The
overread is read-only but SIGSEGVs when it crosses an unmapped page,
intermittently under the normal allocator and deterministically under
guard malloc.  Discovered via task-mc-restrict-segv; the production
mitigation is ``skycolor/optics.py::contiguous_transpose`` with the site
inventory in ``tests/test_blas_layout_guard.py``.

Run (crashes = bug still present on this numpy/Accelerate build):

    # deterministic under guard malloc
    DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib \\
        uv run python validation/tools/accelerate_dgemm_overread_probe.py

    # unguarded: the plain run exercises the same kernels for a while
    uv run python validation/tools/accelerate_dgemm_overread_probe.py

Subtests: ``trans`` (the crashing shape, B = (16, 3).T view) and ``contig``
(the clean kernel, bitwise-identical results at m >= 64 -- the property
that makes the mitigation value-neutral).  After a numpy upgrade a clean
``trans`` run under libgmalloc means the upstream bug is fixed and
``contiguous_transpose`` can be retired.
"""
from __future__ import annotations

import sys

import numpy as np

rng = np.random.default_rng(0)
d = rng.random((16, 3))          # C-contiguous (16, 3): the disk orientation
sub = sys.argv[1] if len(sys.argv) > 1 else "trans"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

if sub == "contig":
    b = np.ascontiguousarray(d.T)
    for i in range(N):
        m = int(rng.integers(64, 24000))
        a = rng.random((m, 3))
        np.clip(a @ b, -1.0, 1.0)
    print(f"contig: {N} iters OK (clean kernel)", flush=True)
elif sub == "trans":
    for i in range(N):
        m = int(rng.integers(64, 24000))
        a = rng.random((m, 3))
        np.clip(a @ d.T, -1.0, 1.0)
    print(f"trans: {N} iters OK (overread did not fault on this run)",
          flush=True)
else:
    raise SystemExit(f"unknown subtest {sub!r} (use trans|contig)")
