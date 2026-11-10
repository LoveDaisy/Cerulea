"""Generate the in-repo Mie anchor arrays for task-cold-optics (plan D3).

The batch-vectorised Mie kernels (``mie_coefficients_batch`` and the batched
moment / phase-table kernels) are verified against arrays pinned in
``tests/data/mie_anchor.npz``.  Two provenance classes live in that file:

* kernel level (``coef_*``) -- a representative (x, m) grid through the
  scalar ``mie_coefficients`` (bitwise class; the scalar kernel is unchanged
  by the task, so these stay scalar-path truth across regenerations);
* aerosol level -- the clear/dusty production surfaces through ``presets``
  (rc_ext/rc_sca/asym bitwise class; delta_m beta/f, phase-table and single
  radiance tolerance class under the task's D2 declared-tolerance route).
  These are generated through the *current* production path, so after the
  batch-vectorisation each regeneration re-pins them to the batched tree;
  the scalar equivalence at re-pin time must be evidenced in the recorded
  reason (dev period: ref_checkpoint 24/24).

Regenerate ONLY with an explicit recorded reason in the task progress log --
never to make a failing comparison pass (same rule as the golden fixtures).

Run from the repository root:

    uv run python tools/make_mie_anchor_arrays.py

Writes ``tests/data/mie_anchor.npz`` plus ``tests/data/mie_anchor.meta.json``
(generating tree, numpy version, UTC+8 timestamp, provenance class split and
regeneration rule) so the provenance of the Git-tracked arrays stays
retrievable in-repo and every regeneration produces the schema-complete
sidecar.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets  # noqa: E402
from skycolor.aerosol import DustAerosol, HinterlandAerosol  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402
from skycolor.mie import mie_coefficients  # noqa: E402
from skycolor.rt.single import sky_radiance as single  # noqa: E402

OUT = ROOT / "tests" / "data" / "mie_anchor.npz"

# representative grid: x -> 0 boundary, clear/rural max x, dusty fine/coarse
# mode extremes, a synthetic 500 boundary, absorbing and dispersive indices
X_GRID = np.array([1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 8.4, 51.5, 167.0, 352.1, 500.0])
M_GRID = np.array([
    1.53 + 0.006j,    # rural / water-soluble
    1.55 + 0.001j,    # mineral dust near the visible minimum
    1.324 + 3200.0 / 550.0**2 + 1e-8j,   # liquid water @ 550 nm
    1.75 + 0.45j,     # strongly absorbing soot
    1.43 + 1e-7j,     # stratospheric sulfate
])

SUN_ZENITH_DEG = 80.0
VIEW_ZENITH_DEG = np.array([0.0, 20.0, 45.0, 70.0])
VIEW_AZIMUTH_DEG = np.array([0.0, 90.0, 180.0, 270.0])
N_MOMENTS_DIRECT = 36
TABLE_SAMPLE_WL = (380.0, 550.0, 780.0)
#: wavelengths whose phase-table scalars are pinned.  The full-grid capture
#: lives in the task scratchpad ref_capture (dev-time); the in-repo anchor
#: keeps a spanning subset so the default-collection test stays affordable
#: (one dusty table build costs ~0.4 s serial and inflates ~4x under
#: 12-worker memory-bandwidth contention).
TABLE_SCALAR_WL = (380.0, 410.0, 440.0, 470.0, 505.0, 550.0, 620.0, 700.0,
                   780.0)


def kernel_level() -> dict:
    out = {}
    xs = np.concatenate([np.full(M_GRID.size, xv) for xv in X_GRID])
    ms = np.concatenate([M_GRID for _ in X_GRID])
    a = np.zeros((xs.size, 1), dtype=complex)
    b = np.zeros((xs.size, 1), dtype=complex)
    qext = np.zeros(xs.size)
    qsca = np.zeros(xs.size)
    nmax = np.zeros(xs.size, dtype=int)
    for j, (xv, mv) in enumerate(zip(xs, ms)):
        c = mie_coefficients(float(xv), complex(mv))
        a = np.pad(a, ((0, 0), (0, max(0, c.nmax + 1 - a.shape[1]))))
        b = np.pad(b, ((0, 0), (0, max(0, c.nmax + 1 - b.shape[1]))))
        a[j, : c.nmax + 1] = c.a[: c.nmax + 1]
        b[j, : c.nmax + 1] = c.b[: c.nmax + 1]
        qext[j] = c.qext
        qsca[j] = c.qsca
        nmax[j] = c.nmax
    out["coef_x"] = xs
    out["coef_m"] = ms
    out["coef_a"] = a
    out["coef_b"] = b
    out["coef_qext"] = qext
    out["coef_qsca"] = qsca
    out["coef_nmax"] = nmax
    return out


def _preset_level(name: str, factory) -> dict:
    wl = WL_COARSE
    atm = factory(wl)
    aer = atm.aerosols[0].aerosol
    out = {}
    out[f"{name}_single"] = single(atm, SUN_ZENITH_DEG, VIEW_ZENITH_DEG,
                                   VIEW_AZIMUTH_DEG)
    res = aer.delta_m(wl, 32)
    out[f"{name}_delta_m_beta"] = res.beta
    out[f"{name}_delta_m_f"] = res.f
    ext, sca = aer.relative_coefficients(wl)
    out[f"{name}_rc_ext"] = ext
    out[f"{name}_rc_sca"] = sca
    out[f"{name}_asym"] = aer.asymmetry(wl)
    out[f"{name}_beta_direct"] = aer.legendre_moments(wl, N_MOMENTS_DIRECT)
    wl_sub = np.array(TABLE_SCALAR_WL)
    peak = np.empty(wl_sub.size)
    norm = np.empty(wl_sub.size)
    for i, w in enumerate(wl_sub):
        t = aer.phase_table(w)
        peak[i] = t.p_peak
        norm[i] = t.normalisation()
    out[f"{name}_p_peak_wl"] = wl_sub
    out[f"{name}_p_peak"] = peak
    out[f"{name}_p_norm"] = norm
    for w in TABLE_SAMPLE_WL:
        out[f"{name}_table_p{int(w)}"] = aer.phase_table(w).p
    return out


def main() -> None:
    data = kernel_level()
    data.update(_preset_level("clear", presets.clear))
    data.update(_preset_level("dusty", presets.dusty))
    # sanity: the nodes used by the presets, for the manifest
    data["_meta_nodes_clear"] = np.array([len(HinterlandAerosol()._nodes)])
    data["_meta_nodes_dusty"] = np.array([len(DustAerosol()._nodes)])
    np.savez(OUT, **data)
    tree = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    meta = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))
                                     ).isoformat(timespec="seconds"),
        "git_head_at_generation": tree,
        "numpy": np.__version__,
        "provenance": (
            "kernel-level coef_* arrays are scalar-path truth (the scalar "
            "kernel is unchanged); preset-level arrays are the production "
            "path's output at git_head_at_generation -- their scalar "
            "equivalence at re-pin time must be evidenced by the recorded "
            "reason for this regeneration in the task progress log"
        ),
        "regeneration_rule": (
            "regenerate only with an explicit recorded reason in the task "
            "progress log -- never to make a failing comparison pass "
            "(same rule as the golden fixtures)"
        ),
    }
    (ROOT / "tests" / "data" / "mie_anchor.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(data)} arrays)")
    print(f"tree: {tree}")
    print(f"numpy: {np.__version__}")


if __name__ == "__main__":
    main()
