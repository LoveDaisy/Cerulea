"""Aureole attribution probe -- task-aureole-delta-fit Step 1 (plan gate G1).

Runs the pre-registered configurations of plan §4 Step 1 against the check3
h=45 deg scene.  The scene construction is reused from the check3 single
source (``validation.common`` grid/atmosphere helpers + the check3
relative-luminance entry); the probe only varies the SOS representation
settings through the production ``solve()`` request layer.

Process isolation (empirical containment, not a physics choice): repeated
``solve(approximation='spherical-mc')`` calls on a ``_restrict``-subset
atmosphere in ONE process corrupt the heap and SIGSEGV after ~4-8 calls
(evidence: ``mc_repeat_segfault_stress.py`` in this directory -- 4 clean
reps then SEGV at a different site each time; full-spectrum atmosphere 8
reps clean; crash sites are manifestations inside the MC transport flush,
not the source.  The corruption is INTERMITTENT: an identical 30-rep run
later the same day completed clean, so one clean run is not evidence of
absence).  Each MC config therefore runs in a fresh worker
subprocess performing at most two transports (wall-clock pass +
instrumented pass).  The root cause is a production finding, out of this
task's scope; it is reported in the task records for an owner-level
spin-off.

Moment-series terminus (measured, see ``moment_tail`` diagnostics): the
Mie amplitude S(mu) is a finite sum of polynomials of degree <= nmax, so
|S|^2 is a polynomial of degree <= 2*nmax and the
Legendre moment series of this mixture terminates EXACTLY at l=30
(nmax <= 15 for the rural mode set); beta_l = 0 for l >= 31.  The delta-M
renormalisation is therefore the identity here (f = beta_32 = 0), the P1
"128-moment" configuration lifts no content (its +0.07% share is the
n_mu 32->64 quadrature effect), and T2 is degenerate in N (identical at
N=32 and N=128 by construction).  For large-x members (dust, clouds)
nmax is large enough that the 32-moment truncation does lose forward
energy and delta-M is load-bearing -- the moment-series terminus moves
with particle size.  This is recorded as part of the stop-loss
attribution handed to the owner.

Configs
-------
B0   default 32-moment delta-M SOS on the full check3 grid.  Mechanical
     anchor: the h=45deg "horizon sun-side / zenith" dev must reproduce the
     value recorded in ``validation/results.json`` within 0.1 pp (plan Step 1;
     same-link same-scene deterministic SOS).  B0 failing aborts the probe
     before P1/P2/P3 run (and writes ``probe_results_anchorfail.json``,
     never overwriting the committed evidence).
B0c  the same B0 request issued through ``solve()`` on the probe's meridian
     cut views, cross-checked against the full-grid map rows (tight numeric
     tolerance, not bitwise: BLAS shape-dependent reduction order per
     learnings) -- mechanically proves the P1/P2 manipulation differs from
     B0 only by the SOS settings.  Its spectral cut (``b0c_rad`` in the
     npz) is the B0 spectral angular profile (same-link verified).
P1   ``SOSConfig(n_mu=64, n_moments=128)``  (resolution up, nested f down;
     with the moment terminus at l=30 this reduces to the n_mu effect)
P2   ``SOSConfig(delta_m=False)``           (raw moments, no renormalisation)
P3   ``approximation='spherical-mc'``       (exact phase sampling + spherical
     transport; fixed seed, 14-wl stride-30 subset of ``WL_COARSE``; MC
     stderr propagated to the ratio for the 3-sigma discipline; worker-
     isolated)
T2   aureole-band residual of the truncated Legendre series vs the exact
     mixture phase function (reference data; consumed by G1 only where the
     plan mandates it, and only with a non-zero moment tail -- with the
     terminus at l=30 it is degenerate in N)
T1   ``SOSConfig(n_mu=128, n_moments=256)`` single-point trend (only on the
     plan's ambiguous-cell trigger)
CTL  full-spectrum low-photon MC control run (only on the plan's 0.05-band
     trigger; supplies the spectral-sampling systematic upper bound;
     worker-isolated)

Meridian cut (>=13 views covering gamma in [1, 90] deg per plan): the check3
grid columns at az=0 (sun azimuth) and az=180 (anti-solar) plus an
aureole-resolved supplement at gamma = 1, 1.5, 2, 3, 4, 6, 8 deg on both
sides of the sun along az=0.  The sun-side horizon (z=90, az=0) and the
zenith (z=0, az=0) -- the two cells of the check3 ratio -- are cut members.
Both records the plan requires are kept per config: the photopic profile
(``*_lum``) and the spectral angular profile (``*_rad``, n_view x n_wl;
``p3`` on its 14-wl subset).

Cost protocol (plan Step 1, r2-Suggestion2 / r3-Minor2): two passes per
config -- pass 1 measures wall clock without instrumentation (the cost
criterion is the *relative wall-clock ratio*), pass 2 measures the
``tracemalloc`` peak (memory is auxiliary/qualitative only; learnings:
``ru_maxrss`` is not a mechanical ruler on macOS).

T2 formula (pinned here per the plan's implement-period delegation; the
denominator is pinned by plan review new-r1-Minor3): on the layer-0 mixture
phase function at 550 nm,

    denominator = int_{gamma in [1,10] deg} p_exact(mu) dmu
    numerator   = int_{gamma in [1,10] deg} |p_N(mu) - p_exact(mu)| dmu
    T2          = numerator / denominator

with p_exact the true mixture phase (the Rayleigh moment series is finite,
hence exact; the aerosol phase is the production exact evaluator
``aerosol.phase``) and p_N the raw truncated Legendre series at N moments
(N=128 = P1's truncation is the gate value; N=32 recorded as reference).
The L1 residual matches the denominator's "integral energy" semantics; the
raw series (not delta-M) is measured because delta-M is itself a candidate
remedy, not the convergence limit being tested.

P3 systematic rule (plan G1; review r2-Minor1 one-line recipe): whenever the
3-sigma-counted P3 share is the deciding quantity of a gate comparison
(1/3 or 1/4) and its distance to that gate is <= 0.05, a full-spectrum
low-photon MC control run is issued (at most once) and

    upper_bound_share = (|ratio_ctl - ratio_P3| + 3*sigma_combined) / (B0-CIE)
    corrected_share   = share_P3 - upper_bound_share

with ``sigma_combined = sqrt(sigma_ctl^2 + sigma_P3^2)``; the corrected
share participates in that comparison.  The band check is applied
symmetrically at every gate P3 adjudicates (the branch-2 entry test at 1/3
and the 1/4 boundary classification whenever P3 supplies the argmax).  If
the measured |ratio difference| alone exceeds 0.05 in share units, the run
flags the 0.05 band as miscalibrated (review new-r2-Sug1 adaptive clause)
for a progress-recorded amendment request.

G1 gate (plan Step 1, complete partition):
  branch 1  share_P1 >= 1/3 or share_P2 >= 1/3      -> axis A significant,
             Step 2 (P3 does not adjudicate; no control run)
  branch 2  both < 1/3 and counted share_P3 >= 1/3  -> ambiguous cell:
             T2 > 0.20 -> Step 2 (C3-leaning); else T1: |ratio_T1-ratio_P1|
             >= 0.20*(B0-P1) -> Step 2, else stop-loss
  branch 3  R in [1/4, 1/3)                          -> Step 2, weak evidence
  branch 4  R < 1/4                                  -> stop-loss
with R = max of the counted shares (P3 counted only when share > 3 sigma;
an uncounted P3 participates in no comparison).

Run (driver):
  PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_aureole.py
Outputs (``probe_results.json`` / ``probe_costs.csv`` / ``probe_profiles.npz``)
land in ``--out-dir`` (default: this directory).  A rerun OVERWRITES the
committed 2026-11-11 evidence in the default location -- pass
``--out-dir <dir>`` for side-by-side runs, or restore with ``git checkout``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

# NOTE: ``scratchpad/`` is a symlink to the parent checkout in worktrees, so
# ``Path(__file__).resolve()`` would silently resolve through it and put the
# PARENT repo first on sys.path.  ``absolute()`` anchors on the cwd without
# following the symlink.
REPO = Path(__file__).absolute().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation import common as C  # noqa: E402
from validation.check3_cie_luminance import (  # noqa: E402
    model_relative_luminance,
    N_AZ,
    N_ZEN,
)
from skycolor.rt.solver import (  # noqa: E402
    HYBRID_SOS,
    SPHERICAL_MC,
    MCConfig,
    SOSConfig,
    solve,
    sky_radiance_total,
)

OUT = Path(__file__).absolute().parent
SUN_ZEN = 45.0                    # check3 h=45 deg reference elevation
WL550_IDX = 34                    # WL_COARSE[34] == 550.0
SUBSET_STRIDE = 6                 # WL_COARSE[::6] -> 14 wavelengths (~13 per plan)
NEAR_GAMMA = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)   # aureole supplement views
T2_BAND = (1.0, 10.0)             # aureole band, deg
T2_THRESH = 0.20                  # plan pre-registered (review-revisable)
GATES = (1.0 / 3.0, 0.25)         # plan pre-registered
BAND = 0.05                       # plan pre-registered P3 systematic band
MOMENT_TAIL_FROM = 33             # diagnostic: max|beta_l| for l >= 33


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# scene helpers (check3 single construction source)
# ---------------------------------------------------------------------------
def build_cut():
    """Meridian-plane cut: >=13 views covering gamma in [1, 90] deg.

    Order: az=0 grid column (37 views; contains the zenith at index 0 and the
    sun-side horizon at index 36), aureole supplement (14 views), az=180 grid
    column (37 views).
    """
    z_grid, a_grid = C.sky_grid(N_ZEN, N_AZ)
    zv = z_grid[:, 0]                       # 0..90, step 2.5
    assert a_grid[0, 0] == 0.0 and a_grid[0, 36] == 180.0
    near_z = np.concatenate([45.0 - np.array(NEAR_GAMMA),
                             45.0 + np.array(NEAR_GAMMA)])
    cut_z = np.concatenate([zv, near_z, zv])
    cut_a = np.concatenate([np.zeros(37), np.zeros(near_z.size),
                            np.full(37, 180.0)])
    idx = dict(az0=slice(0, 37), near=slice(37, 51), az180=slice(51, 88),
               zenith=0, horizon=36)
    return cut_z, cut_a, zv, idx


def lum_of(rad: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Photopic luminance [cd/m2] of (n_view, n_wl) spectral radiance."""
    return C.luminance(np.asarray(rad, dtype=float), wl)


def ratio_of(lum: np.ndarray, idx) -> float:
    """Sun-side horizon / zenith ratio from a cut luminance vector."""
    return float(lum[idx["horizon"]] / lum[idx["zenith"]])


def trapezoid_weights(wl: np.ndarray) -> np.ndarray:
    w = np.full(wl.size, float(np.mean(np.diff(wl))))
    w[0] = w[-1] = 0.5 * (wl[1] - wl[0])
    return w


def lum_stderr(stderr: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """MC stderr propagated to photopic luminance per view (1 sigma).

    Independent-wavelength approximation (anchors are independent MC runs;
    spectrally interpolated samples correlate weakly -- recorded as an
    approximation, the 3-sigma discipline uses this as the scale).
    """
    w = trapezoid_weights(wl)
    return C.K_M * np.sqrt(np.sum((w[None, :] * C.photopic(wl) * stderr) ** 2,
                                  axis=1))


def sigma_ratio_of(stderr: np.ndarray, wl: np.ndarray, lum: np.ndarray,
                   idx) -> float:
    """MC stderr propagated to the horizon/zenith ratio (1 sigma)."""
    lum_std = lum_stderr(stderr, wl)
    h, z = idx["horizon"], idx["zenith"]
    rel2 = (lum_std[h] / lum[h]) ** 2 + (lum_std[z] / lum[z]) ** 2
    return float(ratio_of(lum, idx) * np.sqrt(rel2))


def timed(fn):
    """Pass 1: wall clock only (no instrumentation)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def instrumented(fn):
    """Pass 2: tracemalloc peak (auxiliary memory reference)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, wall, peak


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def energy_identity(atmo, n_mu: int, n_moments: int, delta_m: bool) -> float:
    """max |int p dmu - 2| over layers/wavelengths for a config's moment set.

    The Gauss rule integrates the energy identity exactly for moments below
    ``2 * n_mu``; this checks it numerically on the moment set each config
    actually feeds the kernel.
    """
    from scipy.special import assoc_legendre_p_all

    n = max(2, n_mu) + (n_mu % 2)
    x, w = np.polynomial.legendre.leggauss(n)
    beta = (atmo.delta_m_moments(n_moments)[0] if delta_m
            else atmo.phase_moments(n_moments))       # (nz, N, nwl)
    l_max = beta.shape[1] - 1
    p = assoc_legendre_p_all(l_max, 0, x)[0, :, 0, :]        # (L+1, n_mu)
    scaled = beta * (2.0 * np.arange(l_max + 1) + 1.0)[None, :, None]
    integral = np.einsum("i,klw,li->kw", w, scaled, p)
    return float(np.max(np.abs(integral - 2.0)))


def moment_tail(atmo, n_moments: int, layer: int = 0) -> dict:
    """Moment-series terminus diagnostic (review r1 Major 2).

    The Mie amplitude is a polynomial of degree <= 2*nmax, so the raw
    Legendre moments vanish EXACTLY beyond l = 2*nmax; for this mixture the
    terminus is l=30.  Records how many moments carry content and the tail
    magnitude above ``MOMENT_TAIL_FROM`` -- the mechanical proof of whether a
    higher-moment configuration lifts anything at all.
    """
    beta = atmo.phase_moments(n_moments)[layer, :, WL550_IDX]
    nonzero = int(np.count_nonzero(beta))
    tail_slice = beta[MOMENT_TAIL_FROM - 1:]
    tail = float(np.max(np.abs(tail_slice))) if tail_slice.size else 0.0
    live = np.flatnonzero(beta)
    terminus = int(live[-1]) if live.size else -1
    return {"n_moments": n_moments, "nonzero": nonzero,
            "terminus_l": terminus,
            f"max_abs_beta_l{MOMENT_TAIL_FROM - 1}_plus": tail,
            "beta_last_nonzero": float(beta[terminus]) if live.size else 0.0}


def aod_invariants(atmo) -> dict:
    """Layer AOD550 values + the coefficient-integrated column at 550 nm.

    The coefficients are per metre and the heights in km, so the column
    integral is ``trapz(ext, z_km * 1e3)``; the assertion ties it to the
    layer AOD (review r1 Minor: the old km-integral read 1000x small and
    silently failed its obvious purpose).
    """
    lays = [(lay.aerosol.name, float(lay.aod550)) for lay in atmo.aerosols]
    z_fine = np.linspace(0.0, float(atmo.z_bounds[-1]), 4001)
    ext550 = atmo.aerosol_coefficients(z_fine)[0][:, WL550_IDX]
    column = float(np.trapezoid(ext550, z_fine * 1e3))   # metres: column AOD
    layer_aod = float(sum(a for _n, a in lays))
    ok = abs(column - layer_aod) <= 1e-3 * max(layer_aod, 1e-300)
    assert ok, (f"AOD550 column invariant failed: column={column} vs "
                f"layer sum {layer_aod}")
    return {"layers": lays, "layer_aod_sum": layer_aod,
            "column_550": column, "column_matches_layers": bool(ok)}


# ---------------------------------------------------------------------------
# T2: aureole-band residual of the truncated series
# ---------------------------------------------------------------------------
def t2_aureole_residual(atmo, n_values=(32, 128), layer=0):
    """T2 per the pinned formula (see module docstring), per truncation N."""
    from scipy.special import assoc_legendre_p_all

    mu_lo, mu_hi = np.cos(np.radians(T2_BAND[1])), np.cos(np.radians(T2_BAND[0]))
    x, w = np.polynomial.legendre.leggauss(4096)
    mu_band = 0.5 * (mu_hi - mu_lo) * x + 0.5 * (mu_hi + mu_lo)
    w_band = 0.5 * (mu_hi - mu_lo) * w

    props = atmo.layer_properties()
    b_ray = float(props["beta_ray"][layer, WL550_IDX])
    b_as = float(props["beta_as"][layer, WL550_IDX])
    b_sca = float(props["beta_sca"][layer, WL550_IDX])

    # exact mixture phase: the Rayleigh moment series is finite (exact)
    ray_b = np.asarray(atmo._ray_phase, dtype=float)
    p_ray = np.zeros_like(mu_band)
    for l in range(ray_b.size):
        p_ray += (2 * l + 1) * ray_b[l] * np.polynomial.legendre.legval(
            mu_band, [0.0] * l + [1.0])
    ae = atmo.aerosols[0].aerosol
    p_ae = np.asarray(ae.phase(mu_band, 550.0), dtype=float).ravel()
    p_exact = (b_ray * p_ray + b_as * p_ae) / b_sca

    out = {}
    for n_mv in n_values:
        beta = atmo.phase_moments(n_mv)[layer, :, WL550_IDX]   # raw moments
        p_leg = assoc_legendre_p_all(n_mv - 1, 0, mu_band)[0, :, 0, :]
        p_n = np.einsum("l,li->i", (2.0 * np.arange(n_mv) + 1.0) * beta, p_leg)
        denom = float(np.sum(w_band * p_exact))
        numer = float(np.sum(w_band * np.abs(p_n - p_exact)))
        out[n_mv] = {"denominator": denom, "numerator": numer,
                     "t2": numer / denom, "layer": layer, "wl_nm": 550.0}
    return out


# ---------------------------------------------------------------------------
# MC worker (fresh process per MC config)
# ---------------------------------------------------------------------------
def run_worker(name: str, args) -> int:
    """Run ONE MC config in this fresh process and write a fragment file.

    Each worker performs at most two MC transports (pass 1 + instrumented
    pass 2) -- far below the ~4-8-call corruption threshold observed in
    ``mc_repeat_segfault_stress.py``.  The fragment lands in the driver's
    out-dir (``--fragment-dir``) and is unlinked by the driver after read.
    """
    atmo = C.atmosphere("clear", C.WL_COARSE)
    cut_z, cut_a, zv, idx = build_cut()
    if name == "p3":
        sub = np.arange(0, C.WL_COARSE.size, SUBSET_STRIDE)
        atmo_run = atmo._restrict(sub)
        wl_run = C.WL_COARSE[sub]
    elif name == "ctl":
        atmo_run, wl_run = atmo, C.WL_COARSE
    else:
        raise ValueError(f"worker does not handle config {name!r}")

    def fn():
        return solve(atmo_run, SUN_ZEN, cut_z, cut_a, sun_azimuth_deg=0.0,
                     approximation=SPHERICAL_MC,
                     mc_config=MCConfig(quality=args.mc_quality, seed=args.seed))

    log(f"--- {name}: worker pass 1 (wall clock) ---")
    res, wall = timed(fn)
    log(f"--- {name}: worker pass 2 (tracemalloc) ---")
    _res2, wall_i, peak = instrumented(fn)
    rad = np.asarray(res.radiance, dtype=float)
    l = lum_of(rad, wl_run)
    ratio = ratio_of(l, idx)
    sig = sigma_ratio_of(np.asarray(res.stderr, dtype=float), wl_run, l, idx)
    info = res.diagnostics
    entry = {"wall_s_clean": wall, "wall_s_instrumented": wall_i,
             "mem_peak_mb": peak / 1e6, "ratio": ratio, "sigma_ratio": sig,
             "wl_subset": [float(w) for w in wl_run],
             "mc_quality": args.mc_quality, "seed": args.seed,
             "n_zen_run": info.get("n_zen_run"), "n_az_run": info.get("n_az_run"),
             "n_anchor_run": info.get("n_anchor_run"),
             "n_photons_run": info.get("n_photons_run"),
             "bandwidth": info.get("bandwidth")}
    # concentration-flavoured diagnostic: per-cut-view relative stderr of the
    # photopic luminance (the honest "interval + concentration" pair; the
    # engine's stderr also carries the smoother's 3% bias floor)
    lum_std = lum_stderr(np.asarray(res.stderr, dtype=float), wl_run)
    rel = lum_std / np.maximum(l, 1e-300)
    entry["rel_stderr_max"] = float(np.max(rel))
    entry["rel_stderr_median"] = float(np.median(rel))
    log(f"{name}: ratio={ratio:.4f} +/- {sig:.4f} (rel stderr max "
        f"{entry['rel_stderr_max']:.3f} / median {entry['rel_stderr_median']:.3f})")
    frag = {"config": name, "entry": entry, "lum": l.tolist(),
            "rad": rad.tolist(), "wl": wl_run.tolist()}
    (Path(args.fragment_dir) / f"probe_fragment_{name}.json").write_text(
        json.dumps(frag, indent=2, default=str))
    return 0


def spawn_worker(name: str, args, attempts: int = 2, mc_quality=None):
    """Run one MC config in a fresh subprocess; return (entry, lum, rad).

    A worker crash (the residual heap-corruption risk) consumes one attempt;
    a fresh process restarts the corruption budget from zero.  The CTL
    control run pins its own budget (``mc_quality="low"``, the plan's
    "full-spectrum low-photon" semantics) independently of P3's preset.
    """
    quality = args.mc_quality if mc_quality is None else mc_quality
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).absolute()), "--worker", name,
             "--mc-quality", quality, "--seed", str(args.seed),
             "--fragment-dir", args.fragment_dir],
            env={**os.environ, "PYTHONPATH": str(REPO)},
            capture_output=True, text=True)
        frag_path = Path(args.fragment_dir) / f"probe_fragment_{name}.json"
        if proc.returncode == 0 and frag_path.exists():
            frag = json.loads(frag_path.read_text())
            frag_path.unlink()          # no stray debris in the evidence dir
            log(f"worker {name}: {frag['entry']['ratio']:.4f} +/- "
                f"{frag['entry']['sigma_ratio']:.4f}")
            return (frag["entry"], np.asarray(frag["lum"], dtype=float),
                    np.asarray(frag["rad"], dtype=float))
        log(f"worker {name} attempt {attempt}/{attempts} FAILED "
            f"(exit {proc.returncode}); stderr tail:\n{proc.stderr[-600:]}")
    raise RuntimeError(f"worker {name} failed after {attempts} attempts")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="b0,b0c,p1,p2,p3",
                    help="comma list among b0,b0c,p1,p2,p3,t1,ctl")
    ap.add_argument("--mc-quality", default="low",
                    help="MC QUALITY preset for P3/CTL (D4: existing presets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", default=None,
                    help="internal: run one MC config in this fresh process")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: this script's directory; "
                         "a rerun there overwrites the committed evidence)")
    ap.add_argument("--fragment-dir", default=None,
                    help="worker fragment directory (default: out-dir; "
                         "fragments are unlinked after read)")
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir).absolute() if args.out_dir else OUT
    args.fragment_dir = (Path(args.fragment_dir).absolute()
                         if args.fragment_dir else str(args.out_dir))
    if args.worker:
        return run_worker(args.worker, args)
    out_dir = args.out_dir
    wanted = [s.strip().lower() for s in args.configs.split(",") if s.strip()]
    # B0 anchor is the precondition of everything downstream (plan Step 1)
    if any(k in wanted for k in ("p1", "p2", "p3", "t1", "ctl")) \
            and "b0" not in wanted:
        wanted.insert(0, "b0")

    # -- D6 mechanical recheck: same link -----------------------------------
    assert C.MULTIPLE is True, "SKYCOLOR_MULTIPLE must stay at its default"
    assert C.sky_radiance is sky_radiance_total, "D6 link assertion failed"

    meta = dict(seed=args.seed, mc_quality=args.mc_quality,
                wl_subset_stride=SUBSET_STRIDE, near_gamma=list(NEAR_GAMMA),
                t2_band=list(T2_BAND), t2_threshold=T2_THRESH,
                band=BAND, gates=list(GATES),
                moment_tail_from=MOMENT_TAIL_FROM,
                worker_isolation="MC configs run in fresh subprocesses "
                                 "(heap-corruption containment, see docstring)")
    results = {"meta": meta, "configs": {}, "invariants": {}, "g1": None,
               "t2": None}

    # -- mechanical anchor single source ------------------------------------
    rj = json.loads((REPO / "validation" / "results.json").read_text())
    anchor_metric = next(m for c in rj["checks"] if c["key"] == "check3"
                         for m in c["metrics"]
                         if m["name"] == "horizon sun-side / zenith, h=45deg")
    anchor_dev = float(anchor_metric["dev"])
    results["anchor"] = {"results_json_dev": anchor_dev,
                         "results_json_model": float(anchor_metric["model"]),
                         "results_json_reference":
                             float(anchor_metric["reference"])}
    log(f"anchor from results.json: dev={anchor_dev:.6f}%")

    atmo = C.atmosphere("clear", C.WL_COARSE)
    results["invariants"]["aod_before"] = aod_invariants(atmo)
    cut_z, cut_a, zv, idx = build_cut()
    gamma_cut = C.angular_distance(cut_z, cut_a, SUN_ZEN, 0.0)
    cie_cut = C.cie110_relative_luminance(cut_z[:, None], cut_a[:, None],
                                          SUN_ZEN)[:, 0]
    covered = gamma_cut[(gamma_cut >= 1.0) & (gamma_cut <= 90.0)]
    assert covered.size >= 13, f"cut covers only {covered.size} views in [1,90]"
    results["cut"] = {"n_views": int(cut_z.size),
                      "n_views_gamma_1_90": int(covered.size)}
    log(f"cut: {cut_z.size} views, {covered.size} of them in gamma [1,90] deg")

    ratio_cie = float(C.cie110_relative_luminance(
        np.array([[90.0]]), np.array([[0.0]]), SUN_ZEN)[0, 0])
    results["ratio_cie"] = ratio_cie
    log(f"CIE sun-side horizon/zenith (h=45): {ratio_cie:.6f}")

    lum = {}          # config -> cut luminance vector (npz material)
    rad_cut = {}      # config -> cut spectral radiance (npz material)
    b0_map = {}       # full-grid B0 map (npz material)
    control = {"corrected": None}   # 0.05-band control-run cache (run once)

    def run_config(name, fn):
        log(f"--- {name}: pass 1 (wall clock) ---")
        out, wall = timed(fn)
        log(f"--- {name}: pass 2 (tracemalloc) ---")
        _out2, wall_i, peak = instrumented(fn)
        entry = {"wall_s_clean": wall, "wall_s_instrumented": wall_i,
                 "mem_peak_mb": peak / 1e6}
        results["configs"][name] = entry
        log(f"{name}: wall={wall:.2f}s (instrumented {wall_i:.2f}s, "
            f"peak {peak / 1e6:.1f} MB)")
        return out, entry

    # -- B0: anchor on the full check3 grid ---------------------------------
    if "b0" in wanted:
        (z_f, a_f, rel_f, lz_f), entry = run_config(
            "b0", lambda: model_relative_luminance(SUN_ZEN))
        cie_f = C.cie110_relative_luminance(z_f, a_f, SUN_ZEN)
        ratio_b0 = float(rel_f[-1, 0])
        dev_b0 = 100.0 * (ratio_b0 / float(cie_f[-1, 0]) - 1.0)
        entry.update(ratio=ratio_b0, dev_vs_cie=dev_b0, lz=float(lz_f),
                     map_shape=list(rel_f.shape))
        ok = abs(dev_b0 - anchor_dev) <= 0.1
        entry["anchor_ok"] = bool(ok)
        results["anchor"]["probe_dev"] = dev_b0
        results["ratio_b0"] = ratio_b0
        b0_map["lum"] = rel_f * lz_f
        b0_map["rel"] = rel_f
        log(f"B0 anchor: dev={dev_b0:.6f}% vs results.json {anchor_dev:.6f}% "
            f"(delta {dev_b0 - anchor_dev:+.2e} pp) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            # never overwrite the committed evidence with failed-run data
            (out_dir / "probe_results_anchorfail.json").write_text(
                json.dumps(results, indent=2, default=str))
            log("B0 anchor failed -- probe stops before P1/P2/P3 (plan Step 1); "
                "results written to probe_results_anchorfail.json")
            return 2

    # -- deterministic SOS on the cut ---------------------------------------
    def sos_cut(name, sos_cfg):
        res, entry = run_config(name, lambda: solve(
            atmo, SUN_ZEN, cut_z, cut_a, sun_azimuth_deg=0.0,
            approximation=HYBRID_SOS, sos_config=sos_cfg))
        rad = np.asarray(res.radiance, dtype=float)
        l = lum_of(rad, C.WL_COARSE)
        lum[name] = l
        rad_cut[name] = rad
        ratio = ratio_of(l, idx)
        eff = res.effective["sos"]
        entry.update(ratio=ratio,
                     dev_vs_cie=100.0 * (ratio / ratio_cie - 1.0),
                     n_mu_eff=int(eff["n_mu"]),
                     n_moments_eff=int(eff["n_moments"]),
                     n_moments_optics=(None if eff["n_moments_optics"] is None
                                       else int(eff["n_moments_optics"])),
                     delta_m=bool(eff["delta_m"]), orders=int(eff["orders_run"]),
                     converged=bool(eff["converged"]))
        return res

    if "b0c" in wanted:
        sos_cut("b0c", SOSConfig())
        if "lum" in b0_map:
            map_lum = b0_map["lum"]
            got = np.concatenate([lum["b0c"][:37], lum["b0c"][51:]])
            want = np.concatenate([map_lum[:, 0], map_lum[:, 36]])
            rel = float(np.max(np.abs(got - want)
                               / np.maximum(np.abs(want), 1e-300)))
            results["configs"]["b0c"]["cross_check_max_rel"] = rel
            results["configs"]["b0c"]["cross_check_ok"] = bool(rel <= 1e-12)
            log(f"B0c cross-check vs full-grid rows: max rel {rel:.2e} "
                f"({'OK' if rel <= 1e-12 else 'FAIL'})")
        results["ratio_b0_cut"] = ratio_of(lum["b0c"], idx)

    if "p1" in wanted:
        sos_cut("p1", SOSConfig(n_mu=64, n_moments=128))
    if "p2" in wanted:
        sos_cut("p2", SOSConfig(delta_m=False))
    if "t1" in wanted:
        sos_cut("t1", SOSConfig(n_mu=128, n_moments=256))
        results["invariants"]["energy_identity_t1"] = energy_identity(
            atmo, 128, 256, True)
        log(f"energy identity [t1]: "
            f"{results['invariants']['energy_identity_t1']:.3e}")

    # -- P3: spherical MC in an isolated worker ------------------------------
    if "p3" in wanted:
        entry, l, rad = spawn_worker("p3", args)
        results["configs"]["p3"] = entry
        lum["p3"] = l
        rad_cut["p3"] = rad
        entry["sigma_share"] = (entry["sigma_ratio"]
                                / (results["ratio_b0"] - ratio_cie))
        results["ratio_p3"] = entry["ratio"]
        log(f"P3: ratio={entry['ratio']:.4f} +/- {entry['sigma_ratio']:.4f} "
            f"(sigma share {entry['sigma_share']:.4f}, "
            f"quality={args.mc_quality})")

    # -- invariants ----------------------------------------------------------
    results["invariants"]["aod_after"] = aod_invariants(atmo)
    same = (results["invariants"]["aod_before"] ==
            results["invariants"]["aod_after"])
    results["invariants"]["aod_invariant_ok"] = bool(same)
    log(f"AOD550 invariant before/after identical: {same}")
    for name, n_mu, n_mom, dm in (("b0", 32, 32, True), ("p1", 64, 128, True),
                                  ("p2", 32, 32, False)):
        if name in wanted:
            key = f"energy_identity_{name}"
            results["invariants"][key] = energy_identity(atmo, n_mu, n_mom, dm)
            log(f"energy identity [{name}]: max|int p dmu - 2| = "
                f"{results['invariants'][key]:.3e}")
    # moment-series terminus diagnostics (review r1 Major 2): the proof of
    # whether each higher-moment configuration lifts any content at all
    for name, n in (("b0", 32), ("p1", 128), ("t1", 256)):
        if name in wanted:
            key = f"moment_tail_{name}"
            results["invariants"][key] = moment_tail(atmo, n)
            log(f"moment tail [{name}] (n={n}): {results['invariants'][key]}")

    # -- T2 reference data ----------------------------------------------------
    results["t2"] = t2_aureole_residual(atmo)
    log("T2 (layer-0 mixture, 550nm, band [1,10] deg): "
        + ", ".join(f"N={n}: {v['t2']:.4f}" for n, v in results["t2"].items()))

    # -- G1 ------------------------------------------------------------------
    if all(k in results["configs"] for k in ("b0", "p1", "p2", "p3")):
        g = evaluate_g1(results, args, control, lum, rad_cut, spawn_worker)
        if g.get("t1_trigger"):
            if "t1" not in results["configs"]:
                sos_cut("t1", SOSConfig(n_mu=128, n_moments=256))
                results["invariants"]["energy_identity_t1"] = energy_identity(
                    atmo, 128, 256, True)
                results["invariants"]["moment_tail_t1"] = moment_tail(atmo, 256)
            d1 = abs(results["configs"]["t1"]["ratio"]
                     - results["configs"]["p1"]["ratio"])
            thresh = 0.20 * (results["ratio_b0"]
                             - results["configs"]["p1"]["ratio"])
            g["t1"] = {"delta1": d1, "threshold": thresh,
                       "significant": bool(d1 >= thresh)}
            if d1 >= thresh:
                g["verdict"] = ("ambiguous cell, T2 <= 0.20 -> T1 shows a "
                                "significant trend (Delta1 >= 20% of B0-P1) "
                                "-> Step 2, C3-leaning")
            else:
                g["verdict"] = ("ambiguous cell, T2 <= 0.20 -> T1 plateau "
                                "(Delta1 < 20% of B0-P1) -> axis B dominant, "
                                "STOP-LOSS")
            log(f"T1: ratio={results['configs']['t1']['ratio']:.4f}; "
                f"Delta1={d1:.4f} vs threshold {thresh:.4f}")
        results["g1"] = g
        log(f"G1 branch {g.get('branch')}: {g.get('verdict')}")

    # -- outputs --------------------------------------------------------------
    (out_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    npz = {f"{name}_lum": l for name, l in lum.items()}
    for name, r in rad_cut.items():
        npz[f"{name}_rad"] = r
    npz["cut_z"] = cut_z
    npz["cut_az"] = cut_a
    npz["cut_gamma"] = gamma_cut
    npz["cut_cie_rel"] = cie_cut
    if "lum" in b0_map:
        npz["b0_map_lum"] = b0_map["lum"]
        npz["b0_map_rel"] = b0_map["rel"]
    np.savez_compressed(out_dir / "probe_profiles.npz", **npz)

    with (out_dir / "probe_costs.csv").open("w") as fh:
        fh.write("config,wall_clean_s,wall_instrumented_s,mem_peak_mb\n")
        for name, e in results["configs"].items():
            fh.write(f"{name},{e['wall_s_clean']:.3f},"
                     f"{e['wall_s_instrumented']:.3f},{e['mem_peak_mb']:.1f}\n")

    log("probe complete -> probe_results.json / probe_profiles.npz / "
        "probe_costs.csv")
    return 0


def evaluate_g1(results, args, control, lum, rad_cut, mc_runner):
    """Pre-registered G1 partition (plan Step 1; see module docstring).

    The 0.05-band control run applies symmetrically at every gate P3
    actually adjudicates: the branch-2 entry test at 1/3, and the 1/4
    boundary classification whenever P3 supplies the argmax (a P3-decided
    1/4 comparison decides between gray zone and stop-loss, so the
    uncorrected reading must not classify it -- review r1 Major 1).  The
    control runs at most once per probe (cached in ``control``).

    ``mc_runner(name, args, mc_quality=None) -> (entry, lum, rad)`` runs one
    MC config (the CTL control run) in an isolated worker.
    """
    ratio_b0 = results["ratio_b0"]
    ratio_cie = results["ratio_cie"]
    b0c = ratio_b0 - ratio_cie
    g = {"B0_minus_CIE": b0c, "shares": {}, "notes": []}

    share = {}
    for k in ("p1", "p2", "p3"):
        share[k] = (ratio_b0 - results["configs"][k]["ratio"]) / b0c
    sig3 = results["configs"]["p3"]["sigma_share"]
    share["p3_sigma"] = sig3
    share["p3_counted"] = bool(share["p3"] > 3.0 * sig3)
    g["shares"] = share
    log(f"shares: P1={share['p1']:.4f} P2={share['p2']:.4f} "
        f"P3={share['p3']:.4f} (3sigma={3 * sig3:.4f}, "
        f"counted={share['p3_counted']})")

    def p3_effective(gate):
        """P3 share for one gate comparison, with the 0.05-band control run."""
        if control["corrected"] is not None:
            return control["corrected"], True
        s = share["p3"]
        if abs(s - gate) > BAND:
            return s, False
        log(f"P3 share {s:.4f} within {BAND} of gate {gate:.4f} -> "
            "control run (full spectrum, low photons, isolated worker)")
        entry, l, rad = mc_runner("ctl", args, mc_quality="low")
        results["configs"]["ctl"] = entry
        lum["ctl"] = l
        rad_cut["ctl"] = rad
        sig_p3 = results["configs"]["p3"]["sigma_ratio"]
        sig_comb = float(np.sqrt(entry["sigma_ratio"] ** 2 + sig_p3 ** 2))
        dev_share = abs(entry["ratio"]
                        - results["configs"]["p3"]["ratio"]) / b0c
        upper = dev_share + 3.0 * sig_comb / b0c
        corrected = s - upper
        entry["dev_share"] = dev_share
        entry["sigma_combined_share"] = sig_comb / b0c
        entry["upper_bound_share"] = upper
        entry["corrected_share"] = corrected
        g["control_run"] = {k: entry[k] for k in
                            ("ratio", "sigma_ratio", "dev_share",
                             "sigma_combined_share", "upper_bound_share",
                             "corrected_share", "wall_s_clean")}
        if dev_share > BAND:
            g["notes"].append(
                f"0.05-band miscalibration: measured spectral-sampling "
                f"deviation {dev_share:.4f} > 0.05 in share units; amendment "
                f"request to be recorded in progress.md (review new-r2-Sug1)")
        log(f"control run: ratio={entry['ratio']:.4f}+/-"
            f"{entry['sigma_ratio']:.4f}; upper bound share {upper:.4f}; "
            f"corrected P3 share {corrected:.4f}")
        control["corrected"] = corrected
        return corrected, True

    # branch 1: P1 or P2 >= 1/3 (P3 does not adjudicate this branch)
    if share["p1"] >= GATES[0] or share["p2"] >= GATES[0]:
        g["branch"] = 1
        if share["p3_counted"] and share["p3"] >= GATES[0]:
            g["p3_reference"] = ("P3 counted and also >= 1/3; T2 below is "
                                 "selection reference (plan branch-1 note)")
        g["verdict"] = ("axis A significant -> Step 2 (no control run: P3 "
                        "does not adjudicate this branch)")
        return g

    if not share["p3_counted"]:
        # P3 contributes nothing; R = max(P1, P2), all < 1/3 here
        counted = {"p1": share["p1"], "p2": share["p2"]}
        r = max(counted.values())
        g["R"] = r
        g["branch"] = 3 if r >= GATES[1] else 4
        g["verdict"] = ("P3 not counted (<= 3 sigma); "
                        + ("gray zone -> Step 2 with weak evidence"
                           if g["branch"] == 3 else
                           "R < 1/4 -> stop-loss: neither measured axis "
                           "explains the gap (see attribution)"))
        return g

    # P3 adjudicates: branch-2 entry test at the 1/3 gate
    s3_eff, _ = p3_effective(GATES[0])
    if s3_eff >= GATES[0]:
        g["branch"] = 2
        g["t2_gate_value"] = t2_gate_value(results, g)
        if g["t2_gate_value"]["t2"] > T2_THRESH:
            g["verdict"] = ("ambiguous cell -> axis A unresolved -> "
                            "Step 2, C3-leaning (T2 > 0.20)")
        else:
            g["t1_trigger"] = True
            g["verdict"] = ("ambiguous cell, T2 <= 0.20 -> T1 confirmation "
                            "point required before the exit verdict")
        return g

    # R = max of counted shares.  The 1/4 boundary comparison (gray vs
    # stop-loss): whenever P3 supplies the argmax, P3 adjudicates that
    # comparison and the 0.05-band rule applies to the 1/4 gate too.
    counted = {"p1": share["p1"], "p2": share["p2"], "p3": s3_eff}
    if counted["p3"] == max(counted.values()):
        s3_eff, _ = p3_effective(GATES[1])
        counted["p3"] = s3_eff
        if s3_eff >= GATES[0]:
            # correction only subtracts, so this is unreachable; asserted
            # rather than silently misrouted
            raise AssertionError("corrected P3 share crossed 1/3 upward")
    r = max(counted.values())
    g["R"] = r
    if r >= GATES[1]:
        g["branch"] = 3
        g["verdict"] = ("gray zone [1/4, 1/3) -> Step 2 with weak evidence; "
                        "C2/C4 preferred, C3 only if T2 > 0.20")
        return g
    g["branch"] = 4
    g["verdict"] = ("R < 1/4 -> stop-loss: neither measured axis explains "
                    "the gap (see attribution); evidence attached")
    return g


def t2_gate_value(results, g):
    """T2 gate consumption with the non-zero-tail precondition (r1 Major 2).

    With the moment terminus at l=30 the T2 series is degenerate in N; the
    gate reading is only meaningful when the moment tail above l=32 carries
    content.  A degenerate tail is recorded and the gate value is flagged.
    """
    tail = (results["invariants"].get("moment_tail_p1")
            or results["invariants"].get("moment_tail_b0"))
    # direct predicate on the measured tail magnitude, not a count proxy
    # (an interior zero would break the count assumption; review r2 Sug3)
    degenerate = tail is not None and tail[f"max_abs_beta_l{MOMENT_TAIL_FROM - 1}_plus"] == 0.0
    t2 = results["t2"][128]["t2"]
    value = {"t2": t2, "tail_degenerate": bool(degenerate)}
    if degenerate:
        value["note"] = ("moment series terminates below l=33: T2 is "
                         "degenerate in N and cannot tie-break axis A "
                         "unresolvedness; escalation must rely on T1")
        g["notes"].append(value["note"])
        log("T2 gate: DEGENERATE (moment tail zero) -- recorded")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
