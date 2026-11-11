# Aureole attribution probe (task-aureole-delta-fit, 2026-11-11)

Pre-registered attribution probe for the check3 h=45deg sun-side
horizon/zenith excess (model 8.815 vs CIE 110-1994 3.652, dev +141.406%,
`validation/results.json`).  Promoted from the task scratchpad per the plan's
evidence-persistence rule (review new-r1-Minor5): every number quoted in the
check3 note and the stop-loss record reproduces from this directory; the
committed JSON/NPZ/CSV are the 2026-11-11 run; the physical chain is
unchanged since git d0b956b (skycolor has zero diff throughout), and the
committed artifacts were produced by the HEAD-version probe script (the
schema includes the review-r1 `moment_tail_*` diagnostics and the spectral
`*_rad` arrays -- older probe revisions do not emit them).

## Question and answer

The ruler audit (task-ruler-audit, `validation/RULER_AUDIT.md`, 2026-11-10)
confirmed the +141% is a real model-vs-CIE gap and narrowed the candidates to
the phase-function representation (axis A: delta-M / 32-moment truncation)
and the plane-parallel diffuse transport geometry (axis B).  This probe runs
the pre-registered configurations against the check3 scene and evaluates the
plan's G1 gate:

| config | change | horizon/zenith ratio | share of the gap |
|---|---|---|---|
| B0 | production default (32-moment delta-M SOS) | 8.81524 | — |
| P1 | `n_mu=64, n_moments=128` | 8.81188 | +0.065% — see the terminus finding: **no moment content above l=30, so this is the n_mu 32→64 quadrature effect** |
| P2 | `delta_m=False` (raw moments) | 8.81524 (B0 to 1 ulp; share −3.4e-16, the BLAS reduction-order floor) | ~0 — delta-M is exactly the identity here because f = β₃₂ = 0 (moment-level anchor) |
| P3 | `spherical-mc` (exact phase + spherical transport, low preset, seed 0) | 8.7760 ± 0.0418 (1σ) | +0.76% ± 0.81% (below 3σ, not counted) |
| T2 | aureole-band (γ∈[1°,10°]) truncated-series residual, layer 0 @550nm | — | 0.00012, **degenerate in N** (identical at N=32 and N=128) |

**Moment-series terminus (measured, review r1 Major 2):** the Mie scattering
scattering amplitude S(μ) is a finite sum of polynomials of degree ≤ nmax,
so |S|² is a polynomial of degree ≤ 2·nmax and the raw Legendre
moment series of this mixture terminates EXACTLY at l=30 (nmax ≤ 15 for the
rural mode set); β_l = 0 for l ≥ 31.  The "32-moment representation limit"
hypothesis is therefore vacuous for this aerosol — the truncated series IS
the exact phase function.  For large-x members (dust, cloud droplets) nmax is
large enough that the 32-moment truncation does lose forward energy and
delta-M is load-bearing there; the terminus moves with particle size.

G1 verdict (pre-registered complete partition): **R = 0.00065 < 1/4 →
stop-loss**.  Neither axis moves the ratio: the gap is not
representation-repairable.

Supplementary decomposition (`probe_decompose.py`, deterministic SOS):

* single scattering alone: ratio **7.535** (already +106% over CIE),
* multiple-scattering field alone: ratio 14.14; the diffuse enhancement is
  **horizon-weighted** in this model (horizon ×1.45 vs zenith ×1.24) --
  the opposite direction of the CIE narrative, where multiple scattering
  brightens the zenith relative to the horizon.

Residual attribution: the excess concentrates in (i) the first-order horizon
brightening vs the empirical CIE 110 gradation curve and (ii) a
horizon-weighted diffuse enhancement shared by the plane-parallel SOS and the
spherical-MC transport alike.  The moment-series terminus is recorded as an
explicit part of the attribution handed to the owner.  Owner re-adjudication
pending (plan stop-loss branch); the production default is unchanged by this
task.

## Reproduce

```bash
# the pre-registered probe (B0 anchor must reproduce the results.json dev
# bitwise; MC configs run in fresh worker subprocesses -- see below)
PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_aureole.py

# the single/multiple decomposition of the ratio
PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_decompose.py
```

Outputs (`probe_results.json`, `probe_costs.csv`, `probe_profiles.npz`) land
in `--out-dir` (default: this directory).  **A rerun in the default location
overwrites the committed 2026-11-11 evidence** (wall-clock/memory fields
change; the physics numbers are deterministic) -- pass `--out-dir <dir>` for
side-by-side runs or restore with `git checkout`.  A failed B0 anchor writes
`probe_results_anchorfail.json` and never overwrites the committed evidence.

## Cost (relative wall-clock criterion, plan Step 1)

See `probe_costs.csv` (the committed 2026-11-11 run; numbers below are
mechanically exported from it).  B0 full grid 32.9s; cut-only
SOS runs 1.20-3.14s
(128-moment P1 3.14s, i.e. ~2.6x B0 cut -- one-hand cost
evidence for any future higher-moment candidate, subject to the terminus
finding above); P3 low-budget MC 257s per pass.

## Known containment (not a physics choice)

Repeated `solve(approximation='spherical-mc')` calls on a
`_restrict`-subset atmosphere in ONE process corrupt the heap and SIGSEGV
after ~4-8 calls.  Reproduction assets (tracked here):

* `mc_repeat_segfault_stress.py` -- 30 repeats of the same tiny-budget call;
  observed 4 clean reps then SIGSEGV on 2026-11-11 (crash site drifts:
  manifestations inside `rt/mc.py` `_flush` -> `solar_source.py`, never the
  same line).  **Intermittent**: an otherwise identical 30-rep run later the
  same day completed clean -- absence of a crash in one run is not evidence
  of absence (the full-spectrum control stayed clean in both runs);
* `mc_restrict_segfault_bisect.py` -- view-subset/atmosphere bisection:
  the crash is independent of the view subset and does NOT reproduce with
  the full-spectrum atmosphere (8 reps clean), narrowing it to the
  `_restrict` interaction.

The root cause is a production finding, out of this task's scope, reported
to the owner for an independent spin-off (`twilight._sub_atmosphere` uses the
same `_restrict` mechanism).
