# Aureole attribution probe (task-aureole-delta-fit, 2026-11-11)

Pre-registered attribution probe for the check3 h=45deg sun-side
horizon/zenith excess (model 8.815 vs CIE 110-1994 3.652, dev +141.406%,
`validation/results.json`).  Promoted from the task scratchpad per the plan's
evidence-persistence rule (review new-r1-Minor5): every number quoted in the
check3 note and the stop-loss record reproduces from this directory.

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
| P1 | `n_mu=64, n_moments=128` | 8.81188 | +0.065% |
| P2 | `delta_m=False` (raw moments) | 8.81524 (bitwise B0) | ~0 |
| P3 | `spherical-mc` (exact phase + spherical transport, low preset, seed 0) | 8.7760 ± 0.0418 (1σ) | +0.76% ± 0.81% (below 3σ, not counted) |
| T2 | aureole-band (γ∈[1°,10°]) truncated-series residual, layer 0 @550nm | — | 0.00012 at N=32 and N=128 |

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
spherical-MC transport alike.  Owner re-adjudication pending (plan stop-loss
branch); the production default is unchanged by this task.

## Reproduce

```bash
# the pre-registered probe (B0 anchor must reproduce the results.json dev
# bitwise; MC configs run in fresh worker subprocesses -- see below)
PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_aureole.py

# the single/multiple decomposition of the ratio
PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_decompose.py
```

Outputs land next to the scripts (`probe_results.json`, `probe_costs.csv`,
`probe_profiles.npz`); the committed files are the 2026-11-11 run
(git d0b956b tree).

## Cost (relative wall-clock criterion, plan Step 1)

See `probe_costs.csv`.  B0 full grid 34.8s; cut-only SOS runs 1.2-3.2s
(128-moment P1 3.2s, i.e. ~2.6x B0 cut -- one-hand cost evidence for any
future higher-moment candidate); P3 low-budget MC 261s per pass.

## Known containment (not a physics choice)

Repeated `solve(approximation='spherical-mc')` calls on a
`_restrict`-subset atmosphere in ONE process corrupt the heap and SIGSEGV
after ~4-8 calls (repro: task scratchpad `probe_mc_stress.py`; full-spectrum
atmosphere is clean).  The probe therefore runs each MC config in a fresh
worker subprocess.  The root cause is a production finding, out of this
task's scope, reported to the owner.
