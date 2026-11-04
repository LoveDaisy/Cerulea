# Deep-twilight reference: −9°/−12° sea-level zenith (task-deep-twilight-reference)

A defined, reproducible and budget-honest reference calculation for the
zenith spectral radiance at −9° and −12° solar elevation, replacing the
single low-budget Monte Carlo numbers quoted in `validation/results.json`
as *the* deep-twilight values. Everything below is recomputable from the
frozen data directory `validation/data/deep_twilight_reference/`; the
summary is verified against the raw records bit-consistently
(`--verify-only`). Process ledger and review trail live in the scratchpad
task directory.

**Merge scope (2026-11-05 owner decision):** the producing instrument —
`tools/deep_twilight_reference.py`, `tools/summarize_deep_twilight_reference.py`,
`tools/twilight_second_order.py`, their tests, and the `skycolor/rt/mc.py`
diagnostics sink — is **not merged into main**. It lives on branch
`task/deep-twilight-reference` (tip `dda8673`); only this document, the
ROADMAP/STATUS entries, and the frozen data directory entered main. The
reproduce commands below must run from that branch.

## Contract (pre-registered in `protocol.json`)

* Scenario: `presets.clear` (AOD550 0.05, 300 DU ozone, albedo 0.1,
  ground = observer = 0 km) on 380–780 nm / 5 nm (81 wavelengths); zenith
  view; sun azimuth 0.
* Source: the archived unified-source finite solar disk table
  (`phase3-clear-81-ground0-refined`, explicit label + SHA256, never
  rebuilt). Deterministic first order from `single_zenith` under the same
  source; higher orders from `mc_radiance_split` on native straight
  diffuse paths, 50-order cap and RR defaults.
* Statistics: the sampling unit is one independent complete spectrum;
  per budget the mean, sample SD and Student-t 95 % conditional interval
  (df = 15) over 16 seeds, plus a 5000-resample bootstrap of complete
  repeats (seed 90000) as sensitivity. Photopic luminance
  `683·trapz(L·V, wl)` integrates each spectrum before any statistics.
* Precision gates (pre-registered, tied to the probe-measured finite-disk
  scales): anchor half-widths ≤ scale/2 at both budgets, luminance
  half-width ≤ 0.5 %, budget drift ≤ half of each target, no verdict flip
  between interval methods, max single-path share ≤ 1 %, top-0.1 % ≤ 50 %,
  and no all-zero records. Failing any item marks the depth
  *budget-limited: unresolved*; a stable reference alone never implies the
  source correction is resolved.

## Formal results (16 seeds × {400k, 1.6M} photons/λ, 3.51 h wall-clock)

Direct estimate (`split_total` = finite-source deterministic single + MC
higher orders), conditional Student-t 95 % half-widths at the anchors:

| depth | budget | 450 nm | 550 nm | 650 nm | luminance [cd/m²] |
|---|---|---|---|---|---|
| −9° | 400k | 6.740e−7 ± 2.32e−7 (34 %) | 1.459e−7 ± 2.99e−8 (20 %) | 1.516e−7 ± 2.48e−8 (16 %) | 0.013444 ± 4.7 % |
| −9° | 1.6M | 6.188e−7 ± 9.55e−8 (15 %) | 1.708e−7 ± 1.77e−8 (10 %) | 1.370e−7 ± 1.25e−8 (9 %) | 0.013471 ± 2.3 % |
| −12° | 400k | 1.692e−8 ± 3.19e−8 (189 %) | 1.019e−8 ± 9.15e−9 (90 %) | 1.321e−8 ± 7.49e−9 (57 %) | 0.000993 ± 12.0 % |
| −12° | 1.6M | 5.942e−8 ± 3.60e−8 (61 %) | 1.321e−8 ± 3.71e−9 (28 %) | 8.974e−9 ± 2.81e−9 (31 %) | 0.000932 ± 9.7 % |

**Verdict: both depths are *budget-limited: unresolved*** at the
pre-registered anchor scales (which demand ≤ 2e−9 / 5e−10 absolute at −9°
and ≤ 1.5e−10 / 3.5e−11 / 5e−11 at −12°) and the 0.5 % luminance gate.
Every tail-concentration gate fails as well: at 1.6 M the worst
single-path share is 0.65 (−9°) and 1.0 (−12°), and the top 0.1 % of paths
carry effectively all of the higher-order signal — the deep-twilight
higher-order estimate is heavy-tail dominated at every budget reached
here, exactly as the planning probes predicted. The −12° 400k batch
contains all-zero anchor records (photon-budget detection limits,
"no-tail-evidence") and fails by construction. Bootstrap and Student
intervals agree; no verdict flips. Anchor means drift between budgets at
−9° beyond the half-scale target (e.g. 450 nm: 6.74e−7 → 6.19e−7, i.e.
−8 %) — consistent with rare-path mass still arriving between 400k and
1.6M, so even the point estimates are not budget-stable at the anchor
scale. Luminance drift passes at −9° (0.20 % ≤ 0.25 %) and fails at −12°
(6.6 %).

**Finite-disk source correction (paired, same seeds, 1.6 M):** at −9° the
paired (disk − point) interval excludes zero at all three anchors:
450 nm +4.69e−9 ± 1.08e−9 (+0.76 %), 550 nm +1.65e−9 ± 1.79e−10 (+0.97 %),
650 nm +1.24e−9 ± 2.55e−10 (+0.91 %) — **identified**, matching the ~1 %
scale the planning probes suggested. At −12° only 550 nm is identified
(+8.37e−11 ± 6.63e−11, +0.63 %); the 450 nm and 650 nm intervals contain
zero. This is the correction identification the task was built to
adjudicate; the −12° 450/650 nm corrections remain *unresolved*.

Luminance context: 0.0135 cd/m² (−9°) and 0.00093 cd/m² (−12°) under this
contract. The historical `validation/results.json` values (0.0182 /
0.00076) come from the production engine's anchor+smoothing path with a
different transport contract; they are not reconciled here and the
differences are not attributed.

## Exactly-two-scattering candidate (plan Step 3)

`tools/twilight_second_order.py` estimates the exactly-two-scatter,
no-ground component as an explicit two-vertex cascade integral (zenith
view; finite-source illumination shared with the MC; chord transmittance
with opaque-Earth occlusion; strictly positive proposals). Correctness was
established *before* any effect claim: zero scattering → 0, thin-atmosphere
density-squared scaling, closure against an independent nested Gauss
reference, and closure against the analog MC's own per-path
second-scatter component at −6° within combined Monte Carlo error. The MC
closure test caught three defects the synthetic closures could not
(a missing second-vertex `b_sca`, a volume-vs-cascade measure error, and a
4π-normalised phase used as a probability density) — recorded in the task
ledger; the analog-survival albedos are intentionally absent because they
are survival probabilities, already folded into the scattering densities.

Effect (16 seeds × 2 tiers × 2 depths on the frozen 3-wavelength planning
table, seeds 80000+; one rest MC per repeat, cost bookkeeping corrected in
review round 1): the comparison anchor is `total_analog` — the same-table,
same-seed analog estimator in the same records — with the full
three-wavelength repeat as the shared cost unit, judged by
**variance × repeat wall-clock** (fixed-budget efficiency: a slower
estimator with equal variance must not score better). The measured result
is *mixed and marginal*: variance×time ratios (candidate/analog, 1.6M
tier) are 0.94/0.93/1.07 at −9° (450/550/650 nm) and 0.89/1.04/1.15 at
−12° — the candidate wins 450 nm at both depths and 550 nm at −9° by
6–11 %, loses 650 nm at both depths and 550 nm at −12° by 4–15 %, and no
ratio approaches the order-of-magnitude gain a tail-dominated regime would
need. The L2 component itself is quasi-deterministic (tier-to-tier drift
≈ 1 %). At −12° neither estimator resolves anything (both intervals remain
tail-dominated; the rest-component second moment *grows* with budget as
more tail mass is seen). The full81 direct baseline is reported only as a
*distribution reference* across the table/stream boundary — it is not a
cost comparison, and no equal-wall-clock claim is made against it.
Per-path subtraction (candidate L2 + rest = analog total − second-scatter
analog) is exact by construction; totals are never computed as
differences of independent means. Correctness rests on the closures above
plus a three-independent-repeat MC closure at −6° (single scenario, not a
tail-coverage certificate).

## Sensitivity (plan Step 4)

Same-seed paired refinements at 100k photons/λ (seeds 110000+): solar
disk 2×8 → 4×16 (fingerprints unchanged, drift ≤ 1 %), order cap 50 → 100
(exactly zero — no path reaches the cap), conservative RR, MC fine optical
grid and phase-table refinements (drifts statistically indistinguishable
from zero; the sampling noise floor at this budget is ±20–200 % at −12°,
far above every discretisation effect), and deterministic single-scattering
refinements (Gauss order, density step: relative drifts ≤ 1.6e−5).
Discretisation effects are therefore *below the noise floor* of this
regime — a weak statement at −12°, where that floor is itself the problem.
The archived unified-source heldout evidence was supplemented with a fresh
independent check of the exact archived table (`table-heldout.json`:
grazing-point angle error ≤ 9.7e−5°, effective source relative error
≤ 0.40 %, zero shadow disagreements). Curved diffuse transport remains an
explicit unknown, not a run under this budget.

## Cost and budget accounting

| stage | wall-clock | cap |
|---|---|---|
| direct (64 records) | 7657 s | 4 h shared with pairs |
| source-pairs (32 records) | 4965 s | (shared) |
| candidate (64 records) | 1304 s | 2 h |
| sensitivity (32 records + singles) | 140 s | 2 h |

Erratum: Darwin records of the first formal wave archived
`max_rss_bytes` with a 1024× unit error (fixed in the runner); divide
archived values by 1024 — the frozen records are not rewritten, the
summary carries the note.

Total ≈ 3.9 h against the plan's ≈ 8 h implementation envelope; each
record pins its own seconds, peak RSS, thread environment, code and
source-table SHA256s.

## Reproduce

The instrument is not part of main; check it out from the archive branch
first (data and docs exist there identically):

```bash
git worktree add ../deep-twilight-ref task/deep-twilight-reference
cd ../deep-twilight-ref
uv run python -m tools.summarize_deep_twilight_reference \
    --directory validation/data/deep_twilight_reference --verify-only
uv run pytest -q tests/test_deep_twilight_reference.py \
    tests/test_twilight_second_order.py
```

The stage runners are append-only and resume missing seeds; re-running a
completed stage is a no-op. Regression: full pytest 338 passed;
`validation.run_all` 3 PASS / 3 FAIL identical to the documented baseline;
`validation/results.json` untouched (`regression-run-all.log`).

## What this changes and what it does not

`validation/results.json` and all production defaults are untouched. The
reference is a diagnostic instrument: it quantifies how far the
higher-order twilight signal is from being budget-resolved (≈ 10–30 %
conditional intervals at 1.6 M per λ, heavy-tail dominated) and it
identifies the finite-disk source correction where the data allows
(−9° fully, −12° at 550 nm). Closing the −12° anchors needs an estimator
that removes the tail dominance, not more of the same budget; the
candidate is a measured step in that direction, and the diagnostics sink
(`mc_radiance_split(diagnostics=True)`) is the instrument that made the
decomposition visible. Both are available for follow-up tasks; neither is
wired into any production path.
