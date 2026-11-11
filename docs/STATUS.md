# Status

Current production behavior and limits (2026-11-08, unified production/presentation entry).
Methods and frozen evidence are linked in [the documentation guide](README.md);
completed work and proposed next scopes are in [ROADMAP.md](ROADMAP.md).

A reader-level mainline narrative (Chinese) lives in [physics/MAINLINE.zh.md](physics/MAINLINE.zh.md); this file stays the single source for current behavior and validation limits.

## What works

| Component | Production behavior |
|---|---|
| Spectral radiance, 378–780 nm | Rayleigh + Mie + O₃ + NO₂, spherical first-order geometry |
| Solar source | One unified finite-disk source for every backend: apparent directions, continuous refracted columns, occultation, 16′ disk and gain (`skycolor/solar_source.py`, contract in [physics/SOLAR_SOURCE.md](physics/SOLAR_SOURCE.md)) |
| Single scattering | Unified source at every scattering point; viewing rays remain geometric (retained approximation) |
| Multiple scattering | Successive orders, delta-M scaled diffuse moments (raw-extinction sun), plane-parallel diffuse field; per-disk-beam Fourier source |
| Deep twilight (below −8°) | Spherical MC split estimator; unified-source first order + straight-flight higher orders |
| Aerosols | Eight-member Mie library, analytic Legendre moments, RH growth; AOD550 invariant under wavelength slicing |
| Refraction | On by default; two-branch solar columns and occlusion at the model's opaque bottom |
| Clouds | One homogeneous plane-parallel deck, delta-Eddington; per-beam top/base/mixed illumination from the unified source |
| Horizontal heterogeneity | Radial aerosol field, first order only |
| Request entry | `rt.solver.solve`: four finite approximations (spherical-single / plane-sos / hybrid-sos / spherical-mc), capability/support checks before kernels, per-component arrays plus effective settings; the three legacy entry points are thin forwarders |
| Presentation | `render.solve_dome` prepares one result for display (regular dome grid, ground albedo/irradiance, apparent sun geometry); `render_result`/`chart_table` are pure consumers; the CLI render/chart share one preparation and print a provenance summary (`format_result_summary`); solar time/numeric inputs resolve through `scene.resolve_sun` to the pre-refraction direction |
| Colour and cameras | CIE 1931 → Bradford → tone mapping; panorama, fisheye, perspective |

Frozen `Medium` and `Scene` define the atmosphere and observing scenario;
`Atmosphere` is their facade. Shared species/column algebra lives in `optics`.
`rt.solver.solve` is the single authoritative composition and returns a
`SkyResult`; all spectral fields can
use the same renderer. `at_altitude` truncates the domain without renormalising
columns, and solar occlusion uses the same `r_bounds[0]` bottom. Contracts and
shared-physics validation blind spots: [ARCHITECTURE.md](ARCHITECTURE.md).

## Validation snapshot

The [approximation story](physics/APPROXIMATIONS.md) separates orders,
propagation and numerics using two real current-production scenes; the
[curated figure catalogue](../figures/README.md) owns versions and display
parameters. These examples explain behavior, not absolute accuracy.

The latest committed [six-check results](../validation/results.json) are
**3 PASS / 3 FAIL** (unified solar source; see [physics/SOLAR_SOURCE.md](physics/SOLAR_SOURCE.md)). Passing regression tests do not close these physical gaps.
Use `uv run pytest -q` for the current test count; the M18 full run is preserved
in [CURVED_TRANSPORT_AUDIT.md](twilight/CURVED_TRANSPORT_AUDIT.md).

| Check | Recorded result |
|---|---|
| 1 — Rayleigh optical depth | PASS; RMS 0.48% vs Bodhaine (1999) |
| 2 — Direct/diffuse illuminance | PASS; DNI +4–6%, DHI +6.8%/+3.0% vs CIE 110 at scene-mapped turbidity (ruler-fix; contract analysis in [RULER_AUDIT](../validation/RULER_AUDIT.md)) |
| 3 — CIE relative luminance | FAIL; upper sky RMS 10.6%, sun-side horizon +142% at sun 45° |
| 4 — Horizon chromaticity | PASS, marginal; Δxy = 0.0184 |
| 5 — Twilight, first order only | FAIL in deep twilight; does not test the full engine |
| 6 — Koomen twilight | FAIL; −6° deterministic 0.55×, −9°/−12° engine 1.15×/0.75× — the round-4 contact-band disk fix raised the deep-twilight MC NEE (old 2×8 disk under-integrated partially occulted beams; deterministic SS unchanged) |

Read-only audit of the reference contracts behind checks 2/3/6 (premises,
definitions, solar-axis conventions, transcription): 2026-11-10, see
[validation/RULER_AUDIT.md](../validation/RULER_AUDIT.md) — **conclusions live
there** (this pointer is not a second source; no physical numbers changed).
The ruler-fix correction landed 2026-11-11: check2's DHI reference contract
was re-mapped to the scene turbidity per the audit's adjudication and the two
DHI metrics re-adjudicated PASS, backed by the `ruler_audit_recheck.py`
legality gates.

Check6's deep points are one low-budget seed, not converged predictions. The
[deep-twilight reference](twilight/DEEP_TWILIGHT_REFERENCE.md) quantifies
what those points actually are: at 16 seeds × 1.6M photons/λ the −9°/−12°
zenith anchors are still heavy-tail dominated (9–61 % conditional
intervals), so no deep number in this file is a converged prediction. The
older engine ratios 0.85–0.87× at −6° and 0.54× Patat at −9° predate the
[AOD550 anchor-optics fix](twilight/AEROSOL_GRID_AUDIT.md); they are historical benchmarks.
Independent numerical checks also cover Mie moments, SOS/MC agreement and
cloud energy balance; their scope is described in the method documents.

## Twilight exploration: what changed

Two production defects were fixed: raised-ground solar occlusion (M13) and
wavelength-dependent AOD550 renormalisation (M16). For the clear, refraction-on,
−6° zenith comparison, the deterministic 2.4 km/sea-level ratio is now **1.027**;
its old 1.208 and low-budget engine 1.01 values precede the occlusion fix.
[Altitude audit and reference conditions](twilight/TWILIGHT_ALTITUDE_AUDIT.md).

The following references use a unified finite-disk solar source, 81 wavelengths,
eight independent seeds and a −6° zenith view. Values are cd m⁻²; ± denotes
empirical seed standard error. They are **experimental**, with different
transport budgets; their difference does not establish a curvature effect.

| Experiment | Sea level | 2.4 km | High/sea |
|---|---|---|---|
| [Unified source, straight diffuse paths](twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md) | 0.49323±0.00308 | 0.45977±0.00347 | 0.93215±0.00748 |
| [Continuous diffuse refraction](twilight/CURVED_TRANSPORT_AUDIT.md) | 0.489±0.008 | 0.467±0.011 | 0.956±0.036 |

Source/first-order/MC closure and independent curved-flight checks passed within
their stated scope. The curvature correction's sign and station residuals remain
unresolved; an unexplained native exit 139 is recorded in the transport audit.
These backends have not replaced production defaults.

## Remaining limits

- Production backends share the unified finite-disk solar source
  (`physics/SOLAR_SOURCE.md` §Known deviations records the adjudicated
  residuals). Remaining inverse-angle table sparsity and refracted disk
  angular gain are tracked there; the entry consolidation
  (task-simulation-entry) made `rt.solver.solve` the single composition
  of single+SOS/MC/cloud.
- Absolute twilight validation needs matched solar-axis metadata, site atmospheres
  and instrument responses. Patat's total V-band sky includes night background.
  The limited filter-only conversion changes present −6° spectra by 0.5–0.7%;
  station ratios are not altitude-only targets. [Reference audit](twilight/TWILIGHT_REFERENCE_AUDIT.md).
  The [2026-11-04 raw-observation follow-up](twilight/MATCHED_OBSERVATIONS.md)
  recovers candidate timestamps/site geometry and identifies a wrong-site
  ephemeris in published rural spectra; full observation matching is pending.
- Rare-path MC uncertainty remains large in deep twilight. A single error bar or
  a better fit does not identify missing scattering or calibration error.
- Horizon luminance remains unresolved, but no longer for lack of an aureole
  hypothesis: the pre-registered attribution probe (task-aureole-delta-fit,
  2026-11-11) excluded moment representation and transport geometry (G1
  stop-loss, R = 0.00065) and shifted attribution to first-order horizon
  brightening vs the empirical CIE 110 gradation plus a horizon-weighted
  diffuse enhancement (`validation/tools/aureole_attribution/`).  Delta-fit,
  polarisation, structured clouds and ice-crystal halos remain absent.
- M11 removed repeated Mie work through shared optics and a 256-state cache;
  task-cold-optics then batch-vectorised the cold build itself (bitwise
  coefficient kernels, ulp-level declared-tolerance moment/table kernels).
  Cold dusty `delta_m(32)` fell from 39.4 s to about 3 s; see
  [M11 measurements](ROADMAP.md#m11--mie-working-set-caching-and-shared-preset-optics-2026-10-31)
  and `tools/perf_bench.py`, which separates startup from warm calls.
