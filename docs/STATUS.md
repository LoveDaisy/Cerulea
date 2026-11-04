# Status

Current production behavior and limits after M18 (2026-11-01).
Methods and frozen evidence are linked in [the documentation guide](README.md);
completed work and proposed next scopes are in [ROADMAP.md](ROADMAP.md).

## What works

| Component | Production behavior |
|---|---|
| Spectral radiance, 378–780 nm | Rayleigh + Mie + O₃ + NO₂, spherical first-order geometry |
| Single scattering | Solar attenuation/shadow can refract; viewing rays and phase angles remain geometric |
| Multiple scattering | Successive orders, delta-M scaled, plane-parallel diffuse field |
| Deep twilight (below −8°) | Spherical MC split estimator; refracted first order + geometric higher orders |
| Aerosols | Eight-member Mie library, analytic Legendre moments, RH growth; AOD550 invariant under wavelength slicing |
| Refraction | On by default; two-branch solar columns and occlusion at the model's opaque bottom |
| Clouds | One homogeneous plane-parallel deck, delta-Eddington |
| Horizontal heterogeneity | Radial aerosol field, first order only |
| Colour and cameras | CIE 1931 → Bradford → tone mapping; panorama, fisheye, perspective |

Frozen `Medium` and `Scene` define the atmosphere and observing scenario;
`Atmosphere` is their facade. Shared species/column algebra lives in `optics`.
Explicit `rt.solver.solve` dispatch returns `SkyResult`; all spectral fields can
use the same renderer. `at_altitude` truncates the domain without renormalising
columns, and solar occlusion uses the same `r_bounds[0]` bottom. Contracts and
shared-physics validation blind spots: [ARCHITECTURE.md](ARCHITECTURE.md).

## Validation snapshot

The latest committed [six-check results](../validation/results.json) are
**3 PASS / 3 FAIL**. Passing regression tests do not close these physical gaps.
Use `uv run pytest -q` for the current test count; the M18 full run is preserved
in [CURVED_TRANSPORT_AUDIT.md](twilight/CURVED_TRANSPORT_AUDIT.md).

| Check | Recorded result |
|---|---|
| 1 — Rayleigh optical depth | PASS; RMS 0.48% vs Bodhaine (1999) |
| 2 — Direct/diffuse illuminance | FAIL; DNI +4–6%, DHI +41–46% vs CIE 110 |
| 3 — CIE relative luminance | FAIL; upper sky RMS 10.6%, sun-side horizon +142% at sun 45° |
| 4 — Horizon chromaticity | PASS, marginal; Δxy = 0.0184 |
| 5 — Twilight, first order only | FAIL in deep twilight; does not test the full engine |
| 6 — Koomen twilight | PASS within stated tolerances; −6° deterministic 0.69×, −9°/−12° engine 1.12×/0.59× |

Check6's deep points are one low-budget seed, not converged predictions. The
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

- Production solar prescriptions differ between backends; inverse-angle tables
  are sparse and refracted disk angular gain is absent. The experimental source
  addresses these contracts; production integration remains proposed work.
- Absolute twilight validation needs matched solar-axis metadata, site atmospheres
  and instrument responses. Patat's total V-band sky includes night background.
  The limited filter-only conversion changes present −6° spectra by 0.5–0.7%;
  station ratios are not altitude-only targets. [Reference audit](twilight/TWILIGHT_REFERENCE_AUDIT.md).
  The [2026-11-04 raw-observation follow-up](twilight/MATCHED_OBSERVATIONS.md)
  recovers candidate timestamps/site geometry and identifies a wrong-site
  ephemeris in published rural spectra; full observation matching is pending.
- Rare-path MC uncertainty remains large in deep twilight. A single error bar or
  a better fit does not identify missing scattering or calibration error.
- Horizon luminance and the 32-moment aureole remain unresolved; delta-fit,
  polarisation, structured clouds and ice-crystal halos are absent.
- M11 removed repeated Mie work through shared optics and a 256-state cache.
  Cold dusty optics still need optimisation; [M11 measurements](ROADMAP.md#m11--mie-working-set-caching-and-shared-preset-optics-2026-10-31)
  and `tools/perf_bench.py` separate startup from warm calls.
