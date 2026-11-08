# Roadmap

*Where the project is, and where it could go.  Written to be read two months
from now.*

The organising question has not changed: **ignoring clouds and polarisation,
can we predict the colour of the sky from first principles, and can we say how
confident we are?**  Everything below is measured against that.

This tracked document records completed milestones and their evidence, including
conclusions superseded by later work.  [STATUS.md](STATUS.md) is the current
snapshot; topic documents carry reproducible methods, results and limitations.
Session handoffs in `scratchpad/` are temporary navigation aids, not a second
source of project state.  Once consumed, their durable findings belong here or
in the relevant topic document; their historical instructions are retired.

---

## Milestones reached

### M1 — the backbone (single scattering)

Exact spherically-symmetric single scattering with the *curved* solar beam,
which is what makes twilight possible at all.  Spectral data tables (CIE 1931,
ASTM G173 AM0, JPL-2006 O₃/NO₂) are committed so the code runs offline.
Colour pipeline: CIE XYZ → Bradford adaptation → ACES tone mapping.

*Validated:* Rayleigh optical depth vs Bodhaine (1999), RMS 0.48 %.
*Code:* `rt/single.py`, `spectra/`, `color.py`, `render.py`.

### M2 — multiple scattering

Successive orders of scattering in a plane-parallel atmosphere with a
pseudo-spherical beam: Fourier azimuth expansion, Gauss-Legendre nodes, a
Lambertian ground.  An independent Monte Carlo was written alongside it purely
as a validator.

*Validated:* energy conservation 2.3 × 10⁻⁴; MS/SS at the zenith 0.31, matched
by the Monte Carlo to 1 %.
*Code:* `rt/multiple.py`, `rt/mc.py`.

### M3 — validation and honesty

Five independent checks against published references, plus a harness that
reproduces them (`validation/`).  This is the milestone that turned the project
from "looks plausible" into "we know where it is wrong": the limb is 2× too
bright, the solar aureole was too weak, and twilight below −8° collapsed.

### M4 — the aerosol library and real Mie optics

Eight Mie-backed aerosol members with **analytic** Legendre moments (Gaunt/Wigner
3j, no angular quadrature) and delta-M scaling.  Removed the Henyey-Greenstein
approximation, which was wrong by an order of magnitude inside 1° for dust.

*Validated:* moments vs quadrature 2 × 10⁻⁸; `beta_1` = Bohren-Huffman `g`
to 10⁻¹⁰.
*Code:* `mie.py`, `aerosol.py`, `docs/physics/AEROSOL.md`.

### M5 — the spherical twilight engine

The deep twilight needs light that travels *sideways* for hundreds of
kilometres.  A split-estimator Monte Carlo (exact deterministic first order +
Monte Carlo for orders ≥ 2) on a coarse angular grid closes the gap.

*Validated:* −9° goes from 24× too dark to 1.39× of Patat et al. (2006); −12°
from 1500× to 1.45×.  (M10 later moved these to 0.54× and 0.37×: the 1.39/1.45
were the one-branch chord.)
*Code:* `rt/twilight.py`, `docs/twilight/TWILIGHT.md`.

### M6 — refraction, taken seriously (**now the default**)

Ray-traced refraction (33′ at the horizon, matching Sæmundsson above 5°), then
the full radiative-transfer treatment: the Bouguer invariant `n r sin θ = a`
turns the curved path length into the same closed form as a straight chord with
`b_k = a/n_k`, so the whole correction costs one division per shell rather than
a ray trace per photon.

*Validated:* the per-shell formula against numerical path integration to 0.1 %
(1661 km vs 1663 km for a horizontal ray); the apparent-elevation table against
the ray tracer.
*Code:* `refraction.py`, `data/refraction.csv`, `docs/physics/REFRACTION.md`.
*Consequence:* twilight is 2–5× brighter.  The absolute level that this exposed
was later traced to a two-branch chord bug (M10), not to refraction itself.

### M7 — a cloud deck

A plane-parallel homogeneous cloud layer: Mie droplets, delta-M, a
closed-form delta-Eddington two-stream and a spherical terminator that decides
which part of the deck is still lit.

*Validated:* albedo against an independent Monte Carlo within 0.1–0.7 %
(`g = 0`) and 1 % (`g = 0.85`, τ ≥ 5); `R + T + direct = 1` to 10⁻¹².
*Code:* `cloud.py`, `rt/cloud.py`, `docs/physics/CLOUD.md`.

### M9 — delta-M

`atmosphere.delta_m_properties()` applies the Wiscombe rescaling and
`rt/multiple.py` uses it by default.  Fine aerosols are unaffected (f = 0);
dust changes the multiple-scattering field by 2.7×, because 35 % of its
scattered energy lives in the forward peak that the 32-moment truncation was
discarding.

### M8b — a second twilight dataset and a new altitude question

A single reference cannot separate a model error from a reference bias.  A
second, independent, *photopic* twilight measurement (Koomen et al. 1952,
sea level) enabled a comparison closer to the model's observable.  At this
stage the geometric model was 1.14× Koomen, and the difference from the
high-altitude Patat reference was attributed to altitude.  M10 subsequently
invalidated that explanation: the 1.14× agreement partly cancelled a chord
bug against missing refraction.  The two-site comparison remains useful, but
it is not a controlled experiment that changes only observer altitude; air
clarity, bandpass and calibration also differ.

The durable result is the independent reference dataset and an unresolved
altitude-response question, not closure of the twilight problem.
See `validation/SECOND_DATASET.md` and
[the altitude audit](twilight/TWILIGHT_ALTITUDE_AUDIT.md).

### M8 — the physics of red skies and blue suns

Two questions that turned out to have clean answers:

* **Red afterglow** is not a low-level haze effect.  A boundary-layer aerosol
  makes a sunset *dimmer and whiter*; the glow needs a scattering layer above
  the Earth's shadow — stratospheric sulphate after a volcano, or cirrus.
* **The blue sun** needs a non-monotonic extinction spectrum, which an Ångström
  power law can never produce and a narrow Mie size distribution can.

*Figures:* `08_volcanic_twilight.png`, `09_blue_sun.png`.

### M10 — the solar chord: one branch short

The twilight level was not a physical excess.  For a scattering point whose
sun is below the local horizontal, the ray to the sun crosses its perigee: the
light traverses the far branch (perigee → space) and the near branch
(perigee → point).  `path_lengths_sun` returned only one perigee → space
traversal, i.e. it stopped at the perigee and dropped the near-side
perigee → point segment.  Every grazing twilight ray was therefore
under-attenuated by half of its dense lower-atmosphere column.

*Verified:* the geometric column at 40 km, χ = 96°, against a numerical line
integral: 8.67 × 10³⁰ vs 4.35 × 10³⁰ m⁻², ratio 1.99.  The refracted column
at 38/47/60 km against an RK4 ray trace: 0.50 before, 0.995/1.010/0.984 after.
*Consequence:* the −6° zenith moves from 1.42 to 0.44 cd m⁻² (deterministic)
and from 0.99 to 0.55 (spherical engine); model/Koomen goes from 2.20 to 0.69
(deterministic) or 0.85 (engine); model/Patat from 2.75 to 0.86 or 1.07.  The
old geometric 1.14× was a partial cancellation between this bug (too bright)
and the absence of refraction (too dark).
*Code:* `geometry.py`, `refraction.py`, `tests/test_geometry.py`,
`tests/test_refraction.py`.
*Lesson:* the Monte Carlo and the engine share `path_lengths_sun`, so their
1 % agreement could not catch this; the independent check has to be a line
integral or a ray trace.

### M11 — Mie working-set caching and shared preset optics (2026-10-31)

Profiling separated optical pre-processing from radiative transfer.  The
per-wavelength LRU held eight Mie states while the usual 5 nm working grid
contained 81 wavelengths, so repeated traversals rebuilt every state.  The
capacity was raised to 256, and preset/default atmospheres began sharing
their aerosol definitions and caches.

*Recorded measurements:* Python 3.12 on the development machine, 81 wavelengths,
clear atmosphere unless specified.  Warm single scattering for one direction
fell from 0.353 to 0.003 s; a second `delta_m_properties(32)` from 6.810 to
0.002 s; dusty moments through a second preset instance from 40.858 s to
below 1 ms.  Ten saved output arrays were bit-identical; the milestone's
full test run passed.  Peak RSS rose from 171 to 194 MiB.  These are historical
measurements, not timing requirements for every machine.

*Limits:* cold Mie construction was unchanged; the dusty moment build still
cost about 40 s and a full cold dusty one-direction solve about 178 s in the
recorded workloads.  A 401-wavelength grid exceeds the 256-state capacity.
Refraction-table interpolation was measured at sub-millisecond cost for the
single-scattering query size, so the proposed interpolation rewrite was
withdrawn.  No reduction in the default moment order was used.

*Commits:* `6003af8` (capacity), `dbc49b8` (sharing), merge `f366245`.
*Reproduce:* `uv run python tools/perf_bench.py`; it reports separate cold and
repeat operations.  Shared definitions became frozen in M12.

### M12 — explicit core boundaries and consistent site truncation (2026-10-31)

The architecture refactor established frozen `Medium` and `Scene` definitions,
an `Atmosphere` facade, single-source species-column algebra in `optics`,
explicit backend dispatch returning `SkyResult`, and a solver-independent
spectral-field input to the existing renderer.  Custom density profiles are
injected data rather than subclass/monkey-patch protocols.  Aerosol/cloud
definitions are frozen with replacement APIs; refraction-table lifetime and
identity follow medium construction rather than pressure/elevation proxies.

The intentional physical behavior change was `at_altitude(z0)`: truncate the
layer grid at the site, put the ground at z0, preserve the density profiles and
do not renormalise any column.  This removes double pressure scaling and the
inconsistent ground locations of single scattering, SOS and MC.  At 2.4 km,
pressure is 756.26 hPa and density equals the original profile at that height.
A raised observer over a sea-level domain remains a distinct balloon scenario.

*Verified at the milestone:* seven golden-equivalence cases, extension cases,
and a full run of 257 tests passed at `3401930`; repeat optics timings retained
M11's gains.  The six-check physical validation kept the prior 3 PASS / 3 FAIL
pattern: the diffuse-illuminance, CIE distribution and single-scattering
twilight checks still failed.  Preserved regression results do not resolve
those physical discrepancies.  At −6°, the recorded high-site/sea-level ratio
was 1.208 for the deterministic solver and 1.007 for the low-budget spherical
engine (seed 0).  The engine point estimate does not establish a statistically
significant brightening, and neither result by itself identifies the cause of
the difference from the approximately 0.80 cross-site reference ratio.  These
ratios describe the M12 baseline: M13 subsequently found that solar-visibility
tests still allowed rays through the raised ground.  The earlier claim that
no construction issue remained, and the attribution to a missing low-altitude
diffuse component, were therefore premature.

*Commits:* `f8f07ac` (optics), `55a3fca` / `12f148f` (medium/scene), `22e847f`
(site truncation), `451d877` (results/display), `ddb2e16` (frozen optics),
merge `a15555b`.  See [ARCHITECTURE.md](ARCHITECTURE.md) for current contracts.
The temporary 2026-10-30 handoff has been consumed; M11/M12 supersede its
instructions to implement caching or truncation.  The remaining physical
questions and follow-up experiments live in
[TWILIGHT_ALTITUDE_AUDIT.md](twilight/TWILIGHT_ALTITUDE_AUDIT.md).

### M13 — raised-ground solar occlusion and altitude diagnostics (2026-10-31)

An independent audit of the high-site comparison found a second boundary
inconsistency after M12: transport ended at the raised ground, but several
solar-visibility tests still treated sea level as the opaque surface.  The
geometric masks, refracted grazing invariant and shadow tables now use the
model bottom radius.  Single scattering, SOS, MC and cloud illumination follow
that boundary; the M10 two-branch solar-column integration is unchanged.

*Measured effect:* for `clear`, 81 wavelengths (380–780 nm in 5 nm steps),
refraction on, sun −6°, zenith view, the sea-level deterministic luminance
remains 0.444279 cd m⁻².  At 2.4 km the single-scattering component changes
from 0.411475 to 0.391830 and the SOS component from 0.125253 to 0.064520
cd m⁻².  Their total changes from 0.536728 to 0.456350, so the high/low ratio
falls from 1.208 to **1.027**.  This is a zenith-twilight effect: the former
claim that raised-ground shadow alignment mattered only near the horizon was
incorrect.  Multi-seed spherical-engine results and their photon budgets are
reported separately in [the audit](twilight/TWILIGHT_ALTITUDE_AUDIT.md).

The audit also distinguishes the experiment from the references.  The often
quoted 0.80 is a Maryland/Paranal ratio spanning different studies and
observables.  Koomen's original same-instrument station table instead gives
0.022/0.060 ≈ 0.37 at −6°, but the stations also differ in observation season
and atmospheric transparency.  Neither is a controlled derivative obtained
by removing only the bottom 2.4 km of one atmosphere.  Fixing the occlusion
defect does not by itself validate the remaining physical model or identify
the cause of its difference from either station comparison.

*Code and evidence:* `rt/single.py`, `rt/multiple.py`, `rt/mc.py`,
`rt/cloud.py`, `refraction.py`, `tests/test_raised_ground.py`, and
`tools/twilight_altitude_audit.py`.  The topic report records the source audit,
controlled experiments and limits of the conclusion.
*Fix commit:* `c45a03b`.

---

### M14 — isolate solar-source approximations (2026-11-01)

Paired, full-spectrum direct MC holds every diffuse path fixed while changing
only solar attenuation/shadow to the current single-scattering prescription.
Higher orders increase 29.0% at sea level and 25.5% at 2.4 km; the hybrid
altitude ratio changes 0.9568±0.0071 → 0.9292±0.0072 (empirical standard errors).
All 16 mode pairs have identical next-event position fingerprints, and their
geometric baselines reproduce M13's spectra bit-for-bit.

Independent RK shooting and a separate radial-curvature integral then show
that this source prescription itself needs refinement: sparse inverse-table
angles can miss by 0.1–0.3°, and the refracted solar disk's angular amplitude
is not represented by transmittance alone.  At −6°, the controlled sea-level
first order changes from 0.3684 to 0.3506 with refined angles, then 0.2719
with a point-beam angular Jacobian; a vertical finite-disk approximation gives
0.2772 cd m⁻².  These are source ablations, not a new all-order sky prediction.

*Outcome:* validate one angular/energy-consistent solar source before
calibrating absolute levels or investing in a new diffuse solver.  Production
defaults and prior physical validation remain unchanged in this phase.
Tools, raw data, assumptions and independent checks are in
[SOLAR_SOURCE_AUDIT.md](twilight/SOLAR_SOURCE_AUDIT.md).
*Experiment commit:* `af235a7`.

### M15 — one finite solar source, independently checked (2026-11-01)

Continuous radial columns, a shadow-aligned adaptive direction table, the
angular Jacobian and a two-dimensional 16′ solar disk now form one experimental
source.  Independent Cartesian ray tubes check energy, separate RK density
integrals check columns, and a spline-knot counterexample tightened the angular
oracle before differentiation.  Source gain and detector transfer consistently
use conservation of reduced radiance `L/n²`.

Across 48 full-spectrum records (81 wavelengths, 100k paths/wavelength/seed,
eight seeds, two heights, three source definitions), source-only variants have
identical transport fingerprints.  The unified source gives SS+MS
0.493233±0.003081 / 0.459769±0.003474 cd m⁻² and high/sea 0.932154±0.007479.
Deterministic and MC photopic first orders differ by 0.12/0.23 seed-SE.
These are straight-diffuse-path reference experiments, not a new production
default or an identified station-altitude law.

*Code checkpoint:* `7ed71a6`; *evidence:* `2255b46` and
[UNIFIED_SOLAR_SOURCE_AUDIT.md](twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md).

### M16 — wavelength slicing preserves AOD550 (2026-11-01)

An independent definition check found that monochromatic Mie queries reset
the extinction reference to the queried wavelength.  Twilight's anchor loop
therefore changed the medium.  Mie now always divides by actual `ce(550 nm)`;
the already-normalized Angstrom law no longer gets an interpolated correction.
Eight members' full81/full401 optical arrays are bit-identical; all sampled
single/subset queries now match their full-grid values exactly.

New tests give 13 fail / 3 pass before, 16 pass after; related tests 28 pass.  Full
regression gives 313 pass plus one expected old twilight golden, which passed
after isolated attribution/update; the other six golden files are unchanged.
Six physical checks remain 3 PASS / 3 FAIL; only check6's two low-budget deep points
change.  The corrected definition, not better observational fit, justifies the
update.  *Fix:* `5d6f793`; *evidence:* `73031e3` and
[AEROSOL_GRID_AUDIT.md](twilight/AEROSOL_GRID_AUDIT.md).

### M17 — establish what the twilight references measure (2026-11-01)

Primary papers distinguish Patat's total V-band sky background from pure
solar scattering, and show both station comparisons confound height with
other conditions.  A historical night-background illustration contributes
0.0475% at −6° and 16.8% at −12°; it is not a measured same-night subtraction.
The solar true/apparent and centre/limb contracts remain unverified.

Offline synthesis of all 48 existing spectra through an explicitly limited
ESO V-filter/CALSPEC approximation changes unified −6° total luminance by only
−0.46%/−0.65%, height ratio by −0.19%.  It does not explain the remaining
10–25% cross-scene residual, nor replace the missing complete instrument
response.  No atmosphere, axis shift or zero point was fitted.  See
[TWILIGHT_REFERENCE_AUDIT.md](twilight/TWILIGHT_REFERENCE_AUDIT.md).

### M18 — continuous diffuse refraction and a bounded conclusion (2026-11-01)

Thin-index Snell layers were rejected after concrete artificial-TIR/duct
counterexamples.  Continuous arc-length RK flights now compose with M15's
source.  Independent Hamiltonian trajectories and oblique DOP853 first-order
integrals check directions, optical depth, first boundary and reduced-radiance
transfer.  Source-table construction also now rejects wrong-medium bindings;
the original experiment loader had already checked those contracts.

At −6°, 81 wavelengths, eight seeds, curved SS+MS is
0.489009±0.007777 / 0.467331±0.011337 cd m⁻², high/sea 0.955669±0.035776.
Compared with matched-source global-straight controls, corrections are
−0.63%±1.63% /+3.64%±2.67%; their conditional 95% intervals include zero.
This budget does not identify the curvature correction's sign or explain
the observational residual.  Actual MC-vertex 16→64 solar-disk checks and
2→1km flight refinement are much smaller than sampling uncertainty.

The final full regression passes 316 tests; production physical checks remain
3 PASS / 3 FAIL.  One native process exit 139 remains unexplained, with saved partial
records and a successful faulthandler continuation; it is not concealed as a
completed run.  The experimental backend has not replaced production defaults.
See [CURVED_TRANSPORT_AUDIT.md](twilight/CURVED_TRANSPORT_AUDIT.md) and its frozen evidence.

### Next work justified by this exploration

1. **One production solar-source definition.** Replace the known mixed source
   prescriptions with the independently checked direction/column/energy
   contract, retaining separate transport verification.  A deliverable must
   preserve the declared atmosphere under spectral restriction and expose
   approximation limits across the intended scene range.
2. **Stable higher-order uncertainty.** Obtain reproducible estimates and
   useful confidence intervals for rare-path twilight at a practical runtime.
   More photons alone may be expensive; judge any estimator or acceleration
   by time to a stable physical estimate, not throughput or one reported SE.
3. **Matched absolute validation.** Acquire observing-time/solar-axis metadata,
   the instrument response and site atmospheric conditions before assigning
   the cross-study residual to pressure, aerosols or missing scattering.
   [Raw-observation probes on 2026-11-04](twilight/MATCHED_OBSERVATIONS.md)
   locate concrete spectral and same-twilight candidates and recover selected
   geometry, including a verified wrong-site ephemeris in published rural
   spectra. Instrument, background and atmospheric matching remain open.
4. **Cold optical startup remains separate.** M11's warm-cache success stands;
   optimise cold Mie work independently while preserving the physical contract.

These are proposed engineering/research scopes, not scratchpad tasks already
started.  The completed exploration and raw evidence are durable here; local
session handoffs are not a substitute for this record.

## Open questions that matter now

1. **Production source consistency.** Done: one unified finite-disk solar
   source (apparent direction, continuous refracted columns, occultation,
   16' disk, gain) now feeds single scattering, SOS, twilight MC, clouds
   and direct illumination (`skycolor/solar_source.py`; contract and the
   2026-11-06 adjudicated deviations in [physics/SOLAR_SOURCE.md](physics/SOLAR_SOURCE.md)).
   Remaining candidates, each awaiting a separate go/no-go: a DOP853
   ray-tube referee for near-caustic knot rows, and a per-height vectorized
   dome query kernel (the standing dome cost breach).
   [Source audit](twilight/SOLAR_SOURCE_AUDIT.md), [unified source](twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md).
2. **Absolute twilight brightness and altitude response.** Domain truncation,
   raised-ground occlusion and AOD550 slicing are fixed. The remaining station
   residual cannot be assigned to missing low-level scattering, calibration or
   site pressure. Rare-path uncertainty and matched observing conditions still
   need to be separated. [Final transport audit](twilight/CURVED_TRANSPORT_AUDIT.md).
3. **Observation contracts.** True/apparent solar altitude, disk centre/limb,
   instrument response and site conditions remain unverified. A better fit does
   not determine the solar-axis convention. The limited V-filter synthesis in
   M17 is complete; full-response synthesis and same-night background handling
   are still open. [Reference audit](twilight/TWILIGHT_REFERENCE_AUDIT.md).
4. **The limb and aureole.** CIE 110 horizon luminance differs by about a factor
   of two; a measured clear-sky spectral/luminance scan is needed to distinguish
   model and reference error. Mie/delta-M are present, but 32-moment aureole
   resolution and delta-fit remain open.
5. **Runtime to a reliable result.** M11 removed repeated optical work. Cold
   Mie construction and rare-path MC variance are separate remaining costs;
   profile them separately and judge changes by time to a stable estimate.

---

## Possible directions

Ordered by what they would teach us per unit of effort, not by novelty.

### A. Polarisation (the largest missing physics)

The Rayleigh sky is strongly polarised — up to ~70 % at 90° scattering — with
neutral points (Babinet, Brewster, Arago) whose positions move with the sun and
with the aerosol.  This is:
* completely absent from the current scalar model;
* easy to *photograph* (a polarising filter and three exposures), so it is
  unusually well suited to validation;
* a genuine extension of the radiative transfer (Stokes vectors, a
  polarised phase matrix, and the multiple-scattering solver carrying four
  components instead of one).

Effort: large; a separate physics milestone once the present validation
questions have been scoped.

### B. Comparison with real photographs

Find photographs with known time, location, lens, exposure and white balance;
reproduce the same camera; compare chromaticity maps.  Attractive because it is
the validation a reader will trust most, but methodologically dirty: the camera
response, the tone curve, the white balance and the lens transmission all enter.
A careful version would need a calibration target in the frame.

### C. Three-dimensional clouds

Cumulus, silver linings, crepuscular rays, cloud shadows.  This is volumetric
rendering — a 3D grid, delta tracking, denoising — and it is a different project
from "the colour of the sky".  A thin 3D layer (cirrus with structure) might be
tractable; a cumulus field is not.

### D. Faster cold optics and twilight transport

For cold dusty scenes, Mie tables and moments dominate; warm/repeat optical
work is largely cached (M11).  Vectorised kernels or validated lookup tables
target that startup cost.  Deep-twilight MC has a different bottleneck: rare
light-carrying paths and high variance.  Better estimators, wavelength reuse
or GPU transport should be judged on time to a stable estimate.  A faster SOS
alone would not remove either bottleneck; profile each intended workload
before selecting its optimisation.

### E. Engineering

* **Interactive viewer**: a self-contained web page with a precomputed look-up
  table over (sun elevation × aerosol × ozone), letting a reader scrub through
  the parameter space.  Cheap to build, high delight-per-effort.
* **Video**: a proper time-lapse with exposure adaptation (the 26-frame GIF is
  a proof of concept).
* **A CLI/API clean-up**: presets are ad hoc; a declarative scenario format
  (`{sun: -4, atmosphere: volcanic(0.1), camera: fisheye}`) would make the
  experiments readable.
* **Packaging**: publish to PyPI, add a `--json` output for scripting.

### F. Physics extensions that are cheap

* Ice-crystal phase functions for cirrus (currently equivalent spheres, so no
  22° halo, no sundogs).
* A proper aureole: delta-fit instead of delta-M, and an instrument
  point-spread function for the solar disk.
* Airglow, so that −18° is not zero.
* Site-specific atmospheres (Paranal, Mauna Kea) to settle the twilight
  photometry question above.
* Spectral surface albedo instead of a grey Lambertian ground.

---

## What "done" would look like

For the original question — *what colour is the sky, and why?* — the model
reproduces many mechanisms. Quantitative closure still requires:

* one checked production solar source and stable twilight uncertainty;
* a measured clear-sky spectral radiance set to test the CIE fit at the limb;
* a controlled altitude-response comparison that separates site conditions,
  observable definition, transport approximations and sampling error (the
  domain truncation and raised-ground occlusion are fixed; the physical
  interpretation remains open);
* polarisation, which is the one large piece of physics still missing.

Everything else on this page is either a different project (3D clouds) or a
presentation concern (viewer, video, packaging).

### M19 — deep-twilight reference: budgets, gates, paired source correction (2026-11-04)

`task-deep-twilight-reference` turned the single-seed deep points into a
pre-registered reference: fixed scenario/medium/source contract on the
archived full81 finite-disk table, 16 seeds × {400k, 1.6M} photons/λ ×
{−9°, −12°} plus same-seed point-source pairing (3.51 h wall-clock), all
records append-only with per-record cost/hash provenance and a summary
that recomputes bit-consistently from raw (`--verify-only`).

Outcomes. Both depths are **budget-limited: unresolved** at the
pre-registered anchor scales: 1.6M conditional half-widths are 9–15 % per
anchor at −9° and 28–61 % at −12°, the top-0.1 % of paths carry the whole
higher-order signal (heavy-tail dominated), and −12° 400k contains
all-zero records (detection limits). The paired finite-disk source
correction is **identified** at −9° (+0.76/+0.97/+0.91 % at 450/550/650)
and at −12° 550 nm only (+0.63 %). An exactly-two-scattering candidate
estimator was validated against independent closures including the analog
MC's own per-path decomposition at −6°, then measured: narrower intervals
at −9° (550 nm half-width ratio 0.55 at 1.6M), no rescue at −12°. An
opt-in read-only diagnostics sink (`mc_radiance_split(diagnostics=True)`)
provides order×ground NEE accounting, per-path decomposition and tail
concentration; default-off is bit-identical. Discretisation refinements
(disk nodes, order cap, RR, fine grids, single-scattering quadrature) sit
below the sampling noise floor. Full pytest 338 passed; run_all baseline
unchanged (3 PASS / 3 FAIL); `validation/results.json` untouched.
Evidence: [DEEP_TWILIGHT_REFERENCE.md](twilight/DEEP_TWILIGHT_REFERENCE.md),
data + protocol + manifest in
`validation/data/deep_twilight_reference/`.

Open: closing the −12° anchors needs tail-defeating estimators (the
candidate mitigates, the rest dominates); curved diffuse transport and
continuous-wavelength quadrature remain explicit unknowns of the
reference contract.

### M20 — one production story, finite approximation branches (2026-11-08)

The architecture consolidation keeps four implemented routes behind
`rt.solver.solve`, not an arbitrary physics-switch framework. Solar numeric/time
input resolves once to the pre-refraction direction; `render.solve_dome` prepares
one `SkyResult`/`SkyPresentation`, then images and tables are pure consumers.
The unified finite-disk source is already production code; continuous diffuse
refraction remains a historical research route. The task-7 estimator/diagnostic
source described in M19 was **not** adopted into the production MC: only its
frozen data/docs were integrated; see the [retired index](../experiments/retired/README.md).

[APPROXIMATIONS](physics/APPROXIMATIONS.md) now separates medium, source,
propagation, refraction, orders, algorithm and numerical/statistical limits.
The [thin example](../experiments/approximations/README.md) actually ran 30°/3°
skies and three fixed low-budget daytime MC seeds in 121.73 s total, preserving
spectra/components, XYZ/xy/Lv, cost and diagnostics. It verifies shared-component
closure, not improved observational accuracy. The [catalogue](../figures/README.md)
contains four newly generated, inspected, common-exposure images with complete
commands and source version; all 26 earlier candidates were byte-preserved in
phenomenon/history topics rather than overwritten or re-certified.

Regression: full tree 526 passed; research-free tree **526 passed with no skip**
in its own uv environment. The latter removes every experiment/research package
and all tools except the one original maintenance file dynamically consumed by
the default test harness (`tools/make_scheduling_weights.py`); deleting it too
was tested and caused collection errors, so that invalid minimal-tree assumption
was corrected without copying code or deleting tests. Real CLI spectra/table/
PNG/summary closure and 32 source-check records agree across the two trees.
Source checks retain only the already-adjudicated RK direction failures;
run_all retains 2P/4F with all outputs byte-identical. Current runtime and the
**364.54 s full-suite performance guard breach** are recorded separately from
functional success in [FAST_SUITE](../validation/FAST_SUITE.md); this milestone
is not a new ≤300 s certification. No production/default-test/physical-baseline
or dependency change was made by the representation task.

The useful simplification is operational: render the README scene, read the
branch map and recompute the saved spectra, or locate historical work by its
fixed version. Deep-twilight tail coverage, absolute observation matching,
full curved propagation and current-suite performance remain explicit limits;
none was solved by re-labelling a picture.
