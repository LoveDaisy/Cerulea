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
*Code:* `mie.py`, `aerosol.py`, `docs/AEROSOL.md`.

### M5 — the spherical twilight engine

The deep twilight needs light that travels *sideways* for hundreds of
kilometres.  A split-estimator Monte Carlo (exact deterministic first order +
Monte Carlo for orders ≥ 2) on a coarse angular grid closes the gap.

*Validated:* −9° goes from 24× too dark to 1.39× of Patat et al. (2006); −12°
from 1500× to 1.45×.  (M10 later moved these to 0.54× and 0.37×: the 1.39/1.45
were the one-branch chord.)
*Code:* `rt/twilight.py`, `docs/TWILIGHT.md`.

### M6 — refraction, taken seriously (**now the default**)

Ray-traced refraction (33′ at the horizon, matching Sæmundsson above 5°), then
the full radiative-transfer treatment: the Bouguer invariant `n r sin θ = a`
turns the curved path length into the same closed form as a straight chord with
`b_k = a/n_k`, so the whole correction costs one division per shell rather than
a ray trace per photon.

*Validated:* the per-shell formula against numerical path integration to 0.1 %
(1661 km vs 1663 km for a horizontal ray); the apparent-elevation table against
the ray tracer.
*Code:* `refraction.py`, `data/refraction.csv`, `docs/REFRACTION.md`.
*Consequence:* twilight is 2–5× brighter.  The absolute level that this exposed
was later traced to a two-branch chord bug (M10), not to refraction itself.

### M7 — a cloud deck

A plane-parallel homogeneous cloud layer: Mie droplets, delta-M, a
closed-form delta-Eddington two-stream and a spherical terminator that decides
which part of the deck is still lit.

*Validated:* albedo against an independent Monte Carlo within 0.1–0.7 %
(`g = 0`) and 1 % (`g = 0.85`, τ ≥ 5); `R + T + direct = 1` to 10⁻¹².
*Code:* `cloud.py`, `rt/cloud.py`, `docs/CLOUD.md`.

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
[the altitude audit](TWILIGHT_ALTITUDE_AUDIT.md).

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
[TWILIGHT_ALTITUDE_AUDIT.md](TWILIGHT_ALTITUDE_AUDIT.md).

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
reported separately in [the audit](TWILIGHT_ALTITUDE_AUDIT.md).

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

## Open questions that matter now

These are not roadmap items, they are the things we do not currently
understand.  Each is worth resolving before adding new physics.

1. **Resolved, but the old answer was a bug artifact: the solar-altitude
   convention.**  The previous version of this page found H1 and H2 nearly
   degenerate.  That was computed with the one-branch chord (M10), which
   over-brightened twilight by ~2× and happened to compensate the 0.5° axis
   shift.  After the fix the two readings are far apart (ratios to Koomen):

   | reading | −3° | −6° |
   |---|---|---|
   | H1: reference axis is *true* altitude, refraction on | 0.93× | **0.69×** |
   | H2: reference axis is *apparent* altitude, refraction on | — | 0.29× |
   | geometric model (refraction off, true axis) | 0.75× | 0.46× |

   H2 is now much too dark.  The references almost certainly tabulate the
   **true** (ephemeris) solar altitude, so use H1; refraction stays on by
   default, and the geometric mode is kept only to reproduce old numbers.  The
   convention is now a ~10 % systematic, not a factor of two.
2. **The twilight level was a chord bug, not a site effect and not a physical
   excess.**  A second, independent, *photopic* measurement — Koomen et al.
   (1952), JOSA 42, 353, Maryland, 30 m — lets the model be compared like for
   like at sea level.  With the two-branch chord (M10) the −6° zenith is
   **0.69×** Koomen (deterministic) or **0.85×** (spherical engine), and
   0.86× / 1.07× Patat.  The old 2.20× (and the 1.14× geometric before it)
   were artifacts: the missing near-side perigee→point segment made every
   grazing ray too bright, and the geometric mode's missing refraction partly
   cancelled it.  The residual 15–30 % is inside the 1950s absolute
   calibration (±0.2–0.4 mag) and the plane-parallel multiple-scattering
   approximation; it is no longer a factor-of-two problem.  See
   `validation/SECOND_DATASET.md` and the sixth addendum of
   `validation/ADDENDUM.md`.

   **The altitude comparison remains unresolved, but its construction defect
   is now demonstrated.**  M12 fixed domain truncation and double scaling;
   M13 additionally made solar occlusion respect that raised ground.  The
   deterministic high-site/sea-level ratio at −6° changes from the historical
   1.208× to 1.027×.  The old low-budget engine ratio 1.01× also predates the
   occlusion fix; updated multi-seed results are in
   [the audit](TWILIGHT_ALTITUDE_AUDIT.md).  An observed station ratio is not a
   controlled altitude-only target: the 0.80× Maryland/Paranal comparison mixes
   studies and observables, while Koomen's same-instrument 0.37× comparison
   also changes season and transparency.  Transport approximations, sampling
   uncertainty and site conditions remain to be separated before attributing
   the residual to missing low-altitude multiple scattering.

3. **The limb.**  The model's horizon is ≈ 2× brighter than the CIE 110 fit.
   The CIE formula is a daylighting standard, not a physical model, so this may
   be a reference problem — but a measured clear-sky luminance scan would settle
   it, and we do not have one.
4. **The V band vs the photopic integral.**  Twilight spectra are strongly
   coloured; comparing a photopic integral with a Johnson-V magnitude is a
   10–20 % systematic that nobody has quantified here.
5. **Performance.**  The bottleneck is Mie pre-processing, not the
   multiple-scattering solve: a cold dusty ``delta_m(32)`` costs ~40 s on
   the reference machine (phase tables + analytic Legendre moments), while
   the transfer itself takes seconds.  The per-wavelength state cache now
   holds the whole 81-wavelength working grid (capacity 8 → 256) and the
   preset aerosols are shared across atmospheres, so warm and repeat calls
   are nearly free: warm single clear 1 dir 0.353 → 0.003 s, a second
   ``delta_m_properties(32)`` 6.8 → 0.002 s, a second ``presets.dusty``
   build ~41 s → < 1 ms (``tools/perf_bench.py``; all values bit-identical
   to the pre-change code).  Cold starts still pay the full Mie build and
   need vectorised kernels or a lookup table.

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

For the original question — *what colour is the sky, and why?* — the project is
close.  What remains is not more phenomena but **tighter error bars**:

* a measured clear-sky spectral radiance set to replace the CIE fit at the limb;
* a controlled altitude-response comparison that separates site conditions,
  observable definition, transport approximations and sampling error (the
  domain truncation and raised-ground occlusion are fixed; the physical
  interpretation remains open);
* polarisation, which is the one large piece of physics still missing.

Everything else on this page is either a different project (3D clouds) or a
presentation concern (viewer, video, packaging).
