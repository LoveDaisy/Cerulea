# Roadmap

*Where the project is, and where it could go.  Written to be read two months
from now.*

The organising question has not changed: **ignoring clouds and polarisation,
can we predict the colour of the sky from first principles, and can we say how
confident we are?**  Everything below is measured against that.

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

### M8b — the twilight question closed by a second dataset

A single reference cannot separate a model error from a reference bias.  A
second, independent, *photopic* twilight measurement (Koomen et al. 1952,
sea level) showed that the model is 1.14× the measurement like-for-like, and
that the remaining factor is the 2400 m altitude of the original reference.
(M10 later showed the 1.14 was a partial cancellation; see below.)
The search also turned up the one thing that is wrong: the model has almost no
altitude response.  See `validation/SECOND_DATASET.md`.

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

   **The defect the second dataset exposed is still open, and the chord fix
   changed its sign.**  `at_altitude` has been rewritten as a consistent
   truncation (architecture-refactor Step 3c): the layer grid starts at the
   site, the ground sits there, and nothing is renormalised, so all three
   backends share one ground.  Under that construction the deterministic model
   is 1.21× *brighter* at 2.4 km and the spherical engine 1.01×, while the
   data (Koomen Maryland 30 m vs Patat Paranal 2400 m; Patat's own 961 vs
   743 hPa pair) are 0.80× *darker* — still the wrong sign, with no
   construction-degree-of-freedom left.  The missing low-atmosphere multiple
   scattering remains the physical candidate.  Open item: the refraction
   shadow thresholds are still built for sea-level grazing rays.

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

Effort: large.  It is the natural "M9".

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

### D. A faster multiple-scattering solver

The plane-parallel SOS is the bottleneck for dusty and cloudy scenes.  Options:
vectorise harder, use a lower-order two-stream for the diffuse field, or move
the whole solver to the GPU.  This is what would make an interactive viewer
possible.

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
* an altitude-dependent twilight with the right *sign* (the chord fix resolved
  the absolute level, but the model is still brighter at 2.4 km where the data
  are darker; `at_altitude` needs auditing first);
* polarisation, which is the one large piece of physics still missing.

Everything else on this page is either a different project (3D clouds) or a
presentation concern (viewer, video, packaging).
