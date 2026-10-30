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
from 1500× to 1.45×.
*Code:* `rt/twilight.py`, `docs/TWILIGHT.md`.

### M6 — refraction, taken seriously

Ray-traced refraction (33′ at the horizon, matching Sæmundsson above 5°), then
the full radiative-transfer treatment: the Bouguer invariant `n r sin θ = a`
turns the curved path length into the same closed form as a straight chord with
`b_k = a/n_k`, so the whole correction costs one division per shell rather than
a ray trace per photon.

*Validated:* the per-shell formula against numerical path integration to 0.1 %
(1661 km vs 1663 km for a horizontal ray).
*Code:* `refraction.py`, `data/refraction.csv`, `docs/REFRACTION.md`.

### M7 — a cloud deck

A plane-parallel homogeneous cloud layer: Mie droplets, delta-M, a
closed-form delta-Eddington two-stream and a spherical terminator that decides
which part of the deck is still lit.

*Validated:* albedo against an independent Monte Carlo within 0.1–0.7 %
(`g = 0`) and 1 % (`g = 0.85`, τ ≥ 5); `R + T + direct = 1` to 10⁻¹².
*Code:* `cloud.py`, `rt/cloud.py`, `docs/CLOUD.md`.

### M8 — the physics of red skies and blue suns

Two questions that turned out to have clean answers:

* **Red afterglow** is not a low-level haze effect.  A boundary-layer aerosol
  makes a sunset *dimmer and whiter*; the glow needs a scattering layer above
  the Earth's shadow — stratospheric sulphate after a volcano, or cirrus.
* **The blue sun** needs a non-monotonic extinction spectrum, which an Ångström
  power law can never produce and a narrow Mie size distribution can.

*Figures:* `08_volcanic_twilight.png`, `09_blue_sun.png`.

---

## Open questions that matter now

These are not roadmap items, they are the things we do not currently
understand.  Each is worth resolving before adding new physics.

1. **What does "solar elevation" mean in the twilight reference data?**  With
   geometric shadows the model matches Patat et al. at −6° to 1.5×; with full
   refraction it is 3.3× high.  Either the reference uses apparent altitudes, or
   something else in the twilight model is ~40 % too bright.  This single
   convention decides whether refraction should be on by default.
2. **The limb.**  The model's horizon is ≈ 2× brighter than the CIE 110 fit.
   The CIE formula is a daylighting standard, not a physical model, so this may
   be a reference problem — but a measured clear-sky luminance scan would settle
   it, and we do not have one.
3. **The V band vs the photopic integral.**  Twilight spectra are strongly
   coloured; comparing a photopic integral with a Johnson-V magnitude is a
   10–20 % systematic that nobody has quantified here.
4. **Performance.**  A dusty atmosphere takes ~100 s for one multiple-scattering
   solve.  That is a documentation-grade solver, not an interactive one.

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
* a decided convention for the twilight reference;
* polarisation, which is the one large piece of physics still missing.

Everything else on this page is either a different project (3D clouds) or a
presentation concern (viewer, video, packaging).
