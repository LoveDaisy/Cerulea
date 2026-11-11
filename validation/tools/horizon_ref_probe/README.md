# Horizon reference probe (task-horizon-ref-probe, 2026-11-11)

G0 source-screening record for the check3 sun-side horizon/zenith attribution
question.  The check3 h=45 deg sun-side horizon/zenith excess (model 8.815 vs
CIE 110-1994 3.652, +141%, `validation/results.json`) has two live candidate
root causes -- a real model-side first-order horizon brightening, or a CIE 110
gradation-curve domain-edge artefact -- and the pre-registered discriminator
was an independent measured sun-side near-horizon H/Z ratio with a zenith in
the same scan and same-field turbidity.  This directory persists the outcome
of the frozen source-screening step: **no public original measurement
satisfies the admission contract within the screening budget, so gate G0
fails and the verdict is U (inadjudicable) -- stop-loss**.  No matched model
runs and no A/B/C attribution were performed (the plan forbids attribution
computation without a qualified source); `skycolor/`, `validation/results.json`
and check3 metrics are untouched by this task.

## Files

* `sources.json` -- the machine-authoritative screening record: the frozen
  admission contract, the 12-candidate screening table with per-candidate
  verdicts and reasons, reference-side independence notes, actual screening
  costs, and the domain-gap statement.  All numbers quoted in
  `validation/RULER_AUDIT.md` (section "太阳侧近地平独立实测对照探针") and in
  the check3 note reproduce from this JSON.
* `observations.csv` -- observation-level extracts for the candidates whose
  full text was actually read (Granada 2001 geometry points; Wuttke 2006
  anti-solar H/Z scans).  Every row carries its admission status; no
  qualifying observation exists, so no row is admitted to main adjudication.

## Admission contract (frozen, pre-registered)

Restated from the reviewed task plan (sections 3.1/3.3; constants unchanged
at implementation time; `sources.json` is the machine copy):

* original paper, daytime clear sky (solar elevation > 0 deg),
* sun-side direction |relative azimuth| <= 5 deg, view elevation in
  [0 deg, 5 deg] at its actual direction/FOV (never extrapolated to 0 deg),
* zenith in the same scan, same-field measured turbidity/AOD,
* photopic luminance or full-visible-band spectral radiance (narrow-band
  ratios are auxiliary only),
* reference-side independence from the CIE 110 fitting corpus with a recorded
  basis (post-1994 new acquisition, or an original held-out record;
  publication date alone is insufficient),
* G0: >= 1 qualified scan; |ln(M/C)| >= ln(1.5) stable over the uncertainty
  set; observed-error half-width u <= ln(1.2) and |ln(M/C)| >= 3u.

## Screening outcome

12 candidates screened (6 full texts read, plus abstract/table-level and
citation-level screens), 13 searches, 6 full-text acquisitions, ~45 min wall
clock -- within the 12-candidate / 90-min pre-registered budget.  Zero
qualified.  The closest candidate:

* **Granada 2001** (Hernandez-Andres, Romero, Lee; JOSA A 18, 412;
  DOI 10.1364/JOSAA.18.000412) -- the only public campaign with the right
  geometry: solar-meridian (f_rel = 0 deg) scans that include the zenith, with
  additional sun-side view-elevation points at h = 5 deg and h = 10 deg
  (confirmed visually in Figs. 5(a) and 7(a)), full 380-780 nm spectra, and a
  defensible independence basis (1998 new acquisition).  It fails the
  same-field turbidity gate (no measured AOD; qualitative mention only) and
  publishes chromaticity figures without numeric luminance values, so no H/Z
  ratio can be extracted from the original either.  Auxiliary only, per
  contract.
* The remaining candidates fail geometry (anti-solar principal plane
  [Wuttke 2006], horizon-only [Knestrick & Curcio 1967], sun-relative only
  [ESCAPE Dome C 2024], no near-horizon points [NBSIR 85-3251],
  zenith-only/twilight [Koomen 1952, Tousey & Hulburt 1947 measured part]),
  readability within budget (Lee 1994, Packer & Lock 1951, Knestrick &
  Curcio 1967 -- paywalled, no open copy), or spectral coverage (UV/narrow
  band class).

## Domain gap

No public original dataset combines a sun-side near-horizon direction
(|Daz| <= 5 deg, elevation <= 5 deg), the zenith in the same scan, same-field
AOD/turbidity, and photopic/full-visible coverage.  Daylighting-lineage
scans (the CIE 110 fitting corpus lineage) conventionally stop ~6 deg above
the horizon; atmospheric-optics campaigns favour the anti-solar principal
plane; site surveys sample sun-relative distances without zenith.  Until such
a source exists, the +141% attribution stays open: the measured evidence
needed to separate "model-side excess" from "CIE 110 domain-edge artefact"
does not exist in the screened public literature, and neither branch may be
claimed.  check3 therefore carries a domain note ("太阳侧近地平缺独立实测对照")
instead of a verdict change; `validation/RULER_AUDIT.md` holds the full
statement.  Future options (owner adjudication): request the Granada spectra
plus any concurrent aerosol data from the authors, or locate/commission a
scan with concurrent AOD; this file's admission contract is the frozen gate
for any such re-run.

## Provenance and reproduction

Downloaded originals live under `scratchpad/task-horizon-ref-probe/sources_raw/`
(gitignored; copyright).  Per-candidate acquisition URLs, access dates and
SHA-256 hashes are recorded in `sources.json`.  To re-verify a hash:

```bash
curl -sL -o /tmp/h2001.pdf "http://www.ugr.es/~colorimg/pdfs/josaa_2001_412.pdf"
shasum -a 256 /tmp/h2001.pdf   # f97b21ea...828a0319c (see sources.json)
```

Paywalled originals (Lee 1994, Packer & Lock 1951, Knestrick & Curcio 1967,
Tousey & Hulburt 1947 full text) were not downloaded; their screening level
is recorded honestly in `sources.json` and they do not enter any verdict.

## Deliberate omissions

No `probe.py` / `verify.py` / `preregistered.json` exist here: the probe
machinery (matched runs, classifier, closure checks) is specified in the task
plan for the qualified-source path only, and building it without a single
qualifying source would be an empty framework plus an unrunnable attribution
pipeline.  If a qualified source is later adjudicated by the owner, bootstrap
a new task and re-freeze the gates from this directory's contract.
