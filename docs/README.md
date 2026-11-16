# Documentation guide

Start with the project-wide documents below. Topic folders contain the detailed
methods, experimental conditions and historical evidence.

## Project-wide documents

| Document | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Current production behavior and validation limits |
| [SUMMARY.zh.md](SUMMARY.zh.md) | Chinese overview of the model and conclusions |
| [ROADMAP.md](ROADMAP.md) | Completed milestones, open questions and proposed next work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module boundaries, API contracts and approximation limits |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Branch workflow, change verification and local merge procedure |
| [REPORT.md](REPORT.md) | Physics narrative and recorded figures; benchmark values retain their original conditions |

Setup and run commands live in [the project README](../README.md) and
[RUNNING.md](../RUNNING.md). Agent entry rules live in [AGENTS.md](../AGENTS.md).

## Topics and evidence

| Location | Scope |
|---|---|
| [Mainline story](physics/MAINLINE.zh.md) | Reader-level chapter arc (Chinese) from coarse physics to the produced image, with figures |
| [Layer story](physics/LAYER_STORY.md) | The progressive ladder: each rung adds one ingredient from Rayleigh to the measured sky; figures and data via the [layerstory theme](../experiments/layerstory/README.md) |
| [Approximation story](physics/APPROXIMATIONS.md) | Shared target, branch map and actual controlled comparison; [production figures](../figures/README.md) own versions/commands |
| [physics/](physics/README.md) | Aerosol/Mie, clouds, refraction, successive orders and independent MC methods |
| [twilight/](twilight/README.md) | Production twilight engine, historical profile sweeps and M13–M18 audit sequence |
| [validation/results.json](../validation/results.json) | Latest committed six-check regression snapshot, including existing failures |
| [REPORT.md](../validation/REPORT.md) | Original single-scattering baseline; the five-check table is historical |
| [ADDENDUM.md](../validation/ADDENDUM.md) | Chronological changes and locally marked superseded interpretations |
| [SECOND_DATASET.md](../validation/SECOND_DATASET.md) | Koomen transcription and historical model comparisons, with current audit pointers |
| [FAST_SUITE.md](../validation/FAST_SUITE.md) | Two-tier suite ledger: tier design, measured shapes, scheduling weights and acceptance pairs |
| [Fine-grid solve cost](performance/FINE_GRID_SOLVE_COST.md) | 4/6/8-worker production-chain calibration, byte checks and failed ≤8h admission; no retained production optimization |
| [RULER_AUDIT.md](../validation/RULER_AUDIT.md) | Adjudication matrix for the physical checks: which FAILs are contract gaps vs measurement questions |

Current summaries carry conclusions and links. Topic reports own methods,
conditions, uncertainties and raw-evidence references; preserve historical
measurements with their baseline. Update STATUS and the relevant contract/report
when behavior changes, and ROADMAP when a milestone completes. Local `scratchpad/`
holds workflow records rather than another project-status document.

Language convention: reader-facing walkthroughs and the AI entry ([AGENTS.md](../AGENTS.md))
are Chinese; method contracts, code and docstrings are English.
