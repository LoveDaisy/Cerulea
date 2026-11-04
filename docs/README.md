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
| [physics/](physics/README.md) | Aerosol/Mie, clouds, refraction, successive orders and independent MC methods |
| [twilight/](twilight/README.md) | Production twilight engine, historical profile sweeps and M13–M18 audit sequence |
| [validation/results.json](../validation/results.json) | Latest committed six-check regression snapshot, including existing failures |
| [REPORT.md](../validation/REPORT.md) | Original single-scattering baseline; the five-check table is historical |
| [ADDENDUM.md](../validation/ADDENDUM.md) | Chronological changes and locally marked superseded interpretations |
| [SECOND_DATASET.md](../validation/SECOND_DATASET.md) | Koomen transcription and historical model comparisons, with current audit pointers |

Current summaries carry conclusions and links. Topic reports own methods,
conditions, uncertainties and raw-evidence references; preserve historical
measurements with their baseline. Update STATUS and the relevant contract/report
when behavior changes, and ROADMAP when a milestone completes. Local `scratchpad/`
holds workflow records rather than another project-status document.
