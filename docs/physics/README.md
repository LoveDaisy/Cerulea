# Physics and numerical methods

These notes describe the production methods and their recorded checks. Current
API/approximation contracts are in [ARCHITECTURE.md](../ARCHITECTURE.md), current
physical-check limits in [STATUS.md](../STATUS.md).

| Document | Scope |
|---|---|
| [LAYER_STORY.md](LAYER_STORY.md) | The progressive layer ladder (Rayleigh → … → observation comparison); each rung's visible difference attributed against its pinned baseline |
| [APPROXIMATIONS.md](APPROXIMATIONS.md) | Common target, finite branches, real same-entry comparisons and interpretation limits |
| [SOLAR_SOURCE.md](SOLAR_SOURCE.md) | The unified finite-disk solar source: contract, per-backend consumption and validation entry points |
| [AEROSOL.md](AEROSOL.md) | Mie library, analytic moments, humidity growth and delta-M |
| [MULTIPLE_SCATTERING.md](MULTIPLE_SCATTERING.md) | Plane-parallel successive-orders method |
| [MC_REFERENCE.md](MC_REFERENCE.md) | Independent MC transport and estimator; shared-physics blind spots are declared in ARCHITECTURE |
| [REFRACTION.md](REFRACTION.md) | Production solar-path refraction and two-branch chord geometry |
| [CLOUD.md](CLOUD.md) | Homogeneous cloud deck and delta-Eddington method |

The [twilight topic](../twilight/README.md) carries the split-estimator engine and
source/transport audits. [Documentation guide](../README.md).
