# Physics and numerical methods

These notes describe the production methods and their recorded checks. Current
API/approximation contracts are in [ARCHITECTURE.md](../ARCHITECTURE.md), current
physical-check limits in [STATUS.md](../STATUS.md).

| Document | Scope |
|---|---|
| [AEROSOL.md](AEROSOL.md) | Mie library, analytic moments, humidity growth and delta-M |
| [MULTIPLE_SCATTERING.md](MULTIPLE_SCATTERING.md) | Plane-parallel successive-orders method |
| [MC_REFERENCE.md](MC_REFERENCE.md) | Independent MC transport and estimator; shared-physics blind spots are declared in ARCHITECTURE |
| [REFRACTION.md](REFRACTION.md) | Production solar-path refraction and two-branch chord geometry |
| [CLOUD.md](CLOUD.md) | Homogeneous cloud deck and delta-Eddington method |

The [twilight topic](../twilight/README.md) carries the split-estimator engine and
source/transport audits. [Documentation guide](../README.md).
