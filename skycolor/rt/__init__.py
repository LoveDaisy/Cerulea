from .single import sky_radiance, direct_sun_transmittance, species_columns  # noqa: F401

# Lazy re-exports of the authoritative entry (``rt.solver.solve``) and the
# result contract.  An eager import here would pull the full backend chain
# (multiple / twilight / mc) into every ``import skycolor``, which the
# cold-start work deliberately keeps off the plain import path -- so the
# names resolve on first access instead (PEP 562).
_LAZY_EXPORTS = {"solve", "sky_radiance_total", "SingleConfig", "SOSConfig",
                 "MCConfig", "SkyResult"}


def __getattr__(name):
    if name == "SkyResult":
        from .result import SkyResult
        return SkyResult
    if name in _LAZY_EXPORTS:
        from . import solver
        return getattr(solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_EXPORTS)
