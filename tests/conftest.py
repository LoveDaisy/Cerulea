"""Shared test fixtures.

Currently holds only the transitional thin-atmosphere factory for the
architecture-refactor task (plan Step 3a).  It is scheduled for deletion in
Step 3b, when the real mechanism -- per-species density callables injected
at construction -- replaces it.
"""
from __future__ import annotations

import pytest

from skycolor.atmosphere import Atmosphere


@pytest.fixture
def thin_atmosphere_factory():
    """Build atmospheres with non-physical scaled cross-sections.

    **Test-only transitional fixture (plan Step 3a; deleted in Step 3b).**
    The returned atmosphere is NOT physical: its molecular cross-sections
    are rescaled after construction by patching the frozen Medium's derived
    arrays.  Such objects must never enter ``aerosol.shared()``, any
    identity-keyed cache, or any physical validation -- they exist so the
    "thin atmosphere" tests can probe solver behaviour in a weakly scattering
    limit now that ``_sigma_*`` are read-only.
    """
    def make(wl, ozone_du=300.0, aerosols=(), scale_ray=1.0, scale_o3=1.0,
             scale_no2=1.0) -> Atmosphere:
        atmo = Atmosphere(wl, ozone_du=ozone_du, aerosols=list(aerosols))
        medium = atmo._medium
        object.__setattr__(medium, "_sigma_ray", medium._sigma_ray * scale_ray)
        object.__setattr__(medium, "_sigma_o3", medium._sigma_o3 * scale_o3)
        object.__setattr__(medium, "_sigma_no2", medium._sigma_no2 * scale_no2)
        return atmo

    return make
