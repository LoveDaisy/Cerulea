"""Named atmospheric scenarios.

Every preset returns a fully configured :class:`skycolor.atmosphere.Atmosphere`
given a spectral grid.  They are meant to be starting points: the interesting
question is what happens between them.
"""
from __future__ import annotations

import numpy as np

from .atmosphere import AerosolLayer, Atmosphere, ExponentialProfile, GaussianProfile
from .aerosol import (
    DustAerosol,
    HazeAerosol,
    HinterlandAerosol,
    MaritimeAerosol,
    SmokeAerosol,
    UrbanAerosol,
)


def pristine(wl, ozone_du: float = 300.0) -> Atmosphere:
    """Very clean air: Rayleigh + ozone only."""
    return Atmosphere(wl, ozone_du=ozone_du, aerosols=[])


def clear(wl, aod: float = 0.05, ozone_du: float = 300.0) -> Atmosphere:
    """Clean continental background."""
    return Atmosphere(wl, ozone_du=ozone_du,
                      aerosols=[AerosolLayer(HinterlandAerosol(), aod,
                                             ExponentialProfile(1.5))])


def hazy(wl, aod: float = 0.30, rh: float = 0.7, ozone_du: float = 300.0) -> Atmosphere:
    """Humid boundary layer: the classic milky horizon."""
    return Atmosphere(wl, ozone_du=ozone_du,
                      aerosols=[AerosolLayer(HazeAerosol(rh), aod,
                                             ExponentialProfile(1.2))])


def urban(wl, aod: float = 0.35, ozone_du: float = 320.0) -> Atmosphere:
    return Atmosphere(wl, ozone_du=ozone_du,
                      aerosols=[AerosolLayer(UrbanAerosol(), aod,
                                             ExponentialProfile(1.0))])


def dusty(wl, aod: float = 0.6, elevated: bool = True) -> Atmosphere:
    """Desert dust: a deep boundary layer plus an elevated residual layer."""
    layers = [AerosolLayer(DustAerosol(), 0.6 * aod, ExponentialProfile(2.0))]
    if elevated:
        layers.append(AerosolLayer(DustAerosol(), 0.4 * aod,
                                   GaussianProfile(3.5, 1.5)))
    return Atmosphere(wl, ozone_du=290.0, aerosols=layers)


def smoky(wl, aod: float = 0.5, ozone_du: float = 320.0) -> Atmosphere:
    return Atmosphere(wl, ozone_du=ozone_du,
                      aerosols=[AerosolLayer(SmokeAerosol(), aod,
                                             ExponentialProfile(2.0))])


def marine(wl, aod: float = 0.12) -> Atmosphere:
    return Atmosphere(wl, ozone_du=280.0,
                      aerosols=[AerosolLayer(MaritimeAerosol(), aod,
                                             ExponentialProfile(0.8))])


PRESETS = {
    "pristine": pristine,
    "clear": clear,
    "hazy": hazy,
    "urban": urban,
    "dusty": dusty,
    "smoky": smoky,
    "marine": marine,
}
