"""The observing scenario: where the observer stands and what the ground does.

A ``Scene`` bundles the three quantities that describe *where the model is
being evaluated from* rather than what the atmosphere is made of:

* ``observer_elevation_km`` -- the radius the RT backends integrate from;
* ``ground_albedo`` -- the spectral Lambertian albedo below the observer;
* ``refraction`` -- whether the direct-beam geometry bends rays.

It is frozen: a scene is replaced wholesale, never patched.  The spectral
grid lives in the :class:`~skycolor.medium.Medium`; ``ground_albedo`` is
stored as an array already broadcast to the medium's wavelength grid by the
caller (:class:`skycolor.atmosphere.Atmosphere` does this in its
constructor), so the two can only desynchronise through the facade's
public setters, which re-broadcast as needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scene:
    """The observing scenario (see the module docstring)."""

    observer_elevation_km: float = 0.0
    ground_albedo: np.ndarray | float = 0.1
    refraction: bool = True
