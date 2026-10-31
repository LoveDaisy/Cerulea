"""The result contract of the solver entry point.

``SkyResult`` carries the spectral radiance together with everything needed
to trace what produced it: the method that ran, the resolved request
options, the statistical error of Monte Carlo parts, the component
decomposition where one exists, and backend diagnostics.  Statistical
error, convergence and physical approximation stay *separate* fields --
they are different kinds of uncertainty and are never merged into a single
confidence number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SkyResult:
    """Spectral sky radiance plus its provenance.

    Attributes
    ----------
    radiance : ndarray ``(n_view, n_wl)``
        Total spectral radiance [W m^-2 sr^-1 nm^-1].
    wl : ndarray ``(n_wl,)``
        Wavelength grid [nm].
    method : str
        Which backend served the request: ``"clear"``, ``"twilight"`` or
        ``"cloud"``.
    options : dict
        The resolved request (sun position, solver toggles, quality,
        seed, ...).  What the solver actually used, not a copy of the call.
    stderr : ndarray or None
        1-sigma Monte Carlo statistical error of ``radiance``
        (twilight method only; None elsewhere).  Excludes systematic and
        physical-model error by construction.
    components : dict or None
        Component decomposition when the backend provides one (cloud:
        ``clear`` / ``cloud`` / ``transmittance`` / illumination / optics).
    diagnostics : dict
        Backend diagnostics: scattering ``orders`` summed and ``converged``
        flag for the SOS solver, ``undetected_wl`` and grid info for the
        twilight Monte Carlo, ...
    """

    radiance: np.ndarray
    wl: np.ndarray
    method: str
    options: dict = field(default_factory=dict)
    stderr: np.ndarray | None = None
    components: dict | None = None
    diagnostics: dict = field(default_factory=dict)
