"""The result contract of the solver entry point.

``SkyResult`` carries the spectral radiance together with everything needed
to trace what produced it: the legacy method name and the canonical
approximation that ran, the resolved request options, the statistical error
of Monte Carlo parts, the component decomposition where one exists, backend
diagnostics, and the *actual* physics/numerics of every computed component.
Statistical error, convergence and physical approximation stay *separate*
fields -- they are different kinds of uncertainty and are never merged into
a single confidence number.
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
        Legacy backend name that served the request: ``"clear"``,
        ``"twilight"`` or ``"cloud"``.  Kept for compatibility; the canonical
        approximation is :attr:`approximation`.
    options : dict
        The resolved request (sun position, solver toggles, quality,
        seed, ...).  What the solver actually used, not a copy of the call.
    stderr : ndarray or None
        Monte Carlo parts only (twilight/spherical-mc; None elsewhere).  It
        is a *composite* error estimate: the MC standard error propagated
        through the angular smoother and the wavelength interpolation, with
        the smoother's systematic resolution bias (a 3 % floor,
        ``rt.twilight.MS_BIAS``) added in quadrature.  It is therefore not a
        pure sample standard error and *not* a total physical error of the
        model; systematic approximations of the physics stay outside it.
        Deterministic paths return None.  The raw per-anchor MC statistics
        (before smoothing) are in ``diagnostics["coarse_errors"]``.
    components : dict or None
        Component decomposition when one exists (cloud:
        ``clear`` / ``cloud`` / ``transmittance`` / illumination / optics;
        clear combos: the selected ``single`` and ``multiple`` arrays with
        disabled terms ``None``).  ``components`` reproduces ``radiance``
        exactly; extra sub-components recorded for inspection are not
        additional sums.
    diagnostics : dict
        Backend diagnostics: scattering ``orders`` summed and ``converged``
        flag for the SOS solver, ``undetected_wl`` and grid info for the
        twilight Monte Carlo, ...
    approximation : str
        Canonical approximation that ran: ``"spherical-single"``,
        ``"plane-sos"``, ``"hybrid-sos"`` or ``"spherical-mc"``.
    approximations : dict
        Per computed component, the kernel that produced it and that
        kernel's standing approximations (finite-disk source, geometry,
        propagation convention, delta-M/moment truncation, horizontal-field
        coverage).  Only components that actually ran are recorded.
    effective : dict
        Effective numeric settings per kernel group (``single`` / ``sos`` /
        ``mc``) -- the discretisation *after* kernel-side clamping, plus the
        iteration/statistics outcome.  Request values live in ``options``;
        this field records what the kernels actually used.
    """

    radiance: np.ndarray
    wl: np.ndarray
    method: str
    options: dict = field(default_factory=dict)
    stderr: np.ndarray | None = None
    components: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    approximation: str = ""
    approximations: dict = field(default_factory=dict)
    effective: dict = field(default_factory=dict)
