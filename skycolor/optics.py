"""Shared optical semantics: species listing and column algebra.

Every radiative-transfer backend needs the same three things: the ordered
species listing of an atmosphere (air, ozone, NO2, then one entry per
aerosol layer), the per-species column amounts along shell path lengths,
and the optical depth of a column vector.  They live here as the single
source, so backends stop importing each other's private helpers.

What is deliberately *not* here is discretisation: how a solver samples
path lengths (Gauss quadrature per shell, boundary transmittances, MC fine
grids and delta-tracking transport) stays with the solver -- see the task
plan for the split rationale.  The shared contract is the species ordering
and the column algebra; the numeric operation sequences below reproduce the
historical ``rt.single`` implementations exactly, so structural migrations
stay bit-exact (equivalence criterion A of the task plan).

The ``atmo`` argument is duck-typed: anything exposing ``n_air``/``n_o3``/
``n_no2``, ``aerosols`` (with ``aod550``/``profile``/``aerosol``) and the
``_sigma_*`` cross-section arrays works -- the frozen ``Medium`` is the
canonical such object, and this module is the one place that reads those
attributes.
"""
from __future__ import annotations

import numpy as np

from .constants import R_EARTH


def shell_species_weights(atmo, r_bounds: np.ndarray) -> np.ndarray:
    """Per-shell column-weight matrix, ``(n_shell, 3 + n_aerosol_layers)``.

    Column ``j`` holds the density of species ``j`` -- molecules m^-3 for
    the gases, 550 nm optical depth per metre for the aerosol layers --
    evaluated at the shell mid radius ``sqrt(r_lo * r_hi)``.  Multiplying
    shell path lengths [m] by this matrix yields column amounts in
    molecules m^-2 (gases) / optical depth (aerosols).
    """
    r_mid = np.sqrt(r_bounds[:-1] * r_bounds[1:])
    z_mid = (r_mid - R_EARTH) / 1e3
    cols = [atmo.n_air(z_mid), atmo.n_o3(z_mid), atmo.n_no2(z_mid)]
    for lay in atmo.aerosols:
        cols.append(lay.aod550 * lay.profile(z_mid) / 1000.0)
    return np.stack(cols, axis=1)


def species_extinction_weights(atmo, aerosol_ext=None) -> np.ndarray:
    """Extinction weight of every species at every wavelength.

    ``(n_species, n_wl)``: cross-sections [m^2] for the gases, the relative
    550-nm-normalised extinction coefficient for the aerosol layers.
    ``aerosol_ext`` may pass precomputed aerosol curves (as returned by
    ``Aerosol.relative_coefficients(atmo.wl)[0]``, e.g. the MC keeps one
    copy per trace); by default they are computed here.
    """
    if aerosol_ext is None:
        aerosol_ext = [lay.aerosol.relative_coefficients(atmo.wl)[0]
                       for lay in atmo.aerosols]
    return np.stack([atmo._sigma_ray, atmo._sigma_o3, atmo._sigma_no2]
                    + [np.asarray(e, dtype=float) for e in aerosol_ext])


def shell_species_columns(lengths: np.ndarray,
                          shell_densities: np.ndarray) -> np.ndarray:
    """Column amounts of every species along shell path lengths.

    ``lengths`` is ``(n_shell, n_points)`` [m] and ``shell_densities`` is
    ``(n_shell, n_species)`` as produced by :func:`shell_species_weights`;
    the result is ``(n_points, n_species)``.  Each species column is a
    separate matrix-vector product: the operation sequence is kept
    identical to the historical ``rt.single.species_columns`` so golden
    fixtures stay bit-exact -- a single matmul would reassociate the sums.
    """
    return np.stack([lengths.T @ shell_densities[:, j]
                     for j in range(shell_densities.shape[1])], axis=1)


def tau_from_columns(columns: np.ndarray, ext_weights: np.ndarray) -> np.ndarray:
    """Optical depth of per-species column amounts, ``(n_points, n_wl)``.

    ``columns`` is ``(n_points, n_species)`` and ``ext_weights`` is
    ``(n_species, n_wl)`` (use :func:`species_extinction_weights`).  The
    gas terms (air, ozone, NO2) are summed first, then the aerosol layers
    in registry order -- the same summation order, and hence the same
    rounding, as the historical ``rt.single._tau_from_columns``.
    """
    ext = np.asarray(ext_weights, dtype=float)
    if ext.ndim != 2:
        raise ValueError(f"ext_weights must be (n_species, n_wl), got {ext.shape}")
    tau = columns[:, 0:1] * ext[0]
    for j in range(1, ext.shape[0]):
        tau = tau + columns[:, j:j + 1] * ext[j]
    return tau


def contiguous_transpose(a: np.ndarray) -> np.ndarray:
    """C-contiguous transposed operand for matmul.

    macOS arm64 numpy wheels (2.5.4 verified) dispatch matmul to the system
    Accelerate BLAS, whose dgemm OVERREADS past the end of an operand
    buffer when the *second* operand is F-contiguous (a transposed view)
    and the inner dimension is k=3 -- the Cartesian-vector case every
    geometry matmul uses and the ``(rows, 3) @ (3, 3)`` shape of the
    colour-matrix transforms.  The overread is read-only (no value is
    corrupted) but SIGSEGVs whenever it crosses into an unmapped page,
    which is heap-layout dependent: intermittent in production, immediate
    and deterministic under libgmalloc.  Forcing the transposed operand
    C-contiguous selects the clean kernel; for the production row counts
    (m >= 64) the two kernels are bitwise-identical
    (tests/test_blas_layout_guard.py pins the contract and the site
    inventory, and ``scratchpad/task-mc-restrict-segv/findings.md``
    carries the probe evidence).
    """
    return np.ascontiguousarray(np.asarray(a).T)
