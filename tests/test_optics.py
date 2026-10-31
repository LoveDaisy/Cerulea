"""Property tests for the shared species/column algebra (skycolor.optics).

The module is the single source of the species listing (air, ozone, NO2,
then one entry per aerosol layer) and the column algebra used by every RT
backend.  These tests pin the algebraic properties that must hold for any
atmosphere, independent of the solvers:

1. zero path lengths give zero columns and zero optical depth;
2. the column layout matches the species registry (order and units);
3. tau_from_columns is the exact weighted sum of the columns;
4. per-species contributions are exactly separable (a single non-zero
   column gives exactly that species' cross-section times the column).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor.aerosol import AngstromAerosol  # noqa: E402
from skycolor.atmosphere import (  # noqa: E402
    AerosolLayer,
    Atmosphere,
    ExponentialProfile,
)
from skycolor.constants import R_EARTH  # noqa: E402
from skycolor.optics import (  # noqa: E402
    shell_species_columns,
    shell_species_weights,
    species_extinction_weights,
    tau_from_columns,
)

WL = np.array([400.0, 550.0, 650.0])
N_POINTS = 4


@pytest.fixture
def atmo():
    """Rayleigh + ozone + NO2 + two aerosol layers: exercises the full registry."""
    return Atmosphere(
        WL,
        ozone_du=300.0,
        no2_column=1.0e16,
        aerosols=[
            AerosolLayer(AngstromAerosol(name="fine", alpha=1.2, ssa=0.9, g=0.7),
                         aod550=0.08, profile=ExponentialProfile(1.5)),
            AerosolLayer(AngstromAerosol(name="coarse", alpha=0.5, ssa=0.8, g=0.8),
                         aod550=0.12, profile=ExponentialProfile(3.0)),
        ],
    )


def _rng_lengths(atmo, seed=7):
    n_shell = len(atmo.r_bounds) - 1
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0e3, 5.0e4, size=(n_shell, N_POINTS))


def test_zero_path_gives_zero_tau(atmo):
    n_shell = len(atmo.r_bounds) - 1
    weights = shell_species_weights(atmo, atmo.r_bounds)
    cols = shell_species_columns(np.zeros((n_shell, N_POINTS)), weights)
    assert cols.shape == (N_POINTS, 3 + 2)
    assert np.all(cols == 0.0)
    assert np.all(tau_from_columns(cols, species_extinction_weights(atmo)) == 0.0)


def test_column_layout_matches_registry(atmo):
    """Columns are shell densities integrated along the given path lengths,
    in the registry order air / ozone / NO2 / aerosol layers."""
    lengths = _rng_lengths(atmo)
    weights = shell_species_weights(atmo, atmo.r_bounds)
    assert weights.shape == (len(atmo.r_bounds) - 1, 5)

    r_mid = np.sqrt(atmo.r_bounds[:-1] * atmo.r_bounds[1:])
    z_mid = (r_mid - R_EARTH) / 1e3
    expected = np.stack([atmo.n_air(z_mid), atmo.n_o3(z_mid), atmo.n_no2(z_mid)], axis=1)
    np.testing.assert_array_equal(weights[:, :3], expected)
    for j, lay in enumerate(atmo.aerosols):
        np.testing.assert_array_equal(weights[:, 3 + j],
                                      lay.aod550 * lay.profile(z_mid) / 1000.0)

    cols = shell_species_columns(lengths, weights)
    for j in range(5):
        np.testing.assert_array_equal(cols[:, j], lengths.T @ weights[:, j])


def test_tau_is_exact_weighted_sum(atmo):
    lengths = _rng_lengths(atmo)
    cols = shell_species_columns(lengths, shell_species_weights(atmo, atmo.r_bounds))
    ext = species_extinction_weights(atmo)
    assert ext.shape == (5, WL.size)
    np.testing.assert_array_equal(ext[0], atmo._sigma_ray)
    np.testing.assert_array_equal(ext[1], atmo._sigma_o3)
    np.testing.assert_array_equal(ext[2], atmo._sigma_no2)

    tau = tau_from_columns(cols, ext)
    # elementwise: tau[i, k] == sum_j cols[i, j] * ext[j, k], computed exactly
    manual = np.zeros((N_POINTS, WL.size))
    for j in range(5):
        manual += cols[:, j:j + 1] * ext[j][None, :]
    np.testing.assert_array_equal(tau, manual)


def test_species_contributions_are_separable(atmo):
    """Only one non-zero column -> tau is exactly that species' term."""
    ext = species_extinction_weights(atmo)
    for j in range(5):
        cols = np.zeros((N_POINTS, 5))
        cols[:, j] = 2.5
        tau = tau_from_columns(cols, ext)
        np.testing.assert_array_equal(tau, np.outer(cols[:, j], ext[j]))


def test_ext_weights_accept_precomputed_aerosol_curves(atmo):
    """Passing relative_coefficients explicitly must give the same matrix."""
    rel = [lay.aerosol.relative_coefficients(atmo.wl)[0] for lay in atmo.aerosols]
    np.testing.assert_array_equal(species_extinction_weights(atmo, rel),
                                  species_extinction_weights(atmo))
