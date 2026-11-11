"""AOD550 defines a material spectrum, independent of the query grid."""
from __future__ import annotations

import numpy as np
import pytest

from skycolor.aerosol import AngstromAerosol, LogNormalMode, MieAerosol, get_aerosol
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile


LIBRARY_NAMES = [
    "rural", "urban", "maritime", "desert_dust", "biomass_burning",
    "volcanic_stratospheric", "cirrus", "water_cloud",
]


@pytest.mark.slow(reason="cost: 12-worker parallel 2.5 s (cold-optics listing); fast-tier budget 2 s")
@pytest.mark.parametrize("name", LIBRARY_NAMES)
def test_mie_grid_partition_preserves_the_material_spectrum(name):
    aerosol = get_aerosol(name)
    wl = np.arange(380., 781., 5.)
    full = np.array(aerosol.relative_coefficients(wl))
    # The single-wavelength path is what twilight's MC anchor loops consume.
    single = np.array([aerosol.relative_coefficients([w]) for w in wl])[:, :, 0].T
    np.testing.assert_array_equal(single, full)
    for indices in [np.flatnonzero(wl != 550.), np.array([54, 14, 54])]:
        subset = np.array(aerosol.relative_coefficients(wl[indices]))
        np.testing.assert_array_equal(subset, full[:, indices])
    # Bracketing the reference wavelength does not make interpolated extinction
    # at 550 nm equal to extinction calculated at 550 nm.
    three = np.array(aerosol.relative_coefficients([450., 550., 650.]))
    np.testing.assert_array_equal(aerosol.relative_coefficients([450., 650.]),
                                  three[:, [0, 2]])


@pytest.mark.parametrize("wl", [[450.], [650.], [450., 650.], [650., 450.],
                                [380., 545., 555., 780.]])
def test_angstrom_coefficients_equal_the_analytic_aod550_law(wl):
    aerosol = AngstromAerosol(alpha=1.3, ssa=.94, ssa_slope=1e-4)
    wl = np.array(wl)
    ext, sca = aerosol.relative_coefficients(wl)
    truth = (wl / 550.) ** -1.3
    np.testing.assert_array_equal(ext, truth)
    np.testing.assert_array_equal(sca, truth * np.clip(.94 + 1e-4 * (wl - 550.), 0., 1.))


@pytest.mark.parametrize("kind", ["mie", "angstrom"])
def test_atmosphere_restrict_preserves_optical_coefficients(kind):
    aerosol = (get_aerosol("rural") if kind == "mie" else
               AngstromAerosol(alpha=1.3, ssa=.94, ssa_slope=1e-4))
    atmo = Atmosphere(np.array([450., 550., 650.]), aerosols=[
        AerosolLayer(aerosol, .12, ExponentialProfile(1.5))])
    full = atmo.layer_properties()
    heights = np.array([0., .037, 1., 2.4, 10.])
    ext, sca = atmo.aerosol_coefficients(heights)
    for index in [[0], [1], [2], [0, 2], [2, 0, 2]]:
        restricted = atmo._restrict(index)
        for key, values in restricted.layer_properties().items():
            if key != "z":
                np.testing.assert_array_equal(values, full[key][:, index], err_msg=key)
        got_ext, got_sca = restricted.aerosol_coefficients(heights)
        np.testing.assert_array_equal(got_ext, ext[:, index])
        np.testing.assert_array_equal(got_sca, sca[:, index])


def test_nonpositive_mie_reference_keeps_the_existing_unit_fallback():
    # No particles: the true reference cross section is zero.
    aerosol = MieAerosol(modes=[LogNormalMode(.2, 2., weight=0.)], n_radii=8)
    ext, sca = aerosol.relative_coefficients([450., 650.])
    np.testing.assert_array_equal(ext, np.zeros(2))
    np.testing.assert_array_equal(sca, np.zeros(2))
    empty = aerosol.relative_coefficients([])
    assert all(values.size == 0 for values in empty)
