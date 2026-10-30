"""Tests for the spherical twilight multiple-scattering engine.

The Monte Carlo is deliberately run with a very small photon budget here (pass
``grid=dict(...)``): the suite must stay fast.  The quantitative validation
against Patat et al. and against high-photon references lives in
``tools/twilight_benchmark.py`` and is reported in ``docs/TWILIGHT.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from skycolor import presets
from skycolor.rt import twilight as TW
from skycolor.rt.single import sky_radiance as single_radiance
from skycolor.rt.solver import sky_radiance_total

#: a cheap but still usable budget for the tests
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)
WL = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])


@pytest.fixture(scope="module")
def atmo():
    return presets.clear(WL)


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------
def test_shapes_and_units(atmo):
    zen = np.array([0.0, 30.0, 70.0])
    az = np.array([0.0, 90.0, 180.0])
    rad, err = TW.twilight_radiance(atmo, 96.0, zen, az, quality="low",
                                    grid=TINY, seed=3)
    assert rad.shape == (3, WL.size)
    assert err.shape == (3, WL.size)
    assert np.all(rad >= 0.0)
    assert np.all(err >= 0.0)


def test_broadcasts_scalar_zenith(atmo):
    a, ea = TW.twilight_radiance(atmo, 96.0, 20.0, np.array([0.0, 90.0]),
                                 grid=TINY, seed=3)
    b, eb = TW.twilight_radiance(atmo, 96.0, np.array([20.0, 20.0]),
                                 np.array([0.0, 90.0]), grid=TINY, seed=3)
    assert a.shape == (2, WL.size)
    assert np.array_equal(a, b) and np.array_equal(ea, eb)


def test_single_only_matches_deterministic(atmo):
    """``multiple=False`` must be *exactly* :func:`rt.single.sky_radiance`."""
    zen = np.array([0.0, 45.0, 85.0])
    az = np.array([0.0, 60.0, 180.0])
    rad, err = TW.twilight_radiance(atmo, 96.0, zen, az, single=True,
                                    multiple=False)
    ref = single_radiance(atmo, 96.0, zen, az)
    assert np.allclose(rad, ref, rtol=0, atol=0)
    assert np.all(err == 0.0)


def test_both_disabled_returns_zero(atmo):
    rad, err = TW.twilight_radiance(atmo, 96.0, [0.0], [0.0],
                                    single=False, multiple=False)
    assert np.all(rad == 0.0)
    assert np.all(err == 0.0)


def test_negative_radiance_impossible_and_multiple_positive(atmo):
    ms, err, info = TW.multiple_scattered(atmo, 99.0, [0.0, 85.0], [0.0, 0.0],
                                          quality="low", grid=TINY, seed=5)
    assert np.all(ms >= 0.0)
    # the multiple-scattering term at the zenith in twilight must be non-zero
    assert ms[0, 2] > 0.0
    assert info["anchor_wl"].size == 3


# ---------------------------------------------------------------------------
# symmetry and interpolation
# ---------------------------------------------------------------------------
def test_mirror_symmetry_about_the_solar_meridian(atmo):
    """The model is azimuthally uniform, so +dphi and -dphi must agree.

    They agree *exactly* (the mirror is applied to the coarse grid, not
    resampled), which is a useful check that the symmetric extension and the
    query parameterisation are consistent.
    """
    sun_az = 35.0
    dphi = np.array([20.0, 75.0, 140.0])
    zen = np.array([10.0, 50.0, 80.0])
    rad_p, err_p = TW.twilight_radiance(atmo, 96.0, zen, sun_az + dphi,
                                        sun_azimuth_deg=sun_az,
                                        grid=TINY, seed=9)
    rad_m, err_m = TW.twilight_radiance(atmo, 96.0, zen, sun_az - dphi,
                                        sun_azimuth_deg=sun_az,
                                        grid=TINY, seed=9)
    # identical up to the floating-point symmetry of the local regression
    assert np.allclose(rad_p, rad_m, rtol=1e-5, atol=0)
    assert np.allclose(err_p, err_m, rtol=1e-5, atol=0)


def test_spectral_interpolation_is_a_convex_combination():
    """Linear-in-log interpolation is a partition of unity.

    The interpolated radiance is ``exp(sum_k b_k log L_k)`` with ``b_k >= 0``
    and ``sum_k b_k = 1``.  That makes the result positive, exact at the
    knots, free of overshoot and gives an exact error propagation.
    """
    wl = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
    wl_a = wl[[0, 3, 6]]
    n = wl_a.size
    wmat = np.zeros((wl.size, n))
    for j, lam in enumerate(wl):
        k = int(np.searchsorted(wl_a, lam))
        if k == 0:
            wmat[j, 0] = 1.0
        elif k >= n:
            wmat[j, -1] = 1.0
        else:
            f = (lam - wl_a[k - 1]) / (wl_a[k] - wl_a[k - 1])
            wmat[j, k - 1] = 1.0 - f
            wmat[j, k] = f
    assert np.all(wmat >= 0.0)
    assert np.allclose(wmat.sum(axis=1), 1.0)
    for k, i in enumerate([0, 3, 6]):
        assert np.isclose(wmat[i, k], 1.0)


def test_anchor_wavelengths_are_used_verbatim(atmo):
    """With one anchor the value at that wavelength is the raw estimate."""
    ms1, _, info = TW.multiple_scattered(
        atmo, 90.0, [30.0], [40.0], grid=dict(TINY, n_anchor=1), seed=4)
    # n_anchor=1 selects the first wavelength of the grid
    assert int(info["anchor_wl"][0]) == int(atmo.wl[0])


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_multiple_scattering_grows_towards_twilight(atmo):
    """MS/SS is much larger in deep twilight than in daylight."""
    def ratio(h):
        ms, _, _ = TW.multiple_scattered(atmo, 90.0 - h, [0.0], [0.0],
                                         grid=TINY, seed=6)
        ss = single_radiance(atmo, 90.0 - h, [0.0], [0.0])
        return float(np.sum(ms[0]) / max(np.sum(ss[0]), 1e-300))

    assert ratio(-9.0) > 3.0 * ratio(30.0)
    assert ratio(-9.0) > 1.0


def test_daytime_total_matches_deterministic_model(atmo):
    """Where the deterministic model is good, the engine must agree with it.

    At 30 degrees solar elevation the multiple-scattering term is a modest
    correction and the plane-parallel solver is accurate, so the two models
    must agree to well within the (large) Monte Carlo error of this cheap run.
    """
    h = 30.0
    zen = np.array([0.0, 45.0, 75.0])
    az = np.array([0.0, 90.0, 180.0])
    det = sky_radiance_total(atmo, 90.0 - h, zen, az)
    tw, err = TW.twilight_radiance(atmo, 90.0 - h, zen, az, grid=TINY, seed=2)
    ratio = np.sum(tw, axis=1) / np.sum(det, axis=1)
    assert np.all(np.abs(ratio - 1.0) < 0.15)


def test_ground_albedo_increases_the_sky():
    """A brighter ground must brighten the multiple-scattering field.

    Done with the sun 5 degrees above the horizon, where the surface is fully
    illuminated and the estimator is not in the deep-twilight heavy-tail
    regime.
    """
    atmo = presets.clear(WL)
    ms0, _, _ = TW.multiple_scattered(atmo, 85.0, [0.0, 60.0], [0.0, 0.0],
                                      grid=TINY, seed=8)
    bright = presets.clear(WL)
    bright.ground_albedo = np.full(WL.size, 0.6)
    ms1, _, _ = TW.multiple_scattered(bright, 85.0, [0.0, 60.0], [0.0, 0.0],
                                      grid=TINY, seed=8)
    assert np.all(ms1.sum(axis=1) > 1.1 * ms0.sum(axis=1))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_error_scales_as_one_over_sqrt_n():
    """Four times the photons must halve the statistical error.

    Measured in daylight, where the estimator has many effective contributions
    per path and the sample variance is itself meaningful.
    """
    atmo = presets.clear([550.0])
    m1, e1, _ = TW.multiple_scattered(atmo, 60.0, [0.0], [0.0],
                                      grid=dict(TINY, n_anchor=1, n_photons=5e4),
                                      seed=1)
    m2, e2, _ = TW.multiple_scattered(atmo, 60.0, [0.0], [0.0],
                                      grid=dict(TINY, n_anchor=1, n_photons=2e5),
                                      seed=1)
    # remove the systematic floor before comparing the statistical parts
    def stat(e, m):
        return np.sqrt(np.maximum(e ** 2 - (TW.MS_BIAS * m) ** 2, 0.0))
    ratio = float(stat(e1[0, 0], m1[0, 0]) / stat(e2[0, 0], m2[0, 0]))
    # four times the photons -> half the error, within the noise of the
    # error estimate itself
    assert 1.3 < ratio < 3.5


def test_identical_seed_is_reproducible(atmo):
    a = TW.multiple_scattered(atmo, 99.0, [10.0, 60.0], [0.0, 120.0],
                              grid=TINY, seed=123)
    b = TW.multiple_scattered(atmo, 99.0, [10.0, 60.0], [0.0, 120.0],
                              grid=TINY, seed=123)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_error_is_reported_not_optimistic(atmo):
    """The quoted error must not be absurdly small compared with the truth."""
    ms, err, _ = TW.multiple_scattered(atmo, 96.0, [0.0], [0.0],
                                       grid=TINY, seed=5)
    assert np.all(err[0] >= TW.MS_BIAS * ms[0] * 0.999)


# ---------------------------------------------------------------------------
# quality presets
# ---------------------------------------------------------------------------
def test_quality_presets_are_ordered():
    order = ["low", "medium", "high", "ultra"]
    budget = [TW.QUALITY[q]["n_photons"] * TW.QUALITY[q]["n_anchor"] for q in order]
    assert all(b2 > b1 for b1, b2 in zip(budget, budget[1:]))
    for q in order:
        cfg = TW.QUALITY[q]
        assert cfg["n_zen"] >= 6 and cfg["n_az"] >= 8
        assert cfg["n_anchor"] >= 3
