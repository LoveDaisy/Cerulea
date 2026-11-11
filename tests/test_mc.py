"""Validation tests for the independent Monte Carlo reference solver.

The tests are deliberately numerical rather than structural: each one checks a
physical property that the deterministic code either predicts independently or
that follows from conservation laws.

Run with ``pytest tests/test_mc.py`` (a few seconds) or
``pytest tests/test_mc.py -s`` to see the reported multiple-scattering split.
"""
from __future__ import annotations


import numpy as np
import pytest

from skycolor import presets, spectra
from skycolor.atmosphere import AerosolLayer, Atmosphere
from skycolor.rt import mc, single


# Full-tier exclusive by species dimension: every test here drives the
# Monte-Carlo reference to statistical convergence (see docs/DEVELOPMENT.md).
pytestmark = pytest.mark.slow(
    reason="kind:mc - Monte-Carlo statistical convergence is full-tier exclusive")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def scaled_atmosphere(atmo, factor: float) -> Atmosphere:
    """Return a copy of ``atmo`` with every optical depth scaled by ``factor``.

    Both the molecular densities and the aerosol optical depths are scaled,
    so the result is a genuinely thinner atmosphere, not just a fainter one.
    Data path (plan Step 3b): the scaled profiles are injected as callables,
    which is linear in the column amounts, so the result matches the
    historical cross-section scaling to within floating-point association.
    Assumes the input ozone/NO2 scale linearly with ``ozone_du``/
    ``no2_column`` (true for the built-in and callable-shaped profiles).
    """
    layers = [AerosolLayer(lay.aerosol, aod550=lay.aod550 * factor,
                           profile=lay.profile, horizontal=lay.horizontal)
              for lay in atmo.aerosols]
    return Atmosphere(
        atmo.wl, ozone_du=atmo.ozone_du * factor,
        no2_column=atmo.no2_column * factor, aerosols=layers,
        ground_albedo=atmo.ground_albedo,
        surface_pressure=atmo.surface_pressure, z_bounds_km=atmo.z_bounds,
        elevation_km=atmo.elevation_km, refraction=atmo.refraction,
        air_profile=lambda z: atmo.n_air(z) * factor,
    )


def global_power_balance(albedo, wl, n_theta=12, n_mu=10, n_az=4,
                         n_photons=60_000, seed=0):
    """Integrate the radiative energy budget of the whole planet.

    Returns ``(P_in, P_up_toa, P_absorbed)`` in W nm^-1 for a conservative
    (Rayleigh-only, ozone-free) atmosphere.  The angular integrals use
    Gauss-Legendre quadrature in the cosine of the solar zenith angle (the
    field is axially symmetric about the solar direction) and the hemispheric
    flux helper for each solar position.
    """
    atmo = Atmosphere(wl, ozone_du=0.0, aerosols=[], ground_albedo=albedo)
    r_top = atmo.r_bounds[-1]
    r_ground = atmo.r_bounds[0]

    gx, gw = np.polynomial.legendre.leggauss(n_theta)
    mu = 0.5 * (gx + 1.0)          # cos(solar zenith) on the day side
    wmu = 0.5 * gw
    theta = np.degrees(np.arccos(mu))

    # upward flux at the top of the atmosphere
    atmo.elevation_km = atmo.z_bounds[-1]
    p_up = 0.0
    for th, wt in zip(theta, wmu):
        flux, _ = mc.mc_flux(atmo, th, up=True, n_mu=n_mu, n_az=n_az,
                             n_photons=n_photons, seed=seed)
        p_up += wt * flux[0]
    p_up *= 2.0 * np.pi * r_top**2

    # downward flux at the surface: diffuse from the MC plus the direct beam
    atmo.elevation_km = 0.0
    p_down = 0.0
    for th, wt in zip(theta, wmu):
        flux, _ = mc.mc_flux(atmo, th, up=False, n_mu=n_mu, n_az=n_az,
                             n_photons=n_photons, seed=seed + 1)
        direct = float(single.direct_sun_transmittance(atmo, th)[0])
        direct *= spectra.solar_irradiance(wl)[0] * max(np.cos(np.radians(th)), 0.0)
        p_down += wt * (flux[0] + direct)
    p_absorbed = (1.0 - albedo) * p_down * 2.0 * np.pi * r_ground**2

    p_incident = spectra.solar_irradiance(wl)[0] * np.pi * r_ground**2
    return p_incident, p_up, p_absorbed


# ---------------------------------------------------------------------------
# 1. consistency with the analytic single-scattering solver
# ---------------------------------------------------------------------------
def test_thin_atmosphere_matches_single_scattering():
    """In a very thin atmosphere the total radiance is single scattering.

    The atmosphere is made 100x thinner, so multiple scattering and the
    ground reflection are both below 1 % of the signal (the ground albedo is
    set to zero because ``single.sky_radiance`` does not model the ground).
    """
    wl = np.array([400.0, 500.0, 550.0])
    zenith = np.array([0.0, 60.0, 85.0])
    atmo = presets.pristine(wl)
    atmo.ground_albedo = np.zeros(len(wl))
    thin = scaled_atmosphere(atmo, 1e-2)

    rad, err = mc.mc_radiance(thin, 60.0, zenith, 0.0,
                              n_photons=500_000, seed=11)
    ref = single.sky_radiance(thin, 60.0, zenith, 0.0)

    # multiple scattering + ground reflection are < 1 % here, well inside the
    # Monte Carlo error bars
    z = (rad - ref) / np.maximum(err, 1e-30)
    assert np.all(np.abs(z) < 3.0), f"z-scores outside 3 sigma: {z}"
    assert np.all(rad > 0.0)

    # the same comparison for the *first-order* part of the MC estimator
    split = mc.mc_radiance_split(thin, 60.0, zenith, 0.0,
                                 n_photons=500_000, seed=11)
    z1 = (split["single"] - ref) / np.maximum(split["single_err"], 1e-30)
    assert np.all(np.abs(z1) < 3.0), f"first-order z-scores: {z1}"
    assert np.all(split["multiple"] < 0.02 * split["single"])


def test_single_scattering_part_in_a_thick_atmosphere():
    """The first-order split reproduces the deterministic solver everywhere.

    This is a stronger test than the thin-atmosphere one: the single-scattered
    part of the MC estimator must equal ``sky_radiance`` in a *normal*
    atmosphere, where multiple scattering is large.
    """
    wl = np.array([450.0, 550.0])
    zenith = np.array([0.0, 60.0])
    atmo = presets.clear(wl)
    atmo.ground_albedo = np.zeros(len(wl))

    split = mc.mc_radiance_split(atmo, 60.0, zenith, 0.0,
                                 n_photons=1_000_000, seed=21)
    ref = single.sky_radiance(atmo, 60.0, zenith, 0.0)
    rel = np.abs(split["single"] - ref) / ref
    assert np.all(rel < 0.03), f"single-scattering mismatch: {rel}"
    # multiple scattering is much larger here
    assert np.all(split["multiple"] > 0.05 * split["single"])


# ---------------------------------------------------------------------------
# 2. energy conservation
# ---------------------------------------------------------------------------
def test_energy_conservation_conservative_atmosphere():
    """For a conservative atmosphere no energy may be created or destroyed.

    A Rayleigh-only, ozone-free atmosphere is conservative, so the power
    reflected to space plus the power absorbed by the surface must equal the
    intercepted solar power:

        P_up(TOA) + P_absorbed(surface) = S * pi * R_earth^2

    With a black surface (albedo 0) the absorbed part dominates; with a white
    surface (albedo 1) everything must come back out of the top.  The
    integration is over the whole sphere (the local flux at a single point is
    not conserved in spherical geometry, the global integral is).
    """
    wl = np.array([550.0])
    for albedo in (0.0, 1.0):
        p_inc, p_up, p_abs = global_power_balance(albedo, wl)
        balance = (p_up + p_abs) / p_inc
        assert abs(balance - 1.0) < 0.03, (
            f"energy not conserved for albedo {albedo}: "
            f"P_up={p_up/p_inc:.4f}, P_abs={p_abs/p_inc:.4f}, sum={balance:.4f}")
        if albedo == 0.0:
            # a black surface must absorb most of the incoming power
            assert 0.85 < p_abs / p_inc < 0.95
            assert 0.03 < p_up / p_inc < 0.15
        else:
            assert p_abs == 0.0
            assert 0.95 < p_up / p_inc < 1.05


# ---------------------------------------------------------------------------
# 3. smoothness and statistical error
# ---------------------------------------------------------------------------
def test_no_nans_no_negatives_and_error_scaling():
    """Radiance is finite and non-negative; stderr falls like 1/sqrt(N)."""
    wl = np.array([400.0, 550.0, 700.0])
    atmo = presets.clear(wl)
    zenith = np.array([0.0, 45.0, 75.0, 88.0])

    errs = []
    for n in (20_000, 80_000, 320_000):
        rad, err = mc.mc_radiance(atmo, 60.0, zenith, 30.0,
                                  n_photons=n, seed=5)
        assert np.all(np.isfinite(rad))
        assert np.all(np.isfinite(err))
        assert np.all(rad > 0.0)
        assert np.all(err > 0.0)
        errs.append(err)

    # quadrupling the photon count halves the error
    for a, b in zip(errs[:-1], errs[1:]):
        ratio = a / b
        assert np.all(ratio > 1.5), ratio
        assert np.all(ratio < 2.5), ratio

    # the three estimates must agree within a few sigma of each other
    rad_small, err_small = mc.mc_radiance(atmo, 60.0, zenith, 30.0,
                                          n_photons=20_000, seed=5)
    rad_large, err_large = mc.mc_radiance(atmo, 60.0, zenith, 30.0,
                                          n_photons=320_000, seed=5)
    z = (rad_small - rad_large) / np.maximum(np.hypot(err_small, err_large), 1e-30)
    assert np.all(np.abs(z) < 4.0), z


# ---------------------------------------------------------------------------
# 4. magnitude of multiple scattering
# ---------------------------------------------------------------------------
def test_multiple_scattering_magnitude(capsys):
    """Report the zenith radiance split for the clear-sky reference case.

    ``presets.clear`` at 550 nm, solar zenith 60 deg, zenith view.  The
    architecture document expects the multiple-scattering contribution to be
    of order 10-20 % of the single-scattering value for a clear sky; the
    ground albedo adds a few more percent.
    """
    wl = np.array([550.0])
    atmo = presets.clear(wl)
    split = mc.mc_radiance_split(atmo, 60.0, [0.0], 0.0,
                                 n_photons=1_000_000, seed=1)
    total = split["total"][0, 0]
    l1 = split["single"][0, 0]
    lms = split["multiple"][0, 0]
    ref = single.sky_radiance(atmo, 60.0, [0.0], 0.0)[0, 0]

    with capsys.disabled():
        print("\nclear sky, 550 nm, sun zenith 60 deg, zenith view")
        print(f"  total     {total:.5f} +/- {split['total_err'][0, 0]:.5f} W m^-2 sr^-1 nm^-1")
        print(f"  single    {l1:.5f} +/- {split['single_err'][0, 0]:.5f}"
              f"   (deterministic {ref:.5f})")
        print(f"  multiple  {lms:.5f} +/- {split['multiple_err'][0, 0]:.5f}")
        print(f"  MS / SS = {100 * lms / l1:.1f} %"
              f"   MS / total = {100 * lms / total:.1f} %")

    # the single-scattering part agrees with the deterministic solver
    assert abs(l1 - ref) / ref < 0.02
    # multiple scattering is a real but sub-dominant contribution
    assert 0.05 < lms / l1 < 0.60
    # and the two parts add up
    assert abs(total - l1 - lms) < 1e-12 + 1e-6 * total


def test_ground_albedo_increases_the_diffuse_sky():
    """A brighter ground adds reflected light to the sky (monotonic)."""
    wl = np.array([550.0])
    zenith = np.array([0.0, 70.0])
    totals = []
    for albedo in (0.0, 0.3):
        atmo = presets.clear(wl)
        atmo.ground_albedo = np.full(len(wl), albedo)
        rad, _ = mc.mc_radiance(atmo, 60.0, zenith, 0.0,
                                n_photons=300_000, seed=8)
        totals.append(rad)
    assert np.all(totals[1] > totals[0])
