"""Validation tests for multiple scattering (orders >= 2).

Run from the worktree root with

    .venv/bin/python -m pytest tests/test_multiple.py -q

The tests encode the acceptance criteria:

1. an empty (non-scattering) atmosphere returns exactly zero;
2. energy conservation for a conservative atmosphere;
3. the magnitude of MS/SS for a clear sky, and its growth towards the horizon;
4. smoothness / non-negativity of the view dependence;
5. a sane twilight value.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor import presets  # noqa: E402
from skycolor.aerosol import AngstromAerosol  # noqa: E402
from skycolor.atmosphere import (  # noqa: E402
    AerosolLayer,
    Atmosphere,
    ExponentialProfile,
)
from skycolor.rt import multiple, single  # noqa: E402

WL550 = np.array([550.0])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _conservative_atmosphere(wl, albedo: float) -> Atmosphere:
    """Rayleigh + a purely scattering aerosol, no ozone, no NO2."""
    aerosol = AngstromAerosol(name="conservative", alpha=1.0, ssa=1.0, g=0.6)
    return Atmosphere(
        wl,
        ozone_du=0.0,
        aerosols=[AerosolLayer(aerosol, aod550=0.3,
                               profile=ExponentialProfile(1.5))],
        ground_albedo=albedo,
    )


# ---------------------------------------------------------------------------
# 1. empty atmosphere
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_empty_atmosphere_is_exactly_zero():
    """No Rayleigh and no aerosol -> no scattering -> exactly zero."""
    atmo = Atmosphere(np.array([450.0, 550.0, 650.0]), ozone_du=300.0,
                      aerosols=[],
                      air_profile=lambda z: np.zeros_like(np.asarray(z, dtype=float)))
    ms = multiple.multiple_scattering(
        atmo, 60.0, [0.0, 30.0, 60.0, 85.0], [0.0, 90.0, 180.0, 270.0]
    )
    assert ms.shape == (4, 3)
    assert np.all(ms == 0.0)


# ---------------------------------------------------------------------------
# 2. energy conservation
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
@pytest.mark.parametrize("albedo", [0.0, 1.0])
def test_energy_conservation(albedo):
    """Reflected + transmitted = incident for a conservative atmosphere.

    The run uses 48 streams and a tight order tolerance so that the residual
    is the discretisation error of the solver, not the order truncation.
    With albedo 1 the ground reflects everything, so the net downward flux at
    the bottom must also vanish.
    """
    wl = np.array([450.0, 550.0, 650.0])
    atmo = _conservative_atmosphere(wl, albedo)
    res = multiple.solve(atmo, 0.0, None, n_mu=48, max_orders=200, tol=1e-10)

    incident = res.incident_flux
    net_down = res.flux_down_boa + res.direct_down_boa - res.flux_up_boa
    balance = (res.flux_up_toa + net_down) / incident

    assert res.converged
    np.testing.assert_allclose(balance, 1.0, atol=1e-3)
    if albedo == 1.0:
        np.testing.assert_allclose(net_down / incident, 0.0, atol=1e-6)


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_energy_conservation_thin_rayleigh_only():
    """A Rayleigh-only conservative atmosphere also closes its budget."""
    wl = np.array([450.0, 550.0, 650.0])
    atmo = Atmosphere(wl, ozone_du=0.0, aerosols=[], ground_albedo=1.0)
    res = multiple.solve(atmo, 0.0, None, n_mu=48, max_orders=200, tol=1e-10)
    net_down = res.flux_down_boa + res.direct_down_boa - res.flux_up_boa
    balance = (res.flux_up_toa + net_down) / res.incident_flux
    np.testing.assert_allclose(balance, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. magnitude and horizon trend
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_zenith_magnitude_clear_sky():
    """MS/SS at the zenith, 550 nm, sun zenith 60 deg: 0.08 .. 0.35.

    The reference value from the validation agent is 0.25 .. 0.45; the
    measured value here is about 0.31 for `presets.clear` (AOD 0.05).  A
    turbidity-3 atmosphere, which is what the CIE reference uses, is closer
    to `presets.hazy` and gives a larger ratio.
    """
    atmo = presets.clear(WL550)
    ss = single.sky_radiance(atmo, 60.0, [0.0], 0.0)[0, 0]
    ms = multiple.multiple_scattering(atmo, 60.0, [0.0], 0.0)[0, 0]
    ratio = ms / ss
    assert 0.08 <= ratio <= 0.35, f"MS/SS={ratio:.3f}"


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_ms_ss_increases_towards_horizon():
    """MS/SS grows towards the horizon away from the sun.

    On the sun side the single-scattered forward peak keeps the ratio
    non-monotonic; the diffuse multiple scattering is relatively more
    important at 90 deg azimuth, which is the standard 'blue sky' geometry.
    """
    atmo = presets.clear(WL550)
    vza = np.array([0.0, 30.0, 60.0, 85.0, 88.0])
    ss = single.sky_radiance(atmo, 60.0, vza, 90.0)[:, 0]
    ms = multiple.multiple_scattering(atmo, 60.0, vza, 90.0)[:, 0]
    ratio = ms / ss
    assert np.all(np.diff(ratio) > 0.0), ratio
    assert 0.08 <= ratio[0] <= 0.35
    assert ratio[-1] > 0.6


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_horizon_headline_value():
    """Clear sky, 550 nm, sza 45, vza 88, sun side: MS/SS ~ 0.40."""
    atmo = presets.clear(WL550)
    ss = single.sky_radiance(atmo, 45.0, [88.0], 0.0)[0, 0]
    ms = multiple.multiple_scattering(atmo, 45.0, [88.0], 0.0)[0, 0]
    ratio = ms / ss
    assert 0.30 <= ratio <= 0.55, f"horizon MS/SS={ratio:.3f}"


# ---------------------------------------------------------------------------
# 4. smoothness and non-negativity
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_smooth_and_nonnegative():
    atmo = presets.clear(np.arange(400.0, 701.0, 20.0))
    i550 = int(np.argmin(np.abs(atmo.wl - 550.0)))
    vza = np.linspace(0.0, 90.0, 181)

    ms = multiple.multiple_scattering(atmo, 45.0, vza, 90.0)
    assert np.all(np.isfinite(ms))
    assert np.all(ms >= 0.0)

    d2 = np.diff(ms[:, i550], 2)
    scale = np.max(np.abs(ms[:, i550]))
    assert np.max(np.abs(d2)) < 0.05 * scale

    # azimuthal smoothness at the horizon
    az = np.linspace(0.0, 360.0, 361)
    ms_az = multiple.multiple_scattering(atmo, 45.0, np.full_like(az, 88.0), az)
    assert np.all(np.isfinite(ms_az))
    assert np.all(ms_az >= 0.0)
    assert np.max(np.abs(np.diff(ms_az[:, i550]))) < 0.1 * np.max(ms_az[:, i550])


# ---------------------------------------------------------------------------
# 5. twilight
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_twilight_is_positive_and_bounded():
    atmo = presets.clear(WL550)
    ms = multiple.multiple_scattering(atmo, 96.0, [0.0], 0.0)[0, 0]
    ss = single.sky_radiance(atmo, 96.0, [0.0], 0.0)[0, 0]
    assert ms > 0.0
    assert ms < 1e-3                      # not absurdly large
    assert ms < 10.0 * ss                 # of the same order as SS


# ---------------------------------------------------------------------------
# extra regression checks
# ---------------------------------------------------------------------------
@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_first_order_matches_single_scattering():
    """The first-order quadrature reproduces `rt.single` (plane-parallel view)."""
    atmo = presets.clear(WL550)
    vza = np.array([0.0, 30.0, 60.0])
    res = multiple.solve(atmo, 60.0, vza, 0.0, n_mu=32, single_view=True)
    ref = single.sky_radiance(atmo, 60.0, vza, 0.0)[:, 0]
    np.testing.assert_allclose(res.radiance_single[:, 0], ref, rtol=0.03)


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_optically_thick_horizon_converges():
    """Hazy atmosphere, vza 88 deg (tau along the line of sight ~ 4)."""
    atmo = presets.hazy(np.array([550.0]))
    res = multiple.solve(atmo, 45.0, [88.0], 0.0, n_mu=32)
    assert res.converged
    assert res.orders <= 40
    assert np.all(np.isfinite(res.radiance)) and res.radiance[0, 0] > 0.0
    ms48 = multiple.multiple_scattering(atmo, 45.0, [88.0], 0.0, n_mu=48)
    assert abs(ms48[0, 0] / res.radiance[0, 0] - 1.0) < 0.03


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_scalar_zenith_with_azimuth_array():
    """A scalar zenith angle can be paired with an array of azimuths."""
    atmo = presets.clear(WL550)
    az = np.array([0.0, 90.0, 180.0])
    ms = multiple.multiple_scattering(atmo, 45.0, 60.0, az)
    assert ms.shape == (3, 1)
    assert np.all(ms > 0.0)
    # the sun-side value must differ from the anti-sun value
    assert abs(ms[0, 0] - ms[2, 0]) > 0.0


@pytest.mark.slow(reason="kind:grid - large-grid / full-solve integration is full-tier exclusive")
def test_direct_beam_projection_supports_many_modes():
    """The per-beam Fourier azimuth table must size itself by n_modes.

    The round-4 reviewer's probe: the hardcoded 64-row table crashed with
    IndexError for any legally configured n_modes > 64.
    """
    atmo = Atmosphere(WL550, ozone_du=300.0, aerosols=[],
                      refraction=False, ground_albedo=0.0)
    res = multiple.solve(atmo, 65.0, np.array([0.0, 45.0]),
                         np.array([0.0, 90.0]), n_mu=64, n_modes=65,
                         n_moments=65, max_orders=1, single_view=True)
    assert np.all(np.isfinite(res.radiance))
    assert np.all(res.radiance >= 0.0)


# ---------------------------------------------------------------------------
# gen3-R2: the public TOA incident flux shares the unified plane-flux seam
# ---------------------------------------------------------------------------
def _toa_cap_reference(e_c_deg, alpha_deg=16.0 / 60.0):
    """Independent vacuum cap integral: analytic azimuth, quad radial.

    For one (rho, phi) beam the plane projection is ``A + B cos(phi)``
    with ``A = sin(e_c) cos(rho)``, ``B = cos(e_c) sin(rho)``; its
    positive-part azimuth integral is analytic.  Normalised by the
    centre-plane convention ``pi sin(alpha)^2``.
    """
    from scipy.integrate import quad
    a = np.radians(alpha_deg)
    e_c = np.radians(e_c_deg)

    def radial(rho):
        A = np.sin(e_c) * np.cos(rho)
        B = np.cos(e_c) * np.sin(rho)
        if B == 0.0:
            az = 2.0 * np.pi * max(A, 0.0)
        elif A >= B:
            az = 2.0 * np.pi * A
        elif A <= -B:
            az = 0.0
        else:
            phi0 = np.arccos(-A / B)
            az = 2.0 * (A * phi0 + B * np.sin(phi0))
        return az * np.cos(rho) * np.sin(rho)

    val, _ = quad(radial, 0.0, a, epsabs=1e-20, epsrel=1e-11, limit=200)
    return val / (np.pi * np.sin(a) ** 2)


def test_toa_incident_flux_resolves_plane_boundary():
    """`incident_flux` reads the receiver-plane-aware kernel at the top.

    gen3-R2: solve still summed the fixed coarse disk with per-beam max
    truncation for this public diagnostic -- +20.7% at sun -0.2 deg and a
    false zero at -0.25 deg against an independent cap integral.  The
    seam now matches the same reference at all three angles.
    """
    vac = Atmosphere(np.array([550.0]), z_bounds_km=[0.0, 200.0],
                     aerosols=[], ozone_du=0.0, no2_column=0.0,
                     refraction=False,
                     air_profile=lambda z: np.zeros_like(np.asarray(z, float)))
    from skycolor import spectra
    f_inf = float(spectra.solar_irradiance(np.array([550.0]))[0])
    for e in (-0.25, -0.2, 0.0):
        res = multiple.solve(vac, 90.0 - e, 45.0, 135.0, n_mu=6)
        ker = float(np.ravel(res.incident_flux)[0]) / f_inf
        ref = _toa_cap_reference(e)
        assert ker > 0.0, (e, ker, ref)
        assert abs(ker / ref - 1.0) < 2e-3, (e, ker, ref)
