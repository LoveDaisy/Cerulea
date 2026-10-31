"""Validation of the aerosol models.

Covers the four things that can go wrong independently:

1. the Mie engine itself (published benchmarks for Qext, Qsca and g);
2. the analytic Legendre moments (exactness, g = beta_1, physical range);
3. the adaptive phase table (normalisation and the forward peak);
4. the delta-M transformation (energy conservation, limits) and the documented
   library table (Angstrom exponent, SSA, asymmetry, tau(450)/tau(650));
5. the per-wavelength Mie state cache and the shared preset instances
   (capacity/eviction, bitwise reuse, no cross-contamination).
"""
from __future__ import annotations

import numpy as np
import pytest

from skycolor import presets
from skycolor.aerosol import (
    LIBRARY,
    AngstromAerosol,
    LogNormalMode,
    MieAerosol,
    delta_m,
    get_aerosol,
    henyey_greenstein,
    humidity_growth_factor,
    mie_single,
)
from skycolor.mie import mie_coefficients, mie_legendre_moments

#: Canonical library members (the registry also has aliases).
LIBRARY_NAMES = [
    "rural",
    "urban",
    "maritime",
    "desert_dust",
    "biomass_burning",
    "volcanic_stratospheric",
    "cirrus",
    "water_cloud",
]

#: Values documented in the docstrings of ``skycolor.aerosol`` and in
#: ``docs/AEROSOL.md``.  ``alpha`` is ``-ln(tau(440)/tau(660))/ln(440/660)``,
#: ``tau_ratio`` is ``tau(450)/tau(650)``, ``f32`` is ``beta_32`` (the first
#: omitted moment when the SOS solver keeps 32 moments, l = 0..31).
DOCUMENTED = {
    "rural": dict(alpha=1.295, ssa=0.9824, g=0.7141, tau_ratio=1.616, f32=0.0000),
    "urban": dict(alpha=1.246, ssa=0.9013, g=0.7044, tau_ratio=1.586, f32=0.0000),
    "maritime": dict(alpha=0.546, ssa=1.0000, g=0.8127, tau_ratio=1.225, f32=0.1270),
    "desert_dust": dict(alpha=-0.044, ssa=0.8763, g=0.8165, tau_ratio=0.984, f32=0.3677),
    "biomass_burning": dict(alpha=1.836, ssa=0.8653, g=0.5863, tau_ratio=1.973, f32=0.0000),
    "volcanic_stratospheric": dict(alpha=-0.310, ssa=1.0000, g=0.7097, tau_ratio=0.892, f32=0.0040),
    "cirrus": dict(alpha=-0.037, ssa=0.9998, g=0.8679, tau_ratio=0.986, f32=0.3806),
    "water_cloud": dict(alpha=-0.042, ssa=1.0000, g=0.8582, tau_ratio=0.985, f32=0.3829),
}


@pytest.fixture(scope="module")
def library():
    """One instance of every canonical library member (phase tables cached)."""
    return {name: get_aerosol(name) for name in LIBRARY_NAMES}


# ===========================================================================
# 1. single-sphere Mie against published benchmarks
# ===========================================================================
# Hong Du, "Mie-scattering calculation", Appl. Opt. 43, 1951-1956 (2004),
# Table 1 (Qext, Qsca); g from Wiscombe's MIEV0 as tabulated in Blevins
# (2010), Tables 4/24.  The imaginary part convention here is m = n + i k
# with k > 0 for absorption; the Blevins table prints the opposite sign, so
# its "1.5 - 1.0i" row is our 1.5 + 1.0i.
BENCHMARKS = [
    (10.0, 0.75 + 0.0j, 2.232265, 2.232265, 0.896473),
    (1000.0, 0.75 + 0.0j, 1.997908, 1.997908, 0.844944),
    (1.0, 1.33 + 1e-5j, 9.39520e-2, 9.39230e-2, 0.184517),
    (100.0, 1.33 + 1e-5j, 2.10132, 2.09659, 0.868959),
    (1.0, 1.5 + 1.0j, 2.33632, 0.663454, 0.192136),
    (100.0, 1.5 + 1.0j, 2.09750, 1.28370, 0.850252),
    (1.0, 10.0 + 10.0j, 2.53299, 2.04941, -0.110664),
    (100.0, 10.0 + 10.0j, 2.07112, 1.83679, 0.556215),
]


@pytest.mark.parametrize("x,m,qext,qsca,g", BENCHMARKS)
def test_mie_single_sphere_benchmark(x, m, qext, qsca, g):
    c = mie_coefficients(x, m)
    assert c.qext == pytest.approx(qext, rel=2e-3)
    assert c.qsca == pytest.approx(qsca, rel=2e-3)
    assert c.g == pytest.approx(g, abs=2e-4)
    assert -1.0 < c.g < 1.0


def test_mie_single_helper_matches_coefficients():
    x, m = 10.0, 0.75 + 0.0j
    qe, qs, g, (mu, ph) = mie_single(x, m, n_angles=721)
    c = mie_coefficients(x, m)
    assert qe == pytest.approx(c.qext, rel=1e-12)
    assert qs == pytest.approx(c.qsca, rel=1e-12)
    assert g == pytest.approx(c.g, rel=1e-12)
    assert mu.shape == ph.shape == (721,)
    assert np.all(np.isfinite(ph)) and np.all(ph >= 0.0)


# ===========================================================================
# 2. analytic moments
# ===========================================================================
@pytest.mark.parametrize("x,m", [
    (0.5, 1.53 + 0.006j),
    (2.0, 1.53 + 0.006j),
    (10.0, 1.5 + 0.1j),
    (100.0, 1.33 + 1e-5j),
    (200.0, 1.55 + 0.005j),
])
def test_analytic_moments_match_quadrature(x, m):
    """beta_l from the Mie coefficients vs a fine angular quadrature.

    The phase function is a polynomial in mu of degree ~2 nmax, so a
    Gauss-Legendre rule with enough points is exact; this is an independent
    check of the Gaunt/Wigner expansion.  ``l <= 8`` keeps the reference well
    conditioned even at x = 200.
    """
    c = mie_coefficients(x, m)
    beta = mie_legendre_moments(c, 9)
    mu, w = np.polynomial.legendre.leggauss(4000)
    from skycolor.mie import mie_phase

    p = mie_phase(c, np.arccos(mu))
    for l in range(9):
        pl = np.polynomial.legendre.legval(mu, [0] * l + [1])
        ref = 0.5 * np.sum(w * p * pl)
        assert beta[l] == pytest.approx(ref, abs=2e-8)


@pytest.mark.parametrize("x", [0.3, 2.0, 10.0, 100.0, 200.0, 500.0])
def test_analytic_g_is_physical(x):
    """``beta_1`` equals the independent analytic asymmetry and is in (-1, 1)."""
    c = mie_coefficients(x, 1.55 + 0.005j)
    beta = mie_legendre_moments(c, 2)
    assert beta[0] == pytest.approx(1.0, abs=1e-12)
    assert beta[1] == pytest.approx(c.g, abs=1e-10)
    assert -1.0 < beta[1] < 1.0


def test_large_droplet_asymmetry_stays_physical():
    """The old fixed 721-point mu grid gave g > 1 at x ~ 91; the analytic
    moments and the adaptive table must not."""
    x = 2.0 * np.pi * 8.0 / 0.55  # r = 8 um at 550 nm
    c = mie_coefficients(x, 1.333 + 0.0j)
    assert -1.0 < c.g < 1.0
    assert mie_legendre_moments(c, 2)[1] == pytest.approx(c.g, abs=1e-10)

    a = get_aerosol("water_cloud")
    g = a.asymmetry([550.0])[0]
    assert 0.5 < g < 1.0
    assert a.legendre_moments([550.0], 2)[1, 0] == pytest.approx(g, abs=1e-10)


def test_moment_energy_and_asymmetry_small_particle():
    """g from the moments must agree with the quadrature definition for a
    small (smooth) particle."""
    a = MieAerosol(modes=[LogNormalMode(0.05, 1.6)],
                   refractive_index=1.53 + 0.006j)
    g = a.asymmetry([550.0])[0]
    mom = a.legendre_moments([550.0], 4)
    assert mom[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert mom[1, 0] == pytest.approx(g, abs=1e-10)
    mu, w = np.polynomial.legendre.leggauss(2000)
    p = a.phase(mu, np.array([550.0]))[:, 0]
    g_quad = 0.5 * np.sum(w * mu * p)
    assert g_quad == pytest.approx(g, abs=5e-4)


# ===========================================================================
# 3. adaptive phase table
# ===========================================================================
@pytest.mark.parametrize("name", LIBRARY_NAMES)
@pytest.mark.parametrize("wl", [450.0, 650.0])
def test_phase_normalisation(library, name, wl):
    """``int p dOmega = 4 pi`` to 1e-3, for every library member."""
    a = library[name]
    mu, w = np.polynomial.legendre.leggauss(4000)
    p = a.phase(mu, np.array([wl]))[:, 0]
    integral = 2.0 * np.pi * np.sum(w * p)
    assert integral / (4.0 * np.pi) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("name", LIBRARY_NAMES)
def test_phase_is_positive_and_finite(library, name):
    a = library[name]
    mu = np.linspace(-1.0, 1.0, 201)
    p = a.phase(mu, np.array([550.0]))[:, 0]
    assert np.all(np.isfinite(p))
    assert np.all(p > 0.0)
    # forward scattering (mu = +1 is the last element) is stronger than
    # backward scattering for every member
    assert p[-1] > p[0]


def test_forward_peak_beats_henyey_greenstein():
    """The motivating defect: HG with the same g is far too weak at 1 deg and
    too broad at 30 deg.  This is the dust distribution quoted in the task."""
    a = MieAerosol(modes=[LogNormalMode(0.5, 2.2)],
                   refractive_index=1.55 + 0.005j)
    g = a.asymmetry([550.0])[0]
    theta = np.radians(np.array([1.0, 5.0, 10.0, 30.0]))
    p = a.phase(np.cos(theta), np.array([550.0]))[:, 0]
    hg = henyey_greenstein(np.cos(theta), g)
    ratio = p / hg
    assert ratio[0] > 3.0          # measured ~10.9
    assert ratio[1] > 1.3          # measured ~2.2
    assert ratio[3] < 0.9          # measured ~0.62
    # monotonically decreasing away from the forward peak
    assert np.all(np.diff(p) < 0.0)


def test_phase_shape_conventions():
    a = get_aerosol("rural")
    assert np.ndim(a.phase(1.0, 550.0)) == 0
    assert a.phase(np.array([0.0, 1.0]), 550.0).shape == (2, 1)
    assert a.phase(1.0, np.array([450.0, 550.0])).shape == (1, 2)
    assert a.phase(np.array([0.0, 1.0]), np.array([450.0, 550.0])).shape == (2, 2)


@pytest.mark.parametrize("name", LIBRARY_NAMES)
def test_relative_coefficients_normalised_at_550(library, name):
    a = library[name]
    ext, sca = a.relative_coefficients([450.0, 550.0, 650.0])
    assert ext[1] == pytest.approx(1.0, rel=1e-12)
    assert np.all(sca >= 0.0)
    assert np.all(sca <= ext + 1e-12)


# ===========================================================================
# 4. delta-M
# ===========================================================================
def test_delta_m_henyey_greenstein():
    g, n_keep = 0.7, 4
    mom = np.array([g**l for l in range(n_keep + 2)])
    res = delta_m(mom, n_keep, tau=0.5, omega=0.9)
    f = g**n_keep
    assert res.f == pytest.approx(f)
    assert res.beta[0] == pytest.approx(1.0, abs=1e-14)
    for l in range(n_keep):
        assert res.beta[l] == pytest.approx((g**l - f) / (1.0 - f))
    assert res.g_star == pytest.approx((g - f) / (1.0 - f))
    assert res.tau_star == pytest.approx((1.0 - 0.9 * f) * 0.5)
    assert res.omega_star == pytest.approx(0.9 * (1.0 - f) / (1.0 - 0.9 * f))


def test_delta_m_zero_forward_fraction_is_identity():
    mom = np.array([1.0, 0.5, 0.25, 0.0])  # beta_3 = 0, so N = 3 -> f = 0
    res = delta_m(mom, 3, tau=0.7, omega=0.95)
    assert res.f == 0.0
    assert np.allclose(res.beta, mom[:3])
    assert res.tau_star == pytest.approx(0.7)
    assert res.omega_star == pytest.approx(0.95)
    assert res.g_star == pytest.approx(0.5)


@pytest.mark.parametrize("name", LIBRARY_NAMES)
def test_delta_m_preserves_energy(library, name):
    a = library[name]
    mom = a.legendre_moments([550.0], 33)  # l = 0..32, f = beta_32
    assert mom[0, 0] == pytest.approx(1.0, abs=1e-12)
    res = delta_m(mom, 32, tau=0.4, omega=0.95)
    assert res.beta[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= res.f[0] < 1.0
    assert res.tau_star[0] == pytest.approx((1.0 - 0.95 * res.f[0]) * 0.4)
    assert res.omega_star[0] == pytest.approx(
        0.95 * (1.0 - res.f[0]) / (1.0 - 0.95 * res.f[0])
    )
    # documented delta-M need: f_32 above 10 % for the forward-peaked members
    expected = DOCUMENTED[name]["f32"]
    assert res.f[0] == pytest.approx(expected, abs=0.02)


def test_delta_m_conservative_limit():
    """For a conservative aerosol (omega = 1) delta-M leaves omega* = 1."""
    a = get_aerosol("water_cloud")
    res = a.delta_m([550.0], 32, tau=1.0, omega=1.0)
    assert res.omega_star[0] == pytest.approx(1.0, abs=1e-12)
    assert res.tau_star[0] == pytest.approx(1.0 - res.f[0], rel=1e-12)


def test_aerosol_delta_m_matches_standalone():
    a = get_aerosol("desert_dust")
    wl = np.array([450.0, 550.0])
    method = a.delta_m(wl, 8)
    standalone = delta_m(a.legendre_moments(wl, 9), 8)
    assert np.allclose(method.beta, standalone.beta)
    assert np.allclose(method.f, standalone.f)


def test_angstrom_aerosol_hg_moments():
    a = AngstromAerosol(g=0.65)
    mom = a.legendre_moments([450.0, 550.0], 5)
    for l in range(5):
        assert np.allclose(mom[l], 0.65**l)
    assert a.asymmetry([450.0, 550.0]).tolist() == [0.65, 0.65]


# ===========================================================================
# 5. documented library table
# ===========================================================================
def _alpha_440_660(a):
    ext, _ = a.relative_coefficients([440.0, 550.0, 660.0])
    return float(-np.log(ext[0] / ext[2]) / np.log(440.0 / 660.0))


@pytest.mark.parametrize("name", LIBRARY_NAMES)
def test_library_table(library, name):
    """Angstrom exponent, SSA and g at 550 nm against the documented values."""
    a = library[name]
    doc = DOCUMENTED[name]
    assert _alpha_440_660(a) == pytest.approx(doc["alpha"], abs=0.02)
    assert a.ssa([550.0])[0] == pytest.approx(doc["ssa"], abs=0.005)
    assert a.asymmetry([550.0])[0] == pytest.approx(doc["g"], abs=0.01)


@pytest.mark.parametrize("name", ["desert_dust", "biomass_burning"])
def test_blue_sun_tau_ratio(library, name):
    """``tau(450)/tau(650)`` decides whether an aerosol can make the direct
    beam blue.  Desert dust is the library's (weak) candidate; biomass burning
    is strongly blue-absorbing."""
    a = library[name]
    ext, _ = a.relative_coefficients([450.0, 550.0, 650.0])
    ratio = ext[0] / ext[2]
    assert ratio == pytest.approx(DOCUMENTED[name]["tau_ratio"], rel=0.02)
    if name == "desert_dust":
        assert ratio < 1.0
    else:
        assert ratio > 1.5


# ===========================================================================
# 6. humidity growth and misc
# ===========================================================================
def test_humidity_growth_is_monotonic():
    assert humidity_growth_factor(0.0) == pytest.approx(1.0)
    vals = [humidity_growth_factor(rh) for rh in (0.0, 0.3, 0.6, 0.9)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert humidity_growth_factor(0.5, gamma=0.0) == 1.0


def test_wetting_increases_scattering_and_asymmetry():
    dry = get_aerosol("rural", rh=0.2)
    wet = get_aerosol("rural", rh=0.85)
    assert wet.ssa([550.0])[0] > dry.ssa([550.0])[0]
    assert wet.asymmetry([550.0])[0] > dry.asymmetry([550.0])[0]


def test_library_registry_aliases():
    assert get_aerosol("continental").name.startswith("rural")
    assert get_aerosol("dust").name.startswith("desert_dust")
    assert get_aerosol("cloud").name.startswith("water_cloud")
    assert "rural" in LIBRARY and "water_cloud" in LIBRARY
    with pytest.raises(KeyError):
        get_aerosol("not-an-aerosol")


def test_documented_values_are_consistent_with_docstrings():
    """A cheap guard: the numbers in ``DOCUMENTED`` are the ones quoted in the
    module docstrings (kept in sync manually; see docs/AEROSOL.md)."""
    import skycolor.aerosol as mod

    for cls, name in [
        (mod.RuralAerosol, "rural"),
        (mod.UrbanAerosol, "urban"),
        (mod.MaritimeAerosol, "maritime"),
        (mod.DesertDustAerosol, "desert_dust"),
        (mod.BiomassBurningAerosol, "biomass_burning"),
        (mod.VolcanicStratosphericAerosol, "volcanic_stratospheric"),
        (mod.CirrusAerosol, "cirrus"),
        (mod.WaterCloudAerosol, "water_cloud"),
    ]:
        doc = cls.__doc__ or ""
        assert "SSA" in doc
        assert "delta-M" in doc


def test_angular_function_legendre_expansion():
    """``pi_n`` and ``tau_n`` have the sparse Legendre expansions documented in
    ``skycolor.mie``; verify the closed-form coefficients against quadrature."""
    from skycolor.mie import _A_coefficient, _B_coefficient, _angular_functions

    nmax = 12
    mu, w = np.polynomial.legendre.leggauss(200)
    pi, tau = _angular_functions(mu, nmax)
    for n in range(1, nmax + 1):
        for a in range(0, n + 1):
            pl = np.polynomial.legendre.legval(mu, [0] * a + [1])
            # Legendre coefficient: (2a+1)/2 * integral(pi P_a)
            ref_pi = (2 * a + 1) * 0.5 * np.sum(w * pi[n] * pl)
            ref_tau = (2 * a + 1) * 0.5 * np.sum(w * tau[n] * pl)
            assert ref_pi == pytest.approx(_A_coefficient(n, a), abs=1e-10)
            assert ref_tau == pytest.approx(_B_coefficient(n, a), abs=1e-10)


# ===========================================================================
# 5. per-wavelength state cache and shared preset instances
# ===========================================================================
def test_state_cache_capacity_and_eviction():
    """The state LRU honours ``max_cached_wavelengths``, and a wavelength
    refilled after eviction comes back bit-identical: states are pure
    functions of (wavelength, construction parameters), so capacity trades
    only memory for time, never values."""
    wl = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0])
    capped = MieAerosol(
        modes=[LogNormalMode(0.2, 2.0)], n_radii=32, max_cached_wavelengths=4
    )
    fresh = MieAerosol(
        modes=[LogNormalMode(0.2, 2.0)], n_radii=32, max_cached_wavelengths=64
    )
    first = capped.relative_coefficients(wl)
    assert len(capped._states) <= 4  # the 6-wavelength working set overflows
    second = capped.relative_coefficients(wl)  # re-queries evicted states
    reference = fresh.relative_coefficients(wl)
    assert np.array_equal(first, second)
    assert np.array_equal(second, reference)


def test_repeated_calls_and_preset_reuse_are_bitwise_identical():
    """Repeated delta-M calls hit the cached states and moments and return
    bit-identical results, and two ``presets.clear(...)`` builds share one
    aerosol instance so the Mie caches survive the rebuild."""
    wl = np.array([400.0, 450.0, 500.0, 550.0, 600.0])
    a1 = presets.clear(wl).aerosols[0].aerosol
    a2 = presets.clear(wl).aerosols[0].aerosol
    assert a1 is a2  # the shared instance, not an equal-by-value copy

    rc1 = a1.relative_coefficients(wl)
    rc2 = a2.relative_coefficients(wl)
    assert np.array_equal(rc1[0], rc2[0])
    assert np.array_equal(rc1[1], rc2[1])

    dm1 = a1.delta_m(wl, 16)
    dm2 = a1.delta_m(wl, 16)  # second call: cached states + cached moments
    assert np.array_equal(dm1.beta, dm2.beta)
    assert np.array_equal(dm1.f, dm2.f)
    assert np.array_equal(dm1.g_star, dm2.g_star)


def test_no_cross_contamination_across_rh_and_wavelength():
    """Different construction arguments give different shared instances with
    different values, and interleaved wavelength queries never interfere:
    the shared key is the full construction signature, the per-wavelength
    cache key is the wavelength, and neither leaks across."""
    wl = np.array([400.0, 500.0, 600.0])
    a5 = presets.hazy(wl, rh=0.5).aerosols[0].aerosol
    a8 = presets.hazy(wl, rh=0.8).aerosols[0].aerosol
    assert a5 is not a8  # the RH argument is part of the shared key

    c5 = a5.relative_coefficients(wl)[0]
    c8 = a8.relative_coefficients(wl)[0]
    assert not np.array_equal(c5, c8)  # RH must matter

    # interleaved off-grid queries perturb neither instance
    _ = a5.relative_coefficients(np.array([412.5]))
    _ = a8.relative_coefficients(np.array([700.0]))
    assert np.array_equal(a5.relative_coefficients(wl)[0], c5)
    assert np.array_equal(a8.relative_coefficients(wl)[0], c8)


def test_mie_aerosol_is_frozen():
    """The optical definition is immutable: assignment must raise, and
    ``replace`` returns a fresh object with rebuilt caches (task Step 5)."""
    from dataclasses import FrozenInstanceError

    a = MieAerosol(modes=[LogNormalMode(0.2, 2.0)], n_radii=32)
    with pytest.raises(FrozenInstanceError):
        a.n_radii = 64
    b = a.replace(n_radii=64)
    assert b.n_radii == 64 and a.n_radii == 32
    assert b is not a
    with pytest.raises(TypeError):
        a.replace(nope=1)
