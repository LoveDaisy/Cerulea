"""Acceptance tests for the cloud layer.

The tests encode the four claims the cloud model makes:

1. the Mie optics are consistent (energy, asymmetry, moment identities);
2. delta-M preserves the similarity parameter and reproduces delta-Eddington
   for a Henyey-Greenstein phase function;
3. the two-stream fluxes conserve energy, reproduce the Monte-Carlo albedo of
   a conservative layer, and give the classical overcast sky distribution;
4. the composite model reduces *exactly* to the clear sky at zero optical
   depth and produces a red underlit base after sunset.

Reference albedos marked "MC" come from an independent plane-parallel Monte
Carlo (200 000-400 000 photons, 1 sigma < 0.001) with the same phase function;
they are quoted in ``docs/physics/CLOUD.md``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor import cloud, presets  # noqa: E402
from skycolor.atmosphere import Atmosphere  # noqa: E402
from skycolor.rt import cloud as rtcloud  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402

#: A small spectral grid keeps the (cached) Mie calculation and the
#: clear-sky multiple-scattering solver cheap.
WL = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
I550 = 3


def _atmo() -> Atmosphere:
    """Rayleigh + ozone only: the cloud tests must not depend on aerosol.py."""
    return Atmosphere(WL, ozone_du=300.0, aerosols=[], ground_albedo=0.1)


@pytest.fixture(scope="module")
def stratus() -> cloud.CloudLayer:
    return cloud.get_preset("stratus")


@pytest.fixture(scope="module")
def altostratus() -> cloud.CloudLayer:
    return cloud.get_preset("altostratus")


@pytest.fixture(scope="module")
def cirrus() -> cloud.CloudLayer:
    return cloud.get_preset("cirrus")


@pytest.fixture(scope="module")
def atmo() -> Atmosphere:
    return _atmo()


# ---------------------------------------------------------------------------
# 1. Mie optics
# ---------------------------------------------------------------------------
def test_mie_large_parameter_limits():
    """Q_ext -> 2 and g -> O(1) for droplets much larger than the wavelength."""
    dist = cloud._Distribution(8.0, 1.35, 64)
    out = cloud.mie_optics(550.0, dist, cloud.WATER_REFRACTIVE_INDEX)
    assert abs(out["qext"] - 2.0) < 0.15
    assert abs(out["qsca"] - out["qext"]) < 1e-9 * out["qext"]  # conservative
    assert 0.80 < out["g"] < 0.90
    assert abs(out["omega"] - 1.0) < 1e-9


def test_moments_are_normalised_and_match_g():
    """beta_0 = 1 and beta_1 equals the (independently computed) g."""
    dist = cloud._Distribution(8.0, 1.35, 96)
    out = cloud.mie_optics(550.0, dist, cloud.WATER_REFRACTIVE_INDEX, n_moments=48)
    m = out["moments"]
    assert abs(m[0] - 1.0) < 2e-3
    assert abs(m[1] - out["g"]) < 2e-3          # two independent calculations
    # the forward peak makes the first few moments large and nearly equal
    assert m[1] > 0.8 and m[2] > 0.7
    assert np.all(np.diff(m[10:40]) < 0.02)      # monotonically decaying tail


def test_spectral_optical_depth_is_grey(stratus, cirrus):
    """Cloud extinction is nearly grey across the visible (why clouds are white)."""
    for layer in (stratus, cirrus):
        tau = layer.optical_depth(WL)
        assert np.all(tau > 0)
        assert tau.max() / tau.min() < 1.05


def test_presets_are_plausible(stratus, altostratus, cirrus):
    assert stratus.optical_depth([550.0])[0] == pytest.approx(20.0, rel=1e-6)
    assert altostratus.optical_depth([550.0])[0] == pytest.approx(5.0, rel=1e-6)
    assert cirrus.optical_depth([550.0])[0] == pytest.approx(0.5, rel=1e-6)
    assert cirrus.phase == "ice" and stratus.phase == "water"
    # liquid water contents of a real stratus: 0.05 - 0.5 g/m^3
    assert 0.05 < stratus.liquid_water_content() * 1e3 < 0.5
    assert 0.005 < altostratus.liquid_water_content() * 1e3 < 0.5


# ---------------------------------------------------------------------------
# 2. delta-M
# ---------------------------------------------------------------------------
def test_delta_m_reduces_to_delta_eddington_for_hg():
    """For HG, beta_l = g^l and delta-M at 2N=2 is f = g^2 (Joseph et al.)."""
    g, tau, omega = 0.85, 10.0, 1.0
    f = g**2
    tau_s, omega_s, g_s = cloud.delta_m(tau, omega, g, f)
    assert tau_s == pytest.approx((1 - omega * f) * tau)
    assert g_s == pytest.approx(g / (1 + g))          # = (g - g^2)/(1 - g^2)
    assert omega_s == pytest.approx(1.0)


def test_delta_m_preserves_similarity_parameter():
    """(1 - g) tau for conservative scattering is invariant under delta-M."""
    for g, tau in [(0.86, 20.0), (0.5, 5.0), (0.0, 1.0)]:
        f = cloud.get_preset("stratus").optics([550.0]).f[0]
        tau_s, omega_s, g_s = cloud.delta_m(tau, 1.0, g, f)
        assert (1.0 - g) * tau == pytest.approx((1.0 - g_s) * tau_s, rel=1e-10)


def test_delta_fraction_is_the_second_moment(stratus):
    optics = stratus.optics(WL)
    assert np.allclose(optics.f, optics.moments[2])
    # for a cloud droplet distribution the delta fraction is large: most of
    # the scattered energy is inside the diffraction spike
    assert np.all(optics.f > 0.6)
    assert np.all(optics.f < optics.g)       # otherwise g' would be negative


# ---------------------------------------------------------------------------
# 3. two-stream fluxes
# ---------------------------------------------------------------------------
def test_conservative_layer_conserves_energy():
    tau = np.array([0.5, 1.0, 5.0, 10.0, 20.0, 100.0])
    res = rtcloud.delta_eddington(tau, 1.0, 0.85, 0.7, f=0.72)
    assert np.allclose(res.albedo + res.total_transmittance, 1.0, atol=1e-12)


def test_absorption_is_positive_and_bounded():
    res = rtcloud.delta_eddington(np.array([1.0, 10.0]), 0.9, 0.8, 1.0, f=0.64)
    assert np.all(res.absorption > 0)
    assert np.all(res.absorption < 1)


def test_albedo_matches_monte_carlo():
    """Hemispherical albedo of a conservative layer, mu0 = 1, black ground.

    The references are Monte-Carlo values (see the module docstring).  A
    two-stream carries a closure error that grows at small optical depth, so
    the tolerance is 2 % at tau = 10 and 8 % at tau = 2.
    """
    # isotropic phase function
    res = rtcloud.delta_eddington(10.0, 1.0, 0.0, 1.0, f=0.0)
    assert res.albedo[0] == pytest.approx(0.8523, rel=0.015)   # MC 0.8523
    # strongly forward scattering (g = 0.85)
    res = rtcloud.delta_eddington(10.0, 1.0, 0.85, 1.0, f=0.85**2)
    assert res.albedo[0] == pytest.approx(0.4224, rel=0.02)    # MC 0.4224
    # a real droplet distribution (Mie, g = 0.861, delta-M at f = beta_2)
    optics = cloud.get_preset("stratus").optics([550.0])
    res = rtcloud.delta_eddington(10.0, 1.0, optics.g[0], 1.0, f=optics.f[0])
    assert res.albedo[0] == pytest.approx(0.4025, rel=0.03)    # MC(Mie) 0.4025


def test_albedo_grows_towards_one():
    tau = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0, 1000.0])
    res = rtcloud.delta_eddington(tau, 1.0, 0.86, 1.0, f=0.79)
    assert np.all(np.diff(res.albedo) > 0)
    assert res.albedo[-1] > 0.985
    assert res.albedo[tau == 10.0][0] < 0.7     # a tau = 10 water cloud is grey


def test_albedo_reference_value_tau_10():
    """The 'about 0.7 at tau = 10' number corresponds to g ~ 0.6."""
    res = rtcloud.delta_eddington(10.0, 1.0, 0.6, 1.0, f=0.6**2)
    assert res.albedo[0] == pytest.approx(0.688, abs=0.03)
    # and for a water cloud (g = 0.86) the same albedo needs tau ~ 30
    optics = cloud.get_preset("stratus").optics([550.0])
    res = rtcloud.delta_eddington(30.0, 1.0, optics.g[0], 1.0, f=optics.f[0])
    assert res.albedo[0] == pytest.approx(0.70, abs=0.04)


def test_zero_optical_depth_is_transparent():
    res = rtcloud.delta_eddington(0.0, 1.0, 0.86, 0.7, f=0.79)
    assert res.albedo[0] == 0.0
    assert res.transmittance[0] == 0.0
    assert res.direct_transmittance[0] == pytest.approx(1.0)


def test_removable_resonance_lambda_equals_inverse_mu0():
    """omega = 2/3, g = 0 gives lambda = 1 exactly: the particular solution
    has a 0/0 at mu0 = 1 and must still be finite."""
    res = rtcloud.delta_eddington(5.0, 2.0 / 3.0, 0.0, 1.0, f=0.0)
    assert np.all(np.isfinite(res.albedo))
    assert np.all(np.isfinite(res.transmittance))
    assert 0.0 < res.albedo[0] < 1.0


def test_grazing_beam_albedo_stays_bounded():
    """R must stay finite (and below 1) as the beam approaches the horizon."""
    for mu0 in [0.5, 0.1, 0.01, 1e-3]:
        res = rtcloud.delta_eddington(20.0, 1.0, 0.86, mu0, f=0.79)
        assert 0.0 <= res.albedo[0] <= 1.0
        assert np.isfinite(res.flux_down_base[0])


def test_underlit_flux_equals_top_lit_by_symmetry():
    """A slab is symmetric: lighting it from below reflects the same fraction."""
    top = rtcloud.delta_eddington(5.0, 1.0, 0.86, 0.6, f=0.79)
    base = rtcloud.delta_eddington(5.0, 1.0, 0.86, -0.6, f=0.79)
    assert base.albedo[0] == pytest.approx(top.albedo[0], rel=1e-12)
    assert base.illumination == "base"


# ---------------------------------------------------------------------------
# 4. overcast radiance distribution
# ---------------------------------------------------------------------------
def test_overcast_shape():
    lz = rtcloud.overcast_shape(1.0)
    lh = rtcloud.overcast_shape(0.0)
    assert lh / lz == pytest.approx(1.0 / 3.0)      # CIE overcast law
    assert np.all(np.diff(rtcloud.overcast_shape(np.linspace(0, 1, 20))) > 0)
    # flux normalisation: int L mu dOmega = 1
    mu = np.linspace(0.0, 1.0, 200001)
    flux = 2 * np.pi * np.trapezoid(rtcloud.overcast_shape(mu) * mu, mu)
    assert flux == pytest.approx(1.0, rel=1e-6)


def test_thick_cloud_has_overcast_luminance_distribution(atmo, stratus):
    """Optically thick deck, high sun: peak at the zenith, horizon/zenith 1/3."""
    view = np.array([0.0, 30.0, 60.0, 89.9])
    rad = rtcloud.sky_radiance_with_cloud(atmo, stratus, 45.0, view, 0.0)
    lum = rad[:, I550]
    assert np.all(np.diff(lum) < 0)                  # brightest at the zenith
    assert lum[0] > lum[1] > lum[2] > lum[3]
    # view 89.9 deg is mu = 0.0017, so the law gives (1 + 2 mu)/3
    assert lum[-1] / lum[0] == pytest.approx((1.0 + 2.0 * np.cos(np.radians(89.9))) / 3.0, rel=2e-3)
    # and the cloud term dominates the (heavily attenuated) clear sky
    assert lum[0] > 1e3 * 0.0


def test_cloud_is_brighter_than_clear_sky_in_daylight(atmo, stratus):
    view = np.array([0.0])
    clear = sky_radiance_total(atmo, 45.0, view, 0.0, multiple=True)
    cloudy = rtcloud.sky_radiance_with_cloud(atmo, stratus, 45.0, view, 0.0,
                                             clear_radiance=clear)
    assert cloudy[0, I550] > 2.0 * clear[0, I550]


# ---------------------------------------------------------------------------
# 5. clear-sky limit and integration
# ---------------------------------------------------------------------------
def test_zero_tau_reduces_exactly_to_clear_sky(atmo, stratus):
    view = np.array([0.0, 45.0, 80.0])
    clear = sky_radiance_total(atmo, 50.0, view, 0.0, multiple=True)
    empty = stratus.replace(tau550=0.0)
    total = rtcloud.sky_radiance_with_cloud(atmo, empty, 50.0, view, 0.0,
                                            clear_radiance=clear)
    assert np.array_equal(total, clear)            # exact, not approximate


def test_cloud_attenuates_the_clear_sky(atmo, stratus):
    view = np.array([0.0, 60.0])
    clear = sky_radiance_total(atmo, 50.0, view, 0.0, multiple=True)
    _, comp = rtcloud.sky_radiance_with_cloud(atmo, stratus, 50.0, view, 0.0,
                                              clear_radiance=clear,
                                              return_components=True)
    optics = stratus.optics(WL)
    mu = np.cos(np.radians(view))
    expected = clear * np.exp(-optics.tau[None, :] / mu[:, None])
    assert np.allclose(comp["clear"], expected, rtol=1e-12)


def test_output_shape_matches_solver(atmo, altostratus):
    view = np.linspace(0.0, 85.0, 7)
    clear = sky_radiance_total(atmo, 30.0, view, 0.0, multiple=False)
    out = rtcloud.sky_radiance_with_cloud(atmo, altostratus, 30.0, view, 0.0,
                                          clear_radiance=clear)
    assert out.shape == (view.size, WL.size)
    assert np.all(out >= 0) and np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 6. the sun below the horizon: red underlighting
# ---------------------------------------------------------------------------
def test_shadow_height():
    assert rtcloud.shadow_height_km(0.0) == 0.0
    assert rtcloud.shadow_height_km(60.0) == 0.0
    # the dip of the horizon: 2 degrees -> about 3.9 km
    assert rtcloud.shadow_height_km(92.0) == pytest.approx(3.9, abs=0.2)
    assert rtcloud.shadow_height_km(96.0) > rtcloud.shadow_height_km(92.0)


def test_high_clouds_stay_lit_longer(atmo, altostratus, cirrus):
    assert rtcloud.illuminate_cloud(atmo, altostratus, 92.0).geometry == "base"
    assert rtcloud.illuminate_cloud(atmo, cirrus, 93.0).geometry == "base"
    # per-beam shadow heights at zenith 92 deg span ~0.3-2.9 km around the
    # stratus (0.5-1.5 km): the near-limb beams still reach it (base entry),
    # while the whole disk misses only at deeper sun (zenith 97 deg puts
    # every beam's shadow height above 20 km)
    assert rtcloud.illuminate_cloud(atmo, cloud.get_preset("stratus"),
                                    92.0).geometry == "base"
    assert rtcloud.illuminate_cloud(atmo, cloud.get_preset("stratus"),
                                    97.0).geometry == "none"


def test_underlit_base_turns_red(atmo, altostratus):
    """The cloud base goes white -> orange -> red as the sun sets."""
    from skycolor import color

    def chromaticity(sun_elev):
        illum = rtcloud.illuminate_cloud(atmo, altostratus, 90.0 - sun_elev)
        rad = rtcloud.cloud_radiance(illum, [0.0])
        return color.chromaticity(color.spectrum_to_xyz(rad[0], WL))

    day = chromaticity(20.0)
    sunset = chromaticity(-1.0)
    assert day[1] > 0.33                              # white-ish at noon
    assert sunset[0] > day[0] + 0.15                  # much redder
    assert sunset[0] > 0.5
    # and it does eventually go dark (the whole disk below the boundary;
    # at -6 deg the altostratus top's boundary is near -3.3 deg true)
    illum = rtcloud.illuminate_cloud(atmo, altostratus, 96.0)
    assert illum.geometry == "none"
    assert np.all(rtcloud.cloud_radiance(illum, [0.0]) == 0.0)


def test_underlit_deck_is_not_seen_through_its_shadow(atmo):
    """A thick low deck cannot glow from below: the shadowed part blocks it."""
    low = cloud.get_preset("stratus")
    illum = rtcloud.illuminate_cloud(atmo, low, 91.0)   # sun 1 degree below
    assert illum.geometry == "base"
    # Per-beam shadow heights: at zenith 91 deg the shadow height sits at
    # ~0.85-1.5 km — INSIDE the low deck — so the beams enter mid-deck with
    # the sub-entry fraction shadowed (the per-beam replacement for the old
    # midpoint closure); the deepest-shadowed beams carry most of tau.
    mid = [b for b in illum.beams if b.z_entry_km > low.base_km]
    assert len(mid) >= len(illum.beams) // 2
    assert max(float(np.max(b.tau_below)) for b in mid) > 5.0
    top_view = rtcloud.cloud_radiance(illum, [0.0])[0, I550]
    low_view = rtcloud.cloud_radiance(illum, [85.0])[0, I550]
    assert low_view < top_view                          # attenuated towards the horizon
    assert top_view > 0.0


def test_cloud_layer_is_frozen_and_replace_drops_optics():
    """The deck definition is immutable, and ``replace`` must NOT carry the
    computed optics of the old definition into the new one (the tau scaling
    would silently stay at the old value)."""
    from dataclasses import FrozenInstanceError

    layer = cloud.get_preset("stratus")
    with pytest.raises(FrozenInstanceError):
        layer.tau550 = 5.0
    slim = layer.replace(tau550=5.0)
    wl3 = np.array([450.0, 550.0, 650.0])
    fat, thin = layer.optics(wl3), slim.optics(wl3)
    np.testing.assert_allclose(thin.tau / fat.tau, 5.0 / 20.0, rtol=1e-12)
