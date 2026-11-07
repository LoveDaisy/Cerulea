"""Representative extension scenarios (task plan Step 6 / issue E7).

Each scenario exercises one extension seam end-to-end and records what the
extension *touches*:

(i)   swap the ozone profile            -> tests/test_profiles.py (data
      injection reaching single/SOS/twilight through one definition);
(ii)  add a minimal ABSORBING constituent through the species protocol ->
      this file: an ssa~0 aerosol layer (the black-carbon pattern) enters
      single scattering, SOS and the Monte Carlo with **zero solver
      changes** -- the only files touched are this test and the new
      fixture class defined inside it;
(iii) balloon vs plateau scenarios      -> this file: the two are distinct
      constructions (observer raised vs model truncated at the site) with
      distinguishable physics;
(iv)  redefining a shared preset parameter via ``replace`` -> this file:
      the shared instance stays bit-identical, the variant carries the
      change, cache identities stay separate;
(v)  one display path for clear and cloudy skies -> tests/test_result.py
      (render field seam) and this file (render_scene composition).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor import aerosol, cloud, mie, presets, render  # noqa: E402
from skycolor.atmosphere import (  # noqa: E402
    AerosolLayer,
    Atmosphere,
    ExponentialProfile,
)
from skycolor.rt import mc  # noqa: E402
from skycolor.rt.cloud import sky_radiance_with_cloud  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.single import direct_sun_transmittance  # noqa: E402

WL = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])


# ---------------------------------------------------------------------------
# (ii) a minimal absorbing constituent enters every backend, zero solver diff
# ---------------------------------------------------------------------------
class _BlackCarbonLike(aerosol.Aerosol):
    """TEST-ONLY minimal absorbing constituent: grey absorption with a
    spectrally flat cross-section normalised to 1 at 550 nm, essentially no
    scattering (ssa = 0.05).  Deliberately minimal: it uses only the public
    ``Aerosol`` protocol (relative_coefficients / phase / legendre_moments).
    """

    def relative_coefficients(self, wl_nm):
        wl = np.asarray(wl_nm, dtype=float)
        ext = np.full_like(wl, 1.0)          # grey extinction, normalised
        sca = np.full_like(wl, 0.05)         # ssa = 0.05: almost pure absorption
        return ext, sca

    def phase(self, mu, wl_nm):
        mu = np.asarray(mu, dtype=float)
        return np.full_like(mu, 1.0)         # isotropic; irrelevant at ssa=0.05

    def legendre_moments(self, wl_nm, n_moments: int = 32):
        m = np.zeros((n_moments, len(np.asarray(wl_nm))))
        m[0] = 1.0
        return m


def test_absorbing_constituent_reaches_all_backends():
    """The absorbing layer changes the direct beam, the single-scattered sky,
    the SOS field and the Monte Carlo -- without touching any solver file."""
    base = presets.clear(WL)
    with_bc = presets.clear(WL)
    with_bc.aerosols = list(with_bc.aerosols) + [
        AerosolLayer(_BlackCarbonLike(), aod550=0.10,
                     profile=ExponentialProfile(1.0)),
    ]

    # direct beam: the constituent is grey, so its per-wavelength extinction
    # along the same ray is a constant (a plane-parallel estimate lands within
    # the shell-geometry correction, ~0.2 %)
    T0 = direct_sun_transmittance(base, 60.0)
    T1 = direct_sun_transmittance(with_bc, 60.0)
    ratio = T1 / T0
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)   # greyness
    mu0 = np.cos(np.radians(60.0))
    np.testing.assert_allclose(ratio[0], np.exp(-0.10 / mu0), rtol=3e-3)

    # single scattering + SOS: darkening is visible in the sky itself
    r0 = sky_radiance_total(base, 60.0, 30.0, 120.0)
    r1 = sky_radiance_total(with_bc, 60.0, 30.0, 120.0)
    assert np.any(r1 < r0)

    # Monte Carlo: same constituent, same direction, statistically visible
    mc0 = mc.mc_radiance_split(base, 60.0, 30.0, 120.0,
                               n_photons=400_000, seed=0)
    mc1 = mc.mc_radiance_split(with_bc, 60.0, 30.0, 120.0,
                               n_photons=400_000, seed=0)
    # well above the MC noise for total radiance at these settings
    assert mc1["total"].mean() < mc0["total"].mean()


# ---------------------------------------------------------------------------
# (iii) balloon vs plateau: two distinct observer semantics
# ---------------------------------------------------------------------------
def test_balloon_and_plateau_are_distinct_constructions():
    """Balloon: observer raised, model domain still starts at sea level.
    Plateau: the model itself is truncated at the site (ground = site).
    Both have the observer at 5 km; their physics differ because the domain
    below the observer stays in the balloon's field but not the plateau's."""
    balloon = Atmosphere(WL, elevation_km=5.0)
    plateau = Atmosphere(WL).at_altitude(5.0)

    assert balloon.elevation_km == plateau.elevation_km == 5.0
    assert balloon.z_bounds[0] == 0.0              # sea-level ground below
    assert plateau.z_bounds[0] == 5.0              # ground = the site itself
    # both backends agree on where the observer is
    assert balloon.r_bounds[0] == pytest.approx(6371e3 + 0.0)
    assert plateau.r_bounds[0] == pytest.approx(6371e3 + 5000.0)

    # the plateau sky loses the 0-5 km column's scattering; the balloon sky
    # still sees that air horizontally -> measurably different (spherical
    # single scattering, which has no plane-observer restriction)
    r_balloon = sky_radiance_total(balloon, 60.0, 30.0, 120.0, multiple=False)
    r_plateau = sky_radiance_total(plateau, 60.0, 30.0, 120.0, multiple=False)
    assert not np.allclose(r_balloon, r_plateau)
    # a *full* (SOS-including) balloon computation is rejected since the
    # entry consolidation: the plane-parallel diffuse field lives at the
    # model bottom and cannot represent the below-observer air, so silently
    # evaluating it there is no longer allowed (single-spotted semantic
    # correction, task-simulation-entry plan section 3.4)
    with pytest.raises(ValueError, match="model bottom"):
        sky_radiance_total(balloon, 60.0, 30.0, 120.0)
    # the truncated domain's direct beam is not darker (same column above)
    T_balloon = direct_sun_transmittance(balloon, 60.0)
    T_plateau = direct_sun_transmittance(plateau, 60.0)
    np.testing.assert_allclose(T_balloon, T_plateau, rtol=1e-9)


# ---------------------------------------------------------------------------
# (iv) shared preset parameter changes flow through replace, identities apart
# ---------------------------------------------------------------------------
def test_shared_preset_redefinition_is_isolated():
    from skycolor.aerosol import LogNormalMode

    base = aerosol.shared(aerosol.MieAerosol)   # default modes: hashable key
    atmo1 = Atmosphere(WL, aerosols=[AerosolLayer(base, aod550=0.1,
                                                  profile=ExponentialProfile(1.5))])
    r_before = sky_radiance_total(atmo1, 60.0, 30.0, 120.0)

    # redefine through replace: the shared instance is untouched, the variant
    # is a distinct cache identity.  (A knob outside the distribution's
    # truncation -- e.g. r_max_um 30 -> 10 for an r_g=0.05 um mode -- would
    # legitimately produce a bit-identical sky; changing the mode changes it.)
    variant = base.replace(modes=[LogNormalMode(0.5, 2.0)])
    assert variant is not base
    atmo2 = Atmosphere(WL, aerosols=[AerosolLayer(variant, aod550=0.1,
                                                  profile=ExponentialProfile(1.5))])
    r_variant = sky_radiance_total(atmo2, 60.0, 30.0, 120.0)

    # the original sky is bit-identical to its pre-redefinition value
    r_after = sky_radiance_total(atmo1, 60.0, 30.0, 120.0)
    assert np.array_equal(r_before, r_after)
    # ... and the variant genuinely differs
    assert not np.array_equal(r_variant, r_before)


# ---------------------------------------------------------------------------
# (v) one display path for clear and cloudy skies
# ---------------------------------------------------------------------------
def test_one_display_path_for_clear_and_cloud():
    layer = cloud.get_preset("stratus")
    atmo = presets.clear(WL)
    sza = 60.0

    def cloudy_field(zen, az_rel):
        return sky_radiance_with_cloud(atmo, layer, sza, zen, az_rel)

    clear_rgb = render.render_scene(atmo, sun_elevation_deg=90.0 - sza,
                                    camera="panorama", size=(60, 24),
                                    sun_disk=False)
    cloudy_rgb = render.render_scene(atmo, sun_elevation_deg=90.0 - sza,
                                     camera="panorama", size=(60, 24),
                                     sun_disk=False, radiance_field=cloudy_field)
    # same pipeline call, same output contract, physically different skies
    assert clear_rgb.shape == cloudy_rgb.shape
    assert not np.allclose(clear_rgb, cloudy_rgb)
    # the clear sky is BLUE (blue channel dominates red); a thick stratus
    # swaps the Rayleigh colour for flat grey -- both much closer to white
    sky = slice(0, 12)                      # upper-half pixels only
    ratio_clear = clear_rgb[sky, ..., 0].mean() / clear_rgb[sky, ..., 2].mean()
    ratio_cloudy = cloudy_rgb[sky, ..., 0].mean() / cloudy_rgb[sky, ..., 2].mean()
    assert abs(1.0 - ratio_clear) > 2.0 * abs(1.0 - ratio_cloudy)


# ---------------------------------------------------------------------------
# anchoring: the two independent Mie kernels agree for small size parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.05, 0.2, 1.0, 3.0])
def test_mie_kernels_cross_check_small_x(x):
    """``cloud.mie_coefficients`` (batched, numerical moments) and
    ``mie.mie_coefficients`` (single-particle, analytic moments) are
    deliberately separate implementations with different regimes; for the
    small-x overlap region they must anchor each other."""
    m = 1.53 + 0.006j
    single = mie.mie_coefficients(x, m)
    batch = cloud.mie_coefficients(np.array([x]), m)
    assert single.qext == pytest.approx(float(batch["qext"][0]), rel=1e-8)
    assert single.qsca == pytest.approx(float(batch["qsca"][0]), rel=1e-8)
    assert single.g == pytest.approx(float(batch["g"][0]), rel=1e-7)
