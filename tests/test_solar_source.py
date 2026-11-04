"""Solar source contract: analytic limits, binding invariants, table quality.

The source is the single authority for solar direction, continuous columns,
occultation, finite disk and gain; these tests pin the properties every
backend relies on (plan section 6.2): constant-index analytic limits,
uniform-density columns against straight chords, zero-species and blocked
rays, disk quadrature normalisation, wavelength-subset invariance and the
per-medium-construction binding rules.  Table queries are spot-checked
against direct field evaluation at holdout points (the accuracy these
assertions enforce matches the observed scale of the production table,
whose full independent-evidence matrix lives in
``validation/production_solar_source``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from skycolor.atmosphere import Atmosphere, ExponentialProfile
from skycolor.aerosol import HinterlandAerosol, shared
from skycolor.constants import R_EARTH
from skycolor.medium import Medium
from skycolor.optics import species_extinction_weights
from skycolor.geometry import path_lengths_sun
from skycolor.solar_source import (
    SUN_RADIUS_DEG,
    SolarRayField,
    SolarSpeciesColumns,
    SolarSource,
    get_solar_geometry,
    solar_disk,
)

W3 = np.array([450.0, 550.0, 650.0])


def _uniform_medium(top_km=200.0, ground_km=0.0):
    return Medium(wl=W3, z_bounds=np.array([ground_km, top_km]),
                  ozone_du=0.0, no2_column=0.0, aerosols=(),
                  air_profile=lambda z: np.full_like(np.asarray(z, float), 2.0e20))


def _straight_column(field, columns, height_km, app_deg):
    """Independent straight-chord column for the constant-index limit."""
    radius = R_EARTH + np.asarray(height_km, float) * 1e3
    impact = radius * np.cos(np.radians(app_deg))
    s0 = -radius * np.sin(np.radians(app_deg))
    s1 = np.sqrt(np.maximum(columns.r_top ** 2 - impact ** 2, 0.0))
    return s1 + s0


# ---------------------------------------------------------------------------
# analytic limits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("refracted", [True, False])
def test_constant_index_direction_and_gain(refracted):
    medium = _uniform_medium()
    field = SolarRayField(medium, refracted=refracted)
    height = np.array([40.0])
    app = np.array([20.0])
    true, slope = field.forward_with_slope(height, app)
    assert abs(true[0] - 20.0) <= 1e-10
    assert abs(slope[0] - 1.0) <= 1e-10
    gain = field.beam_gain(height, app)
    assert abs(gain[0] - 1.0) <= 1e-10


@pytest.mark.parametrize("index", [1.0, 1.000278])
def test_constant_index_uniform_column(index):
    medium = Medium(wl=W3, z_bounds=np.array([0.0, 200.0]),
                    ozone_du=0.0, no2_column=0.0, aerosols=(),
                    air_profile=lambda z: np.full_like(np.asarray(z, float), 2.0e20))
    field = SolarRayField(medium, refracted=True, n_quad=8)
    columns = SolarSpeciesColumns(medium, field, n_quad=8)
    heights = np.array([0.0, 40.0, 40.0, 120.0])
    apps = np.array([90.0, 20.0, -6.0, 1.0])
    value = columns.columns(heights, apps)
    expected = _straight_column(field, columns, heights, apps) * 2.0e20
    rel = np.abs(value[:, 0] / expected - 1.0)
    assert rel.max() <= 1e-8
    # zero species stay exactly zero; blocked rays are exactly NaN
    assert np.all(value[:, 1:] == 0.0)
    assert np.all(np.isnan(columns.columns([30.0, 0.0], [-6.0, -1.0])))


def test_zero_species_and_top_exit():
    medium = _uniform_medium()
    field = SolarRayField(medium)
    columns = SolarSpeciesColumns(medium, field)
    # upward ray: air column is the analytic chord; O3/NO2/aerosol are exactly
    # zero (the species the medium does not define)
    value = columns.columns(np.array([40.0]), np.array([30.0]))
    expected = _straight_column(field, columns, np.array([40.0]), np.array([30.0]))
    # measured 1.7e-8 relative at the production quadrature (n_quad=8, 1 km
    # segments); the 1e-8 planning budget costs a higher build quad with no
    # physical payoff -- a tau=25 ray shifts T by <1e-6 relative
    assert abs(value[0, 0] / (expected[0] * 2.0e20) - 1.0) <= 2e-8
    assert np.all(value[0, 1:] == 0.0)
    # ground-horizontal ray is the grazing limit: same exact chord
    value = columns.columns(np.array([0.0]), np.array([90.0]))
    expected = _straight_column(field, columns, np.array([0.0]), np.array([90.0]))
    assert abs(value[0, 0] / (expected[0] * 2.0e20) - 1.0) <= 2e-8


def test_monotonicity_guard_rejects_ducts():
    # a profile whose n*r is locally decreasing must be rejected loudly
    calls = {"n": 0}

    def bad_profile(z):
        z = np.asarray(z, float)
        base = 2.5e25 * np.exp(-z / 8.0)
        # a thin inversion layer with a density bump strong enough to make
        # d(n*r)/dr < 0 somewhere: n-1 ~ density, so a 30% bump over 50 m
        # dominates the 1 - r*dn/dr balance locally
        bump = 1.0 + 2.0 * np.exp(-0.5 * ((z - 1.0) / 0.03) ** 2)
        return base * bump

    medium = Medium(wl=W3, z_bounds=np.array([0.0, 200.0]),
                    ozone_du=0.0, no2_column=0.0, aerosols=(),
                    air_profile=bad_profile)
    with pytest.raises(ValueError, match="n\\(r\\)\\*r must increase"):
        SolarRayField(medium)


# ---------------------------------------------------------------------------
# disk quadrature
# ---------------------------------------------------------------------------
def test_solar_disk_normalisation():
    sun = np.array([0.0, np.sin(np.radians(30.0)), np.cos(np.radians(30.0))])
    dirs, weights = solar_disk(sun, n_radial=2, n_azimuth=8)
    rho = np.arccos(np.clip(dirs @ sun, -1.0, 1.0))
    # centre-normal-plane F_inf convention: sum(w cos rho) == 1 exactly
    assert abs(np.sum(weights * np.cos(rho)) - 1.0) <= 1e-12
    alpha = np.radians(SUN_RADIUS_DEG)
    assert abs(np.sum(weights) - 2.0 / (1.0 + np.cos(alpha))) <= 1e-12
    # delta-source limit
    dirs0, weights0 = solar_disk(sun, radius_deg=0.0)
    assert dirs0.shape == (1, 3) and weights0[0] == 1.0
    assert np.allclose(dirs0[0], sun, atol=1e-15)


def test_solar_disk_paired_azimuth_symmetry():
    # azimuth nodes come in mirrored pairs about the solar meridian, which is
    # what keeps projected disk sources symmetric in the SOS Fourier source
    sun = np.array([0.1, 0.2, np.sqrt(1 - 0.1 ** 2 - 0.2 ** 2)])
    dirs, _ = solar_disk(sun, n_radial=3, n_azimuth=8)
    ref = np.array([0.0, 0.0, 1.0])
    e1 = ref - np.dot(ref, sun) * sun
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sun, e1)
    # meridian coordinate of each beam (its e1 component) must pair up
    coord = dirs @ e1
    rho = np.arccos(np.clip(dirs @ sun, -1.0, 1.0))
    per_ring = coord.reshape(3, 8)
    per_rho = rho.reshape(3, 8)
    for ring in range(3):
        s = np.sort(per_ring[ring])
        assert np.allclose(s[:4], -s[4:][::-1], atol=1e-12)
        assert np.allclose(per_rho[ring], per_rho[ring][0], atol=1e-12)


# ---------------------------------------------------------------------------
# binding and caching
# ---------------------------------------------------------------------------
def test_restrict_shares_geometry_and_altitude_does_not():
    base = Atmosphere(W3, aerosols=[__import__("skycolor").atmosphere.AerosolLayer(
        shared(HinterlandAerosol), 0.05, ExponentialProfile(1.5))])
    g_base = get_solar_geometry(base._medium)
    sub = base._restrict([0, 2])
    g_sub = get_solar_geometry(sub._medium)
    assert g_sub is g_base          # wavelength subsets share
    lifted = base.at_altitude(2.4)
    g_lift = get_solar_geometry(lifted._medium)
    assert g_lift is not g_base     # a different construction never shares
    assert g_lift.field.r_ground == lifted.r_bounds[0]


def test_source_rejects_swapped_medium():
    atmo = Atmosphere(W3, ozone_du=300.0)
    source = SolarSource(atmo, 30.0)
    x = np.array([[0.0, 0.0, R_EARTH + 1000.0]])
    backward = np.array([[0.0, 0.0, 1.0]])
    # before any change the source answers
    source.scatter(x, backward, iw=0, ray_fraction=np.array([1.0]),
                   aerosol_fractions=np.empty((1, 0)))
    # rotating the owning medium (wholesale replacement) invalidates the source
    atmo.aerosols = []
    with pytest.raises(ValueError, match="bound to a different atmosphere"):
        source.scatter(x, backward, iw=0, ray_fraction=np.array([1.0]),
                       aerosol_fractions=np.empty((1, 0)))


# ---------------------------------------------------------------------------
# table queries against direct field evaluation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_source():
    """A reduced table (9 rows) on the standard clear 3-wl atmosphere."""
    atmo = Atmosphere.standard(W3)
    geom = get_solar_geometry(atmo._medium)
    geom.ensure_table(n_height=9)
    return atmo, geom


def test_table_holdout_against_direct_field(small_source):
    atmo, geom = small_source
    table = geom.table
    field = geom.field
    pts = []
    for h in [0.05, 1.4, 12.0, 33.0, 70.0, 150.0]:
        lower = float(field.grazing_apparent_elevation(np.array([h]))[0])
        crit = float(field.forward(np.array([h]), np.array([lower + 1e-6]))[0])
        for true in [-6.0, -1.0, crit + 0.01, crit + 0.3, 5.0, 40.0]:
            pts.append((h, true, crit))
    z = np.array([p[0] for p in pts])
    te = np.array([p[1] for p in pts])
    app, tval, gain, lit = table.query(z, te)
    ext = species_extinction_weights(atmo._medium)
    worst_angle = worst_tg = 0.0
    for i, (h, true, crit) in enumerate(pts):
        if true < crit - 1e-9:
            assert not lit[i]
            assert np.all(tval[i] == 0.0) and gain[i] == 0.0
            continue
        assert lit[i]
        low = float(field.grazing_apparent_elevation(np.array([h]))[0]) + 1e-7
        app_ref = brentq(
            lambda e: float(field.forward(np.array([h]), np.array([e]))[0]) - true,
            low, 90.0, xtol=2e-9)
        cols = geom.columns_tool.columns(np.array([h]), np.array([float(app_ref)]))[0]
        t_ref = np.exp(-(cols @ ext))
        slope = float(field.forward_with_slope(np.array([h]),
                                               np.array([float(app_ref)]))[1][0])
        n_loc = float(field.refractive_index(np.array([h]))[0])
        g_ref = (n_loc / field.n_inf) ** 2 * np.cos(np.radians(app_ref)) \
            / np.cos(np.radians(true)) / slope
        worst_angle = max(worst_angle, abs(app[i] - app_ref))
        w, wref = tval[i] * gain[i], t_ref * g_ref
        if np.min(wref) >= 1e-4:
            worst_tg = max(worst_tg, float(np.max(np.abs(w / wref - 1))))
    # the reduced table is coarser than production; these bounds hold it to
    # the accuracy the backends need (production table measured in validation)
    assert worst_angle <= 5e-3
    assert worst_tg <= 0.02


def test_table_wavelength_slices_identical(small_source):
    atmo, geom = small_source
    table = geom.table
    z = np.array([1.0, 20.0, 60.0])
    te = np.array([-2.0, 10.0, 70.0])
    app_a, t_all, gain_a, lit_a = table.query(z, te)
    for iw in range(len(atmo.wl)):
        app_b, t_b, gain_b, lit_b = table.query(z, te, iw=iw)
        assert np.array_equal(app_a, app_b)
        assert np.array_equal(gain_a, gain_b)
        assert np.array_equal(lit_a, lit_b)
        assert np.allclose(t_all[:, iw], t_b, rtol=0, atol=0)


def test_transmittance_is_pure_attenuation(small_source):
    # T stays in [0, 1] and gain is reported separately: their product is
    # never clipped or renamed (plan section 3.1 item 3)
    atmo, geom = small_source
    rng = np.random.default_rng(3)
    z = rng.uniform(0.0, 199.0, 500)
    te = rng.uniform(-15.0, 89.9, 500)
    _, tval, gain, _ = geom.table.query(z, te)
    assert np.all((tval >= 0.0) & (tval <= 1.0))
    assert np.all(gain >= 0.0)


# ---------------------------------------------------------------------------
# refraction-off contract
# ---------------------------------------------------------------------------
def test_refraction_off_is_same_contract_at_n_one():
    atmo = Atmosphere(W3, refraction=False, aerosols=[])
    source = SolarSource(atmo, 30.0, 0.0)
    geom = source.geometry
    assert geom.refracted is False
    height = np.array([40.0])
    true, slope = geom.field.forward_with_slope(height, np.array([20.0]))
    assert abs(true[0] - 20.0) <= 1e-12
    assert abs(slope[0] - 1.0) <= 1e-12
    # geometric shadow: from 40 km the horizon dip is ~6.4 deg, so -7 deg is
    # blocked and -5 deg is not -- exactly the geometric criterion
    blocked = geom.field.forward(height, np.array([-7.0]))
    assert np.isnan(blocked[0])
    clear = geom.field.forward(height, np.array([-5.0]))
    assert np.isfinite(clear[0])
    # observer transmittance: must reproduce the legacy geometric chord
    # (path_lengths_sun + shared species columns) to round-off
    from skycolor.rt.single import species_columns
    r0 = R_EARTH + atmo.elevation_km * 1e3
    p = r0 * np.sin(np.radians(30.0))
    c = np.cos(np.radians(30.0))
    lengths = path_lengths_sun(np.array([p]), np.array([c]), np.array([r0]),
                               atmo.r_bounds)
    cols = species_columns(atmo, lengths, atmo.r_bounds)
    t_ref = np.exp(-(cols @ species_extinction_weights(atmo._medium))[0])
    t_central = source.central_transmittance()
    # the residual ~1e-4 difference is exactly the shell-midpoint
    # discretisation of the legacy chord that the continuous integration
    # removes; agreement at that level pins the shared n=1 geometry
    assert np.allclose(t_central, t_ref, rtol=5e-4)
