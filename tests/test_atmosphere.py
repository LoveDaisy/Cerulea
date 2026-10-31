import numpy as np
import pytest

from skycolor.atmosphere import Atmosphere, us76_pressure, us76_temperature
from skycolor.constants import DU, R_EARTH, WL_COARSE


def test_us76_reference_values():
    z = np.array([0.0, 5.0, 11.0, 20.0, 32.0, 50.0, 71.0])
    p = us76_pressure(z)
    expected = np.array([101325, 54020, 22632, 5475, 868, 76.0, 3.96])
    assert np.allclose(p / expected, 1.0, rtol=0.02)


def test_temperature_layers():
    assert us76_temperature(np.array([0.0]))[0] == pytest.approx(288.15)
    assert us76_temperature(np.array([15.0]))[0] == pytest.approx(216.65)
    assert us76_temperature(np.array([25.0]))[0] == pytest.approx(221.65)


def test_ozone_column_matches_du():
    from skycolor.constants import Z_TOA

    at = Atmosphere(WL_COARSE, ozone_du=300.0, aerosols=[])
    z = np.linspace(0, Z_TOA / 1e3, 20000)
    col = np.trapezoid(at.n_o3(z), z * 1e3)
    assert col / DU == pytest.approx(300.0, rel=1e-6)


def test_rayleigh_tau():
    at = Atmosphere(WL_COARSE, aerosols=[])
    z = np.linspace(0, 120, 8000)
    tau = np.trapezoid(at.rayleigh_coefficient(z), z * 1e3, axis=0)
    i = np.argmin(abs(at.wl - 550))
    assert 0.095 < tau[i] < 0.099


def test_aerosol_column_normalisation():
    from skycolor.aerosol import HinterlandAerosol
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile

    at = Atmosphere(WL_COARSE, aerosols=[AerosolLayer(HinterlandAerosol(), 0.25,
                                                      ExponentialProfile(1.2))])
    beta = at.aerosol_coefficients(at.z_mid)[0]
    tau = np.sum(beta * np.diff(at.z_bounds)[:, None] * 1e3, axis=0)
    i = np.argmin(abs(at.wl - 550))
    assert tau[i] == pytest.approx(0.25, rel=0.02)


def test_phase_moments_normalised():
    at = Atmosphere.standard(WL_COARSE)
    m = at.phase_moments(8)
    assert np.allclose(m[:, 0, :], 1.0)


# ---------------------------------------------------------------------------
# facade completeness (architecture-refactor Step 3a)
# ---------------------------------------------------------------------------
#: every name the pre-Medium ``Atmosphere`` exposed, captured at the
#: pre-refactor baseline so the facade cannot silently drop one
#: ``_prepare_ozone``/``_prepare_no2``/``_o3_scale``/``_no2_scale`` were
#: captured in the pre-refactor surface; they are *removed* now (they served
#: only the retired inheritance protocol and were dead state on the facade),
#: which test_removed_transitional_protocol_enforces below pins mechanically.
FACADE_METHODS = frozenset({
    "_restrict", "aerosol_coefficients",
    "at_altitude", "delta_m_moments", "delta_m_properties", "layer_properties",
    "n_air", "n_no2", "n_o3", "no2_coefficient", "ozone_coefficient",
    "phase_moments", "pressure", "r_bounds", "rayleigh_coefficient",
    "standard", "temperature",
})
FACADE_ATTRIBUTES = frozenset({
    "wl", "ozone_du", "no2_column", "aerosols", "surface_pressure",
    "elevation_km", "refraction", "z_bounds", "z_mid", "ground_albedo",
    "_sigma_ray", "_sigma_o3", "_sigma_no2", "_ray_phase",
})


def test_facade_preserves_public_surface():
    """Every pre-refactor name must still resolve on the facade."""
    for name in FACADE_METHODS:
        assert hasattr(Atmosphere, name), name
    at = Atmosphere(np.array([400.0, 550.0, 650.0]), ozone_du=300.0, aerosols=[])
    for name in FACADE_ATTRIBUTES:
        assert hasattr(at, name), name


def test_medium_arrays_are_read_only():
    """``_sigma_*``/``wl`` are derived Medium state: assignment must fail
    loudly instead of silently desynchronising the facade from its Medium."""
    at = Atmosphere(np.array([400.0, 550.0]), ozone_du=300.0, aerosols=[])
    with pytest.raises(AttributeError):
        at._sigma_ray = at._sigma_ray
    with pytest.raises(AttributeError):
        at.wl = at.wl
    # the aerosols setter remains the sanctioned redefinition path; the
    # getter returns an immutable tuple (appending raises loudly)
    at.aerosols = []
    assert at.aerosols == ()
    with pytest.raises(AttributeError):
        at.aerosols.append(1)


# ---------------------------------------------------------------------------
# at_altitude: consistent truncation semantics (docs/ARCHITECTURE.md)
# ---------------------------------------------------------------------------
def test_at_altitude_truncates_at_the_site():
    a = Atmosphere.standard(WL_COARSE)
    b = a.at_altitude(2.4)
    assert b.z_bounds[0] == pytest.approx(2.4)
    assert b.r_bounds[0] == pytest.approx(R_EARTH + 2400.0)
    # site pressure is the US76 value at 2.4 km -- no double scaling
    assert b.pressure(2.4) == pytest.approx(75625.67696512025, rel=1e-3)
    # density at the site equals the base atmosphere's -- no renormalisation
    assert b.n_air(2.4) == pytest.approx(a.n_air(2.4), rel=1e-3)
    # no column rescaling: ozone and aerosol amounts are carried untouched
    assert b.ozone_du == a.ozone_du
    assert b.aerosols[0].aod550 == pytest.approx(a.aerosols[0].aod550)
    # the observer coincides with the bottom boundary for every backend
    assert b.elevation_km == 2.4


def test_at_altitude_zero_is_identity():
    a = Atmosphere.standard(WL_COARSE)
    b = a.at_altitude(0.0)
    np.testing.assert_array_equal(b.z_bounds, a.z_bounds)
    assert b.pressure(2.4) == pytest.approx(a.pressure(2.4))
    assert b.n_air(2.4) == pytest.approx(a.n_air(2.4))


def test_elevation_truncates_aerosol_column():
    """AOD above the site is the untouched profile's remaining column, not a
    rescaled copy (the old construction shrank it a second time)."""
    from skycolor.aerosol import HinterlandAerosol
    from skycolor.atmosphere import AerosolLayer, ExponentialProfile
    from skycolor.constants import R_EARTH

    wl = np.array([550.0])
    aod = 0.25
    at = Atmosphere(wl, aerosols=[AerosolLayer(HinterlandAerosol(), aod,
                                               ExponentialProfile(1.5))])
    high = at.at_altitude(3.0)
    assert high.z_bounds[0] == pytest.approx(3.0)
    beta = high.aerosol_coefficients(high.z_mid)[0]
    tau = np.sum(beta * np.diff(high.z_bounds)[:, None] * 1e3, axis=0)[0]
    remaining = float(np.exp(-3.0 / 1.5))   # fraction of an exp(1.5 km) column above 3 km
    # the layer-midpoint quadrature on the 1 km first layer is ~1.7 % coarse
    # for this profile; the point is "no second rescale", not exactness
    assert tau == pytest.approx(aod * remaining, rel=3e-2)


def test_at_altitude_rejects_out_of_domain_sites():
    """z0 at/above the model top (or below 0) would produce an empty layer
    stack; fail at the construction, not in a distant solver."""
    a = Atmosphere.standard(WL_COARSE)
    for bad in (200.0, 500.0, -1.0):
        with pytest.raises(ValueError, match="outside the model domain"):
            a.at_altitude(bad)


def test_at_altitude_layers_are_fresh_copies():
    """The derived atmosphere owns fresh AerosolLayer copies: mutating a
    derived layer must not leak into the source scene (issue F6)."""
    a = Atmosphere.standard(WL_COARSE)
    b = a.at_altitude(2.4)
    b.aerosols[0].aod550 = 0.99          # AerosolLayer stays a mutable record
    assert a.aerosols[0].aod550 == pytest.approx(0.05)   # source unaffected
    assert b.aerosols[0].aod550 == pytest.approx(0.99)


def test_removed_transitional_protocol_enforces():
    """Mechanical gate against a third round of "the record says deleted but
    the code is still there" (code review rounds 1-2, Major 1): the retired
    transitional protocol must have no remnants in skycolor/ -- no
    _prepare_* methods on the facade, no transitional-tense docstrings, and
    no facade-level _o3_scale/_no2_scale (the authoritative scales live on
    the Medium and are consumed by its n_o3/n_no2)."""
    import pathlib

    assert not hasattr(Atmosphere, "_prepare_ozone")
    assert not hasattr(Atmosphere, "_prepare_no2")
    at = Atmosphere(np.array([400.0, 550.0]), ozone_du=300.0, aerosols=[])
    assert not hasattr(at, "_o3_scale")
    assert not hasattr(at, "_no2_scale")
    banned = ("transitional", "_prepare_ozone", "_prepare_no2")
    root = pathlib.Path(__file__).resolve().parents[1] / "skycolor"
    for path in root.rglob("*.py"):
        text = path.read_text().lower()
        for token in banned:
            assert token not in text, f"{path.name} contains {token!r}"
