"""Vertical-profile variants used by experiment 12.

The production model fixes two vertical profiles that the twilight brightness
is sensitive to: the ozone number density (a climatological table in
:mod:`skycolor.atmosphere`) and the air temperature/density (US Standard
Atmosphere 1976).  Both are *shapes*, not column amounts -- the column of ozone
is constrained by Dobson measurements, the surface pressure by the site --
yet neither shape is constrained by any measurement used in this project.

This module builds variants of those shapes **without touching**
``skycolor/``.  Every variant is a thin constructor over the data extension
points of :class:`~skycolor.atmosphere.Atmosphere` -- ``ozone_profile`` /
``air_profile`` callables injected into the frozen ``Medium`` -- so the
custom profiles flow through *every* backend (single scattering, SOS, the
twilight engine's wavelength subsets) with no patching and no subclassing:

``OzoneShapeAtmosphere``
    Arbitrary ozone density shape, renormalised to a prescribed column.  The
    climatological profile itself is read back from the model
    (``Atmosphere(ozone_du=1).n_o3``) so there is only one authoritative copy
    of the table.

``OffsetTemperatureAtmosphere``
    US76 temperature with a constant additive offset above ``z_anchor_km``.
    The pressure is re-integrated hydrostatically so that the surface stays
    anchored -- the point is to change the *scale height* between 20 and
    50 km, not the surface pressure or the total column.

``BandDensityAtmosphere``
    A contribution-function diagnostic: ``n_air`` multiplied by ``factor``
    inside an altitude band, breaking hydrostatic consistency on purpose.

All variants use the same layer grid as the production atmosphere, so the
radiative-transfer solvers need no special-casing.  The twilight engine
picks up the injected profiles through ``Atmosphere._restrict`` (its
native wavelength-subset path), so no runtime patching of the engine is
needed.

Run ``uv run python experiments/12_profile_sensitivity.py`` for the sweep.
"""
from __future__ import annotations

import numpy as np

from skycolor.atmosphere import (
    Atmosphere,
    THERMOSPHERE_SCALE_HEIGHT_KM,
    THERMOSPHERE_START_KM,
    us76_pressure,
    us76_temperature,
)
from skycolor.constants import DU, G0, M_AIR, P0, Z_TOA

__all__ = [
    "OzoneShapeAtmosphere",
    "OffsetTemperatureAtmosphere",
    "BandDensityAtmosphere",
    "climatological_shape",
    "shifted_shape",
    "gaussian_shape",
    "tropospheric_shape",
    "ozone_diagnostics",
    "air_diagnostics",
]

#: ``G0 M / R*`` in K km^-1, the same constant the US76 implementation uses
#: (``_GMR`` in ``skycolor/medium.py``).  Recomputed here rather than
#: imported because it is only needed by this experiment.
_GMR = G0 * M_AIR / 8.31432 * 1000.0  # ~34.16 K/km

#: One-DU reference atmosphere; ``.n_o3(z)`` is the climatological *shape*
#: with its absolute scale carried by the 1 DU normalisation.  Used so that
#: shifted/segmented variants can be expressed relative to the table without
#: importing the private ``_O3_PROFILE`` array.
_PROBE = Atmosphere(np.array([550.0]), ozone_du=1.0, aerosols=[])

#: height grid for the column diagnostics (0.05 km, 0-200 km)
_Z_DIAG = np.linspace(0.0, Z_TOA / 1e3, 4001)


# ---------------------------------------------------------------------------
# Ozone shapes (arbitrary units; the constructors normalise to the column)
# ---------------------------------------------------------------------------
def climatological_shape(z_km):
    """The climatological ozone shape [m^-3], normalised to a 1 DU column."""
    return _PROBE.n_o3(np.asarray(z_km, dtype=float))


def shifted_shape(delta_km: float):
    """The climatological shape translated in height by ``delta_km``.

    ``n(z) = n_clim(z - delta)``.  Below the surface the table is extended at
    its 0 km value (the same clamp ``_log_interp`` applies), so an upward
    shift fills the lowest kilometres with surface-level ozone instead of
    leaving a void.  The result is renormalised to the requested column.
    """
    dz = float(delta_km)

    def shape(z_km):
        return _PROBE.n_o3(np.asarray(z_km, dtype=float) - dz)

    shape.delta_km = dz
    return shape


def gaussian_shape(sigma_km: float, center_km: float = 25.0):
    """A Gaussian ozone layer centred at ``center_km`` with width ``sigma_km``.

    Normalisation is irrelevant (the constructors renormalise); this is the
    *shape*, so ``sigma`` is a geometric (1/e in density) half-width in km.
    """
    c, s = float(center_km), float(sigma_km)

    def shape(z_km):
        z = np.asarray(z_km, dtype=float)
        return np.exp(-0.5 * ((z - c) / s) ** 2)

    shape.center_km = c
    shape.sigma_km = s
    return shape


def tropospheric_shape(trop_du: float, split_km: float = 10.0,
                       total_du: float = 300.0):
    """``trop_du`` of ozone below ``split_km``, the remainder above.

    Both segments keep the climatological *relative* shape, so this lever
    changes only the partition of the column between the troposphere and the
    stratosphere.
    """
    trop = float(trop_du)
    split = float(split_km)
    z = _Z_DIAG
    s = climatological_shape(z)
    below = z < split
    col_below = float(np.trapezoid(np.where(below, s, 0.0), z * 1e3))
    col_above = float(np.trapezoid(np.where(~below, s, 0.0), z * 1e3))

    def shape(z_km):
        zz = np.asarray(z_km, dtype=float)
        s_zz = climatological_shape(zz)
        lo = zz < split
        return np.where(
            lo,
            trop * DU * s_zz / col_below,
            (total_du - trop) * DU * s_zz / col_above,
        )

    shape.trop_du = trop
    shape.split_km = split
    return shape


# ---------------------------------------------------------------------------
# Data-path constructors (custom profiles as injected callables)
# ---------------------------------------------------------------------------
def OzoneShapeAtmosphere(wl_nm, shape, ozone_du: float = 300.0, **kw):
    """An atmosphere with an arbitrary ozone vertical *shape*.

    The shape (arbitrary units) is column-normalised to ``ozone_du`` by the
    medium -- the same semantics the historical ``OzoneShapeAtmosphere``
    subclass implemented by overriding ``_prepare_ozone``/``n_o3``.  Thin
    constructor over ``Atmosphere(ozone_profile=...)``.
    """
    return Atmosphere(wl_nm, ozone_du=ozone_du, ozone_profile=shape, **kw)


def OffsetTemperatureAtmosphere(wl_nm, delta_t: float = 0.0, z_anchor_km: float = 20.0,
                                ramp_km: float = 0.0, **kw):
    """US76 with a constant temperature offset above ``z_anchor_km``.

    The pressure is re-integrated hydrostatically from the (unperturbed)
    pressure at the anchor, so ``p(0)`` and ``p(z_anchor)`` are unchanged and
    only the density between the anchor and the model top is redistributed.
    The offset is discontinuous at the anchor by construction ("a simple
    additive offset, keeping the surface anchored"); a positive ``ramp_km``
    keeps the density continuous there and is used only as a robustness
    check.  Thin constructor over ``Atmosphere(air_profile=...)``: the
    profile callable replicates the historical subclass math
    operation-for-operation (the migration dual-run was bit-exact).
    """
    anchor = float(z_anchor_km)
    ramp = float(ramp_km)
    surface_pressure = float(kw.get("surface_pressure", P0))
    z_tab = np.arange(anchor, Z_TOA / 1e3 + 1e-9, 0.02)

    def temperature(z_km):
        z = np.asarray(z_km, dtype=float)
        if ramp <= 0.0:
            w = (z > anchor).astype(float)
        else:
            w = np.clip((z - anchor) / ramp, 0.0, 1.0)
        return us76_temperature(z) + delta_t * w

    t = temperature(z_tab)
    exponent = np.concatenate((
        [0.0],
        np.cumsum(_GMR * np.diff(z_tab) / (0.5 * (t[:-1] + t[1:]))),
    ))
    p_anchor = float(us76_pressure(np.array([anchor]))[0])
    p_log = np.log(p_anchor) - exponent

    def n_air(z_km):
        z = np.asarray(z_km, dtype=float)
        base = us76_pressure(z) * (surface_pressure / P0)
        perturbed = np.exp(np.interp(z, z_tab, p_log))
        p = np.where(z > anchor, perturbed, base)
        t_z = temperature(z)
        n = p / (1.380649e-23 * t_z)
        z_top = np.array(THERMOSPHERE_START_KM)
        p_top = (np.exp(np.interp(z_top, z_tab, p_log))
                 if z_top > anchor
                 else us76_pressure(z_top) * (surface_pressure / P0))
        n0 = p_top / (1.380649e-23 * temperature(z_top))
        tail = n0 * np.exp(-(z - THERMOSPHERE_START_KM) / THERMOSPHERE_SCALE_HEIGHT_KM)
        return np.where(z > THERMOSPHERE_START_KM, tail, n)

    return Atmosphere(wl_nm, air_profile=n_air, **kw)


def BandDensityAtmosphere(wl_nm, band_km, factor: float = 1.05, **kw):
    """Multiply ``n_air`` by ``factor`` inside ``[z0_km, z1_km]``.

    This is a *diagnostic*, not a physical atmosphere: it breaks hydrostatic
    consistency on purpose, because the question it answers is "how much does
    the radiance care about the air density in this altitude band?".  The
    finite-difference response at ``factor = 1 +/- eps`` is the logarithmic
    contribution function ``d ln L / d ln n`` of that band.  Thin constructor
    over ``Atmosphere(air_profile=...)``.
    """
    z0, z1 = (float(band_km[0]), float(band_km[1]))
    base = Atmosphere(wl_nm, **kw)

    def n_air(z_km):
        n = base.n_air(z_km)
        z = np.asarray(z_km, dtype=float)
        return np.where((z >= z0) & (z < z1), n * factor, n)

    return Atmosphere(wl_nm, air_profile=n_air, **kw)


# ---------------------------------------------------------------------------
# Column / shape diagnostics
# ---------------------------------------------------------------------------
def ozone_diagnostics(atmo, split_km: float = 10.0) -> dict:
    """Column, peak height, centroid, median height and tropospheric DU."""
    z = _Z_DIAG
    n = np.asarray(atmo.n_o3(z), dtype=float)
    col = float(np.trapezoid(n, z * 1e3))
    below = z < split_km
    trop = float(np.trapezoid(np.where(below, n, 0.0), z * 1e3))
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (n[:-1] + n[1:]) * np.diff(z * 1e3))))
    total = cum[-1]
    h50 = float(np.interp(0.5 * total, cum, z))
    return {
        "ozone_column_du": col / DU,
        "tropospheric_du": trop / DU,
        "peak_km": float(z[int(np.argmax(n))]),
        "centroid_km": float(np.trapezoid(n * z, z * 1e3) / col) if col > 0 else 0.0,
        "median_km": h50,
    }


def air_diagnostics(atmo) -> dict:
    """Vertical Rayleigh optical depth and the 20-50 km density weight."""
    z = _Z_DIAG
    n = np.asarray(atmo.n_air(z), dtype=float)
    col = float(np.trapezoid(n, z * 1e3))
    band = (z >= 20.0) & (z <= 50.0)
    col_band = float(np.trapezoid(np.where(band, n, 0.0), z * 1e3))
    #: density at the 35 km terminator height at -6 deg solar elevation
    n35 = float(atmo.n_air(np.array([35.0]))[0])
    from skycolor import spectra

    sigma550 = float(spectra.rayleigh_cross_section(np.array([550.0]))[0])
    return {
        "air_column_molecules_m2": col,
        "rayleigh_tau_550": col * sigma550,
        "air_column_20_50_frac": col_band / col,
        "n_air_35km": n35,
    }
