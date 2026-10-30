"""Vertical-profile variants used by experiment 12.

The production model fixes two vertical profiles that the twilight brightness
is sensitive to: the ozone number density (a climatological table in
:mod:`skycolor.atmosphere`) and the air temperature/density (US Standard
Atmosphere 1976).  Both are *shapes*, not column amounts -- the column of ozone
is constrained by Dobson measurements, the surface pressure by the site --
yet neither shape is constrained by any measurement used in this project.

This module builds variants of those shapes **without touching**
``skycolor/``.  Everything is an :class:`~skycolor.atmosphere.Atmosphere`
subclass that replaces exactly one profile and leaves the rest of the model
(the Rayleigh cross-sections, the solar beam, the RT integrators) untouched:

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

All variants use the same layer grid as the production atmosphere, so the
radiative-transfer solvers need no special-casing.  Each class also carries a
``_restrict(index)`` method (the wavelength subset used by the spherical
engine's per-anchor Monte Carlo) and ``install_twilight_profile_support``
patches the engine to use it -- see the function's docstring for why.

Run ``uv run python experiments/12_profile_sensitivity.py`` for the sweep.
"""
from __future__ import annotations

import numpy as np

from skycolor.atmosphere import Atmosphere, us76_pressure, us76_temperature
from skycolor.constants import DU, G0, M_AIR, P0, Z_TOA

__all__ = [
    "OzoneShapeAtmosphere",
    "OffsetTemperatureAtmosphere",
    "BandDensityAtmosphere",
    "climatological_shape",
    "shifted_shape",
    "gaussian_shape",
    "tropospheric_shape",
    "install_twilight_profile_support",
    "ozone_diagnostics",
    "air_diagnostics",
]

#: ``G0 M / R*`` in K km^-1, the same constant the US76 implementation uses
#: (``_GMR`` in ``skycolor/atmosphere.py``).  Recomputed here rather than
#: imported because it is only needed by this experiment.
_GMR = G0 * M_AIR / 8.31432 * 1000.0  # ~34.16 K/km

#: One-DU reference atmosphere; ``.n_o3(z)`` is the climatological *shape*
#: with its absolute scale carried by the 1 DU normalisation.  Used so that
#: shifted/segmented variants can be expressed relative to the table without
#: importing the private ``_O3_PROFILE`` array.
_PROBE = Atmosphere(np.array([550.0]), ozone_du=1.0, aerosols=[])

#: height grid for the column diagnostics (0.05 km, 0-200 km)
_Z_DIAG = np.linspace(0.0, Z_TOA / 1e3, 4001)


def _restricted_kwargs(atmo, index):
    """Constructor keywords that copy ``atmo`` onto the wavelength subset."""
    return dict(
        no2_column=atmo.no2_column,
        aerosols=list(atmo.aerosols),
        ground_albedo=np.asarray(atmo.ground_albedo, dtype=float)[index],
        surface_pressure=atmo.surface_pressure,
        z_bounds_km=np.asarray(atmo.z_bounds, dtype=float),
        elevation_km=atmo.elevation_km,
        refraction=atmo.refraction,
    )


# ---------------------------------------------------------------------------
# Ozone shapes (arbitrary units; the class normalises to the column)
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

    Normalisation is irrelevant (the class renormalises); this is the *shape*,
    so ``sigma`` is a geometric (1/e in density) half-width in km.
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


class OzoneShapeAtmosphere(Atmosphere):
    """An atmosphere with an arbitrary ozone vertical shape.

    Parameters
    ----------
    wl_nm :
        Spectral grid.
    shape : callable
        Ozone number density in arbitrary units; renormalised so that the
        vertical column equals ``ozone_du``.
    ozone_du :
        Prescribed column [DU].
    **kw :
        Passed to :class:`~skycolor.atmosphere.Atmosphere`.
    """

    def __init__(self, wl_nm, shape, ozone_du: float = 300.0, **kw):
        self._o3_shape = shape
        super().__init__(wl_nm, ozone_du=ozone_du, **kw)

    def _prepare_ozone(self) -> None:
        """Column-normalise the user shape (the table is irrelevant here)."""
        col = float(np.trapezoid(self._o3_shape(_Z_DIAG), _Z_DIAG * 1e3))
        self._o3_scale = self.ozone_du * DU / col

    def n_o3(self, z_km):
        return self._o3_shape(np.asarray(z_km, dtype=float)) * self._o3_scale

    def _restrict(self, index):
        """Same variant on a subset of the spectral grid (engine support)."""
        idx = np.asarray(index, dtype=int)
        return type(self)(
            np.asarray(self.wl, dtype=float)[idx], self._o3_shape,
            ozone_du=self.ozone_du, **_restricted_kwargs(self, idx))


# ---------------------------------------------------------------------------
# Temperature / density above 20 km
# ---------------------------------------------------------------------------
class OffsetTemperatureAtmosphere(Atmosphere):
    """US76 with a constant temperature offset above ``z_anchor_km``.

    The pressure is re-integrated hydrostatically from the (unperturbed)
    pressure at the anchor, so ``p(0)`` and ``p(z_anchor)`` are unchanged and
    only the density *between 20 and 50 km* -- the part of the atmosphere the
    terminator sweeps through at twilight -- is redistributed.  The offset is
    discontinuous at the anchor by construction ("a simple additive offset,
    keeping the surface anchored").
    """

    def __init__(self, wl_nm, delta_t: float = 0.0, z_anchor_km: float = 20.0,
                 ramp_km: float = 0.0, **kw):
        self.delta_t = float(delta_t)
        self.z_anchor_km = float(z_anchor_km)
        #: width of a linear ramp above the anchor [km].  ``0`` is a step
        #: (the "simple additive offset"); a positive value keeps the density
        #: continuous at the anchor and is used only as a robustness check.
        self.ramp_km = float(ramp_km)
        super().__init__(wl_nm, **kw)
        self._build_pressure_table()

    # -- profiles ------------------------------------------------------------
    def temperature(self, z_km):
        z = np.asarray(z_km, dtype=float)
        if self.ramp_km <= 0.0:
            w = (z > self.z_anchor_km).astype(float)
        else:
            w = np.clip((z - self.z_anchor_km) / self.ramp_km, 0.0, 1.0)
        return us76_temperature(z) + self.delta_t * w

    def _build_pressure_table(self) -> None:
        """Hydrostatic pressure from the anchor to the top, on a fine grid."""
        z = np.arange(self.z_anchor_km, Z_TOA / 1e3 + 1e-9, 0.02)
        t = self.temperature(z)
        # p_{i+1} = p_i exp(-GMR dz / T_mid), second order in dz
        exponent = np.concatenate((
            [0.0],
            np.cumsum(_GMR * np.diff(z) / (0.5 * (t[:-1] + t[1:]))),
        ))
        p_anchor = float(us76_pressure(np.array([self.z_anchor_km]))[0])
        self._p_z = z
        self._p_log = np.log(p_anchor) - exponent

    def pressure(self, z_km):
        z = np.asarray(z_km, dtype=float)
        base = us76_pressure(z) * (self.surface_pressure / P0)
        perturbed = np.exp(np.interp(z, self._p_z, self._p_log))
        return np.where(z > self.z_anchor_km, perturbed, base)

    def _restrict(self, index):
        """Same variant on a subset of the spectral grid (engine support)."""
        idx = np.asarray(index, dtype=int)
        return type(self)(
            np.asarray(self.wl, dtype=float)[idx], self.delta_t,
            z_anchor_km=self.z_anchor_km, ramp_km=self.ramp_km,
            **_restricted_kwargs(self, idx))


# ---------------------------------------------------------------------------
# Band-density probe (a contribution function, not a physical atmosphere)
# ---------------------------------------------------------------------------
class BandDensityAtmosphere(Atmosphere):
    """Multiply ``n_air`` by ``factor`` inside ``[z0_km, z1_km]``.

    This is a *diagnostic*, not a physical atmosphere: it breaks hydrostatic
    consistency on purpose, because the question it answers is "how much does
    the radiance care about the air density in this altitude band?".  The
    finite-difference response at ``factor = 1 +/- eps`` is the logarithmic
    contribution function ``d ln L / d ln n`` of that band.
    """

    def __init__(self, wl_nm, band_km, factor: float = 1.05, **kw):
        self.band_km = (float(band_km[0]), float(band_km[1]))
        self.factor = float(factor)
        super().__init__(wl_nm, **kw)

    def n_air(self, z_km):
        n = super().n_air(z_km)
        z = np.asarray(z_km, dtype=float)
        z0, z1 = self.band_km
        return np.where((z >= z0) & (z < z1), n * self.factor, n)

    def _restrict(self, index):
        """Same variant on a subset of the spectral grid (engine support)."""
        idx = np.asarray(index, dtype=int)
        return type(self)(
            np.asarray(self.wl, dtype=float)[idx], self.band_km,
            factor=self.factor, **_restricted_kwargs(self, idx))


def install_twilight_profile_support():
    """Let the spherical engine's Monte Carlo see these subclass overrides.

    ``skycolor.rt.twilight._sub_atmosphere`` rebuilds a **plain**
    ``Atmosphere`` from ``ozone_du`` / ``surface_pressure`` before running the
    single-wavelength Monte Carlo, which silently discards any subclass
    override of ``n_o3`` or ``n_air``.  The engine's first-order term is
    evaluated on the real object and does see the override, so a naive run
    returns a *hybrid*: a corrected single-scattering term plus an uncorrected
    multiple-scattering term.

    The fix is a one-line change upstream (call ``atmo._restrict(index)`` when
    it exists).  ``skycolor/`` is read-only for this experiment, so the
    function below patches the module attribute at run time and returns the
    module.  Call it before ``twilight_radiance``; it is idempotent.
    """
    from skycolor.rt import twilight as _tw

    if getattr(_tw, "_profile_restrict_patch", False):
        return _tw
    original = _tw._sub_atmosphere

    def patched(atmo, index):
        if not hasattr(atmo, "_restrict"):
            return original(atmo, index)
        idx = np.asarray(index, dtype=int)
        if np.array_equal(idx, np.arange(np.asarray(atmo.wl).size)):
            return atmo
        return atmo._restrict(idx)

    _tw._sub_atmosphere = patched
    _tw._profile_restrict_patch = True
    return _tw


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
