"""The physical medium: composition, discretisation and layer processing.

``Medium`` is a frozen value object holding (composition, wavelength grid,
layer boundaries) together with everything derived from them: the species
cross-sections, the density profiles and the layer-averaged optical
processing (``layer_properties`` / ``phase_moments`` / ``delta_m_*``).
Because every quantity it exposes is a pure function of its definitions,
its derived arrays live and die with the object -- there is no cross-object
mutable state, which is what makes frozen media safe to share and to cache
against (the aliasing hazards the task plan's F6 describes cannot arise).

What is deliberately *not* here: the observing scenario (observer height,
ground albedo, refraction) -- that lives in :class:`skycolor.scene.Scene`,
and :class:`skycolor.atmosphere.Atmosphere` is the facade pairing the two.
Custom profiles enter as data at construction (``ozone_profile`` /
``no2_profile`` / ``air_profile`` callables); every processing method reads
the medium's own profiles.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import spectra
from .constants import (
    DU,
    G0,
    M_AIR,
    P0,
    R_EARTH,
    T0,
    Z_TOA,
)

# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976
# ---------------------------------------------------------------------------
#: (base height [km], base temperature [K], lapse rate [K/km], base pressure [Pa])
_US76 = [
    (0.0, 288.15, -6.5, 101325.0),
    (11.0, 216.65, 0.0, 22632.06),
    (20.0, 216.65, 1.0, 5474.889),
    (32.0, 228.65, 2.8, 868.0187),
    (47.0, 270.65, 0.0, 110.9063),
    (51.0, 270.65, -2.8, 66.93887),
    (71.0, 214.65, -2.0, 3.956420),
    (84.852, 186.87, 0.0, 0.3734),
]
_GMR = G0 * M_AIR / 8.31432 * 1000.0  # ~34.16 K/km (z and lapse rate in km)


def us76_temperature(z_km: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 temperature [K]."""
    z = np.asarray(z_km, dtype=float)
    out = np.empty_like(z)
    for i, (hb, tb, lapse, _) in enumerate(_US76):
        top = _US76[i + 1][0] if i + 1 < len(_US76) else 1e9
        m = (z >= hb) & (z < top)
        out[m] = tb + lapse * (z[m] - hb)
    out[z >= _US76[-1][0]] = _US76[-1][1]
    out[z < 0] = T0
    return out


def us76_pressure(z_km: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 pressure [Pa]."""
    z = np.asarray(z_km, dtype=float)
    out = np.full_like(z, 1e-12)
    for i, (hb, tb, lapse, pb) in enumerate(_US76):
        top = _US76[i + 1][0] if i + 1 < len(_US76) else 1e9
        m = (z >= hb) & (z < top)
        dz = z[m] - hb
        if lapse == 0.0:
            out[m] = pb * np.exp(-_GMR * dz / tb)
        else:
            out[m] = pb * (tb / (tb + lapse * dz)) ** (_GMR / lapse)
    m = z >= _US76[-1][0]
    hb, tb, _, pb = _US76[-1]
    out[m] = pb * np.exp(-_GMR * (z[m] - hb) / tb)
    out[z < 0] = P0
    return out


def us76_number_density(z_km: np.ndarray) -> np.ndarray:
    """Dry-air number density [m^-3] from the US Standard Atmosphere."""
    return us76_pressure(z_km) / (1.380649e-23 * us76_temperature(z_km))


# ---------------------------------------------------------------------------
# Default vertical grid
# ---------------------------------------------------------------------------
#: layer boundaries in km; fine near the surface and through the ozone layer,
#: coarser in the upper mesosphere where nothing happens optically.
DEFAULT_Z_BOUNDS = np.array([
    0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 21.5, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 38.0, 41.0, 44.0,
    47.0, 50.0, 53.0, 56.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
    100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 150.0, 160.0,
    175.0, 200.0,
])

#: Above the US76 range the thermosphere is hot and tenuous; a simple
#: exponential tail with this scale height reproduces the observed density
#: (10^17 m^-3 at 120 km, ~10^14 at 200 km) well enough for twilight, whose
#: light below about -10 deg solar elevation comes from these altitudes.
THERMOSPHERE_SCALE_HEIGHT_KM = 15.0
THERMOSPHERE_START_KM = 120.0


# ---------------------------------------------------------------------------
# Climatological profiles
# ---------------------------------------------------------------------------
#: mid-latitude ozone number density [molecules cm^-3] vs height [km]
_O3_PROFILE = np.array([
    [0.0, 4.0e11], [5.0, 5.6e11], [10.0, 8.0e11], [13.0, 1.6e12],
    [15.0, 2.9e12], [17.0, 3.9e12], [20.0, 4.6e12], [22.0, 4.9e12],
    [25.0, 4.9e12], [28.0, 4.4e12], [30.0, 3.9e12], [33.0, 3.1e12],
    [35.0, 2.4e12], [38.0, 1.6e12], [40.0, 1.1e12], [42.0, 7.5e11],
    [45.0, 5.0e11], [48.0, 3.2e11], [50.0, 2.2e11], [55.0, 1.0e11],
    [60.0, 5.0e10], [70.0, 1.5e10], [80.0, 5.0e9], [120.0, 1.0e9],
    [200.0, 1.0e7],
])

#: stratospheric NO2 number density [molecules cm^-3] vs height [km]
_NO2_PROFILE = np.array([
    [0.0, 1.0e9], [10.0, 2.0e9], [15.0, 5.0e9], [20.0, 1.2e10],
    [25.0, 1.5e10], [30.0, 1.2e10], [35.0, 7.0e9], [40.0, 3.0e9],
    [50.0, 8.0e8], [60.0, 2.0e8], [120.0, 1.0e7], [200.0, 1.0e5],
])


def _log_interp(table: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Log-linear interpolation of ``table`` (col 0 = height, col 1 = value)."""
    zz = np.asarray(z, dtype=float)
    zt, vt = table[:, 0], np.log(table[:, 1])
    return np.exp(np.interp(np.clip(zz, zt[0], zt[-1]), zt, vt))


def _column_normalisation(table: np.ndarray) -> float:
    """Column [m^-2] of a number-density table [cm^-3] over the model domain."""
    z = np.linspace(0, Z_TOA / 1e3, 20000)
    n = _log_interp(table, z) * 1e6  # m^-3
    return float(np.trapezoid(n, z * 1e3))


#: column-normalisation grid for user-supplied profile *shapes*.  This is the
#: grid the profile-shape experiments have always used (``_Z_DIAG`` there);
#: keeping it makes the data path bit-comparable with the historical subclass
#: path.  The built-in tables above normalise on their own finer grid, which
#: their numbers have always been pinned to.
_PROFILE_COLUMN_GRID = np.linspace(0.0, Z_TOA / 1e3, 4001)


def _column_of_shape(shape: Callable[[np.ndarray], np.ndarray]) -> float:
    """Column [m^-2] of a user profile *shape* in arbitrary units."""
    return float(np.trapezoid(np.asarray(shape(_PROFILE_COLUMN_GRID), dtype=float),
                              _PROFILE_COLUMN_GRID * 1e3))


# ---------------------------------------------------------------------------
# The medium
# ---------------------------------------------------------------------------
class _RefractionToken:
    """Identity sentinel for the refraction-table cache.

    A dedicated class (rather than a bare ``object()``) because cache keys
    must be weak-referenceable: the cache holds its entries only while some
    living medium holds the token.
    """


@dataclass(frozen=True)
class Medium:
    """A spherically stratified atmosphere as a frozen value object.

    Parameters
    ----------
    wl : array
        Spectral grid in nanometres.
    z_bounds : array
        Layer boundaries of the discretisation [km].
    ozone_du : float
        Total ozone column in Dobson units.
    no2_column : float
        Total NO2 column in molecules m^-2 (0 disables it).
    aerosols : sequence of AerosolLayer
    surface_pressure : float
        Sea-level pressure scaling of the air profile.
    ozone_profile, no2_profile : callable, optional
        Vertical *shape* of the species in arbitrary units, column-normalised
        to ``ozone_du`` / ``no2_column`` exactly like the built-in tables.
        This is the sanctioned extension point for custom profiles (the
        inheritance-based protocol it replaces was retired with Step 3b).
    air_profile : callable, optional
        Complete dry-air number density [m^-3] as a function of height [km];
        replaces the US76 pressure/temperature construction outright.  When
        given, the ``pressure()``/``temperature()`` accessors still report
        the *default* US76 profiles and no longer describe this medium.
    """

    wl: np.ndarray
    z_bounds: np.ndarray
    ozone_du: float = 300.0
    no2_column: float = 0.0
    aerosols: Sequence = ()
    surface_pressure: float = P0
    ozone_profile: Callable[[np.ndarray], np.ndarray] | None = None
    no2_profile: Callable[[np.ndarray], np.ndarray] | None = None
    air_profile: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        # identity token of the density definition, handed to the refraction
        # table cache (see :mod:`skycolor.refraction`).  Deliberately NOT a
        # dataclass field: a freshly constructed medium gets a fresh token
        # ("separately constructed media never share refraction tables"),
        # while restrict() shares the source's token through copy.copy --
        # a wavelength subset never changes the density definition.
        object.__setattr__(self, "_refraction_token", _RefractionToken())
        # profile normalisations
        o3_col = (_column_of_shape(self.ozone_profile) if self.ozone_profile is not None
                  else _column_normalisation(_O3_PROFILE))
        no2_col = (_column_of_shape(self.no2_profile) if self.no2_profile is not None
                   else _column_normalisation(_NO2_PROFILE))
        object.__setattr__(self, "_o3_scale", self.ozone_du * DU / o3_col)
        object.__setattr__(self, "_no2_scale",
                           (self.no2_column / no2_col) if no2_col > 0 else 0.0)
        # cached spectral quantities
        object.__setattr__(self, "_sigma_ray", spectra.rayleigh_cross_section(self.wl))
        object.__setattr__(self, "_sigma_o3", spectra.ozone_cross_section(self.wl))
        object.__setattr__(self, "_sigma_no2", spectra.no2_cross_section(self.wl))
        object.__setattr__(self, "_ray_phase", spectra.rayleigh_phase_moments())

    # -- derived grids --------------------------------------------------------
    @property
    def z_mid(self) -> np.ndarray:
        return 0.5 * (self.z_bounds[1:] + self.z_bounds[:-1])

    @property
    def r_bounds(self) -> np.ndarray:
        return R_EARTH + self.z_bounds * 1e3

    @property
    def refraction_token(self) -> object:
        """Identity token of the density definition (see ``__post_init__``)."""
        return self._refraction_token

    # -- vertical profiles ----------------------------------------------------
    def pressure(self, z_km):
        z = np.asarray(z_km, dtype=float)
        return us76_pressure(z) * (self.surface_pressure / P0)

    def temperature(self, z_km):
        return us76_temperature(np.asarray(z_km, dtype=float))

    def n_air(self, z_km) -> np.ndarray:
        """Dry-air number density [m^-3], with a thermospheric tail."""
        if self.air_profile is not None:
            return np.asarray(self.air_profile(np.asarray(z_km, dtype=float)),
                              dtype=float)
        pressure = self.pressure
        temperature = self.temperature
        z = np.asarray(z_km, dtype=float)
        n = pressure(z) / (1.380649e-23 * temperature(z))
        n0 = pressure(np.array(THERMOSPHERE_START_KM)) / (
            1.380649e-23 * temperature(np.array(THERMOSPHERE_START_KM)))
        tail = n0 * np.exp(-(z - THERMOSPHERE_START_KM) / THERMOSPHERE_SCALE_HEIGHT_KM)
        return np.where(z > THERMOSPHERE_START_KM, tail, n)

    def n_o3(self, z_km):
        if self.ozone_profile is not None:
            return (np.asarray(self.ozone_profile(np.asarray(z_km, dtype=float)),
                               dtype=float) * self._o3_scale)
        return _log_interp(_O3_PROFILE, z_km) * 1e6 * self._o3_scale

    def n_no2(self, z_km):
        if self.no2_profile is not None:
            return (np.asarray(self.no2_profile(np.asarray(z_km, dtype=float)),
                               dtype=float) * self._no2_scale)
        return _log_interp(_NO2_PROFILE, z_km) * 1e6 * self._no2_scale

    # -- optical properties ---------------------------------------------------
    def rayleigh_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        """Rayleigh scattering coefficient [m^-1]."""
        n = self.n_air(z_km)
        s = self._sigma_ray if wl_nm is None else spectra.rayleigh_cross_section(np.asarray(wl_nm))
        return np.multiply.outer(np.atleast_1d(n), s)

    def ozone_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        n = self.n_o3(z_km)
        s = self._sigma_o3 if wl_nm is None else spectra.ozone_cross_section(np.asarray(wl_nm))
        return np.multiply.outer(np.atleast_1d(n), s)

    def no2_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        n = self.n_no2(z_km)
        s = self._sigma_no2 if wl_nm is None else spectra.no2_cross_section(np.asarray(wl_nm))
        return np.multiply.outer(np.atleast_1d(n), s)

    def aerosol_coefficients(self, z_km) -> tuple[np.ndarray, np.ndarray]:
        """Aerosol extinction and scattering coefficients [m^-1].

        Returns ``(beta_ext, beta_sca)`` with shape ``(n_z, n_wl)``.
        """
        z = np.atleast_1d(np.asarray(z_km, dtype=float))
        ext = np.zeros((len(z), len(self.wl)))
        sca = np.zeros_like(ext)
        for lay in self.aerosols:
            rel_ext, rel_sca = lay.aerosol.relative_coefficients(self.wl)
            w = lay.aod550 * lay.profile(z) / 1000.0
            ext += np.multiply.outer(w, rel_ext)
            sca += np.multiply.outer(w, rel_sca)
        return ext, sca

    # -- layer processing -----------------------------------------------------
    def layer_properties(self, z_low=None, z_high=None) -> dict:
        """Optical properties evaluated at the layer mid-points.

        Returns a dict with ``beta_ray``, ``beta_ae`` (aerosol extinction),
        ``beta_as`` (aerosol scattering), ``beta_o3``, ``beta_no2``,
        ``beta_ext`` (total), ``beta_sca`` (total), all shape ``(n_layer, n_wl)``.
        """
        z = self.z_mid if z_low is None else 0.5 * (z_low + z_high)
        beta_ray = self.rayleigh_coefficient(z)
        beta_o3 = self.ozone_coefficient(z)
        beta_no2 = self.no2_coefficient(z)
        beta_ae, beta_as = self.aerosol_coefficients(z)
        beta_sca = beta_ray + beta_as
        beta_ext = beta_ray + beta_ae + beta_o3 + beta_no2
        return dict(
            z=z, beta_ray=beta_ray, beta_ae=beta_ae, beta_as=beta_as,
            beta_o3=beta_o3, beta_no2=beta_no2,
            beta_ext=beta_ext, beta_sca=beta_sca,
        )

    def phase_moments(self, n_moments: int = 32) -> np.ndarray:
        """Legendre moments of the layer-averaged phase function.

        Shape ``(n_layer, n_moments, n_wl)``; ``p(cos t) = sum_l (2l+1) b_l P_l``.

        These are the *raw* moments.  A discrete-ordinates code that truncates
        at ``n_moments`` must use :meth:`delta_m_moments` instead, otherwise the
        forward peak that lives beyond the truncation is silently discarded --
        which for dust and cloud droplets is 37 % of the scattered energy.
        """
        props = self.layer_properties()
        nz, nwl = props["beta_ray"].shape
        out = np.zeros((nz, n_moments, nwl))
        beta_sca = np.maximum(props["beta_sca"], 1e-300)
        # Rayleigh contribution
        for l, b in enumerate(np.pad(self._ray_phase, (0, n_moments - len(self._ray_phase)))):
            out[:, l, :] += b * props["beta_ray"] / beta_sca
        # aerosol contribution
        for lay in self.aerosols:
            _, rel_sca = lay.aerosol.relative_coefficients(self.wl)
            mom = lay.aerosol.legendre_moments(self.wl, n_moments)  # (n_mom, n_wl)
            w = lay.aod550 * lay.profile(self.z_mid) / 1000.0  # (nz,) bulk extinction scale
            bs = np.multiply.outer(w, rel_sca)  # absolute aerosol scattering coeff
            for l in range(n_moments):
                out[:, l, :] += mom[l][None, :] * bs / beta_sca
        return out

    def delta_m_moments(self, n_moments: int = 32):
        """delta-M-scaled moments and the forward fraction.

        Returns ``(moments, f)`` with ``moments`` of shape
        ``(n_layer, n_moments, n_wl)`` and ``f`` of shape
        ``(n_layer, n_wl)``.  ``f = beta_{n_moments}`` is the fraction of the
        scattered energy that the truncated expansion would lose.  The
        transformation (Wiscombe 1977) is

            f        = beta_N
            beta*_l  = (beta_l - f) / (1 - f)          l = 0 .. N-1
            tau*     = tau (1 - omega f)
            omega*   = omega (1 - f) / (1 - omega f)

        Use :meth:`delta_m_properties` to get the rescaled optical properties
        consistently with these moments.
        """
        raw = self.phase_moments(n_moments + 1)
        f = np.clip(raw[:, n_moments, :], 0.0, 0.999)
        beta = (raw[:, :n_moments, :] - f[:, None, :]) / (1.0 - f[:, None, :])
        beta[:, 0, :] = 1.0
        return beta, f

    def delta_m_properties(self, n_moments: int = 32) -> dict:
        """Layer optical properties rescaled by delta-M, plus the moments."""
        props = self.layer_properties()
        beta, f = self.delta_m_moments(n_moments)
        beta_ext = np.maximum(props["beta_ext"], 0.0)
        beta_sca = np.maximum(props["beta_sca"], 0.0)
        omega = np.where(beta_ext > 1e-300, beta_sca / np.maximum(beta_ext, 1e-300), 0.0)
        new_ext = beta_ext * (1.0 - omega * f)
        new_sca = beta_sca * (1.0 - f)
        out = dict(props)
        out["beta_ext"] = new_ext
        out["beta_sca"] = new_sca
        out["beta_ray"] = props["beta_ray"] * (1.0 - f)   # only used via the moments
        out["delta_m_f"] = f
        out["moments"] = beta
        return out

    # -- wavelength subsets ---------------------------------------------------
    def restrict(self, index) -> "Medium":
        """A copy of this medium restricted to the wavelengths ``index``.

        The wavelength-dependent arrays are *sliced*, not recomputed, both for
        speed and because slicing is bit-exact where a recomputation through
        the tables is only guaranteed elementwise-equal.  Everything the
        restriction does not touch (profiles, aerosols, ``z_bounds``, and the
        refraction identity token -- a wavelength subset never changes the
        density definition) is shared with the source medium.
        """
        idx = np.asarray(index, dtype=int)
        new = copy.copy(self)
        object.__setattr__(new, "wl", self.wl[idx])
        object.__setattr__(new, "_sigma_ray", self._sigma_ray[idx])
        object.__setattr__(new, "_sigma_o3", self._sigma_o3[idx])
        object.__setattr__(new, "_sigma_no2", self._sigma_no2[idx])
        return new
