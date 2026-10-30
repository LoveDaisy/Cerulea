"""Vertical structure of the atmosphere and its optical properties.

The model atmosphere is a set of concentric spherical shells.  Within each
shell the composition is described by continuous profiles so that the
radiative-transfer integrators can evaluate the local optical properties at an
arbitrary height:

* dry air (US Standard Atmosphere 1976) -> Rayleigh scattering
* ozone (climatological profile, normalised to a column amount in DU)
* nitrogen dioxide (optional, stratospheric)
* one or more aerosol layers, each with a normalised vertical profile

The *aerosol amount* is specified as an optical depth at 550 nm, which is the
quantity that is actually measured by sun photometers, while the spectral
dependence and the scattering properties come from a
:class:`~skycolor.aerosol.Aerosol` object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import spectra
from .constants import (
    DU,
    G0,
    M_AIR,
    P0,
    R_EARTH,
    R_SPECIFIC_AIR,
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
# Ozone climatology
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


# ---------------------------------------------------------------------------
# Vertical profiles for aerosols
# ---------------------------------------------------------------------------
class VerticalProfile:
    """A normalised vertical distribution of aerosol (integrates to 1)."""

    def __call__(self, z_km: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def column(self, z0: float = 0.0, z1: float = np.inf) -> float:
        """Fraction of the column between ``z0`` and ``z1`` (km)."""
        zz = np.linspace(max(z0, 0.0), min(z1, Z_TOA / 1e3), 4000)
        v = self(zz)
        return float(np.trapezoid(v, zz))


class ExponentialProfile(VerticalProfile):
    """Exponential profile ``exp(-z/H)`` truncated at the model top."""

    def __init__(self, scale_height_km: float = 1.0):
        self.H = float(scale_height_km)

    def __call__(self, z_km):
        return np.exp(-np.clip(np.asarray(z_km, dtype=float), 0, None) / self.H) / self.H


class GaussianProfile(VerticalProfile):
    """A layer of aerosol centred at ``center_km`` with a geometric width."""

    def __init__(self, center_km: float, sigma_km: float):
        self.c = float(center_km)
        self.s = float(sigma_km)

    def __call__(self, z_km):
        z = np.asarray(z_km, dtype=float)
        g = np.exp(-0.5 * ((z - self.c) / self.s) ** 2) / (self.s * np.sqrt(2 * np.pi))
        return np.where(z < 0, 0.0, g)


# ---------------------------------------------------------------------------
# Horizontal structure
# ---------------------------------------------------------------------------
class HorizontalField:
    """Radial modulation of the aerosol amount around the observer.

    ``weight(distance_km)`` multiplies the aerosol optical depth.  This is a
    deliberately simple way of representing "weather near me is different from
    the weather on the horizon", which is a first-order effect on the colour of
    the sky near the horizon.
    """

    def weight(self, distance_km):  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class UniformField(HorizontalField):
    def weight(self, distance_km):
        return np.ones_like(np.asarray(distance_km, dtype=float))

    def describe(self) -> str:
        return "uniform"


class RadialSplitField(HorizontalField):
    """Aerosol amount ``near`` within ``radius_km`` and ``far`` beyond it.

    The transition is a smooth (cosine) ramp of width ``transition_km``.
    """

    def __init__(self, radius_km: float, near: float = 1.0, far: float = 0.0,
                 transition_km: float = 20.0):
        self.r = float(radius_km)
        self.near = float(near)
        self.far = float(far)
        self.t = max(float(transition_km), 1e-6)

    def weight(self, distance_km):
        d = np.asarray(distance_km, dtype=float)
        x = np.clip((d - self.r) / self.t + 0.5, 0.0, 1.0)
        s = 0.5 - 0.5 * np.cos(np.pi * x)
        return self.near + (self.far - self.near) * s

    def describe(self) -> str:
        return f"radial split r={self.r:.0f}km near={self.near} far={self.far}"


# ---------------------------------------------------------------------------
# Aerosol layers
# ---------------------------------------------------------------------------
@dataclass
class AerosolLayer:
    """One aerosol population with a given optical depth and vertical shape."""

    aerosol: "object"  # skycolor.aerosol.Aerosol
    aod550: float = 0.05
    profile: VerticalProfile = field(default_factory=lambda: ExponentialProfile(1.2))
    horizontal: HorizontalField = field(default_factory=UniformField)

    def beta_550(self, z_km):
        """Extinction coefficient at 550 nm [m^-1]."""
        return self.aod550 * self.profile(z_km) / 1000.0


# ---------------------------------------------------------------------------
# The atmosphere
# ---------------------------------------------------------------------------
class Atmosphere:
    """A spherically stratified atmosphere with spectral optical properties.

    Parameters
    ----------
    wl_nm : array
        Spectral grid in nanometres.
    ozone_du : float
        Total ozone column in Dobson units.
    no2_column : float
        Total NO2 column in molecules m^-2 (0 disables it).
    aerosols : list of AerosolLayer
    ground_albedo : float or array
        Lambertian surface albedo (scalar or spectral).
    surface_pressure : float
        Sea-level pressure scaling; use :meth:`at_altitude` to place the
        observer on a mountain.
    z_bounds_km : array, optional
        Layer boundaries of the discretisation.
    """

    def __init__(
        self,
        wl_nm: Sequence[float],
        ozone_du: float = 300.0,
        no2_column: float = 0.0,
        aerosols: Sequence[AerosolLayer] | None = None,
        ground_albedo=0.1,
        surface_pressure: float = P0,
        z_bounds_km: np.ndarray | None = None,
        elevation_km: float = 0.0,
        refraction: bool = True,
    ):
        self.wl = np.asarray(wl_nm, dtype=float)
        self.ozone_du = float(ozone_du)
        self.no2_column = float(no2_column)
        self.aerosols = list(aerosols or [])
        self.surface_pressure = float(surface_pressure)
        self.elevation_km = float(elevation_km)
        #: Whether the direct-beam geometry includes atmospheric refraction
        #: (see :mod:`skycolor.refraction`).  **On by default**: the ray
        #: bending is real, the paths are integrated along the *bent* chord
        #: (the Bouguer invariant gives a closed form per shell), and the
        #: terminator is where it belongs.  Set ``False`` to reproduce the
        #: geometric-shadow numbers quoted in older reports.
        self.refraction = bool(refraction)
        self.z_bounds = np.array(z_bounds_km if z_bounds_km is not None else DEFAULT_Z_BOUNDS,
                                 dtype=float)
        self.z_mid = 0.5 * (self.z_bounds[1:] + self.z_bounds[:-1])
        self._prepare_ozone()
        self._prepare_no2()

        if np.ndim(ground_albedo) == 0:
            self.ground_albedo = np.full(self.wl.shape, float(ground_albedo))
        else:
            self.ground_albedo = np.broadcast_to(np.asarray(ground_albedo, float), self.wl.shape).copy()

        # cached spectral quantities
        self._sigma_ray = spectra.rayleigh_cross_section(self.wl)
        self._sigma_o3 = spectra.ozone_cross_section(self.wl)
        self._sigma_no2 = spectra.no2_cross_section(self.wl)
        self._ray_phase = spectra.rayleigh_phase_moments()

    # -- construction helpers ------------------------------------------------
    @classmethod
    def standard(cls, wl_nm, **kw) -> "Atmosphere":
        """A clean mid-latitude atmosphere: 300 DU ozone, 0.05 AOD boundary layer."""
        from .aerosol import HinterlandAerosol, shared

        kw.setdefault("aerosols", [AerosolLayer(shared(HinterlandAerosol), aod550=0.05,
                                                profile=ExponentialProfile(1.5))])
        return cls(wl_nm, ozone_du=300.0, **kw)

    def at_altitude(self, elevation_km: float) -> "Atmosphere":
        """Return a copy of this atmosphere seen from ``elevation_km``.

        The pressure profile is renormalised to the hydrostatic pressure at
        that altitude and the optical-depth normalisation of every aerosol
        layer is adjusted for the column that remains above the observer.
        """
        z0 = float(elevation_km)
        p0 = float(us76_pressure(np.array([z0]))[0])
        new = Atmosphere(
            self.wl, ozone_du=self.ozone_du, no2_column=self.no2_column,
            aerosols=[], ground_albedo=self.ground_albedo,
            surface_pressure=p0, z_bounds_km=self.z_bounds,
            elevation_km=z0, refraction=self.refraction,
        )
        # rescale aerosol layers to the column remaining above the observer
        for lay in self.aerosols:
            frac = lay.profile.column(z0, np.inf)
            new.aerosols.append(AerosolLayer(lay.aerosol, aod550=lay.aod550 * frac,
                                             profile=lay.profile, horizontal=lay.horizontal))
        # rescale ozone / no2 to the column above the observer
        new._o3_segment = None
        return new

    def _prepare_ozone(self) -> None:
        # column of the tabulated profile, used to normalise to `ozone_du`
        z = np.linspace(0, Z_TOA / 1e3, 20000)
        n = _log_interp(_O3_PROFILE, z) * 1e6  # m^-3
        col = np.trapezoid(n, z * 1e3)
        self._o3_scale = self.ozone_du * DU / col

    def _prepare_no2(self) -> None:
        z = np.linspace(0, Z_TOA / 1e3, 20000)
        n = _log_interp(_NO2_PROFILE, z) * 1e6
        col = np.trapezoid(n, z * 1e3)
        self._no2_scale = (self.no2_column / col) if col > 0 else 0.0

    # -- vertical profiles ---------------------------------------------------
    def pressure(self, z_km):
        z = np.asarray(z_km, dtype=float)
        return us76_pressure(z) * (self.surface_pressure / P0)

    def temperature(self, z_km):
        return us76_temperature(np.asarray(z_km, dtype=float))

    def n_air(self, z_km):
        """Dry-air number density [m^-3], with a thermospheric tail."""
        z = np.asarray(z_km, dtype=float)
        n = self.pressure(z) / (1.380649e-23 * self.temperature(z))
        n0 = self.pressure(np.array(THERMOSPHERE_START_KM)) / (
            1.380649e-23 * self.temperature(np.array(THERMOSPHERE_START_KM)))
        tail = n0 * np.exp(-(z - THERMOSPHERE_START_KM) / THERMOSPHERE_SCALE_HEIGHT_KM)
        return np.where(z > THERMOSPHERE_START_KM, tail, n)

    def n_o3(self, z_km):
        return _log_interp(_O3_PROFILE, z_km) * 1e6 * self._o3_scale

    def n_no2(self, z_km):
        return _log_interp(_NO2_PROFILE, z_km) * 1e6 * self._no2_scale

    # -- optical properties --------------------------------------------------
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
        raw = self.phase_moments(n_moments + 1)          # (nz, N+1, nwl)
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

    def _restrict(self, index):
        """A copy of this atmosphere restricted to the wavelengths ``index``.

        Used by the single-wavelength Monte Carlo loops.  Copying (rather than
        rebuilding) preserves subclass overrides such as a custom ozone or
        density profile, which a rebuild used to drop silently.
        """
        import copy

        new = copy.copy(self)
        idx = np.asarray(index, dtype=int)
        new.wl = self.wl[idx]
        new.ground_albedo = np.asarray(self.ground_albedo)[idx]
        new._sigma_ray = self._sigma_ray[idx]
        new._sigma_o3 = self._sigma_o3[idx]
        new._sigma_no2 = self._sigma_no2[idx]
        return new

    # -- column helpers ------------------------------------------------------
    @property
    def r_bounds(self) -> np.ndarray:
        return R_EARTH + self.z_bounds * 1e3

    def __repr__(self) -> str:
        return (f"Atmosphere(nwl={len(self.wl)}, ozone={self.ozone_du} DU, "
                f"aerosol_layers={len(self.aerosols)})")
