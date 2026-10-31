"""Facade over the physical medium with the observing scenario attached.

The physical medium -- composition, profiles, discretisation and the
layer-averaged optical processing -- lives in :class:`skycolor.medium.Medium`,
a frozen value object, and the observing scenario in
:class:`skycolor.scene.Scene`.  :class:`Atmosphere` is the long-standing
public entry point: it pairs the two frozen parts and forwards every query,
so the rest of the library keeps its ``atmo`` interface unchanged.

The model atmosphere is a set of concentric spherical shells.  Within each
shell the composition is described by continuous profiles so that the
radiative-transfer integrators can evaluate the local optical properties at
an arbitrary height:

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

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Sequence

import numpy as np

from .constants import P0, R_EARTH, Z_TOA
from .medium import (  # noqa: F401  (re-exported for backward compatibility)
    DEFAULT_Z_BOUNDS,
    Medium,
    THERMOSPHERE_SCALE_HEIGHT_KM,
    THERMOSPHERE_START_KM,
    us76_number_density,
    us76_pressure,
    us76_temperature,
)
from .scene import Scene

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
# The facade
# ---------------------------------------------------------------------------
class Atmosphere:
    """A spherically stratified atmosphere with spectral optical properties.

    The physical medium is a frozen :class:`skycolor.medium.Medium` and the
    observing scenario a frozen :class:`skycolor.scene.Scene`; this object
    attaches the two and forwards every query, so the rest of the library
    keeps its ``atmo`` interface unchanged.  Both parts are replaced
    wholesale on redefinition (through the property setters), never patched.

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
    ozone_profile, no2_profile : callable, optional
        Vertical *shape* of the species (arbitrary units), column-normalised
        to ``ozone_du`` / ``no2_column`` exactly like the built-in tables --
        the sanctioned extension point for custom gas profiles.
    air_profile : callable, optional
        Complete dry-air number density [m^-3] vs height [km]; replaces the
        US76 pressure/temperature construction.  With it in place,
        ``pressure()``/``temperature()`` keep reporting the *default* US76
        profiles and no longer describe this atmosphere.
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
        ozone_profile=None,
        no2_profile=None,
        air_profile=None,
    ):
        self._medium = Medium(
            wl=np.asarray(wl_nm, dtype=float),
            z_bounds=np.array(z_bounds_km if z_bounds_km is not None else DEFAULT_Z_BOUNDS,
                              dtype=float),
            ozone_du=float(ozone_du),
            no2_column=float(no2_column),
            aerosols=tuple(aerosols or ()),
            surface_pressure=float(surface_pressure),
            ozone_profile=ozone_profile,
            no2_profile=no2_profile,
            air_profile=air_profile,
        )
        #: Whether the direct-beam geometry includes atmospheric refraction
        #: (see :mod:`skycolor.refraction`).  **On by default**: the ray
        #: bending is real, the paths are integrated along the *bent* chord
        #: (the Bouguer invariant gives a closed form per shell), and the
        #: terminator is where it belongs.  Set ``False`` to reproduce the
        #: geometric-shadow numbers quoted in older reports.
        if np.ndim(ground_albedo) == 0:
            albedo = np.full(self.wl.shape, float(ground_albedo))
        else:
            albedo = np.broadcast_to(np.asarray(ground_albedo, float),
                                     self.wl.shape).copy()
        self._scene = Scene(observer_elevation_km=float(elevation_km),
                            ground_albedo=albedo, refraction=bool(refraction))

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

        The model is **truncated at the site**: the layer grid starts at
        ``z0``, the ground (and with it every solver's bottom boundary)
        sits at ``z0``, and the observer radius coincides with the first
        shell boundary.  The single-scattering solver integrates from the
        observer, the SOS solver and the Monte Carlo fix their ground at
        ``z_bounds[0]`` -- truncation is what makes those three agree for
        the first time.  Nothing is renormalised: the column above the site
        is already the integral of the untouched profiles over the
        truncated domain, so rescaling ozone/NO2 or aerosol columns (the
        old behaviour) would shrink it a second time, and
        ``surface_pressure`` keeps its meaning as the scaling of the whole
        US76 profile rather than "the pressure at the ground".

        This replaces the old double pressure scaling and inconsistent
        bottom boundary.  Solar occultation also uses this same opaque
        bottom radius, including the refracted grazing-ray thresholds.
        Truncation raises a globally spherical surface; it does not model
        a local mountain above surrounding lowland.  Cross-site brightness
        ratios are not controlled predictions of this operation (see
        ``docs/TWILIGHT_ALTITUDE_AUDIT.md``).

        ``elevation_km=0`` returns an equivalent atmosphere (a fresh copy,
        as the method always has).  The derived atmosphere owns fresh
        ``AerosolLayer`` copies, so mutating one scene's layer never leaks
        into the source scene.
        """
        z0 = float(elevation_km)
        if not 0.0 <= z0 < float(self.z_bounds[-1]):
            raise ValueError(
                f"elevation_km={z0} outside the model domain "
                f"[0, {float(self.z_bounds[-1])} km)")
        zb = (np.concatenate(([z0], self.z_bounds[self.z_bounds > z0]))
              if z0 > 0 else self.z_bounds)
        medium = self._medium
        layers = [AerosolLayer(lay.aerosol, aod550=lay.aod550,
                               profile=lay.profile, horizontal=lay.horizontal)
                  for lay in self.aerosols]
        return Atmosphere(
            self.wl, ozone_du=self.ozone_du, no2_column=self.no2_column,
            aerosols=layers, ground_albedo=self.ground_albedo,
            surface_pressure=self.surface_pressure, z_bounds_km=zb,
            elevation_km=z0, refraction=self.refraction,
            ozone_profile=medium.ozone_profile,
            no2_profile=medium.no2_profile,
            air_profile=medium.air_profile,
        )

    # -- medium delegation -----------------------------------------------------
    #   scalar/array attributes of the medium are exposed as read-only
    #   properties: the medium is replaced wholesale on redefinition, never
    #   patched in place.
    @property
    def wl(self) -> np.ndarray:
        return self._medium.wl

    @property
    def z_bounds(self) -> np.ndarray:
        return self._medium.z_bounds

    @property
    def z_mid(self) -> np.ndarray:
        return self._medium.z_mid

    @property
    def ozone_du(self) -> float:
        return self._medium.ozone_du

    @property
    def no2_column(self) -> float:
        return self._medium.no2_column

    @property
    def surface_pressure(self) -> float:
        return self._medium.surface_pressure

    @property
    def aerosols(self) -> tuple:
        """The aerosol layers of the medium, as a read-only tuple.

        Element mutation would pierce into the frozen medium
        (``atmo.aerosols[0].aod550 = ...``), so the collection itself is
        returned immutable: appending raises ``AttributeError`` loudly
        instead of silently writing into a discarded copy.  Redefine the
        layer list through the setter (wholesale replacement).
        """
        return self._medium.aerosols

    @aerosols.setter
    def aerosols(self, layers) -> None:
        self._medium = _dc_replace(self._medium, aerosols=tuple(layers))

    @property
    def _sigma_ray(self) -> np.ndarray:
        return self._medium._sigma_ray

    @property
    def _sigma_o3(self) -> np.ndarray:
        return self._medium._sigma_o3

    @property
    def _sigma_no2(self) -> np.ndarray:
        return self._medium._sigma_no2

    @property
    def _ray_phase(self) -> np.ndarray:
        return self._medium._ray_phase

    # -- scenario delegation ----------------------------------------------------
    @property
    def elevation_km(self) -> float:
        return self._scene.observer_elevation_km

    @elevation_km.setter
    def elevation_km(self, value) -> None:
        self._scene = _dc_replace(self._scene,
                                  observer_elevation_km=float(value))

    @property
    def refraction(self) -> bool:
        return self._scene.refraction

    @refraction.setter
    def refraction(self, value) -> None:
        self._scene = _dc_replace(self._scene, refraction=bool(value))

    @property
    def ground_albedo(self) -> np.ndarray:
        return self._scene.ground_albedo

    @ground_albedo.setter
    def ground_albedo(self, value) -> None:
        if np.ndim(value) == 0:
            albedo = np.full(self.wl.shape, float(value))
        else:
            albedo = np.broadcast_to(np.asarray(value, float), self.wl.shape).copy()
        self._scene = _dc_replace(self._scene, ground_albedo=albedo)

    # -- vertical profiles ---------------------------------------------------
    def pressure(self, z_km):
        return self._medium.pressure(z_km)

    def temperature(self, z_km):
        return self._medium.temperature(z_km)

    def n_air(self, z_km):
        """Dry-air number density [m^-3], with a thermospheric tail."""
        return self._medium.n_air(z_km)

    def n_o3(self, z_km):
        return self._medium.n_o3(z_km)

    def n_no2(self, z_km):
        return self._medium.n_no2(z_km)

    # -- optical properties --------------------------------------------------
    def rayleigh_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        """Rayleigh scattering coefficient [m^-1]."""
        return self._medium.rayleigh_coefficient(z_km, wl_nm)

    def ozone_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        return self._medium.ozone_coefficient(z_km, wl_nm)

    def no2_coefficient(self, z_km, wl_nm=None) -> np.ndarray:
        return self._medium.no2_coefficient(z_km, wl_nm)

    def aerosol_coefficients(self, z_km) -> tuple[np.ndarray, np.ndarray]:
        """Aerosol extinction and scattering coefficients [m^-1].

        Returns ``(beta_ext, beta_sca)`` with shape ``(n_z, n_wl)``.
        """
        return self._medium.aerosol_coefficients(z_km)

    def layer_properties(self, z_low=None, z_high=None) -> dict:
        """Optical properties evaluated at the layer mid-points.

        Returns a dict with ``beta_ray``, ``beta_ae`` (aerosol extinction),
        ``beta_as`` (aerosol scattering), ``beta_o3``, ``beta_no2``,
        ``beta_ext`` (total), ``beta_sca`` (total), all shape ``(n_layer, n_wl)``.
        """
        return self._medium.layer_properties(z_low, z_high)

    def phase_moments(self, n_moments: int = 32) -> np.ndarray:
        """Legendre moments of the layer-averaged phase function.

        Shape ``(n_layer, n_moments, n_wl)``; ``p(cos t) = sum_l (2l+1) b_l P_l``.

        These are the *raw* moments.  A discrete-ordinates code that truncates
        at ``n_moments`` must use :meth:`delta_m_moments` instead, otherwise the
        forward peak that lives beyond the truncation is silently discarded --
        which for dust and cloud droplets is 37 % of the scattered energy.
        """
        return self._medium.phase_moments(n_moments)

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
        return self._medium.delta_m_moments(n_moments)

    def delta_m_properties(self, n_moments: int = 32) -> dict:
        """Layer optical properties rescaled by delta-M, plus the moments."""
        return self._medium.delta_m_properties(n_moments)

    def _restrict(self, index):
        """A copy of this atmosphere restricted to the wavelengths ``index``.

        Used by the single-wavelength Monte Carlo loops.  Copying (rather than
        rebuilding) preserves subclass overrides such as a custom ozone or
        density profile, which a rebuild used to drop silently.
        """
        import copy

        new = copy.copy(self)
        new._medium = self._medium.restrict(index)
        idx = np.asarray(index, dtype=int)
        new._scene = _dc_replace(
            self._scene, ground_albedo=np.asarray(self._scene.ground_albedo)[idx])
        return new

    # -- column helpers ------------------------------------------------------
    @property
    def r_bounds(self) -> np.ndarray:
        return R_EARTH + self.z_bounds * 1e3

    def __repr__(self) -> str:
        return (f"Atmosphere(nwl={len(self.wl)}, ozone={self.ozone_du} DU, "
                f"aerosol_layers={len(self.aerosols)})")
