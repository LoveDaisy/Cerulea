"""Radiative transfer through a plane-parallel cloud deck.

The flux solver is a **delta-Eddington two-stream** model: the phase function
is delta-scaled (see :func:`skycolor.cloud.delta_m`) and the resulting
two-stream equations have a closed-form solution for a homogeneous layer.  The
same solution covers a beam entering from above (the sun above the horizon)
and a beam entering from below (the sun below the horizon but the deck still
sunlit because the Earth's surface curves away underneath it).

The angular radiance is then rebuilt from the fluxes:

* light transmitted from above is distributed with the **CIE overcast sky**
  law ``L(theta) = L_zenith (1 + 2 cos theta) / 3``; its horizon/zenith ratio
  is 1/3 by construction, it peaks at the zenith, and the scale is fixed by the
  transmitted diffuse flux.
* light scattered back out of an **underlit** base is taken to be isotropic.
  A grazing beam illuminating a slab from below is an intrinsically
  three-dimensional geometry (the beam enters through the side of the deck),
  so the two-stream has nothing to say about its angular distribution.  What
  it *does* say -- and what matters here -- is the total flux and its colour.

The total sky radiance seen from below the deck is

.. math::

    L(\\lambda) = L_{\\rm clear}(\\lambda) e^{-\\tau_c(\\lambda)/\\mu_v}
                + L_{\\rm cloud}(\\lambda, \\mu_v)

The first term is the clear sky seen through the layer (zero cloud optical
depth reduces the whole expression *exactly* to the clear sky); the second is
the cloud itself, illuminated from above or below.

Approximations, all of them deliberate
--------------------------------------
* Plane-parallel, homogeneous, infinite in the horizontal.  No cloud sides, no
  finite deck, no 3-D transport.
* The layer is treated as an overlay: the clear-sky field is computed without
  the cloud and then attenuated.  This is exact for the direct beam and
  approximate for the diffuse field, which in reality is partly created
  *below* the cloud (by the ground and the boundary layer).
* The two-stream closure is the Eddington one.  It conserves energy exactly
  and reproduces the hemispherical albedo of a homogeneous layer to a few per
  cent for ``tau >~ 5`` (checked against a Monte Carlo in ``tests/test_cloud.py``);
  at ``tau ~ 1`` the error can reach 10 %.
* The angular distribution is *assumed*, not solved.  A multi-stream solver
  would give the real radiance field, including the brightening towards the
  sun through a thin deck.
* Single scattering by the cloud is included only through its contribution to
  the two-stream source, i.e. the direct transmitted solar beam is not part of
  the sky radiance (it is a delta function in angle).
* No thermal emission (irrelevant in the visible), no polarisation, no
  refraction inside the droplet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import spectra
from ..cloud import CloudLayer, CloudOptics, delta_m
from ..constants import R_EARTH
from ..geometry import is_sunlit, path_lengths_sun
from ..optics import (
    shell_species_columns,
    shell_species_weights,
    species_extinction_weights,
    tau_from_columns,
)

__all__ = [
    "TwoStream",
    "CloudIllumination",
    "delta_eddington",
    "sun_transmittance_to_height",
    "illuminate_cloud",
    "cloud_radiance",
    "sky_radiance_with_cloud",
    "overcast_shape",
]

_TINY = 1e-12


# ---------------------------------------------------------------------------
# delta-Eddington two-stream
# ---------------------------------------------------------------------------
@dataclass
class TwoStream:
    """Fluxes of a homogeneous layer, per unit incident beam irradiance.

    All fluxes are irradiances on a horizontal surface, normalised by the
    incident beam irradiance *on that surface* (``mu0 * E``).  The beam is
    assumed to enter through the top; use :func:`delta_eddington` with a
    negative ``mu0`` for illumination from below.

    Attributes
    ----------
    albedo : ndarray
        Reflected flux / incident flux.
    transmittance : ndarray
        Diffuse flux emerging from the far side / incident flux.
    direct_transmittance : ndarray
        Transmittance of the *delta-scaled* beam, ``exp(-tau'/mu0)``.  Under
        delta-M the forward-scattered light (a fraction ``f`` of the
        scattering) is moved into this beam, so it is larger than the
        physically unscattered ``exp(-tau/mu0)``; it is what closes the energy
        budget ``R + T + direct = 1``.
    total_transmittance : ndarray
        ``transmittance + direct_transmittance``.
    """

    tau: np.ndarray
    omega: np.ndarray
    g: np.ndarray
    f: np.ndarray
    mu0: float
    tau_scaled: np.ndarray
    omega_scaled: np.ndarray
    g_scaled: np.ndarray
    flux_down_top: np.ndarray
    flux_up_top: np.ndarray
    flux_down_base: np.ndarray
    flux_up_base: np.ndarray
    albedo: np.ndarray
    transmittance: np.ndarray
    direct_transmittance: np.ndarray
    total_transmittance: np.ndarray
    illumination: str = "top"
    semi_infinite: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def absorption(self) -> np.ndarray:
        return 1.0 - self.albedo - self.total_transmittance

    def __repr__(self) -> str:
        return (f"TwoStream(illumination={self.illumination!r}, "
                f"albedo={np.mean(self.albedo):.3f}, "
                f"T={np.mean(self.transmittance):.3f})")


#: Above this value of ``lambda * tau`` the growing exponential has amplified
#: round-off past the point where the transmitted flux (of order
#: ``exp(-lambda tau)``) can be recovered, so the semi-infinite solution is
#: used instead.  The transmitted flux is then < 1.5e-8 and dropping it is
#: physically irrelevant.
_DEEP = 18.0


def delta_eddington(tau, omega, g, mu0, f=0.0, ground_albedo=0.0, incident=1.0):
    """Closed-form delta-Eddington two-stream solution for a homogeneous layer.

    Parameters
    ----------
    tau, omega, g : array_like
        Physical (unscaled) optical depth, single-scattering albedo and
        asymmetry parameter.
    mu0 : float
        Cosine of the solar zenith angle.  Positive: the beam enters through
        the top.  Negative: it enters through the base (the layer is lit from
        below).  Zero is treated as no direct beam.
    f : array_like
        Delta-M fraction, i.e. the Legendre moment at the truncation order of
        the solver.  For a two-stream, ``f = beta_2``; see
        :func:`skycolor.cloud.delta_m`.
    ground_albedo : array_like
        Lambertian albedo of the surface below the layer (0 for a black
        surface or for a layer over space).
    incident : array_like
        Beam irradiance normal to the beam at the illuminated surface
        [W m^-2 nm^-1].

    Returns
    -------
    :class:`TwoStream`

    Notes
    -----
    The equations solved are, with ``B = E exp(-tau/mu0)``,

    .. math::

        dF_\\downarrow/d\\tau &= -\\gamma_1 F_\\downarrow + \\gamma_2 F_\\uparrow + b_3 B \\\\
        dF_\\uparrow/d\\tau   &= -\\gamma_2 F_\\downarrow + \\gamma_1 F_\\uparrow - b_4 B

    with the Eddington coefficients

    .. math::

        \\gamma_1 = (7 - \\omega(4+3g))/4, \\quad
        \\gamma_2 = -(1 - \\omega(4-3g))/4, \\\\

        b_3 = \\frac{\\omega}{2}(1 + \\tfrac32 g\\mu_0), \\quad
        b_4 = \\frac{\\omega}{2}(1 - \\tfrac32 g\\mu_0).

    ``M = [[-gamma1, gamma2], [-gamma2, gamma1]]`` satisfies
    ``M^2 = lambda^2 I`` with ``lambda^2 = gamma1^2 - gamma2^2``, so the
    propagator is a cosh/sinh pair.  For conservative scattering
    (``omega = 1``) ``lambda = 0`` and the propagator degenerates to
    ``I + M tau``; the closed form below is written so that this limit is
    taken analytically rather than numerically (the naive eigenvector solution
    divides by ``lambda`` and is useless for clouds, which *are* conservative).
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    g = np.atleast_1d(np.asarray(g, dtype=float))
    f_arr = np.broadcast_to(np.asarray(f, dtype=float), tau.shape)
    ag = np.broadcast_to(np.asarray(ground_albedo, dtype=float), tau.shape)
    inc = np.broadcast_to(np.asarray(incident, dtype=float), tau.shape)
    mu0 = float(mu0)
    mirrored = mu0 < 0.0
    mu0 = abs(mu0)
    if mu0 == 0.0:
        # no direct beam: nothing to scatter
        z = np.zeros_like(tau)
        return TwoStream(tau=tau, omega=omega, g=g, f=f_arr, mu0=0.0,
                         tau_scaled=z, omega_scaled=omega, g_scaled=g,
                         flux_down_top=z, flux_up_top=z, flux_down_base=z,
                         flux_up_base=z, albedo=z, transmittance=z,
                         direct_transmittance=z, total_transmittance=z,
                         illumination="none", semi_infinite=np.zeros_like(tau, bool))

    tau_s, om, gs = delta_m(tau, omega, g, f_arr)
    gamma1 = (7.0 - om * (4.0 + 3.0 * gs)) / 4.0
    gamma2 = -(1.0 - om * (4.0 - 3.0 * gs)) / 4.0
    b3 = 0.5 * om * (1.0 + 1.5 * gs * mu0)
    b4 = 0.5 * om * (1.0 - 1.5 * gs * mu0)

    m11 = -gamma1
    m12 = gamma2
    m21 = -gamma2
    m22 = gamma1
    lam = np.sqrt(np.maximum(gamma1**2 - gamma2**2, 0.0))

    c1 = b3 * inc
    c2 = -b4 * inc
    a = 1.0 / mu0

    def propagator(t):
        """``exp(M t)`` as a 2x2 tuple of arrays."""
        ch = np.cosh(lam * t)
        big = lam > 1e-8
        ratio = np.where(big, np.sinh(lam * t) / np.where(big, lam, 1.0), t)
        return (m11 * ratio + ch, m12 * ratio,
                m21 * ratio, m22 * ratio + ch)

    def particular(t):
        """``P(t) = int_0^t exp(M (t-s)) c exp(-a s) ds``."""
        big = lam > 1e-8
        lam_safe = np.where(big, lam, 1.0)
        e1 = np.exp(lam * t)
        e2 = np.exp(-lam * t)
        ea = np.exp(-a * t)
        # the big branch is evaluated everywhere and masked; where lambda == 0
        # (or lambda == 1/mu0, a removable resonance) it produces inf/nan that
        # np.where discards.
        with np.errstate(divide="ignore", invalid="ignore"):
            gc_big = 0.5 * ((e1 - ea) / (lam_safe + a) + (e2 - ea) / (a - lam_safe))
            gs_big = 0.5 * ((e1 - ea) / (lam_safe + a) - (e2 - ea) / (a - lam_safe))
            # lam == a is a removable 0/0: the limit of (e2 - ea)/(a - lam) is
            # t e^{-lam t}.  Evaluate it explicitly; without this a layer whose
            # lambda happens to equal 1/mu0 returns nan.
            near = np.abs(a - lam_safe) < 1e-6
            residue = t * e2
            gc_near = 0.5 * ((e1 - ea) / (lam_safe + a) + residue)
            gs_near = 0.5 * ((e1 - ea) / (lam_safe + a) - residue)
        gc_big = np.where(near, gc_near, gc_big)
        gs_big = np.where(near, gs_near, gs_big)
        gc_small = (1.0 - ea) / a
        gs_small = t / a - (1.0 - ea) / a**2
        gc = np.where(big, gc_big, gc_small)
        gs = np.where(big, gs_big / lam_safe, gs_small)
        p1 = gc * c1 + gs * (m11 * c1 + m12 * c2)
        p2 = gc * c2 + gs * (m21 * c1 + m22 * c2)
        return p1, p2

    # --- deep layer: drop the growing mode --------------------------------
    deep = lam * tau_s > _DEEP

    # semi-infinite solution (used where deep)
    denom_p = (m11 + a) * (m22 + a) - m12 * m21
    denom_p = np.where(np.abs(denom_p) > _TINY, denom_p, 1.0)
    p1 = (-c1 * (m22 + a) + m12 * c2) / denom_p
    p2 = (m21 * c1 - (m11 + a) * c2) / denom_p
    safe_g2 = np.where(np.abs(gamma2) > 1e-10, gamma2, 1.0)
    alpha = np.where(np.abs(gamma2) > 1e-10, (gamma1 - lam) / safe_g2, 0.0)
    c_semi = -p1
    e_tau = np.exp(-a * tau_s)
    e_lam = np.exp(-lam * tau_s)
    semi_down_base = p1 * e_tau + c_semi * e_lam
    semi_up_base = p2 * e_tau + c_semi * alpha * e_lam
    semi_down_top = p1 + c_semi
    semi_up_top = p2 + c_semi * alpha

    # --- finite layer ------------------------------------------------------
    e11, e12, e21, e22 = propagator(tau_s)
    pt1, pt2 = particular(tau_s)
    denom = e22 - ag * e12
    denom = np.where(np.abs(denom) > _TINY, denom, 1.0)
    x0 = (ag * (pt1 + mu0 * inc * e_tau) - pt2) / denom
    fin_down_top = np.zeros_like(tau_s)
    fin_up_top = x0
    fin_down_base = e12 * x0 + pt1
    fin_up_base = e22 * x0 + pt2

    down_top_c = np.where(deep, semi_down_top, fin_down_top)
    up_top_c = np.where(deep, semi_up_top, fin_up_top)
    down_base_c = np.where(deep, semi_down_base, fin_down_base)
    up_base_c = np.where(deep, semi_up_base, fin_up_base)

    # Reflection and transmission are always measured relative to the side the
    # beam enters: `reflected` leaves through that side, `transmitted` through
    # the far side.  This is invariant under the mirroring below.
    incident_flux = mu0 * inc
    reflected = up_top_c
    transmitted = down_base_c
    albedo = reflected / np.maximum(incident_flux, _TINY)
    transmittance = transmitted / np.maximum(incident_flux, _TINY)
    direct = np.exp(-tau_s / mu0)

    if mirrored:
        # t -> tau* - t, down <-> up.  The entry side is the real base, so the
        # *downward* diffuse flux at the base (what an observer underneath
        # sees) is the light reflected back out of the entry side.
        down_top = up_base_c
        up_top = down_base_c
        down_base = up_top_c
        up_base = down_top_c
        illumination = "base"
    else:
        down_top, up_top = down_top_c, up_top_c
        down_base, up_base = down_base_c, up_base_c
        illumination = "top"

    return TwoStream(
        tau=tau, omega=omega, g=g, f=f_arr, mu0=mu0,
        tau_scaled=tau_s, omega_scaled=om, g_scaled=gs,
        flux_down_top=down_top, flux_up_top=up_top,
        flux_down_base=down_base, flux_up_base=up_base,
        albedo=albedo, transmittance=transmittance,
        direct_transmittance=direct,
        total_transmittance=transmittance + direct,
        illumination=illumination, semi_infinite=deep,
    )


# ---------------------------------------------------------------------------
# solar illumination of the deck
# ---------------------------------------------------------------------------
def sun_transmittance_to_height(atmo, height_km: float, sun_zenith_deg: float):
    """Direct solar transmittance from space to ``height_km``.

    The sun ray is traced through the spherical shells, so a point above the
    observer's horizon but below the observer's shadow boundary is handled
    correctly.  Returns ``(transmittance, sunlit)`` where ``sunlit`` is a bool
    telling whether the geometric ray misses the Earth.
    """
    r = R_EARTH + float(height_km) * 1e3
    sza = np.radians(float(sun_zenith_deg))
    c = np.cos(sza)
    p = r * np.sin(sza)
    lengths = path_lengths_sun(np.array([p]), np.array([c]), np.array([r]),
                               atmo.r_bounds)[:, 0]
    cols = shell_species_columns(lengths[:, None],
                                 shell_species_weights(atmo, atmo.r_bounds))
    tau = tau_from_columns(cols, species_extinction_weights(atmo))[0]
    lit = bool(is_sunlit(np.array([p]), np.array([c]))[0])
    return (np.exp(-tau), lit) if lit else (np.zeros_like(tau), False)


@dataclass
class CloudIllumination:
    """How a cloud deck is lit, and what the two-stream makes of it.

    Attributes
    ----------
    geometry : {"top", "base", "none"}
        ``top``: the sun is above the observer's horizon, the beam enters
        through the cloud top.
        ``base``: the sun is below the horizon but part of the deck is still
        sunlit, because the Earth curves away underneath it; the beam enters
        through the bottom of the *sunlit* part.  This is the red
        underlighting case.
        ``none``: the whole deck is inside the Earth's shadow.
    z_entry_km : float
        Altitude at which the beam enters the layer (the cloud top, or the
        shadow boundary for an underlit deck).
    tau_below : ndarray
        Optical depth between the cloud base and ``z_entry_km``.  For an
        underlit deck the light has to traverse this shadowed part of the deck
        before it reaches the observer, so it is attenuated by
        ``exp(-tau_below / mu_view)``.
    """

    geometry: str
    sun_zenith_deg: float
    mu0: float
    sunlit_top: bool
    sunlit_base: bool
    transmittance_top: np.ndarray
    transmittance_base: np.ndarray
    transmittance_entry: np.ndarray
    beam_flux: np.ndarray
    two_stream: TwoStream
    optics: CloudOptics
    z_entry_km: float = 0.0
    tau_below: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shadow_km: float = 0.0

    @property
    def is_lit(self) -> bool:
        return self.geometry != "none"


def shadow_height_km(sun_zenith_deg: float) -> float:
    """Altitude of the terminator of a spherical Earth.

    A point at height ``z`` is sunlit when its solar ray misses the Earth,
    i.e. when ``(R + z) sin(theta) > R``.  The terminator is therefore at
    ``z = R (1/sin(theta) - 1)`` for ``theta > 90``, and at the ground for a
    sun above the horizon.
    """
    theta = np.radians(float(sun_zenith_deg))
    if theta <= np.pi / 2.0:
        return 0.0
    s = np.sin(theta)
    return float(R_EARTH * (1.0 / max(s, 1e-9) - 1.0) / 1000.0)


def illuminate_cloud(atmo, cloud: CloudLayer, sun_zenith_deg: float,
                     ground_albedo=None) -> CloudIllumination:
    """Direct-beam illumination of ``cloud`` and the two-stream solution.

    The deck is treated as a single homogeneous slab.  With the sun above the
    horizon the beam enters through the top and the ground below is
    Lambertian.  With the sun below the horizon the deck is still partly
    sunlit: the terminator cuts it at :func:`shadow_height_km`, so the layer is
    split into a *shadowed* part below and a *lit* part above, and the beam
    enters the bottom of the lit part.  That is the geometry that produces red
    cloud bases after sunset -- the light is reddened by the long grazing path
    through the atmosphere, the deck is lit while the ground is already dark.

    If the terminator is above the cloud top the whole deck is in shadow and
    ``geometry`` is ``"none"``; something else (the twilight sky, horizontal
    transport from the still-lit part of the atmosphere) would be needed to
    light it, and a plane-parallel model has neither.
    """
    wl = np.asarray(atmo.wl, dtype=float)
    optics = cloud.optics(wl)
    ag = (atmo.ground_albedo if ground_albedo is None
          else np.broadcast_to(np.asarray(ground_albedo, dtype=float), wl.shape))
    t_top, lit_top = sun_transmittance_to_height(atmo, cloud.top_km, sun_zenith_deg)
    t_base, lit_base = sun_transmittance_to_height(atmo, cloud.base_km, sun_zenith_deg)
    mu = float(np.cos(np.radians(float(sun_zenith_deg))))
    f0 = spectra.solar_irradiance(wl)
    zeros = np.zeros_like(f0)

    z_shadow = shadow_height_km(sun_zenith_deg)

    if z_shadow >= cloud.top_km or abs(mu) < 1e-8:
        # the whole deck is inside the Earth's shadow (or the sun is exactly
        # on the horizon, where the beam crosses a horizontal slab over an
        # infinite path and deposits nothing)
        geometry = "none"
        beam = zeros
        t_entry = zeros
        z_entry = float(cloud.base_km)
        tau_below = optics.tau
        ts = delta_eddington(optics.tau, optics.omega, optics.g, 0.0,
                             f=optics.f, ground_albedo=0.0, incident=beam)
    elif z_shadow <= cloud.base_km:
        # the whole deck is sunlit: the beam comes from above if the sun is
        # above the horizon, otherwise from below
        z_entry = float(cloud.base_km if mu < 0.0 else cloud.top_km)
        t_entry, _ = sun_transmittance_to_height(atmo, z_entry, sun_zenith_deg)
        beam = f0 * t_entry
        tau_below = np.zeros_like(f0)
        if mu < 0.0:
            geometry = "base"
            ts = delta_eddington(optics.tau, optics.omega, optics.g, mu,
                                 f=optics.f, ground_albedo=0.0, incident=beam)
        else:
            geometry = "top"
            ts = delta_eddington(optics.tau, optics.omega, optics.g, mu,
                                 f=optics.f, ground_albedo=ag, incident=beam)
    else:
        # the terminator cuts the deck: the part above it is lit from below,
        # the part below it is in shadow.  The illumination is evaluated at
        # the middle of the sunlit part, which is a 1-D closure: in reality
        # the beam enters the lit part through its side, with a transmittance
        # that varies strongly with height.
        geometry = "base"
        z_entry = z_shadow
        z_lit_mid = 0.5 * (z_shadow + cloud.top_km)
        t_entry, _ = sun_transmittance_to_height(atmo, z_lit_mid, sun_zenith_deg)
        beam = f0 * t_entry
        frac = float(np.clip((cloud.top_km - z_shadow) / (cloud.top_km - cloud.base_km),
                             0.0, 1.0))
        tau_below = optics.tau * (1.0 - frac)
        ts = delta_eddington(optics.tau * frac, optics.omega, optics.g, mu,
                             f=optics.f, ground_albedo=0.0, incident=beam)

    return CloudIllumination(
        geometry=geometry, sun_zenith_deg=float(sun_zenith_deg), mu0=abs(mu),
        sunlit_top=lit_top, sunlit_base=lit_base,
        transmittance_top=t_top, transmittance_base=t_base,
        transmittance_entry=t_entry, beam_flux=beam, two_stream=ts,
        optics=optics, z_entry_km=z_entry, tau_below=tau_below,
        shadow_km=z_shadow,
    )


# ---------------------------------------------------------------------------
# angular radiance model
# ---------------------------------------------------------------------------
def overcast_shape(mu_view):
    """CIE overcast sky: ``L = L_zenith (1 + 2 mu)/3``, horizon/zenith = 1/3.

    Normalised so that ``int L mu dOmega = F_down``, i.e. returns
    ``L / F_down = 3 (1 + 2 mu) / (7 pi)``.
    """
    mu = np.clip(np.asarray(mu_view, dtype=float), 0.0, 1.0)
    return 3.0 * (1.0 + 2.0 * mu) / (7.0 * np.pi)


def cloud_radiance(illum: CloudIllumination, view_zenith_deg,
                   isotropic_underlit: bool = True):
    """Radiance of the cloud base seen from below, ``(n_view, n_wl)``.

    The observer is always underneath the deck, so the source is the
    *downward diffuse flux at the base* in both cases: the transmitted light
    for a deck lit from above, the light reflected back out of the base for an
    underlit deck.  For a deck lit from above that flux is spread with the
    overcast law; for an underlit deck it is spread isotropically (see the
    module docstring for why the two-stream cannot say more than that).
    """
    view = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
    mu = np.clip(np.cos(np.radians(view)), 1e-6, 1.0)
    ts = illum.two_stream
    flux = ts.flux_down_base[None, :]
    # light leaving an underlit deck still has to cross the shadowed part of
    # the deck below the shadow boundary
    below = np.exp(-np.asarray(illum.tau_below)[None, :] / mu[:, None])
    if illum.geometry == "top":
        return flux * overcast_shape(mu)[:, None]
    if illum.geometry == "base":
        if isotropic_underlit:
            return np.broadcast_to(flux / np.pi, (view.size, flux.shape[1])) * below
        return flux * overcast_shape(mu)[:, None] * below
    return np.zeros((view.size, illum.optics.wl.size))



# ---------------------------------------------------------------------------
# the drop-in replacement for rt.solver.sky_radiance_total
# ---------------------------------------------------------------------------
def sky_radiance_with_cloud(
    atmo,
    cloud_layer: CloudLayer,
    sun_zenith_deg: float,
    view_zenith_deg,
    view_azimuth_deg,
    sun_azimuth_deg: float = 0.0,
    multiple: bool = True,
    n_gauss: int = 3,
    chunk: int = 512,
    clear_radiance=None,
    ground_albedo=None,
    include_clear: bool = True,
    include_cloud: bool = True,
    return_components: bool = False,
):
    """Total sky radiance with a cloud deck, ``(n_view, n_wl)``.

    Drop-in replacement for :func:`skycolor.rt.solver.sky_radiance_total`::

        L = L_clear * exp(-tau_cloud / mu_view) + L_cloud

    Parameters
    ----------
    atmo : Atmosphere
        Clear-sky atmosphere (the cloud is not part of it).
    cloud_layer : CloudLayer
    sun_zenith_deg : float
    view_zenith_deg, view_azimuth_deg : array_like
    sun_azimuth_deg : float
    multiple : bool
        Passed to the clear-sky solver.
    clear_radiance : ndarray, optional
        Pre-computed ``(n_view, n_wl)`` clear-sky field, to avoid re-running
        the multiple-scattering solver for every call.
    ground_albedo : float or ndarray, optional
        Overrides ``atmo.ground_albedo`` for the cloud-ground coupling.
    return_components : bool
        Also return a dict with the clear and cloud terms and the diagnostics.

    Returns
    -------
    ndarray ``(n_view, n_wl)``, or ``(radiance, components)``.
    """
    from .solver import sky_radiance_total

    view = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
    n_view = view.size
    wl = np.asarray(atmo.wl, dtype=float)

    if clear_radiance is None and include_clear:
        clear_radiance = sky_radiance_total(
            atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=sun_azimuth_deg, multiple=multiple,
            n_gauss=n_gauss, chunk=chunk)
    if clear_radiance is None:
        clear_radiance = np.zeros((n_view, len(wl)))
    clear_radiance = np.asarray(clear_radiance, dtype=float)

    optics = cloud_layer.optics(wl)
    mu_view = np.clip(np.cos(np.radians(view)), 1e-6, 1.0)
    tau_view = optics.tau[None, :] / mu_view[:, None]
    trans = np.exp(-tau_view)

    illum = illuminate_cloud(atmo, cloud_layer, sun_zenith_deg,
                             ground_albedo=ground_albedo)
    cloud_term = cloud_radiance(illum, view) if include_cloud else np.zeros_like(clear_radiance)

    total = clear_radiance * trans + cloud_term
    if not include_clear:
        total = cloud_term

    if return_components:
        return total, dict(clear=clear_radiance * trans, cloud=cloud_term,
                           illumination=illum, optics=optics,
                           transmittance=trans)
    return total
