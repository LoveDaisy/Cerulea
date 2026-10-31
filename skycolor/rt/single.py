"""Single scattering in a spherically stratified atmosphere.

This is the backbone of the model.  It is *exact* (up to quadrature) for
first-order scattering, which is the process that carries almost all of the
colour information in a clear sky: the solar spectrum is attenuated along the
sun ray, the scattered light is attenuated along the line of sight, and the
angular dependence comes from the Rayleigh and Mie phase functions.

Two things make this more than a textbook formula:

* the atmosphere is **spherical** -- the solar zenith angle, the air density
  and the scattering angle are all evaluated at the true position of each
  scattering event, so the terminator, the twilight arch and the slanted
  horizon glow appear without any extra machinery;
* the *direct* solar beam is attenuated along its own curved path, which means
  a point at 40 km altitude is still sunlit when the observer's ground is
  already 8 degrees past sunset.  That is exactly what produces civil twilight.

Multiple scattering is added separately (see :mod:`skycolor.rt.multiple`).
"""
from __future__ import annotations

import numpy as np

from .. import spectra
from ..constants import R_EARTH
from ..geometry import is_sunlit, path_lengths_sun, path_lengths_sun_array
from ..optics import (
    shell_species_columns,
    shell_species_weights,
    species_extinction_weights,
    tau_from_columns,
)


def _quadrature_points(ub: np.ndarray, n_gauss: int):
    """Gauss points inside every shell segment, ordered along the ray.

    ``ub`` has shape ``(n_shell + 1, n_ray)``.  Returns ``u`` and ``du`` of
    shape ``(n_shell * n_gauss, n_ray)``, ordered from the observer outwards.
    """
    gx, gw = np.polynomial.legendre.leggauss(n_gauss)
    lo, hi = ub[:-1], ub[1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    u = mid[:, None, :] + gx[None, :, None] * half[:, None, :]  # (ns, ng, nr)
    du = gw[None, :, None] * half[:, None, :] * np.ones_like(u)
    ns, ng, nr = u.shape
    return u.reshape(ns * ng, nr), du.reshape(ns * ng, nr)


def _local_optics_h(atmo, z_km, h_km, wl_mask=None):
    """Local optical properties at height ``z`` including the horizontal field."""
    z = np.atleast_1d(np.asarray(z_km, dtype=float))
    h = np.atleast_1d(np.asarray(h_km, dtype=float))
    n_air = atmo.n_air(z)
    beta_ray = n_air[:, None] * atmo._sigma_ray[None, :]
    beta_o3 = atmo.n_o3(z)[:, None] * atmo._sigma_o3[None, :]
    beta_no2 = atmo.n_no2(z)[:, None] * atmo._sigma_no2[None, :]
    beta_as = np.zeros_like(beta_ray)
    beta_ae = np.zeros_like(beta_ray)
    for lay in atmo.aerosols:
        rel_ext, rel_sca = lay.aerosol.relative_coefficients(atmo.wl)
        w = lay.aod550 * lay.profile(z) / 1000.0 * lay.horizontal.weight(h)
        beta_ae += w[:, None] * rel_ext[None, :]
        beta_as += w[:, None] * rel_sca[None, :]
    return beta_ray, beta_as, beta_ae, beta_o3, beta_no2


def _tau_from_columns(atmo, cols: np.ndarray) -> np.ndarray:
    """Optical depth of species column amounts; thin wrapper over the shared
    column algebra in :mod:`skycolor.optics` (the single source of the
    species listing and its summation order)."""
    return tau_from_columns(cols, species_extinction_weights(atmo))


def species_columns(atmo, lengths: np.ndarray, r_bounds: np.ndarray) -> np.ndarray:
    """Per-species column amounts along a set of shell path lengths.

    ``lengths`` has shape ``(n_shell, n_points)``.  Returns ``(n_points,
    3 + n_aerosol_layers)``: air, ozone, NO2 (molecules m^-2) followed by one
    column per aerosol layer (550 nm extinction optical depth).  Thin
    wrapper over the shared algebra in :mod:`skycolor.optics`.
    """
    return shell_species_columns(lengths, shell_species_weights(atmo, r_bounds))


def direct_sun_transmittance(atmo, sun_zenith_deg: float) -> np.ndarray:
    """Direct solar transmittance at the observer, shape ``(n_wl,)``."""
    r0 = R_EARTH + atmo.elevation_km * 1e3
    c = np.cos(np.radians(sun_zenith_deg))
    p_sun = r0 * np.sin(np.radians(sun_zenith_deg))
    lengths = path_lengths_sun(np.array([p_sun]), np.array([c]), np.array([r0]),
                               atmo.r_bounds)[:, 0]
    cols = species_columns(atmo, lengths[:, None], atmo.r_bounds)
    tau = _tau_from_columns(atmo, cols)[0]
    lit = is_sunlit(p_sun, c)
    return np.exp(-tau) if lit else np.zeros_like(tau)


def sky_radiance(
    atmo,
    sun_zenith_deg: float,
    view_zenith_deg,
    view_azimuth_deg,
    sun_azimuth_deg: float = 0.0,
    n_gauss: int = 3,
    chunk: int = 128,
) -> np.ndarray:
    """Single-scattered sky radiance for a set of viewing directions.

    The integrand is evaluated on a Gauss quadrature inside every atmospheric
    shell, and the whole (sample, ray, wavelength) block is evaluated with
    vectorised numpy.

    Parameters
    ----------
    atmo : Atmosphere
    sun_zenith_deg : float
        Solar zenith angle (0 = overhead, 90 = horizon, >90 = twilight).
    view_zenith_deg, view_azimuth_deg : array_like
        Viewing direction (zenith angle from the local vertical, azimuth in
        degrees clockwise from north).
    sun_azimuth_deg : float
    n_gauss : int
        Quadrature points per atmospheric shell (3 is plenty).
    chunk : int
        Rays per block; trades memory for speed.

    Returns
    -------
    ndarray ``(n_view, n_wl)``, spectral radiance [W m^-2 sr^-1 nm^-1].
    """
    view_zen = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
    view_az = np.broadcast_to(np.asarray(view_azimuth_deg, dtype=float),
                              view_zen.shape).ravel()
    n_view = view_zen.size
    n_wl = len(atmo.wl)
    out = np.zeros((n_view, n_wl))

    r0 = R_EARTH + atmo.elevation_km * 1e3
    r_bounds = atmo.r_bounds
    r_toa = r_bounds[-1]
    r_mid = np.sqrt(r_bounds[:-1] * r_bounds[1:])
    z_mid = (r_mid - R_EARTH) / 1e3
    n_air_mid = atmo.n_air(z_mid)
    n_o3_mid = atmo.n_o3(z_mid)
    n_no2_mid = atmo.n_no2(z_mid)
    aer_mid = [lay.aod550 * lay.profile(z_mid) / 1000.0 for lay in atmo.aerosols]
    aer_optics = [lay.aerosol.relative_coefficients(atmo.wl) for lay in atmo.aerosols]

    s_ray, s_o3, s_no2 = atmo._sigma_ray, atmo._sigma_o3, atmo._sigma_no2

    use_refraction = bool(getattr(atmo, "refraction", False))
    if use_refraction:
        from ..refraction import refractivity_of

        n_layer = 1.0 + refractivity_of(atmo, z_mid)
        n_surface = 1.0 + refractivity_of(atmo, atmo.elevation_km)
    else:
        n_layer = None
        n_surface = 1.0
    zs = np.radians(sun_zenith_deg)
    azs = np.radians(sun_azimuth_deg)
    shat = np.array([np.sin(zs) * np.sin(azs), np.sin(zs) * np.cos(azs), np.cos(zs)])
    F0 = spectra.solar_irradiance(atmo.wl) / (4.0 * np.pi)

    for start in range(0, n_view, chunk):
        sl = slice(start, min(start + chunk, n_view))
        zv = np.radians(view_zen[sl])
        azv = np.radians(view_az[sl])
        d = np.stack([np.sin(zv) * np.sin(azv), np.sin(zv) * np.cos(azv), np.cos(zv)], axis=1)
        p = np.maximum(r0 * np.sin(zv), 1e-9)
        d0 = r0 * np.cos(zv)
        u_top = np.sqrt(np.maximum(r_toa**2 - p**2, 0.0))
        s_b = np.sqrt(np.maximum(r_bounds[:, None] ** 2 - p[None, :] ** 2, 0.0))
        ub = np.clip(s_b, d0[None, :], u_top[None, :])
        u, du = _quadrature_points(ub, n_gauss)  # (n_pts, n_ray)

        r_pts = np.sqrt(p[None, :] ** 2 + u**2)
        z_pts = (r_pts - R_EARTH) / 1e3
        x_vec = np.empty(u.shape + (3,))
        x_vec[..., :] = d[None, :, :]
        x_vec *= (u - d0[None, :])[..., None]
        x_vec[..., 2] += r0
        h_pts = np.hypot(x_vec[..., 0], x_vec[..., 1]) / 1e3

        mu_sun = np.clip(x_vec @ shat / np.maximum(r_pts, 1e-9), -1.0, 1.0)
        p_sun = r_pts * np.sqrt(np.maximum(1.0 - mu_sun**2, 0.0))
        if use_refraction:
            from ..refraction import geometry as _refr_geometry, refractivity_of

            # the light bends on its way in, so the direction at the point is
            # the *apparent* one and the Bouguer invariant uses it
            e_true = np.degrees(np.arcsin(mu_sun))
            e_app = _refr_geometry().apparent_elevation(e_true, z_pts)
            mu_app = np.sin(np.radians(e_app))
            p_sun = (1.0 + refractivity_of(atmo, z_pts)) * r_pts * np.cos(np.radians(e_app))
            lit = (mu_app >= 0.0) | (p_sun > n_surface * R_EARTH)
            c_sun = mu_app
        else:
            lit = is_sunlit(p_sun, mu_sun)
            c_sun = mu_sun

        # ---- local optical properties, shape (n_pts, n_ray, n_wl) ----------
        beta_ray = atmo.n_air(z_pts)[..., None] * s_ray
        beta_o3 = atmo.n_o3(z_pts)[..., None] * s_o3
        beta_no2 = atmo.n_no2(z_pts)[..., None] * s_no2
        beta_ae = np.zeros_like(beta_ray)
        beta_as = np.zeros_like(beta_ray)
        aer_w = []
        for lay, (rel_ext, rel_sca) in zip(atmo.aerosols, aer_optics):
            w = (lay.aod550 * lay.profile(z_pts) / 1000.0
                 * lay.horizontal.weight(h_pts))  # (n_pts, n_ray)
            aer_w.append(w)
            beta_ae += w[..., None] * rel_ext
            beta_as += w[..., None] * rel_sca
        beta_sca = beta_ray + beta_as
        b_ext = beta_ray + beta_ae + beta_o3 + beta_no2

        # ---- optical depth from the observer (trapezoid along the ray) -----
        first = 0.5 * b_ext[0] * du[0][:, None]
        if u.shape[0] > 1:
            dt = 0.5 * (b_ext[1:] + b_ext[:-1]) * du[1:, :, None]
            tau = np.concatenate([first[None], first[None] + np.cumsum(dt, axis=0)], axis=0)
        else:
            tau = first[None]

        # ---- solar optical depth ------------------------------------------
        if use_refraction:
            from ..refraction import sun_path_lengths

            lengths = sun_path_lengths(p_sun, c_sun, r_pts, r_bounds, n_layer)
        else:
            lengths = path_lengths_sun_array(p_sun, mu_sun, r_pts, r_bounds)  # (ns, np, nr)
        tau_sun = (np.einsum("kpr,k->pr", lengths, n_air_mid)[..., None] * s_ray
                   + np.einsum("kpr,k->pr", lengths, n_o3_mid)[..., None] * s_o3
                   + np.einsum("kpr,k->pr", lengths, n_no2_mid)[..., None] * s_no2)
        for j, (rel_ext, rel_sca) in enumerate(aer_optics):
            col = np.einsum("kpr,k->pr", lengths, aer_mid[j])
            tau_sun = tau_sun + col[..., None] * rel_ext
        tau_sun = np.where(lit[..., None], tau_sun, 1e3)

        # ---- phase function ------------------------------------------------
        cos_theta = np.clip(d @ shat, -1.0, 1.0)
        p_ray = spectra.rayleigh_phase(cos_theta)[None, :, None]
        ph = p_ray * (beta_ray / np.maximum(beta_sca, 1e-300))
        for j, lay in enumerate(atmo.aerosols):
            pj = lay.aerosol.phase(cos_theta, atmo.wl)
            if pj.ndim == 1:
                pj = pj[:, None]
            beta_as_j = aer_w[j][..., None] * aer_optics[j][1]
            ph = ph + pj[None, :, :] * (beta_as_j / np.maximum(beta_sca, 1e-300))

        contrib = beta_sca * ph * np.exp(-tau_sun - tau) * du[..., None]
        out[sl] += contrib.sum(axis=0) * F0
    return out
