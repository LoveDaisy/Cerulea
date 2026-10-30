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

from dataclasses import dataclass

import numpy as np

from .. import spectra
from ..constants import R_EARTH
from ..geometry import is_sunlit, path_lengths_sun


@dataclass
class ScatteringResult:
    radiance: np.ndarray
    tau_sun_observer: np.ndarray | None = None


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


def _local_optics(atmo, z_km, wl_mask=None):
    """Local scattering/extinction coefficients at height ``z`` [m^-1].

    Returns ``(beta_ray, beta_as, beta_ae, beta_o3, beta_no2)``, each of shape
    ``(n_points, n_wl)``.
    """
    z = np.atleast_1d(np.asarray(z_km, dtype=float))
    n_air = atmo.n_air(z)
    beta_ray = n_air[:, None] * atmo._sigma_ray[None, :]
    beta_o3 = atmo.n_o3(z)[:, None] * atmo._sigma_o3[None, :]
    beta_no2 = atmo.n_no2(z)[:, None] * atmo._sigma_no2[None, :]
    beta_as = np.zeros_like(beta_ray)
    beta_ae = np.zeros_like(beta_ray)
    for lay in atmo.aerosols:
        rel_ext, rel_sca = lay.aerosol.relative_coefficients(atmo.wl)
        w = lay.aod550 * lay.profile(z) / 1000.0
        beta_ae += w[:, None] * rel_ext[None, :]
        beta_as += w[:, None] * rel_sca[None, :]
    return beta_ray, beta_as, beta_ae, beta_o3, beta_no2


def _local_optics_h(atmo, z_km, h_km, wl_mask=None):
    """Same as :func:`_local_optics` but with the horizontal aerosol field."""
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
    tau = (cols[:, 0:1] * atmo._sigma_ray[None, :]
           + cols[:, 1:2] * atmo._sigma_o3[None, :]
           + cols[:, 2:3] * atmo._sigma_no2[None, :])
    for j, lay in enumerate(atmo.aerosols):
        rel_ext, _ = lay.aerosol.relative_coefficients(atmo.wl)
        tau = tau + cols[:, 3 + j:4 + j] * rel_ext[None, :]
    return tau


def species_columns(atmo, lengths: np.ndarray, r_bounds: np.ndarray) -> np.ndarray:
    """Per-species column amounts along a set of shell path lengths.

    ``lengths`` has shape ``(n_shell, n_points)``.  Returns ``(n_points,
    3 + n_aerosol_layers)``: air, ozone, NO2 (molecules m^-2) followed by one
    column per aerosol layer (550 nm extinction optical depth).
    """
    r_mid = np.sqrt(r_bounds[:-1] * r_bounds[1:])
    z_mid = (r_mid - R_EARTH) / 1e3
    cols = [lengths.T @ atmo.n_air(z_mid),
            lengths.T @ atmo.n_o3(z_mid),
            lengths.T @ atmo.n_no2(z_mid)]
    for lay in atmo.aerosols:
        cols.append(lengths.T @ (lay.aod550 * lay.profile(z_mid) / 1000.0))
    return np.stack(cols, axis=1)


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
    chunk: int = 512,
) -> np.ndarray:
    """Single-scattered sky radiance for a set of viewing directions.

    Parameters
    ----------
    atmo : Atmosphere
    sun_zenith_deg : float
        Solar zenith angle (0 = overhead, 90 = horizon, >90 = twilight).
    view_zenith_deg, view_azimuth_deg : array_like
        Viewing direction (zenith angle from local vertical, azimuth in degrees
        clockwise from north).
    sun_azimuth_deg : float

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

    zs = np.radians(sun_zenith_deg)
    azs = np.radians(sun_azimuth_deg)
    shat = np.array([np.sin(zs) * np.sin(azs), np.sin(zs) * np.cos(azs), np.cos(zs)])
    F0 = spectra.solar_irradiance(atmo.wl) / (4.0 * np.pi)

    for start in range(0, n_view, chunk):
        sl = slice(start, min(start + chunk, n_view))
        zv = np.radians(view_zen[sl])
        azv = np.radians(view_az[sl])
        d = np.stack([np.sin(zv) * np.sin(azv), np.sin(zv) * np.cos(azv), np.cos(zv)], axis=1)
        mu_v = np.cos(zv)
        p = np.maximum(r0 * np.sin(zv), 1e-9)
        d0 = r0 * mu_v
        u_top = np.sqrt(np.maximum(r_toa**2 - p**2, 0.0))

        s_b = np.sqrt(np.maximum(r_bounds[:, None] ** 2 - p[None, :] ** 2, 0.0))
        ub = np.clip(s_b, d0[None, :], u_top[None, :])
        u, du = _quadrature_points(ub, n_gauss)

        r_pts = np.sqrt(p[None, :] ** 2 + u**2)  # (n_pts, n_ray)
        z_pts = (r_pts - R_EARTH) / 1e3
        x_vec = (u - d0[None, :])[..., None] * d  # (n_pts, n_ray, 3)
        x_vec[..., 2] += r0
        h_pts = np.hypot(x_vec[..., 0], x_vec[..., 1]) / 1e3  # horizontal distance [km]

        mu_sun = np.einsum("prc,pc->pr", x_vec, shat[None, :]) / np.maximum(r_pts, 1e-9)
        mu_sun = np.clip(mu_sun, -1.0, 1.0)
        p_sun = r_pts * np.sqrt(np.maximum(1.0 - mu_sun**2, 0.0))
        lit = is_sunlit(p_sun, mu_sun)

        cos_theta = np.clip(d @ shat, -1.0, 1.0)  # constant along each ray
        p_ray = spectra.rayleigh_phase(cos_theta)  # (n_ray,)
        p_aer = [lay.aerosol.phase(cos_theta, atmo.wl) for lay in atmo.aerosols]

        tau = None
        prev = None
        for i in range(u.shape[0]):
            br, bas, bae, bo3, bno2 = _local_optics_h(atmo, z_pts[i], h_pts[i])
            b_ext = br + bae + bo3 + bno2
            if tau is None:
                tau = 0.5 * b_ext * du[i][:, None]
            else:
                tau = tau + 0.5 * (b_ext + prev) * du[i][:, None]
            prev = b_ext

            b_sca = br + bas
            ph = p_ray[:, None] * (br / np.maximum(b_sca, 1e-300))
            for j, lay in enumerate(atmo.aerosols):
                w = lay.aod550 * lay.profile(z_pts[i]) / 1000.0 * lay.horizontal.weight(h_pts[i])
                _, rel_sca = lay.aerosol.relative_coefficients(atmo.wl)
                beta_as_j = w[:, None] * rel_sca[None, :]
                ph = ph + p_aer[j] * (beta_as_j / np.maximum(b_sca, 1e-300))

            lengths = path_lengths_sun(p_sun[i], mu_sun[i], r_pts[i], r_bounds)
            cols = species_columns(atmo, lengths, r_bounds)
            tau_sun = _tau_from_columns(atmo, cols)
            tau_sun = np.where(lit[i][:, None], tau_sun, 1e3)
            out[sl] += (b_sca * ph * np.exp(-tau_sun - tau) * du[i][:, None]) * F0[None, :]
    return out
