"""Single scattering in a spherically stratified atmosphere.

This is the backbone of the model.  It is *exact* (up to quadrature) for
first-order scattering, which is the process that carries almost all of the
colour information in a clear sky.  Two things make this more than a
textbook formula:

* the atmosphere is **spherical** -- the solar zenith angle, the air density
  and the scattering angle are all evaluated at the true position of each
  scattering event, so the terminator, the twilight arch and the slanted
  horizon glow appear without any extra machinery;
* the *direct* solar beam is the unified finite-disk source of
  :mod:`skycolor.solar_source`: every scattering point consumes the same
  apparent directions, continuous refracted columns, occultation and
  angular-domain gain as every other backend, with the phase argument taken
  from the *apparent* beam direction.

The viewing ray itself stays a straight line with trapezoid optical depth --
retaining the view quadrature separate from the refracted solar source is a
standing approximation of this backend (see ``CAPABILITIES``).
"""
from __future__ import annotations

import numpy as np

from .. import spectra
from ..constants import R_EARTH
from ..solar_source import SolarSource

#: standing approximations of this backend (see rt.solver.solve for the
#: request-level checks)
CAPABILITIES = {
    "scattering": "single scattering, exact spherical-shell quadrature",
    "solar_source": "unified finite disk (apparent directions, continuous "
                    "refracted columns, occultation, local gain)",
    "horizontal_field": "applied to view-path coefficients at every point",
    "refraction": "solar side fully refracted; straight viewing rays and "
                  "geometric view geometry (retained approximation)",
}


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


def species_columns(atmo, lengths: np.ndarray, r_bounds: np.ndarray) -> np.ndarray:
    """Per-species column amounts along a set of shell path lengths.

    ``lengths`` has shape ``(n_shell, n_points)``.  Returns ``(n_points,
    3 + n_aerosol_layers)``: air, ozone, NO2 (molecules m^-2) followed by one
    column per aerosol layer (550 nm extinction optical depth).  Thin
    wrapper over the shared algebra in :mod:`skycolor.optics`; kept for the
    diagnostics and independent checks that deliberately compare the shell
    chord discretisation against the source's continuous columns.
    """
    from ..optics import shell_species_columns, shell_species_weights

    return shell_species_columns(lengths, shell_species_weights(atmo, r_bounds))


def direct_sun_transmittance(atmo, sun_zenith_deg: float) -> np.ndarray:
    """Direct solar transmittance at the observer, shape ``(n_wl,)``.

    Pure attenuation of the unified source's central beam (zero when the
    central ray hits the opaque ground).  This is a centre-beam diagnostic:
    the disk-integrated flux on a horizontal surface is
    :meth:`skycolor.solar_source.SolarSource.direct_irradiance`.
    """
    source = SolarSource(atmo, float(sun_zenith_deg), 0.0)
    return np.asarray(source.central_transmittance(), dtype=float)


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
    vectorised numpy.  The solar part comes from one
    :class:`~skycolor.solar_source.SolarSource` bound to this atmosphere and
    sun direction; its table (built lazily once per medium) supplies the
    apparent directions, continuous refracted columns, occultation and gain
    at every scattering point.

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

    source = SolarSource(atmo, float(sun_zenith_deg), float(sun_azimuth_deg))

    r0 = R_EARTH + atmo.elevation_km * 1e3
    r_bounds = atmo.r_bounds
    r_toa = r_bounds[-1]

    for start in range(0, n_view, chunk):
        sl = slice(start, min(start + chunk, n_view))
        zv = np.radians(view_zen[sl])
        azv = np.radians(view_az[sl])
        d = np.stack([np.sin(zv) * np.sin(azv), np.sin(zv) * np.cos(azv),
                      np.cos(zv)], axis=1)
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

        # ---- local optical properties, shape (n_pts, n_ray, n_wl) ----------
        beta_ray = atmo.n_air(z_pts)[..., None] * atmo._sigma_ray
        beta_o3 = atmo.n_o3(z_pts)[..., None] * atmo._sigma_o3
        beta_no2 = atmo.n_no2(z_pts)[..., None] * atmo._sigma_no2
        beta_ae = np.zeros_like(beta_ray)
        beta_as = np.zeros_like(beta_ray)
        aer_w = []
        aer_optics = [lay.aerosol.relative_coefficients(atmo.wl)
                      for lay in atmo.aerosols]
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
            tau = np.concatenate([first[None], first[None] + np.cumsum(dt, axis=0)],
                                 axis=0)
        else:
            tau = first[None]

        # ---- unified solar source at every scattering point ----------------
        # The kernel integrates T * G * p_mix over the disk beams with the
        # phase argument from the *apparent* beam direction; ray fractions
        # carry the local scattering mixture (4-pi normalised, aerosols in
        # registry order).
        n_pts = u.shape[0]
        n_ray = u.shape[1]
        safe_sca = np.maximum(beta_sca, 1e-300)
        ray_fraction = (beta_ray / safe_sca).reshape(n_pts * n_ray, n_wl)
        if atmo.aerosols:
            aerosol_fractions = np.stack(
                [(w[..., None] * aer_optics[j][1]) / safe_sca
                 for j, w in enumerate(aer_w)],
                axis=-1).reshape(n_pts * n_ray, len(atmo.aerosols), n_wl)
        else:
            aerosol_fractions = np.empty((n_pts * n_ray, 0, n_wl))
        illumination = source.scatter(
            x_vec.reshape(n_pts * n_ray, 3),
            np.broadcast_to(d[None, :, :], (n_pts, n_ray, 3)).reshape(n_pts * n_ray, 3),
            iw=None, ray_fraction=ray_fraction,
            aerosol_fractions=aerosol_fractions,
        ).reshape(n_pts, n_ray, n_wl)

        contrib = (beta_sca * illumination * np.exp(-tau) * du[..., None])
        out[sl] += contrib.sum(axis=0) * spectra.solar_irradiance(atmo.wl) / (4.0 * np.pi)
    return out
