"""Unified finite solar source on straight diffuse paths (experimental).

The point-source table supplies one consistent apparent direction, continuous
column attenuation, and local radiance/solid-angle gain. A true-solid-angle
solar-disk quadrature includes partial occultation without renormalising the
visible part. Detector transfer uses reduced radiance L/n^2; straight diffuse
paths remain an explicitly retained transport approximation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PchipInterpolator

from skycolor import spectra
from skycolor.constants import R_EARTH
from skycolor.optics import species_extinction_weights
from skycolor.rt import mc


def solar_disk(sun_hat, n_radial=2, n_azimuth=8, radius_deg=16.0 / 60.0):
    """Directions and radiance weights for unit center-normal solar irradiance.

    For a uniform spherical cap, L_sun=F_inf/(pi sin(alpha)^2). Thus weights
    sum to 2/(1+cos(alpha)), while sum(weight*cos(rho)) is exactly one.
    Setting radius_deg=0 returns the delta-source limit.
    """
    sun = np.asarray(sun_hat, float)
    sun = sun / np.linalg.norm(sun)
    if radius_deg == 0:
        return sun[None], np.ones(1)
    if n_radial < 1 or n_azimuth < 4 or not 0 < radius_deg < 90:
        raise ValueError("positive radial order, >=4 azimuths and radius in (0,90) required")
    alpha = np.radians(radius_deg)
    x, w = np.polynomial.legendre.leggauss(n_radial)
    cos_rho = 1 - 0.5 * (x + 1) * (1 - np.cos(alpha))
    sin_rho = np.sqrt(np.maximum(1 - cos_rho**2, 0))
    reference = np.array([0., 0., 1.]) if abs(sun[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = reference - np.dot(reference, sun) * sun
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sun, e1)
    phi = 2 * np.pi * (np.arange(n_azimuth) + 0.5) / n_azimuth
    transverse = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    directions = (cos_rho[:, None, None] * sun +
                  sin_rho[:, None, None] * transverse).reshape(-1, 3)
    weights = np.repeat(w / ((1 + np.cos(alpha)) * n_azimuth), n_azimuth)
    np.testing.assert_allclose(np.sum(weights * (directions @ sun)), 1., atol=2e-14)
    return directions, weights


class FiniteSolarSource:
    """Evaluate local solar phase/ground kernels, including detector n^2 transfer."""

    def __init__(self, atmo, table, *, n_radial=2, n_azimuth=8,
                 radius_deg=16.0 / 60.0, chunk=1024):
        self.atmo = atmo
        self.table = table
        if not np.array_equal(atmo.wl, table.wl):
            raise ValueError("The explicit solar source and medium must share their wavelength grid")
        self._medium_identity = getattr(atmo, "_medium", None)
        self._scene_identity = getattr(atmo, "_scene", None)
        self.n_radial = n_radial
        self.n_azimuth = n_azimuth
        self.radius_deg = radius_deg
        self.chunk = chunk
        self.observer_n = float(table.mapping.refractive_index(atmo.elevation_km))
        self.fingerprint = hashlib.sha256()
        self.n_calls = self.n_points = 0
        self._disk = None
        self._original_transmittance = mc._sun_transmittance

    def validate_transport(self, atmo):
        """This experimental clone accepts its exact construction, never an anchor subset."""
        if (atmo is not self.atmo or
                getattr(atmo, "_medium", None) is not self._medium_identity or
                getattr(atmo, "_scene", None) is not self._scene_identity or
                not np.array_equal(atmo.wl, self.table.wl)):
            raise ValueError("Solar-source transport requires the unchanged construction medium; "
                             "build a separate source for restricted/modified atmospheres")

    def _record(self, points):
        self.fingerprint.update(np.asarray(points.shape, dtype=np.int64).tobytes())
        self.fingerprint.update(np.ascontiguousarray(points).tobytes())
        self.n_calls += 1
        self.n_points += len(points)

    def legacy_transmittance(self, atmo, x, sun_hat, dens, tau_w):
        self._record(x)
        return self._original_transmittance(atmo, x, sun_hat, dens, tau_w)

    def _geometry(self, x, sun_hat, iw):
        if self._disk is None:
            self._disk = (np.asarray(sun_hat).copy(), solar_disk(
                sun_hat, self.n_radial, self.n_azimuth, self.radius_deg))
        assert np.array_equal(self._disk[0], sun_hat), "one source instance per solar geometry"
        directions, weights = self._disk[1]
        radius = np.linalg.norm(x, axis=1)
        radial = x / radius[:, None]
        height = np.clip((radius - R_EARTH) / 1e3,
                         self.atmo.z_bounds[0], self.atmo.z_bounds[-1])
        true_mu = np.clip(radial @ directions.T, -1, 1)
        true_e = np.degrees(np.arcsin(true_mu))
        app, transmission, gain, lit = self.table.query(height[:, None], true_e, iw=iw)
        app = np.radians(app)
        local_n = self.table.mapping.refractive_index(height)
        # The solar table gain is LOCAL L. Propagate reduced L/n^2 to the
        # observer, so the n_vertex^2 factor is neither lost nor counted twice.
        gain = gain * (self.observer_n / local_n[:, None])**2
        weighted_gain = gain * weights
        return radial, directions, true_mu, app, transmission, weighted_gain, lit

    def scatter(self, x, backward, sun_hat, iw, ray_fraction, aerosol_fractions):
        """Solar angular integral of T*G*phase, with mixtures normalised to 4pi.

        For scalar iw, fractions have shapes (N,), (N,n_aerosol). For iw=None
        they have (N,n_wl), (N,n_aerosol,n_wl); output is (N,) or (N,n_wl).
        No F_sun/(4pi) or local single-scattering albedo is included here.
        """
        self._record(x)
        n_wl = len(self.atmo.wl)
        out = np.zeros((len(x), n_wl)) if iw is None else np.zeros(len(x))
        wavelengths = self.atmo.wl if iw is None else np.array([self.atmo.wl[iw]])
        for start in range(0, len(x), self.chunk):
            sl = slice(start, min(start + self.chunk, len(x)))
            radial, directions, mu, app, transmission, gain, lit = self._geometry(x[sl], sun_hat, iw)
            active = np.flatnonzero(np.any(lit, axis=1))
            if not active.size:
                continue
            # Twilight has many completely shadowed collisions. Avoid Mie
            # phase evaluations there without changing the sampled paths.
            positions = start + active
            radial, mu, app = radial[active], mu[active], app[active]
            transmission, gain = transmission[active], gain[active]
            dr = np.einsum("ij,ij->i", backward[positions], radial)[:, None]
            horizontal_dot = (backward[positions] @ directions.T - mu * dr) / np.sqrt(
                np.maximum(1 - mu**2, 1e-30))
            phase_mu = np.cos(app) * horizontal_dot + np.sin(app) * dr
            phase_mu = np.where(np.abs(mu) > 1 - 1e-14, np.sign(mu) * dr, phase_mu)
            phase_mu = np.clip(phase_mu, -1, 1)
            p_ray = spectra.rayleigh_phase(phase_mu)
            if iw is None:
                phase = p_ray[..., None] * ray_fraction[positions, None, :]
            else:
                phase = p_ray * ray_fraction[positions, None]
            for j, layer in enumerate(self.atmo.aerosols):
                aerosol_phase = np.asarray(layer.aerosol.phase(phase_mu.ravel(), wavelengths))
                aerosol_phase = aerosol_phase.reshape(phase_mu.shape + (len(wavelengths),))
                if iw is None:
                    phase += aerosol_phase * aerosol_fractions[positions, j, None, :]
                else:
                    phase += aerosol_phase[..., 0] * aerosol_fractions[positions, j, None]
            if iw is None:
                out[positions] = np.sum(transmission * gain[..., None] * phase, axis=1)
            else:
                out[positions] = np.sum(transmission * gain * phase, axis=1)
        return out

    def ground(self, x, sun_hat, iw):
        """Detector-referred solar angular integral of T*G*max(mu_app,0)."""
        self._record(x)
        out = np.zeros(len(x))
        for start in range(0, len(x), self.chunk):
            sl = slice(start, min(start + self.chunk, len(x)))
            _, _, _, app, transmission, gain, _ = self._geometry(x[sl], sun_hat, iw)
            out[sl] = np.sum(transmission * gain * np.maximum(np.sin(app), 0), axis=1)
        return out


@contextmanager
def mc_solar_source(source, *, legacy=False):
    """Scope source-only changes to a tracer clone, leaving every RNG call intact."""
    code = inspect.getsource(mc._trace_wavelength)
    anchor = "    n_view = dirs.shape[0]"
    assert code.count(anchor) == 1
    code = code.replace(anchor, "    _audit_source.validate_transport(atmo)\n" + anchor)
    if legacy:
        namespace = dict(mc.__dict__, _audit_source=source,
                         _sun_transmittance=source.legacy_transmittance)
    else:
        a = "            mu_sun = normal @ sun_hat"
        b = "            contrib[alive[gi]] += g_contrib"
        assert code.count(a) == code.count(b) == 1
        begin, end = code.index(a), code.index(b)
        code = code[:begin] + (
            "            illumination = _audit_source.ground(xg, sun_hat, iw)\n"
            "            g_contrib = wa[gi] * (rho / np.pi) * solar * illumination\n"
        ) + code[end:]
        a = "                # next-event estimate towards the sun"
        b = "                contrib[alive[ci]] += nee"
        assert code.count(a) == code.count(b) == 1
        begin, end = code.index(a), code.index(b)
        code = code[:begin] + (
            "                ray_fraction = b_ray_c / np.maximum(b_sca_c, tiny)\n"
            "                aerosol_fractions = np.column_stack([\n"
            "                    value / np.maximum(b_sca_c, tiny) for value in b_as_c\n"
            "                ]) if b_as_c else np.empty((len(ci), 0))\n"
            "                illumination = _audit_source.scatter(\n"
            "                    xn[ci], da[ci], sun_hat, iw, ray_fraction, aerosol_fractions)\n"
            "                nee = wa[ci] * albedo * illumination * f0\n"
        ) + code[end:]
        namespace = dict(mc.__dict__, _audit_source=source)
    exec(compile(code, str(mc.__file__), "exec"), namespace)
    with patch.object(mc, "_trace_wavelength", namespace["_trace_wavelength"]):
        yield source


def single_zenith(atmo, source, sun_zenith_deg, *, n_gauss=8, segment_km=0.5,
                  density_step_km=0.025):
    """Continuous-density straight-view first order for the same finite source.

    View optical depths come from antiderivatives of fine density PCHIPs;
    source integration uses a separate per-segment Gauss quadrature. Neither
    the production shell/view cumulative approximation nor its solar lookup
    is used. Returns (n_wl,) radiance plus integration metadata.
    """
    bottom = float(atmo.elevation_km)
    top = float(atmo.z_bounds[-1])
    density_grid = np.unique(np.r_[np.arange(bottom, top, density_step_km), atmo.z_bounds, top])
    density_grid = density_grid[density_grid >= bottom]
    fields = [atmo.n_air(density_grid), atmo.n_o3(density_grid), atmo.n_no2(density_grid)]
    fields += [lay.aod550 * lay.profile(density_grid) / 1000 for lay in atmo.aerosols]
    density = np.stack(fields, axis=-1)
    primitive = PchipInterpolator(density_grid, density, axis=0).antiderivative()
    edges = np.unique(np.r_[np.arange(bottom, top, segment_km), atmo.z_bounds, top])
    edges = edges[edges >= bottom]
    gx, gw = np.polynomial.legendre.leggauss(n_gauss)
    half = np.diff(edges) / 2
    z = ((edges[:-1] + edges[1:])[:, None] / 2 + half[:, None] * gx).ravel()
    ds = (half[:, None] * gw * 1000).ravel()
    columns = (primitive(z) - primitive(bottom)) * 1000
    extinction = species_extinction_weights(atmo)
    tau = columns @ extinction
    ray = atmo.n_air(z)[:, None] * atmo._sigma_ray
    aerosol = []
    for layer in atmo.aerosols:
        _, sca = layer.aerosol.relative_coefficients(atmo.wl)
        aerosol.append(layer.aod550 * layer.profile(z)[:, None] / 1000 * sca)
    total_sca = ray + sum(aerosol, start=np.zeros_like(ray))
    ray_fraction = ray / np.maximum(total_sca, 1e-300)
    aerosol_fractions = np.stack(aerosol, axis=1) / np.maximum(
        total_sca[:, None, :], 1e-300) if aerosol else np.empty((len(z), 0, len(atmo.wl)))
    x = np.column_stack([np.zeros((len(z), 2)), R_EARTH + z * 1000])
    backward = np.tile([0., 0., 1.], (len(z), 1))
    angle = np.radians(sun_zenith_deg)
    sun_hat = np.array([0., np.sin(angle), np.cos(angle)])
    illumination = source.scatter(x, backward, sun_hat, None, ray_fraction, aerosol_fractions)
    rad = np.sum(total_sca * illumination * np.exp(-tau) * ds[:, None], axis=0)
    rad *= spectra.solar_irradiance(atmo.wl) / (4 * np.pi)
    return rad, dict(n_points=len(z), n_gauss=n_gauss, segment_km=segment_km,
                     density_step_km=density_step_km)
