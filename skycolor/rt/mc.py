"""Independent Monte Carlo reference solver for the total spectral sky radiance.

This module exists to *validate* the deterministic radiative-transfer code.  It
is written from scratch on purpose: apart from the pure geometry helpers in
:mod:`skycolor.geometry` (ray/shell intersections, solar position) it shares no
code with :mod:`skycolor.rt.single`.  It is slow and slightly noisy, but it is
simple enough that its correctness can be checked by reading it.

What it computes
----------------
The *total* (all scattering orders) spectral radiance at the observer for a
spherically stratified atmosphere, including a Lambertian ground of albedo
``atmo.ground_albedo``.  Units are W m^-2 sr^-1 nm^-1, the same convention as
the rest of the package.

Estimator
---------
Backward (adjoint) path tracing from the camera.  For one wavelength the
radiance is

    L(Omega_v) = sum over scattering vertices x_k of
                     w_k * (b_sca/b_ext)(x_k) * p_mix(Omega_v . Omega_sun)
                     * T_sun(x_k) * S / (4 pi)

where ``w_k`` is the throughput of the backward path (1 for the first vertex,
times the ground albedo for every ground bounce), ``p_mix`` is the local
mixture of the Rayleigh and aerosol phase functions (both normalised to
4*pi), ``T_sun`` is the direct solar transmittance from the vertex to space and
``S`` is the extraterrestrial solar spectral irradiance.  The path is sampled
with the physical extinction: the distance to the next collision is
exponentially distributed in optical depth, so no explicit
``exp(-tau)`` factor appears.  Absorption is treated analogously (a collision
survives with probability equal to the single-scattering albedo), while the
next-event estimate is always evaluated with the deterministic albedo factor.
A Lambertian ground contributes (a) a next-event term ``rho/pi * S * mu_sun *
T_sun`` for the directly illuminated surface and (b) a continuation with
weight multiplied by ``rho`` and a cosine-weighted reflected direction.

Spectral correctness
--------------------
Wavelengths are completely independent: each wavelength gets its own
extinction profile, majorants, single-scattering albedo, phase functions and
ground albedo.  A single reference wavelength is never used to transport all
wavelengths.

Geometry and sampling
---------------------
Rays are traced as straight lines in three dimensions through concentric
spherical shells (no refraction).  The free flight is sampled with **delta
tracking**: the atmosphere is divided into a few coarse layers; inside a layer
the extinction is bounded by a majorant ``b_max`` and steps of length
``-ln(xi)/b_max`` are taken.  A step either reaches a layer boundary (the
majorant changes) or is a "trial" at which a real collision is accepted with
probability ``b_ext(x)/b_max``.  Because the tabulated extinction is
log-linearly interpolated, the majorant (the maximum of that tabulated field
inside the layer, with a small safety factor) is a true upper bound, so the
sampler is unbiased.  Layer boundaries are crossed one shell at a time, and
rays that hit the Earth's surface are reflected (or absorbed) there.

Sampling budget
---------------
``n_photons`` is the **total** number of backward paths per wavelength; it is
split as evenly as possible among the requested view directions.  The work per
call is therefore proportional to ``n_wl * n_photons``, independently of the
number of view directions.  For a single direction all photons are used for
that direction, which is the right behaviour for validation runs; for a large
viewing grid use fewer photons (the per-direction standard error grows like
``1/sqrt(n_photons/n_view)``).  A 100-direction x 80-wavelength call with the
default budget takes about half a minute on a laptop.

Statistical error
-----------------
The returned ``stderr`` is the sample standard deviation of the per-path
radiance estimators divided by sqrt(number of paths for that direction).  It
is a 1-sigma error bar on the Monte Carlo mean and does **not** include
systematic errors (shell quadrature of the solar beam, truncation at
``max_scatter``, interpolation of the extinction table), which are much smaller
than the statistical error in normal use.

Limitations
-----------
* The direct solar beam is not part of the radiance (as in the deterministic
  single-scattering code): a view direction pointing at the sun returns only
  the scattered light.
* Paths are truncated after ``max_scatter`` atmospheric scattering events;
  the omitted orders are negligible for visible wavelengths but the
  truncation is a (small) negative bias in very thick atmospheres.
* Refraction, polarisation and thermal emission are not modelled.
"""
from __future__ import annotations

import numpy as np

from .. import spectra
from ..constants import DEPOLARIZATION, R_EARTH
from ..geometry import is_sunlit, path_lengths_sun

__all__ = ["mc_radiance", "mc_radiance_split", "mc_flux"]


# ---------------------------------------------------------------------------
# numerical settings
# ---------------------------------------------------------------------------
#: boundaries of the coarse layers used by the delta-tracking sampler [km],
#: measured from the bottom of the model atmosphere.  Deliberately coarse: every
#: layer boundary costs one step for every photon, while the majorant inside a
#: layer only costs a few extra null collisions.
_MC_LAYERS_KM = np.array([0.0, 1.0, 4.0, 12.0, 40.0, 120.0])

#: number of intervals in the fine optical-properties table (quadratic in z)
_N_FINE = 700

#: number of points used to tabulate an aerosol phase function for sampling
_N_PHASE = 2049

#: safety factor applied to the majorant (the tabulated field is log-linearly
#: interpolated, so its maximum inside a layer is attained on a grid point)
_MAJORANT_SAFETY = 1.005

#: Russian roulette: below this weight a path is killed and renormalised
_RR_WEIGHT = 0.05
#: Russian roulette on the path length starts after this many scatterings
_RR_START = 10
#: survival probability per scattering beyond ``_RR_START``
_RR_SURVIVE = 0.75

#: displacement used to step just across a shell boundary [m]
_EPS_MOVE = 1.0e-4

#: hard iteration guard (a bug should raise instead of hanging)
_MAX_STEPS = 200_000


# ---------------------------------------------------------------------------
# geometry / atmosphere tables
# ---------------------------------------------------------------------------
def _layer_bounds(atmo) -> np.ndarray:
    """Radius boundaries of the coarse delta-tracking layers [m]."""
    z0 = float(atmo.z_bounds[0])
    z1 = float(atmo.z_bounds[-1])
    z = z0 + _MC_LAYERS_KM
    z = z[(z > z0) & (z < z1)]
    z = np.concatenate([[z0], z, [z1]])
    return R_EARTH + z * 1.0e3


def _fine_grid(atmo) -> np.ndarray:
    """Heights [km] of the optical-properties table.

    Quadratically spaced (dense near the ground) and including the layer
    boundaries of the atmosphere and of the delta-tracking grid.
    """
    z0 = float(atmo.z_bounds[0])
    z1 = float(atmo.z_bounds[-1])
    u = np.linspace(0.0, 1.0, _N_FINE + 1)
    z = z0 + (z1 - z0) * u**2
    z = np.concatenate([z, np.asarray(atmo.z_bounds, float), _layer_bounds_z(atmo)])
    z = np.clip(z, z0, z1)
    return np.unique(z)


def _layer_bounds_z(atmo) -> np.ndarray:
    z0 = float(atmo.z_bounds[0])
    z1 = float(atmo.z_bounds[-1])
    z = z0 + _MC_LAYERS_KM
    return z[(z > z0) & (z < z1)]


def _aerosol_spectral(atmo):
    """Relative extinction/scattering of every aerosol layer, shape (n_wl,)."""
    rel = []
    for lay in atmo.aerosols:
        ext, sca = lay.aerosol.relative_coefficients(atmo.wl)
        rel.append((np.asarray(ext, float), np.asarray(sca, float)))
    return rel


def _shell_species_density(atmo) -> np.ndarray:
    """Per-shell species column weights, shape ``(n_shell, 3 + n_aerosol)``.

    Multiplying the per-shell geometric path lengths by this matrix gives the
    column amounts (molecules m^-2 for the gases, 550 nm optical depth for the
    aerosol layers) used to evaluate the direct solar beam.
    """
    rb = atmo.r_bounds
    r_mid = np.sqrt(rb[:-1] * rb[1:])
    z_mid = (r_mid - R_EARTH) / 1.0e3
    cols = [atmo.n_air(z_mid), atmo.n_o3(z_mid), atmo.n_no2(z_mid)]
    for lay in atmo.aerosols:
        cols.append(lay.aod550 * lay.profile(z_mid) / 1000.0)
    return np.stack(cols, axis=1)


def _species_tau_weights(atmo, iw: int, rel) -> np.ndarray:
    """Optical-depth weight of every species at wavelength index ``iw``."""
    w = [atmo._sigma_ray[iw], atmo._sigma_o3[iw], atmo._sigma_no2[iw]]
    for ext, _ in rel:
        w.append(ext[iw])
    return np.asarray(w, dtype=float)


def _sun_transmittance(atmo, x, sun_hat, dens, tau_w) -> np.ndarray:
    """Direct solar transmittance from points ``x`` to space.

    ``x`` has shape ``(n, 3)``; returns shape ``(n,)``.  The per-shell path
    lengths are the exact spherical-shell chord lengths; the density inside a
    shell is evaluated at the geometric mean radius (the same quadrature the
    deterministic solver uses).
    """
    r = np.sqrt(np.einsum("ij,ij->i", x, x))
    mu = np.clip(x @ sun_hat / r, -1.0, 1.0)
    p = r * np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    lengths = path_lengths_sun(p, mu, r, atmo.r_bounds)  # (n_shell, n)
    cols = lengths.T @ dens                               # (n, n_species)
    tau = cols @ tau_w
    lit = is_sunlit(p, mu)
    return np.where(lit, np.exp(-tau), 0.0)


# ---------------------------------------------------------------------------
# scattering helpers
# ---------------------------------------------------------------------------
def _sample_rayleigh_mu(rng, n: int) -> np.ndarray:
    """Draw cos(theta) from the Rayleigh phase function (Newton inversion)."""
    gamma = DEPOLARIZATION / (2.0 - DEPOLARIZATION)
    a = 1.0 + 3.0 * gamma
    b = 1.0 - gamma
    norm = 2.0 * a + 2.0 * b / 3.0
    u = rng.random(n)
    mu = 2.0 * u - 1.0
    for _ in range(4):
        cdf = (a * (mu + 1.0) + b * (mu**3 + 1.0) / 3.0) / norm
        pdf = (a + b * mu * mu) / norm
        mu = np.clip(mu - (cdf - u) / np.maximum(pdf, 1e-12), -1.0, 1.0)
    return mu


def _phase_table(atmo, iw: int, j: int):
    """Tabulate aerosol ``j``'s phase function at wavelength ``iw``.

    Returns ``(mu_grid, cdf)`` with the cumulative distribution of cos(theta),
    suitable for inverse-transform sampling.
    """
    mu = np.linspace(-1.0, 1.0, _N_PHASE)
    wl = np.array([atmo.wl[iw]], dtype=float)
    p = np.asarray(atmo.aerosols[j].aerosol.phase(mu, wl), dtype=float)
    p = np.nan_to_num(p.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(mu))])
    if cdf[-1] <= 0.0:  # degenerate phase function: isotropic fallback
        return mu, (mu + 1.0) * 0.5
    return mu, cdf / cdf[-1]


def _rotate(d: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate unit vectors ``d`` by scattering angle ``acos(mu)`` and azimuth phi."""
    n = d.shape[0]
    ref = np.zeros((n, 3))
    ref[:, 2] = 1.0
    bad = np.abs(d[:, 2]) > 0.9
    ref[bad, 0] = 1.0
    ref[bad, 2] = 0.0
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    out = mu[:, None] * d
    out += st[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    return out


def _interp_log(z, z_grid, log_values):
    """Log-linear interpolation of a positive tabulated coefficient."""
    return np.exp(np.interp(z, z_grid, log_values))


# ---------------------------------------------------------------------------
# the tracer
# ---------------------------------------------------------------------------
def _trace_wavelength(atmo, iw, dirs, counts, sun_hat, max_scatter, rng,
                      split, dens, rel):
    """Trace all paths of one wavelength.

    Returns a dict of ``(n_view,)`` arrays: ``total``, ``total_var`` and, when
    ``split`` is true, ``single``, ``single_var``, ``multiple``, ``multiple_var``.
    """
    n_view = dirs.shape[0]
    n_tot = int(counts.sum())
    out = {k: np.zeros(n_view) for k in
           ("total", "total_var", "single", "single_var", "multiple", "multiple_var")}

    r0 = R_EARTH + atmo.elevation_km * 1.0e3
    rb = _layer_bounds(atmo)
    rt = rb[-1]
    n_layer = len(rb) - 1
    z_top = float(atmo.z_bounds[-1])
    z_grid = _fine_grid(atmo)

    # --- optical properties on the fine grid, for this wavelength only ------
    z = z_grid
    b_ray = atmo.n_air(z) * atmo._sigma_ray[iw]
    b_o3 = atmo.n_o3(z) * atmo._sigma_o3[iw]
    b_no2 = atmo.n_no2(z) * atmo._sigma_no2[iw]
    b_ae = np.zeros_like(b_ray)
    b_as = []
    for lay, (ext, sca) in zip(atmo.aerosols, rel):
        w = lay.aod550 * lay.profile(z) / 1000.0
        b_ae = b_ae + w * ext[iw]
        b_as.append(w * sca[iw])
    b_ext = b_ray + b_o3 + b_no2 + b_ae
    b_sca = b_ray + np.sum(b_as, axis=0) if b_as else b_ray.copy()
    del b_o3, b_no2, b_ae

    tiny = 1e-300
    log_b_ext = np.log(np.maximum(b_ext, tiny))
    log_b_ray = np.log(np.maximum(b_ray, tiny))
    log_b_sca = np.log(np.maximum(b_sca, tiny))
    log_b_as = [np.log(np.maximum(x, tiny)) for x in b_as]

    # --- majorant per coarse layer -----------------------------------------
    major = np.empty(n_layer)
    for i in range(n_layer):
        inside = (z >= (rb[i] - R_EARTH) / 1e3 - 1e-12) & (z <= (rb[i + 1] - R_EARTH) / 1e3 + 1e-12)
        major[i] = _MAJORANT_SAFETY * float(np.max(b_ext[inside]))
    if not np.any(major > 0.0):
        return out

    # --- phase-function tables for sampling --------------------------------
    phase_tables = [_phase_table(atmo, iw, j) for j in range(len(atmo.aerosols))]

    # --- bookkeeping -------------------------------------------------------
    vid = np.repeat(np.arange(n_view), counts)
    x = np.zeros((n_tot, 3))
    x[:, 2] = r0
    d = dirs[vid].copy()
    w = np.ones(n_tot)
    n_scat = np.zeros(n_tot, dtype=np.int64)
    n_ground = np.zeros(n_tot, dtype=np.int64)
    contrib = np.zeros(n_tot)
    c_single = np.zeros(n_tot) if split else None
    c_mult = np.zeros(n_tot) if split else None

    solar = float(spectra.solar_irradiance(np.array([atmo.wl[iw]]))[0])
    f0 = solar / (4.0 * np.pi)
    rho = float(atmo.ground_albedo[iw])
    tau_w = _species_tau_weights(atmo, iw, rel)

    alive = np.arange(n_tot)
    n_steps = 0

    while alive.size:
        n_steps += 1
        if n_steps > _MAX_STEPS:  # pragma: no cover - safety net
            raise RuntimeError("Monte Carlo tracer did not converge")

        xa = x[alive]
        da = d[alive]
        wa = w[alive]
        sa = n_scat[alive]
        ga = n_ground[alive]
        na = alive.size

        r = np.sqrt(np.einsum("ij,ij->i", xa, xa))
        b = np.einsum("ij,ij->i", xa, da)

        li = np.searchsorted(rb, r, side="right") - 1
        np.clip(li, 0, n_layer - 1, out=li)
        r_lo = rb[li]
        r_up = rb[li + 1]
        bmax = major[li]

        s = -np.log(np.maximum(rng.random(na), 1e-300)) / bmax
        disc_up = b * b + (r_up - r) * (r_up + r)
        s_up = -b + np.sqrt(np.maximum(disc_up, 0.0))
        disc_lo = b * b - (r - r_lo) * (r + r_lo)
        s_lo = np.where((b < 0.0) & (disc_lo > 0.0),
                        -b - np.sqrt(np.maximum(disc_lo, 0.0)), np.inf)
        s_exit = np.minimum(s_up, s_lo)
        trial = s < s_exit

        # ground: the ray reaches the bottom boundary of the lowest layer
        ground = (~trial) & (li == 0) & np.isfinite(s_lo) & (s_lo <= s_up)
        step = np.where(trial, s, s_exit + _EPS_MOVE)
        step = np.where(ground, s_exit, step)

        xn = xa + step[:, None] * da
        rn = np.sqrt(np.einsum("ij,ij->i", xn, xn))

        d_new = da.copy()
        w_new = wa.copy()
        sc_new = sa.copy()
        gd_new = ga.copy()
        kill = np.zeros(na, dtype=bool)

        # --- escape through the top ----------------------------------------
        escaped = (((r >= rt - 1e-9) & (b >= 0.0))
                   | ((~trial) & (~ground) & (rn >= rt - 1e-9)))
        kill |= escaped

        # --- ground events --------------------------------------------------
        gi = np.nonzero(ground)[0]
        if gi.size:
            xg = xn[gi]
            rg = rn[gi]
            normal = xg / rg[:, None]
            mu_sun = normal @ sun_hat
            mu_pos = np.maximum(mu_sun, 0.0)
            t_sun = _sun_transmittance(atmo, xg, sun_hat, dens, tau_w)
            # direct beam reflected by the Lambertian surface (rho/pi * S * mu * T)
            g_contrib = wa[gi] * (rho / np.pi) * solar * mu_pos * t_sun
            contrib[alive[gi]] += g_contrib
            if split:
                c_mult[alive[gi]] += g_contrib
            # continuation: multiply by the albedo, sample cosine-weighted
            w_new[gi] = w_new[gi] * rho
            mu_r = np.sqrt(np.maximum(rng.random(gi.size), 0.0))
            phi_r = 2.0 * np.pi * rng.random(gi.size)
            d_new[gi] = _rotate(normal, mu_r, phi_r)
            gd_new[gi] += 1
            # lift just above the surface
            xn[gi] = normal * (rg + 1e-6)[:, None]
            if rho <= 0.0:  # black surface: nothing can be reflected
                kill[gi] = True

        # --- trial steps (real collision or null collision) ------------------
        ti = np.nonzero(trial & ~escaped)[0]
        if ti.size:
            zt = np.clip((rn[ti] - R_EARTH) / 1e3, float(atmo.z_bounds[0]), z_top)
            p_real = np.exp(np.interp(zt, z_grid, log_b_ext) - np.log(bmax[ti]))
            p_real = np.minimum(np.maximum(p_real, 0.0), 1.0)
            real = rng.random(ti.size) < p_real
            ci = ti[real]

            if ci.size:
                # local optical properties (consistent with the tracked medium)
                zc = np.clip((rn[ci] - R_EARTH) / 1e3, float(atmo.z_bounds[0]), z_top)
                b_ext_c = _interp_log(zc, z_grid, log_b_ext)
                b_ray_c = _interp_log(zc, z_grid, log_b_ray)
                b_sca_c = _interp_log(zc, z_grid, log_b_sca)
                b_as_c = [_interp_log(zc, z_grid, lb) for lb in log_b_as]
                albedo = np.clip(b_sca_c / np.maximum(b_ext_c, tiny), 0.0, 1.0)

                # next-event estimate towards the sun
                mu_v = np.clip(np.einsum("ij,j->i", da[ci], sun_hat), -1.0, 1.0)
                t_sun = _sun_transmittance(atmo, xn[ci], sun_hat, dens, tau_w)
                p_mix = spectra.rayleigh_phase(mu_v) * (b_ray_c / np.maximum(b_sca_c, tiny))
                for j, lb in enumerate(b_as_c):
                    f_j = lb / np.maximum(b_sca_c, tiny)
                    p_j = np.asarray(atmo.aerosols[j].aerosol.phase(
                        mu_v, np.array([atmo.wl[iw]]))).reshape(-1)
                    p_mix = p_mix + p_j * f_j
                nee = wa[ci] * albedo * p_mix * t_sun * f0
                contrib[alive[ci]] += nee
                if split:
                    first = (sc_new[ci] + 1 == 1) & (gd_new[ci] == 0)
                    c_single[alive[ci[first]]] += nee[first]
                    c_mult[alive[ci[~first]]] += nee[~first]

                sc_new[ci] += 1

                # analog absorption / scattering
                surv = rng.random(ci.size) < albedo
                si = ci[surv]
                kill[ci[~surv]] = True

                if si.size:
                    # choose the scattering species
                    fracs = np.empty((si.size, 1 + len(b_as_c)))
                    fracs[:, 0] = b_ray_c[surv] / np.maximum(b_sca_c[surv], tiny)
                    for j in range(len(b_as_c)):
                        fracs[:, j + 1] = b_as_c[j][surv] / np.maximum(b_sca_c[surv], tiny)
                    fracs = np.maximum(fracs, 0.0)
                    fracs /= np.maximum(fracs.sum(axis=1, keepdims=True), tiny)
                    cum = np.cumsum(fracs, axis=1)
                    pick = (rng.random(si.size)[:, None] > cum[:, :-1]).sum(axis=1)

                    mu_s = np.empty(si.size)
                    is_ray = pick == 0
                    if is_ray.any():
                        mu_s[is_ray] = _sample_rayleigh_mu(rng, int(is_ray.sum()))
                    for j, (mug, cdf) in enumerate(phase_tables):
                        m = pick == j + 1
                        if m.any():
                            mu_s[m] = np.interp(rng.random(int(m.sum())), cdf, mug)
                    phi_s = 2.0 * np.pi * rng.random(si.size)
                    d_new[si] = _rotate(da[si], mu_s, phi_s)

                    # termination: order cap
                    over = sc_new[si] >= max_scatter
                    kill[si[over]] = True

                    # Russian roulette on the weight
                    wk = w_new[si]
                    low = wk < _RR_WEIGHT
                    if low.any():
                        idx = np.nonzero(low)[0]
                        dead = rng.random(idx.size) >= wk[idx] / _RR_WEIGHT
                        kill[si[idx[dead]]] = True
                        w_new[si[idx]] = _RR_WEIGHT

                    # Russian roulette on the path length
                    long = sc_new[si] >= _RR_START
                    if long.any():
                        idx = np.nonzero(long)[0]
                        alive_rr = rng.random(idx.size) < _RR_SURVIVE
                        kill[si[idx[~alive_rr]]] = True
                        w_new[si[idx]] /= _RR_SURVIVE

        # --- commit ---------------------------------------------------------
        x[alive] = xn
        d[alive] = d_new
        w[alive] = w_new
        n_scat[alive] = sc_new
        n_ground[alive] = gd_new
        alive = alive[~kill]

    # --- statistics per direction ------------------------------------------
    cnt = np.bincount(vid, minlength=n_view).astype(float)
    for name, arr in (("total", contrib), ("single", c_single), ("multiple", c_mult)):
        if arr is None:
            continue
        s1 = np.bincount(vid, weights=arr, minlength=n_view)
        s2 = np.bincount(vid, weights=arr * arr, minlength=n_view)
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1.0), 0.0)
        var = np.where(cnt > 1,
                       np.maximum(s2 - s1 * s1 / np.maximum(cnt, 1.0), 0.0)
                       / np.maximum(cnt - 1.0, 1.0),
                       0.0)
        out[name] = mean
        out[name + "_var"] = var / np.maximum(cnt, 1.0)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _prepare(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
             sun_azimuth_deg, n_photons):
    view_zen = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
    view_az = np.broadcast_to(np.asarray(view_azimuth_deg, dtype=float),
                              view_zen.shape).ravel()
    n_view = view_zen.size
    if n_view == 0:
        raise ValueError("at least one view direction is required")
    if int(n_photons) < 1:
        raise ValueError("n_photons must be >= 1")
    s = np.radians(float(sun_zenith_deg))
    a = np.radians(float(sun_azimuth_deg))
    sun_hat = np.array([np.sin(s) * np.sin(a), np.sin(s) * np.cos(a), np.cos(s)])
    z = np.radians(view_zen)
    az = np.radians(view_az)
    dirs = np.stack([np.sin(z) * np.sin(az), np.sin(z) * np.cos(az), np.cos(z)], axis=1)
    counts = np.full(n_view, int(n_photons) // n_view, dtype=np.int64)
    counts[: int(n_photons) % n_view] += 1
    if int(n_photons) < n_view:
        counts = np.maximum(counts, 1)
    return dirs, counts, sun_hat


def _run(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
         sun_azimuth_deg, n_photons, seed, max_scatter, split):
    max_scatter = max(1, int(max_scatter))
    dirs, counts, sun_hat = _prepare(atmo, sun_zenith_deg, view_zenith_deg,
                                     view_azimuth_deg, sun_azimuth_deg, n_photons)
    n_view = dirs.shape[0]
    n_wl = len(atmo.wl)
    keys = ["total", "total_var"] + (["single", "single_var", "multiple", "multiple_var"]
                                     if split else [])
    acc = {k: np.zeros((n_view, n_wl)) for k in keys}

    dens = _shell_species_density(atmo)
    rel = _aerosol_spectral(atmo)
    rngs = np.random.SeedSequence(int(seed)).spawn(n_wl)

    for iw in range(n_wl):
        res = _trace_wavelength(atmo, iw, dirs, counts, sun_hat, int(max_scatter),
                                np.random.default_rng(rngs[iw]), split, dens, rel)
        for k in keys:
            acc[k][:, iw] = res[k]
    for k in acc:
        if k.endswith("_var"):
            acc[k] = np.sqrt(np.maximum(acc[k], 0.0))
    return acc


def mc_radiance(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                sun_azimuth_deg=0.0, n_photons=200_000, seed=0,
                max_scatter=50):
    """Total spectral sky radiance and its Monte Carlo standard error.

    Parameters
    ----------
    atmo : skycolor.atmosphere.Atmosphere
        The model atmosphere.  ``atmo.ground_albedo`` (possibly spectral) is
        used; the observer sits at ``atmo.elevation_km``.
    sun_zenith_deg : float
        Solar zenith angle in degrees (0 = overhead, >90 = twilight).
    view_zenith_deg, view_azimuth_deg : array_like
        Viewing directions in degrees, azimuth clockwise from north.  The two
        arguments are broadcast together.
    sun_azimuth_deg : float
        Solar azimuth in degrees, clockwise from north.
    n_photons : int
        Total number of backward paths **per wavelength**, split evenly over
        the view directions.  See the module docstring.
    seed : int
        Seed of the random number generator; wavelengths use independent
        streams derived from it.
    max_scatter : int
        Maximum number of atmospheric scattering events per path.

    Returns
    -------
    radiance, stderr : ndarray ``(n_view, n_wl)``
        Spectral radiance [W m^-2 sr^-1 nm^-1] and its 1-sigma statistical
        uncertainty (same units).
    """
    acc = _run(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
               sun_azimuth_deg, n_photons, seed, max_scatter, split=False)
    return acc["total"], acc["total_var"]


def mc_radiance_split(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                      sun_azimuth_deg=0.0, n_photons=200_000, seed=0,
                      max_scatter=50):
    """Total, single-scattered and multiple-scattered radiance separately.

    The single-scattered part is defined exactly as in
    :func:`skycolor.rt.single.sky_radiance`: paths with one atmospheric
    scattering event before reaching the camera and no ground reflection.
    Everything else (orders >= 2, and any path that has bounced off the
    ground) is returned as ``multiple``.

    Returns
    -------
    dict
        Keys ``total``, ``single``, ``multiple`` and ``total_err``,
        ``single_err``, ``multiple_err``, each an ``(n_view, n_wl)`` array in
        W m^-2 sr^-1 nm^-1.
    """
    acc = _run(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
               sun_azimuth_deg, n_photons, seed, max_scatter, split=True)
    return {
        "total": acc["total"],
        "single": acc["single"],
        "multiple": acc["multiple"],
        "total_err": acc["total_var"],
        "single_err": acc["single_var"],
        "multiple_err": acc["multiple_var"],
    }


def mc_flux(atmo, sun_zenith_deg, sun_azimuth_deg=0.0, up=True,
            n_mu=12, n_az=4, n_photons=200_000, seed=0, max_scatter=50):
    """Hemispheric spectral flux at the observer's position.

    The flux is the angular integral of the radiance over the upper
    (``up=True``) or lower (``up=False``) hemisphere, ``F = int L mu dOmega``.
    By the invariance of radiance this can be evaluated with the backward
    tracer by looking into the opposite hemisphere: an observer at the top of
    the atmosphere looking down measures the upward radiance, and an observer
    at the ground looking up measures the downward radiance.

    The angular integral uses Gauss-Legendre quadrature in ``mu`` and a
    uniform grid in azimuth.  The returned error combines the Monte Carlo
    errors of the individual directions assuming they are independent.

    Returns
    -------
    flux, stderr : ndarray ``(n_wl,)``
        Spectral flux [W m^-2 nm^-1] and its 1-sigma statistical error.
    """
    gx, gw = np.polynomial.legendre.leggauss(int(n_mu))
    mu = 0.5 * (gx + 1.0)
    wmu = 0.5 * gw
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_az), endpoint=False)

    zen = np.arccos(np.clip(mu, -1.0, 1.0)) * 180.0 / np.pi
    if up:
        zen = 180.0 - zen
    zen = np.repeat(zen, len(phi))
    az = np.tile(phi * 180.0 / np.pi, len(mu))
    weight = np.repeat(wmu * mu, len(phi))  # (n_mu*n_az,)

    rad, err = mc_radiance(atmo, sun_zenith_deg, zen, az,
                           sun_azimuth_deg=sun_azimuth_deg,
                           n_photons=n_photons, seed=seed,
                           max_scatter=max_scatter)
    pref = 2.0 * np.pi / len(phi)
    flux = pref * (weight @ rad)
    var = (pref**2) * ((weight[:, None] ** 2 * err**2).sum(axis=0))
    return flux, np.sqrt(np.maximum(var, 0.0))
