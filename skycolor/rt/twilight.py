"""Spherical multiple scattering for twilight: a usable Monte Carlo engine.

The deterministic production model is exact spherical **single** scattering
(:mod:`skycolor.rt.single`) plus plane-parallel **multiple** scattering
(:mod:`skycolor.rt.multiple`).  That combination reproduces twilight down to
about -6 degrees solar elevation and then collapses, because the diffuse field
is horizontally homogeneous while the light that reaches the zenith at -9
degrees has travelled several hundred kilometres horizontally from the sunlit
twilight arch.

This module keeps the exact spherical first-order term and replaces *only* the
multiple-scattering part with the spherical Monte Carlo of
:mod:`skycolor.rt.mc`.  The result is

    L(Omega_v) = L_1(Omega_v)            [deterministic, exact, no variance]
               + L_{>=2}(Omega_v)        [Monte Carlo, spherical]

so the largest, noisiest and most wasteful component of a plain Monte Carlo run
is removed from the estimator.  ``rt.single`` is evaluated on the *requested*
view directions (its cost is negligible), while the Monte Carlo estimate of
``L_{>=2}`` is computed on a coarse angular grid, smoothed on the sphere and
interpolated to the requested directions, and interpolated in wavelength.

The API is deliberately close to :func:`skycolor.rt.solver.sky_radiance_total`::

    radiance, stderr = twilight_radiance(
        atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
        sun_azimuth_deg=0.0, quality="medium", seed=0,
        single=True, multiple=True)

Both returned arrays have shape ``(n_view, n_wl)`` and units
W m^-2 sr^-1 nm^-1.  ``stderr`` is the 1-sigma statistical uncertainty of the
*multiple-scattering* part; the single-scattering part is deterministic and
contributes no variance.

Estimator design
----------------
1. **Split estimator.**  The Monte Carlo returns ``multiple`` from
   :func:`skycolor.rt.mc.mc_radiance_split`, i.e. every path that scattered
   twice or more *or* touched the ground.  That is exactly
   ``total - single`` as defined by ``rt.single``, so adding it to the
   deterministic first-order term is unbiased for the total radiance.  The
   first-scattering contribution (numerically the largest and the noisiest) is
   never estimated stochastically.
2. **Coarse angular grid + spherical smoothing.**  The multiple-scattering
   field is diffuse and smooth in direction.  It is therefore estimated on a
   ``n_zen x n_az`` grid covering zenith angles 0..89 degrees and *half* the
   azimuth range; the field is symmetric about the solar meridian
   ``L(theta, phi_s + dphi) = L(theta, phi_s - dphi)`` because both the
   atmosphere and the ground are azimuthally uniform (in the same sense as
   ``rt.multiple``: the aerosol ``HorizontalField`` is not used).  A
   local-linear regression on the sphere, in log space, smooths the noisy
   coarse values and interpolates them to the requested directions.  The
   regression weights give the error propagation for free.
3. **Spectral interpolation.**  Wavelengths are independent Monte Carlo
   problems, but the multiple-scattering spectrum is smooth.  ``n_anchor``
   wavelengths spanning the model grid are computed exactly and ``log L`` is
   interpolated linearly in wavelength.  The single-scattering term is still
   evaluated at *every* wavelength, so the sharp spectral structure (ozone,
   NO2, Rayleigh) is not degraded.
4. **Error estimate.**  ``stderr`` combines the Monte Carlo standard error of
   the coarse grid with the linear smoother weights and the wavelength
   interpolation weights.  It is a statistical error only; the systematic
   errors are listed in ``docs/TWILIGHT.md``.

Quality presets
---------------
==========  =========  =========  =========  =================================
quality     n_zen      n_az       anchors    photons per anchor wavelength
==========  =========  =========  =========  =================================
``"low"``   6          8          5          4.0e6
``"medium"``8          12         7          1.2e7
``"high"``  10         16         10         4.0e7
``"ultra"`` 12         20         13         1.2e8
==========  =========  =========  =========  =================================

The photon count is the total per anchor wavelength and is split evenly over
the coarse grid, so the standard error per coarse direction scales as
``1/sqrt(n_photons / n_direction)``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..atmosphere import Atmosphere
from . import mc as _mc
from .single import sky_radiance as _single

__all__ = ["twilight_radiance", "multiple_scattered", "QUALITY"]

#: quality presets: (n_zen, n_az, n_anchor, photons per anchor wavelength)
QUALITY = {
    "low": dict(n_zen=6, n_az=8, n_anchor=5, n_photons=4.0e6),
    "medium": dict(n_zen=8, n_az=12, n_anchor=7, n_photons=1.2e7),
    "high": dict(n_zen=10, n_az=16, n_anchor=10, n_photons=4.0e7),
    "ultra": dict(n_zen=12, n_az=20, n_anchor=13, n_photons=1.2e8),
}

#: largest zenith angle of the coarse grid (the exact horizon is singular)
_MAX_COARSE_ZENITH = 89.0

#: largest photon block handed to a single Monte Carlo call (memory bound)
_MAX_BLOCK = 2_000_000

#: smoothing bandwidth, in units of the median nearest-neighbour spacing
_BANDWIDTH = 1.6

#: minimum number of coarse points in a local regression neighbourhood
_MIN_NEIGHBOURS = 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_directions(view_zenith_deg, view_azimuth_deg):
    """Broadcast the view angles and return the unit vectors and the angles."""
    zen = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
    az = np.broadcast_to(np.asarray(view_azimuth_deg, dtype=float), zen.shape).ravel()
    z = np.radians(zen)
    a = np.radians(az)
    vec = np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)], axis=1)
    return zen, az, vec


def _sub_atmosphere(atmo, index):
    """A copy of ``atmo`` restricted to ``wl[index]``.

    The aerosol objects and the vertical grid are shared; only the spectral
    grid is sliced, which is all that a single-wavelength Monte Carlo run
    needs.
    """
    wl = np.asarray(atmo.wl, dtype=float)
    idx = np.asarray(index, dtype=int)
    if np.array_equal(idx, np.arange(wl.size)):
        return atmo
    ground = np.asarray(atmo.ground_albedo, dtype=float)
    return Atmosphere(
        wl[idx],
        ozone_du=getattr(atmo, "ozone_du", 300.0),
        no2_column=getattr(atmo, "no2_column", 0.0),
        aerosols=list(atmo.aerosols),
        ground_albedo=ground[idx],
        surface_pressure=getattr(atmo, "surface_pressure", 101325.0),
        z_bounds_km=np.asarray(atmo.z_bounds, dtype=float),
        elevation_km=float(getattr(atmo, "elevation_km", 0.0)),
    )


def _coarse_grid(n_zen, n_az):
    """Coarse viewing grid: (zenith, azimuth offset from the sun) in degrees.

    The azimuth range is the half circle ``[0, 180]``; the full sky follows
    from the mirror symmetry about the solar meridian.
    """
    n_zen = max(2, int(n_zen))
    n_az = max(2, int(n_az))
    # denser towards the horizon, where the field varies fastest
    u = np.linspace(0.0, 1.0, n_zen)
    zen = _MAX_COARSE_ZENITH * u ** 1.25
    az = np.linspace(0.0, 180.0, n_az)
    zz, aa = np.meshgrid(zen, az, indexing="ij")
    return zz.ravel(), aa.ravel()


def _unit_vectors(zen_deg, az_deg, sun_zenith_deg, sun_azimuth_deg):
    """Unit vectors for a viewing direction given in *absolute* azimuth.

    The azimuth stored on the coarse grid is a *relative* offset from the sun;
    it is converted to an absolute azimuth before the geometry is formed.
    """
    return _as_directions(zen_deg, az_deg + sun_azimuth_deg)[2]


def _angular_distance(a, b):
    """Great-circle distance between unit vectors, clipped for round-off."""
    return np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def _tangent_basis(n):
    """Two orthonormal vectors spanning the tangent plane at ``n``."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def _local_linear_weights(coarse_vec, query_vec, bandwidth=None):
    """Smoother matrix of a local-linear regression on the sphere.

    Returns ``W`` with shape ``(n_query, n_coarse)`` such that a field ``f``
    on the coarse directions is estimated at the query directions by
    ``W @ f``.  The regression is linear in the three-dimensional embedding,
    which makes it exact for a field that is linear in the direction cosines
    and removes the first-order bias of a plain kernel average.
    """
    coarse_vec = np.atleast_2d(coarse_vec)
    query_vec = np.atleast_2d(query_vec)
    n_q = query_vec.shape[0]
    n_c = coarse_vec.shape[0]
    if n_c == 0:
        raise ValueError("the coarse grid is empty")
    if bandwidth is None:
        d = _angular_distance(coarse_vec, coarse_vec)
        np.fill_diagonal(d, np.inf)
        bandwidth = _BANDWIDTH * float(np.median(np.min(d, axis=1)))
    bandwidth = max(float(bandwidth), 1e-3)

    W = np.zeros((n_q, n_c))
    k = min(n_c, max(_MIN_NEIGHBOURS, 14))
    for q in range(n_q):
        nq = query_vec[q]
        gamma = np.arccos(np.clip(coarse_vec @ nq, -1.0, 1.0))
        order = np.argsort(gamma)
        idx = order[:k]
        # keep a few extra points so that the kernel has a chance to decay
        far = order[k:min(n_c, k + 6)]
        idx = np.concatenate([idx, far])
        d = gamma[idx]
        w = np.exp(-0.5 * (d / bandwidth) ** 2)
        if not np.any(w > 1e-8):  # pragma: no cover - degenerate grid
            w = np.ones_like(d)
        t1, t2 = _tangent_basis(nq)
        delta = coarse_vec[idx] - nq
        X = np.column_stack([np.ones(idx.size), delta @ t1, delta @ t2])
        M = X.T @ (w[:, None] * X)
        try:
            # smoother row = e0^T (X^T W X)^-1 X^T W
            Y = np.linalg.solve(M, (X * w[:, None]).T)         # (3, n_idx)
            row = Y[0]
        except np.linalg.LinAlgError:  # pragma: no cover
            row = w / w.sum()
        W[q, idx] = row
    return W, bandwidth


# ---------------------------------------------------------------------------
# Monte Carlo on the coarse grid
# ---------------------------------------------------------------------------
def _mc_coarse(atmo, sun_zenith_deg, sun_azimuth_deg, zen, az_rel, n_anchor,
               n_photons, seed):
    """Multiple-scattering radiance on the coarse grid at anchor wavelengths.

    Returns
    -------
    ms : ndarray ``(n_anchor, n_dir)``
    err : ndarray ``(n_anchor, n_dir)``
    wl_anchor : ndarray ``(n_anchor,)``
    """
    wl = np.asarray(atmo.wl, dtype=float)
    n_wl = wl.size
    if n_wl == 0:
        raise ValueError("the atmosphere has no wavelengths")
    # anchor wavelengths, evenly spread over the model grid (indices, no dupes)
    n_anchor = int(min(max(int(n_anchor), 1), n_wl))
    if n_anchor == n_wl:
        anchors = np.arange(n_wl)
    else:
        anchors = np.unique(np.round(np.linspace(0, n_wl - 1, n_anchor)).astype(int))

    zen = np.asarray(zen, dtype=float)
    az_abs = np.asarray(az_rel, dtype=float) + float(sun_azimuth_deg)
    n_dir = zen.size
    ms = np.zeros((anchors.size, n_dir))
    err = np.zeros((anchors.size, n_dir))

    per_dir = max(1, int(round(float(n_photons) / n_dir)))
    total = per_dir * n_dir
    n_block = max(1, int(np.ceil(total / _MAX_BLOCK)))
    for k, iw in enumerate(anchors):
        sub = _sub_atmosphere(atmo, [iw])
        acc_w = np.zeros(n_dir)
        acc_m = np.zeros(n_dir)
        acc_v = np.zeros(n_dir)
        for b in range(n_block):
            block = total // n_block + (1 if b < total % n_block else 0)
            res = _mc.mc_radiance_split(
                sub, sun_zenith_deg, zen, az_abs,
                sun_azimuth_deg=sun_azimuth_deg,
                n_photons=int(block), seed=int(seed) + 1000 * k + b,
            )
            # every direction has the same number of paths, so the pooled mean
            # is the arithmetic mean of the block means
            acc_w += 1.0
            acc_m += res["multiple"][:, 0]
            acc_v += res["multiple_err"][:, 0] ** 2
        ms[k] = acc_m / acc_w
        err[k] = np.sqrt(acc_v) / acc_w
    return ms, err, wl[anchors]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def multiple_scattered(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                       sun_azimuth_deg=0.0, quality="medium", seed=0,
                       grid=None):
    """Monte Carlo estimate of the multiple-scattering radiance (orders >= 2).

    Parameters are those of :func:`twilight_radiance`; ``grid`` optionally
    overrides the quality preset with a dict containing ``n_zen``, ``n_az``,
    ``n_anchor`` and ``n_photons``.

    Returns
    -------
    ms, stderr, info : ndarray, ndarray, dict
        ``ms`` and ``stderr`` are ``(n_view, n_wl)`` in
        W m^-2 sr^-1 nm^-1.  ``info`` records the coarse grid, the anchor
        wavelengths and the smoothing bandwidth.
    """
    zen, az, vec = _as_directions(view_zenith_deg, view_azimuth_deg)
    cfg = dict(QUALITY["medium" if quality is None else quality])
    if grid:
        cfg.update(grid)
    wl = np.asarray(atmo.wl, dtype=float)

    cz, ca = _coarse_grid(cfg["n_zen"], cfg["n_az"])
    ms_c, err_c, wl_a = _mc_coarse(
        atmo, float(sun_zenith_deg), float(sun_azimuth_deg), cz, ca,
        cfg["n_anchor"], cfg["n_photons"], int(seed),
    )
    coarse_vec = _unit_vectors(cz, ca, float(sun_zenith_deg), float(sun_azimuth_deg))

    # Query directions in the same (relative-azimuth) parameterisation, folded
    # onto [0, 180] by the mirror symmetry about the solar meridian.
    rel_az = az - float(sun_azimuth_deg)
    rel_fold = np.abs((rel_az + 180.0) % 360.0 - 180.0)
    query_vec = _as_directions(zen, rel_fold + float(sun_azimuth_deg))[2]

    n_anchor = wl_a.size
    n_view = zen.size
    # --- smooth in direction, anchor by anchor -----------------------------
    ms_a = np.zeros((n_anchor, n_view))
    sig_a = np.zeros((n_anchor, n_view))
    bandwidth = None
    Wq, bandwidth = _local_linear_weights(coarse_vec, query_vec)
    for k in range(n_anchor):
        vals = ms_c[k]
        errs = err_c[k]
        scale = float(np.max(vals)) if vals.size else 0.0
        if scale <= 0.0:
            continue                       # nothing was scattered at all
        floor = 1e-10 * scale
        z = np.log(np.maximum(vals, floor))
        s = np.where(vals > 0, errs / np.maximum(vals, floor), 1.0)
        zq = Wq @ z
        var = (Wq ** 2) @ (s ** 2)
        ms_a[k] = np.exp(zq)
        sig_a[k] = ms_a[k] * np.sqrt(np.maximum(var, 0.0))

    # --- interpolate in wavelength -----------------------------------------
    ms = np.zeros((n_view, wl.size))
    stderr = np.zeros((n_view, wl.size))
    if n_anchor == 1:
        ms[:] = ms_a[0][:, None]
        stderr[:] = sig_a[0][:, None]
    else:
        loga = np.log(np.maximum(ms_a, 1e-300))
        # interpolation weights of the PCHIP basis in wavelength (n_wl, n_anchor);
        # valid for every direction because the knots do not depend on it
        wmat = np.zeros((wl.size, n_anchor))
        for k in range(n_anchor):
            e = np.zeros(n_anchor)
            e[k] = 1.0
            wmat[:, k] = PchipInterpolator(wl_a, e, extrapolate=True)(wl)
        ms = np.exp(np.minimum(wmat @ loga, 700.0)).T
        stderr = np.sqrt(np.maximum((wmat ** 2) @ (sig_a ** 2), 0.0)).T

    info = dict(
        n_zen=cfg["n_zen"], n_az=cfg["n_az"], n_anchor=cfg["n_anchor"],
        n_photons=cfg["n_photons"], anchor_wl=wl_a, bandwidth=bandwidth,
        coarse_zenith=cz, coarse_azimuth=ca,
    )
    return ms, stderr, info


def twilight_radiance(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                      sun_azimuth_deg=0.0, quality="medium", seed=0,
                      single=True, multiple=True):
    """Total spectral sky radiance in twilight, with a statistical error bar.

    This is the production entry point of the spherical twilight engine.  It
    returns the exact spherical first-order term plus a Monte Carlo estimate of
    all higher orders; see the module docstring for the estimator design.

    Parameters
    ----------
    atmo : skycolor.atmosphere.Atmosphere
        The model atmosphere.  ``ground_albedo`` and ``elevation_km`` are used.
    sun_zenith_deg : float
        Solar zenith angle (0 = overhead, 90 = horizon, > 90 = twilight).
    view_zenith_deg, view_azimuth_deg : array_like
        Viewing directions, degrees, azimuth clockwise from north.  The two
        arguments broadcast against each other.
    sun_azimuth_deg : float
        Solar azimuth, degrees clockwise from north.
    quality : {"low", "medium", "high", "ultra"}
        Photon budget / angular resolution preset (see the module docstring).
    seed : int
        Seed of the Monte Carlo; results are reproducible for a given seed.
    single, multiple : bool
        Include the deterministic first-order / the Monte Carlo higher orders.

    Returns
    -------
    radiance, stderr : ndarray ``(n_view, n_wl)``
        Spectral radiance and its 1-sigma statistical uncertainty, in
        W m^-2 sr^-1 nm^-1.

    Notes
    -----
    * Above about -4 degrees solar elevation the deterministic model
      (:func:`skycolor.rt.solver.sky_radiance_total`) is faster and more
      accurate; use this function for the deep twilight where that model
      fails.
    * ``stderr`` covers the Monte Carlo part only.  The first-order term is
      deterministic (up to its own quadrature) and carries no statistical
      error.
    * The Monte Carlo models a *horizontally uniform* atmosphere, as
      ``rt.multiple`` does; a non-uniform aerosol ``HorizontalField`` affects
      the first-order term only.
    """
    zen, az, _ = _as_directions(view_zenith_deg, view_azimuth_deg)
    n_view = zen.size
    n_wl = len(atmo.wl)
    total = np.zeros((n_view, n_wl))
    err = np.zeros((n_view, n_wl))

    if single:
        total += _single(atmo, sun_zenith_deg, zen, az,
                         sun_azimuth_deg=sun_azimuth_deg)
    if multiple:
        ms, ms_err, _ = multiple_scattered(
            atmo, sun_zenith_deg, zen, az, sun_azimuth_deg=sun_azimuth_deg,
            quality=quality, seed=seed,
        )
        total += ms
        err += ms_err
    return total, err
