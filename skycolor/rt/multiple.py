"""Multiple scattering (all scattering orders >= 2).

The model is a successive-orders-of-scattering (SOS) solver in a
**plane-parallel** atmosphere illuminated by a **pseudo-spherical** direct
beam.  It returns only the contribution of scattering orders >= 2; the exact
first-order term is produced by :mod:`skycolor.rt.single`.

Method
------
* The vertical grid is the one used by :class:`skycolor.atmosphere.Atmosphere`
  (54 layers between the ground and 120 km).  The diffuse field is
  plane-parallel, so the optical depth of a layer is
  ``beta_ext * dz``.  Rays at the observer are plane-parallel as well.
* The angular dependence is expanded in a Fourier series in azimuth and
  discretised with Gauss-Legendre nodes in ``mu = cos(zenith)``, in the same
  spirit as DISORT.  The phase function's Legendre moments come from
  :meth:`Atmosphere.phase_moments`.  The azimuthal expansion is exact (up to
  the moment truncation) -- no m = 0 approximation.
* The **direct beam** is treated pseudo-spherically: for the point of the
  vertical column at radius ``r`` the solar ray has impact parameter
  ``p = r sin(sza)`` and local direction cosine ``mu = cos(sza)``.  Its
  transmittance is computed with the spherical helpers of
  :mod:`skycolor.rt.single` and :mod:`skycolor.geometry`, so rays that graze
  the Earth (twilight) are handled correctly, and layers inside the Earth's
  shadow receive no beam.
* The beam source is the physical beam radiance ``F_sun T(z)`` at the layer
  mid-point.  (An alternative normalisation by the actual transmittance
  difference between layer boundaries would close the column energy budget
  exactly, but it over-weights grazing twilight rays by a large factor, so it
  is deliberately not used.  See ``docs/MULTIPLE_SCATTERING.md``.)
* The recursion is
  ``L^(1) = transfer(source = J_direct)``,
  ``L^(n) = transfer(source = scattering(L^(n-1)))`` with a Lambertian
  ground reflection.  At order 1 the upward boundary also contains the
  reflection of the direct beam, so paths ``beam -> ground -> scatter`` are
  included.  The returned multiple-scattering radiance is
  ``view_integral(scattering(L_total))`` where ``L_total`` is the converged
  diffuse field, i.e. exactly ``sum_{n>=2} L^(n)``.
* Scattering orders are summed until the relative change of the field drops
  below ``tol`` (default 1e-3) or ``max_orders`` (default 40) is reached.

Approximations and known limitations
------------------------------------
* Plane-parallel diffuse transport.  The direct beam is spherical, the
  diffuse field is not.  At large solar zenith angles the diffuse geometry
  near the horizon is therefore approximate; the exact spherical horizon is
  still provided by the first-order term.
* The phase function is truncated at ``n_moments`` Legendre moments.  With
  the default 32 moments the truncation error is of order 1e-4 for the
  aerosol models in this package (g <= 0.75); the forward peak is the most
  sensitive part.  The number of moments is capped at ``2 * n_mu`` so that
  the Gauss quadrature integrates the energy identity ``int p dmu = 2``
  exactly, which is what makes the ``m = 0`` operator energy conserving.
* Polarisation is neglected, and the ground is Lambertian.
* Horizontal homogeneity is assumed for the diffuse field, so the
  ``HorizontalField`` of the aerosol layers does not enter here (it is used by
  the single-scattering module).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import assoc_legendre_p_all

from .. import spectra
from ..constants import R_EARTH
from ..geometry import is_sunlit, path_lengths_sun
from .single import _tau_from_columns, species_columns

__all__ = ["SOSResult", "solve", "multiple_scattering"]

#: default numerical settings
_N_MU = 32
_N_MODES = 12
_N_MOMENTS = 32
_MAX_ORDERS = 40
_TOL = 1e-3
_CHUNK = 256


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SOSResult:
    """Diagnostics returned by :func:`solve`.

    Attributes
    ----------
    radiance : ndarray (n_view, n_wl)
        Multiple-scattering radiance (orders >= 2) [W m^-2 sr^-1 nm^-1].
    radiance_single : ndarray or None
        The same quadrature applied to the *first* order only.  Useful for
        validation against :func:`skycolor.rt.single.sky_radiance`.
    flux_up_toa : ndarray (n_wl,)
        Upward diffuse flux at the top of the atmosphere.
    flux_down_boa, flux_up_boa : ndarray (n_wl,)
        Downward/upward diffuse flux at the bottom of the atmosphere.
    direct_down_boa : ndarray (n_wl,)
        Direct solar irradiance reaching the ground (zero in shadow).
    incident_flux : ndarray (n_wl,)
        Direct solar irradiance at the top of the atmosphere, ``F_sun * mu0``.
    orders : int
        Number of scattering orders that were summed.
    converged : bool
        Whether the ``tol`` criterion was met before ``max_orders``.
    """

    radiance: np.ndarray
    radiance_single: np.ndarray | None
    flux_up_toa: np.ndarray
    flux_down_boa: np.ndarray
    flux_up_boa: np.ndarray
    direct_down_boa: np.ndarray
    incident_flux: np.ndarray
    orders: int
    converged: bool


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _gauss_nodes(n_mu: int):
    """Gauss-Legendre nodes/weights on [-1, 1], with an even node count.

    An even count guarantees that no node sits exactly at ``mu = 0`` (where
    the plane-parallel characteristic would be singular).
    """
    n_mu = max(2, int(n_mu))
    if n_mu % 2:
        n_mu += 1
    x, w = np.polynomial.legendre.leggauss(n_mu)
    return x, w


def _log_factorial_ratio(l_max: int, m: int) -> np.ndarray:
    """``(l-m)! / (l+m)!`` for ``l = 0..l_max`` (zero for ``l < m``)."""
    from scipy.special import gammaln

    out = np.zeros(l_max + 1)
    if m > l_max:
        return out
    ls = np.arange(m, l_max + 1)
    out[m:] = np.exp(gammaln(ls - m + 1.0) - gammaln(ls + m + 1.0))
    return out


def _sun_transmittance_boundaries(atmo, sun_zenith_deg: float) -> np.ndarray:
    """Direct-beam transmittance at every layer boundary, shape ``(nz+1, n_wl)``.

    For the point of the vertical column at radius ``r`` the solar ray has
    impact parameter ``p = r sin(sza)`` and direction cosine
    ``cos(sza)``.  The optical depth is integrated along the true spherical
    ray with the helpers of :mod:`skycolor.rt.single`, and rays that
    intersect the Earth are masked to zero transmittance.
    """
    rb = atmo.r_bounds
    sza = np.radians(float(sun_zenith_deg))
    mu0 = np.cos(sza)
    p = rb * np.sin(sza)
    c = np.full_like(rb, mu0)
    lengths = path_lengths_sun(p, c, rb, rb)  # (n_shell, n_boundary)
    cols = species_columns(atmo, lengths, rb)
    tau = _tau_from_columns(atmo, cols)
    if getattr(atmo, "refraction", False):
        from ..refraction import get_tables

        z_b = (rb - R_EARTH) / 1e3
        crit = get_tables(atmo).shadow_zenith_true(z_b)
        lit = c > np.cos(np.radians(crit))
    else:
        lit = is_sunlit(p, c)
    return np.where(lit[:, None], np.exp(-tau), 0.0)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve(
    atmo,
    sun_zenith_deg,
    view_zenith_deg=None,
    view_azimuth_deg=0.0,
    sun_azimuth_deg=0.0,
    *,
    n_mu: int = _N_MU,
    n_modes: int = _N_MODES,
    n_moments: int = _N_MOMENTS,
    max_orders: int = _MAX_ORDERS,
    tol: float = _TOL,
    chunk: int = _CHUNK,
    single_view: bool = False,
) -> SOSResult:
    """Run the SOS solver and return radiance plus flux diagnostics.

    Parameters are the same as :func:`multiple_scattering`; ``single_view``
    additionally evaluates the first-order field with the same quadrature.
    """
    wl = np.asarray(atmo.wl, dtype=float)
    n_wl = wl.size
    z_bounds = np.asarray(atmo.z_bounds, dtype=float)
    nz = z_bounds.size - 1

    # -- view directions -----------------------------------------------------
    if view_zenith_deg is None:
        n_view = 0
        vza = np.zeros(0)
        vaz = np.zeros(0)
    else:
        vza = np.atleast_1d(np.asarray(view_zenith_deg, dtype=float)).ravel()
        vaz_in = np.asarray(view_azimuth_deg, dtype=float)
        if vza.size == 1 and vaz_in.size > 1:
            vza = np.full(vaz_in.size, vza[0])
        vaz = np.broadcast_to(vaz_in, vza.shape).ravel()
        n_view = vza.size

    # -- layer optics --------------------------------------------------------
    props = atmo.layer_properties()
    beta_ext = np.maximum(props["beta_ext"], 0.0)
    beta_sca = np.maximum(props["beta_sca"], 0.0)
    dz = np.diff(z_bounds) * 1000.0
    dtau = beta_ext * dz[:, None]                       # (nz, n_wl)
    safe_ext = np.maximum(beta_ext, 1e-300)
    omega = np.where(beta_ext > 1e-300, beta_sca / safe_ext, 0.0)
    f_sun = spectra.solar_irradiance(wl)                # (n_wl,)

    mu0 = float(np.cos(np.radians(float(sun_zenith_deg))))
    mu_s = -mu0                                          # propagation cosine

    # -- pseudo-spherical direct beam ---------------------------------------
    t_bounds = _sun_transmittance_boundaries(atmo, sun_zenith_deg)  # (nz+1,n_wl)
    # Physical beam radiance at the layer mid-points.  The direct beam is a
    # delta function in angle with amplitude F_sun * T; this is the source
    # that enters the scattering integral.  (Normalising by the vertical
    # transmittance difference instead would close the column energy budget
    # exactly, but it over-weights grazing twilight rays by a large factor,
    # so the physical amplitude is used.)
    t_mid = 0.5 * (t_bounds[1:] + t_bounds[:-1])
    b_beam = f_sun[None, :] * t_mid
    # direct irradiance incident on the ground
    e_direct = f_sun * max(-mu_s, 0.0) * t_bounds[0]
    incident = f_sun * max(-mu_s, 0.0)
    direct_down = e_direct.copy()

    zeros_flux = np.zeros(n_wl)
    empty_rad = np.zeros((n_view, n_wl))

    if not np.any(beta_sca > 0.0):
        # no scattering at all: multiple scattering is identically zero
        return SOSResult(
            radiance=empty_rad,
            radiance_single=empty_rad.copy() if single_view else None,
            flux_up_toa=zeros_flux.copy(),
            flux_down_boa=zeros_flux.copy(),
            flux_up_boa=zeros_flux.copy(),
            direct_down_boa=direct_down,
            incident_flux=incident,
            orders=0,
            converged=True,
        )

    # -- angular discretisation ---------------------------------------------
    x, w = _gauss_nodes(n_mu)
    # ensure moments are integrable exactly by the Gauss rule
    n_moments = int(min(max(int(n_moments), 1), 2 * x.size))
    n_modes = int(min(max(int(n_modes), 1), n_moments))
    l_max = n_moments - 1
    m_max = n_modes - 1

    n_down = int(np.count_nonzero(x < 0.0))
    n_up = x.size - n_down
    x_down = x[:n_down]
    x_up = x[n_down:]
    w_down = w[:n_down]
    w_up = w[n_down:]

    # Legendre moments of the layer-averaged phase function
    beta = np.asarray(atmo.phase_moments(n_moments), dtype=float)  # (nz,L,n_wl)
    beta = beta[:, :n_moments, :]

    # associated Legendre functions at the Gauss nodes and at mu_s
    p_nodes = assoc_legendre_p_all(l_max, m_max, x)      # (1,L+1,2M+1,n_mu)
    p_sun = assoc_legendre_p_all(l_max, m_max, float(mu_s))
    if p_sun.ndim == 3:                                  # scalar input
        p_sun = p_sun[:, :, :, None]

    # Fourier weights W_l^m = (2l+1) beta_l (l-m)!/(l+m)!
    lfac = (2.0 * np.arange(l_max + 1) + 1.0)
    w_modes = []
    q_nodes = []
    q_sun = []
    for m in range(n_modes):
        c_lm = _log_factorial_ratio(l_max, m)
        wm = beta * (lfac * c_lm)[None, :, None]         # (nz,L,n_wl)
        w_modes.append(wm)
        q_nodes.append(p_nodes[0, :, m, :])              # (L,n_mu)
        q_sun.append(p_sun[0, :, m, 0])                  # (L,)

    # -- transmission factors (wavelength dependent) -------------------------
    trans_down = np.exp(-dtau[:, None, :] / (-x_down)[None, :, None])
    trans_up = np.exp(-dtau[:, None, :] / x_up[None, :, None])
    half_down = np.sqrt(trans_down)
    half_up = np.sqrt(trans_up)
    flux_w_down = w_down * (-x_down)                     # (n_down,)
    flux_w_up = w_up * x_up

    # -- first-order (direct beam) source ------------------------------------
    # S_dir,m = (omega/4pi) * B * (2-delta_m0) * sum_l W_l^m P_l^m(mu)P_l^m(mu_s)
    j_direct = np.zeros((nz, n_modes, x.size, n_wl))
    for m in range(n_modes):
        a = w_modes[m] * q_sun[m][None, :, None]
        jm = np.einsum("klw,li->kiw", a, q_nodes[m])     # (nz,n_mu,n_wl)
        factor = omega * b_beam * (2.0 if m > 0 else 1.0) / (4.0 * np.pi)
        j_direct[:, m] = jm * factor[:, None, :]

    # -- iterate successive orders ------------------------------------------
    field_tot = np.zeros((nz, n_modes, x.size, n_wl))
    field_prev = np.zeros_like(field_tot)
    flux_up_toa = np.zeros(n_wl)
    flux_down_boa = np.zeros(n_wl)
    flux_up_boa = np.zeros(n_wl)

    e_down_prev = np.zeros(n_wl)
    j_order = j_direct
    orders = 0
    converged = False

    for order in range(1, int(max_orders) + 1):
        orders = order

        # ---- scattering source of the requested order ---------------------
        if order > 1:
            j_order = _scatter_source(field_prev, w_modes, q_nodes, w, omega)

        # ---- downward sweep (TOA -> ground) -------------------------------
        l_down = np.zeros((n_modes, n_down, n_wl))
        l_down_mid = np.zeros((nz, n_modes, n_down, n_wl))
        jd = j_order[:, :, :n_down, :]
        for k in range(nz - 1, -1, -1):
            e = trans_down[k][None, :, :]
            eh = half_down[k][None, :, :]
            jk = jd[k]
            l_down_mid[k] = l_down * eh + jk * (1.0 - eh)
            l_down = l_down * e + jk * (1.0 - e)

        e_down = 2.0 * np.pi * np.einsum("i,iw->w", flux_w_down,
                                         l_down[0])
        flux_down_boa += e_down

        # ---- upward sweep (ground -> TOA) ---------------------------------
        l_ground = np.zeros((n_modes, n_up, n_wl))
        # only the azimuthally symmetric mode carries a flux; a Lambertian
        # surface reflects the total (direct + diffuse) downward irradiance.
        refl = (atmo.ground_albedo / np.pi)[None, :] * (
            e_down_prev + (e_direct if order == 1 else 0.0)
        )[None, :]
        l_ground[0] = refl
        flux_up_boa += np.pi * l_ground[0, 0]

        l_up = l_ground
        l_up_mid = np.zeros((nz, n_modes, n_up, n_wl))
        ju = j_order[:, :, n_down:, :]
        for k in range(nz):
            e = trans_up[k][None, :, :]
            eh = half_up[k][None, :, :]
            jk = ju[k]
            l_up_mid[k] = l_up * eh + jk * (1.0 - eh)
            l_up = l_up * e + jk * (1.0 - e)

        flux_up_toa += 2.0 * np.pi * np.einsum("i,iw->w", flux_w_up, l_up[0])

        # ---- accumulate and test convergence ------------------------------
        field_prev[:, :, :n_down, :] = l_down_mid
        field_prev[:, :, n_down:, :] = l_up_mid
        field_tot[:, :, :n_down, :] += l_down_mid
        field_tot[:, :, n_down:, :] += l_up_mid
        e_down_prev = e_down

        if order > 1:
            num = (np.sum(l_down_mid ** 2, axis=(0, 1, 2))
                   + np.sum(l_up_mid ** 2, axis=(0, 1, 2)))
            den = np.sum(field_tot ** 2, axis=(0, 1, 2))
            rel = np.sqrt(num / np.maximum(den, 1e-300))
            if float(np.max(rel)) < tol:
                converged = True
                break

    # -- evaluate the view directions ---------------------------------------
    radiance = np.zeros((n_view, n_wl))
    radiance_single = np.zeros((n_view, n_wl)) if single_view else None

    if n_view > 0:
        # Legendre transform of the converged total field (per mode)
        g_tot = []
        for m in range(n_modes):
            lm = field_tot[:, m, :, :]                   # (nz,n_mu,n_wl)
            g = np.einsum("li,kiw->klw", q_nodes[m] * w[None, :], lm)
            g_tot.append(g)

        dphi = np.radians(vaz - float(sun_azimuth_deg))

        for start in range(0, n_view, max(1, int(chunk))):
            sl = slice(start, min(start + max(1, int(chunk)), n_view))
            mu_v = -np.cos(np.radians(vza[sl]))          # propagation cosine
            mu_v = np.minimum(mu_v, -1e-6)               # horizon clamp
            absmu = -mu_v
            nc = mu_v.size

            # transmittances along the view ray (from the ground upwards)
            win = np.empty((nz, nc, n_wl))
            tr = np.ones((nc, n_wl))
            for k in range(nz):
                e = np.exp(-dtau[k][None, :] / absmu[:, None])
                win[k] = (1.0 - e) * tr
                tr = tr * e

            p_view = assoc_legendre_p_all(l_max, m_max, mu_v)
            if p_view.ndim == 3:
                p_view = p_view[:, :, :, None]

            j_ms = np.zeros((nz, nc, n_wl))
            j_ss = np.zeros((nz, nc, n_wl))
            for m in range(n_modes):
                qv = p_view[0, :, m, :]                  # (L,nc)
                cos_m = np.cos(m * dphi[sl])              # (nc,)
                # multiple scattering: scatter the total diffuse field
                a = (w_modes[m] * g_tot[m]).transpose(0, 2, 1).reshape(
                    nz * n_wl, l_max + 1)
                jv = (a @ qv).reshape(nz, n_wl, nc)
                jv *= (0.5 * omega)[:, :, None]
                jv *= cos_m[None, None, :]
                j_ms += jv.transpose(0, 2, 1)

                if single_view:
                    a0 = (w_modes[m] * q_sun[m][None, :, None]).transpose(
                        0, 2, 1).reshape(nz * n_wl, l_max + 1)
                    jv0 = (a0 @ qv).reshape(nz, n_wl, nc)
                    jv0 *= (omega * b_beam
                            * (2.0 if m > 0 else 1.0) / (4.0 * np.pi)
                            )[:, :, None]
                    jv0 *= cos_m[None, None, :]
                    j_ss += jv0.transpose(0, 2, 1)

            radiance[sl] = np.einsum("kvw,kvw->vw", win, j_ms)
            if single_view:
                radiance_single[sl] = np.einsum("kvw,kvw->vw", win, j_ss)

    return SOSResult(
        radiance=radiance,
        radiance_single=radiance_single,
        flux_up_toa=flux_up_toa,
        flux_down_boa=flux_down_boa,
        flux_up_boa=flux_up_boa,
        direct_down_boa=direct_down,
        incident_flux=incident,
        orders=orders,
        converged=converged,
    )


def _scatter_source(field, w_modes, q_nodes, weights, omega):
    """Scattering source of a diffuse field, all Fourier modes.

    ``field`` has shape ``(nz, n_modes, n_mu, n_wl)`` and is given at layer
    mid-points.  Returns the source at the same points, same shape.
    """
    nz, n_modes, n_mu, n_wl = field.shape
    l_max1 = q_nodes[0].shape[0]
    out = np.zeros_like(field)
    for m in range(n_modes):
        lm = field[:, m, :, :]                           # (nz,n_mu,n_wl)
        g = np.einsum("li,kiw->klw", q_nodes[m] * weights[None, :], lm)
        wg = (w_modes[m] * g).transpose(0, 2, 1).reshape(
            nz * n_wl, l_max1)
        jm = (wg @ q_nodes[m]).reshape(nz, n_wl, n_mu)
        out[:, m] = (jm * (0.5 * omega)[:, :, None]).transpose(0, 2, 1)
    return out


def multiple_scattering(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                        sun_azimuth_deg=0.0, **options) -> np.ndarray:
    """Radiance from scattering orders >= 2, shape ``(n_view, n_wl)``.

    Parameters
    ----------
    atmo : skycolor.atmosphere.Atmosphere
    sun_zenith_deg : float
    view_zenith_deg, view_azimuth_deg : array_like
    sun_azimuth_deg : float
    **options
        ``n_mu`` (Gauss nodes, default 32), ``n_modes`` (Fourier modes,
        default 12), ``n_moments`` (Legendre moments, default 32),
        ``max_orders`` (default 40), ``tol`` (default 1e-3) and ``chunk``
        (view chunk size, default 256).

    Returns
    -------
    ndarray ``(n_view, n_wl)`` in W m^-2 sr^-1 nm^-1.
    """
    res = solve(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                sun_azimuth_deg=sun_azimuth_deg, **options)
    return res.radiance
