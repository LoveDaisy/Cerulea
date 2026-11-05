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
* The **direct beam** is the unified finite-disk solar source of
  :mod:`skycolor.solar_source`, projected per disk beam: at each layer the
  beam keeps its own apparent elevation, continuous-column transmittance,
  occultation and local gain, and the Fourier source sums the beams with
  their disk weights and azimuth factors ``cos(m*phi_b)``.  The solar
  transmittance is always the raw physical extinction -- delta-M scales only
  the diffuse moments, never the sun.  (The alternative normalisation by the
  transmittance *difference* between layer boundaries would close the column
  energy budget exactly, but it over-weights grazing twilight rays by a
  large factor, so the physical amplitude is kept; see
  ``docs/physics/MULTIPLE_SCATTERING.md``.)
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

#: standing approximations of this backend
CAPABILITIES = {
    "scattering": "orders >= 2, plane-parallel discrete ordinates (delta-M)",
    "domain": "ground fixed at z_bounds[0]; atmosphere.elevation_km is not read",
    "horizontal_field": "not applied",
    "solar_source": "unified finite disk, per-beam apparent direction and "
                    "amplitude projected onto the Fourier source; raw-extinction "
                    "sun T with delta-M reserved for the diffuse moments",
    "refraction": "solar side fully refracted; plane-parallel diffuse "
                  "transport stays straight (retained approximation)",
}

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


def _direct_beam_tables(atmo, source, n_modes: int):
    """Per-beam direct-beam tables on the pseudo-spherical column.

    Returns ``(t_mid, gain_mid, mu_s, app_mid, t_ground, gain_ground,
    app_ground, cos_m, weights)`` where ``t_mid``/``gain_mid`` have shape
    ``(nz, n_beam, n_wl)`` (layer mid-points), ``mu_s``/``app_mid`` shape
    ``(nz, n_beam)`` and the ground triplets ``(n_beam, n_wl)`` /
    ``(n_beam,)``.  ``mu_s`` is the beam's *apparent* propagation cosine at
    the layer (negative above the horizon) and ``cos_m[m, b]`` the disk
    azimuth factor ``cos(m * phi_b)`` used by the Fourier projection.
    """
    medium = atmo._medium
    rb = atmo.r_bounds
    z_b = (rb - R_EARTH) / 1e3
    z_mid = 0.5 * (z_b[1:] + z_b[:-1])
    directions, weights = source.disk()
    n_beam = directions.shape[0]
    up = np.array([0.0, 0.0, 1.0])
    # each beam's true elevation is constant along the column (the local up
    # of the pseudo-spherical column is z-hat everywhere)
    true_e = np.degrees(np.arcsin(np.clip(directions @ up, -1.0, 1.0)))
    zq = np.concatenate([z_mid, z_b[:1]])                     # mids + ground
    heights = np.broadcast_to(zq[:, None], (zq.size, n_beam))
    true_grid = np.broadcast_to(true_e[None, :], heights.shape)
    app, t_all, gain_all, _lit = source.vertex(heights, true_grid)
    t_mid, gain_mid, app_mid = t_all[:-1], gain_all[:-1], app[:-1]
    t_ground, gain_ground, app_ground = t_all[-1], gain_all[-1], app[-1]
    mu_s = -np.sin(np.radians(app_mid))                       # propagation cosine
    # Fourier azimuth factors: each beam's LOCAL azimuth offset from the sun
    # meridian (the whole disk spans only ~16 arcmin / sin(zenith) of local
    # azimuth, so cos(m*d_az) ~ 1 for every mode -- the disk is azimuthally
    # delta-like even though it wraps the full circle *around the sun*)
    a_rad = np.radians(float(source.sun_azimuth_deg))
    m_hat = np.array([np.sin(a_rad), np.cos(a_rad), 0.0])
    p_hat = np.array([np.cos(a_rad), -np.sin(a_rad), 0.0])
    d_h = directions - np.outer(directions @ up, up)
    az_rel = np.arctan2(d_h @ p_hat, d_h @ m_hat)
    # sized by the effective n_modes (not a hardcoded cap): solve() accepts
    # any n_modes its quadrature supports, and the projection indexes one row
    # per mode
    cos_m = np.cos(np.outer(np.arange(n_modes), az_rel))
    return (t_mid, gain_mid, mu_s, app_mid, t_ground, gain_ground, app_ground,
            cos_m, weights)


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
    delta_m: bool = True,
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
    # delta-M: the truncated Legendre expansion loses the forward peak, which
    # for dust and cloud droplets is ~37 % of the scattered energy.  Scaling
    # the optical depth and the moments is the standard cure (Wiscombe 1977);
    # without it the multiple-scattering field of a dusty sky is wrong.
    props = atmo.delta_m_properties(n_moments) if delta_m else atmo.layer_properties()
    beta_ext = np.maximum(props["beta_ext"], 0.0)
    beta_sca = np.maximum(props["beta_sca"], 0.0)
    dz = np.diff(z_bounds) * 1000.0
    dtau = beta_ext * dz[:, None]                       # (nz, n_wl)
    safe_ext = np.maximum(beta_ext, 1e-300)
    omega = np.where(beta_ext > 1e-300, beta_sca / safe_ext, 0.0)
    f_sun = spectra.solar_irradiance(wl)                # (n_wl,)

    # -- unified finite-disk direct beam, per disk beam ---------------------
    from ..solar_source import SolarSource

    source = SolarSource(atmo, float(sun_zenith_deg), float(sun_azimuth_deg))
    (t_mid_b, gain_mid_b, mu_s_b, app_mid_b, t_ground_b, gain_ground_b,
     app_ground_b, cos_m_all, beam_w) = _direct_beam_tables(atmo, source, n_modes)
    n_beam = beam_w.size
    # physical amplitude: F_sun * T * G per beam at the layer mid-points.
    # (Normalising by the vertical transmittance difference instead would
    # close the column energy budget exactly, but it over-weights grazing
    # twilight rays by a large factor, so the physical amplitude is used.)
    b_beam = f_sun[None, None, :] * t_mid_b * gain_mid_b[..., None]  # (nz,nb,n_wl)
    # direct irradiance incident on the ground: plain local per-beam sum at
    # the domain bottom (z_bounds[0]) — the same reference as the layer
    # sources, independent of the facade's elevation_km (this backend
    # declares ground = z_bounds[0] and does not read elevation_km)
    z0 = float(z_bounds[0])
    directions_g, _ = source.disk()
    true_e_g = np.degrees(np.arcsin(np.clip(directions_g @ np.array([0.0, 0.0, 1.0]),
                                            -1.0, 1.0)))
    app_g, t_g, gain_g, lit_g = source.vertex(np.full(len(directions_g), z0),
                                              true_e_g)
    e_direct = f_sun * np.sum(
        beam_w[:, None] * lit_g[:, None] * t_g * gain_g[:, None]
        * np.maximum(np.sin(np.radians(app_g)), 0.0)[:, None], axis=0)
    up_hat = np.array([0.0, 0.0, 1.0])
    directions, _ = source.disk()
    incident = f_sun * np.sum(beam_w * np.maximum(directions @ up_hat, 0.0))
    direct_down = e_direct.copy()
    mu_s = -float(np.cos(np.radians(float(sun_zenith_deg))))  # view-frame mu0 (diagnostics)

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
    if delta_m:
        beta = np.asarray(props["moments"], dtype=float)
    else:
        beta = np.asarray(atmo.phase_moments(n_moments), dtype=float)
    beta = beta[:, :n_moments, :]

    # associated Legendre functions at the Gauss nodes and at each beam's
    # apparent propagation cosine per layer (scipy appends the x axis last
    # for array input)
    p_nodes = assoc_legendre_p_all(l_max, m_max, x)      # (1,L+1,2M+1,n_mu)
    mu_flat = np.ascontiguousarray(mu_s_b.T.reshape(-1))  # (nb*nz,), b-major
    p_beams = assoc_legendre_p_all(l_max, m_max, mu_flat)[0]   # (L+1,2M+1,nb*nz)
    p_beams = p_beams[:, :n_modes, :].reshape(l_max + 1, n_modes, n_beam, nz)

    # Fourier weights W_l^m = (2l+1) beta_l (l-m)!/(l+m)!
    lfac = (2.0 * np.arange(l_max + 1) + 1.0)
    w_modes = []
    q_nodes = []
    for m in range(n_modes):
        c_lm = _log_factorial_ratio(l_max, m)
        wm = beta * (lfac * c_lm)[None, :, None]         # (nz,L,n_wl)
        w_modes.append(wm)
        q_nodes.append(p_nodes[0, :, m, :])              # (L,n_mu)

    # per-beam Fourier projection of the direct source, one mode at a time:
    #   C_m[k,l,w] = sum_b beam_w·T·G · P_l^m(mu_s[b,k]) · cos(m*phi_b)
    # kept (nz,L,n_wl) transient so nothing materialises the full
    # (nz,L,M,n_wl) tensor
    # Plain local radiance bookkeeping: this backend's diffuse paths are the
    # retained straight-transport approximation, along which plain radiance
    # (not L/n^2) is the invariant, so layer sources consume G_local directly
    # and no reference-face conversion is applied anywhere — fluxes are
    # local/BOA/TOA quantities as swept.  The single/MC kernels carry the
    # audit's detector transfer (n_obs/n_pt)^2 instead; the ~1e-4 convention
    # difference between the two is part of the declared SOS-vs-SS
    # propagation error (plan 6.3) and is documented in
    # docs/physics/SOLAR_SOURCE.md.
    beam_wg = beam_w[None, :, None] * t_mid_b * gain_mid_b[..., None]  # (nz, nb, n_wl)
    source_factor = omega * f_sun[None, :]                     # (nz, n_wl)

    def _direct_coeff(m):
        pm = p_beams[:, m, :, :]                               # (L+1, nb, nz)
        return np.einsum("kbw,lbk->klw", beam_wg,
                         pm * cos_m_all[m][None, :, None])

    # -- transmission factors (wavelength dependent) -------------------------
    trans_down = np.exp(-dtau[:, None, :] / (-x_down)[None, :, None])
    trans_up = np.exp(-dtau[:, None, :] / x_up[None, :, None])
    half_down = np.sqrt(trans_down)
    half_up = np.sqrt(trans_up)
    flux_w_down = w_down * (-x_down)                     # (n_down,)
    flux_w_up = w_up * x_up

    # -- first-order (direct beam) source ------------------------------------
    # S_dir,m = (omega F /4pi)(2-delta_m0) sum_l W_l^m C_m[k,l,w] P_l^m(mu)
    j_direct = np.zeros((nz, n_modes, x.size, n_wl))
    for m in range(n_modes):
        a = w_modes[m] * _direct_coeff(m)
        jm = np.einsum("klw,li->kiw", a, q_nodes[m])     # (nz,n_mu,n_wl)
        factor = source_factor * (2.0 if m > 0 else 1.0) / (4.0 * np.pi)
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
                    a0 = (w_modes[m] * _direct_coeff(m)).transpose(
                        0, 2, 1).reshape(nz * n_wl, l_max + 1)
                    jv0 = (a0 @ qv).reshape(nz, n_wl, nc)
                    jv0 *= (source_factor
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
