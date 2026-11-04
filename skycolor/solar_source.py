"""The production solar source: one authoritative definition of the sun.

Every radiative-transfer backend (single scattering, SOS, the Monte Carlo
engines, the cloud deck and the direct-illumination paths) must consume the
same solar physics: the *apparent* direction of each beam of the finite
solar disk, the *continuous* species columns along the refracted chord, the
ground occultation of that chord, and the angular-domain gain that converts
true-solid-angle disk integrals into local flux.  This module is the single
authority for all of them; unifying the source does *not* unify the diffuse
transport approximation each backend keeps (plane-parallel SOS, straight
Monte Carlo flights, two-stream clouds -- those remain and are documented
where they live).

Layering (each layer consumes only the ones above it):

1. :class:`SolarRayField` -- the refractive index profile ``n(r)`` of one
   density definition and the radial integration machinery: forward
   apparent->true mapping, its *analytic* same-machinery derivative (the
   two share quadrature nodes and spline evaluations), grazing/occultation
   thresholds and turning-point geometry.  Monotonic ``n*r`` only; ducting
   media raise ``ValueError`` rather than silently producing multi-image
   geometry.
2. :class:`SolarSpeciesColumns` -- continuous air/O3/NO2/aerosol columns
   along the same rays (no shell-chord or shell-average approximation).
   Wavelength independent: extinction is applied by the caller from the
   caller's actual cross sections, so wavelength slices never change the
   geometry.
3. :class:`PointSourceTable` -- a shadow-aligned PCHIP table of (direction,
   columns) versus height, refined until every interval carries a certificate
   for angle, slope and optical depth.  The slope certificate is a knot
   *placement* criterion (0.5% interpolant/slope consistency, well inside
   the 1% heldout transmittance-gain budget); actual gain accuracy is
   measured against independent references by the validation suites.  The
   beam gain is the derivative of the *same* direction interpolant -- a
   second interpolated gain field is never built.  Built lazily, cached per
   medium-construction token.
4. :class:`SolarGeometry` -- the token-cached bundle of 1-3 for one medium
   construction and refraction mode.  ``refraction=False`` is the same
   contract at ``n = 1`` (identity mapping, geometric shadow, straight
   columns), not a degraded path.
5. :class:`SolarSource` -- binds a geometry to an observing scenario and one
   true solar direction, adds the finite disk (radius 16' by default) and
   exposes the consumer kernels: per-vertex attenuation/direction/gain
   queries, the scattering kernel, the ground-incidence kernel and pure
   central-beam attenuation for the legacy transmittance APIs.

Physical bookkeeping (single source of truth for the ``n^2`` factors):

* ``G_local = (n_point/n_inf)^2 * dOmega_app/dOmega_true`` concentrates the
  true-solid-angle disk irradiance ``F_inf`` (defined on the beam-centre
  normal plane) into local flux; per beam the disk weights satisfy
  ``sum(w * cos(rho)) = 1`` under that normalisation, and partial
  occultation removes beams without renormalising the rest.
* Transport along the retained straight diffuse paths conserves the reduced
  radiance ``L/n^2``.  Scattering and ground reflection happen at one local
  ``n``, so the source-to-detector conversion collapses to the single factor
  ``(n_observer/n_inf)^2 * dOmega_app/dOmega_true`` already contained in
  ``G_local * (n_observer/n_point)^2``; kernels apply it once and callers
  never multiply a second ``n^2``.  A constant-index (``n`` possibly != 1)
  medium recovers ``G = 1`` exactly.
* Transmittance ``T = exp(-tau)`` is always the pure attenuation of the
  physical columns and is reported separately from gain; nothing here
  renames ``T*G`` or clips it to 1.
"""
from __future__ import annotations

import time
import weakref
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from . import spectra
from .constants import N_STD_AIR, R_EARTH
from .optics import species_extinction_weights
from .refraction import REFRACTIVITY_STD

#: tabulation top of the refractive profile [m]; above it n is n_inf
_PROFILE_TOP_M = 1.2e6
#: profile tabulation resolution inherited from RefractiveProfile
_PROFILE_POINTS = 4000

#: integration segment scheme [km] shared by the field and column integrals:
#: 1 km through the atmosphere (fine enough that the Gauss quadrature of the
#: smooth density profiles is converged; validated against the independent
#: Cartesian RK chain), 5 km in the negligible upper mesosphere.
_SEGMENT_KM_MID = 1.0
_SEGMENT_MID_TOP_KM = 120.0
_SEGMENT_KM_HIGH = 5.0

#: angular radius of the solar disk [deg]
SUN_RADIUS_DEG = 16.0 / 60.0

#: half-width of the horizon-kink band [deg] inside which the slope
#: certificate is exempt (see ``PointSourceTable._build_row``)
_HORIZON_KINK_GUARD_DEG = 0.05


# ---------------------------------------------------------------------------
# 1. radial field: n(r) and the ray integrals
# ---------------------------------------------------------------------------
class SolarRayField:
    """Refractive index field and radial solar-ray integrals for one medium.

    The field tabulates ``n(r)`` from the medium's air density
    (Gladstone-Dale) or is identically ``n_inf`` for the ``refraction=False``
    contract.  Rays are integrated in radius with Bouguer's invariant
    ``a = n(r0) r0 cos(e_app)``; the integrand singularity at a turning point
    is removed by the ``r = r_lower + u^2`` substitution and the
    ``q(r) - a`` cancellation near a tangent is handled by a local Taylor
    difference of ``q`` (the same stable scheme the experimental audit
    validated).  This preserves the two-branch physics of the M10 chord
    repair: a ray with the source below the local horizontal traverses
    turning point -> space *and* turning point -> point, i.e. ``up + 2*down``
    of optical depth, now with continuous densities instead of shell chords.

    ``with_slope`` integrals evaluate ``d(bending)/d(apparent)`` analytically
    on the same quadrature nodes,

    ``dI/da = (-n'/n) q^2 / (q^2 - a^2)^{3/2},  da/d(app) = -q0 sin(app)``,

    so the beam-gain certificates never depend on finite-difference noise.
    """

    # -- construction helpers --------------------------------------------------
    @staticmethod
    def _tabulation_levels_km(medium) -> np.ndarray:
        """Radii where n(r) is tabulated, in km above sea level.

        These levels are *also* the integration segment boundaries, so no
        quadrature segment ever crosses a spline knot: the curvature and
        slope integrands involve n'' (a jump at every interior knot), and
        integrating across one is exactly what made the knot-sensitive
        slopes of the experimental audit non-convergent.  Spacing follows
        the atmosphere: 1 km through the troposphere and stratosphere (knot
        alignment, not segment size, is what protects the slope integrand --
        the analytic derivative is exact per segment), 5 km in the upper
        mesosphere, 50 km in the exosphere where n is indistinguishable from
        n_inf.
        """
        ground_km = float(medium.z_bounds[0])
        profile_top_km = _PROFILE_TOP_M / 1e3
        levels = np.unique(np.concatenate([
            np.arange(0.0, _SEGMENT_MID_TOP_KM + _SEGMENT_KM_MID, _SEGMENT_KM_MID),
            np.arange(_SEGMENT_MID_TOP_KM, 220.0 + _SEGMENT_KM_HIGH, _SEGMENT_KM_HIGH),
            np.arange(220.0, profile_top_km + 50.0, 50.0),
            np.asarray(medium.z_bounds, dtype=float),
            [11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852, 86.0],
            [ground_km, profile_top_km],
        ]))
        return levels[(levels >= 0.0) & (levels <= profile_top_km)]

    def __init__(self, medium, *, refracted: bool = True,
                 n_quad: int = 6, chunk: int = 512):
        self.refracted = bool(refracted)
        if n_quad < 2 or chunk < 1:
            raise ValueError("n_quad>=2 and chunk>=1 are required")
        self.n_quad = int(n_quad)
        self.chunk = int(chunk)
        self.r_ground = float(medium.r_bounds[0])
        self.r_optical_top = float(medium.r_bounds[-1])
        ground_km = float(medium.z_bounds[0])

        if refracted:
            levels = self._tabulation_levels_km(medium)
            r = R_EARTH + levels * 1e3
            n = 1.0 + REFRACTIVITY_STD * medium.n_air((r - R_EARTH) / 1e3) / N_STD_AIR
        else:
            levels = np.arange(0.0, _PROFILE_TOP_M / 1e3 + 50.0, 50.0)
            r = R_EARTH + levels * 1e3
            n = np.ones_like(r)
        if not np.all(np.diff(r) > 0) or not np.all(n > 0):
            raise ValueError("profile radii must increase and n must be positive")
        if not r[0] <= self.r_ground < r[-1]:
            raise ValueError("opaque ground must lie inside the refractive profile")
        self._n = CubicSpline(r, n)
        self._dn = self._n.derivative()
        self._d2n = self._n.derivative(2)
        self.r = r
        self.n = n
        self.r_top = float(r[-1])
        self.n_inf = float(n[-1])
        self.q_ground = float(self._n(self.r_ground) * self.r_ground)

        # n(r)*r must be strictly increasing: turning-point inversion and the
        # single-image geometry both require it.  q' = n + r n' is cubic per
        # spline interval, so its extrema are quadratic roots; testing those
        # plus the knots rejects ducts across the whole spline, not only at
        # tabulation nodes.
        c0, c1, c2, _ = self._n.c
        ri = r[:-1]
        qa = 12.0 * c0
        qb = 6.0 * (ri * c0 + c1)
        qc = 2.0 * (ri * c1 + c2)
        disc = qb * qb - 4.0 * qa * qc
        roots = []
        for sign in (-1.0, 1.0):
            root = np.divide(-qb + sign * np.sqrt(np.maximum(disc, 0.0)),
                             2.0 * qa, out=np.full_like(qa, np.nan), where=qa != 0.0)
            mask = (disc >= 0.0) & (root > 0.0) & (root < np.diff(r))
            roots.append(ri[mask] + root[mask])
        root = np.divide(-qc, qb, out=np.full_like(qb, np.nan),
                         where=(qa == 0.0) & (qb != 0.0))
        mask = (root > 0.0) & (root < np.diff(r))
        roots.append(ri[mask] + root[mask])
        check = np.concatenate([r, *roots])
        if np.any(self._n(check) + check * self._dn(check) <= 0):
            raise ValueError(
                "n(r)*r must increase: ducts/multiple turning points are "
                "unsupported; build the scene without such a profile")

        self._q_nodes = r * n
        if np.any(np.diff(self._q_nodes) <= 0):
            raise ValueError("profile n*r samples must increase")

        # Integration boundaries: the tabulation levels themselves (see
        # ``_tabulation_levels_km``), so segments and spline knots coincide.
        self._edges = r
        self._gx, self._gw = np.polynomial.legendre.leggauss(self.n_quad)

    # -- basic accessors ------------------------------------------------------
    def refractive_index(self, height_km):
        """n at height(s) [km]; no extrapolation outside the profile."""
        radius = R_EARTH + np.asarray(height_km, dtype=float) * 1e3
        if np.any(radius < self.r[0]) or np.any(radius > self.r_top):
            raise ValueError("height lies outside the refractive profile")
        return self._n(radius)

    def grazing_apparent_elevation(self, height_km):
        """Apparent elevation of the ground-grazing solar ray (occultation edge).

        The exact tangent ray has invariant ``q_ground``; evaluating a
        hair above it avoids strict-mask round-off at the boundary.
        """
        radius = R_EARTH + np.asarray(height_km, dtype=float) * 1e3
        if np.any(radius < self.r_ground) or np.any(radius > self.r_top):
            raise ValueError("height lies outside ground/profile-top bounds")
        ratio = self.q_ground / (self._n(radius) * radius)
        return -np.degrees(np.arccos(np.minimum(ratio, 1.0)))

    def _turning_radius(self, invariant):
        """Radius where ``n(r) r = invariant`` (root of a monotone function)."""
        r = np.interp(invariant, self._q_nodes, self.r)
        for _ in range(4):
            n = self._n(r)
            r -= (n * r - invariant) / (n + r * self._dn(r))
        bad = (r < self.r_ground) | (r > self.r_top) | (
            np.abs(self._n(r) * r - invariant) > 1e-7)
        for i in np.flatnonzero(bad):
            j = int(np.searchsorted(self._q_nodes, invariant[i]))
            r[i] = brentq(lambda x: float(self._n(x) * x - invariant[i]),
                          self.r[j - 1], self.r[j], xtol=1e-8)
        return r

    # -- the integrals --------------------------------------------------------
    def _integrate(self, lower, upper, invariant, initial_gap, with_slope,
                   moving=None):
        """Curvature integrals over one leg; ``r = lower + u^2`` at tangents.

        Returns ``(bending, slope_integral)`` where ``slope_integral`` is
        ``integral(dI/da) dr`` with ``I`` the bending integrand
        ``a (-n'/n) / sqrt(q^2 - a^2)``.  ``slope_mode="none"`` skips the
        slope integral; otherwise ``moving`` (a per-ray bool array, defaults
        to all-true) marks rays whose lower limit is the a-dependent turning
        point.  That case is subtle: the lower limit moves with ``a``
        (``dr_t/da = 1/q'(r_t)``), so differentiating at fixed ``q`` is
        wrong -- the naive ``q^2 (q^2-a^2)^{-3/2}`` form diverges like
        ``u^-2`` where the true total derivative stays finite (the ``u^2``
        substitution absorbs the moving boundary, so ``q(r_t(a)+u^2) - a``
        has no ``O(da)`` term).  See the "moving" branch below.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.broadcast_to(np.asarray(upper, dtype=float), lower.shape)
        nseg = self._edges.size - 1
        # Tangent-band fission: every ray whose leg starts at its own turning
        # point gets that first (variable-width) segment split into four
        # u^2-graded pieces, so the resolution of the near-tangent region does
        # not jump when a turning radius crosses a segment boundary.  Without
        # this the analytic slope carries tiny app-discontinuities there and
        # the table's slope certificate can never converge through one.
        # Fixed-radius legs simply leave the extra slots zero-width.
        lo = np.empty((lower.size, nseg + 4))
        hi = np.empty_like(lo)
        lo[:, 4:] = np.maximum(self._edges[:-1][None, :], lower[:, None])
        hi[:, 4:] = np.minimum(self._edges[1:][None, :], upper[:, None])
        moving = np.asarray(moving, dtype=bool) if moving is not None else \
            np.zeros(lower.size, dtype=bool)
        if moving.any():
            e1 = self._edges[np.clip(np.searchsorted(self._edges, lower), 1, nseg)]
            band_hi = np.minimum(e1, np.broadcast_to(upper, lower.shape))
            frac_lo = np.array([0.0, 1.0, 4.0, 9.0]) / 16.0
            frac_hi = np.array([1.0, 4.0, 9.0, 16.0]) / 16.0
            band = (band_hi - lower)[:, None]
            lo[:, :4] = np.where(moving[:, None],
                                 lower[:, None] + band * frac_lo[None, :], lower[:, None])
            hi[:, :4] = np.where(moving[:, None],
                                 lower[:, None] + band * frac_hi[None, :], lower[:, None])
            # the global segment containing the tangent band is replaced by
            # the fission pieces: clip the global grid to start at band_hi
            lo[:, 4:] = np.maximum(lo[:, 4:],
                                   np.where(moving[:, None], band_hi[:, None], lower[:, None]))
        else:
            lo[:, :4] = lower[:, None]
            hi[:, :4] = lower[:, None]
        hi = np.maximum(hi, lo)
        ulo = np.sqrt(np.maximum(lo - lower[:, None], 0.0))
        uhi = np.sqrt(np.maximum(hi - lower[:, None], 0.0))
        half = 0.5 * (uhi - ulo)
        # Segments clipped away entirely (hi <= lo) leave zero-width slots
        # whose nodes sit exactly on the tangent, where q - a == 0 to the
        # last bit; they carry no integral weight but would poison the
        # 3/2-power slope integrand with 0 * inf.  Mask them out.
        alive = (half > 0.0)[..., None]
        u = 0.5 * (uhi + ulo)[..., None] + half[..., None] * self._gx
        delta = u * u
        r = lower[:, None, None] + delta
        n = self._n(r)
        dn = self._dn(r)
        q = n * r
        qdiff = q - invariant[:, None, None]
        # Direct subtraction of two ~6e6 m invariants loses relative accuracy
        # in the first node near a tangent; the Taylor form removes that.
        nb = self._n(lower)
        dnb = self._dn(lower)
        q1 = nb + lower * dnb                    # q'(r_lower)
        q2 = 2.0 * dnb + lower * self._d2n(lower)
        stable = initial_gap[:, None, None] + delta * (
            q1[:, None, None] + 0.5 * delta * q2[:, None, None])
        qdiff = np.where(delta < 0.1, stable, qdiff)
        base = np.maximum(qdiff * (q + invariant[:, None, None]), 1e-250)
        inv_n = -dn / n
        two_u = 2.0 * u
        sqrt_base = np.sqrt(base)
        weight = self._gw * half[..., None]
        bending = np.sum(np.where(alive,
                                  invariant[:, None, None] * inv_n / sqrt_base * two_u,
                                  0.0) * weight, axis=(1, 2))
        if not with_slope:
            return bending, np.zeros_like(bending)
        invariant = invariant[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # "fixed" rays start at a given radius: r does not move with a and
            # the u-range ends at a fixed upper radius, so a plain partial
            # derivative at fixed r is the whole story.
            fixed_d = np.where(alive, inv_n * (q * q) / (base * sqrt_base), 0.0)
            fixed_sum = np.sum(fixed_d * two_u * weight, axis=(1, 2))
            if not np.any(moving):
                return bending, fixed_sum
            # "moving" rays start at their a-dependent turning point.  Nodes
            # and the u-domain both shift with r_t(a); differentiate at fixed
            # u and add the boundary term of the moving upper limit:
            #   d/da g|_u = 2u r_t' [a K/sqrt(base) - a inv_n q q'/base^{3/2}]
            #             + 2u inv_n q^2 / base^{3/2}
            #   boundary  = -r_t' a inv_n(r_upper) / sqrt(base(r_upper))
            # The two 1/u^2 pieces cancel at u -> 0 exactly (finite
            # angular Jacobian), so the discretised form is regular.
            qp = n + r * dn                       # q'(r) at the nodes
            d2n = self._d2n(r)
            big_k = (dn * dn - n * d2n) / (n * n)  # d(-n'/n)/dr
            irt = (1.0 / q1)[:, None, None]        # dr_t/da
            base32 = base * sqrt_base
            moving_d = np.where(alive,
                                invariant * irt * big_k / sqrt_base
                                - invariant * irt * inv_n * q * qp / base32
                                + inv_n * (q * q) / base32,
                                0.0)
            moving_sum = np.sum(moving_d * two_u * weight, axis=(1, 2))
            # boundary term at the fixed upper radius (moving rays only)
            n_up = self._n(upper)
            dn_up = self._dn(upper)
            inv_up = (-dn_up / n_up)
            base_up = np.maximum(
                (n_up * upper) ** 2 - invariant.reshape(-1) ** 2, 1e-250)
            moving_sum = moving_sum - (irt.reshape(-1) * invariant.reshape(-1)
                                       * inv_up / np.sqrt(base_up))
            slope_int = np.where(moving, moving_sum, fixed_sum)
        return bending, slope_int

    def forward_with_slope(self, height_km, apparent_elevation_deg):
        """True elevation and ``d(true)/d(apparent)`` per apparent ray.

        Rays intersecting the opaque ground return ``(nan, nan)``; blocked
        rays have no source.  The slope is exact (same-machinery analytic
        derivative), positive away from caustics.
        """
        height, app = np.broadcast_arrays(
            np.asarray(height_km, dtype=float),
            np.asarray(apparent_elevation_deg, dtype=float))
        r0 = R_EARTH + height.ravel() * 1e3
        ea = app.ravel()
        if np.any(~np.isfinite(r0)) or np.any(~np.isfinite(ea)):
            raise ValueError("height and apparent elevation must be finite")
        if np.any(r0 < self.r_ground - 1e-7) or np.any(r0 > self.r_top):
            raise ValueError("ray start lies outside ground/profile-top bounds")
        if np.any(np.abs(ea) > 90.0):
            raise ValueError("apparent elevation must lie in [-90, 90] degrees")
        q0 = self._n(r0) * r0
        invariant = q0 * np.cos(np.radians(ea))
        down = ea < 0.0
        valid = (~down) | (invariant > self.q_ground)
        out = np.full(ea.shape, np.nan)
        slope = np.full(ea.shape, np.nan)
        for start in range(0, ea.size, self.chunk):
            idx = np.arange(start, min(start + self.chunk, ea.size))
            idx = idx[valid[idx]]
            if not idx.size:
                continue
            d = down[idx]
            base = r0[idx].copy()
            gap = 2.0 * q0[idx] * np.sin(0.5 * np.radians(ea[idx])) ** 2
            if d.any():
                base[d] = self._turning_radius(invariant[idx[d]])
                gap[d] = 0.0
            # Up-going rays integrate from a fixed radius; down-going rays
            # start at their turning point, whose radius moves with the
            # invariant and therefore needs the total derivative.
            bending, slope_int = self._integrate(base, self.r_top, invariant[idx],
                                                 gap, True, d)
            if d.any():
                b2, s2 = self._integrate(base[d], r0[idx[d]], invariant[idx[d]],
                                         gap[d], True,
                                         np.ones(int(d.sum()), dtype=bool))
                bending[d] += b2
                slope_int[d] += s2
            out[idx] = ea[idx] - np.degrees(bending)
            # d(bending)/d(app) = da/d(app) * integral(dI/da) = -q0 sin(app) * S
            slope[idx] = 1.0 + q0[idx] * np.sin(np.radians(ea[idx])) * slope_int
        return out.reshape(height.shape), slope.reshape(height.shape)

    def forward(self, height_km, apparent_elevation_deg):
        """True elevation per apparent ray; NaN when the ray hits the ground."""
        return self.forward_with_slope(height_km, apparent_elevation_deg)[0]

    def apparent_of_true(self, height_km, true_elevation_deg):
        """Apparent elevation of a source at ``true_elevation_deg``.

        Root solve of the monotone forward map (the table is the fast path;
        this direct evaluation exists for central-beam diagnostics and as
        the reference the table is validated against).  Returns NaN for
        sources below the shadow boundary.
        """
        height, true = np.broadcast_arrays(
            np.asarray(height_km, dtype=float),
            np.asarray(true_elevation_deg, dtype=float))
        zz, et = height.ravel(), true.ravel()
        lower = self.grazing_apparent_elevation(zz)
        crit = self.forward(zz, lower + 1e-6)
        out = np.full(zz.shape, np.nan)
        lit = et >= crit - 1e-9
        if np.any(lit):
            hi = np.where(lit, 90.0, np.nan)
            for i in np.flatnonzero(lit):
                if et[i] >= 90.0:
                    out[i] = 90.0
                    continue
                lo = lower[i] + 1e-7
                out[i] = brentq(
                    lambda e: float(self.forward(zz[i], np.array([e]))[0] - et[i]),
                    lo, 90.0, xtol=2e-9)
        return out.reshape(height.shape)

    def turning_radius(self, invariant):
        """Public alias of the turning-point solver (radius where ``n(r) r`` equals
        the invariant).  Requires a monotone ``n*r`` medium (constructor-checked)."""
        return self._turning_radius(np.asarray(invariant, dtype=float))

    def beam_gain(self, height_km, apparent_elevation_deg):
        """Local point-source gain ``(n/n_inf)^2 dOmega_app/dOmega_true``.

        Exact same-machinery derivative; NaN where blocked or at a caustic
        (non-positive slope).  Not a complete refracted transport correction
        for diffuse paths -- those stay straight by the retained approximation.
        """
        true, slope = self.forward_with_slope(height_km, apparent_elevation_deg)
        height = np.broadcast_arrays(np.asarray(height_km, dtype=float),
                                     np.asarray(apparent_elevation_deg, dtype=float))[0]
        n = self._n(R_EARTH + np.asarray(height, dtype=float) * 1e3)
        app = np.asarray(apparent_elevation_deg, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = ((n / self.n_inf) ** 2
                    * np.cos(np.radians(app)) / np.cos(np.radians(true)) / slope)
        return np.where((slope > 0.0) & np.isfinite(slope), gain, np.nan)


# ---------------------------------------------------------------------------
# 2. continuous species columns
# ---------------------------------------------------------------------------
class SolarSpeciesColumns:
    """Continuous air/O3/NO2/aerosol columns along refracted solar rays.

    Integrates the actual density callables (no shell chords, no shell
    averages) with the same geometry machinery as :class:`SolarRayField`.
    Column order follows :mod:`skycolor.optics`: air, ozone, NO2, then one
    aerosol entry per layer (550 nm optical depth).  The ``HorizontalField``
    is deliberately *not* applied -- solar columns stay radial, matching the
    declared approximation of every backend that consumes them.

    The integration stops at the model top; the field above it is vacuum for
    extinction purposes (the geometric bending integral continues to the
    refractive profile top, which is the field's business, not this class's).
    """

    def __init__(self, medium, field: SolarRayField, *, n_quad: int = 8,
                 chunk: int = 256):
        if n_quad < 2 or chunk < 1:
            raise ValueError("n_quad>=2 and chunk>=1 are required")
        # weak on purpose: this object lives in the geometry cache, whose key
        # is the medium's own token -- a strong back-reference here would pin
        # the medium (and with it the cache entry) forever
        self._medium_ref = weakref.ref(medium)
        self.field = field
        self.n_quad = int(n_quad)
        self.chunk = int(chunk)
        self.r_ground = float(medium.r_bounds[0])
        self.r_top = float(medium.r_bounds[-1])
        if self.field.r_ground != self.r_ground or self.r_top > self.field.r_top:
            raise ValueError("field must share the ground and cover the optical model")
        self.n_species = 3 + len(medium.aerosols)
        ground_km = (self.r_ground - R_EARTH) / 1e3
        top_km = (self.r_top - R_EARTH) / 1e3
        # Density-formula boundaries are integration boundaries, not a
        # piecewise-constant approximation of the densities.
        levels = np.unique(np.concatenate([
            np.arange(ground_km, min(_SEGMENT_MID_TOP_KM, top_km) + _SEGMENT_KM_MID,
                      _SEGMENT_KM_MID),
            np.arange(_SEGMENT_MID_TOP_KM, top_km + _SEGMENT_KM_HIGH, _SEGMENT_KM_HIGH),
            np.asarray(medium.z_bounds, dtype=float),
            [11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852, 86.0],
            [ground_km, top_km],
        ]))
        levels = levels[(levels >= ground_km) & (levels <= top_km)]
        self._edges = R_EARTH + levels * 1e3
        self._gx, self._gw = np.polynomial.legendre.leggauss(self.n_quad)

    @property
    def medium(self):
        medium = self._medium_ref()
        if medium is None:
            raise RuntimeError(
                "the owning medium of this solar geometry has been collected")
        return medium

    def _density_columns(self, height):
        medium = self.medium
        yield medium.n_air(height)
        yield medium.n_o3(height)
        yield medium.n_no2(height)
        for layer in medium.aerosols:
            yield layer.aod550 * layer.profile(height) / 1000.0

    def columns(self, height_km, apparent_elevation_deg) -> np.ndarray:
        """Columns along forward rays, ``broadcast_shape + (n_species,)``.

        Blocked rays (intersecting the opaque ground) return NaN for every
        species; an unblocked ray pointing out at the top returns 0.
        """
        height, app = np.broadcast_arrays(
            np.asarray(height_km, dtype=float),
            np.asarray(apparent_elevation_deg, dtype=float))
        radius = R_EARTH + height.ravel() * 1e3
        ea = app.ravel()
        if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(ea)):
            raise ValueError("height and apparent elevation must be finite")
        if np.any(radius < self.r_ground - 1e-7) or np.any(radius > self.r_top + 1e-7):
            raise ValueError("ray start lies outside the optical model")
        if np.any(np.abs(ea) > 90.0):
            raise ValueError("apparent elevation must lie in [-90, 90] degrees")
        q0 = self.field._n(radius) * radius
        invariant = q0 * np.cos(np.radians(ea))
        down = ea < 0.0
        valid = (~down) | (invariant > self.field.q_ground)
        result = np.full((ea.size, self.n_species), np.nan)
        for start in range(0, ea.size, self.chunk):
            index = np.arange(start, min(start + self.chunk, ea.size))
            index = index[valid[index]]
            if not len(index):
                continue
            descending = down[index]
            lower = radius[index].copy()
            gap = 2.0 * q0[index] * np.sin(0.5 * np.radians(ea[index])) ** 2
            if descending.any():
                lower[descending] = self.field._turning_radius(
                    invariant[index[descending]])
                gap[descending] = 0.0
            value = self._integral(lower, self.r_top, invariant[index], gap)
            if descending.any():
                value[descending] += self._integral(
                    lower[descending], radius[index[descending]],
                    invariant[index[descending]], gap[descending])
            result[index] = value
        return result.reshape(height.shape + (self.n_species,))

    def _integral(self, lower, upper, invariant, initial_gap):
        lower = np.asarray(lower, dtype=float)
        upper = np.broadcast_to(np.asarray(upper, dtype=float), lower.shape)
        lo = np.minimum(np.maximum(self._edges[:-1][None, :], lower[:, None]),
                        upper[:, None])
        hi = np.minimum(np.maximum(self._edges[1:][None, :], lower[:, None]),
                        upper[:, None])
        ulo = np.sqrt(np.maximum(lo - lower[:, None], 0.0))
        uhi = np.sqrt(np.maximum(hi - lower[:, None], 0.0))
        half = 0.5 * (uhi - ulo)
        u = 0.5 * (uhi + ulo)[..., None] + half[..., None] * self._gx
        delta = u * u
        radius = lower[:, None, None] + delta
        n = self.field._n(radius)
        q = n * radius
        qdiff = q - invariant[:, None, None]
        nb = self.field._n(lower)
        dnb = self.field._dn(lower)
        q1 = nb + lower * dnb
        q2 = 2.0 * dnb + lower * self.field._d2n(lower)
        stable = initial_gap[:, None, None] + delta * (
            q1[:, None, None] + 0.5 * delta * q2[:, None, None])
        qdiff = np.where(delta < 0.1, stable, qdiff)
        denominator = np.sqrt(np.maximum(
            qdiff * (q + invariant[:, None, None]), 1e-300))
        # ds/dr = n*r / sqrt((n*r)^2 - a^2); dr/du = 2u.
        weight = q / denominator * (2.0 * u) * self._gw * half[..., None]
        height = (radius - R_EARTH) / 1e3
        result = np.empty((len(lower), self.n_species))
        for index, density in enumerate(self._density_columns(height)):
            result[:, index] = np.sum(np.asarray(density) * weight, axis=(1, 2))
        return result


# ---------------------------------------------------------------------------
# 3. the shadow-aligned point-source table
# ---------------------------------------------------------------------------
@dataclass
class _TableRow:
    """One height row: knots in normalised true-elevation space."""

    t: np.ndarray                 # (true - critical) / (90 - critical), ascending
    direction: np.ndarray         # PCHIP coefficients, (4, n_knot)
    columns: np.ndarray           # PCHIP coefficients, (4, n_knot, n_species)


class PointSourceTable:
    """Shadow-aligned table of direction, columns and gain versus (h, true elev).

    Rows are stacked in ``sqrt(height - ground)`` (grazing sensitivity is
    near the ground); inside a row the knots are refined until every
    interval simultaneously satisfies the angle, slope and optical-depth
    certificates.  Both angular endpoints are exact: the ground-grazing ray
    anchors the lower endpoint and 90 degrees the upper.  The stored
    direction is the *apparent* elevation as a function of normalised true
    elevation; the gain is the derivative of that same interpolant, never a
    second interpolated field.  The slope certificate compares the
    interpolant derivative against the field's analytic derivative, so the
    certificate does not inherit finite-difference noise.

    The table is wavelength independent (species columns in physical units);
    callers apply their own extinction.  It serves one medium construction:
    the geometry cache owns it, and the constructor identity (refraction
    token) is the only sharing contract -- separately constructed media
    build their own tables.
    """

    def __init__(self, medium, field: SolarRayField, columns: SolarSpeciesColumns,
                 *, n_height: int = 65, angle_tolerance_deg: float = 1e-4,
                 derivative_tolerance: float = 0.005, tau_tolerance: float = 0.005,
                 max_refine: int = 20, progress: bool = False):
        self.field = field
        self.columns_tool = columns
        self.ground_km = float(medium.z_bounds[0])
        self.top_km = float(medium.z_bounds[-1])
        self.wl = np.asarray(medium.wl, dtype=float)
        self.extinction = species_extinction_weights(medium)
        # 550 nm anchors the column interpolation conditioning; the anchor is
        # picked by the nearest wavelength *index* so the scale is invariant
        # under wavelength reordering (np.interp would silently change with
        # an unsorted grid).  When 550 nm is absent (beyond 25 nm) the
        # strongest channel per species takes over; the scale is value-free:
        # log1p/expm1 round-trips exactly for any positive scale.
        k = int(np.argmin(np.abs(self.wl - 550.0)))
        if abs(float(self.wl[k]) - 550.0) <= 25.0:
            ref = self.extinction[:, k]
        else:
            ref = np.max(self.extinction, axis=1)
        self.column_scale = np.where(ref > 0, ref,
                                     np.maximum(np.max(self.extinction, axis=1), 1e-300))
        self.column_scale[self.column_scale == 0] = 1.0
        self.constant_index = bool(np.all(field.n == field.n_inf))
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.derivative_tolerance = float(derivative_tolerance)
        self.tau_tolerance = float(tau_tolerance)
        self.max_refine = int(max_refine)
        self.n_height = int(n_height)
        self.build_seconds = 0.0
        self.build_diagnostics: list[dict] = []

        u = np.linspace(0.0, np.sqrt(self.top_km - self.ground_km), max(2, n_height))
        special = np.asarray(medium.z_bounds, dtype=float)
        special = special[(special >= self.ground_km) & (special <= self.top_km)]
        # Near the ground the columns of grazing rays amplify small column
        # errors exponentially in T; resolve that regime explicitly instead
        # of certifying angles only (the historical heldout lesson).
        low = np.arange(0.0, min(1.0, u[-1]) + 1e-10, 0.05)
        middle = np.arange(1.0, min(np.sqrt(5.0), u[-1]) + 1e-10, 0.1)
        heights = np.unique(np.concatenate([
            self.ground_km + u * u, self.ground_km + low * low,
            self.ground_km + middle * middle, special]))
        # the u^2 lattice can land within one ulp of a z_bounds special (e.g.
        # the model top); a zero-width height cell would divide 0/0 in the
        # row interpolation below
        keep = np.concatenate([[True], np.diff(heights) > 1e-9])
        self.heights = heights[keep]
        self.height_axis = np.sqrt(self.heights - self.ground_km)

        started = time.perf_counter()
        rows: list[_TableRow] = []
        criticals: list[float] = []
        for height in self.heights:
            row, critical, diagnostic = self._build_row(float(height))
            rows.append(row)
            criticals.append(critical)
            self.build_diagnostics.append(diagnostic)
            if progress:
                print(f"solar table row h={height:.2f}: {diagnostic}", flush=True)
        self.rows = rows
        self.critical = np.asarray(criticals)
        self.build_seconds = time.perf_counter() - started

    # -- construction ---------------------------------------------------------
    def _initial_knots(self, lower: float) -> np.ndarray:
        """Initial apparent-angle knots for one row.

        The grid the experimental audit validated: half-degree steps near the
        horizon and through the first few degrees, coarser above.  Rows are
        built independently: sharing knots across heights was measured to
        ratchet the knot set monotonically upward (refinement only ever
        adds), tripling the final table size and its build cost.
        """
        # Just above the grazing endpoint the mapping concentrates nearly all
        # of its curvature into the first PCHIP interval; queries at true
        # elevations crit+0.001 deg land there, so the interval must start
        # resolved instead of waiting for refinement rounds to discover it.
        span = 0.0 - lower
        near = lower + span * np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
        app = np.unique(np.concatenate([
            [lower, 0.0, 90.0], near, np.arange(lower, 0, 0.5),
            np.arange(0, 5.0001, 0.5), np.arange(7, 20.0001, 2),
            np.arange(30, 90, 10)]))
        return np.unique(np.clip(app, lower, 90.0))

    def _build_row(self, height):
        t0 = time.perf_counter()
        lower = float(self.field.grazing_apparent_elevation(np.array([height]))[0]) + 1e-6
        app = self._initial_knots(lower)
        true, slope = self.field.forward_with_slope(np.full(app.shape, height), app)
        cols = self.columns_tool.columns(height, app)
        critical = float(true[0])
        diagnostic = {}
        # Per-interval certificates, evaluated incrementally: only intervals
        # whose interpolant stencil changed since their last clean check are
        # re-evaluated.  Splitting an interval invalidates the four-knot
        # stencil of its neighbours (PCHIP endpoint slopes are global per
        # row), which is exactly what the dirty marking below encodes.
        clean = np.zeros(app.size - 1, dtype=bool)
        worst_detail = {}
        for iteration in range(self.max_refine + 1):
            evals = np.flatnonzero(~clean)
            if not evals.size:
                break
            t = (true - critical) / (90.0 - critical)
            d = (app - lower) / (90.0 - lower)
            direction = PchipInterpolator(t, d)
            column = PchipInterpolator(t, np.log1p(cols * self.column_scale), axis=0)
            midapp = 0.5 * (app[:-1] + app[1:])[evals]
            midtrue, midslope = self.field.forward_with_slope(
                np.full(midapp.shape, height), midapp)
            midcols = self.columns_tool.columns(height, midapp)
            midt = (midtrue - critical) / (90.0 - critical)
            predapp = lower + (90.0 - lower) * direction(midt)
            # Certificate: the interpolant's d(app)/d(true) must reproduce the
            # field's analytic d(true)/d(app).  Both endpoints are exact, so
            # this is a pointwise check at the scale of each interval.
            dd_dt = direction.derivative()(midt)
            d_app_d_true = dd_dt * (90.0 - lower) / (90.0 - critical)
            with np.errstate(divide="ignore", invalid="ignore"):
                derivative_error = np.abs(1.0 / (d_app_d_true * midslope) - 1.0)
            angle_error = np.abs(predapp - midapp)
            predcols = np.maximum(np.expm1(column(midt)), 0.0) / self.column_scale
            tau = midcols @ self.extinction
            tau_error = np.abs((predcols - midcols) @ self.extinction)
            # Extremely opaque rays are irrelevant to a direct source; the
            # log-column representation still preserves positivity there.
            active = tau < 25.0
            optical_error = np.max(np.where(active, tau_error, 0.0), axis=1)
            # Near the horizon kink (apparent elevation ~ 0, the branch point
            # between up-going and turning rays) the analytic slope carries an
            # endpoint singularity whose cancellation the integration grid
            # cannot resolve for |app| below its node spacing in angle; the
            # slope certificate there is unresolvable by construction, while
            # the gain is carried by the same-interpolant derivative whose
            # horizon accuracy the experimental audit validated at even 0.5
            # degree knot spacing.  Exempt the slope check in that band; the
            # angle and optical-depth certificates remain active everywhere.
            slope_checkable = np.abs(midapp) > _HORIZON_KINK_GUARD_DEG
            bad = ((angle_error > self.angle_tolerance_deg)
                   | (optical_error > self.tau_tolerance)
                   | ((derivative_error > self.derivative_tolerance)
                      & slope_checkable))
            diagnostic = dict(height=height, knots=len(app),
                              refinement_rounds=iteration,
                              max_midpoint_angle_error_deg=float(np.max(angle_error)),
                              max_midpoint_derivative_relative_error=float(np.max(derivative_error)),
                              max_midpoint_optical_depth_error=float(np.max(optical_error)),
                              unresolved_intervals=int(np.count_nonzero(bad)),
                              seconds=time.perf_counter() - t0)
            clean[evals[~bad]] = True
            if not bad.any():
                break
            if iteration == self.max_refine:
                worst = np.argsort(derivative_error)[-5:]
                worst_detail = dict(worst_app_intervals=np.column_stack([
                    app[:-1][evals[worst]], app[1:][evals[worst]]]).tolist(),
                    worst_derivative_error=derivative_error[worst].tolist())
                raise RuntimeError(
                    f"Solar source row did not meet interpolation tolerances: "
                    f"{diagnostic}; {worst_detail}")
            # Split the failing intervals together with their immediate
            # neighbours (the audit's resolution-balancing rule): a tiny
            # failing interval wedged against a wide certified one makes the
            # PCHIP endpoint derivative at the tiny interval ill-conditioned,
            # and bisecting only the tiny side spirals without converging.
            # Unevaluated neighbours are evaluated now, since their midpoints
            # become knots.
            cand = np.zeros(app.size - 1, dtype=bool)
            cand[evals[bad]] = True
            cand[1:] |= cand[:-1]
            cand[:-1] |= cand[1:]
            split_at = np.flatnonzero(cand)
            # one fresh midpoint evaluation per split interval (re-checking
            # the few already-evaluated ones costs less than bookkeeping)
            m_app_all = 0.5 * (app[:-1] + app[1:])[split_at]
            m_true_all, _ = self.field.forward_with_slope(
                np.full(m_app_all.shape, height), m_app_all)
            m_cols_all = self.columns_tool.columns(height, m_app_all)
            app = np.insert(app, split_at + 1, m_app_all)
            true = np.insert(true, split_at + 1, m_true_all)
            cols = np.insert(cols, split_at + 1, m_cols_all, axis=0)
            clean = np.insert(clean, split_at + 1, False)
            stale = []
            offset = 0
            for s in split_at:
                stale.extend([s + offset - 1, s + offset,
                              s + offset + 1, s + offset + 2])
                offset += 1
            clean[np.clip(np.asarray(stale, dtype=int), 0, clean.size - 1)] = False
        row = _TableRow(t=t, direction=direction.c, columns=column.c)
        return row, critical, diagnostic

    # -- queries --------------------------------------------------------------
    @staticmethod
    def _evaluate_row(row: _TableRow, t):
        i = np.clip(np.searchsorted(row.t, t, side="right") - 1, 0, len(row.t) - 2)
        dx = t - row.t[i]
        c = row.direction[:, i]
        value = ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]
        derivative = (3 * c[0] * dx + 2 * c[1]) * dx + c[2]
        c = row.columns[:, i, :]
        x = dx[:, None]
        columns = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
        return value, derivative, columns

    def query_fields(self, height_km, true_elevation_deg):
        """Return ``(app_deg, columns, gain_local, lit)`` with broadcasting.

        ``columns`` has shape ``broadcast_shape + (n_species,)``.  Shadowed
        points return ``lit=False``, zero columns and zero gain; ``app`` is
        the grazing elevation there (the direction a just-clearing ray would
        have), never a made-up number.
        """
        z, true = np.broadcast_arrays(np.asarray(height_km, dtype=float),
                                      np.asarray(true_elevation_deg, dtype=float))
        shape = z.shape
        zz, et = z.ravel(), true.ravel()
        if np.any(~np.isfinite(zz)) or np.any(~np.isfinite(et)):
            raise ValueError("height and true elevation must be finite")
        if np.any(zz < self.ground_km - 1e-8) or np.any(zz > self.top_km + 1e-8):
            raise ValueError("source query lies outside the model")
        if np.any(np.abs(et) > 90 + 1e-8):
            raise ValueError("true solar elevation must be within [-90, 90]")
        zz = np.clip(zz, self.ground_km, self.top_km)
        et = np.clip(et, -90.0, 90.0)
        u = np.sqrt(zz - self.ground_km)
        i = np.clip(np.searchsorted(self.height_axis, u, side="right") - 1,
                    0, len(self.height_axis) - 2)
        w = (u - self.height_axis[i]) / (self.height_axis[i + 1] - self.height_axis[i])
        critical = (1 - w) * self.critical[i] + w * self.critical[i + 1]
        lower = self.field.grazing_apparent_elevation(zz) + 1e-6
        if self.constant_index:
            # The identity angular map and the Cartesian ground shadow are
            # analytic in a uniform index (including a non-vacuum far field);
            # do not contaminate that limit with height interpolation.
            critical = lower - 1e-6
        t = (et - critical) / (90.0 - critical)
        lit = t >= 0
        app = lower.copy()
        gain = np.zeros_like(app)
        columns = np.zeros((len(zz), len(self.column_scale)))
        if lit.any():
            ind = np.flatnonzero(lit)
            value = np.zeros(len(ind))
            derivative = np.zeros(len(ind))
            col = np.zeros((len(ind), len(self.column_scale)))
            for side in (0, 1):
                rows = i[ind] + side
                weight = (1 - w[ind]) if side == 0 else w[ind]
                # contiguous per-row groups via argsort: the boolean-mask
                # variant cost O(rows x points) and dominated wide queries
                # (dome-shaped batches span ~all height rows)
                order = np.argsort(rows, kind="stable")
                sorted_rows = rows[order]
                boundaries = np.flatnonzero(np.diff(sorted_rows)) + 1
                starts = np.concatenate([[0], boundaries])
                ends = np.concatenate([boundaries, [rows.size]])
                for s0, e0 in zip(starts, ends):
                    ids = order[s0:e0]
                    val, der, cc = self._evaluate_row(self.rows[sorted_rows[s0]],
                                                      t[ind[ids]])
                    value[ids] += weight[ids] * val
                    derivative[ids] += weight[ids] * der
                    col[ids] += weight[ids, None] * cc
            app[ind] = lower[ind] + (90.0 - lower[ind]) * value
            # d(app)/d(true) from the same interpolant that produced app.
            slope = (90.0 - lower[ind]) / (90.0 - critical[ind]) * derivative
            n = self.field.refractive_index(zz[ind])
            near_vertical = np.abs(et[ind] - 90.0) < 1e-6
            ratio = np.divide(np.cos(np.radians(app[ind])),
                              np.cos(np.radians(et[ind])),
                              out=slope.copy(), where=~near_vertical)
            gain[ind] = (n / self.field.n_inf) ** 2 * ratio * slope
            if self.constant_index:
                app[ind] = et[ind]
                gain[ind] = 1.0
            columns[ind] = np.maximum(np.expm1(col), 0.0) / self.column_scale
        return (app.reshape(shape), columns.reshape(shape + (len(self.column_scale),)),
                gain.reshape(shape), lit.reshape(shape))

    def query(self, height_km, true_elevation_deg, iw=None):
        """Return ``(app_deg, T, gain_local, lit)``; ``iw=None`` adds all wl.

        ``T = exp(-columns . extinction)`` is the pure attenuation, zeroed
        where shadowed; gain is returned separately (their product is never
        named transmittance and never clipped).
        """
        app, columns, gain, lit = self.query_fields(height_km, true_elevation_deg)
        if iw is None:
            tau = columns @ self.extinction
            transmittance = np.where(lit[..., None], np.exp(-tau), 0.0)
        else:
            if not 0 <= int(iw) < len(self.wl):
                raise IndexError("wavelength index outside the source table")
            tau = columns @ self.extinction[:, int(iw)]
            transmittance = np.where(lit, np.exp(-tau), 0.0)
        return app, transmittance, gain, lit


# ---------------------------------------------------------------------------
# 4. token-cached geometry
# ---------------------------------------------------------------------------
class SolarGeometry:
    """Ray field, columns and (lazily) the point-source table of one medium.

    Cached per medium-construction token and refraction mode in a weak
    dictionary: entries live exactly as long as some living medium holds the
    token, and this object holds no strong reference to the medium (only a
    weakref for binding validation), so a dead medium's geometry is reclaimed
    instead of pinning it.
    """

    def __init__(self, medium, *, refracted: bool = True,
                 build_table: bool | None = None):
        self.medium_ref = weakref.ref(medium)
        self.refracted = bool(refracted)
        self.field = SolarRayField(medium, refracted=refracted)
        self.columns_tool = SolarSpeciesColumns(medium, self.field)
        self.ground_km = float(medium.z_bounds[0])
        self.top_km = float(medium.z_bounds[-1])
        self.n_species = self.columns_tool.n_species
        self._table: PointSourceTable | None = None
        self._table_build_lock = False
        if build_table:
            self.ensure_table()

    @property
    def table(self) -> PointSourceTable:
        if self._table is None:
            raise RuntimeError(
                "solar source table not built yet; call ensure_table() first")
        return self._table

    def ensure_table(self, **kw) -> PointSourceTable:
        if self._table is None:
            medium = self.medium_ref()
            if medium is None:
                raise RuntimeError("the owning medium is gone; rebuild the source")
            self._table = PointSourceTable(medium, self.field, self.columns_tool, **kw)
        return self._table

    def validate_medium(self, medium) -> None:
        if self.medium_ref() is not medium:
            raise ValueError(
                "solar source bound to a different medium construction; "
                "build a source from the current atmosphere")


_GEOMETRY_CACHE: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()


def get_solar_geometry(medium, *, refracted: bool = True) -> SolarGeometry:
    """Cached :class:`SolarGeometry` for one medium construction.

    Identity is the medium's construction token (the same rule as the
    refraction tables): wavelength subsets share, independent constructions
    never do.  The refraction mode is part of the key -- ``refraction=False``
    is a distinct contract (the ``n = 1`` limit), not a flag on one table.
    """
    per_mode = _GEOMETRY_CACHE.get(medium.refraction_token)
    if per_mode is None:
        per_mode = {}
        _GEOMETRY_CACHE[medium.refraction_token] = per_mode
    geom = per_mode.get(bool(refracted))
    if geom is None:
        geom = SolarGeometry(medium, refracted=bool(refracted))
        per_mode[bool(refracted)] = geom
    return geom


# ---------------------------------------------------------------------------
# 5. the scenario-bound source with consumer kernels
# ---------------------------------------------------------------------------
class SolarSource:
    """The finite-disk solar source bound to one atmosphere and sun direction.

    One instance serves all backends of one solve: the disk quadrature
    (uniform spherical cap, radius 16' by default, weights satisfying
    ``sum(w cos rho) = 1`` under the centre-normal-plane ``F_inf``
    normalisation; ``radius_deg=0`` is the delta-source limit used for
    validation), the observer's detector-transfer factor, and the kernels:

    * :meth:`vertex` -- per-point direction/columns/T/gain from the table;
    * :meth:`scatter` -- the angular integral ``sum_beams T*G*p_mix`` with
      mixtures normalised to ``4*pi``, including the observer transfer
      (the single ``n^2`` factor of the whole chain); no ``F_sun/(4pi)``
      and no local single-scattering albedo are included;
    * :meth:`ground` -- detector-referred ``sum_beams T*G*max(sin app, 0)``
      for horizontal incidence;
    * :meth:`direct_irradiance` -- flux on an arbitrary plane, the seam the
      renderer and the validation DNI use;
    * :meth:`central_transmittance` -- pure central-beam attenuation
      (direct evaluation, no table build) for the legacy transmittance APIs.

    Binding discipline: every kernel validates that the caller's atmosphere
    is still the construction medium/scene, so a swapped medium or scene can
    never silently query a stale source.
    """

    def __init__(self, atmo, sun_zenith_deg: float, sun_azimuth_deg: float = 0.0, *,
                 radius_deg: float = SUN_RADIUS_DEG,
                 n_radial: int = 4, n_azimuth: int = 16, chunk: int = 65536):
        medium = atmo._medium
        self._atmo_ref = weakref.ref(atmo)
        self._medium = medium
        self._scene = atmo._scene
        self.geometry = get_solar_geometry(medium, refracted=bool(atmo.refraction))
        self.radius_deg = float(radius_deg)
        self.n_radial = int(n_radial)
        self.n_azimuth = int(n_azimuth)
        self.chunk = max(1, int(chunk))
        self.sun_zenith_deg = float(sun_zenith_deg)
        self.sun_azimuth_deg = float(sun_azimuth_deg)
        zs = np.radians(self.sun_zenith_deg)
        az = np.radians(self.sun_azimuth_deg)
        self.sun_hat = np.array([np.sin(zs) * np.sin(az),
                                 np.sin(zs) * np.cos(az),
                                 np.cos(zs)])
        self.observer_n = float(self.geometry.field.refractive_index(
            np.array([atmo.elevation_km]))[0])
        self._disk_cache: dict[bytes, tuple] = {}
        self._crit_cache: dict[float, float] = {}
        self.n_calls = 0
        self.n_points = 0

    # -- binding --------------------------------------------------------------
    def _validate(self, atmo) -> None:
        if (atmo is not self._atmo_ref()
                or atmo._medium is not self._medium
                or atmo._scene is not self._scene):
            raise ValueError(
                "solar source bound to a different atmosphere construction or "
                "scene; rebuild the source for the current atmosphere")

    def disk(self) -> tuple[np.ndarray, np.ndarray]:
        """True disk-beam directions ``(n_beam, 3)`` and weights ``(n_beam,)``."""
        key = self.sun_hat.tobytes()
        cached = self._disk_cache.get(key)
        if cached is None:
            cached = solar_disk(self.sun_hat, self.n_radial, self.n_azimuth,
                                self.radius_deg)
            self._disk_cache[key] = cached
        return cached

    def _flux_disk_for_points(self, x):
        """Refined flux disk when every point shares one occultation band."""
        heights = np.clip((np.linalg.norm(x, axis=1) - R_EARTH) / 1e3,
                          self._medium.z_bounds[0], self._medium.z_bounds[-1])
        if float(heights.max()) - float(heights.min()) < 1e-6:
            return self._flux_disk(float(heights[0]))
        return self.disk()

    def _critical_true(self, height_km: float) -> float:
        """True elevation of the ground-grazing ray at ``height_km``."""
        cache = self._crit_cache
        if height_km in cache:
            return cache[height_km]
        field = self.geometry.field
        lower = float(field.grazing_apparent_elevation(np.array([height_km]))[0])
        crit = float(field.forward(np.array([height_km]),
                                   np.array([lower + 1e-6]))[0])
        cache[height_km] = crit
        return crit

    def _flux_disk(self, height_km):
        """Disk for single-height flux kernels, refined near the occultation band.

        When the shadow boundary cuts through the disk at this height, the
        coarse product quadrature integrates the partially lit cap poorly
        (tens of per cent of the tiny signal).  Densify the radial nodes
        around the boundary ring and widen the azimuth grid; the weights keep
        the same centre-normal normalisation, so this is pure refinement.
        """
        crit = self._critical_true(float(height_km))
        elev = 90.0 - self.sun_zenith_deg
        if abs(elev - crit) >= 3.0 * self.radius_deg:
            return self.disk()
        key = (self.sun_hat.tobytes(), "refined", round(float(height_km), 3))
        cached = self._disk_cache.get(key)
        if cached is None:
            # the horizontal-flux integrand steps across the horizon line, so
            # the azimuth resolution dominates the error: a fine azimuth grid
            # (single-point flux calls can afford thousands of beams) with a
            # boundary-split radial rule
            n_r, n_a = 32, 768
            cos_crit = np.cos(np.radians(min(abs(elev - crit), self.radius_deg)))
            lo_nodes, lo_w = np.polynomial.legendre.leggauss(max(2, n_r // 2))
            hi_nodes, hi_w = np.polynomial.legendre.leggauss(max(2, n_r // 2))
            # inner segment [0, rho_crit] and outer [rho_crit, radius], in
            # cos(rho) where the cap area measure is linear
            c_in = cos_crit + (1.0 - cos_crit) * (0.5 * (lo_nodes + 1.0))
            w_in = (1.0 - cos_crit) * 0.5 * lo_w
            c_out = cos_crit + (np.cos(np.radians(self.radius_deg)) - cos_crit) \
                * (0.5 * (hi_nodes + 1.0))
            w_out = (np.cos(np.radians(self.radius_deg)) - cos_crit) * 0.5 * hi_w
            cos_r = np.concatenate([c_in, c_out])
            # the same convention as the plain disk: raw leggauss weights
            # (the segment spans live in the node mapping), divided by
            # (1 + cos(alpha)) * n_azimuth
            w_r = np.concatenate([lo_w, hi_w]) \
                / ((1.0 + np.cos(np.radians(self.radius_deg))) * n_a)
            alpha = np.radians(self.radius_deg)
            sin_rho = np.sqrt(np.maximum(1 - np.asarray(cos_r) ** 2, 0))
            reference = (np.array([0.0, 0.0, 1.0])
                         if abs(self.sun_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
            e1 = reference - np.dot(reference, self.sun_hat) * self.sun_hat
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(self.sun_hat, e1)
            phi = 2 * np.pi * (np.arange(n_a) + 0.5) / n_a
            transverse = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
            directions = (np.asarray(cos_r)[:, None, None] * self.sun_hat
                          + sin_rho[:, None, None] * transverse).reshape(-1, 3)
            weights = np.repeat(np.asarray(w_r), n_a)
            cached = (directions, weights)
            self._disk_cache[key] = cached
        return cached

    # -- kernels --------------------------------------------------------------
    def vertex(self, height_km, true_elevation_deg, *, iw=None):
        """Table query for arbitrary (height, true elevation) points.

        Returns ``(app_deg, T, gain_local, lit)``; builds the table on first
        use.  ``T`` has a final wavelength axis unless ``iw`` is given.  The
        attenuation is evaluated with THIS atmosphere's cross sections -- the
        geometry (and its table) is shared across wavelength subsets, so the
        table's own stored extinction belongs to whichever construction built
        it and must never leak into another medium's query.
        """
        self.n_calls += 1
        table = self.geometry.ensure_table()
        app, columns, gain, lit = table.query_fields(height_km, true_elevation_deg)
        ext = species_extinction_weights(self._medium)
        if iw is None:
            tau = columns @ ext
            transmittance = np.where(lit[..., None], np.exp(-tau), 0.0)
        else:
            if not 0 <= int(iw) < ext.shape[1]:
                raise IndexError("wavelength index outside the source table")
            tau = columns @ ext[:, int(iw)]
            transmittance = np.where(lit, np.exp(-tau), 0.0)
        return app, transmittance, gain, lit

    def _geometry_arrays(self, x, iw, disk=None):
        """Per-point per-beam geometry: true elevations, app, T, weighted gain.

        ``x`` is ``(n, 3)`` in the Earth-centred frame with the observer on
        the +z axis.  ``disk`` overrides the beam set (contact-band refined
        flux kernels); the gain carries the disk weights and the single
        observer transfer ``(n_obs / n_point)^2``; nothing else in the
        chain may apply an ``n^2`` factor.
        """
        directions, weights = disk if disk is not None else self.disk()
        medium = self._medium
        radius = np.linalg.norm(x, axis=1)
        radial = x / radius[:, None]
        height = np.clip((radius - R_EARTH) / 1e3,
                         medium.z_bounds[0], medium.z_bounds[-1])
        true_mu = np.clip(radial @ directions.T, -1.0, 1.0)
        true_e = np.degrees(np.arcsin(true_mu))
        app, columns, gain, lit = self.geometry.ensure_table().query_fields(
            height[:, None], true_e)
        ext = species_extinction_weights(medium)
        if iw is None:
            tau = columns @ ext
            transmission = np.where(lit[..., None], np.exp(-tau), 0.0)
        else:
            tau = columns @ ext[:, int(iw)]
            transmission = np.where(lit, np.exp(-tau), 0.0)
        app_rad = np.radians(app)
        local_n = self.geometry.field.refractive_index(height)
        gain = gain * (self.observer_n / local_n[:, None]) ** 2
        weighted_gain = gain * weights
        return radial, directions, true_mu, app_rad, transmission, weighted_gain, lit

    def scatter(self, x, backward, *, iw=None, ray_fraction, aerosol_fractions):
        """Solar angular integral of ``T * G * p_mix`` over the disk beams.

        For scalar ``iw`` the fractions have shapes ``(n,)`` / ``(n, n_ae)``
        and the output is ``(n,)``; for ``iw=None`` they have ``(n, n_wl)``
        / ``(n, n_ae, n_wl)`` and the output is ``(n, n_wl)``.  The phase
        argument is the *apparent* beam direction composed with the backward
        ray, not the geometric one.
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        n_wl = len(atmo.wl)
        out = np.zeros((len(x), n_wl)) if iw is None else np.zeros(len(x))
        wavelengths = atmo.wl if iw is None else np.array([atmo.wl[iw]])
        # full-spectrum batches materialise (points, beams, n_wl) arrays;
        # shrink the step so the peak transient stays bounded
        step = self.chunk if iw is not None else max(2048, self.chunk // 8)
        for start in range(0, len(x), step):
            sl = slice(start, min(start + step, len(x)))
            radial, directions, mu, app, transmission, gain, lit = \
                self._geometry_arrays(x[sl], iw)
            active = np.flatnonzero(np.any(lit, axis=1))
            if not active.size:
                continue
            # Twilight has many fully shadowed collisions; skipping their Mie
            # phase evaluations does not change any sampled path.
            positions = start + active
            radial, mu, app = radial[active], mu[active], app[active]
            transmission, gain = transmission[active], gain[active]
            dr = np.einsum("ij,ij->i", backward[positions], radial)[:, None]
            horizontal_dot = (backward[positions] @ directions.T - mu * dr) / np.sqrt(
                np.maximum(1 - mu ** 2, 1e-30))
            phase_mu = np.cos(app) * horizontal_dot + np.sin(app) * dr
            phase_mu = np.where(np.abs(mu) > 1 - 1e-14, np.sign(mu) * dr, phase_mu)
            phase_mu = np.clip(phase_mu, -1.0, 1.0)
            p_ray = spectra.rayleigh_phase(phase_mu)
            if iw is None:
                phase = p_ray[..., None] * ray_fraction[positions, None, :]
            else:
                phase = p_ray * ray_fraction[positions, None]
            for j, layer in enumerate(atmo.aerosols):
                aerosol_phase = np.asarray(
                    layer.aerosol.phase(phase_mu.ravel(), wavelengths))
                if aerosol_phase.ndim == 1:
                    # wavelength-independent phase functions return one value
                    # per direction; broadcast across the band
                    aerosol_phase = np.broadcast_to(
                        aerosol_phase.reshape(phase_mu.shape + (1,)),
                        phase_mu.shape + (len(wavelengths),))
                else:
                    aerosol_phase = aerosol_phase.reshape(
                        phase_mu.shape + (len(wavelengths),))
                if iw is None:
                    phase = phase + aerosol_phase * aerosol_fractions[positions, j, None, :]
                else:
                    phase = phase + aerosol_phase[..., 0] * aerosol_fractions[positions, j, None]
            if iw is None:
                out[positions] = np.sum(transmission * gain[..., None] * phase, axis=1)
            else:
                out[positions] = np.sum(transmission * gain * phase, axis=1)
        self.n_points += len(x)
        return out

    def ground(self, x, *, iw=None):
        """Detector-referred solar angular integral of ``T*G*max(sin app, 0)``."""
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        out = np.zeros(len(x)) if iw is not None else np.zeros((len(x), len(atmo.wl)))
        for start in range(0, len(x), self.chunk):
            sl = slice(start, min(start + self.chunk, len(x)))
            _, _, _, app, transmission, gain, _ = self._geometry_arrays(
                x[sl], iw, disk=self._flux_disk_for_points(x[sl]))
            if iw is None:
                gain = gain[..., None]     # gain is wavelength independent
                app = app[..., None]
            out[sl] = np.sum(transmission * gain * np.maximum(np.sin(app), 0.0),
                             axis=1)
        return out

    def direct_irradiance(self, x, normal, *, iw=None):
        """Spectral flux through a plane at the points ``x``.

        ``normal`` is the plane's unit normal in the Earth-centred frame;
        the flux is ``F_inf * sum_beams w*T*G*max(apparent_beam . normal, 0)``
        (detector-referred through the same single ``n^2`` factor).
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        directions, _ = self.disk()
        f_inf = spectra.solar_irradiance(atmo.wl)
        n_pts = len(x)
        out = np.zeros((n_pts, len(atmo.wl))) if iw is None else np.zeros(n_pts)
        for start in range(0, n_pts, self.chunk):
            sl = slice(start, min(start + self.chunk, n_pts))
            _, _, _, app, transmission, gain, _ = self._geometry_arrays(x[sl], iw)
            # The apparent direction stays in the vertical plane of the true
            # direction (radial refraction preserves azimuth), so its unit
            # vector is the true beam direction tilted from the radial by
            # (true -> apparent elevation).
            radial = x[sl] / np.linalg.norm(x[sl], axis=1)[:, None]
            sin_true = np.clip(radial @ directions.T, -1, 1)     # sin(true elev)
            cos_true = np.sqrt(np.maximum(1.0 - sin_true ** 2, 1e-300))
            # The apparent beam direction is  sin(app)*radial + cos(app)*e_h
            # (app = apparent *elevation*, radians); resolve the unit normal
            # against that basis:
            n_dot_radial = np.einsum("ij,j->i", radial, normal)[:, None]
            n_dot_beam = normal @ directions.T                   # (n, n_beam)
            horizontal = (n_dot_beam - n_dot_radial * sin_true) / cos_true
            app_dot_normal = (np.sin(app) * n_dot_radial
                              + np.cos(app) * horizontal)
            if iw is None:
                gain = gain[..., None]     # gain is wavelength independent
                app_dot_normal = app_dot_normal[..., None]
            weighted = transmission * gain * np.maximum(app_dot_normal, 0.0)
            if iw is None:
                out[sl] = weighted.sum(axis=1) * f_inf[None, :]
            else:
                out[sl] = weighted.sum(axis=1) * f_inf[iw]
        return out

    def central_transmittance(self, height_km=None, *, iw=None):
        """Pure central-beam attenuation ``T`` at ``height_km`` (default: observer).

        Direct field evaluation -- no table build -- so diagnostics stay
        cheap even before any scattering kernel has run.  Returns 0 when the
        central ray hits the opaque ground.  This is a centre-beam
        diagnostic; the disk-integrated flux is :meth:`direct_irradiance`.
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        medium = self._medium
        if height_km is None:
            height_km = atmo.elevation_km
        height = float(height_km)
        field = self.geometry.field
        true_e = 90.0 - self.sun_zenith_deg
        lower = float(field.grazing_apparent_elevation(np.array([height]))[0])
        crit = float(field.forward(np.array([height]), np.array([lower + 1e-6]))[0])
        if true_e < crit - 1e-9 or height < medium.z_bounds[0] - 1e-9:
            return 0.0 if iw is not None else np.zeros(len(medium.wl))
        if true_e >= 90.0:
            app = 90.0
        else:
            app = float(field.apparent_of_true(np.array([height]),
                                               np.array([true_e]))[0])
        columns = self.geometry.columns_tool.columns(
            np.array([height]), np.array([app]))[0]
        extinction = species_extinction_weights(medium)
        if iw is not None:
            tau = float(columns @ extinction[:, int(iw)])
        else:
            tau = columns @ extinction
        return np.exp(-tau)


# ---------------------------------------------------------------------------
# disk quadrature
# ---------------------------------------------------------------------------
def solar_disk(sun_hat, n_radial: int = 2, n_azimuth: int = 8,
               radius_deg: float = SUN_RADIUS_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Directions and radiance weights for unit centre-normal irradiance.

    For a uniform spherical cap of angular radius ``alpha``,
    ``L_sun = F_inf / (pi sin(alpha)^2)``, so the Legendre x azimuth product
    weights sum to ``2 / (1 + cos(alpha))`` while ``sum(weight * cos(rho))``
    is exactly one -- the centre-normal-plane ``F_inf`` convention.  Partial
    occultation simply drops beams; the surviving set is never renormalised.
    ``radius_deg = 0`` returns the delta-source limit.
    """
    sun = np.asarray(sun_hat, dtype=float)
    sun = sun / np.linalg.norm(sun)
    if radius_deg == 0:
        return sun[None], np.ones(1)
    if n_radial < 1 or n_azimuth < 4 or not 0 < radius_deg < 90:
        raise ValueError("positive radial order, >=4 azimuths and radius in (0,90) required")
    alpha = np.radians(radius_deg)
    x, w = np.polynomial.legendre.leggauss(n_radial)
    cos_rho = 1 - 0.5 * (x + 1) * (1 - np.cos(alpha))
    sin_rho = np.sqrt(np.maximum(1 - cos_rho ** 2, 0))
    reference = np.array([0.0, 0.0, 1.0]) if abs(sun[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = reference - np.dot(reference, sun) * sun
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sun, e1)
    phi = 2 * np.pi * (np.arange(n_azimuth) + 0.5) / n_azimuth
    transverse = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    directions = (cos_rho[:, None, None] * sun
                  + sin_rho[:, None, None] * transverse).reshape(-1, 3)
    weights = np.repeat(w / ((1 + np.cos(alpha)) * n_azimuth), n_azimuth)
    np.testing.assert_allclose(np.sum(weights * (directions @ sun)), 1.0, atol=2e-14)
    return directions, weights
