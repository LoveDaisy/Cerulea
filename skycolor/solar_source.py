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
#: 1 km through the atmosphere, 5 km in the negligible upper mesosphere.
#: The columns reproduce the RK chain to ~1.3e-4 relative; the field's
#: direction anchor saturates at ~2.67e-4 deg against the 2e-4 deg plan
#: budget because a 0.3 km lower-level refinement (tried and reverted in the
#: round-3 review) resolves genuine near-caustic folds that a monotone
#: PCHIP tabulation cannot certify -- an owner-level design decision, not a
#: wiring bug (see docs/physics/SOLAR_SOURCE.md).
_SEGMENT_KM_MID = 1.0
_SEGMENT_MID_TOP_KM = 120.0
_SEGMENT_KM_HIGH = 5.0

#: request-independent physical anchors for the shared-table build [nm]:
#: the column-interpolation conditioning is anchored at 550 nm, the vacuum
#: and optical-depth certificates at 380 nm (the working band's strongest
#: extinction).  Both are evaluated from the species data directly -- the
#: registrar's wavelength grid must never influence the shared build
#: (review round 4, Major table-build spectral dependence).
_TABLE_SCALE_WL_NM = 550.0
_TABLE_CERT_WL_NM = 380.0

#: angular radius of the solar disk [deg]
SUN_RADIUS_DEG = 16.0 / 60.0

#: half-width of the horizon-kink band [deg] inside which the slope
#: certificate is exempt (see ``PointSourceTable._build_row``)
_HORIZON_KINK_GUARD_DEG = 0.05

#: width below which a knot interval is treated as caustic-noise-exempt
#: (see ``PointSourceTable._build_row``)
_CAUSTIC_GUARD_DEG = 1e-3

#: true-elevation span below which an interval is caustic-flat [deg]
_CAUSTIC_NOISE_DEG = 2e-4

#: query-kernel evaluation block size [points].  The flat gather+Horner pass
#: runs on chunks of this many lit points: full-batch temporaries
#: ((4, P, n_species) coefficient gathers) spill the cache and measured
#: ~30% slower on the dome benchmark; 64k points keep the working set
#: on-chip while still amortizing the per-block numpy overhead.
_QUERY_BLOCK = 1 << 16


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
        the atmosphere: 1 km from the ground through the upper stratosphere
        (knot alignment, not segment size, is what protects the slope
        integrand -- the analytic derivative is exact per segment), 5 km in
        the upper mesosphere and 50 km in the exosphere where n is
        indistinguishable from n_inf.  (A 0.3 km lower-level refinement was
        tried and reverted in the round-3 review: it resolves genuine
        near-caustic folds in high rows that a monotone PCHIP tabulation
        cannot represent -- an owner-level design decision recorded in
        docs/physics/SOLAR_SOURCE.md, not dead wiring.)
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
        # the medium (and with it the cache entry) forever.  Wavelength slices
        # share the token, so every sharing medium is registered and density
        # queries resolve against the first live one.
        self._medium_refs = [weakref.ref(medium)]
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
        # the columns' integrand is n (C1, no curvature term) — the 1 km
        # scheme reproduces the RK chain to 1.3e-4, within the 1e-3 budget
        # (the reverted 0.3 km lower-level experiment concerned only the
        # field's slope integrand; see the module constants comment)
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

    def register_medium(self, medium) -> None:
        if all(ref() is not medium for ref in self._medium_refs):
            self._medium_refs.append(weakref.ref(medium))

    @property
    def medium(self):
        live = [ref for ref in self._medium_refs if ref() is not None]
        self._medium_refs = live
        if not live:
            raise RuntimeError(
                "the owning medium of this solar geometry has been collected")
        return live[0]()

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


@dataclass
class _FlatRows:
    """All rows' knots and PCHIP coefficients concatenated, with offsets.

    One flat copy per table so a query can gather every point's coefficient
    stencil with contiguous first-axis takes instead of looping over row
    groups (the row-group Python loop dominated wide dome-shaped batches).
    The arrays are bitwise copies of the row arrays -- concatenation and
    transpose only move bytes -- and ``knot_base`` / ``coef_base`` locate
    each row's first knot / segment inside the flat arrays (a row has
    ``row_len`` knots and ``row_len - 1`` segments, so the two offsets
    differ).  The coefficient layout puts each segment's four coefficients
    side by side (``(M, 4[, n_species])``, transposed from the row-major
    ``(4, M, ...)`` PCHIP layout): a gather then reads whole stencils with
    one take, which measured ~2.5x faster than gathering a middle axis.
    """

    t: np.ndarray          # (K_total,) concatenated row knots
    direction: np.ndarray  # (4, M_total,) per-plane direction coefficients
    columns: np.ndarray    # (M_total, 4, n_species,) per-segment stencils
    knot_base: np.ndarray  # (n_rows,) flat offset of each row's first knot
    coef_base: np.ndarray  # (n_rows,) flat offset of each row's first segment
    row_len: np.ndarray    # (n_rows,) knots per row


def _anchor_extinction(medium, wl_nm: float) -> np.ndarray:
    """Species extinction evaluated at one fixed anchor wavelength.

    The registrar's wavelength grid must never leak into the shared table's
    build (whichever slice registers first would otherwise set the
    conditioning and the refinement certificates for every later query), so
    the build anchors are evaluated from the species data functions directly
    at fixed physical wavelengths.  Returns ``(n_species, 1)``.
    """
    wl = np.array([float(wl_nm)])
    aerosol_ext = [lay.aerosol.relative_coefficients(wl)[0]
                   for lay in medium.aerosols]
    return np.stack([spectra.rayleigh_cross_section(wl),
                     spectra.ozone_cross_section(wl),
                     spectra.no2_cross_section(wl)]
                    + [np.asarray(e, dtype=float) for e in aerosol_ext])


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
                 max_refine: int = 45, progress: bool = False):
        self.field = field
        self.columns_tool = columns
        self.ground_km = float(medium.z_bounds[0])
        self.top_km = float(medium.z_bounds[-1])
        self.wl = np.asarray(medium.wl, dtype=float)
        # Request-independent physical anchors (plan 3.2; review round 4):
        # the table must come out identical whichever wavelength slice of the
        # shared construction registers first, so the interpolation
        # conditioning (550 nm) and the vacuum / optical-depth certificates
        # (380 nm, the working band's strongest extinction) are evaluated
        # from the species data directly and never from the registrar's
        # grid.  Queries apply the querying atmosphere's own cross sections.
        self.extinction = _anchor_extinction(medium, _TABLE_CERT_WL_NM)
        scale_ref = _anchor_extinction(medium, _TABLE_SCALE_WL_NM)[:, 0]
        self.column_scale = np.where(scale_ref > 0, scale_ref,
                                     np.full_like(scale_ref, 1e-300))
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
        # Rows are READ-ONLY after construction: the flat query cache
        # (:meth:`flat_rows`) is materialized once from them and relies on
        # the row arrays never being mutated afterwards.
        self.rows = rows
        self.critical = np.asarray(criticals)
        self._flat_rows_cache: _FlatRows | None = None
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
        # Rows in the near-vacuum upper thermosphere (both the vertical and
        # the grazing optical depth above the height under 1e-6 — every
        # direction's T within 1e-6 of unity): the map is the identity to
        # float precision, and rows whose grazing ray dips into the lower
        # atmosphere keep their real (refracted, attenuated) treatment.
        z_top = self.top_km
        # vacuum criterion on the WORST direction: a grazing ray from this
        # height traverses far more atmosphere than the vertical — a height
        # with tau_vert < 1e-6 can still have grazing tau ~1e-3, and its
        # near-horizon directions must keep their real attenuation
        cols_vert = self.columns_tool.columns(np.array([height]),
                                              np.array([90.0]))[0]
        lower0 = float(self.field.grazing_apparent_elevation(np.array([height]))[0])
        cols_graze = self.columns_tool.columns(np.array([height]),
                                               np.array([lower0 + 1e-3]))[0]
        tau_vert = float(np.max(cols_vert @ self.extinction))
        tau_graze = float(np.max(cols_graze @ self.extinction))
        if max(tau_vert, tau_graze) < 1e-6:
            app = np.array([lower, 0.5 * (lower + 90.0), 90.0])
            true, slope = self.field.forward_with_slope(np.full(3, height), app)
            cols = self.columns_tool.columns(height, app)
            critical = float(true[0])
            t = (true - critical) / (90.0 - critical)
            d = (app - lower) / (90.0 - lower)
            direction = PchipInterpolator(t, d)
            column = PchipInterpolator(t, np.log1p(cols * self.column_scale), axis=0)
            diagnostic = dict(height=height, knots=3, refinement_rounds=0,
                              max_midpoint_angle_error_deg=0.0,
                              max_midpoint_derivative_relative_error=0.0,
                              max_midpoint_optical_depth_error=0.0,
                              unresolved_intervals=0,
                              seconds=time.perf_counter() - t0, vacuum=True)
            row = _TableRow(t=t, direction=direction.c, columns=column.c)
            return row, critical, diagnostic
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
            # The slope certificate is exempt in the horizon-kink band (a
            # branch point whose slope is unresolvable at float precision).
            # Near-caustic intervals (the map's slope crosses zero, e.g.
            # rays from high rows whose perigee dips into the finely
            # resolved lower atmosphere) show sqrt-type convergence: the
            # bisection continues and the error falls off as sqrt(width),
            # so refinement does converge — it just needs the depth.
            slope_checkable = (np.abs(midapp) > _HORIZON_KINK_GUARD_DEG)
            bad = (((angle_error > self.angle_tolerance_deg)
                    | (optical_error > self.tau_tolerance)
                    | ((derivative_error > self.derivative_tolerance)
                       & slope_checkable)))
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
            # near a caustic the map's true elevation can repeat between
            # distinct knots at float precision.  Keep the first knot of each
            # equal-true run and drop the zero-span intervals between runs:
            # knots drop by k, intervals drop by the same k, so the
            # len(clean) == len(app) - 1 invariant survives.
            drop_knot = np.concatenate([[False], np.diff(true) <= 0])
            drop_int = drop_knot[1:]
            if drop_knot.any():
                app, true, cols = (app[~drop_knot], true[~drop_knot],
                                   cols[~drop_knot])
                clean = clean[~drop_int]
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
    def flat_rows(self) -> _FlatRows:
        """Lazily materialize the flattened query arrays (once per table).

        Cheap relative to the build (~ms of concatenation for ~13k knots);
        like the row arrays themselves the flat copy lives and dies with the
        table object, whose ownership is the geometry cache's token contract.
        """
        if self._flat_rows_cache is None:
            lens = np.fromiter((row.t.size for row in self.rows), dtype=np.intp,
                               count=len(self.rows))
            knot_base = np.zeros(lens.size, dtype=np.intp)
            np.cumsum(lens[:-1], out=knot_base[1:])
            coef_base = np.zeros(lens.size, dtype=np.intp)
            np.cumsum(lens[:-1] - 1, out=coef_base[1:])
            self._flat_rows_cache = _FlatRows(
                t=np.concatenate([row.t for row in self.rows]),
                direction=np.concatenate([row.direction for row in self.rows],
                                         axis=1),
                columns=np.concatenate(
                    [np.ascontiguousarray(row.columns.transpose(1, 0, 2))
                     for row in self.rows]),
                knot_base=knot_base, coef_base=coef_base, row_len=lens)
        return self._flat_rows_cache

    @staticmethod
    def _segment_indices(flat: _FlatRows, row, t):
        """Per-point in-row segment index, bitwise-equal to the per-row probe.

        For each point, the index ``np.clip(np.searchsorted(its row's knots,
        t, side="right") - 1, 0, row_len - 2)`` returns.  Points are grouped
        by row with a stable argsort so each group's probe runs through the
        C ``searchsorted`` on a zero-copy view of that row's knots inside
        the flat array -- the exact primitive and comparison semantics of
        the per-row implementation, hence identical segment selection by
        construction.  ``row`` indexes the table rows, ``t`` holds the
        normalised true elevations of the same points.
        """
        order = np.argsort(row, kind="stable")
        sorted_rows = row[order]
        t_sorted = t[order]
        knot_base, row_len = flat.knot_base, flat.row_len
        boundaries = np.flatnonzero(np.diff(sorted_rows)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [row.size]))
        j_sorted = np.empty(row.size, dtype=np.intp)
        for s0, e0 in zip(starts, ends):
            r = int(sorted_rows[s0])
            n = int(row_len[r])
            knots = flat.t[knot_base[r]:knot_base[r] + n]
            j_sorted[s0:e0] = np.clip(
                np.searchsorted(knots, t_sorted[s0:e0], side="right") - 1,
                0, n - 2)
        j = np.empty_like(j_sorted)
        j[order] = j_sorted
        return j

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
        # height-axis quantities (spline evals, row selection) are computed
        # per UNIQUE height: consumers query (height x beam) grids where the
        # height repeats once per beam, so this dedupes ~n_beam-fold
        zz_unique, inverse = np.unique(zz, return_inverse=True)
        u_u = np.sqrt(zz_unique - self.ground_km)
        i_u = np.clip(np.searchsorted(self.height_axis, u_u, side="right") - 1,
                      0, len(self.height_axis) - 2)
        w_u = ((u_u - self.height_axis[i_u])
               / (self.height_axis[i_u + 1] - self.height_axis[i_u]))
        critical_u = (1 - w_u) * self.critical[i_u] + w_u * self.critical[i_u + 1]
        lower_u = self.field.grazing_apparent_elevation(zz_unique) + 1e-6
        if self.constant_index:
            # The identity angular map and the Cartesian ground shadow are
            # analytic in a uniform index (including a non-vacuum far field);
            # do not contaminate that limit with height interpolation.
            critical_u = lower_u - 1e-6
        i = i_u[inverse]
        w = w_u[inverse]
        critical = critical_u[inverse]
        lower = lower_u[inverse]
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
            flat = self.flat_rows()
            t_lit = t[ind]
            row0 = i[ind]
            nsp = len(self.column_scale)
            P = ind.size
            bs = min(_QUERY_BLOCK, P)
            cc_sten = np.empty((bs, 4, nsp))
            b_c0, b_c1, b_c2, b_c3 = (np.empty(bs) for _ in range(4))
            b_val, b_der, b_t = np.empty(bs), np.empty(bs), np.empty(bs)
            tmp = np.empty((bs, nsp))
            dir0, dir1, dir2, dir3 = (flat.direction[k] for k in range(4))
            # Side 0 runs to completion before side 1 and each point sees
            # exactly one multiply-add per side -- the row-group loop's
            # per-point floating-point order, bitwise.  Inside a side, one
            # flat vectorized pass over cache-sized blocks with per-block
            # scratch reused across blocks (take(out=) / ufunc out=):
            # allocating the (block, 4, n_species) gathers per block cost
            # more in page faults than the arithmetic itself.  The Horner
            # chains keep the row-group implementation's exact operation
            # sequence; only the grouping of the Python-level work moved.
            for side in (0, 1):
                weight = (1 - w[ind]) if side == 0 else w[ind]
                row = row0 + side
                j = self._segment_indices(flat, row, t_lit)
                cb = flat.coef_base[row] + j
                dx = t_lit - flat.t[flat.knot_base[row] + j]
                for s in range(0, P, _QUERY_BLOCK):
                    e = min(s + _QUERY_BLOCK, P)
                    b = e - s
                    cbs = cb[s:e]
                    dxb = dx[s:e]
                    # per-plane direction take -> contiguous (b,) buffers
                    c0 = dir0.take(cbs, out=b_c0[:b])
                    c1 = dir1.take(cbs, out=b_c1[:b])
                    c2 = dir2.take(cbs, out=b_c2[:b])
                    c3 = dir3.take(cbs, out=b_c3[:b])
                    val = b_val[:b]
                    np.multiply(c0, dxb, out=val)
                    np.add(val, c1, out=val)
                    np.multiply(val, dxb, out=val)
                    np.add(val, c2, out=val)
                    np.multiply(val, dxb, out=val)
                    np.add(val, c3, out=val)
                    der = b_der[:b]
                    np.multiply(3.0, c0, out=der)
                    np.multiply(der, dxb, out=der)
                    np.multiply(2.0, c1, out=b_t[:b])
                    np.add(der, b_t[:b], out=der)
                    np.multiply(der, dxb, out=der)
                    np.add(der, c2, out=der)
                    # column stencils: one first-axis take of (b, 4, nsp)
                    cc = np.take(flat.columns, cbs, axis=0, out=cc_sten[:b])
                    x = dxb[:, None]
                    cols_val = tmp[:b]
                    np.multiply(cc[:, 0], x, out=cols_val)
                    np.add(cols_val, cc[:, 1], out=cols_val)
                    np.multiply(cols_val, x, out=cols_val)
                    np.add(cols_val, cc[:, 2], out=cols_val)
                    np.multiply(cols_val, x, out=cols_val)
                    np.add(cols_val, cc[:, 3], out=cols_val)
                    wb = weight[s:e]
                    value[s:e] += wb * val
                    derivative[s:e] += wb * der
                    col[s:e] += wb[:, None] * cols_val
            app[ind] = lower[ind] + (90.0 - lower[ind]) * value
            # d(app)/d(true) from the same interpolant that produced app.
            slope = (90.0 - lower[ind]) / (90.0 - critical[ind]) * derivative
            n_u = self.field.refractive_index(zz_unique)
            n = n_u[inverse[ind]]
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

    def app_of_true(self, height_km, true_elevation_deg):
        """Apparent elevation only -- the exact map :meth:`query_fields` reads.

        The projection-boundary solves evaluate the angular map tens of
        times per candidate beam; columns and gain would be dead weight
        there.  Reading through ``query_fields`` keeps a single
        interpolation authority, so the resolved boundary region and the
        discrete beam integrand cannot drift apart.
        """
        return self.query_fields(height_km, true_elevation_deg)[0]

    def query(self, height_km, true_elevation_deg, extinction):
        """Return ``(app_deg, T, gain_local, lit)`` for one extinction set.

        The table stores wavelength-independent species columns;
        ``extinction`` — the CALLER's cross-section weighting, ``(n_species,)``
        or ``(n_species, n_wl)`` — turns them into the pure attenuation
        ``T = exp(-columns . extinction)``, zeroed where shadowed; gain is
        returned separately (their product is never named transmittance and
        never clipped).  The table's own anchor extinction drives only the
        build certificates, never a query.
        """
        app, columns, gain, lit = self.query_fields(height_km, true_elevation_deg)
        tau = columns @ extinction
        if extinction.ndim == 2:
            transmittance = np.where(lit[..., None], np.exp(-tau), 0.0)
        else:
            transmittance = np.where(lit, np.exp(-tau), 0.0)
        return app, transmittance, gain, lit


# ---------------------------------------------------------------------------
# 4. token-cached geometry
# ---------------------------------------------------------------------------
class SolarGeometry:
    """Ray field, columns and (lazily) the point-source table of one medium.

    Cached per medium-construction token and refraction mode in a weak
    dictionary: the entry lives exactly as long as ANY living medium sharing
    the token (wavelength slices share the token with their parent).  The
    geometry keeps weak references to every medium that ever resolved it and
    resolves density queries against the first live one -- the entry and its
    table are reclaimed when the last sharing medium dies.
    """

    def __init__(self, medium, *, refracted: bool = True,
                 build_table: bool | None = None):
        self.medium_refs = [weakref.ref(medium)]
        self.refracted = bool(refracted)
        self.field = SolarRayField(medium, refracted=refracted)
        self.columns_tool = SolarSpeciesColumns(medium, self.field)
        self.ground_km = float(medium.z_bounds[0])
        self.top_km = float(medium.z_bounds[-1])
        self.n_species = self.columns_tool.n_species
        self._table: PointSourceTable | None = None
        self._critical_curve = None
        if build_table:
            self.ensure_table()

    def critical_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated critical-elevation curve, shared per construction.

        Batch classification and the adaptive-disk construction both consume
        this curve per scattering point, so it must be accurate where the
        boundary lives: near the ground the critical elevation falls off
        like ``-sqrt(2h/R)``, which is LINEAR in ``u = sqrt(h - ground)`` --
        the lattice is uniform in ``u`` below 3 km (linear interpolation is
        exact there to round-off) and progressively coarser above, with a
        PCHIP through the exact field roots on the ``u`` axis.  A linear
        height lattice here interpolated the 10 m critical elevation with a
        0.06 deg error (review round 5, Major cache geometry).
        """
        if self._critical_curve is None:
            ground = float(self.ground_km)
            top = float(self.top_km)
            u_top = np.sqrt(max(top - ground, 1e-9))
            u = np.unique(np.concatenate([
                np.linspace(0.0, np.sqrt(min(3.0, top - ground)), 121),
                np.linspace(np.sqrt(min(3.0, top - ground)),
                            np.sqrt(min(30.0, top - ground)), 41),
                np.linspace(np.sqrt(min(30.0, top - ground)), u_top, 45)]))
            heights = ground + u * u
            crits = np.array([self._exact_critical_true(float(h))
                              for h in heights])
            interp = PchipInterpolator(u, crits)

            def curve(h_arr):
                uu = np.sqrt(np.clip(np.asarray(h_arr, dtype=float) - ground,
                                     0.0, None))
                return interp(uu)

            self._critical_curve = curve
        return self._critical_curve

    def _exact_critical_true(self, height_km: float) -> float:
        field = self.field
        lower = float(field.grazing_apparent_elevation(np.array([height_km]))[0])
        return float(field.forward(np.array([height_km]),
                                   np.array([lower + 1e-6]))[0])

    def register_medium(self, medium) -> None:
        """Track another medium sharing this geometry's construction token."""
        if all(ref() is not medium for ref in self.medium_refs):
            self.medium_refs.append(weakref.ref(medium))
        self.columns_tool.register_medium(medium)

    def resolve_medium(self):
        """The first live medium sharing this geometry, or None."""
        live = [ref for ref in self.medium_refs if ref() is not None]
        self.medium_refs = live
        return live[0]() if live else None

    @property
    def table(self) -> PointSourceTable:
        if self._table is None:
            raise RuntimeError(
                "solar source table not built yet; call ensure_table() first")
        return self._table

    def ensure_table(self) -> PointSourceTable:
        """Build (once) the shared production table -- fixed configuration.

        The shared geometry serves exactly one table configuration: the
        production defaults of :class:`PointSourceTable`.  There is
        deliberately no configuration entry point (gen3-R1 review): the
        keyword arguments this method once accepted were consumed only on
        the first call, so a low-density probe silently re-tuned every
        later production query on the same medium -- a cache-identity
        violation (plan 3.2) that moved T by -5.4% in the reviewer's
        probe.  Custom-density probes build a standalone
        :class:`PointSourceTable` instead; the shared production table is
        not reconfigurable.
        """
        if self._table is None:
            medium = self.resolve_medium()
            if medium is None:
                raise RuntimeError("the owning medium is gone; rebuild the source")
            self._table = PointSourceTable(medium, self.field, self.columns_tool)
        return self._table

    def validate_medium(self, medium) -> None:
        if all(ref() is not medium for ref in self.medium_refs):
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
    else:
        geom.register_medium(medium)
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
                 n_radial: int = 2, n_azimuth: int = 8, chunk: int = 65536):
        medium = atmo._medium
        self._atmo_ref = weakref.ref(atmo)
        self._medium = medium
        self._scene = atmo._scene
        self.geometry = get_solar_geometry(medium, refracted=bool(atmo.refraction))
        self.geometry.register_medium(medium)
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
        self._plane_zero_cache: dict[float, float] = {}
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

    def _position_heights(self, x):
        """Heights of Cartesian query points under one shared domain contract.

        Every position-consuming kernel routes through here (gen3-R2):
        coordinates must be finite and the radius must lie between the
        opaque ground and the model top.  Only a +-1 m rounding band snaps
        to the nearest boundary -- MC ground intersections land there;
        anything beyond raises, matching the table's own query domain.
        Silent clipping used to map an underground point onto the surface
        scene and return nonzero illumination for it (plan 6.1:
        out-of-domain queries fail loudly, they are not reinterpreted).
        """
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("query positions must be finite")
        heights = (np.linalg.norm(x, axis=1) - R_EARTH) / 1e3
        ground = float(self._medium.z_bounds[0])
        top = float(self._medium.z_bounds[-1])
        tol = 1e-3
        if np.any(heights < ground - tol) or np.any(heights > top + tol):
            raise ValueError(
                "query position lies outside the model domain "
                f"[{ground}, {top}] km (beyond the 1 m rounding band)")
        return np.clip(heights, ground, top)

    def _classify_points(self, x):
        """Exact per-point band data — no rounding anywhere.

        Returns ``(in_band, radial, e_loc_deg, crit_deg, height_km)``; the
        round-5 review showed cached rounded geometry shifting the sun and
        re-cutting visible beams, so the construction consumes these exact
        values and nothing else.
        """
        radius = np.linalg.norm(x, axis=1)
        radial = x / np.maximum(radius[:, None], 1e-300)
        heights = self._position_heights(x)
        dot = np.clip(radial @ self.sun_hat, -1.0, 1.0)
        e_loc = np.degrees(np.arcsin(dot))
        # within 1e-12 of the local zenith/nadir the elevation's tangent
        # half-angle is below 1.4e-6 rad: pin to exactly +-90 degrees.
        # sqrt(1 - dot^2) there is pure cancellation noise (dot = 1 - eps
        # loses half its digits), which used to tilt the constructed disk
        # by ~1e-8 rad in a scene-dependent way and break rotation
        # equivariance (gen3-R3); the exact pole is the honest limit.
        e_loc = np.where(dot > 1.0 - 1e-12, 90.0, e_loc)
        e_loc = np.where(dot < -(1.0 - 1e-12), -90.0, e_loc)
        crit = self.geometry.critical_curve()(heights)
        in_band = np.abs(e_loc - crit) < self.radius_deg
        return in_band, radial, e_loc, crit, heights

    def _chunk_for_nbeam(self, n_beam, n_wl, full_spectrum):
        # Per-chunk fixed cost (frame + adaptive-grid construction + table
        # query) needs amortisation, but the floor must never override the
        # element budget that bounds transient memory.  Full-spectrum mode
        # is the memory contract the gen3-R1 review pinned: the 2.5M-element
        # budget decides alone (6144 x 81 -> 5 points; the old 64 floor
        # forced 1.2 GiB).  Single-wavelength arrays are n_beam x 1 (a
        # 64-point x 6144-beam chunk is ~3 MB), so the 64 amortisation
        # floor is memory-safe there -- dropping it collapses twilight MC
        # ground flushes to 10-point chunks and slowed the golden twilight
        # run ~20x.
        budget = 2_500_000 if full_spectrum else self.chunk
        per_point = n_beam * (n_wl if full_spectrum else 1)
        floor = 1 if full_spectrum else 64
        return max(floor, min(self.chunk, budget // max(per_point, 1)))

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

    def critical_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Lazily built critical-elevation curve, shared per construction."""
        return self.geometry.critical_curve()

    def shadow_height(self, true_elevation_deg: float) -> float:
        """Height where the vertical column enters Earth's shadow for one beam.

        Solves ``critical_true(z) = true_elevation_deg`` along the observer's
        vertical column by bisection (``critical_true`` decreases
        monotonically with height: from higher up, grazing rays dip deeper).
        Below the root the column point is occluded for that beam direction;
        above it the beam lights the point.  This -- not the perigee of some
        other ray -- is the occultation boundary every deck consumer must
        classify against (review round 4, Major cloud lit-segment).
        """
        elev = float(true_elevation_deg)
        if elev >= 0.0:
            raise ValueError(
                "shadow_height is the below-horizon occultation root; "
                "descending beams (elevation >= 0) enter through the top")
        z_low = float(self._medium.z_bounds[0])
        z_high = float(self._medium.z_bounds[-1])
        if elev < self._critical_true(z_high) - 1e-9:
            raise ValueError(
                f"beam elevation {elev:.4f} deg is below the deepest grazing "
                f"ray at the model top; the whole column is occluded")
        for _ in range(80):
            z_mid = 0.5 * (z_low + z_high)
            if self._critical_true(z_mid) > elev:
                z_low = z_mid      # grazing still above the beam: root higher
            else:
                z_high = z_mid
        return 0.5 * (z_low + z_high)

    def _center_plane_projection(self, plane, radial, s_loc, c_loc, height_km,
                                 e_loc_deg):
        """Projection of the apparent disk-CENTRE direction onto ``plane``.

        The apparent beam direction lives in the beam's vertical plane
        (``sin(app) * radial + cos(app) * h_hat``), so the centre projection
        is ``sin(app_c) * (n.radial) + cos(app_c) * n.h_hat_c`` with the
        apparent centre elevation read from the table -- the same angular
        map the beam integrand consumes.  Its magnitude decides whether the
        receiver plane cuts the disk interior (|projection| < sin alpha).
        At a vertical sun the horizontal component is undefined; the centre
        direction is then radial and the bare ``n.radial`` is exact.
        """
        a_dn = np.einsum("ij,ij->i", radial, plane)
        ns = plane @ self.sun_hat
        app_c = self.geometry.ensure_table().app_of_true(height_km, e_loc_deg)
        app_c = np.where(np.isfinite(app_c), app_c, 0.0)
        safe = c_loc > 1e-12
        ch_n = np.where(safe, (ns - s_loc * a_dn) / np.maximum(c_loc, 1e-300), 0.0)
        return np.where(safe,
                        np.sin(np.radians(app_c)) * a_dn
                        + np.cos(np.radians(app_c)) * ch_n,
                        a_dn)

    def _plane_zero_true_elevation(self, height_km):
        """True elevation whose apparent elevation is zero at ``height_km``.

        For a horizontal receiver the plane projection boundary is the
        apparent horizon (``app = 0``); the angular map is monotone, so one
        scalar root solve per unique height suffices and heights repeat
        across batches (ground flushes), so the roots are cached on the
        source.  A constant-index medium is its own identity map: the
        boundary is exactly true elevation zero.
        """
        cached = self._plane_zero_cache
        out = np.empty(height_km.shape)
        for h in np.unique(height_km):
            key = float(h)
            val = cached.get(key)
            if val is None:
                table = self.geometry.ensure_table()
                if table.constant_index:
                    val = 0.0
                else:
                    lower = self._critical_true(key)

                    def f(e, _h=key, _t=table):
                        return float(_t.app_of_true(np.array([_h]),
                                                    np.array([e]))[0])
                    a = lower + 1e-9
                    # at ground level the apparent horizon coincides with
                    # the shadow boundary (the grazing ray appears AT the
                    # horizon), so the lit angular map never crosses zero:
                    # the plane cut degenerates onto the earth cut -- pin
                    # the root to the boundary instead of bracketing
                    val = lower if f(a) >= 0.0 \
                        else brentq(f, a, 90.0, xtol=1e-9)
                cached[key] = val
            out = np.where(height_km == key, val, out)
        return out

    def _plane_interval(self, plane, radial, t, t_r, height_km, s_loc, c_loc,
                        alpha, lit_lo, lit_hi, plane_cuts):
        """Intersect the earth-lit radial interval with the receiver plane's
        ``apparent beam . normal`` sign interval, per point and azimuth.

        Along one azimuth the apparent beam direction sweeps its vertical
        plane monotonically as rho grows, so ``app . n`` has at most one
        zero on the cap; the zero is bracketed on ``[0, alpha]`` and
        bisected with the same apparent map the beam integrand consumes
        (the table), so the resolved region and the discrete integrand
        cannot drift.  Azimuths entirely above the plane keep the earth
        interval; entirely below collapse to an empty interval.  A plane
        containing the sweep degenerates to zero projection everywhere and
        integrates to zero, which is the correct grazing result.

        Horizontal receivers (``|normal . radial| = 1`` -- the production
        ground, MC and renderer seam) have ``app . n = +-sin(app)``: the
        boundary is the cached per-height apparent-horizon root and the
        per-azimuth bisection reuses the closed-form ``dz`` profile of the
        earth-shadow construction -- no angular-map evaluations in the
        loop.  Only genuinely tilted planes pay the generic per-iteration
        map evaluation (rare: validation DNI planes and experiments).
        """
        p_lo = np.zeros_like(t_r)
        p_hi = np.full_like(t_r, alpha)
        cuts = np.flatnonzero(plane_cuts)
        if cuts.size:
            a_dn = np.einsum("ij,ij->i", radial, plane)[cuts]
            horizontal = np.abs(a_dn) >= 1.0 - 1e-9
            flat = cuts[horizontal]
            tilt = cuts[~horizontal]
            if flat.size:
                sgn = np.sign(a_dn[horizontal])[:, None]
                tgt = np.sin(np.radians(
                    self._plane_zero_true_elevation(height_km[flat])))[:, None]
                s_l = s_loc[flat][:, None]
                t_r_i = t_r[flat]
                lo = np.zeros_like(t_r_i)
                hi = np.full_like(t_r_i, alpha)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    dz = np.cos(mid) * s_l + np.sin(mid) * t_r_i
                    move_lo = np.where(t_r_i > 0.0, dz < tgt, dz > tgt)
                    lo = np.where(move_lo, mid, lo)
                    hi = np.where(move_lo, hi, mid)
                root = 0.5 * (lo + hi)
                g0 = sgn * (s_l - tgt)
                ga = sgn * (np.cos(alpha) * s_l
                            + np.sin(alpha) * t_r_i - tgt)
                above_all = (g0 > 0.0) & (ga > 0.0)
                sub_lo = np.where(above_all, 0.0,
                                  np.where(ga >= 0.0, root, 0.0))
                sub_hi = np.where(above_all, alpha,
                                  np.where(ga >= 0.0, alpha,
                                           np.where(g0 > 0.0, root, 0.0)))
                p_lo[flat] = sub_lo
                p_hi[flat] = sub_hi
            if tilt.size:
                ns = plane @ self.sun_hat
                nt = np.einsum("naj,nj->na", t, plane)[tilt]
                app_map = self.geometry.ensure_table().app_of_true
                h = height_km[tilt]
                s_l = s_loc[tilt][:, None]
                t_r_i = t_r[tilt]
                a_dn_t = a_dn[~horizontal][:, None]

                def proj(cos_rho, sin_rho):
                    dz = cos_rho * s_l + sin_rho * t_r_i
                    eps = np.degrees(np.arcsin(np.clip(dz, -1.0, 1.0)))
                    app = np.radians(app_map(h[:, None], eps))
                    nd = cos_rho * ns[tilt][:, None] + sin_rho * nt
                    ct = np.sqrt(np.maximum(1.0 - dz * dz, 1e-300))
                    horizontal_part = (nd - a_dn_t * dz) / ct
                    return np.sin(app) * a_dn_t + np.cos(app) * horizontal_part

                g0 = proj(1.0, 0.0)
                ga = proj(np.cos(alpha), np.sin(alpha))
                lo = np.zeros_like(t_r_i)
                hi = np.full_like(t_r_i, alpha)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    gm = proj(np.cos(mid), np.sin(mid))
                    increasing = ga > g0
                    move_lo = np.where(increasing, gm < 0.0, gm > 0.0)
                    lo = np.where(move_lo, mid, lo)
                    hi = np.where(move_lo, hi, mid)
                root = 0.5 * (lo + hi)
                above_all = (g0 > 0.0) & (ga > 0.0)
                sub_lo = np.where(above_all, 0.0,
                                  np.where(ga >= 0.0, root, 0.0))
                sub_hi = np.where(above_all, alpha,
                                  np.where(ga >= 0.0, alpha,
                                           np.where(g0 > 0.0, root, 0.0)))
                p_lo[tilt] = sub_lo
                p_hi[tilt] = sub_hi
        lit_lo = np.maximum(np.clip(lit_lo, 0.0, alpha),
                            np.clip(p_lo, 0.0, alpha))
        lit_hi = np.minimum(np.clip(lit_hi, 0.0, alpha),
                            np.clip(p_hi, 0.0, alpha))
        return lit_lo, lit_hi

    def _local_grid(self, radial, u1, u2, e_loc_deg, crit_deg, n_r, n_az,
                    plane=None, height_km=None):
        """Exact per-point adaptive disk, closed form, computed per chunk.

        ``e_loc``/``crit`` are the point's own local sun elevation and
        critical elevation (degrees, exact -- no cache-key rounding: the
        round-5 review showed rounded values moving the sun and re-cutting
        visible beams).  In ``(rho, phi)`` cap coordinates the elevation
        profile along an azimuth is monotone, so each (point, azimuth) has
        exactly one crossing ``rho*`` of the critical elevation (vectorized
        bisection); the radial Gauss-Legendre rule covers that azimuth's
        lit interval exactly where the discontinuity sits.  Two topologies:
        centre shadowed (``delta < 0``) concentrates azimuths on the lit arc
        and lights ``[rho*, alpha]``; centre visible (``delta > 0``) spans
        the full circle and lights ``[0, rho*]``.  Weights keep the plain
        disk's centre-normal convention with each azimuth's true measure.

        With ``plane`` (per-point receiver normals; the flux kernels) the
        earth-lit interval is additionally intersected with the receiver
        plane's ``apparent beam . normal`` sign interval
        (:meth:`_plane_interval`) -- a second physical cut the coarse disk
        cannot resolve -- and affected points span the full azimuth circle
        because that boundary is not confined to the earth-lit arc.
        ``plane=None`` (scatter) keeps the earth-only construction
        bit-for-bit.

        Returns ``(directions (n, nb, 3) earth-frame, weights (n, nb),
        sin_elev (n, nb))``.
        """
        alpha = np.radians(self.radius_deg)
        e_r = np.radians(e_loc_deg)
        c_r = np.radians(crit_deg)
        s_loc, c_loc = np.sin(e_r), np.cos(e_r)
        centre_lit = (e_loc_deg - crit_deg) > 0.0
        # a receiver plane cutting the disk interior needs the full circle:
        # its boundary is not confined to the earth-lit arc
        if plane is not None:
            plane_cuts = np.abs(self._center_plane_projection(
                plane, radial, s_loc, c_loc, height_km, e_loc_deg)) \
                < np.sin(alpha)
        # azimuths: centre visible -> full circle; centre shadowed -> the
        # lit arc |phi| < phi_half where the rim elevation still exceeds crit.
        # Both branches are per point -- an in-band chunk mixes topologies.
        den = c_loc * np.sin(alpha)
        cos_ph_half = np.where(den > 1e-300,
                               (np.sin(c_r) - s_loc * np.cos(alpha))
                               / np.where(den > 1e-300, den, 1.0), 1.0)
        phi_half = np.where(centre_lit, np.pi,
                            np.arccos(np.clip(cos_ph_half, -1.0, 1.0)))
        if plane is not None:
            phi_half = np.where(plane_cuts, np.pi, phi_half)
        az_measure_ratio = np.where(centre_lit, 1.0 / n_az,
                                    phi_half / (np.pi * n_az))
        # phi grid per point, centred on the local sun-tilt direction
        j = (np.arange(n_az) + 0.5)[None, :]
        phi = -phi_half[:, None] + (2.0 * phi_half[:, None] / n_az) * j
        # per-point earth-frame cap basis: t(phi) = cos(phi) t1 + sin(phi) u2
        # with t1 = c_loc * radial - s_loc * u1 the tilt toward higher
        # elevation; t1 and u2 are both exactly orthogonal to sun_hat, so
        # cos(rho) sun_hat + sin(rho) t is a unit direction for every beam
        cos_phi = np.cos(phi)[:, :, None]
        sin_phi = np.sin(phi)[:, :, None]
        t1 = c_loc[:, None] * radial - s_loc[:, None] * u1        # (n, 3)
        t = cos_phi * t1[:, None, :] + sin_phi * u2[:, None, :]   # (n, n_az, 3)
        # radial component of t(phi) is frame-free: radial*t1 = c_loc and
        # radial*u2 = 0, so sin(e_b) = cos(rho) s_loc + sin(rho) c_loc cos(phi)
        # -- the profile elevation along each azimuth, monotone in rho
        t_r = c_loc[:, None] * np.cos(phi)                        # (n, n_az)
        rho_lo = np.zeros((e_r.shape[0], n_az))
        rho_hi = np.full((e_r.shape[0], n_az), alpha)
        for _ in range(60):
            mid = 0.5 * (rho_lo + rho_hi)
            # dz stays (n, n_az): a trailing axis here would promote the
            # bisection state one rank per iteration and hang the loop
            dz = np.cos(mid) * s_loc[:, None] + np.sin(mid) * t_r
            g = np.arcsin(np.clip(dz, -1.0, 1.0)) - c_r[:, None]
            increasing = t_r > 0.0
            move_lo = np.where(increasing, g < 0.0, g > 0.0)
            rho_lo = np.where(move_lo, mid, rho_lo)
            rho_hi = np.where(move_lo, rho_hi, mid)
        rho_star = 0.5 * (rho_lo + rho_hi)
        ga = np.arcsin(np.clip(
            np.cos(alpha) * s_loc[:, None] + np.sin(alpha) * t_r,
            -1.0, 1.0)) - c_r[:, None]
        lit = centre_lit[:, None]
        lit_lo = np.where(lit, 0.0, rho_star)         # centre lit: inner
        lit_hi = np.where(lit, np.where(ga >= 0.0, alpha, rho_star), alpha)
        # never-lit azimuths (empty interval) keep their slot with zero
        # dc-span, so every point of the chunk shares the rectangular
        # (n_az, n_r) beam layout
        lit_lo = np.clip(lit_lo, 0.0, alpha)
        lit_hi = np.clip(lit_hi, 0.0, alpha)
        if plane is not None:
            lit_lo, lit_hi = self._plane_interval(
                plane, radial, t, t_r, height_km, s_loc, c_loc,
                alpha, lit_lo, lit_hi, plane_cuts)
        # radial Gauss-Legendre nodes per azimuth's lit interval, in
        # cos(rho) where the cap area measure is linear; the dc-span factor
        # carries each azimuth's lit fraction of the cap
        x_g, w_g = np.polynomial.legendre.leggauss(max(2, int(n_r)))
        cos_lo = np.cos(lit_lo)                           # >= cos_hi
        cos_hi = np.cos(lit_hi)
        c_grid = cos_hi[:, :, None] + (cos_lo - cos_hi)[:, :, None] \
            * (0.5 * (x_g + 1.0))[None, None, :]           # (n, n_az, n_r)
        dc = np.maximum(cos_lo - cos_hi, 0.0)
        dc_total = 1.0 - np.cos(alpha)
        w = (dc[:, :, None] / dc_total) * w_g[None, None, :] \
            * (az_measure_ratio[:, None, None] / (1.0 + np.cos(alpha)))
        rho = np.arccos(np.clip(c_grid, -1.0, 1.0))
        # earth-frame directions: d = cos(rho) sun_hat + sin(rho) t(phi);
        # the arc azimuths are measured from the local sun-tilt direction
        directions = (np.cos(rho)[:, :, :, None] * self.sun_hat[None, None, None, :]
                      + np.sin(rho)[:, :, :, None] * t[:, :, None, :]).reshape(
                          e_loc_deg.shape[0], n_az * n_r, 3)
        sin_elev = (np.cos(rho) * s_loc[:, None, None]
                    + np.sin(rho) * t_r[:, :, None]).reshape(
                        e_loc_deg.shape[0], n_az * n_r)
        weights = w.reshape(e_loc_deg.shape[0], n_az * n_r)
        return directions, weights, sin_elev

    def disk_for_column(self, height_km: float, *, n_r: int = 16,
                        n_az: int = 384):
        """Adaptive disk in the EARTH frame for a point on the observer's
        vertical column (decks): the column's local frame is the observer
        frame, so one exact construction serves the slab."""
        curve = self.geometry.critical_curve()
        z = np.array([float(height_km)])
        crit = float(curve(z)[0])
        e_loc = 90.0 - self.sun_zenith_deg
        if abs(e_loc - crit) >= self.radius_deg:
            return self.disk()
        up = np.array([0.0, 0.0, 1.0])
        radial = up[None, :]
        hz = self.sun_hat - np.dot(self.sun_hat, up) * up
        n = np.linalg.norm(hz)
        u1 = (hz / n)[None, :] if n > 1e-12 \
            else np.array([[np.sin(np.radians(self.sun_zenith_deg + 1.0)), 0.0, 0.0]])
        u2 = np.cross(radial, u1)
        e_loc_arr = np.array([e_loc])
        crit_arr = np.array([crit])
        directions, weights, _ = self._local_grid(
            radial, u1, u2, e_loc_arr, crit_arr, n_r, n_az)
        return directions[0], weights[0]

    def vertex(self, height_km, true_elevation_deg, *, iw=None):
        """Table query for arbitrary (height, true elevation) points.

        Returns ``(app_deg, T, gain_local, lit)``; builds the table on first
        use.  ``T`` has a final wavelength axis unless ``iw`` is given.  The
        attenuation is evaluated with THIS atmosphere's cross sections -- the
        geometry (and its table) is shared across wavelength subsets, so the
        table's own stored extinction belongs to whichever construction built
        it and must never leak into another medium's query.
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
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

    def _geometry_arrays(self, x, iw, disk=None, local=False):
        """Per-point per-beam geometry: true elevations, app, T, weighted gain.

        ``x`` is ``(n, 3)`` in the Earth-centred frame with the observer on
        the +z axis.  ``disk`` overrides the beam set (contact-band refined
        flux kernels); the gain carries the disk weights and -- unless
        ``local`` -- the single observer transfer ``(n_obs / n_point)^2``;
        nothing else in the chain may apply an ``n^2`` factor.  ``local``
        reads the plane flux at the point itself (SOS boundaries), which
        must not depend on where the facade's observer happens to be.
        """
        directions, weights = disk if disk is not None else self.disk()
        medium = self._medium
        radius = np.linalg.norm(x, axis=1)
        radial = x / radius[:, None]
        height = self._position_heights(x)
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
        gain = gain * weights
        if not local:
            gain = gain * (self.observer_n / self.geometry.field
                           .refractive_index(height)[:, None]) ** 2
        return radial, directions, true_mu, app_rad, transmission, gain, lit

    @staticmethod
    def _apparent_plane_projection(normal, radial, directions, sin_true,
                                   app_rad):
        """Apparent-direction flux projection onto the unit-normal plane.

        Refraction preserves each beam's horizontal direction, so the
        apparent beam direction is ``sin(app) * radial + cos(app) * h_hat``
        with ``h_hat`` the TRUE beam's horizontal unit vector; the plane
        projection follows without building apparent vectors.  Shared by
        the coarse and contact-band branches so the two cannot drift
        (review gen2 R1: the coarse branch broadcast beam 0's projection
        to the whole disk).
        """
        if normal.ndim == 2:                           # per-point normals (n, 3)
            n_dot_radial = np.einsum("ij,ij->i", radial, normal)[:, None]
            if directions.ndim == 2:                   # shared disk (nb, 3)
                n_dot_beam = (directions @ normal.T).T             # (n, nb)
            else:                                      # per point (n, nb, 3)
                n_dot_beam = np.einsum("nbj,nj->nb", directions, normal)
        else:                                          # one shared normal (3,)
            n_dot_radial = np.einsum("ij,j->i", radial, normal)[:, None]
            if directions.ndim == 2:                   # shared disk (nb, 3)
                n_dot_beam = np.einsum("bj,j->b", directions, normal)[None, :]
            else:                                      # per point (n, nb, 3)
                n_dot_beam = np.einsum("nbj,j->nb", directions, normal)
        cos_true = np.sqrt(np.maximum(1.0 - sin_true ** 2, 0.0))
        # A beam within 1e-8 rad of the local vertical has no horizontal
        # direction: sqrt(1 - sin_true^2) there is pure cancellation noise
        # (sin_true rounds to +-1, the denominator dives to ~1e-150) while
        # the numerator is the residue of two ~equal dot products (~1e-16),
        # so the naive division amplified a single beam to ~1e116 times the
        # solar irradiance at perfectly legal query points (gen3-R4).  The
        # honest radial limit is exact -- the projection is
        # sin(app) * (n . radial) alone -- and the discarded true
        # horizontal term is bounded by cos(app) * 1e-8, far below every
        # budget in this contract.
        radial_lim = cos_true < 1e-8
        horizontal = (n_dot_beam - n_dot_radial * sin_true) \
            / np.where(radial_lim, 1.0, cos_true)
        horizontal = np.where(radial_lim, 0.0, horizontal)
        return np.sin(app_rad) * n_dot_radial + np.cos(app_rad) * horizontal

    def _local_frame_basis(self, x):
        """Per-point local frame: radial, sun-horizontal and cross vectors.

        ``u1`` points along the sun's horizontal component at the point,
        ``u2 = radial x u1``; ``radial``/``u1``/``u2`` are exactly orthonormal
        in both branches.  A point whose local sun elevation is +-90 degrees
        (sun on the vertical) has no horizontal component: the fallback picks,
        per point, the coordinate axis least aligned with ``radial``, projects
        it into the tangent plane and normalises.  The former fixed ``+x``
        fallback was not orthogonal to ``radial`` -- it could even be parallel
        -- so the constructed cap stopped being a unit disk and any plane flux
        depended on the global orientation, up to a false zero (gen3-R3;
        plan 6.1 rotation equivariance and zenith degeneracy).
        """
        radius = np.linalg.norm(x, axis=1)
        radial = x / np.maximum(radius[:, None], 1e-300)
        dot = np.einsum("j,ij->i", self.sun_hat, radial)
        hz = self.sun_hat[None, :] - dot[:, None] * radial
        n = np.linalg.norm(hz, axis=1)
        axis = np.argmin(np.abs(radial), axis=1)
        e = np.zeros_like(radial)
        e[np.arange(radial.shape[0]), axis] = 1.0
        fallback = e - np.einsum("ij,ij->i", e, radial)[:, None] * radial
        fallback /= np.maximum(
            np.linalg.norm(fallback, axis=1), 1e-300)[:, None]
        safe = (n > 1e-12)[:, None]
        u1 = np.where(safe, hz / np.maximum(n, 1e-300)[:, None], fallback)
        u2 = np.cross(radial, u1)
        return radial, u1, u2

    def scatter(self, x, backward, *, iw=None, ray_fraction, aerosol_fractions,
                disk=None):
        """Solar angular integral of ``T * G * p_mix`` over the disk beams.

        For scalar ``iw`` the fractions have shapes ``(n,)`` / ``(n, n_ae)``
        and the output is ``(n,)``; for ``iw=None`` they have ``(n, n_wl)``
        / ``(n, n_ae, n_wl)`` and the output is ``(n, n_wl)``.  The phase
        argument is the *apparent* beam direction composed with the backward
        ray, not the geometric one.  ``disk`` overrides the per-point beam
        selection with an earth-frame set (validation reference quadratures
        only; production callers get the exact per-point contact-band grid).
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        n_wl = len(atmo.wl)
        out = np.zeros((len(x), n_wl)) if iw is None else np.zeros(len(x))
        wavelengths = atmo.wl if iw is None else np.array([atmo.wl[iw]])
        if disk is not None:
            self._scatter_coarse_chunk(x, backward, np.arange(len(x)),
                                       disk[0], disk[1], iw, wavelengths, atmo,
                                       ray_fraction, aerosol_fractions, out)
            self.n_points += len(x)
            return out
        in_band, radial, e_loc, crit, _h = self._classify_points(x)
        n_r, n_az = 8, 64
        n_beam = n_r * n_az
        # out-of-band points: the coarse production disk (their per-beam lit
        # drop is exact whenever the boundary does not cut the cap)
        coarse_idx = np.flatnonzero(~in_band)
        if coarse_idx.size:
            step = self._chunk_for_nbeam(16, n_wl, iw is None)
            for st in range(0, coarse_idx.size, step):
                pos = coarse_idx[st:st + step]
                dsk = self.disk()
                self._scatter_coarse_chunk(x[pos], backward[pos], pos,
                                           dsk[0], dsk[1], iw, wavelengths,
                                           atmo, ray_fraction,
                                           aerosol_fractions, out)
        # in-band points: exact per-point grid, chunked by the elements budget
        band_idx = np.flatnonzero(in_band)
        if band_idx.size:
            step = self._chunk_for_nbeam(n_beam, n_wl, iw is None)
            for st in range(0, band_idx.size, step):
                pos = band_idx[st:st + step]
                self._scatter_local_chunk(x[pos], backward[pos], pos, n_r, n_az,
                                          iw, wavelengths, atmo, ray_fraction,
                                          aerosol_fractions, out)
        self.n_points += len(x)
        return out

    def _scatter_coarse_chunk(self, x_chunk, backward_chunk, pos, directions,
                              weights, iw, wavelengths, atmo, ray_fraction,
                              aerosol_fractions, out):
        radial, _, mu, app, transmission, gain, lit = self._geometry_arrays(
            x_chunk, iw, disk=(directions, weights))
        active = np.flatnonzero(np.any(lit, axis=1))
        if not active.size:
            return
        # Twilight has many fully shadowed collisions; skipping their Mie
        # phase evaluations does not change any sampled path.
        pos = pos[active]
        radial, mu, app = radial[active], mu[active], app[active]
        transmission, gain = transmission[active], gain[active]
        # rows of the passed chunk follow ``active``, not ``pos``: ``pos``
        # indexes the caller's full batch (coarse points are a filtered,
        # non-contiguous subset there), so the backward rays are the chunk's
        # own rows in the same order
        bw = backward_chunk[active]
        dr = np.einsum("ij,ij->i", bw, radial)[:, None]
        horizontal_dot = (bw @ directions.T
                          - mu * dr) / np.sqrt(np.maximum(1 - mu ** 2, 1e-30))
        phase_mu = np.cos(app) * horizontal_dot + np.sin(app) * dr
        phase_mu = np.where(np.abs(mu) > 1 - 1e-14, np.sign(mu) * dr, phase_mu)
        phase_mu = np.clip(phase_mu, -1.0, 1.0)
        p_ray = spectra.rayleigh_phase(phase_mu)
        if iw is None:
            phase = p_ray[..., None] * ray_fraction[pos, None, :]
        else:
            phase = p_ray * ray_fraction[pos, None]
        for j, layer in enumerate(atmo.aerosols):
            aerosol_phase = np.asarray(
                layer.aerosol.phase(phase_mu.ravel(), wavelengths))
            if aerosol_phase.ndim == 1:
                aerosol_phase = np.broadcast_to(
                    aerosol_phase.reshape(phase_mu.shape + (1,)),
                    phase_mu.shape + (len(wavelengths),))
            else:
                aerosol_phase = aerosol_phase.reshape(
                    phase_mu.shape + (len(wavelengths),))
            if iw is None:
                phase = phase + aerosol_phase * aerosol_fractions[pos, j, None, :]
            else:
                phase = phase + aerosol_phase[..., 0] * aerosol_fractions[pos, j, None]
        if iw is None:
            out[pos] = np.sum(transmission * gain[..., None] * phase, axis=1)
        else:
            out[pos] = np.sum(transmission * gain * phase, axis=1)

    def _scatter_local_chunk(self, x_chunk, backward_chunk, pos, n_r, n_az,
                             iw, wavelengths, atmo, ray_fraction,
                             aerosol_fractions, out):
        """One chunk of in-band points on the exact per-point grid: the
        construction, the vertex query and the phase evaluation are all
        row-wise in the point, so chunking never changes a value."""
        radial, u1, u2 = self._local_frame_basis(x_chunk)
        in_band, radial, e_loc, crit, _h = self._classify_points(x_chunk)
        directions, weights, sin_e = self._local_grid(
            radial, u1, u2, e_loc, crit, n_r, n_az)
        true_e = np.degrees(np.arcsin(sin_e))
        medium = self._medium
        height = self._position_heights(x_chunk)
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
        gain = gain * (self.observer_n / local_n[:, None]) ** 2 * weights
        active = np.flatnonzero(np.any(lit, axis=1))
        if not active.size:
            return
        gpos = pos[active]
        bw = backward_chunk[active]
        rad_a = radial[active]
        dr = np.einsum("ij,ij->i", bw, rad_a)[:, None]
        # horizontal projection of each point-beam direction: subtract the
        # radial component of the backward ray, then project onto the
        # per-point directions (mu = sin e_b from the construction); every
        # row follows ``active`` so the skipped fully-shadowed points keep
        # the chunk's arrays aligned
        horizontal_dot = np.einsum(
            "ij,ibj->ib",
            bw - rad_a * dr,
            directions[active]) \
            / np.sqrt(np.maximum(1.0 - sin_e[active] ** 2, 1e-30))
        phase_mu = np.cos(app_rad[active]) * horizontal_dot \
            + np.sin(app_rad[active]) * dr
        phase_mu = np.clip(phase_mu, -1.0, 1.0)
        p_ray = spectra.rayleigh_phase(phase_mu)
        if iw is None:
            phase = p_ray[..., None] * ray_fraction[gpos, None, :]
        else:
            phase = p_ray * ray_fraction[gpos, None]
        for j, layer in enumerate(atmo.aerosols):
            aerosol_phase = np.asarray(
                layer.aerosol.phase(phase_mu.ravel(), wavelengths))
            if aerosol_phase.ndim == 1:
                aerosol_phase = np.broadcast_to(
                    aerosol_phase.reshape(phase_mu.shape + (1,)),
                    phase_mu.shape + (len(wavelengths),))
            else:
                aerosol_phase = aerosol_phase.reshape(
                    phase_mu.shape + (len(wavelengths),))
            if iw is None:
                phase = phase + aerosol_phase * aerosol_fractions[gpos, j, None, :]
            else:
                phase = phase + aerosol_phase[..., 0] * aerosol_fractions[gpos, j, None]
        if iw is None:
            out[gpos] = np.sum(transmission[active] * gain[active][..., None]
                               * phase, axis=1)
        else:
            out[gpos] = np.sum(transmission[active] * gain[active] * phase,
                               axis=1)

    def _flux_band(self, x, normals):
        """Points whose flux needs the exact per-point grid.

        The shared coarse disk is exact only while neither physical
        boundary crosses the disk interior: the earth-shadow boundary
        (``|e_loc - crit| < alpha``) or the receiver-plane projection
        (|apparent centre . normal| < sin alpha -- gen3-R1 review).  Out of
        both bands every beam sits on one side of both cuts and the
        per-beam ``max(., 0)`` drop is exact.

        A fully shadowed disk (``e_loc - crit <= -alpha``) has zero flux
        through every kernel, so the plane test is masked by
        ``lit_possible`` there -- both routes return zero and the exact
        grid must not be spent on dark twilight ground points.  (The
        centre's apparent reading below the shadow boundary is the grazing
        placeholder, not the sun's position, so the raw plane test would
        misflag every such point.)
        """
        in_band, radial, e_loc, crit, heights = self._classify_points(x)
        e_r = np.radians(e_loc)
        adn_c = self._center_plane_projection(normals, radial,
                                              np.sin(e_r), np.cos(e_r),
                                              heights, e_loc)
        lit_possible = (e_loc - crit) > -self.radius_deg
        plane_band = (np.abs(adn_c) < np.sin(np.radians(self.radius_deg))) \
            & lit_possible
        return in_band | plane_band

    def _flux_local_chunk(self, x_chunk, normals, pos, n_r, n_az, iw, f_inf,
                          out, local):
        """One chunk of flux-band points on the exact per-point grid.

        ``ground`` and ``direct_irradiance`` share this kernel: both the
        earth-shadow and the receiver-plane boundaries are resolved by the
        construction, so ``max(., 0)`` only removes sign noise at the
        roots.  ``f_inf`` is ``None`` for ``ground`` (normalised integral,
        no spectrum) and the ``F_inf`` grid for ``direct_irradiance``.
        """
        radial, u1, u2 = self._local_frame_basis(x_chunk)
        in_band, radial, e_loc, crit, heights = self._classify_points(x_chunk)
        directions, weights, sin_e = self._local_grid(
            radial, u1, u2, e_loc, crit, n_r, n_az,
            plane=normals, height_km=heights)
        true_e = np.degrees(np.arcsin(sin_e))
        medium = self._medium
        app, columns, gain, lit = self.geometry.ensure_table().query_fields(
            heights[:, None], true_e)
        ext = species_extinction_weights(medium)
        app_rad = np.radians(app)
        adn = self._apparent_plane_projection(normals, radial, directions,
                                              sin_e, app_rad)
        gain = gain * weights
        if not local:
            gain = gain * (self.observer_n / self.geometry.field
                           .refractive_index(heights)[:, None]) ** 2
        if iw is None:
            tau = columns @ ext
            transmission = np.where(lit[..., None], np.exp(-tau), 0.0)
            weighted = transmission * gain[..., None] \
                * np.maximum(adn[..., None], 0.0)          # (n,nb,n_wl)
            total = weighted.sum(axis=1)
            out[pos] = total * f_inf[None, :] if f_inf is not None else total
        else:
            tau = columns @ ext[:, int(iw)]
            transmission = np.where(lit, np.exp(-tau), 0.0)
            weighted = transmission * gain * np.maximum(adn, 0.0)
            total = weighted.sum(axis=1)
            out[pos] = total * f_inf[iw] if f_inf is not None else total

    def _flux(self, x, normals, *, iw, local, f_inf):
        """Shared flux driver of ``ground`` and ``direct_irradiance``.

        ``normals`` are per-point unit plane normals: ``ground`` passes the
        local vertical, ``direct_irradiance`` one shared normal broadcast
        over the points.  Flux-band points (either boundary cutting the
        disk) go through the exact per-point kernel; the rest keep the
        shared coarse disk, exact there because every beam sits on one
        side of both boundaries.
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        out = np.zeros(len(x)) if iw is not None else np.zeros((len(x), len(atmo.wl)))
        band = self._flux_band(x, normals)
        n_beam = 16 * 384
        step = self._chunk_for_nbeam(n_beam, len(atmo.wl), iw is None)
        coarse_idx = np.flatnonzero(~band)
        if coarse_idx.size:
            dsk = self.disk()
            for st in range(0, coarse_idx.size, step):
                pos = coarse_idx[st:st + step]
                directions = dsk[0]
                _, _, _, app, transmission, gain, _ = self._geometry_arrays(
                    x[pos], iw, disk=(directions, dsk[1]), local=local)
                radial = x[pos] / np.linalg.norm(x[pos], axis=1)[:, None]
                sin_true = np.clip(radial @ directions.T, -1, 1)
                app_dot_normal = self._apparent_plane_projection(
                    normals[pos], radial, directions, sin_true, app)
                if iw is None:
                    gain = gain[..., None]
                    app_dot_normal = app_dot_normal[..., None]
                weighted = transmission * gain * np.maximum(app_dot_normal, 0.0)
                total = weighted.sum(axis=1)
                if iw is None:
                    out[pos] = (total * f_inf[None, :] if f_inf is not None
                                else total)
                else:
                    out[pos] = (total * f_inf[iw] if f_inf is not None
                                else total)
        band_idx = np.flatnonzero(band)
        for st in range(0, band_idx.size, step):
            pos = band_idx[st:st + step]
            self._flux_local_chunk(x[pos], normals[pos], pos, 16, 384, iw,
                                   f_inf, out, local)
        return out

    def ground(self, x, *, iw=None, local=False):
        """Solar angular integral of ``T*G*max(sin app, 0)`` on the horizontal.

        Default: detector-referred through the single ``n^2`` observer
        transfer (renderer/ground-display seam).  ``local=True`` reads the
        horizontal flux at the point itself -- no observer dependence --
        which is the accounting SOS's bottom boundary must use to match its
        layer sources (review gen2 R1).  The integral shares the
        direct-irradiance kernel and resolves both the earth-shadow and
        the horizon-plane projection boundary of the disk (gen3-R1): a
        disk straddling the plane no longer relies on coarse per-beam
        ``max`` truncation.
        """
        x = np.asarray(x, dtype=float)
        radial = x / np.linalg.norm(x, axis=1)[:, None]
        return self._flux(x, np.ascontiguousarray(radial), iw=iw, local=local,
                          f_inf=None)

    def direct_irradiance(self, x, normal, *, iw=None, local=False):
        """Spectral flux through a plane at the points ``x``.

        ``normal`` is the plane's unit normal in the Earth-centred frame;
        the flux is ``F_inf * sum_beams w*T*G*max(apparent_beam . normal, 0)``.
        By default the flux is detector-referred through the single
        ``n^2`` observer transfer; ``local=True`` reads the plane flux at
        the point itself (SOS boundaries), which must not depend on where
        the facade's observer happens to be.  The full-spectrum default
        (renderer seam) and the per-wavelength calls share one (point,
        beam) projection; only the spectral multiplication gains a
        wavelength axis.  When the disk straddles the plane (or the earth
        shadow), the per-point grid resolves the exact boundary region
        instead of coarse ``max`` truncation (gen3-R1).
        """
        atmo = self._atmo_ref()
        if atmo is None:
            raise RuntimeError("the owning atmosphere is gone; rebuild the source")
        self._validate(atmo)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        f_inf = spectra.solar_irradiance(atmo.wl)
        normals = np.ascontiguousarray(np.broadcast_to(normal, (len(x), 3)))
        return self._flux(np.asarray(x, dtype=float), normals, iw=iw,
                          local=local, f_inf=f_inf)

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
