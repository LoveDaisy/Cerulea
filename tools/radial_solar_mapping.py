"""Experimental solar-angle mapping by independent radial refraction integrals.

This is a numerical reference for a monotonic ``q(r) = n(r) r`` atmosphere,
not a replacement for production ``skycolor.refraction``.  It integrates
ray curvature in radius instead of stepping Cartesian RK4 trajectories.
Only rays which clear the opaque model bottom are evaluated; others return
NaN.  The atmosphere above the supplied profile top is treated as constant n.

Example::

    mapping = RadialSolarMapping(atmo, n_quad=16)
    true_elevation = mapping.forward([38., 47.], [-5.48734, -5.76770])
    gain = mapping.beam_gain([38., 47.], [-5.48734, -5.76770])

Angles/heights are degrees/km in the public API.  Array inputs broadcast.
The beam gain is a point-source, single-image angular Jacobian; finite-disk
partial occultation and diffuse-path refraction are deliberately not included.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from skycolor.constants import R_EARTH
from skycolor.refraction import RefractiveProfile


class RadialSolarMapping:
    """Vectorised forward solar-angle mapping for one density definition.

    Parameters
    ----------
    atmo
        Defines the density profile and opaque ground ``r_bounds[0]``.
    profile
        Optional prebuilt object exposing one-dimensional ``r`` and ``n``
        arrays.  A new CubicSpline is built from these arrays, so the radial
        integration is independent of the RK4 implementation, but the two
        methods intentionally use the same refractive-index profile.
    n_quad
        Gauss-Legendre nodes per fixed physical-height segment.
    segment_km
        Segment spacing below 120 km.  Spline-knot interpolation error is
        separate from this integration spacing and ``n_quad``.
    chunk
        Maximum rays handled together, bounding integration-array memory.

    Notes
    -----
    Requires positive n and strictly increasing n*r; ducting / multi-image
    media are rejected.  The constructor checks every interval's extrema of
    d(n*r)/dr, including between profile knots, before inverting n*r.
    """

    def __init__(self, atmo, *, profile=None, n_quad: int = 16,
                 segment_km: float = 0.5, chunk: int = 128):
        profile = RefractiveProfile(atmo) if profile is None else profile
        self.r = np.asarray(profile.r, dtype=float)
        self.n = np.asarray(profile.n, dtype=float)
        self.r_ground = float(atmo.r_bounds[0])
        if self.r.ndim != 1 or self.r.size < 4 or self.n.shape != self.r.shape:
            raise ValueError("profile r/n must be matching 1-D arrays with >=4 nodes")
        if not np.all(np.diff(self.r) > 0) or not np.all(self.n > 0):
            raise ValueError("profile radii must increase and refractive index must be positive")
        if not self.r[0] <= self.r_ground < self.r[-1]:
            raise ValueError("opaque ground must lie inside the refractive profile")
        if n_quad < 2 or segment_km <= 0 or chunk < 1:
            raise ValueError("n_quad>=2, segment_km>0 and chunk>=1 are required")
        self._n = CubicSpline(self.r, self.n)
        self._dn = self._n.derivative()
        self._d2n = self._n.derivative(2)
        # n is cubic per interval, hence q'=d(n*r)/dr is cubic and its
        # extrema are the real roots of a quadratic.  Testing those plus
        # knots establishes monotonic q over the entire spline, rather
        # than assuming monotonic tabulation nodes exclude a narrow duct.
        c0, c1, c2, _ = self._n.c
        ri = self.r[:-1]
        qa = 12.0 * c0
        qb = 6.0 * (ri * c0 + c1)
        qc = 2.0 * (ri * c1 + c2)
        disc = qb * qb - 4.0 * qa * qc
        roots = []
        for sign in (-1.0, 1.0):
            root = np.divide(-qb + sign * np.sqrt(np.maximum(disc, 0.0)),
                             2.0 * qa, out=np.full_like(qa, np.nan), where=qa != 0.0)
            mask = (disc >= 0.0) & (root > 0.0) & (root < np.diff(self.r))
            roots.append(ri[mask] + root[mask])
        root = np.divide(-qc, qb, out=np.full_like(qb, np.nan),
                         where=(qa == 0.0) & (qb != 0.0))
        mask = (root > 0.0) & (root < np.diff(self.r))
        roots.append(ri[mask] + root[mask])
        check = np.concatenate([self.r, *roots])
        if np.any(self._n(check) + check * self._dn(check) <= 0):
            raise ValueError("n(r)*r must increase: ducts/multiple turning points are unsupported")
        self.n_quad = int(n_quad)
        self.chunk = int(chunk)
        self.r_top = float(self.r[-1])
        self.n_inf = float(self.n[-1])
        self.q_ground = float(self._n(self.r_ground) * self.r_ground)
        # The inverse table is a root initialiser, never the root authority.
        self._q_nodes = self.r * self.n
        if np.any(np.diff(self._q_nodes) <= 0):
            raise ValueError("profile n*r samples must increase")
        h0 = (self.r_ground - R_EARTH) / 1e3
        htop = (self.r_top - R_EARTH) / 1e3
        levels = np.unique(np.concatenate([
            np.arange(0.0, min(120.0, htop) + segment_km, segment_km),
            np.arange(120.0, min(200.0, htop) + 5.0, 5.0),
            np.arange(200.0, htop + 100.0, 100.0),
            [h0, htop],
        ]))
        levels = levels[(levels >= h0) & (levels <= htop)]
        # Curvature derivatives are piecewise polynomials. Integrating across
        # spline knots can leave tiny angle errors which become large noise
        # when differentiated for the beam Jacobian near a turning point.
        knots = self.r[(self.r >= self.r_ground) &
                       (self.r <= min(self.r_top, R_EARTH + 200e3))]
        self._edges = np.unique(np.r_[R_EARTH + levels * 1e3, knots])
        self._gx, self._gw = np.polynomial.legendre.leggauss(self.n_quad)

    def refractive_index(self, height_km):
        """Index of the supplied profile at height(s), with no extrapolation."""
        radius = R_EARTH + np.asarray(height_km, dtype=float) * 1e3
        if np.any(radius < self.r[0]) or np.any(radius > self.r_top):
            raise ValueError("height lies outside the refractive profile")
        return self._n(radius)

    def grazing_apparent_elevation(self, height_km):
        """Apparent elevation of the ground-tangent solar ray.

        The exact tangent is the occultation boundary.  Evaluate forward a
        small positive angle above it to avoid strict-mask roundoff.
        """
        radius = R_EARTH + np.asarray(height_km, dtype=float) * 1e3
        if np.any(radius < self.r_ground) or np.any(radius > self.r_top):
            raise ValueError("height lies outside ground/profile-top bounds")
        ratio = self.q_ground / (self.refractive_index(height_km) * radius)
        return -np.degrees(np.arccos(np.minimum(ratio, 1.0)))

    def _turning_radius(self, invariant):
        r = np.interp(invariant, self._q_nodes, self.r)
        # q is monotonic; four Newton corrections greatly exceed the accuracy
        # needed from the ~300 m initial inverse table interpolation.
        for _ in range(4):
            n = self._n(r)
            r -= (n * r - invariant) / (n + r * self._dn(r))
        bad = (r < self.r_ground) | (r > self.r_top) | (
            np.abs(self._n(r) * r - invariant) > 1e-7)
        # Ordinary atmospheric profiles converge above.  A nearly critical
        # but still monotonic custom profile can require a bracketed solve.
        for i in np.flatnonzero(bad):
            j = int(np.searchsorted(self._q_nodes, invariant[i]))
            r[i] = brentq(lambda x: float(self._n(x) * x - invariant[i]),
                          self.r[j - 1], self.r[j], xtol=1e-8)
        return r

    def _integral(self, lower, upper, invariant, initial_gap):
        """Positive curvature integral, with r=lower+u² removing tangent roots."""
        lower = np.asarray(lower)
        upper = np.broadcast_to(upper, lower.shape)
        lo = np.maximum(self._edges[:-1][None, :], lower[:, None])
        hi = np.minimum(self._edges[1:][None, :], upper[:, None])
        hi = np.maximum(hi, lo)
        ulo = np.sqrt(np.maximum(lo - lower[:, None], 0.0))
        uhi = np.sqrt(np.maximum(hi - lower[:, None], 0.0))
        half = 0.5 * (uhi - ulo)
        u = 0.5 * (uhi + ulo)[..., None] + half[..., None] * self._gx
        delta = u * u
        r = lower[:, None, None] + delta
        n = self._n(r)
        dn = self._dn(r)
        q = n * r
        qdiff = q - invariant[:, None, None]
        # Direct subtraction of two ~6e6 m invariants loses relative accuracy
        # in the first quadrature node near a tangent.  A local Taylor
        # difference removes that cancellation and the tiny root residual.
        nb = self._n(lower)
        dnb = self._dn(lower)
        q1 = nb + lower * dnb
        q2 = 2.0 * dnb + lower * self._d2n(lower)
        stable = initial_gap[:, None, None] + delta * (
            q1[:, None, None] + 0.5 * delta * q2[:, None, None])
        qdiff = np.where(delta < 0.1, stable, qdiff)
        denom = np.sqrt(np.maximum(qdiff * (q + invariant[:, None, None]), 1e-300))
        integrand = invariant[:, None, None] * (-dn / n) / denom * (2.0 * u)
        values = np.sum(integrand * self._gw * half[..., None], axis=(1, 2))
        return values

    def forward(self, height_km, apparent_elevation_deg):
        """True elevation for each apparent ray; NaN if it intersects ground.

        Inputs broadcast; output has their broadcast shape.  Rays start
        within the supplied profile, with apparent elevation in [-90, 90].
        """
        height, app = np.broadcast_arrays(np.asarray(height_km, dtype=float),
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
        for start in range(0, ea.size, self.chunk):
            idx = np.arange(start, min(start + self.chunk, ea.size))
            idx = idx[valid[idx]]
            if not idx.size:
                continue
            d = down[idx]
            base = r0[idx].copy()
            gap = 2.0 * q0[idx] * np.sin(0.5 * np.radians(ea[idx]))**2
            if d.any():
                base[d] = self._turning_radius(invariant[idx[d]])
                gap[d] = 0.0
            bending = self._integral(base, self.r_top, invariant[idx], gap)
            if d.any():
                bending[d] += self._integral(
                    base[d], r0[idx[d]], invariant[idx[d]], gap[d])
            out[idx] = ea[idx] - np.degrees(bending)
        return out.reshape(height.shape)

    def jacobian(self, height_km, apparent_elevation_deg, *, step_deg: float = 0.001):
        """d(true elevation)/d(apparent elevation), by symmetric difference.

        NaN when either neighbour is blocked.  No extrapolation across the
        shadow boundary is performed.  At +/-90 degrees use an interior
        apparent direction instead of this central difference.
        """
        if step_deg <= 0:
            raise ValueError("step_deg must be positive")
        app = np.asarray(apparent_elevation_deg, dtype=float)
        return (self.forward(height_km, app + step_deg)
                - self.forward(height_km, app - step_deg)) / (2.0 * step_deg)

    def beam_gain(self, height_km, apparent_elevation_deg, *, step_deg: float = 0.001):
        """Point-source solar angular gain (n/n_inf)² dOmega_app/dOmega_true.

        Requires a positive local angular Jacobian, away from caustics and
        disk/ground partial occultation.  It is not a complete refracted
        transport correction when diffuse photon paths remain straight.
        """
        height, app = np.broadcast_arrays(np.asarray(height_km, dtype=float),
                                          np.asarray(apparent_elevation_deg, dtype=float))
        true = self.forward(height, app)
        slope = self.jacobian(height, app, step_deg=step_deg)
        n = self._n(R_EARTH + height * 1e3)
        gain = ((n / self.n_inf)**2 * np.cos(np.radians(app))
                / np.cos(np.radians(true)) / slope)
        return np.where(slope > 0.0, gain, np.nan)
