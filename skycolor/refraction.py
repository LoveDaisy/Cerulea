"""Atmospheric refraction: apparent solar position and the twilight shadow.

Refraction does three things to a sky model, and they are worth separating:

1. **It moves the sun.**  The atmosphere bends light toward the denser medium,
   so everything is seen higher than it is.  At the horizon the shift is about
   half a degree; the sun sets two to three minutes later than the geometry
   says (much longer at high latitudes), and its disk is flattened into the
   "Etruscan vase" shape.
2. **It moves the terminator.**  The boundary between sunlit and shadowed air
   is raised, so every twilight stage happens slightly later in solar
   elevation than the geometric calculation predicts.  Since twilight
   brightness changes by roughly a factor of two per degree, this is a real
   (if modest) effect.
3. **It bends the line of sight.**  A sky element just above the geometric
   horizon is actually a little higher, and the whole sky is compressed into
   slightly less than a hemisphere.  This matters for rendering and for
   comparing with photographs.

The implementation traces real rays through the refractive atmosphere rather
than using an empirical fit, because the usual fits (Saemundsson, Bennett) are
only defined for objects *above* the horizon and we need the geometry of rays
that graze or pass below it.

Rays obey the ray equation for a spherically symmetric medium,

    d/ds (n u) = grad n,          n = n(r),  grad n = n'(r) rhat

integrated with a fixed-step Runge-Kutta scheme in a 2-D Cartesian frame.
The refractive index follows Gladstone-Dale,
``n - 1 = (n0 - 1) * rho / rho0`` with ``n0 - 1 = 2.78e-4`` at 15 degC and
1013.25 hPa (Edlen, at 550 nm).

The integrated refraction at the horizon comes out at 30 arcminutes, which
agrees with Saemundsson's fit (29') for the US Standard Atmosphere.  The
often-quoted 34' is a mean over a wide range of temperature profiles (28'-40');
the difference between 30' and 34' is 4 arcminutes, i.e. about 1/8 of the
effect being modelled.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import N_STD_AIR, R_EARTH

#: refractivity of standard dry air at 550 nm, 15 degC, 1013.25 hPa
REFRACTIVITY_STD = 2.78e-4

#: how far above the surface the refractive profile is tabulated [m]
_PROFILE_TOP = 1.2e6


class _Profile:
    """Cached refractive index profile ``n(r)`` for one atmosphere."""

    def __init__(self, atmo, n_points: int = 4000):
        self.r = np.linspace(R_EARTH, R_EARTH + _PROFILE_TOP, n_points)
        z = (self.r - R_EARTH) / 1e3
        self.n = 1.0 + REFRACTIVITY_STD * atmo.n_air(z) / N_STD_AIR
        # A cubic spline keeps the derivative smooth across the tabulated
        # points; a piecewise-linear interpolation would make dn/dr
        # discontinuous on a 300 m scale and destroy the RK4 convergence.
        self._n = CubicSpline(self.r, self.n)
        self._dn = self._n.derivative()

    def n_of(self, r):
        return float(self._n(r))

    def dndr_of(self, r):
        return float(self._dn(r))


def _trace(profile: _Profile, r0: float, theta0: float, r_earth: float,
           r_top: float, ds: float, max_steps: int, stop_on_ground: bool = True):
    """Trace one ray; returns ``(hit_earth, r_min, theta_inf, bending)``.

    ``stop_on_ground=False`` integrates through a grazing contact with the
    surface, which is what the shadow-threshold calculation needs (the ray is
    by construction tangent to it, and stopping there would truncate half the
    bending).
    """
    y = np.array([0.0, r0])
    u = np.array([np.sin(theta0), np.cos(theta0)])
    u /= np.linalg.norm(u)

    def accel(pos, direction):
        r = np.linalg.norm(pos)
        n = profile.n_of(r)
        g = profile.dndr_of(r) / n * (pos / r)
        return direction, g - direction * np.dot(direction, g)

    total = 0.0
    hit = False
    r_min = r0
    for step in range(max_steps):
        r = np.linalg.norm(y)
        if r < r_min:
            r_min = r
        if stop_on_ground and step > 0 and r <= r_earth * (1.0 + 1e-9):
            hit = True
            break
        if r >= r_top:
            break
        k1y, k1u = accel(y, u)
        k2y, k2u = accel(y + 0.5 * ds * k1y, u + 0.5 * ds * k1u)
        k3y, k3u = accel(y + 0.5 * ds * k2y, u + 0.5 * ds * k2u)
        k4y, k4u = accel(y + ds * k3y, u + ds * k3u)
        y = y + ds / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        u_new = u + ds / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        u_new /= np.linalg.norm(u_new)
        total += np.arccos(np.clip(np.dot(u, u_new), -1.0, 1.0))
        u = u_new
    theta_inf = float(np.arctan2(u[0], u[1]))
    return hit, r_min, theta_inf, total


def trace_ray(atmo, r0: float, theta0: float, r_earth: float = R_EARTH,
              r_top: float = R_EARTH + 1.2e6, ds: float = 2000.0,
              max_steps: int = 100000):
    """Trace one ray through the refractive atmosphere.

    Parameters
    ----------
    r0 : start radius [m]
    theta0 : *apparent* zenith angle at the start point [rad]; 0 is straight
        up, pi/2 is horizontal, > pi/2 points below the local horizon.

    Returns
    -------
    dict with ``hit_earth``, ``r_min``, ``theta_inf`` (the zenith angle the
    ray makes with the initial local vertical once it reaches space, i.e. the
    *true* zenith angle of the source) and ``bending`` (total turning, rad).
    """
    profile = _Profile(atmo)
    hit, r_min, theta_inf, bending = _trace(profile, r0, theta0, r_earth, r_top,
                                            ds, max_steps)
    return dict(hit_earth=hit, r_min=r_min, theta_inf=theta_inf, bending=bending)


def boundary_theta(profile: _Profile, r0: float, r_earth: float = R_EARTH) -> float:
    """Apparent zenith angle of the ray from ``r0`` that just grazes the Earth.

    Bouguer's invariant ``n r sin(theta) = const`` fixes this exactly: the
    grazing ray has ``n(R) R`` as its constant, so
    ``sin(theta) = n(R) R / (n(r0) r0)``.  No ray tracing is needed.
    """
    a = profile.n_of(r_earth) * r_earth
    s = a / (profile.n_of(r0) * r0)
    if s >= 1.0:
        return np.pi / 2
    # the ray points *down* towards the horizon, so take the obtuse solution
    return float(np.pi - np.arcsin(s))


class RefractionTables:
    """Apparent/true conversion and shadow thresholds for one atmosphere."""

    def __init__(self, atmo, n_elev: int = 61, alt_km=None):
        self.atmo = atmo
        profile = _Profile(atmo)
        self.profile = profile
        self._n_elev = n_elev
        self._elev_table = None

        # --- shadow threshold vs altitude -------------------------------
        if alt_km is None:
            alt_km = np.array([0.0, 0.5, 1, 2, 3, 5, 7, 10, 13, 16, 20, 25,
                               30, 38, 47, 56, 70, 90, 120])
        self.alt_km = np.asarray(alt_km, dtype=float)
        sh_true, sh_app = [], []
        for h in self.alt_km:
            r0 = R_EARTH + h * 1e3
            th = boundary_theta(profile, r0)
            _, _, _, bend = _trace(profile, r0, th - 3e-5, R_EARTH,
                                   R_EARTH + 1.2e6, 2000.0, 400000,
                                   stop_on_ground=False)
            sh_app.append(th)
            sh_true.append(th + bend)   # true = apparent + bending
        self._shadow_true = np.degrees(np.array(sh_true))
        self._shadow_app = np.degrees(np.array(sh_app))

    # -- solar position -----------------------------------------------------
    def _build_elevation_table(self):
        elev = np.radians(np.linspace(0.05, 90.0, self._n_elev))
        true = []
        for e in elev:
            _, _, _, bend = _trace(self.profile, R_EARTH * (1 + 1e-9),
                                   np.pi / 2 - e, R_EARTH, R_EARTH + 1.2e6,
                                   2000.0, 400000)
            true.append(e - bend)
        self._elev_table = (np.degrees(np.array(true)), np.degrees(elev))

    @property
    def true_elev(self):
        if self._elev_table is None:
            self._build_elevation_table()
        return self._elev_table[0]

    @property
    def elev(self):
        if self._elev_table is None:
            self._build_elevation_table()
        return self._elev_table[1]

    def apparent_elevation(self, true_elev_deg):
        """Apparent elevation [deg] of a source at ``true_elev_deg``."""
        t = np.asarray(true_elev_deg, dtype=float)
        return np.interp(t, self.true_elev, self.elev,
                         left=self.elev[0], right=self.elev[-1])

    def apparent_zenith(self, true_zenith_deg):
        return 90.0 - self.apparent_elevation(90.0 - np.asarray(true_zenith_deg, float))

    def refraction(self, true_elev_deg):
        t = np.asarray(true_elev_deg, dtype=float)
        return self.apparent_elevation(t) - t

    # -- shadow -------------------------------------------------------------
    def shadow_zenith_true(self, z_km):
        """True solar zenith angle at which a point at ``z_km`` becomes shadowed."""
        return np.interp(np.asarray(z_km, dtype=float), self.alt_km, self._shadow_true)

    def shadow_zenith_apparent(self, z_km):
        """Same threshold expressed in apparent solar zenith angle."""
        return np.interp(np.asarray(z_km, dtype=float), self.alt_km, self._shadow_app)

    def refraction_extension(self, z_km):
        """Degrees of extra solar depression that still illuminate a point at z."""
        z = np.asarray(z_km, dtype=float)
        geometric = 90.0 + np.degrees(np.arccos(R_EARTH / (R_EARTH + z * 1e3)))
        return self.shadow_zenith_true(z) - geometric

    def horizon_dip(self, z_km):
        """Dip of the visible horizon [deg]."""
        z = np.maximum(np.asarray(z_km, dtype=float), 0.0)
        return np.degrees(np.arccos(R_EARTH / (R_EARTH + z * 1e3)))


_CACHE: dict[tuple, RefractionTables] = {}


def _key(atmo):
    # refraction only depends on the density profile, which is the same for
    # every atmosphere with the same surface pressure and observer altitude
    return (round(float(getattr(atmo, "surface_pressure", 101325.0)), 3),
            round(float(getattr(atmo, "elevation_km", 0.0)), 6))


def get_tables(atmo) -> RefractionTables:
    """Cached :class:`RefractionTables` for the density profile of ``atmo``."""
    key = _key(atmo)
    if key not in _CACHE:
        _CACHE[key] = RefractionTables(atmo)
    return _CACHE[key]


def apparent_solar_zenith(atmo, true_zenith_deg, refraction=True):
    """Convert a true (ephemeris) solar zenith angle to the apparent one."""
    if not refraction:
        return true_zenith_deg
    return get_tables(atmo).apparent_zenith(true_zenith_deg)


# ---------------------------------------------------------------------------
# the refractive geometry used by the radiative transfer
# ---------------------------------------------------------------------------
#: grid of the packaged refraction table
def _load_table():
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "data" / "refraction.csv"
    return np.loadtxt(path, delimiter=",", skiprows=1)


class RefractionGeometry:
    """Turn the packaged refraction table into fast vectorised lookups.

    The table gives, for a grid of altitudes and *apparent* elevations, the
    *true* (asymptotic) elevation of the source.  Radiative transfer needs the
    inverse: given the geometric solar elevation at a point, what is the
    apparent one, and what is the resulting Bouguer invariant?
    """

    def __init__(self):
        raw = _load_table()
        self.alts = np.unique(raw[:, 0])
        self.true_grid = np.linspace(-10.0, 90.0, 1200)
        self.app = np.full((len(self.alts), len(self.true_grid)), np.nan)
        for i, z in enumerate(self.alts):
            m = raw[:, 0] == z
            et, ea = raw[m, 2], raw[m, 1]
            o = np.argsort(et)
            et, ea = et[o], ea[o]
            inside = (self.true_grid >= et[0]) & (self.true_grid <= et[-1])
            self.app[i, inside] = np.interp(self.true_grid[inside], et, ea)
            # outside the tabulated range the refraction is either negligible
            # (high elevations) or the point is in shadow (low ones); extend
            # with slope 1 / the endpoint slope so the interpolation is smooth
            self.app[i, self.true_grid < et[0]] = ea[0] + (self.true_grid[self.true_grid < et[0]] - et[0])
            above = self.true_grid > et[-1]
            self.app[i, above] = self.true_grid[above] - (et[-1] - ea[-1])

    def apparent_elevation(self, true_elev_deg, alt_km):
        """Apparent elevation [deg] for a given true elevation and altitude."""
        te = np.asarray(true_elev_deg, dtype=float)
        z = np.atleast_1d(np.asarray(alt_km, dtype=float))
        z = np.broadcast_to(z, te.shape)
        te_flat = te.reshape(-1)
        z_flat = z.reshape(-1)
        # interpolate along the true-elevation axis for every altitude row,
        # then blend across altitude
        rows = np.stack([np.interp(te_flat, self.true_grid, self.app[i])
                         for i in range(len(self.alts))])          # (n_alt, N)
        zi = np.interp(z_flat, self.alts, np.arange(len(self.alts)))
        i0 = np.clip(np.floor(zi).astype(int), 0, len(self.alts) - 1)
        i1 = np.clip(i0 + 1, 0, len(self.alts) - 1)
        w = zi - i0
        idx = np.arange(te_flat.size)
        vals = rows[i0, idx] * (1 - w) + rows[i1, idx] * w
        return vals.reshape(te.shape)


_GEOM: RefractionGeometry | None = None


def geometry() -> RefractionGeometry:
    """Cached :class:`RefractionGeometry` built from the packaged table."""
    global _GEOM
    if _GEOM is None:
        _GEOM = RefractionGeometry()
    return _GEOM


def refractivity_of(atmo, z_km):
    """Refractivity ``n - 1`` at height ``z``."""
    return REFRACTIVITY_STD * atmo.n_air(z_km) / N_STD_AIR


def sun_path_lengths(a, mu_app, r_point, r_bounds, n_layer):
    """Per-shell path lengths of a *refracted* solar ray.

    ``a`` is the Bouguer invariant ``n r sin(theta)``, ``mu_app`` the cosine of
    the apparent zenith angle at the point and ``n_layer`` the refractive index
    of each shell.  Inside a homogeneous shell the ray is a straight chord with
    impact parameter ``b_k = a / n_k``, so the path length has the same closed
    form as a straight ray with ``p`` replaced by ``b_k`` -- which is why the
    whole correction costs one division rather than a ray trace.

    Returns an array of shape ``(n_shell, ...)`` matching ``a``.
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(mu_app, dtype=float)
    r = np.asarray(r_point, dtype=float)
    rb = np.asarray(r_bounds, dtype=float)
    n = np.asarray(n_layer, dtype=float)
    shp = a.shape
    b = a[None] / n.reshape((len(rb) - 1,) + (1,) * a.ndim)          # (ns, ...)
    r_lo = rb[:-1].reshape((-1,) + (1,) * a.ndim)
    r_hi = rb[1:].reshape((-1,) + (1,) * a.ndim)
    big = np.broadcast_to(b, (len(rb) - 1,) + shp)
    s_lo = np.sqrt(np.maximum(r_lo**2 - big**2, 0.0))
    s_hi = np.sqrt(np.maximum(r_hi**2 - big**2, 0.0))
    u_point = r * np.abs(c)
    up = np.maximum(s_hi - np.maximum(s_lo, u_point[None]), 0.0)
    down = np.maximum(np.minimum(s_hi, u_point[None]) - np.minimum(s_lo, u_point[None]), 0.0)
    return up + np.where(c[None] < 0.0, down, 0.0)
