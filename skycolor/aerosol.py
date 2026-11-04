"""Aerosol optical models.

Every aerosol model exposes two *intensive* properties that are independent of
the amount of aerosol in the air:

``relative_coefficients(wl)``
    extinction and scattering coefficients normalised so that the extinction
    at 550 nm equals 1.  The absolute coefficient in a layer is therefore
    ``AOD550 * profile(z) * rel_ext(wl)``.

``phase(mu, wl)``
    scattering phase function normalised so that ``int p dOmega = 4*pi``.

Single scattering albedo, asymmetry parameter and Legendre moments follow from
those two, either analytically (the Mie models) or by quadrature (the base
class).

Three families are provided:

* :class:`AngstromAerosol` -- an analytic power-law model with a
  Henyey-Greenstein phase function.  It is kept for fast sweeps, for
  controlled comparisons with the physical model, and because several parts of
  the project use it as a deliberately simple reference.
* :class:`MieAerosol` -- the physical model: one or more log-normal size
  distributions, a (possibly wavelength-dependent) complex refractive index,
  Mie theory, **analytic** Legendre moments, and an adaptive phase-function
  table that resolves the forward diffraction peak.
* the **named library** at the bottom of this module: ``rural``, ``urban``,
  ``maritime``, ``desert_dust``, ``biomass_burning``,
  ``volcanic_stratospheric``, ``cirrus`` and ``water_cloud``.  Each is a
  :class:`MieAerosol` with documented provenance and documented Angstrom
  exponent, SSA and asymmetry at 550 nm.

The analytic moments and the phase table are described in ``docs/physics/AEROSOL.md``.
The short version is that the phase function is a polynomial in
``mu = cos(theta)``, so its Legendre moments can be obtained *exactly* from the
Mie coefficients with the Gaunt/Wigner 3-j expansion implemented in
:mod:`skycolor.mie`; no angular quadrature is involved.  This is what makes
``g = beta_1`` physical even for a 15 um droplet, whose forward peak is
``~0.002 deg`` wide.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field, fields as dataclasses_fields
from typing import Callable, Sequence

import numpy as np

from .mie import (
    log_normal_nodes,
    mie_coefficients,
    mie_moment_numerators,
    mie_phase,
    phase_table_angles,
)

_GAUSS_MU, _GAUSS_W = np.polynomial.legendre.leggauss(96)


# ===========================================================================
# base class
# ===========================================================================
class Aerosol:
    """Base class for an aerosol population."""

    name = "aerosol"

    # -- to be implemented ---------------------------------------------------
    def relative_coefficients(self, wl_nm) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def phase(self, mu, wl_nm) -> np.ndarray:
        raise NotImplementedError

    # -- derived -------------------------------------------------------------
    def extinction(self, wl_nm) -> np.ndarray:
        return self.relative_coefficients(wl_nm)[0]

    def scattering(self, wl_nm) -> np.ndarray:
        return self.relative_coefficients(wl_nm)[1]

    def ssa(self, wl_nm) -> np.ndarray:
        ext, sca = self.relative_coefficients(wl_nm)
        return np.where(ext > 0, sca / np.maximum(ext, 1e-300), 0.0)

    def asymmetry(self, wl_nm) -> np.ndarray:
        """Asymmetry parameter ``g = <cos theta>``, one value per wavelength."""
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        p = self.phase(_GAUSS_MU, wl)
        if p.ndim == 1:
            p = p[:, None]
        return 0.5 * np.sum(_GAUSS_W[:, None] * _GAUSS_MU[:, None] * p, axis=0)

    def legendre_moments(self, wl_nm, n_moments: int = 32) -> np.ndarray:
        """Coefficients ``beta_l`` of ``p(cos t) = sum_l (2l+1) beta_l P_l``.

        Returns an array of shape ``(n_moments, n_wl)``.
        """
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        p = self.phase(_GAUSS_MU, wl)  # (n_mu, n_wl)
        out = np.zeros((n_moments, len(wl)))
        for l in range(n_moments):
            pl = np.polynomial.legendre.legval(_GAUSS_MU, [0] * l + [1])
            out[l] = 0.5 * np.sum(_GAUSS_W[:, None] * p * pl[:, None], axis=0)
        return out

    # -- delta-M -------------------------------------------------------------
    def delta_m(self, wl_nm, n_moments: int = 32, tau=None, omega=None) -> "DeltaMResult":
        """Delta-M scaling of this aerosol's phase function.

        Computes the moments ``beta_0 .. beta_{n_moments}``, takes
        ``f = beta_{n_moments}`` and returns the renormalised set.  See
        :func:`delta_m` for the transformation and the meaning of ``tau`` and
        ``omega``.
        """
        mom = self.legendre_moments(wl_nm, int(n_moments) + 1)
        return delta_m(mom, n_moments, tau=tau, omega=omega)

    # ``sos_inputs`` is the name used by the multiple-scattering solver: the
    # returned object carries beta*, f, tau*, omega* and g*.
    def sos_inputs(self, wl_nm, n_moments: int = 32, tau=None, omega=None) -> "DeltaMResult":
        """Alias for :meth:`delta_m`; the result is what the SOS solver needs."""
        return self.delta_m(wl_nm, n_moments, tau=tau, omega=omega)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


def henyey_greenstein(mu, g):
    """Henyey-Greenstein phase function normalised to ``int p dOmega = 4 pi``."""
    g = np.asarray(g, dtype=float)
    return (1.0 - g**2) / np.power(1.0 + g**2 - 2.0 * g * np.asarray(mu), 1.5)


# ===========================================================================
# analytic (Angstrom) model
# ===========================================================================
@dataclass
class AngstromAerosol(Aerosol):
    """Power-law extinction with a Henyey-Greenstein phase function.

    Parameters
    ----------
    alpha : float
        Angstrom exponent; ``1.3`` is typical continental haze, ``0.2`` dust,
        ``2.0`` smoke.  The extinction is ``(lambda/550nm)**(-alpha)``.
    ssa : float
        Single scattering albedo at 550 nm (weakly wavelength dependent for
        the sake of realism: absorption is slightly stronger in the blue for
        brown carbon / dust).
    g : float
        Asymmetry parameter.
    ssa_slope : float
        d(ssa)/d(lambda) in 1/nm, default 0 (grey absorption).
    """

    name: str = "angstrom"
    alpha: float = 1.3
    ssa: float = 0.95
    g: float = 0.65
    ssa_slope: float = 0.0

    def relative_coefficients(self, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        ext = (wl / 550.0) ** (-self.alpha)
        s = np.clip(self.ssa + self.ssa_slope * (wl - 550.0), 0.0, 1.0)
        return ext, ext * s

    def phase(self, mu, wl_nm):
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        if np.ndim(wl_nm) == 0:
            out = henyey_greenstein(mu_arr, self.g)
        else:
            wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
            out = henyey_greenstein(mu_arr[:, None], np.full(wl.shape, self.g)[None, :])
        if np.ndim(mu) == 0 and np.ndim(wl_nm) == 0:
            return out[0]
        return out

    def legendre_moments(self, wl_nm, n_moments: int = 32) -> np.ndarray:
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        return np.power(float(self.g), np.arange(n_moments))[:, None] * np.ones_like(wl)[None, :]

    def asymmetry(self, wl_nm) -> np.ndarray:
        return np.full(np.atleast_1d(np.asarray(wl_nm, dtype=float)).shape, float(self.g))


# ===========================================================================
# adaptive phase table
# ===========================================================================
def _phase_u(theta_deg: np.ndarray) -> np.ndarray:
    """Monotone coordinate used by the phase table interpolation.

    ``log10(theta)`` inside the forward degree, linear in ``theta`` from 1 to
    179 degrees, and ``log10(180-theta)`` inside the backward degree.  Linear
    interpolation of ``log p`` in this coordinate is log-log in the forward
    peak (where the phase function is a power law at small angles) and
    log-linear elsewhere.
    """
    t = np.asarray(theta_deg, dtype=float)
    u = np.empty_like(t)
    m_fwd = t < 1.0
    u[m_fwd] = np.log10(np.maximum(t[m_fwd], 1e-300))
    m_mid = (t >= 1.0) & (t <= 179.0)
    u[m_mid] = (t[m_mid] - 1.0) / 178.0 * 4.0
    m_back = t > 179.0
    u[m_back] = 4.0 + (-np.log10(np.maximum(180.0 - t[m_back], 1e-300))) / 4.0
    return u


class PhaseTable:
    """Interpolated, adaptive tabulation of a phase function.

    The table is built on the grid returned by
    :func:`skycolor.mie.phase_table_angles`: log-spaced in scattering angle
    from ``1e-4`` to ``1`` degree, linear to ``179`` degrees, and log-spaced
    again in the last degree.  This resolves the forward diffraction peak of
    spheres up to ``x ~ 200`` (and beyond) instead of the fixed 721-point
    ``mu`` grid used by the old code, which could not.

    ``p(0)`` and ``p(180)`` are stored exactly and returned for angles outside
    the interpolable range.
    """

    def __init__(self, theta_deg: np.ndarray, p: np.ndarray):
        theta_deg = np.asarray(theta_deg, dtype=float)
        p = np.asarray(p, dtype=float)
        if theta_deg.shape != p.shape:
            raise ValueError("theta and p must have the same shape")
        self.theta_deg = theta_deg
        self.p = np.maximum(p, 0.0)
        self.p_peak = float(self.p[0])
        self.p_back = float(self.p[-1])
        self.theta_min = float(theta_deg[1])
        self.theta_max = float(theta_deg[-2])
        # interpolation nodes exclude the two exact endpoints
        self._u = _phase_u(theta_deg[1:-1])
        self._logp = np.log(np.maximum(self.p[1:-1], 1e-300))
        order = np.argsort(self._u)
        self._u = self._u[order]
        self._logp = self._logp[order]

    def __call__(self, mu) -> np.ndarray:
        mu_arr = np.asarray(mu, dtype=float)
        theta = np.degrees(np.arccos(np.clip(mu_arr, -1.0, 1.0)))
        u = _phase_u(theta)
        out = np.exp(np.interp(u, self._u, self._logp))
        out = np.where(theta <= self.theta_min, self.p_peak, out)
        out = np.where(theta >= self.theta_max, self.p_back, out)
        return out

    def normalisation(self) -> float:
        """``int p dOmega`` evaluated on the table itself (should be ``4 pi``)."""
        th = np.radians(self.theta_deg)
        return float(2.0 * np.pi * np.trapezoid(self.p * np.sin(th), th))

    def __repr__(self) -> str:
        return (f"PhaseTable(n={self.p.size}, "
                f"normalisation={self.normalisation()/(4*np.pi):.6f})")


# ===========================================================================
# Mie theory (compatibility helpers)
# ===========================================================================
def _log_normal_weights(r, r_g, sigma_g):
    """Number density weights of a log-normal distribution (unnormalised)."""
    return np.exp(-0.5 * (np.log(r / r_g) / np.log(sigma_g)) ** 2) / (r * np.log(sigma_g))


def mie_single(x: float, m: complex, n_angles: int = 0):
    """Mie coefficients for a single sphere.

    Kept as a small public helper (and for backwards compatibility).  The
    phase function, when requested, is evaluated directly at the requested
    angles -- there is no fixed angular grid, so the forward peak of a large
    sphere is represented correctly.

    Returns
    -------
    qext, qsca, g, (mu, phase)
    """
    coeff = mie_coefficients(x, m)
    if n_angles <= 0:
        return coeff.qext, coeff.qsca, coeff.g, (None, None)
    mu = np.linspace(-1.0, 1.0, int(n_angles))
    phase = mie_phase(coeff, np.arccos(mu))
    return coeff.qext, coeff.qsca, coeff.g, (mu, phase)


# ===========================================================================
# Mie aerosol
# ===========================================================================
@dataclass
class LogNormalMode:
    """A log-normal size distribution.

    Parameters
    ----------
    r_g_um : float
        Geometric mean radius [um] of the *number* distribution.
    sigma_g : float
        Geometric standard deviation (dimensionless, ``> 1``).
    weight : float
        Relative number weight of the mode.
    refractive_index : complex or callable, optional
        Per-mode refractive index.  If ``None`` the aerosol's index is used.
        This makes external mixtures (e.g. water-soluble + soot) possible.
    """

    r_g_um: float  # geometric mean radius [um]
    sigma_g: float  # geometric standard deviation
    weight: float = 1.0  # relative number weight
    refractive_index: Callable | complex | None = None


@dataclass(frozen=True)
class MieAerosol(Aerosol):
    """Aerosol described by log-normal size modes and a refractive index.

    Frozen value object: the optical definition (modes, index, grids) never
    changes after construction, which is what makes ``shared()`` instances
    safe to reuse across atmospheres.  The per-wavelength caches below are
    mutable *contents* behind an immutable binding -- redefining an aerosol
    is :meth:`replace`, never attribute assignment.

    The size distribution is integrated with Mie theory.  The Legendre moments
    of the resulting phase function are computed **analytically** from the Mie
    coefficients (see :mod:`skycolor.mie`), so ``g = beta_1`` is exact and
    always lies in ``(-1, 1)``; the angular phase function is tabulated
    adaptively so that the forward peak of large particles is resolved.

    Parameters
    ----------
    modes : sequence of LogNormalMode
    refractive_index : complex or callable
        ``complex`` for a non-dispersive index, or a callable
        ``m(wl_nm) -> complex``.
    r_min_um, r_max_um : float
        Hard limits on the radius grid.  Together with ``n_sigma`` they
        truncate the log-normal distribution.
    n_radii : int
        Number of radii per mode (trapezoidal rule in ``ln r``).
    n_sigma : float
        The distribution is truncated at ``|ln(r/r_g)| <= n_sigma*ln(sigma_g)``.
    phase_table : bool
        Use the adaptive phase table (default).  If ``False`` the phase
        function is evaluated directly from the Mie coefficients; this is
        exact but much slower.
    n_angles : int
        Deprecated and ignored.  It used to be the size of the fixed
        ``cos(theta)`` grid, which is exactly what the adaptive table
        replaces.  It is kept so old call sites keep working.
    """

    name: str = "mie"
    modes: Sequence[LogNormalMode] = field(
        default_factory=lambda: [LogNormalMode(0.05, 2.0)]
    )
    refractive_index: Callable | complex = 1.53 + 0.006j
    r_min_um: float = 1e-4
    r_max_um: float = 30.0
    n_radii: int = 96
    n_sigma: float = 3.0
    n_angles: int = 0
    use_phase_table: bool = True
    table_forward_points: int | None = None
    table_mid_points: int | None = None
    table_back_points: int = 80
    moments_cache_size: int = 36
    # Per-wavelength Mie states are pure functions of (wavelength,
    # construction parameters), so refilling an evicted state reproduces the
    # evicted values bit for bit -- capacity only trades memory for time.
    # The default must hold the whole working grid: on the 81-wavelength /
    # 5 nm grid (``validation.common.WL_COARSE``) the old default of 8 was
    # smaller than the working set, the LRU thrashed, and *every* call
    # rebuilt all 81 states -- the dominant cost of everything downstream.
    # 256 covers that grid with 3x headroom.  Memory: a state holds one
    # coefficient set per size node (fine modes: ~96 nodes at small ``x``,
    # KB per state); the coarse-mode members (dust / cirrus / cloud, two
    # 128-node modes, ``x`` up to ~500 and hence long a/b arrays) reach
    # ~1 MB per state, so a full cache is at most of order 10^2 MB for one
    # such instance -- and only when it is queried at 256 distinct
    # wavelengths.
    max_cached_wavelengths: int = 256

    def __post_init__(self):
        # frozen dataclass: the internal cache slots are set in place here
        # (contents mutate later, the binding never does)
        object.__setattr__(self, "_nodes", self._build_nodes())
        object.__setattr__(self, "_states", OrderedDict())
        # Phase tables are cheap to keep but expensive to rebuild.  They are
        # cached separately from the per-wavelength coefficient states, with
        # their own (larger) capacity, so a repeated sky_radiance() call over
        # the same wavelengths never rebuilds tables even when the
        # coefficient-state cache evicts between calls.
        object.__setattr__(self, "_tables", OrderedDict())
        object.__setattr__(self, "_max_tables", 512)

    def replace(self, **kw) -> "MieAerosol":
        """A copy of this aerosol with the given *dataclass fields* redefined.

        Field-level only: ``kw`` keys must be dataclass fields of the
        instance's class.  The preset subclasses (``RuralAerosol``,
        ``HazeAerosol``, ...) take different constructor keywords (``rh``,
        ...) that are **not** fields -- to vary a preset knob, construct a
        new instance (``HazeAerosol(rh=0.9)`` / :func:`shared`) rather than
        calling ``replace``.  Note the ``name`` field is not re-derived:
        renamed definitions should also pass ``name=`` explicitly.  The
        derived nodes and the per-wavelength caches are rebuilt for the new
        definition; the old object (and any shared instance of it) is
        untouched.
        """
        field_names = {f.name for f in dataclasses_fields(type(self))}
        unknown = set(kw) - field_names
        if unknown:
            raise TypeError(f"unknown field(s): {sorted(unknown)}")
        new = copy.copy(self)
        for k, v in kw.items():
            object.__setattr__(new, k, v)
        object.__setattr__(new, "_nodes", new._build_nodes())
        object.__setattr__(new, "_states", OrderedDict())
        object.__setattr__(new, "_tables", OrderedDict())
        object.__setattr__(new, "_max_tables", 512)
        return new

    # -- construction --------------------------------------------------------
    def _build_nodes(self):
        nodes = []
        for mode in self.modes:
            if mode.weight == 0.0:
                continue
            r, w, _ = log_normal_nodes(
                mode.r_g_um,
                mode.sigma_g,
                n_radii=self.n_radii,
                n_sigma=self.n_sigma,
                r_min_um=self.r_min_um,
                r_max_um=self.r_max_um,
                weight=mode.weight,
            )
            mi = mode.refractive_index
            for ri, wi in zip(r, w):
                nodes.append((float(ri), float(wi), mi))
        return nodes

    def _index(self, mi, wl_nm: float) -> complex:
        if mi is None:
            mi = self.refractive_index
        if callable(mi):
            return complex(mi(wl_nm))
        return complex(mi)

    # -- per-wavelength state ------------------------------------------------
    def _state(self, wl_nm: float) -> "_State":
        key = round(float(wl_nm), 6)
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
            return state

        coeffs = []
        ce = cs = gnum = 0.0
        x_max = 0.0
        for r, w, mi in self._nodes:
            x = 2.0 * np.pi * r / (float(wl_nm) * 1e-3)
            if x < 1e-4:
                # far below the Rayleigh regime the cross section is
                # negligible; skipping avoids ill-conditioned Bessel recurrences
                continue
            m = self._index(mi, wl_nm)
            c = mie_coefficients(x, m)
            area = np.pi * (r * 1e-6) ** 2
            wt = w * area
            ce += wt * c.qext
            cs += wt * c.qsca
            gnum += wt * c.qsca * c.g
            coeffs.append((c, r, w, area))
            x_max = max(x_max, c.x)
        g = float(gnum / cs) if cs > 0.0 else 0.0
        state = _State(
            wl=key, coeffs=coeffs, ce=float(ce), cs=float(cs), g=g, x_max=x_max
        )
        self._states[key] = state
        while len(self._states) > max(int(self.max_cached_wavelengths), 1):
            self._states.popitem(last=False)
        return state

    # -- public interface ----------------------------------------------------
    def relative_coefficients(self, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        states = [self._state(w) for w in wl]
        ce = np.array([s.ce for s in states])
        cs = np.array([s.cs for s in states])
        if ce.size == 0:
            return ce, cs
        # AOD550 fixes the reference cross section of the material, including
        # monochromatic MC queries and grids that do not contain 550 nm.
        scale = self._state(550.0).ce
        scale = scale if scale > 0.0 else 1.0
        return ce / scale, cs / scale

    def asymmetry(self, wl_nm) -> np.ndarray:
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        return np.array([self._state(w).g for w in wl])

    def legendre_moments(self, wl_nm, n_moments: int = 32) -> np.ndarray:
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        n_moments = int(n_moments)
        out = np.zeros((n_moments, wl.size))
        for i, w in enumerate(wl):
            state = self._state(w)
            need = max(n_moments, int(self.moments_cache_size))
            if state.beta is None or state.beta.size < n_moments:
                state.beta = self._size_averaged_moments(state, need)
            out[:, i] = state.beta[:n_moments]
        return out

    def _size_averaged_moments(self, state: "_State", n_moments: int) -> np.ndarray:
        if not state.coeffs or state.cs <= 0.0:
            out = np.zeros(n_moments)
            out[0] = 1.0
            return out
        num = np.zeros(n_moments)
        den = 0.0
        for c, r, w, area in state.coeffs:
            U = mie_moment_numerators(c, n_moments)  # = qsca * beta_l
            num += (w * area) * U
            den += (w * area) * c.qsca
        if den <= 0.0:
            out = np.zeros(n_moments)
            out[0] = 1.0
            return out
        beta = num / den
        # beta_0 = 1 analytically; remove round-off so delta-M is exactly
        # energy conserving.
        beta[0] = 1.0
        return beta

    # -- phase function ------------------------------------------------------
    def _get_table(self, wl_nm: float) -> PhaseTable:
        key = round(float(wl_nm), 6)
        table = self._tables.get(key)
        if table is not None:
            self._tables.move_to_end(key)
            return table
        table = self._build_table(self._state(key))
        self._tables[key] = table
        while len(self._tables) > self._max_tables:
            self._tables.popitem(last=False)
        return table

    def _build_table(self, state: "_State") -> PhaseTable:
        theta = phase_table_angles(
            state.x_max,
            n_forward=self.table_forward_points,
            n_mid=self.table_mid_points,
            n_back=self.table_back_points,
        )
        th = np.radians(theta)
        psum = np.zeros_like(th)
        den = 0.0
        for c, r, w, area in state.coeffs:
            psum += (w * area * c.qsca) * mie_phase(c, th)
            den += w * area * c.qsca
        if den <= 0.0:
            # no scattering: isotropic normalised phase function
            psum = np.ones_like(th)
            den = 1.0
        return PhaseTable(theta, psum / den)

    def _phase_direct(self, state: "_State", mu: np.ndarray) -> np.ndarray:
        theta = np.arccos(np.clip(mu, -1.0, 1.0))
        psum = np.zeros_like(theta)
        den = 0.0
        for c, r, w, area in state.coeffs:
            psum += (w * area * c.qsca) * mie_phase(c, theta)
            den += w * area * c.qsca
        return psum / den if den > 0.0 else np.ones_like(theta)

    def phase(self, mu, wl_nm):
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        out = np.empty((mu_arr.size, wl.size))
        for i, w in enumerate(wl):
            if self.use_phase_table:
                out[:, i] = self._get_table(w)(mu_arr)
            else:
                out[:, i] = self._phase_direct(self._state(w), mu_arr)
        if np.ndim(mu) == 0 and np.ndim(wl_nm) == 0:
            return float(out[0, 0])
        return out

    def phase_table(self, wl_nm) -> PhaseTable:
        """Return (and cache) the adaptive phase table at ``wl_nm``."""
        return self._get_table(float(wl_nm))

    def phase_normalisation(self, wl_nm) -> float:
        """``int p dOmega`` from the tabulated phase function [should be 4 pi]."""
        return self.phase_table(wl_nm).normalisation()

    # -- named constructors kept for compatibility ---------------------------
    @classmethod
    def water_soluble(cls, rh: float = 0.5):
        """Shettle & Fenn water-soluble aerosol with humidity growth."""
        return rural(rh=rh)

    @classmethod
    def dust(cls):
        return desert_dust()

    @classmethod
    def soot(cls):
        return MieAerosol(
            name="soot",
            modes=[LogNormalMode(0.012, 2.0)],
            refractive_index=_soot_index,
        )

    @classmethod
    def cloud_droplets(cls, r_eff_um: float = 8.0):
        return water_cloud(r_eff_um=r_eff_um)


@dataclass
class _State:
    """Cached per-wavelength optical state of a :class:`MieAerosol`."""

    wl: float
    coeffs: list
    ce: float
    cs: float
    g: float
    x_max: float
    beta: np.ndarray | None = None


# ===========================================================================
# delta-M
# ===========================================================================
@dataclass
class DeltaMResult:
    """Output of :func:`delta_m` / :meth:`Aerosol.delta_m`.

    Attributes
    ----------
    beta : ndarray (n_moments, n_wl)
        Renormalised Legendre moments ``beta*_l = (beta_l - f)/(1-f)``.
    f : ndarray (n_wl,)
        Forward-scattered fraction, ``beta_{n_moments}`` (clipped to
        ``[0, 1-1e-12]`` for numerical safety).
    tau_star, omega_star : ndarray (n_wl,) or None
        Renormalised optical depth and single-scattering albedo.  ``tau*``
        requires both ``tau`` and ``omega``; ``omega*`` requires ``omega``.
    g_star : ndarray (n_wl,)
        Renormalised asymmetry parameter ``beta*_1 = (g-f)/(1-f)``.
    n_moments : int
        Number of moments kept (``l = 0 .. n_moments-1``).
    """

    beta: np.ndarray
    f: np.ndarray
    tau_star: np.ndarray | None
    omega_star: np.ndarray | None
    g_star: np.ndarray
    n_moments: int


def delta_m(moments, n_moments: int | None = None, tau=None, omega=None) -> DeltaMResult:
    """Delta-M scaling of a truncated Legendre expansion.

    Given moments ``beta_0 .. beta_L`` (``L >= n_moments``), the first omitted
    moment is used as the forward-scattered fraction

    .. math:: f = \\beta_{N+1},

    and the truncated set is renormalised as

    .. math::

        \\beta^*_l = \\frac{\\beta_l - f}{1-f} \\quad (l \\le N), \\qquad
        \\tau^* = (1 - \\omega f)\\,\\tau, \\qquad
        \\omega^* = \\frac{\\omega(1-f)}{1-\\omega f}.

    This is the standard delta-M transformation (Wiscombe 1977): removing the
    forward delta function from the phase function and rescaling the optical
    depth and albedo keeps the *scattered* energy unchanged.  ``beta*_0 = 1``
    follows immediately from ``beta_0 = 1``.

    Parameters
    ----------
    moments : array_like
        Shape ``(L+1, n_wl)`` (or ``(L+1,)`` for a single wavelength).
    n_moments : int, optional
        Number of moments to keep, ``N+1``; defaults to ``L`` (i.e. use
        ``beta_L`` as ``f`` and keep everything before it).
    tau, omega : array_like, optional
        Optical depth and single-scattering albedo to rescale.  Scalars,
        ``(n_wl,)`` arrays or broadcastable arrays are accepted.

    Returns
    -------
    DeltaMResult
    """
    mom = np.asarray(moments, dtype=float)
    squeezed = mom.ndim == 1
    if squeezed:
        mom = mom[:, None]
    if mom.ndim != 2:
        raise ValueError("moments must be 1-D or 2-D (n_moments, n_wl)")
    L, nwl = mom.shape
    if n_moments is None:
        n_moments = L - 1
    n_moments = int(n_moments)
    if n_moments < 1 or n_moments >= L:
        raise ValueError(
            f"need 1 <= n_moments < number of moments ({L}); got {n_moments}"
        )

    f_raw = mom[n_moments].copy()
    # f must be a fraction of the energy.  For the library members the raw
    # beta_{N+1} is already in [0, 1); clipping only protects the algebra.
    f = np.clip(f_raw, 0.0, 1.0 - 1e-12)
    one_minus_f = 1.0 - f
    beta = (mom[:n_moments] - f[None, :]) / one_minus_f[None, :]
    beta[0, :] = 1.0
    g_star = beta[1].copy() if n_moments > 1 else np.zeros_like(f)

    def _rescale(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(nwl, float(arr))
        else:
            arr = np.broadcast_to(arr, (nwl,)).astype(float).copy()
        return arr

    tau_arr = _rescale(tau)
    omega_arr = _rescale(omega)
    # tau* needs both tau and omega (the (1 - omega f) factor is the whole
    # point); omega* needs only omega.
    tau_star = None
    omega_star = None
    if omega_arr is not None:
        denom = 1.0 - omega_arr * f
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        omega_star = omega_arr * one_minus_f / denom
        if tau_arr is not None:
            tau_star = (1.0 - omega_arr * f) * tau_arr

    if squeezed:
        beta = beta[:, 0]
        f = f[0]
        g_star = g_star[0]
        if tau_star is not None:
            tau_star = tau_star[0]
        if omega_star is not None:
            omega_star = omega_star[0]
    return DeltaMResult(beta, f, tau_star, omega_star, g_star, n_moments)


# ===========================================================================
# Refractive-index models
# ===========================================================================
def _water_index(wl_nm):
    """Refractive index of liquid water (Cauchy fit, visible)."""
    wl = np.asarray(wl_nm, dtype=float)
    return 1.324 + 3200.0 / wl**2 + 1e-8j


def _ice_index(wl_nm):
    """Refractive index of ice Ih (equivalent-sphere approximation)."""
    wl = np.asarray(wl_nm, dtype=float)
    return 1.305 + 1500.0 / wl**2 + 1e-6j


def _sulfate_index(wl_nm):
    """Refractive index of ~75 % H2SO4/water stratospheric sulfate droplets."""
    wl = np.asarray(wl_nm, dtype=float)
    return 1.430 + 3000.0 / wl**2 + 1e-7j


def _sea_salt_index(wl_nm):
    """Refractive index of dry sea salt (NaCl), weakly dispersive."""
    wl = np.asarray(wl_nm, dtype=float)
    return 1.340 + 1200.0 / wl**2 + 1e-6j


def _dust_index(wl_nm):
    """Iron-oxide-bearing mineral dust.

    ``Im(m)`` is small in the visible and rises again towards the near
    infrared, with a minimum near 550-600 nm.  This is the ingredient that
    produces the anomalous, non-monotonic spectral behaviour of desert dust
    (the extinction does not follow a single power law across the visible).
    The numbers are illustrative but bracket the values compiled by
    Shettle & Fenn (1979) and OPAC (Hess et al. 1998).
    """
    wl = np.asarray(wl_nm, dtype=float)
    k_wl = np.array([300.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0,
                     800.0, 900.0, 1000.0, 1200.0])
    k_val = np.array([0.0040, 0.0028, 0.0020, 0.0014, 0.0010, 0.0008, 0.00086,
                      0.0010, 0.0014, 0.0020, 0.0026, 0.0032])
    k = np.interp(wl, k_wl, k_val)
    n = 1.55 + 0.005 * (550.0 / wl - 1.0)
    return n + 1j * k


def _soot_index(wl_nm):
    """Fresh/aged soot with a brown-carbon component (absorption peaks in blue)."""
    wl = np.asarray(wl_nm, dtype=float)
    k = 0.45 * (550.0 / wl) ** 1.0
    n = 1.75 + 0.02 * (550.0 / wl - 1.0)
    return n + 1j * k


def _brown_carbon_index(wl_nm):
    """Biomass-burning aerosol: moderately absorbing, strongly blue-weighted."""
    wl = np.asarray(wl_nm, dtype=float)
    k = 0.03 * (550.0 / wl) ** 1.2
    n = 1.60 + 0.02 * (550.0 / wl - 1.0)
    return n + 1j * k


# ===========================================================================
# Humidity growth
# ===========================================================================
def humidity_growth_factor(rh: float, gamma: float = 0.35) -> float:
    """Radius growth factor ``f(RH) = (1-RH)^(-gamma)``.

    ``RH`` is a fraction in ``[0, 1)``.  ``gamma`` is about 0.35 for the
    Shettle & Fenn water-soluble aerosol, 0.3 for sea salt and 0.45 for
    sulfate; a value of 0 gives a non-hygroscopic aerosol.
    """
    rh = float(np.clip(rh, 0.0, 0.995))
    if gamma <= 0.0:
        return 1.0
    return float((1.0 - rh) ** (-gamma))


def _grown_index(m_dry, f: float, m_water=1.334 + 1e-8j) -> complex:
    """Volume-weighted (Wiener) refractive index of a wetted particle.

    ``f`` is the radius growth factor, so the water volume fraction is
    ``1 - 1/f^3``.  This is the simplest mixing rule that gets the direction
    right (the wet particle is less absorbing and has a lower real index); it
    is not an exact effective-medium theory.
    """
    v_w = 1.0 - 1.0 / float(f) ** 3
    return complex(m_dry) * (1.0 - v_w) + complex(m_water) * v_w


# ===========================================================================
# Named aerosol library
# ===========================================================================
def _rural_config(rh: float = 0.5):
    f = humidity_growth_factor(rh, 0.35)
    return dict(
        name=f"rural(RH={rh:.0%})",
        modes=[LogNormalMode(0.05 * f, 2.0)],
        refractive_index=_grown_index(1.53 + 0.006j, f),
    )


def _urban_config(rh: float = 0.5):
    f = humidity_growth_factor(rh, 0.35)
    return dict(
        name=f"urban(RH={rh:.0%})",
        modes=[
            LogNormalMode(0.05 * f, 2.0, 0.75,
                          refractive_index=_grown_index(1.53 + 0.006j, f)),
            LogNormalMode(0.04, 2.0, 0.25, refractive_index=_soot_index),
        ],
        refractive_index=1.53 + 0.006j,
    )


def _maritime_config(rh: float = 0.5):
    f = humidity_growth_factor(rh, 0.30)
    m = _grown_index(1.34 + 1e-6j, f)
    return dict(
        name=f"maritime(RH={rh:.0%})",
        modes=[
            LogNormalMode(0.10 * f, 1.8, 0.997, refractive_index=m),
            LogNormalMode(1.5 * f, 1.9, 0.003, refractive_index=m),
        ],
        refractive_index=m,
    )


def _desert_dust_config(rh: float = 0.0):
    f = humidity_growth_factor(rh, 0.05)
    return dict(
        name=f"desert_dust(RH={rh:.0%})",
        modes=[
            LogNormalMode(0.40 * f, 2.0, 0.35, refractive_index=_dust_index),
            LogNormalMode(2.0 * f, 2.2, 0.65, refractive_index=_dust_index),
        ],
        refractive_index=_dust_index,
        n_radii=128,
    )


def _biomass_burning_config(rh: float = 0.3):
    f = humidity_growth_factor(rh, 0.15)
    return dict(
        name=f"biomass_burning(RH={rh:.0%})",
        modes=[LogNormalMode(0.05 * f, 1.8, 1.0, refractive_index=_brown_carbon_index)],
        refractive_index=_brown_carbon_index,
    )


def _volcanic_stratospheric_config(rh: float = 0.0):
    # Stratospheric sulfate droplets are already aqueous; ``rh`` has no
    # meaning at 20 km, so it is accepted and ignored.
    return dict(
        name="volcanic_stratospheric",
        modes=[LogNormalMode(0.5, 1.6, 1.0, refractive_index=_sulfate_index)],
        refractive_index=_sulfate_index,
        n_radii=64,
    )


def _cirrus_config(rh: float = 0.0):
    # Equivalent spheres.  r_eff = r_g exp(2.5 ln(sigma_g)^2) ~ 8 um.
    return dict(
        name="cirrus",
        modes=[LogNormalMode(6.4, 1.35, 1.0, refractive_index=_ice_index)],
        refractive_index=_ice_index,
        n_radii=128,
    )


def _water_cloud_config(r_eff_um: float = 8.0, rh: float = 0.0):
    sigma_g = 1.3
    sig = np.log(sigma_g)
    r_g = float(r_eff_um) / np.exp(2.5 * sig**2)
    return dict(
        name=f"water_cloud(reff={r_eff_um:g}um)",
        modes=[LogNormalMode(r_g, sigma_g, 1.0, refractive_index=_water_index)],
        refractive_index=_water_index,
        n_radii=128,
    )


class RuralAerosol(MieAerosol):
    """Rural / continental background aerosol (Shettle & Fenn water-soluble).

    Provenance: Shettle & Fenn (1979) "Models for the aerosols of the lower
    atmosphere and the effects of humidity variations on their optical
    properties" (AFGL-TR-79-0214), water-soluble model; the size parameters
    are the standard accumulation-mode values used by OPAC (Hess et al. 1998)
    as well, and the humidity growth follows Shettle & Fenn's ``(1-RH)^-0.35``
    law.

    At 550 nm and RH = 50 % this model gives ``alpha(440/660) = 1.30``,
    ``SSA = 0.982`` and ``g = 0.714``.  ``tau(450)/tau(650) = 1.62``: blue is
    removed faster than red, so this aerosol cannot produce a blue sun.  The
    forward peak is broad; delta-M is *not* required (``f_32 < 1e-4``).
    """

    def __init__(self, rh: float = 0.5, **kw):
        super().__init__(**_rural_config(rh), **kw)


class UrbanAerosol(MieAerosol):
    """Urban / polluted continental aerosol: accumulation mode + soot.

    Provenance: illustrative external mixture of the OPAC "water soluble" and
    "soot" components (Hess et al. 1998), with the soot fraction chosen to
    reproduce the measured urban single-scattering albedo of about 0.88-0.92
    at 550 nm (e.g. Dubovik et al. 2002, AERONET inversions).

    At 550 nm and RH = 50 % this model gives ``alpha(440/660) = 1.25``,
    ``SSA = 0.901`` and ``g = 0.704``.  ``tau(450)/tau(650) = 1.59``.  delta-M
    is not required (``f_32 < 1e-4``), but the absorbing soot mode makes the
    single-scattering albedo strongly wavelength dependent.
    """

    def __init__(self, rh: float = 0.5, **kw):
        super().__init__(**_urban_config(rh), **kw)


class MaritimeAerosol(MieAerosol):
    """Maritime aerosol: sea salt, accumulation + coarse mode.

    Provenance: OPAC sea-salt modes (Hess et al. 1998) and Shettle & Fenn
    (1979) maritime model.  The two modes are an accumulation mode
    (``r_g = 0.10 um``, ``sigma_g = 1.8``, 99.7 % of the number) and a coarse
    mode (``r_g = 1.5 um``, ``sigma_g = 1.9``, 0.3 % of the number, most of the
    mass); sea salt is hygroscopic, with ``f(RH) = (1-RH)^-0.30``.

    At 550 nm and RH = 50 % this model gives ``alpha(440/660) = 0.55``,
    ``SSA = 1.000`` and ``g = 0.813``.  ``tau(450)/tau(650) = 1.23``.  The
    coarse mode produces a narrow forward peak; delta-M is recommended
    (``f_32 = 0.127``, i.e. 13 % of the energy is in the omitted forward
    delta).
    """

    def __init__(self, rh: float = 0.5, **kw):
        super().__init__(**_maritime_config(rh), **kw)


class DesertDustAerosol(MieAerosol):
    """Desert dust: accumulation + coarse mode with iron-oxide absorption.

    Provenance: the bimodal size distribution and the real refractive index
    follow OPAC's mineral-dust model (Hess et al. 1998); the imaginary part is
    an illustrative iron-oxide model with a visible minimum and a near-IR rise,
    which reproduces the *anomalous, non-monotonic* extinction of dust in the
    visible (see ``docs/physics/AEROSOL.md``).  Dust is only weakly hygroscopic here
    (``gamma = 0.05``).

    At 550 nm this model gives ``alpha(440/660) = -0.04``, ``SSA = 0.876``
    and ``g = 0.816``.  The negative Angstrom exponent is the anomalous dust
    behaviour: the broad coarse mode plus the iron-oxide absorption makes the
    extinction *increase* slightly towards the red, with
    ``tau(450)/tau(650) = 0.984`` (a weak blue-sun tendency; a narrow coarse
    mode, e.g. ``r ~ 0.5 um`` and ``sigma_g ~ 1.2``, pushes the ratio down to
    about 0.7 -- and a narrower mode to ~0.5 -- but that is not a realistic
    desert-dust width).  The
    coarse mode has a strong forward peak; delta-M is required
    (``f_32 = 0.368``).
    """

    def __init__(self, rh: float = 0.0, **kw):
        super().__init__(**_desert_dust_config(rh), **kw)


class BiomassBurningAerosol(MieAerosol):
    """Biomass-burning aerosol: fine, strongly absorbing, narrow-ish.

    Provenance: illustrative, chosen to reproduce the measured Angstrom
    exponent (``alpha ~ 1.9-2.2``) and single-scattering albedo
    (``SSA ~ 0.80-0.88`` at 550 nm) of fresh smoke (Dubovik et al. 2002;
    Eck et al. 2005).  The imaginary index is a brown-carbon model, so
    absorption increases towards the blue.

    At 550 nm and RH = 30 % this model gives ``alpha(440/660) = 1.84``,
    ``SSA = 0.865`` and ``g = 0.586``.  ``tau(450)/tau(650) = 1.97``: strongly
    blue-absorbing, the opposite of a blue sun.  delta-M is not required
    (``f_32 < 1e-4``).
    """

    def __init__(self, rh: float = 0.3, **kw):
        super().__init__(**_biomass_burning_config(rh), **kw)


class VolcanicStratosphericAerosol(MieAerosol):
    """Volcanic stratospheric sulfate aerosol (r ~ 0.5 um, nearly conservative).

    Provenance: the canonical post-eruption stratospheric sulfuric-acid
    droplet model (75 % H2SO4 by weight; refractive index after Palmer &
    Williams 1975; size distribution after Stothers 1997 and the OPAC
    "sulfate" mode).  This is the aerosol that produces the red/purple
    afterglows after a major eruption.

    At 550 nm this model gives ``alpha(440/660) = -0.31``, ``SSA > 0.999``
    and ``g = 0.710``.  ``tau(450)/tau(650) = 0.892``: the layer is slightly
    grey/red-biased, not a strong blue absorber, so its role in the red
    afterglow is through near-conservative forward scattering rather than
    through spectral extinction.  The forward peak is moderate; delta-M is
    optional (``f_32 = 0.004``).
    """

    def __init__(self, **kw):
        super().__init__(**_volcanic_stratospheric_config(), **kw)


class CirrusAerosol(MieAerosol):
    """Cirrus ice crystals, represented by **equivalent spheres**.

    This is an approximation: real ice crystals are non-spherical (plates,
    columns, aggregates) and their phase function has halos and a strong
    backscatter peak that Mie theory does not reproduce.  Equivalent spheres
    with the same effective radius get the extinction, SSA and the gross
    forward-scattering properties roughly right, which is enough for the sky
    colour; they are not adequate for halo phenomena.

    Provenance: illustrative ice-cloud effective radius (``r_eff ~ 8 um``,
    ``sigma_g = 1.35``), within the range measured by in-situ campaigns
    (e.g. Heymsfield & McFarquhar 1996); refractive index of ice Ih.

    At 550 nm this model gives ``alpha(440/660) = -0.04``, ``SSA = 0.9998``
    and ``g = 0.868``.  ``tau(450)/tau(650) = 0.986``.  The forward peak is
    strong; delta-M is required (``f_32 = 0.381``).
    """

    def __init__(self, **kw):
        super().__init__(**_cirrus_config(), **kw)


class WaterCloudAerosol(MieAerosol):
    """Water cloud droplets (r_eff 5-15 um, very strong forward peak).

    Provenance: maritime/continental stratocumulus droplet spectra are close
    to log-normal with ``sigma_g ~ 1.3`` and ``r_eff = 5-15 um``
    (e.g. Miles et al. 2000); the refractive index is that of liquid water.
    The default is ``r_eff = 8 um``.

    At 550 nm this model gives ``alpha(440/660) = -0.04``, ``SSA = 1.000``
    and ``g = 0.858``.  ``tau(450)/tau(650) = 0.985``.  The forward peak is
    extremely strong -- at ``r = 8 um`` its width is about ``0.3 deg`` -- so
    delta-M is **required** (``f_32 = 0.383``, i.e. 38 % of the energy is in
    the omitted forward delta), and a plain 32-moment expansion without
    delta-M is meaningless.
    """

    def __init__(self, r_eff_um: float = 8.0, **kw):
        super().__init__(**_water_cloud_config(r_eff_um), **kw)


# ---------------------------------------------------------------------------
# shared construction (memoised preset aerosols)
# ---------------------------------------------------------------------------
_SHARED_AEROSOLS: dict = {}


def shared(cls, *args, **kwargs):
    """Return the one shared instance of ``cls`` for these arguments.

    Preset aerosols are pure functions of their constructor arguments, so
    call sites asking for the same aerosol can share a single instance.
    Sharing is what lets the per-wavelength Mie caches survive the frequent
    ``Atmosphere`` rebuilds done by presets and harnesses: without it every
    ``presets.clear(...)`` pays the full Mie pre-processing again even for
    wavelengths that were already computed.

    Read-only by *structure*, not by convention: :class:`MieAerosol` is a
    frozen dataclass, so attribute assignment raises
    ``FrozenInstanceError`` instead of silently leaking into every
    atmosphere that shares the instance.  Redefining an aerosol is
    :meth:`MieAerosol.replace`, which returns a fresh (unshared) object.
    The cache key is ``(cls, args, sorted kwargs)`` -- exact, because the
    preset arguments are simple floats.
    """
    key = (cls, args, tuple(sorted(kwargs.items())))
    instance = _SHARED_AEROSOLS.get(key)
    if instance is None:
        instance = cls(*args, **kwargs)
        _SHARED_AEROSOLS[key] = instance
    return instance


# ---------------------------------------------------------------------------
# library registry
# ---------------------------------------------------------------------------
LIBRARY = {
    "rural": RuralAerosol,
    "continental": RuralAerosol,
    "urban": UrbanAerosol,
    "maritime": MaritimeAerosol,
    "desert_dust": DesertDustAerosol,
    "dust": DesertDustAerosol,
    "biomass_burning": BiomassBurningAerosol,
    "smoke": BiomassBurningAerosol,
    "volcanic_stratospheric": VolcanicStratosphericAerosol,
    "volcanic": VolcanicStratosphericAerosol,
    "cirrus": CirrusAerosol,
    "water_cloud": WaterCloudAerosol,
    "cloud": WaterCloudAerosol,
}


def get_aerosol(name: str, **kw) -> MieAerosol:
    """Return a named library aerosol.

    Examples
    --------
    >>> get_aerosol("rural", rh=0.8).name
    'rural(RH=80%)'
    >>> get_aerosol("desert_dust").name
    'desert_dust(RH=0%)'
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LIBRARY:
        raise KeyError(f"unknown aerosol {name!r}; choose from {sorted(LIBRARY)}")
    return LIBRARY[key](**kw)


# ---------------------------------------------------------------------------
# Backwards-compatible names
# ---------------------------------------------------------------------------
class HinterlandAerosol(RuralAerosol):
    """Clean continental background (old name; now the Mie ``rural`` model)."""

    def __init__(self):
        super().__init__(rh=0.5)


class HazeAerosol(RuralAerosol):
    """Humid continental haze (old name; now the Mie ``rural`` model at ``RH``)."""

    def __init__(self, rh: float = 0.6):
        super().__init__(rh=rh)


class DustAerosol(DesertDustAerosol):
    """Desert dust (old name; now the Mie ``desert_dust`` model)."""

    def __init__(self, aod_scale: float = 1.0):
        super().__init__()


class SmokeAerosol(BiomassBurningAerosol):
    """Biomass burning (old name; now the Mie ``biomass_burning`` model)."""

    def __init__(self):
        super().__init__()


# ``UrbanAerosol`` and ``MaritimeAerosol`` above are already the new library
# models with the old constructor signature (no required arguments).
