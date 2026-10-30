"""One-dimensional, plane-parallel, homogeneous cloud layers.

This module answers a narrow question well: what does a **uniform deck** do to
the colour of the sky?  It describes a single cloud layer by four numbers --
base, top, optical depth at 550 nm and effective radius -- and derives the
single-scattering properties from Mie theory.

What is modelled
----------------
* A slab between ``base_km`` and ``top_km``, horizontally infinite and
  homogeneous.  Stratus, altostratus and cirrus are all "a layer with a
  different optical depth and droplet size".
* Mie scattering by a log-normal droplet size distribution, normalised to the
  **effective radius** ``r_eff = <r^3>/<r^2>``.  Extinction, single-scattering
  albedo, asymmetry parameter and the Legendre moments of the phase function
  all come out of the same calculation.
* **Delta-M scaling** (Wiscombe 1977) of the phase function, which removes the
  diffraction spike that a truncated Legendre series cannot represent.

What is *not* modelled
----------------------
No three-dimensional structure, no cloud streets, no cumulus, no entrainment,
no ice-crystal habit (cirrus uses an equivalent-sphere approximation), no
vertical variation of droplet size inside the deck and no horizontal variation
of optical depth.  See ``docs/CLOUD.md`` for the full list and for what those
omissions cost.

Units
-----
Heights in km (the public convention of this package), radii in micrometres,
optical depths dimensionless, coefficients m^-1.  Wavelengths in nm.

Mie implementation
------------------
The routines below are a self-contained rewrite of the Mie scattering core in
:mod:`skycolor.aerosol` (Bohren & Huffman 1983), vectorised over particle size
and with a per-size truncation of the Riccati-Bessel recurrences.  It is
deliberately *not* imported from ``aerosol.py``: that module is on a separate
evolution path, and a cloud model must not silently change when an aerosol
model is refactored.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

__all__ = [
    "CloudLayer",
    "CloudOptics",
    "PRESETS",
    "get_preset",
    "stratus",
    "altostratus",
    "cirrus",
    "delta_m",
    "mie_coefficients",
    "mie_optics",
]


# ---------------------------------------------------------------------------
# microphysics
# ---------------------------------------------------------------------------
#: Refractive index of liquid water in the visible.  The imaginary part at
#: visible wavelengths is of order 1e-8 (Hale & Querry 1973), i.e. the
#: single-scattering albedo differs from 1 by less than 1e-6.  We set it to
#: exactly zero and document that real clouds are *almost* conservative.
WATER_REFRACTIVE_INDEX = 1.333 + 0.0j
#: Ice, also as an equivalent sphere (Warren 1984 gives 1.31 in the visible).
ICE_REFRACTIVE_INDEX = 1.31 + 0.0j
#: Density of the condensed phase [kg m^-3].
RHO_WATER = 1000.0
RHO_ICE = 917.0

#: Default geometric standard deviation of the droplet size distribution.
#: ``sigma_g = 1.35`` corresponds to an effective variance of 0.09, which is
#: typical of stratocumulus and altostratus (Miles, Verlinde & Clothiaux 2000).
SIGMA_G = {"water": 1.35, "ice": 1.45}

#: Numerical defaults.  See ``docs/CLOUD.md`` for the convergence study that
#: fixes these numbers: ``n_radii`` controls the size integral, ``n_f``/``n_b``
#: the forward-peak and full-sphere angular quadrature.
N_RADII = 128
N_FORWARD = 64
N_BACKWARD = 384
FORWARD_SPAN = 12.0  #: forward quadrature spans theta < FORWARD_SPAN / x_max
N_MOMENTS = 64


class _Distribution:
    """Log-normal droplet size distribution on a Gauss-Legendre grid in ln r."""

    def __init__(self, r_eff_um: float, sigma_g: float, n_radii: int):
        if r_eff_um <= 0:
            raise ValueError("r_eff_um must be positive")
        if sigma_g <= 1.0:
            raise ValueError("sigma_g must be > 1")
        sig = np.log(float(sigma_g))
        # log-normal:  r_eff = r_g exp(5 sigma^2 / 2)
        self.r_g = float(r_eff_um) / np.exp(2.5 * sig**2)
        r_lo = self.r_g * np.exp(-3.5 * sig)
        r_hi = self.r_g * np.exp(3.5 * sig)
        t, w = np.polynomial.legendre.leggauss(int(n_radii))
        ln_r = 0.5 * (np.log(r_hi) - np.log(r_lo)) * (t + 1.0) + np.log(r_lo)
        jac = 0.5 * (np.log(r_hi) - np.log(r_lo))
        r = np.exp(ln_r)
        dens = np.exp(-0.5 * (ln_r - np.log(self.r_g)) ** 2 / sig**2) / (r * sig)
        w = w * jac * dens * r  # integral over dr
        self.radii_um = r
        self.weights = w / w.sum()
        self.r_eff_um = float(np.sum(self.weights * r**3) / np.sum(self.weights * r**2))
        self.sigma_g = float(sigma_g)

    @property
    def mean_sq_radius(self) -> float:
        """``<r^2>`` in m^2."""
        return float(np.sum(self.weights * (self.radii_um * 1e-6) ** 2))

    @property
    def mean_cube_radius(self) -> float:
        """``<r^3>`` in m^3."""
        return float(np.sum(self.weights * (self.radii_um * 1e-6) ** 3))


# ---------------------------------------------------------------------------
# Mie coefficients
# ---------------------------------------------------------------------------
def mie_coefficients(x, m: complex):
    """Mie coefficients ``a_n``, ``b_n`` for an array of size parameters.

    Parameters
    ----------
    x : array_like
        Size parameters ``2 pi r / lambda``.  May contain a wide range of
        values (a drop-size distribution spans a factor of ten).
    m : complex
        Refractive index.

    Returns
    -------
    dict with ``a``, ``b`` of shape ``(n_max + 1, n_x)``, plus ``qext``,
    ``qsca`` and ``g`` (each ``(n_x,)``).

    Notes
    -----
    The Riccati-Bessel functions are evolved with upward recurrences, which
    grow like ``(2n+1)/x`` per step.  Truncating each column at its own
    ``n_max(x) = x + 4 x^(1/3) + 2`` and freezing the inactive columns keeps
    the recurrences stable from ``x = 1`` to ``x = 10^4``; without the mask a
    single shared ``n_max`` overflows for the small droplets in a broad
    distribution.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n_x = x.size
    n_max_i = np.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0).astype(int)
    n_max = int(n_max_i.max())
    x_safe = np.maximum(x, 1e-12)
    rho = m * x

    # logarithmic derivative D_n(rho) by downward recurrence
    D = np.zeros((n_max + 1, n_x), dtype=complex)
    for n in range(n_max, 0, -1):
        act = n_max_i >= n
        d_next = D[n, act]
        D[n - 1, act] = n / rho[act] - 1.0 / (d_next + n / rho[act])

    # Riccati-Bessel psi_n, chi_n by upward recurrence, masked per size
    psi = np.zeros((n_max + 2, n_x))
    chi = np.zeros((n_max + 2, n_x))
    psi[0] = np.sin(x)
    psi[1] = np.sin(x) / x_safe - np.cos(x)
    chi[0] = np.cos(x)
    chi[1] = np.cos(x) / x_safe + np.sin(x)
    for n in range(1, n_max + 1):
        act = n_max_i >= n + 1
        xa = x_safe[act]
        psi[n + 1, act] = (2 * n + 1) / xa * psi[n, act] - psi[n - 1, act]
        chi[n + 1, act] = (2 * n + 1) / xa * chi[n, act] - chi[n - 1, act]
    xi = psi - 1j * chi

    a = np.zeros((n_max + 1, n_x), dtype=complex)
    b = np.zeros((n_max + 1, n_x), dtype=complex)
    for n in range(1, n_max + 1):
        act = n_max_i >= n
        xa = x_safe[act]
        pa = D[n, act] / m + n / xa
        pb = m * D[n, act] + n / xa
        a[n, act] = (pa * psi[n, act] - psi[n - 1, act]) / (pa * xi[n, act] - xi[n - 1, act])
        b[n, act] = (pb * psi[n, act] - psi[n - 1, act]) / (pb * xi[n, act] - xi[n - 1, act])

    n = np.arange(1, n_max + 1)
    qext = 2.0 / x_safe**2 * np.sum((2 * n + 1)[:, None] * np.real(a[1:] + b[1:]), axis=0)
    qsca = (2.0 / x_safe**2
            * np.sum((2 * n + 1)[:, None] * (np.abs(a[1:]) ** 2 + np.abs(b[1:]) ** 2), axis=0))

    an, bn = a[1:], b[1:]
    an1 = np.vstack([a[2:], np.zeros((1, n_x), dtype=complex)])
    bn1 = np.vstack([b[2:], np.zeros((1, n_x), dtype=complex)])
    g_num = np.sum(n[:, None] * (n + 2)[:, None] / (n + 1)[:, None]
                   * np.real(an * np.conj(an1) + bn * np.conj(bn1)), axis=0)
    g_num += np.sum((2 * n + 1)[:, None] / (n * (n + 1))[:, None]
                    * np.real(an * np.conj(bn)), axis=0)
    g = 4.0 / (x_safe**2 * np.maximum(qsca, 1e-300)) * g_num
    return dict(a=a, b=b, qext=qext, qsca=qsca, g=g)


def _mie_amplitudes(theta, a, b):
    """Mie scattering amplitudes ``S1``, ``S2`` at the angles ``theta``.

    Returns arrays of shape ``(n_theta, n_x)``, ready to be combined with the
    size weights.
    """
    mu = np.cos(theta)
    sin_t = np.sin(theta)
    n_max = a.shape[0] - 1
    pi = np.zeros((n_max + 1, mu.size))
    tau = np.zeros((n_max + 1, mu.size))
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, n_max + 1):
        pi[n] = ((2 * n - 1) * mu * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    nn = np.arange(1, n_max + 1)
    c = (2.0 * nn + 1.0) / (nn * (nn + 1.0))
    A = (a[1:] * c[:, None]).T          # (n_x, n) real part of a_n weighted
    B = (b[1:] * c[:, None]).T
    S1 = A @ pi[1:] + B @ tau[1:]
    S2 = A @ tau[1:] + B @ pi[1:]
    # sin_t is not needed explicitly; kept for clarity of the conventions
    del sin_t
    return S1.T, S2.T                   # (n_theta, n_x)


# ---------------------------------------------------------------------------
# angular quadrature and moments
# ---------------------------------------------------------------------------
def _angular_grid(x_max: float, n_forward: int, n_backward: int,
                  forward_span: float):
    """Composite Gauss-Legendre rule on ``theta`` in ``[0, pi]``.

    The forward diffraction peak has an angular width of order ``1/x``; a
    uniform grid in ``cos(theta)`` would need thousands of points to resolve
    it.  Instead the first segment ``[0, forward_span/x_max]`` carries most of
    the points and the rest of the sphere uses a coarser Gauss rule.

    Returns ``(theta, weights * sin(theta))`` so that
    ``integral_0^pi f(theta) sin(theta) dtheta = sum(weights * f)``.
    """
    theta_1 = min(forward_span / x_max, 0.9 * np.pi)
    xf, wf = np.polynomial.legendre.leggauss(int(n_forward))
    xb, wb = np.polynomial.legendre.leggauss(int(n_backward))
    theta = np.concatenate([0.5 * theta_1 * (xf + 1.0),
                            theta_1 + 0.5 * (np.pi - theta_1) * (xb + 1.0)])
    w = np.concatenate([0.5 * theta_1 * wf,
                        0.5 * (np.pi - theta_1) * wb])
    return theta, w * np.sin(theta)


def _legendre_all(mu: np.ndarray, n_moments: int) -> np.ndarray:
    """``P_l(mu)`` for ``l = 0 .. n_moments-1``; shape ``(n_moments, len(mu))``."""
    out = np.zeros((n_moments, np.asarray(mu).size))
    out[0] = 1.0
    if n_moments > 1:
        out[1] = mu
    for l in range(2, n_moments):
        out[l] = ((2 * l - 1) * mu * out[l - 1] - (l - 1) * out[l - 2]) / l
    return out


def mie_optics(
    wl_nm: float,
    distribution: _Distribution,
    refractive_index: complex,
    n_moments: int = N_MOMENTS,
    n_forward: int = N_FORWARD,
    n_backward: int = N_BACKWARD,
    forward_span: float = FORWARD_SPAN,
) -> dict:
    """Single-scattering properties of a droplet distribution at one wavelength.

    Returns a dict with ``qext``, ``qsca``, ``omega`` (single-scattering
    albedo), ``g``, ``moments`` (Legendre coefficients ``beta_l`` normalised as
    ``p(cos t) = sum_l (2l+1) beta_l P_l``) and ``phase`` (the phase function
    on the quadrature grid, for diagnostics).
    """
    r_m = distribution.radii_um * 1e-6
    x = 2.0 * np.pi * r_m / (wl_nm * 1e-9)
    coeff = mie_coefficients(x, refractive_index)
    qext, qsca, g_single = coeff["qext"], coeff["qsca"], coeff["g"]
    area = np.pi * r_m**2
    cross_sca = distribution.weights * area * qsca
    cross_ext = distribution.weights * area * qext
    total_sca = float(cross_sca.sum())
    total_ext = float(cross_ext.sum())

    theta, w_theta = _angular_grid(float(x.max()), n_forward, n_backward,
                                   forward_span)
    s1, s2 = _mie_amplitudes(theta, coeff["a"], coeff["b"])  # (n_theta, n_x)
    phase_single = (2.0 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
                    / (x**2 * np.maximum(qsca, 1e-300))[None, :])
    phase = (phase_single * cross_sca[None, :]).sum(axis=1) / max(total_sca, 1e-300)

    mu = np.cos(theta)
    pl = _legendre_all(mu, n_moments)
    moments = 0.5 * (pl * (w_theta * phase)[None, :]).sum(axis=1)

    return dict(
        qext=total_ext / (distribution.weights * area).sum(),
        qsca=total_sca / (distribution.weights * area).sum(),
        omega=total_sca / max(total_ext, 1e-300),
        g=float((cross_sca * g_single).sum() / max(total_sca, 1e-300)),
        moments=moments,
        phase=phase,
        theta=theta,
    )


# ---------------------------------------------------------------------------
# delta-M
# ---------------------------------------------------------------------------
def delta_m(tau, omega, g, f):
    """Delta-M scaling of a homogeneous layer (Wiscombe 1977).

    The phase function is split into a forward delta of weight ``f`` plus a
    residual whose Legendre series is the original one with the first ``f``
    removed::

        beta'_l = (beta_l - f) / (1 - f)

    so that ``g' = (g - f)/(1 - f)``.  The optical depth and the
    single-scattering albedo are rescaled in the usual way so that the
    *similarity parameter* ``(1 - g) tau`` (conservative scattering) and the
    total extinction are preserved::

        tau'   = (1 - omega f) tau
        omega' = omega (1 - f) / (1 - omega f)

    Parameters
    ----------
    tau, omega, g, f : float or ndarray
        Optical depth, single-scattering albedo, asymmetry parameter and the
        delta fraction (the Legendre moment at the truncation order).

    Returns
    -------
    ``(tau_scaled, omega_scaled, g_scaled)``

    Notes
    -----
    ``f`` must be the moment at the truncation order used by the solver.  For
    a two-stream solver that order is ``2N = 2``, so ``f = beta_2``.  For an
    Henyey-Greenstein phase function ``beta_l = g^l`` and ``f = g^2``, which is
    exactly the classical delta-Eddington scaling of Joseph, Wiscombe &
    Weinman (1976): the delta-M limit of a two-stream *is* delta-Eddington.
    """
    tau = np.asarray(tau, dtype=float)
    omega = np.asarray(omega, dtype=float)
    g = np.asarray(g, dtype=float)
    f = np.asarray(f, dtype=float)
    f = np.clip(f, 0.0, 0.999)
    denom = np.maximum(1.0 - omega * f, 1e-12)
    tau_s = denom * tau
    omega_s = np.clip(omega * (1.0 - f) / denom, 0.0, 1.0)
    g_s = (g - f) / np.maximum(1.0 - f, 1e-12)
    if tau.ndim == 0 and omega.ndim == 0 and g.ndim == 0 and f.ndim == 0:
        return float(tau_s), float(omega_s), float(g_s)
    return tau_s, omega_s, g_s


# ---------------------------------------------------------------------------
# spectral container
# ---------------------------------------------------------------------------
@dataclass
class CloudOptics:
    """Spectral single-scattering properties of a :class:`CloudLayer`.

    All arrays have shape ``(n_wl,)`` except ``moments`` which is
    ``(n_moments, n_wl)``.  ``tau`` is the *physical* (unscaled) extinction
    optical depth at each wavelength; the delta-M scaled quantities are
    provided separately because the two-stream solver works with those.
    """

    wl: np.ndarray
    tau: np.ndarray
    omega: np.ndarray
    g: np.ndarray
    moments: np.ndarray
    f: np.ndarray
    tau_delta: np.ndarray
    omega_delta: np.ndarray
    g_delta: np.ndarray
    qext: np.ndarray
    r_eff_um: float
    sigma_g: float
    phase_kind: str

    @property
    def n_moments(self) -> int:
        return self.moments.shape[0]

    def at(self, wl_nm) -> "CloudOptics":
        """Interpolate the optics onto another spectral grid."""
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        if wl.size == self.wl.size and np.allclose(wl, self.wl):
            return self

        def interp(a):
            return np.interp(wl, self.wl, a)

        mom = np.vstack([interp(row) for row in self.moments])
        tau = interp(self.tau)
        omega = np.clip(interp(self.omega), 0.0, 1.0)
        g = np.clip(interp(self.g), -1.0, 1.0)
        f = np.clip(interp(self.f), 0.0, 0.999)
        tau_s, omega_s, g_s = delta_m(tau, omega, g, f)
        return CloudOptics(wl=wl, tau=tau, omega=omega, g=g, moments=mom,
                           f=f, tau_delta=tau_s, omega_delta=omega_s,
                           g_delta=g_s, qext=interp(self.qext),
                           r_eff_um=self.r_eff_um, sigma_g=self.sigma_g,
                           phase_kind=self.phase_kind)

    def __repr__(self) -> str:
        return (f"CloudOptics(r_eff={self.r_eff_um:.1f}um, "
                f"tau550={np.interp(550.0, self.wl, self.tau):.3g}, "
                f"g={np.interp(550.0, self.wl, self.g):.3f})")


# ---------------------------------------------------------------------------
# the cloud layer
# ---------------------------------------------------------------------------
@dataclass
class CloudLayer:
    """A homogeneous plane-parallel cloud deck.

    Parameters
    ----------
    base_km, top_km : float
        Geometric base and top of the deck.
    tau550 : float
        Vertical extinction optical depth at 550 nm.  This is the primary
        knob: it is what a satellite retrieval or a textbook table quotes.
    r_eff_um : float
        Effective radius ``<r^3>/<r^2>`` of the droplet distribution.
    n_droplets_cm3 : float, optional
        Droplet number concentration.  If given it is used only for derived
        liquid-water-content diagnostics; ``tau550`` remains authoritative.
    phase : {"water", "ice"}
        Condensed phase.  Ice is treated as an equivalent sphere (see the
        module docstring).
    sigma_g : float, optional
        Geometric standard deviation of the log-normal distribution.
    n_moments : int
        Number of Legendre moments of the phase function to compute.
    """

    base_km: float = 0.5
    top_km: float = 1.5
    tau550: float = 20.0
    r_eff_um: float = 8.0
    n_droplets_cm3: float | None = 200.0
    phase: str = "water"
    sigma_g: float | None = None
    n_moments: int = N_MOMENTS
    n_radii: int = N_RADII
    n_forward: int = N_FORWARD
    n_backward: int = N_BACKWARD
    _optics: CloudOptics | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.top_km <= self.base_km:
            raise ValueError("top_km must be above base_km")
        if self.phase not in ("water", "ice"):
            raise ValueError(f"unknown phase {self.phase!r}")
        if self.tau550 < 0:
            raise ValueError("tau550 must be non-negative")

    # -- derived geometry ----------------------------------------------------
    @property
    def thickness_m(self) -> float:
        return (self.top_km - self.base_km) * 1000.0

    @property
    def mid_km(self) -> float:
        return 0.5 * (self.base_km + self.top_km)

    @property
    def refractive_index(self) -> complex:
        return WATER_REFRACTIVE_INDEX if self.phase == "water" else ICE_REFRACTIVE_INDEX

    @property
    def sigma_g_effective(self) -> float:
        return float(self.sigma_g if self.sigma_g is not None else SIGMA_G[self.phase])

    @property
    def density(self) -> float:
        return RHO_WATER if self.phase == "water" else RHO_ICE

    def distribution(self) -> _Distribution:
        return _Distribution(self.r_eff_um, self.sigma_g_effective, self.n_radii)

    # -- microphysical diagnostics ------------------------------------------
    @property
    def extinction_coefficient(self) -> float:
        """Mean extinction coefficient at 550 nm [m^-1]."""
        return self.tau550 / max(self.thickness_m, 1e-9)

    def liquid_water_content(self, qext550: float | None = None) -> float:
        """Liquid/ice water content [kg m^-3].

        ``LWC = tau rho (4/3) <r^3> / (Q_ext <r^2> H)``, i.e. the water needed
        to produce the requested optical depth.  With ``r_eff`` and
        ``Q_ext = 2`` this reduces to ``2 rho r_eff tau / (3 H)``.
        """
        dist = self.distribution()
        q = 2.0 if qext550 is None else qext550
        return (self.tau550 * self.density * (4.0 / 3.0) * dist.mean_cube_radius
                / (q * dist.mean_sq_radius * max(self.thickness_m, 1e-9)))

    # -- optics --------------------------------------------------------------
    def optics(self, wl_nm) -> CloudOptics:
        """Spectral single-scattering properties on ``wl_nm``.

        The Mie calculation is done on a coarse spectral grid (20 nm, always
        including 550 nm) and interpolated, because every quantity varies
        smoothly with wavelength over the visible -- a cloud's optical depth is
        grey to within a few per cent.  The coarse solution is cached.
        """
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        coarse = _coarse_grid(wl)
        key = tuple(np.round(coarse, 6))
        if self._optics is None or not np.array_equal(self._optics.wl, coarse):
            self._optics = self._compute_optics(coarse)
        return self._optics.at(wl)

    def _compute_optics(self, wl: np.ndarray) -> CloudOptics:
        dist = self.distribution()
        n_wl = wl.size
        n_m = self.n_moments
        moments = np.zeros((n_m, n_wl))
        qext = np.zeros(n_wl)
        qsca = np.zeros(n_wl)
        gg = np.zeros(n_wl)
        for i, lam in enumerate(wl):
            out = mie_optics(lam, dist, self.refractive_index, n_moments=n_m,
                             n_forward=self.n_forward, n_backward=self.n_backward)
            moments[:, i] = out["moments"]
            qext[i] = out["qext"]
            qsca[i] = out["qsca"]
            gg[i] = out["g"]
        # normalise the extinction spectrum to exactly tau550 at 550 nm
        q550 = float(np.interp(550.0, wl, qext))
        tau = self.tau550 * qext / max(q550, 1e-300)
        omega = np.clip(qsca / np.maximum(qext, 1e-300), 0.0, 1.0)
        g = np.clip(gg, -1.0, 1.0)
        f = np.clip(moments[2].copy(), 0.0, 0.999)
        tau_s, omega_s, g_s = delta_m(tau, omega, g, f)
        return CloudOptics(wl=wl.copy(), tau=tau, omega=omega, g=g,
                           moments=moments, f=f, tau_delta=tau_s,
                           omega_delta=omega_s, g_delta=g_s, qext=qext,
                           r_eff_um=self.r_eff_um,
                           sigma_g=self.sigma_g_effective,
                           phase_kind=self.phase)

    # -- convenience ---------------------------------------------------------
    def optical_depth(self, wl_nm) -> np.ndarray:
        """Vertical extinction optical depth at ``wl_nm``."""
        return self.optics(wl_nm).tau

    def transmittance(self, wl_nm, mu=1.0) -> np.ndarray:
        """Direct transmittance through the whole deck along zenith cosine ``mu``."""
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
        return np.exp(-self.optics(wl_nm).tau / mu)

    def replace(self, **kw) -> "CloudLayer":
        new = replace(self, **kw)
        new._optics = None
        return new

    def describe(self) -> str:
        lwc = self.liquid_water_content()
        cone = self.n_droplets_cm3
        parts = [f"{self.phase} deck {self.base_km:g}-{self.top_km:g} km",
                 f"tau550={self.tau550:g}",
                 f"r_eff={self.r_eff_um:g} um",
                 f"LWC={lwc * 1e3:.3f} g/m3"]
        if cone is not None:
            parts.append(f"n={cone:g} cm^-3")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"CloudLayer({self.describe()})"


def _coarse_grid(wl: np.ndarray, step: float = 20.0) -> np.ndarray:
    """Coarse spectral grid covering ``wl`` and always containing 550 nm."""
    lo = min(float(wl.min()), 550.0)
    hi = max(float(wl.max()), 550.0)
    n = max(2, int(np.ceil((hi - lo) / step)) + 1)
    grid = np.linspace(lo, hi, n)
    return np.unique(np.concatenate([grid, [550.0]]))


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
def stratus(tau550: float = 20.0, base_km: float = 0.5, top_km: float = 1.5,
            r_eff_um: float = 8.0, **kw) -> CloudLayer:
    """Classic overcast stratus: thick, low, small droplets."""
    return CloudLayer(base_km=base_km, top_km=top_km, tau550=tau550,
                      r_eff_um=r_eff_um, n_droplets_cm3=200.0, phase="water",
                      **kw)


def altostratus(tau550: float = 5.0, base_km: float = 3.0, top_km: float = 5.0,
                r_eff_um: float = 6.0, **kw) -> CloudLayer:
    """Mid-level deck: thinner, so the reddened sunlight gets through."""
    return CloudLayer(base_km=base_km, top_km=top_km, tau550=tau550,
                      r_eff_um=r_eff_um, n_droplets_cm3=150.0, phase="water",
                      **kw)


def cirrus(tau550: float = 0.5, base_km: float = 8.0, top_km: float = 10.0,
           r_eff_um: float = 40.0, **kw) -> CloudLayer:
    """Thin ice cloud.  Ice crystals are approximated by equivalent spheres.

    The effective radius is that of a sphere with the same volume/surface
    ratio as the crystal population; real cirrus phase functions have strong
    halo and (for oriented plates) specular features that this cannot
    reproduce.
    """
    return CloudLayer(base_km=base_km, top_km=top_km, tau550=tau550,
                      r_eff_um=r_eff_um, n_droplets_cm3=0.05, phase="ice",
                      **kw)


PRESETS = {
    "stratus": stratus,
    "altostratus": altostratus,
    "cirrus": cirrus,
}


def get_preset(name: str, **kw) -> CloudLayer:
    """Build one of the named presets (``stratus``, ``altostratus``, ``cirrus``)."""
    try:
        factory = PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown cloud preset {name!r}; choose from {sorted(PRESETS)}") from None
    return factory(**kw)
