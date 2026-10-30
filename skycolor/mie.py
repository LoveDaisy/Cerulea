"""Mie theory core used by :mod:`skycolor.aerosol`.

This module contains the numerical engine only: Mie coefficients for a
homogeneous sphere, the exact Legendre moments of the single-sphere phase
function, and a vectorised phase-function evaluator.  The atmospheric
aerosol classes (size distributions, refractive-index dispersion, delta-M) live
in :mod:`skycolor.aerosol`.

Notation
--------
``x = 2 pi r / lambda``
    size parameter (dimensionless).
``m = n + i k``
    complex refractive index; ``k >= 0`` means absorption.
``a_n, b_n``
    Mie coefficients (Bohren & Huffman 1983, eqs. 4.52).
``p(mu)``
    scattering phase function normalised so that ``int p dOmega = 4 pi``.
    It is expanded as ``p(mu) = sum_l (2l+1) beta_l P_l(mu)``.

The phase function is evaluated with the usual Mie amplitude functions

.. math::

    S_1(\\theta) = \\sum_n \\frac{2n+1}{n(n+1)}
        \\left[a_n \\pi_n(\\cos\\theta) + b_n \\tau_n(\\cos\\theta)\\right],

    S_2(\\theta) = \\sum_n \\frac{2n+1}{n(n+1)}
        \\left[a_n \\tau_n(\\cos\\theta) + b_n \\pi_n(\\cos\\theta)\\right],

    p(\\theta) = \\frac{2\\,(|S_1|^2 + |S_2|^2)}{x^2 Q_\\mathrm{sca}}.

The last normalisation (the factor 2) is the convention used throughout this
project; it gives ``int p dOmega = 4 pi`` and matches
``skycolor.spectra.rayleigh_phase``.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = [
    "MieCoefficients",
    "mie_coefficients",
    "mie_asymmetry",
    "mie_phase",
    "mie_moment_numerators",
    "mie_legendre_moments",
    "gaunt_squared",
    "phase_table_angles",
    "log_normal_nodes",
]


# ===========================================================================
# Mie coefficients
# ===========================================================================
@dataclass(frozen=True)
class MieCoefficients:
    """Mie coefficients and efficiencies for one homogeneous sphere.

    ``a`` and ``b`` are indexed ``1..nmax``; entry 0 is unused and set to zero.
    """

    x: float
    m: complex
    a: np.ndarray
    b: np.ndarray
    qext: float
    qsca: float
    nmax: int

    @property
    def qabs(self) -> float:
        return self.qext - self.qsca

    @property
    def g(self) -> float:
        """Analytic asymmetry parameter ``<cos theta>``."""
        return mie_asymmetry(self.a, self.b, self.x, self.qsca, self.nmax)


def _nmax(x: float) -> int:
    """Wiscombe's truncation criterion, the one used by BHMIE."""
    return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)


def mie_coefficients(x: float, m: complex, n_max: int | None = None) -> MieCoefficients:
    """Mie coefficients for a sphere of size parameter ``x`` and index ``m``.

    The implementation follows Bohren & Huffman (1983): the logarithmic
    derivative ``D_n(mx)`` is computed by downward recurrence (always stable),
    the Riccati-Bessel functions ``psi_n(x)`` and ``chi_n(x)`` by upward
    recurrence, and the series are truncated at
    ``nmax = x + 4 x^(1/3) + 2``.  It is accurate to roughly 1e-6 relative for
    ``x`` up to several thousand, i.e. far beyond the ``x ~ 200`` required by
    the aerosol library.
    """
    x = float(x)
    m = complex(m)
    if x <= 0.0:
        z = np.zeros(1, dtype=complex)
        return MieCoefficients(x, m, z, z, 0.0, 0.0, 0)

    if n_max is None:
        n_max = _nmax(x)
    n_max = max(int(n_max), 1)
    nmx = max(n_max, int(abs(m * x))) + 15

    # --- logarithmic derivative D_n(rho), downward recurrence --------------
    rho = m * x
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / rho - 1.0 / (D[n] + n / rho)

    # --- Riccati-Bessel functions ------------------------------------------
    psi = np.zeros(n_max + 2)
    psi[0] = np.sin(x)
    psi[1] = np.sin(x) / x - np.cos(x)
    for n in range(1, n_max + 1):
        psi[n + 1] = (2 * n + 1) / x * psi[n] - psi[n - 1]

    chi = np.zeros(n_max + 2)
    chi[0] = np.cos(x)
    chi[1] = np.cos(x) / x + np.sin(x)
    for n in range(1, n_max + 1):
        chi[n + 1] = (2 * n + 1) / x * chi[n] - chi[n - 1]

    xi = psi - 1j * chi

    # --- Mie coefficients ---------------------------------------------------
    a = np.zeros(n_max + 1, dtype=complex)
    b = np.zeros(n_max + 1, dtype=complex)
    for n in range(1, n_max + 1):
        dn = D[n]
        pa = dn / m + n / x
        pb = m * dn + n / x
        a[n] = (pa * psi[n] - psi[n - 1]) / (pa * xi[n] - xi[n - 1])
        b[n] = (pb * psi[n] - psi[n - 1]) / (pb * xi[n] - xi[n - 1])

    n_arr = np.arange(1, n_max + 1)
    qext = 2.0 / x**2 * np.sum((2 * n_arr + 1) * np.real(a[1:] + b[1:]))
    qsca = 2.0 / x**2 * np.sum(
        (2 * n_arr + 1) * (np.abs(a[1:]) ** 2 + np.abs(b[1:]) ** 2)
    )
    return MieCoefficients(x, m, a, b, float(qext), float(qsca), n_max)


def mie_asymmetry(
    a: np.ndarray, b: np.ndarray, x: float, qsca: float, nmax: int | None = None
) -> float:
    """Analytic asymmetry parameter from the Mie coefficients.

    This is Bohren & Huffman (1983) eq. (4.74); it is exact and, unlike a
    quadrature of the phase function, remains in ``(-1, 1)`` for any size
    parameter.
    """
    if nmax is None:
        nmax = len(a) - 1
    if nmax <= 0 or qsca <= 0.0 or x <= 0.0:
        return 0.0
    n = np.arange(1, nmax + 1, dtype=float)
    an = a[1 : nmax + 1]
    bn = b[1 : nmax + 1]
    an1 = np.append(a[2 : nmax + 1], 0.0)
    bn1 = np.append(b[2 : nmax + 1], 0.0)
    g_num = np.sum(
        n * (n + 2.0) / (n + 1.0)
        * np.real(an * np.conj(an1) + bn * np.conj(bn1))
    )
    g_num += np.sum((2.0 * n + 1.0) / (n * (n + 1.0)) * np.real(an * np.conj(bn)))
    g = 4.0 / (x * x * qsca) * g_num
    return float(np.clip(g, -1.0, 1.0))


# ===========================================================================
# Analytic Legendre moments
# ===========================================================================
#: The phase function is a polynomial in mu.  The key identity used here is
#:
#:     S_1(mu) = sum_a P_a P_a(mu),   S_2(mu) = sum_a Q_a P_a(mu),
#:
#: where
#:
#:     P_a = sum_n c_n (a_n A_na + b_n B_na),
#:     Q_a = sum_n c_n (b_n A_na + a_n B_na),
#:     c_n = (2n+1)/(n(n+1)),
#:
#: and A_na, B_na are the Legendre coefficients of the angular functions
#: pi_n = P_n' and tau_n.  They have a simple closed form (see
#: ``_A_coefficient`` / ``_B_coefficient``) and are *independent of x and m*.
#: Then
#:
#:     beta_l = (2 / (x^2 Q_sca)) sum_{a,a'} G_{aa'l}
#:              [P_a P_{a'}^* + Q_a Q_{a'}^*],
#:
#: where G_{aa'l} = <a a' l; 0 0 0>^2 is a squared Wigner 3-j symbol.  The
#: selection rules |a-a'| <= l and a+a'+l even make the sum O(nmax * L^2)
#: rather than O(nmax^2 * L).


def _A_coefficient(n: int, a: int) -> float:
    """Legendre coefficient of ``pi_n`` at ``P_a``.

    ``pi_n = P_n'`` and the classical derivative expansion gives
    ``pi_n = sum_{a=n-1,n-3,...} (2a+1) P_a``.
    """
    if a < 0 or a > n - 1 or (n - 1 - a) % 2 != 0:
        return 0.0
    return float(2 * a + 1)


def _B_coefficient(n: int, a: int) -> float:
    """Legendre coefficient of ``tau_n`` at ``P_a``.

    ``tau_n = n mu pi_n - (n+1) pi_{n-1}``.  Expanding ``mu P_k`` with the
    three-term recurrence gives the sparse pattern below; it is used only for
    documentation and for cross-checking the vectorised construction.
    """
    if a < 0 or a > n or (n - a) % 2 != 0:
        return 0.0

    def in_S(m: int, k: int) -> bool:
        """True if ``k`` is in {m-1, m-3, ...} (the support of pi_m)."""
        return 0 <= k <= m - 1 and (m - 1 - k) % 2 == 0

    val = 0.0
    # from n * mu * pi_n:  mu P_k -> (k+1) P_{k+1} + k P_{k-1}
    if in_S(n, a - 1):
        val += n * a
    if in_S(n, a + 1):
        val += n * (a + 1)
    # from -(n+1) * pi_{n-1}
    if in_S(n - 1, a):
        val -= (n + 1) * (2 * a + 1)
    return float(val)


def _gaunt_sq(a: np.ndarray, b: np.ndarray, l: int) -> np.ndarray:
    """Squared Wigner 3-j symbol ``<a b l; 0 0 0>^2`` for arrays ``a, b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    J = a + b + l
    ok = (np.abs(a - b) <= l) & (a + b >= l) & (np.mod(J, 2) == 0)
    out = np.zeros_like(a)
    if not np.any(ok):
        return out
    aa = a[ok]
    bb = b[ok]
    JJ = J[ok]
    logabs = 0.5 * (
        gammaln(JJ - 2 * aa + 1)
        + gammaln(JJ - 2 * bb + 1)
        + gammaln(JJ - 2 * l + 1)
        - gammaln(JJ + 2)
    )
    logabs += (
        gammaln(JJ / 2 + 1)
        - gammaln(JJ / 2 - aa + 1)
        - gammaln(JJ / 2 - bb + 1)
        - gammaln(JJ / 2 - l + 1)
    )
    out[ok] = np.exp(2.0 * logabs)
    return out


def gaunt_squared(a, b, l: int) -> np.ndarray:
    """Public alias for the squared Wigner 3-j symbol ``<a b l; 0 0 0>^2``."""
    return _gaunt_sq(a, b, l)


@functools.lru_cache(maxsize=8)
def _gaunt_bands(l_max: int, a_max: int):
    """Pre-computed ``(d, a0, G)`` bands for every ``l <= l_max``.

    For a given ``l`` the only non-zero entries have ``a' = a + d`` with
    ``d`` in ``{-l, -l+2, ..., l}``.  Each band stores the range of ``a`` and
    the corresponding ``G`` array, so the moment sum becomes a handful of
    vector dot products per ``l``.
    """
    bands = []
    for l in range(l_max + 1):
        per_l = []
        for d in range(-l, l + 1, 2):
            a0 = max(0, -d)
            a1 = min(a_max, a_max - d)
            if a1 < a0:
                continue
            a = np.arange(a0, a1 + 1)
            b = a + d
            G = _gaunt_sq(a, b, l)
            if np.any(G):
                # store as read-only arrays so the cache is safe to share
                G.flags.writeable = False
                per_l.append((int(d), int(a0), G))
        bands.append(tuple(per_l))
    return tuple(bands)


def mie_moment_numerators(
    coeff: MieCoefficients, n_moments: int
) -> np.ndarray:
    """Return ``U_l = Q_sca * beta_l`` for ``l = 0 .. n_moments-1``.

    Dividing by ``Q_sca`` gives the physical moments; keeping the numerator
    separate lets the size average cancel ``Q_sca`` analytically.
    """
    x = coeff.x
    qsca = coeff.qsca
    nmax = coeff.nmax
    if x <= 0.0 or qsca <= 0.0 or nmax <= 0 or n_moments <= 0:
        return np.zeros(n_moments)

    a = coeff.a
    b = coeff.b
    n = np.arange(1, nmax + 1, dtype=float)
    c = np.zeros(nmax + 1)
    c[1:] = (2.0 * n + 1.0) / (n * (n + 1.0))

    # Tail sums S^a_k = sum_{n>=k, n step 2} c_n a_n (same for b).
    Sa = np.zeros(nmax + 5, dtype=complex)
    Sb = np.zeros(nmax + 5, dtype=complex)
    for k in range(nmax, 0, -1):
        Sa[k] = c[k] * a[k] + Sa[k + 2]
        Sb[k] = c[k] * b[k] + Sb[k + 2]

    k = np.arange(nmax + 2)
    ca = np.zeros(nmax + 2)
    ca[1 : nmax + 1] = c[1:]
    aa = np.zeros(nmax + 2, dtype=complex)
    aa[1 : nmax + 1] = a[1:]
    bb = np.zeros(nmax + 2, dtype=complex)
    bb[1 : nmax + 1] = b[1:]

    # P_a, Q_a (the Legendre coefficients of S1 and S2).
    P = (2.0 * k + 1.0) * (Sa[k + 1] - Sb[k + 2]) + k * k * ca * bb
    Q = (2.0 * k + 1.0) * (Sb[k + 1] - Sa[k + 2]) + k * k * ca * aa

    bands = _gaunt_bands(n_moments - 1, max(256, nmax))
    C = 2.0 / (x * x)  # qsca deliberately left in the numerator
    out = np.zeros(n_moments)
    for l in range(n_moments):
        tot = 0.0 + 0.0j
        for d, a0, G in bands[l]:
            a_hi = min(nmax, nmax - d, a0 + G.size - 1)
            if a_hi < a0:
                continue
            n_here = a_hi - a0 + 1
            sl = slice(a0, a_hi + 1)
            sl2 = slice(a0 + d, a_hi + 1 + d)
            tot += np.dot(
                G[:n_here],
                P[sl] * np.conj(P[sl2]) + Q[sl] * np.conj(Q[sl2]),
            )
        out[l] = (C * tot).real
    return out


def mie_legendre_moments(
    coeff: MieCoefficients, n_moments: int
) -> np.ndarray:
    """Physical Legendre moments ``beta_0 .. beta_{n_moments-1}``."""
    U = mie_moment_numerators(coeff, n_moments)
    if coeff.qsca > 0.0:
        U = U / coeff.qsca
    return U


# ===========================================================================
# Phase function on an angular grid
# ===========================================================================
def _angular_functions(mu: np.ndarray, nmax: int):
    """Return ``pi[n]`` and ``tau[n]`` arrays of shape ``(nmax+1, len(mu))``."""
    mu = np.asarray(mu, dtype=float)
    pi = np.zeros((nmax + 1, mu.size))
    tau = np.zeros((nmax + 1, mu.size))
    if nmax >= 1:
        pi[1] = 1.0
        tau[1] = mu
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) * mu * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi, tau


def mie_phase(
    coeff: MieCoefficients,
    theta_rad,
    *,
    mu: bool = False,
) -> np.ndarray:
    """Phase function ``p`` at scattering angles ``theta_rad`` (or ``mu``).

    Returns an array with the same shape as the input.  ``p`` is normalised so
    that ``int p dOmega = 4 pi``.
    """
    if coeff.x <= 0.0 or coeff.qsca <= 0.0:
        return np.zeros_like(np.asarray(theta_rad, dtype=float))
    theta = np.asarray(theta_rad, dtype=float)
    mu_arr = theta if mu else np.cos(theta)
    pi, tau = _angular_functions(mu_arr.ravel(), coeff.nmax)

    a = coeff.a
    b = coeff.b
    nmax = coeff.nmax
    s1 = np.zeros(pi.shape[1], dtype=complex)
    s2 = np.zeros(pi.shape[1], dtype=complex)
    for n in range(1, nmax + 1):
        cf = (2 * n + 1) / (n * (n + 1))
        s1 += cf * (a[n] * pi[n] + b[n] * tau[n])
        s2 += cf * (a[n] * tau[n] + b[n] * pi[n])
    p = 2.0 * (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (coeff.x**2 * coeff.qsca)
    return p.reshape(theta.shape)


# ===========================================================================
# Angular grid for the adaptive phase table
# ===========================================================================
def phase_table_angles(
    x_max: float,
    *,
    forward_decades: float = 4.0,
    n_forward: int | None = None,
    n_mid: int | None = None,
    n_back: int = 80,
) -> np.ndarray:
    """Adaptive grid in scattering angle (degrees) for a phase-function table.

    The grid resolves the forward diffraction peak of the largest sphere:

    * ``0`` and a log-spaced segment from ``1e-4`` to ``1`` degree,
    * a linear segment from ``1`` to ``179`` degrees,
    * a log-spaced segment for the last degree before ``180`` (the glory).

    ``n_forward`` and ``n_mid`` default to values that scale with ``x_max``:
    the forward peak has width ``~1/x_max`` radians and the interference
    structure in the middle has period ``~2 pi / x_max`` radians, so more
    points are needed for larger particles.
    """
    x_max = max(float(x_max), 1e-3)
    if n_forward is None:
        n_forward = int(np.clip(3.0 * x_max, 80, 800))
    if n_mid is None:
        n_mid = int(np.clip(8.0 * x_max, 240, 2400))
    n_forward = max(int(n_forward), 16)
    n_mid = max(int(n_mid), 16)
    n_back = max(int(n_back), 8)
    theta_f = np.logspace(-forward_decades, 0.0, n_forward, endpoint=False)
    theta_m = np.linspace(1.0, 179.0, n_mid, endpoint=True)
    theta_b = 180.0 - np.logspace(0.0, -forward_decades, n_back, endpoint=True)
    theta = np.concatenate(([0.0], theta_f, theta_m, theta_b, [180.0]))
    # drop duplicates introduced at the segment joins
    theta = np.unique(theta)
    return theta


# ===========================================================================
# Size-distribution nodes
# ===========================================================================
def log_normal_nodes(
    r_g_um: float,
    sigma_g: float,
    n_radii: int = 96,
    n_sigma: float = 3.0,
    r_min_um: float = 1e-4,
    r_max_um: float = 30.0,
    weight: float = 1.0,
):
    """Nodes and weights for a truncated log-normal size distribution.

    The quadrature is the trapezoidal rule in ``t = ln(r/r_g)/ln(sigma_g)``,
    which is the natural variable of a log-normal.  The distribution is
    truncated at ``|t| <= n_sigma`` and at the hard limits
    ``[r_min_um, r_max_um]``; weights are renormalised inside the truncated
    support.

    Returns
    -------
    r_um : ndarray
        Radii in micrometres.
    w : ndarray
        Number weights, ``sum(w) = weight``.
    t : ndarray
        The corresponding standardised log-radius values.
    """
    r_g_um = float(r_g_um)
    sigma_g = float(sigma_g)
    if r_g_um <= 0.0 or sigma_g <= 0.0 or n_radii < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    sig = np.log(sigma_g)
    if sig <= 0.0:
        # monodisperse limit: a single radius
        if r_min_um <= r_g_um <= r_max_um:
            return np.array([r_g_um]), np.array([weight]), np.array([0.0])
        return np.zeros(0), np.zeros(0), np.zeros(0)

    t = np.linspace(-n_sigma, n_sigma, int(n_radii))
    r = r_g_um * np.exp(sig * t)
    keep = (r >= r_min_um) & (r <= r_max_um)
    if not np.any(keep):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    r = r[keep]
    t = t[keep]
    w = np.exp(-0.5 * t * t)
    w = w / w.sum() * float(weight)
    return r, w, t
