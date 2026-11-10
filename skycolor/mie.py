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
    "MieCoefficientsBatch",
    "mie_coefficients",
    "mie_coefficients_batch",
    "mie_asymmetry",
    "mie_phase",
    "mie_phase_batch",
    "mie_moment_numerators",
    "mie_moment_numerators_batch",
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


# ---------------------------------------------------------------------------
# Fused multiply-add without hardware FMA (see mie_coefficients_batch)
# ---------------------------------------------------------------------------
def _fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``fl(a*b + c)`` with a single rounding, without hardware FMA.

    The CPython build on this platform contracts ``x + y*z`` inside its
    complex division into a fused multiply-add, so reproducing that division
    bit-for-bit needs an FMA primitive; numpy has none.  Dekker's twoProduct
    (veltkamp split) recovers the exact product error, and a twoSum chain
    collapses ``p + err + c`` so that only the final addition rounds.  Verified
    bitwise against libm ``fma`` (200k random triples) and against CPython's
    contracted division on the real (node x wavelength) pair population
    (task-cold-optics probe, A3c: 0/28512 pairs differ).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    p = a * b
    split = 134217729.0  # 2**27 + 1 (Dekker split for IEEE double)
    ca = split * a
    ah = ca - (ca - a)
    al = a - ah
    cb = split * b
    bh = cb - (cb - b)
    bl = b - bh
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    bb = s - p
    f = (p - (s - bb)) + (c - bb)  # Knuth twoSum: p + c = s + f exactly
    u = f + err
    bb2 = u - f
    g = (f - (u - bb2)) + (err - bb2)  # twoSum: f + err = u + g exactly
    return s + u + g


def _smith_division_constants(re: np.ndarray, im: np.ndarray):
    """Hoisted per-denominator Smith constants for CPython complex division.

    CPython computes ``num / den`` (complex) by Smith's method with FMA
    contraction; for the batched Mie kernel the only CPython-division site is
    ``n / rho`` whose denominator ``rho`` is fixed per size node, so the
    branch decision, ratio and (fused) denominator are computed once per pair
    here and reused for every recurrence step.

    Returns ``(big, ratio1, denom1, ratio2, denom2)``: with ``big`` selecting
    branch 1 (``|re| >= |im|``), ``n / den`` is ``n/denom1 - i*(n*ratio1)/denom1``
    on branch 1 and ``(n*ratio2)/denom2 - i*n/denom2`` on branch 2 (the
    numerators' own FMA sites collapse to plain products for a real numerator).
    """
    re = np.asarray(re, dtype=float)
    im = np.asarray(im, dtype=float)
    big = np.abs(re) >= np.abs(im)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio1 = np.where(big, im / np.where(big, re, 1.0), 0.0)
        denom1 = np.where(big, _fma(im, ratio1, np.where(big, re, 1.0)), 1.0)
        ratio2 = np.where(~big, re / np.where(~big, im, 1.0), 0.0)
        denom2 = np.where(~big, _fma(re, ratio2, np.where(~big, im, 1.0)), 1.0)
    return big, ratio1, denom1, ratio2, denom2


@dataclass(frozen=True)
class MieCoefficientsBatch:
    """Mie coefficients for one wavelength's size nodes, computed in batch.

    Bitwise-equal to calling :func:`mie_coefficients` on each pair
    (task-cold-optics probe A3c, 0/28512 real pairs differing).  The two
    division sites of the scalar path run on different engines: ``n / rho``
    has pure-Python operands and is CPython complex division -- reproduced by
    :func:`_smith_division_constants` -- while ``1.0 / (D[n] + n/rho)`` has a
    numpy complex128 operand and is bitwise-equal to plain numpy array
    division.  Rows are zero beyond each pair's ``n_max``, matching the
    scalar zeros-padded layout.

    Attributes are shaped ``(n_pair,)`` / ``(n_pair, n_max_pad + 1)``.
    """

    x: np.ndarray
    m: np.ndarray
    a: np.ndarray
    b: np.ndarray
    qext: np.ndarray
    qsca: np.ndarray
    n_max: np.ndarray
    nmx: np.ndarray

    @property
    def n_pair(self) -> int:
        return int(self.x.size)


def mie_coefficients_batch(x, m) -> MieCoefficientsBatch:
    """Batched :func:`mie_coefficients` over size parameters ``x``.

    Parameters
    ----------
    x : array_like (n_pair,)
        Size parameters; must be positive (pairs with ``x < 1e-4`` are the
        caller's business, mirroring the scalar early-out).
    m : array_like (n_pair,) complex
        Refractive index per pair.

    Returns
    -------
    MieCoefficientsBatch
        ``a``/``b`` rows are the per-pair coefficient arrays zero-padded to a
        common ``n_max_pad + 1`` columns; ``qext``/``qsca`` are reduced with
        the scalar path's per-pair ``np.sum`` calls so they stay bitwise too.
    """
    x_arr = np.asarray(x, dtype=float)
    m_arr = np.asarray(m, dtype=complex)
    if x_arr.ndim != 1 or m_arr.shape != x_arr.shape:
        raise ValueError("x and m must be 1-D arrays of equal length")
    P = x_arr.size
    if P == 0:
        empty = np.zeros(0)
        return MieCoefficientsBatch(empty, empty.astype(complex),
                                    np.zeros((0, 1), dtype=complex),
                                    np.zeros((0, 1), dtype=complex),
                                    empty, empty, empty.astype(int),
                                    empty.astype(int))

    # per-pair truncation with the scalar (CPython) semantics; the loop is
    # O(n_pair) scalar work, negligible next to the recurrences
    n_max_arr = np.array([max(int(_nmax(float(xv))), 1) for xv in x_arr])
    nmx_arr = np.array(
        [max(int(n_max_arr[p]), int(abs(complex(m_arr[p]) * float(x_arr[p])))) + 15
         for p in range(P)])
    nmx_pad = int(nmx_arr.max())
    n_max_pad = int(n_max_arr.max())

    # --- n / rho via hoisted Smith constants (CPython division site) -------
    rho = m_arr * x_arr
    big, ratio1, denom1, ratio2, denom2 = _smith_division_constants(
        np.real(rho), np.imag(rho))
    # work TRANSPOSED (recurrence-step rows x pair columns): the per-step
    # operands become contiguous rows instead of cache-hostile matrix columns.
    # Branch 1 (|re| >= |im|) is built for all pairs without a where() over
    # the full (nmx, P) matrix -- every atmospheric index in the library is
    # branch 1; a branch-2 pair, if one ever appears, is patched column-wise.
    n_val = np.arange(nmx_pad + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # built into preallocated slots: a chained expression here
        # materialises ~5 full-size temporaries per call
        n_over_rho = np.empty((nmx_pad + 1, P), dtype=complex)
        scr = np.empty((nmx_pad + 1, P))
        np.divide(n_val[:, None], denom1[None, :], out=n_over_rho.real)
        np.multiply(n_val[:, None], ratio1[None, :], out=scr)
        np.divide(scr, denom1[None, :], out=scr)
        np.negative(scr, out=scr)
        n_over_rho.imag = scr
    if not big.all():
        idx2 = np.nonzero(~big)[0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.multiply(n_val[:, None], ratio2[None, idx2], out=scr)
            np.divide(scr, denom2[None, idx2], out=scr)
            n_over_rho[:, idx2].real = scr
            np.divide(n_val[:, None], denom2[None, idx2], out=scr)
            np.negative(scr, out=scr)
            n_over_rho[:, idx2].imag = scr

    # --- D down-recurrence, one masked step per n --------------------------
    # rows beyond a pair's nmx keep their zero seed: the masked updates leave
    # each pair's trajectory starting from its own nmx, like the scalar loop
    D = np.zeros((nmx_pad + 1, P), dtype=complex)
    for n in range(nmx_pad, 0, -1):
        mask = nmx_arr >= n
        tm = n_over_rho[n, mask]
        # 1.0/(D[n] + n/rho): the scalar path's operand is a numpy complex128
        # scalar, i.e. numpy scalar division -- bitwise-equal to array division
        D[n - 1, mask] = tm - 1.0 / (D[n, mask] + tm)

    # --- psi / chi up-recurrence (real arithmetic) --------------------------
    psi = np.zeros((n_max_pad + 2, P))
    chi = np.zeros((n_max_pad + 2, P))
    sx = np.sin(x_arr)
    cx = np.cos(x_arr)
    psi[0] = sx
    chi[0] = cx
    psi[1] = sx / x_arr - cx
    chi[1] = cx / x_arr + sx
    for n in range(1, n_max_pad + 1):
        mask = n_max_arr >= n
        f = ((2 * n + 1) / x_arr)[mask]
        psi[n + 1, mask] = f * psi[n, mask] - psi[n - 1, mask]
        chi[n + 1, mask] = f * chi[n, mask] - chi[n - 1, mask]

    xi = psi - 1j * chi

    # --- a / b combination (transposed; transposed back for the public layout)
    aT = np.zeros((n_max_pad + 1, P), dtype=complex)
    bT = np.zeros((n_max_pad + 1, P), dtype=complex)
    for n in range(1, n_max_pad + 1):
        mask = n_max_arr >= n
        dn = D[n, mask]
        pa = dn / m_arr[mask] + n / x_arr[mask]
        pb = m_arr[mask] * dn + n / x_arr[mask]
        aT[n, mask] = (pa * psi[n, mask] - psi[n - 1, mask]) \
            / (pa * xi[n, mask] - xi[n - 1, mask])
        bT[n, mask] = (pb * psi[n, mask] - psi[n - 1, mask]) \
            / (pb * xi[n, mask] - xi[n - 1, mask])
    a = np.ascontiguousarray(aT.T)
    b = np.ascontiguousarray(bT.T)

    # --- cheap reductions stay per-pair (K2): same np.sum calls as scalar ---
    qext = np.empty(P)
    qsca = np.empty(P)
    for j in range(P):
        xv = float(x_arr[j])
        nx = int(n_max_arr[j])
        n_arr = np.arange(1, nx + 1)
        aj = a[j, 1: nx + 1]
        bj = b[j, 1: nx + 1]
        qext[j] = 2.0 / xv**2 * np.sum((2 * n_arr + 1) * np.real(aj + bj))
        qsca[j] = 2.0 / xv**2 * np.sum(
            (2 * n_arr + 1) * (np.abs(aj) ** 2 + np.abs(bj) ** 2))
    return MieCoefficientsBatch(x_arr, m_arr, a, b, qext, qsca,
                                n_max_arr, nmx_arr)


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


#: cached squared-Wigner weight rows for the per-shift moment reduction:
#: ``G_l(a, a+d)`` depends only on ``(a, d, l)``, so rows are built once for
#: the longest requested ``a`` range and sliced (bitwise-identical prefixes)
_GAUNT_WEIGHT_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _gaunt_weights(d: int, l: int, a_hi: int) -> np.ndarray:
    """Cached ``G_l(a, a+d)`` for ``a`` in ``[max(0, -d), a_hi]``."""
    a0 = max(0, -d)
    key = (d, l)
    arr = _GAUNT_WEIGHT_CACHE.get(key)
    need = a_hi - a0 + 1
    if arr is None or arr.size < need:
        a_arr = np.arange(a0, a_hi + 1, dtype=float)
        arr = _gaunt_sq(a_arr, a_arr + d, l)
        _GAUNT_WEIGHT_CACHE[key] = arr
    return arr[:need]


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


def mie_moment_numerators_batch(
    x: np.ndarray,
    qsca: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_max: np.ndarray,
    n_moments: int = 32,
) -> np.ndarray:
    """Batched :func:`mie_moment_numerators` over a set of size nodes.

    Parameters
    ----------
    x, qsca : ndarray (n_pair,)
    a, b : ndarray (n_pair, n_max_pad + 1) complex
        Batch coefficient rows (as produced by :func:`mie_coefficients_batch`);
        entries beyond each pair's ``n_max`` are zero.
    n_max : ndarray (n_pair,) int
    n_moments : int

    Returns
    -------
    ndarray (n_pair, n_moments)
        ``U_l = Q_sca * beta_l`` per pair.  Only the real part of the scalar
        path's band sum is computed (the scalar discards the imaginary part
        via ``(C * tot).real``), so the reduction runs in real arithmetic.

    Equivalence note (task-cold-optics): the elementwise machinery (stride-2
    tail sums, P/Q assembly) is bitwise-equal to the scalar path, but the
    per-``l`` band reduction cannot be -- the scalar path's ``np.dot`` on
    ~100-element rows dispatches to BLAS with an accumulation order no
    batched form reproduces (probe B2).  The drift is ulp-level and covered
    by the task's D2 declared tolerances downstream.  Pairs are processed in
    geometric ``n_max`` buckets so each bucket's band pass runs at its own
    truncation: the scalar loops every pair at its own ``n_max``, and a
    single pass padded to the batch maximum wastes several-fold work on
    log-uniform size distributions (measured 2.5x -> projected ~10x on the
    moment stage, see the 2026-11-10 re-adjudication entry).
    """
    x_arr = np.asarray(x, dtype=float)
    qsca_arr = np.asarray(qsca, dtype=float)
    a_mat = np.asarray(a, dtype=complex)
    b_mat = np.asarray(b, dtype=complex)
    n_max_arr = np.asarray(n_max, dtype=int)
    n_moments = int(n_moments)
    P = x_arr.size
    out = np.zeros((P, n_moments))
    if P == 0 or n_moments <= 0:
        return out
    nmax_pad = int(n_max_arr.max())
    if nmax_pad <= 0:
        return out

    bands = _gaunt_bands(n_moments - 1, max(256, nmax_pad))
    # transposed (step rows x pair columns) for contiguous per-step vectors
    aT = np.ascontiguousarray(a_mat.T)
    bT = np.ascontiguousarray(b_mat.T)
    C = 2.0 / (x_arr**2)

    # geometric n_max buckets: (lo, hi] with hi doubling from 2
    edges = [2]
    while edges[-1] < nmax_pad:
        edges.append(edges[-1] * 2)
    for lo, hi in zip([0] + edges[:-1], edges):
        hi = min(hi, nmax_pad)
        mask = (n_max_arr <= hi) & (n_max_arr > lo)
        rows = int(np.count_nonzero(mask))
        if rows == 0:
            continue
        idx = np.nonzero(mask)[0]
        aB = aT[: hi + 1, idx]
        bB = bT[: hi + 1, idx]

        n = np.arange(1, hi + 1, dtype=float)
        c = np.zeros(hi + 1)
        c[1:] = (2.0 * n + 1.0) / (n * (n + 1.0))
        # stride-2 tail sums; rows with n_max < kk hold exact zeros in aB/bB,
        # so the sums propagate zeros exactly like the scalar's untouched
        # entries -- no per-step mask needed inside the bucket
        Sa = np.zeros((hi + 5, rows), dtype=complex)
        Sb = np.zeros((hi + 5, rows), dtype=complex)
        for kk in range(hi, 0, -1):
            Sa[kk] = c[kk] * aB[kk] + Sa[kk + 2]
            Sb[kk] = c[kk] * bB[kk] + Sb[kk + 2]

        k = np.arange(hi + 2)
        ca = np.zeros(hi + 2)
        ca[1: hi + 1] = c[1:]
        aa = np.zeros((hi + 2, rows), dtype=complex)
        bb = np.zeros((hi + 2, rows), dtype=complex)
        aa[1: hi + 1] = aB[1: hi + 1]
        bb[1: hi + 1] = bB[1: hi + 1]
        kca = k * k * ca
        Pv = (2.0 * k + 1.0)[:, None] * (Sa[k + 1] - Sb[k + 2]) \
            + kca[:, None] * bb
        Qv = (2.0 * k + 1.0)[:, None] * (Sb[k + 1] - Sa[k + 2]) \
            + kca[:, None] * aa

        # realified band reduction, restructured per shift d: the product
        # P[a]*conj(P[a+d]) + Q[a]*conj(Q[a+d]) depends only on d, not on l,
        # so it is built once per d and contracted against a (a, l) matrix of
        # squared Wigner coefficients with one BLAS gemm per d -- the scalar
        # path's 528 (l, band) dot products collapse to 63 product blocks +
        # 63 gemms (measured: the elementwise band loop is numpy-throughput
        # bound; this form runs at BLAS throughput)
        Pr = np.ascontiguousarray(Pv.real)
        Pi = np.ascontiguousarray(Pv.imag)
        Qr = np.ascontiguousarray(Qv.real)
        Qi = np.ascontiguousarray(Qv.imag)
        Cb = C[idx]
        Lm1 = n_moments - 1
        acc = np.zeros((n_moments, rows))
        # preallocated product buffers: the same (n_here, rows) temporaries
        # are reused across all 63 shifts (without this, hundreds of per-shift
        # full-size allocations churn the allocator's lazy-free pool, which
        # ru_maxrss counts even though nothing is live)
        pbuf = np.empty((hi + 1, rows))
        pbuf2 = np.empty((hi + 1, rows))
        for d in range(-Lm1, Lm1 + 1):
            a0 = max(0, -d)
            a_hi = min(hi, hi - d)
            if a_hi < a0:
                continue
            s1 = slice(a0, a_hi + 1)
            s2 = slice(a0 + d, a_hi + 1 + d)
            n_here = a_hi - a0 + 1
            pb = pbuf[:n_here]
            pb2 = pbuf2[:n_here]
            np.multiply(Pr[s1], Pr[s2], out=pb)
            np.multiply(Pi[s1], Pi[s2], out=pb2)
            pb += pb2
            np.multiply(Qr[s1], Qr[s2], out=pb2)
            pb += pb2
            np.multiply(Qi[s1], Qi[s2], out=pb2)
            pb += pb2
            Gd = np.empty((n_here, n_moments))
            for l in range(n_moments):
                Gd[:, l] = _gaunt_weights(d, l, a_hi)
            acc += Gd.T @ pb
        out[idx, :] = (Cb[None, :] * acc).T
    return out


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


def mie_phase_batch(
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    qsca: np.ndarray,
    n_max: np.ndarray,
    theta_rad: np.ndarray,
) -> np.ndarray:
    """Batched :func:`mie_phase` over one wavelength's nodes.

    The angular functions ``pi_n``/``tau_n`` depend on ``mu`` only, so the
    scalar path recomputing them per size node is pure waste: this kernel
    builds the shared table once at the batch's ``n_max`` (row ``n`` is
    bitwise-equal to the per-node recurrence, which only consumes earlier
    rows) and accumulates ``s1``/``s2`` with a per-step mask.

    Parameters
    ----------
    a, b : ndarray (n_pair, n_max_pad + 1) complex
    x, qsca : ndarray (n_pair,)
    n_max : ndarray (n_pair,) int
    theta_rad : ndarray (n_angle,)

    Returns
    -------
    ndarray (n_pair, n_angle)
        Phase function per pair; rows with ``qsca <= 0`` are zero.

    Equivalence note (task-cold-optics): measured <= 1 ulp (3.1e-16 relative)
    on at most one node per wavelength against the scalar path (probe C);
    the residual is covered by the D2 declared tolerances together with the
    batched moment kernel.
    """
    a_mat = np.asarray(a, dtype=complex)
    b_mat = np.asarray(b, dtype=complex)
    x_arr = np.asarray(x, dtype=float)
    qsca_arr = np.asarray(qsca, dtype=float)
    n_max_arr = np.asarray(n_max, dtype=int)
    th = np.asarray(theta_rad, dtype=float)
    P = x_arr.size
    if P == 0 or th.size == 0:
        return np.zeros((P, th.size))
    nmax_pad = int(n_max_arr.max())
    mu_arr = np.cos(th)
    pi, _ = _angular_functions(mu_arr, nmax_pad)
    # tau rows are recomputed per step from the stored pi rows (two
    # (n_angle,) vector ops) instead of precomputing a second full table --
    # same values, half the angular-table memory
    s1 = np.zeros((P, th.size), dtype=complex)
    s2 = np.zeros((P, th.size), dtype=complex)
    # preallocated step buffers, no per-step mask: rows beyond their n_max
    # hold exact zeros in a/b, so their terms are exact zeros and the full
    # adds leave every row's accumulation bitwise identical to the masked
    # form (and hundreds of full-size temporaries per build do not churn the
    # allocator's lazy-free pool, which ru_maxrss counts)
    t1 = np.empty((P, th.size), dtype=complex)
    for n in range(1, nmax_pad + 1):
        cf = (2 * n + 1) / (n * (n + 1))
        pi_n = pi[n][None, :]
        tau_n = (n * mu_arr * pi[n] - (n + 1) * pi[n - 1])[None, :]
        a_n = a_mat[:, n][:, None]
        b_n = b_mat[:, n][:, None]
        np.multiply(a_n, pi_n, out=t1)
        t1 += b_n * tau_n
        t1 *= cf
        s1 += t1
        np.multiply(a_n, tau_n, out=t1)
        t1 += b_n * pi_n
        t1 *= cf
        s2 += t1
    # |s|^2 assembled in-place into two real scratch rows instead of a chain
    # of full-size complex temporaries
    rbuf = np.empty((P, th.size))
    rbuf2 = np.empty((P, th.size))
    np.multiply(s1.real, s1.real, out=rbuf)
    rbuf += s1.imag * s1.imag
    np.multiply(s2.real, s2.real, out=rbuf2)
    rbuf2 += s2.imag * s2.imag
    rbuf += rbuf2
    rbuf *= 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rbuf /= (x_arr[:, None] ** 2 * qsca_arr[:, None])
    rbuf[~np.isfinite(rbuf)] = 0.0
    return rbuf


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
