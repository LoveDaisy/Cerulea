"""Aerosol optical models.

Every aerosol model exposes two *intensive* properties that are independent of
the amount of aerosol in the air:

``relative_coefficients(wl)``
    extinction and scattering coefficients normalised so that the extinction
    at 550 nm equals 1.  The absolute coefficient in a layer is therefore
    ``AOD550 * profile(z) * rel_ext(wl)``.

``phase(mu, wl)``
    scattering phase function normalised so that ``int p dOmega = 4*pi``.

Single scattering albedo, asymmetry parameter and Legendre moments are derived
from those two by quadrature in the base class, so a new aerosol type only
needs to provide the two methods above.

Two families are provided:

* :class:`AngstromAerosol` -- an analytic model with a power-law spectral
  dependence and a Henyey-Greenstein phase function.  Fast and perfectly
  adequate for haze and dust.
* :class:`MieAerosol` -- a physical model: one or more log-normal size
  distributions, Mie theory (Bohren & Huffman 1983) and a refractive index.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

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
        """Asymmetry parameter ``g = <cos theta>``."""
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        return np.array([0.5 * np.sum(_GAUSS_W * _GAUSS_MU * self.phase(_GAUSS_MU, w)) for w in wl])

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
        ext = ext / np.interp(550.0, wl, ext) if wl.min() <= 550 <= wl.max() else ext
        s = np.clip(self.ssa + self.ssa_slope * (wl - 550.0), 0.0, 1.0)
        return ext, ext * s

    def phase(self, mu, wl_nm):
        mu = np.asarray(mu, dtype=float)
        if np.ndim(wl_nm) == 0:
            return henyey_greenstein(mu, self.g)
        return henyey_greenstein(mu[:, None], np.asarray(wl_nm)[None, :] * 0 + self.g)


class DustAerosol(AngstromAerosol):
    """Desert dust: weak spectral dependence, strong forward scattering."""

    def __init__(self, aod_scale: float = 1.0):
        super().__init__(name="dust", alpha=0.25, ssa=0.93, g=0.75, ssa_slope=-1.5e-4)


class SmokeAerosol(AngstromAerosol):
    """Biomass burning aerosol: strongly absorbing in the blue."""

    def __init__(self):
        super().__init__(name="smoke", alpha=1.9, ssa=0.82, g=0.55, ssa_slope=3.0e-4)


class HazeAerosol(AngstromAerosol):
    """Humid continental haze: fine, almost conservative scattering."""

    def __init__(self, rh: float = 0.6):
        alpha = 1.5 - 0.5 * rh
        g = 0.60 + 0.10 * rh
        super().__init__(name=f"haze(RH={rh:.0%})", alpha=alpha, ssa=0.98, g=g)


class HinterlandAerosol(AngstromAerosol):
    """Clean continental background."""

    def __init__(self):
        super().__init__(name="hinterland", alpha=1.3, ssa=0.96, g=0.65)


class UrbanAerosol(AngstromAerosol):
    def __init__(self):
        super().__init__(name="urban", alpha=1.5, ssa=0.90, g=0.62, ssa_slope=1.0e-4)


class MaritimeAerosol(AngstromAerosol):
    def __init__(self):
        super().__init__(name="maritime", alpha=0.8, ssa=0.98, g=0.75)


# ===========================================================================
# Mie theory
# ===========================================================================
def _log_normal_weights(r, r_g, sigma_g):
    """Number density weights of a log-normal distribution (unnormalised)."""
    return np.exp(-0.5 * (np.log(r / r_g) / np.log(sigma_g)) ** 2) / (r * np.log(sigma_g))


def mie_single(x: float, m: complex, n_angles: int = 0):
    """Mie coefficients for a single sphere.

    Parameters
    ----------
    x : size parameter ``2 pi r / lambda``
    m : complex refractive index
    n_angles : if > 0, also return the phase function on that many equally
        spaced cos(theta) points.

    Returns
    -------
    qext, qsca, g, (mu, phase)
    """
    if x <= 0:
        return 0.0, 0.0, 0.0, (None if n_angles == 0 else (np.array([]), np.array([])))
    nmax = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    nmx = max(nmax, int(abs(m * x))) + 15
    rho = m * x

    # logarithmic derivative D_n(rho) by downward recurrence
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / rho - 1.0 / (D[n] + n / rho)

    # Riccati-Bessel functions
    psi = np.zeros(nmax + 2)
    psi[0] = np.sin(x)
    psi[1] = np.sin(x) / x - np.cos(x)
    for n in range(1, nmax + 1):
        psi[n + 1] = (2 * n + 1) / x * psi[n] - psi[n - 1]
    chi = np.zeros(nmax + 2)
    chi[0] = np.cos(x)
    chi[1] = np.cos(x) / x + np.sin(x)
    for n in range(1, nmax + 1):
        chi[n + 1] = (2 * n + 1) / x * chi[n] - chi[n - 1]
    xi = psi - 1j * chi

    a = np.zeros(nmax + 1, dtype=complex)
    b = np.zeros(nmax + 1, dtype=complex)
    for n in range(1, nmax + 1):
        dn = D[n]
        pa = dn / m + n / x
        pb = m * dn + n / x
        a[n] = (pa * psi[n] - psi[n - 1]) / (pa * xi[n] - xi[n - 1])
        b[n] = (pb * psi[n] - psi[n - 1]) / (pb * xi[n] - xi[n - 1])

    n = np.arange(1, nmax + 1)
    qext = 2.0 / x**2 * np.sum((2 * n + 1) * np.real(a[1:] + b[1:]))
    qsca = 2.0 / x**2 * np.sum((2 * n + 1) * (np.abs(a[1:]) ** 2 + np.abs(b[1:]) ** 2))
    an = a[1:]
    bn = b[1:]
    an1 = np.append(a[2:], 0.0)
    bn1 = np.append(b[2:], 0.0)
    g_num = np.sum(n * (n + 2) / (n + 1) * np.real(an * np.conj(an1) + bn * np.conj(bn1)))
    g_num += np.sum((2 * n + 1) / (n * (n + 1)) * np.real(an * np.conj(bn)))
    g = 4.0 / (x**2 * max(qsca, 1e-300)) * g_num

    if n_angles == 0:
        return qext, qsca, float(g), (None, None)

    mu = np.linspace(-1.0, 1.0, n_angles)
    theta = np.arccos(mu)
    s1, s2 = _mie_amplitude(mu, a, b)
    phase = 2.0 * (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (x**2 * max(qsca, 1e-300))
    return qext, qsca, float(g), (mu, phase)


def _mie_amplitude(mu, a, b):
    """Mie scattering amplitudes S1, S2 at the given cosines."""
    mu = np.asarray(mu, dtype=float)
    smu = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    nmax = len(a) - 1
    # pi_n = P_n^1(mu)/smu via recurrence, tau_n = mu*pi_n - (n+1)*pi_{n-1} ... use
    # standard recurrences (Bohren & Huffman eq. 4.47)
    pi = np.zeros((nmax + 1, len(mu)))
    tau = np.zeros((nmax + 1, len(mu)))
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) * mu * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    s1 = np.zeros(len(mu), dtype=complex)
    s2 = np.zeros(len(mu), dtype=complex)
    for n in range(1, nmax + 1):
        c = (2 * n + 1) / (n * (n + 1))
        s1 += c * (a[n] * pi[n] + b[n] * tau[n])
        s2 += c * (a[n] * tau[n] + b[n] * pi[n])
    return s1, s2


@dataclass
class LogNormalMode:
    """A log-normal size distribution."""

    r_g_um: float  # geometric mean radius [um]
    sigma_g: float  # geometric standard deviation
    weight: float = 1.0  # relative number weight


@dataclass
class MieAerosol(Aerosol):
    """Aerosol described by log-normal size modes and a refractive index.

    Parameters
    ----------
    modes : sequence of LogNormalMode
    refractive_index : complex or callable
        ``complex`` for a non-dispersive index, or a callable
        ``m(wl_nm) -> complex``.
    """

    name: str = "mie"
    modes: Sequence[LogNormalMode] = field(default_factory=lambda: [LogNormalMode(0.05, 2.0)])
    refractive_index: Callable | complex = 1.53 + 0.006j
    r_min_um: float = 0.005
    r_max_um: float = 20.0
    n_radii: int = 48
    n_angles: int = 181

    def __post_init__(self):
        self._cache_wl = None
        self._cache = None

    def _m(self, wl):
        if callable(self.refractive_index):
            return complex(self.refractive_index(wl))
        return complex(self.refractive_index)

    def _compute(self, wl_nm):
        key = tuple(np.round(np.atleast_1d(wl_nm), 6))
        if self._cache_wl is not None and self._cache_wl == key:
            return self._cache
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        r = np.geomspace(self.r_min_um, self.r_max_um, self.n_radii)
        w = np.zeros_like(r)
        for mode in self.modes:
            w = w + mode.weight * _log_normal_weights(r, mode.r_g_um, mode.sigma_g)
        w = w / w.sum()
        Cext = np.zeros(len(wl))
        Csca = np.zeros(len(wl))
        g = np.zeros(len(wl))
        phase = np.zeros((self.n_angles, len(wl)))
        for i, lam in enumerate(wl):
            m = self._m(lam)
            ce = cs = 0.0
            gsum = 0.0
            psum = np.zeros(self.n_angles)
            for j, rj in enumerate(r):
                x = 2 * np.pi * rj / (lam * 1e-3)  # lam in um -> x
                if x < 1e-4:
                    continue
                qe, qs, gj, (mu, ph) = mie_single(x, m, self.n_angles)
                area = np.pi * (rj * 1e-6) ** 2
                ce += w[j] * qe * area
                cs += w[j] * qs * area
                gsum += w[j] * qs * area * gj
                psum += w[j] * qs * area * ph
            Cext[i], Csca[i] = ce, cs
            g[i] = gsum / max(cs, 1e-300)
            phase[:, i] = psum / max(cs, 1e-300)
        self._cache_wl = key
        self._cache = (Cext, Csca, g, phase)
        return self._cache

    def relative_coefficients(self, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        ce, cs, _, _ = self._compute(wl)
        scale = np.interp(550.0, wl, ce) if wl.min() <= 550 <= wl.max() else ce.mean()
        return ce / scale, cs / scale

    def phase(self, mu, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        _, _, _, ph = self._compute(wl)
        grid = np.linspace(-1.0, 1.0, ph.shape[0])
        if np.ndim(mu) == 0 and np.ndim(wl_nm) == 0:
            return np.interp(float(mu), grid, ph[:, 0])
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        out = np.empty((len(mu_arr), len(wl)))
        for i in range(len(wl)):
            out[:, i] = np.interp(mu_arr, grid, ph[:, i])
        return out

    @classmethod
    def water_soluble(cls, rh: float = 0.5):
        """Shettle & Fenn water-soluble aerosol with humidity growth."""
        r_g = 0.05 * (1.0 - rh) ** (-0.35)
        return cls(name=f"water-soluble(RH={rh:.0%})",
                   modes=[LogNormalMode(r_g, 2.0)], refractive_index=1.53 + 0.006j)

    @classmethod
    def dust(cls):
        return cls(name="dust", modes=[LogNormalMode(0.5, 2.2, 0.6), LogNormalMode(4.0, 2.0, 0.4)],
                   refractive_index=1.55 + 0.005j)

    @classmethod
    def soot(cls):
        return cls(name="soot", modes=[LogNormalMode(0.012, 2.0)], refractive_index=1.75 + 0.44j)

    @classmethod
    def cloud_droplets(cls, r_eff_um: float = 8.0):
        return cls(name="cloud", modes=[LogNormalMode(r_eff_um, 1.35)],
                   refractive_index=1.333, r_max_um=40.0)
