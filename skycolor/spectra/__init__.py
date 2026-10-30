"""Spectral data: solar irradiance, colour matching functions, absorption
cross sections and Rayleigh scattering.

Data files live in ``skycolor/data`` and are built by ``tools/build_data.py``
from the raw sources documented in ``skycolor/data/SOURCES.md``.
"""
from __future__ import annotations

import functools
import pathlib

import numpy as np

from ..constants import DEPOLARIZATION, N_STD_AIR

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, delimiter=",", skiprows=1)


@functools.lru_cache(maxsize=None)
def _cie_raw() -> np.ndarray:
    return _load("cie1931_2deg.csv")


@functools.lru_cache(maxsize=None)
def _solar_raw() -> np.ndarray:
    return _load("solar_am0.csv")


@functools.lru_cache(maxsize=None)
def _o3_raw() -> np.ndarray:
    return _load("o3_cross_section.csv")


@functools.lru_cache(maxsize=None)
def _no2_raw() -> np.ndarray:
    return _load("no2_cross_section.csv")


# ---------------------------------------------------------------------------
# resampling helper
# ---------------------------------------------------------------------------
def resample(table: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    """Linearly interpolate columns 1.. of ``table`` onto ``wl_nm``.

    Values outside the tabulated range are set to zero (all our tables vanish
    outside the ultraviolet/visible).
    """
    src = table[:, 0]
    out = np.empty((len(wl_nm), table.shape[1] - 1))
    for j in range(1, table.shape[1]):
        v = np.interp(wl_nm, src, table[:, j], left=0.0, right=0.0)
        out[:, j - 1] = v
    return out


# ---------------------------------------------------------------------------
# colour matching functions
# ---------------------------------------------------------------------------
def cie_xyz(wl_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2-degree colour matching functions, shape ``(n_wl, 3)``."""
    return resample(_cie_raw(), np.asarray(wl_nm, dtype=float))


# ---------------------------------------------------------------------------
# solar spectrum
# ---------------------------------------------------------------------------
def solar_irradiance(wl_nm: np.ndarray) -> np.ndarray:
    """Extraterrestrial solar spectral irradiance [W m^-2 nm^-1]."""
    return resample(_solar_raw(), np.asarray(wl_nm, dtype=float))[:, 0]


# ---------------------------------------------------------------------------
# absorption cross sections
# ---------------------------------------------------------------------------
def ozone_cross_section(wl_nm: np.ndarray) -> np.ndarray:
    """O3 absorption cross section [m^2], JPL-2006 evaluation."""
    return np.maximum(resample(_o3_raw(), np.asarray(wl_nm, dtype=float))[:, 0], 0.0) * 1e-4


def no2_cross_section(wl_nm: np.ndarray) -> np.ndarray:
    """NO2 absorption cross section [m^2] at 220 K (stratospheric)."""
    return np.maximum(resample(_no2_raw(), np.asarray(wl_nm, dtype=float))[:, 0], 0.0) * 1e-4


# ---------------------------------------------------------------------------
# Rayleigh scattering
# ---------------------------------------------------------------------------
def refractive_index_air(wl_nm: np.ndarray) -> np.ndarray:
    """Refractive index of standard dry air (Edlen, as used by Bodhaine 1999).

    Valid from the near ultraviolet to the near infrared; ``wl`` in nm.
    """
    nu2 = (1e3 / np.asarray(wl_nm, dtype=float)) ** 2  # (1/um)^2
    return 1.0 + 1e-8 * (8342.13 + 2406030.0 / (130.0 - nu2) + 15997.0 / (38.9 - nu2))


def rayleigh_cross_section(wl_nm: np.ndarray, depolarization: float = DEPOLARIZATION) -> np.ndarray:
    """Rayleigh scattering cross section per molecule [m^2].

    Bodhaine, Wood, Dutton & Slusser (1999), *On Rayleigh optical depth
    calculations*, J. Atmos. Ocean. Tech. 16, 1854.  At 550 nm this returns
    4.507e-31 m^2, giving a sea-level vertical optical depth of 0.097.
    """
    wl = np.asarray(wl_nm, dtype=float)
    lam = wl * 1e-9
    n = refractive_index_air(wl)
    n2 = n * n
    f = (6.0 + 3.0 * depolarization) / (6.0 - 7.0 * depolarization)
    sigma = (24.0 * np.pi**3 / (N_STD_AIR**2 * lam**4)) * ((n2 - 1.0) / (n2 + 2.0)) ** 2 * f
    return sigma


def rayleigh_phase(cos_theta, depolarization: float = DEPOLARIZATION):
    """Rayleigh phase function normalised so that ``int p dOmega = 4 pi``."""
    gamma = depolarization / (2.0 - depolarization)
    ct2 = np.asarray(cos_theta) ** 2
    return 3.0 / (4.0 * (1.0 + 2.0 * gamma)) * ((1.0 + 3.0 * gamma) + (1.0 - gamma) * ct2)


def rayleigh_phase_moments(depolarization: float = DEPOLARIZATION) -> np.ndarray:
    """Legendre coefficients beta_l of the Rayleigh phase function (l = 0..2).

    ``p(cos t) = sum_l (2l+1) beta_l P_l(cos t)``.
    """
    gamma = depolarization / (2.0 - depolarization)
    b0 = 1.0
    b2 = (1.0 - gamma) / (10.0 * (1.0 + 2.0 * gamma))
    return np.array([b0, 0.0, b2])
