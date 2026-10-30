"""Build the packaged spectral data tables from the raw source files in ``data_raw/``.

Run with::

    python tools/build_data.py

The outputs are small CSV files in ``skycolor/data/`` that are committed to the
repository, so that the simulation itself has no network dependency.  The raw
inputs in ``data_raw/`` are also committed (all are small); see
``skycolor/data/SOURCES.md`` for provenance.
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "skycolor" / "data"

#: Common output grid, 1 nm from 300 to 830 nm.
WL = np.arange(300.0, 831.0, 1.0)


def _write(name: str, header: str, cols: list[np.ndarray]) -> None:
    path = OUT / name
    with path.open("w", newline="") as fh:
        fh.write(header + "\n")
        for row in zip(*cols):
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(cols[0])} rows)")


def build_cie() -> None:
    """CIE 1931 2-degree colour matching functions (CVRL, 1 nm)."""
    data = np.loadtxt(RAW / "ciexyz31_1.csv", delimiter=",")
    wl, xb, yb, zb = data.T
    m = (wl >= 300) & (wl <= 830)
    wl, xb, yb, zb = wl[m], xb[m], yb[m], zb[m]
    # interpolate onto the common grid (the source starts at 360 nm; outside the
    # tabulated range the CMFs are exactly zero).
    def _interp(v):
        out = np.zeros_like(WL)
        ok = (WL >= wl[0]) & (WL <= wl[-1])
        out[ok] = np.interp(WL[ok], wl, v)
        return out

    _write("cie1931_2deg.csv", "wavelength_nm,xbar,ybar,zbar",
           [WL, _interp(xb), _interp(yb), _interp(zb)])


def build_solar() -> None:
    """Extraterrestrial solar spectral irradiance, ASTM G173-03 (AM0 column).

    Units: W m^-2 nm^-1.
    """
    data = np.loadtxt(RAW / "ASTMG173.csv", delimiter=",", skiprows=2)
    wl = data[:, 0]
    e0 = data[:, 1]
    m = (wl >= 300) & (wl <= 830)
    wl, e0 = wl[m], e0[m]

    # The table is tabulated at 0.5 nm below 400 nm and 1 nm above, so linear
    # interpolation onto the common 1 nm grid is exact in the visible.
    out = np.interp(WL, wl, e0)
    _write("solar_am0.csv", "wavelength_nm,irradiance_W_m2_nm", [WL, out])
    print(f"  integrated AM0 irradiance (300-830 nm): {np.trapezoid(out, WL):.1f} W/m2")
    print(f"  integrated AM0 irradiance (400-700 nm): "
          f"{np.trapezoid(out[(WL >= 400) & (WL <= 700)], WL[(WL >= 400) & (WL <= 700)]):.1f} W/m2")


def _read_nc(name: str):
    import netCDF4

    ds = netCDF4.Dataset(RAW / name)
    wl = np.array(ds.variables["wavelength"][:], dtype=float)
    temp = np.array(ds.variables["temperature"][:], dtype=float)
    xs = np.array(ds.variables["cross_section_parameters"][:], dtype=float)
    ds.close()
    return wl, temp, xs


def build_ozone() -> None:
    """O3 absorption cross section, Chappuis/Huggins bands (cm^2 molecule^-1).

    Source: JPL 2006 evaluation as distributed with NCAR TUV-x
    (``data/cross_sections/O3_4.nc``; 295 K and 218 K columns are identical in
    the visible, i.e. the table carries no Chappuis temperature dependence).
    """
    wl, temp, xs = _read_nc("O3_4.nc")
    m = (wl >= 300) & (wl <= 830)
    wl, xs = wl[m], xs[0][m]
    out = np.interp(WL, wl, xs)
    out[WL < wl.min()] = 0.0
    _write("o3_cross_section.csv", "wavelength_nm,sigma_cm2", [WL, out])


def build_no2() -> None:
    """NO2 absorption cross section (cm^2 molecule^-1), 220 K / 294 K columns."""
    wl, temp, xs = _read_nc("NO2_1.nc")
    m = (wl >= 300) & (wl <= 830)
    wl = wl[m]
    a = np.interp(WL, wl, xs[0][m])
    b = np.interp(WL, wl, xs[1][m])
    a[WL < wl.min()] = 0.0
    b[WL < wl.min()] = 0.0
    _write("no2_cross_section.csv", "wavelength_nm,sigma_220K_cm2,sigma_294K_cm2", [WL, a, b])




# ---------------------------------------------------------------------------
# refraction table
# ---------------------------------------------------------------------------
#: altitudes [km] and apparent elevations [deg] of the refraction table
_REFRACT_ALT = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0,
                         20.0, 25.0, 30.0, 38.0, 47.0, 56.0, 70.0, 90.0, 120.0])
_REFRACT_ELEV = np.array([-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])


def build_refraction() -> None:
    """Tabulate the astronomical refraction vs altitude and apparent elevation.

    Columns: ``altitude_km, apparent_elevation_deg, true_elevation_deg``.
    ``true = apparent + refraction``, so the table is monotonic in the
    apparent elevation and can be inverted to go from the geometric solar
    elevation (what an ephemeris gives) to the apparent one (what the light
    actually does).  Entries below the dip of the horizon are omitted: there
    the ray hits the ground and the point is in shadow.
    """
    from skycolor.atmosphere import Atmosphere, us76_pressure, us76_temperature
    from skycolor.constants import N_STD_AIR, R_EARTH
    from skycolor.refraction import REFRACTIVITY_STD, _Profile, _trace

    atmo = Atmosphere(np.array([550.0]), aerosols=[])
    prof = _Profile(atmo)
    rows = []
    for h in _REFRACT_ALT:
        r0 = R_EARTH + h * 1e3
        dip = np.degrees(np.arccos(R_EARTH / r0))
        for e in _REFRACT_ELEV:
            if e < -dip - 1e-9:
                continue          # the ray hits the ground
            hit, _, _, bend = _trace(prof, r0, np.pi / 2 - np.radians(e),
                                     R_EARTH, R_EARTH + 1.2e6, 2000.0, 400000)
            if hit:
                continue
            rows.append((h, e, e - np.degrees(bend)))
    path = OUT / "refraction.csv"
    with path.open("w", newline="") as fh:
        fh.write("altitude_km,apparent_elevation_deg,true_elevation_deg\n")
        for h, e, t in rows:
            fh.write(f"{h:.6g},{e:.6g},{t:.9g}\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_cie()
    build_solar()
    build_ozone()
    build_no2()
    build_refraction()
