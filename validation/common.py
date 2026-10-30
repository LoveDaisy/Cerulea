"""Shared utilities for the sky-colour validation harness.

Every check script is self-contained and can be run either directly::

    /path/to/.venv/bin/python validation/check1_rayleigh.py

or through :mod:`validation.run_all`.  Both paths use the same helpers here.

Conventions (see ``docs/ARCHITECTURE.md``):
* wavelengths in nm,
* spectral radiance in W m^-2 sr^-1 nm^-1,
* illuminance/luminance computed with the CIE 1931 2-degree photopic
  function (``ybar``) and the maximum luminous efficacy ``K_m = 683 lm/W``.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

import os

import numpy as np

# Make the worktree importable no matter which directory the script is run
# from.  ``validation/common.py`` -> parents[1] is the worktree root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
DATA = VAL / "data"
FIG = VAL / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skycolor import presets, spectra  # noqa: E402
from skycolor.constants import DU, P0, R_EARTH  # noqa: E402
from skycolor.rt.single import direct_sun_transmittance  # noqa: E402

#: ``SKYCOLOR_MULTIPLE=1`` switches every check to the full solver
#: (single + multiple scattering) instead of single scattering alone.
if os.environ.get("SKYCOLOR_MULTIPLE", "1") == "1":
    from skycolor.rt.solver import sky_radiance_total as sky_radiance  # noqa: E402
    MULTIPLE = True
else:
    from skycolor.rt.single import sky_radiance  # noqa: E402
    MULTIPLE = False

# ---------------------------------------------------------------------------
# spectral grids
# ---------------------------------------------------------------------------
WL = np.arange(380.0, 781.0, 1.0)          # 1 nm, visible only
WL_COARSE = np.arange(380.0, 781.0, 5.0)   # 5 nm, for large sky sweeps
WL_UV = np.arange(300.0, 701.0, 1.0)       # Rayleigh check range

K_M = 683.0  # lm/W


def photopic(wl=WL):
    """CIE 1931 2-degree photopic luminous efficiency V(lambda)."""
    return spectra.cie_xyz(np.asarray(wl, dtype=float))[:, 1]


def luminance(radiance, wl=WL):
    """Photopic luminance [cd m^-2] of spectral radiance ``(..., n_wl)``."""
    radiance = np.asarray(radiance, dtype=float)
    return K_M * np.trapezoid(radiance * photopic(wl), np.asarray(wl, dtype=float), axis=-1)


def illuminance(irradiance, wl=WL):
    """Photopic illuminance [lx] of spectral irradiance ``(..., n_wl)``."""
    irradiance = np.asarray(irradiance, dtype=float)
    return K_M * np.trapezoid(irradiance * photopic(wl), np.asarray(wl, dtype=float), axis=-1)


# ---------------------------------------------------------------------------
# sky geometry
# ---------------------------------------------------------------------------
def sky_grid(n_zenith=37, n_azimuth=72, zmax=90.0):
    """Regular grid in viewing zenith/azimuth, including the zenith and horizon."""
    z = np.linspace(0.0, zmax, n_zenith)
    a = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)
    return np.meshgrid(z, a, indexing="ij")


def angular_distance(view_zen_deg, view_az_deg, sun_zen_deg, sun_az_deg=0.0):
    """Great-circle angle between view direction(s) and the sun [degrees]."""
    z = np.radians(np.asarray(view_zen_deg, dtype=float))
    a = np.radians(np.asarray(view_az_deg, dtype=float) - sun_az_deg)
    zs = np.radians(float(sun_zen_deg))
    cosg = np.cos(z) * np.cos(zs) + np.sin(z) * np.sin(zs) * np.cos(a)
    return np.degrees(np.arccos(np.clip(cosg, -1.0, 1.0)))


def horizontal_illuminance(radiance, view_zen_deg, view_az_deg, wl=WL):
    """Diffuse horizontal illuminance from a regular sky grid [lx].

    ``radiance`` has shape ``(n_zen, n_az, n_wl)`` (or ``(n_zen, n_az)`` for
    already-integrated luminance).  The zenith angle grid must include 0 and
    90 degrees.
    """
    lum = radiance if np.ndim(radiance) == 2 else luminance(radiance, wl)
    z = np.radians(np.asarray(view_zen_deg, dtype=float)[:, 0])
    # integral over theta of L cos(theta) sin(theta) dtheta, for each azimuth
    integrand = lum * (np.cos(z) * np.sin(z))[:, None]
    per_az = np.trapezoid(integrand, z, axis=0)
    return float(np.mean(per_az) * 2.0 * np.pi)


# ---------------------------------------------------------------------------
# CIE 110-1994 clear-sky luminance distribution
# ---------------------------------------------------------------------------
#: CIE 110-1994 / CIE 22-1973 standard clear sky.  The relative luminance is
#:
#:   L(theta, gamma) / Lz =
#:       (0.91 + 10 exp(-3 gamma) + 0.45 cos^2 gamma)
#:       (1 - exp(-0.32 / cos theta))
#:     / [(0.91 + 10 exp(-3 theta_s) + 0.45 cos^2 theta_s)
#:        (1 - exp(-0.32))]
#:
#: where theta is the viewing zenith angle, gamma the angular distance from
#: the sun and theta_s the solar zenith angle.  This is exactly the
#: "standard clear sky" of CIE 110-1994, also reproduced as eq. (4.31) of
#: Zotti (2007), "Computer graphics in historical and modern sky
#: observations" (TU Wien), which cites CIE 110-1994.


def cie110_indicatrix(gamma_deg):
    g = np.radians(np.asarray(gamma_deg, dtype=float))
    return 0.91 + 10.0 * np.exp(-3.0 * g) + 0.45 * np.cos(g) ** 2


def cie110_gradation(zenith_deg):
    z = np.radians(np.asarray(zenith_deg, dtype=float))
    cosz = np.cos(z)
    # At the horizon cos(theta)=0 and the exponential vanishes; guard the
    # division so that 90 degrees is handled exactly.
    safe = np.where(cosz > 1e-9, cosz, 1e-9)
    return np.where(cosz > 1e-9, 1.0 - np.exp(-0.32 / safe), 1.0)


def cie110_relative_luminance(view_zen_deg, view_az_deg, sun_zen_deg):
    """Relative luminance L(theta,gamma)/Lz for the CIE 110-1994 clear sky."""
    gamma = angular_distance(view_zen_deg, view_az_deg, sun_zen_deg, 0.0)
    zs = float(sun_zen_deg)
    num = cie110_indicatrix(gamma) * cie110_gradation(view_zen_deg)
    den = cie110_indicatrix(zs) * cie110_gradation(0.0)
    return num / den


def cie110_zenith_luminance(sun_elevation_deg, turbidity=3.0):
    """CIE 110-1994 clear-sky zenith luminance [cd m^-2].

    ``Yz = (1.376 T - 1.81) tan(h) + 0.38  [kcd m^-2]``

    with ``T`` the atmospheric turbidity and ``h`` the solar elevation.  This
    empirical relation is quoted in CIE 110-1994 and reproduced by Zotti
    (2007), section 4.5.  The model atmosphere does not have a Linke
    turbidity, so the comparison is made at a stated ``T`` and the
    corresponding model case is chosen to have a similar aerosol load.
    """
    h = math.radians(float(sun_elevation_deg))
    return 1000.0 * ((1.376 * float(turbidity) - 1.81) * math.tan(h) + 0.38)


def cie110_absolute_luminance(view_zen_deg, view_az_deg, sun_elevation_deg, turbidity=3.0):
    """Absolute CIE 110 clear-sky luminance [cd m^-2] on the given grid."""
    sun_zen = 90.0 - float(sun_elevation_deg)
    return cie110_zenith_luminance(sun_elevation_deg, turbidity) * cie110_relative_luminance(
        view_zen_deg, view_az_deg, sun_zen
    )


# ---------------------------------------------------------------------------
# model atmospheres
# ---------------------------------------------------------------------------
def atmosphere(kind="clear", wl=WL_COARSE, aod=0.05, ozone_du=300.0):
    """Named model atmosphere used by the checks.

    ``kind`` is one of the :data:`skycolor.presets.PRESETS` keys,
    ``"pristine"`` (Rayleigh + ozone only) or ``"clear_uniform"`` (the
    ``clear`` aerosol with a height-independent profile, used to separate the
    effect of the vertical aerosol profile).
    """
    wl = np.asarray(wl, dtype=float)
    if kind == "pristine":
        return presets.pristine(wl, ozone_du=ozone_du)
    if kind == "clear":
        return presets.clear(wl, aod=aod, ozone_du=ozone_du)
    if kind == "clear_uniform":
        from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile
        from skycolor.aerosol import HinterlandAerosol

        # A 20 km "well mixed" layer is effectively uniform over the
        # optically relevant atmosphere; the AOD is unchanged.
        return Atmosphere(
            wl, ozone_du=ozone_du,
            aerosols=[AerosolLayer(HinterlandAerosol(), aod, ExponentialProfile(20.0))],
        )
    return presets.PRESETS[kind](wl)


# ---------------------------------------------------------------------------
# result bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class Metric:
    name: str
    model: float
    reference: float
    unit: str = ""
    tolerance: float | None = None      # allowed |deviation|, in ``dev_unit``
    dev: float | None = None            # signed deviation
    dev_kind: str = "rel_pct"           # rel_pct | abs | ratio
    note: str = ""

    @property
    def passed(self) -> bool:
        if self.tolerance is None or self.dev is None:
            return True
        return abs(self.dev) <= self.tolerance

    def dev_text(self) -> str:
        if self.dev is None:
            return "-"
        if self.dev_kind == "rel_pct":
            return f"{self.dev:+.1f}%"
        if self.dev_kind == "ratio":
            return f"{self.dev:.2f}x"
        return f"{self.dev:+.3g}"


@dataclass
class CheckResult:
    key: str
    title: str
    metrics: list[Metric] = field(default_factory=list)
    verdict: str = "SKIP"               # PASS | FAIL | SKIP
    summary: str = ""
    details: dict = field(default_factory=dict)
    error: str | None = None

    def finalise(self):
        if self.error is not None:
            self.verdict = "FAIL"
        elif not self.metrics:
            self.verdict = "SKIP"
        elif all(m.passed for m in self.metrics):
            self.verdict = "PASS"
        else:
            self.verdict = "FAIL"
        return self


def rel_dev(model, reference):
    reference = np.asarray(reference, dtype=float)
    model = np.asarray(model, dtype=float)
    return 100.0 * (model - reference) / np.where(np.abs(reference) > 0, reference, np.nan)


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{float(x):.{nd-1}e}"
    return f"{float(x):.{nd}g}"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_csv(path, header, columns):
    path = pathlib.Path(path)
    arr = np.column_stack([np.asarray(c) for c in columns])
    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for row in arr:
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")
    return path


def save_json(path, obj):
    path = pathlib.Path(path)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    return path


def plt():
    """Matplotlib with the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_standalone(result: CheckResult, detail_lines=None):
    """Print a check result as a table and return the process exit code."""
    print(f"\n{result.key}: {result.title}")
    print("-" * 78)
    if result.error:
        print(result.error)
    for m in result.metrics:
        tol = "-" if m.tolerance is None else f"{m.tolerance:g}"
        print(
            f"  {m.name:<38s} model={fmt(m.model):>10s} ref={fmt(m.reference):>10s} "
            f"{m.unit:<8s} dev={m.dev_text():>8s} tol={tol:>6s} "
            f"{'PASS' if m.passed else 'FAIL'}"
        )
    if detail_lines:
        print()
        for line in detail_lines:
            print(line)
    print(f"  verdict: {result.verdict}")
    return 0 if result.verdict in ("PASS", "SKIP") else 1


def guarded(key, title, fn):
    """Run ``fn`` and turn any exception into a FAIL result (never crash run_all)."""
    try:
        return fn()
    except Exception:  # pragma: no cover - this is the safety net itself
        return CheckResult(key=key, title=title, error=traceback.format_exc()).finalise()
