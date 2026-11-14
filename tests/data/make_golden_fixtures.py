"""Generate the golden bit-exact regression fixtures.

Run from the repository root:

    uv run python tests/data/make_golden_fixtures.py

The fixtures pin the pre-refactor baseline of task-architecture-refactor:
every structural migration step must reproduce them with ``np.array_equal``
(equivalence criterion A of the task plan).  Regenerate only with an
explicit, recorded reason in the task progress log -- never to make a
failing comparison pass.

All scenarios are pinned at ``elevation_km=0`` so the structural criterion
stays decoupled from the altitude-truncation semantics change, which is
verified separately (see ``docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md``).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from skycolor import cloud, presets, render  # noqa: E402
from skycolor.atmosphere import Atmosphere  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402
from skycolor.rt import multiple  # noqa: E402
from skycolor.rt.cloud import sky_radiance_with_cloud  # noqa: E402
from skycolor.rt.mc import mc_radiance_split  # noqa: E402
from skycolor.rt.solver import sky_radiance_total  # noqa: E402
from skycolor.rt.twilight import twilight_radiance  # noqa: E402

WL7 = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
TINY = dict(n_zen=6, n_az=8, n_anchor=3, n_photons=6.0e5)
OUT = ROOT / "tests" / "data"


def _save(name: str, scenario: dict, **arrays) -> None:
    path = OUT / f"golden_{name}.npz"
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["scenario"] = np.array(json.dumps(scenario))
    np.savez_compressed(path, **payload)
    print(f"wrote {path.name} ({path.stat().st_size} bytes)")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    atmo = presets.clear(WL_COARSE)
    rad = sky_radiance_total(atmo, 80.0, 60.0, 120.0)
    _save("clear_direction",
          dict(atmo="presets.clear(WL_COARSE)", entry="rt.solver.sky_radiance_total",
               sun_zenith_deg=80.0, view_zenith_deg=60.0, view_azimuth_deg=120.0,
               elevation_km=0.0),
          radiance=rad, wl=atmo.wl)
    print(f"  clear_direction: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    atmo = presets.clear(WL_COARSE)
    res = multiple.solve(atmo, 65.0, np.array([0.0, 45.0, 85.0]),
                         np.array([0.0, 90.0, 180.0]), n_mu=6,
                         single_view=True)
    _save("sos",
          dict(atmo="presets.clear(WL_COARSE)", entry="rt.multiple.solve",
               sun_zenith_deg=65.0, view_zenith_deg=[0.0, 45.0, 85.0],
               view_azimuth_deg=[0.0, 90.0, 180.0], n_mu=6,
               single_view=True, elevation_km=0.0),
          radiance=res.radiance, radiance_single=res.radiance_single,
          flux_up_toa=res.flux_up_toa, flux_down_boa=res.flux_down_boa,
          flux_up_boa=res.flux_up_boa, direct_down_boa=res.direct_down_boa,
          incident_flux=res.incident_flux,
          orders=np.array(res.orders), converged=np.array(res.converged),
          wl=atmo.wl)
    print(f"  sos: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    atmo = presets.clear(WL7)
    dome = render.sky_dome(atmo, 78.0, n_zenith=4, n_azimuth=5)
    _save("dome",
          dict(atmo="presets.clear(WL7)", entry="render.sky_dome",
               sun_zenith_deg=78.0, n_zenith=4, n_azimuth=5, elevation_km=0.0),
          zenith=dome["zenith"], azimuth_rel=dome["azimuth_rel"],
          radiance=dome["radiance"], wl=dome["wl"])
    print(f"  dome: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    atmo = Atmosphere(WL7, ozone_du=300.0, aerosols=[])
    total, comps = sky_radiance_with_cloud(atmo, cloud.get_preset("stratus"),
                                           75.0, 45.0, 135.0,
                                           return_components=True)
    _save("cloud",
          dict(atmo="Atmosphere(WL7, ozone_du=300, aerosols=[])",
               entry="rt.cloud.sky_radiance_with_cloud",
               cloud="cloud.get_preset('stratus')", sun_zenith_deg=75.0,
               view_zenith_deg=45.0, view_azimuth_deg=135.0, elevation_km=0.0),
          radiance=total, clear=comps["clear"], cloud=comps["cloud"],
          transmittance=comps["transmittance"], wl=atmo.wl)
    print(f"  cloud: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    atmo = presets.clear(WL7)
    rad, err = twilight_radiance(atmo, 96.0, np.array([0.0, 30.0, 70.0]),
                                 np.array([0.0, 90.0, 180.0]),
                                 quality="low", grid=TINY, seed=0)
    _save("twilight",
          dict(atmo="presets.clear(WL7)", entry="rt.twilight.twilight_radiance",
               sun_zenith_deg=96.0, view_zenith_deg=[0.0, 30.0, 70.0],
               view_azimuth_deg=[0.0, 90.0, 180.0], quality="low",
               grid=TINY, seed=0, elevation_km=0.0),
          radiance=rad, stderr=err, wl=atmo.wl)
    print(f"  twilight: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    atmo = presets.clear(WL7)
    res = mc_radiance_split(atmo, 85.0, np.array([30.0, 70.0]),
                            np.array([90.0, 180.0]), n_photons=600_000, seed=0)
    _save("mc",
          dict(atmo="presets.clear(WL7)", entry="rt.mc.mc_radiance_split",
               sun_zenith_deg=85.0, view_zenith_deg=[30.0, 70.0],
               view_azimuth_deg=[90.0, 180.0], n_photons=600000, seed=0,
               elevation_km=0.0),
          total=res["total"], single=res["single"], multiple=res["multiple"],
          total_err=res["total_err"], wl=atmo.wl)
    print(f"  mc: {time.perf_counter() - t0:.1f} s")

    # Truncation semantics (post-at_altitude-fix): a NEW-semantics baseline at
    # z0=2.4 km, generated after the at_altitude fix and never compared
    # against pre-fix fixtures.  It gives the truncated default the same
    # bit-exact protection the elevation=0 fixtures give the old domain.
    t0 = time.perf_counter()
    atmo = presets.clear(WL7).at_altitude(2.4)
    rad = sky_radiance_total(atmo, 85.0, 45.0, 135.0)
    _save("altitude",
          dict(atmo="presets.clear(WL7).at_altitude(2.4)",
               entry="rt.solver.sky_radiance_total",
               sun_zenith_deg=85.0, view_zenith_deg=45.0, view_azimuth_deg=135.0,
               elevation_km=2.4),
          radiance=rad, wl=atmo.wl, z_bounds=atmo.z_bounds)
    print(f"  altitude: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
