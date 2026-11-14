"""The layer story: from Rayleigh to the measured sky, one ingredient at a time.

Seven rungs, each adding exactly one ingredient on top of the previous
medium, all through the public production entry -- this is a thin caller,
not a new solver.  The committed authorities for the ladder and its
comparison baselines are ``docs/physics/LAYER_STORY.md`` (narration) and
the theme README plus ``data/layers.md`` (scene parameters, call record);
authored per the task plan (scratchpad, not tracked).

Run from the repository with ``python -m experiments.layerstory.layer_story``.
The NPZ contains ordinary arrays and literal text, never pickled objects.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import save
from skycolor import cloud, color, presets, render
from skycolor.atmosphere import AerosolLayer, Atmosphere, ExponentialProfile
from skycolor.color import K_M
from skycolor.aerosol import HinterlandAerosol, shared
from skycolor.constants import WL_COARSE
from skycolor.rt.result import SkyResult
from skycolor.rt.solver import MCConfig, solve

THEME = Path(__file__).resolve().parent

#: the working spectral grid (380-780 nm, 5 nm), same as the other themes
WL = WL_COARSE

#: the clean-continental aerosol load every rung from L2 on shares
AOD_CLEAR = 0.05
#: its exponential boundary-layer scale height [km]
HAZE_SCALE_H_KM = 1.5

#: Mid-latitude stratospheric NO2 column [m^-2] (4e15 cm^-2, mid-range of the
#: literature-typical 2-6e15 cm^-2).  The built-in profile *shape* is scaled
#: to this column; the shape's own natural column (~3e16 cm^-2) is an order
#: of magnitude above the typical atmosphere and is NOT used as the value.
NO2_COLUMN = 4.0e19

#: Chart directions for the CSV table (name, zenith deg, relative azimuth deg).
DIRECTIONS = (("zenith", 0.0, 0.0), ("45deg sun-side", 45.0, 0.0),
              ("45deg anti-solar", 45.0, 180.0),
              ("horizon sun-side", 86.25, 0.0),
              ("horizon anti-solar", 86.25, 180.0))

#: Dome grids per scene group.  Day layers render at the approximations
#' theme's resolution; twilight/cloud layers use coarser domes -- their
#: engines price per view direction and the gradients are smooth.
DAY_GRID = dict(n_zenith=25, n_azimuth=37)
TWILIGHT_GRID = dict(n_zenith=11, n_azimuth=23)
CLOUD_GRID = dict(n_zenith=11, n_azimuth=21)

#: MC transport grid: deliberately coarse and budget-limited (qualitative,
#: single seed; no variability study -- see theme README).
MC_GRID = dict(n_zen=3, n_az=4, n_anchor=5, n_photons=2.0e5)

#: Display groups.  Within a group every image shares one exposure locked
#: from the first real result; no per-figure auto exposure or white balance.
#: Day and cloud use the approximations theme's calibrated exposure 0.03
#: (reused verbatim so panels read against the same tonemapping; the cloud
#: deck's zenith Lv ~6.4e3 cd/m2 sits mid-scale there).  Twilight is much
#: darker (zenith Lv ~4.4 cd/m2 at sun -4 deg): its exposure 10.0 was locked
#: from the first real -4 deg solve -- the twilight band readable, headroom
#: kept for the brighter MC panel.
DISPLAY_DAY = dict(camera="perspective", size=(640, 320), hfov_deg=100.0,
                   pitch_deg=14.0, exposure=0.03, tonemap="aces",
                   adaptation=None, sun_disk=False, ground=False)
DISPLAY_TWILIGHT = dict(camera="perspective", size=(640, 320), hfov_deg=100.0,
                        pitch_deg=14.0, exposure=10.0, tonemap="aces",
                        adaptation=None, sun_disk=False, ground=False)
DISPLAY_CLOUD = dict(camera="perspective", size=(640, 320), hfov_deg=100.0,
                     pitch_deg=14.0, exposure=0.03, tonemap="aces",
                     adaptation=None, sun_disk=False, ground=True)
DISPLAY_GROUPS = {"day": DISPLAY_DAY, "twilight": DISPLAY_TWILIGHT,
                  "cloud": DISPLAY_CLOUD}


def _hinterland(aod: float = AOD_CLEAR) -> AerosolLayer:
    return AerosolLayer(shared(HinterlandAerosol), aod,
                        ExponentialProfile(HAZE_SCALE_H_KM))


def build_media():
    """The ladder's three media; each rung switches exactly one ingredient.

    L1 Rayleigh only (no O3, no NO2, no aerosol) -> L2 adds the Hinterland
    boundary-layer aerosol at AOD 0.05 -> L3 adds the 300 DU ozone column
    and the typical NO2 column.  From L3 on, the medium is frozen; later
    rungs only change orders, transport geometry or add a cloud deck.
    """
    return {
        "l1": presets.pristine(WL, ozone_du=0.0),
        "l2": presets.clear(WL, aod=AOD_CLEAR, ozone_du=0.0),
        "l3": Atmosphere(WL, ozone_du=300.0, no2_column=NO2_COLUMN,
                         aerosols=[_hinterland()]),
    }


MEDIA_DESC = {
    "l1": "pristine: Rayleigh only, ozone_du=0, no2_column=0, aerosols=(), "
          "albedo=0.1, sea-level observer, refraction=True",
    "l2": f"clear family minus ozone: HinterlandAerosol AOD550={AOD_CLEAR}, "
          f"exponential H={HAZE_SCALE_H_KM} km, ozone_du=0, no2_column=0, "
          "albedo=0.1",
    "l3": f"HinterlandAerosol AOD550={AOD_CLEAR} (H={HAZE_SCALE_H_KM} km), "
          f"ozone_du=300, no2_column={NO2_COLUMN:.3e} m^-2 (4e15 cm^-2), "
          "albedo=0.1",
}


def _assert_single_ingredient_step(media):
    """Mechanism evidence for the ladder narrative: L3 is exactly L2 plus
    {O3, NO2} -- same albedo, observer, pressure, refraction and the same
    aerosol member/aod/profile; only the gas columns differ.  presets build
    L2 while L3 is hand-constructed, so this is asserted from the objects
    instead of trusted from the constructor calls.
    """
    l2, l3 = media["l2"], media["l3"]
    np.testing.assert_array_equal(l2.ground_albedo, l3.ground_albedo)
    np.testing.assert_allclose(l2.ground_albedo, 0.1)
    assert l2.elevation_km == l3.elevation_km == 0.0
    assert l2.refraction is True and l3.refraction is True
    assert l2.surface_pressure == l3.surface_pressure
    assert l2.ozone_du == 0.0 and l3.ozone_du == 300.0
    assert l2.no2_column == 0.0 and l3.no2_column == NO2_COLUMN
    assert len(l2.aerosols) == len(l3.aerosols) == 1
    a2, a3 = l2.aerosols[0], l3.aerosols[0]
    assert a2.aerosol is a3.aerosol, "the shared member object must be reused"
    assert a2.aod550 == a3.aod550
    assert vars(a2.profile) == vars(a3.profile)
    assert vars(a2.horizontal) == vars(a3.horizontal)
    return ("L3 = L2 + {O3, NO2} verified by object readback: equal albedo "
            "array, elevation/refraction/pressure, the same shared aerosol "
            f"object at AOD550={a3.aod550} with identical profile/horizontal "
            "parameters; only the gas columns differ (0 -> 300 DU, "
            f"0 -> {NO2_COLUMN:.3e} m^-2)")


def _solves(quick: bool):
    """The solve inventory: 10 deterministic + 1 MC (11 total), cold-to-warm.

    Comparison baselines are solved states, per the plan's contrast protocol:
    L2 contrasts L1 (reused), L3 bridges its predecessor's medium at its own
    scene, L4 bridges single-only before adding the second order, L5
    contrasts within the -4 degree scene (single vs MC vs plane world) and
    L6 contrasts a contract-matched hybrid-sos clear/cloud pair.
    """
    day = DAY_GRID if not quick else dict(n_zenith=7, n_azimuth=13)
    twi = TWILIGHT_GRID if not quick else dict(n_zenith=7, n_azimuth=13)
    cld = CLOUD_GRID if not quick else dict(n_zenith=7, n_azimuth=13)
    mc = MC_GRID if not quick else dict(MC_GRID, n_photons=5.0e4)
    return (
        dict(key="l1_rayleigh_30", layer="L1", title="Rayleigh only",
             medium="l1", elevation=30.0, approximation="plane-sos",
             single=True, multiple=False, grid=day, yaw=0.0, display="day"),
        dict(key="l2_aerosol_30", layer="L2", title="+Mie aerosol",
             medium="l2", elevation=30.0, approximation="plane-sos",
             single=True, multiple=False, grid=day, yaw=0.0, display="day"),
        # bridge: the L2 medium re-solved at the L3 scene -- the Chappuis
        # comparison baseline (same scene, medium one rung down)
        dict(key="l3_bridge_3", layer="L3", title="bridge (no O3)",
             medium="l2", elevation=3.0, approximation="plane-sos",
             single=True, multiple=False, views=(0.0,)),
        dict(key="l3_o3_3", layer="L3", title="+O3/NO2",
             medium="l3", elevation=3.0, approximation="plane-sos",
             single=True, multiple=False, views=(0.0,)),
        dict(key="l4_bridge_30", layer="L4", title="bridge (single only)",
             medium="l3", elevation=30.0, approximation="plane-sos",
             single=True, multiple=False, grid=day, yaw=0.0, display="day"),
        dict(key="l4_multiple_30", layer="L4", title="+multiple scattering",
             medium="l3", elevation=30.0, approximation="plane-sos",
             single=True, multiple=True, grid=day, yaw=180.0, display="day"),
        dict(key="l5_single_-4", layer="L5", title="spherical single",
             medium="l3", elevation=-4.0, approximation="spherical-single",
             grid=twi, yaw=0.0, display="twilight"),
        dict(key="l5_mc_-4", layer="L5", title="+MC higher orders",
             medium="l3", elevation=-4.0, approximation="spherical-mc",
             mc=MCConfig(quality="low", seed=7, grid=mc),
             grid=twi, yaw=0.0, display="twilight"),
        # the plane world at -4 degrees is not rejected by the contract and
        # is physically readable: it keeps the upper layers "lit" through
        # grazing-beam column transmittance with no Earth shadow -- the
        # review-pinned contrast figure for "no twilight geometry"
        dict(key="l5_plane_-4", layer="L5", title="plane-parallel world",
             medium="l3", elevation=-4.0, approximation="plane-sos",
             single=True, multiple=False, grid=twi, yaw=0.0,
             display="twilight"),
        dict(key="l6_clear_30", layer="L6", title="clear (same contract)",
             medium="l3", elevation=30.0, approximation="hybrid-sos",
             grid=cld, yaw=0.0, display="cloud"),
        dict(key="l6_cloud_30", layer="L6", title="+stratus deck",
             medium="l3", elevation=30.0, approximation="hybrid-sos",
             cloud_layer="stratus", grid=cld, yaw=0.0, display="cloud"),
    )


def _plain(value):
    """Keep backend diagnostics readable without an object-array/pickle format."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def quantities(radiance, wl):
    """Un-tonemapped XYZ, xy and photopic luminance [cd m^-2]."""
    xyz = color.spectrum_to_xyz(radiance, wl)
    return xyz, color.chromaticity(xyz), K_M * xyz[..., 1]


def display_kwargs(spec):
    """Final render_result arguments for one solve: group + its own yaw."""
    return {**DISPLAY_GROUPS[spec["display"]], "yaw_deg": spec["yaw"]}


def compute(quick=False):
    """Run the ladder once, for real.  Every rung really calls production."""
    media = build_media()
    step_evidence = _assert_single_ingredient_step(media)
    print(f"medium step check: {step_evidence}", flush=True)
    deck = cloud.stratus()  # tau550=20, base 0.5 km, top 1.5 km; observer below
    deck_desc = (f"cloud.stratus(): tau550={deck.tau550}, base {deck.base_km}, "
                 f"top {deck.top_km} km, r_eff={deck.r_eff_um} um, "
                 f"{deck.phase}")
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    data = dict(
        wl=np.asarray(WL), runs=[], quick=np.asarray(bool(quick)),
        version=version,
        medium_step_check=np.asarray(step_evidence),
        config=repr(dict(day_grid=DAY_GRID, twilight_grid=TWILIGHT_GRID,
                         cloud_grid=CLOUD_GRID, mc_grid=MC_GRID,
                         no2_column=NO2_COLUMN, deck=deck_desc)),
        media=repr(MEDIA_DESC),
        display_groups=repr(DISPLAY_GROUPS),
    )
    for spec in _solves(quick):
        print(f"solve {spec['key']} ({spec['title']})", flush=True)
        start = perf_counter()
        if "views" in spec:
            res = solve(media[spec["medium"]], 90.0 - spec["elevation"],
                        np.asarray(spec["views"], dtype=float),
                        np.zeros(len(spec["views"])),
                        sun_azimuth_deg=0.0, approximation=spec["approximation"],
                        single=spec.get("single"), multiple=spec.get("multiple"))
            zenith = np.asarray(spec["views"], dtype=float)
            azimuth = np.zeros(len(spec["views"]))
        else:
            pres = render.solve_dome(
                media[spec["medium"]], spec["elevation"], 0.0,
                approximation=spec["approximation"],
                cloud_layer=deck if spec.get("cloud_layer") else None,
                mc_config=spec.get("mc"),
                single=spec.get("single"), multiple=spec.get("multiple"),
                **spec["grid"])
            res, zenith, azimuth = pres.result, pres.zenith, pres.azimuth_rel
        elapsed = perf_counter() - start
        key = spec["key"]
        data["runs"].append(key)
        data[f"{key}/seconds"] = elapsed
        data[f"{key}/layer"] = spec["layer"]
        data[f"{key}/title"] = spec["title"]
        data[f"{key}/elevation"] = spec["elevation"]
        data[f"{key}/medium"] = spec["medium"]
        data[f"{key}/approximation"] = spec["approximation"]
        data[f"{key}/method"] = res.method
        data[f"{key}/zenith"] = zenith
        data[f"{key}/azimuth_rel"] = azimuth
        data[f"{key}/view_zenith"] = np.repeat(zenith, azimuth.size)
        data[f"{key}/view_azimuth"] = np.tile(azimuth, zenith.size)
        data[f"{key}/summary"] = render.format_result_summary(res)
        if "display" in spec:
            data[f"{key}/display"] = repr(display_kwargs(spec))
            # the solve-bound ground material rides along so --from-npz can
            # rebind a legal ground display without re-solving
            for name in ("ground_albedo", "ground_direct_irradiance"):
                value = getattr(pres, name, None)
                if value is not None:
                    data[f"{key}/{name}"] = value
            data[f"{key}/ground_supported"] = np.asarray(
                bool(getattr(pres, "ground_supported", False)))
        for name in ("options", "effective", "diagnostics", "approximations"):
            data[f"{key}/{name}"] = repr(_plain(getattr(res, name)))
        for name, rad in {"total": res.radiance, **res.components}.items():
            # the overlay also records non-radiance diagnostics (illumination,
            # optics, ...); only (n_view, n_wl) arrays are spectral components
            if not isinstance(rad, np.ndarray) or rad.ndim != 2:
                continue
            data[f"{key}/{name}"] = rad
            for quantity, values in zip(("xyz", "xy", "Lv"), quantities(rad, WL)):
                data[f"{key}/{name}_{quantity}"] = values
        if res.stderr is not None:
            data[f"{key}/stderr"] = res.stderr
        print(f"{elapsed:.3f} s\n{data[f'{key}/summary']}", flush=True)
    data["runs"] = np.asarray(data["runs"])
    check(data)
    return data


def check(data):
    """Same-path closure and the ladder's decomposition identities.

    Closure is per solve (total = sum of its recorded components); adding
    radiance ACROSS rungs is not attempted -- multiple scattering makes the
    optical-depth composition nonlinear, so cross-ingredient additivity is
    physically meaningless.  The identities pin that a bridge really is the
    same state its rung contrasts against, bit for bit.
    """
    for key in data["runs"]:
        # one decomposition family per solve: the overlay records BOTH its
        # trans-attenuated clear+cloud composition and the underlying plain
        # single+multiple clear decomposition -- summing across families
        # would double-count the clear field
        if f"{key}/clear" in data:
            comp_names = ["clear", "cloud"]
        else:
            comp_names = [n for n in ("single", "multiple")
                          if f"{key}/{n}" in data]
        total = data[f"{key}/{comp_names[0]}"].copy()
        for name in comp_names[1:]:
            total += data[f"{key}/{name}"]
        np.testing.assert_array_equal(data[f"{key}/total"], total)
        for part in ("total", *comp_names):
            for name, value in zip(("xyz", "xy", "Lv"),
                                   quantities(data[f"{key}/{part}"], data["wl"])):
                np.testing.assert_allclose(data[f"{key}/{part}_{name}"], value,
                                           rtol=1e-12, atol=0)
    # bridge states are bit-identical to the rung terms they baseline
    np.testing.assert_array_equal(data["l4_bridge_30/total"],
                                  data["l4_multiple_30/single"])
    np.testing.assert_array_equal(data["l5_single_-4/total"],
                                  data["l5_mc_-4/single"])
    # the overlay's plain clear decomposition is the contract-matched clear
    # solve itself (its "clear" component is that field times the deck
    # transmittance -- the attenuation is the physics L6 shows)
    np.testing.assert_array_equal(data["l6_cloud_30/single"]
                                  + data["l6_cloud_30/multiple"],
                                  data["l6_clear_30/total"])


def bind(data, key, part="total"):
    """Bind saved spectra with the production carrier, with no new solve."""
    # the run's own legacy method tag rides along so the rebound presentation
    # stays faithful to the first solve (older archives without the key fall
    # back to "clear" -- display behaviour is identical either way)
    method = str(data[f"{key}/method"]) if f"{key}/method" in data else "clear"
    res = SkyResult(radiance=data[f"{key}/{part}"], wl=data["wl"], method=method)
    kwargs = {}
    if f"{key}/ground_supported" in data and bool(data[f"{key}/ground_supported"]):
        kwargs = dict(ground_supported=True,
                      ground_albedo=data[f"{key}/ground_albedo"],
                      ground_direct_irradiance=data[f"{key}/ground_direct_irradiance"])
    return render.presentation_from_result(res, data[f"{key}/zenith"],
                                           data[f"{key}/azimuth_rel"],
                                           float(data[f"{key}/elevation"]), 0.0,
                                           **kwargs)


def row_index(data, key, zenith_deg, azimuth_deg=0.0):
    """Flat row of the nearest saved direction for one solve.

    Coarse grids may snap a requested direction to a distant node (the
    11-node twilight/cloud grids have no 86.25 deg node, so a "horizon"
    request lands on the exact-horizon 90 deg node); consumers must compare
    the recorded zenith_deg/azimuth_rel_deg, not the direction label.
    """
    zen, az = data[f"{key}/zenith"], data[f"{key}/azimuth_rel"]
    i = int(np.argmin(np.abs(zen - zenith_deg)))
    j = int(np.argmin(np.abs(az - azimuth_deg)))
    return i * len(az) + j


def write_tables(data, directory, csv_name="layers.csv", md_name="layers.md",
                 solve_versions=None, header_notes=()):
    """Write the CSV direction table and the markdown call record.

    ``solve_versions`` switches on per-run provenance lines for archives that
    mix solves from different code states (the v3 augmentation); the default
    single-version batch keeps the original header-only record byte-for-byte.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / csv_name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "layer", "component", "direction", "zenith_deg",
                         "azimuth_rel_deg", "Lv_cd_m-2", "x", "y"))
        for key in data["runs"]:
            key = str(key)
            zen, az = data[f"{key}/zenith"], data[f"{key}/azimuth_rel"]
            if az.size == 1:
                # spectral-row solves: report only the actually solved
                # directions -- five DIRECTIONS labels would silently alias
                # the zenith row and break cross-run direction comparisons
                dirs = tuple((f"solved-view-{i}", float(zz), float(aa))
                             for i, (zz, aa) in enumerate(zip(zen, az)))
            else:
                dirs = DIRECTIONS
            for name, z, a in dirs:
                row = row_index(data, key, z, a)
                for part in ("total", "single", "multiple", "clear", "cloud"):
                    if f"{key}/{part}" not in data:
                        continue
                    writer.writerow((key, data[f"{key}/layer"], part, name,
                                     zen[row // len(az)], az[row % len(az)],
                                     data[f"{key}/{part}_Lv"][row],
                                     *data[f"{key}/{part}_xy"][row]))
    lines = ["# The layer story, actual production calls", "",
             f"Production version: `{data['version']}`", "",
             f"Configuration: `{data['config']}`", "",
             f"Media: `{data['media']}`", "", f"Display groups: `{data['display_groups']}`",
             "", "Order: L1, L2, L3 bridge+state, L4 bridge+full, "
                 "L5 single/MC/plane, L6 clear+cloud.  First call per medium "
                 "is cold; later calls share warmed optics.  No fair speed "
                 "ranking.  MC: qualitative / budget-limited, single seed.",
             "Closure: per-solve component additivity only; cross-ingredient "
             "additivity is not attempted (nonlinear in the optical depths)."]
    if solve_versions is not None:
        lines += ["", "Solve versions: recorded per run below.  Runs sharing "
                      "the production version above belong to the original "
                      "single-version batch; a run with its own solve_version "
                      "is a separately registered augmentation, not part of "
                      "that batch."]
    for note in header_notes:
        lines += ["", note]
    for key in data["runs"]:
        lines += ["", f"## {key}  ({data[f'{key}/layer']} · {data[f'{key}/title']})", "",
                  f"Wall time: {float(data[f'{key}/seconds']):.6f} s", ""]
        if solve_versions is not None:
            lines += [f"Solve version: `{solve_versions[str(key)]}`", ""]
        lines += ["```text", str(data[f"{key}/summary"]), "```"]
        for name in ("options", "effective", "diagnostics", "approximations"):
            lines += ["", f"{name}: `{data[f'{key}/{name}']}`"]
    (directory / md_name).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# the v3 augmentation (task-figures-polish, 2026-11-12): ONE pre-registered
# extra production solve -- the L3 state (+O3/NO2) as a full DAY_GRID dome, so
# the overview carries a rendered L3 panel and the ozone figure gains sky
# views.  The committed 11-run archive keeps its bytes; the augmented archive
# is layer_story_v3.npz with per-run solve provenance.  Nothing here re-runs
# the ladder: compute() and this entry are separate, explicit contracts.
# ---------------------------------------------------------------------------
L3_DOME_RUN = "l3_o3_dome_3"
SOURCE_ARCHIVE = THEME / "data" / "layer_story.npz"
PREREGISTER_PATH = THEME / "data" / "l3_dome_preregister.json"
V3_ARCHIVE = THEME / "data" / "layer_story_v3.npz"
L3_DOME_ZENITH_RTOL = 1e-10
L3_DOME_BUDGET_S = 300.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _code_identity() -> tuple[str, str]:
    """Compute-time code identity: HEAD plus a worktree-cleanliness fingerprint.

    A solve-time HEAD is meaningful only for a clean tree; uncommitted changes
    get a content fingerprint instead of silently posing as HEAD.
    """
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    if not dirty:
        return head, "clean-worktree"
    return head, "dirty:" + hashlib.sha256(
        "\n".join(sorted(dirty)).encode()).hexdigest()[:16]


def l3_dome_spec() -> dict:
    """The pre-registered single-run request (the dome branch of compute, once)."""
    return dict(key=L3_DOME_RUN, layer="L3", title="+O3/NO2 (dome)",
                medium="l3", elevation=3.0, sun_azimuth_deg=0.0,
                approximation="plane-sos", single=True, multiple=False,
                grid=dict(DAY_GRID), yaw=0.0, display="day")


def _assert_l3_medium(atmo) -> str:
    """Object readback of the L3 medium against the registered construction."""
    a = atmo.aerosols[0]
    np.testing.assert_allclose(atmo.ground_albedo, 0.1)
    assert atmo.elevation_km == 0.0 and atmo.refraction is True
    assert atmo.ozone_du == 300.0 and atmo.no2_column == NO2_COLUMN
    assert len(atmo.aerosols) == 1 and a.aod550 == AOD_CLEAR
    assert vars(a.profile) == vars(ExponentialProfile(HAZE_SCALE_H_KM))
    return ("l3 medium readback: ozone_du=300, no2_column="
            f"{NO2_COLUMN:.3e} m^-2, one Hinterland aerosol at "
            f"AOD550={AOD_CLEAR} with H={HAZE_SCALE_H_KM} km profile, "
            "albedo 0.1, sea-level observer, refraction on")


def _validate_preregister(pre: dict, source: Path, source_sha: str) -> None:
    """Every dispatch-blocking field must match the registered request."""
    def fail(what):
        raise SystemExit(f"preregister mismatch: {what} -- no solve may start")

    run = pre["run"]
    spec = l3_dome_spec()
    checks = (
        ("key", spec["key"]), ("layer", spec["layer"]), ("title", spec["title"]),
        ("sun_elevation_deg", spec["elevation"]),
        ("sun_azimuth_deg", spec["sun_azimuth_deg"]),
        ("approximation", spec["approximation"]),
        ("single", spec["single"]), ("multiple", spec["multiple"]),
    )
    for field, expected in checks:
        if run[field] != expected:
            fail(f"{field}: registered {run[field]!r} != code {expected!r}")
    if run["grid"] != {"n_zenith": spec["grid"]["n_zenith"],
                       "n_azimuth": spec["grid"]["n_azimuth"],
                       "min_zenith_deg": 0.0}:
        fail(f"grid: registered {run['grid']}")
    if pre["source_archive"]["sha256"] != source_sha:
        fail(f"source archive sha256 {source_sha[:16]}... does not match the "
             f"registered {pre['source_archive']['sha256'][:16]}...")
    if pre["source_archive"]["path"] != str(source.relative_to(THEME.parents[1])):
        fail(f"source archive path {source} is not the registered one")


def augment_l3_dome(source: Path = SOURCE_ARCHIVE, destination: Path | None = None,
                    preregister: Path = PREREGISTER_PATH) -> dict:
    """Add the single pre-registered L3 dome solve to the archive -- once.

    Exactly one ``render.solve_dome`` call, gated by the tracked preregister
    (source hash, request fields, budget, stop conditions).  The source
    archive is only ever read; the augmented archive ``layer_story_v3.npz``
    plus ``layers_v3.{csv,md}`` land in ``destination`` with per-run solve
    provenance and a run record.  No verification failure, budget breach or
    retry ever produces a second solver call: a failure writes its evidence
    and stops.
    """
    destination = Path(destination) if destination is not None else V3_ARCHIVE.parent
    record_path = destination / "l3_dome_run.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} already exists -- the augmentation "
                         "already ran; a deliberate reproduction must clear it "
                         "explicitly first")
    with np.load(source, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    check(data)
    source_sha = _sha256_file(source)
    _validate_preregister(json.loads(Path(preregister).read_text()),
                          source, source_sha)

    media = build_media()
    _assert_single_ingredient_step(media)
    medium_evidence = _assert_l3_medium(media["l3"])
    spec = l3_dome_spec()
    key = spec["key"]

    print(f"solve {key} ({spec['title']}) -- the single pre-registered "
          "augmentation solve", flush=True)
    start = perf_counter()
    try:
        pres = render.solve_dome(
            media[spec["medium"]], spec["elevation"], spec["sun_azimuth_deg"],
            approximation=spec["approximation"], single=spec["single"],
            multiple=spec["multiple"], **spec["grid"])
    except Exception as exc:  # any solver failure stops the augmentation
        _write_failure_record(destination, preregister, source_sha,
                              dict(elapsed_s=perf_counter() - start,
                                   problems=[f"solve failed: {exc!r}"]))
        raise SystemExit(f"augmentation solve failed: {exc!r} -- the failure "
                         "record is written, no second solve") from exc
    elapsed = perf_counter() - start
    res = pres.result
    wl = np.asarray(data["wl"])
    zenith, azimuth = pres.zenith, pres.azimuth_rel
    total = np.asarray(res.radiance)

    problems = []
    if elapsed > L3_DOME_BUDGET_S:
        problems.append(f"budget: {elapsed:.1f} s > {L3_DOME_BUDGET_S} s cap")
    if total.shape != (zenith.size * azimuth.size, wl.size):
        problems.append(f"shape {total.shape} != "
                        f"({zenith.size * azimuth.size}, {wl.size})")
    if not (np.all(np.diff(zenith) > 0) and np.all(np.diff(azimuth) > 0)
            and zenith[0] == 0.0 and zenith[-1] == 90.0
            and azimuth[0] == 0.0 and azimuth[-1] == 180.0):
        problems.append("grid endpoints not exact 0..90 / 0..180 increasing")
    if not (np.all(np.isfinite(total)) and np.all(total >= 0.0)):
        problems.append("radiance not finite non-negative")
    if problems:
        _write_failure_record(destination, preregister, source_sha,
                              dict(elapsed_s=elapsed, problems=problems))
        raise SystemExit("augmentation verification failed: "
                         + "; ".join(problems) + " -- nothing written but the "
                         "failure record, no second solve")
    # preset repeatability against the archived zenith row, all azimuth nodes
    zen_rows = total.reshape(zenith.size, azimuth.size, wl.size)[0]
    old_zenith = np.asarray(data["l3_o3_3/total"])[0]
    np.testing.assert_allclose(
        zen_rows, np.broadcast_to(old_zenith, zen_rows.shape),
        rtol=L3_DOME_ZENITH_RTOL, atol=0)

    new = dict(data)
    new["runs"] = np.append(np.asarray(data["runs"]), key)
    new[f"{key}/seconds"] = elapsed
    new[f"{key}/layer"] = spec["layer"]
    new[f"{key}/title"] = spec["title"]
    new[f"{key}/elevation"] = spec["elevation"]
    new[f"{key}/medium"] = spec["medium"]
    new[f"{key}/approximation"] = spec["approximation"]
    new[f"{key}/method"] = res.method
    new[f"{key}/zenith"] = zenith
    new[f"{key}/azimuth_rel"] = azimuth
    new[f"{key}/view_zenith"] = np.repeat(zenith, azimuth.size)
    new[f"{key}/view_azimuth"] = np.tile(azimuth, zenith.size)
    new[f"{key}/summary"] = render.format_result_summary(res)
    new[f"{key}/display"] = repr(display_kwargs(spec))
    # the solve-bound ground material rides along like compute() stores it
    for name in ("ground_albedo", "ground_direct_irradiance"):
        value = getattr(pres, name, None)
        if value is not None:
            new[f"{key}/{name}"] = value
    new[f"{key}/ground_supported"] = np.asarray(
        bool(getattr(pres, "ground_supported", False)))
    for name in ("options", "effective", "diagnostics", "approximations"):
        new[f"{key}/{name}"] = repr(_plain(getattr(res, name)))
    for name, rad in {"total": res.radiance, **res.components}.items():
        if not isinstance(rad, np.ndarray) or rad.ndim != 2:
            continue
        new[f"{key}/{name}"] = rad
        for quantity, values in zip(("xyz", "xy", "Lv"), quantities(rad, wl)):
            new[f"{key}/{name}_{quantity}"] = values
    if res.stderr is not None:
        new[f"{key}/stderr"] = res.stderr

    # per-run solve provenance: the 11 ladder runs map to the original global
    # version; the augmentation records its own compute-time identity
    head, fingerprint = _code_identity()
    for old_key in (str(r) for r in data["runs"]):
        new[f"{old_key}/solve_version"] = str(data["version"])
    new[f"{key}/solve_version"] = head
    new[f"{key}/code_fingerprint"] = fingerprint
    try:
        source_rel = str(Path(source).relative_to(THEME.parents[1]))
    except ValueError:
        source_rel = str(source)
    new["source_archive"] = source_rel
    new["source_archive_sha256"] = source_sha
    new["augment_note"] = (
        "runs 1-11 are the original ladder solved at the production version "
        f"below ({str(data['version'])[:7]}); {key} is the single "
        "pre-registered 2026-11-12 augmentation solved at its own recorded "
        "HEAD -- a mixed-source archive, not a 12-solve single version run")
    for k in data:  # the source fields ride along bit-for-bit; "runs" is the
        if k == "runs":  # one intentionally extended key, prefix-checked below
            continue
        if isinstance(data[k], np.ndarray):
            np.testing.assert_array_equal(new[k], data[k])
        else:
            assert new[k] == data[k], k
    old_runs = np.asarray(data["runs"])
    np.testing.assert_array_equal(np.asarray(new["runs"])[:old_runs.size],
                                  old_runs)
    check(new)

    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / V3_ARCHIVE.name, **new)
    solve_versions = {str(r): str(new[f"{r}/solve_version"]) for r in new["runs"]}
    write_tables(new, destination, csv_name="layers_v3.csv",
                 md_name="layers_v3.md", solve_versions=solve_versions,
                 header_notes=[f"Archive provenance: `{new['augment_note']}`",
                               f"Source archive: `{source_rel}` "
                               f"(sha256 `{source_sha[:16]}...`)"])
    record = dict(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run=key, head=head, code_fingerprint=fingerprint,
        elapsed_s=elapsed, budget_s=L3_DOME_BUDGET_S,
        zenith_crosscheck_rtol=L3_DOME_ZENITH_RTOL,
        source_archive=source_rel, source_archive_sha256=source_sha,
        preregister=str(Path(preregister).name),
        preregister_sha256=_sha256_file(Path(preregister)),
        medium_evidence=medium_evidence,
        verification=["shape/endpoints/finite-nonnegative ok",
                      "zenith crosscheck vs l3_o3_3 ok",
                      "old fields bit-equal, check(new) passed"],
        outputs={name: _sha256_file(destination / name) for name in
                 (V3_ARCHIVE.name, "layers_v3.csv", "layers_v3.md")},
    )
    record_path.write_text(json.dumps(record, indent=1))
    print(f"{elapsed:.3f} s\n{new[f'{key}/summary']}\nv3 archive -> "
          f"{destination / V3_ARCHIVE.name}", flush=True)
    return new


def _write_failure_record(destination: Path, preregister, source_sha: str,
                          evidence: dict) -> None:
    """Keep the evidence of a refused augmentation without promoting anything."""
    destination.mkdir(parents=True, exist_ok=True)
    head, fingerprint = _code_identity()
    (destination / "l3_dome_failure.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        head=head, code_fingerprint=fingerprint,
        preregister=str(Path(preregister).name),
        source_archive_sha256=source_sha, **evidence), indent=1))




# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
INK = ("#2a78d6", "#eb6834", "#1baf7a", "#8d6bd8")


def _sky_axis(ax, data, key, title):
    rgb = render.render_result(bind(data, key), **ast.literal_eval(
        str(data[f"{key}/display"])))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _spectrum_axis(ax, data, curves, title, ylabel=True, loglog=False):
    """curves: (key, part, label, style, zenith_deg, azimuth_deg) tuples."""
    for (key, part, label, style, zdeg, adeg), ink in zip(curves, INK):
        wl, rad = data["wl"], data[f"{key}/{part}"]
        row = row_index(data, str(key), zdeg, adeg)
        if loglog:
            ax.loglog(wl, rad[row], style, color=ink, linewidth=2, label=label)
        else:
            ax.plot(wl, rad[row], style, color=ink, linewidth=2, label=label)
    ax.set_title(title, fontsize=11)
    # both axis labels live here so every consumer (log-log or linear) can be
    # read standalone; radiance stays a spectral radiance, never an integral
    ax.set_xlabel("Wavelength [nm]")
    if ylabel:
        ax.set_ylabel("Radiance [W m$^{-2}$ sr$^{-1}$ nm$^{-1}$]")
    if not loglog:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, color="#c3c2b7")
    ax.legend(fontsize=9)


def draw(data, destination):
    """All images use locked per-group display arguments, no auto exposure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)

    # per-layer images, one per rendered solve -- theme-local run outputs,
    # never published to root figures/ (only the two curated figures below)
    theme_figures = THEME / "figures"
    theme_figures.mkdir(parents=True, exist_ok=True)
    for key in data["runs"]:
        if f"{key}/display" not in data:
            continue
        rgb = render.render_result(bind(data, str(key)), **ast.literal_eval(
            str(data[f"{key}/display"])))
        render.save_png(rgb, theme_figures / f"{key}.png")

    # the overview: the whole ladder in one image -- the committed reader's
    # main carrier (published to root figures/ via --publish-figures)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), layout="constrained")
    _sky_axis(axes[0, 0], data, "l1_rayleigh_30", "L1 · Rayleigh only")
    _sky_axis(axes[0, 1], data, "l2_aerosol_30", "L2 · + Mie aerosol")
    _sky_axis(axes[0, 2], data, "l4_multiple_30", "L4 · + multiple (anti-solar)")
    _sky_axis(axes[0, 3], data, "l6_cloud_30", "L6 · + stratus deck")
    _spectrum_axis(
        axes[1, 0], data,
        (("l3_bridge_3", "total", "no O$_3$", "-", 0.0, 0.0),
         ("l3_o3_3", "total", "+ O$_3$/NO$_2$", "-", 0.0, 0.0)),
        "L3 · + O$_3$/NO$_2$ (zenith @ sun 3°)")
    _sky_axis(axes[1, 1], data, "l5_single_-4", "L5 · spherical single")
    _sky_axis(axes[1, 2], data, "l5_mc_-4", "L5 · + MC higher orders")
    _sky_axis(axes[1, 3], data, "l5_plane_-4", "L5 · plane-parallel world")
    fig.suptitle("The layer story · same locked exposure per group · production solves")
    save(fig, destination / "layer_story_overview.png")
    plt.close(fig)

    # the spectral signatures behind the renders (published alongside)
    with plt.rc_context({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
        _spectrum_axis(axes[0, 0], data,
                       (("l1_rayleigh_30", "total", "zenith", "-", 0.0, 0.0),
                        ("l1_rayleigh_30", "total", "horizon", "--", 86.25, 0.0)),
                       "L1 · Rayleigh λ$^{-4}$ colouring (log-log)", loglog=True)
        _spectrum_axis(axes[0, 1], data,
                       (("l1_rayleigh_30", "total", "L1 clean", "-", 86.25, 0.0),
                        ("l2_aerosol_30", "total", "L2 + aerosol", "-", 86.25, 0.0)),
                       "L2 · aerosol whitening (near horizon)")
        _spectrum_axis(axes[0, 2], data,
                       (("l3_bridge_3", "total", "no O$_3$", "-", 0.0, 0.0),
                        ("l3_o3_3", "total", "+ O$_3$/NO$_2$", "-", 0.0, 0.0)),
                       "L3 · Chappuis band (zenith @ 3°)")
        _spectrum_axis(axes[1, 0], data,
                       (("l4_bridge_30", "total", "single only", "-", 45.0, 180.0),
                        ("l4_multiple_30", "total", "+ multiple", "-", 45.0, 180.0)),
                       "L4 · multiple fills the anti-solar sky (45°)")
        _spectrum_axis(axes[1, 1], data,
                       (("l5_single_-4", "total", "spherical single", "-", 0.0, 0.0),
                        ("l5_mc_-4", "total", "+ MC orders", "--", 0.0, 0.0)),
                       "L5 · twilight zenith spectra (sun −4°)")
        xy = axes[1, 2]
        for key, ink in (("l1_rayleigh_30", INK[0]), ("l2_aerosol_30", INK[1]),
                         ("l3_o3_3", INK[2]), ("l4_multiple_30", INK[3]),
                         ("l5_mc_-4", "#0b0b0b")):
            row = row_index(data, key, 0.0)
            xy.plot(data[f"{key}/total_xy"][row, 0], data[f"{key}/total_xy"][row, 1],
                    "o", color=ink, label=str(data[f"{key}/layer"]))
        xy.set_title("L1–L5 zenith chromaticities (each rung its own scene)")
        xy.set_xlabel("CIE x")
        xy.set_ylabel("CIE y")
        xy.grid(alpha=0.3, color="#c3c2b7")
        xy.legend(fontsize=9)
        fig.suptitle("Spectral signatures of each rung (bridge-vs-state pairs)")
        save(fig, destination / "layer_story_spectra.png")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarse bounded smoke run (same solve inventory)")
    parser.add_argument("--publish-figures", action="store_true",
                        help="explicitly write selected production figures to root figures/")
    parser.add_argument("--from-npz", type=Path,
                        help="redraw without solving (saved local arrays)")
    parser.add_argument("--augment-l3-dome", nargs="?", const=str(SOURCE_ARCHIVE),
                        default=None, metavar="SOURCE_NPZ",
                        help="run the single pre-registered L3 dome solve and "
                             "write layer_story_v3.npz + layers_v3.{csv,md} "
                             "(default source: the committed archive); refuses "
                             "to overwrite an existing run record")
    args = parser.parse_args(argv)
    if args.augment_l3_dome is not None:
        if args.quick or args.publish_figures or args.from_npz:
            parser.error("--augment-l3-dome is exclusive of --quick, "
                         "--publish-figures and --from-npz")
        augment_l3_dome(Path(args.augment_l3_dome))
        return
    if args.from_npz:
        with np.load(args.from_npz, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        # Display metadata belongs to the saved experiment, not current defaults.
        if ast.literal_eval(str(data["display_groups"])) != DISPLAY_GROUPS:
            raise ValueError("saved display groups differ from this example; "
                             "do not silently restyle")
        check(data)
    else:
        data = compute(quick=args.quick)
    out = THEME / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not args.from_npz:
        np.savez_compressed(out / "layer_story.npz", **data)
    write_tables(data, out)
    destination = THEME.parents[1] / "figures" if args.publish_figures \
        else THEME / "figures"
    draw(data, destination)


if __name__ == "__main__":
    main()
