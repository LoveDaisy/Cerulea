"""Explicit theme tests for the archive-only figure refresh.

Run explicitly: ``uv run pytest -n 0 -q experiments/tests/test_figures_refresh.py``.
Never in the default suite (experiments/ is outside the default testpaths).

What these pins guarantee, on the real committed archive:

- the redraw path NEVER re-enters the solver, the solar source or any solve
  orchestration entry -- every such entry is stubbed to raise while figures
  are actually generated (contract from the task plan Step 1);
- the fisheye re-projection is genuinely equidistant (r/R = theta/90) through
  the production direction grid, with the quadrant/seam conventions the
  annotations rely on;
- annotations are computed from the same camera grids the pixels came from:
  analytic horizon/locator anchors, and out-of-FOV directions are reported,
  never clamped to the image edge;
- colours reach the production color pipeline with the archive's locked
  group exposures, and the sampled spectra passed in are the archive's own.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # theme-local tests live outside the package
    sys.path.insert(0, str(REPO))

from experiments.layerstory import layer_story as ls  # noqa: E402
from experiments.layerstory import refresh_figures as rf  # noqa: E402
from skycolor import color, render  # noqa: E402
from skycolor.rt.result import SkyResult  # noqa: E402

NPZ = REPO / "experiments/layerstory/data/layer_story.npz"
V3_NPZ = REPO / "experiments/layerstory/data/layer_story_v3.npz"

_SOLVER_ENTRIES = (
    "skycolor.rt.solver.solve",
    "skycolor.render.solve_dome",
    "skycolor.render.render_scene",
    "skycolor.solar_source.SolarSource",
)


def _raiser(*_args, **_kwargs):
    raise AssertionError("a solver/solar-source entry was reached during "
                         "the archive-only redraw -- zero-solve contract broken")


@pytest.fixture(scope="module")
def data():
    return rf.load(NPZ)


@pytest.fixture(scope="module")
def v3():
    """The committed augmented archive (layer_story_v3.npz)."""
    return rf.load(V3_NPZ)


@pytest.fixture(autouse=True)
def no_solve(monkeypatch):
    """Every solver/orchestration entry raises; the redraw must not care."""
    for target in _SOLVER_ENTRIES:
        monkeypatch.setattr(target, _raiser, raising=True)
    monkeypatch.setattr(ls, "solve", _raiser, raising=True)
    monkeypatch.setattr(ls, "compute", _raiser, raising=True)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def test_audit_endpoints_and_capability(data, v3):
    report = rf.audit(data, NPZ)
    dome_runs = [k for k, cap in report["runs"].items() if cap["dome_views"]]
    assert dome_runs == ["l1_rayleigh_30", "l2_aerosol_30", "l4_bridge_30",
                         "l4_multiple_30", "l5_single_-4", "l5_mc_-4",
                         "l5_plane_-4", "l6_clear_30", "l6_cloud_30"]
    for key in ("l3_bridge_3", "l3_o3_3"):
        cap = report["runs"][key]
        assert cap["zenith_nodes"] == 1 and not cap["dome_views"]
        assert cap["zenith_span"] == [0.0, 0.0]
    for key, cap in report["runs"].items():
        if cap["dome_views"]:
            assert cap["zenith_span"] == [0.0, 90.0]
            assert cap["azimuth_span"] == [0.0, 180.0]
            assert cap["rows_match"]
    # the original archive lacks the L3 dome: overview/ozone report it by
    # name instead of crashing, everything else stays feasible
    assert not report["figures"]["overview"]["feasible"]
    assert report["figures"]["overview"]["missing_runs"] == ["l3_o3_dome_3"]
    assert not report["figures"]["ozone"]["feasible"]
    assert report["figures"]["ozone"]["missing_runs"] == ["l3_o3_dome_3"]
    for fig_id in ("rayleigh", "aerosol", "multiple", "twilight", "cloud",
                   "pipeline-spectrum", "pipeline-view"):
        assert report["figures"][fig_id]["feasible"], fig_id
    assert not report["mixed_source"]
    md = rf.render_audit_markdown(report)
    assert "missing: l3_o3_dome_3" in md
    assert "all runs map to" in md
    # the augmented archive adds one full dome run and everything is feasible
    report_v3 = rf.audit(v3, V3_NPZ)
    dome_runs_v3 = [k for k, cap in report_v3["runs"].items()
                    if cap["dome_views"]]
    assert dome_runs_v3 == dome_runs + ["l3_o3_dome_3"]
    cap = report_v3["runs"]["l3_o3_dome_3"]
    assert cap["zenith_nodes"] == 25 and cap["azimuth_nodes"] == 37
    assert all(f["feasible"] for f in report_v3["figures"].values())
    assert report_v3["mixed_source"]
    for fig in report_v3["figures"].values():
        for curve in fig["curves"]:
            assert curve["exact_node"], curve


def test_audit_records_versions_and_hash(data):
    report = rf.audit(data, NPZ)
    assert report["input_sha256"] == rf.sha256_file(NPZ)
    assert report["solve_version"] == str(data["version"])
    # the archive's solve version is pinned: redraws must never re-solve,
    # and the drawing-code version is reported separately in the report
    assert report["solve_version"].startswith("286486e")
    assert "drawing-code version" in rf.render_audit_markdown(report)


# ---------------------------------------------------------------------------
# projection anchors: the grids the panels consume
# ---------------------------------------------------------------------------
def test_fisheye_grid_is_equidistant():
    """r/R = theta/90 anchors: zenith 0, 45 deg at half radius, horizon rim."""
    size = 81
    az, el, mask = render.fisheye_grid(size, max_zenith=90.0,
                                       projection="equidistant")
    cx = cy = (size - 1) / 2.0
    rim = (size - 1) / 2.0
    assert el[int(cy), int(cx)] == pytest.approx(90.0, abs=1e-9)
    half = int(round(rim / 2.0))
    assert el[int(cy), int(cx) - half] == pytest.approx(45.0, abs=0.5)
    assert el[int(round(cy - rim)), int(cx)] == pytest.approx(0.0, abs=0.5)
    assert el[int(cy), 0] == pytest.approx(0.0, abs=0.5)  # left rim
    assert mask[int(cy), int(cx)] and mask[int(cy), 1]  # disk inside, corners out
    assert not mask[0, 0]


def test_fisheye_azimuth_quadrants():
    """The convention the tick placement relies on: 0 top, 90 right, ..."""
    size = 81
    az, el, _ = render.fisheye_grid(size, projection="equidistant")
    c = int((size - 1) / 2.0)
    rim = (size - 1) / 2.0
    assert az[c, int(c + rim)] == pytest.approx(90.0, abs=0.5)
    assert az[int(c + rim), c] == pytest.approx(180.0, abs=0.5)
    assert az[c, c - int(rim)] == pytest.approx(270.0, abs=0.5)
    assert az[c - int(rim), c] == pytest.approx(0.0, abs=0.5)


def test_sample_dome_mirrors_across_the_principal_plane(data):
    """Relative-azimuth seam: symmetric dome -> +1 deg and -1 deg identical."""
    key = "l1_rayleigh_30"
    dome = dict(zenith=data[f"{key}/zenith"],
                azimuth_rel=data[f"{key}/azimuth_rel"],
                radiance=data[f"{key}/total"].reshape(25, 37, 81))
    plus = render.sample_dome(dome, np.array([45.0]), np.array([1.0]))
    minus = render.sample_dome(dome, np.array([45.0]), np.array([-1.0]))
    np.testing.assert_array_equal(plus, minus)
    wrap = render.sample_dome(dome, np.array([45.0]), np.array([179.0]))
    wrap2 = render.sample_dome(dome, np.array([45.0]), np.array([-179.0]))
    np.testing.assert_array_equal(wrap, wrap2)


def test_rect_grid_horizon_and_center_anchor():
    """Analytic anchors on the archived camera: horizon row and a direction."""
    w, h, hfov, pitch, yaw = 640, 320, 100.0, 14.0, 0.0
    az, el = render.perspective_grid(w, h, hfov, pitch, yaw)
    cols, rows = rf.horizon_polyline(el)
    assert cols.size == w  # one crossing per column: a straight vanishing line
    assert np.ptp(rows) < 2.0
    f = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
    expected_row = h / 2.0 + f * np.tan(np.radians(pitch)) - 0.5
    assert float(np.median(rows)) == pytest.approx(expected_row, abs=2.0)
    # a 30-deg-high direction toward the sun resolves to its own pixel
    p = rf.locate(az, el, 0.0, 30.0)
    assert p is not None
    c, r = p
    assert el[r, c] == pytest.approx(30.0, abs=0.5)
    # azimuth wraps at 360: compare as a signed relative angle
    az_rel = (az[r, c] + 180.0) % 360.0 - 180.0
    assert az_rel == pytest.approx(0.0, abs=0.5)


def test_sun_placement_modes():
    """In view -> marker; below horizon -> horizon indicator; out of FOV ->
    outside note (never clamped to the image edge)."""
    az, el = render.perspective_grid(640, 320, 100.0, 14.0, 0.0)
    kind, p = rf.sun_placement(az, el, 0.0, 30.0)
    assert kind == "marker" and p is not None
    kind, p = rf.sun_placement(az, el, 0.0, -4.0)
    assert kind == "horizon" and p is not None
    cols, rows = rf.horizon_polyline(el)
    # the indicator pixel sits on the horizon line at the sun's azimuth
    assert rows[p[0]] == pytest.approx(float(p[1]), abs=2.0)
    # anti-solar camera: the sun azimuth is behind the view
    az180, el180 = render.perspective_grid(640, 320, 100.0, 14.0, 180.0)
    assert rf.sun_placement(az180, el180, 0.0, 30.0) == ("outside", None)
    assert rf.sun_placement(az180, el180, 0.0, -4.0) == ("outside", None)


# ---------------------------------------------------------------------------
# zero-solve figure generation on the real archive
# ---------------------------------------------------------------------------
def _png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def test_sample_figures_render_without_solve(data, monkeypatch):
    """rayleigh + twilight generate with every solver entry stubbed to raise;
    the archive bytes are untouched; the captured color inputs are the
    archive's own spectra at the locked group exposures."""
    seen = []
    twilight_fish = {}

    orig = color.render_rgb

    def spy(rad, wl, **kwargs):
        rad = np.asarray(rad, dtype=float)
        seen.append(dict(shape=rad.shape, **kwargs))
        # hold one fisheye chunk only (the arrays are large); the first
        # 640px-fisheye call at the twilight exposure is l5_single chunk 0
        if (rad.shape == (64 * 640, 81) and kwargs.get("exposure") == 10.0
                and "chunk0" not in twilight_fish):
            twilight_fish["chunk0"] = rad
        return orig(rad, wl, **kwargs)

    monkeypatch.setattr(color, "render_rgb", spy)
    out = rf.render_figures(data, ["rayleigh", "twilight"], NPZ)
    assert len(out) == 2
    for path in out:
        img = _png(path)
        assert img.shape[0] > 300 and img.shape[1] > 300
    assert twilight_fish, "twilight fisheye must render at exposure 10.0"
    rad = twilight_fish["chunk0"]
    fish_calls = [c for c in seen
                  if c["shape"] == (64 * 640, 81) and c["exposure"] == 10.0]
    assert fish_calls[0]["tonemap"] == "aces"
    assert fish_calls[0]["adaptation"] is None
    # the first chunk of that panel equals an independent sample_dome read of
    # the archived total field through the same equidistant grid
    az, el, _ = render.fisheye_grid(640, max_zenith=90.0,
                                    projection="equidistant")
    rel = (az - 0.0 + 180.0) % 360.0 - 180.0
    zen = 90.0 - np.clip(el, 0.0, 90.0)
    dome = dict(zenith=data["l5_single_-4/zenith"],
                azimuth_rel=data["l5_single_-4/azimuth_rel"],
                radiance=data["l5_single_-4/total"].reshape(11, 23, 81))
    expected = render.sample_dome(dome, zen[:64].ravel(), rel[:64].ravel())
    np.testing.assert_allclose(rad, expected, rtol=1e-12, atol=0)
    # the day panels stay on the day group's locked exposure
    assert any(c["exposure"] == 0.03 for c in seen)


def test_regeneration_is_pixel_identical(data, tmp_path):
    """Same archive, same code, same params -> identical decoded pixels."""
    first = rf.render_figures(data, ["rayleigh"], NPZ)[0]
    px1 = _png(first)
    second = rf.render_figures(data, ["rayleigh"], NPZ)[0]
    np.testing.assert_array_equal(px1, _png(second))


def test_all_figures_render_without_solve(v3):
    """The full batch (--figure all equivalent) generates all nine registered
    figures with every solver entry stubbed to raise, and both archives' bytes
    are untouched before/after."""
    sha_before = rf.sha256_file(V3_NPZ)
    sha_old = rf.sha256_file(NPZ)
    out = rf.render_figures(v3, sorted(rf.FIGURES), V3_NPZ)
    assert len(out) == len(rf.FIGURES) == 9
    for path in out:
        assert _png(path).shape[0] > 300
    assert rf.sha256_file(V3_NPZ) == sha_before
    assert rf.sha256_file(NPZ) == sha_old


def test_overview_is_strictly_eight_renders(v3, monkeypatch):
    """Acceptance ④/负向验收: the overview drawing object is 2x4, every axis
    holds an image and no axis holds a legend (curves) -- and the L3 panel
    binds the pre-registered augmentation run, not a pseudo-dome."""
    fig = rf.build_figure(v3, "overview")
    assert len(fig.axes) == 8
    for ax in fig.axes:
        assert len(ax.images) == 1
        assert ax.get_legend() is None
    captured = []
    orig = render.render_result

    def spy(pres, **kwargs):
        captured.append(pres)
        return orig(pres, **kwargs)

    monkeypatch.setattr(render, "render_result", spy)
    rf.build_figure(v3, "overview").canvas.draw()
    l3 = [p for p in captured
          if np.array_equal(np.asarray(p.radiance).reshape(-1, p.wl.size),
                            v3["l3_o3_dome_3/total"])]
    assert len(l3) == 1


def test_manifest_records_command_and_directions(data):
    rf.render_figures(data, ["aerosol"], NPZ)
    manifest = json.loads((rf.OUT / "refresh_manifest.json").read_text())
    entry = manifest[0]
    assert entry["fig_id"] == "aerosol"
    assert "--figure aerosol" in entry["command"]
    assert entry["input_sha256"] == rf.sha256_file(NPZ)
    assert entry["solve_version"] == str(data["version"])
    assert entry["missing_runs"] == []
    for curve in entry["curves"]:
        z, a = curve["requested"]
        assert curve["row"] == ls.row_index(data, curve["key"], z, a)
    # per-run solve provenance: the archive's global version unless the run
    # carries its own
    for key, version in entry["run_solve_versions"].items():
        expected = (str(data[f"{key}/solve_version"])
                    if f"{key}/solve_version" in data else str(data["version"]))
        assert version == expected


def test_manifest_distinguishes_mixed_sources(v3):
    """The ozone manifest entry names the two provenances separately: the
    archived zenith-row curves keep the original batch version, the sky views
    carry the augmentation run's own solve version."""
    rf.render_figures(v3, ["ozone"], V3_NPZ)
    manifest = json.loads((rf.OUT / "refresh_manifest.json").read_text())
    entry = next(e for e in manifest if e["fig_id"] == "ozone")
    versions = entry["run_solve_versions"]
    assert set(versions.values()) >= {str(v3["version"]),
                                      str(v3["l3_o3_dome_3/solve_version"])}
    assert versions["l3_o3_3"] == str(v3["version"])
    assert versions["l3_o3_dome_3"] == str(v3["l3_o3_dome_3/solve_version"])


def test_plain_run_writes_only_theme_outputs(data):
    """A plain run never touches root figures/ or the historical snapshots."""
    out = rf.render_figures(data, ["cloud"], NPZ)
    assert len(out) == 1
    theme_out = (rf.OUT / "figures").resolve()
    assert out[0].resolve().parent == theme_out


def test_publish_names_are_the_registered_v3_set():
    """Publishing writes only the registry's own *_v3.png names (nine figures
    including the overview); the retired v1/v2 root names are never republished
    or overwritten by it."""
    expected = {"layer_story_overview_v3.png",
                "layer_story_rayleigh_v3.png", "layer_story_aerosol_v3.png",
                "layer_story_ozone_v3.png", "layer_story_multiple_v3.png",
                "layer_story_twilight_v3.png", "layer_story_cloud_v3.png",
                "layer_story_pipeline_spectrum_v3.png",
                "layer_story_pipeline_view_v3.png"}
    assert {f.publish_name for f in rf.FIGURES.values()} == expected
    for name in expected:
        assert name.startswith("layer_story_") and name.endswith("_v3.png")
    retired = {"layer_story_overview.png", "layer_story_spectra.png",
               "layer_story_rayleigh_v2.png", "layer_story_aerosol_v2.png",
               "layer_story_ozone_v2.png", "layer_story_multiple_v2.png",
               "layer_story_twilight_v2.png", "layer_story_cloud_v2.png",
               "layer_story_pipeline_spectrum_v2.png",
               "layer_story_pipeline_view_v2.png"}
    assert not retired & {f.publish_name for f in rf.FIGURES.values()}


def test_historical_snapshots_match_recorded_hashes():
    """historical/README.md is the hash registry of the retired figure bytes
    (v1 montages and the v2 per-position set): every recorded snapshot file
    must exist and match its recorded sha256 prefix."""
    import re

    text = (REPO / "experiments/layerstory/historical/README.md").read_text()
    rows = re.findall(r"\| `([^`]+\.png)` \| `([0-9a-f]{16})` \|", text)
    assert len(rows) >= 10, "the historical README must record all snapshots"
    for name, short in rows:
        path = REPO / "experiments/layerstory/historical" / name
        assert path.exists(), name
        assert rf.sha256_file(path).startswith(short), name


def test_ground_consumption_matches_registry(data, monkeypatch):
    """The registry's ground decision reaches render_result: the cloud pair
    renders through the production ground path consuming each run's own
    solve-bound materials; pipeline-view renders ground=False (plan §3) --
    captured consumption, not declarations."""
    calls = []
    orig = render.render_result

    def spy(pres, **kwargs):
        calls.append((pres, dict(kwargs)))
        return orig(pres, **kwargs)

    monkeypatch.setattr(render, "render_result", spy)
    rf.render_figures(data, ["cloud", "pipeline-view"], NPZ)
    grounds = [(pres, kw) for pres, kw in calls if kw.get("ground") is True]
    noground = [(pres, kw) for pres, kw in calls
                if kw.get("ground") is False]
    assert len(grounds) == 2 and len(noground) == 1
    matched = set()
    for pres, _kw in grounds:
        for key in ("l6_clear_30", "l6_cloud_30"):
            if (np.array_equal(pres.ground_albedo,
                               data[f"{key}/ground_albedo"])
                    and np.array_equal(pres.ground_direct_irradiance,
                                       data[f"{key}/ground_direct_irradiance"])):
                matched.add(key)
    assert matched == {"l6_clear_30", "l6_cloud_30"}
    pres, _kw = noground[0]
    np.testing.assert_array_equal(pres.radiance.reshape(-1, pres.wl.size),
                                  data["l6_clear_30/total"])


# ---------------------------------------------------------------------------
# the v3 augmentation contract (task-figures-polish): exactly ONE
# pre-registered solve, gated by the tracked preregister; every test here
# stubs render.solve_dome -- the real solve ran once, outside the tests
# ---------------------------------------------------------------------------
def _stub_solve_dome(captured: list, radiance=None):
    """A render.solve_dome stub: capture the request, then abort -- or bind a
    synthetic dome field (assembled from archived arrays, never computed)."""
    def stub(atmo, sun_el, sun_az, **kwargs):
        captured.append(dict(atmo=atmo, sun_el=sun_el, sun_az=sun_az, **kwargs))
        if radiance is None:
            # SystemExit passes through augment's solver-failure wrapper
            # untouched: capture and abort before any real compute
            raise SystemExit("captured the pre-registered request -- "
                             "aborting before any real compute")
        res = SkyResult(radiance=radiance, wl=ls.WL, method="clear",
                        options=dict(sun_zenith_deg=90.0 - sun_el,
                                     sun_azimuth_deg=sun_az,
                                     approximation="plane-sos",
                                     single=True, multiple=False),
                        components={"single": radiance},
                        approximation="plane-sos")
        return render.presentation_from_result(
            res, np.linspace(0.0, 90.0, 25), np.linspace(0.0, 180.0, 37),
            sun_el, sun_az)
    return stub


def test_augment_request_contract(data, monkeypatch, tmp_path):
    """The single dispatch: correct medium, scene, grid and no MC/cloud --
    captured, then aborted before any real compute."""
    captured = []
    monkeypatch.setattr(render, "solve_dome", _stub_solve_dome(captured))
    with pytest.raises(SystemExit, match="captured"):
        ls.augment_l3_dome(destination=tmp_path)
    assert len(captured) == 1
    call = captured[0]
    atmo = call["atmo"]
    assert call["sun_el"] == 3.0 and call["sun_az"] == 0.0
    assert atmo.ozone_du == 300.0 and atmo.no2_column == ls.NO2_COLUMN
    assert atmo.aerosols[0].aod550 == ls.AOD_CLEAR
    assert call["approximation"] == "plane-sos"
    assert call["single"] is True and call["multiple"] is False
    assert call["n_zenith"] == 25 and call["n_azimuth"] == 37
    assert "mc_config" not in call and "cloud_layer" not in call


def test_augment_refuses_mismatched_preregister(data, monkeypatch, tmp_path):
    """A tampered preregister stops the dispatch before any solver call."""
    captured = []
    monkeypatch.setattr(render, "solve_dome", _stub_solve_dome(captured))
    pre = json.loads(ls.PREREGISTER_PATH.read_text())
    pre["run"]["grid"]["n_zenith"] = 46
    tampered = tmp_path / "preregister.json"
    tampered.write_text(json.dumps(pre))
    with pytest.raises(SystemExit, match="preregister mismatch"):
        ls.augment_l3_dome(destination=tmp_path, preregister=tampered)
    assert captured == []


def test_augment_refuses_existing_run_record(tmp_path):
    (tmp_path / "l3_dome_run.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        ls.augment_l3_dome(destination=tmp_path)


def test_augment_assembles_v3_on_stubbed_field(data, monkeypatch, tmp_path):
    """Full assembly over a synthetic dome tiled from the archived zenith row:
    the real zenith crosscheck passes by construction, old fields ride along
    bit-for-bit, provenance is recorded, and the written archive re-reads
    under the same guards as the committed one."""
    radiance = np.tile(np.asarray(data["l3_o3_3/total"])[0], (925, 1))
    captured = []
    monkeypatch.setattr(render, "solve_dome", _stub_solve_dome(captured, radiance))
    new = ls.augment_l3_dome(destination=tmp_path)
    runs = [str(r) for r in new["runs"]]
    assert len(runs) == 12 and runs[-1] == "l3_o3_dome_3"
    assert runs[:-1] == [str(r) for r in data["runs"]]
    with np.load(NPZ, allow_pickle=False) as old:
        for k in old.files:
            if k == "runs":  # extended by the augmentation run, prefix-checked
                continue
            np.testing.assert_array_equal(new[k], old[k], err_msg=k)
    for r in runs[:-1]:
        assert str(new[f"{r}/solve_version"]) == str(data["version"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert str(new["l3_o3_dome_3/solve_version"]) == head
    assert str(new["l3_o3_dome_3/code_fingerprint"])
    assert str(new["l3_o3_dome_3/display"]) == repr(ls.display_kwargs(
        ls.l3_dome_spec()))
    record = json.loads((tmp_path / "l3_dome_run.json").read_text())
    assert record["run"] == "l3_o3_dome_3" and record["head"] == head
    assert record["elapsed_s"] <= ls.L3_DOME_BUDGET_S
    v3 = tmp_path / "layer_story_v3.npz"
    assert v3.exists() and (tmp_path / "layers_v3.csv").exists()
    assert (tmp_path / "layers_v3.md").exists()
    assert record["outputs"]["layer_story_v3.npz"] == rf.sha256_file(v3)
    reloaded = rf.load(v3)
    assert str(reloaded["runs"][-1]) == "l3_o3_dome_3"
    zen_rows = reloaded["l3_o3_dome_3/total"].reshape(25, 37, -1)[0]
    np.testing.assert_allclose(
        zen_rows, np.broadcast_to(np.asarray(data["l3_o3_3/total"])[0],
                                  zen_rows.shape),
        rtol=ls.L3_DOME_ZENITH_RTOL, atol=0)


def test_augment_refuses_bad_field_and_keeps_evidence(data, monkeypatch,
                                                       tmp_path):
    """Both refusal paths keep evidence and never write an archive: a solver
    failure (unbindable field) and a verification failure (negative radiance)
    each stop after exactly one call."""
    unbindable = np.tile(np.asarray(data["l3_o3_3/total"])[0], (900, 1))
    negative = np.tile(np.asarray(data["l3_o3_3/total"])[0], (925, 1))
    negative[500, 40] = -1e-3
    for bad, pattern in ((unbindable, "solve failed"),
                         (negative, "verification failed")):
        captured = []
        monkeypatch.setattr(render, "solve_dome", _stub_solve_dome(captured, bad))
        with pytest.raises(SystemExit, match=pattern):
            ls.augment_l3_dome(destination=tmp_path)
        assert len(captured) == 1
        assert (tmp_path / "l3_dome_failure.json").exists()
        assert not (tmp_path / "layer_story_v3.npz").exists()
        (tmp_path / "l3_dome_failure.json").unlink()  # next loop iteration


# ---------------------------------------------------------------------------
# the committed v3 archive itself (read-only: the real solve ran exactly once)
# ---------------------------------------------------------------------------
def test_augmented_archive_contract(data, v3):
    """The committed v3 archive carries the old 11 runs bit-for-bit plus the
    one preregistered L3 dome run, with per-run solve provenance and a run
    record consistent with the tracked files."""
    with np.load(NPZ, allow_pickle=False) as old:
        for k in old.files:
            if k == "runs":
                continue
            np.testing.assert_array_equal(v3[k], old[k], err_msg=k)
    runs = [str(r) for r in v3["runs"]]
    assert runs[:-1] == [str(r) for r in data["runs"]]
    assert runs[-1] == "l3_o3_dome_3"
    cap = rf.run_capability(v3, "l3_o3_dome_3")
    assert cap["dome_views"] and cap["zenith_nodes"] == 25
    assert cap["azimuth_nodes"] == 37
    pre = json.loads(ls.PREREGISTER_PATH.read_text())
    record = json.loads((ls.V3_ARCHIVE.parent / "l3_dome_run.json").read_text())
    assert record["run"] == "l3_o3_dome_3"
    assert str(v3["l3_o3_dome_3/solve_version"]) == record["head"]
    assert record["elapsed_s"] <= ls.L3_DOME_BUDGET_S
    assert pre["budget"]["wall_clock_s"] == ls.L3_DOME_BUDGET_S
    assert float(v3["l3_o3_dome_3/elevation"]) == pre["run"]["sun_elevation_deg"]
    assert v3["l3_o3_dome_3/zenith"].size == pre["run"]["grid"]["n_zenith"]
    assert v3["l3_o3_dome_3/azimuth_rel"].size == pre["run"]["grid"]["n_azimuth"]
    # stored preset repeatability identity, read back from the two archives
    zen_rows = np.asarray(v3["l3_o3_dome_3/total"]).reshape(25, 37, -1)[0]
    np.testing.assert_allclose(
        zen_rows,
        np.broadcast_to(np.asarray(data["l3_o3_3/total"])[0], zen_rows.shape),
        rtol=ls.L3_DOME_ZENITH_RTOL, atol=0)
    for r in runs[:-1]:
        assert str(v3[f"{r}/solve_version"]) == str(data["version"])
    assert str(v3["l3_o3_dome_3/code_fingerprint"]) == record["code_fingerprint"]
    assert str(v3["source_archive_sha256"]) == rf.sha256_file(NPZ)
    assert str(v3["source_archive_sha256"]) == pre["source_archive"]["sha256"]
    for name, sha in record["outputs"].items():
        assert rf.sha256_file(ls.V3_ARCHIVE.parent / name) == sha


# ---------------------------------------------------------------------------
# v3 geometry: camera-relative angle axes, square fisheye box, per-run notes
# ---------------------------------------------------------------------------
def test_angle_axes_analytic_anchors():
    """α/β ticks sit exactly on the archived pinhole mapping, only inside the
    open field, with honest signs (α rightward, β upward on shrinking rows)."""
    from matplotlib.figure import Figure

    display = dict(size=(640, 320), hfov_deg=100.0)
    info = rf.angle_axes(Figure().add_subplot(), display)
    f = (640 / 2.0) / np.tan(np.radians(50.0))
    pos = dict(info["horizontal"])
    assert set(pos) == {-40, -20, 0, 20, 40}  # ±49.98° edges exclude ±60
    assert pos[0] == pytest.approx((640 - 1) / 2, abs=1e-9)
    assert pos[40] == pytest.approx((640 - 1) / 2 + f * np.tan(np.radians(40)),
                                    rel=1e-12)
    assert pos[40] > pos[-40]
    vert = dict(info["vertical"])
    assert set(vert) == {-20, 0, 20}  # ±30.72° edges exclude ±40
    assert vert[20] == pytest.approx((320 - 1) / 2 - f * np.tan(np.radians(20)),
                                     rel=1e-12)
    assert vert[20] < vert[0] < vert[-20]  # rows grow downward, beta upward


def test_angle_axes_are_camera_relative():
    """The tick mapping reads only size/hfov: identical for yaw 0 and yaw 180
    cameras, while the production direction field points those same columns at
    opposite world azimuths -- camera-relative axes, absolute annotations."""
    from matplotlib.figure import Figure

    display = dict(size=(640, 320), hfov_deg=100.0)
    info0 = rf.angle_axes(Figure().add_subplot(), display)
    info180 = rf.angle_axes(Figure().add_subplot(), display)
    assert info0 == info180
    az0, el0 = render.perspective_grid(640, 320, 100.0, 14.0, 0.0)
    az180, el180 = render.perspective_grid(640, 320, 100.0, 14.0, 180.0)
    cx = int((640 - 1) / 2)
    cy = int((320 - 1) / 2)
    # the alpha=0 half-pixel column aims at the camera yaw; the beta=0 row
    # sits at the pitch -- production readback of the pixel-centre convention
    # (integer neighbours of the 319.5 column agree within ~0.11 deg)
    signed = lambda az: (az + 180.0) % 360.0 - 180.0
    assert np.abs(signed(az0[:, cx])).max() == pytest.approx(0.11, abs=0.05)
    assert np.abs(signed(az180[:, cx]) - 180.0).max() == pytest.approx(
        0.11, abs=0.05)
    assert el0[cy, cx] == pytest.approx(14.0, abs=0.15)
    assert el180[cy, cx] == pytest.approx(14.0, abs=0.15)
    # a different legal camera moves the ticks with it
    other = rf.angle_axes(Figure().add_subplot(),
                          dict(size=(640, 320), hfov_deg=90.0))
    assert other["focal_px"] == pytest.approx(320.0 / np.tan(np.radians(45.0)))
    assert dict(other["horizontal"])[20] != dict(info0["horizontal"])[20]


def test_rect_fisheye_geometry_after_draw(data):
    """Final drawn canvas: the fisheye axes box is square with exact symmetric
    image limits and no pixel ticks; the rect panel keeps exact image limits
    with the camera-relative angle tick labels."""
    fig = rf.build_figure(data, "rayleigh")
    fig.canvas.draw()
    fish = rect = None
    for ax in fig.axes:
        if not ax.images:
            continue
        span = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        if span > 600:
            fish = ax
        else:
            rect = ax
    assert fish is not None and rect is not None
    bb = fish.get_window_extent()
    assert bb.width == pytest.approx(bb.height, abs=1.5)
    assert tuple(fish.get_xlim()) == (-0.5, 639.5)
    assert tuple(fish.get_ylim()) == (639.5, -0.5)
    assert fish.get_xticks().size == 0 and fish.get_yticks().size == 0
    assert tuple(rect.get_xlim()) == (-0.5, 639.5)
    assert tuple(rect.get_ylim()) == (319.5, -0.5)
    f = (640 / 2.0) / np.tan(np.radians(50.0))
    for axis, get_ticks, get_labels, angle_of in (
            ("x", rect.get_xticks, lambda: [t.get_text() for t in
                                            rect.get_xticklabels()],
             lambda lab: np.radians(int(lab.rstrip("°")))),
            ("y", rect.get_yticks, lambda: [t.get_text() for t in
                                            rect.get_yticklabels()],
             lambda lab: np.radians(int(lab.rstrip("°"))))):
        labels = get_labels()
        assert {lab.rstrip("°") for lab in labels} <= {"-40", "-20", "0",
                                                       "20", "40"}
        for x, lab in zip(get_ticks(), labels):
            a = angle_of(lab)
            if axis == "x":
                expected = (640 - 1) / 2 + f * np.tan(a)
            else:
                expected = (320 - 1) / 2 - f * np.tan(a)
            assert x == pytest.approx(expected, abs=1.0), (axis, lab)
    assert "camera-relative" in rect.get_xlabel()
    assert "camera-relative" in rect.get_ylabel()


def test_footnote_and_labels_are_per_run(data):
    """The figure note is generated from each run's own grid and zero cells:
    a 25 x 37 day figure says 25 x 37, and only runs with horizon zeros get
    the truncation sentence with the corrected out-of-window wording."""
    ray = rf.run_footnote(data, ["l1_rayleigh_30"])
    assert "25 x 37" in ray and "11 x 23" not in ray
    assert "truncation" not in ray
    twi = rf.run_footnote(data, ["l5_single_-4", "l5_mc_-4", "l5_plane_-4"])
    assert "11 x 23" in twi
    assert "l5_single_-4: horizon-row truncation zeros" in twi
    assert "147–180" in twi
    assert "outside the rect frame's azimuth window" in twi
    assert "l5_plane_-4: horizon row zero at all azimuths" in twi
    assert twi.count("truncation zeros") == 1  # MC run has no zero sentence
    fig = rf.build_figure(data, "rayleigh")
    assert "25 x 37" in fig._suptitle.get_text()
    fig2 = rf.build_figure(data, "twilight")
    suptitle = fig2._suptitle.get_text()
    assert "11 x 23" in suptitle and "147–180" in suptitle


def test_spectrum_axis_labels(data):
    """Every spectrum panel is standalone-readable: wavelength x-label and the
    spectral-radiance y-label (log-log and linear alike)."""
    fig = rf.build_figure(data, "rayleigh")
    spec = next(ax for ax in fig.axes if ax.get_legend() is not None)
    assert spec.get_xlabel() == "Wavelength [nm]"
    assert "W m$^{-2}$ sr$^{-1}$ nm$^{-1}$" in spec.get_ylabel()
