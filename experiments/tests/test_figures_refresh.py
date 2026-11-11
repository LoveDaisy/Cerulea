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
def test_audit_endpoints_and_capability(data):
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
    assert all(f["feasible"] for f in report["figures"].values())
    for fig in report["figures"].values():
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


def test_all_figures_render_without_solve(data):
    """The full batch (--figure all equivalent) generates every registered
    figure with every solver entry stubbed to raise, and the archive bytes
    are untouched before/after."""
    sha_before = rf.sha256_file(NPZ)
    out = rf.render_figures(data, sorted(rf.FIGURES), NPZ)
    assert len(out) == len(rf.FIGURES) == 8
    for path in out:
        assert _png(path).shape[0] > 300
    assert rf.sha256_file(NPZ) == sha_before


def test_manifest_records_command_and_directions(data):
    rf.render_figures(data, ["aerosol"], NPZ)
    manifest = json.loads((rf.OUT / "refresh_manifest.json").read_text())
    entry = manifest[0]
    assert entry["fig_id"] == "aerosol"
    assert "--figure aerosol" in entry["command"]
    assert entry["input_sha256"] == rf.sha256_file(NPZ)
    assert entry["solve_version"] == str(data["version"])
    for curve in entry["curves"]:
        z, a = curve["requested"]
        assert curve["row"] == ls.row_index(data, curve["key"], z, a)


def test_plain_run_writes_only_theme_outputs(data):
    """A plain run never touches root figures/ or the historical snapshots."""
    out = rf.render_figures(data, ["cloud"], NPZ)
    assert len(out) == 1
    theme_out = (rf.OUT / "figures").resolve()
    assert out[0].resolve().parent == theme_out


def test_publish_names_are_the_registered_v2_set():
    """Publishing writes only the registry's own *_v2.png names, root originals
    (overview/spectra) are never republished or overwritten by it."""
    expected = {"layer_story_rayleigh_v2.png", "layer_story_aerosol_v2.png",
                "layer_story_ozone_v2.png", "layer_story_multiple_v2.png",
                "layer_story_twilight_v2.png", "layer_story_cloud_v2.png",
                "layer_story_pipeline_spectrum_v2.png",
                "layer_story_pipeline_view_v2.png"}
    assert {f.publish_name for f in rf.FIGURES.values()} == expected
    for name in expected:
        assert name.startswith("layer_story_") and name.endswith("_v2.png")
        assert not Path(name).name in ("layer_story_overview.png",
                                       "layer_story_spectra.png")


def test_historical_snapshots_are_byte_identical():
    """The tracked v1 snapshots equal the root originals byte for byte."""
    import hashlib

    root = REPO / "figures"
    hist = REPO / "experiments/layerstory/historical"
    pairs = [("layer_story_overview.png",
              "historical_layer_story_overview_v1.png"),
             ("layer_story_spectra.png",
              "historical_layer_story_spectra_v1.png")]
    for original, snapshot in pairs:
        a = hashlib.sha256((root / original).read_bytes()).hexdigest()
        b = hashlib.sha256((hist / snapshot).read_bytes()).hexdigest()
        assert a == b, snapshot


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
