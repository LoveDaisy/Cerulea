"""Small real production calls plus final-display and artifact round-trip checks.

Explicit theme suite only: never import this module from the default suite.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

# Like the CSSP theme suite, explicit pytest works without installing research.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest

from experiments.approximations import compare as C
from skycolor import color, render
from skycolor.rt import solver


@pytest.fixture(scope="module")
def real_data():
    return C.compute(n_zenith=3, n_azimuth=3,
                     wl=np.array([450.0, 550.0, 650.0]), seeds=())


def test_calls_reach_authoritative_solver(monkeypatch):
    calls = []
    original = solver.solve

    def spy(atmo, sun_z, zen, az, **kwargs):
        calls.append((atmo, sun_z, zen.copy(), az.copy(), kwargs.copy()))
        return original(atmo, sun_z, zen, az, **kwargs)

    monkeypatch.setattr(solver, "solve", spy)
    data = C.compute(2, 2, np.array([450.0, 550.0, 650.0]), seeds=())
    assert len(calls) == 6
    assert len({id(c[0]) for c in calls}) == 1
    for group, sun_z in ((calls[:3], 60.0), (calls[3:], 87.0)):
        assert [c[4]["approximation"] for c in group] == [
            "plane-sos", "hybrid-sos", "spherical-single"]
        for atmo, z, zen, az, kwargs in group:
            assert z == sun_z and atmo.refraction
            np.testing.assert_array_equal(zen, group[0][2])
            np.testing.assert_array_equal(az, group[0][3])
            assert kwargs["sos_config"] is None  # retain production defaults
    C.check(data)


def test_roundtrip_and_csv_recompute(real_data, tmp_path):
    archive = tmp_path / "data.npz"
    np.savez_compressed(archive, **real_data)
    with np.load(archive, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    C.check(data)
    C.write_tables(data, tmp_path)
    with (tmp_path / "representative.csv").open() as f:
        for row in csv.DictReader(f):
            key, part = row["run"], row["component"]
            zen, az = data[f"{key}/zenith"], data[f"{key}/azimuth_rel"]
            i = np.flatnonzero(zen == float(row["zenith_deg"]))[0]
            j = np.flatnonzero(az == float(row["azimuth_rel_deg"]))[0]
            rad = data[f"{key}/{part}"][i*len(az)+j]
            xyz = color.spectrum_to_xyz(rad, data["wl"])
            np.testing.assert_allclose(float(row["Lv_cd_m-2"]), 683*xyz[1],
                                       rtol=1e-12, atol=0)
            np.testing.assert_allclose([float(row["x"]), float(row["y"])],
                                       color.chromaticity(xyz), rtol=1e-12, atol=0)


def test_final_display_boundary_and_no_resolve(real_data, monkeypatch, tmp_path):
    original = render.render_result
    calls = []

    def spy(pres, **kwargs):
        calls.append(kwargs.copy())
        return original(pres, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("redraw must not solve again")

    monkeypatch.setattr(render, "render_result", spy)
    monkeypatch.setattr(render, "solve_dome", forbidden)
    monkeypatch.setattr(solver, "solve", forbidden)
    C.draw(real_data, tmp_path)
    assert len(calls) == 6 and all(k == C.DISPLAY for k in calls)
    assert C.DISPLAY["exposure"] == 0.03
    assert C.DISPLAY["adaptation"] is None
    assert not C.DISPLAY["sun_disk"] and not C.DISPLAY["ground"]
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "sky_03.png", "sky_30.png", "approximations.png", "spectra.png"}


def test_default_vs_explicit_publish_destination(real_data, monkeypatch, tmp_path):
    theme = tmp_path / "experiments" / "approximations"
    monkeypatch.setattr(C, "THEME", theme)
    monkeypatch.setattr(C, "compute", lambda: real_data)
    destinations = []
    monkeypatch.setattr(C, "draw", lambda data, dest: destinations.append(dest))
    C.main([])
    assert destinations == [theme / "figures"]
    assert not (tmp_path / "figures").exists()
    C.main(["--from-npz", str(theme / "out/representative.npz"), "--publish-figures"])
    assert destinations[-1] == tmp_path / "figures"


def test_bad_component_does_not_pass_dark_absolute_tolerance(real_data):
    bad = dict(real_data)
    key = "e3_hybrid-sos/total"
    bad[key] = real_data[key].copy() * 1e-20
    with pytest.raises(AssertionError):
        C.check(bad)
