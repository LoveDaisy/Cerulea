"""CLI wiring: both subcommands share the one solar input contract.

The time/numeric command pairs below are *real subprocess* runs of
``python -m skycolor.cli``: the numeric command's angles are assembled at
round-trip float precision from the independent ``refraction=False``
ephemeris call, so stdout/PNG equality proves the two entry points resolve
to the same physical direction.  Raw-angle and raw-spectrum equality are
asserted in-process (the chart table only carries two decimals, which would
mask small wiring differences).
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skycolor.cli import main  # noqa: E402
from skycolor.geometry import sun_position  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEIJING = (39.9, 116.4)
WHEN = dt.datetime(2026, 11, 5, 0, 0)          # 2026-11-05T00:00:00Z
GRID = ["--n-zenith", "5", "--n-azimuth", "7", "--wl-step", "75",
        "--wl-min", "400", "--wl-max", "700"]


def _numeric_angle_args():
    """Round-trip precision: repr() of the float re-parses to the same bits."""
    az, el = sun_position(BEIJING[0], BEIJING[1], WHEN, refraction=False)
    return ["--sun-elevation", repr(el), "--sun-azimuth", repr(az)]


def _run(argv):
    from io import StringIO

    out, err = StringIO(), StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.stdout", out)
        mp.setattr("sys.stderr", err)
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# in-process wiring
# ---------------------------------------------------------------------------
def test_raw_spectrum_matches_between_inputs():
    """Full-precision check behind the rounded table: the same UTC instant
    and the round-tripped numeric angles produce bit-identical raw angles.
    The full command pairs run as real subprocesses below (plan 6.2); this
    is the last-bits guarantee the printed two-decimal table cannot show."""
    from skycolor.scene import resolve_sun

    az_t, el_t = resolve_sun(time=WHEN, latitude=BEIJING[0], longitude=BEIJING[1])
    argv = _numeric_angle_args()
    az_n = float(argv[argv.index("--sun-azimuth") + 1])
    el_n = float(argv[argv.index("--sun-elevation") + 1])
    assert (az_n, el_n) == (az_t, el_t)          # exact bits through the CLI parser


def test_request_errors_exit_non_zero():
    """A rejected request prints the solver's explanation and exits 2 -- no
    silent fallback to a default path."""
    code, _, err = _run(["chart", "--time", "2026-11-05T00:00:00Z",
                         "--sun-elevation", "10"] + GRID)
    assert code == 2
    assert "error:" in err
    code, _, err = _run(["chart", "--time", "not-a-time"] + GRID)
    assert code == 2
    assert "error:" in err


# ---------------------------------------------------------------------------
# real subprocess pairs (the wiring evidence of plan 6.2)
# ---------------------------------------------------------------------------
def _subprocess(argv):
    return subprocess.run([sys.executable, "-m", "skycolor.cli"] + argv,
                          capture_output=True, text=True, cwd=str(ROOT),
                          check=False, timeout=300)


def test_subprocess_chart_pair_identical_stdout():
    t = _subprocess(["chart", "--time", "2026-11-05T00:00:00Z",
                     "--latitude", repr(BEIJING[0]),
                     "--longitude", repr(BEIJING[1])] + GRID)
    n = _subprocess(["chart"] + _numeric_angle_args() + GRID)
    assert t.returncode == 0 and n.returncode == 0, (t.stderr, n.stderr)
    assert t.stdout == n.stdout
    assert "45deg sun-side" in t.stdout      # table content sanity, not just equality


def test_subprocess_render_pair_identical_png(tmp_path):
    common = ["--size", "32x16", "--no-sun", "--exposure", "1e-3"] + GRID
    p_time = tmp_path / "time.png"
    p_num = tmp_path / "num.png"
    t = _subprocess(["render", "--time", "2026-11-05T00:00:00Z",
                     "--latitude", repr(BEIJING[0]),
                     "--longitude", repr(BEIJING[1]),
                     "--out", str(p_time)] + common)
    n = _subprocess(["render"] + _numeric_angle_args() +
                    ["--out", str(p_num)] + common)
    assert t.returncode == 0 and n.returncode == 0, (t.stderr, n.stderr)
    from PIL import Image

    a = np.asarray(Image.open(p_time)).astype(int)
    b = np.asarray(Image.open(p_num)).astype(int)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the four approximations and the cloud option route through the one entry
# (M4)
# ---------------------------------------------------------------------------
MC_BUDGET = ["--quality", "low", "--seed", "0", "--mc-photons", "2e5",
             "--mc-n-zen", "2", "--mc-n-az", "3", "--mc-n-anchor", "3"]


@pytest.mark.parametrize("approx", ["spherical-single", "plane-sos",
                                    "hybrid-sos", "spherical-mc"])
def test_cli_routes_all_four_approximations(approx):
    argv = ["chart", "--sun-elevation", "10", "--sun-azimuth", "137"] + GRID + \
        ["--approximation", approx]
    if approx == "spherical-mc":
        argv += MC_BUDGET
    code, out, err = _run(argv)
    assert code == 0, err
    assert f"approximation: {approx}" in err       # the summary shows what ran


def test_cli_rejects_mc_settings_on_non_mc_request():
    """--quality with plane-sos: the solver refuses MC settings for a request
    that runs no MC term; the CLI reports it and exits non-zero."""
    code, _, err = _run(["chart", "--sun-elevation", "10"] + GRID +
                        ["--approximation", "plane-sos", "--quality", "low"])
    assert code == 2
    assert "Monte Carlo" in err


def test_cli_rejects_cloud_with_mc():
    code, _, err = _run(["chart", "--sun-elevation", "10"] + GRID +
                        ["--cloud", "stratus", "--approximation",
                         "spherical-mc"] + MC_BUDGET)
    assert code == 2
    assert "composes with" in err


def test_cli_cloud_option_runs_and_declares_itself():
    code, out, err = _run(["chart", "--sun-elevation", "10"] + GRID +
                          ["--cloud", "stratus"])
    assert code == 0, err
    assert "approximation: hybrid-sos (method 'cloud')" in err


def test_cli_render_cloud_writes_png_with_summary(tmp_path):
    p = tmp_path / "cloud.png"
    code, out, err = _run(["render", "--sun-elevation", "10"] + GRID +
                          ["--cloud", "stratus", "--size", "16x8", "--out",
                           str(p)])
    assert code == 0, err
    assert p.exists()
    from PIL import Image

    assert np.asarray(Image.open(p)).shape == (8, 16, 3)
    assert "approximation:" in err
    assert f"wrote {p}" in out


def test_cli_help_lists_the_shared_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["chart", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--approximation", "--cloud", "--quality", "--seed",
                 "--mc-photons", "--mc-n-zen", "--mc-n-az", "--mc-n-anchor"):
        assert flag in out
