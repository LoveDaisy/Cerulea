"""Research-area boundary regression for the default suite.

Structural checks only -- the *runtime* isolation proof (a minimal tree with
no research paths present, running production/tests/validation) lives in the
task verification records.  These tests pin the invariants that keep it true:

* the production package never imports tools/experiments;
* default validation never imports tools;
* the default test collection stays inside ``tests/`` and contains no
  research-theme modules;
* experiment themes own their outputs (no import-time side effects, no root
  ``figures`` defaults), and check6 writes into ``validation/figures``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"

RESEARCH_TOPS = {"tools", "experiments"}

ACTIVE_THEMES = {
    "day": "day_cycle",
    "twilight": "twilight",
    "aerosol": "aerosol",
    "heterogeneity": "heterogeneity",
    "altitude": "altitude_latitude",
    "bluesun": "blue_sun",
    "clouds": "clouds",
}


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("path", sorted((REPO / "skycolor").rglob("*.py")))
def test_production_package_never_imports_research_areas(path):
    overlap = _imports_of(path) & RESEARCH_TOPS
    assert not overlap, f"{path.relative_to(REPO)} imports {sorted(overlap)}"


@pytest.mark.parametrize("path", sorted((REPO / "validation").glob("*.py")))
def test_default_validation_never_imports_tools(path):
    overlap = _imports_of(path) & RESEARCH_TOPS
    assert not overlap, f"{path.relative_to(REPO)} imports {sorted(overlap)}"


@pytest.mark.parametrize("path", sorted(TESTS_DIR.glob("test_*.py")))
def test_default_test_modules_stay_out_of_research_areas(path):
    overlap = _imports_of(path) & RESEARCH_TOPS
    assert not overlap, f"{path.relative_to(REPO)} imports {sorted(overlap)}"


def test_research_theme_tests_live_outside_default_collection():
    # the CSSP suite is theme-local; nothing under tests/ may shadow it
    assert not (TESTS_DIR / "test_cssp_observation_inputs.py").exists()
    assert (REPO / "experiments/cssp/tests/test_cssp_observation_inputs.py").exists()
    config = (REPO / "pyproject.toml").read_text()
    assert 'testpaths = ["tests"]' in config


def test_experiments_common_has_no_import_side_effects():
    src = (REPO / "experiments/common.py").read_text()
    header = src.split("def save", 1)[0]
    assert "mkdir" not in header, "common must not create directories on import"
    assert "FIG =" not in header and "OUT =" not in header, \
        "common must not carry a global root-figures default"
    # save() takes an explicit destination path
    assert "def save(fig, path)" in src


@pytest.mark.parametrize("theme,module", sorted(ACTIVE_THEMES.items()))
def test_theme_scripts_own_their_outputs(theme, module):
    src = (REPO / "experiments" / theme / f"{module}.py").read_text()
    assert "THEME = Path(__file__).resolve().parent" in src
    assert 'from common import' not in src, "old repo-root import must be gone"
    assert "save(fig, \"" not in src, "save() calls must pass an explicit path"


def test_check6_writes_into_validation_figures():
    src = (REPO / "validation/check6_second_dataset.py").read_text()
    assert 'C.FIG / "13_twilight_datasets.png"' in src
    assert 'C.ROOT / "figures"' not in src


def test_research_tools_are_retired_from_active_trees():
    for retired in ["tools/point_solar_table.py", "tools/unified_solar_transport.py",
                    "experiments/twilight_figure.py",
                    "experiments/06_gallery.py", "experiments/07_timelapse.py"]:
        assert not (REPO / retired).exists(), retired
    # frozen entry points remain readable as documentation
    assert (REPO / "experiments/retired/README.md").exists()
    assert (REPO / "experiments/cssp/README.md").exists()
