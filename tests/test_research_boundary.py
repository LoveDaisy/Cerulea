"""Research-area boundary regression for the default suite.

Production-side invariants only: these hold whether or not the research
areas exist on disk (the minimal acceptance tree deliberately excludes
them).  Assertions about research-directory *layout* (theme scripts, CSSP
test location, retired README) live in the theme-local
``experiments/tests/test_layout_contracts.py``, which runs explicitly and is
never collected by the default suite.

* the production package never imports tools/experiments;
* default validation never imports tools;
* the default test collection stays inside ``tests/`` and contains no
  research-theme modules;
* check6 writes into ``validation/figures``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"

RESEARCH_TOPS = {"tools", "experiments"}


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


def test_default_collection_never_shadows_research_modules():
    # the CSSP suite is theme-local: the default suite must not carry a copy
    assert not (TESTS_DIR / "test_cssp_observation_inputs.py").exists()
    config = (REPO / "pyproject.toml").read_text()
    assert 'testpaths = ["tests"]' in config


def test_check6_writes_into_validation_figures():
    src = (REPO / "validation/check6_second_dataset.py").read_text()
    assert 'C.FIG / "13_twilight_datasets.png"' in src
    assert 'C.ROOT / "figures"' not in src


def test_research_tools_are_retired_from_active_trees():
    for retired in ["tools/point_solar_table.py", "tools/unified_solar_transport.py",
                    "experiments/twilight_figure.py",
                    "experiments/06_gallery.py", "experiments/07_timelapse.py"]:
        assert not (REPO / retired).exists(), retired
