"""Research-layout contracts (theme-local; never in the default suite).

Run explicitly: ``uv run pytest experiments/tests -q``.  These pin the
layout agreements between the research themes and the shared helpers; the
default production suite deliberately does not depend on any of them (see
``tests/test_research_boundary.py`` for the production-side invariants).
"""
from __future__ import annotations
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

ACTIVE_THEMES = {
    "day": "day_cycle",
    "twilight": "twilight",
    "aerosol": "aerosol",
    "heterogeneity": "heterogeneity",
    "altitude": "altitude_latitude",
    "bluesun": "blue_sun",
    "clouds": "clouds",
    "layerstory": "layer_story",
}


def test_cssp_suite_is_theme_local():
    assert (REPO / "experiments/cssp/tests/test_cssp_observation_inputs.py").exists()
    assert (REPO / "experiments/cssp/cssp_observation_inputs.py").exists()


def test_experiments_common_has_no_import_side_effects():
    src = (REPO / "experiments/common.py").read_text()
    header = src.split("def save", 1)[0]
    assert "mkdir" not in header, "common must not create directories on import"
    assert "FIG =" not in header and "OUT =" not in header, \
        "common must not carry a global root-figures default"
    assert "def save(fig, path)" in src


@pytest.mark.parametrize("theme,module", sorted(ACTIVE_THEMES.items()))
def test_theme_scripts_own_their_outputs(theme, module):
    src = (REPO / "experiments" / theme / f"{module}.py").read_text()
    assert "THEME = Path(__file__).resolve().parent" in src
    assert 'from common import' not in src, "old repo-root import must be gone"
    assert 'save(fig, "' not in src, "save() calls must pass an explicit path"


def test_frozen_entry_points_readable():
    assert (REPO / "experiments/retired/README.md").exists()
    assert (REPO / "experiments/cssp/README.md").exists()
