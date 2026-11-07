"""Attach measured-cost scheduling groups for pytest-xdist.

See ``tests/scheduling.py`` for the packing policy.  Each item gets an
``xdist_group`` mark naming its bin, so ``--dist=loadgroup`` sends the whole
bin to one worker.  Assignment depends only on the collected nodeid list,
which is identical in every xdist worker, so all workers agree on the
grouping without coordination.  Serial runs (``-n 0`` or no xdist) skip the
marking entirely.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

# The validation package lives in the repo root next to skycolor/ and is not
# installed; put the root on sys.path so tests can import it directly.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduling import assign_bins, load_weights  # noqa: E402


def _worker_count(config):
    """xdist worker count from the actual configuration, or None if serial."""
    if not config.pluginmanager.hasplugin("xdist"):
        return None
    value = config.getvalue("numprocesses")
    if value is None:
        return None
    if isinstance(value, str):  # "auto" and similar
        return os.cpu_count() or 1
    return int(value) or None


def pytest_collection_modifyitems(config, items):
    n = _worker_count(config)
    if not n or not items:
        return
    modules, nodes = load_weights()
    nodeids = [item.nodeid for item in items]
    bin_of = assign_bins(nodeids, n)
    for item in items:
        key = item.nodeid if item.nodeid in nodes else item.nodeid.split("::")[0]
        item.add_marker(pytest.mark.xdist_group(f"bin{bin_of[key]}"))
