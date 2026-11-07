"""Attach measured-cost scheduling groups for pytest-xdist.

``tests/scheduling.py`` owns the packing policy; this hook only applies it.
Each item gets an ``xdist_group`` mark naming its bin, so ``--dist=loadgroup``
sends the whole bin to one worker.  Assignment depends only on the collected
nodeid list, which is identical in every xdist worker, so all workers agree
on the grouping without coordination.  Serial runs (``-n 0``, no ``-n``, or
no xdist plugin) skip the marking entirely.
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

from scheduling import assign_bins  # noqa: E402


def _worker_count(config):
    """Number of xdist workers actually in play, or ``None`` when serial.

    The three execution contexts need distinct sources for N:

    * Inside an xdist worker process ``config.workerinput["workercount"]``
      is the real worker count.  Reading ``numprocesses`` here silently
      disables the whole schedule: xdist's own ``setup_config`` (3.8.0
      ``remote.py``) nulls ``config.option.numprocesses`` in workers, and
      distributed runs block controller-side collection, so a hook that
      only trusts the option never marks anything in a real ``-n 12`` run.
    * On the controller — a real distributed run's configuration step, and
      any single-process invocation whose addopts carry ``-n N`` (this
      includes ``--collect-only``, which collects locally) — the
      ``numprocesses`` option is authoritative.
    * ``-n 0`` or an unset value means serial: no groups.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return None
    if hasattr(config, "workerinput"):
        return int(config.workerinput["workercount"])
    value = config.getvalue("numprocesses")
    if value is None:
        return None
    if isinstance(value, str):  # "auto" and similar
        return os.cpu_count() or 1
    return int(value) or None


# tryfirst is load-bearing: xdist's worker-side hookimpl reads these marks and
# encodes each bin into the collection it reports as a ``nodeid@binN`` suffix,
# which is the *only* channel the controller's LoadGroupScheduling sees
# (distributed runs block controller-side collection entirely).  Attached too
# late, the marks exist but the suffixes are empty and ``--dist=loadgroup``
# silently degrades to per-test load distribution.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    n = _worker_count(config)
    if not n or not items:
        return
    # assign_bins returns full-nodeid -> bin; consume it verbatim.  Deriving
    # module keys here would re-implement (and drift from) the packing in
    # tests/scheduling.py — the one semantic implementation lives there.
    bin_of = assign_bins([item.nodeid for item in items], n)
    for item in items:
        item.add_marker(pytest.mark.xdist_group(f"bin{bin_of[item.nodeid]}"))
