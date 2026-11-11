"""Fast-tier slow filter + measured-cost scheduling groups for pytest-xdist.

``tests/scheduling.py`` owns both policies; this hook only applies them, in
one ``pytest_collection_modifyitems`` pass with a fixed order:

1. *Fast-tier filter* — unless an explicit selection (``-m`` / ``-k`` /
   ``::`` nodeid) or ``SKYCOLOR_FULL_SUITE`` says otherwise, ``slow``-marked
   tests are deselected (in-place mutation + ``pytest_deselected`` report).
   Filtering must run before packing so the bins below only ever cover the
   population that will actually run, and the same decision must be reached
   in the controller and in every xdist worker — hence the environment
   variable channel owned by ``tests/scheduling.py``.
2. *Group packing* — each surviving item gets an ``xdist_group`` mark naming
   its bin, so ``--dist=loadgroup`` sends the whole bin to one worker.
   Assignment depends only on the collected nodeid list, which is identical
   in every xdist worker, so all workers agree on the grouping without
   coordination.  Serial runs (``-n 0``, no ``-n``, or no xdist plugin) skip
   the marking entirely.
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

from scheduling import assign_bins, fast_tier_active  # noqa: E402


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
# silently degrades to per-test load distribution.  The slow filter shares
# this hookimpl so "deselect, then pack" is a fixed intra-function order —
# no cross-plugin ordering assumption.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    if fast_tier_active(config) and items:
        kept, deselected = [], []
        for item in items:
            (deselected if item.get_closest_marker("slow") is not None else kept).append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = kept
    n = _worker_count(config)
    if not n or not items:
        return
    # assign_bins returns full-nodeid -> bin; consume it verbatim.  Deriving
    # module keys here would re-implement (and drift from) the packing in
    # tests/scheduling.py — the one semantic implementation lives there.
    bin_of = assign_bins([item.nodeid for item in items], n)
    for item in items:
        item.add_marker(pytest.mark.xdist_group(f"bin{bin_of[item.nodeid]}"))
