"""Ruler audit recheck -- independent recomputations for RULER_AUDIT.md.

This entry point belongs to the read-only对照合同 audit of check2 / check3 /
check6 (task-ruler-audit).  It recomputes the numbers quoted in
``validation/RULER_AUDIT.md`` from the committed model and reference data, so
every figure in the report can be regenerated with one command::

    uv run python -m validation.ruler_audit_recheck            # all sections
    uv run python -m validation.ruler_audit_recheck --only check2

It is deliberately NOT part of ``run_all.CHECKS`` and NOT collected by pytest:
the audit adds no runtime to the validation harness.  It reads the model only
through the same public helpers the checks use (``validation.common``,
production solver entry points) and never writes anything except stdout.
Physical numbers of ``validation/results.json`` are inputs to this audit, not
outputs; the recheck never touches them.

Evidence-grade legend used in the report:
* 原文前提核对 -- premise read from the original publication (or its archived
  full-text audit, ``docs/twilight/TWILIGHT_REFERENCE_AUDIT.md``),
* 复算       -- computed here from committed data/models,
* 推断       -- inference, explicitly labelled as such.
A second-hand reproduction of a premise is marked 复现（二级）.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, ".")

from validation import common as C  # noqa: E402

SECTIONS = ("check2", "check3", "check6")


def _hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# check2 -- CIE 110 DHI / G173 / Bird reference premises
# ---------------------------------------------------------------------------
def recheck_check2() -> dict:
    _hdr("check2 -- illuminance reference contract")
    raise NotImplementedError("filled in Step 2")


# ---------------------------------------------------------------------------
# check3 -- CIE 110 relative distribution premises
# ---------------------------------------------------------------------------
def recheck_check3() -> dict:
    _hdr("check3 -- relative distribution reference contract")
    raise NotImplementedError("filled in Step 3")


# ---------------------------------------------------------------------------
# check6 -- Koomen (1952) observation contract
# ---------------------------------------------------------------------------
def recheck_check6() -> dict:
    _hdr("check6 -- Koomen observation contract")
    raise NotImplementedError("filled in Step 4")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=SECTIONS, default=None,
                    help="run a single section (default: all)")
    args = ap.parse_args()
    which = SECTIONS if args.only is None else (args.only,)
    for name in which:
        fn = {"check2": recheck_check2, "check3": recheck_check3,
              "check6": recheck_check6}[name]
        fn()
    print("\nall requested sections completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
