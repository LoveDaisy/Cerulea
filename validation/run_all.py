"""Run every validation check and print a summary table.

Exit status
-----------
* 0 -- every check that ran passed (skipped checks do not fail the run),
* 1 -- at least one check that ran failed its tolerance.

Execution model
---------------
One global ``ProcessPoolExecutor`` (spawn, ``MAX_WORKERS`` workers) created
in :func:`main`: check6's spherical-engine Monte-Carlo points are dispatched
first so they overlap the deterministic work, then checks 1-5 run in
workers; check6 itself (deterministic points, figure, altitude response)
runs in the parent and consumes the engine futures.  Only the parent writes
``validation/results.json``; worker exceptions propagate -- a crashed check
never masquerades as a physical FAIL and no partial summary is saved.
``checkN.run()`` and the standalone commands remain the serial reference;
the same authoritative functions execute in both paths.

Usage::

    cd "/path/to/skycolor-wt/validation"
    /path/to/.venv/bin/python validation/run_all.py
"""
from __future__ import annotations

import concurrent.futures as futures
import importlib
import multiprocessing as mp
import sys
from collections.abc import Mapping

sys.path.insert(0, ".")

from validation import common as C  # noqa: E402

CHECKS = [
    ("validation.check1_rayleigh", "check1", "Rayleigh optical depth (Bodhaine 1999)"),
    ("validation.check2_illuminance", "check2", "Direct/diffuse clear-sky illuminance"),
    ("validation.check3_cie_luminance", "check3", "CIE 110 clear-sky luminance distribution"),
    ("validation.check4_chromaticity", "check4", "Sky chromaticity vs measured spectra"),
    ("validation.check5_twilight", "check5", "Twilight zenith luminance"),
    ("validation.check6_second_dataset", "check6", "Twilight vs Koomen (1952)"),
]

#: Measured optimum for the reference development machine (12 cores): one
#: global pool shared by checks and engine points, no nested pools.
MAX_WORKERS = 8


def _run_check(module_name, key, title):
    """Worker body for checks 1-5: import and run one check.

    Module-level on purpose: spawn workers unpickle it by qualified name.
    A missing check module keeps the historical SKIP semantics; any other
    error propagates to the parent.
    """
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith(module_name.split(".")[-1]):
            return C.CheckResult(key=key, title=title,
                                 summary="not implemented (skipped)").finalise()
        raise
    return mod.run()


def _engine_point_worker(h):
    """Worker body for one check6 spherical-engine Monte-Carlo point."""
    from validation import check6_second_dataset as c6

    return c6.engine_point(h)


class _EnginePoints(Mapping):
    """Lazy read-only view over the dispatched engine-point futures.

    First access to a depth waits for its worker and memoises the arrays,
    so repeated lookups inside check6 never rerun the Monte-Carlo.
    """

    def __init__(self, pending):
        self._pending = dict(pending)
        self._resolved = {}

    def __getitem__(self, h):
        if h not in self._resolved:
            self._resolved[h] = self._pending[h].result()
        return self._resolved[h]

    def __iter__(self):
        return iter(self._pending)

    def __len__(self):
        return len(self._pending)


def main():
    results_by_key = {}
    ctx = mp.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                     mp_context=ctx) as pool:
        from validation import check6_second_dataset as c6

        # Engine points first: the Monte-Carlo workers overlap the checks.
        engine_futures = {h: pool.submit(_engine_point_worker, h)
                          for h in c6.engine_depths()}
        check_futures = {}
        for module_name, key, title in CHECKS:
            if key == "check6":
                continue  # deterministic/figure work stays in the parent
            check_futures[key] = pool.submit(_run_check, module_name, key, title)

        results_by_key["check6"] = c6.run(engine_points=_EnginePoints(engine_futures))
        for module_name, key, title in CHECKS:
            if key == "check6":
                continue
            results_by_key[key] = check_futures[key].result()
    results = [results_by_key[key] for _m, key, _t in CHECKS]

    print("=" * 96)
    print("SKY-COLOUR VALIDATION SUMMARY")
    print("=" * 96)
    print(f"{'check':<8s} {'status':<7s} {'what':<52s} key numbers")
    print("-" * 96)
    for r in results:
        print(f"{r.key:<8s} {r.verdict:<7s} {r.title[:52]:<52s} {r.summary}")
    print("-" * 96)

    # metric-level table for failed checks (the interesting part)
    failed = [r for r in results if r.verdict == "FAIL"]
    if failed:
        print("\nFAILED METRICS")
        for r in failed:
            print(f"\n[{r.key}] {r.title}")
            if r.error:
                print(r.error)
            for m in r.metrics:
                if not m.passed:
                    print(
                        f"  {m.name}: model={C.fmt(m.model)} ref={C.fmt(m.reference)} "
                        f"{m.unit} dev={m.dev_text()} tol={m.tolerance}"
                    )

    payload = {
        "checks": [
            {
                "key": r.key,
                "title": r.title,
                "verdict": r.verdict,
                "summary": r.summary,
                "error": r.error,
                "metrics": [
                    {
                        "name": m.name,
                        "model": m.model,
                        "reference": m.reference,
                        "unit": m.unit,
                        "dev": m.dev,
                        "dev_kind": m.dev_kind,
                        "tolerance": m.tolerance,
                        "passed": m.passed,
                    }
                    for m in r.metrics
                ],
            }
            for r in results
        ]
    }
    C.save_json(C.VAL / "results.json", payload)

    n_fail = sum(r.verdict == "FAIL" for r in results)
    n_pass = sum(r.verdict == "PASS" for r in results)
    n_skip = sum(r.verdict == "SKIP" for r in results)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
