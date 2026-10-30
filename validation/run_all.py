"""Run every validation check and print a summary table.

Exit status
-----------
* 0 -- every check that ran passed (skipped checks do not fail the run),
* 1 -- at least one check that ran failed its tolerance.

Usage::

    cd "/path/to/skycolor-wt/validation"
    /path/to/.venv/bin/python validation/run_all.py
"""
from __future__ import annotations

import importlib
import sys

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


def main():
    results = []
    for module_name, key, title in CHECKS:
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.endswith(module_name.split(".")[-1]):
                results.append(
                    C.CheckResult(
                        key=key, title=title,
                        summary="not implemented (skipped)",
                    ).finalise()
                )
                continue
            raise
        results.append(mod.run())

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
