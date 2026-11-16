"""Tool-local bookkeeping/byte contracts, without production dome solves.

Run explicitly with ``uv run python -m pytest -n 0 tools/tests``. The default
production suite intentionally cannot import tools (research boundary).
kind: bookkeeping/contract; cost: five synthetic tests, 0.11 s combined.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from tools import fine_grid_cost as cost


def synthetic_run(p=4):
    rows = []
    for i, item in enumerate(cost.worklist()):
        rows.append(dict(**item, pid=i % p, arrays={"single": {"sha256": "same"}},
                         instrumented=False, diagnostics={"converged": True},
                         effective={"single": {"n_gauss": 3, "chunk": 512}},
                         cold_medium=i < p, complete_s=10., worker_offset_s=40.))
    return dict(workers=p, records=rows, pool_wall_s=41., startup_s=1.)


def test_fixed_work_and_projection():
    jobs = cost.worklist()
    assert len(jobs) == 16
    assert len({tuple(j["request"].values()) for j in jobs[8:]}) == 8
    assert {j["request"]["sun_elevation_deg"] for j in jobs} >= {-6, -3, -.125, .25, 3, 30, 90}
    for p in (4, 6, 8):
        run = synthetic_run(p)
        result = cost.projection(run)
        assert result["states"] == 1848
        assert result["component_volumes"] == 48
        assert result["B_s"] == 241
        assert result["adopted_s"] == max(1848*41/16, 241+int(np.ceil(1848/p))*10+40)/.8
        assert result["cold_medium_builds"] == 24


def test_incomplete_and_different_worker_outputs_rejected():
    original = synthetic_run()
    bad = deepcopy(original)
    bad["records"].pop()
    with pytest.raises(ValueError, match="completion"):
        cost.validate_run(bad)
    bad = deepcopy(original)
    bad["records"][0]["arrays"]["single"]["sha256"] = "different"
    with pytest.raises(ValueError, match="byte difference"):
        cost.validate_run(bad)
    bad = deepcopy(original)
    bad["records"][0]["request"]["n_zenith"] = 31
    with pytest.raises(ValueError, match="changed request"):
        cost.validate_run(bad)
    bad = deepcopy(original)
    for row in bad["records"]:
        row["pid"] = 0
    with pytest.raises(ValueError, match="worker"):
        cost.validate_run(bad)


def test_byte_contract_and_budget_no_reset(tmp_path):
    a = np.array([0.])
    assert cost.array_signature(a) != cost.array_signature(-a)
    assert cost.array_signature(a) != cost.array_signature(a.astype("float32"))
    assert cost.array_signature(a) != cost.array_signature(a.reshape(1, 1))
    with cost.budget(tmp_path, "one"):
        pass
    with cost.budget(tmp_path, "different-output-name"):
        pass
    assert len(cost.load(tmp_path / "budget.json")) == 2
    cost.dump(tmp_path / "budget.json", [dict(status="DONE", elapsed_s=14400)])
    with pytest.raises(ValueError, match="exhausted"):
        with cost.budget(tmp_path, "three"):
            pytest.fail("must reject before executing")
    cost.dump(tmp_path / "budget.json", [dict(status="RUNNING")])
    with pytest.raises(ValueError, match="interrupted"):
        with cost.budget(tmp_path, "four"):
            pytest.fail("must reject before executing")


def test_resource_pressure_and_completion_record(tmp_path, monkeypatch):
    samples = [dict(swap="used = 10.00M", vm_stat="Swapouts: 100.",
                    rss_tree_bytes=100, load=[4., 3., 2.]),
               dict(swap="used = 9.00M", vm_stat="Swapouts: 100.",
                    rss_tree_bytes=120, load=[5., 3., 2.])]
    path = tmp_path / "resources.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    resources = cost.resource_summary(path)
    assert resources["memory_eligible"]
    assert resources["swap_growth_mb"] == 0
    samples[1]["vm_stat"] = "Swapouts: 101."
    path.write_text("\n".join(json.dumps(s) for s in samples))
    assert not cost.resource_summary(path)["memory_eligible"]

    archive = tmp_path / "state.npz"
    np.savez(archive, single=np.array([1.]))
    row = dict(archive=str(archive), archive_sha256=cost.sha(archive), pid=42,
               request={"example": True}, arrays=cost.archive_signature(archive))
    cost.dump(archive.with_suffix(".json"), row)
    cost.dump(archive.with_suffix(".started.json"), dict(pid=42, request=row["request"]))
    monkeypatch.setattr(cost, "check_arrays", lambda data: None)
    cost.verify_record(row)
    archive.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError):
        cost.verify_record(row)
    cost.dump(archive.with_suffix(".json"), row)
    np.savez(archive, single=np.array([2.]))
    with pytest.raises(ValueError, match="changed output archive"):
        cost.verify_record(row)


def test_portable_report_arithmetic_is_recomputed(tmp_path):
    runs = [synthetic_run(p) for p in (4, 6, 8, 4, 6, 8)]
    records = []
    for run in runs:
        row = cost.projection(run)
        row.update(resources={"memory_eligible": True}, adoptable=row["cost_eligible"])
        records.append(row)
    ref = {k: runs[0]["records"][0][k] for k in ("request", "arrays", "effective", "diagnostics")}
    data = dict(schema_version=1, runs=runs, references=[ref]*8, replay=[],
                decision=dict(projections=records, target_supported=True),
                compute_charged_s=10., audit_and_export_allowance_s=60.,
                budget_ledger=[dict(elapsed_s=10.)])
    path = tmp_path / "report.json"
    cost.dump(path, data)
    cost.recheck_report(path)
    data["decision"]["projections"][0]["component_volumes"] = 96
    cost.dump(path, data)
    with pytest.raises(ValueError, match="projection arithmetic"):
        cost.recheck_report(path)
