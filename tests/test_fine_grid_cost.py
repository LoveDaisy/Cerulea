"""Cheap bookkeeping/byte-contract probes, without production dome solves."""
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
