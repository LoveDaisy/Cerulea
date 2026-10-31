"""Read-only coarse/global-majorant mechanism and sampling-control audit.

Run uv run python -m tools.check_majorant_equivalence. Frozen inputs and source
snapshots are tracked; no scratchpad, git-history lookup or new MC is required.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from skycolor import presets, spectra
from skycolor.constants import R_EARTH
from skycolor.rt import mc
from tools.continuous_curved_mc import rewrite_transport
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT/"validation/data/curved_transport_audit"
DEFAULT_SOURCE = ROOT/"validation/data/unified_solar_source_audit"


def run_audit(directory: Path, source_dir: Path) -> dict:
    """Evaluate bounds/cloned code and reintegrate independent/seed-paired records."""
    archive = directory/"majorant-control"
    snapshots = archive/"snapshots"
    frozen_hashes = json.loads((archive/"inputs.sha256.json").read_text())
    for name, expected in frozen_hashes.items():
        if hashlib.sha256((archive/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen majorant-control input changed: {name}")
    for name, expected in json.loads((source_dir/"inputs.sha256.json").read_text()).items():
        if hashlib.sha256((source_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen phase-three solar-source input changed: {name}")
    if (snapshots/"mc_596aaba.py").read_bytes() != (ROOT/"skycolor/rt/mc.py").read_bytes():
        raise ValueError("The MC implementation differs from the audited frozen baseline")
    wl = np.arange(380.,781.,5.)
    base = presets.clear(wl)
    # Reexecute only the baseline normalization method, on the same material states.
    # The library diff since phase three contains aerosol.py alone.
    baseline_aerosol = (snapshots/"aerosol_596aaba.py").read_text()
    old_class = next(node for node in ast.parse(baseline_aerosol).body
                     if isinstance(node,ast.ClassDef) and node.name=="MieAerosol")
    old_method = next(node for node in old_class.body
                      if isinstance(node,ast.FunctionDef) and node.name=="relative_coefficients")
    old_namespace = {"np":np}
    exec(compile(ast.Module(body=[old_method],type_ignores=[]),"baseline-relative-coefficients","exec"),
         old_namespace)
    manifest = json.loads((source_dir/"tables-manifest.json").read_text())
    rows = []
    for height in [0.,2.4]:
        atmo = base if height == 0 else base.at_altitude(height)
        entry = manifest["unified"][str(height)]
        table = PointSolarTable.load(source_dir/entry["path"],atmo,expected_label=entry["label"])
        source = FiniteSolarSource(atmo,table)
        captured = {}
        def capture(code, medium):
            rewritten,extras = rewrite_transport(code,medium,bend=False)
            captured.update(native=code,global_straight=rewritten,extras=extras)
            return rewritten,extras
        with mc_solar_source(source,transport=capture):
            pass  # Clone only. No photons, free flights or RNG draws.
        native,global_code = captured["native"],captured["global_straight"]
        optical_a = "    # --- optical properties"
        optical_b = "    # --- bookkeeping"
        assert native[native.index(optical_a):native.index(optical_b)] == global_code[
            global_code.index(optical_a):global_code.index(optical_b)]
        tail = "        # --- ground events"
        expected_tail = native[native.index(tail):].replace("da[ci]","arrival[ci]").replace(
            "da[si]","arrival[si]")
        assert expected_tail == global_code[global_code.index(tail):]
        assert "_fine_grid" not in captured["extras"]
        z = mc._fine_grid(atmo)
        rb = mc._layer_bounds(atmo)
        global_rb = captured["extras"]["_layer_bounds"](atmo)
        assert rb[0] == global_rb[0] and rb[-1] == global_rb[-1]
        rel = mc._aerosol_spectral(atmo)
        old_rel = [old_namespace["relative_coefficients"](layer.aerosol,atmo.wl)
                   for layer in atmo.aerosols]
        for previous,current in zip(old_rel,rel):
            assert all(np.array_equal(a,b) for a,b in zip(previous,current))
        def extinction(heights):
            beta = (atmo.n_air(heights)[:,None]*atmo._sigma_ray
                    +atmo.n_o3(heights)[:,None]*atmo._sigma_o3
                    +atmo.n_no2(heights)[:,None]*atmo._sigma_no2)
            for layer,(ext,_) in zip(atmo.aerosols,rel):
                beta += (layer.aod550*layer.profile(heights)/1000.)[:,None]*ext
            return beta
        beta = extinction(z)
        majors = np.array([1.005*beta[(z >= (lo-R_EARTH)/1000-1e-12)
                                    &(z <= (hi-R_EARTH)/1000+1e-12)].max(axis=0)
                           for lo,hi in zip(rb[:-1],rb[1:])])
        fractions = np.linspace(0.,1.,33)
        dense = (z[:-1,None]+np.diff(z)[:,None]*fractions).ravel()
        # Include both sides of every boundary and use actual radius/height recovery.
        dense = np.unique(np.clip(np.r_[dense,((rb[:,None]-R_EARTH)/1000
                            +np.array([-1e-8,0.,1e-8])).ravel()],z[0],z[-1]))
        radius = R_EARTH+1000*dense
        dense = (radius-R_EARTH)/1000
        li = np.clip(np.searchsorted(rb,radius,side="right")-1,0,len(rb)-2)
        interpolated = np.column_stack([mc._interp_log(dense,z,np.log(beta[:,i]))
                                        for i in range(len(wl))])
        native_ratio = interpolated/majors[li]
        global_major = 1.005*beta.max(axis=0)
        global_ratio = interpolated/global_major
        continuous_ratio = extinction(dense)/majors[li]
        assert np.max(native_ratio)<1 and np.max(global_ratio)<1
        rows.append(dict(ground_km=height,fine_grid_points=len(z),dense_heights=len(dense),
            wavelength_count=len(wl),coarse_bounds_km=((rb-R_EARTH)/1000).tolist(),
            global_bounds_km=((global_rb-R_EARTH)/1000).tolist(),
            optical_and_phase_table_build_code_identical=True,
            ground_scatter_RR_order_cap_commit_code_identical_after_direction_rename=True,
            global_fine_grid_function_is_native=True,
            phase3_and_current_full81_aerosol_coefficients_bit_identical=True,
            max_native_interpolated_beta_over_major=float(native_ratio.max()),
            max_global_interpolated_beta_over_major=float(global_ratio.max()),
            native_majorant_violations=int((native_ratio>1).sum()),
            global_majorant_violations=int((global_ratio>1).sum()),
            max_native_continuous_beta_over_major=float(continuous_ratio.max()),
            native_max_boundary_skip_optical_depth_per_event=float(beta.max()*mc._EPS_MOVE),
            fine_grid_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
            beta_ext_sha256=hashlib.sha256(beta.tobytes()).hexdigest()))

    old = []
    for path in (source_dir/"raw").glob("*.jsonl"):
        old.extend(json.loads(s) for s in path.read_text().splitlines() if s.strip())
    new = [json.loads(s) for s in (directory/"raw/global-straight-high-stat.jsonl").read_text().splitlines()]
    v = spectra.cie_xyz(wl)[:,1]
    photopic = lambda s: float(683*np.trapezoid(np.asarray(s)*v,wl))
    stats = []
    for h in [0.,2.4]:
        a = [r for r in old if r["mode"]=="unified" and r["elevation"]==h]
        b = [r for r in new if r["mode"]=="unified" and r["elevation"]==h]
        assert len(a)==len(b)==8
        assert {r["seed"] for r in a}.isdisjoint({r["seed"] for r in b})
        assert all(np.array_equal(r["deterministic_single"],a[0]["deterministic_single"])
                   for r in a+b)
        assert len({r["point_table_sha256"] for r in a+b})==1
        x,y = [np.array([photopic(r["fields"]["split_total"]) for r in group]) for group in [a,b]]
        ex,ey = [values.std(ddof=1)/np.sqrt(8) for values in [x,y]]
        stats.append(dict(height_km=h,native_mean=float(x.mean()),native_se=float(ex),
            global_mean=float(y.mean()),global_se=float(ey),
            independent_difference=float(y.mean()-x.mean()),
            combined_se=float(np.hypot(ex,ey)),
            difference_over_combined_se=float((y.mean()-x.mean())/np.hypot(ex,ey)),
            deterministic_SS_all_bit_identical=True,source_NPZ_sha_identical=True))
    old_source = (snapshots/"unified_solar_transport_7ed71a62.py").read_text()
    assert hashlib.sha256(old_source.encode()).hexdigest() == a[0]["source_hashes"][
        "unified_solar_transport.py"]
    old_ast,new_ast = [ast.parse(text) for text in [old_source,
        (ROOT/"tools/unified_solar_transport.py").read_text()]]
    def physics_nodes(tree):
        wanted = {"solar_disk","single_zenith","_geometry","scatter","ground"}
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                if isinstance(node,ast.FunctionDef) and node.name in wanted}
    assert physics_nodes(old_ast) == physics_nodes(new_ast)
    result = dict(rows=rows,independent_seed_comparison=stats,
        source_solar_disk_geometry_scatter_ground_and_deterministic_SS_AST_identical_to_phase3=True,
        native_max_scatter_default=inspect.signature(mc.mc_radiance_split).parameters["max_scatter"].default,
        constants={key:getattr(mc,key) for key in ["_N_FINE","_MAJORANT_SAFETY","_EPS_MOVE",
            "_MAX_STEPS","_RR_WEIGHT","_RR_START","_RR_SURVIVE"]},
        baseline_mc_byte_identical=True,
        implementation_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in map(Path,["skycolor/rt/mc.py","tools/continuous_curved_mc.py",
                                  "tools/unified_solar_transport.py"])},
        scope="No MC run; source/flight clones and fine-grid coefficient evaluations only")

    # Same seed labels define paired independent batches; their trajectories
    # need not coincide. Estimate uncertainty from the observed eight differences.
    controls = [json.loads(line) for line in
        (directory/"raw/native-new-seeds-ground2p4.jsonl").read_text().splitlines() if line.strip()]
    controls = sorted(controls,key=lambda row: row["seed"])
    comparison = sorted([row for row in new if row["elevation"] == 2.4],
                        key=lambda row: row["seed"])
    assert [r["seed"] for r in controls] == list(range(310,318))
    assert [r["seed"] for r in comparison] == list(range(310,318))
    before = (snapshots/"unified_solar_transport_before_guard.py").read_text()
    after = (snapshots/"unified_solar_transport_after_guard.py").read_text()
    added_guard = "".join(line[1:] for line in
        (archive/"source-binding-guard.patch").read_text().splitlines(keepends=True)
        if line.startswith("+") and not line.startswith("+++"))
    assert after.count(added_guard) == 1 and after.replace(added_guard,"",1) == before
    assert physics_nodes(ast.parse(before)) == physics_nodes(ast.parse(after))
    before_hash,after_hash = [hashlib.sha256(text.encode()).hexdigest() for text in [before,after]]
    for native,global_row in zip(controls,comparison):
        assert native["config"]["transport"] == "native"
        assert global_row["config"]["transport"] == "global-straight"
        for row in [native,global_row]:
            assert row["config"]["photons"] == 100000
            assert row["mode"] == "unified" and row["sun_elevation"] == -6
            np.testing.assert_array_equal(row["wavelengths"],wl)
            assert row["config"]["disk_radial"] == 2 and row["config"]["disk_azimuth"] == 8
        assert native["point_table_sha256"] == global_row["point_table_sha256"]
        np.testing.assert_array_equal(native["deterministic_single"],global_row["deterministic_single"])
        assert native["source_hashes"]["unified_solar_transport.py"] == after_hash
        assert global_row["source_hashes"]["unified_solar_transport.py"] == before_hash
        for name in set(native["source_hashes"]) & set(global_row["source_hashes"]):
            if name != "unified_solar_transport.py":
                assert native["source_hashes"][name] == global_row["source_hashes"][name]
    native_values = np.array([photopic(row["fields"]["split_total"]) for row in controls])
    global_values = np.array([photopic(row["fields"]["split_total"]) for row in comparison])
    differences = native_values-global_values
    paired_se = float(differences.std(ddof=1)/np.sqrt(8))
    control_result = dict(native_new_seeds_mean=float(native_values.mean()),
        native_new_seeds_se=float(native_values.std(ddof=1)/np.sqrt(8)),
        native_minus_global=float(differences.mean()),paired_se=paired_se,
        z=float(differences.mean()/paired_se),difference_values=differences.tolist())
    expected_control = json.loads((directory/"majorant-control-summary.json").read_text())
    for key,value in control_result.items():
        np.testing.assert_allclose(value,expected_control[key],rtol=1e-13,atol=1e-15)
    result["new_native_seed_control"] = control_result
    result["source_binding_guard"] = dict(before_sha256=before_hash,after_sha256=after_hash,
        archived_patch_exactly_adds_constructor_preconditions=True,
        source_numerical_methods_AST_identical=True,
        note="Hash changes between the global and new native batches are guard-only; "
             "the existing driver already bound saved tables to their construction medium")
    result["sampling_scope"] = ("Historical independent-seed 2.244-SE contrast is retained. "
        "New same-seed batch contrast is 0.871 SE without identical photon paths. "
        "Neither contrast certifies rare-tail convergence or excludes every systematic bias.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=DEFAULT_DIRECTORY)
    parser.add_argument("--source-data",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    result = run_audit(args.directory,args.source_data)
    output = args.out or args.directory/"majorant-control/majorant-equivalence-probe.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(independent_seed_comparison=result["independent_seed_comparison"],
                         new_native_seed_control=result["new_native_seed_control"]),indent=2))


if __name__ == "__main__":
    main()
