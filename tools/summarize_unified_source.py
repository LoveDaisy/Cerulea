"""Recompute the frozen phase-three source audit from independent seed records.

Run from the repository root: uv run python -m tools.summarize_unified_source.
No Monte Carlo is rerun. Input hashes, source-table contracts, spectral integrals,
seed coverage and paired-path fingerprints are checked before summary/plot output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from skycolor import presets, spectra
from tools.point_solar_table import PointSolarTable


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "validation/data/unified_solar_source_audit"
MODES = ("geometric", "unified", "legacy")
HEIGHTS = (0.0, 2.4)


def moments(values):
    """Mean and standard error over independent seed-level estimates."""
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1)/np.sqrt(len(values))),
                replicates=values.tolist())


def summarize(folder: Path) -> dict:
    """Validate and reduce the frozen 48-record, 3-source, 2-height experiment."""
    hashes = json.loads((folder/"inputs.sha256.json").read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen audit input changed: {name}")
    rows = []
    for path in sorted((folder/"raw").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    keys = [(r["mode"], r["elevation"], r["seed"]) for r in rows]
    expected = {(mode, height, seed) for mode in MODES for height in HEIGHTS
                for seed in range(200, 208)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Require exactly one record per mode/height/independent seed (48 total)")
    lookup = dict(zip(keys, rows))
    wavelengths = np.arange(380., 781., 5.)
    cmf_y = spectra.cie_xyz(wavelengths)[:, 1]

    def photopic(spectrum):
        return float(683*np.trapezoid(np.asarray(spectrum)*cmf_y, wavelengths))

    # Re-integrate all spectra; do not accept the recorded aggregate as evidence.
    max_integral_difference = 0.0
    for row in rows:
        np.testing.assert_array_equal(row["wavelengths"], wavelengths)
        if (row["config"]["photons"] != 100000 or row["sun_elevation"] != -6
                or row["config"].get("transport", "native") != "native"):
            raise ValueError("Unexpected scene or path budget")
        if row["source_hashes"] != rows[0]["source_hashes"]:
            raise ValueError("Frozen experiment used inconsistent source implementations")
        for field, stored in row["photopic_integrals"].items():
            value = photopic(row["fields"][field])
            max_integral_difference = max(max_integral_difference, abs(value-stored))
            np.testing.assert_allclose(value, stored, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(np.asarray(row["fields"]["split_total"]),
            np.asarray(row["deterministic_single"])+np.asarray(row["fields"]["multiple"]),
            rtol=2e-14, atol=1e-20)
    fingerprints = []
    for height in HEIGHTS:
        for seed in range(200, 208):
            paired = [lookup[mode, height, seed] for mode in MODES]
            signatures = {(r["path_fingerprint"], r["n_nee_calls"], r["n_nee_points"])
                          for r in paired}
            if len(signatures) != 1:
                raise ValueError(f"Source modes changed geometric paths at {height=}, {seed=}")
            fingerprints.append(dict(height_km=height, seed=seed,
                                     fingerprint=paired[0]["path_fingerprint"]))
        if len({lookup["unified", height, seed]["path_fingerprint"]
                for seed in range(200, 208)}) != 8:
            raise ValueError("Repeated seed/path fingerprint")

    # Keep the explicit label and medium signature contract when checking saved tables.
    manifest = json.loads((folder/"tables-manifest.json").read_text())
    base = presets.clear(wavelengths)
    checked_tables = {}
    for mode in MODES:
        for height in HEIGHTS:
            entry = manifest[mode][str(height)]
            relative = Path(entry["path"])
            if relative.is_absolute():
                raise ValueError("Archived table manifest must be relocatable")
            path = folder/relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for seed in range(200, 208):
                if lookup[mode, height, seed]["point_table_sha256"] != digest:
                    raise ValueError("Recorded run did not use this frozen source table")
            if str(relative) not in checked_tables:
                atmo = base if height == 0 else base.at_altitude(height)
                table = PointSolarTable.load(path, atmo, expected_label=entry["label"])
                checked_tables[str(relative)] = dict(label=table.label, sha256=digest,
                    wavelength_count=len(table.wl), height_rows=len(table.heights))

    result = dict(record_count=len(rows), seeds=list(range(200, 208)),
        paths_per_seed_per_wavelength=100000, wavelengths_nm=wavelengths.tolist(),
        sun_elevation_deg=-6., observer_view="zenith", heights_km=list(HEIGHTS),
        uncertainty="one standard error across eight independent seed replicates; "
                    "paired delta method for ratios; excludes deterministic/systematic error",
        transport="native straight geometric MC diffuse paths; continuous refracted solar source",
        checks=dict(sha256_inputs=len(hashes), spectral_integral_max_abs_difference=max_integral_difference,
                    matching_source_fingerprints=fingerprints, tables=checked_tables), modes={})
    for mode in MODES:
        by_height = {}
        totals = []
        for height in HEIGHTS:
            group = [lookup[mode, height, seed] for seed in range(200, 208)]
            for row in group[1:]:
                np.testing.assert_array_equal(row["deterministic_single"], group[0]["deterministic_single"])
            deterministic = photopic(group[0]["deterministic_single"])
            fields = {field: moments([photopic(r["fields"][field]) for r in group])
                      for field in ("single", "multiple", "total", "split_total")}
            closure = fields["single"]
            single_spectra = np.array([r["fields"]["single"] for r in group])
            spectral_se = single_spectra.std(axis=0, ddof=1)/np.sqrt(len(group))
            spectral_z = ((single_spectra.mean(axis=0)-group[0]["deterministic_single"])
                          / spectral_se)
            worst = int(np.argmax(np.abs(spectral_z)))
            by_height[str(height)] = dict(deterministic_single=deterministic, fields=fields,
                single_closure=dict(mc_minus_deterministic=closure["mean"]-deterministic,
                    discrepancy_in_seed_se=(closure["mean"]-deterministic)/closure["se"],
                    spectral_max_abs_seed_se=float(np.abs(spectral_z[worst])),
                    spectral_worst_wavelength_nm=float(wavelengths[worst]),
                    spectral_seed_se=spectral_z.tolist()))
            totals.append(np.array(fields["split_total"]["replicates"]))
        low, high = totals
        ratio = high.mean()/low.mean()
        # A ratio of means differs from a mean of per-seed ratios. Its linear
        # influence retains the covariance induced by the height seed pairing.
        influence = (high-ratio*low)/low.mean()
        result["modes"][mode] = dict(heights=by_height, height_ratio=dict(
            estimate=float(ratio), se=float(influence.std(ddof=1)/np.sqrt(len(low)))),
            physically_closed_source=(mode != "legacy"))
    result["provenance"] = dict(heads=sorted({r["head"] for r in rows}),
        library_diff_sha256=sorted({r["library_diff_sha256"] for r in rows}),
        recorded_source_hashes=rows[0]["source_hashes"],
        note="Raw phase-three records precede the aerosol 550-nm normalization fix. "
             "Every frozen grid includes 550 nm, so that fix leaves these optics bit-identical.")
    return result


def plot_summary(summary: dict, path: Path):
    """Plot source changes and height responses; bars show seed SE only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Geometric source", "Unified source", "Legacy (mixed)"]
    colours = ["#526d82", "#b34b38", "#8c8680"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    x = np.arange(3)
    for height, offset, marker in [(0., -.08, "o"), (2.4, .08, "s")]:
        values = [summary["modes"][mode]["heights"][str(height)]["fields"]["split_total"]
                  for mode in MODES]
        left.errorbar(x+offset, [v["mean"] for v in values], yerr=[v["se"] for v in values],
            fmt=marker, color="#263a48" if height == 0 else "#b34b38", capsize=4,
            label=f"Ground = observer = {height:g} km")
    ratios = [summary["modes"][mode]["height_ratio"] for mode in MODES]
    for i, value in enumerate(ratios):
        right.errorbar(i, value["estimate"], yerr=value["se"], fmt="o", color=colours[i], capsize=4)
    right.axhline(1, color="#a8a8a8", linewidth=1, linestyle="--")
    left.set(ylabel="Zenith luminance (cd m$^{-2}$)", ylim=(.35, .60),
             title="Deterministic SS + MC multiple scattering")
    right.set(ylabel="Luminance at 2.4 km / sea level", ylim=(.89, 1.015),
              title="Height response (paired seed SE)")
    for axis in [left, right]:
        axis.set_xticks(x, labels, rotation=12)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, fontsize=8)
    fig.suptitle("Solar elevation -6 deg, clear AOD550=0.05, 380-780 nm; 8 independent seeds",
                 fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    summary = summarize(args.data)
    (args.data/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot_summary(summary, args.data/"summary.png")
    print(json.dumps(dict(records=summary["record_count"],
        source_modes={mode: summary["modes"][mode]["height_ratio"] for mode in MODES}), indent=2))


if __name__ == "__main__":
    main()
