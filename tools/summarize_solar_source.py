"""Recompute paired solar-source audit statistics and its figure, without RT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/data/solar_source_audit"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def ratio(a, b):
    """Ratio mean(b)/mean(a) and paired delta-method empirical standard error."""
    a, b = np.asarray(a), np.asarray(b)
    r = float(b.mean() / a.mean())
    se = float(np.std((b - r * a) / a.mean(), ddof=1) / np.sqrt(a.size))
    return dict(value=r, se=se)


def main():
    rows = load("full-paired.jsonl")
    keyed = {(r["mode"], r["elevation"], r["seed"]): r for r in rows}
    assert len(keyed) == len(rows) == 32
    seeds = list(range(100, 108))
    summary = dict(standard_scene="clear, -6 deg zenith, 380-780 nm / 5 nm, ground 0/2.4 km",
                   uncertainty="Across 8 independent seeds; statistical error only",
                   heights={}, altitude_ratios={})
    for height in [0, 2.4]:
        group = {}
        for mode in ["geometric", "attenuation"]:
            r = [keyed[(mode, height, seed)] for seed in seeds]
            group[mode] = {key: np.array([x["photopic_integrals"][key] for x in r])
                           for key in ["single", "multiple", "split_total", "total"]}
        assert all(keyed[("geometric", height, s)]["path_fingerprint"] ==
                   keyed[("attenuation", height, s)]["path_fingerprint"] for s in seeds)
        before, after = group["geometric"]["multiple"], group["attenuation"]["multiple"]
        diff = after - before
        summary["heights"][str(height)] = dict(
            means={m: {k: float(v.mean()) for k, v in fields.items()}
                   for m, fields in group.items()},
            standard_errors={m: {k: float(v.std(ddof=1) / np.sqrt(len(v)))
                                 for k, v in fields.items()} for m, fields in group.items()},
            multiple_ratio=ratio(before, after),
            multiple_paired_difference=float(diff.mean()),
            multiple_difference_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
            all_paths_identical=True)
    for mode in ["geometric", "attenuation"]:
        lo = [keyed[(mode, 0, s)]["photopic_integrals"]["split_total"] for s in seeds]
        hi = [keyed[(mode, 2.4, s)]["photopic_integrals"]["split_total"] for s in seeds]
        summary["altitude_ratios"][mode] = ratio(lo, hi)
    old_r, new_r = summary["altitude_ratios"].values()
    # Influence for the difference of two ratios on the same replicate paths.
    influences = []
    for mode, r in zip(["geometric", "attenuation"], [old_r, new_r]):
        lo = np.array([keyed[(mode, 0, s)]["photopic_integrals"]["split_total"] for s in seeds])
        hi = np.array([keyed[(mode, 2.4, s)]["photopic_integrals"]["split_total"] for s in seeds])
        influences.append((hi - r["value"] * lo) / lo.mean())
    summary["altitude_ratio_change"] = dict(
        value=new_r["value"] - old_r["value"],
        se=float(np.std(influences[1] - influences[0], ddof=1) / np.sqrt(8)))
    single = load("single-radial-refined.jsonl")
    summary["single_refined"] = [dict(mode=r["mode"], elevation=r["elevation"],
        photopic=r["photopic"], n_gauss=r["n_gauss"], n_quad=r["n_quad"]) for r in single]
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    ax = axes[0]
    x, labels, ss, ms = [], [], [], []
    for j, height in enumerate([0, 2.4]):
        for i, mode in enumerate(["geometric", "attenuation"]):
            vals = summary["heights"][str(height)]["means"][mode]
            ms.append(vals["multiple"])
            ss.append(vals["split_total"] - vals["multiple"])
            x.append(2 * j + i)
            labels.append(f"{'Sea' if height == 0 else '2.4 km'}\n{'Baseline' if i == 0 else 'T-only'}")
    ax.bar(x, ss, color="#4679a2", label="Fixed refracted first order")
    ax.bar(x, ms, bottom=ss, color="#db9b49", label="Geometric-path MC higher orders")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.8)
    ax.set_title("MC solar attenuation/shadow ablation")
    ax.set_ylabel("Zenith luminance [cd m$^{-2}$]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    modes = ["packaged", "radial_attenuation", "radial_beam", "vertical_disk_17"]
    values = {r["mode"]: r["photopic"] for r in single if r["elevation"] == 0}
    ax.bar(range(4), [values[m] for m in modes],
           color=["#8c969f", "#4679a2", "#788e72", "#9e8767"])
    for j, mode in enumerate(modes):
        ax.text(j, values[mode] + 0.008, f"{values[mode]:.3f}", ha="center")
    ax.set_xticks(range(4), ["Packaged\nangles", "Radial\nangles", "+ Point-beam\nJacobian", "+ Vertical-disk\napprox."])
    ax.set_ylim(0, 0.46)
    ax.set_title("Single scattering: solar-source components")
    ax.set_ylabel("Sea-level luminance [cd m$^{-2}$]")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures/15_solar_source_audit.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary["altitude_ratios"], indent=2))


if __name__ == "__main__":
    main()
