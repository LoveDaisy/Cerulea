"""Summarise archived altitude runs and redraw the audit figure (no RT run).

``uv run python -m tools.summarize_twilight_altitude``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/data/twilight_altitude_audit"


def read(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]


def summarise(rows, keys):
    samples = {height: {row["seed"]: row for row in rows if row["elevation"] == height}
               for height in (0.0, 2.4)}
    for height in samples:
        assert len(samples[height]) == sum(row["elevation"] == height for row in rows), \
            "Duplicate seed records: use a fresh output file for a rerun"
    seeds = sorted(samples[0.0])
    assert seeds == sorted(samples[2.4]), "Both heights need the same independent seeds"
    n = len(seeds)
    assert n >= 2
    result = {"independent_seeds": seeds, "fields": {}}
    for key in keys:
        lo = np.array([samples[0.0][s]["photopic_integrals"][key] for s in seeds])
        hi = np.array([samples[2.4][s]["photopic_integrals"][key] for s in seeds])
        ratio = float(hi.mean() / lo.mean())
        # Paired delta-method influence, without assuming common-path coupling.
        influence = (hi - ratio * lo) / lo.mean()
        se = float(np.std(influence, ddof=1) / np.sqrt(n))
        halfwidth = float(t.ppf(0.975, n - 1) * se)
        result["fields"][key] = dict(
            sea_mean=float(lo.mean()), high_mean=float(hi.mean()),
            sea_se=float(lo.std(ddof=1) / np.sqrt(n)),
            high_se=float(hi.std(ddof=1) / np.sqrt(n)),
            ratio_of_means=ratio, ratio_se=se,
            empirical_t95_interval=[ratio - halfwidth, ratio + halfwidth],
            sea_replicates=lo.tolist(), high_replicates=hi.tolist())
    return result


def main():
    summary = dict(
        units="cd m^-2, photopic, 380-780 nm at 5 nm spacing",
        standard_scene=dict(sun_elevation_deg=-6, view_zenith_deg=0,
                            ground_elevation_km=[0.0, 2.4], aod550=0.05,
                            ozone_du=300, ground_albedo=0.1),
        uncertainty=("Across independent seed replicates; paired delta method for "
                     "ratio of means. Empirical t intervals are not a proof of tail "
                     "convergence and exclude model/site/photometric errors."),
        direct=summarise(read("full-direct"),
            ["single", "multiple", "total", "hybrid_total", "geometric_split_total"]),
    )
    engine_path = DATA / "engine-low.jsonl"
    if engine_path.exists():
        summary["engine_low"] = summarise(read("engine-low"), ["total", "multiple"])
    before, after = read("baseline-deterministic"), read("corrected-deterministic")
    summary["deterministic"] = dict(before=before[1]["photopic_integrals"],
                                     after=after[1]["photopic_integrals"],
                                     sea=after[0]["photopic_integrals"])
    summary["deterministic"]["ratio_before"] = (
        before[1]["photopic_integrals"]["total"] / before[0]["photopic_integrals"]["total"])
    summary["deterministic"]["ratio_after"] = (
        after[1]["photopic_integrals"]["total"] / after[0]["photopic_integrals"]["total"])
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    ax = axes[0]
    records = [after[0], before[1], after[1]]
    ss = np.array([r["photopic_integrals"]["single"] for r in records])
    ms = np.array([r["photopic_integrals"]["multiple"] for r in records])
    x = np.arange(3)
    ax.bar(x, ss, color="#4679a2", label="First order")
    ax.bar(x, ms, bottom=ss, color="#db9b49", label="SOS higher orders")
    for j, total in enumerate(ss + ms):
        ax.text(j, total + 0.008, f"{total:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, ["Sea level", "2.4 km\nbefore", "2.4 km\ncorrected"])
    ax.set_title("Opaque-ground correction, deterministic")
    ax.set_ylabel("Zenith luminance [cd m$^{-2}$]")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    for j, (key, label, color) in enumerate([
        ("geometric_split_total", "Geometric split", "#788e72"),
        ("hybrid_total", "Refracted first + geometric higher", "#4679a2"),
    ]):
        field = summary["direct"]["fields"][key]
        ax.errorbar(np.array([0, 1]) + (j - 0.5) * 0.08,
                    [field["sea_mean"], field["high_mean"]],
                    yerr=[field["sea_se"], field["high_se"]],
                    fmt="o-", capsize=4, color=color,
                    label=f"{label}\nratio {field['ratio_of_means']:.3f}")
    ax.set_xticks([0, 1], ["Sea level", "2.4 km"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.3, 0.65)
    ax.set_title("Direct MC, 81 wavelengths, 8 seeds")
    ax.set_ylabel("Zenith luminance [cd m$^{-2}$]")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures/14_twilight_altitude_audit.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v["ratio_of_means"] for k, v in
                      summary["direct"]["fields"].items()}, indent=2))


if __name__ == "__main__":
    main()
