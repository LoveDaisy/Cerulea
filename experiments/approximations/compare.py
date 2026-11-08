"""Two controlled skies through the public production entry; not a new solver.

Run from the repository with ``python -m experiments.approximations.compare``.
The NPZ contains ordinary arrays and literal text, never pickled objects.
"""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from experiments.common import save
from skycolor import color, presets, render
from skycolor.constants import WL_COARSE
from skycolor.rt.result import SkyResult
from skycolor.rt.solver import MCConfig

THEME = Path(__file__).resolve().parent
DISPLAY = dict(camera="perspective", size=(640, 320), hfov_deg=100.0,
               pitch_deg=14.0, yaw_deg=0.0, exposure=0.03, tonemap="aces",
               adaptation=None, sun_disk=False, ground=False)
SEEDS = (7, 17, 27)
DIRECTIONS = (("zenith", 0.0, 0.0), ("45 sun", 45.0, 0.0),
              ("45 anti", 45.0, 180.0), ("near horizon sun", 86.25, 0.0),
              ("near horizon anti", 86.25, 180.0))


def _plain(value):
    """Keep backend diagnostics readable without an object-array/pickle format."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def quantities(radiance, wl):
    """Un-tonemapped XYZ, xy and photopic luminance [cd m^-2]."""
    xyz = color.spectrum_to_xyz(radiance, wl)
    return xyz, color.chromaticity(xyz), 683.0 * xyz[..., 1]


def compute(n_zenith=25, n_azimuth=37, wl=WL_COARSE, seeds=SEEDS):
    """Fixed 30/3 degree experiment. Every branch really calls solve_dome.

    Defaults retain the production single/SOS discretisation. MC is deliberately
    coarse and budget-limited; its repeated calls demonstrate variability only.
    """
    atmo = presets.clear(wl, aod=0.05, ozone_du=300.0)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    data = dict(wl=np.asarray(wl), runs=[], version=version,
                display=repr(DISPLAY),
                medium=repr(dict(preset="clear", aod550=0.05, ozone_du=300.0,
                                 no2_column=atmo.no2_column, refraction=atmo.refraction,
                                 z_bounds_km=atmo.z_bounds.tolist(),
                                 albedo=atmo.ground_albedo.tolist(), observer_km=0.0,
                                 aerosol="HinterlandAerosol; exponential H=1.5 km")))
    for elevation in (30, 3):
        requests = [(name, {}) for name in
                    ("plane-sos", "hybrid-sos", "spherical-single")]
        if elevation == 30:
            requests += [("spherical-mc", dict(mc_config=MCConfig(
                quality="low", seed=seed,
                grid=dict(n_zen=3, n_az=4, n_anchor=5, n_photons=200000))))
                         for seed in seeds]
        for approximation, kwargs in requests:
            seed = kwargs["mc_config"].seed if kwargs else None
            key = f"e{elevation}_{approximation}" + (f"_s{seed}" if seed is not None else "")
            print(f"solve {key} (first call cold, subsequent shared optics warm)", flush=True)
            start = perf_counter()
            pres = render.solve_dome(atmo, elevation, 0.0, n_zenith=n_zenith,
                                     n_azimuth=n_azimuth, approximation=approximation,
                                     **kwargs)
            elapsed = perf_counter() - start
            res = pres.result
            data["runs"].append(key)
            data[f"{key}/seconds"] = elapsed
            data[f"{key}/elevation"] = elevation
            data[f"{key}/zenith"] = pres.zenith
            data[f"{key}/azimuth_rel"] = pres.azimuth_rel
            data[f"{key}/view_zenith"] = np.repeat(pres.zenith, pres.azimuth_rel.size)
            data[f"{key}/view_azimuth"] = np.tile(pres.azimuth_rel, pres.zenith.size)
            data[f"{key}/summary"] = render.format_result_summary(res)
            for name in ("options", "effective", "diagnostics", "approximations"):
                data[f"{key}/{name}"] = repr(_plain(getattr(res, name)))
            for name, rad in {"total": res.radiance, **res.components}.items():
                if rad is None:
                    continue
                data[f"{key}/{name}"] = rad
                for quantity, values in zip(("xyz", "xy", "Lv"), quantities(rad, pres.wl)):
                    data[f"{key}/{name}_{quantity}"] = values
            if res.stderr is not None:
                data[f"{key}/stderr"] = res.stderr
            print(f"{elapsed:.3f} s\n{data[f'{key}/summary']}", flush=True)
    data["runs"] = np.asarray(data["runs"])
    check(data)
    return data


def check(data):
    """Assert same-path closure, not accuracy against nature or an MC reference."""
    for key in data["runs"]:
        total = data[f"{key}/single"].copy()
        if f"{key}/multiple" in data:
            total += data[f"{key}/multiple"]
        np.testing.assert_array_equal(data[f"{key}/total"], total)
        for part in ("single", "multiple", "total"):
            if f"{key}/{part}" not in data:
                continue
            for name, value in zip(("xyz", "xy", "Lv"),
                                   quantities(data[f"{key}/{part}"], data["wl"])):
                np.testing.assert_allclose(data[f"{key}/{part}_{name}"], value,
                                           rtol=1e-12, atol=0)
    for elevation in (30, 3):
        plane, hybrid, single = [f"e{elevation}_{name}" for name in
                                 ("plane-sos", "hybrid-sos", "spherical-single")]
        np.testing.assert_array_equal(data[f"{plane}/multiple"], data[f"{hybrid}/multiple"])
        np.testing.assert_array_equal(data[f"{single}/total"], data[f"{hybrid}/single"])


def bind(data, key, part="total"):
    """Bind saved spectra with the production carrier, with no new solve."""
    res = SkyResult(radiance=data[f"{key}/{part}"], wl=data["wl"], method="clear")
    return render.presentation_from_result(res, data[f"{key}/zenith"],
                                           data[f"{key}/azimuth_rel"],
                                           float(data[f"{key}/elevation"]), 0.0)


def write_tables(data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "representative.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "component", "direction", "zenith_deg", "azimuth_rel_deg",
                         "Lv_cd_m-2", "x", "y"))
        for key in data["runs"]:
            zen, az = data[f"{key}/zenith"], data[f"{key}/azimuth_rel"]
            for name, z, a in DIRECTIONS:
                i, j = np.argmin(abs(zen-z)), np.argmin(abs(az-a))
                row = i * len(az) + j
                for part in ("single", "multiple", "total"):
                    if f"{key}/{part}" not in data:
                        continue
                    writer.writerow((key, part, name, zen[i], az[j],
                                     data[f"{key}/{part}_Lv"][row],
                                     *data[f"{key}/{part}_xy"][row]))
    lines = ["# Actual production calls", "", f"Production version: `{data['version']}`",
             "", "Order: plane, hybrid, single, MC seeds 7/17/27 (30°), then 3°.",
             "First call cold; later calls share warmed optical caches. No fair speed ranking.",
             "MC: qualitative / budget-limited; concentration diagnostics not provided.",
             "stderr includes smoother bias floor, not total physical error or Lv confidence.",
             "", f"Medium: `{data['medium']}`", "", f"Display: `{data['display']}`"]
    for key in data["runs"]:
        lines += ["", f"## {key}", "", f"Wall time: {float(data[f'{key}/seconds']):.6f} s",
                  "", "```text", str(data[f"{key}/summary"]), "```"]
        for name in ("options", "effective", "diagnostics", "approximations"):
            lines += ["", f"{name}: `{data[f'{key}/{name}']}`"]
    (directory / "representative.md").write_text("\n".join(lines) + "\n")


def draw(data, destination):
    """All images use identical final render_result arguments, no auto exposure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.7), layout="constrained")
    for row, elevation in enumerate((30, 3)):
        for col, name in enumerate(("spherical-single", "plane-sos", "hybrid-sos")):
            rgb = render.render_result(bind(data, f"e{elevation}_{name}"), **DISPLAY)
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"sun {elevation}° · {name}", fontsize=11)
            axes[row, col].axis("off")
            if name == "hybrid-sos":
                render.save_png(rgb, destination / f"sky_{elevation:02d}.png")
    fig.suptitle("Same atmosphere, camera and exposure (0.03) · no sun marker / ground")
    save(fig, destination / "approximations.png")
    plt.close(fig)

    # Spectral shape and magnitude need lines, not a colour-only surrogate.
    # Three fixed identities; dashes + labels also work in monochrome.
    palette = ("#2a78d6", "#eb6834", "#1baf7a")
    with plt.rc_context({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        curves = (("plane-sos", "single", "plane single", "-"),
                  ("hybrid-sos", "single", "spherical single", "--"),
                  ("hybrid-sos", "total", "hybrid total", ":"))
        for row, elevation in enumerate((30, 3)):
            for col, target in enumerate((0.0, 86.25)):
                ax = axes[row, col]
                for (name, part, label, style), ink in zip(curves, palette):
                    key = f"e{elevation}_{name}"
                    zen, az = data[f"{key}/zenith"], data[f"{key}/azimuth_rel"]
                    i = np.argmin(abs(zen-target)) * len(az)
                    ax.plot(data["wl"], data[f"{key}/{part}"][i], style,
                            color=ink, linewidth=2, label=label)
                actual = zen[np.argmin(abs(zen-target))]
                ax.set_title(f"Sun {elevation}° · view zenith {actual:g}° · sun-side")
                ax.set_xlabel("Wavelength [nm]")
                ax.set_ylabel("Radiance [W m$^{-2}$ sr$^{-1}$ nm$^{-1}$]")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.3, color="#c3c2b7")
                ax.legend(fontsize=9)
        fig.suptitle("Added orders and changed observation propagation are different comparisons")
        save(fig, destination / "spectra.png")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-figures", action="store_true",
                        help="explicitly write selected production figures to root figures/")
    parser.add_argument("--from-npz", type=Path, help="redraw without solving (saved local arrays)")
    args = parser.parse_args(argv)
    if args.from_npz:
        with np.load(args.from_npz, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        # Display metadata belongs to the saved experiment, not current defaults.
        if ast.literal_eval(str(data["display"])) != DISPLAY:
            raise ValueError("saved display differs from this example; do not silently restyle")
        check(data)
    else:
        data = compute()
    out = THEME / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not args.from_npz:
        np.savez_compressed(out / "representative.npz", **data)
    write_tables(data, out)
    destination = THEME.parents[1] / "figures" if args.publish_figures else THEME / "figures"
    draw(data, destination)


if __name__ == "__main__":
    main()
