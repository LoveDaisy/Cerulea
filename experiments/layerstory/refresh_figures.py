"""Redraw the layer-story figures from the committed archive -- zero solve.

The committed ``data/layer_story.npz`` is the single numeric source: this
entry only reads it back (``allow_pickle=False`` ordinary arrays), reuses the
production display primitives (``layer_story.bind`` / ``render.sample_dome``
/ ``color.render_rgb`` / ``render.render_result``) and never re-enters the
solver.  Figure definitions live in one registry below -- it is the
authoritative source for both generation and the reproduce manifest, so no
curve/camera definition is duplicated between scripts.

``--audit-only`` writes the coverage report: per-run grid shape/endpoints,
row-count and endpoint verification (no clamp patching -- a missing endpoint
disqualifies the view instead of being padded), ground-material availability,
sun metadata readback and per-figure feasibility against the registry.
``--figure <id>`` (with optional ``--publish-figures`` for the root
``figures/`` copies) renders the registered figures; a plain run writes only
theme-local ``out/`` outputs and never touches root or historical files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.layerstory import layer_story as ls

THEME = Path(__file__).resolve().parent
DEFAULT_NPZ = THEME / "data" / "layer_story.npz"
OUT = THEME / "out"

#: relative-azimuth node spacing tolerance when deciding "exact node"
NODE_TOL = 1e-9


def load(path: Path = DEFAULT_NPZ) -> dict:
    """Read the archive with the same guards as the theme's --from-npz path."""
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if ast.literal_eval(str(data["display_groups"])) != ls.DISPLAY_GROUPS:
        raise ValueError("saved display groups differ from this example; "
                         "do not silently restyle")
    ls.check(data)
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    """The *drawing* code version -- never presented as the solve version."""
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   text=True).strip()


# ---------------------------------------------------------------------------
# figure registry -- the single authoritative definition of every new figure
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Curve:
    """One spectrum curve: archive row at (zenith, relative azimuth)."""
    key: str
    part: str
    zenith: float
    azimuth: float
    label: str
    style: str


@dataclass(frozen=True)
class SpectrumPanelDef:
    curves: tuple[Curve, ...]
    loglog: bool
    title: str


@dataclass(frozen=True)
class ViewPanelDef:
    """One rendered view of an archive run: archived perspective camera or a
    zenith-centred equidistant 180-degree fisheye resampled from the same
    dome (display resampling, not a new computation)."""
    key: str
    part: str
    camera: str          # "rect" | "fisheye"
    ground: bool
    title: str


@dataclass(frozen=True)
class FigureDef:
    fig_id: str
    publish_name: str    # root figures/ name used by --publish-figures
    layout: str          # "spectrum_pair" | "spectrum" | "triple_pair" |
                         # "rect_pair" | "single_rect"
    spectrum: SpectrumPanelDef | None
    views: tuple[ViewPanelDef, ...]
    consumers: tuple[str, ...]   # the doc positions this figure serves
    note: str


_FISH = "fisheye (equidistant, 180 deg, zenith-centred)"
_RECT = "perspective (archived camera)"

FIGURES: dict[str, FigureDef] = {
    "rayleigh": FigureDef(
        "rayleigh", "layer_story_rayleigh_v2.png", "spectrum_pair",
        SpectrumPanelDef(
            (Curve("l1_rayleigh_30", "total", 0.0, 0.0, "zenith", "-"),
             Curve("l1_rayleigh_30", "total", 86.25, 0.0,
                   "horizon sun-side", "--")),
            loglog=True,
            title="L1 · Rayleigh spectrum (log-log): zenith vs near-horizon"),
        (ViewPanelDef("l1_rayleigh_30", "total", "rect", False,
                      f"L1 · Rayleigh only — {_RECT}"),
         ViewPanelDef("l1_rayleigh_30", "total", "fisheye", False,
                      f"L1 · Rayleigh only — {_FISH}")),
        ("docs/physics/LAYER_STORY.md §L1", "docs/physics/MAINLINE.zh.md 图2"),
        "the spectrum explains the colour; the dome pair locates zenith/horizon"),
    "aerosol": FigureDef(
        "aerosol", "layer_story_aerosol_v2.png", "spectrum",
        SpectrumPanelDef(
            (Curve("l1_rayleigh_30", "total", 86.25, 0.0, "L1 clean", "-"),
             Curve("l2_aerosol_30", "total", 86.25, 0.0, "L2 + aerosol", "-")),
            loglog=False,
            title="L2 · aerosol whitening (near horizon, sun-side)"),
        (),
        ("docs/physics/LAYER_STORY.md §L2", "docs/physics/MAINLINE.zh.md 图3"),
        "same-scene spectral contrast; geometry is established by `rayleigh`"),
    "ozone": FigureDef(
        "ozone", "layer_story_ozone_v2.png", "spectrum",
        SpectrumPanelDef(
            (Curve("l3_bridge_3", "total", 0.0, 0.0, "no O$_3$", "-"),
             Curve("l3_o3_3", "total", 0.0, 0.0, "+ O$_3$/NO$_2$", "-")),
            loglog=False,
            title="L3 · Chappuis band (zenith, sun 3°)"),
        (),
        ("docs/physics/LAYER_STORY.md §L3", "docs/physics/MAINLINE.zh.md 图4"),
        "zenith rows only -- the archive has no L3 dome, so no sky view here"),
    "multiple": FigureDef(
        "multiple", "layer_story_multiple_v2.png", "spectrum",
        SpectrumPanelDef(
            (Curve("l4_bridge_30", "total", 45.0, 180.0, "single only", "-"),
             Curve("l4_multiple_30", "total", 45.0, 180.0, "+ multiple", "-")),
            loglog=False,
            title="L4 · multiple fills the anti-solar sky (45° anti-solar, "
                  "rel. azimuth 180°)"),
        (),
        ("docs/physics/LAYER_STORY.md §L4", "docs/physics/MAINLINE.zh.md 图5"),
        "anti-solar means relative azimuth 180 deg, not a 90 deg scattering angle"),
    "twilight": FigureDef(
        "twilight", "layer_story_twilight_v2.png", "triple_pair",
        None,
        (ViewPanelDef("l5_single_-4", "total", "rect", False,
                      "L5 · spherical single — " + _RECT),
         ViewPanelDef("l5_mc_-4", "total", "rect", False,
                      "L5 · + MC higher orders — " + _RECT),
         ViewPanelDef("l5_plane_-4", "total", "rect", False,
                      "L5 · plane-parallel world — " + _RECT),
         ViewPanelDef("l5_single_-4", "total", "fisheye", False,
                      "spherical single — " + _FISH),
         ViewPanelDef("l5_mc_-4", "total", "fisheye", False,
                      "+ MC higher orders — " + _FISH),
         ViewPanelDef("l5_plane_-4", "total", "fisheye", False,
                      "plane-parallel world — " + _FISH)),
        ("docs/physics/LAYER_STORY.md §L5", "docs/physics/MAINLINE.zh.md 图6"),
        "three -4 deg states, locked twilight exposure 10.0; MC panel is "
        "single-seed / coarse-grid qualitative (no new convergence evidence)"),
    "cloud": FigureDef(
        "cloud", "layer_story_cloud_v2.png", "rect_pair",
        None,
        (ViewPanelDef("l6_clear_30", "total", "rect", True,
                      "L6 · clear (same contract) — " + _RECT),
         ViewPanelDef("l6_cloud_30", "total", "rect", True,
                      "L6 · + stratus deck — " + _RECT)),
        ("docs/physics/LAYER_STORY.md §L6", "docs/physics/MAINLINE.zh.md 图7"),
        "contract-matched pair, both with the solve-bound deck-lit floor, "
        "locked cloud exposure 0.03; no fisheye (no ground in a 180 deg dome)"),
    "pipeline-spectrum": FigureDef(
        "pipeline-spectrum", "layer_story_pipeline_spectrum_v2.png", "spectrum",
        SpectrumPanelDef(
            (Curve("l6_clear_30", "total", 0.0, 0.0, "zenith", "-"),
             Curve("l6_clear_30", "total", 81.0, 0.0,
                   "near-horizon sun-side (81°)", "--")),
            loglog=False,
            title="hybrid-sos input spectra @ sun 30° (the two directions "
                  "imaged beside)"),
        (),
        ("docs/physics/MAINLINE.zh.md 图8",),
        "only the two directions of the SAME run that figure 9 images"),
    "pipeline-view": FigureDef(
        "pipeline-view", "layer_story_pipeline_view_v2.png", "single_rect",
        None,
        (ViewPanelDef("l6_clear_30", "total", "rect", False,
                      "hybrid-sos @ sun 30° — " + _RECT + ", no ground"),),
        ("docs/physics/MAINLINE.zh.md 图9",),
        "same run/component as pipeline-spectrum (single source pair); "
        "ground=False like the day display convention"),
}


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _run_options(data, key) -> dict:
    return ast.literal_eval(str(data[f"{key}/options"]))


def run_capability(data, key) -> dict:
    """What this archived run can and cannot support, from its own fields.

    Endpoint coverage is asserted, not clipped: a dome grid that misses a
    0/90 or 0/180 endpoint disqualifies camera views outright (the display
    binding's own contract) instead of being padded.
    """
    zen = np.asarray(data[f"{key}/zenith"], dtype=float)
    az = np.asarray(data[f"{key}/azimuth_rel"], dtype=float)
    total = np.asarray(data[f"{key}/total"], dtype=float)
    wl = np.asarray(data["wl"], dtype=float)
    parts = sorted(p for p in ("single", "multiple", "clear", "cloud",
                               "transmittance")
                   if f"{key}/{p}" in data)
    opts = _run_options(data, key)
    tol = 1e-6
    endpoints_ok = bool(
        zen.size >= 2 and az.size >= 2
        and np.all(np.diff(zen) > 0) and np.all(np.diff(az) > 0)
        and abs(zen[0]) <= tol and abs(zen[-1] - 90.0) <= tol
        and abs(az[0]) <= tol and abs(az[-1] - 180.0) <= tol)
    rows_match = bool(total.shape == (zen.size * az.size, wl.size))
    has_display = f"{key}/display" in data
    ground_supported = bool(data[f"{key}/ground_supported"]) \
        if f"{key}/ground_supported" in data else False
    ground_material = all(f"{key}/{m}" in data
                          for m in ("ground_albedo",
                                    "ground_direct_irradiance"))
    dome_views = bool(endpoints_ok and rows_match)
    return dict(
        zenith_nodes=int(zen.size), azimuth_nodes=int(az.size),
        zenith_span=[float(zen[0]), float(zen[-1])],
        azimuth_span=[float(az[0]), float(az[-1])],
        total_shape=list(total.shape), rows_match=rows_match,
        endpoints_ok=endpoints_ok,
        dome_views=dome_views,
        components=parts,
        has_display_record=has_display,
        ground_supported=ground_supported,
        ground_material=ground_material,
        ground_display=bool(dome_views and ground_supported
                            and ground_material),
        elevation_deg=float(data[f"{key}/elevation"]),
        sun_zenith_deg=opts.get("sun_zenith_deg"),
        sun_azimuth_deg=opts.get("sun_azimuth_deg"),
        refraction=opts.get("refraction"),
        method_recorded=f"{key}/method" in data,
    )


def curve_actual_node(data, curve: Curve) -> tuple[float, float, int]:
    """The archive node a requested curve direction lands on (and its row)."""
    row = ls.row_index(data, curve.key, curve.zenith, curve.azimuth)
    az = np.asarray(data[f"{curve.key}/azimuth_rel"], dtype=float)
    i, j = divmod(row, az.size)
    zen = np.asarray(data[f"{curve.key}/zenith"], dtype=float)
    return float(zen[i]), float(az[j]), row


def figure_feasibility(data, fig: FigureDef) -> dict:
    """Per-figure feasibility against the archive's real capabilities."""
    caps = {k: run_capability(data, k) for k in
            {c.key for c in (fig.spectrum.curves if fig.spectrum else ())}
            | {v.key for v in fig.views}}
    curves = []
    for c in (fig.spectrum.curves if fig.spectrum else ()):
        z, a, row = curve_actual_node(data, c)
        curves.append(dict(key=c.key, part=c.part, requested=[c.zenith,
                                                             c.azimuth],
                           actual_node=[z, a], row=int(row),
                           exact_node=bool(abs(z - c.zenith) <= NODE_TOL
                                           and abs(a - c.azimuth) <= NODE_TOL)))
    views = [dict(key=v.key, camera=v.camera, ground=v.ground,
                 dome_views=caps[v.key]["dome_views"],
                 ground_display=caps[v.key]["ground_display"],
                 feasible=bool(caps[v.key]["dome_views"]
                               and (not v.ground
                                    or caps[v.key]["ground_display"]))
                 and (caps[v.key]["has_display_record"] or v.camera == "fisheye"))
             for v in fig.views]
    spectrum_feasible = None if fig.spectrum is None else all(
        f"{c.key}/{c.part}" in data for c in fig.spectrum.curves)
    return dict(
        fig_id=fig.fig_id, publish_name=fig.publish_name,
        consumers=list(fig.consumers), note=fig.note,
        curves=curves, views=views,
        spectrum_feasible=spectrum_feasible,
        feasible=bool(spectrum_feasible is not False
                      and all(v["feasible"] for v in views)),
    )


def audit(data, npz_path: Path) -> dict:
    """The Step-0 coverage audit: what the archive really holds."""
    runs = [str(r) for r in data["runs"]]
    caps = {k: run_capability(data, k) for k in runs}
    figures = {k: figure_feasibility(data, f) for k, f in FIGURES.items()}
    return dict(
        input_file=str(npz_path),
        input_sha256=sha256_file(npz_path),
        input_bytes=npz_path.stat().st_size,
        solve_version=str(data["version"]),
        quick=bool(data["quick"]),
        wl_nodes=int(np.asarray(data["wl"]).size),
        n_runs=len(runs),
        runs=caps,
        figures=figures,
        closure_check="layer_story.check(data) passed (per-solve additivity "
                      "+ XYZ/xy/Lv identities, bridge identities)",
        zero_solve="audit consumed np.load + layer_story.check only; no "
                   "solver, solar-source or solve_dome entry was reachable "
                   "from this path",
        sun_metadata_note="options carry the pre-refraction input direction "
                          "(sun_zenith_deg/sun_azimuth_deg); apparent disk "
                          "geometry is NOT archived and is not reconstructed",
    )


def _capability_line(cap: dict) -> str:
    dome = "full half dome" if cap["dome_views"] else \
        (f"NO dome views ({cap['zenith_nodes']}x{cap['azimuth_nodes']})"
         if not cap["endpoints_ok"] or not cap["rows_match"]
         else "no dome (single row)")
    ground = "ground display" if cap["ground_display"] else "no ground display"
    return f"{dome}, {ground}"


def render_audit_markdown(report: dict) -> str:
    lines = [
        "# layerstory archive coverage audit (refresh_figures)",
        "",
        f"- input: `{report['input_file']}`",
        f"- sha256: `{report['input_sha256']}` ({report['input_bytes']} bytes)",
        f"- solve version: `{report['solve_version']}` (compute-time HEAD)",
        f"- drawing-code version: `{git_head()}` (redraw only -- NOT the "
        "solve version)",
        f"- runs: {report['n_runs']}, wavelength nodes: {report['wl_nodes']}, "
        f"quick: {report['quick']}",
        f"- closure: {report['closure_check']}",
        f"- zero solve: {report['zero_solve']}",
        f"- sun metadata: {report['sun_metadata_note']}",
        "",
        "## Per-run coverage",
        "",
        "| run | grid (zen x az) | span | total shape | capabilities | "
        "components | display record |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, cap in report["runs"].items():
        lines.append(
            f"| `{key}` | {cap['zenith_nodes']} x {cap['azimuth_nodes']} | "
            f"zen {cap['zenith_span'][0]}..{cap['zenith_span'][1]}, "
            f"az {cap['azimuth_span'][0]}..{cap['azimuth_span'][1]} | "
            f"{cap['total_shape']} | {_capability_line(cap)} | "
            f"{', '.join(cap['components']) or '-'} | "
            f"{'yes' if cap['has_display_record'] else 'no'} |")
    lines += [
        "",
        "L3 runs are zenith-row solves: they support spectrum curves only, "
        "never a dome view.  CSV/Lv tables stay sampled-direction summaries; "
        "the dome authority is the NPZ arrays.",
        "",
        "## Figure feasibility (registry vs archive)",
        "",
        "| figure | feasible | curves (requested -> node) | views | consumers |",
        "|---|---|---|---|---|",
    ]
    for fig in report["figures"].values():
        curve_txt = "; ".join(
            f"{c['key']}/{c['part']} ({c['requested'][0]},{c['requested'][1]})"
            f" -> ({c['actual_node'][0]},{c['actual_node'][1]})"
            + ("" if c["exact_node"] else " SNAPPED")
            for c in fig["curves"]) or "-"
        view_txt = "; ".join(
            f"{v['key']} {v['camera']} ground={v['ground']}"
            f" [{'ok' if v['feasible'] else 'REFUSED'}]"
            for v in fig["views"]) or "-"
        lines.append(f"| `{fig['fig_id']}` | {'yes' if fig['feasible'] else 'NO'}"
                     f" | {curve_txt} | {view_txt} | "
                     f"{', '.join(fig['consumers'])} |")
    lines.append("")
    return "\n".join(lines)


def run_audit(npz_path: Path) -> dict:
    data = load(npz_path)
    report = audit(data, npz_path)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_coverage.md").write_text(render_audit_markdown(report))
    (OUT / "audit_coverage.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit and redraw the layer-story figures from the "
                    "committed archive (zero solve)")
    parser.add_argument("--from-npz", type=Path, default=DEFAULT_NPZ,
                        help="archive to read (default: committed data)")
    parser.add_argument("--audit-only", action="store_true",
                        help="write the coverage report and exit")
    args = parser.parse_args(argv)
    report = run_audit(args.from_npz)
    infeasible = [k for k, f in report["figures"].items() if not f["feasible"]]
    print(f"audit -> {OUT / 'audit_coverage.md'}")
    for key, cap in report["runs"].items():
        print(f"  {key}: {_capability_line(cap)}")
    if infeasible:
        raise SystemExit(f"figures infeasible against the archive: "
                         f"{infeasible}")
    if not args.audit_only:
        raise SystemExit("figure generation lands with the next milestone; "
                         "use --audit-only for now")


if __name__ == "__main__":
    main()
