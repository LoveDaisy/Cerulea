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
from skycolor import color, render

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


_FISH = "fisheye (equidistant, zenith-centred, 180°)"
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
        "anti-solar means relative azimuth 180°, not a 90° scattering angle"),
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
                      "spherical single\n" + _FISH),
         ViewPanelDef("l5_mc_-4", "total", "fisheye", False,
                      "+ MC higher orders\n" + _FISH),
         ViewPanelDef("l5_plane_-4", "total", "fisheye", False,
                      "plane-parallel world\n" + _FISH)),
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
        "contract-matched pair, each with its own solve-bound floor lighting, "
        "locked cloud exposure 0.03; no fisheye (no ground in a 180° dome)"),
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
    """Audit convenience entry: load, audit, write the report files."""
    report = audit(load(npz_path), npz_path)
    write_audit(report)
    return report


def write_audit(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_coverage.md").write_text(render_audit_markdown(report))
    (OUT / "audit_coverage.json").write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------
# drawing (Step 1+): archive-only re-projection and computed annotations
#
# Two projection families, both fed from the SAME bound presentation:
# - "rect": the archived perspective camera via render.render_result (the
#   production display path, unchanged).
# - "fisheye": a zenith-centred equidistant 180-degree view rebuilt through
#   production primitives only -- directions from render.fisheye_grid's
#   equidistant branch (r/R = theta/90, horizon on the rim), sampling via
#   render.sample_dome with render_result's own relative-azimuth convention,
#   colouring via color.render_rgb with the group's locked exposure.  No
#   interpolation or colour code is written here.
#
# Every annotation pixel (horizon, azimuth ticks, sun direction) is computed
# from the same camera/direction grid the image was rendered from -- nothing
# is hand-placed.  A direction outside the view is reported as such, never
# clamped to the image edge.
# ---------------------------------------------------------------------------
_ANN_INK = "#111111"
_TICK_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)


def fisheye_rgb(pres, exposure: float, size: int = 640, chunk: int = 64):
    """Equidistant fisheye RGB of a bound presentation, plus its direction grid.

    ``exposure`` is the archived display group's locked value -- the same
    value the rect panel of the pair renders with, never an auto exposure.
    Row-chunked so the (size, size, n_wl) spectral intermediate never fully
    materialises; chunking is exact because an explicit exposure makes
    color.render_rgb per-pixel pure (no global statistic crosses chunks).
    Returns (rgb, azimuth, elevation, mask) so annotations reuse the exact
    grid the pixels came from.
    """
    az, el, mask = render.fisheye_grid(size, max_zenith=90.0,
                                       projection="equidistant")
    dome = dict(zenith=pres.zenith, azimuth_rel=pres.azimuth_rel,
                radiance=pres.radiance)
    rel = (az - pres.sun_azimuth_deg + 180.0) % 360.0 - 180.0
    zen = 90.0 - np.clip(el, 0.0, 90.0)
    rgb = np.zeros((size, size, 3))
    for r0 in range(0, size, chunk):
        rows = slice(r0, min(r0 + chunk, size))
        rad = render.sample_dome(dome, zen[rows].ravel(), rel[rows].ravel())
        block = color.render_rgb(rad, pres.wl, exposure=exposure,
                                 tonemap="aces", adaptation=None,
                                 white_xy=color.D65_XY)
        rgb[rows] = block.reshape(rows.stop - rows.start, size, 3)
    rgb[~mask] = 0.0
    return rgb, az, el, mask


def _angular_distance(az, el, az_t, el_t) -> np.ndarray:
    """Great-circle distance [deg] from every grid direction to one target."""
    daz = np.radians((az - az_t + 180.0) % 360.0 - 180.0)
    e1, e2 = np.radians(el), np.radians(el_t)
    cosd = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(daz)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def locate(az, el, az_t, el_t, tol: float = 2.0):
    """Pixel (col, row) of the grid direction nearest a target, or None.

    In view means a pixel within ``tol`` degrees of the target -- an
    out-of-FOV direction is never snapped to the image edge.
    """
    d = _angular_distance(az, el, az_t, el_t)
    row, col = np.unravel_index(int(np.argmin(d)), d.shape)
    if d[row, col] <= tol:
        return int(col), int(row)
    return None


def horizon_polyline(el: np.ndarray):
    """The el == 0 crossing of a camera direction field (one point/column).

    Vectorised sign change + linear interpolation; a pinhole camera's horizon
    is a straight vanishing line, so the sampled crossings trace it.  Columns
    entirely above or below the horizon simply contribute nothing.
    """
    h, w = el.shape
    above = el >= 0.0
    flip = above[:-1] != above[1:]
    rows = np.argmax(flip, axis=0)
    has = flip[rows, np.arange(w)]
    cols = np.arange(w)[has]
    r = rows[has]
    y0, y1 = el[r, cols], el[r + 1, cols]
    t = np.where(y0 != y1, y0 / np.where(y0 != y1, y0 - y1, 1.0), 0.0)
    return cols.astype(float), r + t


def azimuth_tick_pixel(az, el, tick_az: float, tol: float = 0.25):
    """Pixel of a tick azimuth, placed at its closest-to-horizon crossing."""
    d = np.abs((az - tick_az + 180.0) % 360.0 - 180.0)
    near = d <= tol
    if not near.any():
        return None
    cost = np.where(near, np.abs(el), np.inf)
    row, col = np.unravel_index(int(np.argmin(cost)), az.shape)
    return int(col), int(row)


def sun_placement(az, el, sun_az: float, sun_el: float):
    """Where the pre-refraction sun *input direction* annotation goes.

    - ("marker", pixel): direction inside the view -- hollow cross there.
    - ("horizon", pixel): sun below the horizon -- the horizon point at the
      sun's azimuth; the below-horizon disk itself is never drawn.
    - ("outside", None): azimuth out of the FOV -- text note only, no edge
      clamping.

    Cloud scenes keep the same marker semantics: it is an azimuth hint for
    the input direction, never a claim that the disk shines through the deck.
    """
    if sun_el < 0.0:
        d = _angular_distance(az, el, sun_az, 0.0)
        near = np.abs(el) <= 1.5
        if near.any():
            dd = np.where(near, d, np.inf)
            row, col = np.unravel_index(int(np.argmin(dd)), d.shape)
            if np.isfinite(dd[row, col]) and dd[row, col] <= 1.0:
                return "horizon", (int(col), int(row))
        return "outside", None
    p = locate(az, el, sun_az, sun_el)
    if p is not None:
        return "marker", p
    return "outside", None


def _sun_label(kind: str, sun_az: float, sun_el: float) -> str:
    if kind == "marker":
        return f"sun input dir (az {sun_az:.0f}°, el {sun_el:.1f}°)"
    if kind == "horizon":
        return (f"sun input az {sun_az:.0f}° — {abs(sun_el):.1f}° below "
                "horizon")
    return f"sun input az {sun_az:.0f}°, el {sun_el:.1f}° — out of view"


def _ann_text(ax, x, y, text, offset=(5, 7)):
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                fontsize=8, color=_ANN_INK,
                bbox=dict(fc="#ffffff", ec="none", alpha=0.65, pad=1.2))


def _draw_sun(ax, placement, sun_az: float, sun_el: float) -> None:
    kind, p = placement
    if p is None:
        _ann_text(ax, 4, 14, _sun_label("outside", sun_az, sun_el),
                  offset=(2, -4))
        return
    if kind == "marker":
        ax.plot(*p, marker="+", markersize=13, markeredgewidth=1.8,
                color=_ANN_INK)
        _ann_text(ax, *p, _sun_label(kind, sun_az, sun_el))
    else:  # "horizon": boundary direction indicator, never a disk
        ax.plot(*p, marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor=_ANN_INK, markeredgewidth=1.4)
        _ann_text(ax, *p, _sun_label(kind, sun_az, sun_el), offset=(5, -12))


def _plt():
    """Agg-bound pyplot (same local-import convention as layer_story.draw)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def annotate_rect(ax, az, el, sun_az: float, sun_el: float) -> None:
    """Horizon line, azimuth ticks and the sun direction note on a rect view."""
    cols, rows = horizon_polyline(el)
    ax.plot(cols, rows, color=_ANN_INK, linewidth=1.1, alpha=0.85)
    for tick in _TICK_AZIMUTHS:
        p = azimuth_tick_pixel(az, el, tick)
        if p is None:
            continue
        c, r = p
        ax.plot([c, c], [r - 5, r + 5], color=_ANN_INK, linewidth=1.1)
        _ann_text(ax, c, r, f"{tick:.0f}°", offset=(-8, 6))
    _draw_sun(ax, sun_placement(az, el, sun_az, sun_el), sun_az, sun_el)


def annotate_fisheye(ax, az, el, mask, sun_az: float, sun_el: float,
                     size: int) -> None:
    """Horizon rim, azimuth ticks and the sun direction note on a fisheye."""
    from matplotlib.patches import Circle

    cx = cy = (size - 1) / 2.0
    # equidistant: zen = r * 90, so the horizon (zen 90) sits exactly on the
    # r = 1 rim -- the same normalisation fisheye_grid used for the pixels
    rim = (size - 1) / 2.0
    ax.add_patch(Circle((cx, cy), rim, fill=False, ec=_ANN_INK,
                        linewidth=1.2))
    sun = sun_placement(az, el, sun_az, sun_el)
    for tick in _TICK_AZIMUTHS:
        # a below-horizon sun indicator at the same rim azimuth replaces the
        # tick (its label already names that azimuth); markers live inside
        # the dome and never collide with rim ticks
        if (sun[0] == "horizon"
                and abs((tick - sun_az + 180.0) % 360.0 - 180.0) <= 1.0):
            continue
        p = locate(az, el, tick, 0.0, tol=1.0)
        if p is None:
            continue
        dx, dy = p[0] - cx, p[1] - cy
        n = np.hypot(dx, dy)
        u = (dx / n, dy / n)
        ax.plot([p[0], p[0] + 6 * u[0]], [p[1], p[1] + 6 * u[1]],
                color=_ANN_INK, linewidth=1.1)
        # labels sit just inside the rim: always inside the image, never
        # clipped at the square's edge, bbox keeps them readable on sky
        lp = (p[0] - 16 * u[0], p[1] - 16 * u[1])
        _ann_text(ax, *lp, f"{tick:.0f}°", offset=(0, 0))
    _draw_sun(ax, sun, sun_az, sun_el)


def _display_of(data, key: str) -> dict:
    return ast.literal_eval(str(data[f"{key}/display"]))


def _sun_of(data, key: str) -> tuple[float, float]:
    opts = _run_options(data, key)
    return float(opts["sun_azimuth_deg"]), 90.0 - float(opts["sun_zenith_deg"])


def rect_panel(ax, data, vdef: ViewPanelDef, overrides=None) -> dict:
    """The archived perspective view through the untouched production path.

    ``vdef.ground`` is the authoritative ground decision for the panel and is
    passed to render_result explicitly -- the archive display record alone
    would silently re-impose the solving run's own choice (the cloud-group
    record carries ground=True), contradicting the registry/manifest/caption
    declarations.  A ground panel additionally requires the solve-bound
    materials; a no-ground panel renders the no-ground display convention
    below the horizon.
    """
    key = vdef.key
    display = {**_display_of(data, key), "ground": vdef.ground,
               **(overrides or {})}
    pres = ls.bind(data, key, vdef.part)
    if vdef.ground and not (pres.ground_supported
                            and pres.ground_albedo is not None
                            and pres.ground_direct_irradiance is not None):
        raise ValueError(f"{key}: ground display needs the solve-bound "
                         "materials (audit would have refused this view)")
    rgb = render.render_result(pres, **display)
    ax.imshow(rgb)
    w, h = display["size"]
    az, el = render.perspective_grid(w, h, display["hfov_deg"],
                                     display["pitch_deg"],
                                     display["yaw_deg"])
    annotate_rect(ax, az, el, *_sun_of(data, key))
    ax.set_title(vdef.title, fontsize=10)
    return dict(camera="rect", key=key, part=vdef.part, ground=vdef.ground,
                display=display, pixel_size=[w, h])


def fisheye_panel(ax, data, vdef: ViewPanelDef, size: int = 640) -> dict:
    """The equidistant re-projection of the same archived dome."""
    display = _display_of(data, vdef.key)
    pres = ls.bind(data, vdef.key, vdef.part)
    rgb, az, el, mask = fisheye_rgb(pres, float(display["exposure"]), size)
    ax.imshow(rgb)
    annotate_fisheye(ax, az, el, mask, *_sun_of(data, vdef.key), size)
    if "mc" in vdef.key:
        # the archive's single-seed MC stores the zenith as one discrete
        # direction estimate whose value sits ~46% below the 9-degree row it
        # interpolates against; in the equidistant fisheye that discontinuity
        # renders as a dark central cell.  Annotate it so a reader sees an
        # MC-quality property, not sky physics.
        cx = cy = (size - 1) / 2.0
        ax.annotate("MC single-seed zenith cell",
                    xy=(cx, cy), xytext=(cx + 108, cy - 86),
                    arrowprops=dict(arrowstyle="-", color=_ANN_INK, lw=0.9),
                    fontsize=7.5, color=_ANN_INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=_ANN_INK, lw=0.6, alpha=0.85))
    ax.set_title(vdef.title, fontsize=10)
    return dict(camera="fisheye", key=vdef.key, part=vdef.part, ground=False,
                display=dict(exposure=display["exposure"],
                             tonemap=display["tonemap"],
                             adaptation=display["adaptation"]),
                pixel_size=[size, size])


def spectrum_panel(ax, data, spec: SpectrumPanelDef) -> dict:
    """Curves straight from archived rows via the theme's own axis helper."""
    curves = tuple((c.key, c.part, c.label, c.style, c.zenith, c.azimuth)
                   for c in spec.curves)
    ls._spectrum_axis(ax, data, curves, spec.title, loglog=spec.loglog)
    return dict(curves=[dict(key=c.key, part=c.part, zenith=c.zenith,
                             azimuth=c.azimuth, label=c.label)
                        for c in spec.curves],
                loglog=spec.loglog)


# figure-level style: the same paper tone the theme's spectra figure uses
_STYLE = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
          "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"}


def build_figure(data, fig_id: str):
    """Assemble one registered figure from the archive (matplotlib, no solve)."""
    plt = _plt()
    fig_def = FIGURES[fig_id]
    with plt.rc_context(_STYLE):
        if fig_def.layout == "spectrum_pair":
            fig = plt.figure(figsize=(10.8, 8.2), layout="constrained")
            gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0),
                                  height_ratios=(0.62, 1.0))
            spectrum_panel(fig.add_subplot(gs[:, 0]), data, fig_def.spectrum)
            rect_panel(fig.add_subplot(gs[0, 1]), data, fig_def.views[0])
            fisheye_panel(fig.add_subplot(gs[1, 1]), data, fig_def.views[1])
        elif fig_def.layout == "spectrum":
            fig = plt.figure(figsize=(7.4, 4.8), layout="constrained")
            spectrum_panel(fig.add_subplot(), data, fig_def.spectrum)
        elif fig_def.layout == "triple_pair":
            fig = plt.figure(figsize=(12.6, 7.8), layout="constrained")
            gs = fig.add_gridspec(2, 3, height_ratios=(0.62, 1.0))
            for j, vdef in enumerate(fig_def.views[:3]):
                rect_panel(fig.add_subplot(gs[0, j]), data, vdef)
            for j, vdef in enumerate(fig_def.views[3:]):
                fisheye_panel(fig.add_subplot(gs[1, j]), data, vdef)
        elif fig_def.layout == "rect_pair":
            fig = plt.figure(figsize=(11.6, 3.8), layout="constrained")
            for j, vdef in enumerate(fig_def.views):
                rect_panel(fig.add_subplot(1, 2, j + 1), data, vdef)
        elif fig_def.layout == "single_rect":
            fig = plt.figure(figsize=(8.4, 4.6), layout="constrained")
            rect_panel(fig.add_subplot(), data, fig_def.views[0])
        else:
            raise ValueError(f"unknown layout {fig_def.layout}")
        title = fig_def.note
        if fig_def.layout in ("spectrum_pair", "triple_pair"):
            title += ("\nFish-eye rim: the archived dome grid is 11 x 23 "
                      "(8.2-deg azimuth cells); horizon-row truncation zeros "
                      "are archive content, rendered as-is.")
        fig.suptitle(title, fontsize=9.5)
    return fig


def _manifest_entry(data, fig_id: str, npz_path: Path, publish: bool,
                    input_sha: str) -> dict:
    fig_def = FIGURES[fig_id]
    feas = figure_feasibility(data, fig_def)
    try:
        rel = npz_path.relative_to(THEME.parents[1])
    except ValueError:
        rel = npz_path
    command = (f"uv run python -m experiments.layerstory.refresh_figures "
               f"--from-npz {rel} --figure {fig_id}")
    if publish:
        command += " --publish-figures"
    return dict(fig_id=fig_id, publish_name=fig_def.publish_name,
                consumers=list(fig_def.consumers), note=fig_def.note,
                command=command, curves=feas["curves"], views=feas["views"],
                input_sha256=input_sha, solve_version=str(data["version"]),
                drawing_version=git_head())


def _save_bytes(fig, destinations) -> None:
    """Render ONCE, write the identical bytes everywhere.

    savefig(bbox_inches="tight") runs a layout pass that can mutate a
    constrained-layout figure's state, so a second savefig of the same fig
    is not guaranteed to reproduce the first render byte for byte.  The
    published copies must be pixel-identical to the theme run outputs, so
    the figure is rendered to a buffer once and the same bytes are written
    to every destination (same dpi/tight convention as experiments.common.save).
    """
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    data = buf.getvalue()
    for path in destinations:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            shown = path.relative_to(THEME.parents[1])
        except ValueError:
            shown = path
        print(f"  figure -> {shown}")


def render_figures(data, fig_ids, npz_path: Path,
                   publish: bool = False) -> list[Path]:
    """Generate the requested figures; theme out/ always, root only on publish.

    A plain run never touches root figures/ or the historical snapshots --
    publishing is explicit and writes only the registry's own v2 names.
    """
    plt = _plt()

    figures_dir = OUT / "figures"
    root = THEME.parents[1] / "figures"
    input_sha = sha256_file(npz_path)
    written = []
    manifest = []
    for fig_id in fig_ids:
        fig = build_figure(data, fig_id)
        theme_path = figures_dir / f"{fig_id}.png"
        destinations = [theme_path]
        if publish:
            destinations.append(root / FIGURES[fig_id].publish_name)
        _save_bytes(fig, destinations)
        plt.close(fig)
        written.append(theme_path)
        manifest.append(_manifest_entry(data, fig_id, npz_path, publish,
                                        input_sha))
    (OUT / "refresh_manifest.json").write_text(json.dumps(manifest, indent=1))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit and redraw the layer-story figures from the "
                    "committed archive (zero solve)")
    parser.add_argument("--from-npz", type=Path, default=DEFAULT_NPZ,
                        help="archive to read (default: committed data)")
    parser.add_argument("--audit-only", action="store_true",
                        help="write the coverage report and exit")
    parser.add_argument("--figure", action="append", metavar="ID",
                        help="registry figure id, or 'all' (repeatable)")
    parser.add_argument("--publish-figures", action="store_true",
                        help="also write the named v2 files to root figures/")
    args = parser.parse_args(argv)
    if args.figure is None and not args.audit_only:
        parser.error("nothing to do: pass --audit-only and/or --figure "
                     "<id|all>")
    data = load(args.from_npz)
    report = audit(data, args.from_npz)
    write_audit(report)
    print(f"audit -> {OUT / 'audit_coverage.md'}")
    for key, cap in report["runs"].items():
        print(f"  {key}: {_capability_line(cap)}")
    infeasible = [k for k, f in report["figures"].items() if not f["feasible"]]
    if infeasible:
        raise SystemExit(f"figures infeasible against the archive: "
                         f"{infeasible}")
    if args.figure is None:
        return
    ids = sorted(FIGURES) if "all" in args.figure else list(args.figure)
    unknown = [i for i in ids if i not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure ids: {unknown}")
    for path in render_figures(data, ids, args.from_npz,
                               publish=args.publish_figures):
        print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
