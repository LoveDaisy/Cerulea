"""Experiments 09 and 10 -- what a cloud deck does to the colour of the sky.

(a) clear sky vs stratus vs altostratus at several sun elevations;
(b) the sunset sequence of an altostratus deck: the red underlit base;
(c) the hemispherical albedo of a homogeneous layer against optical depth,
    compared with the delta-Eddington analytic solution and with Monte Carlo.

The cloud radiance comes from :func:`skycolor.rt.cloud.sky_radiance_with_cloud`,
bound into a ``radiance_field`` closure and displayed through the standard
``render`` pipeline (:func:`render.sky_dome` / :func:`render.render_scene`)
-- one display path for clear and cloudy skies; this script only binds the
medium and the deck into the field callable the display layer consumes.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiments.common import WL, chromaticity_of, save

THEME = Path(__file__).resolve().parent          # experiments/<theme>
FIG = THEME / "figures"      # theme-owned figure output (not the root candidates)
OUT = THEME / "out"          # theme-owned numeric output (gitignored)
from skycolor import cloud, color, presets, render
from skycolor.color import K_M
from skycolor.constants import WL_DEFAULT
from skycolor.rt.cloud import (delta_eddington, illuminate_cloud,
                               sky_radiance_with_cloud)
from skycolor.rt.solver import sky_radiance_total

#: Monte-Carlo reference albedos (conservative layer, mu0 = 1, black ground).
#: 2e5-4e5 photons, 1 sigma <= 0.001; see docs/physics/CLOUD.md.
MC_G0 = np.array([0.3406, 0.5165, 0.7372, 0.8523, 0.9206])
MC_G085 = np.array([0.0424, 0.0913, 0.2372, 0.4224, 0.6176])
MC_TAU = np.array([1.0, 2.0, 5.0, 10.0, 20.0])


# ---------------------------------------------------------------------------
# medium-side bindings (the display pipeline itself lives in render.py)
# ---------------------------------------------------------------------------
def _cloud_field(atmo, cloud_layer, sun_zenith_deg):
    """Bind the medium and the deck into the ``radiance_field`` callable that
    :func:`render.sky_dome` / :func:`render.render_scene` consume; the
    display layer sees only ``(zen, az_rel) -> (n_view, n_wl)``."""
    def field(zen, az_rel):
        return sky_radiance_with_cloud(atmo, cloud_layer, sun_zenith_deg,
                                       zen, az_rel)
    return field


def _cloud_direct_irradiance(atmo, cloud_layer, sun_zenith_deg):
    """Deck-attenuated ground irradiance -- thin forwarder.

    The implementation moved to the library
    (:func:`skycolor.rt.cloud.ground_direct_irradiance`, task M3) so the
    production display and these figures share exactly one beam-sum; this
    wrapper keeps the experiment call sites unchanged.
    """
    from skycolor.rt.cloud import ground_direct_irradiance as _library_helper

    return _library_helper(atmo, cloud_layer, sun_zenith_deg)


def _cloud_panorama(atmo, cloud_layer, sun_elevation_deg, size=(420, 170),
                    exposure=None, sun_disk=False):
    """One panorama through the standard render path, with the deck-attenuated
    ground; ``exposure=None`` renders with render_scene's own auto-exposure.
    ``cloud_layer=None`` renders the plain clear sky through the built-in
    field (render's default path)."""
    sza = 90.0 - float(sun_elevation_deg)
    has_deck = cloud_layer is not None
    return render.render_scene(
        atmo, sun_elevation_deg=float(sun_elevation_deg), camera="panorama",
        size=size, exposure=exposure, tonemap="aces", adaptation=None,
        sun_disk=sun_disk, ground=True,
        radiance_field=_cloud_field(atmo, cloud_layer, sza) if has_deck else None,
        ground_direct_irradiance=_cloud_direct_irradiance(atmo, cloud_layer, sza)
        if has_deck else None)


def _cloud_dome(atmo, cloud_layer, sun_zenith_deg, n_zenith=40, n_azimuth=49):
    """A cloudy sky dome through :func:`render.sky_dome` (for exposure pick)."""
    return render.sky_dome(atmo, float(sun_zenith_deg), n_zenith=n_zenith,
                           n_azimuth=n_azimuth,
                           radiance_field=_cloud_field(atmo, cloud_layer,
                                                       float(sun_zenith_deg)))


def _exposure(dome, percentile=97.0):
    """One exposure for a whole figure, so brightness differences survive."""
    lum = color.luminance(color.spectrum_to_xyz(dome["radiance"], dome["wl"]))
    pos = lum[lum > 0]
    return 1.0 / max(np.percentile(pos, percentile), 1e-12)


# ---------------------------------------------------------------------------
# (a) clear vs overcast
# ---------------------------------------------------------------------------
def figure_comparison(atmo):
    elevations = [40.0, 20.0, 8.0, 2.0, -1.0]
    scenarios = [("clear", None), ("stratus", cloud.get_preset("stratus")),
                 ("altostratus", cloud.get_preset("altostratus"))]
    # one exposure from the clear noon sky, shared by every panel
    ref = render.sky_dome(atmo, 90.0 - elevations[0])
    exposure = _exposure(ref)
    print(f"  shared exposure = {exposure:.3e}")

    panels = []
    for name, layer in scenarios:
        row = []
        for elev in elevations:
            img = _cloud_panorama(atmo, layer, elev, exposure=exposure,
                                  sun_disk=(layer is None and elev > 0.0))
            row.append(img)
            if layer is not None:
                prof = sky_radiance_with_cloud(atmo, layer, 90.0 - elev,
                                               [0.0], [0.0])
                lum = color.luminance(color.spectrum_to_xyz(prof[0], atmo.wl))
                print(f"  {name:12s} sun {elev:+5.1f} deg  zenith luminance "
                      f"{lum:9.1f} cd/m2")
        panels.append(row)

    ny, nx = panels[0][0].shape[:2]
    gap = 4
    canvas = np.zeros((3 * ny + 4 * gap, len(elevations) * (nx + gap) - gap, 3))
    for i, row in enumerate(panels):
        for j, img in enumerate(row):
            y0 = i * (ny + gap)
            x0 = j * (nx + gap)
            canvas[y0:y0 + ny, x0:x0 + nx] = img

    fig, ax = plt.subplots(figsize=(15.0, 8.2))
    ax.imshow(canvas)
    ax.set_xticks([j * (nx + gap) + nx / 2 for j in range(len(elevations))])
    ax.set_xticklabels([f"sun {e:+.0f}°" for e in elevations], fontsize=9)
    ax.set_yticks([i * (ny + gap) + ny / 2 for i in range(3)])
    ax.set_yticklabels(["clear", "stratus (τ=20)", "altostratus (τ=5)"], fontsize=10)
    ax.set_title("clear sky vs overcast: the deck swaps the Rayleigh blue for a flat grey\n"
                 "(panorama, sun at azimuth 0, shared exposure, no chromatic adaptation)",
                 fontsize=10)
    ax.tick_params(length=0)
    save(fig, FIG / "10_cloud_clear_vs_overcast.png")
    plt.close(fig)
    return exposure


def figure_fisheye(atmo):
    """A fisheye of the overcast sky, to show the overcast luminance law."""
    layer = cloud.get_preset("stratus")
    dome = _cloud_dome(atmo, layer, 90.0 - 45.0, n_zenith=48, n_azimuth=61)
    rgb = render.render_scene(atmo, sun_elevation_deg=45.0, camera="fisheye",
                              size=(420, 420), exposure=_exposure(dome), tonemap="aces",
                              adaptation=None, sun_disk=False, ground=False,
                              radiance_field=_cloud_field(atmo, layer, 45.0))

    # luminance profile across the sky: the (1 + 2 cos theta)/3 law
    zen = np.linspace(0.0, 90.0, 91)
    prof = sky_radiance_with_cloud(atmo, layer, 90.0 - 45.0, zen, 0.0)
    lum = color.luminance(color.spectrum_to_xyz(prof, atmo.wl))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    axes[0].imshow(rgb)
    axes[0].set_title("overcast fisheye, sun 45° (equal-area projection)", fontsize=9)
    axes[0].axis("off")
    axes[1].plot(zen, lum / lum[0], "o-", ms=3, label="model")
    mu = np.cos(np.radians(zen))
    axes[1].plot(zen, (1 + 2 * mu) / 3, "k--", lw=1, label="(1+2 cos θ)/3")
    axes[1].set_xlabel("view zenith angle [deg]")
    axes[1].set_ylabel("L / L_zenith")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("overcast luminance distribution", fontsize=9)
    save(fig, FIG / "10_cloud_overcast_fisheye.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) sunset sequence
# ---------------------------------------------------------------------------
def figure_sunset(atmo):
    elevations = [5.0, 1.0, -0.5, -1.5, -2.2, -3.0]
    scenarios = [("altostratus 3-5 km, τ=5", cloud.get_preset("altostratus")),
                 ("cirrus 8-10 km, τ=0.5", cloud.get_preset("cirrus"))]
    # Each panel is exposed for its own sky, like a photographer would: a
    # single exposure across the sequence would leave everything after sunset
    # black.  The luminance printed on each panel carries the absolute scale.
    panels, notes = [], []
    for name, layer in scenarios:
        row, row_notes = [], []
        for elev in elevations:
            sza = 90.0 - elev
            dome = _cloud_dome(atmo, layer, sza)
            row.append(_cloud_panorama(atmo, layer, elev,
                                       exposure=_exposure(dome)))
            illum = illuminate_cloud(atmo, layer, sza)
            if illum.geometry == "none":
                row_notes.append((elev, "none", None, 0.0))
                print(f"  {name:26s} sun {elev:+5.1f}°  in the Earth's shadow")
                continue
            # the colour of the *cloud itself*, not of the sky behind it
            _, comp = sky_radiance_with_cloud(atmo, layer, sza, [0.0], 0.0,
                                              return_components=True)
            xy, lum = chromaticity_of(comp["cloud"][0])
            row_notes.append((elev, illum.geometry, xy, lum))
            print(f"  {name:26s} sun {elev:+5.1f}°  lit={illum.geometry:4s} "
                  f"z_shadow={illum.shadow_km:4.1f} km  base xy=({xy[0]:.3f},{xy[1]:.3f}) "
                  f"L={lum:8.2f} cd/m2")
        panels.append(row)
        notes.append(row_notes)

    ny, nx = panels[0][0].shape[:2]
    gap = 4
    canvas = np.zeros((2 * ny + 3 * gap, len(elevations) * (nx + gap) - gap, 3))
    for i, row in enumerate(panels):
        for j, img in enumerate(row):
            canvas[i * (ny + gap):i * (ny + gap) + ny, j * (nx + gap):j * (nx + gap) + nx] = img

    fig, ax = plt.subplots(figsize=(15.0, 8.4))
    ax.imshow(canvas)
    ax.set_xticks([j * (nx + gap) + nx / 2 for j in range(len(elevations))])
    ax.set_xticklabels([f"sun {e:+.1f}°" for e in elevations], fontsize=9)
    ax.set_yticks([i * (ny + gap) + ny / 2 for i in range(2)])
    ax.set_yticklabels([s[0] for s in scenarios], fontsize=9)
    for i, row in enumerate(notes):
        for j, (elev, geom, xy, lum) in enumerate(row):
            label = ("shadow" if geom == "none"
                     else f"{geom} ({xy[0]:.2f},{xy[1]:.2f})  {lum:.0f} cd/m²")
            ax.text(j * (nx + gap) + 5, i * (ny + gap) + ny - 14, label,
                    fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1.5, lw=0))
    ax.set_title("sunset sequence: the deck is still lit after the ground is dark\n"
                 "cloud-base chromaticity (x,y) inset; each panel exposed for its own sky",
                 fontsize=10)
    ax.tick_params(length=0)
    save(fig, FIG / "10_cloud_sunset_sequence.png")
    plt.close(fig)
    return notes


def figure_albedo(atmo):
    taus = np.geomspace(0.1, 500.0, 60)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # ---- (1) albedo vs optical depth -------------------------------------
    ax = axes[0]
    for g, style in [(0.0, "-"), (0.5, "-"), (0.6, "--"), (0.861, "-")]:
        R = delta_eddington(taus, 1.0, g, 1.0, f=g**2).albedo
        ax.plot(taus, R, style, lw=1.6, label=f"delta-Eddington, g={g:.2f}")
    ax.plot(MC_TAU, MC_G0, "o", ms=5, mfc="none", label="Monte Carlo, g=0")
    ax.plot(MC_TAU, MC_G085, "s", ms=5, mfc="none", label="Monte Carlo, g=0.85")
    ax.axhline(0.7, color="0.6", ls=":", lw=1)
    ax.axvline(10.0, color="0.6", ls=":", lw=1)
    ax.plot([10.0], [0.69], "k*", ms=9)
    ax.annotate("τ=10, g=0.6 → 0.69\n(the 'about 0.7' reference)",
                (10, 0.69), fontsize=7, xytext=(14, 0.42),
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("water droplets: τ=10, g=0.86 → 0.40", (10, 0.4025), fontsize=7,
                xytext=(1.2, 0.10),
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xscale("log")
    ax.set_xlabel("optical depth at 550 nm")
    ax.set_ylabel("reflected fraction")
    ax.set_title("cloud albedo vs optical depth\n(conservative layer, sun overhead)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)

    # ---- (2) how long the deck stays lit ---------------------------------
    ax = axes[1]
    elevations = np.array([20.0, 10.0, 4.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0])
    for name, layer in [("altostratus", cloud.get_preset("altostratus")),
                        ("cirrus", cloud.get_preset("cirrus"))]:
        top = ([], [])
        base = ([], [])
        for elev in elevations:
            illum = illuminate_cloud(atmo, layer, 90.0 - elev)
            L = sky_radiance_with_cloud(atmo, layer, 90.0 - elev, [0.0], 0.0,
                                        return_components=True)[1]["cloud"][0]
            lum = K_M * color.spectrum_to_xyz(L, atmo.wl)[1]
            target = top if illum.geometry == "top" else base
            target[0].append(elev)
            target[1].append(max(lum, 1e-3))
        ax.semilogy(top[0], top[1], "o-", ms=4, label=f"{name}, lit from above")
        ax.semilogy(base[0], base[1], "s--", ms=4, mfc="none",
                    label=f"{name}, lit from below")
    clear = []
    for elev in elevations:
        L = sky_radiance_total(atmo, 90.0 - elev, [0.0], 0.0, multiple=True)[0]
        clear.append(K_M * color.spectrum_to_xyz(L, atmo.wl)[1])
    ax.semilogy(elevations, clear, "k:", lw=1, label="clear zenith")
    ax.axvline(0.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("sun elevation [deg]")
    ax.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.set_title("the deck stays lit after the ground is dark\n"
                 "(the jump at 0° is the top-lit / underlit switch)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # ---- (3) the colour is borrowed from the light source ----------------
    # A 1 nm grid is needed here: at 5 nm the solar Fraunhofer lines alias into
    # noise, which is a sampling artefact and not part of the model.  Only the
    # cloud term is needed, so no multiple-scattering solve is involved.
    ax = axes[2]
    wl1 = WL_DEFAULT
    atmo1 = presets.clear(wl1)
    layer = cloud.get_preset("altostratus")
    for elev, col in [(20.0, "#444"), (0.5, "#999"), (-1.0, "tab:orange"), (-2.0, "tab:red")]:
        _, comp = sky_radiance_with_cloud(atmo1, layer, 90.0 - elev, [0.0], 0.0,
                                          include_clear=False, return_components=True)
        rad = comp["cloud"][0]
        smooth = np.convolve(rad / rad.max(), np.ones(9) / 9.0, mode="same")
        ax.plot(wl1[4:-4], smooth[4:-4], color=col,
                label=f"sun {elev:+.1f}°  ({comp['illumination'].geometry}-lit)")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("normalised radiance")
    ax.set_title("the deck borrows its colour from the light source\n"
                 "(altostratus base spectrum, normalised, 9 nm boxcar)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, FIG / "10_cloud_albedo_and_sunset.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(parts=("09", "fisheye", "sunset", "albedo")):
    wl = WL  # 5 nm grid, 380-780 nm (shared with the other experiments)
    atmo = presets.clear(wl)

    if "09" in parts:
        print("[09] clear vs overcast")
        figure_comparison(atmo)
    if "fisheye" in parts:
        figure_fisheye(atmo)
    if "sunset" in parts:
        print("[10] sunset sequence")
        figure_sunset(atmo)
    if "albedo" in parts:
        print("[10] albedo and sunset photometry")
        figure_albedo(atmo)

    # numeric summary for the docs and the tests
    optics = cloud.get_preset("stratus").optics(np.array([550.0]))
    taus = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 200.0])
    res = delta_eddington(taus, 1.0, optics.g[0], 1.0, f=optics.f[0])
    print("  stratus: g=%.4f  delta-M f=%.4f  ->  g'=%.4f, tau'/tau=%.4f"
          % (optics.g[0], optics.f[0], res.g_scaled[0],
             (1.0 - optics.omega[0] * optics.f[0])))
    for t, r in zip(taus, res.albedo):
        print(f"    tau={t:6.1f}  R={r:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "10_clouds.npz", wl=wl, tau=taus, albedo=res.albedo,
             mc_tau=MC_TAU, mc_g0=MC_G0, mc_g085=MC_G085,
             stratus_g=optics.g, stratus_f=optics.f)
    print("done")


if __name__ == "__main__":
    import sys

    main(tuple(sys.argv[1:]) or ("09", "fisheye", "sunset", "albedo"))
