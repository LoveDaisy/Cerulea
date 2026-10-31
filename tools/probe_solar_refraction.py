"""Independent RK4/root-solved point probes of the packaged solar lookup.

No library state or optical baseline is modified. Run with
``uv run python -m tools.probe_solar_refraction --out out/solar-probe.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import brentq

from skycolor import presets
from skycolor.constants import R_EARTH
from skycolor.optics import shell_species_weights, species_extinction_weights
from skycolor.refraction import (
    boundary_theta, geometry, refractive_profile, refractivity_of,
    sun_path_lengths, trace_ray,
)


def root_direction(atmo, profile, height_km, solar_zenith_deg, ground_radius, ds=2000.0):
    """Invert the public RK4 escape direction; return None inside the shadow."""
    r0 = R_EARTH + height_km * 1e3
    target = np.radians(solar_zenith_deg)
    maximum = boundary_theta(profile, r0, ground_radius) - 1e-5

    def residual(theta):
        ray = trace_ray(atmo, r0, theta, r_earth=ground_radius, profile=profile, ds=ds)
        if ray["hit_earth"]:
            raise RuntimeError("The nominally above-tangent bracket hit ground")
        return ray["theta_inf"] - target

    edge_residual = residual(maximum)
    if edge_residual < 0:
        return None, float(np.degrees(edge_residual))
    theta = brentq(residual, 0.5 * np.pi - 0.01, maximum, xtol=1e-10)
    return theta, float(np.degrees(edge_residual))


def rk_columns(atmo, profile, height_km, theta, ground_radius, ds=2000.0):
    """Integrate densities with independent RK4 positions and scalar columns.

    The same continuous profile governs the ray; no shell chord helper is
    used. Species are sampled at all four RK stages and integrated with the
    corresponding RK weights. The path stops at the model top or ground.
    """
    y = np.array([0.0, R_EARTH + height_km * 1e3])
    direction = np.array([np.sin(theta), np.cos(theta)])
    top = float(atmo.r_bounds[-1])
    min_radius = np.linalg.norm(y)
    samples = []

    def rhs(position, unit):
        radius = np.linalg.norm(position)
        gradient = profile.dndr_of(radius) / profile.n_of(radius) * position / radius
        return unit, gradient - unit * np.dot(unit, gradient)

    for _ in range(20000):
        radius = np.linalg.norm(y)
        min_radius = min(min_radius, radius)
        if radius < ground_radius:
            return dict(blocked=True, perigee_km=float((min_radius-R_EARTH)/1e3))
        if radius >= top:
            break
        p1 = y
        k1y, k1u = rhs(p1, direction)
        p2 = y + ds * 0.5 * k1y
        k2y, k2u = rhs(p2, direction + ds * 0.5 * k1u)
        p3 = y + ds * 0.5 * k2y
        k3y, k3u = rhs(p3, direction + ds * 0.5 * k2u)
        p4 = y + ds * k3y
        k4y, k4u = rhs(p4, direction + ds * k3u)
        samples.append([np.linalg.norm(p) for p in (p1, p2, p3, p4)])
        y = y + ds / 6 * (k1y + 2*k2y + 2*k3y + k4y)
        direction = direction + ds / 6 * (k1u + 2*k2u + 2*k3u + k4u)
        direction /= np.linalg.norm(direction)
    else:
        raise RuntimeError("Column ray did not reach the top")

    z = (np.asarray(samples) - R_EARTH) / 1e3
    columns = []
    for density in [atmo.n_air(z), atmo.n_o3(z), atmo.n_no2(z)] + [
        layer.aod550 * layer.profile(z) / 1000.0 for layer in atmo.aerosols
    ]:
        columns.append(float(np.sum(density * np.array([1., 2., 2., 1.])) * ds / 6))
    tau = float(np.asarray(columns) @ species_extinction_weights(atmo)[:, 0])
    return dict(blocked=False, perigee_km=float((min_radius-R_EARTH)/1e3),
                species_columns=columns, tau550=tau, transmittance550=float(np.exp(-tau)))


def packaged_point(atmo, profile, height, zenith, ground_radius):
    apparent = float(geometry().apparent_elevation(np.array([90.-zenith]),
                                                   np.array([height]))[0])
    theta = np.radians(90.-apparent)
    radius = R_EARTH + height*1e3
    invariant = (1.+refractivity_of(atmo, height))*radius*np.cos(np.radians(apparent))
    ground_n = profile.n_of(ground_radius)
    lit = bool(apparent >= 0 or invariant > ground_n * ground_radius)
    zmid = (np.sqrt(atmo.r_bounds[:-1] * atmo.r_bounds[1:])-R_EARTH)/1e3
    lengths = sun_path_lengths(np.array([invariant]), np.array([np.sin(np.radians(apparent))]),
                              np.array([radius]), atmo.r_bounds,
                              1.+refractivity_of(atmo, zmid))[:, 0]
    columns = lengths @ shell_species_weights(atmo, atmo.r_bounds)
    tau = float(columns @ species_extinction_weights(atmo)[:, 0])
    ray = trace_ray(atmo, radius, theta, r_earth=ground_radius, profile=profile, ds=2000.)
    return dict(apparent_elevation_deg=apparent, invariant_m=float(invariant), lit=lit,
                tau550=tau if lit else None,
                transmittance550=float(np.exp(-tau)) if lit else 0.,
                rk_hit_ground=ray["hit_earth"],
                rk_true_zenith_deg=float(np.degrees(ray["theta_inf"])),
                direction_error_deg=float(np.degrees(ray["theta_inf"])-zenith)
                                    if not ray["hit_earth"] else None,
                rk_at_same_apparent=rk_columns(atmo, profile, height, theta, ground_radius))


def angular_gain_probe(atmo, profile, height, zenith, apparent):
    """Point-source angular Jacobian, away from a ground-shadow boundary."""
    radius = R_EARTH + height*1e3
    rows = []
    for ds in [2000., 1000.]:
        for halfwidth in [0.01, 0.005]:
            elevations = []
            for e in [apparent-halfwidth, apparent+halfwidth]:
                ray = trace_ray(atmo, radius, np.radians(90.-e), profile=profile, ds=ds)
                assert not ray["hit_earth"]
                elevations.append(float(90.-np.degrees(ray["theta_inf"])))
            inverse_slope = 2*halfwidth/(elevations[1]-elevations[0])
            gain = (profile.n_of(radius)**2 * np.cos(np.radians(apparent)) /
                    np.cos(np.radians(90.-zenith)) * inverse_slope)
            rows.append(dict(step_m=ds, halfwidth_app_deg=halfwidth,
                             dapp_dtrue=inverse_slope, point_beam_gain=float(gain)))
    table_de = 0.005
    et = 90.-zenith
    vals = geometry().apparent_elevation(np.array([et-table_de, et+table_de]),
                                        np.array([height, height]))
    table_slope = float((vals[1]-vals[0])/(2*table_de))
    return dict(height_km=height, true_zenith_deg=zenith,
                root_apparent_elevation_deg=apparent,
                packaged_dapp_dtrue=table_slope, rk_derivatives=rows,
                caveat="Point angular Jacobian only; no finite disk or horizon truncation")


def refinement_pilot(atmo, profile):
    """Hold out midpoints in small exact-RK inversion tables, not a new LUT."""
    heights = np.arange(30., 51.)
    exact_app = []
    for height in heights:
        theta, _ = root_direction(atmo, profile, height, 96., R_EARTH)
        assert theta is not None
        exact_app.append(90.-np.degrees(theta))
    exact_app = np.asarray(exact_app)
    packaged = geometry().apparent_elevation(np.full_like(heights, -6.), heights)
    mask_2km = (heights % 2 == 0)
    mask_4km = ((heights-30) % 4 == 0)
    result = dict(
        altitude_sweep=dict(heights_km=heights.tolist(), exact_app_deg=exact_app.tolist(),
                            packaged_app_deg=packaged.tolist(),
                            packaged_max_error_deg=float(np.max(np.abs(packaged-exact_app)))),
    )
    for label, mask in [("2km", mask_2km), ("4km", mask_4km)]:
        pred = np.interp(heights, heights[mask], exact_app[mask])
        error = pred[~mask]-exact_app[~mask]
        result["altitude_sweep"][label] = dict(held_out_heights=heights[~mask].tolist(),
                                               error_deg=error.tolist(),
                                               max_abs_error_deg=float(np.max(np.abs(error))))
    true_elev = np.arange(-6.3, -5.69, 0.1)
    exact = []
    for elev in true_elev:
        theta, _ = root_direction(atmo, profile, 38., 90.-elev, R_EARTH)
        assert theta is not None
        exact.append(90.-np.degrees(theta))
    exact = np.asarray(exact)
    knots = np.arange(len(true_elev)) % 2 == 0
    prediction = np.interp(true_elev, true_elev[knots], exact[knots])
    error = prediction[~knots]-exact[~knots]
    result["angular_sweep"] = dict(
        height_km=38., true_elev_deg=true_elev.tolist(), exact_app_deg=exact.tolist(),
        knot_spacing_true_deg=0.2, held_out_error_deg=error.tolist(),
        max_abs_error_deg=float(np.max(np.abs(error))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refinement-pilot", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    base = presets.clear([550.])
    profile = refractive_profile(base)
    if args.refinement_pilot:
        result = refinement_pilot(base, profile)
        result["seconds"] = time.perf_counter()-started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result, indent=2))
        return
    rows = []
    for ground in [0., 2.4]:
        atmo = base if ground == 0 else base.at_altitude(ground)
        rg = float(atmo.r_bounds[0])
        for zenith, heights in [(96., [20., 30., 36., 38., 40., 47., 50.]),
                                (99., [70., 90.])]:
            for height in heights:
                packaged = packaged_point(atmo, profile, height, zenith, rg)
                theta, margin = root_direction(atmo, profile, height, zenith, rg)
                root = None if theta is None else dict(
                    apparent_elevation_deg=float(90.-np.degrees(theta)),
                    rk=rk_columns(atmo, profile, height, theta, rg))
                row = dict(ground_km=ground, height_km=height, true_zenith_deg=zenith,
                           packaged=packaged, rk_root=root, edge_true_minus_target_deg=margin)
                rows.append(row)
                print(json.dumps(dict(ground=ground, z=height, zenith=zenith,
                    table_error_deg=packaged["direction_error_deg"],
                    table_t=packaged["transmittance550"],
                    root_t=0. if root is None else root["rk"]["transmittance550"])), flush=True)

    # Exact apparent/altitude nodes should still regenerate from the same RK4.
    table = np.loadtxt(Path(__file__).resolve().parents[1] / "skycolor/data/refraction.csv",
                       delimiter=",", skiprows=1)
    nodes = []
    for height, apparent in [(30., -4.), (38., -5.), (47., -5.), (47., -6.), (70., -7.)]:
        selected = table[(table[:, 0] == height) & (table[:, 1] == apparent)]
        if not len(selected):
            continue
        raw_true = float(selected[0, 2])
        ray = trace_ray(base, R_EARTH+height*1e3, np.radians(90.-apparent),
                        profile=profile, ds=2000.)
        nodes.append(dict(height_km=height, apparent_elevation_deg=apparent,
                          raw_true_elevation_deg=raw_true,
                          rk_true_elevation_deg=float(90.-np.degrees(ray["theta_inf"])),
                          delta_deg=float(90.-np.degrees(ray["theta_inf"])-raw_true),
                          builder_true_elevation_deg=float(apparent-np.degrees(ray["bending"])),
                          builder_delta_deg=float(apparent-np.degrees(ray["bending"])-raw_true)))
    jacobians = []
    for row in rows:
        if row["ground_km"] == 0 and row["true_zenith_deg"] == 96 and row["height_km"] in [38., 40., 47.]:
            jacobians.append(angular_gain_probe(
                base, profile, row["height_km"], 96., row["rk_root"]["apparent_elevation_deg"]))
    result = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  wavelength_nm=550., ray_step_m=2000., rows=rows, original_nodes=nodes,
                  angular_gain_probes=jacobians,
                  seconds=time.perf_counter()-started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("node_regeneration", nodes, "seconds", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
