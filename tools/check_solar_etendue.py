"""Check solar beam amplitude using a parallel-ray tube's physical area.

The reference starts equally spaced *parallel* rays on a far transverse
plane, steps Cartesian trajectories with DOP853, and measures their area on
the local transverse plane. Lossless energy gives G = A_far / A_local,
without differentiating a solar-angle map or inserting an n^2 factor.
This is compared with the radial-integral n^2 angular-Jacobian expression.

Run ``uv run python -m tools.check_solar_etendue --out out/solar-etendue.json``.
No production source is modified and no atmospheric MC is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from skycolor.atmosphere import Atmosphere
from skycolor.constants import N_STD_AIR, R_EARTH
from skycolor.refraction import RefractiveProfile, boundary_theta
from tools.radial_solar_mapping import RadialSolarMapping


def _rhs(profile):
    def rhs(_, state):
        pos, direction = state[:3], state[3:]
        radius = np.linalg.norm(pos)
        grad = profile.dndr_of(radius) / profile.n_of(radius) * pos / radius
        return np.concatenate([direction, grad - direction * np.dot(direction, grad)])
    return rhs


def _integrate(profile, position, direction, event, *, max_step=10_000., rtol=2e-13):
    result = solve_ivp(
        _rhs(profile), (0., 15_000_000.), np.concatenate([position, direction]),
        method="DOP853", rtol=rtol,
        atol=np.array([2e-8, 2e-8, 2e-8, 2e-15, 2e-15, 2e-15]),
        max_step=max_step, events=event,
    )
    if not result.success or not len(result.y_events[0]):
        raise RuntimeError(f"Ray did not reach its target: {result.message}")
    return result.y_events[0][0]


def tube_gain(atmo, height_km, true_elevation, *, offsets_m=(5., 20., 80.), rtol=2e-13):
    """Solve central direction, then send adjacent parallel rays inward."""
    profile = RefractiveProfile(atmo)
    point = np.array([0., 0., R_EARTH + height_km * 1e3])
    radius_top = float(profile.r[-1])

    def escape(_, state):
        return np.linalg.norm(state[:3]) - radius_top
    escape.terminal = True
    escape.direction = 1

    def outward(elevation):
        angle = np.radians(elevation)
        direction = np.array([np.cos(angle), 0., np.sin(angle)])
        return _integrate(profile, point, direction, escape, rtol=rtol)

    def residual(elevation):
        ray = outward(elevation)
        unit = ray[3:] / np.linalg.norm(ray[3:])
        return np.degrees(np.arcsin(unit[2])) - true_elevation

    tangent = 90. - np.degrees(boundary_theta(profile, point[2], atmo.r_bounds[0]))
    app = brentq(residual, tangent + 1e-5, min(89., true_elevation + 10.), xtol=2e-10)
    centre = outward(app)
    far_direction = centre[3:] / np.linalg.norm(centre[3:])
    # Move beyond the profile top so every offset begins in constant n.
    far_centre = centre[:3] + 100_000. * far_direction
    local_direction = np.array([np.cos(np.radians(app)), 0., np.sin(np.radians(app))])
    far_y = np.array([0., 1., 0.])
    far_meridian = np.cross(far_y, far_direction)

    def local_plane(_, state):
        return float(np.dot(state[:3] - point, local_direction))
    local_plane.terminal = True
    local_plane.direction = -1

    central_back = _integrate(profile, far_centre, -far_direction, local_plane, rtol=rtol)
    rows = []
    for delta in offsets_m:
        derivatives = []
        for basis in [far_meridian, far_y]:
            plus = _integrate(profile, far_centre + delta * basis, -far_direction,
                              local_plane, rtol=rtol)
            minus = _integrate(profile, far_centre - delta * basis, -far_direction,
                               local_plane, rtol=rtol)
            derivatives.append((plus[:3] - minus[:3]) / (2 * delta))
        area_ratio = abs(float(np.dot(np.cross(*derivatives), local_direction)))
        rows.append(dict(offset_m=delta, local_over_far_area=area_ratio,
                         beam_gain_from_energy=1. / area_ratio))
    radial = RadialSolarMapping(atmo, n_quad=16)
    angular_gain = float(radial.beam_gain(height_km, app, step_deg=0.001))
    return dict(height_km=height_km, true_elevation_deg=true_elevation,
                apparent_elevation_deg=float(app),
                closure_true_elevation_deg=float(true_elevation + residual(app)),
                centre_return_error_m=float(np.linalg.norm(central_back[:3] - point)),
                n_point=profile.n_of(point[2]), n_inf=profile.n_of(profile.r[-1]),
                angular_formula_gain=angular_gain,
                area_tube_results=rows,
                max_relative_gain_difference=float(max(
                    abs(r["beam_gain_from_energy"] / angular_gain - 1) for r in rows)))


def disk_normalisation(radius_arcmin=16., n_radial=8, n_azimuth=16):
    """Exact spherical disk solid angle and a centre-normal irradiance check."""
    alpha = np.radians(radius_arcmin / 60.)
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    cosrho = 0.5 * (1 + np.cos(alpha)) + 0.5 * (1 - np.cos(alpha)) * nodes
    # Normalised true-solid-angle average, including the azimuth quadrature.
    mean_cos = float(np.sum(weights[:, None] * cosrho[:, None]
                            * np.ones((1, n_azimuth))) / (2 * n_azimuth))
    mean_one = float(np.sum(weights[:, None] * np.ones((1, n_azimuth)))
                     / (2 * n_azimuth))
    scalar_to_normal = 2. / (1 + np.cos(alpha))
    return dict(radius_arcmin=radius_arcmin, radial_nodes=n_radial,
                azimuth_nodes=n_azimuth,
                solid_angle_sr=float(2 * np.pi * (1 - np.cos(alpha))),
                normalised_weights_sum=mean_one, mean_cosrho=mean_cos,
                scalar_angular_over_normal_irradiance=float(scalar_to_normal),
                recovered_normal_irradiance=float(mean_cos * scalar_to_normal),
                note="For F_inf normal to the disk centre, L_inf=F_inf/(pi sin(alpha)^2).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=2e-13)
    args = parser.parse_args()
    started = time.perf_counter()
    atmo = Atmosphere([550.], aerosols=[], ozone_du=0)
    standard = []
    for height, elev in [(38., -6.), (40., -6.), (47., -6.), (0.5, 10.)]:
        row = tube_gain(atmo, height, elev, rtol=args.rtol)
        standard.append(row)
        print(json.dumps(row), flush=True)
    constant = []
    for density in [0., N_STD_AIR]:
        uniform = Atmosphere([550.], aerosols=[], ozone_du=0,
            air_profile=lambda z, d=density: np.full_like(np.asarray(z, float), d))
        constant.append(tube_gain(uniform, 38., -6., offsets_m=(5., 20.), rtol=args.rtol))
    result = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  method="DOP853 parallel-ray-tube area, no angular derivative in energy gain",
                  integration_rtol=args.rtol,
                  standard_profile=standard, constant_profiles=constant,
                  solar_disk=disk_normalisation(), seconds=time.perf_counter() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("elapsed", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
