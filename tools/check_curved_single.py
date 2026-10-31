"""Independent DOP853 first-order oracle for the experimental continuous MC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import presets, spectra
from skycolor.constants import R_EARTH
from skycolor.refraction import RefractiveProfile
from skycolor.rt import mc
from tools.continuous_curved_mc import rewrite_transport
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source, solar_disk


class OpticalField:
    """The MC's continuous log-linear coefficients, with no transport reuse."""

    def __init__(self, atmo, iw):
        self.z = mc._fine_grid(atmo)
        self.atmo, self.iw = atmo, iw
        self.rel = [a.aerosol.relative_coefficients(atmo.wl) for a in atmo.aerosols]
        ray = atmo.n_air(self.z) * atmo._sigma_ray[iw]
        extinction = ray + atmo.n_o3(self.z) * atmo._sigma_o3[iw]
        extinction += atmo.n_no2(self.z) * atmo._sigma_no2[iw]
        aerosol = []
        for layer, (ext, sca) in zip(atmo.aerosols, self.rel):
            weight = layer.aod550 * layer.profile(self.z) / 1000
            extinction += weight * ext[iw]
            aerosol.append(weight * sca[iw])
        total = ray + sum(aerosol, start=np.zeros_like(ray))
        self.log = np.log(np.maximum(np.array([extinction, total, ray, *aerosol]), 1e-300))

    def values(self, z):
        return np.array([np.exp(np.interp(z, self.z, field)) for field in self.log])


def deterministic(atmo, source, field, sun_hat, direction, *, step_m, n_gauss=8):
    """Integrate dL=F/(4pi)*beta_sca*phase_source*exp(-tau)*ds."""
    profile = RefractiveProfile(atmo)
    index = CubicSpline(profile.r, profile.n)
    derivative = index.derivative()
    def rhs(_, y):
        x, d = y[:3], y[3:6]
        radius = np.linalg.norm(x)
        radial = x / radius
        gradient = float(derivative(radius) / index(radius)) * radial
        z = (radius - R_EARTH) / 1000
        extinction = float(np.exp(np.interp(z, field.z, field.log[0])))
        return np.r_[d, gradient - d * (d @ gradient), extinction]
    def top(_, y):
        return np.linalg.norm(y[:3]) - atmo.r_bounds[-1]
    top.terminal, top.direction = True, 1
    y0 = np.r_[0., 0., R_EARTH + 1000 * atmo.elevation_km, direction, 0.]
    solution = solve_ivp(rhs, (0., 2e7), y0, method="DOP853", events=top,
        dense_output=True, max_step=step_m, rtol=2e-10,
        atol=np.array([1e-5]*3 + [1e-12]*3 + [1e-10]))
    if not solution.success or not len(solution.t_events[0]):
        raise RuntimeError(solution.message)
    distance = float(solution.t_events[0][0])
    edges = np.r_[np.arange(0., distance, step_m), distance]
    gx, gw = np.polynomial.legendre.leggauss(n_gauss)
    half = np.diff(edges) / 2
    s = ((edges[1:] + edges[:-1])[:, None] / 2 + half[:, None] * gx).ravel()
    ds = (half[:, None] * gw).ravel()
    y = solution.sol(s)
    x, backward, tau = y[:3].T, y[3:6].T, y[6]
    z = (np.linalg.norm(x, axis=1) - R_EARTH) / 1000
    _, sca, ray, *aerosol = field.values(z)
    fractions = np.array(aerosol).T / sca[:, None]
    kernel = source.scatter(x, backward, sun_hat, field.iw, ray/sca, fractions)
    radiance = float(np.sum(ds * sca * kernel * np.exp(-tau)) *
                     spectra.solar_irradiance([atmo.wl[field.iw]])[0] / (4*np.pi))
    invariant = index(np.linalg.norm(x, axis=1)) * np.linalg.norm(np.cross(x, backward), axis=1)
    nz = invariant[invariant > 0]
    stats = dict(radiance=radiance, step_m=step_m, n_gauss=n_gauss,
        distance_m=distance, final_tau=float(solution.y_events[0][0, 6]),
        n_quad=len(s), n_ode=solution.nfev,
        direction_norm_error=float(np.max(abs(np.linalg.norm(backward, axis=1)-1))),
        max_bouguer_relative_drift=float(np.max(abs(nz/nz[0]-1))) if len(nz) else 0.)
    # A sparse independent vector reconstruction audits both the phase sign and
    # the local L/n^2 transfer, without calling source._geometry.
    active = np.flatnonzero(kernel > 0)
    check = active[np.linspace(0, len(active)-1, min(7, len(active))).astype(int)]
    phase_checks = []
    directions, weights = solar_disk(sun_hat, source.n_radial, source.n_azimuth, source.radius_deg)
    for j in check:
        radial = x[j] / np.linalg.norm(x[j])
        mu = directions @ radial
        app, transmission, gain, _ = source.table.query(z[j], np.degrees(np.arcsin(mu)), field.iw)
        tangent = (directions - mu[:, None]*radial) / np.sqrt(1-mu**2)[:, None]
        local_sun = np.cos(np.radians(app))[:, None]*tangent + np.sin(np.radians(app))[:, None]*radial
        cosine = np.clip(local_sun @ backward[j], -1., 1.)
        phase = spectra.rayleigh_phase(cosine) * ray[j]/sca[j]
        for k, layer in enumerate(atmo.aerosols):
            phase += np.asarray(layer.aerosol.phase(cosine, [atmo.wl[field.iw]])).reshape(-1) * fractions[j,k]
        transfer = (source.observer_n / source.table.mapping.refractive_index(z[j]))**2
        expected = float(np.sum(weights * transmission * gain * phase * transfer))
        phase_checks.append(dict(height_km=float(z[j]), source_kernel=float(kernel[j]),
            vector_kernel=expected, relative_error=expected/kernel[j]-1,
            reduced_radiance_factor=float(transfer)))
    stats["phase_and_reduced_radiance_checks"] = phase_checks
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=7300)
    parser.add_argument("--photons-per-view", type=int, default=5000)
    parser.add_argument("--wavelength", type=float, default=550.)
    parser.add_argument("--views", type=float, nargs="+", default=[0., 70., 85.])
    parser.add_argument("--steps", type=float, nargs="+", default=[2000., 1000.])
    parser.add_argument("--table", type=Path, default=ROOT / "validation/data/"
        "unified_solar_source_audit/tables/point-source-clear81-ground0-refined.npz")
    parser.add_argument("--table-label", default="phase3-clear-81-ground0-refined")
    args = parser.parse_args()
    if args.seeds < 2 or args.photons_per_view < 1 or any(s <= 0 for s in args.steps):
        parser.error("at least two seeds, positive photon count and positive steps are required")
    started = time.perf_counter()
    atmo = presets.clear(np.arange(380., 781., 5.))
    iw = int(np.flatnonzero(atmo.wl == args.wavelength)[0])
    path = args.table.resolve()
    table = PointSolarTable.load(path, atmo, expected_label=args.table_label)
    field = OpticalField(atmo, iw)
    views = args.views
    dirs, counts, sun_hat = mc._prepare(atmo, 96., views, [0.]*len(views), 0.,
                                      args.photons_per_view*len(views))
    report = dict(wavelength_nm=args.wavelength, full_grid=atmo.wl.tolist(),
        table_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        table_label=table.label, table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sun_elevation_deg=-6., ground_km=0., views_deg=views,
        optical_definition="MC fine-grid log-linear beta fields, full81 relative coefficients",
        single_semantics="first atmospheric collision and zero previous ground reflections; max_scatter=1",
        deterministic=[], mc=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["seconds"] = time.perf_counter()-started
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    source = FiniteSolarSource(atmo, table)
    for step in args.steps:
        for view, direction in zip(views, dirs):
            result = deterministic(atmo, source, field, sun_hat, direction, step_m=step)
            result["view_deg"] = view
            report["deterministic"].append(result)
            save()
            print(json.dumps({k:result[k] for k in ["step_m","view_deg","radiance","n_quad"]}), flush=True)
    dens = mc._shell_species_density(atmo)
    for step in args.steps:
        for seed in range(args.seed_start, args.seed_start+args.seeds):
            source = FiniteSolarSource(atmo, table)
            flights = []
            def transport(code, medium):
                code, extras = rewrite_transport(code, medium, max_step_m=step)
                flights.append(extras["_continuous_flight"])
                return code, extras
            before = time.perf_counter()
            with mc_solar_source(source, transport=transport):
                result = mc._trace_wavelength(atmo, iw, dirs, counts, sun_hat, 1,
                    np.random.default_rng(seed), True, dens, field.rel)
            assert np.array_equal(result["total"], result["single"])
            assert not np.any(result["multiple"])
            one = dict(step_m=step, seed=seed, counts=counts.tolist(),
                single=result["single"].tolist(), single_var=result["single_var"].tolist(),
                multiple=result["multiple"].tolist(), seconds=time.perf_counter()-before,
                total_equals_single=True,
                rk_ray_steps=flights[0].rk_ray_steps)
            report["mc"].append(one)
            save()
            print(json.dumps(one), flush=True)
    report["comparison"] = []
    for step in args.steps:
        rows = [r for r in report["mc"] if r["step_m"] == step]
        means = np.array([r["single"] for r in rows])
        analytic = np.array([r["radiance"] for r in report["deterministic"] if r["step_m"] == step])
        mean, se = means.mean(axis=0), means.std(axis=0, ddof=1)/np.sqrt(len(rows))
        reported = np.sqrt(np.sum([r["single_var"] for r in rows], axis=0))/len(rows)
        report["comparison"].append(dict(step_m=step, mc_mean=mean.tolist(),
            seed_se=se.tolist(), reported_se=reported.tolist(), deterministic=analytic.tolist(),
            relative_difference=(mean/analytic-1).tolist(),
            z_score=[float(d/s) if s > 0 else None for d,s in zip(mean-analytic,se)]))
    z = np.array([0., .037, .5, 1., 2.4, 5., 10., 20., 30., 40., 50., 80., 100., 150., 199.])
    ae, aerosol_sca = atmo.aerosol_coefficients(z)
    ray = atmo.rayleigh_coefficient(z)
    direct_ext = ray + ae + atmo.ozone_coefficient(z) + atmo.no2_coefficient(z)
    direct_sca = ray + aerosol_sca
    represented = field.values(z)
    report["medium_probe"] = dict(heights_km=z.tolist(),
        extinction_log_grid_vs_direct_max_relative=float(np.max(abs(represented[0]/direct_ext[:,iw]-1))),
        scattering_log_grid_vs_direct_max_relative=float(np.max(abs(represented[1]/direct_sca[:,iw]-1))),
        full_vs_restricted_aerosol_bitwise=bool(np.array_equal(
            ae[:,[iw]], atmo._restrict([iw]).aerosol_coefficients(z)[0])))
    names = ["tools/check_curved_single.py", "tools/continuous_curved_mc.py",
             "tools/unified_solar_transport.py", "skycolor/aerosol.py", "skycolor/rt/mc.py"]
    report["source_sha256"] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    report["head"] = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    save()


if __name__ == "__main__":
    main()
