"""Single-scattering source ablation with independently integrated solar angles.

This tool retains the production shell optical-depth formula, geometric view
and geometric phase. It varies the solar angle prescription and point-beam
angular amplitude only. An optional vertical solar-disk marginal is an
approximation to a uniform finite disk, not an exact 2-D disk ray integral.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from skycolor import presets, refraction, spectra
from skycolor.constants import R_EARTH
from skycolor.rt import single
from tools.radial_solar_mapping import RadialSolarMapping


class InverseSolarMapping:
    """Solve the monotonic radial forward map; do not extrapolate into shadow."""

    def __init__(self, atmo, n_quad=8):
        self.forward = RadialSolarMapping(atmo, n_quad=n_quad)
        self.atmo = atmo
        self.last = None

    def apparent_elevation(self, true_elev, height):
        et, z = np.broadcast_arrays(np.asarray(true_elev, float), np.asarray(height, float))
        nr = self.forward._n(R_EARTH + z.ravel() * 1e3) * (R_EARTH + z.ravel() * 1e3)
        lower = -np.degrees(np.arccos(np.minimum(self.forward.q_ground / nr, 1.0))) + 1e-6
        crit = self.forward.forward(z.ravel(), lower)
        lit = et.ravel() >= crit
        app = lower - 0.001
        gain = np.zeros_like(app)
        if lit.any():
            zz, target, low = z.ravel()[lit], et.ravel()[lit], lower[lit]
            value = np.clip(refraction.geometry().apparent_elevation(target, zz), low, 89.99)
            for _ in range(8):
                f = self.forward.forward(zz, value) - target
                if np.max(np.abs(f)) < 2e-8:
                    break
                minus = np.maximum(value - 0.001, low)
                plus = np.minimum(value + 0.001, 89.999)
                slope = (self.forward.forward(zz, plus) - self.forward.forward(zz, minus)) / (
                    plus - minus)
                value = np.clip(value - f / slope, low, 89.99)
            residual = self.forward.forward(zz, value) - target
            assert np.max(np.abs(residual)) < 1e-6, "Solar inverse failed to converge"
            minus = np.maximum(value - 0.001, low)
            plus = np.minimum(value + 0.001, 89.999)
            slope = (self.forward.forward(zz, plus) - self.forward.forward(zz, minus)) / (
                plus - minus)
            n = self.forward._n(R_EARTH + zz * 1e3)
            gain[lit] = ((n / self.forward.n_inf)**2 * np.cos(np.radians(value))
                         / np.cos(np.radians(target)) / slope)
            app[lit] = value
        self.last = dict(app=app.reshape(z.shape), gain=gain.reshape(z.shape),
                         lit=lit.reshape(z.shape))
        return self.last["app"]

    def amplitude(self, true_elev, height, app):
        assert np.array_equal(self.last["app"], app)
        return self.last["gain"]


def radiance(atmo, sun_elevation, mode, inverse, n_gauss):
    """Scope numerical changes to a cloned function; original source is untouched."""
    if mode == "packaged":
        return single.sky_radiance(atmo, 90 - sun_elevation, [0], [0], n_gauss=n_gauss)[0]
    source = inspect.getsource(single.sky_radiance)
    old = "e_app = _refr_geometry().apparent_elevation(e_true, z_pts)"
    assert source.count(old) == 1
    source = source.replace(old, "e_app = _audit_inverse.apparent_elevation(e_true, z_pts)")
    if mode == "radial_beam":
        old = "out[sl] += contrib.sum(axis=0) * F0"
        assert source.count(old) == 1
        source = source.replace(old,
            "contrib *= _audit_inverse.amplitude(e_true, z_pts, e_app)[..., None]\n"
            "        out[sl] += contrib.sum(axis=0) * F0")
    namespace = dict(single.__dict__, _audit_inverse=inverse)
    exec(compile(source, str(Path(single.__file__).resolve()), "exec"), namespace)
    return namespace["sky_radiance"](
        atmo, 90 - sun_elevation, [0], [0], n_gauss=n_gauss)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-gauss", type=int, default=12)
    parser.add_argument("--n-quad", type=int, default=8)
    parser.add_argument("--disk-nodes", type=int, nargs="*", default=[])
    parser.add_argument("--elevations", type=float, nargs="+", default=[0, 2.4])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    wl = np.arange(380., 781., 5.)
    v = spectra.cie_xyz(wl)[:, 1]
    base = presets.clear(wl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for height in args.elevations:
        atmo = base if height == 0 else base.at_altitude(height)
        inverse = InverseSolarMapping(atmo, args.n_quad)
        modes = ["packaged", "radial_attenuation", "radial_beam"]
        for mode in modes + [f"vertical_disk_{n}" for n in args.disk_nodes]:
            started = time.perf_counter()
            if mode.startswith("vertical_disk_"):
                n = int(mode.rsplit("_", 1)[1])
                # Gauss-Chebyshev second kind integrates the vertical marginal
                # 2/pi sqrt(1-u^2) of a uniform circular disk; weights sum to 1.
                k = np.arange(1, n + 1)
                offsets = (16.0 / 60.0) * np.cos(k * np.pi / (n + 1))
                weights = 2.0 / (n + 1) * np.sin(k * np.pi / (n + 1))**2
                assert abs(weights.sum() - 1) < 1e-12
                rad = sum(weight * radiance(atmo, -6 + offset, "radial_beam", inverse,
                                            args.n_gauss) for offset, weight in
                          zip(offsets, weights))
            else:
                rad = radiance(atmo, -6, mode, inverse, args.n_gauss)
            record = dict(head=head, mode=mode, elevation=height, wavelengths=wl.tolist(),
                n_gauss=args.n_gauss, n_quad=args.n_quad,
                radiance=rad.tolist(), photopic=float(683 * np.trapezoid(rad * v, wl)),
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                seconds=time.perf_counter() - started)
            with args.out.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({key: record[key] for key in
                              ["mode", "elevation", "photopic", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
