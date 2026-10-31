"""Controlled solar-source ablations on unchanged straight-path Monte Carlo.

This experimental tool changes only the next-event solar attenuation/shadow.
It copies the current tracer function into an isolated namespace; no library
source or global helper is modified. The public entry is scoped to that copy
for one call, so runs are sequential, not thread-safe. Refraction amplitudes,
local phase angles and diffuse-path bending are deliberately not added.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from skycolor import presets, refraction, spectra
from skycolor.constants import R_EARTH
from skycolor.rt import mc
from skycolor.rt.single import sky_radiance


class SolarAttenuation:
    """Packaged solar bending prescription; transport and phase stay geometric."""

    def __init__(self, atmo, mode):
        self.mode = mode
        self.original = mc._sun_transmittance
        self.fingerprint = hashlib.sha256()
        self.n_calls = 0
        self.n_points = 0
        self.r_bounds = atmo.r_bounds
        mid = np.sqrt(self.r_bounds[:-1] * self.r_bounds[1:])
        self.n_layer = 1 + refraction.refractivity_of(atmo, (mid - R_EARTH) / 1e3)
        self.ground_invariant = (1 + refraction.refractivity_of(
            atmo, atmo.z_bounds[0])) * self.r_bounds[0]

    def __call__(self, atmo, x, sun_hat, dens, tau_w):
        self.fingerprint.update(np.asarray(x.shape, dtype=np.int64).tobytes())
        self.fingerprint.update(np.ascontiguousarray(x).tobytes())
        self.n_calls += 1
        self.n_points += len(x)
        if self.mode == "geometric":
            return self.original(atmo, x, sun_hat, dens, tau_w)
        r = np.linalg.norm(x, axis=1)
        mu = np.clip(x @ sun_hat / r, -1, 1)
        z = (r - R_EARTH) / 1e3
        e_true = np.degrees(np.arcsin(mu))
        e_app = refraction.geometry().apparent_elevation(e_true, z)
        mu_app = np.sin(np.radians(e_app))
        invariant = (1 + refraction.refractivity_of(atmo, z)) * r * np.cos(
            np.radians(e_app))
        lit = (mu_app >= 0) | (invariant > self.ground_invariant)
        lengths = refraction.sun_path_lengths(invariant, mu_app, r,
                                             self.r_bounds, self.n_layer)
        tau = (lengths.T @ dens) @ tau_w
        return np.where(lit, np.exp(-tau), 0)


@contextmanager
def solar_source(atmo, mode):
    provider = SolarAttenuation(atmo, mode)
    source = inspect.getsource(mc._trace_wavelength)
    namespace = dict(mc.__dict__, _sun_transmittance=provider)
    exec(compile(source, str(Path(mc.__file__).resolve()), "exec"), namespace)
    tracer = namespace["_trace_wavelength"]
    with patch.object(mc, "_trace_wavelength", tracer):
        yield provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["geometric", "attenuation"],
                        choices=["geometric", "attenuation"])
    parser.add_argument("--elevations", type=float, nargs="+", default=[0, 2.4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--photons", type=int, default=500_000)
    parser.add_argument("--wavelengths", type=float, nargs="+",
                        default=[450, 500, 550, 600, 650])
    parser.add_argument("--sun", type=float, default=-6)
    args = parser.parse_args()
    wl = np.array(args.wavelengths)
    base = presets.clear(wl)
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "skycolor"], cwd=root)
    v = spectra.cie_xyz(wl)[:, 1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for height in args.elevations:
        atmo = base if height == 0 else base.at_altitude(height)
        ss = sky_radiance(atmo, 90 - args.sun, [0], [0])[0]
        for seed in args.seeds:
            paired_fingerprint = None
            for mode in args.modes:
                start = time.perf_counter()
                with solar_source(atmo, mode) as source:
                    result = mc.mc_radiance_split(
                        atmo, 90 - args.sun, [0], [0], seed=seed,
                        n_photons=args.photons)
                fp = source.fingerprint.hexdigest()
                if paired_fingerprint is None:
                    paired_fingerprint = fp
                else:
                    assert paired_fingerprint == fp, "Solar ablation changed photon paths"
                fields = {key: value[0] for key, value in result.items()}
                fields["split_total"] = ss + fields["multiple"]
                integrals = {key: float(683 * np.trapezoid(value * v, wl))
                             for key, value in fields.items() if not key.endswith("_err")}
                record = dict(head=head, source_diff_sha256=hashlib.sha256(diff).hexdigest(),
                    tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mode=mode, elevation=height, seed=seed, photons_per_wavelength=args.photons,
                    wavelengths=wl.tolist(), sun_elevation=args.sun,
                    path_fingerprint=fp, n_nee_calls=source.n_calls, n_nee_points=source.n_points,
                    deterministic_single=ss.tolist(),
                    fields={k: val.tolist() for k, val in fields.items()},
                    photopic_integrals=integrals, seconds=time.perf_counter() - start)
                with args.out.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps({k: record[k] for k in
                    ["mode", "elevation", "seed", "photopic_integrals", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
