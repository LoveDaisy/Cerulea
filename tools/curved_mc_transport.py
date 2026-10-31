"""Rejected thin-index-layer Snell experiment, retained to reproduce a counterexample.

The original delta-tracking/scattering/RNG/source code is retained.  Straight
segments inside thin concentric layers are refracted at their interfaces.
Near grazing incidence the discontinuous index creates artificial total
internal reflection and ducts absent from the continuous monotonic n*r model.
Do not use its all-order estimates as a smooth-atmosphere physical reference.
The accepted experimental flight is in continuous_curved_mc.py. Interface
transmission here has no Fresnel loss and uses reduced radiance L/n^2.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from skycolor.constants import R_EARTH
from skycolor.refraction import RefractiveProfile, trace_ray
from skycolor.rt import mc


def snell_interface(position, direction, n_inside, n_outside, *, bend=True):
    """Transmit or internally reflect unit directions at a spherical interface.

    Arrays broadcast over rays.  Returns ``(direction_out, tir)`` with shapes
    ``(N,3)`` and ``(N,)``.  The inward/outward sign comes from the incident
    direction at the actual crossing point, not its sign at the segment start.
    Equal indices and bend=False preserve the input direction bit-for-bit.
    """
    x = np.asarray(position, dtype=float)
    d = np.asarray(direction, dtype=float)
    ni, no = np.broadcast_arrays(np.asarray(n_inside, float), np.asarray(n_outside, float))
    ni, no = np.broadcast_to(ni, (len(d),)), np.broadcast_to(no, (len(d),))
    if np.any(ni <= 0) or np.any(no <= 0):
        raise ValueError("refractive indices must be positive")
    result = d.copy()
    tir = np.zeros(len(d), dtype=bool)
    change = (ni != no) if bend else np.zeros(len(d), dtype=bool)
    if not change.any():
        return result, tir
    index = np.flatnonzero(change)
    normal = x[index] / np.linalg.norm(x[index], axis=1, keepdims=True)
    mu = np.einsum("ij,ij->i", d[index], normal)
    tangent = d[index] - mu[:, None] * normal
    transmitted_tangent = (ni[index] / no[index])[:, None] * tangent
    sine2 = np.einsum("ij,ij->i", transmitted_tangent, transmitted_tangent)
    reflected = sine2 > 1.0
    sign = np.where(mu >= 0.0, 1.0, -1.0)
    result[index] = transmitted_tangent + (sign * np.sqrt(np.maximum(1.0-sine2, 0.0)))[:, None] * normal
    if reflected.any():
        ri = index[reflected]
        result[ri] = d[ri] - 2.0 * mu[reflected, None] * normal[reflected]
        tir[ri] = True
    return result, tir


class SnellLayers:
    """A single optical definition plus thin transport boundaries/index values."""

    def __init__(self, atmo, layer_km=0.5, *, bend=True):
        if not np.isfinite(layer_km) or layer_km <= 0:
            raise ValueError("layer_km must be finite and positive")
        self.atmo = atmo
        self.layer_km = float(layer_km)
        self.bend = bool(bend)
        z0, z1 = float(atmo.z_bounds[0]), float(atmo.z_bounds[-1])
        # Transport interfaces are independent of the density/source shell
        # discretisation.  Keep that original grid in the fine optical table,
        # rather than pinning an irregular first index layer during refinement.
        heights = np.unique(np.r_[np.arange(z0, z1, layer_km), z1])
        # Distinct floating-point heights can map to the same Earth-centred
        # radius.  Remove those zero-width layers before indexing interfaces.
        self.rb = np.unique(R_EARTH + heights * 1e3)
        self.heights = (self.rb - R_EARTH) / 1e3
        self.profile = RefractiveProfile(atmo)
        self.n_profile = CubicSpline(self.profile.r, self.profile.n)
        mid = np.sqrt(self.rb[:-1] * self.rb[1:])
        self.n = self.n_profile(mid)
        self.fine = np.unique(np.r_[mc._fine_grid(atmo), self.heights])
        self.boundary_events = self.refracted_events = self.tir_events = self.radial_nudges = 0

    def layer_bounds(self, atmo):
        """The clone's replacement _layer_bounds; wavelength restriction is safe."""
        if atmo.r_bounds[0] != self.rb[0] or atmo.r_bounds[-1] != self.rb[-1]:
            raise ValueError("one SnellLayers instance per medium/domain")
        return self.rb

    def fine_grid(self, atmo):
        self.layer_bounds(atmo)
        return self.fine

    def crossing(self, xa, da, distance, layer, next_layer, old_xn):
        """Handle layer boundaries without touching collisions, weights or RNG.

        Existing straight-step positions are retained for equal-index events.
        A rare near-tangent roundoff correction moves radially by at most EPS
        to ensure a reflected path stays on its original side of the boundary.
        """
        self.boundary_events += len(layer)
        ni, no = self.n[layer], self.n[next_layer]
        changed = (ni != no) if self.bend else np.zeros(len(layer), bool)
        if not changed.any():
            return old_xn, da, np.zeros(len(layer), bool), changed
        boundary = xa + distance[:, None] * da
        new_d, tir = snell_interface(boundary, da, ni, no, bend=self.bend)
        out_x = old_xn.copy()
        out_x[changed] = boundary[changed] + mc._EPS_MOVE * new_d[changed]
        # Directions near a critical angle can move less than one radial ULP
        # when offset only along the ray; classify the intended side explicitly.
        hit_radius = np.where(next_layer > layer, self.rb[layer + 1], self.rb[layer])
        side = np.where((next_layer > layer) ^ tir, 1.0, -1.0)
        new_radius = np.linalg.norm(out_x, axis=1)
        bad_side = changed & (side * (new_radius - hit_radius) <= 1e-8)
        if bad_side.any():
            out_x[bad_side] *= ((hit_radius[bad_side] + side[bad_side] * mc._EPS_MOVE)
                               / new_radius[bad_side])[:, None]
        self.refracted_events += int(np.count_nonzero(changed & ~tir))
        self.tir_events += int(np.count_nonzero(tir))
        self.radial_nudges += int(np.count_nonzero(bad_side))
        return out_x, new_d, tir, changed


def rewrite_transport(code: str, atmo, layer_km=0.5, bend=True):
    """Return rewritten tracer source and namespace entries for thin-layer RT.

    ``code`` may already contain a different solar NEE prescription.  Exact
    anchor assertions refuse unknown tracer layouts.  No RNG/scattering/source
    block is replaced, and this function never edits a library module.
    """
    for anchor in ["    rb = _layer_bounds(atmo)", "    z_grid = _fine_grid(atmo)",
                   "        kill |= escaped\n"]:
        assert code.count(anchor) == 1, f"unexpected tracer anchor: {anchor!r}"
    assert "_curved_layers" not in code, "transport rewrite already applied"
    layers = SnellLayers(atmo, layer_km, bend=bend)
    anchor = "        kill |= escaped\n"
    insertion = '''
        # Thin-layer Snell boundary event: source/collision/RNG remain original.
        if _curved_layers.bend:
            bi = np.flatnonzero((~trial) & (~ground) & (~escaped))
            if bi.size:
                next_li = np.where(s_lo[bi] <= s_up[bi], li[bi] - 1, li[bi] + 1)
                bx, bd, _, changed = _curved_layers.crossing(
                    xa[bi], da[bi], s_exit[bi], li[bi], next_li, xn[bi])
                if changed.any():
                    ci_boundary = bi[changed]
                    xn[ci_boundary] = bx[changed]
                    d_new[ci_boundary] = bd[changed]
                    rn[ci_boundary] = np.linalg.norm(bx[changed], axis=1)
'''
    return code.replace(anchor, anchor + insertion), dict(
        _layer_bounds=layers.layer_bounds, _fine_grid=layers.fine_grid,
        _curved_layers=layers)


def trace_unscattered(layers, height_km, apparent_elevation_deg, max_steps=200000):
    """A no-collision probe using the same boundary event, not the MC estimator."""
    x = np.array([[0., 0., R_EARTH + height_km*1e3]])
    angle = np.radians(apparent_elevation_deg)
    d = np.array([[np.cos(angle), 0., np.sin(angle)]])
    initial_layer = int(np.clip(np.searchsorted(layers.rb, np.linalg.norm(x), side="right")-1,
                                0, len(layers.n)-1))
    invariant = float(layers.n[initial_layer] * np.linalg.norm(np.cross(x[0], d[0])))
    max_relative_drift = 0.0
    for step in range(max_steps):
        radius = float(np.linalg.norm(x))
        b = float(x[0] @ d[0])
        li = int(np.clip(np.searchsorted(layers.rb, radius, side="right")-1, 0, len(layers.n)-1))
        rlo, rup = layers.rb[li:li+2]
        sup = -b + np.sqrt(max(b*b + (rup-radius)*(rup+radius), 0.0))
        disc = b*b - (radius-rlo)*(radius+rlo)
        slo = -b-np.sqrt(max(disc, 0.0)) if b < 0 and disc > 0 else np.inf
        sexit = min(sup, slo)
        ground = li == 0 and np.isfinite(slo) and slo <= sup
        if ground:
            return dict(hit_ground=True, steps=step, max_relative_invariant_drift=max_relative_drift)
        xn = x + (sexit + mc._EPS_MOVE)*d
        if np.linalg.norm(xn) >= layers.rb[-1] - 1e-9:
            return dict(hit_ground=False, steps=step+1,
                true_elevation_deg=float(np.degrees(np.arcsin(np.clip(d[0, 2], -1, 1)))),
                max_relative_invariant_drift=max_relative_drift,
                boundary_events=layers.boundary_events, tir_events=layers.tir_events,
                radial_nudges=layers.radial_nudges)
        target = li-1 if slo <= sup else li+1
        xn, dn, tir, changed = layers.crossing(
            x, d, np.array([sexit]), np.array([li]), np.array([target]), xn)
        next_li = li if tir[0] else target
        actual = float(layers.n[next_li] * np.linalg.norm(np.cross(xn[0], dn[0])))
        max_relative_drift = max(max_relative_drift, abs(actual/invariant-1))
        x, d = xn, dn
    raise RuntimeError("unscattered thin-layer ray did not terminate")


def main():
    from skycolor import presets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fine", action="store_true",
                        help="also probe 0.0625/0.03125/0.015625 km index layers")
    args = parser.parse_args()
    base = presets.clear([550.])
    rows = []
    for ground in [0., 2.4]:
        atmo = base if ground == 0 else base.at_altitude(ground)
        profile = RefractiveProfile(atmo)
        for height, app in [(38., -5.487342769121213), (47., -5.7677034867361385),
                            (ground, 0.2), (ground, 5.0), (ground, 30.0)]:
            rk = trace_ray(atmo, R_EARTH+height*1e3, np.radians(90-app),
                           r_earth=atmo.r_bounds[0], r_top=atmo.r_bounds[-1],
                           profile=profile, ds=500.)
            reference = 90.0 - np.degrees(rk["theta_inf"])
            steps = [0.5, 0.25, 0.125] + ([0.0625, 0.03125, 0.015625] if args.fine else [])
            for layer in steps:
                model = SnellLayers(atmo, layer)
                result = trace_unscattered(model, height, app)
                result.update(ground_km=ground, height_km=height, app_elevation_deg=app,
                              layer_km=layer, rk_true_elevation_deg=float(reference),
                              angle_error_deg=result.get("true_elevation_deg", np.nan)-reference)
                rows.append(result)

    # Snell's law and reflection have an algebraic conservation oracle.
    point = np.array([[0., 0., R_EARTH+40e3]])
    angle = np.radians(60.)
    d = np.array([[np.sin(angle), 0., np.cos(angle)]])
    reflected, tir = snell_interface(point, d, [1.5], [1.0])
    np.testing.assert_allclose(reflected, [[d[0, 0], 0., -d[0, 2]]], atol=1e-14)
    assert tir[0]
    transmitted, tir = snell_interface(point, d, [1.00028], [1.00025])
    assert not tir[0]
    np.testing.assert_allclose(1.00028*np.linalg.norm(np.cross(point[0], d[0])),
        1.00025*np.linalg.norm(np.cross(point[0], transmitted[0])), rtol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(transmitted), 1., atol=2e-15)
    same, _ = snell_interface(point, d, [1.5], [1.5])
    np.testing.assert_array_equal(same, d)
    same, _ = snell_interface(point, d, [1.5], [1.0], bend=False)
    np.testing.assert_array_equal(same, d)
    rng = np.random.default_rng(314)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = np.broadcast_to(point, directions.shape)
    ni, no = np.full(256, 1.00028), np.full(256, 1.00001)
    refracted, reflected = snell_interface(points, directions, ni, no)
    actual_index = np.where(reflected, ni, no)
    before = ni[:, None] * np.cross(points, directions)
    after = actual_index[:, None] * np.cross(points, refracted)
    np.testing.assert_allclose(before, after, rtol=2e-13, atol=2e-8)
    np.testing.assert_allclose(np.linalg.norm(refracted, axis=1), 1., atol=2e-15)

    # A constant transport index leaves the same fine-grid MC paths exactly
    # unchanged, including all random draws and source/scattering values.
    results = []
    for bend in [False, True]:
        code, extras = rewrite_transport(inspect.getsource(mc._trace_wavelength), base, 0.5, bend)
        extras["_curved_layers"].n[:] = 1.
        namespace = dict(mc.__dict__, **extras)
        exec(compile(code, str(mc.__file__), "exec"), namespace)
        with patch.object(mc, "_trace_wavelength", namespace["_trace_wavelength"]):
            results.append(mc.mc_radiance_split(base, 96., [0.], [0.], n_photons=512, seed=29))
    assert all(np.array_equal(results[0][key], results[1][key]) for key in results[0])
    result = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows, snell_and_tir_checks=True, equal_index_exact_direction=True,
        constant_index_mc_bit_exact=True,
        note="Thin-index-layer diffuse transport only; source remains its own continuous definition")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(max_angle_error_deg=max(abs(r["angle_error_deg"]) for r in rows),
        finest_max_angle_error_deg=max(abs(r["angle_error_deg"]) for r in rows if r["layer_km"]==0.125),
        constant_index_mc_bit_exact=True), indent=2))


if __name__ == "__main__":
    main()
