"""Continuous-index RK flights for experimental Monte Carlo transport.

The tracer keeps its continuous fine-grid extinction and uses one global
delta-tracking majorant.  Only geometric free flight is replaced: RK4
substeps contain no RNG/scattering/source calls.  Boundary events use the
first ground/top root of a cubic-Hermite path, including a possible interior
radial minimum.  Interface jumps, artificial TIR and Fresnel factors do not
exist in this continuous-index model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from skycolor.constants import R_EARTH
from skycolor.refraction import trace_ray
from skycolor.rt import mc
from tools.radial_solar_mapping import RadialSolarMapping


class ContinuousCurvedFlight:
    """Advance physical arc lengths, retaining reduced-radiance path weights.

    ``advance(x,d,s)`` returns (position, arrival_direction, trial, ground,
    escaped).  For constant n, including a constant non-vacuum far field,
    or bend=False the analytic straight flight is used exactly.  The optional
    profile is for independent numerical limiting cases, not a density change.
    """

    def __init__(self, atmo, max_step_m=2000.0, *, bend=True, profile=None):
        if not np.isfinite(max_step_m) or max_step_m <= 0:
            raise ValueError("max_step_m must be finite and positive")
        self.mapping = RadialSolarMapping(atmo, profile=profile, n_quad=4)
        self.r_ground = float(atmo.r_bounds[0])
        self.r_top = float(atmo.r_bounds[-1])
        self.rb = np.array([self.r_ground, self.r_top])
        self.max_step_m = float(max_step_m)
        self.bend = bool(bend)
        self.constant_index = bool(np.all(self.mapping.n == self.mapping.n[0]))
        self.calls = self.rk_steps = self.rk_ray_steps = self.turn_min_checks = 0
        self.ground_hits = self.top_escapes = 0
        self.last_distance = None

    def layer_bounds(self, atmo):
        if atmo.r_bounds[0] != self.r_ground or atmo.r_bounds[-1] != self.r_top:
            raise ValueError("one continuous-flight instance per medium/domain")
        return self.rb

    @staticmethod
    def _dot(a, b):
        return np.einsum("ij,ij->i", a, b)

    def _rhs(self, position, direction):
        radius = np.sqrt(self._dot(position, position))
        index = self.mapping._n(radius)
        gradient = (self.mapping._dn(radius) / index / radius)[:, None] * position
        acceleration = gradient - self._dot(direction, gradient)[:, None] * direction
        return direction, acceleration

    def _rk4(self, x, d, h):
        step = h[:, None]
        k1x, k1d = self._rhs(x, d)
        k2x, k2d = self._rhs(x + 0.5*step*k1x, d + 0.5*step*k1d)
        k3x, k3d = self._rhs(x + 0.5*step*k2x, d + 0.5*step*k2d)
        k4x, k4d = self._rhs(x + step*k3x, d + step*k3d)
        xn = x + step/6 * (k1x + 2*k2x + 2*k3x + k4x)
        dn = d + step/6 * (k1d + 2*k2d + 2*k3d + k4d)
        dn /= np.sqrt(self._dot(dn, dn))[:, None]
        return xn, dn

    @staticmethod
    def _hermite(x, d, a, b, h, theta):
        t = theta[:, None]
        position = x + (h*theta)[:, None] * (d + t*(a + t*b))
        direction = d + t*(2*a + 3*t*b)
        return position, direction

    def _boundary_event(self, x, d, xn, dn, h):
        """Earliest root of a monotonic-q radial path's Hermite interpolant.

        A monotonic n*r medium has no radial maximum, but can have a minimum.
        An endpoint-only test misses a substep entering and leaving the ground.
        The derivative sign brackets that minimum independently of endpoint
        radii; the first crossing is then bracketed before the minimum.
        """
        r0 = np.sqrt(self._dot(x, x))
        r1 = np.sqrt(self._dot(xn, xn))
        v0, v1 = self._dot(x, d), self._dot(xn, dn)
        turning = (v0 < 0) & (v1 > 0)
        candidate = (r1 <= self.r_ground) | (r1 >= self.r_top) | turning
        event_t = np.full(len(x), np.inf)
        event_kind = np.zeros(len(x), dtype=np.int8)
        index = np.flatnonzero(candidate)
        if not len(index):
            return event_t, event_kind, None
        xx, dd, hh = x[index], d[index], h[index]
        delta = (xn[index] - xx) / hh[:, None]
        aa = 3*delta - 2*dd - dn[index]
        bb = -2*delta + dd + dn[index]
        minimum = np.ones(len(index))
        ti = np.flatnonzero(turning[index])
        if len(ti):
            self.turn_min_checks += len(ti)
            lo, hi = np.zeros(len(ti)), np.ones(len(ti))
            for _ in range(40):
                mid = 0.5*(lo+hi)
                px, pd = self._hermite(xx[ti], dd[ti], aa[ti], bb[ti], hh[ti], mid)
                descending = self._dot(px, pd) < 0
                lo = np.where(descending, mid, lo)
                hi = np.where(descending, hi, mid)
            minimum[ti] = 0.5*(lo+hi)
        px, _ = self._hermite(xx, dd, aa, bb, hh, minimum)
        min_radius = np.sqrt(self._dot(px, px))
        ground = min_radius <= self.r_ground
        top = r1[index] >= self.r_top
        # At a surface launch, only a ray initially entering that surface
        # has a zero-time event; a tangent/outgoing launch is allowed.
        ground |= (r0[index] <= self.r_ground) & (v0[index] < 0)
        top |= (r0[index] >= self.r_top) & (v0[index] >= 0)
        for hit, radius, upper, kind in [
            (ground, self.r_ground, minimum, 1),
            (top, self.r_top, np.ones(len(index)), 2),
        ]:
            j = np.flatnonzero(hit)
            if not len(j):
                continue
            lo, hi = np.zeros(len(j)), upper[j].copy()
            for _ in range(40):
                mid = 0.5*(lo+hi)
                pos, _ = self._hermite(xx[j], dd[j], aa[j], bb[j], hh[j], mid)
                radial = np.sqrt(self._dot(pos, pos)) - radius
                before = radial > 0 if kind == 1 else radial < 0
                lo = np.where(before, mid, lo)
                hi = np.where(before, hi, mid)
            theta = 0.5*(lo+hi)
            immediate = ((r0[index[j]] <= radius) & (v0[index[j]] < 0) if kind == 1
                         else (r0[index[j]] >= radius) & (v0[index[j]] >= 0))
            theta[immediate] = 0.0
            select = theta < event_t[index[j]]
            event_t[index[j[select]]] = theta[select]
            event_kind[index[j[select]]] = kind
        return event_t, event_kind, (index, aa, bb)

    def _straight(self, x, d, s):
        radius = np.sqrt(self._dot(x, x))
        b = self._dot(x, d)
        top = -b + np.sqrt(np.maximum(b*b + (self.r_top-radius)*(self.r_top+radius), 0))
        disc = b*b - (radius-self.r_ground)*(radius+self.r_ground)
        ground = np.where((b < 0) & (disc >= 0),
                          -b-np.sqrt(np.maximum(disc, 0)), np.inf)
        boundary = np.minimum(top, ground)
        trial = s < boundary
        travel = np.where(trial, s, boundary)
        hit_ground = (~trial) & (ground <= top)
        escaped = (~trial) & ~hit_ground
        self.last_distance = travel
        self.ground_hits += int(hit_ground.sum())
        self.top_escapes += int(escaped.sum())
        return x + travel[:, None]*d, d.copy(), trial, hit_ground, escaped

    def advance(self, x, d, distance):
        """Advance sampled free paths or stop at the first physical boundary."""
        position = np.asarray(x, float).copy()
        direction = np.asarray(d, float).copy()
        remaining = np.broadcast_to(np.asarray(distance, float), (len(position),)).copy()
        if np.any(~np.isfinite(remaining)) or np.any(remaining < 0):
            raise ValueError("flight distances must be finite and nonnegative")
        self.calls += 1
        if not self.bend or self.constant_index:
            return self._straight(position, direction, remaining)
        n = len(position)
        trial, ground, escaped = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        travel = np.zeros(n)
        alive = np.arange(n)
        for _ in range(200000):
            if not len(alive):
                break
            h = np.minimum(remaining[alive], self.max_step_m)
            zero = h == 0
            if zero.any():
                trial[alive[zero]] = True
                alive = alive[~zero]
                h = h[~zero]
                if not len(alive):
                    break
            xa, da = position[alive], direction[alive]
            xn, dn = self._rk4(xa, da, h)
            self.rk_steps += 1
            self.rk_ray_steps += len(alive)
            theta, kind, hermite = self._boundary_event(xa, da, xn, dn, h)
            event = kind != 0
            if event.any():
                ci, aa, bb = hermite
                lookup = np.full(len(alive), -1, dtype=int)
                lookup[ci] = np.arange(len(ci))
                j = np.flatnonzero(event)
                loc = lookup[j]
                px, pd = self._hermite(xa[j], da[j], aa[loc], bb[loc], h[j], theta[j])
                radius = np.where(kind[j] == 1, self.r_ground, self.r_top)
                px *= (radius / np.sqrt(self._dot(px, px)))[:, None]
                pd /= np.sqrt(self._dot(pd, pd))[:, None]
                xn[j], dn[j] = px, pd
                ground[alive[j]] = kind[j] == 1
                escaped[alive[j]] = kind[j] == 2
                travel[alive[j]] += h[j]*theta[j]
            position[alive], direction[alive] = xn, dn
            no_event = ~event
            travel[alive[no_event]] += h[no_event]
            remaining[alive[no_event]] = np.maximum(remaining[alive[no_event]]-h[no_event], 0)
            done = no_event & (remaining[alive] == 0)
            trial[alive[done]] = True
            alive = alive[no_event & ~done]
        else:
            raise RuntimeError("continuous RK flight exceeded its substep bound")
        self.last_distance = travel
        self.ground_hits += int(ground.sum())
        self.top_escapes += int(escaped.sum())
        return position, direction, trial, ground, escaped


def rewrite_transport(code: str, atmo, max_step_m=2000, bend=True):
    """Replace flight geometry and direct source/scatter uses of its direction.

    Input may already contain root's unified NEE source.  The existing single
    random free-path draw remains; RK/dense boundary handling draws no random
    numbers.  Scattering phase/sampling uses the ARRIVAL direction.
    """
    assert "_continuous_flight" not in code and "_curved_layers" not in code
    start = "        r = np.sqrt(np.einsum(\"ij,ij->i\", xa, xa))"
    end = "        d_new = da.copy()\n"
    assert code.count(start) == code.count(end) == 1
    lo, hi = code.index(start), code.index(end)+len(end)
    replacement = '''        bmax = np.full(na, major[0])
        s = -np.log(np.maximum(rng.random(na), 1e-300)) / bmax
        xn, arrival, trial, ground, escaped = _continuous_flight.advance(xa, da, s)
        rn = np.sqrt(np.einsum("ij,ij->i", xn, xn))
        d_new = arrival.copy()
'''
    code = code[:lo] + replacement + code[hi:]
    start = "        # --- escape through the top"
    end = "        kill |= escaped\n"
    assert code.count(start) == code.count(end) == 1
    lo, hi = code.index(start), code.index(end)+len(end)
    code = code[:lo] + "        kill |= escaped\n" + code[hi:]
    assert code.count("da[ci]") >= 1 and code.count("da[si]") == 1
    code = code.replace("da[ci]", "arrival[ci]").replace("da[si]", "arrival[si]")
    flight = ContinuousCurvedFlight(atmo, max_step_m, bend=bend)
    return code, dict(_layer_bounds=flight.layer_bounds, _continuous_flight=flight)


def main():
    from skycolor import presets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = presets.clear([550.])
    rows = []
    for ground in [0., 2.4]:
        atmo = base if ground == 0 else base.at_altitude(ground)
        for height, app in [(38., -5.487342769121213), (47., -5.7677034867361385),
                            (ground, 0.2), (ground, 30.)]:
            ref = trace_ray(atmo, R_EARTH+height*1e3, np.radians(90-app),
                            r_earth=atmo.r_bounds[0], r_top=atmo.r_bounds[-1], ds=250.)
            reference = 90-np.degrees(ref["theta_inf"])
            x = np.array([[0., 0., R_EARTH+height*1e3]])
            angle = np.radians(app)
            d = np.array([[np.cos(angle), 0., np.sin(angle)]])
            for step in [4000., 2000., 1000.]:
                flight = ContinuousCurvedFlight(atmo, step)
                q0 = float(flight.mapping.refractive_index(height) * np.linalg.norm(np.cross(x,d)))
                px, pd, trial, hit, escaped = flight.advance(x, d, [4e6])
                qe = float(flight.mapping._n(np.linalg.norm(px)) * np.linalg.norm(np.cross(px,pd)))
                rows.append(dict(ground_km=ground, height_km=height, apparent_deg=app,
                    max_step_m=step, rk_true_elevation_deg=float(reference),
                    true_elevation_deg=float(np.degrees(np.arcsin(pd[0,2]))),
                    angle_error_deg=float(np.degrees(np.arcsin(pd[0,2]))-reference),
                    relative_bouguer_error=qe/q0-1 if q0 else 0.,
                    escaped=bool(escaped[0]), hit_ground=bool(hit[0]),
                    travel_m=float(flight.last_distance[0]), rk_ray_steps=flight.rk_ray_steps))

    constant = []
    rng = np.random.default_rng(91)
    x = np.zeros((32,3)); x[:,2] = R_EARTH+40e3
    d = rng.normal(size=(32,3)); d /= np.linalg.norm(d,axis=1,keepdims=True)
    s = np.geomspace(1.,4e6,32)
    for n in [1.,1.000278]:
        profile = SimpleNamespace(r=np.linspace(R_EARTH,R_EARTH+1.2e6,4000),n=np.full(4000,n))
        straight = ContinuousCurvedFlight(base,bend=False,profile=profile).advance(x,d,s)
        curved = ContinuousCurvedFlight(base,bend=True,profile=profile).advance(x,d,s)
        assert all(np.array_equal(a,b) for a,b in zip(straight,curved))
        constant.append(dict(n=n,all_fields_bit_exact=True))

    # Both ends of one 2 km step are outside Earth, but its interior enters it.
    x = np.array([[0.,0.,R_EARTH+0.01]])
    angle = np.radians(-0.005)
    d = np.array([[np.cos(angle),0.,np.sin(angle)]])
    event_rows = []
    for step in [4000.,2000.,1000.]:
        flight = ContinuousCurvedFlight(base,step)
        px,pd,trial,ground,escaped = flight.advance(x,d,[2000.])
        assert ground[0] and not trial[0] and not escaped[0]
        assert abs(np.linalg.norm(px)-R_EARTH)<2e-8
        event_rows.append(dict(max_step_m=step,ground_hit=True,
            first_hit_distance_m=float(flight.last_distance[0]),turn_min_checks=flight.turn_min_checks))
    miss_flight = ContinuousCurvedFlight(base,2000.)
    miss_x = x.copy(); miss_x[:,2] = R_EARTH+0.05
    _,_,miss_trial,miss_ground,miss_top = miss_flight.advance(miss_x,d,[2000.])
    assert miss_trial[0] and not miss_ground[0] and not miss_top[0]
    top_flight = ContinuousCurvedFlight(base,2000.)
    top_x = np.array([[0.,0.,top_flight.r_top-0.01]])
    _,_,top_trial,top_ground,top_escape = top_flight.advance(top_x,[[0.,0.,1.]],[2000.])
    assert top_escape[0] and not top_trial[0] and not top_ground[0]
    assert abs(top_flight.last_distance[0]-0.01)<2e-6
    # A radial arc-length integral independently checks the first ground event.
    mapping = ContinuousCurvedFlight(base).mapping
    rg = R_EARTH
    qg = float(mapping._n(rg)*rg)
    invariant = float(mapping._n(rg+0.01)*(rg+0.01)*np.cos(angle))
    q1 = float(mapping._n(rg)+rg*mapping._dn(rg))
    q2 = float(2*mapping._dn(rg)+rg*mapping._d2n(rg))
    def integrand(delta):
        radius = rg+delta
        q = float(mapping._n(radius)*radius)
        gap = (qg-invariant)+delta*(q1+0.5*delta*q2)
        return q/np.sqrt(gap*(q+invariant))
    arc, _ = quad(integrand,0.,0.01,epsabs=1e-7)

    # Validate arrival-direction rewiring and the global-majorant clone compiles.
    code,extras = rewrite_transport(inspect.getsource(mc._trace_wavelength),base,2000.,False)
    assert "_rotate(arrival[si]" in code and "da[ci]" not in code
    namespace = dict(mc.__dict__,**extras)
    exec(compile(code,str(mc.__file__),"exec"),namespace)
    with patch.object(mc,"_trace_wavelength",namespace["_trace_wavelength"]):
        sample = mc.mc_radiance_split(base,96.,[0.],[0.],n_photons=512,seed=17)
    assert all(np.isfinite(v).all() for v in sample.values())
    original_namespace = dict(mc.__dict__,_layer_bounds=extras["_layer_bounds"])
    exec(compile(inspect.getsource(mc._trace_wavelength),str(mc.__file__),"exec"),original_namespace)
    with patch.object(mc,"_trace_wavelength",original_namespace["_trace_wavelength"]):
        straight_reference = mc.mc_radiance_split(base,96.,[0.],[0.],n_photons=512,seed=17)
    assert all(np.array_equal(sample[k],straight_reference[k]) for k in sample)

    path_invariants = []
    for height,app in [(38.,-5.487342769121213),(0.,0.2)]:
        for step in [4000.,2000.,1000.]:
            flight = ContinuousCurvedFlight(base,step)
            xx = np.array([[0.,0.,R_EARTH+height*1e3]])
            dd = np.array([[np.cos(np.radians(app)),0.,np.sin(np.radians(app))]])
            initial = float(flight.mapping.refractive_index(height)*np.linalg.norm(np.cross(xx,dd)))
            maximum = 0.0
            for _ in range(200000):
                xx,dd,trial,ground,escaped = flight.advance(xx,dd,[step])
                q = float(flight.mapping._n(np.linalg.norm(xx))*np.linalg.norm(np.cross(xx,dd)))
                maximum = max(maximum,abs(q/initial-1))
                if ground[0] or escaped[0]:
                    break
            else:
                raise RuntimeError("invariant probe did not leave the atmosphere")
            path_invariants.append(dict(height_km=height,app_elevation_deg=app,max_step_m=step,
                                        maximum_relative_bouguer_error=maximum))

    angles = rng.uniform(0.5,90.,1000)
    direction = np.column_stack([np.cos(np.radians(angles)),np.zeros(1000),
                                  np.sin(np.radians(angles))])
    position = np.zeros((1000,3)); position[:,2] = R_EARTH
    distance = rng.exponential(50e3,1000)
    flight = ContinuousCurvedFlight(base,2000.)
    started = time.perf_counter()
    flight.advance(position,direction,distance)
    seconds = time.perf_counter()-started
    result = dict(head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,constant_index=constant,near_tangent_ground=event_rows,
        near_tangent_miss_correct=True,top_first_event_correct=True,
        radial_reference_ground_distance_m=arc,clone_smoke_finite=True,
        straight_global_majorant_mc_bit_exact=True,path_invariants=path_invariants,
        performance=dict(rays=1000,max_step_m=2000.,seconds=seconds,
            rk_ray_steps=flight.rk_ray_steps,turn_min_checks=flight.turn_min_checks))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(max_angle_error_deg=max(abs(r["angle_error_deg"]) for r in rows),
        max_bouguer_error=max(abs(r["relative_bouguer_error"]) for r in rows),
        near_tangent=event_rows,radial_ground_distance_m=arc,
        performance=result["performance"]),indent=2))


if __name__ == "__main__":
    main()
