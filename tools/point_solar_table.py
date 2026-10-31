"""Experimental shadow-aligned tables for a continuous refracted point sun.

The two angular endpoints are preserved by storing normalized true/app angle
distances from the ground-grazing ray. Height interpolation is linear in
sqrt(height above ground). The local beam gain is differentiated from that
same angular interpolant; it is never an independently interpolated field.

Only single-image atmospheres with monotonic n*r are supported. These tables
are explicit per-experiment artifacts, not a general custom-profile cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import PchipInterpolator

from skycolor.constants import R_EARTH
from skycolor.optics import species_extinction_weights
from skycolor.refraction import RefractiveProfile
from tools.continuous_solar_columns import ContinuousSolarColumns
from tools.radial_solar_mapping import RadialSolarMapping


def _signature(atmo):
    """Readable, sampled experimental definition; not a callable identity proof."""
    z = np.unique(np.concatenate([np.asarray(atmo.z_bounds),
                                 np.linspace(atmo.z_bounds[0], atmo.z_bounds[-1], 1025)]))
    values = [z, np.asarray(atmo.wl), np.asarray(atmo.z_bounds), atmo.n_air(z),
              atmo.n_o3(z), atmo.n_no2(z), species_extinction_weights(atmo)]
    values.extend(layer.aod550 * layer.profile(z) / 1000.0 for layer in atmo.aerosols)
    h = hashlib.sha256()
    for value in values:
        arr = np.ascontiguousarray(value, dtype=np.float64)
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return dict(sample_sha256=h.hexdigest(), n_species=3 + len(atmo.aerosols),
                ground_km=float(atmo.z_bounds[0]), top_km=float(atmo.z_bounds[-1]),
                wavelengths=np.asarray(atmo.wl).tolist(),
                limitation="Sampled signature does not prove arbitrary callable equality; "
                           "load only the explicitly labelled frozen experiment definition")


class PointSolarTable:
    """Per-definition continuous point-sun attenuation, direction and local gain.

    Query returns ``(app_deg, T, gain_local, lit)`` with broadcasting. With
    ``iw=None`` T has an additional final wavelength axis. Gain contains
    ``(n_point/n_inf)^2``; an adjoint detector-transfer factor is not included.
    Saved tables require the same explicit experiment label and medium when
    loaded; no automatic parameter cache or custom-callable deduplication is
    attempted.
    """

    def __init__(self, atmo, *, n_height=65, angle_tolerance_deg=2e-4,
                 derivative_tolerance=0.002, tau_tolerance=0.005, n_quad=8,
                 max_refine=20, label="explicit-standard-experiment", refracted=True,
                 profile=None, progress=False):
        started = time.perf_counter()
        self.atmo = atmo
        if not refracted and profile is not None:
            raise ValueError("Use refracted=False or an explicit profile, not both")
        if not refracted:
            samples = RefractiveProfile(atmo)
            profile = SimpleNamespace(r=samples.r, n=np.ones_like(samples.n))
        self.mapping = RadialSolarMapping(atmo, profile=profile, n_quad=n_quad)
        self.constant_index = bool(np.all(self.mapping.n == self.mapping.n_inf))
        self.column_tool = ContinuousSolarColumns(atmo, mapping=self.mapping, n_quad=n_quad)
        self.ground_km = float(atmo.z_bounds[0])
        self.top_km = float(atmo.z_bounds[-1])
        self.wl = np.asarray(atmo.wl, dtype=float)
        self.extinction = species_extinction_weights(atmo)
        ref = np.array([np.interp(550.0, self.wl, row) for row in self.extinction])
        self.column_scale = np.where(ref > 0, ref,
            np.maximum(np.max(self.extinction, axis=1), 1e-300))
        self.column_scale[self.column_scale == 0] = 1.0
        self.label = str(label)
        self.definition = _signature(atmo)
        self.parameters = dict(n_height=int(n_height), angle_tolerance_deg=angle_tolerance_deg,
            derivative_tolerance=derivative_tolerance, tau_tolerance=tau_tolerance,
            n_quad=int(n_quad), max_refine=int(max_refine), grazing_epsilon_deg=1e-6,
            refracted=bool(refracted), explicit_profile=profile is not None,
            low_height_grid="sqrt(h) step .05 below 1km, .1 from 1 to 5km")
        u = np.linspace(0, np.sqrt(self.top_km - self.ground_km), max(2, int(n_height)))
        special = np.asarray(atmo.z_bounds, dtype=float)
        special = special[(special >= self.ground_km) & (special <= self.top_km)]
        # A first heldout probe found up to 7% transmission error at tens of
        # metres above ground despite excellent angular interpolation. Near
        # grazing, a small column error is magnified exponentially; resolve
        # that observed low-height regime rather than certifying angles only.
        low = np.arange(0, min(1.0, u[-1]) + 1e-10, .05)
        middle = np.arange(1.0, min(np.sqrt(5.0), u[-1]) + 1e-10, .1)
        self.heights = np.unique(np.concatenate([
            self.ground_km + u*u, self.ground_km + low*low,
            self.ground_km + middle*middle, special]))
        self.height_axis = np.sqrt(self.heights - self.ground_km)
        self.critical = []
        self.rows = []
        self.build_diagnostics = []
        for height in self.heights:
            row, critical, diagnostic = self._build_row(float(height))
            self.rows.append(row)
            self.critical.append(critical)
            self.build_diagnostics.append(diagnostic)
            if progress:
                print(json.dumps(dict(label=self.label, **diagnostic)), flush=True)
        self.critical = np.asarray(self.critical)
        self.build_seconds = time.perf_counter() - started

    def _build_row(self, height):
        lower = float(self.mapping.grazing_apparent_elevation(height)) + 1e-6
        app = np.unique(np.concatenate([
            [lower, 0.0, 90.0], np.arange(lower, 0, 0.5),
            np.arange(0, 5.0001, .5), np.arange(7, 20.0001, 2), np.arange(30, 90, 10)]))
        app = app[(app >= lower) & (app <= 90)]
        true = self.mapping.forward(height, app)
        columns = self.column_tool.columns(height, app)
        critical = float(true[0])
        diagnostic = {}
        for iteration in range(self.parameters["max_refine"] + 1):
            inv = PchipInterpolator(true, app)
            col = PchipInterpolator(true, np.log1p(columns*self.column_scale), axis=0)
            midapp = .5*(app[:-1] + app[1:])
            midtrue = self.mapping.forward(height, midapp)
            midcols = self.column_tool.columns(height, midapp)
            predapp = inv(midtrue)
            # Compare point derivatives at the scale of the interval being
            # certified. A fixed .001-degree difference is a box average;
            # comparing it to arbitrarily fine PCHIP cells need not converge
            # at a narrow physical profile transition.
            step = np.minimum(.001, .2*(app[1:]-app[:-1]))
            step = np.minimum(np.maximum(step, 1e-5), .2*(midapp-lower))
            slope = (self.mapping.forward(height, midapp + step) -
                     self.mapping.forward(height, midapp - step)) / (2*step)
            derivative_error = np.abs(inv.derivative()(midtrue)*slope - 1)
            angle_error = np.abs(predapp - midapp)
            predcols = np.maximum(np.expm1(col(midtrue)), 0)/self.column_scale
            tau = midcols @ self.extinction
            tau_error = np.abs((predcols-midcols) @ self.extinction)
            # Extremely opaque rays are irrelevant to a direct source; the
            # log-column representation still preserves positivity there.
            active = tau < 25.0
            optical_error = np.max(np.where(active, tau_error, 0), axis=1)
            bad = ((angle_error > self.parameters["angle_tolerance_deg"]) |
                   (derivative_error > self.parameters["derivative_tolerance"]) |
                   (optical_error > self.parameters["tau_tolerance"]))
            diagnostic = dict(height=height, knots=len(app), refinement_rounds=iteration,
                max_midpoint_angle_error_deg=float(np.max(angle_error)),
                max_midpoint_derivative_relative_error=float(np.max(derivative_error)),
                max_midpoint_optical_depth_error=float(np.max(optical_error)),
                unresolved_intervals=int(np.count_nonzero(bad)))
            if not bad.any():
                break
            if iteration == self.parameters["max_refine"]:
                worst = np.argsort(derivative_error)[-5:]
                detail = dict(worst_app_intervals=np.column_stack([
                    app[:-1][worst], app[1:][worst]]).tolist(),
                    worst_derivative_error=derivative_error[worst].tolist())
                raise RuntimeError(f"Source row did not meet interpolation tolerances: {diagnostic}; {detail}")
            # PCHIP endpoint slopes depend on neighbouring intervals. Keep
            # their resolution balanced instead of repeatedly bisecting only
            # one side of a rapidly varying angular derivative.
            marked = bad.copy()
            marked[:-1] |= bad[1:]
            marked[1:] |= bad[:-1]
            bad = marked
            app = np.concatenate([app, midapp[bad]])
            true = np.concatenate([true, midtrue[bad]])
            columns = np.concatenate([columns, midcols[bad]])
            order = np.argsort(app)
            app, true, columns = app[order], true[order], columns[order]
        t = (true-critical)/(90-critical)
        d = (app-lower)/(90-lower)
        direction = PchipInterpolator(t, d)
        column = PchipInterpolator(t, np.log1p(columns*self.column_scale), axis=0)
        return dict(t=t, direction=direction.c, columns=column.c), critical, diagnostic

    @staticmethod
    def _evaluate_row(row, t):
        i = np.clip(np.searchsorted(row["t"], t, side="right")-1, 0, len(row["t"])-2)
        dx = t-row["t"][i]
        c = row["direction"][:, i]
        value = ((c[0]*dx+c[1])*dx+c[2])*dx+c[3]
        derivative = (3*c[0]*dx+2*c[1])*dx+c[2]
        c = row["columns"][:, i, :]
        x = dx[:, None]
        columns = ((c[0]*x+c[1])*x+c[2])*x+c[3]
        return value, derivative, columns

    def query_fields(self, height_km, true_elevation_deg):
        """Return app, continuous species columns, local gain and lit mask."""
        z, true = np.broadcast_arrays(np.asarray(height_km, float),
                                     np.asarray(true_elevation_deg, float))
        shape = z.shape
        zz, et = z.ravel(), true.ravel()
        if np.any(~np.isfinite(zz)) or np.any(~np.isfinite(et)):
            raise ValueError("height and true elevation must be finite")
        if np.any(zz < self.ground_km-1e-8) or np.any(zz > self.top_km+1e-8):
            raise ValueError("source query lies outside the model")
        if np.any(np.abs(et) > 90+1e-8):
            raise ValueError("true solar elevation must be within [-90,90]")
        zz = np.clip(zz, self.ground_km, self.top_km)
        et = np.clip(et, -90, 90)
        u = np.sqrt(zz-self.ground_km)
        i = np.clip(np.searchsorted(self.height_axis, u, side="right")-1,
                    0, len(self.height_axis)-2)
        w = (u-self.height_axis[i])/(self.height_axis[i+1]-self.height_axis[i])
        critical = (1-w)*self.critical[i] + w*self.critical[i+1]
        lower = self.mapping.grazing_apparent_elevation(zz) + 1e-6
        if self.constant_index:
            # The identity angular map and Cartesian ground shadow are
            # analytic in a uniform index, including a non-vacuum far field.
            # Do not contaminate that limiting case with height interpolation.
            critical = lower-1e-6
        t = (et-critical)/(90-critical)
        lit = t >= 0
        app = lower-.001
        gain = np.zeros_like(app)
        columns = np.zeros((len(zz), len(self.column_scale)))
        if lit.any():
            ind = np.flatnonzero(lit)
            value = np.zeros(len(ind))
            derivative = np.zeros(len(ind))
            col = np.zeros((len(ind), len(self.column_scale)))
            for side in (0, 1):
                rows = i[ind]+side
                weight = (1-w[ind]) if side == 0 else w[ind]
                for row_index in np.unique(rows):
                    ids = np.flatnonzero(rows == row_index)
                    val, der, cc = self._evaluate_row(self.rows[row_index], t[ind[ids]])
                    value[ids] += weight[ids]*val
                    derivative[ids] += weight[ids]*der
                    col[ids] += weight[ids, None]*cc
            app[ind] = lower[ind] + (90-lower[ind])*value
            slope = (90-lower[ind])/(90-critical[ind])*derivative
            n = self.mapping.refractive_index(zz[ind])
            near_vertical = np.abs(et[ind]-90) < 1e-6
            ratio = np.divide(np.cos(np.radians(app[ind])), np.cos(np.radians(et[ind])),
                              out=slope.copy(), where=~near_vertical)
            gain[ind] = (n/self.mapping.n_inf)**2 * ratio * slope
            if self.constant_index:
                app[ind] = et[ind]
                gain[ind] = 1.0
            columns[ind] = np.maximum(np.expm1(col), 0)/self.column_scale
        return (app.reshape(shape), columns.reshape(shape+(len(self.column_scale),)),
                gain.reshape(shape), lit.reshape(shape))

    def query(self, height_km, true_elevation_deg, iw=None):
        """Return (app_deg, T, gain_local, lit); iw=None adds all wavelengths."""
        app, columns, gain, lit = self.query_fields(height_km, true_elevation_deg)
        if iw is None:
            tau = columns @ self.extinction
            transmittance = np.where(lit[..., None], np.exp(-tau), 0)
        else:
            if not 0 <= int(iw) < len(self.wl):
                raise IndexError("wavelength index outside the source table")
            tau = columns @ self.extinction[:, int(iw)]
            transmittance = np.where(lit, np.exp(-tau), 0)
        return app, transmittance, gain, lit

    def save(self, path):
        """Write an explicit experiment table; no pickled objects are stored."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(heights=self.heights, critical=self.critical,
            extinction=self.extinction, column_scale=self.column_scale,
            profile_r=self.mapping.r, profile_n=self.mapping.n)
        for i, row in enumerate(self.rows):
            for name, value in row.items():
                arrays[f"row_{i}_{name}"] = value
        arrays["metadata"] = np.array(json.dumps(dict(label=self.label,
            definition=self.definition, parameters=self.parameters,
            build_seconds=self.build_seconds, build_diagnostics=self.build_diagnostics)))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, atmo, *, expected_label):
        """Load only an explicitly labelled frozen experiment definition."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["label"] != expected_label or meta["definition"] != _signature(atmo):
                raise ValueError("Saved solar table does not match this labelled experiment")
            obj = cls.__new__(cls)
            obj.atmo, obj.label = atmo, expected_label
            profile = SimpleNamespace(r=data["profile_r"].copy(), n=data["profile_n"].copy())
            obj.mapping = RadialSolarMapping(atmo, profile=profile,
                                            n_quad=meta["parameters"]["n_quad"])
            obj.constant_index = bool(np.all(obj.mapping.n == obj.mapping.n_inf))
            obj.column_tool = ContinuousSolarColumns(atmo, mapping=obj.mapping,
                n_quad=meta["parameters"]["n_quad"])
            for name in ["heights", "critical", "extinction", "column_scale"]:
                setattr(obj, name, data[name].copy())
            obj.wl = np.asarray(atmo.wl, float)
            obj.ground_km, obj.top_km = float(atmo.z_bounds[0]), float(atmo.z_bounds[-1])
            obj.height_axis = np.sqrt(obj.heights-obj.ground_km)
            obj.parameters, obj.definition = meta["parameters"], meta["definition"]
            obj.build_seconds, obj.build_diagnostics = meta["build_seconds"], meta["build_diagnostics"]
            obj.rows = [{name: data[f"row_{i}_{name}"].copy()
                         for name in ["t", "direction", "columns"]}
                        for i in range(len(obj.heights))]
        return obj


def main():
    """Build/load an explicitly labelled pilot and evaluate independent holdouts."""
    import argparse
    from scipy.optimize import brentq
    from skycolor import presets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--n-height", type=int, default=17)
    parser.add_argument("--geometric", action="store_true")
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[450., 550., 650.])
    args = parser.parse_args()
    atmo = presets.clear(args.wavelengths)
    if args.table.exists():
        table = PointSolarTable.load(args.table, atmo, expected_label=args.label)
    else:
        table = PointSolarTable(atmo, n_height=args.n_height, label=args.label,
                                refracted=not args.geometric, progress=True)
        table.save(args.table)
    points = []
    for height in [0.037, 1.337, 10.37, 29.37, 37.3, 41.37, 69.3, 89.3, 150.37, 199.9]:
        if not table.ground_km <= height <= table.top_km:
            continue
        lower = float(table.mapping.grazing_apparent_elevation(height)) + 1e-6
        crit = float(table.mapping.forward(height, lower))
        for true in [-6., -9., crit-.001, crit+.001, crit+.05, crit+.5, 20., 89.9, 90.]:
            points.append((height, true, lower, crit))
    reference = []
    for z, true, lower, crit in points:
        if true < crit:
            reference.append(dict(height=z, true_elevation=true, lit=False,
                                  app=None, gain=0., columns=np.zeros(len(table.column_scale))))
            continue
        app = (90.0 if true == 90 else brentq(
            lambda e: float(table.mapping.forward(z, e))-true, lower, 90., xtol=2e-9))
        lo, hi = max(app-.0001, lower), min(app+.0001, 90.)
        invder = (hi-lo)/float(table.mapping.forward(z, hi)-table.mapping.forward(z, lo))
        n = float(table.mapping.refractive_index(z))
        cosine_ratio = invder if true == 90 else np.cos(np.radians(app))/np.cos(np.radians(true))
        gain = (n/table.mapping.n_inf)**2 * cosine_ratio * invder
        reference.append(dict(height=z, true_elevation=true, lit=True, app=float(app),
                              gain=float(gain), columns=table.column_tool.columns(z, app)))
    z = np.array([p[0] for p in points])
    true = np.array([p[1] for p in points])
    app, transmission, gain, lit = table.query(z, true)
    rows = []
    for i, ref in enumerate(reference):
        expected_t = np.exp(-ref["columns"] @ table.extinction) if ref["lit"] else np.zeros(len(table.wl))
        active = expected_t > 1e-9
        relative_t = np.divide(transmission[i]-expected_t, expected_t,
                               out=np.zeros_like(expected_t), where=active)
        effective = expected_t >= 1e-4
        weighted = transmission[i]*gain[i]
        weighted_ref = expected_t*ref["gain"]
        source_relative = np.divide(weighted-weighted_ref, weighted_ref,
            out=np.zeros_like(weighted_ref), where=weighted_ref >= 1e-4)
        row = {k: v for k, v in ref.items() if k != "columns"}
        row.update(table_lit=bool(lit[i]), table_app=float(app[i]), table_gain=float(gain[i]),
            app_error_deg=float(app[i]-ref["app"]) if lit[i] and ref["lit"] else None,
            gain_relative_error=float(gain[i]/ref["gain"]-1) if lit[i] and ref["gain"] else None,
            transmission=transmission[i].tolist(), reference_transmission=expected_t.tolist(),
            max_active_transmission_relative_error=float(np.max(np.abs(relative_t))),
            max_effective_transmission_relative_error=float(np.max(
                np.where(effective, np.abs(relative_t), 0))),
            max_absolute_transmission_error=float(np.max(np.abs(transmission[i]-expected_t))),
            max_effective_source_relative_error=float(np.max(np.abs(source_relative))),
            max_absolute_source_error=float(np.max(np.abs(weighted-weighted_ref))))
        rows.append(row)
    rng = np.random.default_rng(41)
    count = 100_000
    z_perf = table.ground_km + (table.top_km-table.ground_km)*rng.random(count)**2
    true_perf = rng.uniform(-15, 89.9, count)
    started = time.perf_counter()
    table.query(z_perf, true_perf, iw=0)
    elapsed = time.perf_counter()-started
    loaded = PointSolarTable.load(args.table, atmo, expected_label=args.label)
    after = loaded.query(z, true)
    roundtrip = all(np.array_equal(a, b) for a, b in zip((app, transmission, gain, lit), after))
    assert roundtrip
    accepted = [r for r in rows if r["lit"] and r["table_lit"]]
    result = dict(parameters=table.parameters, height_rows=len(table.heights),
        total_angular_knots=sum(len(row["t"]) for row in table.rows), build_seconds=table.build_seconds,
        build_diagnostics=table.build_diagnostics, validation_rows=rows,
        max_app_error_deg=max(abs(r["app_error_deg"]) for r in accepted),
        max_gain_relative_error=max(abs(r["gain_relative_error"]) for r in accepted),
        max_transmission_relative_error=max(r["max_active_transmission_relative_error"] for r in rows),
        max_effective_transmission_relative_error=max(
            r["max_effective_transmission_relative_error"] for r in rows),
        max_absolute_transmission_error=max(r["max_absolute_transmission_error"] for r in rows),
        max_effective_source_relative_error=max(r["max_effective_source_relative_error"] for r in rows),
        max_absolute_source_error=max(r["max_absolute_source_error"] for r in rows),
        shadow_disagreements=sum(r["lit"] != r["table_lit"] for r in rows),
        roundtrip_bit_identical=roundtrip, query_100k_seconds=elapsed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ["validation_rows", "build_diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
