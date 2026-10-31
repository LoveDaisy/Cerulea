"""Continuous species columns on refracted solar rays, by radial quadrature.

This experimental reference integrates the actual density callables.  It
does not use shell chord lengths or shell-average densities.  The refractive
profile and turning-point solver are intentionally shared with
``RadialSolarMapping``; Cartesian RK4 density integrals provide the independent
path/integration check.  Run the recorded checks with::

    uv run python -m tools.continuous_solar_columns --out out/columns-validation.json

No production library or existing validation baseline is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from skycolor.constants import R_EARTH
from tools.radial_solar_mapping import RadialSolarMapping


class ContinuousSolarColumns:
    """Integrate continuous species densities along one family of solar rays.

    Parameters
    ----------
    atmo
        Supplies the physical densities, aerosol layers and model boundaries.
    mapping
        Optional ``RadialSolarMapping`` for the same refractive definition and
        opaque bottom.  Its spline and private turning-point solver are reused
        explicitly; column integration is separate from its curvature integral.
    profile
        Optional r/n samples, passed to a newly constructed mapping.  This is
        useful for independent constant-index limiting cases.  It cannot be
        combined with ``mapping``.
    n_quad, segment_km, chunk
        Gauss nodes per physical-height segment, segment size in km and maximum
        number of rays processed together.  All ray/species samples are batched;
        only the small species list is iterated in Python.

    Notes
    -----
    Air/O3/NO2 columns are molecules m^-2; aerosol columns are optical depth at
    550 nm, in the same order as ``skycolor.optics``.  Integration stops at the
    model top, even if the refractive profile extends much further.  The
    supplied medium must have monotonic n*r, as required by the mapping.
    """

    def __init__(self, atmo, *, mapping=None, profile=None, n_quad: int = 8,
                 segment_km: float = 0.5, chunk: int = 128):
        if mapping is not None and profile is not None:
            raise ValueError("pass mapping or profile, not both")
        if n_quad < 2 or segment_km <= 0 or chunk < 1:
            raise ValueError("n_quad>=2, segment_km>0 and chunk>=1 are required")
        self.atmo = atmo
        self.mapping = mapping if mapping is not None else RadialSolarMapping(
            atmo, profile=profile, n_quad=n_quad, segment_km=segment_km, chunk=chunk)
        self.r_ground = float(atmo.r_bounds[0])
        self.r_top = float(atmo.r_bounds[-1])
        if self.mapping.r_ground != self.r_ground or self.r_top > self.mapping.r_top:
            raise ValueError("mapping must share the ground and cover the optical model")
        self.n_quad = int(n_quad)
        self.chunk = int(chunk)
        self.n_species = 3 + len(atmo.aerosols)
        h0, htop = (np.array([self.r_ground, self.r_top]) - R_EARTH) / 1e3
        # Density/temperature formula boundaries are integration boundaries,
        # not a piecewise-constant approximation to those densities.
        levels = np.unique(np.concatenate([
            np.arange(h0, htop, segment_km), np.asarray(atmo.z_bounds, float),
            [11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852, 86.0], [h0, htop],
        ]))
        levels = levels[(levels >= h0) & (levels <= htop)]
        self._edges = R_EARTH + levels * 1e3
        self._gx, self._gw = np.polynomial.legendre.leggauss(self.n_quad)

    def _density_columns(self, height):
        yield self.atmo.n_air(height)
        yield self.atmo.n_o3(height)
        yield self.atmo.n_no2(height)
        for layer in self.atmo.aerosols:
            yield layer.aod550 * layer.profile(height) / 1000.0

    def _integral(self, lower, upper, invariant, initial_gap):
        """One radial leg; r=lower+u² regularises a tangent at lower."""
        lower = np.asarray(lower)
        upper = np.broadcast_to(upper, lower.shape)
        lo = np.minimum(np.maximum(self._edges[:-1][None, :], lower[:, None]),
                        upper[:, None])
        hi = np.minimum(np.maximum(self._edges[1:][None, :], lower[:, None]),
                        upper[:, None])
        ulo = np.sqrt(np.maximum(lo - lower[:, None], 0.0))
        uhi = np.sqrt(np.maximum(hi - lower[:, None], 0.0))
        half = 0.5 * (uhi - ulo)
        u = 0.5 * (uhi + ulo)[..., None] + half[..., None] * self._gx
        delta = u * u
        radius = lower[:, None, None] + delta
        n = self.mapping._n(radius)
        q = n * radius
        qdiff = q - invariant[:, None, None]
        # The mapping uses the same stable q(r)-a near a tangent.  Sharing
        # that local geometry avoids inventing a second turning-point rule.
        nb = self.mapping._n(lower)
        dnb = self.mapping._dn(lower)
        q1 = nb + lower * dnb
        q2 = 2.0 * dnb + lower * self.mapping._d2n(lower)
        stable = initial_gap[:, None, None] + delta * (
            q1[:, None, None] + 0.5 * delta * q2[:, None, None])
        qdiff = np.where(delta < 0.1, stable, qdiff)
        denominator = np.sqrt(np.maximum(
            qdiff * (q + invariant[:, None, None]), 1e-300))
        # ds/dr = n*r/sqrt((n*r)^2-a^2); multiply by dr/du=2u.
        weight = q / denominator * (2.0 * u) * self._gw * half[..., None]
        height = (radius - R_EARTH) / 1e3
        result = np.empty((len(lower), self.n_species))
        for index, density in enumerate(self._density_columns(height)):
            result[:, index] = np.sum(np.asarray(density) * weight, axis=(1, 2))
        return result

    def columns(self, height_km, apparent_elevation_deg) -> np.ndarray:
        """Columns on forward rays, shape ``broadcast_shape + (n_species,)``.

        Heights are km and apparent elevations degrees.  Starts must be inside
        the optical domain.  A ray intersecting the opaque ground returns NaN
        for every species; an unblocked ray at the top pointing out returns 0.
        """
        height, app = np.broadcast_arrays(np.asarray(height_km, dtype=float),
                                          np.asarray(apparent_elevation_deg, dtype=float))
        radius = R_EARTH + height.ravel() * 1e3
        ea = app.ravel()
        if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(ea)):
            raise ValueError("height and apparent elevation must be finite")
        if np.any(radius < self.r_ground - 1e-7) or np.any(radius > self.r_top + 1e-7):
            raise ValueError("ray start lies outside the optical model")
        if np.any(np.abs(ea) > 90.0):
            raise ValueError("apparent elevation must lie in [-90, 90] degrees")
        q0 = self.mapping._n(radius) * radius
        invariant = q0 * np.cos(np.radians(ea))
        down = ea < 0.0
        valid = (~down) | (invariant > self.mapping.q_ground)
        result = np.full((ea.size, self.n_species), np.nan)
        for start in range(0, ea.size, self.chunk):
            index = np.arange(start, min(start + self.chunk, ea.size))
            index = index[valid[index]]
            if not len(index):
                continue
            descending = down[index]
            lower = radius[index].copy()
            gap = 2.0 * q0[index] * np.sin(0.5 * np.radians(ea[index]))**2
            if descending.any():
                lower[descending] = self.mapping._turning_radius(invariant[index[descending]])
                gap[descending] = 0.0
            value = self._integral(lower, self.r_top, invariant[index], gap)
            if descending.any():
                value[descending] += self._integral(
                    lower[descending], radius[index[descending]],
                    invariant[index[descending]], gap[descending])
            result[index] = value
        return result.reshape(height.shape + (self.n_species,))


def main():
    """Reproduce RK comparisons, constant-index limits and warm timings."""
    from skycolor import presets
    from skycolor.atmosphere import Atmosphere
    from skycolor.optics import species_extinction_weights

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "validation/data/solar_source_audit/geometry-probe.json")
                           .read_text())
    base = presets.clear([550.0])
    rows = []
    for ground in [0.0, 2.4]:
        atmo = base if ground == 0 else base.at_altitude(ground)
        selected = [r for r in reference["rows"] if r["ground_km"] == ground
                    and r["true_zenith_deg"] == 96.0
                    and r["height_km"] in [30.0, 38.0, 40.0, 47.0]
                    and r["rk_root"] is not None]
        heights = np.array([r["height_km"] for r in selected])
        apparent = np.array([r["rk_root"]["apparent_elevation_deg"] for r in selected])
        extinction = species_extinction_weights(atmo)[:, 0]
        for n_quad in [4, 8, 16]:
            tool = ContinuousSolarColumns(atmo, n_quad=n_quad)
            columns = tool.columns(heights, apparent)
            for ref, value in zip(selected, columns):
                rk = ref["rk_root"]["rk"]
                ref_columns = np.array(rk["species_columns"])
                relative = np.divide(value - ref_columns, ref_columns,
                    out=np.zeros_like(value), where=ref_columns != 0.0)
                tau = float(value @ extinction)
                transmittance = float(np.exp(-tau))
                rows.append(dict(ground_km=ground, height_km=ref["height_km"],
                    apparent_elevation_deg=ref["rk_root"]["apparent_elevation_deg"],
                    n_quad=n_quad, columns=value.tolist(),
                    rk_columns=ref_columns.tolist(), column_relative_error=relative.tolist(),
                    tau550=tau, transmittance550=transmittance,
                    rk_transmittance550=rk["transmittance550"],
                    transmittance_relative_error=transmittance/rk["transmittance550"] - 1.0))

    analytic = []
    density = 2.0e20
    for index in [1.0, 1.000278]:
        atmo = Atmosphere([550.], aerosols=[], ozone_du=0., no2_column=0.,
                         air_profile=lambda z: np.full_like(np.asarray(z, float), density))
        profile = SimpleNamespace(r=np.linspace(R_EARTH, R_EARTH + 1.2e6, 4000),
                                  n=np.full(4000, index))
        tool = ContinuousSolarColumns(atmo, profile=profile, n_quad=16)
        heights = np.array([0., 40., 40., 120.])
        apparent = np.array([90., 20., -6., 1.])
        radius = R_EARTH + heights * 1e3
        impact = radius * np.cos(np.radians(apparent))
        # Straight Cartesian ray intersection, independent of shell helpers.
        path = -radius * np.sin(np.radians(apparent)) + np.sqrt(
            np.maximum(tool.r_top**2 - impact**2, 0.))
        value = tool.columns(heights, apparent)
        expected = path * density
        relative = np.divide(value[:, 0] - expected, expected,
                             out=np.zeros_like(expected), where=expected != 0.)
        np.testing.assert_allclose(value[:, 0], expected, rtol=3e-9, atol=1e8)
        np.testing.assert_array_equal(value[:, 1:], 0.)
        blocked = tool.columns([30., 0.], [-6., -1.])
        assert np.isnan(blocked).all()
        analytic.append(dict(refractive_index=index,
            max_relative_air_error=float(np.max(np.abs(relative))),
            zero_species_exact=True, blocked_rays_all_nan=True))

    tool = ContinuousSolarColumns(base, n_quad=8)
    heights = np.full(1000, 40.0)
    apparent = np.linspace(-5.2, -5.8, 1000)
    tool.columns(heights[:2], apparent[:2])
    elapsed = []
    for _ in range(2):
        started = time.perf_counter()
        value = tool.columns(heights, apparent)
        elapsed.append(time.perf_counter() - started)
    result = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mapping_sha256=hashlib.sha256((root / "tools/radial_solar_mapping.py").read_bytes())
                              .hexdigest(),
        reference="validation/data/solar_source_audit/geometry-probe.json",
        rows=rows, constant_index_checks=analytic,
        performance=dict(rays=1000, n_quad=8, seconds=elapsed,
                         finite_all_species=bool(np.isfinite(value).all())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(
        max_relative_t550_error=max(abs(r["transmittance_relative_error"]) for r in rows),
        constant_index_checks=analytic, performance=result["performance"]), indent=2))


if __name__ == "__main__":
    main()
