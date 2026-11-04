"""Correctness tests for the exactly-two-scattering candidate estimator.

The plan (task-deep-twilight-reference Step 3) requires, before any effect
claim:

1. zero scattering must return exactly 0;
2. a thin pure-scattering atmosphere must scale with the square of the
   density (all transmittances ~ 1, albedo ~ 1 -> L2 ~ b_sca^2);
3. an n=1-style uniform slab with isotropic-like kernels must close against
   an independent nested quadrature over (z1, r, mu, phi) -- the reference
   computes the volume integral in Earth-centred spherical coordinates with
   its own dense-sampled occlusion test and shares *no weight-derivation
   code* with the proposal;
4. the real-field adapter must reproduce the analog Monte Carlo's per-path
   exactly-two-scatter-no-ground component (diagnostics sink) at -6 deg,
   multi-seed, within combined Monte Carlo error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor.constants import R_EARTH
from tools.twilight_second_order import (
    SecondOrderFields, second_order_samples, sun_hat_for, zenith_view_tau)

WL3 = np.array([450.0, 550.0, 650.0])


# ---------------------------------------------------------------------------
# synthetic uniform-slab scenario (constant b_sca, pure scattering)
# ---------------------------------------------------------------------------
class SyntheticScenario:
    """Constant-coefficient slab with d2/mu1-sensitive synthetic kernels.

    ``illum(x2, d2) = exp(-kappa z2) * (1 + 0.5 (d2 . sun_hat))`` and
    ``phase_mix(mu1, z1) = 1 + 0.5 mu1`` exercise both direction channels;
    ``f0`` and all normalisations are internal to the scenario (the sampler
    must reproduce whatever convention the fields define).
    """

    def __init__(self, beta=5.0e-5, z_top_km=20.0, kappa=2.0e-5,
                 sun_elevation=-9.0):
        self.beta = beta                      # b_sca = b_ext [1/m]
        self.z_top_km = z_top_km
        self.kappa = kappa                    # illumination decay [1/m]
        self.sun_hat = sun_hat_for(sun_elevation)
        self.f0 = 1.7e-3
        self.calls = {"illum": 0, "chord": 0}

    def fields(self):
        beta, kappa = self.beta, self.kappa

        def illumination(x2, prop_dir, rf, af):
            # prop_dir = unit(x1 -> x2), the backward-path propagation dir
            self.calls["illum"] += 1
            z2 = (np.sqrt(np.einsum("ij,ij->i", x2, x2)) - R_EARTH) / 1e3
            return np.exp(-kappa * z2 * 1e3) * (1.0 + 0.5 * (prop_dir @ self.sun_hat))

        def chord(x1, x2):
            self.calls["chord"] += 1
            diff = x2 - x1
            length = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            # independent occlusion test: dense midpoint minimum radius
            t = np.linspace(0.0, 1.0, 201)[None, :, None]
            pts = x1[:, None, :] + t * diff[:, None, :]
            radius = np.sqrt(np.einsum("ijk,ijk->ij", pts, pts))
            blocked = radius.min(axis=1) < R_EARTH
            return np.where(blocked, 0.0, np.exp(-beta * length))

        return SecondOrderFields(
            z_top_km=self.z_top_km, ground_radius=R_EARTH,
            top_radius=R_EARTH + self.z_top_km * 1e3,
            t_view=lambda z: np.exp(-beta * z * 1e3),
            b_sca=lambda z: np.full(np.shape(z), beta),
            b_ext=lambda z: np.full(np.shape(z), beta),
            phase_mix=lambda mu, z: 1.0 + 0.5 * np.asarray(mu),
            illumination=illumination,
            ray_fraction=lambda z: np.ones(np.shape(z)),
            aerosol_fractions=lambda z: np.zeros(np.shape(z) + (0,)),
            chord_transmittance=chord,
            f0=self.f0)

    # ---- independent reference (nested Gauss over angles and path) --------
    # x1-centred product quadrature: Gauss-Legendre over the direction cos
    # mu, azimuth phi and the path length s in [0, S_max(direction)], with
    # all weight factors (dmu * dphi * ds cascade measure -- no Jacobian:
    # the x2 integral is per direction per unit path, not per volume --
    # and the S_max root geometry) re-derived here from scratch.  Shares
    # with the proposal only the *problem definition*: the injected field
    # callables and the model domain radii -- no proposal weight code.
    def reference(self, n_z=32, n_mu=64, n_phi=48, n_s=48):
        f = self.fields()
        ground, top = self.z_top_km * 0.0 + R_EARTH, R_EARTH + self.z_top_km * 1e3
        z_nodes, z_w = np.polynomial.legendre.leggauss(n_z)
        z1 = 0.5 * self.z_top_km * (z_nodes + 1.0)
        dz = 0.5 * self.z_top_km * z_w
        mu_nodes, mu_w = np.polynomial.legendre.leggauss(n_mu)
        phis = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
        dphi = 2.0 * np.pi / n_phi
        st = np.sqrt(np.maximum(1.0 - mu_nodes ** 2, 0.0))
        omegas = np.stack([np.repeat(st, n_phi) * np.tile(np.sin(phis), n_mu),
                           np.repeat(st, n_phi) * np.tile(np.cos(phis), n_mu),
                           np.repeat(mu_nodes, n_phi)], axis=1)
        dir_w = np.repeat(mu_w, n_phi) * dphi  # (n_mu*n_phi,)
        n_dir = omegas.shape[0]
        gx, gw = np.polynomial.legendre.leggauss(n_s)
        total = 0.0
        chunk = 512  # directions per batch to bound memory
        for iz in range(n_z):
            x1 = np.array([0.0, 0.0, R_EARTH + z1[iz] * 1e3])
            r1 = float(np.linalg.norm(x1))
            for start in range(0, n_dir, chunk):
                om = omegas[start:start + chunk]
                dw = dir_w[start:start + chunk]
                bvec = om @ x1  # (n_dir,) distance-along-ray projection
                s_top = -bvec + np.sqrt(np.maximum(
                    bvec ** 2 + top ** 2 - r1 * r1, 0.0))
                disc = bvec ** 2 + ground ** 2 - r1 * r1
                raw = np.where(disc > 0.0,
                               -bvec - np.sqrt(np.maximum(disc, 0.0)), np.inf)
                s_ground = np.where(raw > 0.0, raw, np.inf)
                s_max = np.minimum(s_top, s_ground)
                s_nodes = 0.5 * s_max[:, None] * (gx[None, :] + 1.0)
                s_weights = 0.5 * s_max[:, None] * gw[None, :]
                x2 = (x1[None, None, :]
                      + s_nodes[..., None] * om[:, None, :]).reshape(-1, 3)
                z2 = (np.sqrt(np.einsum("ij,ij->i", x2, x2)) - R_EARTH) / 1e3
                mu1 = np.repeat(om[:, 2], n_s)
                t12 = f.chord_transmittance(
                    np.broadcast_to(x1, x2.shape), x2)
                illum = f.illumination(x2, np.broadcast_to(
                    om[:, None, :], (om.shape[0], n_s, 3)).reshape(-1, 3),
                    np.ones(len(x2)), np.zeros((len(x2), 0)))
                integrand = (f.t_view(np.asarray(z1[iz]))
                             * f.b_sca(np.asarray(z1[iz])) * 1.0e3
                             * f.phase_mix(mu1, np.asarray(z1[iz]))
                             * t12 * f.b_sca(z2) * illum * f.f0)
                node_w = (s_weights.ravel()
                          * np.repeat(dw, n_s))
                total += float(np.dot(integrand, node_w)) * dz[iz]
        return total


@pytest.mark.parametrize("beta", [5.0e-5])
def test_uniform_slab_closes_against_independent_quadrature(beta):
    scenario = SyntheticScenario(beta=beta)
    fields = scenario.fields()
    n_samples = 240_000
    result = second_order_samples(fields, scenario.sun_hat, n_samples,
                                  seed=11, batch=60_000)
    ref = scenario.reference()
    se = np.sqrt(max(result["second_moment"] - result["L2"] ** 2, 0.0)
                 / result["n_samples"])
    quad_bias = 0.02  # midpoint phi grid on a mildly asymmetric integrand
    tol = max(4.0 * se, quad_bias * abs(ref))
    assert result["L2"] > 0.0
    # within the sampler's s-domain the ground limit is enforced by S_max
    # (rays stop at the ground entry), so no chord can be Earth-shadowed;
    # the occlusion machinery itself is exercised in the dedicated test below
    assert result["occluded"] == 0
    assert abs(result["L2"] - ref) <= tol, (
        result["L2"], ref, se, result["occluded"], result["dark_illum"])


def test_segment_occlusion_and_chord_transmittance():
    """Chords whose perigee dips below the opaque Earth are blocked."""
    from tools.twilight_second_order import chord_transmittance, segment_occluded
    from skycolor import presets

    atmo = presets.clear(WL3)
    r_ground = float(atmo.r_bounds[0])
    top = np.array([0.0, 0.0, R_EARTH + 30e3])       # 30 km up
    # a chord to a point 2000 km away and *below* the start level passes
    # deep enough to dip under the surface (its perigee radius < r_ground)
    x_far = np.array([2.0e6, 0.0, 0.0])              # inside the Earth sphere
    assert bool(segment_occluded(top[None, :], x_far[None, :], r_ground)[0])
    assert float(chord_transmittance(atmo, 1, top[None, :], x_far[None, :])[0]) == 0.0
    # a straight horizontal chord does NOT follow the curvature: it stays at
    # (and rises above) its starting radius, so it is unblocked
    x_flat = top + np.array([2.0e6, 0.0, 0.0])
    assert not bool(segment_occluded(top[None, :], x_flat[None, :], r_ground)[0])
    # a short downward chord from 30 km stays above the surface: unblocked
    x_low = np.array([0.0, 0.0, R_EARTH + 5e3])
    assert not bool(segment_occluded(top[None, :], x_low[None, :], r_ground)[0])
    t_clear = float(chord_transmittance(atmo, 1, top[None, :], x_low[None, :])[0])
    assert 0.0 < t_clear <= 1.0
    # endpoints exactly on the surface never count as occluded (the sampler
    # stops rays at the ground entry, so equality is the boundary case)
    x_ground = np.array([0.0, 0.0, R_EARTH])
    assert not bool(segment_occluded(top[None, :], x_ground[None, :], r_ground)[0])


def test_zero_scattering_is_exactly_zero():
    scenario = SyntheticScenario()
    fields = scenario.fields()
    fields.b_sca = lambda z: np.zeros(np.shape(z))
    fields.b_ext = lambda z: np.ones(np.shape(z))
    result = second_order_samples(fields, scenario.sun_hat, 20_000, seed=5)
    assert result["sum"] == 0.0
    assert result["L2"] == 0.0


def test_thin_atmosphere_scales_with_density_squared():
    # beta small enough that beta * (longest chord ~ 1e6 m) << 1, so every
    # transmittance is ~1 and the integrand is exactly proportional to b_sca^2
    scenario1 = SyntheticScenario(beta=1.0e-7)
    scenario2 = SyntheticScenario(beta=2.0e-7)
    r1 = second_order_samples(scenario1.fields(), scenario1.sun_hat,
                              150_000, seed=21, batch=75_000)
    r2 = second_order_samples(scenario2.fields(), scenario2.sun_hat,
                              150_000, seed=21, batch=75_000)
    ratio = r2["L2"] / r1["L2"]
    rel1 = np.sqrt(max(r1["second_moment"] - r1["L2"] ** 2, 0.0)
                   / r1["n_samples"]) / r1["L2"]
    rel2 = np.sqrt(max(r2["second_moment"] - r2["L2"] ** 2, 0.0)
                   / r2["n_samples"]) / r2["L2"]
    tol = 4.0 * np.hypot(rel1, rel2) + 0.02  # + thin-limit transmittance bias
    assert abs(ratio - 4.0) <= tol, (ratio, rel1, rel2)


def test_zenith_view_tau_matches_fine_trapezoid():
    from skycolor import presets

    atmo = presets.clear(WL3)
    z = np.linspace(0.0, 50.0, 9)
    tau = zenith_view_tau(atmo, 1, z)
    zz = np.linspace(0.0, 50.0, 400_001)
    b_ray = atmo.n_air(zz) * atmo._sigma_ray[1]
    b_o3 = atmo.n_o3(zz) * atmo._sigma_o3[1]
    b_no2 = atmo.n_no2(zz) * atmo._sigma_no2[1]
    ext, _sca = atmo.aerosols[0].aerosol.relative_coefficients(atmo.wl)
    b_ae = (atmo.aerosols[0].aod550 * atmo.aerosols[0].profile(zz) / 1000.0
            * ext[1])
    b_ext = b_ray + b_o3 + b_no2 + b_ae          # [1/m], zz in km
    tau_ref = np.array([
        float(np.trapezoid(b_ext[zz <= zi], zz[zz <= zi])) * 1e3  # km -> m
        for zi in z])
    np.testing.assert_allclose(tau, tau_ref, rtol=2e-4)


@pytest.mark.parametrize("mc_seed,cand_seed", [(202, 303), (204, 305), (206, 307)])
def test_real_adapter_matches_mc_second_component(mc_seed, cand_seed):
    """-6 deg, 550 nm: candidate vs the MC's own analog c2nt component.

    The plan requires a MULTI-seed closure; each pair keeps its own result
    and must agree within its own combined Monte Carlo error.  This is a
    single-scenario, three-independent-repeat check -- not a tail-coverage
    certificate.
    """
    from skycolor import presets
    from skycolor.rt import mc
    from tools.point_solar_table import PointSolarTable
    from tools.twilight_second_order import estimate
    from tools.unified_solar_transport import (
        FiniteSolarSource, mc_solar_source)

    atmo = presets.clear(WL3)
    table = PointSolarTable.load(
        ROOT / "validation/data/deep_twilight_reference/probe/point-source.npz",
        atmo, expected_label="deep-twilight-plan-probe-450-550-650-clear-v1")
    source = FiniteSolarSource(atmo, table)
    with mc_solar_source(source):
        result = mc.mc_radiance_split(atmo, 96.0, [0.0], [0.0],
                                      n_photons=300_000, seed=mc_seed,
                                      diagnostics=True)
    n_paths = 300_000
    d = result["diagnostics"][1]["per_view"][0]
    c2nt_mean = d["s1"][1] / n_paths
    if c2nt_mean <= 0.0:
        pytest.skip("no analog second-scatter signal at this budget")
    c2nt_second = np.array(d["s2"])[1][1] / n_paths
    se_mc = np.sqrt(max(c2nt_second - c2nt_mean ** 2, 0.0) / n_paths)

    candidate = estimate(atmo, 1, FiniteSolarSource(atmo, table), -6.0,
                         120_000, seed=cand_seed)
    se_cand = np.sqrt(max(candidate["second_moment"] - candidate["L2"] ** 2, 0.0)
                      / candidate["n_samples"])
    tol = 5.0 * np.hypot(se_mc, se_cand) + 1e-30
    assert abs(candidate["L2"] - c2nt_mean) <= tol, (
        candidate["L2"], c2nt_mean, se_mc, se_cand)
