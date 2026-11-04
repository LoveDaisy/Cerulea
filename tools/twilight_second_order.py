"""Exactly-two-scattering candidate estimator for deep twilight (Step 3).

This module is a *bounded experiment* of task-deep-twilight-reference: it
targets the part of the multiple-scattering radiance that the analog Monte
Carlo attributes to paths with exactly two atmospheric scattering events and
no ground bounce, and estimates it as an explicit two-vertex volume integral
instead of relying on the rare analog paths.  The plan pre-registers one such
candidate; there is no second algorithm and no promised speedup -- the
candidate must first pass independent correctness checks, then compete with
the direct estimator at equal wall-clock, and a negative result is a valid
delivery.

Quantity (zenith view, sea-level observer, per wavelength ``iw``):

    L2 = f0 * INT dz1  T_v(z1) b_sca(x1) p_mix(mu1; x1)
              * INT dOmega2 INT ds2 T12(x1, x2) b_sca(x2) illum(x2, dir)

with

* ``x1 = (0, 0, R_E + z1)`` the first vertex on the zenith view ray,
* ``T_v`` view transmittance observer -> x1 (continuous-density
  antiderivative, the same construction ``single_zenith`` uses),
* ``mu1`` the scattering angle at x1 between the backward propagation
  direction (up, towards the observer) and the direction towards x2,
* ``p_mix`` the scattering mixture at x1 as a *probability density per
  steradian* -- the MC's 4pi-normalized phase functions divided by 4pi
  (in the MC this factor is absorbed by sampling the direction from the
  phase function, here it appears explicitly),
* ``T12`` chord transmittance x1 -> x2 including opaque-Earth occlusion
  (a chord whose perigee dips below the model bottom contributes zero),
* ``dir = (x2 - x1)/|x2 - x1|`` the backward-path propagation direction at
  x2 (the convention ``source.scatter`` and ``single_zenith`` use),
* ``illum(x2, dir)`` the *same* ``FiniteSolarSource.scatter`` kernel the MC
  NEE uses (solar angular integral of T*G*phase mixture with detector
  transfer) -- the candidate shares the reference's source contract,
* ``f0 = S(iw) / (4 pi)`` exactly as in the MC tracer.

There are no explicit single-scattering albedo factors: the analog MC
carries them as survival probabilities (``kill`` when absorption wins), so
they enter through the physical scattering densities ``b_sca`` that both
the MC sampling and this estimator integrate against.

The x2 integral is a *cascade* integral over solid angle and path length
``INT dOmega2 INT ds2`` -- the radiance cascade weights contributions per
direction per unit path, NOT per unit volume; the inverse-square distance
factor ``1/s2`` that relates this to a volume density appears only if one
rewrites ``dOmega2 ds2`` as ``dV / s2**2``.  The sampler draws (omega2, s2)
directly, so the weight carries no Jacobian beyond ``S_max`` inverting the
uniform path proposal.

Estimator: sequential importance sampling with strictly positive proposal
density everywhere,

    q(z1)     = 0.5 * Uniform[0, z_top] + 0.5 * TruncExp(z_exp, [0, z_top])
    q(omega2) = f_iso / (4 pi) + (1 - f_iso) * UniformCap(sun_hat, alpha)
    q(s2)     = Uniform[0, S_max(omega2)]

(the cap proposal only *concentrates* samples towards the sunlit side; the
isotropic component keeps every direction reachable, so the finite disk's
partially visible region near the horizon is never lost even though the
disk centre is below the horizon).  The full weight of one sample is

    w = [T_v(z1) b_sca(x1) / q(z1)]
      * [a(x1) p_mix(mu1; x1) / q(omega2)]
      * [T12 b_sca(x2) a(x2) illum(x2, d2) s2^2 S_max]
      * f0

where ``S_max`` inverts the uniform s2 proposal (``S_max`` is the distance
from x1 to leave the atmospheric shell: top exit or ground entry, whichever
comes first, so x2 always lies inside the model).
``illum`` and ``p_mix`` are normalised to 4 pi exactly as in the MC, so L2
is directly comparable to the MC's per-path second-scatter component
(the diagnostics sink's ``second_no_touch`` sums): the reference estimator
is ``MC_rest = MC_multiple - MC_second_no_touch`` *per path*, and the
combined total is ``single_det + L2 + mean(MC_rest)``.

Optical fields are injected (:class:`SecondOrderFields`) so the correctness
suite can evaluate closed-form scenarios (uniform slab, isotropic phase,
synthetic illumination) against an independent nested quadrature that shares
no weight-derivation code with the proposal.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable

import numpy as np

from skycolor import spectra
from skycolor.constants import R_EARTH
from skycolor.optics import species_extinction_weights

#: proposal defaults (documented normalisations, not tuned results)
Z_EXP_KM = 3.0              # truncated-exponential scale for the first vertex
F_ISOTROPIC = 0.5           # isotropic share of the omega2 proposal
CONE_HALF_ANGLE_DEG = 30.0  # sun-directed cap half-angle of the omega2 proposal
#: z1 is proposed in km while the first-vertex integrand T_v * b_sca is a
#: per-meter path integral; this factor converts dz1[km] to meters exactly once
Z1_KM_TO_M = 1.0e3


@dataclasses.dataclass
class SecondOrderFields:
    """Injected optical fields of the two-vertex integrand.

    All callables are vectorised over leading sample axes; heights are km
    above the model bottom, positions are Earth-centred metres.
    """

    z_top_km: float    # model top above the observer along the zenith
    ground_radius: float  # opaque-Earth radius [m] (model bottom)
    top_radius: float  # model top radius [m]
    t_view: Callable[[np.ndarray], np.ndarray]              # z [km] -> T
    b_sca: Callable[[np.ndarray], np.ndarray]               # z -> b_sca [1/m]
    b_ext: Callable[[np.ndarray], np.ndarray]               # z -> b_ext [1/m]
    phase_mix: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (mu, z) -> p_mix
    illumination: Callable[..., np.ndarray]
    # (x2 (n,3), prop_dir (n,3) = unit(x1->x2), ray_fraction (n,),
    #  aerosol_fractions (n, n_a)) -> (n,); prop_dir is the backward-path
    # propagation direction at x2, matching source.scatter's ``backward``
    ray_fraction: Callable[[np.ndarray], np.ndarray]        # z -> b_ray/b_sca
    aerosol_fractions: Callable[[np.ndarray], np.ndarray]   # z -> (n, n_a)
    chord_transmittance: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # (x1 (n,3), x2 (n,3)) -> T12 with opaque-Earth occlusion
    f0: float          # S(iw) / (4 pi)


# ---------------------------------------------------------------------------
# real-field adapter (zenith view, one wavelength, finite solar source)
# ---------------------------------------------------------------------------
def zenith_view_tau(atmo, iw: int, z: np.ndarray) -> np.ndarray:
    """Optical depth observer -> height ``z`` [km] along the zenith.

    Continuous-density antiderivative of the species columns -- the same
    construction as ``single_zenith`` (fine density PCHIP antiderivative, no
    shell quadrature).  Heights are measured from the model bottom.
    """
    from scipy.interpolate import PchipInterpolator

    bottom = float(atmo.elevation_km)
    top = float(atmo.z_bounds[-1])
    grid = np.unique(np.r_[np.arange(bottom, top, 0.025), atmo.z_bounds, top])
    grid = grid[grid >= bottom]
    fields = [atmo.n_air(grid), atmo.n_o3(grid), atmo.n_no2(grid)]
    fields += [lay.aod550 * lay.profile(grid) / 1000 for lay in atmo.aerosols]
    density = np.stack(fields, axis=-1)
    primitive = PchipInterpolator(grid, density, axis=0).antiderivative()
    columns = (primitive(np.asarray(z, float)) - primitive(bottom)) * 1000
    return columns @ species_extinction_weights(atmo)[:, int(iw)]


def _iw_optics(atmo, iw: int):
    """Per-wavelength scattering/absorption profiles on a fixed height grid."""
    z = np.linspace(float(atmo.z_bounds[0]), float(atmo.z_bounds[-1]), 2001)
    b_ray = atmo.n_air(z) * atmo._sigma_ray[iw]
    b_o3 = atmo.n_o3(z) * atmo._sigma_o3[iw]
    b_no2 = atmo.n_no2(z) * atmo._sigma_no2[iw]
    b_as = []
    b_ae = np.zeros_like(b_ray)
    for lay in atmo.aerosols:
        ext, sca = lay.aerosol.relative_coefficients(atmo.wl)
        w = lay.aod550 * lay.profile(z) / 1000.0
        b_as.append(w * sca[iw])
        b_ae = b_ae + w * ext[iw]
    b_sca = b_ray + sum(b_as, start=np.zeros_like(b_ray))
    b_ext = b_sca + b_o3 + b_no2 + b_ae
    return z, b_ray, b_as, b_sca, b_ext


def real_fields(atmo, iw: int, source, sun_hat: np.ndarray) -> SecondOrderFields:
    """Bind the integrand fields to the benchmark atmosphere and the *same*
    ``FiniteSolarSource`` instance the MC runs under.

    Profile pieces are evaluated by log-linear interpolation on a fixed
    height grid -- the same interpolation the MC tracer uses for its local
    optical properties, so the candidate sees the medium exactly as the MC
    does.
    """
    z_grid, b_ray, b_as, b_sca, b_ext = _iw_optics(atmo, iw)
    lb_ray = np.log(np.maximum(b_ray, 1e-300))
    lb_sca = np.log(np.maximum(b_sca, 1e-300))
    lb_ext = np.log(np.maximum(b_ext, 1e-300))
    lb_as = [np.log(np.maximum(v, 1e-300)) for v in b_as]
    wl_arr = np.array([float(atmo.wl[iw])])

    def _interp(z, log_values):
        return np.exp(np.interp(z, z_grid, log_values))

    def b_sca_of(z):
        return _interp(z, lb_sca)

    def b_ext_of(z):
        return _interp(z, lb_ext)

    def phase_mix(mu, z):
        # probability density per steradian: the MC's 4pi-normalized mixture
        # phase (spectra.rayleigh_phase + aerosol phases) divided by 4pi
        mix = spectra.rayleigh_phase(mu) * (_interp(z, lb_ray)
                                            / np.maximum(b_sca_of(z), 1e-300))
        for j, lb in enumerate(lb_as):
            p_j = np.asarray(atmo.aerosols[j].aerosol.phase(mu, wl_arr)).reshape(-1)
            mix = mix + p_j * (_interp(z, lb) / np.maximum(b_sca_of(z), 1e-300))
        return mix / (4.0 * np.pi)

    def ray_fraction_of(z):
        return _interp(z, lb_ray) / np.maximum(b_sca_of(z), 1e-300)

    def aerosol_fractions_of(z):
        return np.stack([_interp(z, lb) / np.maximum(b_sca_of(z), 1e-300)
                         for lb in lb_as], axis=-1)

    def illumination(x2, d2, rf, af):
        return source.scatter(x2, d2, sun_hat, iw, rf, af)

    return SecondOrderFields(
        z_top_km=float(atmo.z_bounds[-1] - atmo.z_bounds[0]),
        ground_radius=float(atmo.r_bounds[0]),
        top_radius=float(atmo.r_bounds[-1]),
        t_view=lambda z: np.exp(-zenith_view_tau(atmo, iw, z)),
        b_sca=b_sca_of,
        b_ext=b_ext_of,
        phase_mix=phase_mix,
        illumination=illumination,
        ray_fraction=ray_fraction_of,
        aerosol_fractions=aerosol_fractions_of,
        chord_transmittance=lambda x1, x2: chord_transmittance(atmo, iw, x1, x2),
        f0=float(spectra.solar_irradiance(wl_arr)[0] / (4.0 * np.pi)),
    )


def sun_hat_for(sun_elevation_deg: float, sun_azimuth_deg: float = 0.0) -> np.ndarray:
    s = np.radians(90.0 - sun_elevation_deg)
    a = np.radians(sun_azimuth_deg)
    return np.array([np.sin(s) * np.sin(a), np.sin(s) * np.cos(a), np.cos(s)])


# ---------------------------------------------------------------------------
# segment (chord) geometry: occlusion and transmittance
# ---------------------------------------------------------------------------
def segment_occluded(x1: np.ndarray, x2: np.ndarray, ground_radius: float):
    """True where the straight chord x1 -> x2 dips below the opaque Earth."""
    diff = x2 - x1
    length = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    d = diff / np.maximum(length[:, None], 1e-300)
    r1 = np.sqrt(np.einsum("ij,ij->i", x1, x1))
    r2 = np.sqrt(np.einsum("ij,ij->i", x2, x2))
    b = np.einsum("ij,ij->i", x1, d)
    t_peri = -b
    rp = np.sqrt(np.maximum(r1 * r1 - b * b, 0.0))
    inside = (t_peri > 0.0) & (t_peri < length)
    min_radius = np.where(inside, rp, np.minimum(r1, r2))
    return min_radius < ground_radius


def segment_shell_lengths(r_bounds: np.ndarray, x1: np.ndarray, d: np.ndarray,
                          length: np.ndarray):
    """Length of segments inside the *ball* of each boundary radius [m].

    ``x1`` (n, 3), unit ``d`` (n, 3), segment ``length`` (n,); returns
    ``(n, n_bounds)`` where column ``k`` is the path length inside the
    sphere of radius ``r_bounds[k]``.  Differencing consecutive columns
    yields per-shell lengths.  Entry/exit solutions of |x1 + s d| = r_k,
    clipped to the segment.
    """
    r1 = np.sqrt(np.einsum("ij,ij->i", x1, x1))
    b = np.einsum("ij,ij->i", x1, d)
    disc = b[:, None] ** 2 - (r1 ** 2)[:, None] + r_bounds[None, :] ** 2
    root = np.sqrt(np.maximum(disc, 0.0))
    s_in = np.clip(-b[:, None] - root, 0.0, length[:, None])
    s_out = np.clip(-b[:, None] + root, 0.0, length[:, None])
    return s_out - s_in


def chord_transmittance(atmo, iw: int, x1: np.ndarray, x2: np.ndarray):
    """Chord transmittance x1 -> x2; zero where Earth-shadowed.

    Shell discretisation and density evaluation follow the MC's
    ``_sun_transmittance`` convention (path length per shell, density at the
    shell geometric mean radius) so the candidate lives in the same
    discretised medium as the rest of the reference.
    """
    from skycolor.optics import shell_species_columns, shell_species_weights

    occluded = segment_occluded(x1, x2, atmo.r_bounds[0])
    diff = x2 - x1
    length = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    d = diff / np.maximum(length[:, None], 1e-300)
    # per-boundary clipped crossings give the length *inside each ball*;
    # differencing consecutive boundaries yields the per-shell lengths
    # (n, n_shell) that the shared column algebra expects as (n_shell, n)
    ball = segment_shell_lengths(atmo.r_bounds, x1, d, length)
    lengths = (ball[:, 1:] - ball[:, :-1]).T
    cols = shell_species_columns(lengths,
                                 shell_species_weights(atmo, atmo.r_bounds))
    tau = cols @ species_extinction_weights(atmo)[:, int(iw)]
    return np.where(occluded, 0.0, np.exp(-tau))


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------
def _proposal_z1(rng, n, z_top, z_exp):
    """Mixture 0.5*U[0,z_top] + 0.5*TruncExp(1/z_exp) on [0, z_top].

    Returns (z1, pdf); pdf is strictly positive on the whole interval.
    """
    use_uniform = rng.random(n) < 0.5
    z_uniform = rng.random(n) * z_top
    lam = 1.0 / z_exp
    norm = -np.expm1(-lam * z_top)
    z_truncexp = -np.log1p(-rng.random(n) * norm) / lam
    z = np.where(use_uniform, z_uniform, z_truncexp)
    pdf = 0.5 / z_top + 0.5 * lam * np.exp(-lam * z) / norm
    return z, pdf


def _angles_to_dirs(mu, phi):
    st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    return np.stack([st * np.sin(phi), st * np.cos(phi), mu], axis=1)


def _rotate_around(axis, mu, phi):
    """Unit vectors at angle acos(mu) from ``axis`` with azimuth ``phi``."""
    n = mu.shape[0]
    a = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    return (mu[:, None] * a[None, :]
            + st[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))


def _proposal_omega(rng, n, sun_hat, f_iso, cone_deg):
    """Isotropic + uniform-cap mixture on the sphere; returns (dirs, pdf).

    Cap density: 1 / (2 pi (1 - cos alpha)) inside the cap, else 0; the
    isotropic component keeps the proposal strictly positive everywhere.
    """
    use_iso = rng.random(n) < f_iso
    mu_iso = 1.0 - 2.0 * rng.random(n)
    d_iso = _angles_to_dirs(mu_iso, 2.0 * np.pi * rng.random(n))
    cos_alpha = np.cos(np.radians(cone_deg))
    mu_cap = 1.0 - rng.random(n) * (1.0 - cos_alpha)
    d_cap = _rotate_around(sun_hat, mu_cap, 2.0 * np.pi * rng.random(n))
    dirs = np.where(use_iso[:, None], d_iso, d_cap)
    in_cap = (dirs @ sun_hat) >= cos_alpha
    pdf = f_iso * 0.25 / np.pi + (1.0 - f_iso) * in_cap / (
        2.0 * np.pi * (1.0 - cos_alpha))
    return dirs, pdf


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
def second_order_samples(fields: SecondOrderFields, sun_hat: np.ndarray,
                         n_samples: int, seed: int, *, batch: int = 65536,
                         z_exp: float = Z_EXP_KM, f_iso: float = F_ISOTROPIC,
                         cone_deg: float = CONE_HALF_ANGLE_DEG) -> dict:
    """Draw importance samples of the two-vertex integrand.

    Returns sums, second moment, extremes and diagnostic counts.  Samples
    whose chord x1 -> x2 crosses the opaque Earth contribute zero through
    T12 (counted in ``occluded``, never dropped); ``dark_illum`` counts
    second vertices where the finite-source illumination is exactly zero
    (shadow).
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    total_sq = 0.0
    max_abs = 0.0
    n_occluded = 0
    n_dark = 0
    done = 0
    started = time.perf_counter()
    while done < n_samples:
        m = min(batch, n_samples - done)
        z1, pdf_z = _proposal_z1(rng, m, fields.z_top_km, z_exp)
        omega, pdf_o = _proposal_omega(rng, m, sun_hat, f_iso, cone_deg)
        x1 = np.zeros((m, 3))
        x1[:, 2] = R_EARTH + z1 * 1e3
        b = np.einsum("ij,ij->i", x1, omega)
        r1 = np.sqrt(np.einsum("ij,ij->i", x1, x1))
        s_top = -b + np.sqrt(np.maximum(b * b + fields.top_radius ** 2 - r1 * r1, 0.0))
        disc_g = b * b + fields.ground_radius ** 2 - r1 * r1
        # only a *forward* root counts: an upward ray intersects the ground
        # sphere behind the vertex (negative root), which is not a limit
        raw_ground = np.where(disc_g > 0.0,
                              -b - np.sqrt(np.maximum(disc_g, 0.0)), np.inf)
        s_ground = np.where(raw_ground > 0.0, raw_ground, np.inf)
        s_max = np.minimum(s_top, s_ground)
        if not np.all(s_max > 0.0):
            raise ValueError("degenerate proposal geometry (S_max <= 0)")
        s_vertex = rng.random(m) * s_max
        x2 = x1 + s_vertex[:, None] * omega
        d2 = (x1 - x2) / np.maximum(
            np.sqrt(np.einsum("ij,ij->i", x1 - x2, x1 - x2))[:, None], 1e-300)
        z2 = (np.sqrt(np.einsum("ij,ij->i", x2, x2)) - R_EARTH) / 1e3
        mu1 = omega @ np.array([0.0, 0.0, 1.0])  # backward direction at x1: up
        occluded = segment_occluded(x1, x2, fields.ground_radius)
        # illumination takes the *backward-path propagation* direction at x2
        # (x1 -> x2), the same convention as source.scatter/backward and as
        # single_zenith (which passes the observer-to-vertex direction)
        illum = fields.illumination(x2, omega, fields.ray_fraction(z2),
                                    fields.aerosol_fractions(z2))
        t12 = fields.chord_transmittance(x1, x2)
        # The single-scattering albedos are analog SURVIVAL probabilities in
        # the MC, not weight factors: integrating against the physical
        # scattering densities b_sca(x1), b_sca(x2) already includes them
        # (b_ext * b_sca/b_ext = b_sca).  Adding albedos here would count
        # every survival twice.
        value = ((fields.t_view(z1) * fields.b_sca(z1) * Z1_KM_TO_M / pdf_z)
                 * (fields.phase_mix(mu1, z1) / pdf_o)
                 * t12 * fields.b_sca(z2) * illum
                 * s_max * fields.f0)
        total += float(value.sum())
        total_sq += float(np.dot(value, value))
        max_abs = max(max_abs, float(np.abs(value).max()))
        n_occluded += int(np.count_nonzero(occluded))
        n_dark += int(np.count_nonzero(illum == 0.0))
        done += m
    seconds = time.perf_counter() - started
    return dict(n_samples=done, sum=total, sum_sq=total_sq,
                L2=total / done, second_moment=total_sq / done,
                max_abs=max_abs, occluded=n_occluded, dark_illum=n_dark,
                seconds=seconds)


def estimate(atmo, iw: int, source, sun_elevation_deg: float, n_samples: int,
             seed: int, **kwargs) -> dict:
    """Estimate L2 on the real benchmark atmosphere for one wavelength."""
    fields = real_fields(atmo, iw, source, sun_hat_for(sun_elevation_deg))
    result = second_order_samples(fields, sun_hat_for(sun_elevation_deg),
                                  n_samples, seed, **kwargs)
    result["seed"] = int(seed)
    result["iw"] = int(iw)
    result["wl"] = float(atmo.wl[iw])
    return result
