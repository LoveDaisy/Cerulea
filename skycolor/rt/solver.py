"""The authoritative production radiative-transfer entry point.

:func:`solve` is the *one* place that composes the model's algorithm kernels
into a complete sky-radiance computation.  A request names a finite
*approximation* (``approximation=``), selects scattering orders with
``single``/``multiple``, and may extend the scene with ``cloud_layer``;
order selection, support-domain checks, component assembly and capability
recording happen here exactly once.  The historical full-calculation entry
points (:func:`sky_radiance_total`, :func:`rt.twilight.twilight_radiance`,
:func:`rt.cloud.sky_radiance_with_cloud`) are thin forwarders on top of it,
so there is no second place where ``single + multiple`` can be composed
differently.

Finite approximations (all consume the same unified finite-disk solar
source; the source side is refracted in every term, the viewing/diffuse
side follows the kernel's own convention):

=================  ==========================  ===============================
``approximation``  first order                 higher orders
=================  ==========================  ===============================
``spherical-       spherical-shell straight    none (requesting ``multiple``
single``           viewing rays                explicitly is rejected)
                   (:mod:`rt.single`)
``plane-sos``      the SOS kernel's own first  the same SOS run's orders >= 2
                   order (``rt.multiple.solve  (plane-parallel diffuse
                   `` with ``single_view``)    transport, delta-M)
``hybrid-sos``     spherical single            plane-parallel SOS
                   (the historical "clear"
                   combination -- default)
``spherical-mc``   spherical single            spherical straight-line Monte
                                               Carlo split term
                                               (:func:`rt.twilight.multiple_scattered`)
=================  ==========================  ===============================

``spherical-mc`` is an *algorithm choice*, not extra physics: its split
estimator defines "higher orders" exactly as the other backends do (total
minus first order).  ``plane-sos``'s source is the same finite disk
projected per beam -- a pseudo-spherical illumination, not a strict
plane-parallel one.

The legacy ``method="clear" | "twilight" | "cloud"`` strings remain as
aliases resolved at this single place: ``clear`` -> ``hybrid-sos``,
``twilight`` -> ``spherical-mc``, ``cloud`` -> the cloud overlay over the
``hybrid-sos`` base.  Numerical settings arrive either through the
historical flat kwargs (``n_gauss``, ``chunk``, ``quality``, ``seed``,
``grid``) or through the grouped :class:`SingleConfig` / :class:`SOSConfig`
/ :class:`MCConfig`; giving the same setting twice is an error, so no
priority has to be remembered.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..atmosphere import UniformField
from . import multiple as _multiple
from . import twilight as _twilight
from .cloud import CAPABILITIES as _CLOUD_CAPABILITIES
from .cloud import sky_radiance_with_cloud as _sky_radiance_with_cloud
from .multiple import CAPABILITIES as _SOS_CAPABILITIES
from .result import SkyResult
from .single import CAPABILITIES as _SINGLE_CAPABILITIES
from .single import sky_radiance as _single
from .twilight import CAPABILITIES as _MC_CAPABILITIES
from .twilight import QUALITY as _MC_QUALITY

__all__ = ["solve", "sky_radiance_total", "SingleConfig", "SOSConfig",
           "MCConfig", "SPHERICAL_SINGLE", "PLANE_SOS", "HYBRID_SOS",
           "SPHERICAL_MC"]

#: canonical finite approximations
SPHERICAL_SINGLE = "spherical-single"
PLANE_SOS = "plane-sos"
HYBRID_SOS = "hybrid-sos"
SPHERICAL_MC = "spherical-mc"
_APPROXIMATIONS = (SPHERICAL_SINGLE, PLANE_SOS, HYBRID_SOS, SPHERICAL_MC)

#: legacy method aliases -> the canonical approximation of the clear base
#: run (``cloud`` additionally applies the cloud overlay on top)
_METHOD_ALIASES = {"clear": HYBRID_SOS, "twilight": SPHERICAL_MC,
                   "cloud": HYBRID_SOS}

#: wrapper-level defaults of the historical signature.  These are the
#: established production defaults (they predate this module and are what
#: every existing caller and golden number relies on); kernel defaults
#: apply only where the wrapper has never had an opinion (all of
#: :class:`SOSConfig`, and the MC grid beyond ``quality``/``seed``).
_N_GAUSS = 3
_CHUNK = 512
_QUALITY = "medium"
_SEED = 0


# ---------------------------------------------------------------------------
# grouped numerical configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SingleConfig:
    """Numerical settings of the spherical single-scattering kernel.

    ``None`` fields fall back to the historical wrapper defaults
    (``n_gauss=3``, ``chunk=512``).  ``chunk`` here is the single-kernel
    ray block size and is unrelated to :attr:`SOSConfig.chunk`.
    """
    n_gauss: int | None = None
    chunk: int | None = None


@dataclass(frozen=True)
class SOSConfig:
    """Numerical settings of the plane-parallel SOS kernel.

    ``None`` fields fall back to the kernel's own authoritative defaults
    (32 mu nodes, 12 Fourier modes, 32 moments, 40 orders, ``tol=1e-3``,
    chunk 256, delta-M on); they are deliberately not re-stated here so
    there is exactly one source for each default.
    """
    n_mu: int | None = None
    n_modes: int | None = None
    n_moments: int | None = None
    max_orders: int | None = None
    tol: float | None = None
    chunk: int | None = None
    delta_m: bool | None = None


@dataclass(frozen=True)
class MCConfig:
    """Numerical settings of the spherical Monte Carlo kernel.

    ``quality`` must name a preset of ``rt.twilight.QUALITY``; ``grid``
    overrides individual preset keys.  ``None`` fields fall back to the
    historical wrapper defaults (``quality="medium"``, ``seed=0``, no
    grid override).
    """
    quality: str | None = None
    seed: int | None = None
    grid: dict | None = None


@dataclass(frozen=True)
class _Request:
    """Fully resolved request (internal)."""
    approximation: str
    overlay: bool
    single: bool
    multiple: bool
    mc_runs: bool
    sos_runs: bool
    n_gauss: int
    chunk: int
    quality: str
    seed: int
    grid: dict | None
    sos_kwargs: dict


def _component(kernel: str, capabilities: dict) -> dict:
    """Provenance record of one computed component."""
    return {"kernel": kernel, "approximations": dict(capabilities)}


# ---------------------------------------------------------------------------
# request resolution (all errors here happen before any kernel runs)
# ---------------------------------------------------------------------------
def _resolve_request(
    method: str | None,
    approximation: str | None,
    single: bool | None,
    multiple: bool | None,
    n_gauss: int | None,
    chunk: int | None,
    quality: str | None,
    seed: int | None,
    grid: dict | None,
    single_config: SingleConfig | None,
    sos_config: SOSConfig | None,
    mc_config: MCConfig | None,
) -> _Request:
    # -- alias / approximation resolution ------------------------------------
    if method is not None and method not in _METHOD_ALIASES:
        raise ValueError(f"unknown method {method!r}; "
                         "expected 'clear', 'twilight' or 'cloud'")
    if approximation is not None and approximation not in _APPROXIMATIONS:
        raise ValueError(f"unknown approximation {approximation!r}; expected "
                         f"one of {', '.join(_APPROXIMATIONS)}")
    overlay = method == "cloud"
    alias_approx = _METHOD_ALIASES.get(method) if method is not None else None
    if overlay:
        # the cloud overlay is defined over the hybrid-sos base combination
        if approximation is not None and approximation != HYBRID_SOS:
            raise ValueError(
                f"the cloud overlay composes with approximation="
                f"{HYBRID_SOS!r} (or an explicit precomputed clear field), "
                f"not {approximation!r}")
        base_approx = HYBRID_SOS
    elif approximation is not None and alias_approx is not None:
        if approximation != alias_approx:
            raise ValueError(
                f"method={method!r} and approximation={approximation!r} "
                "disagree; pass one of them")
        base_approx = approximation
    else:
        base_approx = approximation if approximation is not None else alias_approx
        if base_approx is None:
            base_approx = HYBRID_SOS

    # -- order selection ------------------------------------------------------
    single_r = True if single is None else bool(single)
    if multiple is None:
        # spherical-single has no higher-order term; everything else defaults
        # to both orders
        multiple_r = base_approx != SPHERICAL_SINGLE
    else:
        multiple_r = bool(multiple)
        if base_approx == SPHERICAL_SINGLE and multiple_r:
            raise ValueError(
                "approximation 'spherical-single' has no higher-order term; "
                "drop approximation and use 'hybrid-sos' for single + SOS")
    if not single_r and not multiple_r:
        raise ValueError("solve() with single=False and multiple=False "
                         "computes nothing; enable at least one term")

    # -- grouped vs flat settings: same setting twice is an error -------------
    sc = single_config if single_config is not None else SingleConfig()
    oc = sos_config if sos_config is not None else SOSConfig()
    mc = mc_config if mc_config is not None else MCConfig()

    def merge(flat, grouped, flat_name, group_name):
        if flat is not None and grouped is not None:
            raise ValueError(f"{flat_name} and {group_name} are the same "
                             "setting given twice; pass one")
        return flat if flat is not None else grouped

    n_gauss_r = merge(n_gauss, sc.n_gauss, "n_gauss", "single_config.n_gauss")
    chunk_r = merge(chunk, sc.chunk, "chunk", "single_config.chunk")
    quality_r = merge(quality, mc.quality, "quality", "mc_config.quality")
    seed_r = merge(seed, mc.seed, "seed", "mc_config.seed")
    grid_r = merge(grid, mc.grid, "grid", "mc_config.grid")
    n_gauss_r = _N_GAUSS if n_gauss_r is None else int(n_gauss_r)
    chunk_r = _CHUNK if chunk_r is None else int(chunk_r)

    # -- applicability: settings for kernels that will not run are errors ----
    mc_runs = base_approx == SPHERICAL_MC and multiple_r
    sos_runs = (base_approx == PLANE_SOS
                or (base_approx == HYBRID_SOS and multiple_r))
    single_runs = single_r or overlay   # the cloud overlay always includes it
    if not mc_runs:
        given = [name for name, val in (("quality", quality), ("seed", seed),
                                        ("grid", grid)) if val is not None]
        if mc_config is not None:
            given += [f"mc_config.{f}" for f in ("quality", "seed", "grid")
                      if getattr(mc, f) is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} configure the Monte Carlo kernel, but "
                "this request runs no MC term (higher orders are plane-parallel "
                "SOS or absent); drop them or pick approximation='spherical-mc'")
    if not sos_runs:
        if sos_config is not None and any(
                getattr(oc, f) is not None
                for f in ("n_mu", "n_modes", "n_moments", "max_orders", "tol",
                          "chunk", "delta_m")):
            raise ValueError(
                "sos_config configures the plane-parallel SOS kernel, but "
                "this request runs no SOS term; drop it or pick a "
                "plane-parallel higher-order approximation")
    if not single_runs:
        given = [name for name, val in (("n_gauss", n_gauss), ("chunk", chunk))
                 if val is not None]
        if single_config is not None:
            given += [f"single_config.{f}" for f in ("n_gauss", "chunk")
                      if getattr(sc, f) is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} configure the single-scattering kernel, "
                "but this request computes no spherical single term")

    # -- MC preset / grid validation (before any expensive work) --------------
    quality_r = _QUALITY if quality_r is None else str(quality_r)
    seed_r = _SEED if seed_r is None else int(seed_r)
    if quality_r not in _MC_QUALITY:
        raise ValueError(f"unknown quality {quality_r!r}; "
                         f"expected one of {sorted(_MC_QUALITY)}")
    grid_keys = set(next(iter(_MC_QUALITY.values())))
    if grid_r is not None:
        unknown = set(grid_r) - grid_keys
        if unknown:
            raise ValueError(f"unknown grid keys {sorted(unknown)}; expected "
                             f"a subset of {sorted(grid_keys)}")
        grid_r = dict(grid_r)

    # -- SOS kwargs: pass through only what the caller actually set -----------
    sos_kwargs = {}
    for field_name, value in (("n_mu", oc.n_mu), ("n_modes", oc.n_modes),
                              ("n_moments", oc.n_moments),
                              ("max_orders", oc.max_orders), ("tol", oc.tol),
                              ("chunk", oc.chunk), ("delta_m", oc.delta_m)):
        if value is not None:
            sos_kwargs[field_name] = value

    return _Request(approximation=base_approx, overlay=overlay,
                    single=single_r, multiple=multiple_r,
                    mc_runs=mc_runs, sos_runs=sos_runs,
                    n_gauss=n_gauss_r, chunk=chunk_r, quality=quality_r,
                    seed=seed_r, grid=grid_r, sos_kwargs=sos_kwargs)


def _require_bottom_observer(atmo) -> None:
    """Plane-parallel terms are defined only at the model bottom.

    ``Atmosphere.at_altitude`` truncates the domain, so a raised *station*
    has ``elevation_km == z_bounds[0]`` and qualifies by construction.  An
    observer raised on an untruncated domain (``atmo.elevation_km = ...``)
    has no plane interpretation -- the diffuse field and its view integral
    live at the domain bottom -- so such requests fail here instead of being
    silently evaluated at the wrong height.  ``spherical-single`` has no
    plane term and is the approximation for balloon-style geometries.
    """
    z0 = float(np.asarray(atmo.z_bounds).ravel()[0])
    z_obs = float(atmo.elevation_km)
    if abs(z_obs - z0) > 1e-9 * max(1.0, abs(z0)):
        raise ValueError(
            f"plane-parallel terms require the observer at the model bottom: "
            f"elevation_km={z_obs} but z_bounds[0]={z0}.  Use "
            f"atmosphere.at_altitude() for a raised station, or "
            f"approximation='spherical-single' (no plane term) for a raised "
            f"observer on the full domain")


def _check_horizontal_field(atmo, approximation: str, multiple_r: bool) -> None:
    """Horizontal-field coverage: applied by spherical single, ignored by
    the plane/MC higher-order kernels.

    ``plane-sos`` uses the field in *no* component, so a non-uniform field
    is rejected rather than silently ignored; ``hybrid-sos`` /
    ``spherical-mc`` keep the historical partial support (first-order term
    only) with a warning.
    """
    if not any(not isinstance(lay.horizontal, UniformField)
               for lay in getattr(atmo, "aerosols", ())):
        return
    if approximation == PLANE_SOS:
        raise ValueError(
            "plane-sos uses the aerosol HorizontalField in no component; "
            "a non-uniform field would be silently ignored.  Use "
            "'hybrid-sos' or 'spherical-single' (view-side field support) "
            "or a uniform field")
    if multiple_r:
        kernel = ("the Monte Carlo multiple scattering"
                  if approximation == SPHERICAL_MC
                  else "the plane-parallel SOS multiple scattering")
        warnings.warn(
            f"a non-uniform aerosol HorizontalField affects the first-order "
            f"term only; {kernel} assumes a horizontally uniform atmosphere",
            RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# historical ndarray entry -- now a thin forwarder
# ---------------------------------------------------------------------------
def sky_radiance_total(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                       sun_azimuth_deg=0.0, multiple=True, single=True,
                       n_gauss=3, chunk=512):
    """Total spectral sky radiance (single + multiple scattering).

    Thin forwarder to :func:`solve` with the ``hybrid-sos`` approximation;
    the returned array is bit-identical to the pre-consolidation
    implementation (one ``rt.single`` call plus one SOS run, added in that
    order).  The historical empty request (both switches off) returns
    ``None`` here -- kept for compatibility, it runs no kernel -- while
    :func:`solve` itself rejects it.  ``n_gauss``/``chunk`` are forwarded
    only when the single term actually runs (they configure no other
    kernel; the old implementation silently ignored them otherwise).
    """
    if not single and not multiple:
        return None   # historical empty request, documented; no kernel run
    return solve(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                 sun_azimuth_deg=sun_azimuth_deg, method="clear",
                 single=bool(single), multiple=bool(multiple),
                 **({"n_gauss": n_gauss, "chunk": chunk} if single else {})
                 ).radiance


# ---------------------------------------------------------------------------
# the authoritative entry
# ---------------------------------------------------------------------------
def solve(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
          sun_azimuth_deg=0.0, *,
          method: str | None = None,
          approximation: str | None = None,
          cloud_layer=None,
          single: bool | None = None,
          multiple: bool | None = None,
          n_gauss: int | None = None,
          chunk: int | None = None,
          quality: str | None = None,
          seed: int | None = None,
          grid: dict | None = None,
          single_config: SingleConfig | None = None,
          sos_config: SOSConfig | None = None,
          mc_config: MCConfig | None = None) -> SkyResult:
    """Resolve one request, run the needed kernels once each, return a
    traceable :class:`SkyResult`.

    Parameters
    ----------
    method : ``"clear"``, ``"twilight"`` or ``"cloud"``, optional
        Legacy aliases, resolved to :attr:`approximation` at this single
        place (``clear`` -> ``hybrid-sos``, ``twilight`` -> ``spherical-mc``,
        ``cloud`` -> cloud overlay over ``hybrid-sos``).  Giving ``method``
        and a disagreeing ``approximation`` is an error.
    approximation : one of the four canonical names, optional
        Defaults to ``hybrid-sos`` (the historical clear combination).
    cloud_layer : CloudLayer, optional
        Cloud overlay (legacy ``method="cloud"``).  Requires ``cloud_layer``
        when that alias is used.
    single, multiple : bool, optional
        Order selection; ``None`` (not given) resolves to the approximation
        default (both orders, except ``spherical-single`` which is
        single-only).  ``spherical-single`` with an explicit ``multiple=True``
        is rejected.  Both false computes nothing and is rejected.
    n_gauss, chunk : single-kernel settings
    quality, seed, grid : Monte Carlo settings (``grid`` keys must name
        preset fields of ``rt.twilight.QUALITY``; unknown keys are errors)
    single_config, sos_config, mc_config : grouped settings
        Same settings as the flat kwargs above; passing one setting through
        both channels is an error.  ``None`` fields use kernel defaults.

    Raises
    ------
    ValueError
        Unknown method/approximation/quality/grid key; ``method="cloud"``
        without ``cloud_layer``; disagreeing ``method``/``approximation``;
        the same numerical setting given twice; settings for kernels the
        request does not run; an observer off the model bottom on any
        plane-parallel request (use ``atmosphere.at_altitude``); a
        non-uniform aerosol ``HorizontalField`` on ``plane-sos`` (no
        component applies it).

    Returns
    -------
    SkyResult
        ``radiance`` for the legacy combinations is bit-identical to the
        historical entry point called with the same arguments.
    """
    req = _resolve_request(method, approximation, single, multiple,
                           n_gauss, chunk, quality, seed, grid,
                           single_config, sos_config, mc_config)

    options: dict = dict(sun_zenith_deg=float(sun_zenith_deg),
                         sun_azimuth_deg=float(sun_azimuth_deg),
                         single=bool(req.single), multiple=bool(req.multiple),
                         approximation=req.approximation)
    diagnostics: dict = {}
    effective: dict = {}
    approximations: dict = {}
    stderr = None
    components = None

    if req.overlay:
        # -- cloud overlay (legacy composition; consolidated in the solver in
        #    a later step -- the numbers and the options bookkeeping are the
        #    historical ones)
        if cloud_layer is None:
            raise ValueError("method='cloud' requires cloud_layer")
        options.pop("single")   # the deck composition always includes it
        options.update(n_gauss=int(req.n_gauss), chunk=int(req.chunk),
                       cloud=cloud_layer.describe())
        total, components = _sky_radiance_with_cloud(
            atmo, cloud_layer, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=sun_azimuth_deg, multiple=req.multiple,
            n_gauss=req.n_gauss, chunk=req.chunk, return_components=True)
        approximations["single"] = _component(
            "rt.single.sky_radiance", _SINGLE_CAPABILITIES)
        approximations["multiple"] = _component(
            "rt.multiple.solve", _SOS_CAPABILITIES)
        approximations["cloud"] = _component("rt.cloud", _CLOUD_CAPABILITIES)
        return SkyResult(radiance=total, wl=atmo.wl,
                         method="cloud", options=options, stderr=stderr,
                         components=components, diagnostics=diagnostics,
                         approximation=req.approximation,
                         approximations=approximations, effective=effective)

    if req.approximation == SPHERICAL_MC:
        # -- spherical single + spherical straight-line MC split term.
        #    (Routing note: this goes through the twilight engine's own
        #    entry today; the consolidation into single + multiple_scattered
        #    is the next step and keeps these numbers bit-identical.)
        options.update(quality=req.quality, seed=int(req.seed),
                       grid=dict(req.grid) if req.grid else None)
        _check_horizontal_field(atmo, req.approximation, req.multiple)
        total, err, info = _twilight.twilight_radiance(
            atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=sun_azimuth_deg, quality=req.quality,
            seed=req.seed, single=req.single, multiple=req.multiple,
            grid=req.grid, return_info=True)
        stderr = err
        diagnostics.update(info)
        if req.multiple:
            effective["mc"] = dict(
                quality=req.quality, seed=req.seed,
                grid=dict(req.grid) if req.grid else None,
                n_zen=info["n_zen"], n_az=info["n_az"],
                n_anchor=info["n_anchor"], n_photons=info["n_photons"],
                bandwidth=info["bandwidth"])
            approximations["multiple"] = _component(
                "rt.twilight.multiple_scattered", _MC_CAPABILITIES)
        if req.single:
            approximations["single"] = _component(
                "rt.single.sky_radiance", _SINGLE_CAPABILITIES)
        return SkyResult(radiance=total, wl=atmo.wl, method="twilight",
                         options=options, stderr=stderr, components=components,
                         diagnostics=diagnostics, approximation=req.approximation,
                         approximations=approximations, effective=effective)

    # -- deterministic combinations (spherical single / SOS parts) ----------
    if req.sos_runs:
        _require_bottom_observer(atmo)
    _check_horizontal_field(atmo, req.approximation, req.multiple)

    ss = None
    first = None
    ms = None
    sos = None
    if req.approximation == PLANE_SOS:
        # one SOS run serves both terms: ``single_view`` adds the plane
        # first-order evaluation to the same iteration -- no second kernel
        # call, and the explicit existing cost of a single-only request
        sos = _multiple.solve(atmo, sun_zenith_deg, view_zenith_deg,
                              view_azimuth_deg, sun_azimuth_deg=sun_azimuth_deg,
                              single_view=bool(req.single), **req.sos_kwargs)
        if req.single:
            first = sos.radiance_single
        if req.multiple:
            ms = sos.radiance
    else:
        if req.single:
            ss = _single(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                         sun_azimuth_deg=sun_azimuth_deg, n_gauss=req.n_gauss,
                         chunk=req.chunk)
        if req.multiple:
            sos = _multiple.solve(atmo, sun_zenith_deg, view_zenith_deg,
                                  view_azimuth_deg, sun_azimuth_deg=sun_azimuth_deg,
                                  **req.sos_kwargs)
            ms = sos.radiance

    total = None
    if ss is not None:
        total = ss
    if first is not None:
        total = first if total is None else total + first
    if ms is not None:
        total = ms if total is None else total + ms

    components = {"single": first if req.approximation == PLANE_SOS else ss,
                  "multiple": ms}
    if req.single:
        options.update(n_gauss=int(req.n_gauss), chunk=int(req.chunk))
        effective["single"] = {"n_gauss": int(req.n_gauss),
                               "chunk": int(req.chunk)}
        approximations["single"] = _component(
            "rt.single.sky_radiance", _SINGLE_CAPABILITIES)
    if sos is not None:
        diagnostics["orders"] = int(sos.orders)
        diagnostics["converged"] = bool(sos.converged)
        effective["sos"] = {
            "n_mu": int(sos.n_mu_eff), "n_modes": int(sos.n_modes_eff),
            "n_moments": int(sos.n_moments_eff), "tol": float(sos.tol),
            "max_orders": int(sos.max_orders), "delta_m": bool(sos.delta_m),
            "orders_run": int(sos.orders), "converged": bool(sos.converged)}
        approximations["multiple"] = _component(
            "rt.multiple.solve(single_view=True)" if req.approximation == PLANE_SOS
            else "rt.multiple.solve", _SOS_CAPABILITIES)
        if req.approximation == PLANE_SOS and req.single:
            approximations["single"] = _component(
                "rt.multiple.solve(single_view=True)", _SOS_CAPABILITIES)

    return SkyResult(radiance=total, wl=atmo.wl, method="clear",
                     options=options, stderr=stderr, components=components,
                     diagnostics=diagnostics, approximation=req.approximation,
                     approximations=approximations, effective=effective)
