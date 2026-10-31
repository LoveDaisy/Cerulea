"""High-level radiative transfer entry points.

:func:`sky_radiance_total` is the historical ``(n_view, n_wl)`` ndarray
entry for the clear sky.  :func:`solve` is the structured entry: it selects
one backend *explicitly* (``method="clear" | "twilight" | "cloud"`` -- there
is no automatic switching), applies the capability checks of
:data:`skycolor.rt.twilight.CAPABILITIES` and friends, and returns a
:class:`~skycolor.rt.result.SkyResult` whose diagnostics trace the orders,
convergence, statistical error and component decomposition behind the
numbers.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..atmosphere import UniformField
from . import multiple as _multiple
from . import twilight as _twilight
from .cloud import sky_radiance_with_cloud as _sky_radiance_with_cloud
from .result import SkyResult
from .single import sky_radiance as _single


def sky_radiance_total(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                       sun_azimuth_deg=0.0, multiple=True, single=True,
                       n_gauss=3, chunk=512):
    """Total spectral sky radiance (single + multiple scattering)."""
    out = None
    if single:
        out = _single(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                      sun_azimuth_deg=sun_azimuth_deg, n_gauss=n_gauss, chunk=chunk)
    if multiple:
        ms = _multiple.multiple_scattering(atmo, sun_zenith_deg, view_zenith_deg,
                                           view_azimuth_deg, sun_azimuth_deg=sun_azimuth_deg)
        out = ms if out is None else out + ms
    return out


def solve(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
          sun_azimuth_deg=0.0, *,
          method: str = "clear",
          cloud_layer=None,
          single: bool = True,
          multiple: bool = True,
          n_gauss: int = 3,
          chunk: int = 512,
          quality: str = "medium",
          seed: int = 0,
          grid: dict | None = None) -> SkyResult:
    """Run one backend *by name* and return a traceable :class:`SkyResult`.

    Parameters
    ----------
    method : ``"clear"``, ``"twilight"`` or ``"cloud"``
        ``"clear"`` is single + SOS multiple scattering (the
        :func:`sky_radiance_total` semantics); ``"twilight"`` is the
        spherical engine (:func:`skycolor.rt.twilight.twilight_radiance`);
        ``"cloud"`` composes a deck with the clear solver
        (:func:`skycolor.rt.cloud.sky_radiance_with_cloud`) and requires
        ``cloud_layer``.  There is no automatic method switching: the
        caller decides.
    single, multiple : bool
        Include the first-order / higher-order terms.
    quality, seed, grid : twilight options (ignored by other methods).

    Returns
    -------
    SkyResult
        ``radiance`` is bit-identical to the corresponding legacy entry
        point called with the same arguments.
    """
    options: dict = dict(sun_zenith_deg=float(sun_zenith_deg),
                         sun_azimuth_deg=float(sun_azimuth_deg),
                         single=bool(single), multiple=bool(multiple))
    diagnostics: dict = {}
    stderr = None
    components = None

    if method == "clear":
        options.update(n_gauss=int(n_gauss), chunk=int(chunk))
        total = None
        if single:
            total = _single(atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
                            sun_azimuth_deg=sun_azimuth_deg, n_gauss=n_gauss,
                            chunk=chunk)
        if multiple:
            ms = _multiple.solve(atmo, sun_zenith_deg, view_zenith_deg,
                                 view_azimuth_deg, sun_azimuth_deg=sun_azimuth_deg)
            diagnostics["orders"] = int(ms.orders)
            diagnostics["converged"] = bool(ms.converged)
            total = ms.radiance if total is None else total + ms.radiance
    elif method == "twilight":
        options.update(quality=str(quality), seed=int(seed),
                       grid=dict(grid) if grid else None)
        if any(not isinstance(lay.horizontal, UniformField)
               for lay in getattr(atmo, "aerosols", ())):
            warnings.warn(
                "twilight: a non-uniform aerosol HorizontalField affects the "
                "first-order term only; the Monte Carlo multiple scattering "
                "assumes a horizontally uniform atmosphere",
                RuntimeWarning, stacklevel=2)
        total, err, info = _twilight.twilight_radiance(
            atmo, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=sun_azimuth_deg, quality=quality, seed=seed,
            single=single, multiple=multiple, grid=grid, return_info=True)
        stderr = err
        diagnostics.update(info)
    elif method == "cloud":
        if cloud_layer is None:
            raise ValueError("method='cloud' requires cloud_layer")
        options.update(n_gauss=int(n_gauss), chunk=int(chunk),
                       cloud=cloud_layer.describe())
        total, components = _sky_radiance_with_cloud(
            atmo, cloud_layer, sun_zenith_deg, view_zenith_deg, view_azimuth_deg,
            sun_azimuth_deg=sun_azimuth_deg, multiple=multiple,
            n_gauss=n_gauss, chunk=chunk, return_components=True)
    else:
        raise ValueError(f"unknown method {method!r}; "
                         "expected 'clear', 'twilight' or 'cloud'")

    return SkyResult(radiance=total, wl=atmo.wl, method=method,
                     options=options, stderr=stderr, components=components,
                     diagnostics=diagnostics)
