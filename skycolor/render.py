"""Rendering: one solve, then pure display of the result.

The production flow is one-directional::

    resolve_sun -> one solve (``solve_dome``) -> SkyPresentation -> outputs

:func:`solve_dome` runs the authoritative entry (:func:`rt.solver.solve`)
exactly once on the regular half-dome grid and binds the *display* material
the result alone cannot carry: the grid layout, the resolved solar direction,
the adopted source's apparent sun position/visibility, the ground albedo and
the matching ground direct irradiance.  The binding lives in a
:class:`SkyPresentation`; every consumer afterwards (:func:`render_result`,
the chart table) is a pure function of those arrays -- switching cameras or
re-printing tables never re-enters the solver or the solar source.

``SkyResult`` stays the only authority on solver facts (approximation,
diagnostics, effective settings); the carrier holds no second copy of them
and no runnable solver callback.  Existing convenience entries
(:func:`sky_dome`, :func:`sky_radiance_grid`, :func:`render_scene`) keep
their signatures and ndarray contracts and are thin forwarders on this path,
so there is no second place where dome + cameras + color can be assembled
differently.  The ``radiance_field`` injection seam stays: external fields
flow through :func:`presentation_from_field` (provenance ``"external"``,
honestly recorded, never presented as a solver result).
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace

import numpy as np

from . import color


# ---------------------------------------------------------------------------
# sky dome
# ---------------------------------------------------------------------------
def _single_config(n_gauss: int, chunk: int):
    """Grouped single-kernel settings for the thin forwarders (lazy import:
    the solver tree is heavy and color-only users should not pay for it)."""
    from .rt.solver import SingleConfig

    return SingleConfig(n_gauss=n_gauss, chunk=chunk)


def sky_dome(atmo, sun_zenith_deg: float, n_zenith: int = 46, n_azimuth: int = 91,
             n_gauss: int = 3, chunk: int = 512, min_zenith_deg: float = 0.0,
             multiple: bool = True, radiance_field=None):
    """Radiance of the whole sky on a (zenith, relative azimuth) grid.

    The atmosphere is horizontally homogeneous, so the radiance only depends on
    the angle between the viewing direction and the sun; the grid covers
    relative azimuths 0..180 degrees.  Returns a dict with ``zenith``,
    ``azimuth_rel``, ``radiance`` (n_zenith, n_azimuth, n_wl), ``wl`` and
    ``sun_zenith_deg``.

    Thin forwarder: without ``radiance_field`` this is one
    :func:`solve_dome` request (the authoritative solver entry) converted
    back to the historical dict shape; with ``radiance_field`` the external
    field is sampled once through the same preparation path.  The dome
    numbers are unchanged by the restructuring.

    ``radiance_field`` decouples the display from the solver: a callable
    ``(zen_flat, az_rel_flat) -> (n_view, n_wl)`` consulted instead of the
    built-in solver.  The medium and the solar geometry are bound
    in the caller's closure, so the display layer needs no notion of what
    produced the field (a cloudy sky, an experiment, a validation harness).
    With a field given, ``atmo`` is used only for the ``wl`` metadata.
    """
    pres = solve_dome(atmo, 90.0 - float(sun_zenith_deg), 0.0,
                      n_zenith=n_zenith, n_azimuth=n_azimuth,
                      min_zenith_deg=min_zenith_deg, multiple=multiple,
                      single_config=_single_config(n_gauss, chunk),
                      radiance_field=radiance_field)
    if pres.result is not None:
        _warn_result_limits(pres.result)
    return dict(zenith=pres.zenith, azimuth_rel=pres.azimuth_rel,
                radiance=pres.radiance, wl=pres.wl,
                sun_zenith_deg=sun_zenith_deg)


def sky_radiance_grid(atmo, sun_zenith_deg, Z, A, n_gauss=3, chunk=512,
                      sun_azimuth_deg=0.0, multiple=None, radiance_field=None,
                      *, approximation=None, cloud_layer=None,
                      single_config=None, sos_config=None, mc_config=None):
    """Radiance on an explicit (zenith, relative azimuth) meshgrid.

    Without ``radiance_field`` a thin forwarder to one :func:`rt.solver.solve`
    request (the historical ndarray shape is the reshaped result).  Legacy
    calls (no keyword-only solver option given) keep the historical request
    shape exactly: ``multiple`` defaults to True and the single-kernel
    settings 3/512 go through as before.  Giving any solver option switches
    to pass-through mode: only explicitly set values are forwarded and the
    entry point's own defaults/validation decide the rest -- no backend
    alias or combination is parsed here.  With a field, the field is
    consulted instead (its coordinate convention is unchanged: absolute
    view azimuth is the caller's business, the grid is relative to the
    sun).
    """
    if radiance_field is not None:
        rad = np.asarray(radiance_field(np.ravel(Z), np.ravel(A)), dtype=float)
        return rad.reshape(np.shape(Z) + (rad.shape[-1],))

    from .rt.solver import solve

    shape = np.shape(Z)
    new_options = any(opt is not None for opt in
                      (approximation, cloud_layer, single_config, sos_config,
                       mc_config))
    kwargs: dict = {}
    if new_options:
        if multiple is not None:
            kwargs["multiple"] = multiple
        kwargs.update(approximation=approximation, cloud_layer=cloud_layer,
                      single_config=single_config, sos_config=sos_config,
                      mc_config=mc_config)
    else:
        kwargs.update(single=True,
                      multiple=True if multiple is None else bool(multiple),
                      n_gauss=n_gauss, chunk=chunk)
    res = solve(atmo, sun_zenith_deg, np.ravel(Z),
                np.ravel(A) + sun_azimuth_deg,
                sun_azimuth_deg=sun_azimuth_deg, **kwargs)
    _warn_result_limits(res)
    return res.radiance.reshape(shape + (res.radiance.shape[-1],))


def sample_dome(dome: dict, view_zenith, view_azimuth_rel) -> np.ndarray:
    """Bilinear interpolation of a sky dome onto new directions."""
    zen = dome["zenith"]
    az = dome["azimuth_rel"]
    rad = dome["radiance"]
    zv = np.clip(view_zenith, zen[0], zen[-1])
    av = np.clip(np.abs(view_azimuth_rel), az[0], az[-1])
    iz = np.clip(np.searchsorted(zen, zv) - 1, 0, len(zen) - 2)
    ia = np.clip(np.searchsorted(az, av) - 1, 0, len(az) - 2)
    fz = ((zv - zen[iz]) / (zen[iz + 1] - zen[iz]))[..., None]
    fa = ((av - az[ia]) / (az[ia + 1] - az[ia]))[..., None]
    r00 = rad[iz, ia]
    r01 = rad[iz, ia + 1]
    r10 = rad[iz + 1, ia]
    r11 = rad[iz + 1, ia + 1]
    return (r00 * (1 - fz) * (1 - fa) + r01 * (1 - fz) * fa
            + r10 * fz * (1 - fa) + r11 * fz * fa)


def hemispherical_irradiance(radiance: np.ndarray, zenith_deg, azimuth_rel_deg):
    """Diffuse irradiance on a horizontal surface from a sky dome.

    ``E = int L cos(theta) dOmega``.  With ``mu = cos(theta)`` and
    ``dOmega = -dmu dphi`` this is ``int int L mu dmu dphi``.  The ``mu`` factor
    is easy to forget and doubles the answer if missing -- see
    ``tests/test_render.py``.
    """
    mu = np.cos(np.radians(np.asarray(zenith_deg, dtype=float)))
    phi = np.radians(np.asarray(azimuth_rel_deg, dtype=float))
    # integrate over mu first (mu is decreasing, hence the sign)
    over_mu = np.abs(np.trapezoid(np.asarray(radiance) * mu[:, None, None], mu, axis=0))
    return np.trapezoid(over_mu, phi, axis=0) * 2.0  # 0..pi -> full circle


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def panorama_grid(width: int, height: int, min_elevation: float = -8.0,
                  max_elevation: float = 90.0):
    azimuth = np.linspace(0.0, 360.0, width, endpoint=False)
    elevation = np.linspace(max_elevation, min_elevation, height)
    A, E = np.meshgrid(azimuth, elevation, indexing="xy")
    return A, E


def fisheye_grid(size: int, max_zenith: float = 90.0, projection: str = "equal_area",
                 center=None):
    y, x = np.mgrid[0:size, 0:size].astype(float)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    if center is not None:
        cx, cy = center
    r = np.hypot(x - cx, y - cy) / ((size - 1) / 2.0)
    mask = r <= 1.0
    if projection == "equal_area":
        zen = np.degrees(np.arcsin(np.clip(r, 0, 1)))
    else:
        zen = r * max_zenith
    azimuth = (np.degrees(np.arctan2(x - cx, -(y - cy)))) % 360.0
    return azimuth, 90.0 - zen, mask


def perspective_grid(width: int, height: int, hfov_deg: float = 90.0,
                     pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """Pinhole camera looking towards ``yaw`` at ``pitch`` (degrees)."""
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    x, y = np.meshgrid(np.arange(width) + 0.5 - width / 2.0,
                       height / 2.0 - (np.arange(height) + 0.5), indexing="xy")
    dx, dy, dz = x / f, y / f, np.ones_like(x)
    n = np.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx / n, dy / n, dz / n
    p = np.radians(pitch_deg)
    v = np.radians(yaw_deg)
    dy2 = dy * np.cos(p) + dz * np.sin(p)   # +pitch looks up
    dz2 = -dy * np.sin(p) + dz * np.cos(p)
    dx3 = dx * np.cos(v) + dz2 * np.sin(v)
    dz3 = -dx * np.sin(v) + dz2 * np.cos(v)
    elevation = np.degrees(np.arcsin(np.clip(dy2, -1, 1)))
    azimuth = (np.degrees(np.arctan2(dx3, dz3))) % 360.0
    return azimuth, elevation


# ---------------------------------------------------------------------------
# the presentation carrier
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyPresentation:
    """Everything displaying one sky needs, bound at preparation time.

    The carrier pairs a :class:`~skycolor.rt.result.SkyResult` (or, for the
    ``"external"`` provenance, an honestly-unknown substitute) with the
    regular grid it was computed on and the scene's display material.  It is
    the *only* input of the pure display functions (:func:`render_result`,
    :func:`chart_table`): once bound, switching cameras, tone maps or output
    formats re-enters nothing.  ``SkyResult`` remains the sole authority on
    solver facts -- the carrier copies none of its diagnostics and holds no
    solver callback, so nothing here can silently re-solve.

    Snapshots, not live references: ``wl``, ``ground_albedo`` and
    ``ground_direct_irradiance`` are copied at construction, so mutating the
    facade afterwards cannot change an already-bound presentation.
    ``radiance`` is a reshape *view* of the result's array (one spectral
    field, no second copy); treat it as read-only.

    ``sun_elevation_deg``/``sun_azimuth_deg`` are the pre-refraction input
    direction; ``sun_apparent_elevation_deg``/``sun_visible`` are the
    adopted source's apparent geometry at the observer (``None`` when not
    bound -- an external field does not own a source), consumed only by the
    decorative sun-disk marker, never by the numeric outputs.
    """

    zenith: np.ndarray                 # (n_zen,) grid, copied
    azimuth_rel: np.ndarray            # (n_az,) grid, copied
    wl: np.ndarray                     # (n_wl,), copied
    radiance: np.ndarray               # (n_zen, n_az, n_wl), reshape view
    sun_elevation_deg: float           # true (pre-refraction) input direction
    sun_azimuth_deg: float
    provenance: str                    # "solve" | "external"
    result: object | None              # SkyResult when provenance == "solve"
    cloud: bool                        # the scene contains a cloud deck
    sun_apparent_elevation_deg: float | None = None
    sun_visible: bool | None = None
    ground_albedo: np.ndarray | None = None       # (n_wl,), copied
    ground_direct_irradiance: np.ndarray | None = None  # (n_wl,), copied
    ground_supported: bool = False     # observer sits on the opaque model bottom

    @property
    def sun_zenith_deg(self) -> float:
        return 90.0 - self.sun_elevation_deg


def _check_grid(zenith, azimuth_rel, radiance, n_wl) -> np.ndarray:
    """Validate grid/field shape compatibility, return the 3-D reshape."""
    zen = np.asarray(zenith, dtype=float).ravel()
    az = np.asarray(azimuth_rel, dtype=float).ravel()
    rad = np.asarray(radiance, dtype=float)
    n_view = zen.size * az.size
    if rad.ndim == 2:
        if rad.shape[0] != n_view:
            raise ValueError(
                f"radiance has {rad.shape[0]} view rows but the grid is "
                f"{zen.size} x {az.size} = {n_view}; reshape/bind only a "
                "matching result")
    elif rad.ndim == 3:
        if rad.shape[:2] != (zen.size, az.size):
            raise ValueError(
                f"radiance grid {rad.shape[:2]} does not match the given "
                f"grid ({zen.size}, {az.size})")
    else:
        raise ValueError(
            f"radiance must be (n_view, n_wl) or (n_zen, n_az, n_wl), "
            f"got shape {rad.shape}")
    if rad.shape[-1] != n_wl:
        raise ValueError(
            f"radiance has {rad.shape[-1]} wavelengths but the grid binds "
            f"{n_wl}")
    return rad.reshape(zen.size, az.size, rad.shape[-1])


def _check_sun_matches_result(result, sun_elevation_deg, sun_azimuth_deg):
    """The bound direction must be the direction the result was solved at."""
    opt = getattr(result, "options", None) or {}
    if "sun_zenith_deg" in opt:
        want = 90.0 - float(sun_elevation_deg)
        if abs(float(opt["sun_zenith_deg"]) - want) > 1e-9:
            raise ValueError(
                f"result was solved at sun zenith {float(opt['sun_zenith_deg'])} "
                f"but the presentation binds elevation {sun_elevation_deg} "
                f"(zenith {want}); bind the matching direction")
    if "sun_azimuth_deg" in opt:
        if abs(float(opt["sun_azimuth_deg"]) - float(sun_azimuth_deg)) > 1e-9:
            raise ValueError(
                f"result was solved at sun azimuth "
                f"{float(opt['sun_azimuth_deg'])} but the presentation binds "
                f"{sun_azimuth_deg}; bind the matching direction")


def presentation_from_result(result, zenith, azimuth_rel,
                             sun_elevation_deg: float,
                             sun_azimuth_deg: float = 0.0, *,
                             ground_albedo=None,
                             ground_direct_irradiance=None,
                             sun_apparent_elevation_deg: float | None = None,
                             sun_visible: bool | None = None,
                             ground_supported: bool = False,
                             ) -> SkyPresentation:
    """Bind a *ready* :class:`SkyResult` to its grid for display.

    Pure reshape/bind: no solver runs, no backend choice is taken here, no
    missing display material is silently recomputed.  The result's grid
    shape and its recorded solar direction must match the arguments; the
    ground display additionally needs ``ground_albedo`` and a matching
    ``ground_direct_irradiance`` bound by the caller (compute them once in
    the caller's preparation step -- not here).  Missing material keeps the
    sky display working (:func:`render_result` with ``ground=False``) and
    makes the ground display fail loudly instead of pairing this sky with
    someone else's beam.
    """
    from .rt.result import SkyResult

    if not isinstance(result, SkyResult):
        raise TypeError(
            "presentation_from_result binds a SkyResult; use "
            "presentation_from_field for an externally produced field")
    rad = _check_grid(zenith, azimuth_rel, result.radiance,
                      int(np.asarray(result.wl).size))
    _check_sun_matches_result(result, sun_elevation_deg, sun_azimuth_deg)
    return SkyPresentation(
        zenith=np.array(zenith, dtype=float).ravel().copy(),
        azimuth_rel=np.array(azimuth_rel, dtype=float).ravel().copy(),
        wl=np.array(result.wl, dtype=float).ravel().copy(),
        radiance=rad,
        sun_elevation_deg=float(sun_elevation_deg),
        sun_azimuth_deg=float(sun_azimuth_deg),
        provenance="solve",
        result=result,
        cloud=getattr(result, "method", "") == "cloud",
        sun_apparent_elevation_deg=(None if sun_apparent_elevation_deg is None
                                    else float(sun_apparent_elevation_deg)),
        sun_visible=(None if sun_visible is None else bool(sun_visible)),
        ground_albedo=(None if ground_albedo is None
                       else np.array(ground_albedo, dtype=float).ravel().copy()),
        ground_direct_irradiance=(None if ground_direct_irradiance is None
                                  else np.array(ground_direct_irradiance,
                                                dtype=float).ravel().copy()),
        ground_supported=bool(ground_supported),
    )


def presentation_from_field(field, atmo, sun_elevation_deg: float,
                            sun_azimuth_deg: float = 0.0, n_zenith: int = 46,
                            n_azimuth: int = 91, min_zenith_deg: float = 0.0, *,
                            ground_albedo=None,
                            ground_direct_irradiance=None,
                            note: str = "") -> SkyPresentation:
    """Bind an externally produced spectral field onto the regular grid.

    This is the public shape of the historical ``radiance_field`` injection
    seam: the field callable ``(zen_flat, az_rel_flat) -> (n_view, n_wl)``
    is sampled **once** (relative azimuths, the grid's own convention) and
    the presentation records provenance ``"external"`` -- its capabilities
    and statistics are unknown, never presented as a solver result.  No
    solar-source query happens here; if the sky is displayed together with
    the ground, a matching ``ground_direct_irradiance`` must be bound (or
    the retired ``ground_direct_transmittance`` passed at display time),
    otherwise :func:`render_result` with ``ground=True`` raises.
    """
    zen = np.linspace(min_zenith_deg, 90.0, n_zenith)
    az = np.linspace(0.0, 180.0, n_azimuth)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    rad = np.asarray(field(np.ravel(Z), np.ravel(A)), dtype=float)
    wl = np.asarray(atmo.wl, dtype=float).ravel()
    if rad.shape != (zen.size * az.size, wl.size):
        raise ValueError(
            f"the field returned {rad.shape}, expected "
            f"({zen.size * az.size}, {wl.size}) for this grid and medium")
    rad3 = rad.reshape(zen.size, az.size, wl.size)
    if ground_albedo is None:
        # the historical default: the facade's ground albedo lights the
        # display ground unless the caller overrides it
        ground_albedo = np.asarray(atmo.ground_albedo, dtype=float).ravel()
    return SkyPresentation(
        zenith=zen.copy(), azimuth_rel=az.copy(), wl=wl.copy(),
        radiance=rad3,
        sun_elevation_deg=float(sun_elevation_deg),
        sun_azimuth_deg=float(sun_azimuth_deg),
        provenance="external",
        result=None,
        cloud=False,
        sun_apparent_elevation_deg=None,
        sun_visible=None,
        ground_albedo=(None if ground_albedo is None
                       else np.array(ground_albedo, dtype=float).ravel().copy()),
        ground_direct_irradiance=(None if ground_direct_irradiance is None
                                  else np.array(ground_direct_irradiance,
                                                dtype=float).ravel().copy()),
        ground_supported=True,
    )


def _bind_display_material(atmo, result, zenith, azimuth_rel,
                           sun_elevation_deg, sun_azimuth_deg,
                           cloud_layer=None):
    """One preparation-time binding of the adopted source's display geometry
    and the matching ground material (clear scenes; the cloud deck's
    beam-attenuated ground irradiance is bound by the same step once its
    helper is used here).  Runs before any display function, so displays
    stay pure."""
    from .solar_source import SolarSource

    sun_zenith_deg = 90.0 - float(sun_elevation_deg)
    source = SolarSource(atmo, sun_zenith_deg, float(sun_azimuth_deg))
    z_obs = float(atmo.elevation_km)
    app_e, _, _, lit = source.vertex(np.array([z_obs]),
                                     np.array([float(sun_elevation_deg)]))
    z0 = float(np.asarray(atmo.z_bounds).ravel()[0])
    ground_supported = abs(z_obs - z0) <= 1e-9 * max(1.0, abs(z0))
    albedo = None
    e_dir = None
    if ground_supported:
        albedo = np.asarray(atmo.ground_albedo, dtype=float).ravel().copy()
        if result.method == "cloud":
            # the deck's beam-attenuated ground lighting: the one library
            # implementation of the adopted prescription (single-source with
            # the experiments since M3)
            from .rt.cloud import ground_direct_irradiance as _deck_ground

            e_dir = _deck_ground(atmo, cloud_layer, sun_zenith_deg)
        else:
            # the plain unified-source ground seam at the opaque model bottom
            x_ground = np.array([[0.0, 0.0, float(atmo.r_bounds[0])]])
            e_dir = np.asarray(
                source.direct_irradiance(x_ground,
                                         np.array([0.0, 0.0, 1.0]))[0],
                dtype=float)
    return presentation_from_result(
        result, zenith, azimuth_rel, sun_elevation_deg, sun_azimuth_deg,
        ground_albedo=albedo, ground_direct_irradiance=e_dir,
        sun_apparent_elevation_deg=float(np.asarray(app_e).ravel()[0]),
        sun_visible=bool(np.asarray(lit).ravel()[0]),
        ground_supported=ground_supported)


def solve_dome(atmo, sun_elevation_deg: float, sun_azimuth_deg: float = 0.0,
               n_zenith: int = 46, n_azimuth: int = 91,
               min_zenith_deg: float = 0.0, *,
               approximation=None, cloud_layer=None, single=None,
               multiple=None, single_config=None, sos_config=None,
               mc_config=None, radiance_field=None,
               ground_direct_irradiance=None) -> SkyPresentation:
    """One authoritative solve on the regular half-dome grid, prepared for
    display.

    The grid (zenith x relative azimuth, the historical layout) is sent to
    :func:`rt.solver.solve` **once** -- the absolute view azimuths are the
    relative grid plus the same solar azimuth, and no backend alias or
    combination is resolved here.  The returned carrier binds the result,
    the grid, the adopted source's apparent sun geometry and the matching
    ground material, so every later display is a pure function of bound
    arrays.  ``radiance_field`` replaces the solve with one external-field
    sampling (:func:`presentation_from_field` semantics);
    ``ground_direct_irradiance`` overrides the preparation-bound ground
    irradiance wholesale (the historical seam, e.g. a deck-attenuated
    beam).
    """
    if radiance_field is not None:
        return presentation_from_field(
            radiance_field, atmo, sun_elevation_deg, sun_azimuth_deg,
            n_zenith, n_azimuth, min_zenith_deg,
            ground_direct_irradiance=ground_direct_irradiance)
    from .rt.solver import solve

    zen = np.linspace(min_zenith_deg, 90.0, n_zenith)
    az = np.linspace(0.0, 180.0, n_azimuth)
    Z, A = np.meshgrid(zen, az, indexing="ij")
    result = solve(atmo, 90.0 - float(sun_elevation_deg), Z.ravel(),
                   A.ravel() + float(sun_azimuth_deg),
                   sun_azimuth_deg=float(sun_azimuth_deg),
                   approximation=approximation, cloud_layer=cloud_layer,
                   single=single, multiple=multiple,
                   single_config=single_config, sos_config=sos_config,
                   mc_config=mc_config)
    pres = _bind_display_material(atmo, result, zen, az,
                                  float(sun_elevation_deg),
                                  float(sun_azimuth_deg),
                                  cloud_layer=cloud_layer)
    if ground_direct_irradiance is not None:
        pres = _dc_replace(pres, ground_direct_irradiance=np.array(
            ground_direct_irradiance, dtype=float).ravel().copy())
    return pres


# ---------------------------------------------------------------------------
# pure display of a bound presentation
# ---------------------------------------------------------------------------
#: the chart table's conventional directions (name, zenith, relative azimuth)
DEFAULT_CHART_ROWS = (("zenith", 0.0, 0.0), ("45deg sun-side", 45.0, 0.0),
                      ("45deg anti-solar", 45.0, 180.0),
                      ("horizon sun-side", 88.0, 0.0),
                      ("horizon anti-solar", 88.0, 180.0))


def chart_table(pres: SkyPresentation, rows=DEFAULT_CHART_ROWS):
    """Nearest-grid-node chromaticity/luminance rows from a presentation.

    Pure consumption of the bound arrays: the spectrum of each row comes
    from ``pres.radiance``/``pres.wl``, never from a new solve.  Each row
    reports the *actual* sampled grid angles, so a nearest neighbour is
    never passed off as the exact requested direction.
    """
    out = []
    for name, zen, az in rows:
        i = int(np.argmin(np.abs(pres.zenith - zen)))
        j = int(np.argmin(np.abs(pres.azimuth_rel - az)))
        xyz = color.spectrum_to_xyz(pres.radiance[i, j], pres.wl)
        xy = color.chromaticity(xyz)
        out.append((name, float(pres.zenith[i]), float(pres.azimuth_rel[j]),
                    float(xy[0]), float(xy[1]), float(683.0 * xyz[1])))
    return out


def _sun_marker_geometry(pres: SkyPresentation, sun_disk):
    """Resolve the decorative sun-disk request against the bound geometry.

    The marker is a saturated-white *annotation*, not radiance: it never
    enters charts or ground energy.  ``None`` (auto) draws it only for a
    solver-prepared clear sky whose source geometry is bound and visible;
    cloud scenes default to no marker (a thick deck does not carry an
    unattenuated white disk), ``False`` disables, ``True`` forces (and
    requires the bound apparent geometry).
    """
    if sun_disk is False:
        return None
    auto = sun_disk is None
    if auto and (pres.provenance != "solve" or pres.cloud):
        return None
    if (pres.sun_apparent_elevation_deg is None
            or pres.sun_visible is None
            or not pres.sun_visible):
        if auto:
            return None
        raise ValueError(
            "sun_disk=True needs the presentation's bound apparent solar "
            "geometry, and the source must be visible at the observer")
    return (pres.sun_azimuth_deg, pres.sun_apparent_elevation_deg)


# ---------------------------------------------------------------------------
# diagnostics through to the consumer
# ---------------------------------------------------------------------------
def format_result_summary(source) -> str:
    """A human-readable provenance/limitations summary of one result.

    Reads **only** fields the result itself carries: what actually ran, its
    effective numerics, its convergence/statistics state and the kernels'
    standing approximations.  Missing information is reported as unknown,
    never inferred or filled in from the request.  Accepts a
    :class:`SkyPresentation` or a bare :class:`~skycolor.rt.result.SkyResult`.
    """
    result = getattr(source, "result", None)
    if result is None:
        result = source
    lines = []
    approx = getattr(result, "approximation", "") or "unknown"
    method = getattr(result, "method", "") or "unknown"
    lines.append(f"approximation: {approx} (method {method!r})")
    opt = getattr(result, "options", None) or {}
    if "sun_zenith_deg" in opt:
        lines.append(f"sun: zenith {float(opt['sun_zenith_deg']):.3f} deg, "
                     f"azimuth {float(opt.get('sun_azimuth_deg', 0.0)):.3f} deg "
                     "(input direction pre-refraction)")
    lines.append("source-side refraction: "
                 + ("on" if opt.get("refraction") else "off"))
    approximations = getattr(result, "approximations", None) or {}
    if approximations:
        lines.append("components:")
        for name, comp in approximations.items():
            lines.append(f"  {name}: {comp.get('kernel', 'unknown')} "
                         f"(source_refraction={comp.get('source_refraction')})")
            for key, text in (comp.get("approximations", None) or {}).items():
                lines.append(f"    {key}: {text}")
    else:
        lines.append("components: unknown (no provenance recorded)")
    effective = getattr(result, "effective", None) or {}
    if effective:
        parts = "; ".join(f"{k} {v}" for k, v in effective.items())
        lines.append(f"effective: {parts}")
    else:
        lines.append("effective: unknown")
    d = getattr(result, "diagnostics", None) or {}
    if "orders" in d:
        conv = d.get("converged")
        conv_txt = "yes" if conv is True else ("NO" if conv is False else "unknown")
        note = ("  -- multiple scattering truncated; treat with caution"
                if conv is False else "")
        lines.append(f"sos: orders={d['orders']} converged={conv_txt}{note}")
    if "n_photons" in d or "undetected_wl" in d:
        mc_eff = effective.get("mc", None) or {}
        lines.append(
            "mc: photons_run={} anchors_run={} coarse grid {}x{} seed {}".format(
                mc_eff.get("n_photons_run", "unknown"),
                mc_eff.get("n_anchor_run", "unknown"),
                mc_eff.get("n_zen_run", "unknown"),
                mc_eff.get("n_az_run", "unknown"),
                mc_eff.get("seed", "unknown")))
        und = d.get("undetected_wl")
        und = [] if und is None else np.asarray(und, dtype=float).ravel()
        if und.size:
            shown = ", ".join(f"{float(w):.0f}" for w in und[:8])
            more = " ..." if und.size > 8 else ""
            lines.append(
                f"mc undetected wavelengths: {und.size} [{shown}{more}] nm -- "
                "no Monte Carlo contribution detected; report as unresolved, "
                "not zero")
    stderr = getattr(result, "stderr", None)
    if stderr is not None:
        arr = np.asarray(stderr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if (opt.get("mc_run", True) is False) or not finite.size or not np.any(finite):
            lines.append("mc stderr: identically zero (no Monte Carlo term ran)")
        else:
            lines.append(
                "mc stderr (W m-2 sr-1 nm-1): median {:.3e}, max {:.3e} -- "
                "composite estimate: the MC standard error propagated through "
                "the angular smoother and wavelength interpolation plus a ~3% "
                "smoother resolution-bias floor (in quadrature); not the total "
                "physical error of the model, and the photon budget is not "
                "tail coverage".format(float(np.median(finite)),
                                       float(finite.max())))
    return "\n".join(lines)


def _warn_result_limits(result) -> None:
    """Visible limitations for the convenience ndarray entries.

    Non-convergence, undetected Monte Carlo wavelengths and the statistical
    nature of MC stderr must not hide behind a dict the user never opens.
    The presentation carrier path stays silent here: its consumers read
    :func:`format_result_summary` instead.
    """
    import warnings

    d = getattr(result, "diagnostics", None) or {}
    if d.get("converged") is False:
        warnings.warn(
            f"SOS did not converge ({d.get('orders', '?')} orders): the "
            "plane-parallel multiple-scattering term is truncated",
            RuntimeWarning, stacklevel=3)
    und = d.get("undetected_wl")
    if und is not None and np.size(und):
        warnings.warn(
            f"Monte Carlo detected no contribution at {np.size(und)} "
            "wavelength(s): treat them as unresolved, not zero",
            RuntimeWarning, stacklevel=3)
    stderr = getattr(result, "stderr", None)
    if stderr is not None:
        opt = getattr(result, "options", None) or {}
        arr = np.asarray(stderr, dtype=float)
        if (opt.get("mc_run", True) is not False) and np.any(arr > 0):
            warnings.warn(
                "higher orders are a Monte Carlo statistical estimate: stderr "
                "is a composite (smoother + interpolation + bias floor), not "
                "the total physical error",
                RuntimeWarning, stacklevel=3)


def render_result(pres: SkyPresentation, *, camera: str = "panorama",
                  size=(1000, 500), exposure=None, tonemap: str = "aces",
                  adaptation=None, reference_percentile: float = 92.0,
                  sun_disk=None, ground: bool = True, hfov_deg: float = 90.0,
                  pitch_deg: float = 0.0, yaw_deg: float = 0.0, white_xy=None,
                  preserve_chromaticity: bool = True,
                  ground_direct_transmittance=None) -> np.ndarray:
    """Render a bound presentation to an RGB image.  Pure: no solve, no
    solar-source query -- every spectral value comes from ``pres``.

    The ground is the Lambertian model bottom lit by the bound direct
    irradiance plus the diffuse integral of the *same* bound dome.  An
    external-field presentation must carry a matching
    ``ground_direct_irradiance`` (or the retired
    ``ground_direct_transmittance``) for ``ground=True`` -- a custom sky is
    never silently paired with the clear-sky beam; giving both overrides is
    an error, and ``ground=False`` skips the check entirely.  A solver
    presentation whose observer is off the model bottom has no ground
    display (raise instead of computing ground lighting at another height).
    """
    import warnings

    if camera == "panorama":
        az, el = panorama_grid(size[0], size[1])
        mask = None
    elif camera == "fisheye":
        az, el, mask = fisheye_grid(size[0])
    elif camera == "perspective":
        az, el = perspective_grid(size[0], size[1], hfov_deg, pitch_deg, yaw_deg)
        mask = None
    else:
        raise ValueError(f"unknown camera {camera}")

    e_dir = pres.ground_direct_irradiance
    if ground:
        if ground_direct_transmittance is not None:
            if e_dir is not None:
                raise ValueError(
                    "pass either the bound ground_direct_irradiance or the "
                    "deprecated ground_direct_transmittance, not both")
            warnings.warn(
                "ground_direct_transmittance is deprecated by the unified "
                "solar source; prepare the presentation with "
                "ground_direct_irradiance instead (it carries the disk "
                "geometry)", DeprecationWarning, stacklevel=2)
            from .spectra import solar_irradiance

            T = np.asarray(ground_direct_transmittance, dtype=float)
            cosz = max(np.cos(np.radians(pres.sun_zenith_deg)), 0.0)
            e_dir = solar_irradiance(pres.wl) * T * cosz
        if e_dir is None:
            if pres.provenance == "solve" and not pres.ground_supported:
                raise ValueError(
                    "the presentation's observer is off the model bottom, so "
                    "there is no ground display; use ground=False")
            raise ValueError(
                "the presentation carries no ground_direct_irradiance; bind "
                "matching ground lighting at preparation time (or display "
                "with ground=False)")

    dome = dict(zenith=pres.zenith, azimuth_rel=pres.azimuth_rel,
                radiance=pres.radiance)
    rel = (az - pres.sun_azimuth_deg + 180.0) % 360.0 - 180.0
    above = el >= 0.0
    rad = sample_dome(dome, 90.0 - np.clip(el, 0.0, 90.0), rel)
    if ground:
        e_diff = hemispherical_irradiance(pres.radiance, pres.zenith,
                                          pres.azimuth_rel)
        albedo = pres.ground_albedo
        if albedo is None:
            raise ValueError(
                "the presentation carries no ground_albedo; bind it at "
                "preparation time (or display with ground=False)")
        e_total = np.broadcast_to(np.asarray(e_dir, dtype=float),
                                  (pres.wl.size,)).copy() + e_diff
        Lg = np.asarray(albedo, dtype=float) / np.pi * e_total
        rad = np.where(above[..., None], rad, Lg[None, None, :])

    if white_xy is None:
        white_xy = color.D65_XY
    if adaptation is None:
        white_xy = color.D65_XY
    rgb = color.render_rgb(rad, pres.wl, exposure=exposure, tonemap=tonemap,
                           adaptation=adaptation, white_xy=white_xy,
                           reference_percentile=reference_percentile,
                           preserve_chromaticity=preserve_chromaticity)

    marker = _sun_marker_geometry(pres, sun_disk)
    if marker is not None:
        m_az, m_el = marker
        ang = np.hypot(((az - m_az + 180) % 360 - 180) * np.cos(np.radians(el)),
                       el - m_el)
        rgb[ang < 0.30] = 1.0
    if mask is not None:
        rgb = rgb * mask[..., None]
    return rgb


# ---------------------------------------------------------------------------
# high level renderer (thin: prepare once -> render_result)
# ---------------------------------------------------------------------------
def ground_radiance(atmo, dome, sun_zenith_deg, ground_albedo=None,
                    direct_transmittance=None, *, direct_irradiance=None):
    """Spectral radiance of a Lambertian ground illuminated by sun + sky.

    Legacy ndarray entry, kept for direct callers: the default direct term
    is the unified source's horizontal direct irradiance at the observer's
    ground.  ``direct_irradiance`` overrides it wholesale -- e.g. a cloud
    deck's beam-attenuated irradiance.  ``direct_transmittance`` is the
    retired centre-beam override: it still works for explicit old callers
    (multiplied by F*cos(z) exactly as before) but is deprecated; passing it
    together with ``direct_irradiance`` is an error, and nothing defaults to
    it.  New display code binds the irradiance on a presentation and uses
    :func:`render_result` instead.
    """
    import warnings

    from .rt.single import direct_sun_transmittance  # noqa: F401  (re-exported seam)
    from .solar_source import SolarSource
    from .spectra import solar_irradiance

    if direct_irradiance is not None and direct_transmittance is not None:
        raise ValueError(
            "pass either direct_irradiance or the deprecated "
            "direct_transmittance, not both")
    albedo = (atmo.ground_albedo if ground_albedo is None
              else np.full(len(atmo.wl), ground_albedo))
    E_diff = hemispherical_irradiance(dome["radiance"], dome["zenith"], dome["azimuth_rel"])
    if direct_irradiance is not None:
        E_dir = np.broadcast_to(np.asarray(direct_irradiance, dtype=float),
                                (len(atmo.wl),)).copy()
    elif direct_transmittance is not None:
        warnings.warn(
            "direct_transmittance is deprecated by the unified solar source; "
            "pass direct_irradiance instead (it carries the disk geometry)",
            DeprecationWarning, stacklevel=2)
        T = np.asarray(direct_transmittance, dtype=float)
        cosz = max(np.cos(np.radians(sun_zenith_deg)), 0.0)
        E_dir = solar_irradiance(atmo.wl) * T * cosz
    else:
        source = SolarSource(atmo, float(sun_zenith_deg), 0.0)
        x_ground = np.array([[0.0, 0.0, atmo.r_bounds[0] + (atmo.elevation_km - atmo.z_bounds[0]) * 1e3]])
        E_dir = np.asarray(source.direct_irradiance(x_ground, np.array([0.0, 0.0, 1.0]))[0],
                           dtype=float)
    return np.asarray(albedo) / np.pi * (E_dir + E_diff), E_dir + E_diff


def render_scene(atmo, sun_elevation_deg: float, sun_azimuth_deg: float = 0.0,
                 camera: str = "panorama", size=(1000, 500), exposure=None,
                 tonemap: str = "aces", adaptation=None,
                 reference_percentile: float = 92.0, sun_disk=True,
                 ground: bool = True, n_zenith: int = 50, n_azimuth: int = 91,
                 hfov_deg: float = 90.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 white_xy=None, preserve_chromaticity: bool = True,
                 radiance_field=None, ground_direct_transmittance=None,
                 *, ground_direct_irradiance=None, approximation=None,
                 cloud_layer=None, single_config=None, sos_config=None,
                 mc_config=None) -> np.ndarray:
    """Render a complete scene (sky + ground + sun disk) to an RGB image.

    Thin forwarder: prepares the presentation once (:func:`solve_dome`, or
    one sampling of ``radiance_field``) and delegates to
    :func:`render_result`; the ndarray return contract is unchanged.  The
    keyword-only solver options (``approximation``/``cloud_layer``/grouped
    configs) pass through item by item -- no backend combination is
    assembled here.  ``radiance_field`` / ``ground_direct_irradiance`` keep
    their seam meaning: a non-default field usually needs a matching ground
    irradiance (a cloudy sky also attenuates the direct beam), otherwise
    the ground display fails loudly instead of pairing the custom sky with
    the clear beam.  ``ground_direct_transmittance`` is the retired
    override; see :func:`render_result`.
    """
    pres = solve_dome(atmo, sun_elevation_deg, sun_azimuth_deg,
                      n_zenith=n_zenith, n_azimuth=n_azimuth,
                      approximation=approximation, cloud_layer=cloud_layer,
                      single_config=single_config, sos_config=sos_config,
                      mc_config=mc_config, radiance_field=radiance_field,
                      ground_direct_irradiance=ground_direct_irradiance)
    if pres.result is not None:
        _warn_result_limits(pres.result)
    return render_result(pres, camera=camera, size=size, exposure=exposure,
                         tonemap=tonemap, adaptation=adaptation,
                         reference_percentile=reference_percentile,
                         sun_disk=sun_disk, ground=ground, hfov_deg=hfov_deg,
                         pitch_deg=pitch_deg, yaw_deg=yaw_deg,
                         white_xy=white_xy,
                         preserve_chromaticity=preserve_chromaticity,
                         ground_direct_transmittance=ground_direct_transmittance)


def save_png(rgb: np.ndarray, path):
    from PIL import Image

    Image.fromarray(color.to_uint8(rgb)).save(path)
    return path


def save_hdr(radiance: np.ndarray, wl, path):
    np.savez_compressed(path, radiance=radiance, wl=wl)
    return path
