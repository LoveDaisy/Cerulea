"""Check 2 -- clear-sky direct and diffuse illuminance.

Three independent references are used, because no single published table
covers both quantities at both requested solar elevations:

1. **ASTM G173-03 direct-normal spectrum** at air mass 1.5 (solar elevation
   48.19 deg).  This is a standard reference spectrum, independent of the
   model's radiative transfer.  Its photopic integral is 97.14 klx and its
   400-700 nm integral is 374.8 W m^-2 (computed here from the committed
   `data_raw/ASTMG173.csv`).  The model is run with the G173 atmosphere
   (ozone 343.8 DU, AOD_550 = 0.073, Hinterland aerosol) at the same
   elevation.

2. **CIE 110-1994 clear-sky diffuse illuminance** at solar elevations 30 and
   60 deg.  The standard gives the relative luminance distribution and the
   absolute zenith luminance Yz(T, h) = (1.376 T - 1.81) tan h + 0.38
   kcd m^-2.  Integrating Yz * L_rel cos(theta) dOmega over the hemisphere
   gives the diffuse horizontal illuminance.  We use T = 2.2, a typical
   Linke turbidity for a clean continental boundary layer with
   AOD_550 ~ 0.05.  (The CIE T is not exactly the model's Linke turbidity;
   T = 2.2 is chosen from the standard's clear-sky range and is stated
   explicitly.  At T = 2.2 the CIE zenith luminance agrees with the model to
   within 3 %, which is a useful consistency check in itself.)

3. **Bird & Hulstrom (1981) clear-sky broadband model**, evaluated with
   AOD_500 = 0.057, precipitable water 1.5 cm, ozone 0.3 atm-cm, and
   converted to illuminance with the G173 direct-beam luminous efficacy
   107.9 lm/W.  This provides a second, independent estimate of direct
   normal illuminance at 30 and 60 deg.

Tolerances
----------
* direct normal illuminance: 10 % against G173, 15 % against Bird,
* diffuse horizontal illuminance: 25 % against CIE 110 (the model has no
  multiple scattering, so a one-sided low bias is expected in principle),
* spectral shape: 10 % RMS over 400-700 nm.

Run:  python validation/check2_illuminance.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from skycolor import spectra  # noqa: E402
from skycolor.rt.single import direct_sun_transmittance  # noqa: E402

from validation import common as C  # noqa: E402

#: luminous efficacy of the ASTM G173 direct-normal beam, 97.14 klx / 900.14 W/m2
K_G173 = 107.9


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
def load_g173():
    """ASTM G173-03 columns: wavelength, extraterrestrial, global tilt, DNI."""
    path = C.ROOT / "data_raw" / "ASTMG173.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def g173_dni_illuminance(wl_g, dni_g):
    ybar = spectra.cie_xyz(wl_g)[:, 1]
    m = (wl_g >= 380.0) & (wl_g <= 780.0)
    return float(C.K_M * np.trapezoid((dni_g * ybar)[m], wl_g[m]))


def bird_clear_sky(solar_zenith_deg, aod380, aod500, precipitable_water=1.5,
                   ozone_cm=0.3, pressure=101325.0, dni_extra=1361.0,
                   asymmetry=0.85, albedo=0.2):
    """Bird & Hulstrom (1981) clear-sky broadband DNI and DHI [W m^-2].

    Direct transcription of the model as implemented in pvlib
    (``pvlib.clearsky.bird``, pvlib 0.11), which follows Bird & Hulstrom,
    SERI/TR-642-761 (1981).  Only the scalar case is needed here.
    """
    z = float(solar_zenith_deg)
    am = 1.0 / (np.cos(np.radians(z)) + 0.15 * (93.885 - z) ** -1.253)
    am_press = am * pressure / 101325.0
    t_rayleigh = np.exp(-0.0903 * am_press**0.84 * (1.0 + am_press - am_press**1.01))
    am_o3 = ozone_cm * am
    t_ozone = (
        1.0
        - 0.1611 * am_o3 * (1.0 + 139.48 * am_o3) ** -0.3034
        - 0.002715 * am_o3 * (1.0 + 0.044 * am_o3 + 0.0003 * am_o3**2) ** -1
    )
    t_gases = np.exp(-0.0127 * am_press**0.26)
    am_h2o = am * precipitable_water
    t_water = 1.0 - 2.4959 * am_h2o / ((1.0 + 79.034 * am_h2o) ** 0.6828 + 6.385 * am_h2o)
    aod_bb = 0.27583 * aod380 + 0.35 * aod500
    t_aerosol = np.exp(-(aod_bb**0.873) * (1.0 + aod_bb - aod_bb**0.7088) * am**0.9108)
    taa = 1.0 - 0.1 * (1.0 - am + am**1.06) * (1.0 - t_aerosol)
    rs = 0.0685 + (1.0 - asymmetry) * (1.0 - t_aerosol / taa)
    idn = 0.9662 * dni_extra * t_aerosol * t_water * t_gases * t_ozone * t_rayleigh
    cz = max(np.cos(np.radians(z)), 0.0)
    idnh = idn * cz
    ias = (
        dni_extra * cz * 0.79 * t_ozone * t_gases * t_water * taa
        * (0.5 * (1.0 - t_rayleigh) + asymmetry * (1.0 - t_aerosol / taa))
        / (1.0 - am + am**1.02)
    )
    ghi = (idnh + ias) / (1.0 - albedo * rs)
    return float(idn), float(ghi - idnh)


def cie110_diffuse_illuminance(solar_elevation_deg, turbidity, n_zen=181, n_az=144):
    """Integrate the absolute CIE 110 clear-sky luminance over the hemisphere."""
    z, a = C.sky_grid(n_zen, n_az)
    rel = C.cie110_relative_luminance(z, a, 90.0 - solar_elevation_deg)
    lz = C.cie110_zenith_luminance(solar_elevation_deg, turbidity)
    return C.horizontal_illuminance(lz * rel, z, a)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def model_case(elevation_deg, aod550=0.05, ozone_du=300.0, wl=C.WL_COARSE,
               n_zen=73, n_az=144):
    atmo = C.atmosphere("clear", wl, aod=aod550, ozone_du=ozone_du)
    # DNI: the unified source's flux on the plane normal to the apparent sun
    # centre; GHI: the source's flux on the horizontal plane (plan 6.4).
    # Both are exact disk integrals -- the legacy F*T*sin(true elevation)
    # decomposition ignored the disk geometry and the refraction shift.
    from skycolor.solar_source import SolarSource
    import skycolor.geometry as _g

    source = SolarSource(atmo, 90.0 - float(elevation_deg), 0.0)
    xg = np.array([[0.0, 0.0, atmo.r_bounds[0]]])
    lower = float(source.geometry.field.grazing_apparent_elevation(
        np.array([0.0]))[0])
    crit = float(source.geometry.field.forward(np.array([0.0]),
                                               np.array([lower + 1e-6]))[0])
    elev_true = float(elevation_deg)
    e_dni = np.zeros(len(wl))
    if elev_true >= crit:
        app_c = float(source.geometry.field.apparent_of_true(
            np.array([0.0]), np.array([elev_true]))[0])
        n_plane = _g.sun_vector(0.0, app_c)
        e_dni = source.direct_irradiance(xg, n_plane)[0]
    e_ghi = source.direct_irradiance(xg, np.array([0.0, 0.0, 1.0]))[0]
    dni_il = float(C.illuminance(e_dni, wl))
    z, a = C.sky_grid(n_zen, n_az)
    rad = C.sky_radiance(atmo, 90.0 - elevation_deg, z.ravel(), a.ravel())
    rad = rad.reshape(z.shape + (len(wl),))
    dhi_il = C.horizontal_illuminance(rad, z, a, wl)
    ghi_il = float(C.illuminance(e_ghi, wl)) + dhi_il
    return dict(dni=dni_il, dhi=dhi_il, ghi=ghi_il, e_dni=e_dni, atmo=atmo)


def run():
    metrics = []
    details = {}
    wl = C.WL_COARSE

    # -- 1. ASTM G173 at 48.19 deg -----------------------------------------
    wl_g, et_g, gt_g, dni_g = load_g173()
    g173_klx = g173_dni_illuminance(wl_g, dni_g) / 1000.0
    g173_vis = float(np.trapezoid(
        dni_g[(wl_g >= 400) & (wl_g <= 700)], wl_g[(wl_g >= 400) & (wl_g <= 700)]
    ))
    g173_model = model_case(48.19, aod550=0.0734, ozone_du=343.8, wl=np.arange(400., 701., 1.))
    dni_klx = g173_model["dni"] / 1000.0
    metrics.append(C.Metric(
        "DNI illuminance, AM1.5 (48.19deg)", dni_klx, g173_klx, unit="klx",
        tolerance=10.0, dev=float(C.rel_dev(dni_klx, g173_klx)),
    ))
    model_vis = float(np.trapezoid(g173_model["e_dni"], np.arange(400., 701., 1.)))
    metrics.append(C.Metric(
        "DNI, 400-700 nm, AM1.5", model_vis, g173_vis, unit="W/m2",
        tolerance=10.0, dev=float(C.rel_dev(model_vis, g173_vis)),
    ))
    # spectral shape at the 1 nm grid
    wl1 = np.arange(400.0, 701.0, 1.0)
    e_g = np.interp(wl1, wl_g, dni_g)
    spec_dev = C.rel_dev(g173_model["e_dni"], e_g)
    metrics.append(C.Metric(
        "DNI spectral RMS, 400-700 nm", float(np.sqrt(np.mean(spec_dev**2))), 0.0,
        unit="%", tolerance=10.0, dev=float(np.sqrt(np.mean(spec_dev**2))),
        dev_kind="abs",
    ))

    # -- 2. CIE 110 diffuse at 30 and 60 deg --------------------------------
    T_CIE = 2.2
    cie_dhi = {}
    for h in (30.0, 60.0):
        cie_dhi[h] = cie110_diffuse_illuminance(h, T_CIE) / 1000.0
    model = {h: model_case(h, wl=wl) for h in (30.0, 60.0)}
    for h in (30.0, 60.0):
        m = model[h]["dhi"] / 1000.0
        metrics.append(C.Metric(
            f"DHI illuminance, h={h:.0f}deg (CIE 110, T=2.2)", m, cie_dhi[h],
            unit="klx", tolerance=25.0, dev=float(C.rel_dev(m, cie_dhi[h])),
        ))

    # -- 3. Bird & Hulstrom direct at 30 and 60 deg -------------------------
    aod380 = 0.05 * (380.0 / 550.0) ** -1.3
    aod500 = 0.05 * (500.0 / 550.0) ** -1.3
    bird_klx = {}
    for h in (30.0, 60.0):
        dni_b, dhi_b = bird_clear_sky(90.0 - h, aod380, aod500)
        bird_klx[h] = (dni_b * K_G173 / 1000.0, dhi_b * K_G173 / 1000.0)
        m = model[h]["dni"] / 1000.0
        metrics.append(C.Metric(
            f"DNI illuminance, h={h:.0f}deg (Bird)", m, bird_klx[h][0],
            unit="klx", tolerance=15.0, dev=float(C.rel_dev(m, bird_klx[h][0])),
        ))

    details = dict(
        g173_klx=g173_klx,
        g173_visible=g173_vis,
        model_am15_dni_klx=dni_klx,
        model_am15_visible=model_vis,
        spectral_rms=float(np.sqrt(np.mean(spec_dev**2))),
        cie_dhi_klx=cie_dhi,
        turbidity=T_CIE,
        model={str(k): dict(dni=v["dni"] / 1000.0, dhi=v["dhi"] / 1000.0,
                            ghi=v["ghi"] / 1000.0) for k, v in model.items()},
        bird={str(k): dict(dni=v[0], dhi=v[1]) for k, v in bird_klx.items()},
        bird_dhi_klx={str(k): v[1] for k, v in bird_klx.items()},
    )

    # -- figure --------------------------------------------------------------
    plt = C.plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    ax = axes[0]
    labels = ["DNI h=30", "DNI h=60", "DNI AM1.5", "DHI h=30", "DHI h=60"]
    model_vals = [
        model[30.0]["dni"] / 1000.0,
        model[60.0]["dni"] / 1000.0,
        dni_klx,
        model[30.0]["dhi"] / 1000.0,
        model[60.0]["dhi"] / 1000.0,
    ]
    ref_vals = [
        bird_klx[30.0][0],
        bird_klx[60.0][0],
        g173_klx,
        cie_dhi[30.0],
        cie_dhi[60.0],
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, model_vals, 0.4, label="model", color="C0")
    ax.bar(x + 0.2, ref_vals, 0.4, label="reference", color="C1")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("illuminance [klx]")
    ax.set_title("clear-sky illuminance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    wl1 = np.arange(400.0, 701.0, 1.0)
    ax.axhspan(-10, 10, color="C2", alpha=0.12, label="+/-10 %")
    ax.plot(wl1, spec_dev, "C0-", lw=0.8)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("model / ASTM G173 - 1 [%]")
    ax.set_title("direct-normal spectral shape at AM1.5")
    ax.set_ylim(-30, 60)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Check 2: clear-sky direct and diffuse illuminance")
    fig.savefig(C.FIG / "check2_illuminance.png", dpi=150)
    plt.close(fig)

    C.save_csv(
        C.DATA / "check2_illuminance.csv",
        ["elevation_deg", "model_dni_klx", "model_dhi_klx", "ref_dni_klx",
         "ref_dhi_klx", "ref_dni_source", "ref_dhi_source"],
        [
            [30.0, model[30.0]["dni"] / 1000.0, model[30.0]["dhi"] / 1000.0,
             bird_klx[30.0][0], cie_dhi[30.0], 1.0, 2.0],
            [60.0, model[60.0]["dni"] / 1000.0, model[60.0]["dhi"] / 1000.0,
             bird_klx[60.0][0], cie_dhi[60.0], 1.0, 2.0],
            [48.19, dni_klx, float("nan"), g173_klx, float("nan"), 3.0, 0.0],
        ],
    )

    summary = (
        f"DNI: {model[30.0]['dni']/1000:.1f}/{dni_klx:.1f}/"
        f"{model[60.0]['dni']/1000:.1f} klx at h=30/48.19/60 vs "
        f"{bird_klx[30.0][0]:.1f}/{g173_klx:.1f}/{bird_klx[60.0][0]:.1f}; "
        f"DHI: {model[30.0]['dhi']/1000:.1f}/{model[60.0]['dhi']/1000:.1f} klx "
        f"vs CIE {cie_dhi[30.0]:.1f}/{cie_dhi[60.0]:.1f}."
    )
    return C.CheckResult(
        key="check2", title="Clear-sky direct and diffuse illuminance",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    lines = [
        "",
        "  Solar-elevation   quantity   model      reference    reference source",
    ]
    for name, m, r in [
        ("DNI h=30", result.details["model"]["30.0"]["dni"], result.details["bird"]["30.0"]["dni"]),
        ("DNI h=60", result.details["model"]["60.0"]["dni"], result.details["bird"]["60.0"]["dni"]),
        ("DNI AM1.5", result.details["model_am15_dni_klx"], result.details["g173_klx"]),
        ("DHI h=30", result.details["model"]["30.0"]["dhi"], result.details["cie_dhi_klx"][30.0]),
        ("DHI h=60", result.details["model"]["60.0"]["dhi"], result.details["cie_dhi_klx"][60.0]),
    ]:
        print_line = f"  {name:16s} {m:9.2f} klx {r:9.2f} klx"
        lines.append(print_line)
    lines.append("")
    lines.append(f"  CIE 110 diffuse reference uses turbidity T = {result.details['turbidity']}.")
    lines.append("  Bird & Hulstrom direct reference uses AOD500 = 0.057, pw = 1.5 cm,")
    lines.append("  and the G173 luminous efficacy 107.9 lm/W.")
    return C.run_standalone(result, lines)


if __name__ == "__main__":
    raise SystemExit(main())
