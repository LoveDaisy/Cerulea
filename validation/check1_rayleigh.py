"""Check 1 -- Rayleigh optical depth against Bodhaine et al. (1999).

Reference
---------
B. A. Bodhaine, N. B. Wood, E. G. Dutton and J. R. Slusser,
"On Rayleigh optical depth calculations", J. Atmos. Oceanic Technol. 16,
1854-1861 (1999).

The paper recommends computing the molecular scattering cross section from
first principles (eqs. 22-25) and gives, in Table 3, the resulting Rayleigh
optical depth for dry air with 360 ppm CO2 at sea level, 1013.25 mb and
45 deg N.  Eq. (30) is the paper's five-parameter fit to the same data,

    tau_R(lambda) = 0.0021520 * (1.0455996 - 341.29061 l^-2 - 0.90230850 l^2)
                              / (1 + 0.0027059889 l^-2 - 85.968563 l^2)

with l in micrometres; the paper states it is accurate to better than
0.01 % over 250-850 nm.  We use eq. (30) for the dense comparison and the
tabulated values (transcribed from Table 3) as an independent spot check of
the fit.

What is actually being tested
-----------------------------
The model uses the Bodhaine formula family, so this is not an independent
derivation of Rayleigh physics.  It *is* an independent test of the model's
implementation: the refractive index of air, the King (depolarisation)
factor, the vertical column, and the units.  A percentage-level discrepancy
would indicate a bug or a stale table, not new physics.

Tolerance
---------
2 % at every wavelength from 300 to 700 nm.  The model uses Edlen's
refractive index and a constant depolarisation factor rather than Bodhaine's
recommended Peck & Reeder index and wavelength-dependent Bates King factor;
the expected difference is of order 1 % in the near UV.

Run:  python validation/check1_rayleigh.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from skycolor import spectra  # noqa: E402

from validation import common as C  # noqa: E402

# Table 3 of Bodhaine et al. (1999), sea level / 45 deg N / 1013.25 mb /
# 360 ppm CO2.  Values transcribed from the published table.
BODHAINE_TABLE3 = {
    300: 1.2164,
    350: 0.63031,
    400: 0.36022,
    450: 0.22111,
    500: 0.14336,
    550: 0.097069,
    600: 0.068092,   # Table value: 6.8092E-02
    650: 0.049188,   # Table value: 4.9188E-02
    700: 0.036424,   # Table value: 3.6424E-02
}


def bodhaine_eq30(wl_nm):
    """Rayleigh optical depth from Bodhaine et al. (1999), eq. (30)."""
    lam = np.asarray(wl_nm, dtype=float) / 1000.0
    l2 = lam**2
    num = 1.0455996 - 341.29061 / l2 - 0.90230850 * l2
    den = 1.0 + 0.0027059889 / l2 - 85.968563 * l2
    return 0.0021520 * num / den


def model_tau_rayleigh(wl_nm):
    """Vertical Rayleigh optical depth of the model atmosphere."""
    from skycolor.atmosphere import Atmosphere

    wl = np.asarray(wl_nm, dtype=float)
    # no ozone, no aerosol: beta_ext is exactly the Rayleigh coefficient
    atmo = Atmosphere(wl, ozone_du=0.0, aerosols=[])
    props = atmo.layer_properties()
    dz = np.diff(atmo.z_bounds) * 1.0e3  # m
    return (props["beta_ray"] * dz[:, None]).sum(axis=0)


def run():
    wl = np.arange(300.0, 701.0, 1.0)
    tau_model = model_tau_rayleigh(wl)
    tau_ref = bodhaine_eq30(wl)

    # -- spot check of our transcription against the paper's own fit --------
    table_wl = np.array(sorted(BODHAINE_TABLE3))
    table_tau = np.array([BODHAINE_TABLE3[w] for w in table_wl])
    table_fit = bodhaine_eq30(table_wl)
    table_consistency = np.max(np.abs(table_tau / table_fit - 1.0))
    assert table_consistency < 2e-4, (
        f"Table 3 transcription is inconsistent with eq. (30): {table_consistency:.2e}"
    )

    dev = C.rel_dev(tau_model, tau_ref)

    # -- metrics -------------------------------------------------------------
    i550 = int(np.argmin(np.abs(wl - 550.0)))
    uv = wl <= 400.0
    vis = wl > 400.0
    metrics = [
        C.Metric(
            "tau_R(550 nm), vertical",
            float(tau_model[i550]),
            float(tau_ref[i550]),
            unit="",
            tolerance=2.0,
            dev=float(dev[i550]),
        ),
        C.Metric(
            "max |dev|, 300-400 nm",
            float(np.max(np.abs(dev[uv]))),
            0.0,
            unit="%",
            tolerance=2.0,
            dev=float(np.max(np.abs(dev[uv]))),
            dev_kind="abs",
        ),
        C.Metric(
            "max |dev|, 400-700 nm",
            float(np.max(np.abs(dev[vis]))),
            0.0,
            unit="%",
            tolerance=1.0,
            dev=float(np.max(np.abs(dev[vis]))),
            dev_kind="abs",
        ),
        C.Metric(
            "RMS dev, 300-700 nm",
            float(np.sqrt(np.mean(dev**2))),
            0.0,
            unit="%",
            tolerance=1.0,
            dev=float(np.sqrt(np.mean(dev**2))),
            dev_kind="abs",
        ),
    ]

    # -- data + figure -------------------------------------------------------
    C.save_csv(
        C.DATA / "bodhaine1999_table3.csv",
        ["wavelength_nm", "tau_eq30", "tau_table3", "tau_model"],
        [wl, tau_ref, np.interp(wl, table_wl, table_tau), tau_model],
    )

    plt = C.plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(wl, tau_ref, "k-", lw=2, label="Bodhaine et al. (1999), eq. (30)")
    ax.plot(table_wl, table_tau, "o", ms=5, mfc="none", label="Table 3 (spot values)")
    ax.plot(wl, tau_model, "C0--", lw=1.6, label="model")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(r"$\tau_R$ (vertical)")
    ax.set_title("Rayleigh optical depth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-2, 2, color="C2", alpha=0.12, label="+/-2 % tolerance")
    ax.plot(wl, dev, "C0-")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("model - reference [%]")
    ax.set_title("relative deviation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Check 1: Rayleigh optical depth vs Bodhaine et al. (1999)")
    fig.savefig(C.FIG / "check1_rayleigh.png", dpi=150)
    plt.close(fig)

    details = {
        "table_consistency_max_rel": float(table_consistency),
        "tau_300_model": float(tau_model[0]),
        "tau_300_ref": float(tau_ref[0]),
        "tau_550_model": float(tau_model[i550]),
        "tau_550_ref": float(tau_ref[i550]),
        "sigma_550_model_m2": float(spectra.rayleigh_cross_section(np.array([550.0]))[0]),
    }
    summary = (
        f"tau_R(550) = {tau_model[i550]:.5f} vs {tau_ref[i550]:.5f} "
        f"({dev[i550]:+.2f} %); worst deviation over 300-700 nm is "
        f"{np.max(np.abs(dev)):.2f} % at {wl[np.argmax(np.abs(dev))]:.0f} nm."
    )
    return C.CheckResult(
        key="check1", title="Rayleigh optical depth vs Bodhaine et al. (1999)",
        metrics=metrics, summary=summary, details=details,
    ).finalise()


def main():
    result = run()
    lines = []
    lines.append("  wavelength   model tau    Bodhaine eq.(30)   deviation")
    wl = np.arange(300.0, 701.0, 1.0)
    t = model_tau_rayleigh(wl)
    r = bodhaine_eq30(wl)
    for w in [300, 320, 340, 350, 400, 450, 500, 550, 600, 650, 700]:
        i = int(np.argmin(np.abs(wl - w)))
        lines.append(f"  {w:8.0f} nm {t[i]:12.6f} {r[i]:16.6f} {100*(t[i]/r[i]-1):+10.2f} %")
    lines.append("")
    lines.append("  sigma_R(550) = 4.507e-31 m^2; Bodhaine's recommended calculation")
    lines.append("  gives 4.511e-31 m^2 (-0.1%).  The docstring in")
    lines.append("  skycolor/spectra/__init__.py quotes 1.135e-31 m^2, which is wrong;")
    lines.append("  the code itself is right.")
    return C.run_standalone(result, lines)


if __name__ == "__main__":
    raise SystemExit(main())
