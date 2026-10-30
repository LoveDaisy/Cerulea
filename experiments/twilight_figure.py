"""Rebuild the deep-twilight figure from the saved sweep (no Monte Carlo).

`twilight_nocturnal.py` produces the data (12 minutes of Monte Carlo) and a
preview figure.  The preview's fisheye panels are rendered from the engine's
*coarse* angular grid and show the grid cells as polygons, which is honest but
useless; this script redraws the quantitative panels from the saved JSON and
omits them.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, save

D = json.loads((OUT / "11_twilight_deep.json").read_text())
h = -np.array(D["elevations"])
ss = np.array(D["single"])
det = np.array(D["deterministic"])
patat = {float(k): v for k, v in D["patat"].items()}
pk = np.array(sorted(patat))
pv = np.array([patat[k] for k in pk])

# The engine points below are the *production* run (5 wavelength anchors x
# 3e7 photons), from docs/TWILIGHT.md section 2.2.  The JSON written by the
# --quick preview run is 20x cheaper and its engine values carry 40-70 %
# statistical error, so they are not used for the figure.
ENG = {  # solar elevation [deg]: (luminance, sigma)
    -3.0: (37.80, 0.24),
    -6.0: (0.978, 0.018),
    -9.0: (0.0238, 0.0043),
    -12.0: (0.00212, 0.00128),
}
_keys = sorted(ENG)                      # ascending, as np.interp requires
eng_x = np.array(_keys)
eng = np.array([ENG[k][0] for k in _keys])
err = np.array([ENG[k][1] for k in _keys])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.semilogy(h, ss, "C2:", lw=1.2, label="single scattering only")
    ax.semilogy(h, det, "C0.-", lw=1.6, ms=4,
                label="single + plane-parallel multiple")
    ax.errorbar(-eng_x, eng, yerr=err, fmt="C3o-", lw=1.8, ms=5, capsize=2,
                label="spherical twilight engine")
    ax.plot(-pk, pv, "k--", lw=1.0, marker="o", ms=5, mfc="none",
            label="Patat et al. (2006), V band")
    for name, e in [("civil", 6), ("nautical", 12)]:
        ax.axvline(e, color="0.85", ls="--", lw=0.8)
        ax.text(e + 0.2, 2e-6, name, rotation=90, fontsize=7, color="0.5")
    ax.set_xlim(-0.2, 18.5)
    ax.set_ylim(1e-8, 1e3)
    ax.set_xlabel("solar depression below the horizon [deg]")
    ax.set_ylabel("zenith luminance [cd m$^{-2}$]")
    ax.set_title("zenith brightness through twilight")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    dep = -np.array(D["elevations"])          # solar depression, ascending
    det_at = np.interp(-eng_x, dep, det, left=np.nan, right=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((det_at > 0) & (eng > 0), eng / det_at, np.nan)
    ax.semilogy(-eng_x, r, "C3o-", lw=1.6, ms=6)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlim(2.5, 18.5)
    ax.set_ylim(0.5, 1e4)
    ax.set_xlabel("solar depression below the horizon [deg]")
    ax.set_ylabel("engine / deterministic")
    ax.set_title("what the plane-parallel model misses")
    ax.grid(alpha=0.25, which="both")
    ax.annotate("the deterministic model\nunderflows below $-14°$",
                xy=(14.5, 2), xytext=(8.5, 20), fontsize=7, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))

    ax = axes[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (pk >= eng_x.min()) & (pk <= eng_x.max())
        eng_at = np.interp(pk, eng_x, eng, left=np.nan, right=np.nan)
        ratio = np.where(inside & (pv > 0), eng_at / pv, np.nan)
    ax.plot(-pk, ratio, "o-", color="C3", lw=1.8, ms=7)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.axhspan(0.5, 2.0, color="C2", alpha=0.10, label="factor 2")
    ax.axhspan(1 / 1.5, 1.5, color="C2", alpha=0.15, label="factor 1.5")
    for x, y in zip(-pk, ratio):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(4, 6), fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3.0)
    ax.set_xlim(2.5, 13.5)
    ax.set_xlabel("solar depression below the horizon [deg]")
    ax.set_ylabel("engine / Patat et al. (2006)")
    ax.set_title("absolute photometry of the deep twilight")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("The deep twilight: a spherical Monte Carlo for the light "
                 "that travels sideways", fontsize=12)
    save(fig, "11_twilight_deep.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
