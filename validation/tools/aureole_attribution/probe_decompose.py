"""Supplementary stop-loss decomposition: single vs multiple in the ratio.

The pre-registered probe (probe_aureole.py in this directory) ruled out the
moment representation (P1/P2) and the diffuse-transport geometry (P3) as
sources of the check3 h=45deg horizon/zenith excess.  This script decomposes
the B0 ratio into its deterministic components -- exact spherical single
scattering alone vs the SOS multiple-scattering field alone -- so the
residual attribution records WHERE the 8.8 vs CIE 3.65 ratio comes from.
Three deterministic SOS calls on the probe's meridian cut (single-only,
multiple-only, and the total used for the per-view closure check
``single + multiple == total``); no MC (no worker isolation needed).

Run: PYTHONPATH=. uv run python validation/tools/aureole_attribution/probe_decompose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).absolute().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation import common as C  # noqa: E402
from skycolor.rt.solver import (  # noqa: E402
    HYBRID_SOS, SPHERICAL_SINGLE, SOSConfig, solve,
)
from probe_aureole import build_cut, ratio_of, SUN_ZEN  # noqa: E402

OUT = Path(__file__).absolute().parent


def main() -> None:
    atmo = C.atmosphere("clear", C.WL_COARSE)
    cut_z, cut_a, zv, idx = build_cut()

    ss = solve(atmo, SUN_ZEN, cut_z, cut_a, sun_azimuth_deg=0.0,
               approximation=SPHERICAL_SINGLE).radiance
    ms = solve(atmo, SUN_ZEN, cut_z, cut_a, sun_azimuth_deg=0.0,
               approximation=HYBRID_SOS, single=False,
               sos_config=SOSConfig()).radiance
    tot = solve(atmo, SUN_ZEN, cut_z, cut_a, sun_azimuth_deg=0.0,
                approximation=HYBRID_SOS, sos_config=SOSConfig()).radiance

    # per-view closure: single + multiple == total (solver contract)
    clos = float(np.max(np.abs(ss + ms - tot)
                        / np.maximum(np.abs(tot), 1e-300)))
    l_ss = C.luminance(ss, C.WL_COARSE)
    l_ms = C.luminance(ms, C.WL_COARSE)
    l_tot = C.luminance(tot, C.WL_COARSE)
    h, z = idx["horizon"], idx["zenith"]
    out = {
        "closure_max_rel": clos,
        "zenith": {"single": float(l_ss[z]), "multiple": float(l_ms[z]),
                   "total": float(l_tot[z]),
                   "mult_frac_of_total": float(l_ms[z] / l_tot[z])},
        "horizon_sun": {"single": float(l_ss[h]), "multiple": float(l_ms[h]),
                        "total": float(l_tot[h]),
                        "mult_frac_of_total": float(l_ms[h] / l_tot[h])},
        "ratio_single_only": float(l_ss[h] / l_ss[z]),
        "ratio_multiple_only": float(l_ms[h] / l_ms[z]),
        "ratio_total": float(l_tot[h] / l_tot[z]),
        "mult_enhancement": {"zenith": float(l_tot[z] / l_ss[z]),
                             "horizon": float(l_tot[h] / l_ss[h])},
    }
    print(json.dumps(out, indent=1))
    (OUT / "probe_decompose.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
