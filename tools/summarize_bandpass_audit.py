"""Postprocess frozen solar-source spectra through the ESO BESS_V filter.

This is a filter-only experiment, not a reconstruction of FORS1's calibrated
Johnson-Cousins response. No atmosphere, solar angle, or fitted zeropoint is
changed. Inputs remain offline, immutable files. Run from the repository root
with ``uv run python -m tools.summarize_bandpass_audit``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from skycolor import spectra


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_SOURCES = REPO_ROOT / "validation/data/unified_solar_source_audit/raw"
AUDIT_DATA = REPO_ROOT / "validation/data/twilight_reference_audit"
REFERENCES = AUDIT_DATA / "references"
AB_ZERO_JY = 3631.0
C_NM_S = 299792458.0 * 1e9
OMEGA_ARCSEC2 = (np.pi / (180.0 * 3600.0)) ** 2
VEGA_V = 0.031


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    resolved = path.resolve()
    label = str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)
    return dict(path=label, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def fits_header(data: bytes, start: int) -> tuple[dict, int]:
    """Read scalar FITS cards; reject rather than guess unfamiliar structures."""
    header = {}
    for pos in range(start, len(data), 80):
        card = data[pos:pos+80].decode("ascii")
        if card[:8].strip() == "END":
            return header, ((pos + 80 + 2879) // 2880) * 2880
        if card[8:10] != "= ":
            continue
        raw = card[10:].split("/", 1)[0].strip()
        if raw.startswith("'"):
            value = raw.strip("'").strip()
        elif raw in {"T", "F"}:
            value = raw == "T"
        else:
            value = float(raw.replace("D", "E")) if "." in raw or "E" in raw else int(raw)
        header[card[:8].strip()] = value
    raise ValueError("FITS header has no END card")


def read_calspec_vega(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the specific CALSPEC scalar WAVELENGTH/FLUX table without astropy.

    Expected columns are double Angstrom and float FLAM. FLAM -> W/m2/nm
    contributes 1e-7 / 1e-4 * 10 = .01. This intentionally is not a general
    FITS reader; unfamiliar schemas fail explicitly.
    """
    data = path.read_bytes()
    primary, extension_start = fits_header(data, 0)
    if primary["NAXIS"] != 0:
        raise ValueError("Expected an empty CALSPEC primary HDU")
    h, offset = fits_header(data, extension_start)
    expected = dict(XTENSION="BINTABLE", NAXIS=2, PCOUNT=0, GCOUNT=1,
                    TTYPE1="WAVELENGTH", TFORM1="1D", TUNIT1="ANGSTROMS",
                    TTYPE2="FLUX", TFORM2="1E", TUNIT2="FLAM")
    if any(h.get(key) != value for key, value in expected.items()):
        raise ValueError("Unrecognized CALSPEC wavelength/flux schema")
    dtype = np.dtype(dict(names=["wavelength", "flux"], formats=[">f8", ">f4"],
                         offsets=[0, 8], itemsize=h["NAXIS1"]))
    table = np.frombuffer(data, dtype=dtype, count=h["NAXIS2"], offset=offset)
    wl, flux = table["wavelength"].astype(float) / 10, table["flux"].astype(float) * .01
    if not np.all(np.diff(wl) > 0) or not np.all(np.isfinite(flux)) or np.any(flux < 0):
        raise ValueError("Invalid CALSPEC wavelength or flux")
    return wl, flux


class FilterApproximation:
    """Quadrature for spectral radiance (n_wl,) in W m^-2 sr^-1 nm^-1."""

    def __init__(self, wl, filter_data, vega_wl, vega_flux):
        self.wl = np.asarray(wl)
        lo, hi = self.wl[[0, -1]]
        fw, ft = filter_data.T
        if not (fw[0] <= lo < hi <= fw[-1]):
            raise ValueError("Filter data do not cover the model spectrum")
        # Piecewise-linear spectra and transmission: lambda*L*T is cubic.
        # Segment at all three data sets' knots; Gauss3 integrates it exactly.
        knots = np.unique(np.concatenate(([lo, hi], self.wl,
            fw[(fw > lo) & (fw < hi)], vega_wl[(vega_wl > lo) & (vega_wl < hi)])))
        x, w = np.polynomial.legendre.leggauss(3)
        a, b = knots[:-1, None], knots[1:, None]
        self.x = ((a + b) / 2 + (b - a) / 2 * x).ravel()
        self.q = ((b - a) / 2 * w).ravel()
        self.transmission = np.interp(self.x, fw, ft)
        self.weights = {
            "photon": self.q * self.transmission * self.x,
            "energy": self.q * self.transmission,
        }
        # 1 Jy = 1e-26 W/m2/Hz; lambda is in nm here and in d(lambda).
        ab_flux = AB_ZERO_JY * 1e-26 * C_NM_S / self.x**2
        vega = np.interp(self.x, vega_wl, vega_flux)
        if not np.all(vega > 0):
            raise ValueError("Vega spectrum is not positive over the integrated band")
        self.ab_zeros = {key: float(weight @ ab_flux) for key, weight in self.weights.items()}
        self.vega_zeros = {key: float(weight @ vega) for key, weight in self.weights.items()}
        self.y = 683.0 * spectra.cie_xyz(self.wl)[:, 1]
        self.ab_minus_vega_V = {
            key: float(-2.5 * np.log10(self.vega_zeros[key] / self.ab_zeros[key]) - VEGA_V)
            for key in self.weights}

    def evaluate(self, radiance) -> dict:
        rad = np.asarray(radiance, dtype=float)
        if rad.shape != self.wl.shape or not np.all(np.isfinite(rad)) or np.any(rad < 0):
            raise ValueError("Radiance must be finite, nonnegative, and match wavelengths")
        native_y = float(np.trapezoid(rad * self.y, self.wl))
        xyz = np.trapezoid(rad[:, None] * spectra.cie_xyz(self.wl), self.wl, axis=0)
        xy = xyz[:2] / xyz.sum()
        out = dict(photopic_cd_m2=native_y, cie_x=float(xy[0]), cie_y=float(xy[1]))
        sampled = np.interp(self.x, self.wl, rad)
        for convention, weight in self.weights.items():
            signal = float(weight @ sampled)
            ab_ratio = OMEGA_ARCSEC2 * signal / self.ab_zeros[convention]
            v_ratio = OMEGA_ARCSEC2 * signal / self.vega_zeros[convention]
            m_ab = float(-2.5 * np.log10(ab_ratio))
            m_v = float(VEGA_V - 2.5 * np.log10(v_ratio))
            v_equiv = float(1.08e5 * 10**(-.4 * m_v))
            prefix = convention + "_"
            out.update({prefix + "signal": signal, prefix + "AB_mag_arcsec2": m_ab,
                prefix + "V_approx_mag_arcsec2": m_v,
                prefix + "V_equiv_cd_m2": v_equiv,
                prefix + "V_equiv_over_photopic": v_equiv / native_y})
        return out


def mean_sem(values) -> dict:
    data = np.asarray(values, dtype=float)
    return dict(mean=float(np.mean(data)), sem=float(np.std(data, ddof=1) / np.sqrt(len(data)))
                if len(data) > 1 else None)


def paired_ratio(numerator, denominator) -> dict:
    """Ratio of means with seed-paired delta-method sampling error."""
    a, b = np.asarray(numerator), np.asarray(denominator)
    ratio = np.mean(a) / np.mean(b)
    residual = (a - ratio * b) / np.mean(b)
    return dict(ratio_of_means=float(ratio), sem_delta_paired=float(
        np.std(residual, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", default=[ARCHIVED_SOURCES / name for name in (
        "unified-ground0.jsonl", "all-sources-ground2p4.jsonl", "baselines-ground0.jsonl")])
    p.add_argument("--filter", type=Path, default=REFERENCES / "fors-bess-v.txt")
    p.add_argument("--vega", type=Path, default=REFERENCES / "alpha_lyr_stis_011.fits")
    p.add_argument("--out", type=Path, default=AUDIT_DATA / "bandpass-audit.json")
    args = p.parse_args()
    frozen_inputs = json.loads((AUDIT_DATA / "inputs.sha256.json").read_text())
    for path in [*args.inputs, args.filter, args.vega]:
        identity = fingerprint(path)
        expected = frozen_inputs.get(identity["path"])
        if expected is not None and identity["sha256"] != expected:
            raise ValueError(f"Archived audit input changed: {identity['path']}")
    records = [dict(json.loads(line), input_file=fingerprint(path)["path"]) for path in args.inputs
               for line in path.read_text().splitlines() if line.strip()]
    wl = np.asarray(records[0]["wavelengths"])
    if any(not np.array_equal(r["wavelengths"], wl) for r in records):
        raise ValueError("Inputs have different wavelength grids")
    filter_data = np.loadtxt(args.filter)
    vega_wl, vega_flux = read_calspec_vega(args.vega)
    response = FilterApproximation(wl, filter_data, vega_wl, vega_flux)
    per_seed, groups = [], defaultdict(list)
    seen = set()
    closure_error = 0.0
    for record in records:
        identity = (record["sun_elevation"], record["elevation"], record["mode"], record["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate sun/height/mode/seed: {identity}")
        seen.add(identity)
        fields = {key: value for key, value in record["fields"].items() if not key.endswith("_err")}
        fields["deterministic_single"] = record["deterministic_single"]
        metrics = {key: response.evaluate(value) for key, value in fields.items()}
        for key, expected in record["photopic_integrals"].items():
            error = abs(metrics[key]["photopic_cd_m2"] / expected - 1)
            closure_error = max(closure_error, error)
            if error > 1e-12:
                raise ValueError("Photopic postprocessing disagrees with frozen record")
        if not np.allclose(fields["split_total"],
                           np.asarray(fields["multiple"]) + record["deterministic_single"],
                           rtol=1e-13, atol=0):
            raise ValueError("split_total is not deterministic SS + MC multiple")
        row = dict(sun=identity[0], elevation=identity[1], mode=identity[2], seed=identity[3],
                   input_file=record["input_file"], path_fingerprint=record["path_fingerprint"],
                   metrics=metrics)
        per_seed.append(row)
        groups[identity[:3]].append(row)
    summaries = []
    for (sun, height, mode), group in sorted(groups.items()):
        summary = dict(sun=sun, elevation=height, mode=mode, n_seeds=len(group),
                       seeds=sorted(r["seed"] for r in group), fields={})
        for field, metrics in group[0]["metrics"].items():
            summary["fields"][field] = {key: mean_sem([r["metrics"][field][key] for r in group])
                                        for key in metrics}
            summary["fields"][field]["photon_band_correction"] = paired_ratio(
                [r["metrics"][field]["photon_V_equiv_cd_m2"] for r in group],
                [r["metrics"][field]["photopic_cd_m2"] for r in group])
        summaries.append(summary)

    pairs = []
    keys = sorted(groups)
    for a in keys:
        for b in keys:
            label = None
            if a[0] == b[0] and a[1] > b[1] and a[2] == b[2]:
                label = "high_over_low"
            if a[:2] == b[:2] and a[2] == "unified" and b[2] in {"legacy", "geometric"}:
                label = "unified_over_" + b[2]
            if label is None:
                continue
            ga = {r["seed"]: r for r in groups[a]}
            gb = {r["seed"]: r for r in groups[b]}
            seeds = sorted(set(ga) & set(gb))
            if len(seeds) != len(ga) or len(seeds) != len(gb):
                raise ValueError("Pairs do not share all seeds")
            result = dict(comparison=label, numerator=list(a), denominator=list(b), seeds=seeds,
                same_paths=all(ga[s]["path_fingerprint"] == gb[s]["path_fingerprint"] for s in seeds),
                fields={})
            for field in ga[seeds[0]]["metrics"]:
                result["fields"][field] = {}
                for metric in ("photopic_cd_m2", "photon_signal", "energy_signal"):
                    result["fields"][field][metric] = paired_ratio(
                        [ga[s]["metrics"][field][metric] for s in seeds],
                        [gb[s]["metrics"][field][metric] for s in seeds])
                band = result["fields"][field]
                band["photon_ratio_over_photopic_ratio"] = (
                    band["photon_signal"]["ratio_of_means"] /
                    band["photopic_cd_m2"]["ratio_of_means"])
            pairs.append(result)

    # Unit and linearity probes do not use sky/reference fitting.
    ab_radiance = AB_ZERO_JY * 1e-26 * C_NM_S / wl**2 / OMEGA_ARCSEC2
    flat_ab = response.evaluate(ab_radiance)
    first = np.asarray(records[0]["fields"]["split_total"])
    one, twice = response.evaluate(first), response.evaluate(2*first)
    linearity_error = abs(twice["photon_AB_mag_arcsec2"] - one["photon_AB_mag_arcsec2"]
                          + 2.5*np.log10(2))
    # A second integration in frequency uses f_nu/(h*nu), rather than
    # reusing the wavelength weights or the CALSPEC-derived V zeropoint.
    nu = np.geomspace(C_NM_S/wl[-1], C_NM_S/wl[0], 100001)
    lam = C_NM_S / nu
    transmission_nu = np.interp(lam, filter_data[:, 0], filter_data[:, 1])
    observed_fnu = OMEGA_ARCSEC2 * np.interp(lam, wl, first) * lam**2 / C_NM_S
    frequency_ratio = (np.trapezoid(observed_fnu * transmission_nu / nu, nu) /
                       (AB_ZERO_JY * 1e-26 * np.trapezoid(transmission_nu / nu, nu)))
    frequency_mag = -2.5 * np.log10(frequency_ratio)
    frequency_error = abs(frequency_mag - one["photon_AB_mag_arcsec2"])
    if (abs(flat_ab["photon_AB_mag_arcsec2"]) > 1e-4 or linearity_error > 1e-12
            or frequency_error > 1e-7):
        raise ValueError("AB-zero or amplitude-scaling probe failed")
    result = dict(
        label="ESO BESS_V filter-only approximation; not calibrated FORS1 Johnson-Cousins V",
        conventions=dict(model_radiance="W m^-2 sr^-1 nm^-1", wavelength="nm",
            integration_range_nm=[float(wl[0]), float(wl[-1])],
            transmission="ESO filter only; QE=1 within model range; zero outside",
            photon="integral(lambda * L_lambda * T d_lambda); hc cancels in ratios",
            energy="integral(L_lambda * T d_lambda); diagnostic calorimetric convention",
            AB_zero_Jy=AB_ZERO_JY, steradian_per_arcsec2=OMEGA_ARCSEC2,
            Vega_V_adopted=VEGA_V, Vega_FLAM_to_W_m2_nm=.01,
            V_equivalent="108000 * 10**(-0.4 * V_approx); not photopic luminance",
            uncertainty="SEM across seeds; pair SEM uses ratio-of-means delta method; excludes model/response error"),
        provenance=dict(inputs=[fingerprint(f) for f in args.inputs],
            filter=fingerprint(args.filter), vega=fingerprint(args.vega),
            script=fingerprint(Path(__file__)),
            record_heads=sorted(set(r["head"] for r in records)),
            library_diff_sha256=sorted(set(r["library_diff_sha256"] for r in records))),
        response=dict(AB_minus_V_approx=response.ab_minus_vega_V,
            photon_AB_denominator=response.ab_zeros["photon"],
            photon_Vega_denominator=response.vega_zeros["photon"],
            n_quadrature_nodes=len(response.x)),
        checks=dict(max_relative_photopic_closure_error=closure_error,
            interpolated_flat_fnu_AB_magnitude=flat_ab["photon_AB_mag_arcsec2"],
            doubling_magnitude_error=linearity_error,
            frequency_domain_AB_magnitude=frequency_mag,
            wavelength_vs_frequency_magnitude_difference=frequency_error),
        summaries=summaries, pairs=pairs, per_seed=per_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with args.out.with_suffix(".csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["sun", "elevation", "mode", "field", "n_seeds", "photopic", "photopic_sem",
            "mean_seed_photon_AB_mag", "mean_seed_photon_V_approx_mag", "photon_V_equiv",
            "V_equiv_over_photopic_ratio_of_means", "band_correction_sem"])
        for group in summaries:
            for field, m in group["fields"].items():
                writer.writerow([group["sun"], group["elevation"], group["mode"], field, group["n_seeds"],
                    m["photopic_cd_m2"]["mean"], m["photopic_cd_m2"]["sem"],
                    m["photon_AB_mag_arcsec2"]["mean"], m["photon_V_approx_mag_arcsec2"]["mean"],
                    m["photon_V_equiv_cd_m2"]["mean"],
                    m["photon_band_correction"]["ratio_of_means"],
                    m["photon_band_correction"]["sem_delta_paired"]])
    print(json.dumps(dict(out=str(args.out), n_records=len(records), checks=result["checks"],
                          response=result["response"]), indent=2))
    for group in summaries:
        values = group["fields"]["split_total"]
        print(group["elevation"], group["mode"], "Y", values["photopic_cd_m2"],
              "Veq", values["photon_V_equiv_cd_m2"],
              "C", values["photon_band_correction"])
    for pair in pairs:
        print(pair["comparison"], pair["numerator"], pair["denominator"],
              pair["fields"]["split_total"])


if __name__ == "__main__":
    main()
