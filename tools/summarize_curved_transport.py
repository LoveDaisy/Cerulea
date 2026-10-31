"""Summarize independent-seed curved-flight controls without fitting data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from skycolor import spectra


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def moments(values):
    values = np.asarray(values)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, seed_standard_error=se,
        student_95_interval=[mean-t.ppf(.975, n-1)*se, mean+t.ppf(.975, n-1)*se]
            if se is not None else None,
        values=values.tolist(),
        first_half_mean=float(values[:max(n//2, 1)].mean()),
        second_half_mean=float(values[n//2:].mean()))


def ratio(high, low):
    high, low = np.asarray(high), np.asarray(low)
    r = high.mean()/low.mean()
    return dict(ratio=float(r), paired_delta_standard_error=float(
        np.std(high-r*low, ddof=1)/np.sqrt(len(high))/low.mean()))


def values(rows, field):
    return np.array([r['photopic_integrals'][field] for r in rows])


def summarize(rows):
    out = {field: moments(values(rows, field))
        for field in ['single', 'multiple', 'split_total', 'total']}
    wl = np.array(rows[0]['wavelengths'])
    weights = np.empty_like(wl)
    weights[0], weights[-1] = (wl[1]-wl[0])/2, (wl[-1]-wl[-2])/2
    weights[1:-1] = (wl[2:]-wl[:-2])/2
    weights *= 683*spectra.cie_xyz(wl)[:, 1]
    for field in ['single', 'multiple', 'total']:
        variances = [np.sum((weights*np.array(r['fields'][field+'_err']))**2) for r in rows]
        out[field]['within_path_predicted_se_of_seed_mean'] = float(
            np.sqrt(np.sum(variances))/len(rows))
    ss = float(np.trapezoid(np.array(rows[0]['deterministic_single'])*
                           spectra.cie_xyz(wl)[:, 1], wl)*683)
    out['deterministic_single'] = ss
    out['single']['closure_seed_sigma'] = ((out['single']['mean']-ss)/
        out['single']['seed_standard_error']) if len(rows)>1 else None
    out['seconds'] = float(sum(r['seconds'] for r in rows))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--straight-file', default='global-straight.jsonl')
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    manifest_path = args.directory.parent/'inputs.sha256.json'
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text()).items():
            actual = hashlib.sha256((args.directory.parent/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Frozen curved-flight input changed: {name}')
    names = [args.straight_file, 'curved-ground0.jsonl', 'curved-ground2p4.jsonl']
    rows = sum((read(args.directory/name) for name in names), [])
    wavelengths = np.arange(380., 781., 5.)
    cie_y = spectra.cie_xyz(wavelengths)[:, 1]
    table_hashes = {}
    for row in rows:
        np.testing.assert_array_equal(row['wavelengths'], wavelengths)
        config = row['config']
        assert row['mode'] == 'unified' and row['sun_elevation'] == -6
        assert config['disk_radial'] == 2 and config['disk_azimuth'] == 8
        assert config['disk_radius'] == 16/60
        assert config['ss_gauss'] == 8 and config['ss_segment'] == .5
        assert row['source_hashes'] == rows[0]['source_hashes']
        height = float(row['elevation'])
        table_hashes.setdefault(height, row['point_table_sha256'])
        assert table_hashes[height] == row['point_table_sha256']
        for key, stored in row['photopic_integrals'].items():
            computed = 683*np.trapezoid(np.array(row['fields'][key])*cie_y, wavelengths)
            np.testing.assert_allclose(computed, stored, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(row['fields']['split_total'],
            np.array(row['deterministic_single'])+np.array(row['fields']['multiple']),
            rtol=2e-14, atol=1e-20)
    grouped = {}
    for r in rows:
        key = (r['config']['transport'], float(r['elevation']))
        grouped.setdefault(key, []).append(r)
    assert len({(r['config']['transport'], r['elevation'], r['seed']) for r in rows}) == len(rows)
    seeds = None
    for key, group in grouped.items():
        group.sort(key=lambda r: r['seed'])
        current = [r['seed'] for r in group]
        if seeds is None:
            seeds = current
        assert current == seeds, (key, current, seeds)
    out = dict(seeds=seeds, frozen_manifest_checked=manifest_path.exists(), input_sha256={name:hashlib.sha256(
        (args.directory/name).read_bytes()).hexdigest() for name in names}, assumptions=[
        'Standard errors describe these independent seed batches, not unknown tail mass.',
        'Student intervals are conditional diagnostics, not guarantees for heavy-tailed transport.',
        'Identical seeds retain covariance; curved paths are not identical to straight paths.',
        'No observational zero point, solar axis or atmosphere was fitted.'], groups={})
    for (transport, height), group in grouped.items():
        out['groups'][f'{transport}/{height}'] = summarize(group)
    for transport in ['continuous', 'global-straight']:
        out[transport+'_height_ratio'] = ratio(
            values(grouped[(transport, 2.4)], 'split_total'),
            values(grouped[(transport, 0.)], 'split_total'))
    for height in [0., 2.4]:
        bent, straight = grouped[('continuous', height)], grouped[('global-straight', height)]
        a, b = values(bent, 'split_total'), values(straight, 'split_total')
        out[f'curvature/{height}'] = dict(
            difference=moments(a-b), ratio=ratio(a, b))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    if args.figure:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        colors = ['#304d60', '#b65038']
        for index, height in enumerate([0., 2.4]):
            for j, transport in enumerate(['global-straight', 'continuous']):
                stats = out['groups'][f'{transport}/{height}']['split_total']
                axes[0].errorbar(index+(j-.5)*.14, stats['mean'],
                    yerr=stats['seed_standard_error'], color=colors[j],
                    marker=['o', 's'][j], capsize=4,
                    label=['Straight, 100k', 'Curved, 10k'][j] if index == 0 else None)
            change = out[f'curvature/{height}']['ratio']
            axes[1].errorbar(index, 100*(change['ratio']-1),
                yerr=100*change['paired_delta_standard_error'],
                color=colors[1], marker='o', capsize=4)
        for ax in axes:
            ax.set_xticks([0, 1], ['0 km', '2.4 km'])
            ax.set_xlabel('Ground = observer altitude')
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('Zenith SS + MC higher orders [cd m$^{-2}$]')
        axes[0].legend(frameon=False, title='Paths / wavelength / seed')
        axes[1].axhline(0, color='.5', linestyle='--', linewidth=1)
        axes[1].set_ylabel('Curved / straight minus one [%]')
        fig.suptitle('Unified finite Sun at -6 deg; 81 wavelengths; 8 seeds; bars = seed SEM')
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=170)
        plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k!='groups'}, indent=2))


if __name__ == '__main__':
    main()
