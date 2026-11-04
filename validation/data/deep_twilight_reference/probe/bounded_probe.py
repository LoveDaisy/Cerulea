"""Read-only planning probe; all generated artifacts stay beside this script."""
from pathlib import Path
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
START = time.perf_counter()

def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def timeout(signum, frame):
    raise TimeoutError('Owner-authorized 1800-second question/build limit reached')

signal.signal(signal.SIGALRM, timeout)
import numpy as np
from scipy.stats import t
import scipy
import skycolor
from skycolor import presets
from skycolor.rt import mc
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source, single_zenith
assert Path(skycolor.__file__).resolve().parent == ROOT / 'skycolor'
FILES = ['skycolor/rt/mc.py', 'tools/point_solar_table.py',
         'tools/unified_solar_transport.py', 'tools/continuous_solar_columns.py',
         'tools/radial_solar_mapping.py']
config = dict(wavelengths=[450., 550., 650.], suns=[-9., -12.],
    photons_per_wavelength=[1000, 4000, 16000], replicates=8,
    seed_rule='10000 + sun_index*1000 + budget_index*100 + repeat_index',
    n_height=33, derivative_tolerance=.005, disk_radial=2, disk_azimuth=8,
    max_scatter=50, transport='native straight diffuse / refracted finite sun',
    scenario='clear AOD550 .05 O3 300DU ground=observer=0 default albedo .1',
    purpose='Cost and conditional seed/budget variability, NOT photopic integral or tail certification',
    timeout_table_seconds=1800, timeout_all_mc_seconds=1800,
    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
    platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
    hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], text=True).splitlines(),
    thread_env={k: os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
    import_path=skycolor.__file__,
    head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    git_status=subprocess.check_output(['git','status','--porcelain'], text=True),
    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
save('config.json', config)
records = []
try:
    signal.alarm(1800)
    started = time.perf_counter()
    atmo = presets.clear(np.array(config['wavelengths']))
    table = PointSolarTable(atmo, n_height=33, derivative_tolerance=.005,
                           label='deep-twilight-plan-probe-450-550-650-clear-v1', progress=True)
    table.save(OUT / 'point-source.npz')
    signal.alarm(0)
    source_hash = hashlib.sha256((OUT/'point-source.npz').read_bytes()).hexdigest()
    save('build.json', dict(seconds=time.perf_counter()-started,
         table_seconds=table.build_seconds, parameters=table.parameters,
         definition=table.definition, rows=len(table.heights), sha256=source_hash,
         ground_albedo=np.asarray(atmo.ground_albedo).tolist()))
    signal.alarm(1800)
    mc_started = time.perf_counter()
    # Use one table, separate source instances because each caches its sun direction.
    singles = {}
    for sun in config['suns']:
        ts = time.perf_counter()
        value, info = single_zenith(atmo, FiniteSolarSource(atmo, table), 90-sun)
        singles[sun] = value
        save(f'single-{int(-sun)}.json', dict(sun=sun, spectrum=value.tolist(),
             info=info, seconds=time.perf_counter()-ts))
    # Interleave depths within a budget; every batch has an independent seed.
    with (OUT/'raw.jsonl').open('x') as stream:
        for ib, photons in enumerate(config['photons_per_wavelength']):
            for repeat in range(config['replicates']):
                for isun, sun in enumerate(config['suns']):
                    seed = 10000 + isun*1000 + ib*100 + repeat
                    source = FiniteSolarSource(atmo, table)
                    ts = time.perf_counter()
                    with mc_solar_source(source):
                        result = mc.mc_radiance_split(atmo, 90-sun, [0], [0],
                            n_photons=photons, seed=seed, max_scatter=50)
                    row = dict(sun=sun, seed=seed, photons_per_wavelength=photons,
                        seconds=time.perf_counter()-ts,
                        fields={k: v[0].tolist() for k,v in result.items()},
                        split_total=(singles[sun]+result['multiple'][0]).tolist(),
                        source_sha256=source_hash, fingerprint=source.fingerprint.hexdigest(),
                        nee_calls=source.n_calls, nee_points=source.n_points)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    records.append(row)
                    print(json.dumps(row), flush=True)
    signal.alarm(0)
    groups=[]
    for sun in config['suns']:
        for photons in config['photons_per_wavelength']:
            rows=[r for r in records if r['sun']==sun and r['photons_per_wavelength']==photons]
            x=np.array([r['split_total'] for r in rows])
            mean=x.mean(axis=0)
            se=x.std(axis=0,ddof=1)/np.sqrt(len(x))
            half=t.ppf(.975,len(x)-1)*se
            groups.append(dict(sun=sun, photons_per_wavelength=photons, seeds=[r['seed'] for r in rows],
                mean=mean.tolist(), seed_se=se.tolist(), conditional_t95_halfwidth=half.tolist(),
                relative_halfwidth=(half/mean).tolist(), minimum=x.min(axis=0).tolist(),
                maximum=x.max(axis=0).tolist(), seconds=sum(r['seconds'] for r in rows)))
    save('summary.json',dict(groups=groups, mc_and_single_seconds=time.perf_counter()-mc_started,
         total_seconds=time.perf_counter()-START, status='probe_completed_not_reference_certified',
         limits='No path-tail/order diagnostics; 3 wavelengths are not a luminance measurement; Student intervals conditional on sampled tails'))
except BaseException as exc:
    signal.alarm(0)
    save('failure.json',dict(error=repr(exc), traceback=traceback.format_exc(),
         completed_records=len(records), total_seconds=time.perf_counter()-START))
    raise
