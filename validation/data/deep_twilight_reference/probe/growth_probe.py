"""Bounded budget growth and paired disk correction; reuse the one saved table."""
from pathlib import Path
import hashlib
import json
import signal
import sys
import time
import traceback
import numpy as np
from scipy.stats import t
ROOT=Path.cwd().resolve()
sys.path.insert(0,str(ROOT))
from skycolor import presets
from skycolor.rt import mc
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source, single_zenith
OUT=Path(__file__).resolve().parent
START=time.perf_counter()

def save(name,obj):
    with (OUT/name).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)

def expired(signum,frame):
    raise TimeoutError('1800-second combined growth/seed/disk question limit reached')

config=dict(wavelengths=[450.,550.,650.],suns=[-9.,-12.],budgets=[100000,400000,1600000],
    repeats=8,seed_rule='20000 + sun_index*1000 + budget_index*100 + replicate',
    modes=['disk','point'],max_scatter=50,disk_nodes=[2,8],
    table_sha256=hashlib.sha256((OUT/'point-source.npz').read_bytes()).hexdigest(),
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    inherited_hardware_and_source_config='config.json',timeout_seconds=1800,
    precision_candidate='finite-disk minus point-disk source effect on identical native paths')
save('growth-config.json',config)
records=[]
signal.signal(signal.SIGALRM,expired)
signal.alarm(1800)
try:
    atmo=presets.clear(np.array(config['wavelengths']))
    table=PointSolarTable.load(OUT/'point-source.npz',atmo,
        expected_label='deep-twilight-plan-probe-450-550-650-clear-v1')
    singles={}
    for sun in config['suns']:
        for mode in config['modes']:
            source=FiniteSolarSource(atmo,table,radius_deg=16/60 if mode=='disk' else 0.)
            singles[sun,mode],_=single_zenith(atmo,source,90-sun)
    with (OUT/'growth-raw.jsonl').open('x') as f:
        for ib,n in enumerate(config['budgets']):
            for repeat in range(config['repeats']):
                for isu,sun in enumerate(config['suns']):
                    fp=None
                    for mode in config['modes']:
                        seed=20000+isu*1000+ib*100+repeat
                        source=FiniteSolarSource(atmo,table,radius_deg=16/60 if mode=='disk' else 0.)
                        ts=time.perf_counter()
                        with mc_solar_source(source):
                            result=mc.mc_radiance_split(atmo,90-sun,[0],[0],n_photons=n,seed=seed)
                        fingerprint=source.fingerprint.hexdigest()
                        if fp is not None:
                            assert fingerprint==fp,'Source variants changed native paths'
                        fp=fingerprint
                        row=dict(sun=sun,mode=mode,seed=seed,n=n,seconds=time.perf_counter()-ts,
                            fields={k:v[0].tolist() for k,v in result.items()},
                            deterministic_single=singles[sun,mode].tolist(),
                            split_total=(singles[sun,mode]+result['multiple'][0]).tolist(),
                            fingerprint=fp,nee_points=source.n_points)
                        f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
                        records.append(row)
                        print(json.dumps(dict(sun=sun,mode=mode,seed=seed,n=n,seconds=row['seconds'])),flush=True)
    signal.alarm(0)
    groups=[]
    for sun in config['suns']:
        for n in config['budgets']:
            disk=[r for r in records if r['sun']==sun and r['n']==n and r['mode']=='disk']
            point=[r for r in records if r['sun']==sun and r['n']==n and r['mode']=='point']
            assert [r['seed'] for r in disk]==[r['seed'] for r in point]
            x=np.array([r['split_total'] for r in disk])
            delta=x-np.array([r['split_total'] for r in point])
            half=t.ppf(.975,7)*x.std(axis=0,ddof=1)/np.sqrt(8)
            dh=t.ppf(.975,7)*delta.std(axis=0,ddof=1)/np.sqrt(8)
            groups.append(dict(sun=sun,n=n,mean=x.mean(axis=0).tolist(),half95=half.tolist(),
                relative_half95=(half/x.mean(axis=0)).tolist(),delta_mean=delta.mean(axis=0).tolist(),
                delta_half95=dh.tolist(),minimum=x.min(axis=0).tolist(),maximum=x.max(axis=0).tolist(),
                disk_seconds=sum(r['seconds'] for r in disk),point_seconds=sum(r['seconds'] for r in point)))
    save('growth-summary.json',dict(groups=groups,seconds=time.perf_counter()-START,
         status='completed; conditional intervals, not proof of sampled tail completeness'))
except BaseException as exc:
    signal.alarm(0)
    save('growth-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
        completed_records=len(records),seconds=time.perf_counter()-START))
    raise
