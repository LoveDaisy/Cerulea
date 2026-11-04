"""Observe existing return frames; do not rewrite transport, source, or RNG calls."""
from pathlib import Path
import sys
import json
import time
import signal
import traceback
import hashlib
import numpy as np
ROOT=Path.cwd().resolve(); sys.path.insert(0,str(ROOT))
from skycolor import presets
from skycolor.rt import mc
from tools.point_solar_table import PointSolarTable
from tools.unified_solar_transport import FiniteSolarSource, mc_solar_source
OUT=Path(__file__).resolve().parent
START=time.perf_counter()

def save(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def timeout(signum,frame): raise TimeoutError('Diagnostic question limit 1800 seconds')
signal.signal(signal.SIGALRM,timeout); signal.alarm(1800)
config=dict(n=1600000,seeds=list(range(30000,30008)),suns=[-9.,-12.],
    wavelengths=[450.,550.,650.], timeout_seconds=1800,
    method='sys.setprofile observes source/tracer returns; no code replacement beyond existing mc_solar_source',
    limitations='Vertex order sums reconstructed from unchanged source return and caller weights; final path order is NOT contribution order',
    table_sha256=hashlib.sha256((OUT/'point-source.npz').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
save('diagnostic-config.json',config)
try:
    atmo=presets.clear(np.array(config['wavelengths']))
    table=PointSolarTable.load(OUT/'point-source.npz',atmo,expected_label='deep-twilight-plan-probe-450-550-650-clear-v1')
    def run(sun,seed,n,observe):
        paths=[]; orders=np.zeros((3,51,2)); ground=np.zeros((3,51))
        def observer(frame,event,arg):
            if event!='return': return
            name=frame.f_code.co_name
            caller=frame.f_back
            if name in ('scatter','ground') and caller and caller.f_code.co_name=='_trace_wavelength':
                loc=caller.f_locals; iw=loc['iw']
                if name=='scatter':
                    ci=loc['ci']; value=loc['wa'][ci]*loc['albedo']*arg*loc['f0']
                    order=loc['sc_new'][ci]+1; bounced=loc['gd_new'][ci]>0
                    for g in (0,1):
                        mask=bounced==g
                        orders[iw,:,g]+=np.bincount(order[mask],weights=value[mask],minlength=51)[:51]
                else:
                    gi=loc['gi']; value=loc['wa'][gi]*(loc['rho']/np.pi)*loc['solar']*arg
                    ground[iw]+=np.bincount(loc['sc_new'][gi],weights=value,minlength=51)[:51]
            elif name=='_trace_wavelength' and 'contrib' in frame.f_locals:
                loc=frame.f_locals; c=loc['c_mult']; total=float(c.sum()); npath=len(c)
                k=max(1,int(np.ceil(npath*.001))); top=np.partition(c,npath-k)[-k:]
                denom=float(np.dot(c,c))
                paths.append(dict(iw=loc['iw'],n=npath,nonzero=int(np.count_nonzero(c)),
                    sum=total,sum_sq=denom,max=float(c.max()),
                    max_fraction=float(c.max()/total) if total else None,
                    top_0_1_percent_fraction=float(top.sum()/total) if total else None,
                    effective_contributors=total*total/denom if denom else 0.,
                    capped_paths=int(np.count_nonzero(loc['n_scat']>=50)),
                    final_weight_max=float(loc['w'].max())))
        source=FiniteSolarSource(atmo,table); ts=time.perf_counter()
        with mc_solar_source(source):
            if observe: sys.setprofile(observer)
            try: result=mc.mc_radiance_split(atmo,90-sun,[0],[0],n_photons=n,seed=seed)
            finally: sys.setprofile(None)
        if observe:
            reconstructed=(orders.sum(axis=(1,2))+ground.sum(axis=1))/n
            np.testing.assert_allclose(reconstructed,result['total'][0],rtol=2e-13,atol=1e-30)
        return dict(sun=sun,seed=seed,n=n,seconds=time.perf_counter()-ts,
            fields={k:v[0].tolist() for k,v in result.items()},fingerprint=source.fingerprint.hexdigest(),
            paths=paths,vertex_order_sums=orders.tolist(),ground_order_sums=ground.tolist())
    checks=[]
    for sun in config['suns']:
        off=run(sun,29999,100000,False); on=run(sun,29999,100000,True)
        assert off['fields']==on['fields'] and off['fingerprint']==on['fingerprint']
        checks.append(dict(sun=sun,bitwise_equal=True,off=off,on=on))
    save('diagnostic-controls.json',checks)
    with (OUT/'diagnostic-raw.jsonl').open('x') as f:
        for sun in config['suns']:
            for seed in config['seeds']:
                row=run(sun,seed,config['n'],True)
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
                print(json.dumps(dict(sun=sun,seed=seed,seconds=row['seconds'],paths=row['paths'])),flush=True)
    save('diagnostic-completion.json',dict(seconds=time.perf_counter()-START,status='completed'))
    signal.alarm(0)
except BaseException as exc:
    sys.setprofile(None); signal.alarm(0)
    save('diagnostic-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-START))
    raise
