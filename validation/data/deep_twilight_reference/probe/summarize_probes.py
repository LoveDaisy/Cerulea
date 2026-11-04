"""Recompute probe summaries from raw records, not prior summary assertions."""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.stats import t
P=Path(__file__).resolve().parent
raw=[json.loads(s) for s in (P/'growth-raw.jsonl').read_text().splitlines()]
diag=[json.loads(s) for s in (P/'diagnostic-raw.jsonl').read_text().splitlines()]
assert len(raw)==96 and len(diag)==16
out={'raw_counts':{'growth':len(raw),'diagnostics':len(diag)},'groups':[]}
for sun in (-9.,-12.):
    rows=[r for r in diag if r['sun']==sun]
    orders=np.sum([r['vertex_order_sums'] for r in rows],axis=0)
    ground=np.sum([r['ground_order_sums'] for r in rows],axis=0)
    multiple=orders.sum(axis=(1,2))+ground.sum(axis=1)-orders[:,1,0]
    x=np.array([r['fields']['multiple'] for r in rows])
    se=x.std(axis=0,ddof=1)/np.sqrt(8)
    for iw,wl in enumerate((450,550,650)):
        paths=[r['paths'][iw] for r in rows]
        denom=multiple[iw]
        order_fraction=(orders[iw].sum(axis=1)/denom)
        order_fraction[1]-=orders[iw,1,0]/denom
        row=dict(sun=sun,wl=wl,multiple_mean=float(x[:,iw].mean()),
            half95=float(t.ppf(.975,7)*se[iw]),
            relative_half95=float(t.ppf(.975,7)*se[iw]/x[:,iw].mean()),
            effective_contributors_range=[min(p['effective_contributors'] for p in paths),max(p['effective_contributors'] for p in paths)],
            max_single_path_fraction_range=[min(p['max_fraction'] for p in paths),max(p['max_fraction'] for p in paths)],
            top_0_1_percent_fraction_range=[min(p['top_0_1_percent_fraction'] for p in paths),max(p['top_0_1_percent_fraction'] for p in paths)],
            nonzero_paths=sum(p['nonzero'] for p in paths),
            order_fractions=order_fraction[:11].tolist(),
            order_gt10_fraction=float(order_fraction[11:].sum()),
            touched_ground_fraction=float((orders[iw,:,1].sum()+ground[iw].sum())/denom),
            cap_paths=sum(p['capped_paths'] for p in paths))
        out['groups'].append(row)
    for n in (100000,400000,1600000):
        d=[r for r in raw if r['sun']==sun and r['n']==n and r['mode']=='disk']
        p=[r for r in raw if r['sun']==sun and r['n']==n and r['mode']=='point']
        assert len(d)==len(p)==8
        assert all(a['seed']==b['seed'] and a['fingerprint']==b['fingerprint'] for a,b in zip(d,p))
        for r in d+p:
            np.testing.assert_allclose(np.array(r['fields']['single'])+r['fields']['multiple'],r['fields']['total'],rtol=2e-13,atol=1e-30)
            np.testing.assert_array_equal(np.array(r['deterministic_single'])+r['fields']['multiple'],r['split_total'])
out['diagnostic_seconds']=sum(r['seconds'] for r in diag)
out['sha256']={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['raw.jsonl','growth-raw.jsonl','diagnostic-raw.jsonl','point-source.npz']}
with (P/'verified-probe-summary.json').open('x') as f: json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out,indent=2))
