"""Independent Hamiltonian DOP853 flight and aerosol-sampling diagnostics.

No Monte Carlo is run. The current report is written to tracked audit data by
default; --out can select a separate artifact without changing historical runs.
"""
from pathlib import Path
import argparse
import sys
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "validation/data/curved_transport_audit/final-review/hamiltonian-probes.json"

from skycolor import presets
from skycolor.constants import R_EARTH
from skycolor.rt import mc
from tools.continuous_curved_mc import ContinuousCurvedFlight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ground in [0., 2.4]:
        atmo = presets.clear([450., 550., 650.])
        if ground:
            atmo = atmo.at_altitude(ground)
        mapping = ContinuousCurvedFlight(atmo).mapping
        # p=n*d uses a Hamiltonian momentum variable rather than the flight
        # implementation's projected unit-direction acceleration.
        def rhs(_, state):
            position, momentum = state[:3], state[3:]
            radius = np.linalg.norm(position)
            index = float(mapping._n(radius))
            gradient = float(mapping._dn(radius)) * position/radius
            return np.r_[momentum/index, gradient]
        def hit_ground(_, state):
            return np.linalg.norm(state[:3])-atmo.r_bounds[0]
        def hit_top(_, state):
            return np.linalg.norm(state[:3])-atmo.r_bounds[-1]
        hit_ground.direction, hit_ground.terminal = -1, True
        hit_top.direction, hit_top.terminal = 1, True
        for height, elevation, distance in [
            (ground+.00001, -.005, 2000.),
            (ground+.00005, -.005, 2000.),
            (ground, .2, 2e6),
            (38., -5.487342769121213, 2e6),
            (100., -9.48538509285, 2e6),
            (47., -20., 2e6),
            (ground+2., 89., 25000.),
            (ground+3., -2., 50000.),
        ]:
            xx=np.array([0.,0.,R_EARTH+height*1000])
            e=np.radians(elevation);dd=np.array([np.cos(e),0.,np.sin(e)])
            n0=float(mapping._n(np.linalg.norm(xx)))
            reference=solve_ivp(rhs,(0,distance),np.r_[xx,n0*dd],method='DOP853',
                rtol=2e-12,atol=np.r_[[1e-6]*3,[2e-13]*3],max_step=5000.,
                events=[hit_ground,hit_top])
            if not reference.success:
                raise RuntimeError(reference.message)
            xr,pr=reference.y[:3,-1],reference.y[3:,-1]
            dr=pr/np.linalg.norm(pr)
            kind='ground' if len(reference.t_events[0]) else 'top' if len(reference.t_events[1]) else 'trial'
            for step in [2000.,1000.]:
                flight=ContinuousCurvedFlight(atmo,max_step_m=step)
                x,d,trial,hit,escaped=flight.advance(xx[None],dd[None],[distance])
                actual='ground' if hit[0] else 'top' if escaped[0] else 'trial'
                records.append(dict(ground_km=ground,height_km=height,elevation_deg=elevation,
                    sampled_distance_m=distance,step_m=step,reference_event=kind,event=actual,
                    travel_error_m=float(flight.last_distance[0]-reference.t[-1]),
                    position_error_m=float(np.linalg.norm(x[0]-xr)),
                    direction_error_rad=float(np.linalg.norm(d[0]-dr)),
                    hamiltonian_norm_error=float(abs(np.linalg.norm(pr)/mapping._n(np.linalg.norm(xr))-1)),
                    reference_nfev=reference.nfev))
    aerosol=presets.clear([450.,550.,650.]).aerosols[0].aerosol
    sampler=[]
    for wavelength in [450.,550.,650.]:
        truth=aerosol.legendre_moments([wavelength],n_moments=5)[:,0]
        for count in [2049,8193,32769]:
            mu=np.linspace(-1.,1.,count)
            phase=np.asarray(aerosol.phase(mu,[wavelength])).ravel()
            mass=.5*(phase[1:]+phase[:-1])*np.diff(mu)
            total=mass.sum()
            # Linear inverse-CDF interpolation actually samples piecewise
            # constant pdf in mu, with these exact cell probabilities.
            gx,gw=np.polynomial.legendre.leggauss(3)
            nodes=(mu[1:,None]+mu[:-1,None])/2+np.diff(mu)[:,None]/2*gx
            sampled=[float(np.sum(mass/total * (eval_legendre(k,nodes)@gw)/2)) for k in range(5)]
            sampler.append(dict(wavelength_nm=wavelength,N_PHASE=count,
                half_integral_phase=float(total/2),sampled_moments=sampled,
                analytic_moments=truth.tolist(),moment_errors=(np.array(sampled)-truth).tolist()))
    root=ROOT
    result=dict(checker=dict(path="tools/check_curved_hamiltonian.py",
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),scope='No Monte Carlo; independent Hamiltonian DOP853 flights and exact inverse-CDF cell moments.',
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['tools/continuous_curved_mc.py','tools/unified_solar_transport.py','tools/point_solar_table.py','skycolor/rt/mc.py']},
        flights=records,aerosol_sampling=sampler)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(rows=len(records),event_mismatches=sum(r['event']!=r['reference_event'] for r in records),
        max_position_m=max(r['position_error_m'] for r in records),
        max_travel_m=max(abs(r['travel_error_m']) for r in records),
        max_direction_rad=max(r['direction_error_rad'] for r in records),
        sampler2049=[r for r in sampler if r['N_PHASE']==2049]),indent=2))


if __name__=='__main__':
    main()
