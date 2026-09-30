"""Only predeclared development seeds; never used for learned model comparison."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np
from .environment import RecoveryEnv
from .utils import atomic,config

def one(job):
    c,seed,mode=job;e=RecoveryEnv(c,seed,disturbance_scale=0 if mode=='undisturbed' else 1).prepare()
    while e.tick<c['end_ms']:
        a=e.teacher() if mode=='teacher' else float(mode=='constant_max');e.step(a)
    return dict(seed=seed,mode=mode,**e.metrics())

def run_development(c,path,n=12):
    with ProcessPoolExecutor(c['workers']) as pool:rows=list(pool.map(one,[(c,c['seed_starts']['development']+i,m) for m in ['undisturbed','nominal','teacher','constant_max'] for i in range(n)]))
    atomic(path,rows)
    for m in ['undisturbed','nominal','teacher','constant_max']:
        r=[x for x in rows if x['mode']==m]
        print(m,'success',np.mean([x['success'] for x in r]),'stable',np.mean([x['initial_stable'] for x in r]),'force',np.mean([x['peak_contact_force_n'] for x in r]),flush=True)

if __name__=='__main__':
    import yaml
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run_development(yaml.safe_load(a.config.read_text()),a.output)
