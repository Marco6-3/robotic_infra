"""Shared imitation data; exact command timing and no hidden-state features."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json
import numpy as np
from .environment import InsertionEnv
from .utils import atomic,config,progress


def one(job):
    c,seed,path=job;path=Path(path)
    if path.exists():return
    e=InsertionEnv(c,seed).prepare();rng=np.random.default_rng([seed,491])
    x=list(e.window());y=[];commands=[]
    while not e.done():
        target=e.teacher();y.append(target)
        # Same data for all models; occasional recovery exploration, independent of hidden geometry.
        action=np.clip(target+rng.normal(0,.1,2),-1,1)
        if rng.random()<.10:action=rng.uniform(-1,1,2)
        e.step(action);commands.append(e.last_action.copy());x.append(e.history[-1])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,x=np.asarray(x,dtype=np.float32),y=np.asarray(y,dtype=np.float32).reshape(-1,2),issued=np.asarray(commands).reshape(-1,2),log=e.log)
    path.with_suffix('.tmp').replace(path)
    atomic(path.with_suffix('.json'),dict(seed=seed,hole=e.hole,friction=e.friction,mass=e.mass,kp=e.kp,probes=e.probes,targets=len(y),**e.metrics()))


def collect(run,c):
    for split,n in [('train',c['train_episodes']),('validation',c['validation_episodes'])]:
        with ProcessPoolExecutor(c['workers']) as pool:
            for i,_ in enumerate(pool.map(one,[(c,c['seed_starts'][split]+i,run/'data'/split/f'{i:04}.npz') for i in range(n)])):
                if (i+1)%16==0:progress(run,'collect',split=split,episodes=i+1)
    x=np.concatenate([np.load(p)['x'] for p in (run/'data/train').glob('*.npz')])
    floor=np.r_[np.full(32,c['tactile_quantum_m']),np.full(3,c['q_quantum_m']),np.full(2,.05)]
    np.savez(run/'normalizer.npz',mean=x.mean(0),scale=np.maximum(x.std(0),floor))


def load_windows(run,c,split):
    xs=[];ys=[];h=c['history_ms']//c['sample_ms']+1
    for p in sorted((run/'data'/split).glob('*.npz')):
        z=np.load(p);x=z['x'];y=z['y']
        if len(y):xs.append(np.stack([x[i:i+h] for i in range(len(y))]));ys.append(y)
    if not xs:raise RuntimeError('No preterminal training targets; task gate failed')
    return np.concatenate(xs),np.concatenate(ys)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();collect(a.run,config(a.run))
