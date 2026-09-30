"""Shared behavior data, privileged teacher actions, episode-disjoint seeds."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,time
import numpy as np
from .environment import RecoveryEnv
from .utils import atomic,config,progress

def collect_episode(job):
    c,seed,path=job;path=Path(path)
    if path.exists():return str(path)
    e=RecoveryEnv(c,seed).prepare();rng=np.random.default_rng([seed,113])
    x=list(e.window());y=[];executed=[]
    while e.tick<c['end_ms']:
        target=e.teacher();y.append(target)
        if e.tick%40==0:explore=float(rng.uniform(0,.7)) if rng.random()<.25 else None
        action=explore if explore is not None else float(np.clip(target+rng.normal(0,.025),0,1))
        executed.append(action);e.step(action);x.append(e.history[-1])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,x=np.asarray(x,dtype=np.float32),y=np.asarray(y,dtype=np.float32),executed=executed,log=e.log)
    path.with_suffix('.tmp').replace(path)
    atomic(path.with_suffix('.json'),dict(**e.meta,**e.metrics()))
    return str(path)

def collect(run,c):
    for split,n in [('train',c['train_episodes']),('validation',c['validation_episodes'])]:
        jobs=[(c,c['seed_starts'][split]+i,run/'data'/split/f'{i:04d}.npz') for i in range(n)]
        with ProcessPoolExecutor(c['workers']) as pool:
            for i,_ in enumerate(pool.map(collect_episode,jobs)):
                if (i+1)%12==0:progress(run,'collect',split=split,episodes=i+1)
    # Statistics use only policy-visible TRAIN observations. No labels/state filter.
    all_x=np.concatenate([np.load(p)['x'] for p in sorted((run/'data/train').glob('*.npz'))])
    mean=all_x.mean(0);scale=all_x.std(0)
    floor=np.r_[np.full(32,5e-6),np.full(7,1e-4),np.full(2,1e-4),.05]
    np.savez(run/'normalizer.npz',mean=mean,scale=np.maximum(scale,floor))

def load_windows(run,c,split):
    h=c['history_ms']//c['sample_ms']+1;xs=[];ys=[]
    for p in sorted((run/'data'/split).glob('*.npz')):
        z=np.load(p);x=z['x'];y=z['y']
        xs.append(np.stack([x[i:i+h] for i in range(len(y))]));ys.append(y)
    return np.concatenate(xs),np.concatenate(ys)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();collect(a.run,config(a.run))
