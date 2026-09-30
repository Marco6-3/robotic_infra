"""All conditions act on live MuJoCo; state, force and target never enter student."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from .environment import InsertionEnv
from .policy import Policy,normalize
from .utils import atomic,config,progress,setup

CONDITIONS=['M0','M1','M2','M1_temporal_shuffle','M2_temporal_shuffle','M2_action_shuffle']


def corrupt(x,condition,seeds,ticks):
    x=x.copy()
    for i,(seed,tick) in enumerate(zip(seeds,ticks)):
        rng=np.random.default_rng([int(seed),int(tick),914])
        if condition.endswith('temporal_shuffle'):x[i,:-1]=x[i,rng.permutation(x.shape[1]-1)]
        elif condition.endswith('action_shuffle'):x[i,:,-2:]=x[i,rng.permutation(x.shape[1]),-2:]
    return x


def load_model(run,c,condition,seed):
    m=condition.split('_')[0];p=Policy(c,m).to(c['device'])
    state=torch.load(run/'models'/f'{m}-{seed}'/'final.pt',map_location=c['device'],weights_only=False)
    p.load_state_dict(state['model']);return p.eval()


def simple_action(e,mode):
    if mode=='teacher':return e.teacher()
    if mode in ['nominal','brute_down']:return np.zeros(2)
    if mode=='constant_x':return np.array([1.,0.])
    if mode=='spiral':
        t=(e.tick-e.control_start)/e.c['horizon_ms'];radius=.003*t;angle=5*np.pi*t
        goal=radius*np.array([np.cos(angle),np.sin(angle)])
        return np.clip((goal-e.target)/e.c['max_action_m'],-1,1)
    raise ValueError(mode)


def batch(run,c,condition,ts,seeds,mirrors=None,snapshots=None):
    envs=[]
    for i,s in enumerate(seeds):
        e=InsertionEnv(c,s,1 if mirrors is None else mirrors[i],force_scale=8 if condition=='brute_down' else 1)
        envs.append(e.prepare() if snapshots is None else e.restore(snapshots[i]))
    model=load_model(run,c,condition,ts) if condition.startswith('M') else None
    norm=np.load(run/'normalizer.npz') if model is not None else None
    first_actions={}
    while any(not e.done() for e in envs):
        active=[e for e in envs if not e.done()]
        if model is None:actions=[simple_action(e,condition) for e in active]
        else:
            x=corrupt(np.stack([e.window() for e in active]),condition,[e.seed for e in active],[e.tick for e in active])
            with torch.inference_mode():actions=model(torch.tensor(normalize(x,norm['mean'],norm['scale']),device=c['device'])).cpu().numpy()
        for e,a in zip(active,actions):
            first_actions.setdefault(id(e),np.asarray(a).tolist());e.step(a)
    rows=[dict(seed=e.seed,mirror=e.mirror,train_seed=ts,condition=condition,first_action=first_actions.get(id(e)),**e.metrics()) for e in envs]
    return rows,envs


def evaluate(run,c):
    setup(c)
    for cond in CONDITIONS+['teacher','nominal','constant_x','spiral','brute_down']:
        for ts in c['train_seeds'] if cond.startswith('M') else [-1]:
            path=run/'eval'/f'{cond}-{ts}.json'
            if path.exists():continue
            start=time.perf_counter();rows=[]
            for off in range(0,c['eval_episodes'],c['eval_batch_size']):
                seeds=list(range(c['seed_starts']['test']+off,c['seed_starts']['test']+min(off+c['eval_batch_size'],c['eval_episodes'])))
                out,envs=batch(run,c,cond,ts,seeds);rows+=out
                directory=run/'traces'/f'{cond}-{ts}';directory.mkdir(parents=True,exist_ok=True)
                for e in envs:np.savez_compressed(directory/f'{e.seed}.npz',observations=e.observations,issued=e.actions,log=e.log)
            atomic(path,rows);progress(run,'evaluate',condition=cond,seed=ts,success=np.mean([r['success'] for r in rows]),seconds=time.perf_counter()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();evaluate(a.run,config(a.run))
