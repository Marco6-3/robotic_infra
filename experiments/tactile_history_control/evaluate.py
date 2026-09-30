"""Paired unseen-seed CLOSED LOOP evaluation and frozen input corruptions."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from .environment import RecoveryEnv,JOINT_ORDER
from .policy import Policy,normalize
from .utils import atomic,config,progress,read,setup

CONDITIONS=['M0','M1','M2','M1_temporal_shuffle','M2_temporal_shuffle','M2_action_shuffle']

def corrupt(x,condition,seeds,tick):
    x=x.copy()
    for i,seed in enumerate(seeds):
        rng=np.random.default_rng([int(seed),int(tick),551])
        if condition.endswith('temporal_shuffle'):
            order=rng.permutation(x.shape[1]-1);x[i,:-1]=x[i,order]
        elif condition.endswith('action_shuffle'):
            x[i,:,-1]=x[i,rng.permutation(x.shape[1]),-1]
    return x

def load_model(run,c,condition,seed):
    method=condition.split('_')[0];model=Policy(c,method).to(c['device'])
    state=torch.load(run/'models'/f'{method}-{seed}'/'final.pt',map_location=c['device'],weights_only=False)
    model.load_state_dict(state['model']);return model.eval()

def evaluate_batch(run,c,condition,train_seed,seeds,snapshots=None,variants=None):
    envs=[]
    for i,s in enumerate(seeds):
        e=RecoveryEnv(c,s,0 if variants is None else variants[i])
        envs.append(e.prepare() if snapshots is None else e.restore(snapshots[i]))
    model=None if condition in ['teacher','nominal','constant_max'] else load_model(run,c,condition,train_seed)
    norm=np.load(run/'normalizer.npz') if model is not None else None
    while any(e.tick<c['end_ms'] for e in envs):
        active=[e for e in envs if e.tick<c['end_ms']]
        if model is None:actions=[e.teacher() if condition=='teacher' else float(condition=='constant_max') for e in active]
        else:
            x=np.stack([e.window() for e in active])
            # Each trajectory has its own corruption RNG; independent of batch order.
            x=np.stack([corrupt(x[i:i+1],condition,[e.seed],e.tick)[0] for i,e in enumerate(active)])
            x=normalize(x,norm['mean'],norm['scale'])
            with torch.inference_mode():actions=model(torch.tensor(x,device=c['device'])).cpu().numpy()
        for e,a in zip(active,actions):e.step(float(a))
    return [dict(seed=e.seed,variant=e.variant,train_seed=train_seed,condition=condition,**e.metrics()) for e in envs],envs

def evaluate(run,c):
    setup(c)
    for condition in CONDITIONS+['teacher','nominal','constant_max']:
        training=c['train_seeds'] if condition.startswith('M') else [-1]
        for ts in training:
            path=run/'eval'/f'{condition}-{ts}.json'
            if path.exists():continue
            records=[];start=time.perf_counter()
            for offset in range(0,c['eval_episodes'],c['eval_batch_size']):
                seeds=[c['seed_starts']['test']+i for i in range(offset,min(offset+c['eval_batch_size'],c['eval_episodes']))]
                rows,envs=evaluate_batch(run,c,condition,ts,seeds);records.extend(rows)
                # Full observations and actions permit replay and audit, one file/episode.
                directory=run/'eval_traces'/f'{condition}-{ts}';directory.mkdir(parents=True,exist_ok=True)
                for e in envs:np.savez_compressed(directory/f'{e.seed}.npz',log=e.log,observations=np.asarray(e.observations),joint_order=np.asarray(JOINT_ORDER))
            atomic(path,records);progress(run,'evaluate',condition=condition,seed=ts,success=np.mean([r['success'] for r in records]),seconds=time.perf_counter()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();evaluate(a.run,config(a.run))
