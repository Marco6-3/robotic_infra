"""Predeclared near-current matching; branches resume true MuJoCo state/history."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np
import torch
from .environment import RecoveryEnv
from .evaluate import evaluate_batch
from .utils import atomic,config,progress,read,save_pt,setup

def bank_one(job):
    c,seed,variant=job;e=RecoveryEnv(c,seed,variant).prepare()
    rng=np.random.default_rng([seed,variant,324]);strength=float(rng.uniform(0,.85))
    anchor=e.onset+c['ambiguity']['anchor_after_onset_ms']
    while e.tick<anchor:
        # Different causal actions, same exogenous task; common command for last 5ms.
        action=strength if e.tick<anchor-5 else .12
        e.step(action)
    return dict(seed=seed,variant=variant,snapshot=e.snapshot(),teacher=e.teacher(),prefix_metrics=e.metrics())

def match(bank,c):
    a=c['ambiguity'];scale=np.r_[np.full(32,a['tactile_tolerance_m']),np.full(7,a['arm_tolerance_rad']),np.full(2,a['finger_tolerance_m'])]
    pairs=[];nearest=[]
    for seed in sorted(set(b['seed'] for b in bank)):
        ids=[i for i,b in enumerate(bank) if b['seed']==seed];candidates=[]
        for j,i in enumerate(ids):
            for k in ids[j+1:]:
                diff=(bank[i]['snapshot']['history'][-1,:-1]-bank[k]['snapshot']['history'][-1,:-1])/scale
                rms=float(np.sqrt(np.mean(diff**2)));linf=float(abs(diff).max());gap=abs(bank[i]['teacher']-bank[k]['teacher'])
                candidates.append(dict(seed=seed,left=i,right=k,rms=rms,linf=linf,teacher_gap=gap))
        if candidates:nearest.append(min(candidates,key=lambda p:p['rms']))
        valid=[p for p in candidates if p['rms']<=a['rms_limit'] and p['linf']<=a['linf_limit'] and p['teacher_gap']>=a['teacher_gap']]
        # At most one pair per disturbance seed; no outcome/model-based ranking.
        if valid:pairs.append(min(valid,key=lambda p:p['rms']))
    return pairs,nearest

def run_ambiguity(run,c):
    setup(c);directory=run/'ambiguity';directory.mkdir(exist_ok=True)
    path=directory/'bank.pt'
    if path.exists():bank=torch.load(path,weights_only=False)
    else:
        jobs=[(c,c['seed_starts']['ambiguity']+i,v) for i in range(c['ambiguity']['seeds']) for v in range(c['ambiguity']['variants'])]
        with ProcessPoolExecutor(c['workers']) as pool:bank=list(pool.map(bank_one,jobs))
        save_pt(path,bank)
    pairs,nearest=match(bank,c);atomic(directory/'pairs.json',pairs);atomic(directory/'nearest.json',nearest)
    progress(run,'ambiguity',bank_episodes=len(bank),pairs=len(pairs))
    ids=sorted(set(i for p in pairs for i in [p['left'],p['right']]))
    for method in ['M0','M1','M2']:
        for ts in c['train_seeds']:
            path=directory/f'{method}-{ts}.json'
            if path.exists():continue
            rows=[]
            for offset in range(0,len(ids),c['eval_batch_size']):
                batch=ids[offset:offset+c['eval_batch_size']];items=[bank[i] for i in batch]
                out,envs=evaluate_batch(run,c,method,ts,[b['seed'] for b in items],[b['snapshot'] for b in items],[b['variant'] for b in items])
                for r,i,b,e in zip(out,batch,items,envs):
                    r.update(bank_id=i,teacher=b['teacher'],initial_action=e.log[0][1],initial_action_error=abs(e.log[0][1]-b['teacher']))
                    rows.append(r)
            atomic(path,rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run_ambiguity(a.run,config(a.run))
