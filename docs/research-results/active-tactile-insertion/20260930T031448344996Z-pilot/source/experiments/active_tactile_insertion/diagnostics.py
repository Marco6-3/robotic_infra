"""Constructed mirror probes; distinguish tactile-only from full T/q ambiguity."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np
import torch
from .environment import InsertionEnv
from .evaluate import batch
from .utils import atomic,save_pt,config,progress,setup


def build_pair(job):
    c,seed=job;out=[]
    for sign in [1,-1]:
        e=InsertionEnv(c,seed,mirror=sign).prepare()
        out.append(dict(seed=seed,mirror=sign,snapshot=e.snapshot(),teacher=e.teacher(),metrics=e.metrics()))
    return out


def pair_metrics(a,b,c):
    cfg=c['ambiguity'];x=a['snapshot']['history'];y=b['snapshot']['history']
    dt=(x[:,:32]-y[:,:32])/cfg['tactile_scale_m'];dq=(x[:,32:35]-y[:,32:35])/cfg['q_scale_m']
    both=np.concatenate([dt,dq],-1);aa=np.asarray(a['teacher']);bb=np.asarray(b['teacher'])
    cosine=float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-9))
    valid=not any(e['metrics']['prefix_force_failure'] or e['metrics']['prefix_success'] for e in [a,b])
    tq_rms=float(np.sqrt(np.mean(both**2)));tq_linf=float(abs(both).max())
    return dict(seed=a['seed'],tactile_rms=float(np.sqrt(np.mean(dt**2))),q_rms=float(np.sqrt(np.mean(dq**2))),
                tq_rms=tq_rms,tq_linf=tq_linf,action_history_rms=float(np.sqrt(np.mean((x[:,-2:]-y[:,-2:])**2))),
                expert_cosine=cosine,valid_prefix=valid,
                matched=bool(valid and tq_rms<=cfg['rms_limit'] and tq_linf<=cfg['linf_limit'] and cosine<=cfg['cosine_max']))


def reconstruct_actions(run,c):
    """Diagnostic only. Regression uses q history, predicts issued a_(t-1), never hidden state."""
    sets={}
    for split in ['train','validation']:
        features=[];targets=[]
        for p in sorted((run/'data'/split).glob('*.npz')):
            x=np.load(p)['x'];q=x[:,32:35]
            if len(q)<4:continue
            f=np.concatenate([q[3:]-q[2:-1],q[2:-1]-q[1:-2],q[1:-2]-q[:-3]],1)
            features.append(f);targets.append(x[3:,-2:])
        sets[split]=(np.concatenate(features),np.concatenate(targets))
    tr,ty=sets['train'];va,vy=sets['validation'];mean=tr.mean(0);scale=np.maximum(tr.std(0),1e-6)
    tr=np.c_[(tr-mean)/scale,np.ones(len(tr))];va=np.c_[(va-mean)/scale,np.ones(len(va))]
    w=np.linalg.solve(tr.T@tr+np.eye(tr.shape[1])*1e-3,tr.T@ty);pred=va@w
    active=abs(vy)>.1
    result=dict(validation_r2=float(1-np.sum((pred-vy)**2)/np.sum((vy-vy.mean(0))**2)),
                active_component_direction_accuracy=float(np.mean(np.sign(pred[active])==np.sign(vy[active]))),
                training_observations=len(tr),validation_observations=len(va),policy_inputs_changed=False)
    atomic(run/'diagnostics/action_reconstruction.json',result)
    np.savez(run/'diagnostics/action_reconstruction.npz',weights=w,mean=mean,scale=scale)


def diagnostics(run,c):
    setup(c);directory=run/'diagnostics';directory.mkdir(exist_ok=True)
    path=directory/'bank.pt'
    if path.exists():bank=torch.load(path,weights_only=False)
    else:
        with ProcessPoolExecutor(c['workers']) as p:pairs=list(p.map(build_pair,[(c,c['seed_starts']['ambiguity']+i) for i in range(c['ambiguity']['seeds'])]))
        bank=[a for pair in pairs for a in pair];save_pt(path,bank)
    metadata=[pair_metrics(bank[i],bank[i+1],c) for i in range(0,len(bank),2)]
    atomic(directory/'pairs.json',metadata);progress(run,'diagnostics',constructed_pairs=len(metadata),full_Tq_matches=sum(p['matched'] for p in metadata))
    # All constructed mirror pairs are evaluated; only prequalified full-T/q matches test B.
    for m in ['M0','M1','M2']:
        for ts in c['train_seeds']:
            path=directory/f'{m}-{ts}.json'
            if path.exists():continue
            rows=[]
            for offset in range(0,len(bank),c['eval_batch_size']):
                part=bank[offset:offset+c['eval_batch_size']]
                out,_=batch(run,c,m,ts,[b['seed'] for b in part],[b['mirror'] for b in part],[b['snapshot'] for b in part])
                for r,b in zip(out,part):
                    r['initial_action_error']=float(np.linalg.norm(np.asarray(r['first_action'])-b['teacher'])) if r['first_action'] is not None else None
                    r['teacher_action']=np.asarray(b['teacher']).tolist()
                rows+=out
            atomic(path,rows)
    reconstruct_actions(run,c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();diagnostics(a.run,config(a.run))
