"""Post-freeze probes: no gradients or decisions flow back into representations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from .dataset import load_split, source_config, transform
from .train import setup, tensor_data, load_model, infer
from .evaluate import metric_summary
from .utils import ROOT, read, atomic, check_frozen, sha, now, cluster

def ridge_fit(x, y, alpha=1.):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);ymean=y.mean(0);yscale=np.maximum(y.std(0),1e-6)
    a=(x-mean)/scale;b=(y-ymean)/yscale
    coef=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@b)
    return dict(mean=mean,scale=scale,ymean=ymean,yscale=yscale,coef=coef)

def ridge_predict(model,x):
    return ((x-model['mean'])/model['scale']@model['coef'])*model['yscale']+model['ymean']

def truth_targets(run,config,raw,split):
    if not (run/'FROZEN.json').exists():
        raise RuntimeError('Privileged probes forbidden before MODEL freeze')
    root=run/'episodes/final' if split=='final' else ROOT/config['source_run']/'episodes'/split
    privileged=[]
    for eid,meta in zip(raw['ids'],raw['meta']):
        with np.load(root/eid,allow_pickle=False) as f:
            row=f['labels'][f['labels'][:,0]==source_config(config)['anchor_tick']][0]
            # 1 friction + 3 position + 3 world linear velocity + 6 finger force + contact count.
            privileged.append(np.r_[meta['friction'],row[3:6],row[20:23],row[11:17],row[2]])
    future=raw['future'][:,-1].reshape(len(raw['y']),-1)
    names=['slip_outcome']+[f'future_{h}ms_obs_{d}' for h in config['horizons_ms'] for d in range(50)]
    names+=['friction','object_x','object_y','object_z','velocity_x','velocity_y','velocity_z',
            'left_normal','left_shear_x','left_shear_z','right_normal','right_shear_x','right_shear_z','contact_count']
    return np.c_[raw['y'],future,np.array(privileged)],names

def representations(run,config,regime,raw,shuffled=False):
    with np.load(run/'normalizers'/f'{regime}.npz') as f:norm=dict(f)
    data=transform(raw,norm)
    if shuffled:
        data['x']=data['x'].copy()
        for i,meta in enumerate(raw['meta']):
            order=np.random.default_rng([config['shuffle_seed'],meta['seed'],meta['variant']]).permutation(60)
            data['x'][i,:-1]=data['x'][i,order]
    tensors=tensor_data(data,config['device']);selection=read(run/'FROZEN.json')['selection'][regime]['models']
    result={};probabilities={}
    for name in ['B2','B4']:
        zs=[];es=[];ps=[]
        for record in selection[name]['replicates']:
            model=load_model(run,record,config);p,z,e=infer(model,tensors,config,representations=True)
            zs.append(z);es.append(e);ps.append(p)
        result[name+'_z']=np.concatenate(zs,axis=1);probabilities[name]=np.mean(ps,axis=0)
        if name=='B4':result['current_embedding']=np.concatenate(es,axis=1)
    result['current_raw']=data['x'][:,-1]
    return result,probabilities

def fit_probes(run,config):
    check_frozen(run);setup(config)
    if (run/'FINAL_SEEDS.json').exists():
        raise RuntimeError('Probe fitting forbidden after final seed reservation')
    directory=run/'probes';directory.mkdir(exist_ok=True);validation={};artifacts={}
    for regime in config['regimes']:
        train=load_split(run,config,regime,'train');val=load_split(run,config,regime,'validation')
        yt,names=truth_targets(run,config,train,'train');yv,_=truth_targets(run,config,val,'validation')
        zt,_=representations(run,config,regime,train);zv,_=representations(run,config,regime,val)
        for name,x in zt.items():
            model=ridge_fit(x.astype('float64'),yt,config['probe_ridge'])
            path=directory/f'{regime}-{name}.npz';np.savez(path,**model);artifacts[str(path.relative_to(run))]=sha(path)
            p=ridge_predict(model,zv[name]);validation[f'{regime}-{name}']=metric_summary(yv[:,0],np.clip(p[:,0],0,1))
    atomic(run/'probe_validation.json',validation)
    atomic(run/'PROBES_FROZEN.json',dict(time=now(),model_freeze_sha256=sha(run/'FROZEN.json'),
                                       alpha=config['probe_ridge'],artifacts=artifacts,target_names=names,
                                       used_splits=['train','validation'],hyperparameter_selection=False,
                                       privileged_targets_only_post_model_freeze=True))

def analyze(run,config):
    import csv
    check_frozen(run);setup(config);frozen=read(run/'PROBES_FROZEN.json')
    for rel,h in frozen['artifacts'].items():assert sha(run/rel)==h
    result={};alllatent={}
    with (run/'pairs.csv').open() as f:pair_rows=list(csv.DictReader(f))
    for regime in config['regimes']:
        raw=load_split(run,config,regime,'final');truth,names=truth_targets(run,config,raw,'final')
        zs,normal_probs=representations(run,config,regime,raw);shuffled,shuffle_probs=representations(run,config,regime,raw,shuffled=True)
        group={};prob={};index={eid:i for i,eid in enumerate(raw['ids'])}
        for name,x in zs.items():
            with np.load(run/'probes'/f'{regime}-{name}.npz') as f:model=dict(f)
            p=ridge_predict(model,x);ps=ridge_predict(model,shuffled[name]);prob[name]=np.clip(p[:,0],0,1)
            delta=(p-truth)/model['yscale'];deltas=(ps-truth)/model['yscale']
            per_channel=(delta**2).mean(0);per_channel_shuffled=(deltas**2).mean(0)
            future={}
            for i,horizon in enumerate(config['horizons_ms']):
                for modality,lo,hi in [('tactile',0,32),('q',32,41),('dq',41,50)]:
                    sl=slice(1+i*50+lo,1+i*50+hi)
                    future[f'{horizon}ms_{modality}']=dict(normalized_mse=float(per_channel[sl].mean()),shuffle_normalized_mse=float(per_channel_shuffled[sl].mean()))
            # Scale each representation by train-only per-dimension std before distances.
            normalized=(x-model['mean'])/model['scale'];pairs=[r for r in pair_rows if r['regime']==regime]
            distances=[];seeds=[];opposite=[]
            for r in pairs:
                distances.append(float(np.sqrt(np.mean((normalized[index[r['a']]]-normalized[index[r['b']]])**2))))
                seeds.append(int(r['seed']));opposite.append(r['opposite']=='True')
            distances=np.array(distances);seeds=np.array(seeds);opposite=np.array(opposite,dtype=bool)
            group[name]=dict(outcome=metric_summary(truth[:,0],prob[name]),
                             shuffled_outcome=metric_summary(truth[:,0],np.clip(ps[:,0],0,1)),
                             future_observation=future,
                             privileged_normalized_mse=dict(zip(names[151:],map(float,per_channel[151:]))),
                             privileged_shuffled_normalized_mse=dict(zip(names[151:],map(float,per_channel_shuffled[151:]))),
                             matched_opposite_distance=cluster(distances[opposite],seeds[opposite],config),
                             matched_same_outcome_distance=cluster(distances[~opposite],seeds[~opposite],config))
            alllatent[f'{regime}-{name}']=x
        comparisons={}
        for name in ['B2_z','current_embedding','current_raw']:
            comparisons['B4_z_over_'+name]=cluster((prob[name]-raw['y'])**2-(prob['B4_z']-raw['y'])**2,raw['seeds'],config)
        shuffle={name:dict(normal=metric_summary(raw['y'],normal_probs[name]),shuffled=metric_summary(raw['y'],shuffle_probs[name]),
                           brier_degradation=cluster((shuffle_probs[name]-raw['y'])**2-(normal_probs[name]-raw['y'])**2,raw['seeds'],config)) for name in ['B2','B4']}
        result[regime]=dict(probes=group,comparisons=comparisons,temporal_shuffle=shuffle,
                            distance_warning='RMS distance uses train-standardized dimensions; different embeddings need not have identical geometry; distances alone are not proof of better inference')
    atomic(run/'representation_analysis.json',result);np.savez_compressed(run/'latents.npz',**alllatent)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--fit-probes',action='store_true');a=p.parse_args()
    c=yaml.safe_load((a.run/'config.yaml').read_text());(fit_probes if a.fit_probes else analyze)(a.run,c)
