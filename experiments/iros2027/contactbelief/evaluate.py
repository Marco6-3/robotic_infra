"""Frozen predictions, full metric CIs, strict paired method comparisons."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
import yaml
from experiments.iros2027.i001_v2.data import pair_rows, load_rows
from .dataset import load_split, source_config, transform
from .train import setup, tensor_data, load_model, infer
from .models import parameter_report
from .utils import read, atomic, check_frozen, cluster

METRICS = ['brier', 'auroc', 'balanced_accuracy', 'f1', 'ece']

def weighted_metrics(y, p, weights):
    """Weights may repeat a physics cluster; tied-score AUROC is weighted exactly."""
    w = np.atleast_2d(weights).astype('float64'); y = np.asarray(y); p = np.asarray(p)
    hard = p >= .5; n = w.sum(1); pos = w@y; neg = n-pos
    tp = w@(y*hard); tn = w@((1-y)*~hard); fp = neg-tn; fn = pos-tp
    brier = w@((p-y)**2)/n
    ba = .5*(tp/np.maximum(pos, 1)+tn/np.maximum(neg, 1))
    f1 = 2*tp/np.maximum(2*tp+fp+fn, 1)
    order = np.argsort(p, kind='stable'); unique, inv = np.unique(p[order], return_inverse=True)
    auc = []
    for row in w:
        positive = np.bincount(inv, weights=row[order]*y[order], minlength=len(unique))
        negative = np.bincount(inv, weights=row[order]*(1-y[order]), minlength=len(unique))
        denom = positive.sum()*negative.sum()
        auc.append(float((positive*(np.cumsum(negative)-.5*negative)).sum()/denom) if denom else .5)
    ece = np.zeros(len(w)); bins = np.minimum((p*10).astype(int), 9)
    for b in range(10):
        keep = bins == b
        ece += np.abs(w[:, keep]@(p[keep]-y[keep]))/n
    return np.stack([brier, np.asarray(auc), ba, f1, ece], axis=1)

def bootstrap_weights(seeds, config):
    unique, inv = np.unique(seeds, return_inverse=True)
    draws = np.random.default_rng(config['bootstrap_seed']).integers(len(unique), size=(config['bootstrap_samples'], len(unique)))
    counts = np.stack([np.bincount(row, minlength=len(unique)) for row in draws])
    return counts[:, inv]

def metric_summary(y, p, weights=None):
    point = weighted_metrics(y, p, np.ones(len(y)))[0]
    result = dict(zip(METRICS, map(float, point)))
    result.update(n=len(y), positive=int(np.sum(y)))
    if weights is not None:
        boot = np.concatenate([weighted_metrics(y,p,w) for w in np.array_split(weights,20)])
        result['ci95'] = dict(zip(METRICS, np.quantile(boot,[.025,.975],axis=0).T.tolist()))
    return result

def pair_values(p, pairs, index):
    a = np.array([index[r['a']] for r in pairs], dtype=int); b = np.array([index[r['b']] for r in pairs], dtype=int)
    direction = np.array([r['label_a']-r['label_b'] for r in pairs])
    d = (p[a]-p[b])*direction
    return (d>0).astype(float)+.5*(d==0)

def compare(y, a, b, seeds, pairs, index, config, fixed):
    ps = np.array([r['seed'] for r in pairs]); scores = pair_values(b,pairs,index)-pair_values(a,pairs,index)
    fm = np.array([r['fixed'] for r in pairs], dtype=bool)
    return dict(brier_improvement=cluster((a-y)**2-(b-y)**2,seeds,config),
                pair_improvement=cluster(scores,ps,config),
                fixed_friction_brier=cluster(((a-y)**2-(b-y)**2)[fixed],seeds[fixed],config),
                fixed_friction_pair=cluster(scores[fm],ps[fm],config))

def cost(model, config):
    result = parameter_report(model); e=config['encoder_dim'];h=config['hidden_dim'];t=61
    head=h*(h//2)+(h//2)
    if model.method=='B0': mac=58*e+e*h+head
    elif model.method=='B1': mac=t*58*config['stack_hidden']+config['stack_hidden']*h+head
    elif model.method in ['B2','B4']: mac=t*(58*e+3*(e*h+h*h))+head
    else: mac=t*58*e+config['transformer_layers']*(t*8*e*e+2*t*t*e)+head
    result['approx_outcome_flops']=2*mac
    result['approx_auxiliary_forward_flops']=2*t*3*((h+17)*32+32*50) if model.method=='B4' else 0
    result['flop_convention']='2 FLOPs per multiply-accumulate; ignores bias/normalization/activation; dense full-history inference'
    x=torch.zeros(1,61,58,device=config['device']);times=[]
    with torch.no_grad():
        for _ in range(20): model(x)
        if config['device']=='cuda':torch.cuda.synchronize()
        for _ in range(100):
            start=time.perf_counter();model(x)
            if config['device']=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    result['batch1_latency_ms_median']=float(np.median(times));result['batch1_latency_ms_p95']=float(np.quantile(times,.95))
    result['latency_scope']='single model, preloaded device tensor, full 300ms history; no data IO; CUDA synchronized'
    result['ensemble_replicates']=len(config['training_seeds'])
    result['ensemble_approx_flops']=len(config['training_seeds'])*result['approx_outcome_flops']
    return result

def evaluate(run, config):
    frozen=check_frozen(run);setup(config)
    results={};comparisons={};records=[];allpairs=[];prediction_arrays={};costs={};replicate_results={};shortcut={}
    for regime in config['regimes']:
        raw=load_split(run,config,regime,'final')
        with np.load(run/'normalizers'/f'{regime}.npz') as f:norm=dict(f)
        tensors=tensor_data(transform(raw,norm),config['device']);y=raw['y'];seeds=raw['seeds'];index={s:i for i,s in enumerate(raw['ids'])}
        # The inherited function selects by CURRENT obs only before annotating labels.
        rows=[r for r in load_rows(run,source_config(config),'final') if r['eligible']]
        pairs=pair_rows(rows,source_config(config),regime);allpairs.extend(pairs);op=[r for r in pairs if r['opposite']]
        fixed=np.array([r['fixed_friction'] for r in raw['meta']]);weights=bootstrap_weights(seeds,config)
        models=frozen['selection'][regime]['models'];preds={};individual={};group={}
        for name,selection in models.items():
            pp=[]
            for r in selection['replicates']:
                model=load_model(run,r,config);p=infer(model,tensors,config);pp.append(p)
                prediction_arrays[f'{regime}-{name}-seed{r["seed"]}']=p
                replicate_results[f'{regime}-{name}-seed{r["seed"]}']=metric_summary(y,p)
                if name!='B2matched' and len(pp)==1:costs[f'{regime}-{name}']=cost(model,config)
            p=np.mean(pp,axis=0);preds[name]=p;individual[name]=pp;prediction_arrays[f'{regime}-{name}']=p
            group[name]=metric_summary(y,p,weights)
            group[name]['matched_pair']=cluster(pair_values(p,op,index),[r['seed'] for r in op],config)
            group[name]['fixed_friction']=metric_summary(y[fixed],p[fixed])
            for eid,s,yy,prob in zip(raw['ids'],seeds,y,p):
                records.append(dict(regime=regime,method=name,episode=eid,seed=int(s),label=int(yy),prediction=float(prob)))
        comparisons[regime]={}
        for baseline in ['B0','B1','B2','B2matched','B3']:
            entry=compare(y,preds[baseline],preds['B4'],seeds,op,index,config,fixed)
            entry['training_seed_brier_improvements']=[float(np.mean((a-y)**2-(b-y)**2)) for a,b in zip(individual[baseline],individual['B4'])]
            entry['training_seed_pair_improvements']=[float(np.mean(pair_values(b,op,index)-pair_values(a,op,index))) for a,b in zip(individual[baseline],individual['B4'])]
            comparisons[regime]['B4_over_'+baseline]=entry
        results[regime]=dict(models=group,episodes=len(y),seeds=len(set(seeds)),matched_candidates=len(pairs),
                             opposite_pairs=len(op),pair_seed_coverage=len(set(r['seed'] for r in op)),
                             strongest_generic=frozen['selection'][regime]['strongest_generic'])
        # Predeclared read-only stratification: never selects checkpoints/lambda.
        strata={}
        for lo,hi in [(2325,2360),(2360,2395),(2395,2431)]:
            keep=np.array([lo<=r['probe_start']<hi for r in raw['meta']])
            strata[f'probe_start_{lo}_{hi}']=dict(n=int(keep.sum()),brier_improvement=cluster(((preds['B2']-y)**2-(preds['B4']-y)**2)[keep],seeds[keep],config))
        for lo,hi in [(.45,.5),(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,.75),(.75,.81)]:
            keep=np.array([lo<=r['friction']<hi for r in raw['meta']])
            strata[f'friction_{lo}_{hi}']=dict(n=int(keep.sum()),brier_improvement=cluster(((preds['B2']-y)**2-(preds['B4']-y)**2)[keep],seeds[keep],config))
        shortcut[regime]=dict(strata=strata,majority_prior_brier=float(np.mean((float(frozen['training_positive_rate'])-y)**2)),
                              identifiers_in_inputs=False,unseen_seeds=True,balanced_accuracy_delta=group['B4']['balanced_accuracy']-group['B2']['balanced_accuracy'])
    atomic(run/'metrics.json',results);atomic(run/'bootstrap.json',comparisons);atomic(run/'costs.json',costs)
    atomic(run/'replicate_metrics.json',replicate_results);atomic(run/'shortcut_analysis.json',shortcut)
    np.savez_compressed(run/'predictions.npz',**prediction_arrays)
    for name,rows in [('predictions.csv',records),('pairs.csv',allpairs)]:
        with (run/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['regime','a','b']);writer.writeheader();writer.writerows(rows)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();evaluate(a.run,yaml.safe_load((a.run/'config.yaml').read_text()))
