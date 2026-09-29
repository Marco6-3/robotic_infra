"""Seed-disjoint kernel baselines, finite-resolution pairs and cluster CIs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import yaml


def load_episodes(run):
    config=yaml.safe_load((run/'config.yaml').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    rows=[]
    for path in sorted((run/'episodes').glob('*.npz')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['episode_sha256'][path.name]:
            raise ValueError(f'episode hash mismatch: {path}')
        with np.load(path,allow_pickle=False) as z:
            row={k:z[k].copy() for k in z.files if k!='metadata'}
            row.update(json.loads(str(z['metadata'])))
        # Future labels remain separate; feature views contain causal indices only.
        row['causal']=row['ticks']<=config['anchor_tick']
        assert row['ticks'][row['causal']][-1]==config['anchor_tick']
        rows.append(row)
    return config,rows


def features(rows, ms, *, kind='tactile'):
    n=ms//5+1
    tactile=np.stack([r['tactile'][r['causal']][-n:] for r in rows])
    if kind=='repeat': tactile=np.repeat(tactile[:,-1:],n,axis=1)
    if kind=='shuffle':
        for i in range(len(tactile)):
            tactile[i,:-1]=tactile[i,np.random.default_rng(1781+i).permutation(n-1)]
    if kind=='mismatch':
        # Random whole-history donors from another seed, within this split.
        # A fixed cyclic shift is invalid: sorted rows preserve friction-bin order.
        donor = mismatched_donors(np.array([r['seed'] for r in rows]))
        tactile[:,:-1]=tactile[donor,:-1]
    if kind=='mean':tactile=tactile.mean(axis=1,keepdims=True)
    if kind=='proprio':
        q=np.stack([r['proprio'][r['causal']][-n:] for r in rows])
        tactile=np.concatenate([tactile,q],axis=2)
    return tactile.reshape(len(rows),-1)


def mismatched_donors(seeds):
    rng=np.random.default_rng(np.random.SeedSequence([9881,int(seeds[0])]))
    for _ in range(10000):
        donor=rng.permutation(len(seeds))
        if np.all(seeds[donor]!=seeds):return donor
    raise ValueError('cannot assign distinct-seed historical donors')


def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p);hard=p>=.5
    tp=int(np.sum(hard & (y==1)));tn=int(np.sum(~hard & (y==0)))
    fp=int(np.sum(hard & (y==0)));fn=int(np.sum(~hard & (y==1)))
    pos=p[y==1];neg=p[y==0]
    auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos)*len(neg) else None
    # Tie-grouped average precision, matching a thresholded PR staircase.
    ap=0.;previous_recall=0.
    for threshold in sorted(set(p),reverse=True):
        pred=p>=threshold
        recall=float(np.sum(y[pred]))/max(1,int(y.sum()))
        ap+=(recall-previous_recall)*float(np.mean(y[pred]));previous_recall=recall
    return dict(n=len(y),positive=int(y.sum()),brier=float(np.mean((p-y)**2)),
                balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
                auroc=auc,average_precision=ap if len(pos) else None,tp=tp,tn=tn,fp=fp,fn=fn)


def squared_distance(a,b):
    return np.maximum(((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1],0.)


def fit_predict(train,val,test,y_train,y_val,config):
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-8)
    a=(train-mean)/scale;b=(val-mean)/scale;c=(test-mean)/scale
    d_train=squared_distance(a,a);d_val=squared_distance(b,a);d_test=squared_distance(c,a)
    center=float(np.mean(y_train));best=None
    for length in config['kernel_lengths']:
        kernel=np.exp(-d_train/(2*length**2))
        kv=np.exp(-d_val/(2*length**2))
        for penalty in config['ridge_penalties']:
            alpha=np.linalg.solve(kernel+penalty*np.eye(len(a)),y_train-center)
            pv=np.clip(center+kv@alpha,0,1)
            loss=float(np.mean((pv-y_val)**2))
            if best is None or loss<best['validation_brier']:
                pt=np.clip(center+np.exp(-d_test/(2*length**2))@alpha,0,1)
                best=dict(validation_brier=loss,length=length,penalty=penalty,
                          alpha=alpha,test=pt,validation=pv,mean=mean,scale=scale,
                          normalized_train=a,center=center)
    return best


def bootstrap_delta(y,pa,pb,seeds,config):
    """Positive delta means B has lower Brier than A; independent unit=seed."""
    unique=np.unique(seeds)
    delta=(pa-y)**2-(pb-y)**2
    sums=np.array([delta[seeds==s].sum() for s in unique]);counts=np.array([np.sum(seeds==s) for s in unique])
    rng=np.random.default_rng(config['bootstrap_seed'])
    indices=rng.integers(len(unique),size=(config['bootstrap_samples'],len(unique)))
    boot=sums[indices].sum(1)/counts[indices].sum(1)
    return dict(delta_brier=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),independent_seeds=len(unique))


def alias_pairs(rows,config):
    candidates=[]
    for (seed,probe) in sorted({(r['seed'],r['probe']) for r in rows if r['split']=='test'}):
        group=[r for r in rows if r['split']=='test' and r['seed']==seed and r['probe']==probe and r['eligible']]
        for a,b in itertools.combinations(group,2):
            assert a['future_input_sha256']==b['future_input_sha256']
            fa=a['force_truth'][a['causal']][-1];fb=b['force_truth'][b['causal']][-1]
            qa=a['proprio'][a['causal']][-1];qb=b['proprio'][b['causal']][-1]
            diff=np.abs(qa-qb)
            limits=[config[k] for k in ['alias_arm_position_max_rad','alias_finger_position_max_m','alias_arm_velocity_max_rad_s','alias_finger_velocity_max_m_s']]
            maxima=[float(diff[:7].max()),float(diff[7:9].max()),float(diff[9:16].max()),float(diff[16:18].max())]
            candidates.append(dict(seed=seed,probe=probe,a=a['episode'],b=b['episode'],label_a=a['label'],label_b=b['label'],
                opposite=a['label']!=b['label'],force_max_n=float(np.max(np.abs(fa-fb))),
                force_rms_n=float(np.sqrt(np.mean((fa-fb)**2))),proprio_maxima=maxima,
                proprio_match=all(x<=limit for x,limit in zip(maxima,limits))))
    sensitivity=[]
    for probe in [False,True]:
        for threshold in [.01,.02,.03,.05]:
            subset=[p for p in candidates if p['probe']==probe]
            pairs=[p for p in subset if p['opposite'] and p['proprio_match'] and p['force_max_n']<=threshold]
            sensitivity.append(dict(probe=probe,force_threshold_n=threshold,candidates=len(subset),
                                    pairs=len(pairs),independent_seeds=len({p['seed'] for p in pairs})))
    return candidates,sensitivity


def analyze(run):
    config,all_rows=load_episodes(run)
    # A new analysis directory preserves prior models and reports.
    from datetime import datetime
    out=run/('analysis-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'));out.mkdir()
    (out/'evaluate.py').write_bytes(Path(__file__).read_bytes())
    models=out/'models';models.mkdir()
    summary=dict(run=str(run.resolve()),collection_count=len(all_rows),excluded=[r['episode'] for r in all_rows if not r['eligible']],
                 config_sha256=hashlib.sha256((run/'config.yaml').read_bytes()).hexdigest(),groups={},
                 environment=dict(device='CPU',numpy=np.__version__,workers=0,batch='full kernel; no stochastic minibatches'))
    predictions=[]
    for probe in [False,True]:
        groups={split:[r for r in all_rows if r['probe']==probe and r['split']==split and r['eligible']] for split in ['train','validation','test']}
        assert not (set(r['seed'] for r in groups['train']) & set(r['seed'] for r in groups['test']))
        y={s:np.array([r['label'] for r in rs]) for s,rs in groups.items()}
        seeds=np.array([r['seed'] for r in groups['test']])
        specs=[(f'tactile_{ms}ms',ms,'tactile') for ms in config['history_ms']]+[
            ('current_tactile_proprio',0,'proprio'),('history_tactile_proprio',250,'proprio'),
            ('repeat_current_250ms',250,'repeat'),('shuffle_time_250ms',250,'shuffle'),
            ('mismatch_past_250ms',250,'mismatch'),('mean_last_100ms',100,'mean')]
        results={};fitted={}
        for name,ms,kind in specs:
            x={s:features(rs,ms,kind=kind) for s,rs in groups.items()}
            model=fit_predict(x['train'],x['validation'],x['test'],y['train'],y['validation'],config)
            np.savez_compressed(models/f'probe{int(probe)}-{name}.npz',**model,train_y=y['train'])
            fitted[name]=model
            results[name]=dict(**metrics(y['test'],model['test']),validation_brier=model['validation_brier'],
                               kernel_length=model['length'],ridge_penalty=model['penalty'],input_dimensions=x['train'].shape[1],coefficients=len(y['train']))
            for row,label,pred in zip(groups['test'],y['test'],model['test']):
                predictions.append(dict(probe=probe,model=name,episode=row['episode'],seed=row['seed'],label=int(label),prediction=float(pred)))
        best=min([n for n,ms,k in specs if k=='tactile' and ms>0],key=lambda n:results[n]['validation_brier'])
        comparisons={}
        for base,name in [('tactile_0ms',best),('current_tactile_proprio','history_tactile_proprio'),
                          ('mean_last_100ms',best),('shuffle_time_250ms','tactile_250ms'),
                          ('mismatch_past_250ms','tactile_250ms')]:
            comparisons[f'{name}_over_{base}']=bootstrap_delta(y['test'],fitted[base]['test'],fitted[name]['test'],seeds,config)
        repeat_error=float(np.max(np.abs(fitted['tactile_0ms']['test']-fitted['repeat_current_250ms']['test'])))
        assert repeat_error<1e-8,'repeat-current must be mathematically equivalent'
        # Label-permuted training sanity control; validation still selects hyperparameters.
        x={s:features(rs,250) for s,rs in groups.items()}
        yp=np.random.default_rng(983).permutation(y['train'])
        sanity=fit_predict(x['train'],x['validation'],x['test'],yp,y['validation'],config)
        np.savez_compressed(models/f'probe{int(probe)}-permuted_labels.npz',**sanity,train_y=yp)
        results['permuted_labels']=metrics(y['test'],sanity['test'])
        summary['groups'][str(probe)]=dict(split_counts={s:dict(n=len(v),positive=int(y[s].sum())) for s,v in groups.items()},
            models=results,selected_history=best,comparisons=comparisons,repeat_max_error=repeat_error)
    pairs,sensitivity=alias_pairs(all_rows,config)
    qualified=[p for p in pairs if p['opposite'] and p['proprio_match'] and p['force_max_n']<=config['alias_force_max_n']]
    summary['alias_sensitivity']=sensitivity
    summary['qualified_pairs']=len(qualified)
    summary['qualified_pair_seeds']=len({p['seed'] for p in qualified})
    summary['future_inputs_matched']=all(len({r['future_input_sha256'] for r in all_rows if r['seed']==s})==1 for s in {r['seed'] for r in all_rows})
    probe_results=summary['groups']['True'];best=probe_results['selected_history']
    delta=probe_results['comparisons'][f'{best}_over_tactile_0ms']
    summary['primary_criteria']=dict(alias_exists=any(p['probe'] for p in qualified),history_improves=delta['ci95'][0]>0,
        supported=any(p['probe'] for p in qualified) and delta['ci95'][0]>0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'alias_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
    with (out/'predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    write_report(out,summary,qualified)
    print(json.dumps(dict(output=str(out),criteria=summary['primary_criteria'],qualified_pairs=len(qualified)),indent=2))
    return out


def write_report(out,s,pairs):
    lines=['# I001 controlled benchmark：第一层证据','',f"原始运行：`{s['run']}`",'',
           f"真实 MuJoCo episodes：{s['collection_count']}；anchor 不满足稳定夹持而排除：{len(s['excluded'])}。",
           f"后续输入配对一致：{s['future_inputs_matched']}。合格近似混叠对：{s['qualified_pairs']}，覆盖 {s['qualified_pair_seeds']} 个独立 seed。",'',
           '## 判定','',f"预定义两项判据：`{json.dumps(s['primary_criteria'])}`。",'',
           '支持范围仅为当前刚体力代理、固定探测和物理分布；不能推出精确数学不可观测、动作历史必要性、ContactBelief 架构优势、闭环收益或真实触觉结论。','']
    for probe,g in s['groups'].items():
        lines += [f'## 探测载荷：{probe}','',f"划分：`{g['split_counts']}`。上下文仅按验证集选择：**{g['selected_history']}**。",'',
                  '| 输入 | Brier ↓ | AUROC ↑ | balanced accuracy ↑ | AP ↑ |','| --- | ---: | ---: | ---: | ---: |']
        for name,m in g['models'].items():
            lines.append(f"| {name} | {m['brier']:.4f} | {m['auroc'] if m['auroc'] is None else format(m['auroc'],'.4f')} | {m['balanced_accuracy']:.4f} | {m['average_precision']:.4f} |")
        lines += ['', 'Brier 改善为正表示历史更好；95% CI 按测试 seed 聚类重采样：','']
        for name,c in g['comparisons'].items():lines.append(f"- {name}: {c['delta_brier']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]，n={c['independent_seeds']} seeds。")
        lines += ['',f"重复当前帧与单帧最大预测差：{g['repeat_max_error']:.3g}。",'']
    lines += ['## 近邻阈值敏感性','','| 探测 | 力逐通道阈值 N | 对数 | 独立 seeds | 候选对 |','| --- | ---: | ---: | ---: | ---: |']
    for r in s['alias_sensitivity']:lines.append(f"| {r['probe']} | {r['force_threshold_n']} | {r['pairs']} | {r['independent_seeds']} | {r['candidates']} |")
    lines += ['','匹配使用无噪声力真值验证距离；模型输入始终为独立带噪声观测。标签只用于离线选择相反结果的机制示例，未参与测试集输入或调参。','', '## 可复查近邻示例','']
    for pair in sorted([p for p in pairs if p['probe']],key=lambda p:p['force_max_n'])[:3]:
        lines.append(f"- `{pair['a']}` vs `{pair['b']}`：未来标签 {pair['label_a']}/{pair['label_b']}，当前最大通道力差 {pair['force_max_n']:.6f} N，q/dq 分组最大差 {pair['proprio_maxima']}。")
    lines += ['','## 限制与解释','',
        '- 不同摩擦属于不同隐藏物理参数；这是主动探测后识别接触参数的实验，不是已经证明瞬时动态状态必须用特定网络恢复。',
        '- 近似相等依赖传感分辨率；原始高精度 q/dq 或微小力差可能足够预测，必须同时查看 current_tactile_proprio。',
        '- 若 shuffle_time 同样有效，仅支持历史内容有用，不支持顺序记忆必要；若无探测/静态平均同样有效，历史去噪是替代解释。',
        '- 50 ms slip 标签依赖仿真接触速度和 5 ms 持续阈值，未评估长期掉落、恢复或学习控制。',
        '- 测试来自未见 seed、相同物理范围和单一方块；不存在跨任务、未见物体、外部 benchmark 或真机证据。',
        '- 源快照/配置/资产与 episode 哈希见 manifest；全配对见 alias_pairs.json；每 episode 预测见 predictions.csv；模型见 models/。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();analyze(args.run)

if __name__=='__main__':main()
