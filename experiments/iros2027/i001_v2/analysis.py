"""Frozen-test scores, seed-clustered comparisons and mechanism diagnostics."""
import csv,json
from pathlib import Path
import numpy as np
from .data import load_rows,pair_rows,features
from .model import BASELINES,predict


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);hard=p>=.5
    tp=int(np.sum(hard&(y==1)));tn=int(np.sum(~hard&(y==0)));fp=int(np.sum(hard&(y==0)));fn=int(np.sum(~hard&(y==1)))
    pos=p[y==1];neg=p[y==0]
    auc=float(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg))) if len(pos)*len(neg) else None
    calibration=[];ece=0.
    for lo in np.arange(0,1,.1):
        keep=(p>=lo)&((p<lo+.1) if lo<.89 else (p<=1))
        if keep.any():
            score=float(p[keep].mean());freq=float(y[keep].mean());ece+=keep.mean()*abs(score-freq)
            calibration.append(dict(lower=float(lo),n=int(keep.sum()),prediction=score,frequency=freq))
    return dict(n=len(y),positive=int(y.sum()),brier=float(np.mean((p-y)**2)),auroc=auc,
        balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)) if len(pos)*len(neg) else None,
        f1=2*tp/max(1,2*tp+fp+fn),tp=tp,tn=tn,fp=fp,fn=fn,ece=float(ece),calibration=calibration)


def clustered(values,seeds,c):
    values=np.asarray(values);seeds=np.asarray(seeds);unique=np.unique(seeds)
    if not len(values):return dict(mean=None,ci95=[None,None],seeds=0,n=0)
    sums=np.array([values[seeds==s].sum() for s in unique]);counts=np.array([np.sum(seeds==s) for s in unique])
    rng=np.random.default_rng(c['bootstrap_seed']);idx=rng.integers(len(unique),size=(c['bootstrap_samples'],len(unique)))
    boot=sums[idx].sum(1)/counts[idx].sum(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),seeds=len(unique),n=len(values))


def pair_score(pair,predictions):
    # Ties receive 0.5. Only opposite-outcome matched pairs are discriminative.
    d=(predictions[pair['a']]-predictions[pair['b']])*(pair['label_a']-pair['label_b'])
    return float(d>0)+.5*float(d==0)


def evaluate(run,c):
    assert (run/'FROZEN.json').is_file(),'models must freeze before test access'
    rows=[r for r in load_rows(run,c,'test') if r['eligible']]
    y=np.array([r['label'] for r in rows]);seeds=np.array([r['seed'] for r in rows]);fixed=np.array([r['fixed_friction'] for r in rows])
    results={};boots={};predcsv=[];allpairs=[];predsave={}
    for regime in ['exact','finite']:
        pairs=pair_rows(rows,c,regime);allpairs.extend(pairs);opposite=[p for p in pairs if p['opposite']]
        preds={};group={}
        for baseline in BASELINES:
            with np.load(run/'models'/f'{regime}-{baseline}.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
            p,seconds=predict(m,features(rows,c,regime,baseline));preds[baseline]=p
            byid={r['episode']:float(v) for r,v in zip(rows,p)}
            scores=np.array([pair_score(pair,byid) for pair in opposite]);pseeds=np.array([pair['seed'] for pair in opposite])
            group[baseline]=dict(**metrics(y,p),fixed_friction=metrics(y[fixed],p[fixed]) if fixed.any() else None,
                matched_pair=clustered(scores,pseeds,c),input_dimensions=len(m['mean']),trainable_coefficients=len(m['coef']),
                inference_seconds=seconds,validation_brier=float(m['validation_brier']))
            predsave[f'{regime}-{baseline}']=p
            for r,truth,prob in zip(rows,y,p):predcsv.append(dict(regime=regime,baseline=baseline,episode=r['episode'],seed=r['seed'],label=int(truth),prediction=float(prob)))
        comparisons={}
        for a,b in [('B1','B4'),('B2','B3'),('N1','B4'),('N2','B4')]:
            delta=(preds[a]-y)**2-(preds[b]-y)**2
            pa={r['episode']:float(v) for r,v in zip(rows,preds[a])};pb={r['episode']:float(v) for r,v in zip(rows,preds[b])}
            dp=np.array([pair_score(p,pb)-pair_score(p,pa) for p in opposite]);ps=np.array([p['seed'] for p in opposite])
            fop=[p for p in opposite if p['fixed']]
            comparisons[f'{b}_over_{a}']=dict(brier=clustered(delta,seeds,c),fixed_friction_brier=clustered(delta[fixed],seeds[fixed],c),
                matched_pair=clustered(dp,ps,c),fixed_friction_pair=clustered([pair_score(p,pb)-pair_score(p,pa) for p in fop],[p['seed'] for p in fop],c))
        results[regime]=dict(models=group,matched_pairs=len(pairs),qualified_pairs=len(opposite),qualified_seeds=len(set(p['seed'] for p in opposite)),
            fixed_qualified_pairs=sum(p['fixed'] for p in opposite));boots[regime]=comparisons
    # Predictor does not receive friction. Here it is used only for posthoc stratification.
    friction_diagnostics=[]
    for lo,hi in zip([.45,.50,.55,.60,.65,.70,.75],[.50,.55,.60,.65,.70,.75,.81]):
        keep=np.array([lo<=r['friction']<hi for r in rows])
        if keep.any():friction_diagnostics.append(dict(lower=lo,upper=hi,n=int(keep.sum()),slip_rate=float(y[keep].mean()),
            b1_brier=float(np.mean((predsave['finite-B1'][keep]-y[keep])**2)),b4_brier=float(np.mean((predsave['finite-B4'][keep]-y[keep])**2))))
    (run/'metrics.json').write_text(json.dumps(results,indent=2)+'\n');(run/'bootstrap.json').write_text(json.dumps(boots,indent=2)+'\n')
    for filename,records in [('pairs.csv',allpairs),('predictions.csv',predcsv)]:
        with (run/filename).open('w') as f:
            fields=list(records[0]) if records else ['regime','seed','a','b'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    np.savez_compressed(run/'predictions.npz',**predsave)
    (run/'friction_diagnostics.json').write_text(json.dumps(friction_diagnostics,indent=2)+'\n')
    write_report(run,c,results,boots,friction_diagnostics)
    return results


def positive(entry):return entry['ci95'][0] is not None and entry['ci95'][0]>0


def write_report(run,c,results,boots,friction):
    b=boots['finite'];r=results['finite'];primary=b['B4_over_B1']
    count_ok=r['qualified_pairs']>=c['target_pairs'] and r['qualified_seeds']>=c['target_pair_seeds']
    main=positive(primary['brier']) and positive(primary['matched_pair'])
    controls=positive(b['B4_over_N1']['brier']) and positive(b['B4_over_N2']['brier'])
    action=positive(b['B3_over_B2']['brier']);fixed=positive(primary['fixed_friction_brier']) and positive(primary['fixed_friction_pair'])
    status='supported' if count_ok and main and controls and fixed else ('partially supported' if count_ok and main else 'not supported')
    verdict=dict(status=status,construction_target_met=count_ok,primary_and_pair_gain=main,negative_controls=controls,action_gain=action,fixed_friction_gain=fixed,
        scope='specified observation model and single controlled geometry; no RGB, hardware, new architecture or universal necessity')
    (run/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    progress=json.loads((run/'progress.json').read_text())
    lines=['# I001-v2 正式实验结果','',f'结论：**{status}**。',f'运行目录：`{run}`','',
        f"有限分辨率测试集：{r['qualified_pairs']} 个相反结局匹配对、{r['qualified_seeds']} 个独立 seed；构建门槛通过：{count_ok}。",'',
        '主比较固定 B4 vs B1，300 ms 历史、200 Hz；核宽/正则仅在 validation 选择并在生成 test 前冻结。',
        '配对搜索仅使用当前传感器观测与当前命令；结局/摩擦/物体真值只在候选对固定后作注释。','',
        f"数据统计：`{json.dumps(progress.get('statistics',{}),ensure_ascii=False)}`",'']
    for regime,g in results.items():
        lines += [f'## {regime}','', '| 输入 | Brier ↓ | AUROC ↑ | BA ↑ | F1 ↑ | ECE ↓ | pair accuracy ↑ |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name,m in g['models'].items():
            def fmt(v):return 'NA' if v is None else f'{v:.4f}'
            lines.append('| '+name+' | '+' | '.join(fmt(m[k]) for k in ['brier','auroc','balanced_accuracy','f1','ece'])+' | '+fmt(m['matched_pair']['mean'])+' |')
        lines += ['', '按 physics seed 聚类的配对 bootstrap，改善为正：','']
        for name,comp in boots[regime].items():lines.append(f"- {name}: Brier `{comp['brier']}`；pair accuracy `{comp['matched_pair']}`；固定摩擦 Brier `{comp['fixed_friction_brier']}`；固定摩擦 pair `{comp['fixed_friction_pair']}`。")
    lines += ['', '固定数据下的置信区间不涵盖训练数据重采样；配对共享 episode，通过 seed 聚类避免把所有 pair 当独立样本。',
        '模型均最多 192 个核回归系数，统一零填充/当前帧重复后的 61×58 维输入；重复不增加信息，匹配矩阵计算预算。metrics.json 记录维度、参数量与推理时间。',
        '额外微观力只作为 labels；输入触觉为接触穿透几何生成的 4×4 双指缩进代理，不是力传感器、RGB 或经过标定的真实触觉。','']
    (run/'RESULTS.md').write_text('\n'.join(lines))
    diag=['# I001-v2 机制诊断','',f'判定：`{json.dumps(verdict,ensure_ascii=False)}`','',
        f"固定摩擦主比较：`{primary['fixed_friction_brier']}`；固定摩擦配对：`{primary['fixed_friction_pair']}`。",'',
        f"动作条件化 B3 vs B2：`{b['B3_over_B2']}`。",'',
        'exact 与 finite 使用同一物理 episode；量化/噪声/匹配阈值在正式采集前冻结。有限分辨率不是实际硬件精度。',
        'B1 始终保留 q/dq 和当前命令；禁止根据 test 改传感模型。两种 regime 的差异是预注册的精度敏感性分析。',
        '固定摩擦无增益时不能宣称收益独立于摩擦辨识；细摩擦区间的滑移率与 B1/B4 损失见 friction_diagnostics.json。',
        '没有验证泛化到不同物体、外部 benchmark 或实际机器人。随机主动夹爪探测确实进入 B3/B4；未来输入逐 seed 完全一致。',
        'test 数量只按已冻结的配对数量/覆盖率规则扩展，未根据 test 性能改模型；模型本身和历史长度在首个 test 生成前冻结。',
        '阴性判据触发后不做架构搜索。本结果限制在所检验模型族和任务，不能证明历史在所有接触任务中无用。','']
    if not count_ok:diag += ['**Benchmark construction failure**：到预注册上限仍未获得足够相反结局配对/独立 seed；不能把未达样本门槛的结果升级为 H1 支持。','']
    if not fixed:diag += ['固定摩擦证据不足，不能排除摩擦辨识解释。','']
    if not action:diag += ['动作历史独立增益未成立，不支持 action-conditioning necessity。','']
    (run/'DIAGNOSIS.md').write_text('\n'.join(diag))
    claim='Under the specified observation model and controlled grasping environment, matched current tactile-proprioceptive observations can correspond to different near-future contact outcomes, while causal action-conditioned interaction history provides additional predictive information on unseen physics seeds.' if status=='supported' and action else '本轮未建立完整的 action-conditioned 第一层机制主张。只报告 RESULTS.md 中实际成立的限定比较；不得宣称 ContactBelief 已证实、通用必要性、SOTA 或真实世界有效性。'
    (run/'FIRST_LAYER_CLAIM.md').write_text('# 第一层可使用的主张\n\n'+claim+'\n')
