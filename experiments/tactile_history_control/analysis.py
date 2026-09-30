"""Paired clustered bootstrap, decision gates, inspectable tables and reports."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from .utils import atomic,config,read

METRICS=['success','drop','max_slip_m','final_slip_m','recovery_latency_s','peak_contact_force_n','mean_correction_m']

def matrix(rows,metric):
    ts=sorted(set(r['train_seed'] for r in rows));seeds=sorted(set(r['seed'] for r in rows))
    # Ambiguity variants belong to the same disturbance cluster.
    return np.array([[np.mean([r[metric] for r in rows if r['train_seed']==t and r['seed']==s]) for s in seeds] for t in ts])

def summarize(a,c):
    rng=np.random.default_rng(c['bootstrap_seed']);n=c['bootstrap_samples']
    ti=rng.integers(0,a.shape[0],(n,a.shape[0]));si=rng.integers(0,a.shape[1],(n,a.shape[1]))
    means=a[ti[:,:,None],si[:,None,:]].mean((1,2))
    return dict(mean=float(a.mean()),seed_sd=float(a.mean(1).std(ddof=1)) if len(a)>1 else 0.,
                ci95=np.quantile(means,[.025,.975]).tolist(),per_train_seed=a.mean(1).tolist(),n_train_seeds=a.shape[0],n_disturbance_seeds=a.shape[1])

def contrast(rows,left,right,c):
    if not rows.get(left) or not rows.get(right):return None
    a=matrix(rows[left],'success');b=matrix(rows[right],'success')
    if a.shape!=b.shape:raise ValueError('unpaired comparison')
    result=summarize(a-b,c);result['all_seeds_positive']=bool(np.all((a-b).mean(1)>0))
    return result

def passes(r,c):
    return bool(r and r['mean']>=c['minimum_effect'] and r['ci95'][0]>0 and r['all_seeds_positive'])

def fmt(r,percent=False):
    if r is None:return '未建立'
    mult=100 if percent else 1
    return f"{r['mean']*mult:.2f} [{r['ci95'][0]*mult:.2f}, {r['ci95'][1]*mult:.2f}]"

def analyze(run,c):
    rows={}
    for p in sorted((run/'eval').glob('*.json')):
        for r in read(p):rows.setdefault(r['condition'],[]).append(r)
    summary={k:{m:summarize(matrix(v,m),c) for m in METRICS} for k,v in rows.items()}
    diffs={name:contrast(rows,l,r,c) for name,l,r in [('A_M1_minus_M0','M1','M0'),('B_M2_minus_M1','M2','M1'),('M2_minus_action_shuffle','M2','M2_action_shuffle'),('M1_minus_temporal_shuffle','M1','M1_temporal_shuffle'),('M2_minus_temporal_shuffle','M2','M2_temporal_shuffle')]}
    pairs=read(run/'ambiguity/pairs.json');arows={}
    for p in (run/'ambiguity').glob('M*.json'):
        for r in read(p):arows.setdefault(r['condition'],[]).append(r)
    amb={k:dict(success=summarize(matrix(v,'success'),c),initial_action_error=summarize(matrix(v,'initial_action_error'),c)) for k,v in arows.items() if v}
    adiff=contrast(arows,'M2','M1',c)
    A=passes(diffs['A_M1_minus_M0'],c)
    B=passes(diffs['B_M2_minus_M1'],c) and passes(diffs['M2_minus_action_shuffle'],c) and len(pairs)>=c['ambiguity']['min_pairs'] and passes(adiff,c)
    stable_failures=sum(not r['initial_stable'] for v in rows.values() for r in v)
    status='PROMISING' if A and B else 'WEAK_SIGNAL' if A or B else 'NO_SIGNAL'
    constant=summary['constant_max']['success']['mean'];best=max(summary[k]['success']['mean'] for k in ['M0','M1','M2'])
    ceiling=constant>=best
    if stable_failures:status='CONFOUNDED'
    if status=='PROMISING' and ceiling:status='WEAK_SIGNAL'
    result=dict(status=status,decision='REVISE',hypothesis_A=A,hypothesis_B=B,comparisons=diffs,summary=summary,
                ambiguity=dict(pairs=len(pairs),metrics=amb,M2_minus_M1=adiff),initialization_failures=stable_failures,constant_max_matches_or_beats_learned=ceiling)
    atomic(run/'summary.json',result)
    with (run/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','metric','mean','seed_sd','ci_low','ci_high','n_train','n_disturbances'])
        for k,ms in summary.items():
            for m,r in ms.items():w.writerow([k,m,r['mean'],r['seed_sd'],*r['ci95'],r['n_train_seeds'],r['n_disturbance_seeds']])
    table=['|条件|成功 % [95% CI]|训练种子SD pp|掉落 %|最大滑移 mm|删失时延 ms|峰值单指力 N|平均收紧 mm|','|---|---|---|---|---|---|---|---|']
    for k in ['M0','M1','M2','M1_temporal_shuffle','M2_temporal_shuffle','M2_action_shuffle','teacher','nominal','constant_max']:
        s=summary[k];table.append(f"|{k}|{fmt(s['success'],True)}|{100*s['success']['seed_sd']:.2f}|{100*s['drop']['mean']:.2f}|{1000*s['max_slip_m']['mean']:.2f}|{1000*s['recovery_latency_s']['mean']:.1f}|{s['peak_contact_force_n']['mean']:.2f}|{1000*s['mean_correction_m']['mean']:.2f}|")
    difftext='\n'.join(f'- {k}: {fmt(r,True)} pp；各训练种子差值 {[round(v*100,2) for v in r["per_train_seed"]]}' for k,r in diffs.items())
    seedtext='\n'.join(f'- {k}: {[round(v*100,2) for v in summary[k]["success"]["per_train_seed"]]} %' for k in ['M0','M1','M2'])
    ambtext=f"按冻结规则找到 {len(pairs)} 对（最少要求 {c['ambiguity']['min_pairs']}），来自独立的 ambiguity seeds；未在评估后放宽阈值。\n"
    for k,v in amb.items():ambtext+=f"- {k}: 成功率 {fmt(v['success'],True)}%；初始 teacher action 绝对误差 {fmt(v['initial_action_error'])}。\n"
    ambtext+=f"\nM2−M1 配对成功率差：{fmt(adiff,True)} pp。"
    if len(pairs)<c['ambiguity']['min_pairs']:ambtext+=' 样本不足，Hypothesis B 的 ambiguity 条件未建立。'
    models=[read(p) for p in sorted((run/'models').glob('*/complete.json'))]
    hardware=read(run/'environment.json');elapsed=read(run/'timing.json') if (run/'timing.json').exists() else {}
    failure_text=f"初始化失败计数（包含重复条件）{stable_failures}；正式执行失败文件存在：{(run/'failure.json').exists()}。"
    limitation='只验证刚体 MuJoCo、几何压入量 taxel 和一个 known-pose 单块任务；teacher 仅启发式而非最优控制。共享训练数据的3个初始化不能代表3套独立数据采集。固定600步预算可能欠拟合；BC状态分布偏移、无真实RGB触觉/真机证据，均限制推广。'
    report=f'''# 闭环 tactile history Pilot 结果

状态：**{status}**；决策：**REVISE**。A 支持={A}；B 支持={B}。本次为 reduced-scale exploratory Pilot，非论文最终测试。

## 闭环结果

未见扰动种子上独立运行全部策略和诊断，主结论不由training loss得出。均值跨训练种子；CI同时重采样训练初始化及配对扰动seed，保留条件配对。

{chr(10).join(table)}

每个指标的CI和训练种子SD均在 metrics.csv / summary.json；全成功/全失败时percentile bootstrap会退化，不能据此声称总体概率确定。{c['eval_episodes']}/{c['eval_episodes']}全成功的单seed Wilson 95%区间约{100*c['eval_episodes']/(c['eval_episodes']+1.96**2):.1f}%–100%，显示有限样本不确定性。时延失败样本右删失到episode结束，不能当作全部成功恢复时间。max slip含掉落后的位移，不能解释成接触内微滑移。

训练种子顺序 {c['train_seeds']}：
{seedtext}

## 预先定义的核心差值

{difftext}

A/B均要求≥5pp、95%CI下界>0、各训练种子方向一致。B还要求action shuffle和匹配子集同时通过。此Pilot无差异不构成等效性证明。

## 近当前观测匹配分支

{ambtext}

pair距离/teacher gap见 ambiguity/pairs.json，最近但未必合格候选见 nearest.json，完整恢复状态见bank.pt。teacher gap是启发式动作差异，不是已证明的最优动作差异。匹配阈值允许多个量化档位差异，属于近观测诊断，不能证明观测严格混叠。所有模型从相同保存状态/历史开始闭环运行。匹配组为单独诊断，不替代主test。

## 数据、训练与失败

训练 {c['train_episodes']} episodes，validation {c['validation_episodes']}，每episode {(c['end_ms']-c['start_ms'])//c['sample_ms']} targets，共 {c['train_episodes']*((c['end_ms']-c['start_ms'])//c['sample_ms'])} 训练targets。test {c['eval_episodes']} distinct seeds × {len(c['train_seeds'])} model initializations ×6 learned conditions，外加3种参考策略各{c['eval_episodes']}集；ambiguity bank {c['ambiguity']['seeds']*c['ambiguity']['variants']}集及合格匹配分支。所有原始episode指标保留。

相同参数量：{sorted(set(m['parameters'] for m in models))}；相同步数 {c['train_steps']}，AdamW lr {c['learning_rate']}，batch {c['train_batch_size']}；M0重复当前观测、M1/M0置零action通道。固定最后checkpoint，无test选模。

{failure_text}

恒定最大收紧成功率 {100*constant:.2f}%，是否达到/超过最佳学习策略：{ceiling}。这是任务可被简单控制解决的直接替代解释；不能仅凭teacher恢复成功就宣称历史必要。

## 限制与下一步

{limitation}

若B未通过：当前任务没有证明 past action history 的独立价值。若A未通过：当前任务不足以支持temporal history的闭环改善，更不能声称必要。下一步先检查task/teacher在安全接触力约束下是否存在必须区别处理的近观测状态；只做小规模物理诊断，用新development seeds；不扩大网络、不基于本test反复调参。

## 图、运行与复现

![closed-loop](success.png)
![training](training.png)

硬件/依赖：
```json
{json.dumps(hardware,ensure_ascii=False,indent=2)}
```
阶段计时（秒）：
```json
{json.dumps(elapsed,indent=2)}
```

本次精确配置：
```yaml
{(run/'config.yaml').read_text()}```

恢复原运行：`env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.tactile_history_control.run --resume {run.resolve()}`。
扩大预算的新run命令：`env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.tactile_history_control.run --full-scale`（没有执行；需先明确task诊断是否值得加大规模，新seed分区，不把已看过Pilot test当作未见数据）。source/commit/diff/config/protocol/hash均保存在run中。
'''
    (run/'RESULTS.md').write_text(report)
    claim=f'''# 当前证据支持的 claim

本次在 FR3 MuJoCo disturbed-grasp recovery、双指简化几何taxel、200Hz、61-token、相同{models[0]['parameters']}参数Transformer、共享训练集的{len(c['train_seeds'])}个初始化及{c['eval_episodes']}个未见扰动seed上，M1−M0成功率差为{fmt(diffs['A_M1_minus_M0'],True)}pp，M2−M1为{fmt(diffs['B_M2_minus_M1'],True)}pp。

Hypothesis A：{'达到预定义支持门槛' if A else '未达到预定义支持门槛'}。Hypothesis B：{'达到预定义支持门槛' if B else '未达到预定义支持门槛；当前任务没有证明 past action history 的独立价值'}。

没有真机、RGB触觉、通用操作、最优控制或必要性结论。恒定最大收紧及观察模型仍是重要解释边界。
'''
    (run/'CLAIM.md').write_text(claim)
    (run/'NEGATIVE_RESULT.md').write_text(f'''# 未通过的假设与下一步

A通过={A}；B通过={B}。

{difftext}

{ambtext}

可能原因：当前观测足以反映本任务的接触变化；持续加紧可解决大部分扰动；teacher不显式依赖过去action；关节位置已编码部分动作后果；600步有限训练与BC分布偏移。上述是待检验解释，不把负结果归咎为确定的模型不足。

下一步最小修改：用新的development seeds检验加入明确物理合理接触力上限后，是否能找到T/q近似但合理安全动作不同的可恢复状态对。先比较teacher、恒定收紧与current-only可行性，不训练更大模型。本次不改test、不调参追求M2获胜。
''')
    (run/'REVIEW.md').write_text(f'''# Reviewer 审查

- 数据泄漏：normalizer只取train T/q/action；episode seed分区隔离，无future window，先observe后action。teacher privileged位移/接触速度只产生action label，无额外预测目标。输入42维白名单和spy测试核对。
- 输入局限：taxel由sim contact geometry生成，是显式仿真观测模型，不是真实触觉。没有将force当student输入。
- baseline公平性：M0/M1/M2同初始化、same-step minibatches、optimizer、步数、hidden、参数量 {sorted(set(m['parameters'] for m in models))}；M0重复current保证相同attention算量。缺少adaptive current控制的更强baseline可能影响广泛方法比较，但constant-max已提供简单替代解释。
- temporal leakage：包含当前T/q及a_(t-1)，不含a_t；causal mask和未来扰动不影响过去latent的测试通过。过去token打乱保留当前token。
- seed contamination：train、validation、main test、ambiguity seed分区不同；development范围修改只发生在正式训练前，并保存开发结果。初始化共享数据，CI不把它们当独立采集。
- privileged leakage：qvel/slip/friction/object pose不进入学生；teacher-only标签是明确授权的privileged imitation。ambiguity按teacher动作差值挑诊断pair，属于已声明的诊断定义，不是模型选择；因此不是总体random subset。
- cherry-picking：包含所有主评估seed/条件，保存失败与全部checkpoint曲线；pair阈值不因test结果放宽。匹配不足不补挑成功例。
- test-set tuning：固定末步checkpoint，不依validation/test选模；FROZEN记录checkpoint/config/source hashes；run resume验证冻结源代码。test只评估一次，恢复跳过原子完成的条件。
- 统计：paired two-way bootstrap，3个training initializations和{c['eval_episodes']}个disturbance seeds仍小，不证明等效性；多个诊断未做家族多重性校正，positive结果仍需独立确认。
- 主要科学风险：恒定最大收紧足够强，没有硬性force预算；teacher是启发式，不能称最优动作；匹配对{len(pairs)}个，初始化失败{stable_failures}。仅单任务、几何触觉、有限训练，可能欠拟合/BC covariate shift。
- 实时性：200Hz为仿真时间控制，不等于batch GPU evaluation实现了真机5ms最坏时延。闭环每步等待推理，没有异步丢包/USB模拟。

审查结论：{status}，保持Pilot边界；决策REVISE。详细原始证据见config、manifest、data、eval、ambiguity、metrics.csv与summary.json。
''')
    plots(run,c,summary)
    return result

def plots(run,c,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['M0','M1','M2','M2_action_shuffle','teacher','nominal','constant_max']
    means=np.array([summary[k]['success']['mean'] for k in keys])*100
    lo=np.array([summary[k]['success']['ci95'][0] for k in keys])*100
    hi=np.array([summary[k]['success']['ci95'][1] for k in keys])*100
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(keys,means,color=['#6b8cae']*3+['#ce9557']+['#759c7a']*3)
    ax.errorbar(keys,means,yerr=[means-lo,hi-means],fmt='none',color='black',capsize=4)
    ax.set_ylim(0,105);ax.set_ylabel('Recovery success (%)');ax.tick_params(axis='x',rotation=22);ax.set_title('Held-out closed-loop Pilot; paired clustered bootstrap 95% CI')
    fig.tight_layout();fig.savefig(run/'success.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for p in sorted((run/'models').glob('*/curve.json')):
        curve=read(p)
        for ax,k in zip(axes,['train_mse','validation_mse']):ax.plot([r['step'] for r in curve],[r[k] for r in curve],label=p.parent.name);ax.set_title(k);ax.set_xlabel('Optimizer step');ax.set_yscale('log')
    axes[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(run/'training.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyze(a.run,config(a.run))
