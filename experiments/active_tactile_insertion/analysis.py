"""Frozen safe-insertion comparisons; no test-driven task/model changes."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from experiments.tactile_history_control.analysis import matrix,summarize,contrast,passes,fmt
from .utils import atomic,config,read

METRICS=['success','force_failure','timeout','insertion_depth_m','lateral_error_m','peak_force_n','latency_s','search_path_m']
ORDER=['M0','M1','M2','M1_temporal_shuffle','M2_temporal_shuffle','M2_action_shuffle','teacher','nominal','constant_x','spiral','brute_down']


def plot(run,c,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    keys=['M0','M1','M2','M2_action_shuffle','teacher','nominal','spiral','brute_down']
    y=np.array([summary[k]['success']['mean']*100 for k in keys]);lo=np.array([summary[k]['success']['ci95'][0]*100 for k in keys]);hi=np.array([summary[k]['success']['ci95'][1]*100 for k in keys])
    axes[0].bar(keys,y);axes[0].errorbar(keys,y,yerr=[y-lo,hi-y],fmt='none',color='black',capsize=3);axes[0].set_ylim(0,105);axes[0].set_ylabel('Safe insertion success (%)');axes[0].tick_params(axis='x',rotation=45)
    for p in sorted((run/'models').glob('*/curve.json')):
        r=read(p);axes[1].plot([a['step'] for a in r],[a['validation_mse'] for a in r],label=p.parent.name)
    axes[1].set_yscale('log');axes[1].set_xlabel('Optimizer step');axes[1].set_ylabel('Validation action MSE');axes[1].legend(fontsize=6,ncol=3)
    fig.tight_layout();fig.savefig(run/'results.png',dpi=160);plt.close(fig)
    pairs=read(run/'diagnostics/pairs.json');fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter([p['tactile_rms'] for p in pairs],[p['q_rms'] for p in pairs],c=[p['expert_cosine'] for p in pairs],vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xlabel('Full-history tactile difference (scaled RMS)');ax.set_ylabel('Full-history q difference (scaled RMS)');ax.set_title('Constructed mirror probes: q cannot be omitted')
    fig.tight_layout();fig.savefig(run/'ambiguity.png',dpi=160);plt.close(fig)


def analyze(run,c):
    rows={}
    for p in (run/'eval').glob('*.json'):
        for r in read(p):rows.setdefault(r['condition'],[]).append(r)
    for k,v in rows.items():
        ts=sorted(set(r['train_seed'] for r in v))
        for t in ts:
            seeds=sorted(r['seed'] for r in v if r['train_seed']==t)
            assert seeds==list(range(c['seed_starts']['test'],c['seed_starts']['test']+c['eval_episodes']))
    summary={k:{m:summarize(matrix(v,m),c) for m in METRICS} for k,v in rows.items()}
    differences={name:contrast(rows,a,b,c) for name,a,b in [('A_M1_minus_M0','M1','M0'),('B_M2_minus_M1','M2','M1'),('M2_minus_action_shuffle','M2','M2_action_shuffle'),('M1_minus_temporal_shuffle','M1','M1_temporal_shuffle'),('M2_minus_temporal_shuffle','M2','M2_temporal_shuffle')]}
    pairs=read(run/'diagnostics/pairs.json');matched={p['seed'] for p in pairs if p['matched']};mirror={};qualified={}
    for p in (run/'diagnostics').glob('M*.json'):
        for r in read(p):
            mirror.setdefault(r['condition'],[]).append(r)
            if r['seed'] in matched:qualified.setdefault(r['condition'],[]).append(r)
    ms={k:summarize(matrix(v,'success'),c) for k,v in mirror.items()}
    dmatched=contrast(qualified,'M2','M1',c)
    A=passes(differences['A_M1_minus_M0'],c)
    B=passes(differences['B_M2_minus_M1'],c) and passes(differences['M2_minus_action_shuffle'],c) and len(matched)>=c['ambiguity']['min_pairs'] and passes(dmatched,c)
    status='PROMISING' if B else 'WEAK_SIGNAL' if A else 'NO_SIGNAL';go='GO' if B else 'NO-GO'
    ordinary=[r for k,v in rows.items() if k!='brute_down' for r in v]
    prefix_failure=sum(r['prefix_force_failure'] for r in ordinary);prefix_success=sum(r['prefix_success'] for r in ordinary)
    if prefix_failure or prefix_success:status='CONFOUNDED';go='NO-GO'
    recon=read(run/'diagnostics/action_reconstruction.json')
    result=dict(status=status,decision='REVISE' if go=='NO-GO' else 'CONTINUE TO DEVELOPMENT',go_no_go=go,hypothesis_A=A,hypothesis_B=B,summary=summary,comparisons=differences,
                matched_pairs=len(matched),matched_comparison=dmatched,all_mirror_pairs=ms,action_reconstruction=recon,prefix_failures=prefix_failure,prefix_successes=prefix_success)
    atomic(run/'summary.json',result)
    with (run/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','metric','mean','training_seed_sd','ci_low','ci_high'])
        for k,metrics in summary.items():
            for m,r in metrics.items():w.writerow([k,m,r['mean'],r['seed_sd'],*r['ci95']])
    table=['|条件|安全成功% [95%CI]|超力%|timeout%|深度mm|峰值接触力N|删失时延ms|搜索路径mm|','|---|---|---|---|---|---|---|---|']
    for k in ORDER:
        s=summary[k];table.append(f"|{k}|{fmt(s['success'],True)}|{100*s['force_failure']['mean']:.2f}|{100*s['timeout']['mean']:.2f}|{1000*s['insertion_depth_m']['mean']:.2f}|{s['peak_force_n']['mean']:.2f}|{1000*s['latency_s']['mean']:.1f}|{1000*s['search_path_m']['mean']:.2f}|")
    difftext='\n'.join(f"- {k}: {fmt(v,True)} pp；各training seed差值{[round(x*100,2) for x in v['per_train_seed']]}" for k,v in differences.items())
    pairtext=f"构造 {len(pairs)} 对镜像孔位/动作历史，但仅 {len(matched)} 对满足完整61步T/q匹配门槛（最低{c['ambiguity']['min_pairs']}对）。未删q或重排触觉channel来制造匹配。\n"
    pairtext+='\n'.join(f'- 所有镜像样本 {k}: {fmt(v,True)}%（描述性，不能代替匹配子集）' for k,v in ms.items())
    pairtext+=f"\n合格子集M2−M1：{fmt(dmatched,True)} pp。\n"
    tr=[read(p) for p in (run/'data/train').glob('*.json')];va=[read(p) for p in (run/'data/validation').glob('*.json')]
    targets=sum(r['targets'] for r in tr);counts={k:len(v) for k,v in rows.items()}
    models=[read(p) for p in (run/'models').glob('*/complete.json')]
    parameters=sorted(set(m['parameters'] for m in models));elapsed=read(run/'timing.json');env=read(run/'environment.json')
    failrows=[]
    for group in rows.values():
        for r in group:
            if not r['success']:failrows.append({k:r[k] for k in ['condition','train_seed','seed','force_failure','timeout','peak_force_n','lateral_error_m','prefix_force_failure']})
    atomic(run/'failures.json',failrows)
    resulttext=f'''# Active tactile insertion Pilot 结果

**{status} / {go}**；决策 **{result['decision']}**。A支持={A}，B支持={B}。这是缩小规模、单任务的探索性实验，不是最终论文测试。

## 安全闭环主比较

{chr(10).join(table)}

主评估均为新hole offset与probe sequence的IID种子，未做范围外泛化。CI对训练初始化及配对task seed同时bootstrap；3个初始化共享数据，不是独立采集。全成功/全失败的bootstrap区间退化不能证明总体概率为1/0；{c['eval_episodes']}/{c['eval_episodes']}全成功的单seed Wilson下界约{100*c['eval_episodes']/(c['eval_episodes']+1.96**2):.1f}%。其他指标完整CI/SD见metrics.csv。

{difftext}

主要门槛：至少5pp改善、CI下界>0、三个training seed均正；B还须action shuffle及足量全T/q匹配分支同时通过。不能用较低BC误差替代闭环收益。

## 主动probe与歧义诊断

{pairtext}

q历史重建上一下发动作：validation R²={recon['validation_r2']:.3f}，非微小action分量方向准确率={100*recon['active_component_direction_accuracy']:.2f}%。这是只读diagnostic，不是student新增特征；说明必须检查“q能告诉模型怎样运动”这一替代解释，不单凭此宣称信息完全等价。

原始pair距离、teacher方向、是否合格见diagnostics/pairs.json，physics snapshot见bank.pt。只看T_history相似而忽略q_history，不构成本实验M1/M2的信息隔离对照。镜像构造若不满足完整观测匹配，就如实判定没有建立所需歧义，而不是跑完后扩大阈值。

![diagnostic](ambiguity.png)

## 输入与物理限制

实际物理为三轴Cartesian夹持工装，两个内侧pad与圆截面带半球导入端peg接触；T为pad/shaft几何压入量map。它不是FR3完整链路、RGB触觉、wrench投影或真实传感器标定。q仅三轴encoder，绝无隐藏孔位、被动peg/mount位置、摩擦、速度或接触标签进入student。teacher特权仅生成动作label及evaluation。

孔为24片凸几何近似圆倒角孔，使用软接触；这限制对精密刚性装配/真实材料的推广。插入深度8mm包括5mm圆鼻，圆柱段越过1.5mm倒角喉部约1.5mm。安全力为所有peg-hole接触力模长之和，1ms监测；超8N即终止，不允许超力后成功。probe有统一encoder竖直限位，主phase固定1.5N向下preload、只控制Δx/Δy。接触工装缩小了运动规划变量，但并非高保真光学触觉系统。

## 数据、计算与完整性

训练{len(tr)} episodes / {targets}个非terminal teacher targets；validation {len(va)} episodes；零target train episodes={sum(r['targets']==0 for r in tr)}。主test task seeds={c['eval_episodes']}，训练初始化={c['train_seeds']}。主rollout数={sum(counts.values())}；所有镜像分支rollout数={sum(len(v) for v in mirror.values())}。原始失败完整保留在failures.json，逐trajectory输入/下发action/log见traces。

三方法参数量一致：{parameters}；各{c['train_steps']}steps，batch={c['train_batch_size']}，固定最终checkpoint。validation只记录，不选模。时间、配置、数据、normalizer与checkpoint均有hash；resume禁止改冻结输入。

普通条件prefix force failures（跨条件重复计数）={prefix_failure}，prefix success={prefix_success}。是否存在执行异常文件：{(run/'failure.json').exists()}。所有episode均进入结果，没有按成功率补抽seed。No-contact prefix比例等见原始metrics，不能把无接触经验描述为触觉识别。

## 解释与最小下一步

当前数据{'支持' if B else '没有证明'}在完整T/q历史之上加入past action能带来可重复的安全插入收益。{'M1−M0达到预定门槛。' if A else 'M1−M0也未达到预定门槛。'}随机probe使交互丰富，但不能保证命令与encoder历史提供独立信息；被动对中、BC覆盖不足、teacher特权不可观测或时序分布偏移仍是替代解释。

下一步先用少量新的development seeds检查实际任务中被阻挡/顺应运动能否形成“完整T/q历史相似、issued command不同且安全可行动作相反”的状态对。若不存在，修订科学问题为interaction-conditioned控制的工程价值，或研究明确且真实的执行器内部状态；不得隐藏q让M2胜出，也不直接扩大Transformer/转第二孔型。

## 图与复现

![closed-loop and learning](results.png)

硬件/软件：
```json
{json.dumps(env,ensure_ascii=False,indent=2)}
```
阶段耗时（秒）：
```json
{json.dumps(elapsed,indent=2)}
```
精确配置：
```yaml
{(run/'config.yaml').read_text()}```

恢复：`env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.active_tactile_insertion.run --resume {run.resolve()}`。
扩大预算的新run：`env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.active_tactile_insertion.run --full-scale`；未执行，当前NO-GO不建议自动扩大规模。
'''
    (run/'RESULTS.md').write_text(resulttext)
    (run/'CLAIM.md').write_text(f'''# 当前可支持的结论

仅在本Cartesian工装、几何触觉代理、圆鼻peg/倒角孔、固定安全力和IID held-out seed设置下：M1−M0 safe success差为{fmt(differences['A_M1_minus_M0'],True)}pp；M2−M1为{fmt(differences['B_M2_minus_M1'],True)}pp。

核心action-history独立收益假设{'达到预设门槛' if B else '未被当前实验确立'}。完整T/q历史匹配只有{len(matched)}对；不能用tactile-only相似性替代完整观测歧义。结论{go}，不声称history必要、最优contact Jacobian、真机泛化或新颖性。
''')
    (run/'NEGATIVE_RESULT.md').write_text(f'''# 负结果与不确定性

A支持={A}，B支持={B}；{go}。

{difftext}

{pairtext}

即使模型对action shuffle敏感，也可能是输入分布扰动，不能替代M2>M1。随机probe不是独立信息的数学保证，因为M1包含q；三步q回归的R²={recon['validation_r2']:.3f}也提示检查这一解释。若模型低成功，还应区分teacher不可观测、BC分布偏移、采样覆盖和任务物理问题，不能径直扩模。

下一步最小实验：保留full q，用新development seeds检验物理可恢复、全T/q匹配的相反正确动作对；不足则先改假设。未根据本test修改超力阈值、孔形、模型或匹配规则。
''')
    (run/'REVIEW.md').write_text(f'''# Reviewer审查

- 输入边界：37维=32几何taxel+3encoder q+2上一真实下发命令。observed T不读force/teacher，测试直接阻断这些接口；被动mount、孔位、摩擦、dq都不送student。
- teacher特权：仅MSE action target与evaluation；没有future/contact-state辅助目标。安全终止为任务定义，零target episode不补抽，计数已报告。
- 参数公平：{parameters}，M0重复current、M0/M1屏蔽两action维度；三者相同初始化、batch索引、optimizer、steps、最后checkpoint。不同骨干不参与比较。
- 因果：x_t的a是上一下发命令，clip后记录；先取history/teacher label再step。causal mask和snapshot restore有测试，no future observation。
- seed：development/train/validation/test/ambiguity分区，test与probe使用同seed确定但不同RNG流；全部条件成对。三初始化共享数据，报告cluster bootstrap，不当独立采集。
- test tuning：物理修复/圆鼻导入端/probe限位在学习前完成；模型评估后不改task/匹配门槛。固定最后checkpoint，val仅日志；FROZEN及source/config/data/model hashes核验。
- 歧义诊断：镜像task不等于相同观测，特别是q暴露运动方向。必须完整T/q匹配；本次合格{len(matched)}对。全部镜像结果描述性展示，不把不合格pair宣传为action独立价值证据。
- 强简单基线：nominal、constant+x、spiral、强down均真实闭环；teacher使用孔位不属于可部署student。圆鼻和软接触可能带来被动对中，必须与nominal比较。
- 力安全：1ms物理步检查contact force，超力failure不可恢复为success；固定8N阈值。MuJoCo接触尖峰/软接触材料非真实校准，安全结论仅此仿真。
- 实时：200Hz是仿真时间，不代表实际机器人5ms最坏推理时延。
- 有限证据：单形状、小数据、3初始化、{c['eval_episodes']}个test seeds，CI宽及全成功bootstrap退化限制结论。未做多重检验校正，不据此给paper-level显著性结论。
- 文献只支持任务背景，不支持本M2>M1主张；见LITERATURE.md。本轮没有world model、未来预测或大模型。

审查结论：{status} / {go}，{result['decision']}。保留所有失败、旧grasp结果和本轮开发记录。
''')
    plot(run,c,summary);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyze(a.run,config(a.run))
