"""Preregistered offline gate; no post-test rescue/model reselection."""
from .utils import read, atomic

def positive(entry):
    return entry['ci95'][0] is not None and entry['ci95'][0]>0

def both(entry):
    return positive(entry['brier_improvement']) and positive(entry['pair_improvement'])

def report(run,config):
    metrics=read(run/'metrics.json');boot=read(run/'bootstrap.json');reps=read(run/'representation_analysis.json')
    frozen=read(run/'FROZEN.json');audit=read(run/'final_data_audit.json');shortcuts=read(run/'shortcut_analysis.json')
    finite=boot['finite'];strongest=frozen['selection']['finite']['strongest_generic'];gru=finite['B4_over_B2'];strict=finite['B4_over_B2matched']
    stable=all(v>0 for v in strict['training_seed_brier_improvements']) and sum(v>0 for v in strict['training_seed_pair_improvements'])>=2
    gate1=both(gru) and both(strict) and stable
    gate2=both(finite['B4_over_'+strongest]) and both(finite['B4_over_B3'])
    pr=reps['finite']['comparisons']
    gate3=positive(pr['B4_z_over_B2_z']) or positive(pr['B4_z_over_current_embedding'])
    timing=[v['brier_improvement']['mean'] for k,v in shortcuts['finite']['strata'].items() if k.startswith('probe_start')]
    exact_delta=boot['exact']['B4_over_B2']['brier_improvement']['mean']
    gate4=(positive(gru['fixed_friction_brier']) and positive(gru['fixed_friction_pair'])
           and sum(v is not None and v>0 for v in timing)>=2 and exact_delta>0
           and shortcuts['finite']['balanced_accuracy_delta']>=0
           and positive(reps['finite']['temporal_shuffle']['B4']['brier_degradation']))
    construction=audit['seeds']>=100 and audit['qualified_pairs']>=1000 and audit['qualified_seed_coverage']>=80
    gates=dict(gate1_predictive_objective=gate1,gate2_generic_temporal=gate2,gate3_representation=gate3,
               gate4_shortcut_checks=gate4,adequate_final_construction=construction)
    go=all(gates.values());verdict=dict(go=go,gates=gates,strongest_generic=strongest,
                                      scope='controlled grasp, fixed anchor and 100ms outcome; offline only',
                                      action='closed-loop stage authorized' if go else 'stop method escalation; write negative result')
    atomic(run/'GATE.json',verdict)
    lines=['# ContactBelief-v0 正式方法实验结果','',f'Offline gate：**{"GO" if go else "NO-GO"}**。',
           f"全新 FINAL TEST：{audit['episodes']} episodes / {audit['seeds']} physics seeds；{audit['qualified_pairs']} 相反结果匹配对，覆盖 {audit['qualified_seed_coverage']} seeds。",'',
           f'主要比较为 ContactBelief 对 validation 预先选出的最强通用历史模型 **{strongest}**；同时重点检验 B2 和相同 learning rate 的 B2matched。',
           '主要估计量是三个初始化模型的概率均值；各初始化独立结果和差值保留。所有候选训练相同步数，checkpoint 只按 validation Brier 选择。','']
    for regime in config['regimes']:
        lines += [f'## {regime}','','| 模型 | Brier ↓ | AUROC ↑ | BA ↑ | F1 ↑ | ECE ↓ | pair ↑ |','|---|---:|---:|---:|---:|---:|---:|']
        for name,m in metrics[regime]['models'].items():
            pair=m['matched_pair']['mean'];values=[m[k] for k in ['brier','auroc','balanced_accuracy','f1','ece']]+[pair]
            lines.append('| '+name+' | '+' | '.join('NA' if v is None else f'{v:.4f}' for v in values)+' |')
        lines += ['','所有指标的 physics-seed bootstrap 95% CI 见 metrics.json；改善 CI 见 bootstrap.json。','']
        for name in ['B2','B2matched','B3',frozen['selection'][regime]['strongest_generic']]:
            b=boot[regime]['B4_over_'+name]
            lines.append(f"- B4 vs {name}: Brier 改善 {b['brier_improvement']}；pair 改善 {b['pair_improvement']}。")
    lines += ['','## Gate','','| Gate | 通过 |','|---|---|']+[f'| {k} | {v} |' for k,v in gates.items()]
    lines += ['','B2/B4 outcome encoder、GRU、outcome head 完全相同。B2 分配同一个 auxiliary head 但不更新；costs.json 同时报告分配参数、实际获得梯度的参数、推理参数、FLOPs 近似和同步 GPU latency。',
              'B0 是当前完整 58 通道输入；无 RGB。触觉仍是双指尖几何缩进代理，并非标定的真实触觉图像。',
              '训练/选模未使用旧 test；新 test 在 FROZEN.json 和 PROBES_FROZEN.json 之后生成。真值变量只在模型冻结后的 post-hoc probe / 分层分析使用。',
              '预测 head 的多时域未来命令条件只进入辅助分支，不进入 outcome encoder。lambda 只在既有 validation 选择。',
              '置信区间以 physics seed 为簇，保留簇内 episode/pair 相关性；不包含训练数据重新采样的不确定性。','']
    (run/'RESULTS.md').write_text('\n'.join(lines))
    diag=['# 方法诊断','',f'结论：{"offline positive; closed-loop remains untested" if go else "offline method gate failed"}。','',
          f"严格 lambda=0 对照：{strict}",'',
          f"表征 outcome probe 改善：{pr}",'',
          f"固定摩擦：Brier {gru['fixed_friction_brier']}；pair {gru['fixed_friction_pair']}。",'',
          f"exact B4 vs B2 的 Brier 改善：{exact_delta}；三个探测开始时间分层改善：{timing}。",'',
          'friction/object pose/object velocity/true force/contact count 的线性可解码性在 representation_analysis.json 完整报告；更易解码摩擦不能等同于学得通用 belief。',
          'Temporal shuffle 保留当前帧，只破坏历史顺序，使用冻结的 predictor/probe；其退化可能包括分布外扰动，不能单独证明物理 state 语义。',
          '所有 anchor 均为 2.6s、未来外载方案固定；按探测时间分层并未消除所有固定时序 shortcut。未显式输入 tick、seed 或 variant。',
          'exact/finite 使用相同 episodes 和独立训练的模型；有限精度中的收益可能包含降噪。Gate4 要求 exact 的 Brier 改善同号。',
          '固定摩擦子集检查摩擦 bin shortcut；BA 和类先验基线检查标签不平衡；这些有限检查不能排除所有混杂。',
          '三次训练初始化并非三个独立物理 test。模型容量和训练步数公平不等于训练 FLOPs 相同；辅助目标确实增加训练成本。','']
    (run/'DIAGNOSIS.md').write_text('\n'.join(diag))
    if not go:
        if not both(strict):
            reason='generic recurrent history is sufficient in this experiment; predictive-state training has not shown an independent benefit.'
        else:
            reason='Predictive training has limited offline signals, but the preregistered method gate is not satisfied; this benchmark has not established the full method claim.'
        (run/'NEGATIVE_RESULT.md').write_text('# ContactBelief-v0 负结果 / 未通过项\n\n'+reason+'\n\n失败项：'+', '.join(k for k,v in gates.items() if not v)+'。\n\n未继续扩大模型、增加层数或搜索更多 lambda；未进入 learned closed-loop。保留全部候选、日志、checkpoint、正负结果。\n\n该判断限定于当前数据、模型族、训练预算和受控任务；不证明 predictive objectives 在所有任务中无用。\n')
    (run/'METHOD_CLAIM.md').write_text('Predictive latent-state training improves future contact-state inference over matched generic history encoders in the controlled grasping environment.\n' if go else '本实验未通过预注册 offline method gate，不能宣称 predictive belief state 具有独立且稳定的方法优势。具体成立的比较见 RESULTS.md。\n')
    return verdict
