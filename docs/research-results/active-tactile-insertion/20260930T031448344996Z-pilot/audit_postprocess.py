"""Inspect immutable run results; no additional policy evaluation or selection."""
from pathlib import Path
import json,hashlib
import numpy as np
from experiments.active_tactile_insertion.run import freeze
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
assert freeze(ROOT)==read(ROOT/'FROZEN.json')
rows=[r for p in (ROOT/'eval').glob('*.json') for r in read(p)];branches=[r for p in (ROOT/'diagnostics').glob('M*.json') for r in read(p)]
assert len(rows)==1104 and len(branches)==432
for r in rows:
    if r['success']:
        assert r['peak_force_n']<=8 and not r['force_failure'] and r['insertion_depth_m']>=.008
        if r['lateral_error_m']>=.0004:
            assert any(a['condition']==r['condition'] and a['train_seed']==r['train_seed'] and a['seed']==r['seed'] and a['verified'] for a in read(ROOT/'event-audit.json'))
    assert not (r['success'] and r['force_failure'])
    z=np.load(ROOT/'traces'/f"{r['condition']}-{r['train_seed']}"/f"{r['seed']}.npz")
    if len(z['issued']):np.testing.assert_allclose(z['observations'][61:,-2:],z['issued'],rtol=1e-6,atol=1e-7)
pairs=read(ROOT/'diagnostics/pairs.json');summary=read(ROOT/'summary.json')
report=dict(main_rollouts=len(rows),constructed_pair_rollouts=len(branches),no_unsafe_successes=True,all_command_channels_aligned=True,source_data_model_hashes_match=True,
            matched_pairs=sum(p['matched'] for p in pairs),tactile_rms_range=[min(p['tactile_rms'] for p in pairs),max(p['tactile_rms'] for p in pairs)],q_rms_range=[min(p['q_rms'] for p in pairs),max(p['q_rms'] for p in pairs)])
(ROOT/'final-audit.json').write_text(json.dumps(report,indent=2))
text='''# 冻结结果的补充解释

此文件仅审计已完成结果，不更改任务、模型、种子、匹配标准或结论。由同目录audit_postprocess.py重建。

- 已核对1104次主rollout、432次镜像状态分支。所有success都满足8N接触力界限；不存在force_failure同时标success。成功定义是1ms子步首次达到8mm深度、0.4mm对中并保持20ms，数据保存于5ms控制步末。25个保存末状态的对中误差稍超0.4mm的边界案例，已按保存的真实命令逐1ms回放，全部核实此前成功事件满足条件；成功事件与末记录间隔0–4ms，见event-audit.json。成功事件计数未改，不能把末记录中的几何值当作精确成功事件快照。所有存储观测中的action与上一实际下发命令逐项一致。
- M1−M0为+6.94pp [0.69,14.58]，支持这个Pilot中的history内容收益；但M1 temporal shuffle仍98.61%，相对原顺序仅差1.39pp [0,5.56]。因此不能声称模型必须利用正确时间顺序，也不能据此认定它已学习contact Jacobian。历史样本覆盖或近似无序的T/q集合仍是强解释。
- M2−M1为−2.08pp [−8.33,0]；M2动作shuffle前后平均成功率完全相同（各seed差异并非逐episode相同）。当前数据不支持action history额外价值，不能用M2较低teacher动作MSE替代这个结论。
- M1已有q_history，实际运动方向可从中估计。仅3步q增量的线性回归即可在validation重建上一下发动作R²约0.755、非微小分量方向准确率89.60%。这不是信息完全等价证明，但明确否定“没有a就不知道刚才往哪里动”的默认前提。
- 24个主动构造镜像pair没有一个满足完整T/q-history匹配；不能丢q、交换左右触觉channel或扩大容差来人为制造action优势。
- NO-GO针对“past action在T/q历史之上带来独立收益”的核心假设。它不表示盲插入不能学习：M1已在这48个IID任务seed上144/144安全成功，但这些共享task seeds，不能当144个独立环境样本；也不能把此天花板外推到其他孔型/真机。

最小下一步应优先区分history覆盖与时间顺序的作用，或在真实执行器未跟踪指令的情形验证全T/q-history匹配是否存在。保留全部proprioception，使用新的development seeds；本轮不再训练更大网络或调整测试任务追求M2胜出。
'''
(ROOT/'POSTHOC_AUDIT.md').write_text(text)
for name in ['RESULTS.md','REVIEW.md','NEGATIVE_RESULT.md']:
    p=ROOT/name;s=p.read_text()
    if 'POSTHOC_AUDIT.md' not in s:p.write_text(s+'\n\n补充：[POSTHOC_AUDIT.md](POSTHOC_AUDIT.md) 检查安全成功/动作时间对齐，并讨论history覆盖、时间顺序与q可重建动作的替代解释；未改变指标或实验。\n')
print(json.dumps(report))
