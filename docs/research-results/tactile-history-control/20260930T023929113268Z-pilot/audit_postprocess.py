"""Read-only result audit; does not select pairs, train, or rerun evaluation."""
from pathlib import Path
import csv,hashlib,json
from collections import Counter
import torch
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=[r for p in (ROOT/'eval').glob('*.json') for r in read(p)]
assert len(rows)==1008
for key in set((r['condition'],r['train_seed']) for r in rows):
    group=[r for r in rows if (r['condition'],r['train_seed'])==key]
    assert len(group)==48 and sorted(r['seed'] for r in group)==list(range(940000,940048))
failures=[]
for r in rows:
    if r['success']:continue
    reason='initialization' if not r['initial_stable'] else 'drop' if r['drop'] else 'excessive_slip' if r['max_slip_m']>=.025 else 'not_stable_at_end'
    failures.append({k:r[k] for k in ['condition','train_seed','seed','max_slip_m','drop']}|{'reason':reason})
with (ROOT/'failure_cases.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['condition','train_seed','seed','max_slip_m','drop','reason']);writer.writeheader();writer.writerows(failures)
bank=torch.load(ROOT/'ambiguity/bank.pt',weights_only=False);pairs=read(ROOT/'ambiguity/pairs.json')
branches=[r for p in (ROOT/'ambiguity').glob('M*.json') for r in read(p)]
assert len(branches)==126
pre_failed=[i for p in pairs for i in [p['left'],p['right']] if bank[i]['prefix_metrics']['max_slip_m']>=.025]
table=['|seed|T/q RMS|teacher gap|分支前最大位移 mm（左/右）|M0成功%|M1成功%|M2成功%|','|---|---|---|---|---|---|---|']
for p in pairs:
    prior=[bank[p[k]]['prefix_metrics']['max_slip_m']*1000 for k in ['left','right']]
    success=[]
    for m in ['M0','M1','M2']:
        group=[r for r in branches if r['condition']==m and r['seed']==p['seed']]
        success.append(100*sum(r['success'] for r in group)/len(group))
    table.append(f"|{p['seed']}|{p['rms']:.3f}|{p['teacher_gap']:.3f}|{prior[0]:.2f}/{prior[1]:.2f}|"+'|'.join(f'{s:.1f}' for s in success)+'|')
counts=Counter((r['condition'],r['reason']) for r in failures)
text='''# 冻结结果的补充只读审计

本文件生成于评估完成后；不改训练、checkpoint、匹配规则、原始指标或原始pair。可执行本目录 `audit_postprocess.py` 重建。

## 覆盖范围与失败

主评估1008个rollout，21组condition/initialization，每组完整覆盖同48个seed；近观测分支126个rollout（7对×2状态×3方法×3初始化）。原始主评估失败逐条见 failure_cases.csv；统计如下。

'''+ '\n'.join(f'- {k[0]} / {k[1]}: {v}' for k,v in sorted(counts.items()))+'\n\n## 匹配子集的重要限制\n\n'+'\n'.join(table)+f'''\n
7对中有 {len(pre_failed)} 个分支起点在此前已超过25mm主成功阈值（未达到40mm drop阈值），即使分支控制恢复接触，该episode也不能再满足既定成功标准。这是匹配诊断构造的缺陷；保留该样本，没有看过结果后剔除。该子集既少于预定10对，也不能作为history有独立控制价值的确认性证据。

M2在普通validation上的teacher动作MSE较小，但匹配子集的初始teacher动作误差较大；这与构造子集引入不同先前行为/状态分布相容，不能据此断言具体原因。当前T/q匹配只是带容差近似，不是严格观测混叠证明；teacher动作差也不是最优动作差证明。

M2对action shuffle的成功率差为+7.64pp [1.39,15.28]，显示策略对正确action序列敏感；shuffle也是分布外输入，因此该结果不能替代M2与M1的主比较。M2−M1仅+1.39pp [-4.17,8.33]，未达到预先门槛。NO_SIGNAL针对独立闭环收益假设，不表示模型完全没有使用历史。

结论保持NO_SIGNAL / REVISE。下一步应先修订可恢复匹配状态的开发生成规则，并检查明确安全力预算下简单控制是否仍然解决任务；使用新的development seeds，不回头调整本test或扩大模型。
'''
(ROOT/'POSTHOC_AUDIT.md').write_text(text)
for name in ['RESULTS.md','REVIEW.md','NEGATIVE_RESULT.md']:
    path=ROOT/name;body=path.read_text();link='\n\n补充只读审计：[POSTHOC_AUDIT.md](POSTHOC_AUDIT.md) 包含全部失败分类、逐pair比较，以及一个分支前已超过成功位移阈值的样本限制；未删样本或修改结论。\n'
    if 'POSTHOC_AUDIT.md' not in body:path.write_text(body+link)
freeze=read(ROOT/'FROZEN.json');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/p)==h for p,h in freeze['checkpoints'].items())
assert sha(ROOT/'normalizer.npz')==freeze['normalizer_sha256']
(ROOT/'final-audit.json').write_text(json.dumps(dict(main_rollouts=len(rows),ambiguity_rollouts=len(branches),matched_pairs=len(pairs),already_excessive_slip_at_branch=len(pre_failed),all_condition_seed_grids_complete=True,checkpoint_hashes_unchanged=True),indent=2))
print('Audit complete:',len(rows),'main rollouts,',len(branches),'ambiguity rollouts;',len(pre_failed),'pre-failed branch')
