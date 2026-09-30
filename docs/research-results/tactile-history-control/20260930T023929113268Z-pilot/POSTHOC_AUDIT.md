# 冻结结果的补充只读审计

本文件生成于评估完成后；不改训练、checkpoint、匹配规则、原始指标或原始pair。可执行本目录 `audit_postprocess.py` 重建。

## 覆盖范围与失败

主评估1008个rollout，21组condition/initialization，每组完整覆盖同48个seed；近观测分支126个rollout（7对×2状态×3方法×3初始化）。原始主评估失败逐条见 failure_cases.csv；统计如下。

- M0 / drop: 13
- M1 / drop: 10
- M1 / excessive_slip: 2
- M1_temporal_shuffle / drop: 17
- M2 / drop: 9
- M2 / excessive_slip: 1
- M2_action_shuffle / drop: 18
- M2_action_shuffle / excessive_slip: 1
- M2_action_shuffle / not_stable_at_end: 2
- M2_temporal_shuffle / drop: 17
- M2_temporal_shuffle / excessive_slip: 1
- nominal / drop: 38

## 匹配子集的重要限制

|seed|T/q RMS|teacher gap|分支前最大位移 mm（左/右）|M0成功%|M1成功%|M2成功%|
|---|---|---|---|---|---|---|
|950003|0.486|0.245|0.64/0.30|100.0|100.0|100.0|
|950004|0.613|0.886|6.33/0.37|100.0|100.0|100.0|
|950009|0.187|0.831|13.91/0.26|83.3|50.0|66.7|
|950010|0.413|0.770|0.51/27.76|50.0|50.0|50.0|
|950014|0.394|0.683|24.98/0.39|50.0|50.0|50.0|
|950015|0.582|0.193|0.44/0.23|100.0|100.0|100.0|
|950019|0.782|0.429|0.76/23.67|50.0|50.0|50.0|

7对中有 1 个分支起点在此前已超过25mm主成功阈值（未达到40mm drop阈值），即使分支控制恢复接触，该episode也不能再满足既定成功标准。这是匹配诊断构造的缺陷；保留该样本，没有看过结果后剔除。该子集既少于预定10对，也不能作为history有独立控制价值的确认性证据。

M2在普通validation上的teacher动作MSE较小，但匹配子集的初始teacher动作误差较大；这与构造子集引入不同先前行为/状态分布相容，不能据此断言具体原因。当前T/q匹配只是带容差近似，不是严格观测混叠证明；teacher动作差也不是最优动作差证明。

M2对action shuffle的成功率差为+7.64pp [1.39,15.28]，显示策略对正确action序列敏感；shuffle也是分布外输入，因此该结果不能替代M2与M1的主比较。M2−M1仅+1.39pp [-4.17,8.33]，未达到预先门槛。NO_SIGNAL针对独立闭环收益假设，不表示模型完全没有使用历史。

结论保持NO_SIGNAL / REVISE。下一步应先修订可恢复匹配状态的开发生成规则，并检查明确安全力预算下简单控制是否仍然解决任务；使用新的development seeds，不回头调整本test或扩大模型。
