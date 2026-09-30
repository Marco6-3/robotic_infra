# 负结果与不确定性

A支持=False，B支持=False；NO-GO。

- A_M1_minus_M0: 0.00 [0.00, 0.00] pp；各training seed差值[0.0]
- B_M2_minus_M1: 0.00 [0.00, 0.00] pp；各training seed差值[0.0]
- M2_minus_action_shuffle: 0.00 [0.00, 0.00] pp；各training seed差值[0.0]
- M1_minus_temporal_shuffle: 0.00 [0.00, 0.00] pp；各training seed差值[0.0]
- M2_minus_temporal_shuffle: 0.00 [0.00, 0.00] pp；各training seed差值[0.0]

构造 2 对镜像孔位/动作历史，但仅 0 对满足完整61步T/q匹配门槛（最低10对）。未删q或重排触觉channel来制造匹配。
- 所有镜像样本 M0: 75.00 [50.00, 100.00]%（描述性，不能代替匹配子集）
- 所有镜像样本 M2: 75.00 [50.00, 100.00]%（描述性，不能代替匹配子集）
- 所有镜像样本 M1: 75.00 [50.00, 100.00]%（描述性，不能代替匹配子集）
合格子集M2−M1：未建立 pp。


即使模型对action shuffle敏感，也可能是输入分布扰动，不能替代M2>M1。随机probe不是独立信息的数学保证，因为M1包含q；三步q回归的R²=0.706也提示检查这一解释。若模型低成功，还应区分teacher不可观测、BC分布偏移、采样覆盖和任务物理问题，不能径直扩模。

下一步最小实验：保留full q，用新development seeds检验物理可恢复、全T/q匹配的相反正确动作对；不足则先改假设。未根据本test修改超力阈值、孔形、模型或匹配规则。
