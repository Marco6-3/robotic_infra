# 未通过的假设与下一步

A通过=False；B通过=False。

- A_M1_minus_M0: 0.00 [0.00, 0.00] pp；各训练种子差值 [0.0]
- B_M2_minus_M1: 0.00 [0.00, 0.00] pp；各训练种子差值 [0.0]
- M2_minus_action_shuffle: 0.00 [0.00, 0.00] pp；各训练种子差值 [0.0]
- M1_minus_temporal_shuffle: 0.00 [0.00, 0.00] pp；各训练种子差值 [0.0]
- M2_minus_temporal_shuffle: 0.00 [0.00, 0.00] pp；各训练种子差值 [0.0]

按冻结规则找到 0 对（最少要求 10），来自独立的 ambiguity seeds；未在评估后放宽阈值。

M2−M1 配对成功率差：未建立 pp。 样本不足，Hypothesis B 的 ambiguity 条件未建立。

可能原因：当前观测足以反映本任务的接触变化；持续加紧可解决大部分扰动；teacher不显式依赖过去action；关节位置已编码部分动作后果；600步有限训练与BC分布偏移。上述是待检验解释，不把负结果归咎为确定的模型不足。

下一步最小修改：用新的development seeds检验加入明确物理合理接触力上限后，是否能找到T/q近似但合理安全动作不同的可恢复状态对。先比较teacher、恒定收紧与current-only可行性，不训练更大模型。本次不改test、不调参追求M2获胜。
