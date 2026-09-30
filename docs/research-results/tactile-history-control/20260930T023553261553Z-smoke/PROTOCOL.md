# 冻结前 Pilot 协议

## 问题与假设

问题：200 Hz disturbed grasp recovery 中，300 ms tactile/proprioception 历史是否改善闭环恢复；past issued action 是否在此基础上有独立收益？

主要比较：M1−M0（A），M2−M1（B）的配对恢复成功率差。强替代解释：当前压入量/关节位置已充分；持续加紧就能解决任务；action 通道只引入 teacher 的行为相关性。故额外报告 nominal、privileged heuristic teacher、constant maximum closure。

失败条件：对应差值小于 5 个百分点，或配对 95% bootstrap CI 跨零，或三个训练种子方向不一致，均不认定支持该假设。B 还必须同时满足 M2 优于 action shuffle（同样 5pp、CI 下界大于零）及近当前观测匹配子集 M2 优于 M1（至少 10 个独立扰动种子对，差值至少 5pp、CI 下界大于零）。任何门槛缺证据即不支持。小样本无差异不等于等效性证明。

结论标签：两假设通过且简单基线未解释收益为 PROMISING；部分证据为 WEAK_SIGNAL；均未通过为 NO_SIGNAL；泄漏、初始化失效或任务天花板等使机制无法辨识时可标 CONFOUNDED。负结果停止扩模，决策 REVISE 或 STOP。此任务为探索性 Pilot，不是最终独立论文测试。

## 物理与观测

复用既有 FR3 MuJoCo grasp/lift 和双指内侧 contact geometry taxel renderer。1 kHz 物理、200 Hz observation/action；从 2.4 s 稳定抓持后闭环控制至 3.6 s。扰动水平，方向围绕世界 ±y（切向）各 ±30°，幅度、起始、持续时间、质量、摩擦、初始偏置随机。所有范围见同目录 config.yaml。训练前仅在 development seeds 上检查物理范围，记录调试；正式 test 不参与修改。

T 为两个 4×4 taxel 压入量图；其值由接触几何生成，带 1 μm 噪声和 5 μm 量化，不是 RGB DIGIT、真实触觉或标定力传感器。物理位置为左右 parallel jaw 内侧。q 顺序为 fr3_joint1..7, fr3_finger_joint1..2。输入 42 维 = 32 tactile + 9 q + 1 上一已下发归一化 residual。禁止 dq/ddq/object pose/friction/contact label 进入 student。接触几何的 simulator 来源是明确的 observation-model 假设。

动作 u∈[0,1]，delta_g=−0.012u m；每次下发绝对目标 width=0.032+delta_g。输入 a_(t−1) 是最近一次裁剪后已下发的 u，可精确换算为 width；不是当前待预测动作、teacher label 或执行器实测位移。没有人为 actuator delay，保留原生动力学。无 7DoF 学习控制；机械臂固定已知抓取姿态。

## 训练公平性与因果顺序

t 时刻先读 T/q 和上一条命令，teacher 生成 label，再 student/behavior 下发动作，推进 5 个物理步。窗口 61 个 token，每时刻一个多模态 token；不拼 semantic token。M0 把当前 T/q 重复 61 次；M1 保留 T/q 窗口；两者 action 通道置零。M2 保留全部。相同输入矩阵、正弦位置编码、causal attention、2层128维4头、256 FFN、MLP action head，参数总量严格相同（包括两个零输入通道对应的权重）；M0重复方法允许其使用固定位置，但没有历史内容。

所有方法共享同一批 72 个 behavior episodes，teacher MSE target、AdamW、600 steps、每步 batch64、训练种子17/29/43，同 seed 的初始化与每步样本索引一致。不搜索学习率、不做辅助loss。Teacher privileged displacement/contact slip speed 只生成 teacher action；不以 hidden state 筛选训练样本、选 checkpoint 或调 test。behavior 大部分跟随 teacher，混合短时随机收紧以覆盖 action history；这会有 imitation covariate shift。

只用训练可见输入拟合标准化。16 个 validation episodes 仅记录 action MSE；最终固定第600步 checkpoint，不使用 validation/test 成功率选择模型。episode seed 分区不交叉；三个训练初始化共享训练集，不把它们当三套独立采集数据。

## 评估及诊断

48 个未见扰动种子配对用于全部条件，保留所有 episode（包括初始化失败），无筛选。成功：初始稳定，未曾掉落，最大相对 TCP 位移<25mm，末端连续100ms双侧接触且切向速度<5mm/s。drop：位移>40mm或物体高度<0.46m。报告 max/final slip、峰值单指法向力（privileged evaluation）、平均绝对 corrective target、恢复时延。时延从扰动结束计至首次满足100ms稳定，失败右删失到episode末；主表为有上限时延，不把其解读成全部样本真实恢复时间。

统计：每训练种子单独成功率，跨训练种子均值/样本SD；同时对训练种子及配对 disturbance seeds 重采样的95% bootstrap CI，不把 3×48 episodes 当作144个完全独立样本。单一 teacher/nominal/constant 行按扰动种子 bootstrap。此规模对小效应统计功效有限。

Temporal shuffle：推理时对 past60 tokens 全模态同序随机排列，当前 token 保留。对 M1、M2 分别做。Action shuffle：只打乱 M2 全61个已知动作，不改变T/q；保持动作值多重集。每episode/time固定 RNG，各条件均进行真实闭环 rollout。分布外扰动性能下降只是使用通道的证据，不能单独证明因果价值。

Ambiguity：独立 24 个 disturbance seeds，每个4个不同 preceding action variants；从相同初始化出发，在 onset+40ms 截断，末25ms相同动作。只比较同seed不同variant，当前41维T/q距离按 tactile25μm、arm0.1mrad、finger0.1mm缩放，要求 RMS≤1、L∞≤3，teacher action差≥0.15；按最小观测RMS选每seed最多一对，不按模型表现挑选。该teacher是启发式，不能称最优动作证明。匹配阈值在test前冻结。完整MuJoCo integration state及61步历史分支恢复，M0/M1/M2真实闭环到3.6s，报告成对成功率和初始teacher action误差。若没有或不足10对，明确该诊断未建立，B不通过；不得看test后放宽匹配。

## 审计、恢复与停止

独立 timestamp run，保存 config/protocol/source hashes、Git diff/commit、hardware、依赖版本、dataset、normalizer、optimizer/latest/final checkpoint、metrics、曲线、pair bank。每100步保存，resume不可修改配置/实验源码。冻结后禁止训练。异常记failure.json和traceback；阶段/条件原子完成才跳过。长运行按配置有限循环，不自动扩大规模。结果冻结后修报告不改模型和test。
