# ContactBelief-v0 预注册协议

本文件在正式训练、新 FINAL TEST 之前固定。I001-v2 的协议、数据、报告与结论只读保护；本阶段检验方法，不重复宣称 history 必要性。

## 假设与主要估计量

H_method：加入预测未来可观测交互状态的目标，能比同架构的 outcome-only GRU 以及强通用历史编码器更好地预测 100 ms 内 slip。

主 regime 为既有 finite observation，exact 为预定精度敏感性分析。主比较是 B4 对 validation Brier 最优的 B1/B2/B3，必须同时突出 B4 vs B2 及 B4 vs B2matched（与 B4 相同 learning rate 的 lambda=0 GRU）。最强基线只能由 validation 决定，不能由 final test 重新挑选。

主估计量为三个独立训练初始化的预测概率均值；所有方法都使用三个初始化。各初始化的性能和方法差值单独报告。主要指标为 Brier 降低与 matched opposite-outcome pair discrimination 提升；两者越大越好。

## 数据与禁止泄漏

- 复用 I001-v2 正式运行的 2400 train / 800 validation episodes，原始 physics seed 分割不变。既有 test 不进入本实验训练、选择、最终评分；仅校验其文件哈希以证明未改动。
- 输入 x 为 tactile[32]、q[9]、dq[9]、command[8] 共 58 通道；过去 300 ms，200 Hz，含当前帧，共 61 帧。B0 特意仅使用当前帧，是当前观测对照。
- 时间截止 2600 ms。未来观测、slip 标签、摩擦、物体 pose/state、接触力不进入 outcome encoder。未来观测仅作为辅助监督目标，未来命令仅进入 dynamics head。
- train/validation 的 eligibility 沿用既有数据定义，不按方法得分或 privileged probe 再筛选样本。
- 保留原 tactile 几何缩进代理；没有 RGB，也不声称真实 tactile image。exact/finite 的既有量化和噪声设置不修改。网络计算为 float32。
- 规范化仅拟合 train：每通道历史均值/标准差，以既有 observation quantum 为标准差下界；未来目标为相对当前观测的增量，其尺度由 train 增量标准差拟合。

## 模型与预算

B0 Current-MLP；B1 Stack-MLP（61×58 flatten）；B2 GRU-History；B3 单层、4 heads、64 维 causal Transformer；B4 ContactBelief。

B2/B4 使用完全相同的 58→64 ReLU encoder、单层 GRU64、64→32→1 outcome head、初始化 seed、batch 顺序、AdamW、weight decay、梯度裁剪和 60 epoch 预算。B2 与 B4 均分配小型 dynamics head，但 B2 不对其施加 loss（不获得梯度、不用于推理）。报告 allocated / actually trained / inference 三种参数量，不把 dormant 参数冒充有效容量。

辅助 head：每个 causal prefix 的 z_t 与已知 [t,t+h) 命令的平均值、最后命令、horizon/100 拼接，通过共享的 81→32 ReLU→50 MLP 输出 50 维 tactile+q+dq 的归一化增量。h = 25/50/100 ms；不重建 action、slip 或 privileged state。所有 t 使用同一个 head；future controls 来自预先采样的开环命令计划，仅使用命令通道，不使用未来测量或额外外载真值。

L = binary cross entropy(outcome at anchor) + lambda × mean(MSE_tactile, MSE_q, MSE_dq)。三个 modality 等权，各自通道已规范化。记录 outcome 与三部分 reconstruction loss。B2 的 lambda=0；B4 的正候选仅 [0.1,1.0]。

候选 learning rate 只有 [0.001,0.0003]；三个初始化 [1701,1702,1703]。所有候选跑满 60 epochs，不提前停止；单初始化 checkpoint 按 validation Brier 选择，然后超参数按三初始化集成 validation Brier 选择。没有 architecture search、额外 lambda 搜索、最终测试后的重新选模。B2matched 复用已有同学习率的 B2 checkpoint，不额外训练。

无 AMP，CUDA，batch=256，num_workers=0，确定性 cuDNN；所有模块 optimizer steps 相等。辅助任务训练 FLOPs 额外增加，不能把推理参数相同等同于训练成本相同。完整历史推理 latency 使用 batch=1、GPU 同步、20 warmups/100 repeats，另报告近似 MAC FLOPs 和集成倍数。

## 冻结与全新 FINAL TEST

训练选择完成后写 FROZEN.json，包含 architecture/config/source/checkpoints/normalizers/选择结果哈希。此后禁止训练或修改这些对象。

随后才读取 privileged train/validation targets，拟合只读 probe。Probe 是固定 alpha=1 的线性 ridge，无任何调参；不反向传播到 encoder，不参与 checkpoint、模型或 baseline 选择。保存 PROBES_FROZEN.json 后才预约 FINAL seeds。

通过 OS entropy 在 [100000000,900000000) 中选择连续 120 个从未出现在 runs episode 文件名中的物理 seed，检查不重合；保存实际 seed 清单。每 seed 20 episodes，共 2400 episodes；与 I001-v2 使用同一环境/干预/标签定义。数量固定，不按 test 分数、结果正负或 pair 数调节。逐 episode 校验时间戳、标签和 future input 一致性，校验一次独立确定性重放。

匹配算法复用 I001-v2 只读实现：同 physics seed 下当前传感器观测的量化归一化 L∞≤1、RMS<0.5，当前命令完全相等；先只按当前观测匹配，再注释 outcome/摩擦。所有合格 episode 均评分；pair discrimination 仅在 opposite outcomes 上计算，平分记 0.5。

## 指标与表征

Brier/AUROC/BA/F1/ECE（10 等宽 bins）和 pair discrimination；阈值固定 0.5。全部主要模型指标和方法差值采用 2000 次 physics-seed cluster bootstrap 95% percentile CI，同一个 seed 下的 episodes/pairs 一起抽样。不把共享 episode 的 pairs 当作独立数据；CI 条件于固定训练集，另外报告三次初始化敏感性。

只读 representation probe 比较：B4 z、B2 z、B4 当前 e、原始当前 x。三个初始化的 embedding 直接拼接，不学习额外 aggregator；B4 z/B2 z/current e 均为 192 维。线性 probe 在 train 拟合，在 validation 只记录，在 FINAL 只评分。

Probe targets：outcome，三个 horizon 的未来可观测 tactile/q/dq，以及仅 post-freeze 读取的摩擦、物体位置/线速度、双指真实法向/切向接触力和接触数量。报告每类 train-standardized MSE、outcome Brier；privileged probe 能力不是 human-semantic state 的证明。

另外保留当前帧、随机打乱过去 60 帧，通过同一个冻结 encoder/probe 评估退化；不同 outcome 的 matched pairs 在 train-standardized latent 空间中的 RMS 距离与相同 outcome matched pairs 对照。空间距离本身不能替代可解码性或方法性能。

## 预定 GO / NO-GO

全部满足才继续实现 learned closed-loop：

1. Gate1：B4 相对 validation 最佳 B2 和 B2matched，Brier 与 pair improvement 的 CI 下界均 >0；严格匹配对照的三次初始化 Brier 改善全部 >0，至少两次 pair 改善 >0。
2. Gate2：相对 validation 最强 B1/B2/B3 及 B3 Transformer，两个主要改善 CI 下界均 >0。
3. Gate3：B4 latent 的固定线性 outcome probe 相对 B2 latent 或 current embedding 的 Brier 改善 CI 下界 >0。
4. Gate4：固定摩擦子集两个主要改善 CI 下界 >0；按 probe start [2325,2360)、[2360,2395)、[2395,2431) ms 的至少两层 Brier 改善同号；exact B4 vs B2 Brier 改善 >0；BA 不下降；temporal shuffle 的 B4 Brier 退化 CI 下界 >0。新 seed、输入排除 ID、类先验 Brier 也审计并报告。
5. 数据充分性：≥100 seeds、≥1000 opposite pairs，覆盖≥80 seeds；不足时报告 benchmark construction limitation，不按结果追加采样。

Gate4 是有限检查，不能排除所有 shortcut：尤其固定 anchor/未来外载仍是任务局限。不得把它写成通用证明。

任何 gate 未通过：生成 NEGATIVE_RESULT.md，保留全部失败配置/checkpoint/日志，停止复杂化或新增任务，不实施 learned closed-loop。只有所有 gate 通过才单独设计/实现/冻结控制实验并验证控制结果。Offline positive 不等于 manipulation improvement。

## 可恢复与证据保护

每 epoch 原子保存 optimizer/model/curve/latest；episode 和正式产物独立保存到 runs/contactbelief。独占 run 文件锁；恢复校验源代码和冻结文件哈希。I001-v2 原代码和正式 run 的所有文件（不含 __pycache__/.lock）开始/结束哈希比对。已完成 run 恢复只验证并返回，不覆盖结果。
