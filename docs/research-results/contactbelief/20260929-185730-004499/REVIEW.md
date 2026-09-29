# ContactBelief-v0 人工复核

最终判定：**NO-GO**。保留全部候选与正负结果；不修改冻结模型、超参数、协议或最终测试集，不进入 learned closed-loop。

## 核心方法比较

有限精度下，B4 对完全相同架构的 B2（同时也是同学习率 B2matched）：

- Brier 改善 −0.000427，95% CI [−0.002763, 0.001892]。
- matched-pair accuracy 改善 +0.908 个百分点，95% CI [−0.252, 2.140]。
- 三个初始化的 Brier 改善分别为 +0.001653、−0.000784、−0.002472，没有稳定同号。

validation 预先选出的 strongest generic baseline 是 Stack-MLP，仍作为主比较，不按 final test 重新挑选。Final test 中 Transformer 得分最好，这是额外事实：B4 比 Transformer 的 Brier 差 0.010692，95% CI [0.006305,0.015047]；pair accuracy 差 6.137 个百分点，95% CI [3.744,8.546]。

exact 下 B4 对 B2 有很小的 Brier 改善（0.002320，CI 下界仅 0.0000024），但 pair 改善不成立，也未超过 Stack/Transformer。这不改变 finite primary gate 的失败。

## 表征的正信号与边界

B4 latent outcome probe 的 Brier 0.111973，比当前 embedding 的 0.147293 好；这一差值 CI 为 [0.028445,0.042795]，因此预注册 Gate3 的宽口径通过。

但是 B2 latent 的 probe Brier 已达 0.113979；B4 对它的改善仅 0.002006，CI [−0.000790,0.004630] 跨零。它支持历史表示相对当前帧有信息，**没有建立 predictive objective 相对普通 GRU 的独立表征优势**。

两种 GRU 都能强烈解码摩擦。finite friction 的 train-standardized MSE：B2=0.006094，B4=0.007345，B4 并未更好。固定摩擦子集上的 Brier/pair 改善 CI 也跨零，不能把普通 hidden-parameter identification 写成 ContactBelief 特有能力。

Temporal shuffle 明显破坏两种 GRU 的 outcome/probe 性能，但这也是 distribution shift，不独自证明某种人类语义的 physical state。

## 辅助目标是否真正学到未来预测

训练中的 tactile/q/dq loss 均下降，且按 modality 等权；并非 auxiliary loss 没有反向传播。只读最终 head 审计见 auxiliary_prediction_audit.json。

finite 的全部 causal prefixes 上，+100ms tactile 归一化 MSE 由 persistence 的 1.5072 降到 0.6798。但是在真正进行 outcome 判别的 anchor 上，+100ms tactile MSE 为 5.1812，反而略高于 persistence 的 5.1191；dq 也没有改善。两组指标使用相同 train-derived normalization，但评估时间点不同。

协议中的辅助 loss 对 61 个历史 prefix × 3 个 horizon 等权；只有 35/183 个目标时间超过加载 anchor，其中恰好从 anchor 出发的目标只有 3/183。**可能的解释**是目标改善了多数历史内动力学，却没有有效对齐决定未来 slip 的加载阶段。此项是 post-hoc 诊断，不是已证明的因果归因，也不据此更改 loss、再选 checkpoint 或继续训练。

## 距离与训练预算局限

当前 embedding 有 11/192 个维度在 train 近似常量；以标准差下界归一化后，极少数 final 激活造成很大的距离异常值。相同 outcome pair 的距离均值为 5.558，中位数仅 0.654，最大值超过 12000。因此不能拿不同表示的平均欧氏距离直接宣称“更真实的 latent state”；主要依赖冻结线性 probe 和最终预测指标。详见 embedding_distance_audit.json。

所有单个候选均训练 60 epochs / 600 optimizer steps，且 B2/B4 推理参数均为 30,849；B4 辅助 head 额外训练 4,274 参数。最佳 checkpoint 的 epoch 可不同，因为各自只按 validation 选择。部分最佳模型接近 epoch 60，因此本结论限定在该预注册预算；没有证明所有更长训练的 predictive objectives 无效。

B4 每个 regime 搜索 2 learning rates × 2 positive lambdas × 3 initializations=12 次；其余每种方法各 6 次。B4 的总搜索与训练计算并不更少；同架构推理参数相同不等于训练 FLOPs 相同。主要预测为三模型集成，costs.json 的 latency 明确为单模型、batch=1，另给出集成 FLOPs 倍数，没有把单模型 latency 冒充集成 latency。

## 完整性与复现

全新 final seeds 共 120 个，2400 episodes；804 positives / 1596 negatives；2754 opposite pairs 覆盖 112 seeds。模型冻结、probe 冻结、seed 预约、episode 生成的时间顺序已审计。

68 项 regression tests 通过。I001-v2 的 5722 个受保护文件哈希全部保持不变；单 episode 确定性重放通过；GPU 合成中断恢复的模型、optimizer、loss/validation 日志与连续训练 bitwise 相同。

REPRODUCE.txt 启动的是新实验（会在新模型冻结后重新抽取新 test seeds）；EXACT_FINAL_REPLAY.txt 使用保存的 seeds/权重，在另一个目录重新生成同一 final set。完整 2400-episode replay helper 已提供，但本轮只实际执行单 episode 物理重放，未额外重复全量数据集。

冻结科学结果与 gate 不变；本文件及辅助 head/距离审计是只读 post-hoc 复核。
