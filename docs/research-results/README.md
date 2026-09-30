# 历史受控机器人实验结果

> 本目录保存 `robotic_infra` 在 2026-09-30 scope reset 之前形成的科学证据、负结果和审计材料。它们继续用于 provenance 与复现，但不再定义该仓库的 active research roadmap。新的 research question 应进入独立 research repository。参见 [Research / Infrastructure Boundary](../RESEARCH_BOUNDARY.md)。

这里保存已经完成的实验代码对应的正式证据摘要和可核验结果。原始 `runs/` 仍由 `.gitignore` 排除；本目录是选定证据：早期实验为原始run的逐字节副本，新Pilot则在原输出目录中按清单发布文件，不改写原实验。

| 阶段 | 问题 | 实际结论 | 入口 |
|---|---|---|---|
| I001 | 历史是否比当前触觉更有信息？ | 历史优于带噪单帧力代理，但未超过完整高精度当前观测 | [报告](../../experiments/iros2027/i001/RESULTS.md) |
| I001-v2 | 受控观测模型下，历史能否改善接触结果预测？ | 部分支持：finite 有收益；动作历史独立贡献未成立，exact 当前基线更好 | [正式结果](i001-v2/20260929-171412-151811/RESULTS.md)、[审核结论](i001-v2/20260929-171412-151811/REVIEWED_VERDICT.json)、[限定主张](i001-v2/20260929-171412-151811/FIRST_LAYER_CLAIM.md) |
| ContactBelief-v0 | predictive latent objective 是否比普通 history encoding 更有效？ | **NO-GO**：未稳定优于同架构 GRU，未进入 learned closed-loop | [正式结果](contactbelief/20260929-185730-004499/RESULTS.md)、[负结果](contactbelief/20260929-185730-004499/NEGATIVE_RESULT.md)、[复核诊断](contactbelief/20260929-185730-004499/REVIEW.md) |
| Tactile history control | history能否改善disturbed-grasp恢复？ | **NO_SIGNAL / REVISE**；constant-max 100%，两项history主差值区间跨零 | [记录](tactile-history-control/README.md) |
| Active tactile insertion | past action在完整T/q history之上是否有收益？ | **WEAK_SIGNAL / NO-GO**；M1−M0 +6.94 pp，M2−M1 −2.08 pp；q可重建动作方向 | [记录](active-tactile-insertion/README.md) |

当前研究决策见[执行时触觉反馈与在线动作修正](../research/EXECUTION_TIME_TACTILE_REFINEMENT.md)：复用插入工装，转向fresh feedback及prediction-conditioned correction；新方向尚未运行。旧报告里的后续建议保留为历史记录。

## 数据与冻结边界

I001-v2 使用 5600 episodes / 280 physics seeds。ContactBelief 复用其中 train/validation，在 architecture、超参数、normalization、模型和只读 probe 冻结后，重新生成 120 个全新 seed / 2400 episodes 作为 FINAL TEST。既有 I001-v2 test 没有作为 ContactBelief 的 final test。

ContactBelief 的主要对照由 validation 预先选定为 Stack-MLP；同架构 GRU 和 Temporal Transformer 均保留，不根据 final test 重新选择主比较。配对与 episode 的不确定性按 physics seed 聚类。

上述I001/ContactBelief结论限定于受控抓取预测与观测模型，不含闭环控制改进。新两轮Pilot另有闭环结果，仍限定于各自仿真任务；均无RGB/真实触觉图像、真实机器人、通用manipulation或SOTA主张。

## GitHub 中包含什么

- 实验代码、协议、配置、测试和复现命令。
- 正式报告、完整 baseline 指标、seed bootstrap、预测 CSV、matched-pair CSV、图表。
- 冻结记录、seed 清单、运行/数据哈希、回归与完整性校验。
- ContactBelief 全部 72 个候选的逐 epoch 曲线与完成记录，包括未选中模型。

每个结果目录的 `EXPORT_MANIFEST.json` 记录实际公开文件的 SHA-256。其他 manifest/FROZEN/COMPLETE 是原始运行记录，包含未上传文件的引用，不能将它们误读成完整运行归档清单。

约 6 GB 原始运行目录及 checkpoint、optimizer、latent 二进制仍保留在原工作区，未纳入 Git。仅克隆本仓库可以审阅指标并重新聚合已发布预测，但**不能凭报告副本直接恢复训练或重放冻结模型**；这需要对应的完整原始运行文件。

报告中的本机绝对路径是原始运行 provenance，保持原样。`REPRODUCE.txt` 描述新实验与本地恢复；ContactBelief 的 `EXACT_FINAL_REPLAY.txt` / helper 需要放回原完整 run 目录才可使用。单 episode 物理重放已经验证，未额外执行完整 2400-episode 重放。

代码入口：[I001](../../experiments/iros2027/i001/README.md)、[I001-v2](../../experiments/iros2027/i001_v2/README.md)、[ContactBelief](../../experiments/iros2027/contactbelief/README.md)。

2026-09-30新增两轮Pilot的公开范围、576个证据文件校验、原始源码快照与恢复限制，见[发布说明](PILOT_RELEASE_20260930.md)。新run的原始二进制和权重仍留本地。
