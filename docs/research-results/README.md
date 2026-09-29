# 受控机器人实验结果

这里保存已经完成的实验代码对应的正式证据摘要和可核验结果。原始 `runs/` 仍由 `.gitignore` 排除；本目录是选定产物的逐字节副本，不改写原实验。

| 阶段 | 问题 | 实际结论 | 入口 |
|---|---|---|---|
| I001 | 历史是否比当前触觉更有信息？ | 历史优于带噪单帧力代理，但未超过完整高精度当前观测 | [报告](../../experiments/iros2027/i001/RESULTS.md) |
| I001-v2 | 受控观测模型下，历史能否改善接触结果预测？ | 部分支持：finite 有收益；动作历史独立贡献未成立，exact 当前基线更好 | [正式结果](i001-v2/20260929-171412-151811/RESULTS.md)、[审核结论](i001-v2/20260929-171412-151811/REVIEWED_VERDICT.json)、[限定主张](i001-v2/20260929-171412-151811/FIRST_LAYER_CLAIM.md) |
| ContactBelief-v0 | predictive latent objective 是否比普通 history encoding 更有效？ | **NO-GO**：未稳定优于同架构 GRU，未进入 learned closed-loop | [正式结果](contactbelief/20260929-185730-004499/RESULTS.md)、[负结果](contactbelief/20260929-185730-004499/NEGATIVE_RESULT.md)、[复核诊断](contactbelief/20260929-185730-004499/REVIEW.md) |

## 数据与冻结边界

I001-v2 使用 5600 episodes / 280 physics seeds。ContactBelief 复用其中 train/validation，在 architecture、超参数、normalization、模型和只读 probe 冻结后，重新生成 120 个全新 seed / 2400 episodes 作为 FINAL TEST。既有 I001-v2 test 没有作为 ContactBelief 的 final test。

ContactBelief 的主要对照由 validation 预先选定为 Stack-MLP；同架构 GRU 和 Temporal Transformer 均保留，不根据 final test 重新选择主比较。配对与 episode 的不确定性按 physics seed 聚类。

所有结论限定于该受控抓取环境与观测模型，没有 RGB/真实触觉图像、真实机器人、通用 manipulation、SOTA 或闭环控制改进的主张。

## GitHub 中包含什么

- 实验代码、协议、配置、测试和复现命令。
- 正式报告、完整 baseline 指标、seed bootstrap、预测 CSV、matched-pair CSV、图表。
- 冻结记录、seed 清单、运行/数据哈希、回归与完整性校验。
- ContactBelief 全部 72 个候选的逐 epoch 曲线与完成记录，包括未选中模型。

每个结果目录的 `EXPORT_MANIFEST.json` 记录实际公开文件的 SHA-256。其他 manifest/FROZEN/COMPLETE 是原始运行记录，包含未上传文件的引用，不能将它们误读成完整运行归档清单。

约 6 GB 原始运行目录及 checkpoint、optimizer、latent 二进制仍保留在原工作区，未纳入 Git。仅克隆本仓库可以审阅指标并重新聚合已发布预测，但**不能凭报告副本直接恢复训练或重放冻结模型**；这需要对应的完整原始运行文件。

报告中的本机绝对路径是原始运行 provenance，保持原样。`REPRODUCE.txt` 描述新实验与本地恢复；ContactBelief 的 `EXACT_FINAL_REPLAY.txt` / helper 需要放回原完整 run 目录才可使用。单 episode 物理重放已经验证，未额外执行完整 2400-episode 重放。

代码入口：[I001](../../experiments/iros2027/i001/README.md)、[I001-v2](../../experiments/iros2027/i001_v2/README.md)、[ContactBelief](../../experiments/iros2027/contactbelief/README.md)。
