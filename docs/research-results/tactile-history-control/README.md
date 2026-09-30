# Tactile history 闭环控制实验记录

本目录保存独立的 FR3 MuJoCo disturbed-grasp recovery Pilot；不修改或覆盖 I001 / ContactBelief 结果。

- [正式Pilot](20260930T023929113268Z-pilot/RESULTS.md)：72 train、16 validation、48个未见主test扰动seed，3个训练初始化；包括shuffle控制和独立ambiguity分支。精确配置与原始轨迹在同目录。
- [执行与审计](20260930T023929113268Z-pilot/VALIDATION.md)：测试、开发阶段变更、smoke、输入时间对齐与哈希核对。
- [完整smoke](20260930T023803123901Z-smoke/RESULTS.md)：只用于验证pipeline，不作科学结论。
- [早期smoke失败](20260930T023553261553Z-smoke/failure.json)：标准化dtype错误已修复，保留原始失败记录。

实验代码、中文使用说明、固定协议位于仓库 `experiments/tactile_history_control/`。正式结论以该run的 `CLAIM.md` 为准；后续诊断必须使用新run，不能覆盖或重标这些探索性结果。

## 发布与后续研究

GitHub仅包含按EXPORT_MANIFEST选定的报告、指标、曲线及原始源码快照；原始轨迹、数据与权重保留本地。公开目录不能直接resume，详见[发布与验证说明](../PILOT_RELEASE_20260930.md)。

研究主线已更新为[执行时触觉反馈与在线动作修正](../../research/EXECUTION_TIME_TACTILE_REFINEMENT.md)。旧报告不重写；新方向尚未训练或评估。
