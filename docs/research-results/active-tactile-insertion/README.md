# 主动触觉插入实验记录

- [正式Pilot结果](20260930T031448344996Z-pilot/RESULTS.md)：80 train、16 validation、48个held-out任务、3个训练初始化；结论WEAK_SIGNAL / NO-GO（针对action history独立收益）。
- [当前可支持的claim](20260930T031448344996Z-pilot/CLAIM.md)
- [负结果与下一步](20260930T031448344996Z-pilot/NEGATIVE_RESULT.md)
- [Reviewer审查](20260930T031448344996Z-pilot/REVIEW.md)
- [补充审计](20260930T031448344996Z-pilot/POSTHOC_AUDIT.md)：q可重建命令、时间顺序的替代解释、首次成功事件与控制步末记录的差异。
- [测试与物理开发证据](20260930T031448344996Z-pilot/VALIDATION.md)
- [完整smoke](20260930T031358095718Z-smoke/RESULTS.md)：仅验证pipeline，不作科学结论。

旧disturbed-grasp实验保留在 `../tactile-history-control/`，不覆盖既有结果。新代码位于仓库 `experiments/active_tactile_insertion/`；这是Cartesian接触工装与几何触觉代理，不是完整FR3或真机评价。

## 发布与后续研究

GitHub仅包含按EXPORT_MANIFEST选定的报告、指标、曲线及原始源码快照；原始轨迹、数据与权重保留本地。公开目录不能直接resume，详见[发布与验证说明](../PILOT_RELEASE_20260930.md)。

研究主线已更新为[执行时触觉反馈与在线动作修正](../../research/EXECUTION_TIME_TACTILE_REFINEMENT.md)。旧报告不重写；新方向尚未训练或评估。
