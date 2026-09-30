# Reviewer审查

- 输入边界：37维=32几何taxel+3encoder q+2上一真实下发命令。observed T不读force/teacher，测试直接阻断这些接口；被动mount、孔位、摩擦、dq都不送student。
- teacher特权：仅MSE action target与evaluation；没有future/contact-state辅助目标。安全终止为任务定义，零target episode不补抽，计数已报告。
- 参数公平：[278466]，M0重复current、M0/M1屏蔽两action维度；三者相同初始化、batch索引、optimizer、steps、最后checkpoint。不同骨干不参与比较。
- 因果：x_t的a是上一下发命令，clip后记录；先取history/teacher label再step。causal mask和snapshot restore有测试，no future observation。
- seed：development/train/validation/test/ambiguity分区，test与probe使用同seed确定但不同RNG流；全部条件成对。三初始化共享数据，报告cluster bootstrap，不当独立采集。
- test tuning：物理修复/圆鼻导入端/probe限位在学习前完成；模型评估后不改task/匹配门槛。固定最后checkpoint，val仅日志；FROZEN及source/config/data/model hashes核验。
- 歧义诊断：镜像task不等于相同观测，特别是q暴露运动方向。必须完整T/q匹配；本次合格0对。全部镜像结果描述性展示，不把不合格pair宣传为action独立价值证据。
- 强简单基线：nominal、constant+x、spiral、强down均真实闭环；teacher使用孔位不属于可部署student。圆鼻和软接触可能带来被动对中，必须与nominal比较。
- 力安全：1ms物理步检查contact force，超力failure不可恢复为success；固定8N阈值。MuJoCo接触尖峰/软接触材料非真实校准，安全结论仅此仿真。
- 实时：200Hz是仿真时间，不代表实际机器人5ms最坏推理时延。
- 有限证据：单形状、小数据、3初始化、48个test seeds，CI宽及全成功bootstrap退化限制结论。未做多重检验校正，不据此给paper-level显著性结论。
- 文献只支持任务背景，不支持本M2>M1主张；见LITERATURE.md。本轮没有world model、未来预测或大模型。

审查结论：WEAK_SIGNAL / NO-GO，REVISE。保留所有失败、旧grasp结果和本轮开发记录。


补充：[POSTHOC_AUDIT.md](POSTHOC_AUDIT.md) 检查安全成功/动作时间对齐，并讨论history覆盖、时间顺序与q可重建动作的替代解释；未改变指标或实验。
