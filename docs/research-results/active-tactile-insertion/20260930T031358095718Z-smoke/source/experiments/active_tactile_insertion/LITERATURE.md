# 文献定位（2026-09-30核验）

[ContactWorld v3](https://arxiv.org/html/2606.13877v3) 包含insertion、disassembly、screwing和exploration等任务，比较不同视觉/触觉表示。其主要问题是world-model representation对预测与规划的影响；并不直接建立本实验的M2相对完整T/q-history M1的独立动作信息收益。文中也区分预测稳定性改善与任务成功。

[VT-WAM v1](https://arxiv.org/html/2607.02503v1) 把视觉、触觉形变预测和动作生成联合建模，在含插入的真实接触任务评估；其平台具有夹爪内侧双Xense传感器。其大模型、视觉锚点和附加训练目标与本Pilot不同，不能把其整体收益视为本实验的action-history消融结论。

这两篇工作为接触操作和动态触觉的重要性提供背景，不替代我们的独立闭环比较，不证明本方法新颖性。本轮不引入它们的world-model或flow-matching架构。
