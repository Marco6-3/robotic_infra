# IROS 2027 触觉操作实验

这里保存建立在 `robotic_infra` 通用 FR3/MuJoCo 能力之上的论文实验代码。研究问题的
权威描述仍在 `robot-gripper-iros-2027`；本目录只承载可执行配置、策略、消融和指标。

## 当前完成范围

- I001（触觉时间上下文）和 I002（异步 semantic–tactile 通信）的 Stage 0 配置；
- I001 使用 200 Hz 接触信号/100 Hz fast policy；I002 使用双指视觉触觉 60 Hz/fast policy 60 Hz、semantic policy 5 Hz 的因果调度检查；
- 操作台几何与任务工作区锚点检查；
- 配置 SHA-256、采样数量、最大信息年龄和结论边界报告。
- I002 的延迟/抖动/丢包通道、双向消息合约、可恢复 run 目录和详细实验方案。

运行：

```bash
pixi run generate-model
pixi run iros-stage0
pixi run i002-smoke
```

报告写入 `runs/iros-stage0/<timestamp>/report.json`。通过 Stage 0 只表示配置、场景命名和
因果调度成立，不表示策略学会了抓取、滑移恢复或插入。

## 代码归属

- 可复用的触觉观测、接触仿真、控制器和机器人接口应进入顶层 `src/`；
- I001/I002 专属 policy、共享 token、消融和指标留在本目录；
- 大型数据集与 checkpoint 不提交到 Git，由运行清单记录路径和哈希。

## 下一阶段

1. 基于下方已实现的接触代理任务补充未见物体与扰动测试集；
2. 保留 MuJoCo contact force 作为 privileged label，并实现双指视觉触觉 MuJoCo renderer adapter；
3. 用独立高频记录和因果历史加载器开展 I001 离线预测及闭环消融；
4. 为 I002 加入匹配计算预算的同步/异步 baseline，再开始共享 token 实验。

I002 的完整分阶段步骤、baseline、指标和停止条件见
[`i002/EXPERIMENT_PROTOCOL.md`](i002/EXPERIMENT_PROTOCOL.md)。

## 接触任务与真实快环

新增独立的 [接触代理实验](contact/README.md)：`pixi run contact-foundation`。
已连接 known-pose 抓取/抬升、随机质量/摩擦/扰动、200 Hz 力代理、独立原始记录、
100 Hz 规则残差和 5 Hz 慢意图通道，提供 I001 的因果历史加载器。
该入口不替代 I002 的 DIGIT RGB 配置；视觉触觉渲染和学习实验仍待实现。
