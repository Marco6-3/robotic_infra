# I002 异步 semantic–tactile 通信实验方案

## 1. 研究问题与结论边界

目标是比较低频 semantic policy 与高频 tactile policy 的通信方式，而不是预设异步架构、
共享 token 或双向通信一定更好。第一阶段使用 Franka/MuJoCo；结论首先只适用于该任务、
控制器、传感器抽象和计算预算。

需要检验的假设：

1. 在匹配计算预算下，异步执行能否降低接触事件到修正动作的延迟；
2. coarse action、压缩 token、完整 latent 的性能与通信开销如何变化；
3. semantic context 的 age、延迟、抖动和丢包达到什么程度后性能开始退化；
4. tactile -> semantic 反馈和事件触发更新是否比固定频率调用更有效；
5. 这些差异能否在未见摩擦、质量、扰动和物体上保持。

帧、控制 tick 和 token 不是独立统计样本。主要统计单位是 episode；随机种子、物体和
物理参数用于构建独立重复。

## 2. 分阶段实验步骤

### Stage 0 — 合约与运行时

目的：在没有学习模型和任务得分的情况下排除时间语义和实验管理错误。

1. 固定 controller、tactile、fast policy 和 semantic policy 的独立频率；
2. 所有消息保存 source、send、delivery 和 consume 时间；
3. 注入可复现的 latency、jitter、dropout 和 max staleness；
4. 检查 fast policy 永远看不到未来 semantic 信息；
5. 输出配置 SHA-256、Git commit/dirty 状态、事件 JSONL、指标和 attempt 状态；
6. 中断后只能在配置哈希一致时恢复，并保留新的 attempt，而不是覆盖原报告。

验收命令：

```bash
pixi run generate-model
pixi run iros-stage0
pixi run i002-smoke
```

Stage 0 通过只表示通信基础设施成立，不表示任何策略有效。

### Stage 1 — 双指视觉触觉与任务

默认采用研究中常见的 GelSight/DIGIT 类视觉触觉路线：在平行夹爪左右内指腹各安装一个
传感器，采集两路原始 RGB 触觉图像。参考 DIGIT 合约为 `640x480 @ 60 Hz`；策略可因算力
需要缩放为 `160x120`，但原始数据必须连同 source timestamp 保存。每个 session 为左右
传感器分别采集无接触 background reference，不把两个传感器的背景或标定参数混用。
硬件规格与图像接口以 [DIGIT 官方页面](https://digit.ml/digit.html) 和
[DIGIT 论文](https://arxiv.org/abs/2005.14679) 为参考。

当前 MuJoCo 的左右法向力通道只作为 privileged contact label，不进入默认 policy input。
[TACTO](https://github.com/facebookresearch/tacto) 是视觉触觉仿真的参考实现，但其公开集成以 PyBullet 为主；在完成 MuJoCo adapter、
图像与接触几何校验之前，不把 contact-force proxy 渲染成伪“触觉图像”。

首个任务选择 `disturbed_grasp`：机械臂抓住台面物体后施加可控侧向扰动或改变负载，fast
policy 尝试在物体滑落前修正夹持/局部动作。

步骤：

1. 在 `task_workspace_center` 添加带 free joint 的标准物体；
2. 以 60 Hz 采集左右视觉触觉 RGB，并以独立 source timestamp 对齐；
3. 离线保存 MuJoCo contact force、物体姿态、命令动作和实际控制状态作为 privileged evidence；
4. 定义 slip onset、drop、recovery 和 excessive-force 的自动判据；
5. 先实现不学习的 oracle semantic + rule-based tactile controller；
6. 验证不同延迟和频率确实能改变闭环结果，否则停止训练模型并修正任务。

### Stage 2 — 最小学习系统

分开训练并保存：

1. semantic encoder/policy；
2. tactile history encoder；
3. communication adapter/token compressor；
4. fast residual policy。

训练、验证、闭环评估和导出必须是不同入口。每次训练记录 device、CUDA、batch size、
AMP、worker 数、数据根目录、checkpoint 和最佳验证指标。首次有效 checkpoint 尽早保存，
后续优先恢复而不是重训。

### Stage 3 — 核心 baseline

先固定任务、训练数据、参数量和平均计算预算，只比较：

| ID | 系统 | 通信 |
| --- | --- | --- |
| B0 | slow semantic only | 无 fast policy |
| B1 | fast tactile only | 无 semantic context |
| B2 | synchronous multimodal | 每次联合计算 |
| B3 | async coarse action | 只传 nominal action |
| B4 | async compressed tokens | `K={1,4,16,64}` |
| B5 | async full latent | 不压缩 semantic latent |
| B6 | async bidirectional | 加 tactile summary/event |
| B7 | async event-triggered | 触觉事件触发 semantic update |

不能一次展开所有变量。推荐顺序：B0–B4 确认基本信号，再扫描 token 数量，再做
staleness，最后比较方向和触发方式。

### Stage 4 — 时序与通信鲁棒性

逐项测试并保存实际而不是目标值：

- semantic rate：`2, 5, 10 Hz`；
- fast policy rate：`20, 50, 60, 100, 200 Hz`；其中 60 Hz 是 DIGIT 类采集基线；
- deterministic latency：`0, 20, 50, 100, 200, 500 ms`；
- jitter、随机丢包、burst drop；
- token age gate 和超时后的 fallback；
- 单向与双向消息字节数、每秒调用数和 GPU 时间。

### Stage 5 — 泛化与确认性评估

数据划分按 episode/object/physics seed 完成，禁止把相邻帧随机拆入训练和验证。至少保留：

- 未见摩擦区间；
- 未见质量/质心；
- 未见扰动方向和幅值；
- 未见物体；
- 传感器噪声、偏置和短时缺失。

开发阶段可用 3 个种子排错；形成方法结论前至少使用 10 个独立种子，并报告 episode
均值、置信区间、失败类型和每个 seed 的原始结果。

## 3. 指标

任务指标：episode success、recovery success、drop/slip 次数、完成时间、力峰值和冲量。

时序指标：source-to-consume age、接触到修正动作的端到端延迟、semantic/fast 实际频率、
deadline miss、stale fallback 次数。

计算与通信：前向耗时、GPU 峰值显存、semantic calls/episode、token 数、bytes/message、
bytes/s。比较同步与异步时必须匹配或明确报告总算力预算。

## 4. 运行产物

每个 `runs/i002/<run-id>/` 必须包含：

```text
config.yaml
manifest.json
attempts/0001/events.jsonl
attempts/0001/metrics.jsonl
attempts/0001/summary.json
checkpoints/                 # 开始训练后使用
```

`manifest.json` 记录配置哈希、Git 状态、环境、attempt 和 checkpoint 来源。配置哈希不一致
时不得在原 run 上恢复。

## 5. 停止条件

出现以下任一情况时，不继续扩大模型：

- rule-based fast loop 在无延迟条件下仍不能改善任务；
- 实际触觉/控制频率达不到计划频率且未记录 deadline miss；
- baseline 没有匹配计算预算；
- 结果只来自连续帧而非独立 episode；
- 性能差异无法在多个物理 seed 重现；
- token probe 或可视化是唯一“机制有效”证据。
