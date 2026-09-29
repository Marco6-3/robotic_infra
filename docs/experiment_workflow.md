# 实验工作流：采用成熟机器人学习仓库的组织方式

## 参考项目与采用范围

核对日期：2026-09-29。参考官方公开文档，而不是假定所有实验室都采用同一种软件栈。

| 参考 | 可核实的组织方式 | 本仓库采用方式 |
| --- | --- | --- |
| [robosuite](https://robosuite.ai/docs/overview.html)，由 Stanford SVL、UT Austin RPL、NVIDIA GEAR 等维护使用 | 环境、机器人、对象、传感器、控制器和输入设备模块分离 | 共用 `DisturbedGrasp`、`ContactProxy`、`GripResidual`；预览与批量评估调用同一个 episode 运行函数 |
| [robosuite 环境接口](https://robosuite.ai/docs/modules/environments.html) | 同一环境支持有窗口、无窗口及不同观测配置 | `contact-play` 与 `contact-eval` 共用 YAML 参数；当前不是 robosuite 环境适配器，也未声称兼容 Gym API |
| [robomimic](https://robomimic.github.io/docs/tutorials/configs.html) | 集中配置；通过独立实验配置启动，不直接修改默认配置 | 校验物理、传感器和控制参数，保存配置及代码快照；用户复制配置建立自己的实验 |
| [MoveIt](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html) | RViz 目标交互、规划与执行 | 后续 ROS 规划入口；当前尚未接入，不冒充已完成的可视化功能 |

当前实现采用这些项目的结构原则，没有复制其代码，也没有安装/迁移成 robosuite 或 robomimic。
现有 FR3、MuJoCo、ROS 与记录环境保持独立。无需新增训练框架即可使用本轮入口。

## 从这里开始

```bash
cd /home/mingzhe/Documents/ws/robotic_infra
pixi run contact-play
```

看到机械臂、桌面和橙色方块后，实验自动以半速播放并重复。这个入口不需要另开键盘终端，
也不连接 ROS 命令话题。它是自动实验的交互预览，不是人工末端遥操作。

**按键焦点放在 MuJoCo 窗口：** 空格暂停/继续，`R` 从头重来，`N` 切换下一实验条件，
`Q` 退出。左上角显示当前条件、任务阶段、触觉法向真值、夹爪修正量和掉落状态。
图中文字使用英文以适配 MuJoCo 内置字体。保持 `Geom 0` 开启才能看到桌面。
不要用窗口原生 Control/Physics 控件修改实验；窗口渲染独立模型副本，实验由配置控制。

```bash
# 看固定策略为何掉落
pixi run contact-play --condition nominal
# 只看一遍，正常速度
pixi run contact-play --condition fast --speed 1 --once
# 选其他随机种子
pixi run contact-play --seed 1
```

预览不写数据；暂停和慢放仅改变墙钟速度，物理步长仍为 1 ms。

## 修改实验

复制默认配置，给自己的实验独立命名：

```bash
mkdir -p experiments/local
cp experiments/iros2027/contact/config.yaml experiments/local/my_contact.yaml
```

在自己的文件中改参数，然后运行：

```bash
pixi run contact-play --config experiments/local/my_contact.yaml --condition fast
pixi run contact-eval --config experiments/local/my_contact.yaml
```

| 配置部分 | 可改参数 |
| --- | --- |
| `task` | 质量、摩擦、扰动力峰值范围 |
| `sensor` | 采样率、噪声、允许的信息年龄 |
| `controller` | 快慢频率、夹爪修正步长和上限、力阈值、切向/法向比阈值 |
| `seeds` | 批量运行的随机种子 |
| `conditions` | 固定/快环、触觉延迟、慢消息延迟、扰动倍率 |

当前采样频率必须是 1000 Hz 的整数约数，非法值会报错。该限制属于此接触代理任务，
不更改原 I002 的 60 Hz DIGIT 方案。四秒任务阶段与评价阈值仍是固定协议，修改它们需要
同时修订任务和验收指标；这里只把经验证的常用实验参数开放为配置。

## 查看结果

```bash
pixi run contact-eval
# 用上一步最后打印的 Results 路径替换下面的 RUN_PATH
pixi run contact-report RUN_PATH
```

每次评估生成新的运行目录。查看 `manifest.json` 的实际配置、代码和网格哈希；
`attempts/0001/` 下每个 episode 都有独立的触觉、控制、决策与真值日志。
汇总按 episode 计数，不把连续帧作为独立样本。可视化调试与批量评估使用同一个控制循环。

## 后续接入顺序

1. 先用当前任务确认物理参数、延迟、记录和失败判据，再扩大物体和独立测试种子。
2. I001 增加独立的 dataset/train/evaluate 入口，保留当前规则策略作为 baseline。
3. I002 先比较匹配计算预算的同步/异步策略，再加 learned token 与双向通信。
4. 若需要人工示范，接入末端 IK/OSC 与 SpaceMouse/手柄；避免以逐关节按键作为正式采集方式。
5. ROS 调试需要时接 RViz 的 RobotModel/TF/传感器显示；末端规划需要 MoveIt 配置和轨迹控制器。
6. DIGIT 渲染、RoboTwin 及真实传感器仍是单独的适配阶段。

这些后续能力尚未实现。当前只完成了配置统一、预览、记录、汇总和相关回归检查。
