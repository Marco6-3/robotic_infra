# FR3 MuJoCo 机器人学习平台（v1）

这是 FR3 机器人学习平台的仿真优先实现。项目在 **Ubuntu 24.04 + ROS 2 Jazzy** 目标环境中，通过 `mujoco_ros2_control` 使用 MuJoCo，并让学习接口独立于 ROS 话题名和控制器名。

## 第一次使用：先看自动抓取实验

```bash
cd /home/mingzhe/Documents/ws/robotic_infra
pixi run contact-play     # 看真实物理实验，半速自动重播
pixi run contact-eval     # 无窗口批量运行并保存数据
pixi run contact-report RUN_PATH  # 用上一条输出的 Results 目录查看汇总
```

在仿真窗口按空格暂停、`R` 重来、`N` 下一条件、`Q` 退出，无需另开键盘控制终端。
详细使用方式、配置修改和参考仓库见 [实验工作流](docs/experiment_workflow.md)。

I001 已新增受控观测混叠实验：`pixi run i001-collect`、`pixi run i001-evaluate RUN_PATH`。
[实测结果与结论边界](experiments/iros2027/i001/RESULTS.md)：历史优于带噪声单帧触觉，
但未优于当前触觉加高精度本体状态。
后续 I001-v2 的限定机制证据与 ContactBelief-v0 的 NO-GO 方法结果，见
[受控实验结果索引](docs/research-results/README.md)。
`sim-nvidia` 是 ROS 基础场景入口，`keyboard_policy.py` 只用于关节命令链路测试。


## 当前研究方向（2026-09-30）

主线已转向 **执行时触觉反馈如何修正已规划、尚未执行的动作**：慢规划器输出action chunk，快触觉模块在执行中刷新观测；先验证fresh feedback，再检验预期接触与实际接触的偏差是否改善纠错。

[研究方向与下一轮Pilot设计](docs/research/EXECUTION_TIME_TACTILE_REFINEMENT.md) 对照T-Rex、TacForcing与TacPAC，明确因果时序、强基线及停止条件。**新方向尚未实现或训练**，20/200 Hz只是候选调度设置。

已完成的[active insertion Pilot](docs/research-results/active-tactile-insertion/20260930T031448344996Z-pilot/RESULTS.md) 中，history相对current为+6.94 pp，但past action相对T/q history为−2.08 pp；q历史可较好重建动作方向。保留负结果，暂停围绕action history独立收益扩展。此前[disturbed-grasp Pilot](docs/research-results/tactile-history-control/20260930T023929113268Z-pilot/RESULTS.md) 的constant-max为100%，也一并保留。

两轮代码、逐episode指标、图表与审计已整理；公开范围、依赖和精确恢复限制见[发布说明](docs/research-results/PILOT_RELEASE_20260930.md)。插入实验使用Cartesian工装与几何触觉代理，不代表完整FR3或真机插入。

## 本机运行与策略验收（2026-09-26）

本机环境使用项目 `.pixi`（ROS/MuJoCo）与 `.venv-recording`（LeRobot/ACT/CUDA）
分开管理，无需安装系统 ROS 或修改 NVIDIA 驱动。具体实测结果见 [VALIDATION.md](VALIDATION.md)。
ML 安装和验收入口会清除子进程的 `PYTHONPATH` / `PYTHONHOME`，避免 RoboStack
激活环境将 ROS 的同名 Python 包注入虚拟环境。手动从 `pixi shell` 调用录制环境时也应清除这两个变量。

```bash
cd /home/mingzhe/Documents/ws/robotic_infra
pixi run test
pixi run test-model
pixi run ros-smoke      # 自动构建、启动无窗口 ROS 仿真、测试动作/双相机并退出
pixi run policy-smoke   # ACT 双相机 CUDA 训练/推理冒烟和直接 MuJoCo 闭环
pixi run record-smoke   # 两段视频数据录制并逐帧读回
pixi run iros-stage0    # 检查 I001/I002 配置、场景锚点和多频率因果调度
pixi run i002-smoke     # 检查异步通信、延迟/丢包/staleness 和可恢复运行目录
pixi run sim-nvidia     # 本机双显卡：使用 NVIDIA 打开交互式 MuJoCo 窗口
```

首次使用策略环境时执行 `pixi run install-policy`。该入口选择 PyTorch 2.8.0、
torchvision 0.23.0、CUDA 12.8 和配套 torchcodec 0.7.0；不会让 pip 改写 ROS 环境。
本机完整已验证的 Python 包版本另保存在 `config/requirements-policy-linux64.lock.txt`。
安装器支持 `--index-url`、`--torch-index-url` 和 `--lerobot-source`，可以在网络不稳定时
使用包镜像与已验证的本地 LeRobot checkout，参数只影响本次安装。

`ros-smoke` 默认使用独立的 ROS domain 173 和本机发现范围，输出位于
`runs/ros-*/`；可通过 `ROS_DOMAIN_ID` 指定其他空闲 domain。日志中的实时调度权限提示
不影响普通仿真，但本项目不承诺 1 kHz 硬实时性能。

`policy-smoke` 使用固定提交的 LeRobot ACT 实现和 ResNet18 双相机编码器，
将原始 640×480 图像缩放为 128×96，使用小型 Transformer 配置降低显存占用。
默认进行 20 次优化、检查点保存/读回和 30 帧仿真闭环。输出包含配置、损失、耗时、
显存峰值、检查点 SHA-256 和相机图片，位于 `runs/act-*/`。模型动作在此测试中定义为
相对 home 的关节残差（0.1 rad 尺度）与归一化夹爪宽度，执行前限制范围并统计限幅次数。
这不是仓库录制格式的 9 维绝对动作；接入其他数据集需要对应的动作适配。

```bash
# 继续本脚本自己的检查点；--steps 表示累计优化步数
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python scripts/policy_smoke.py \
  --resume runs/act-时间戳/checkpoint.pt --steps 40
```

此测试仅使用两帧本地脚本样本验证计算和控制链路，没有预训练任务权重，也没有
抓取任务、训练/验证数据集划分或成功率评测。它不证明视觉泛化或操作能力。
上述策略冒烟的基础场景有操作台和接触力代理，但不包含动态任务物体、视觉触觉图像后端或任务判据。
独立的 `contact-play/contact-eval` 已包含动态方块和任务判据；
因此仍不能把策略冒烟测试解释为正式操作实验。

RoboTwin 是另一套基于 SAPIEN 的任务平台，其官方基础环境为 Python 3.10，
需要单独下载任务资产及配置策略；本项目的 FR3/MuJoCo 环境不能直接作为其运行环境。
本次选择 ACT 验证本项目策略链路，未把它记为 RoboTwin 基准通过。
参考 [RoboTwin 安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。

ROS 构建还会自动应用 `patches/mujoco-ros2-control-librt.patch`，为固定上游的共享内存
相机目标补上 Linux `librt` 链接。图形开发库由 Pixi 显式提供；ROS 与直接 MuJoCo
统一从模型 `home` keyframe 启动。腕部相机朝向已修正为夹爪前方工作区，避免手掌外壳
遮挡整个画面。`sim` 保留系统默认显卡选择；本机测试默认路径相机约 12 Hz，
`sim-nvidia` 约 28 Hz，无窗口约 29 Hz。这些是短时测量，不是性能保证。

## v1 固定约定

- Franka Research 3 机械臂和 Franka Hand 夹爪。
- MuJoCo 物理仿真和底层控制频率为 1 kHz（`dt = 0.001 s`）。
- 策略动作和 LeRobot 主时间线为 30 Hz。
- 机械臂动作是 7 个关节位置目标。
- 夹爪动作同时提供米制实际宽度和 `[0, 1]` 归一化值。
- 两个 RGB 相机：固定外部相机和腕部相机，均为 `640x480 @ 30 FPS`。
- I002 默认部署观测是安装在左右内指腹的两路 DIGIT 类 RGB 视觉触觉，参考规格
  `640x480 @ 60 Hz`；策略输入可缩放到 `160x120`，原始视频和 source timestamp 必须保留。
- 两个 MuJoCo fingertip normal force 仅作为 privileged label，不是策略默认输入，也不是
  触觉图像或真实硬件传感器模型。
- 不模拟相机噪声、延迟、抖动和丢帧。
- 录制器使用 LeRobot 视频存储，不把 RGB 帧永久保存成单张图片。
- MoveIt 只用于可选验证，不是策略运行依赖。

低层目标桥支持零阶保持和因果线性插值。默认线性模式在收到新的 30 Hz 目标后开始插值，并在推断出的策略周期内到达目标；需要立即采用零阶保持时使用 `hold` 模式。

## 操作台与实验工作区

生成场景包含一个位于机器人前方的静态可碰撞操作台：台面尺寸为
`0.90 x 0.90 x 0.05 m`，台面高度为 `0.40 m`，并定义
`task_workspace_center = [0.60, 0.00, 0.405] m` 作为任务物体的默认放置锚点。
当前 FR3 模型安装在地面原点；常见的 `0.7--0.8 m` 人体工位会与现有 home 姿态相交，
因此这里使用低矮研究台。该尺寸服务于仿真可达性，不代表最终真机安装尺寸。

纯接口、图像和时序冒烟不依赖操作台；抓取、滑移恢复、插入和接触转换实验必须使用
操作台或等价接触环境。操作台和 touch sensor 本身仍不构成任务，正式实验还需要动态物体、
随机化、传感器噪声模型以及可自动计算的成功/失败条件。

I002 已具备确定性的异步通信 smoke、延迟/抖动/丢包注入、双向消息、staleness gate 和
可恢复 run 目录。完整实验阶梯见
[`experiments/iros2027/i002/EXPERIMENT_PROTOCOL.md`](experiments/iros2027/i002/EXPERIMENT_PROTOCOL.md)。

## 目录结构

```text
src/fr3_robot_api/   与框架无关的 Robot、Action 和 Observation 类型
src/fr3_control/     30 Hz 到 1 kHz 的目标桥
src/fr3_recorder/    相机/状态同步和 LeRobot 适配器
src/fr3_sim/         可选的直接 MuJoCo 冒烟测试 Robot 适配器
src/fr3_description/ ROS 包、控制器/相机配置和启动文件
config/              固定模型和运行时清单
repos/               可复现的第三方 checkout 清单
tests/               与依赖无关的接口和时序测试
experiments/iros2027/ I001/I002 的配置、因果调度检查和后续论文实验
```

## 可复现的模型输入

`repos/models.repos` 固定 MuJoCo Menagerie FR3 模型和官方
`franka_description` 仓库的不可变提交。Menagerie 使用 Git 标签
`v2026.9.0`，不是 Python 包版本。运行仿真前，准备脚本会验证 checkout
提交是否与清单一致。

`config/runtime_manifest.yaml` 还固定 MuJoCo 3.10.0、LeRobot v0.6.1 和选定的
`mujoco_ros2_control` 快照。LeRobot 通过提交固定，避免录制器接口和默认视频
流水线发生无记录漂移。

外部 `mujoco_ros2_control` 仓库单独固定在 `repos/third_party.repos` 中，
不会被提交到本仓库；执行 `vcs import` 后放在 `src/mujoco_ros2_control/`。

## 初始化（Pixi / Ubuntu 24.04 / ROS 2 Jazzy）

仓库使用 Pixi 和 RoboStack 管理 ROS 2 Jazzy、MuJoCo Python、编译工具及
运行时依赖。Pixi 环境支持 `linux-aarch64` 和 `linux-64`，不会修改系统的
ROS 或 Ubuntu 安装。先安装 Pixi，然后在仓库根目录执行：

```bash
# 官方 Pixi 安装器；安装到当前用户目录，不需要 sudo
curl -fsSL https://pixi.sh/install.sh | sh

cd robotic_infra
pixi install --locked
pixi run bootstrap
pixi run verify-pins
pixi run verify-runtime
pixi run build
```

安装完成后，可以在启动 ROS 前运行无窗口控制回路检查：

```bash
pixi run test
pixi run test-model
pixi run smoke-test
```

`build_fr3_hand_scene.py` 在第三方 checkout 存在时读取
`third_party/franka_description/robots/fr3/joint_limits.yaml`，使生成的 MJCF
和 ROS 描述使用相同的机械臂关节限制。夹爪手指关节范围固定为 `0..0.04 m`，
并通过 `fr3_split` tendon 耦合。

展开 ROS xacro 为 URDF 后，可以验证两套描述：

```bash
python3 scripts/validate_model_pair.py --urdf /path/to/fr3_expanded.urdf
```

如果机器没有 MuJoCo Python 运行时，只检查 XML 名称、限制和夹爪几何时使用
`--skip-fk`。

## 启动仿真

Pixi 环境中执行：

```bash
pixi run sim
```

启动文件使用绝对 MJCF 路径，因为 launch 进程不应依赖不确定的工作目录。
底层目标插值默认为 `linear`。launch 自动启动策略桥并使用仿真时钟；可通过
`target_interpolation:=hold` 选择保持模式，或通过 `start_policy_bridge:=false`
停用默认策略桥。MoveIt 尚未集成，不提供无实际作用的启动开关。

键盘策略可以验证 30 Hz 动作话题和 1 kHz 低层桥：

```bash
# 终端 1（已启动则不必重复）
pixi run sim

# 终端 2
pixi shell
source install/setup.bash
python3 scripts/keyboard_policy.py --ros-args -p use_sim_time:=true
```

键盘控制：`1..7` 选择关节，`a/d` 移动关节，`o/c` 调整夹爪，空格回到
固定初始姿态，`q` 退出。键盘发布的数据为
`[q1..q7, width_m, width_normalized]`，不依赖 MoveIt 或 LeRobot。

## 直接 MuJoCo 检查

`fr3_sim.MujocoRobot` 提供直接 MuJoCo 验证路径，可检查生成的 MJCF、相机、
动作响应和数据录制。`reset()` 恢复 home keyframe 并清除上一回合的动作，
`close()` 释放渲染器。ROS 路径仍由 `mujoco_ros2_control + ros2_control` 提供，
两条路径需要分别验收；直接仿真通过不代表 ROS 启动已通过。

## 测试

干净克隆后，不需要 ROS、MuJoCo、第三方模型或 LeRobot 的纯 Python 合约测试：

```bash
python3 -m pytest -q
```

需要 Python 3.12、NumPy 和 pytest。默认跳过带 `integration` 标记的模型测试；
`pixi run test-model` 会先验证固定 checkout 并生成模型，缺少输入时明确失败。
CI 运行纯 Python 测试，ROS 启动和 GPU 驱动兼容性仍需在目标机器验收。

## 数据集语义

录制器使用固定外部相机时间戳作为 30 Hz 数据集时间线。默认只选该时间点之前
最近的腕部帧和状态，允许的最大年龄分别为 40 ms 和 10 ms。
`RecorderConfig(causal=False)` 仅用于有意的离线双向匹配。每帧表示
`(observation_t, action_t)`；下一次 tick 后得到的状态是
`observation_{t+1}`。这样可以明确动作和观测的时序，并为后续更高频率模态
保留扩展空间。帧中保留各模态、动作及决策时间戳；默认决策时间等于外部帧时间，
有推理延迟时显式传入 `decision_timestamp_ns`。动作不得晚于决策时刻，
动作年龄及决策延迟默认均不得超过 40 ms。过期/未来数据返回 `None/False`，
非法动作抛出异常，调用者必须统计拒绝帧数，不能把缺帧伪装成连续实时数据。

LeRobot 的 `timestamp` 为固定 FPS 的名义时间；`timestamp_ns` 保存实际源时间。
1 kHz / 30 Hz 使用 33/34 步交替调度，量化误差小于 1 ms，不再每 33 步触发造成漂移。
动作同时提供米制宽度和归一化宽度时必须一致；控制和录制共用规范化逻辑，
越界命令直接拒绝，不再出现控制被限幅但标签仍记录原始值的情况。

## 录制与读回验收

LeRobot 的 dataset 依赖较多，单独安装到 `.venv-recording`，避免 pip 改动
ROS/Pixi 的锁定环境。固定版本要求 Python 3.12。训练时可使用该环境或另建
与你的 CUDA / JetPack 匹配的训练环境；ARM 上的 PyTorch GPU 支持需另外验收。

```bash
pixi run install-recording
pixi run record-smoke
```

`record-smoke` 用直接 MuJoCo 路径生成两段各 30 帧的脚本运动，保存 LeRobot
视频数据，再逐帧重新读取两路图像、动作和时间戳。成功后在新建的
`datasets/smoke-*` 目录写入 `validation.json`；不覆盖已有目录，不上传数据。
无显示器时默认尝试 `MUJOCO_GL=egl`，需目标机器提供可用的 EGL 驱动。
这是数据管线验收，不是任务示范，也不能用于证明策略训练或任务成功率。

需要改变验收规模时：

```bash
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python scripts/record_smoke.py --episodes 2 --frames 60
```

## 本机 ROS 验收

1. `pixi run build` 成功，`pixi run verify-runtime` 显示 Python/native 与 ROS vendor
   均为 3.10.0。`verify-pins` 检查 `.repos` 的固定提交，生成模型和构建依赖该检查。
2. `pixi run sim` 后，控制器正常激活，两路图像和 `/joint_states` 持续发布。
3. 在另一个已加载 `install/setup.bash` 的终端启动上述键盘策略；首条动作应能驱动
   仿真机器人，改变关节与夹爪目标时状态随之变化。越界/非法动作应被拒绝并记录日志。
4. 暂停/恢复仿真，确认策略桥使用 `/clock`；回拨仿真时钟后丢弃旧目标，等待新动作。

ROS Python 定时器设置为 1 ms，不构成硬实时保证，也不保证每个物理步恰好收到
一次 ROS 消息。高频触觉实验应测量实际延迟，必要时将修正控制放入仿真步回调。
当前仓库仍需为具体机器人学习实验增加任务、示范/训练入口和成功判据。

## 后续替换为真实机器人

`MujocoRobot` 是唯一的仿真器实现。未来的 `FrankaRobot` 可以实现相同的
`Robot` 协议，而不需要修改策略、录制器或数据集代码。v1 不要求两种实现
使用完全相同的 launch 文件或 ROS 话题名。
