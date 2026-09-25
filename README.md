# FR3 MuJoCo 机器人学习平台（v1）

这是 FR3 机器人学习平台的仿真优先实现。项目在 **Ubuntu 24.04 + ROS 2 Jazzy** 目标环境中，通过 `mujoco_ros2_control` 使用 MuJoCo，并让学习接口独立于 ROS 话题名和控制器名。

## v1 固定约定

- Franka Research 3 机械臂和 Franka Hand 夹爪。
- MuJoCo 物理仿真和底层控制频率为 1 kHz（`dt = 0.001 s`）。
- 策略动作和 LeRobot 主时间线为 30 Hz。
- 机械臂动作是 7 个关节位置目标。
- 夹爪动作同时提供米制实际宽度和 `[0, 1]` 归一化值。
- 两个 RGB 相机：固定外部相机和腕部相机，均为 `640x480 @ 30 FPS`。
- 不模拟相机噪声、延迟、抖动和丢帧。
- 录制器使用 LeRobot 视频存储，不把 RGB 帧永久保存成单张图片。
- MoveIt 只用于可选验证，不是策略运行依赖。

低层目标桥支持零阶保持和因果线性插值。默认线性模式在收到新的 30 Hz 目标后开始插值，并在推断出的策略周期内到达目标；需要立即采用零阶保持时使用 `hold` 模式。

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
.venv-recording/bin/python scripts/record_smoke.py --episodes 2 --frames 60
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
