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

`config/runtime_manifest.yaml` 还固定 MuJoCo 3.4.0、LeRobot v0.6.1 和选定的
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

cd fr3_mujoco_platform
pixi install
pixi run bootstrap
pixi run verify-pins
pixi run build
```

安装完成后，可以在启动 ROS 前运行无窗口控制回路检查：

```bash
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
底层目标插值默认记录为 `linear`，MoveIt 验证配置默认关闭。

键盘策略可以验证 30 Hz 动作话题和 1 kHz 低层桥：

```bash
# 终端 1
pixi run build
pixi shell
ros2 run fr3_control fr3_policy_bridge

# 终端 2
pixi shell
python3 scripts/keyboard_policy.py
```

键盘控制：`1..7` 选择关节，`a/d` 移动关节，`o/c` 调整夹爪，空格回到
固定初始姿态，`q` 退出。键盘发布的数据为
`[q1..q7, width_m, width_normalized]`，不依赖 MoveIt 或 LeRobot。

## 直接 MuJoCo 检查

`fr3_sim.MujocoRobot` 是直接 MuJoCo 冒烟测试适配器，可检查生成的 MJCF、
相机渲染和与框架无关的 Robot 接口。生产控制路径仍然是
`mujoco_ros2_control + ros2_control`。

## 测试

不需要 ROS、MuJoCo 或 LeRobot 的纯 Python 合约测试可以运行：

```bash
python3 -m pytest -q
```

运行时请在 Pixi 环境中安装清单指定的 ROS 依赖、MuJoCo Python 绑定和
LeRobot。录制器延迟导入 LeRobot，因此接口、时序和模式测试不需要 GPU、ROS
或视频编码器。

## 数据集语义

录制器使用固定外部相机时间戳作为 30 Hz 数据集时间线。腕部帧和机器人状态
按最近时间戳选择，同时在帧元数据中保留源时间戳。每帧表示
`(observation_t, action_t)`；下一次 tick 后得到的状态是
`observation_{t+1}`。这样可以明确动作和观测的时序，并为后续更高频率模态
保留扩展空间。

## 后续替换为真实机器人

`MujocoRobot` 是唯一的仿真器实现。未来的 `FrankaRobot` 可以实现相同的
`Robot` 协议，而不需要修改策略、录制器或数据集代码。v1 不要求两种实现
使用完全相同的 launch 文件或 ROS 话题名。
