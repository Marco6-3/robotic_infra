# 本机修复与验收记录

日期：2026-09-26。所有通过项均来自本机实际执行。

## I002 基础设施增量验收（2026-09-28）

本次新增内容已在同一工作区重新执行，而不是沿用旧测试结论：

| 检查 | 实测结果 | 本地证据 |
|---|---|---|
| 纯 Python 测试 | 33 通过，7 个集成项按默认规则跳过 | 本次终端输出 |
| MuJoCo 集成测试 | 7 通过，包含操作台和无相机触觉读取 | 本次终端输出 |
| 指端接触力代理 | Franka Hand 左右内侧指腹各有一个 privileged 法向力通道 | 生成的 `scene.xml` 运行时枚举 |
| I002 视觉触觉合约 | 双指 DIGIT 类 RGB，原始 `640x480 @ 60 Hz`，策略可缩放 `160x120` | `tests/test_contracts.py` |
| I002 通信 smoke | 10 次 semantic、120 次 fast decision、40 次 contact event，使用未来信息 0 次 | `runs/i002/20260928-215128-079902-i002_stage0_async_communication/attempts/0001/summary.json` |
| 延迟与丢包 | semantic age P95 200 ms、最大 216.67 ms；feedback 固定种子丢弃 1/40 | 同上 |
| 恢复路径 | 原 run 新增 `attempts/0002`，未覆盖第一次事件和指标 | `runs/i002/20260928-215128-079902-i002_stage0_async_communication/manifest.json` |
| ROS 回归 | 10 包构建、3 控制器、60 动作、双相机约 29.16 Hz | `runs/ros-20260928-123611-878529/validation.json` |

这里的指端力通道是 MuJoCo privileged label；默认策略观测已经选为双指视觉触觉，但
MuJoCo 视觉触觉 renderer 尚未实现，不能声称已经生成或验证 DIGIT 图像。
I002 smoke 仅验证确定性的异步消息合约、延迟/抖动/丢包、staleness、双向反馈和
run 生命周期。尚未添加动态任务物体，也没有执行学习策略、disturbed-grasp 或 baseline
性能比较，不能把 `communication_smoke_passed` 解读为 I002 假设成立。

## 环境与隔离

- Ubuntu 24.04.4 LTS，内核 `7.0.0-34-generic`，NVIDIA 驱动 `595.84`。
- RTX 5060 Laptop GPU，8151 MiB 显存；直接 EGL 查询返回 NVIDIA 渲染器。
- 项目 `.pixi`：Python 3.12.14、ROS 2 Jazzy、MuJoCo Python/native/vendor 均为 3.10.0。
- 项目 `.venv-recording`：独立的 Python 3.12.3 策略/视频环境。
- 未使用 sudo、未修改系统驱动、未改动其他项目环境。

## 已修复

1. Pixi 显式声明 OpenGL/EGL 开发库并更新双平台锁文件，解决相机插件找不到 OpenGL 的错误。
2. 固定上游的共享内存相机实现补上 Linux `librt`，解决 `shm_open` 链接失败；补丁由构建任务幂等应用。
3. ROS 通过 `initial_keyframe=home` 初始化，避免全零姿态越界和策略目标被拒绝。
4. 新增无窗口启动参数及自动 ROS 验收/退出脚本。
5. 修正腕部相机位置和朝向；旧视野被夹爪外壳遮挡，修正后可看到手指和前方地面。
6. 新增 NVIDIA PRIME 窗口启动入口。默认显卡路径约 12 Hz，相同双相机在 NVIDIA 路径约 28 Hz。
7. 为 ACT 增加 CUDA 12.8 依赖约束、检查点和仿真闭环验收脚本。
8. 清除 ML 子进程继承的 `PYTHONPATH` / `PYTHONHOME`，修复 RoboStack 包优先于虚拟环境包加载的问题；安装入口重跑与 `pip check` 均通过。

CUDA 大包下载曾中途断开并从头重试；采用镜像分段下载、完整 SHA-256 校验后再安装。分段脚本与索引哈希清单位于 `runs/system-validation/`。

Franka Git 下载曾多次断开，最终使用官方固定提交归档恢复；196 个文件逐一与该提交的 Git blob ID 比对一致，工作区干净。没有用近似版本替代固定输入。

## 已完成的 ROS / MuJoCo 验收

| 检查 | 实测结果 | 本地证据 |
|---|---|---|
| 纯 Python 测试 | 24 通过 | `runs/system-validation/tests.log` |
| 模型集成测试 | 3 通过，含腕部视野和视线回归 | `runs/system-validation/model-tests.log` |
| 完整 ROS 构建 | 10 个包成功 | `runs/system-validation/build-clean.log` |
| 直接控制 | 2 秒仿真，60 次策略更新 | `runs/system-validation/control-smoke.log` |
| URDF / MJCF 一致性 | 关节/限位/夹爪及零位/home FK/TCP 通过 | `runs/system-validation/model-pair.log` |
| 无窗口 ROS 闭环 | 3 控制器 active，60 次命令，关节位移 0.1182 rad | `runs/ros-20260926-135747-404588/validation.json` |
| 无窗口双相机 | 640×480 RGB，约 29.27 / 28.30 Hz，非空图像 | 同上及目录内 PNG |
| NVIDIA 窗口闭环 | 60 次命令，位移约 0.12 rad，相机约 28.22 / 28.06 Hz | `runs/system-validation/gui-nvidia.json` |

## 策略与录制验收

全部通过。PyTorch `2.8.0+cu128` / CUDA `12.8`，torchvision 0.23.0，torchcodec 0.7.0，LeRobot 0.6.1 固定提交。
安装入口重跑、`pip check` 均通过；CUDA 矩阵前向/反向已在 RTX 5060 的 compute capability 12.0 上实测。
ML 子进程的同名包实际来自 `.venv-recording`，不会被 Pixi 的 PYTHONPATH 覆盖。

| 检查 | 实测结果 | 本地证据 |
|---|---|---|
| ACT 优化 | 20 步、batch 2、两路 RGB；总损失 84.6693 → 1.6161 | `runs/act-20260926-143244-318167/validation.json` |
| ACT 推理与动作闭环 | 30 帧、1 秒仿真、0 次限幅；关节位移范数 0.0678 rad | 同上 |
| ACT 模型推理耗时 | 中位 5.66 ms，P95 5.92 ms | 同上 |
| PyTorch 显存峰值 | 366.9 MiB（allocated，不含驱动/渲染上下文） | 同上 |
| 检查点读回 | 前后动作张量一致；保存原生 pretrained_model 与完整优化器/RNG 状态 | 同上及 checkpoint.pt |
| 续跑 | 从第 20 步继续到第 22 步，再完成 30 帧闭环 | `runs/act-20260926-143325-860688/validation.json` |
| LeRobot 视频 | 两回合各 30 帧；逐帧解码双相机、校验动作及时间戳，60 帧全部通过 | `datasets/smoke-20260926-143250-721010/validation.json` |

最终 ACT 与录制验收顺序执行。推理耗时不含图像渲染、缩放或 ROS 传输，不是严格性能基准。
续跑时中位推理 6.09 ms、P95 6.57 ms。
这是两帧样本上的计算链路检查，总损失下降主要包含 VAE KL 项，不是任务准确率或泛化能力。

完整已安装版本快照为 `config/requirements-policy-linux64.lock.txt`。原始日志、环境快照及源码 SHA-256 索引在 `runs/system-validation/`。
最终初次 ACT 检查点 SHA-256：`4ea03282e2700c1797ed36785243b9039752ff873c18846ce564ef9f9153d75a`。

## 边界与已知提示

- RoboTwin 未安装/运行基准任务。它使用独立的 SAPIEN/Python 3.10 基础环境及任务资产，不能把 FR3/MuJoCo 跑通当成 RoboTwin 通过。见[官方安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。
- ACT 使用小型配置、两帧脚本样本和范围受限动作，仅验证执行链路；没有抓取成功率、视觉泛化或真实机器人结论。
- ROS 日志存在非致命 EGL `0x502` 初始化提示；实际双相机图像已检查。退出时上游统计线程可能提示 ROS context 已失效，验收脚本仍正常退出且无残留仿真进程。
- 普通用户无法启用 FIFO 实时调度；ROS Python 1 ms 定时器不是硬实时保证。窗口模式偶见动作到达同一仿真时间戳被拒绝，不能声称没有消息丢弃或调度延迟。
- 构建包含上游 CMake / setuptools 弃用警告；不等同于构建失败。
- 源码变更尚未提交；`runs/` 与环境目录由 Git 忽略，保留在本机。

## 重跑

```bash
pixi install --locked
pixi run test
pixi run test-model
pixi run ros-smoke
pixi run policy-smoke
pixi run record-smoke
pixi run iros-stage0
pixi run i002-smoke
pixi run sim-nvidia
```

---

## 2026-09-25 历史记录（不代表当前未完成项）


日期：2026-09-25。基线提交：`b2ff1b4ea033b758433a4535addf14f8e6b0d1ef`。

## 已完成

- 纯 Python 合约与回归测试：24 项通过；默认不要求模型或 ROS。
- 模型集成测试：2 项通过，验证生成的 MJCF、机器人动作响应、重置和时间戳检查。
- 固定模型 checkout 核验通过；MuJoCo Python 与其 native runtime 均为 3.10.0。
- 无窗口控制验收：2 秒物理仿真、60 个策略目标，状态有限且关节响应动作。
- EGL 渲染：外部与腕部 RGB 均为 640×480。
- LeRobot v0.6.1 固定提交录制：两段各 30 帧，合计 60 帧；逐帧读回并解码两路视频，
  验证规范化动作、状态有效性、动作与源时间戳、回合重置后的时间线。
- Pixi 锁文件重新解析成功，linux-64 / linux-aarch64 的 Python MuJoCo 和
  `libmujoco` 均锁定 3.10.0。

录制验证使用 Linux x86_64、Python 3.12、CPU PyTorch 2.7.1、torchvision 0.22.1、
PyAV 15.1.0 与 EGL；这不构成 Jetson/CUDA 训练兼容性证明。

## 未完成的本机验收

尝试了完整 `pixi run build`，环境安装阶段下载
`https://prefix.dev/conda-forge/noarch/python-gil-3.12.14-hd8ed1ab_3.conda`
失败，报错为网络代理 tunnel connection failure。因此未能在本次环境中编译或启动
完整 ROS 栈，也未验证真实 DDS 定时、控制器激活和相机发布。
这不是已经确认的代码编译失败；也不能把它视为编译通过。

目标机器请执行 README 的安装、运行时验证和 ROS 验收步骤。
本次验证不包含真实机器人、学习策略训练或任何任务成功率。
