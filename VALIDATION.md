# 修复验收记录

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
