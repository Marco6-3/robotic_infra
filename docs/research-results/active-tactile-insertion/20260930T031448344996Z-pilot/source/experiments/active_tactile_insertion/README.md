# 主动触觉插入 Pilot

核心问题：盲插入时，完整T/q历史之上加入past issued action，能否提高安全插入成功率？保留q作为强baseline，随机probe本身不能保证动作信息独立。

三轴Cartesian夹持工装、圆截面带圆导入端peg、分片近似圆倒角孔；双指内侧2×4×4几何压入量触觉代理。不是完整FR3/ROS闭环，也不是真实光学触觉。主policy只有Δx/Δy，固定向下preload，以深度、8N安全力和时限联合判断成功。

[PROTOCOL.md](PROTOCOL.md) 定义冻结协议；[LITERATURE.md](LITERATURE.md) 明确参考论文与本假设的区别。旧 `tactile_history_control` 实验不改动。

```bash
# 仓库根目录；复用已安装torch/CUDA/MuJoCo/matplotlib的环境
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m pytest -q tests/test_active_tactile_insertion.py
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.active_tactile_insertion.run --smoke
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.active_tactile_insertion.run
# 恢复输出路径；不允许修改冻结实验源码/配置/数据
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.active_tactile_insertion.run --resume /absolute/run
```

默认：80 train、16 validation、48 test task seeds、3 training seeds、800 steps。结果输出 `docs/research-results/active-tactile-insertion/<timestamp>/`，包括RESULTS、CLAIM、NEGATIVE_RESULT、REVIEW、诊断、模型及原始轨迹。

扩大预算的全新run：`python -m experiments.active_tactile_insertion.run --full-scale`。该命令不自动执行，不意味着已有证据支持扩大规模。
