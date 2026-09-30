# Tactile history 闭环控制 Pilot

本实验比较 M0 当前 T/q、M1 历史 T/q、M2 历史 T/q/已下发 action。统一小型 causal Transformer，不扩展 ContactBelief，也不做 backbone 搜索。

实际复用 MuJoCo FR3 抓取/抬升和双指内侧的简化压入量 taxel map；不是 RGB 触觉或真机结论。详细输入边界、teacher权限、预先定义的决策门槛和统计方法见 [PROTOCOL.md](PROTOCOL.md)。

从仓库根目录运行（复用已安装 torch/CUDA/MuJoCo 的独立环境）：

```bash
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m pytest -q tests/test_tactile_history_control.py
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.tactile_history_control.run --smoke
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.tactile_history_control.run
# 使用程序打印的绝对 run 路径续跑；不能修改配置/实验源代码：
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python -m experiments.tactile_history_control.run --resume /absolute/run/path
```

默认正式运行仍是 reduced-scale Pilot：72 train / 16 validation / 48 held-out tasks，3 training seeds，600 optimizer steps。smoke 使用独立种子和极小预算，其结果不是科学结论。输出放在 `docs/research-results/tactile-history-control/<timestamp>/`。

模块：environment（任务/传感/teacher），dataset（behavior collection），policy（模型），train（固定步数与可恢复checkpoint），evaluate（真实闭环），ambiguity（匹配状态分支），ablation_runner（全部控制），analysis（统计/图/四份报告），run（生命周期）。

独立阶段可执行 `python -m experiments.tactile_history_control.{dataset,train,evaluate,ambiguity,analysis} RUN`；正式评估前应由 run 创建 FROZEN.json。扩大规模需新run/new seeds，不能复用此Pilot test为最终独立验证。`--full-scale` 提供一组明确加大的复现实验预算（不会自动执行，且不意味着已获准进入论文评估）。

绘图使用 `matplotlib`（本次运行安装 3.11.2）；完整实际依赖列表保存在每个 run 的 `pip-freeze.txt`，GPU/驱动信息见 `nvidia-smi.txt`。标准化使用训练集均值/标准差，按传感量设置尺度下限，并裁剪到 ±8；参数固定在 `dataset.py` 和 `policy.py`，不由测试集调节。
