# ContactBelief-v0 方法实验

检验显式预测未来交互观测的训练目标，能否优于普通历史编码。与 I001-v2 分离，后者所有代码、数据和正式结论保持只读。

完整流程：复用 train/validation → 相同预算训练五种方法 → validation 选模 → 冻结模型 → 只读表征 probe → 冻结 probe → 新 FINAL seeds → 评分/置信区间/机制诊断 → GO/NO-GO。

```bash
cd /home/mingzhe/Documents/ws/robotic_infra
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-recording/bin/python -m experiments.iros2027.contactbelief.run
```

恢复已有运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-recording/bin/python -m experiments.iros2027.contactbelief.run --resume /absolute/path/to/run
```

使用已安装的项目 PyTorch CUDA / MuJoCo 环境，无需修改 Pixi 或 I001-v2。每个 run 保存配置、源文件快照、环境、全部训练曲线与 optimizer checkpoints、FROZEN.json、PROBES_FROZEN.json、FINAL_SEEDS.json、新 episodes、指标、bootstrap、预测、表征分析、参数/算量/latency、图和复现命令。

`PROTOCOL.md` 定义完整假设、输入边界和 gate。B2 是 lambda=0 的严格 GRU 对照；B4 只增加辅助预测目标。旧 test 不作为最终证据。所有输入均为触觉几何代理与 q/dq/command，不包括 RGB 或 privileged simulator state。

主要产物为 `RESULTS.md`、`DIAGNOSIS.md`、`METHOD_CLAIM.md` 和 `GATE.json`；未过 gate 时生成 `NEGATIVE_RESULT.md` 并停止方法升级。离线通过后才允许单独实施 closed-loop，不把离线分数解释成控制能力。
