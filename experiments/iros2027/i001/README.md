# I001：受控接触观测混叠实验

从现有 FR3 disturbed grasp 出发，检验当前双指力代理相近的状态，在相同后续输入下
是否产生不同滑移结果，以及历史是否提供额外信息。完整假设、排除条件和反证标准见
[PROTOCOL.md](PROTOCOL.md)。这是单个任务的机制实验，不是多任务 benchmark 或 SOTA。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-collect
OPENBLAS_NUM_THREADS=1 pixi run i001-evaluate runs/i001-aliasing/<运行目录>
```

- 采集默认四个 CPU 进程，约 456 个 episode；不训练 GPU 网络。
- 模型为 NumPy RBF kernel ridge，验证集选择超参数和历史长度，模型系数保存到 `models/`。
- `--resume <运行目录>` 恢复完整 episode 检查点，源文件与配置必须一致；半个 episode 重跑。
- 独立分析可重复执行，每次新建 `analysis-*`，保留以往分析。
- `manifest.json` 记录源快照、配置、资产和每个 episode SHA-256。
- 每个 NPZ 包含 2.4 秒之后的 200 Hz 观测、q/dq、控制，1 kHz 特权标签，anchor 仿真状态、未来输入。
  分析只取 anchor 及以前的特征，隐藏物理参数和未来标签不能进入模型。
- 输出 `REPORT.md`、`summary.json`、`predictions.csv`、全部近邻候选 `alias_pairs.json`。

绘图是可选分析步骤，在具有 NumPy/Matplotlib 的 Python 中运行：

```bash
python3 -m experiments.iros2027.i001.plot runs/i001-aliasing/<运行目录>/analysis-<时间>
```

图输出 PNG/PDF；本项目 Pixi 的仿真运行不依赖 Matplotlib。

本机 2026-09-29 的完整结果、反例和后续缺口见 [RESULTS.md](RESULTS.md)。
原始数据独立审计：`pixi run python -m experiments.iros2027.i001.audit RUN_PATH`。
