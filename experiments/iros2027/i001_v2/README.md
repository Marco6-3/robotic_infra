# I001-v2 Active Contact Probe

本目录是独立、可恢复的 controlled mechanism experiment，不覆盖 contact、I001-v1 或原运行结果。
主检验固定为 **B4（完整交互历史）vs B1（完整当前触觉＋本体＋动作）**。
完整科学假设、观测模型、配对、数据停止与反证规则见 [PROTOCOL.md](PROTOCOL.md)。

## 一条命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-v2
```

流程：train/validation 采集 → 数据验证 → 模型选择与冻结 → 生成独立 test →
按匹配对数量扩展 → 全量审计与精确重演 → 预测/聚类 bootstrap → 机制诊断/图表 → 全部 regression tests。

正式配置固定 120 个训练 seed、40 个验证 seed。模型冻结后从新种子生成 test，
直到相反结局匹配对 ≥1000 且覆盖 ≥100 个独立 seed，或总计达到 20,000 episodes。
没有为了使历史获胜而继续调参的循环。

长时间运行可由终端保持，或自行使用 nohup：

```bash
mkdir -p runs/i001-v2
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-v2 > runs/i001-v2/launch.log 2>&1 &
```

启动日志打印 `RUN=...`。运行中读取该目录的 `progress.json` 和 `events.jsonl`。
每个 episode 原子写入；模型逐个保存检查点；锁阻止同时恢复同一运行。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-v2 \
  --resume /绝对路径/运行目录 --config /绝对路径/运行目录/config.yaml
```

恢复要求源文件和配置哈希不变。已完成 episode 不覆盖；损坏、来源变化、test 提前出现会拒绝继续。
完整结束的 run 再次 resume 只返回完成状态，不重新生成结果。

## 数据与输出

- `manifest.json`、`git.diff`、`source/`：Git 状态、配置、源/资产哈希、环境、精确命令与 seed。
- `episodes/{train,validation,test}/*.npz`：完整 200 Hz 观测与 source/available timestamps，
  1 kHz commanded action/q/dq，历史及挑战阶段 1 kHz privileged labels（更早阶段 200 Hz）。
- `exact`/`finite` 的列顺序：双指各 4×4 几何触觉（32维）、机器人 q（9）、dq（9）、命令（8）。
- 触觉是接触几何缩进代理；`mj_contactForce` 只生成标签，不进入 policy feature。
- `controls`：tick、七轴目标＋夹爪宽度、q、dq。
- `labels`：tick、接触切向相对速度、contact count、object position、object−TCP、位移、drop、
  双指真实力六维、object spatial velocity 六维、object quaternion 四维。
- `FROZEN.json`、`models/`：生成 test 前冻结的划分与超参数、可恢复模型。
- `metrics.json`、`bootstrap.json`、`pairs.csv`、`predictions.csv`：包括全部 label-blind 候选对，
  `opposite` 仅为配对后的标签注释，不是距离搜索输入。
- `RESULTS.md`、`DIAGNOSIS.md`、`FIRST_LAYER_CLAIM.md`：正式结果、替代解释、允许使用的限定主张。
- `figures/`、`audit.json`、`regression.json`、`COMPLETE.json`、`REPRODUCE.txt`：图、验证与复现入口。

仿真/训练使用现有项目 Pixi 的 MuJoCo、NumPy；CPU 六进程生成，不需要 GPU 训练。
图表调用本机已有 `/usr/bin/python3` 的 Matplotlib。A/B 精度设置没有真实硬件标定，
没有 RGB 输入、learned controller、新架构、跨物体或真机有效性主张。
