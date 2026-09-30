# 执行与审计证据

## 自动测试

命令：

```bash
.venv-recording/bin/python -m pytest -q tests/test_tactile_history_control.py tests/test_contact_foundation.py tests/test_i001_v2.py
```

结果：18 passed, 5 deselected。包括新实验的9项测试以及复用模块的相关非integration回归；被排除的是旧integration标记测试。本实验的MuJoCo状态恢复/输入隔离测试实际执行了物理仿真，没有mock物理结果。正式采集和闭环评估另外实际运行了完整MuJoCo任务。

新测试覆盖：参数公平、M0/M1对禁用信息不敏感、causal mask、shuffle值保持、上一条命令时间对齐、真实仿真状态恢复、匹配阈值、seed分区、float32标准化、dataset窗口边界、配对bootstrap（部分合并在同一测试函数）。

## 开发与smoke

- development v1：扰动方向覆盖整个水平面，0.8–3.2N；12个seed，nominal成功11/12，teacher12/12，constant-max12/12。
- development v2：仅改变物理任务为±y各30°扇区、1.5–4N；12个seed，nominal2/12，teacher12/12，constant-max12/12。未用学习模型比较来选任务。
- ambiguity development v1：onset+40ms、最后25ms共同动作，12个seed未找到合格pair。
- ambiguity development v2：onset+80ms、最后5ms共同动作，12个seed找到7对；二者均未在main test上操作，未比较学习模型。
- 第一轮smoke `20260930T023553261553Z-smoke` 在训练入口发现numpy标准化输出float64、Torch模型float32不匹配；原失败日志保留。修复为显式float32，增加dtype测试。
- 第二轮smoke `20260930T023803123901Z-smoke` 完整跑通采集、训练、闭环控制、shuffle、匹配诊断与图表/报告。仅4个test seed、3个train steps，不作为研究结果。

正式Pilot开始后未根据测试表现修改任务、模型、训练步数、checkpoint、匹配阈值或任何实验源代码。

## 额外只读审计

- `training-audit.json`：9个模型均279041参数、600步；88个采集episode；全部dataset token的最后action通道对应上一实际下发命令；冻结源码、checkpoint、normalizer哈希匹配。
- `observation-audit.json`：实际输入42维；训练/验证集在±8标准化裁剪前越界元素比例分别约0.055%/0.045%。裁剪是预先固定实现，不根据test修改。
- `model-assets-sha256.json`：44个实际使用的模型资产文件SHA-256，补充scene/model XML的源码manifest。
- `pip-freeze.txt`、`nvidia-smi.txt`、`environment.json`：实际依赖和硬件记录。

稳定性窗口采用200Hz的20个连续稳定观测样本，首末样本相隔95ms；指标以20×5ms记为100ms窗口。因此恢复时延具有5ms采样分辨率，不代表连续时间接触稳定性证明。

`environment.json` 的 num_workers=4 指数据采集进程数；训练窗口预先装入GPU，没有DataLoader后台worker。训练为float32、dropout=0；使用固定步数和按(seed,step)独立生成的batch索引恢复，不承诺跨GPU/跨PyTorch版本位级一致。

完成后的 `run --resume` 已真实执行通过：跳过完成的采集/训练/评估阶段，仅重建报告；冻结检查点与normalizer保持一致。随后执行 `audit_postprocess.py` 重新附加补充只读审计。正式主评估1008个rollout、匹配分支126个rollout均通过完整seed网格核对，见 `final-audit.json`。
