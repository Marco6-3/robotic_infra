# 2026-09-30 Pilot 证据发布说明

此次公开 disturbed-grasp recovery 和 active tactile insertion 两轮独立实验的代码、测试、正式结果、smoke、开发失败、负结果及补充审计。新研究主线见[执行时触觉反馈与在线动作修正](../research/EXECUTION_TIME_TACTILE_REFINEMENT.md)；该方向尚未实现或训练。

## 公开内容与本地内容

五个run合计公开576个证据文件（约3.63 MiB，不含五份导出清单），包括逐episode评估JSON、诊断配对、训练曲线、数据元信息、配置、协议、图表、环境版本及原始源码快照。`EXPORT_MANIFEST.json`分别记录公开文件的字节数/SHA-256，以及未公开本地文件的字节数/SHA-256。

约138 MiB其他产物保留本地：训练/验证NPZ、完整轨迹NPZ、normalizer、checkpoint/optimizer、物理snapshot及主机进程日志等。未上传权重，也没有删掉或重新生成旧实验。未来timestamp运行目录默认被各组`.gitignore`忽略；本次公开文件经清单明确选入。

克隆仓库即可审核报告、逐episode指标并重新聚合；**不能直接用公开目录恢复训练或重放冻结策略**。原报告中的`--resume`、event replay与audit脚本需要完整本地run和匹配源码。报告中的本机路径保持原始provenance；它们不是通用运行路径。

## 源码冻结与本次兼容性改动

原始运行基于提交`05c2707`及当时尚未提交的实验代码，准确源码在每个run的`source/`，由原始`manifest.json`逐文件校验。发布前已合入远端`aec6c9d`的可选仿真依赖隔离修复。

本次只对两份新增测试的入口增加`pytest.importorskip`，让未安装Torch/MuJoCo的轻量环境跳过这些可选实验测试；未改变测试断言、实验算法、模型、任务或旧指标。当前测试文件与冻结快照因此不同；disturbed-grasp所引用的`i001_v2/physics.py`还包含上述远端修复。原始manifest不重写，当前checkout直接`--resume`旧run会按设计拒绝这些源码差异。

精确恢复需单独checkout原始base `05c2707`，把所选run的`source/`按相对路径叠加到该checkout，并提供完整run及对应模型资产。不要在现用工作区覆盖文件，也不要绕过hash检查。FR3模型资产通过原有生成流程获取并按`model-assets-sha256.json`核对；仅两份顶层XML快照不足以替代其引用的全部资产。新run可使用当前checkout生成，不能将其称为旧冻结run的位级恢复。

## 发布验证

完整ML环境执行：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
  .venv-recording/bin/python -m pytest -q -o addopts='' -p no:cacheprovider
```

结果：**90 passed**，包含新增19项实验测试及现有integration检查。

Pixi环境未安装Torch；ROS的`launch_testing`自动加载插件会在模块级skip时提前结束收集。因此Python回归使用以下显式禁用插件命令，结果为**63 passed, 3 skipped**（三份可选Torch测试模块）；这不是ROS启动测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  .pixi/envs/default/bin/python -m pytest -q -ra -o addopts='' -p no:cacheprovider
```

公开文件与冻结快照核验不需要Torch、MuJoCo或原始二进制：

```bash
python scripts/verify_pilot_evidence.py
```

科学结论沿用两轮原报告：disturbed grasp **NO_SIGNAL / REVISE**；active insertion **WEAK_SIGNAL / NO-GO**（针对past action独立收益），保留history内容的有限正信号。仓库测试通过不增加这些科学结论的强度。


在ML解释器中屏蔽Torch/MuJoCo导入、关闭第三方pytest插件后，按默认非integration选择运行contracts：**50 passed, 3 skipped, 13 deselected**。这验证轻量CI的可选依赖边界；不把skip称为仿真通过。

已只复制公开清单中的文件到临时目录，重算两轮正式Pilot的全部summary指标、配对bootstrap差值及判定，与原始summary在1e-12容差内一致；原始报告未改写。复核命令（需前述ML环境）为：

```bash
env -u PYTHONPATH -u PYTHONHOME .venv-recording/bin/python scripts/reaggregate_pilot_evidence.py
```
