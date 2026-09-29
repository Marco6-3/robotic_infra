# I001-v2 预注册协议（正式采集前冻结）

唯一主假设 H1：当前 tactile + q/dq + current action 在指定观测分辨率下匹配时，
过去 action + tactile + proprioception 仍能在未见 physics seeds 上提供未来滑移预测信息。
主比较 **B4 vs B1**；B3 vs B2 为动作条件化的独立检验。阴性结果完整保留。

## 任务与数据

单个 FR3 双指方块抓持，1 kHz MuJoCo，200 Hz 传感采样，300 ms 历史。
每 seed 固定质量（63–67 g）、物体几何、接触 solver time-constant 缩放（0.9–1.1）、未来输入。
20 个变体独立采样初始平面偏移 ±0.3 mm、probe 幅度/方向/时刻/持续时间。
其中 10 个变体固定摩擦 0.54，另 10 个连续均匀摩擦 0.45–0.80；没有离散标签 friction bin 输入。
固定摩擦的选择仅来自 development seed=800000：候选 0.54/0.56/0.58 中
0.54 的十个探测变体有 8 阳性、2 阴性，另两值均 0 阳性；只检查任务标签与安全性，未训练模型。

主动 probe 是 nominal width → ±0.5–1.5 mm → nominal width，梯形轨迹。
起点在 anchor 前 170–275 ms，持续 40–100 ms；方向 tighten/loosen 等概率。
可叠加 0–0.3 N 向下外力。anchor=2.6 s，之后不再 probe；同 seed 的未来命令和
1.5 N 外力轨迹（20 ms 上升）逐元素相同，挑战持续 100 ms。

anchor 夹持条件：高度 >0.49 m，两指法向力 >0.15 N，接触切向速度 <0.003 m/s，
相对 history 起点位移 <15 mm。它仅用于判断任务是否处于有效抓持阶段，统一应用于全部 episode；
**邻居搜索本身绝不读取这些 privileged 值**。报告全部排除样本。
未来标签为 anchor 后 100 ms 内接触切向速度连续 5 ms >0.01 m/s。

## 观测模型（不是硬件规格）

触觉输入为两指内侧局部 x/z 网格的 4×4 penetration-depth 几何代理：
接触位置高斯 splat（sigma=6 mm）与穿透深度的逐格最大值。只用接触几何，**不读取接触力**。
该代理不模拟真实弹性皮肤，不是 RGB、DIGIT、压力/力传感器。真实力、object pose/velocity、
摩擦只进入标签、夹持有效性判断和事后机制分析，不进入特征、归一化、超参数选择或邻居距离。

A exact：未添加传感噪声/量化的 simulator geometry tactile 与机器人 q/dq。
B finite（主 regime）：触觉 5 μm 量化、独立 1 μm 标准差噪声；机械臂 q 1e-5 rad，
指 q 1e-5 m，机械臂 dq 1e-3 rad/s，指 dq 1e-4 m/s 的量化。
命令是已知数字输入，机械臂 1e-5 rad/夹爪 1e-5 m 格点化且配对还要求命令完全相等。
本体不叠加额外噪声；数值在正式采集前固定，不能根据 test 调整。

距离以配置中的每通道 quantum 归一化，要求 L∞ ≤1 且 RMS <0.5；
同 seed 内先仅用当前观测建立候选对，再注释 future label、隐藏参数与物体真值。
qualified pairs 指候选中未来标签相反的对；数量目标不依据模型预测。
exact 用相同距离尺度单独报告；它允许模型利用尺度之内的小信号，所以匹配并不等于数学不可观测。

## 划分与采集停止

development=800000 起；train=810000 起的 120 seeds（2400 episodes），
validation=820000 起的 40 seeds（800 episodes）。两者数量预先固定。
完成验证集选择后保存并哈希全部模型、超参数、协议、划分到 FROZEN.json；
**随后才允许生成 test**，避免读取 test 标签后调参。

test 从 830000 起，最多 840 seeds，20 episodes/seed；全流程上限 20,000 episodes。
每 20 seeds 更新一次 progress、全量计数、阳性率和配对覆盖；最少测试 100 seeds。
有限分辨率 test qualified pairs ≥1000 且覆盖 ≥100 seeds 才按数量目标停止；
否则继续至全局 20,000 上限并报告 benchmark construction failure。
比例不强制为 60/20/20：将剩余预算预先分配给受控配对的独立测试样本。
不根据 test 性能增删样本、重训或更换历史长度。

## 模型与预先固定比较

只使用最多 192 landmarks 的 Nyström RBF ridge，所有 baseline 相同系数预算，
相同核宽 [0.5,1,2]、正则 [0.01,0.1,1] 候选；validation Brier 选择，固定随机种子。
历史模型均 300 ms、200 Hz；没有大型架构搜索，也没有事后上下文长度选择。
各模型统一 61×58 维计算输入：缺失模态填零，当前帧基线重复同一帧 61 次，不增加任何历史信息。
全部使用同一 192-landmark 核路径，以匹配模型参数与矩阵计算预算；核距离采用维度平均平方距离。
另外报告计算输入维度、推理耗时；训练尺度仅来自训练集。

B0 当前触觉；B1 当前触觉+q/dq+action；B2 触觉历史；B3 触觉+动作历史；
B4 触觉+动作+q/dq 历史。N1 保留当前帧、对过去帧共同时间置乱；N2 同 split 跨 seed
随机错配完整过去、保留当前帧；N3 训练标签随机置换。所有当前/历史均严格因果。
随机标签控制不涉及泄漏 test 标签。N2 donor 不可来自同 seed，不得使用保留摩擦排序的循环位移。

## 推断与结论

主指标：Brier 改善(B1−B4)的 physics-seed cluster bootstrap 95% CI 下界 >0；
匹配相反结局 pair 的排序准确率改善(B4−B1)的 CI 下界 >0。
次指标 AUROC、BA、F1、ECE/可靠性 bins、pair accuracy；ties=0.5。
按 seed 聚类 2000 次重采样；模型选择冻结后才一次性评估 test；CI 以已拟合模型为条件。

只有配对数量达标、主预测与配对改善成立、两项历史破坏对照显著变差、固定摩擦同样成立时，
才支持强机制 H1。B3−B2 也有增益才支持更进一步的 action-conditioned claim。
若仅整体历史获益而固定摩擦不成立，则只可报告有限/部分支持并保留摩擦辨识解释。
若主 CI 含零、强当前观测足够、动作无增益、配对构建失败或信息泄漏，不继续调架构追逐胜出。
任务完成时必须输出 negative/partial 结论和未满足的条件，不能把没有支持解释为普遍证伪。
