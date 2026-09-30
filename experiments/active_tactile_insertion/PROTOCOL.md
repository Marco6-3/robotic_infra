# Active tactile insertion：冻结前 Pilot 协议

## 科学问题

能否从主动触觉交互历史推断安全插入所需的二维修正运动？核心比较 M2−M1，即给定完整 tactile/proprioceptive history 后，past issued action 是否提高闭环安全插入成功率。次要比较 M1−M0。强替代解释：q历史已经包含探测方向/位移，当前接触分布足够，或顺应性导致被动对中。

随机probe不保证action独立信息：若q历史足以还原所下发动作，给出a不会增加条件信息。必须保留完整q，不能用删除q、改坐标系、隐藏已测位移或人为动作映射来让M2获胜。

## 物理边界

采用MuJoCo三轴Cartesian夹持工装，隔离运动规划和7DoF IK。不是完整FR3 ROS执行，也不是真机验证。q顺序固定 `carriage_x, carriage_y, carriage_z`，单位m。夹持peg通过三个被动弹性自由度连接到工装，夹爪内侧左右两个pad与peg真实接触；被动mount位置不进入student。

圆截面peg半径5mm、半球导入端半径5mm（胶囊碰撞形状）；圆孔半径5.4mm，由24个凸几何片近似，倒角宽1.2mm、深1.5mm。目标tip插入深度8mm，使圆柱段越过倒角喉部约1.5mm。接触模型 solref=.015 1、solimp=.9 .95 .0005，属于软接触近似，不宣称rigid精密装配或材料真实刚度。

hole offset、摩擦、peg质量、motor刚度按config随机；隐藏孔中心不进入student。1kHz物理，200Hz观测/控制。350ms接触准备后，300ms随机probe；probe阶段用同一encoder-based竖直限位目标，防止策略接管前已完成插入；主控制阶段固定1.5N向下预载，student仅输出Δx/Δy。原始上下游实际接触是否发生作为diagnostic记录，不丢弃未接触episode。

安全力为shaft对hole所有接触点三维接触力模长之和，1ms逐步监测，超过8N立即终止为failure。不是只检查200Hz采样值，也不把夹爪正常夹紧力计入插入力。成功要求深度≥8mm、横向误差<0.4mm、连续20ms保持，且从准备阶段起从未超力，并在主阶段1.6s内完成。成功/超力时停止；其余timeout（卡滞诊断不等同于已知真实contact mode）。成功后不继续施力。

强力baseline把向下驱动提高为12N，但依然使用8N实际接触力标准；如果在无明显接触负载时安全进入，仍记成功，不能为了“暴力必失败”改标签。

## 传感和动作契约

T为双指内侧pad/shaft接触几何的2×4×4压入量map，加入0.5μm噪声、2μm量化。它不是力场投影，不使用mj_contactForce生成T，也不是RGB光学触觉renderer。q为三轴实际encoder位置，5μm量化，无手工dq/ddq；每token为32 T +3 q +2 previous issued commands，共37维。

动作u∈[-1,1]^2，Δxy=40μm·u，每5ms更新有界位置目标。history记录workspace裁剪后的真实下发增量，不是teacher label，也不是实际达到的位移。实际目标和执行器内部力不输入student；q历史可隐式估计运动，此信息不剥夺。动作饱和边界所有模型一致。

teacher只通过privileged hole center和peg tip position得到K误差修正，K=.15并限幅；在接触力超过0.8F_safe时把横移缩为1/4。教师特权只生成动作label及evaluation，不筛选有利训练样本。环境安全终止使部分episode没有动作target，须记录实际数量，不能补抽有利seed。

## 随机probe和三种baseline

每段probe随机方向、0.2–1.0mm目标偏移幅度、30–90ms持续时间及顺序；受同一每步动作限幅，故实际达到幅度可能更小，保存真正下发命令，不把计划幅度当实际运动。

M0重复当前T/q至61 tokens，action通道置零；M1保留61步T/q且action置零；M2保留T/q/a。均同输入投影大小、2层128维4头causal Transformer、256FFN、零dropout、相同2维tanh头，参数量严格相同。固定窗口300ms，一个时刻一个token。

共享80个behavior episodes，teacher加少量独立执行噪声/随机探索，输出teacher动作MSE；16 validation仅记录MSE。三训练初始化17/29/43，同初始参数、每step样本索引、AdamW lr3e-4、batch64、800 steps，固定最终checkpoint。无模型选择、辅助loss、world model或架构搜索。normalizer仅使用train可见输入，物理尺度下限、±8clip固定。

## 评估与统计

48组全新hole offsets与probe sequences，IID来自同范围（不宣称超出训练范围泛化）；全部方法共用任务seed。报告safe success、force failure、timeout、tip depth、lateral error、peak force、search path和latency（失败用horizon右删失）。多个初始化共享同训练数据，不当作独立采集重复。

条件：M0/M1/M2，M1和M2 past temporal shuffle（当前token不动），M2 action shuffle（全61个action二维向量同序打乱，T/q不变）；以及teacher、nominal、constant+x、预定spiral、brute-down。所有learned controls实际闭环执行。

配对bootstrap同时重采样training seed与task seed，4000次，95%CI。A、B主比较均要求平均提升≥5pp、CI下界>0、三个训练seed方向均为正。B还要求M2−action-shuffle满足同一门槛，并在至少10个完整T/q历史匹配pair上提高safe success。全部必要条件通过才GO，部分证据不升级为论文评价。失败为NO-GO/REVISE；“未证明”不等于效果为零或等效。

## 主动构造的history歧义诊断

另取24个独立seed，每seed生成镜像hole offset及镜像probe command的一对真实physics轨迹；不交换/镜像传感器channel，不丢弃q。保存原始MuJoCo状态与61步history。所有48个构造状态都分支M0/M1/M2真实闭环，同时报告是否真的匹配。

匹配在整个61步T/q上进行：taxel尺度10μm、q尺度50μm，RMS≤1、L∞≤3，teacher action cosine≤−0.5；prefix不可已成功/超力。阈值在test前固定。仅T相似而q不同，不合格；teacher是heuristic，不能称最优动作差。若不足10对明确未建立B的关键诊断，不扩大容差、不挑模型赢的pair。

额外只读diagnostic：在train用3步q增量ridge拟合上一issued action，在validation报告R²及大于0.1分量的方向准确率。该finite difference只在diagnostic，不加到任何student输入；无teacher/hidden-state特征。用于检查M1能否从q知道运动，不能单独证明完整信息等价。

## 开发、停止与审计

平底圆柱/较硬接触在开发种子出现碰撞边界瞬态力尖峰；未开始学习即修正为圆导入端和软接触，并加入统一probe竖直限位。低motor刚度使nominal接近全成功，因此保留较高刚度；未用M0/M1/M2成绩选物理配置。完整开发结果保留。开发中8对镜像样本未匹配完整T/q，因此不预先宣称任务必然需要action。

默认为Pilot。工程安全/teacher可行性检查通过后只运行上述固定预算；test后不得修任务/改checkpoint追求正结果。保存config/protocol、源码/hash、Git信息、依赖硬件、数据/动作/轨迹、checkpoint、force-failure、所有negative结果。每200训练步可恢复；eval按完成条件恢复；异常保留。新任务保留旧grasp实验。
