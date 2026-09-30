# Reviewer 审查

- 数据泄漏：normalizer只取train T/q/action；episode seed分区隔离，无future window，先observe后action。teacher privileged位移/接触速度只产生action label，无额外预测目标。输入42维白名单和spy测试核对。
- 输入局限：taxel由sim contact geometry生成，是显式仿真观测模型，不是真实触觉。没有将force当student输入。
- baseline公平性：M0/M1/M2同初始化、same-step minibatches、optimizer、步数、hidden、参数量 [279041]；M0重复current保证相同attention算量。缺少adaptive current控制的更强baseline可能影响广泛方法比较，但constant-max已提供简单替代解释。
- temporal leakage：包含当前T/q及a_(t-1)，不含a_t；causal mask和未来扰动不影响过去latent的测试通过。过去token打乱保留当前token。
- seed contamination：train、validation、main test、ambiguity seed分区不同；development范围修改只发生在正式训练前，并保存开发结果。初始化共享数据，CI不把它们当独立采集。
- privileged leakage：qvel/slip/friction/object pose不进入学生；teacher-only标签是明确授权的privileged imitation。ambiguity按teacher动作差值挑诊断pair，属于已声明的诊断定义，不是模型选择；因此不是总体random subset。
- cherry-picking：包含所有主评估seed/条件，保存失败与全部checkpoint曲线；pair阈值不因test结果放宽。匹配不足不补挑成功例。
- test-set tuning：固定末步checkpoint，不依validation/test选模；FROZEN记录checkpoint/config/source hashes；run resume验证冻结源代码。test只评估一次，恢复跳过原子完成的条件。
- 统计：paired two-way bootstrap，3个training initializations和4个disturbance seeds仍小，不证明等效性；多个诊断未做家族多重性校正，positive结果仍需独立确认。
- 主要科学风险：恒定最大收紧足够强，没有硬性force预算；teacher是启发式，不能称最优动作；匹配对0个，初始化失败0。仅单任务、几何触觉、有限训练，可能欠拟合/BC covariate shift。
- 实时性：200Hz为仿真时间控制，不等于batch GPU evaluation实现了真机5ms最坏时延。闭环每步等待推理，没有异步丢包/USB模拟。

审查结论：NO_SIGNAL，保持Pilot边界；决策REVISE。详细原始证据见config、manifest、data、eval、ambiguity、metrics.csv与summary.json。
