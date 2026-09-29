# ContactBelief-v0 负结果 / 未通过项

generic recurrent history is sufficient in this experiment; predictive-state training has not shown an independent benefit.

失败项：gate1_predictive_objective, gate2_generic_temporal, gate4_shortcut_checks。

未继续扩大模型、增加层数或搜索更多 lambda；未进入 learned closed-loop。保留全部候选、日志、checkpoint、正负结果。

该判断限定于当前数据、模型族、训练预算和受控任务；不证明 predictive objectives 在所有任务中无用。
