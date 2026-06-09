# 扫雷模型指标速查

## 训练时

**Train / Val Loss（MSE）**
概率蒸馏使用 MSE loss。衡量预测概率与求解器真值的偏差。
- loss = 0.001：预测概率与真值平均偏差约 3%（极好）
- loss = 0.010：偏差约 10%（还在学习）
- 越低越好，0 = 完美

**Val Acc（验证准确率）**
covered 格子中 (P>0.5 vs 真值是雷) 的二元准确率。>95% 正常。

**Act Acc（动作准确率）**
每步选 P(mine) 最低的 covered 格，该格是否安全（不是雷）。
- 0.998+ = 每 1000 步错 2 步（优秀）
- 这是**最重要的训练指标**

**Ponder Penalty** (已废弃)
自适应训练中的惩罚项。PonderNet 已于 2026-06-02 废弃，改用固定步长 BPTT。

## 评估时（evaluate.py）

**Won / Lost**
- Won：翻开全部安全格、未中雷
- Lost：踩雷
- 无 Stuck（超时情况极少）

**Action Accuracy**
全局动作准确率 = 安全翻开数 / 总翻开数。0.999 = 每 1000 步错 1 步。
胜率近似受 `action_accuracy ^ avg_steps` 控制，因此高密度长局对单步错误更敏感：
- S1 8×8/10: `0.9994 ^ 17.9 ≈ 99%`
- S4 8×8/25: `0.9981 ^ 23.6 ≈ 95.6%`
- S4 若要接近 99% WR，action_acc 需要约 `0.9996`

**Win Rate**
Won / Total。最直观的最终指标。受样本量影响（10 局可能波动大，建议 100+）。

**Rule-Guard Actions**
启用 `scripts/evaluate.py --rule_guard` 时，评估会优先选择 `ConstraintSolver` 可证明安全的格子，并统计这类动作数量。该指标用于诊断：
- guard 后胜率显著提升：裸模型仍会在基础/子集规则上偶发排序错误
- guard 后胜率变化很小：主要瓶颈在高阶概率排序或训练目标

带 `--rule_guard` 的结果应单独记录为辅助框架成绩，不与裸模型 win rate 混记。

当前 S5 诊断基线：
- 裸模型: 8×8/32, 186/200 WR = 93.00%, action_acc=0.9961
- 裸模型 500 局: 457/500 WR = 91.40%, action_acc=0.9952（评估期间 cache 从 200 扩到 500 boards）
- `--rule_guard` 500 局: 491/500 WR = 98.20%, action_acc=0.9990, `rule_guard_actions=8194`

## Failure Mining（collect_mistakes.py）

`scripts/collect_mistakes.py` 使用裸模型 rollout，不启用 `--rule_guard`；solver 只用于诊断和生成 target。输出分两部分：
- `.npz`: 与 `TrajectoryPool` 兼容的单步 hard-example trajectory，可作为后续 replay source
- `.json`: 诊断摘要和每个保存样本的元信息

错误分类：
- `rule_guard_avoidable`: `ConstraintSolver` 已有可证明 safe cells，但模型没有选这些 safe cells
- `hard_sorting`: 当前没有可证明 safe cells，模型直接点雷
- `calibration_drift`: 模型没有点雷，但选择的 solver target 明显差于当前最优 target

推荐先用固定 eval cache 运行诊断，再决定是否混入训练：

```bash
PYTHONPATH=src uv run python3 scripts/collect_mistakes.py \
  checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
  --output data/mistakes/S5_rule_guard_failures.npz
```

当前 S5 mining 结果：
- 裸模型 rollout: 457/500 WR = 91.40%, 8909 steps
- 保存错题: 441 states
- `rule_guard_avoidable`: 435
- `hard_sorting`: 6
- `calibration_drift`: 13（仅统计，不保存到默认 NPZ）

该分布说明下一步应优先做小比例 hard-example replay，目标是内化 `ConstraintSolver` 已能证明的 safe-cell 排序；lookahead search 暂时排在后面。

S5 mistake replay fine-tune 后：
- S5 裸模型: 484/500 WR = 96.80%, action_acc=0.9982
- S5 `--rule_guard`: 494/500 WR = 98.80%, action_acc=0.9993
- S5 after-mining: 388 saved states, `rule_guard_avoidable=382`, `hard_sorting=6`
- S1 回归: 197/200 WR = 98.50%
- S4/S25 回归: 192/200 WR = 96.00%

解读：
- hard-example replay 显著提升裸模型排序能力，S5 裸胜率 +5.4pp
- `hard_sorting` 未下降，说明剩余极难错误不是当前微调主要解决对象
- 后续二次微调应保守提高 mistake weight 或降低 lr，避免牺牲 S1/S4 泛化

S5 mistake replay fine-tune v2 后：
- S5 裸模型: 486/500 WR = 97.20%, action_acc=0.9985
- S5 `--rule_guard`: 496/500 WR = 99.20%, action_acc=0.9996
- S5 after-mining: 381 saved states, `rule_guard_avoidable=377`, `hard_sorting=4`
- S1 回归: 197/200 WR = 98.50%
- S4/S25 回归: 192/200 WR = 96.00%

解读：
- 第二轮 replay 仍正向，但裸模型仅 +0.4pp，进入平台期
- `rule_guard` 已达 99.20%，辅助框架路线接近 99%+
- 继续同构 replay 的预期收益较低；当前成功基线固定为 `checkpoints/v5_replay_S5_mistake_ft2/best_model.pt`

## Solver-Safe Ranking

`deep_mse_solver_safe_rank` 是 hard-example replay 专用 loss：

```text
deep_mse_solver_safe_rank =
  deep_mse
  + rank_loss_weight * best_safe_rank
  + rank_loss_weight * solver_safe_set_rank
```

其中 `solver_safe_set_rank` 只在样本携带 `solver_safe_masks_*` 时生效。该 mask 由 `scripts/collect_mistakes.py` 调用 `ConstraintSolver` 生成，表示当前状态下所有可证明 safe 的 covered cells。

目标：
- 原 `deep_mse_rank`: 至少一个 target-safe cell 排在 non-safe cells 前面
- 新 `solver_safe_set_rank`: safe set 里的最低 logit 低于 safe set 外的最低 logit，使裸模型 argmin 落入可证明 safe set

注意：最初的全 pairwise 版本“所有可证明 safe cells 都排在 unknown cells 前面”在 S5 上出现负优化，说明该约束过强，会扰动概率校准。当前实现改为 set-min objective，直接对齐动作选择。

当前实验状态：
- 全 pairwise 版本: S5 500 局 480/500 WR = 96.00%，低于 `mistake_ft2`
- set-min 版本: 训练内 100 局 91/100 WR = 91.00%，仍为负优化
- 结论: 暂停 solver-safe ranking 路线；当前最佳仍是 `deep_mse_rank + mistake replay v2`

旧版 mistake NPZ 不包含 `solver_safe_masks_*`，使用该 loss 前需重新运行 failure mining 生成新版错题文件。

## RL 训练时（已从 main 移除）

> RL 代码已从 main 移除，以下指标仅供历史参考。

**ret（Return，总回报）**
一局总分来自即时奖励累加：
- 安全动作：`+1.0`
- 额外 floodfill：`+0.05 * (cells_revealed - 1)`
- 踩雷：`-20.0`
- 通关：无额外奖励

mine_continue 模式下探索期 ret 通常为负；若从监督 checkpoint 微调，理想情况是 eval_wr 不明显回退，ret 缓慢上升。

**loss（REINFORCE loss）**
策略梯度损失。-log(π(action)) × advantage。
- 正值：模型因坏决策被惩罚（正常，前期）
- 负值：模型做对了，但 baseline 追上了（后期常见）
- 绝对值 < 1：稳定训练
- 绝对值 > 5：不稳定，需降 lr 或调温度

**eval_wr（评估胜率）**
每 100 局评估一次，确定性推理。追踪 RL 是否真正提升了能力。

**baseline（EMA 基线）**
指数移动平均的 return。当前实现中它是日志诊断指标，不参与 advantage 计算。应跟随 ret 趋势移动。

## 关键概念

**概率蒸馏**
训练模型输出精确的 P(mine)，而非硬标签（是/否雷）。
优势：保留不确定性信息、支持模糊局面、动作选择更合理。

**迭代 Refinement**
模型输出回灌给自己，检测矛盾、修正推理。
- 训练：固定步长 4, 全 BPTT (无 detach)
- 推理/评估：收敛早停 `max|P_t - P_{t-1}| < 0.01`，上限由 `POLICY.refinement.eval_max_steps` 控制
- V5 架构：constraint residual channels + prev_probs 显式反馈

**收敛步数（refine steps used）**
评估时实际执行的 refinement 次数。V5 不再输出未监督 confidence，提前退出只看
`max|P_t - P_{t-1}| < convergence_eps`。
