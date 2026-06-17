# 训练记录

当前主线: V5 replay curriculum — S1(8×8/10) → S2(15) → S3(20) → S4(25) → S5(32)。

> **2026-06-06 更新**: V5 constraint channels 从 4 个扩为 8 个，当前模型输入为 19ch；主 loss 为 `deep_mse_rank`；主 recipe 为 `v5_curriculum_replay`。
> **2026-06-16 更新**: 数据布局简化 — 训练数据统一平铺在 `data/` 根目录，命名格式 `{W}x{H}_{M}_{index:04d}.npz`；`TrajectoryPool` 通过 `(H, W, mines)` 三元组自动过滤。旧记录中的 `data/S1` 等路径为历史格式。
> **推理策略**: 训练固定 4 步全 BPTT；默认评估根据 `max|P_t - P_{t-1}| < convergence_eps` 早停，当前 S5 100% 组合显式使用 `--refine_steps 5`。

## 日志格式

| 日期 | 阶段 | 设备 | 超参 | Val Loss | Val Acc | 胜率 | Checkpoint | 备注 |
|------|------|------|------|----------|---------|------|------------|------|

---

## 2026-06-13: 本地 M3 Pro 从零完整复现到 100%

### 目标

验证另一台机器出现的异常表格是否可复现：

| Stage | 雷数 | 对方复现胜率 |
|------|------|--------------|
| S1 | 10 | 75% |
| S2 | 15 | 93% |
| S3 | 20 | 21% |
| S4 | 25 | 14% |
| S5 | 32 | 97% |

### 复现设置

| 项目 | 值 |
|------|-----|
| 设备 | 本地 M3 Pro, PyTorch MPS |
| 编排脚本 | `scripts/run_full_repro.py` |
| Prefix | `repro_mps_20260613` |
| 数据 | canonical `data/`, 每阶段 10000 局 strict no-guess (历史格式: `data/S1` 至 `data/S5`) |
| 基础训练 | S1→S5, `deep_mse_rank`, V5 19ch, batch=64, refine=4 |
| 后续训练 | S5 hard-example replay ×2 → denoise replay ×2 |
| 日志 | `logs/repro_mps_20260613_full_repro_20260613_120132.log` |

### 基础 S1→S5 从零复现

| 阶段 | 继承 | 内部 best | 独立 500 局裸评估 | Checkpoint |
|------|------|-----------|-------------------|------------|
| S1 8×8/10 | scratch | 97.0% | 489/500 = 97.80% | `checkpoints/repro_mps_20260613_v5_replay_S1/best_model.pt` |
| S2 8×8/15 | S1 | 96.0% | 479/500 = 95.80% | `checkpoints/repro_mps_20260613_v5_replay_S2/best_model.pt` |
| S3 8×8/20 | S2 | 94.0% | 457/500 = 91.40% | `checkpoints/repro_mps_20260613_v5_replay_S3/best_model.pt` |
| S4 8×8/25 | S3 | 93.5% | 460/500 = 92.00% | `checkpoints/repro_mps_20260613_v5_replay_S4/best_model.pt` |
| S5 8×8/32 | S4 | 94.5% | 475/500 = 95.00% | `checkpoints/repro_mps_20260613_v5_replay_S5/best_model.pt` |

结论：本地从零训练没有复现 S3=21%、S4=14% 的崩盘。另一台机器的低胜率更可能来自 checkpoint 路径、阶段继承、评估参数或数据目录混用，而不是当前训练路线本身。

### S5 hard-example replay 与 denoise replay

| 步骤 | 数据/继承 | 结果 |
|------|-----------|------|
| base mistake mining | `repro_mps_20260613_v5_replay_S5` | 保存 476 条 replay states |
| `mistake_ft` | base S5 + first mistakes, epochs=2 | internal best 98.5% |
| second mistake mining | `mistake_ft` | 保存 404 条 replay states |
| `mistake_ft2` | second mistakes, epochs=1 | internal best 99.0% |
| `denoise_rank` | `deep_mse_denoise_rank`, epochs=2 | internal best 99.0% |
| denoise mistake mining | `denoise_rank` | 保存 306 条 replay states |
| `denoise_rank_ft2` | final denoise replay, epochs=1 | internal best 99.0% |

### 最终评估

Checkpoint: `checkpoints/repro_mps_20260613_v5_replay_S5_denoise_rank_ft2/best_model.pt`

| 难度 | 评估 | 胜率 | Action Acc |
|------|------|------|------------|
| S1 8×8/10 | naked, 500 boards | 497/500 = 99.40% | 0.9997 |
| S2 8×8/15 | naked, 500 boards | 491/500 = 98.20% | 0.9993 |
| S3 8×8/20 | naked, 500 boards | 480/500 = 96.00% | 0.9985 |
| S4 8×8/25 | naked, 500 boards | 485/500 = 97.00% | 0.9988 |
| S5 8×8/32 | naked, 500 boards | 494/500 = 98.80% | 0.9993 |
| S5 8×8/32 | `--preset s5_guarded_100`, 1000 boards | 1000/1000 = 100.00% | 1.0000 |

最终 S5 guarded 指标：

| 项目 | 值 |
|------|-----|
| Rule-guard actions | 16433 |
| Prob-zero-guard actions | 1841 |
| Avg game steps | 18.3 |
| Avg refine steps | 4.8 |
| Loss/Stuck | 0/0 |

结论：完整路线可在本地 MPS 从零复现到 100%。对方异常结果最优先排查：

1. 是否只跑了基础 recipe，或混用了 legacy `--stage/--all`。
2. S3/S4 评估时 checkpoint 是否指向对应阶段输出，而不是旧目录或错误阶段。
3. `data/S1-S5` 是否为 canonical 目录和命名，且没有旧 `.npz` 混入。
4. S5 100% 是否使用 `--preset s5_guarded_100`，即 `refine_steps=5 + rule_guard + prob_zero_guard`。

## 2026-06-06: V5 19ch + ranking/replay curriculum

### S1 sanity baseline

| 项目 | 值 |
|------|-----|
| Recipe | `v5_curriculum_replay` Phase 1 |
| 架构 | V5 constraint residual, 19ch input, 1ch mine logit |
| 设备 | Mac MPS |
| 数据 | `data/S1`, 8×8/10, 10000 局 strict no-guess |
| 超参 | supervised `deep_mse_rank`, lr=3e-4, epochs=5, batch=64, refine=4 |
| 独立评估 | 198/200 WR = 99.00%, action_acc=0.9994, avg_steps=17.9, avg_refine=3.9 |
| Checkpoint | `checkpoints/v5_S1/best_model.pt`；当前等价主线 phase 输出为 `checkpoints/v5_replay_S1/best_model.pt` |
| 结论 | 19ch hard-constraint channels 将 S1 恢复到旧体系 99% 水平，V5 架构方向成立。 |

### S4 replay result

| 项目 | 值 |
|------|-----|
| Recipe | `v5_curriculum_replay` Phase 4 |
| 棋盘 | 8×8/25, 39.1% 密度 |
| 数据 | `data/S4:0.7,data/S1:0.1,data/S2:0.1,data/S3:0.1` |
| 超参 | supervised `deep_mse_rank`, lr=3e-4, epochs=5, batch=64, refine=4 |
| 独立评估 | 191/200 WR = 95.50%, action_acc=0.9981, avg_steps=23.6, avg_refine=4.0 |
| Checkpoint | `checkpoints/v5_replay_S4/best_model.pt` |
| 结论 | S4 胜率主要受逐步 action error 控制：`0.9981^23.6 ≈ 95.6%`。若要 99% WR，action_acc 需接近 0.9996。 |

### S5 max-density result

| 项目 | 值 |
|------|-----|
| Recipe | `v5_curriculum_replay` Phase 5 |
| 棋盘 | 8×8/32, 50.0% 密度 |
| 数据 | `data/S5:0.6,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1` |
| 继承 | `checkpoints/v5_replay_S4/best_model.pt` |
| 超参 | supervised `deep_mse_rank`, lr=3e-4, epochs=5, batch=64, refine=4 |
| 独立评估 | 186/200 WR = 93.00%, action_acc=0.9961, avg_steps=18.1, avg_refine=3.9 |
| 独立评估 500 局 | 457/500 WR = 91.40%, action_acc=0.9952, avg_steps=17.8, avg_refine=3.9 |
| Rule guard 诊断 200 局 | 196/200 WR = 98.00%, action_acc=0.9989, rule_guard_actions=3324, avg_steps=18.5, avg_refine=3.9 |
| Rule guard 诊断 500 局 | 491/500 WR = 98.20%, action_acc=0.9990, rule_guard_actions=8194, avg_steps=18.2, avg_refine=3.9 |
| Checkpoint | `checkpoints/v5_replay_S5/best_model.pt` |
| 注意 | 500 局裸模型评估期间 eval cache 从 200 boards 扩容到 500 boards；严格对照建议在固定 500-board cache 上复跑裸模型一次。 |
| 结论 | 裸模型在 50% 密度 no-guess 上未崩溃；rule guard 稳定提升到 98%+，说明大量剩余错误来自可证明安全格上的排序抖动。 |

直接从 S5 开始：

```bash
PYTHONPATH=src uv run python3 scripts/train_stage.py \
  --recipe v5_curriculum_replay --start_phase 5 --end_phase 5 \
  --arch V5 --device auto --eval_games 200
```

辅助框架诊断：

```bash
PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5/best_model.pt \
  --arch V5 --n_games 500 --device auto --width 8 --height 8 --mines 32 \
  --rule_guard
```

`--rule_guard` 会优先选择 `ConstraintSolver` 可证明安全的格子；它不计入裸模型成绩，用来判断剩余 loss 是基础规则抖动还是高阶排序错误。

Failure mining 诊断：

```bash
PYTHONPATH=src uv run python3 scripts/collect_mistakes.py \
  checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --device auto \
  --board_pool data \
  --output data/mistakes/S5_rule_guard_failures.npz
```

输出的 `.npz` 与 `TrajectoryPool` 兼容，可作为后续 replay source；对应 `.json` 记录 `rule_guard_avoidable`、`hard_sorting`、`calibration_drift` 计数。

Failure mining 500 局结果：

| 项目 | 值 |
|------|-----|
| Rollout | 裸模型，固定 `data/eval_boards_8x8_32.npz` |
| 胜率 | 457/500 WR = 91.40% |
| 总步数 | 8909 |
| 保存错题 | 441 单步 states |
| `rule_guard_avoidable` | 435 |
| `hard_sorting` | 6 |
| `calibration_drift` | 13（仅统计，不保存到 NPZ） |
| 输出 | `data/mistakes/S5_rule_guard_failures.npz` + `.json` |
| 结论 | 剩余错误几乎全部集中在可证明安全格排序失败；下一步优先小比例 hard-example replay，暂不优先上 lookahead search。 |

### S5 mistake replay fine-tune result

| 项目 | 值 |
|------|-----|
| 继承 | `checkpoints/v5_replay_S5/best_model.pt` |
| 数据 | `data/S5:0.55,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_rule_guard_failures.npz:0.05` |
| 超参 | supervised `deep_mse_rank`, lr=1e-4, epochs=2, batch=64, refine=4 |
| Checkpoint | `checkpoints/v5_replay_S5_mistake_ft/best_model.pt` |
| S5 裸模型 500 局 | 484/500 WR = 96.80%, action_acc=0.9982, avg_steps=18.2, avg_refine=4.0 |
| S5 rule guard 500 局 | 494/500 WR = 98.80%, action_acc=0.9993, rule_guard_actions=8241, avg_steps=18.3 |
| S5 after-mining | 484/500 WR = 96.80%, 388 saved states, `rule_guard_avoidable=382`, `hard_sorting=6`, `calibration_drift=11` |
| S1 回归 200 局 | 197/200 WR = 98.50%, action_acc=0.9992, avg_steps=18.8 |
| S4/S25 回归 200 局 | 192/200 WR = 96.00%, action_acc=0.9983, avg_steps=23.9 |
| 结论 | hard-example replay 成立：S5 裸模型从 91.40% 提升到 96.80%，S1/S4 无明显回归。下一步可用 `S5_after_mistake_ft.npz` 做更保守二次微调，优先 lr=5e-5、epochs=1、mistake weight=5%-8%。 |

二次微调命令：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_rank \
  --data_dir "data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_mistake_ft.npz:0.08" \
  --pretrained checkpoints/v5_replay_S5_mistake_ft/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_mistake_ft2 \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 1 --lr 5e-5
```

### S5 mistake replay fine-tune v2 result

| 项目 | 值 |
|------|-----|
| 继承 | `checkpoints/v5_replay_S5_mistake_ft/best_model.pt` |
| 数据 | `data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_mistake_ft.npz:0.08` |
| 超参 | supervised `deep_mse_rank`, lr=5e-5, epochs=1, refine=4 |
| Checkpoint | `checkpoints/v5_replay_S5_mistake_ft2/best_model.pt` |
| 训练内评估 | 98/100 WR = 98.00%, action_acc=0.999 |
| S5 裸模型 500 局 | 486/500 WR = 97.20%, action_acc=0.9985, avg_steps=18.2, avg_refine=4.0 |
| S5 rule guard 500 局 | 496/500 WR = 99.20%, action_acc=0.9996, rule_guard_actions=8256, avg_steps=18.4 |
| S5 after-mining | 486/500 WR = 97.20%, 381 saved states, `rule_guard_avoidable=377`, `hard_sorting=4`, `calibration_drift=11` |
| S1 回归 200 局 | 197/200 WR = 98.50%, action_acc=0.9992, avg_steps=18.8 |
| S4/S25 回归 200 局 | 192/200 WR = 96.00%, action_acc=0.9983, avg_steps=23.9 |
| 结论 | 二次 hard-example replay 继续正向但边际收益变小：S5 裸模型从 96.80% 到 97.20%，rule guard 到 99.20%，S1/S4 无回归。该阶段最佳 checkpoint 为 `v5_replay_S5_mistake_ft2`。不建议继续同构第三轮 replay；solver-safe ranking 已证伪，下一步优先 no-arch denoising refinement。 |

### Hard-Example Replay 基线复现流程

hard-example replay 成功基线为：

```text
checkpoints/v5_replay_S5_mistake_ft2/best_model.pt
S5 naked:      486/500 WR = 97.20%
S5 rule_guard: 496/500 WR = 99.20%
S1 regression: 197/200 WR = 98.50%
S4 regression: 192/200 WR = 96.00%
```

复现前置：
- `data/S1` 到 `data/S5` 均为 strict no-guess 数据，每阶段 10000 trajectories
- `data/eval_boards_8x8_32.npz` 固定为 500 boards
- 继承 checkpoint 为 `checkpoints/v5_replay_S5/best_model.pt`

1. 收集 S5 base checkpoint 的错题：

```bash
PYTHONPATH=src uv run python3 scripts/collect_mistakes.py \
  checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
  --output data/mistakes/S5_rule_guard_failures.npz
```

2. 第一轮 hard-example replay：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_rank \
  --data_dir "data/S5:0.55,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_rule_guard_failures.npz:0.05" \
  --pretrained checkpoints/v5_replay_S5/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_mistake_ft \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 2 --lr 1e-4
```

3. 收集第一轮后的错题：

```bash
PYTHONPATH=src uv run python3 scripts/collect_mistakes.py \
  checkpoints/v5_replay_S5_mistake_ft/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
  --output data/mistakes/S5_after_mistake_ft.npz
```

4. 第二轮保守 hard-example replay：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_rank \
  --data_dir "data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_mistake_ft.npz:0.08" \
  --pretrained checkpoints/v5_replay_S5_mistake_ft/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_mistake_ft2 \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 1 --lr 5e-5
```

5. 固定 cache 评估：

```bash
PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data

PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data --rule_guard

PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --width 8 --height 8 --mines 10 --n_games 200 --board_pool data

PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --width 8 --height 8 --mines 25 --n_games 200 --board_pool data
```

复现注意：
- `train_supervised.py` 已禁止自动启动 background generator；监督训练只读取显式 `--data_dir`
- 如果 S5 加载不是 `10000` trajectories，先修复 `data/S5` 再训练
- 不要覆盖 `checkpoints/v5_replay_S5_mistake_ft2`，它是 hard-example replay 成功基线

### Solver-safe-set ranking exploration

已新增 `deep_mse_solver_safe_rank` 并完成两轮探索，但当前结果为负优化，暂不作为成功基线。

实现要点：
- `scripts/collect_mistakes.py` 新增保存 `solver_safe_masks_*`
- `TrajectoryPool.batch(..., include_solver_safe=True)` 可返回 solver-safe masks
- 普通 S1-S5 replay 样本没有 safe mask 时，该额外 loss 自动为 0
- 初版全 pairwise safe-set ranking 在 S5 500 局下降到 480/500 WR = 96.00%，低于 `mistake_ft2` 的 486/500；当前实现已改为保守 set-min objective，只要求 safe set 内最低 logit 低于 safe set 外最低 logit。
- set-min 版本训练内 100 局为 91/100 WR = 91.00%，仍明显低于 `mistake_ft2`，因此暂停该路线。

历史错题生成命令：

```bash
PYTHONPATH=src uv run python3 scripts/collect_mistakes.py \
  checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
  --output data/mistakes/S5_after_mistake_ft2_solver_safe.npz
```

已证伪训练命令（不要作为当前基线）：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_solver_safe_rank \
  --data_dir "data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_mistake_ft2_solver_safe.npz:0.08" \
  --pretrained checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_solver_safe_rank \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 1 --lr 5e-5
```

### No-Arch Denoising Refinement

新增 `deep_mse_denoise_rank`，目标是在不改变 V5 19ch 架构的前提下，让模型从任意不完美概率图修正回 solver target。该路线借鉴 diffusion/denoising 思想，但不生成雷盘，也不引入 step/noise channel，因此可以继承 hard-example replay checkpoint。

实现要点：
- `MinesweeperTransformer.refine(..., initial_probs=...)` 支持从外部概率图启动；默认仍为 `0.5`，评估行为不变
- 训练时随机采样 denoising initial priors：`0.5`、`target_probs + gaussian noise`、`target_probs/random mix`、轻度 wrong-biased probs
- loss 仍为 `deep_mse + best_safe_rank`，不使用已证伪的 solver-safe ranking
- `v5_replay_S5_denoise_rank` 证明该路线正向：S5 裸模型达到 490/500 WR = 98.00%

训练命令：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_denoise_rank \
  --data_dir "data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_mistake_ft.npz:0.08" \
  --pretrained checkpoints/v5_replay_S5_mistake_ft2/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_denoise_rank \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 1 --lr 5e-5
```

结果：

| 项目 | 值 |
|------|-----|
| Checkpoint | `checkpoints/v5_replay_S5_denoise_rank/best_model.pt` |
| S5 裸模型 500 局 | 490/500 WR = 98.00%, action_acc=0.9989 |
| S5 rule guard 500 局 | 496/500 WR = 99.20% |
| S1 回归 200 局 | 195/200 WR = 97.50% |
| S4/S25 回归 200 局 | 195/200 WR = 97.50% |
| S5 after-mining | 342 saved states, `rule_guard_avoidable=338`, `hard_sorting=4`, `calibration_drift=8` |
| 结论 | denoising refinement 提升裸模型上限，但 S1/S4 略低于 `mistake_ft2`；后续使用更低 lr 和 S5_after_denoise_rank 做保守二次微调。 |

二次 denoise replay 命令：

```bash
PYTHONPATH=src uv run python3 scripts/train.py \
  --mode supervised --arch V5 --loss_type deep_mse_denoise_rank \
  --data_dir "data/S5:0.50,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,data/mistakes/S5_after_denoise_rank.npz:0.10" \
  --pretrained checkpoints/v5_replay_S5_denoise_rank/best_model.pt \
  --save_dir checkpoints/v5_replay_S5_denoise_rank_ft2 \
  --board_width 8 --board_height 8 --board_mines 32 \
  --epochs 1 --lr 2e-5
```

二次结果：

| 项目 | 值 |
|------|-----|
| Checkpoint | `checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt` |
| S5 裸模型 500 局 | 486/500 WR = 97.20% |
| S5 rule guard refine=4 | 497/500 WR = 99.40% |
| S5 rule guard refine=5 | 498/500 WR = 99.60% |
| S5 rule guard refine=6 | 498/500 WR = 99.60% |
| S5 rule guard refine=8 | 497/500 WR = 99.40% |
| 结论 | 二次微调提升了 guarded 策略的稳定性，但裸模型回落到 `mistake_ft2` 水平；最终部署评估采用 `refine_steps=5`。 |

### Current 100% Combo Strategy

最终 100% 路线是 `v5_replay_S5_denoise_rank_ft2` + `rule_guard` + `prob_zero_guard`，不是裸模型 100%。

实现边界：
- `rule_guard`: `ConstraintSolver` 有可证明 safe cells 时，只在 safe set 内交给模型排序
- `prob_zero_guard`: `rule_guard` 无 safe cells 时调用 `ProbabilitySolver`，仅在存在 `P(mine)=0` covered cells 时直接选择该零概率安全格
- `rule_mine_guard`: 仅保留诊断开关，最终 100% 组合不依赖

S5 正式评估命令：

```bash
PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 1000 --board_pool data \
  --refine_steps 5 --rule_guard --prob_zero_guard
```

S5 结果：

| 项目 | 值 |
|------|-----|
| 棋盘 | S5 8×8/32, 50.0% 密度 |
| Eval cache | `data/eval_boards_8x8_32.npz`, 1000 boards |
| Checkpoint | `checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt` |
| 策略 | `--refine_steps 5 --rule_guard --prob_zero_guard` |
| 胜率 | 1000/1000 WR = 100.00% |
| Action accuracy | 1.0000 |
| Rule-guard actions | 16396 |
| Prob-zero-guard actions | 1855 |
| Avg game steps | 18.3 |
| Avg refine steps | 4.9 |

S1/S4 回归命令：

```bash
PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt \
  --width 8 --height 8 --mines 10 --n_games 200 --board_pool data \
  --refine_steps 5 --rule_guard --prob_zero_guard

PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt \
  --width 8 --height 8 --mines 25 --n_games 200 --board_pool data \
  --refine_steps 5 --rule_guard --prob_zero_guard
```

回归结果：

| 阶段 | 结果 |
|------|------|
| S1 8×8/10 | 200/200 WR = 100.00% |
| S4 8×8/25 | 200/200 WR = 100.00% |

### 当前 100% 组合复现流程

推荐直接复现当前组合：

```bash
PYTHONPATH=src uv run python3 scripts/evaluate.py \
  checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 1000 --board_pool data \
  --refine_steps 5 --rule_guard --prob_zero_guard
```

若需从训练开始完整复现：
1. 先按“Hard-Example Replay 基线复现流程”训练到 `v5_replay_S5_mistake_ft2`
2. 运行 `deep_mse_denoise_rank` 训练生成 `v5_replay_S5_denoise_rank`
3. 对 `v5_replay_S5_denoise_rank` 进行 failure mining，生成 `data/mistakes/S5_after_denoise_rank.npz`
4. 运行二次 denoise replay，生成 `v5_replay_S5_denoise_rank_ft2`
5. 使用 `--refine_steps 5 --rule_guard --prob_zero_guard` 在固定 cache 上评估

---

## 2026-06-06: V5 S1 strict no-guess rerun (15ch/Deep-MSE 历史记录)

| 项目 | 值 |
|------|-----|
| Recipe | `v5_s1` |
| 架构 | V5 constraint residual, 15ch input, 1ch mine logit |
| 设备 | Mac MPS |
| 数据 | `data/`, 8×8/10, 5000 局 strict no-guess |
| 生成命令 | `PYTHONPATH=src uv run python3 scripts/generate_data.py --n_samples 5000 --output data --width 8 --height 8 --mines 10 --workers 8 --samples_per_file 2000 --force` |
| 数据校验 | 5000/5014 accepted, `total_ambiguous_cells=0`, `avg_ambig_per_game=0.0` |
| 训练命令 | `PYTHONPATH=src uv run python3 scripts/train_stage.py --recipe v5_s1 --arch V5 --device auto --eval_games 200` |
| 超参 | supervised `deep_mse`, lr=3e-4, epochs=5, batch=64, refine=4 |
| Epoch 5 loss | 0.0033 |
| 训练内评估 | 94/100 WR, action_acc=0.997 |
| 独立评估 | 187/200 WR = 93.50%, action_acc=0.9963, avg_steps=17.6, avg_refine=4.0 |
| Eval cache oracle | `data/eval_boards_8x8_10.npz`: 200/200 WR, `forced_guess_steps=0` |
| Checkpoint | `checkpoints/v5_s1_deep/best_model.pt` |
| 备注 | `generate_no_guess_board()` 现额外要求本项目 `ProbabilitySolver` 可无猜解完；外部 no-guess 但本 solver 会遇到非零最小概率的棋盘会被拒绝。 |

---

## 🆕 当前主线 (2026-05-30/31 重整)

### S1 — 8×8/10雷 规则学习

| 项目 | 值 |
|------|-----|
| 日期 | 2026-05-30 |
| 棋盘 | 8×8, 10 雷 (15.6%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 (并行 16 进程) |
| 超参 | lr=1e-3, wd=3e-4, batch=64, epochs=2, refine=global 1-16 |
| Val Acc | — (待跑) |
| 胜率 | — (待评估) |
| Checkpoint | `checkpoints/S1/` |
| 备注 | 三阶段首步. 并行生成已整合到 scripts/generate_data.py |

### S2 — 8×8/20雷 密度变化

| 项目 | 值 |
|------|-----|
| 日期 | 待跑 |
| 棋盘 | 8×8, 20 雷 (31.2%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=3e-4, wd=3e-4, batch=64, epochs=2, refine=global 1-16 |
| 继承 | `checkpoints/S1/best_model.pt` |
| Checkpoint | `checkpoints/S2/` |

### S3 — 8×8/32雷 高密度 (50%密度)

| 项目 | 值 |
|------|-----|
| 日期 | 2026-06-04 (重定义) |
| 棋盘 | 8×8, 32 雷 (50.0%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=1e-4, wd=3e-4, batch=64, epochs=5, refine=4步BPTT |
| 继承 | `checkpoints/S2/best_model.pt` |
| 评估目标 | V4 架构冲击 50% 密度，验证 grounding + residual 有效性 |
| 已知结果 | 待训练 (V4 架构刚落地) |
| Checkpoint | `checkpoints/S3/` |

> **历史 S3**: 旧 S3 定义为 8×8/25雷 (39.1%)，零样本 10×10/40 达 74% 胜率。已被新定义取代。

---
> **RL 微调已放弃 (2026-06-03)**。代码已从 main 移除，历史追溯使用 git。优先用在线 BCE 复现 99%+ 胜率，再考虑 RL。

---

## ⏳ 历史 (旧多阶段管线，已废弃)

### 2026-05-30: S1 (旧) — 4070S 首训 (refine=1, 5 epochs)

| 项目 | 值 |
|------|-----|
| **Val Loss** | **0.0023** (Epoch 5) |
| **Val Acc** | **97.5%** |
| 备注 | 环境测试用，非最新训练方案 |

### 旧管线结果（Mac M3 Pro MPS）

| 模型 | 任务 | Win Rate | Action Acc |
|------|------|----------|------------|
| S2.5 (refine 5) | 8×8/25 | 99.6% | 1.000 |
| S2.5 (refine 5) | 10×10/40 零样本 | 97.0% | 0.999 |
| S3 (无 refine) | 16×16/40 零样本 | 99.0% | 1.000 |
| 旧混合 checkpoint (refine 4) | 8×8/10 | 94% | - |
| RL (from 旧混合 checkpoint) | 8×8/10 | 73% ↓ | - |

> **RL 退化注记 (2026-05-30)**: 旧混合 checkpoint → RL 微调后从 94% 降至 73%。根因: confidence 头被清零。

---

## 2026-06-01: Refinement 有效性交叉验证

> **实验目的**: 验证 refinement 训练是否对监督学习有效，分离训练与推理的贡献。
> **环境**: 开发机 CPU，S1 (8×8/10雷)，1K 样本，2 epoch，batch_size=64，lr=1e-3，weight_decay=3e-4。

| Exp | 训练配置 | 评估配置 | 胜率 | Act Acc | 备注 |
|-----|---------|---------|------|---------|------|
| A | refine=1 (单步) | max_steps=1 | 55.5% | 97.2% | 纯单步 baseline |
| C | refine=1 (单步) | max_steps=16 | 29.5% | 94.8% | 单步模型 + refinement 推理 = OOD 崩溃 |
| D | refine=8 | max_steps=1 | 48.0% | 96.2% | refinement 训练 + 单步推理 |
| B | refine=8 | max_steps=16 | **81.0%** | **98.9%** | 匹配组合，最优 |

**结论**:
- Refinement 训练 + refinement 推理是黄金组合，必须匹配使用
- 混用导致 OOD（单步模型只见过 prev=0.5，refinement 推理时 prev 为模型输出）
- 纯单步 55.5% → refinement 组合 81.0%，+25.5pp，refinement 明确有效
- S2.5 旧模型 99.6% 的"refine 5"实为 no-op（prev_probs 权重被 zero-pad），纯单步质量达到的
## 2026-06-01/02: 重构与 Self-Play 探索

### 架构清理
- **移除 confidence 头**: `num_classes` 2→1，砍掉 confidence_loss + ponder_penalty，纯 MSE loss
- **早停阈值**: `convergence_eps` 1e-3→0.05，refinement 在 3-5 步内收敛退出
- **温度移除**: `train.py` 和 `train_rl.py` 删除 `--temperature`
- **配置集中**: `src/config/training_policy.py` 统一管理所有 refinement 参数

### 监督预训练（新架构，从头训练）
| 阶段 | 棋盘 | 胜率 | Act Acc | 训练耗时 |
|------|------|------|---------|---------|
| S1 | 8×8/10 | 98.0% | 99.9% | 281s |
| S2 | 8×8/20 | 95.5% | 99.8% | 399s |
| S3 (旧) | 8×8/25 → 10×10/40 零样本 | **92-96%** | 99.8% | 890s |

### Self-Play 探索（以 REINFORCE→self-play MSE 替代 RL 微调）

| # | 起点 | 基线WR | LR | Buffer | 峰值 | 最终 | 结论 |
|---|------|--------|-----|--------|------|------|------|
| 1 | S1 | 0% | 1e-5 | ∞ | 16% | 9% | 无限 buffer → 过拟合崩塌 |
| 2 | S2 | 66% | 1e-5 | 环形1024 | **83%** | 52% | LR 太高 → 冲高后崩塌 |
| 3 | S2 | 66% | 1e-6 | 环形1024 | 76% | **72%** | 稳定 +6，LR 是关键 |
| 4 | S3 | 96% | 1e-6 | 环形1024 | 97% | 92% | 基线太高，无训练信号 |

**核心发现**:
- Self-play MSE 有天花板，S2 最佳甜点只到 76%，追不上纯监督 S3 的 96%
- 纯监督 S1→S2→S3 仍是当前最强方案
- 以 100% 为目标的"非纯监督"路径未找到，self-play + MSE 本质仍是监督学习的低效变体

### Refinement 信息传递验证
- prev_probs 权重 100% 非零，信道正常训练
- 因果性确认：不同 prev 输入 → 不同输出
- 链路一致性：步与步之间输入输出逐 bit 一致（Δ < 1e-7）

## 2026-06-02: V3 架构升级与高密度测试崩溃

### 架构演进 (V3 Dual-Track)
- **隐式记忆 (Hidden State)**: 引入 64 维 `mem_state`，与 CNN 提取的 1 通道局部概率解耦。
- **训练方式变更**: 放弃随机 $k$ 步截断，改为固定步长 (BPTT `train_max_steps=4`) 的展开训练，以极小化计算开销并强制模型利用内部循环。
- **参数变化**: 重新梳理输入通道，总参数量稳定在约 33 万。

### 训练与评估结果
| 阶段 | 棋盘 | 训练配置 | Val Loss | Val Acc | 胜率 (Evaluate) | Act Acc |
|------|------|---------|----------|---------|-----------------|---------|
| S1 | 8×8/10 | BPTT 4步, lr=1e-3 | 0.0027 | 97.6% | 98.5% (197/200) | 0.999 |
| S2 | 8×8/20 | 继承S1, BPTT 4步 | (待补充) | (待补充) | (待补充) | (待补充) |
| S3 (旧V3) | 8×8/32 | 继承S2, BPTT 4步 | 0.0012 | 98.6% | **36.7%** (22/60) | **0.959** |

**核心问题与反思**:
1. **S3 评估阶段发生严重劣化**：在验证集上近乎完美的模型 (Val Acc 98.6%)，在实战中胜率暴跌至 36.7%。
2. 此问题在讨论中经历了多次排查猜测（包括评估状态丢失、训练数据污染等），但均未获得最终确定的结论。
3. 当前模型已处于最新提交状态，有待其他 Agent 进一步审查评估逻辑或数据源是否存在隐藏断层。

---

## 2026-06-04 代码审计 & 架构迭代

### V4 架构修复（FAEX1 上午完成）

| 改动 | 说明 |
|------|------|
| Grounding 注入 | CNN 特征 `features_seq` 在每轮 Transformer 自循环中重新注入 |
| 外部残差 | `mem_seq + Transformer(mem_seq + PE + features)`，Transformer 建模 delta |
| 去末端 LayerNorm | 允许 confidence 信号在循环中增长，不被归一化压缩 |
| Deep Inference | `eval_max_steps`: 4 → 16，训练固定 4 步 BPTT，推理可深度思考 |
| dropout | 0.0 → 0.2 (Transformer Encoder) |
| 配置模块化 | ModelConfig / TrainingConfig / StageConfig / STAGES 拆分 |

### S3 碰撞测试重置

S3 重新定义为 `8×8/32`（50% 密度），直接在新 V4 上验证高密度能力。

### 代码审计结论

Nanobot 对完整训练/评估/模型管线进行了逐行审计：
- **无逻辑 bug** — 训练循环、梯度流、数据管线均正确
- 识别了三个非阻塞问题（记录为 pitfalls #11-13）
- BN batch_size=1 在 2D 卷积场景下被空间维度缓解，不构成阻塞

### 待验证

- [ ] 新 V4（grounding + residual + deep inference）在 S3 8×8/32 上的实际表现
- [ ] eval_max_steps=16 时训练/推理解耦是否导致分布偏移

---

*最后更新: 2026-06-04*
