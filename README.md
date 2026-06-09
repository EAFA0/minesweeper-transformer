# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。当前主线：**V5 replay curriculum `S1 → S2 → S3 → S4 → S5` + S5 hard-example replay**。

当前最佳基线：

```text
checkpoint: checkpoints/v5_replay_S5_mistake_ft2/best_model.pt
S5 8×8/32 naked:      486/500 WR = 97.20%
S5 8×8/32 rule_guard: 496/500 WR = 99.20%
```

完整复现流程见 `docs/training-log.md` 的“当前最佳基线复现流程”。

## 快速开始

```bash
# 1. 安装依赖（editable install）
uv sync

# 2. 全阶段训练（S1 → S2 → S3 → S4 → S5）
uv run python3 scripts/train_stage.py --recipe v5_curriculum_replay --arch V5

# 3. 零样本评估
uv run python3 scripts/evaluate.py checkpoints/v5_replay_S5/best_model.pt \
  --width 10 --height 10 --mines 40

# 4. 单阶段冷启动 (Online 模式)
uv run python3 scripts/train.py \
  --mode online --board_width 8 --board_height 8 --board_mines 10 --n_games 5000

# 5. 离线监督蒸馏 (Supervised 模式)
uv run python3 -m src.data.generator --n_samples 1000
uv run python3 scripts/train.py --mode supervised --epochs 5

# 6. 从 checkpoint 继续
uv run python3 scripts/train.py \
  --pretrained checkpoints/S1/best_model.pt --n_games 500
```

## 全局策略配置

跨训练和评估必须一致的项目参数统一维护在 `src/config/training_policy.py`：

```text
refinement.train_max_steps = 4   # 监督训练 BPTT 全展开（无 detach）
refinement.eval_max_steps  = 4   # 评估/推理与训练一致
refinement.convergence_eps = 0.05
```

`scripts/train.py` 不接受 `--refine` 参数，避免训练与评估间的默认值漂移。

## 训练 Pipeline

V5 replay curriculum：strict no-guess 数据 + Deep-MSE probability distillation + best-safe ranking loss + 全 BPTT refinement。

| 阶段 | 棋盘 | 雷数 | 密度 | 游戏数 |
|------|------|------|------|--------|
| S1 | 8×8 | 10 | 15.6% | 10000 |
| S2 | 8×8 | 15 | 23.4% | 10000 |
| S3 | 8×8 | 20 | 31.3% | 10000 |
| S4 | 8×8 | 25 | 39.1% | 10000 |
| S5 | 8×8 | 32 | 50.0% | 10000 |

每阶段继承前一阶段权重（curriculum transfer）。S2 之后混入低密度 replay，减少纯顺序课程导致的遗忘。冷启动无需预生成数据，`TrajectoryPool` 统一管理后台经验回放池。

## 常用命令

```bash
# 所有命令通过 uv run 确保使用正确的虚拟环境

# 全阶段训练
uv run python3 scripts/train_stage.py --recipe v5_curriculum_replay --arch V5

# 只跑最高密度 S5（继承 checkpoints/v5_replay_S4/best_model.pt）
uv run python3 scripts/train_stage.py \
  --recipe v5_curriculum_replay --start_phase 5 --end_phase 5 --arch V5

# 单阶段
uv run python3 scripts/train_stage.py --stage S1

# 独立训练（冷启动 Online）
uv run python3 scripts/train.py \
  --mode online --board_width 8 --board_height 8 --board_mines 10 --n_games 5000

# 离线监督训练（Supervised）
uv run python3 scripts/train.py \
  --mode supervised --data_dir data/S1 --epochs 5

# 微调已有模型
uv run python3 scripts/train.py \
  --pretrained checkpoints/S1/best_model.pt --n_games 500 --lr 1e-5

# 评估
uv run python3 scripts/evaluate.py checkpoints/v5_replay_S5/best_model.pt \
  --width 10 --height 10 --mines 40 --n_games 1000

# 规则 guard 诊断（辅助框架成绩，和裸模型成绩分开记录）
uv run python3 scripts/evaluate.py checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 200 --rule_guard

# 错题挖掘诊断（裸模型 rollout，输出训练兼容 NPZ + JSON summary）
uv run python3 scripts/collect_mistakes.py checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
  --output data/mistakes/S5_rule_guard_failures.npz

# 仅评估已有 checkpoint
uv run python3 scripts/train_stage.py --stage S5 --eval_only --eval 10 10 40
```

## 项目结构

```text
src/
  game/             扫雷引擎、求解器
  config/          训练/评估策略配置
  data/            数据生成模块（含 generator.py CLI）
  model/           CNN + Transformer + constraint residual refinement (V5)
  training/        训练循环（MSE + online BCE）+ 共享评估模块

scripts/
  train.py           统一训练入口 (支持 --loss_type bce|mse|deep_mse|deep_mse_rank, --recipe)
  train_stage.py     分阶段编排 (S1→S2→S3→S4→S5, --recipe, replay curriculum)
  evaluate.py        独立评估 CLI
  collect_mistakes.py 裸模型 failure mining 诊断，输出 hard-example NPZ
```

## 文档体系

| 文档 | 用途 |
|------|------|
| [AGENTS.md](AGENTS.md) | 核心索引和开发约束 |
| [docs/training-log.md](docs/training-log.md) | 训练实验记录 |
| [docs/architecture.md](docs/architecture.md) | 架构决策 |
| [agents/pitfalls.md](agents/pitfalls.md) | 避坑指南 |

## 说明

- 使用 `uv` 管理依赖，所有命令通过 `uv run python3 scripts/...` 执行。
- `data/`、`checkpoints/`、eval board pool 文件不入库。
- 旧架构 (V1/V1_5/V4) 和 RL 代码已从 main 移除，历史追溯使用 git。
