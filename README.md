# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。当前主线：**监督预训练 `S1 → S2 → S3`** + Online BCE 冷启。

## 快速开始

```bash
# 1. 安装依赖（editable install）
uv sync

# 2. 全阶段训练（S1 → S2 → S3）
uv run python3 scripts/train_stage.py --all

# 3. 零样本评估
uv run python3 scripts/evaluate.py checkpoints/S3/best_model.pt \
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

Online BCE：自验证棋盘池 + frontier BCE loss + 全 BPTT refinement。

| 阶段 | 棋盘 | 雷数 | 密度 | 游戏数 |
|------|------|------|------|--------|
| S1 | 8×8 | 10 | 15.6% | 5000 |
| S2 | 8×8 | 20 | 31.3% | 3000 |
| S3 | 8×8 | 32 | 50.0% | 3000 |

每阶段继承前一阶段权重（curriculum transfer）。冷启动无需预生成数据，`TrajectoryPool` 统一管理后台经验回放池。

## 常用命令

```bash
# 所有命令通过 uv run 确保使用正确的虚拟环境

# 全阶段训练
uv run python3 scripts/train_stage.py --all

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
uv run python3 scripts/evaluate.py checkpoints/S3/best_model.pt \
  --width 10 --height 10 --mines 40 --n_games 1000

# 仅评估已有 checkpoint
uv run python3 scripts/train_stage.py --stage S3 --eval_only --eval 10 10 40
```

## 项目结构

```text
src/
  game/             扫雷引擎、求解器
  config/          训练/评估策略配置
  data/            数据生成模块（含 generator.py CLI）
  model/           CNN + Transformer + iterative refinement (V3 hidden state)
  training/        训练循环（MSE + online BCE）+ 共享评估模块

scripts/
  train.py           统一训练入口 (支持 --loss_type bce|mse)
  train_stage.py     分阶段编排 (S1→S2→S3)
  evaluate.py        独立评估 CLI
  archived/          RL 代码（已归档）
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
- RL 代码已归档至 `scripts/archived/`，优先用监督/online BCE 复现 99%+ 胜率。
