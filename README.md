# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。当前主线：**监督预训练 `S1 → S2 → S3`** + Online BCE 冷启。

## 快速开始

```bash
# 1. 安装依赖（editable install，import model/training/data 随处可用）
uv sync

# 2. 监督预训练主线：S1 → S2 → S3
uv run python3 scripts/train_stage.py --all

# 3. 评估 S3 在目标棋盘上的零样本能力
uv run python3 scripts/evaluate.py checkpoints/S3/best_model.pt \
  --width 10 --height 10 --mines 40

# 4. Online BCE 冷启动 (从零训练，无需预生成数据)
uv run python3 scripts/train.py --mode online --n_games 5000 \
  --board_width 8 --board_height 8 --board_mines 10
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

### 三阶段预训练

| 阶段 | 棋盘 | 雷数 | 密度 | Epoch |
|------|------|------|------|-------|
| S1 | 8×8 | 10 | 15.6% | 2 |
| S2 | 8×8 | 20 | 31.3% | 3 |
| S3 | 8×8 | 25 | 39.1% | 5 |

每阶段继承前一阶段权重（curriculum transfer）。历史/实验阶段 `S1.5/S2.5/S2.75/S3L/S4L` 可通过 `--legacy_stage` 显式运行。

### Online BCE

在自验证棋盘上实时生成数据，对 frontier（已翻开格相邻的 covered 格）计算 BCE loss。全 BPTT refinement。

## 常用命令

```bash
# 所有命令通过 uv run 确保使用正确的虚拟环境

# 单阶段训练
uv run python3 scripts/train_stage.py --stage S1

# 主线全阶段
uv run python3 scripts/train_stage.py --all

# 独立训练（监督 MSE）
uv run python3 scripts/train.py --mode supervised \
  --data_dir data/S1 --epochs 5 --save_dir checkpoints/S1

# 独立训练（Online BCE）
uv run python3 scripts/train.py --mode online \
  --n_games 5000 --board_width 8 --board_height 8 --board_mines 20

# 生成更多数据
uv run python3 scripts/generate_data.py \
  --n_samples 2000 --width 8 --height 8 --mines 20 --output data/S2

# 评估模型
uv run python3 scripts/evaluate.py checkpoints/S3/best_model.pt \
  --width 10 --height 10 --mines 40 --n_games 1000

# 评估已有 checkpoint（跳过训练）
uv run python3 scripts/train_stage.py --stage S3 --eval_only --eval 10 10 40
```

## 项目结构

```text
src/
  game/             扫雷引擎、求解器
  config/          训练/评估策略配置
  data/            数据生成（自验证棋盘 + 概率标签）
  model/           CNN + Transformer + iterative refinement (V3 hidden state)
  training/        训练循环（MSE + online BCE）+ 共享评估模块

scripts/
  train.py           统一训练入口 (--mode supervised|online)
  train_stage.py     分阶段编排 (S1→S2→S3)
  evaluate.py        独立评估 CLI
  generate_data.py   数据生成 CLI
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
