# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。当前主线是 **监督预训练 `S1 -> S2 -> S3`**，Online BCE 冷启实验中。

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 监督预训练主线：S1 -> S2 -> S3
python scripts/train_stage.py --all

# 3. 评估 S3 在目标棋盘上的零样本能力
python scripts/evaluate.py checkpoints/S3/best_model.pt \
  --width 10 --height 10 --mines 40

# 4. Online BCE 冷启动 (从零训练，无需预生成数据)
python scripts/train.py --mode online --n_games 5000 \
  --board_width 8 --board_height 8 --board_mines 10
```

默认 RL pool 路径会按棋盘自动推导为 `rl_boards_10x10_40.npz`。

## 全局策略配置

跨训练和评估必须一致的项目参数统一维护在 `src/config/training_policy.py`：

```text
refinement.train_max_steps = 4   # 监督训练 BPTT 全展开，无 detach
refinement.eval_max_steps  = 4   # 评估/推理与训练一致
refinement.rl_steps        = 16  # RL (已归档)
refinement.convergence_eps = 0.05
```

`scripts/train.py` 不再提供 `--refine` 参数，避免训练与评估之间出现局部默认值漂移。

## 训练 Pipeline

### Phase 1: 概率蒸馏

监督训练学习 `ProbabilitySolver` 输出的精确 `P(mine)` 概率矩阵。

```text
数据: 自验证/无猜棋盘 -> ProbabilitySolver 概率标签
模型: CNN -> Interpolatable PE -> Transformer -> P(mine) + confidence
损失: masked MSE + confidence loss + ponder penalty
增强: D4 旋转/翻转
```

### Phase 2: 三阶段预训练

| 阶段 | 棋盘 | 雷数 | 密度 | Epoch | 继承 |
|------|------|------|------|-------|------|
| S1 | 8×8 | 10 | 15.6% | 2 | 从头 |
| S2 | 8×8 | 20 | 31.3% | 2 | S1 |
| S3 | 8×8 | 25 | 39.1% | 5 | S2 |

历史/实验阶段 `S1.5/S2.5/S2.75/S3L/S4L` 仍可通过 `--legacy_stage` 显式运行，但不属于默认主线。

### Phase 3: RL 微调

RL 使用预生成 self-validated board pool，训练侧只读取 pool，不在训练过程中生成数据。

当前 reward shaping：

```text
safe reveal       = +1.0
extra floodfill   = +0.05 * (cells_revealed - 1)
mine hit          = -20.0
win reward        = none
pre-reveal bonus  = none
```

训练环境使用 `mine_continue=True` 获取密集纠错信号，评估环境使用真实规则：踩雷即结束。

## 常用命令

```bash
# 单阶段训练
python scripts/train_stage.py --stage S1

# 主线全阶段
python scripts/train_stage.py --all

# 显式运行历史阶段
python scripts/train_stage.py --legacy_stage S1.5

# 只评估已有 checkpoint
python scripts/train_stage.py --stage S3 --eval_only --eval 10 10 40

# 构建固定 RL pool
python scripts/generate_rl_pool.py --width 10 --height 10 --mines 40 --target_size 12000

# RL 微调
python scripts/train_rl.py --pretrained checkpoints/S3/best_model.pt --width 10 --height 10 --mines 40
```

## 项目结构

```text
src/minesweeper/       扫雷引擎、约束求解器、概率求解器
src/config/            项目级训练/评估策略配置
src/data/              监督数据和自验证棋盘生成
src/model/             CNN + Transformer + iterative refinement
src/training/          监督训练、RL 环境、RL 训练、RL board pool
scripts/generate_data.py      监督数据生成
scripts/train_stage.py        S1/S2/S3 预训练入口
scripts/evaluate.py           胜率评估入口
scripts/generate_rl_pool.py   RL board pool 构建入口
scripts/train_rl.py           RL 微调入口
```

## 文档体系

| 文档 | 用途 |
|------|------|
| [AGENTS.md](AGENTS.md) | 核心索引和开发约束 |
| [CHANGELOG.md](CHANGELOG.md) | 变更日志 |
| [docs/training-log.md](docs/training-log.md) | 训练实验记录 |
| [docs/architecture.md](docs/architecture.md) | 架构决策 |
| [docs/conventions.md](docs/conventions.md) | 环境和命令约定 |
| [docs/metrics.md](docs/metrics.md) | 指标速查 |
| [agents/pitfalls.md](agents/pitfalls.md) | Agent 避坑指南 |

## 说明

- 使用 `uv` 管理依赖。
- `data/`、`checkpoints/`、board pool 文件不入库。
- mixed 数据路线仍保留为实验能力，但不再是默认推荐路线。
