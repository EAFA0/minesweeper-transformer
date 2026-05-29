# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。**概率蒸馏 → 迭代 refinement → RL 微调**三阶段训练管线。

## 项目结构

```
├── src/
│   ├── minesweeper/
│   │   ├── game.py               # 扫雷引擎：布雷、翻开、flood fill
│   │   ├── probability_solver.py # 精确概率求解器（约束传播 + 连通分量枚举）
│   │   ├── solver.py             # 约束传播求解器（ms-toollib fallback）
│   │   └── constants.py          # 常量与通道定义
│   ├── data/
│   │   ├── generator.py          # 单棋盘数据生成
│   │   ├── mixed_generator.py    # 混合数据生成（可变尺寸 + 密度）
│   │   ├── no_guess.py           # ms-toollib 无猜棋盘生成
│   │   └── self_validated.py     # 自验证棋盘生成（ProbabilitySolver，RL用）
│   ├── model/
│   │   └── architecture.py       # CNN + Transformer + Iterative Refinement
│   └── training/
│       ├── dataset.py            # PyTorch DataLoader（D4 增强）
│       ├── train.py              # 监督训练（自适应随机 k + ponder penalty）
│       ├── rl_env.py             # RL 环境（自验证棋盘 + mine_continue）
│       └── rl_train.py           # REINFORCE 策略梯度训练
├── scripts/
│   ├── generate_data.py          # 数据生成（支持 --mixed）
│   ├── train.py                  # 监督训练 CLI
│   ├── train_stage.py            # 分阶段训练入口（统一所有阶段）
│   ├── train_rl.py               # RL 训练 CLI
│   └── evaluate.py               # 模型评估（自适应 refinement + board pool）
├── docs/
│   └── metrics.md                # 指标速查
├── README.md
└── .gitignore
```

## 快速开始

```bash
# 1. 安装
uv sync

# 2. 推荐：混合数据 + 单次训练（可变尺寸 4-8，密度 10%-50%）
python scripts/generate_data.py --mixed --n_samples 12000 --output data/mixed
python scripts/train.py --data_dir data/mixed --epochs 3 --refine 8 \
  --save_dir checkpoints/S_mixed

# 3. 评估
python scripts/evaluate.py checkpoints/S_mixed/best_model.pt \
  --width 10 --height 10 --mines 40

# 4. RL 微调
python scripts/train_rl.py \
  --pretrained checkpoints/S_mixed/best_model.pt \
  --width 10 --height 10 --mines 40 \
  --mine_continue --total_games 5000
```

> 备选：分阶段训练 `python scripts/train_stage.py --stage S1` → `S2` → `S3`

### 依赖安装

| 平台 | 命令 |
|------|------|
| Apple Silicon | `uv sync` |
| NVIDIA GPU | `uv sync --index-url https://download.pytorch.org/whl/cu124` |
| CPU | `uv sync --index-url https://download.pytorch.org/whl/cpu` |

## 训练 Pipeline

### Phase 1: 概率蒸馏（监督学习）

模型学习 ProbabilitySolver 输出的精确 P(mine) 概率矩阵。

**推荐：混合数据单次训练。** 一次生成可变尺寸（4-8）和密度（10%-50%）的数据，单一训练跑覆盖全部难度，不需要分阶段切换权重。

```
数据: 混合无猜棋盘（可变尺寸 + 密度）→ ProbabilitySolver 计算标签
模型: CNN(3层) → PE → Transformer(4层) → P(mine) + confidence
损失: MSE(probs, target) + 0.3×MSE(conf, conf_target) + 0.1×ponder_penalty
增强: D4 (旋转+翻转, 8×)
```

备选：分阶段密度课程（`train_stage.py --stage S1 S2 S3`），详见阶段表。

### Phase 2: 自适应 Refinement 训练

训练时每 batch 随机抽迭代步数 k ∈ [1, 8]，只对第 k 步算 loss。
Ponder penalty 惩罚深迭代但低置信度。`detach()` 切断 BPTT，训练速度接近单次推理。

推理时置信度 > 0.95 自动早停，上限 12 步。

### Phase 3: RL 微调

从 S3 权重 warm start，REINFORCE 策略梯度在自验证棋盘上微调。

- **自验证棋盘**：ProbabilitySolver 验证每局可解
- **mine_continue**：踩雷 -10 分，游戏继续（不中断 episode）
- **自适应 refinement**：推理时自动多步迭代
- **权重隔离**：checkpoint 保存在 `checkpoints/rl/`

## 阶段表（备选路线）

| 阶段 | 棋盘 | 雷数 | 密度 | Epoch | 继承 |
|------|------|------|------|-------|------|
| S1 | 8×8 | 10 | 15.6% | 1 | 从头 |
| S2 | 8×8 | 20 | 31.3% | 1 | S1 |
| S3 | 10×10 | 40 | 40.0% | 2 | S2 |

可选：`S1.5`(8×8/15), `S2.5`(8×8/25), `S2.75`(8×8/30), `S3L`(12×12/40), `S4L`(16×16/80)

## 数据格式

- **channels**: `(10, H, W)` float32 — [0]covered, [1]flagged, [2:10]numbers 1-8
- **probs**: `(H, W)` float32 — ProbabilitySolver 精确概率
- **mask**: `(H, W)` bool — 参与 loss 的 covered 格

## 模型架构

```
Input (B, 10, H, W) + [prev_probs (B, 1, H, W)]
  → CNN: 3× Conv3×3 + BN + ReLU → (B, 64, H, W)
  → 2D Interpolatable Positional Encoding
  → Transformer Encoder: 4 layers, d=64, nhead=4, d_ff=256
  → Conv1×1 → (B, 2, H, W) → P(mine) + confidence
```

**297K 参数**。InterpolatablePE 支持任意棋盘尺寸。

## 关键结果（旧管线，供参考）

| 模型 | 任务 | Win Rate | Action Acc |
|------|------|----------|------------|
| S2.5 (refine 5) | 8×8/25 | 99.6% | 1.000 |
| S2.5 (refine 5) | 10×10/40 零样本 | **97.0%** | 0.999 |
| S3 (无 refine) | 16×16/40 零样本 | 99.0% | 1.000 |

新管线（自适应训练 + 混合数据）结果待补充。

## train_stage.py 参数

```
python scripts/train_stage.py --stage S2 [OPTIONS]

  --stage S1|S2|S3|...  训练阶段
  --epochs N             覆盖默认 epoch 数
  --resume               续训（自动追加 epoch）
  --refine N             迭代 refinement 最大步数（默认 8）
  --eval_only            仅评估已有 checkpoint
  --force_data           强制重新生成数据
  --device auto          设备选择
```

## 说明

- 使用 `uv` 管理依赖
- `data/` 和 `checkpoints/` 已加入 `.gitignore`
- 评估自动缓存 board pool（`eval_boards_*.npz`）
- 推荐混合数据单次训练替代分阶段：`--mixed --n_samples 12000`
