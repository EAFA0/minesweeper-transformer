# Minesweeper Transformer

CNN + Transformer 混合架构的扫雷 AI。**概率蒸馏 → 密度课程 → 迭代 refinement → RL 微调**四阶段训练管线。

## 项目结构

```
├── src/
│   ├── minesweeper/              # 扫雷游戏引擎
│   │   ├── game.py               # 核心逻辑：布雷、翻开、flood fill
│   │   ├── solver.py             # 约束传播求解器
│   │   ├── probability_solver.py # 精确概率求解器（约束传播 + 连通分量枚举）
│   │   └── constants.py          # 常量与通道定义
│   ├── data/
│   │   ├── generator.py          # 监督学习数据生成管线
│   │   ├── no_guess.py           # ms-toollib 无猜棋盘生成
│   │   └── self_validated.py     # 自验证棋盘生成（ProbabilitySolver，RL 用）
│   ├── model/
│   │   └── architecture.py       # CNN + Transformer + Iterative Refinement
│   └── training/
│       ├── dataset.py            # PyTorch DataLoader（D4 增强）
│       ├── train.py              # 监督训练循环（MSE + action accuracy）
│       ├── rl_env.py             # RL 环境（自验证棋盘 + mine_continue）
│       ├── rl_train.py           # REINFORCE 策略梯度训练
│       └── ppo_train.py          # [DEPRECATED] PPO 实现（保留供参考）
├── scripts/
│   ├── generate_data.py          # 数据生成 CLI
│   ├── train.py                  # 监督训练 CLI
│   ├── train_stage.py            # 分阶段训练入口（统一所有阶段）
│   ├── train_rl.py               # RL 训练 CLI
│   ├── evaluate.py               # 模型评估（自适应 refinement + board pool）
│   └── train_s1.py ~ train_s4.py # [DEPRECATED] 旧 wrapper → 改用 train_stage.py
├── README.md
└── .gitignore
```

## 快速开始

```bash
# 1. 安装
uv sync

# 2. 从头训练：概率蒸馏 + 密度课程
python scripts/train_stage.py --stage S1      # 8×8/10雷 从头训练
python scripts/train_stage.py --stage S1.5    # 8×8/15雷 继承 S1
python scripts/train_stage.py --stage S2      # 8×8/20雷 继承 S1.5
python scripts/train_stage.py --stage S2.5    # 8×8/25雷 继承 S2 (高密度)
python scripts/train_stage.py --stage S2.75   # 8×8/30雷 继承 S2.5 (极限密度)

# 3. 评估（自适应 refinement + board pool 自动开启）
python scripts/evaluate.py checkpoints/S2_5/best_model.pt --no_guess

# 4. RL 微调（从监督权重起步）
python scripts/train_rl.py \
  --pretrained checkpoints/S2/best_model.pt \
  --width 8 --height 8 --mines 20 \
  --mine_continue --total_games 5000
```

### 依赖安装

| 平台 | 命令 |
|------|------|
| Apple Silicon (M1-4) | `uv sync` |
| NVIDIA GPU | `uv sync --index-url https://download.pytorch.org/whl/cu124` |
| CPU | `uv sync --index-url https://download.pytorch.org/whl/cpu` |

## 训练 Pipeline

### Phase 1: 概率蒸馏（监督学习）

模型学习 ProbabilitySolver 输出的精确 P(mine) 概率矩阵。

```
数据: ms-toollib 无猜棋盘 → ProbabilitySolver 计算标签
模型: CNN(3层) → PE → Transformer(4层) → P(mine) + confidence
损失: MSE(probs, target) + 0.3 × MSE(conf, conf_target)
增强: D4 (旋转+翻转, 8×)
```

### Phase 2: 密度课程

固定棋盘大小，逐步提升雷密度来增加约束复杂度。

| 阶段 | 棋盘 | 雷数 | 密度 | 继承 |
|------|------|------|------|------|
| S1 | 8×8 | 10 | 15.6% | 从头 |
| S1.5 | 8×8 | 15 | 23.4% | S1 |
| S2 | 8×8 | 20 | 31.3% | S1.5 |
| S2.5 | 8×8 | 25 | 39.1% | S2 |
| S2.75 | 8×8 | 30 | 46.9% | S2.5 |
| S3 | 12×12 | 40 | 27.8% | S2 |
| S3d | 10×10 | 30 | 30.0% | S2.75 |
| S3.5d | 10×10 | 40 | 40.0% | S3d |
| S4 | 16×16 | 80 | 31.3% | S3 |

统一入口：`python scripts/train_stage.py --stage <STAGE>`。

### Phase 3: 迭代 Refinement

模型输出 2 通道：`[P(mine), confidence]`。推理时自动迭代——把上一步的概率估计回灌给自己，检测矛盾、修正推理，直到概率分布收敛。

```
迭代 1: board + 0.5 → [P₁, C₁]
迭代 2: board + P₁ → [P₂, C₂]  ← 修正矛盾
迭代 3: board + P₂ → [P₃, C₃]  ← 微调确认（C₃ ≈ 1.0 → 停止）
```

- 训练时 unroll 5 步（权重共享），渐进权重
- 推理时自适应停止（conf > 0.95 或分布不再变化）
- 评估自动启用，无需 `--refine` 标记

### Phase 4: RL 微调

从监督权重 warm start，REINFORCE 策略梯度在自验证棋盘上微调。

```bash
python scripts/train_rl.py --pretrained checkpoints/S2/best_model.pt ...
```

关键设计：
- **自验证棋盘**：ProbabilitySolver 验证每局可解（秒级 vs ms-toollib 的 10-50s）
- **mine_continue**：踩雷 -10 分，游戏继续（不中断 episode，信号密集）
- **自适应 refinement**：eval 时自动多步迭代
- **不覆盖权重**：checkpoint 保存在 `checkpoints/rl/`

## 数据格式

训练样本 `(channels, probs, mask)`：

- **channels**: `(10, H, W)` float32
  - `[0]`: covered mask | `[1]`: flagged | `[2:10]`: numbers 1-8 one-hot
- **probs**: `(H, W)` float32 — ProbabilitySolver 精确概率标签
- **mask**: `(H, W)` bool — 参与 loss 的 covered 格

## 模型架构

```
Input (B, 10, H, W)
  → [prev_probs (B, 1, H, W)]  ← 迭代 refinement 回灌
  → CNN: 3× Conv3×3 + BN + ReLU  → (B, 64, H, W)
  → 2D Interpolatable Positional Encoding (支持任意尺寸)
  → Transformer Encoder: 4 layers, d=64, nhead=4, d_ff=256
  → Conv1×1 → (B, 2, H, W) → P(mine) + confidence
```

**297K 参数**。InterpolatablePE 支持训练 12×12 → 推理 16×16（零样本泛化验证通过）。

## 关键结果

| 模型 | 任务 | Win Rate | Action Acc |
|------|------|----------|------------|
| S2.5 (refine 5) | 8×8/25 (本任务) | 99.6% | 1.000 |
| S2.5 (refine 5) | 10×10/40 (零样本) | **97.0%** | 0.999 |
| S3 (无 refine) | 10×10/40 (零样本) | ~75% | 0.992 |
| S3 (无 refine) | 16×16/40 (零样本) | 99.0% | 1.000 |

迭代 refinement 在 10×10/40 上带来 **22 个百分点**的提升（75% → 97%）。

## train_stage.py 完整参数

```
python scripts/train_stage.py --stage S2.5 [OPTIONS]

  --stage S1|S1.5|S2|S2.5|S2.75|S3|...  训练阶段
  --epochs N       覆盖默认 epoch 数
  --resume         从 checkpoint 续训（自动追加 epoch）
  --refine N       迭代 refinement 步数（默认 1）
  --eval_only      仅评估已有 checkpoint
  --random_eval    使用随机棋盘评估（默认无猜）
  --force_data     强制重新生成数据
  --device auto    设备选择
```

## 说明

- **无 pip/pip3**：项目使用 `uv` 管理依赖
- **数据不入库**：`data/` 和 `checkpoints/` 已加入 `.gitignore`
- **评估自动缓存**：`--no_guess` 模式自动保存/加载 board pool（`eval_boards_*.npz`）
- **旧脚本兼容**：`train_s1.py ~ train_s4.py` 仍可用但已废弃，建议直接用 `train_stage.py`
