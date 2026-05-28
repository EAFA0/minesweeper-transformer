# Minesweeper Transformer

基于 CNN + Transformer 混合架构的扫雷 AI，从 0 完整走通深度学习训练全流程。

## 项目结构

```
├── src/
│   ├── minesweeper/           # 扫雷游戏引擎
│   │   ├── game.py            # 核心逻辑：布雷、翻开、flood fill
│   │   ├── solver.py          # 约束传播求解器（用于数据筛选）
│   │   └── constants.py       # 常量与通道定义
│   ├── data/
│   │   └── generator.py       # 训练数据生成管线
│   ├── model/
│   │   └── architecture.py    # CNN + Transformer 模型定义
│   └── training/
│       ├── dataset.py         # PyTorch DataLoader
│       └── train.py           # 训练循环（BCE loss + metric）
├── scripts/
│   ├── generate_data.py       # 数据生成 CLI
│   └── train.py               # 训练 CLI
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

| 平台 | 命令 | 后端 |
|------|------|------|
| Apple Silicon (M1/M2/M3) | `uv sync` | MPS（PyTorch 自带，无需额外安装） |
| NVIDIA GPU | `uv sync --index-url https://download.pytorch.org/whl/cu124` | CUDA |
| 仅 CPU | `uv sync --index-url https://download.pytorch.org/whl/cpu` | CPU |

> 没有 uv？`curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. 生成训练数据

```bash
uv run python scripts/generate_data.py --n_samples 10000
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 10000 | 生成的游戏局数 |
| `--width` | 8 | 棋盘宽度 |
| `--height` | 8 | 棋盘高度 |
| `--mines` | 10 | 雷的数量 |
| `--output` | data/training | 输出目录 |
| `--seed` | 42 | 随机种子 |
| `--no_progress` | false | 关闭进度条 |

输出为 `data/training/data_XXXX.npz`（压缩格式），每个文件包含 100 局游戏的轨迹数据。

### 3. 训练模型

```bash
uv run python scripts/train.py --epochs 50 --device auto
```

> `auto` 会自动检测 CUDA → MPS → CPU。也可手动指定 `--device mps`。
>
> 训练默认启用 D4 数据增强（旋转+翻转，等效 8 倍数据量）。关闭：`--no_augment`。

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--lr_scheduler` | cosine | LR 调度器 (cosine/plateau/none) |
| `--device` | auto | 设备 (cpu/cuda/auto) |
| `--data_dir` | data/training | 训练数据目录 |
| `--save_dir` | checkpoints | 模型保存目录 |

训练过程会自动保存最佳模型到 `checkpoints/best_model.pt`，以及每轮指标到 `checkpoints/metrics.json`。

## 数据格式

每个训练样本为 (channels, labels, mask) 三元组：

- **channels**: `(10, 8, 8)` float32 — 输入特征
  - `[0]`: covered mask（1=未翻开）
  - `[1]`: flagged mask（1=已插旗）
  - `[2:10]`: numbers 1-8 one-hot
- **labels**: `(8, 8)` float32 — 真实雷位（1=雷，0=安全）
- **mask**: `(8, 8)` bool — 参与 loss 的格子（1=covered 格）

## 模型架构

```
Input (B, 10, 8, 8)
  → CNN: 3× Conv3×3 + BN + ReLU  → (B, 64, 8, 8)
  → 2D Learnable Positional Encoding
  → Transformer Encoder: 3 layers, d=64, nhead=4, FF=256
  → Conv1×1 → (B, 1, 8, 8) → Sigmoid → P(mine)
```

**231K 参数**。输出为每个格子的雷概率（0=安全，1=雷）。

## 训练策略

### Phase 1：监督学习冷启动（当前阶段）

- 用约束传播求解器筛选"无猜可解"的棋盘（~59% 收率）
- 真实雷图作为标签（非 solver 推理结果）
- BCE loss，仅 covered 格参与计算
- 自动 pos_weight 平衡 mine/safe 类别
- D4 数据增强：随机旋转/翻转，等效 8 倍数据量
- 正则化：dropout=0.2, weight_decay=3e-4, gradient clipping=1.0

### Phase 2：RL 微调（当前阶段）

REINFORCE 策略梯度算法，在 Phase 1 模型基础上微调：

- **策略**：π(reveal cell i) = softmax(-P(mine)_i / τ)，温度 τ 控制探索程度
- **Reward 设计**：安全翻开 +1，触发 flood fill +3，正确插旗 +2，踩雷 -10，胜利 +20
- **Solver 辅助**：solver 判定安全/雷时给额外 bonus，加速收敛
- **Baseline**：指数移动平均减少梯度方差

```bash
uv run python scripts/train_rl.py --total_games 5000 --device auto
```

训练完成后评估：

```bash
uv run python scripts/evaluate.py checkpoints/rl/rl_final.pt --n_games 1000
```

## 技术决策

- **单头输出 vs 双头**：单头 P(mine) 足够，翻开/插旗是 P(mine) 的直接函数
- **标签来源**：真实雷图而非 solver 结论，让模型学到 solver 之外的隐含规律
- **约束传播求解器**：自定义简化版（trivial + subset），零错误推理
- **通道设计**：10 通道（covered + flagged + 8× number one-hot）
