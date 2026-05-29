# Minesweeper Transformer

基于 CNN + Transformer 混合架构的扫雷 AI，从 0 完整走通深度学习训练全流程。

**当前方向**：概率蒸馏（Probability Distillation）— 训练模型估计每个格子精确的含雷概率，目标 90%+ 单步动作正确率。

## 项目结构

```
├── src/
│   ├── minesweeper/              # 扫雷游戏引擎
│   │   ├── game.py               # 核心逻辑：布雷、翻开、flood fill
│   │   ├── solver.py             # 约束传播求解器（find_safe_and_mines）
│   │   ├── probability_solver.py # 精确概率求解器（枚举所有一致雷配置）
│   │   └── constants.py          # 常量与通道定义
│   ├── data/
│   │   ├── generator.py          # 训练数据生成管线（概率蒸馏）
│   │   └── no_guess.py           # ms-toollib 无猜棋盘生成
│   ├── model/
│   │   └── architecture.py       # CNN + Transformer 模型定义
│   └── training/
│       ├── dataset.py            # PyTorch DataLoader（D4 增强）
│       ├── train.py              # 训练循环（MSE loss + action accuracy）
│       └── rl_train.py           # [DEPRECATED] REINFORCE RL 微调
├── scripts/
│   ├── generate_data.py          # 数据生成 CLI
│   ├── train.py                  # 训练 CLI（通用）
│   ├── train_s1.py ~ train_s4.py # 分阶段训练脚本（curriculum learning）
│   ├── evaluate.py               # 模型评估（胜率 + 动作正确率）
│   └── train_rl.py               # [DEPRECATED] RL 训练 CLI
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
| `--samples_per_file` | 100 | 每文件游戏数 |
| `--no_progress` | false | 关闭进度条 |

输出为 `data/training/data_XXXX.npz`，每个文件包含若干局游戏的轨迹数据。

### 3. 训练模型

```bash
uv run python scripts/train.py --epochs 50 --device auto
```

> `auto` 会自动检测 CUDA → MPS → CPU。也可手动指定 `--device mps`。

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight_decay` | 3e-4 | 权重衰减 |
| `--lr_scheduler` | cosine | LR 调度器 (cosine/plateau/none) |
| `--grad_clip` | 1.0 | 梯度裁剪 norm |
| `--device` | auto | 设备 (cpu/cuda/mps/auto) |
| `--data_dir` | data/training | 训练数据目录 |
| `--save_dir` | checkpoints | 模型保存目录 |
| `--pretrained` | "" | 预训练权重路径（curriculum learning） |

训练默认启用 D4 数据增强（旋转+翻转，等效 8 倍数据量）。关闭：`--no_augment`。

训练过程自动保存最佳模型到 `checkpoints/best_model.pt`，以及每轮指标到 `checkpoints/metrics.json`。

### 4. 评估模型

```bash
uv run python scripts/evaluate.py checkpoints/best_model.pt --n_games 500 --device auto
```

评估指标：
- **胜率 (Win Rate)**：完整通关的游戏占比
- **动作正确率 (Action Accuracy)**：选择的格子确实是安全的步数占比（核心指标，目标 ≥90%）
- **平均步数**：每局平均操作次数

## 数据格式

每个训练样本为 `(channels, probs, mask)` 三元组：

- **channels**: `(10, H, W)` float32 — 输入特征
  - `[0]`: covered mask（1=未翻开）
  - `[1]`: flagged mask（1=已插旗）
  - `[2:10]`: numbers 1-8 one-hot
- **probs**: `(H, W)` float32 — ProbabilitySolver 计算的精确含雷概率（0~1 软标签）
- **mask**: `(H, W)` bool — 参与 loss 的格子（1=covered 格）

## 模型架构

```
Input (B, 10, H, W)
  → CNN: 3× Conv3×3 + BN + ReLU  → (B, 64, H, W)
  → 2D Interpolatable Positional Encoding
  → Transformer Encoder: 3 layers, d=64, nhead=4, d_ff=256
  → Conv1×1 → (B, 1, H, W) → Sigmoid → P(mine)
```

**~296K 参数**。输出为每个格子的雷概率（0=安全，1=雷）。位置编码支持可变棋盘尺寸（16×16 的热启动）。

## 训练策略

### 概率蒸馏（当前方案）

核心思路：不让模型死记"这个格子是雷/不是雷"，而是让它学习"基于当前可见信息，这个格子有多大概率是雷"。

- **数据来源**：ms-toollib 生成的无猜棋盘（保证每步都有可推理的安全格）
- **标签生成**：ProbabilitySolver 对每一步局面枚举所有一致雷配置，计算每个 covered 格的边缘概率
- **标签分布**：~48% 模糊（0<P<1）、~22% 确定安全（P=0）、~30% 确定是雷（P=1）
- **损失函数**：MSE Loss（对概率值直接回归）
- **评估指标**：Action Accuracy — 每步选 P(mine) 最低的 covered 格翻开，检查是否安全
- **数据增强**：D4（旋转+翻转），等效 8× 数据量
- **正则化**：dropout=0.2、weight_decay=3e-4、gradient clipping=1.0

### 多阶段课程训练 (Curriculum Learning)

| 阶段 | 棋盘 | 雷数 | 继承 |
|------|------|------|------|
| S1 | 8×8 | 10 | 从头训练 |
| S1.5 | 8×8 | 15 | S1 权重 |
| S2 | 8×8 | 20 | S1.5 权重 |
| S3 | 12×12 | 40 | S2 权重 |
| S4 | 16×16 | 80 | S3 权重 |

每个阶段的训练脚本：`scripts/train_s1.py` ~ `scripts/train_s4.py`

### 已废弃的方向

- **REINFORCE RL 微调**：credit assignment 混乱，无猜数据涌现后监督学习更高效。相关代码保留在 `src/training/rl_train.py`（已标记 DEPRECATED）。
- **BCE + 硬标签**：用 0/1 二分类训练。模糊局面的梯度互相矛盾，收敛差。已被概率蒸馏取代。
- **全局雷数通道**：11→10 通道，移除剩余雷数/总雷数标量。局部数字边界已隐含该信息，移除后模型更专注空间拓扑。

## 技术决策

- **单头输出**：P(mine) 足矣，翻开/插旗是 P(mine) 的直接函数
- **标签来源**：ProbabilitySolver 枚举产生的精确概率，而非二值雷图
- **No-guess 棋盘**：ms-toollib 保证每步存在可推理安全格，使约束传播求解器能完整走通每局
- **通道设计**：10 通道（covered + flagged + 8× number one-hot），纯空间信息无全局标量
- **可变尺寸**：InterpolatablePositionalEncoding 支持不同棋盘尺寸的迁移学习
