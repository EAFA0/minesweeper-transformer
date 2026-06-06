# 架构决策记录

本文档记录 Minesweeper Transformer 项目的关键架构决策。每个决策包含背景、考虑的方案、最终选择和理由。

## ADR-001: 概率蒸馏 vs 硬标签分类

- **状态**: ✅ 接受
- **日期**: 2026-05-27

### 背景
训练数据需要标签。传统做法是 BCE 二分类（是雷/不是雷）。但扫雷存在 NP-完全的约束一致性难题 — 某些局面的某些格子无法确定是否为雷。

### 考虑的方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A: BCE 硬标签 | 0/1 标签，判断是否是雷 | 简单直接 | 丢失不确定性信息，模糊局面学习困难 |
| B: MSE 概率蒸馏 | 学习 ProbabilitySolver 的精确 P(mine) | 保留不确定性、支持模糊局面 | 需要求解器计算标签 |
| C: 混合 | 确定格用BCE，模糊格用MSE | 两全其美 | 工程复杂度高 |

### 决策
**采用方案 B（MSE 概率蒸馏）**。

理由:
1. 不确定性是扫雷的核心挑战（NP-完全），丢弃概率信息会让模型学习错误的确定性
2. 动作选择时选 P(mine) 最低格优于选"不是雷"格（因为可能没有确定的"不是雷"格）
3. 蒸馏 ProbabilitySolver 的概率分布比硬标签提供了更丰富的训练信号
4. 实际效果验证：概率蒸馏训练出的模型有更强的零样本泛化能力

---

## ADR-002: 通道设计 (10-ch)

- **状态**: ✅ 接受
- **日期**: 2026-05-27 (更新: 2026-06-04)

### 背景
需要设计输入通道来表示棋盘状态。

### 决策
使用 10 通道：
- [0] covered: 1=未翻开, 0=已翻开
- [1] flagged: 1=已标雷, 0=未标
- [2:10] numbers 1-8: one-hot 编码翻开格数字

理由:
1. One-hot 数字通道让 CNN 可以独立学习每种数字的邻域模式
2. 简化设计，移除 `mines_remaining_ratio` 以降低初始训练复杂度
3. 10 通道符合当前 V5 架构的输入层定义

---

## ADR-003: 模型架构 (CNN + Transformer 混合 V5)

- **状态**: ✅ 接受 (V5 为唯一活跃架构)
- **日期**: 2026-05-27 (原始), 2026-06-06 (V5 单架构清理)

### 背景
扫雷同时需要局部模式识别（CNN 擅长的数字-邻域关系）和全局约束推理（Transformer 擅长的远距离依赖）。

### 当前架构: V5 Constraint Residual
CNN (3层 3×3 Conv, 64ch) → InterpolatablePE → Transformer (4层, d=64, 4头) → 2ch 输出 (mine + confidence)。

输入: 10 board + 1 prev_probs + 4 constraint channels = 15 channels

### 历史架构对比 (已从 main 移除)

| 架构 | 描述 | 8×8/10 胜率 | 动作准确率 |
|------|------|------------|-----------|
| V1_5 | CNN 重跑 + prev_probs 显式反馈 | 83.0% | 0.990 |
| V1 | 无 refinement | 74.0% | 0.985 |
| V4 baseline | latent loop | 63.5% | 0.978 |

V4 的 latent loop refinement 在所有消融实验中均未超过 V1（无 refinement），说明 latent loop 不仅无益反而有害。根因是 Transformer 在抽象 latent 空间自循环，永远看不到 decoder 输出，无法进行矛盾检测和修正。

V1/V1_5/V4 代码已从 main 移除，历史追溯使用 git。

---

## ADR-004: 迭代 Refinement

- **状态**: ✅ 接受 (V5 constraint residual 为唯一活跃路线)
- **日期**: 2026-05-28 (更新: 2026-06-06)

### 背景
单次前向推理可能产生不一致的预测。迭代 refinement 让模型反复修正自己的预测。

### 决策
- **主力路线 (V5)**: 保留 V1_5 的 `prev_probs` 显式反馈，并新增 4 个由棋盘规则直接计算的 constraint channels。每步 refinement 根据上一轮概率计算数字约束残差，再喂回 CNN + Transformer，全 BPTT。
- **废弃路线 (V4 latent loop)**: CNN 只跑一次，Transformer 在抽象 latent 空间自循环。消融实验表明此路线在所有变体中均未超过 V1（无 refinement）。

### V5 constraint residual channels (2026-06-06)

V5 的输入为 `10 board + 1 prev_probs + 4 constraints = 15ch`。约束通道不使用真实雷位置，只使用当前可见棋盘和上一轮概率，因此训练与推理一致。

对每个 revealed cell:

```text
target_remaining = revealed_number - flagged_neighbor_count
predicted = sum(prev_probs on adjacent covered cells)
residual = target_remaining - predicted
```

然后将 revealed cells 上的约束投影到相邻 covered cells。当前实现的 4 个通道：

| 通道 | 含义 |
|------|------|
| `mean_residual` | 相邻数字约束平均还缺/多多少雷概率质量 |
| `mean_abs_residual` | 相邻约束不平衡强度 |
| `constraint_count_norm` | 该 covered cell 参与多少个数字约束，除以 8 归一化 |
| `mean_target_remaining` | 相邻数字约束平均还需要解释多少雷 |

理由:
1. `prev_probs` 告诉模型“当前假设是什么”，constraint channels 告诉模型“当前假设和可见规则哪里不平衡”。
2. 扫雷 refinement 本质更接近约束传播，显式残差比纯 latent loop 更符合问题结构。
3. V5 仍保留 CNN + Transformer，局部规则、共享边界和远距离依赖都能继续建模。

### 消融实验数据 (2026-06-05, S1 8×8/10雷, 10000 games, 5 epochs)

| 架构 | Refinement 机制 | 胜率 | action_acc |
|------|----------------|------|------------|
| V1_5 | CNN 重跑 + prev_probs 显式反馈 | 83.0% | 0.990 |
| V1 | 无 refinement | 74.0% | 0.985 |
| V4 baseline | latent loop | 63.5% | 0.978 |

### 根因分析
V4 的 Transformer 在抽象 latent 空间自循环，永远看不到 decoder 输出，无法进行矛盾检测和修正。扫雷 refinement 的本质是矛盾检测（模型预测与 board 约束不一致 → 修正），这需要显式反馈回路。

### 当前策略
- 主力架构: V5 (CNN 重跑 + prev_probs 反馈 + constraint residual channels)
- 推理: `eval_max_steps=4` (与训练步数对齐)，通过 `max|ΔP| < 1e-2` 早停
- 旧架构 (V1/V1_5/V4) 已从 main 移除，历史追溯使用 git

---

## ADR-005: 密度课程 vs 混合数据

- **状态**: ✅ 接受（两条路线并存）
- **日期**: 2026-05-28

### 背景
如何组织训练数据？逐步提升难度（课程学习）还是混合所有难度？

### 决策
两条路线并存，互不冲突:
1. **分阶段课程** (`train_stage.py`): S1(8×8/10) → S2(8×8/20) → S3(10×10/40)，每阶段继承权重
2. **混合数据** (`-m src.data.generator --mixed`): 一次生成可变尺寸+密度，单次训练

当前主力使用分阶段路线（更可控，每阶段验证）。

---

## ADR-006: RL 算法选择 (REINFORCE vs PPO)

- **状态**: ✅ 接受
- **日期**: 2026-05-30（更新 2026-05-31）

### 背景
监督学习后需要 RL 微调提升实战能力。

### 决策
**采用 REINFORCE**。

理由:
1. 从监督权重 warm start，策略已经很好了，不需要 PPO 的稳定性保证
2. 扫雷每局 20-80 步，轨迹短，REINFORCE 的高方差问题不严重
3. 工程简单，更容易调试

### RL 环境设计决策

#### mine_continue: 训练 True, 评估 False
- **训练时 `mine_continue=True`**: 模型踩雷后自动标雷，游戏继续。让模型从错误中学习而不提前终止
- **评估时 `mine_continue=False`**: 踩雷即结束，反映真实扫雷规则下的胜率
- **理由**: 分离训练信号（密集反馈）和评估标准（真实规则）

#### 奖励设计
```
reveal_safe      = +1.0
floodfill_bonus  = +0.05 × max(0, cells_revealed - 1)
hit_mine         = -20.0
win_reward       = none
```
- **即时奖励**: 扫雷动作没有“牺牲当前换未来”的规划收益，训练信号只绑定当前动作结果
- **小权重 floodfill**: 保留快速结束游戏的倾向，但避免模型为追求大 floodfill 奖励而赌博
- **无 win bonus**: 避免最后一步获得与动作无关的巨额梯度
- **无预揭开补分**: 开局已翻格属于环境状态，不归因到模型最后一步动作

#### 固定尺寸 vs 混合
- **focused RL (10×10/40)**: 单一尺寸/密度，训练信号聚焦
- **mixed_env 仅实验使用**: 训练入口默认固定棋盘，显式 `--mixed` 才启用混合池
- **理由**: 主目标是把 S3 checkpoint 在目标分布上推近 100%，固定分布更容易诊断

#### 棋盘池
- `generate_rl_pool.py` 负责预生成 self-validated board pool
- `train_rl.py` 训练侧只读取 pool，不在训练过程中构造或补齐数据
- `first_done=True`: 模型从 after-first-click 态出发（与训练数据一致），避免 OOD 全覆棋盘

#### 提前退出：收敛检测
- 推理时 `max|P_t - P_{t-1}| < 1e-3` → 停止
- 替代置信度方案（confidence head 无显式监督，无法可靠达到 0.95）
- 实测 S3 模型 90% 样本在 5 步内收敛（max 16）

#### Advantage
- 当前实现使用 batch 内即时奖励标准化降低方差
- `baseline` 仅作为训练日志诊断指标，不参与 advantage 计算

---

## ADR-007: 三阶段预训练路线

- **状态**: ✅ 接受
- **日期**: 2026-05-30

### 背景
原来有 S1-S4 分阶段 + mixed 混合路线并存，各自独立运行，造成混淆（Agent 曾用旧 S1 训练后才发现不是最新方案）。

### 决策
统一为三阶段监督预训练课程：
1. **S1 (规则)**: 8×8/10雷, 2 epochs, 学习基本扫雷规则
2. **S2 (密度)**: 8×8/20雷, 2 epochs, 继承 S1 → 学习雷密度可变
3. **S3 (高密度泛化)**: 8×8/32雷 (50%密度), 5 epochs, 继承 S2 → 直接冲击高密度

理由:
1. “先教规则，再教密度，再冲击高密度”的课程设计可控且易复现
2. S3 的 8×8/32 密度 50%，是高密度扫雷的硬骨头
3. 统一入口 `train_stage.py --stage S1/S2/S3`，消除“该跑哪个阶段”的困惑

## 待决策事项

- [ ] 局部窗口注意力 vs 全局注意力（16×16+ 性能优化）
- [ ] 是否需要单独的 value head 做 bootstrapping

---

## ADR-008: 训练 Recipe 系统

- **状态**: ✅ 接受
- **日期**: 2026-06-06

### 背景
训练策略（MSE warmup → online BCE finetune → hybrid）之前分散在 `--mode`/`--loss_type`/`--stage` 等 CLI 参数中手动组合，容易出错且难以复现。

### 决策
引入 **Recipe 系统**：将训练策略抽象为可命名的多阶段 recipe。

```text
RecipePhase: {mode, loss_type, n_games, lr, board, pretrained, save_dir}
TrainingRecipe: {name, phases: [RecipePhase, ...]}
```

预定义 recipe 示例：
- `v5_s1`: supervised MSE warmup (8×8/10, 5000 games) → online BCE finetune (8×8/10, 3000 games)

### 使用方式

```bash
# 查看 recipe 内容
uv run python3 scripts/train.py --recipe v5_s1 --dry_run

# 执行完整 recipe（多阶段自动编排）
uv run python3 scripts/train_stage.py --recipe v5_s1 --arch V5

# 单 phase 执行（调试用）
uv run python3 scripts/train.py --recipe v5_s1 --arch V5
```

### 设计原则
1. Recipe 与 stage 并存，`--recipe` 优先于 `--stage`/`--mode`/`--loss_type`
2. 不传 `--recipe` 时完全向后兼容
3. `train_stage.py` 自动编排多 phase 顺序执行，每 phase 后自动评估
4. Pretrained checkpoint 链自动解析（phase N 默认继承 phase N-1 的 best_model.pt）

---

*最后更新: 2026-06-06 (Recipe 系统新增)*
