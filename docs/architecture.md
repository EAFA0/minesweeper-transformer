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
CNN (3层 3×3 Conv, 64ch) → InterpolatablePE → Transformer (4层, d=64, 4头) → 1ch 输出 (mine logit)。

输入: 10 board + 1 prev_probs + 8 constraint channels = 19 channels

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
- **主力路线 (V5)**: 保留 V1_5 的 `prev_probs` 显式反馈，并新增 8 个由棋盘规则直接计算的 constraint channels。每步 refinement 根据上一轮概率计算数字约束残差与硬规则信号，再喂回 CNN + Transformer，全 BPTT。
- **废弃路线 (V4 latent loop)**: CNN 只跑一次，Transformer 在抽象 latent 空间自循环。消融实验表明此路线在所有变体中均未超过 V1（无 refinement）。

### V5 constraint residual channels (2026-06-06)

V5 的输入为 `10 board + 1 prev_probs + 8 constraints = 19ch`。约束通道不使用真实雷位置，只使用当前可见棋盘和上一轮概率，因此训练与推理一致。

对每个 revealed cell:

```text
target_remaining = revealed_number - flagged_neighbor_count
predicted = sum(prev_probs on adjacent covered cells)
residual = target_remaining - predicted
```

然后将 revealed cells 上的约束投影到相邻 covered cells。当前实现的 8 个通道：

| 通道 | 含义 |
|------|------|
| `mean_residual` | 相邻数字约束平均还缺/多多少雷概率质量 |
| `max_residual` | 相邻约束中最强的正 residual |
| `min_residual` | 相邻约束中最强的负 residual |
| `mean_abs_residual` | 相邻约束不平衡强度 |
| `constraint_count_norm` | 该 covered cell 参与多少个数字约束，除以 8 归一化 |
| `forced_safe_signal` | 任一相邻约束满足 `target_remaining == 0` |
| `forced_mine_signal` | 任一相邻约束满足 `target_remaining == covered_neighbor_count` |
| `min_slack_norm` | 相邻硬规则 slack 的最小值，越低表示约束越接近强制规则 |

理由:
1. `prev_probs` 告诉模型“当前假设是什么”，constraint channels 告诉模型“当前假设和可见规则哪里不平衡”。
2. 扫雷 refinement 本质更接近约束传播，显式残差比纯 latent loop 更符合问题结构。
3. `forced_safe_signal`/`forced_mine_signal` 直接暴露基础扫雷硬规则，减少模型从平均 residual 中反推规则的负担。
4. V5 仍保留 CNN + Transformer，局部规则、共享边界和远距离依赖都能继续建模。

### Solver-Safe Ranking Loss (2026-06-06)

S5 hard-example replay 后，裸模型从 91.40% 提升到 97.20%，但剩余 mined states 仍主要是 `rule_guard_avoidable`。这说明单纯重复采样错题已进入平台期，下一步需要更直接地约束排序目标。

新增 loss 类型 `deep_mse_solver_safe_rank`：

```text
loss = deep_mse
     + rank_loss_weight * best_safe_rank
     + rank_loss_weight * solver_safe_set_rank
```

其中：
- `best_safe_rank`: 原 `deep_mse_rank`，要求至少一个 target P(mine)=0 的候选排在非零目标前面。
- `solver_safe_set_rank`: 新增项，要求 `ConstraintSolver` 可证明 safe set 中的最低 logit 低于 safe set 外的最低 logit，使裸模型 argmin 落入可证明 safe set。

`solver_safe_set_rank` 只使用 failure mining 生成的 `solver_safe_masks_*`，不在训练热路径中临时调用 solver；普通 replay 样本缺少该 mask 时额外项自动为 0。

全 pairwise 版本（所有可证明 safe cells 都排在所有 unknown cells 前面）在 S5 上出现负优化，因此当前实现采用更保守的 set-min objective，优先修正动作选择而不是强行重排全部 covered cells。

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
1. **Replay 分阶段课程** (`train_stage.py`): S1(8×8/10) → S2(8×8/15) → S3(8×8/20) → S4(8×8/25) → S5(8×8/32)，每阶段继承权重，并在 S2 之后混入低密度 replay
2. **混合数据** (`-m src.data.generator --mixed`): 一次生成可变尺寸+密度，单次训练

当前主力使用 replay 分阶段路线（更可控，每阶段验证，并缓解纯顺序课程遗忘）。

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
- 推理时 `max|P_t - P_{t-1}| < 0.01` → 停止
- 不再维护 confidence head；未监督 confidence 会污染 early-stop 决策
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
统一为五阶段 replay 监督预训练课程：
1. **S1 (规则)**: 8×8/10雷，学习基本扫雷规则
2. **S2 (低中密度)**: 8×8/15雷，继承 S1 → 提升约束密度
3. **S3 (中密度)**: 8×8/20雷，继承 S2 → 学习更密集局部约束
4. **S4 (高密度)**: 8×8/25雷，继承 S3 → 对齐旧高胜率密度课程
5. **S5 (最高密度)**: 8×8/32雷，50% 密度碰撞测试

理由:
1. “先教规则，再教密度，再冲击高密度”的课程设计可控且易复现
2. 阶段名只使用 `S1` 到 `S5`，不再引入 `S1.5`/`S2.5` 等历史密度名
3. 主入口为 `train_stage.py --recipe v5_curriculum_replay`，消除“该跑哪个阶段”的困惑

## 待决策事项

- [ ] 局部窗口注意力 vs 全局注意力（16×16+ 性能优化）
- [ ] 是否需要单独的 value head 做 bootstrapping

---

## ADR-008: 训练 Recipe 系统

- **状态**: ✅ 接受
- **日期**: 2026-06-06

### 背景
训练策略之前分散在 `--mode`/`--loss_type`/`--stage` 等 CLI 参数中手动组合，容易出错且难以复现。
2026-06-06 实测显示 pure online BCE finetune 会把 S1 MSE checkpoint 从 83% WR 破坏到约 15% WR，因此不再作为默认 recipe 阶段。

### 决策
引入 **Recipe 系统**：将训练策略抽象为可命名的多阶段 recipe。

```text
RecipePhase: {mode, loss_type, n_games, lr, board, pretrained, save_dir}
TrainingRecipe: {name, phases: [RecipePhase, ...]}
```

预定义 recipe：
- `v5_s1_rank`: S1 sanity baseline，Deep-MSE + best-safe ranking loss，对齐 `argmin P(mine)` 动作选择
- `v5_curriculum_replay`: 当前主线，五阶段 replay curriculum。S2 使用 `70% S2 + 30% S1`，S3 使用 `70% S3 + 15% S1 + 15% S2`，S4 使用 `70% S4 + 10% S1 + 10% S2 + 10% S3`，S5 使用 `60% S5 + 10% S1/S2/S3/S4`

过时 recipe（plain Deep-MSE、non-replay curriculum、MSE baseline）已从主线入口删除；历史实验记录保留在 `docs/training-log.md` 和 git 历史中。

`deep_mse_rank` 的 ranking 项只使用 solver probability target，不读取真实雷位置：

```text
preferred   = covered cells where target P(mine) <= 1e-6
competitors = covered cells where target P(mine) > 1e-6
loss_rank   = mean relu(min_logit(preferred) + margin - logits(competitors))
loss        = deep_mse + 0.1 * loss_rank
```

该项直接优化评估动作 `argmin P(mine)` 的排序，不替代 probability distillation。

`TrajectoryPool` 支持逗号分隔的 weighted replay 数据源：

```text
data/S5:0.6,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1
```

训练时只会对第一个目录启动当前阶段的数据生成；后续目录作为 replay source 只读加载，并按指定权重采样。

数据生成与评估默认使用 `generate_no_guess_board()`。`generate_self_validated_board()` 允许 safe hint，只能用于显式标注的 hint-solvable 探索，不得作为主训练/评估 benchmark。项目默认假设 `data/` 下的训练和评估数据均为 no-guess。

项目 no-guess 合同以本仓库的 `ProbabilitySolver` 为准：生成出的棋盘必须能在每一步找到 `P(mine)=0` 的 covered cell，直到胜利。外部生成器声称 no-guess 但本 solver 无法无猜推进的棋盘会被拒绝。

### 使用方式

```bash
# 查看 recipe 内容
uv run python3 scripts/train.py --recipe v5_curriculum_replay --dry_run

# 执行单阶段 recipe
uv run python3 scripts/train_stage.py --recipe v5_s1_rank --arch V5

# 执行五阶段 replay curriculum
uv run python3 scripts/train_stage.py --recipe v5_curriculum_replay --arch V5

# 只执行 S5，高密度碰撞测试
uv run python3 scripts/train_stage.py \
  --recipe v5_curriculum_replay --start_phase 5 --end_phase 5 --arch V5

# 辅助框架诊断：规则 guard + 模型排序
uv run python3 scripts/evaluate.py checkpoints/v5_replay_S5/best_model.pt \
  --width 8 --height 8 --mines 32 --n_games 200 --rule_guard

# 单 phase 执行（调试用）
uv run python3 scripts/train.py --recipe v5_s1_rank --arch V5
```

### 设计原则
1. Recipe 与 stage 并存，`--recipe` 优先于 `--stage`/`--mode`/`--loss_type`
2. 不传 `--recipe` 时完全向后兼容
3. `train_stage.py` 自动编排 recipe phase 顺序执行，每 phase 后自动评估
4. Pretrained checkpoint 链自动解析（phase N 默认继承 phase N-1 的 best_model.pt）
5. 可用 `--start_phase` / `--end_phase` 只运行 recipe 的一段，适合从已有 checkpoint 直接进入 S5

`--rule_guard` 是评估/部署侧辅助框架：先由确定性 `ConstraintSolver` 找出可证明安全格，再在这些候选内用模型概率排序；没有可证明安全格时退回模型 `argmin P(mine)`。裸模型实验记录默认不启用该开关。

---

*最后更新: 2026-06-06 (Recipe 系统新增)*
