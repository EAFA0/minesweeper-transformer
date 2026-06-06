## [未发布] - 2026-06-05

### 训练管线修复
- **修复 BCE 数值稳定性**: Offline supervised 与 online frontier BCE 均改为 raw logits + `F.binary_cross_entropy_with_logits()`；概率路径只用于 MSE、动作选择和评估。
- **修复 V1_5 单步接口一致性**: `forward()` 初始 `prev_probs` 从 0 改为 0.5，与 `refine(num_steps=1)` 保持一致。
- **补充 V4 logits 入口**: 新增 `forward_logits()`，避免 V4 BCE 路径继续依赖 sigmoid 后概率。

### V5 constraint residual 架构
- **新增 V5**: 在 V1_5 的 `prev_probs` 显式反馈基础上，加入 4 个规则派生 constraint channels，输入共 `10 board + 1 prev_probs + 4 constraints = 15ch`。
- **约束特征**: 对每个已翻开格计算 `target_remaining = number - flagged_neighbors` 与 `residual = target_remaining - sum(adjacent prev_probs)`，再投影到相邻 covered cells。
- **接入训练/评估**: `--arch V5` 已接入 `scripts/train.py`、`scripts/train_stage.py`、`scripts/evaluate.py`、checkpoint 加载和统一训练分发。
- **迁移支持**: V5 可从 V1/V1_5 类 checkpoint 迁移，CNN 首层新增输入通道零初始化。

### V4 消融实验与架构决策
- **V4 latent loop 路线废弃**: 经过系统性消融实验，V4 的 latent loop refinement 在所有变体中均未超过 V1（无 refinement）。根因是 Transformer 在抽象 latent 空间自循环，看不到 decoder 输出，无法进行矛盾检测和修正。
- **V1_5 确认为主力架构**: CNN 重跑 + prev_probs 显式反馈的 refinement 机制是当前唯一有效的迭代修正方案，S1 胜率 83.0%。
- **消融实验数据** (S1 8×8/10雷, 10000 games, 5 epochs):
  - V1_5 (显式反馈): 83.0% WR, 0.990 action_acc
  - V1 (无 refinement): 74.0% WR, 0.985 action_acc
  - V4 baseline: 63.5% WR → +2ch: 70.5% → +LayerNorm: 64.0% → -features_seq: 68.0%
- **修复**: V4 `_transformer_step` 移除 `features_seq` 注入（每步注入原始 CNN 特征导致 refinement 信号被淹没）
- **修复**: V4 Transformer 末层加回 LayerNorm
- **修复**: V4 DecoderHead 从 1ch 恢复为 2ch (mine + confidence)
- **修复**: `eval_max_steps` 从 16 降为 4，与训练步数对齐

## [未发布] - 2026-06-04

### 架构与代码重构
- **统一数据管道 (Unified Data Pipeline)**: 引入 `TrajectoryPool` (`src/training/trajectory_pool.py`) 彻底替代并移除了 `Dataset` 和 `BoardPool` (`TrainBoardPool`)。
- 移除了冗余的 `src/data/generation/` 目录结构，统一到 `src/data/generator.py` 和 `scripts/generate_data.py`
- **后台异步经验池**: `TrajectoryPool` 现在通过 `multiprocessing` 后台守护进程预先推演并存储完整轨迹。前端暴露 `pop()` (提供 `(mines, visible)` 供在线探索) 和 `batch()` (提供 `(channels, probs, mask)` 供离线蒸馏)，彻底打通 Online 与 Offline 的界限，并解决容量不足时的阻塞问题。
- **统一训练入口**: `scripts/train.py` 作为单一 CLI 入口，通过 `--mode online|supervised` 和 `--loss_type bce|mse` 动态路由到 `train.py`(Online BCE) 或 `train_supervised.py`(Offline npz)。
- **数据管线模块化**: `TrajectoryPool` (`trajectory_pool.py`) 统一离线/在线数据生产；`EvalBoardPool` (`eval_pool.py`) 负责评估缓存。旧 `board_pool.py` 和 `Dataset` 已移除。
- **降低圈复杂度**:
  - `src/training/train.py` 引入 `TrainingContext` (Parameter Object 模式) 将近 200 行的主循环拆分为更小的独立函数，解决多返回值和长参数列表问题。
  - `src/game/probability_solver.py` 重构 `_enumerate_exact` 函数，消除嵌套重名函数和高圈复杂度回溯逻辑。

## [历史记录] - 2026-06-02

### 架构重构 (V3/V4 Dual-Track Architecture)
- **隐式记忆 (Hidden State)**: 引入 64 维 `mem_state`，与 CNN 提取的 1 通道局部概率解耦，大幅提升模型在长程逻辑推导（如反证法）时的"草稿本"容量。
- **双轨推演**: CNN 负责处理 `[board, prev_probs]` 提取局部空间特征，Transformer 负责处理 CNN 特征并维护 `mem_state`。
- **Decoder Head**: 引入 1x1 Conv 并在其中设置 `bias=True`，利用偏置项为高密度棋盘提供全局雷概率先验。

### 训练策略变更
- **废弃 PonderNet**: 彻底移除 Halt Loss 和 Ponder Penalty，因为其会导致梯度冲突与模型"懒惰"。
- **固定步长 BPTT**: 训练阶段改为固定步长 (`train_max_steps=4`) 的 BPTT 展开；推理阶段放开限制 (`eval_max_steps=16`) 并通过物理概率差值判断收敛。
- **设备适配**: `--device` 默认值硬编码为 `auto`，无缝兼容 Mac MPS 与 Windows CUDA。

### 已知问题与待办
- **S3 评估阶段胜率劣化**: 模型在 8x8/32 (50% 密度) 难度下，验证集准确率达 98.6% (Val Loss 0.0012)，但实际评估胜率暴跌至 36.7%。当前正在排查数据分布同源性与单步高精度累计误差 (Action Acc 95.9% ^ N) 导致的问题。

# 更新日志

本项目的所有重要变更都将记录在此文件中。

## [未发布] - 2026-05-31

### 新增
- **全局训练策略**: 新增 `src/config/training_policy.py`，统一监督训练、RL、评估的 refine/reward 默认值
- **收敛检测**: early-stop `max|ΔP| < 1e-3` 替代置信度方案，S3 模型 90% 样本 5 步收敛
- **RL 奖励设计**: `safe=+1`, `floodfill_bonus=+0.05/cell`, `hit_mine=-20`, 无 win bonus/预揭开补分
- **固定尺寸 RL 默认路线**: `train_rl.py` 默认固定棋盘，`--mixed` 才启用混合池
- **RL Board Pool 解耦**: `generate_rl_pool.py` 负责构建，训练侧只读 pool

### 变更
- **三阶段路线**: 混合泛化阶段 → S3 (8×8/25), 密度对齐 10×10/40 (39%→40%)
- **refine 策略**: 监督训练 `k ∈ [1, 16]`，评估上限 16 且收敛早停，RL 固定 16 步
- **CLI 参数**: 移除 `scripts/train.py` 和 `scripts/train_rl.py` 的 `--refine`，禁止局部覆盖
- **RL 环境**: 训练 `mine_continue=True`, 评估 `False` (分别优化信号和测量)
- **阶段入口**: `train_stage.py --stage` 只暴露 S1/S2/S3，历史阶段改用 `--legacy_stage`

### Bug 修复
- `generate_data` 并行 worker 参数冲突 (partial 导致 seed→width)
- `load_pretrained` 误报 "migrated output head" 消息 (print 在 if 外)
- 移除 RL 通关奖励和预揭开补分，避免最后一步获得与动作无关的巨额奖励
- eval `mine_continue` 错配 (修复后立刻回退，评估需测真实胜率)
- `first_done=False` OOD 回退 (模型未学过全覆态)

### 结果
- **S3 (8×8/25) 零样本 10×10/40**: 74% 胜率 (1000 局, 0 stuck)
- **纯 RL from scratch**: 20k 局可从负 return 学到正 return，但样本效率低，不作为主线
- **提前退出**: S3 模型 90% early-stop (mean 5.0 steps, max 16)

---

## 历史 (2026-05-27 ~ 2026-05-29)

### 新增
- 项目初始化：CNN + Transformer 混合架构
- 概率蒸馏数据管线 + D4 增强
- 迭代 Refinement 训练（自适应随机 k + ponder penalty）
- RL 微调管线（REINFORCE + mine_continue）
- 分阶段训练入口 `train_stage.py`
- 混合数据生成器（可变尺寸 + 密度）
- Board Pool 评估缓存

### 架构决策
- 标签选用 MSE (ProbabilitySolver 精确概率)，非 BCE
- 通道设计：11 通道 (covered + flagged + 1-8 numbers + mines_remaining_ratio)
- 位置编码：InterpolatablePE 支持可变尺寸
- Refinement 步数：早期实验曾训练固定 4，旧实现推理最大 12 步

### 已知问题
- RL 微调从旧混合 checkpoint 起步曾退化至 73%（根因：checkpoint 迁移时 confidence 头被清零）
- 10K 数据时 train loss 0.377 vs val loss 0.729（过拟合）
