## [未发布] - 2026-06-06

### V5 单架构清理
- **删除旧模型**: `architecture_v1.py`, `architecture_v1_5.py`, `architecture_v5.py` 已删除。V5 提升为唯一 `architecture.py`，类名 `MinesweeperTransformer`。
- **删除 archived 脚本**: `scripts/archived/` 目录已删除。RL 和旧训练脚本历史追溯使用 git。
- **CLI 收敛**: `--arch` 默认值从 V4 改为 V5，choices 仅保留 `["V5"]`。
- **训练/评估去分支**: `train.py` 移除 V1/V4 分支，`evaluate.py` 的 `load_model()` 移除 arch 分发，`utils.py` 的 `build_model()`/`model_forward()` 简化为 V5-only。
- **移除未监督 confidence head**: V5 输出头从 2ch 收敛为 1ch mine logit；refinement early-stop 改为 `max|P_t - P_{t-1}| < convergence_eps`，评估日志记录真实执行步数。
- **V5 constraint channels 重构**: 输入从 `15ch` 扩为 `19ch`，constraint channels 从 4 个平均统计特征改为 8 个硬约束友好特征，新增 `max_residual`、`min_residual`、`forced_safe_signal`、`forced_mine_signal`、`min_slack_norm`。
- **旧 checkpoint 不再迁移**: `load_pretrained()` 改为 strict load；架构 shape 变化直接失败，历史 checkpoint 通过重新训练处理。
- **新增 best-safe ranking loss**: 新增 `deep_mse_rank = deep_mse + 0.1 * ranking_loss`，显式要求 `target P(mine)=0` 的最安全候选 logit 低于非零目标候选；新增 `v5_s1_rank` 与 `v5_curriculum_rank` 对照 recipe。
- **新增 replay curriculum**: `TrajectoryPool` 支持 `data/S2:0.7,data/S1:0.3` 形式的 weighted replay 数据源；新增 `v5_curriculum_replay`，用于缓解 S1→S4 纯顺序训练造成的低密度遗忘。
- **收敛主线 recipe**: 删除过时的 `v5_s1`、`v5_curriculum`、`v5_curriculum_rank`、`v5_s1_mse` 入口；主线只保留 `v5_s1_rank` 与 `v5_curriculum_replay`。
- **新增最高密度 S5**: `v5_curriculum_replay` 扩展到 S5=8×8/32（50% 密度），S5 使用 `60% S5 + 10% S1/S2/S3/S4` replay mix；`--stage` / `--eval --stage` 支持 S5。
- **支持 recipe 分段执行**: `scripts/train_stage.py --recipe ... --start_phase 5 --end_phase 5` 可直接从已有 S4 checkpoint 进入 S5，无需重跑完整课程。
- **训练记录更新**: `docs/training-log.md` 记录 V5 19ch S1 99.0% WR、S4 replay 95.5% WR、S5 93.0% WR，以及 S5 训练/评估命令。
- **新增 rule guard 诊断**: `scripts/evaluate.py --rule_guard` 会优先选择 `ConstraintSolver` 可证明安全格，用于区分基础规则抖动与高阶排序错误；该结果与裸模型成绩分开记录。
- **S5 rule guard 结果**: V5 S5 裸模型 8×8/32 为 186/200 WR = 93.00%；500 局裸评为 457/500 WR = 91.40%（评估期间 cache 从 200 扩到 500 boards）；启用 `--rule_guard` 后 500 局为 491/500 WR = 98.20%，说明可证明安全格排序抖动是主要剩余误差之一。
- **新增 failure mining 诊断脚本**: `scripts/collect_mistakes.py` 使用裸模型 rollout 收集 `rule_guard_avoidable`、`hard_sorting`、`calibration_drift` 状态；输出训练兼容 `.npz` 和诊断 `.json`。S5 500 局结果保存 441 个错题 states，其中 435 个为 `rule_guard_avoidable`、6 个为 `hard_sorting`，后续优先尝试小比例 hard-example replay。
- **S5 mistake replay 微调结果**: 使用 5% hard-example replay 从 `checkpoints/v5_replay_S5/best_model.pt` 微调到 `checkpoints/v5_replay_S5_mistake_ft/best_model.pt`；S5 裸模型从 457/500 WR = 91.40% 提升到 484/500 WR = 96.80%，`--rule_guard` 提升到 494/500 WR = 98.80%，S1/S4 回归分别为 98.50%/96.00%，无明显退化。
- **S5 mistake replay 二次微调结果**: 使用 8% `S5_after_mistake_ft.npz`、lr=5e-5、epochs=1 微调到 `checkpoints/v5_replay_S5_mistake_ft2/best_model.pt`；S5 裸模型提升到 486/500 WR = 97.20%，`--rule_guard` 提升到 496/500 WR = 99.20%，S1/S4 回归维持 98.50%/96.00%。继续同构 replay 收益进入平台期，下一步优先研究 solver-safe-set ranking loss。
- **固定当前最佳基线复现流程**: `docs/training-log.md` 新增 `v5_replay_S5_mistake_ft2` 完整复现命令链：base S5 failure mining → 第一轮 5% mistake replay → after-mining → 第二轮 8% mistake replay → S5/S1/S4 固定 cache 评估。
- **新增 solver-safe-set ranking loss**: `collect_mistakes.py` 现在保存 `solver_safe_masks_*`；`TrajectoryPool.batch(..., include_solver_safe=True)` 可返回该 mask；新增 `deep_mse_solver_safe_rank`，在 `deep_mse_rank` 基础上要求 `ConstraintSolver` safe set 内最低 logit 低于 safe set 外最低 logit。
- **收敛 solver-safe ranking 目标**: 全 pairwise safe-set ranking 在 S5 上负优化（500 局 96.00%，低于 `mistake_ft2` 97.20%）；`compute_solver_safe_set_ranking_loss()` 改为 set-min objective，只要求 safe set 内最低 logit 低于 safe set 外最低 logit，直接对齐 argmin 动作选择。
- **暂停 solver-safe ranking 路线**: set-min 版本训练内 100 局仅 91/100 WR，仍低于 `mistake_ft2`；当前成功基线保持 `deep_mse_rank + mistake replay v2`。
- **新增 no-arch denoising refinement**: `MinesweeperTransformer.refine()` 支持 `initial_probs`；新增 `deep_mse_denoise_rank`，训练时用 `0.5`、noisy target、random mix、wrong-biased probs 作为不完美概率图输入，学习修正回 solver target；不改 19ch 架构，可继承 `v5_replay_S5_mistake_ft2`。
- **Denoising refinement 结果**: `checkpoints/v5_replay_S5_denoise_rank/best_model.pt` 在 S5 500-board cache 上裸模型达到 490/500 WR = 98.00%，超过 `mistake_ft2` 的 97.20%；保守二次 denoise replay 后 `checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt` 在 `--rule_guard --refine_steps 5` 下达到 498/500 WR = 99.60%。
- **新增 probability-zero guard**: `scripts/evaluate.py --prob_zero_guard` 仅在 `--rule_guard` 找不到可证明 safe cells 时调用 `ProbabilitySolver`，且只在存在 `P(mine)=0` covered cells 时接管动作；候选内部直接按 solver probability 选，不再交给模型排序，避免大组件 fallback 破坏模型排序。
- **新增 rule-mine guard 诊断**: `scripts/evaluate.py --rule_mine_guard` 在无 safe cells 但有 solver-proven mines 时虚拟排除这些 mine candidates，不真实插旗；该路径用于诊断，最终 100% 组合不依赖它。
- **最高密度 100% 结果**: `checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt` 使用 `--refine_steps 5 --rule_guard --prob_zero_guard` 在 S5 8×8/32 固定 1000-board cache 上达到 1000/1000 WR = 100.00%，`prob_zero_guard_actions=1855`；S1 8×8/10 与 S4 8×8/25 回归均为 200/200 WR = 100.00%。
- **修复 flood fill flag 处理**: `_flood_fill()` 现在只展开 `CellState.COVERED`，不会穿过 `FLAGGED` cell；避免真实插旗实验污染棋盘状态时违反扫雷规则。
- **停止 supervised 自动生成数据**: `train_supervised.py` 不再启动 background `generate_data.py` 写入 primary data source；离线训练现在只读取显式 `--data_dir`，避免微调时覆盖 `data/S5` 等阶段数据。
- **修复训练入口 mode 路由**: `scripts/train.py` legacy 分支现在会应用 `--mode`，避免 recipe phase 传入 `--mode supervised` 时被默认 `online` 覆盖。
- **文档同步**: AGENTS.md、README.md、architecture.md、conventions.md、metrics.md、training-log.md、docs/README.md 均已更新。

### 训练 Recipe 系统
- **新增 `src/config/recipe_config.py`**: `RecipePhase` + `TrainingRecipe` dataclass，将训练策略抽象为可命名的多阶段 recipe。
- **收敛 `v5_s1` recipe**: 默认仅保留 supervised calibration (8×8/10, 5000 games)；实测 pure online BCE finetune 会将 83% WR checkpoint 退化到约 15% WR，已从默认 recipe 移除。
- **新增 Deep-MSE 主线**: `v5_s1` 改为 `deep_mse`，对每个 refinement step 都监督 solver probability；S1 200局评估 81.5% WR，高于普通 MSE baseline 的 79.5% WR。
- **保留 MSE baseline recipe**: 新增 `v5_s1_mse` 用于复现实验对照；Rank-MSE 试验未带来收益，未纳入默认 recipe。
- **`scripts/train.py`**: 新增 `--recipe` 和 `--dry_run` 参数，支持单 phase 执行；不传 `--recipe` 时向后兼容。
- **`scripts/train_stage.py`**: 新增 `--recipe` 参数，自动编排 recipe phase 顺序执行 + 每 phase 后评估；自动解析 pretrained checkpoint 链。

### No-Guess 数据链路修复
- **恢复严格 no-guess 默认生成器**: `src/data/generator.py`、`training/evaluate.py`、`TrajectoryPool.get_eval_game()` 重新使用 `generate_no_guess_board()`；主训练/评估不再使用带 safe hint 的 `generate_self_validated_board()`。
- **收紧 no-guess 合同**: `generate_no_guess_board()` 现在会额外验证本项目 `ProbabilitySolver` 能无猜解完；训练轨迹生成遇到 `min(P_mine) > 0` 的状态会拒绝该棋盘。
- **统一数据目录**: 默认训练目录保持为 `data/`，eval cache 保持 `eval_boards_{W}x{H}_{M}.npz`；no-guess 是全局默认语义，不通过额外目录名表达。
- **Recipe 显式绑定数据目录**: `RecipePhase` 新增 `data_dir`，`v5_s1` / `v5_s1_mse` 固定读写 `data/`。
- **V5 四阶段课程命名**: 新增 `v5_curriculum`，阶段统一命名为 `S1 → S2 → S3 → S4`，分别对应 8×8/10、15、20、25 雷；不再使用 `S1.5`/`S2.5` 这类小数阶段名。

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
- **阶段入口**: `train_stage.py --stage` 暴露 S1/S2/S3/S4；历史小数阶段不再作为主线入口

### Bug 修复
- `generate_data` 并行 worker 参数冲突 (partial 导致 seed→width)
- `load_pretrained` 现已移除自动迁移，架构 shape 变化要求重新训练 checkpoint
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
