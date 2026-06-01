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
