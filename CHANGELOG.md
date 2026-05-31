# 更新日志

本项目的所有重要变更都将记录在此文件中。

## [未发布] - 2026-05-31

### 新增
- **收敛检测**: early-stop `max|ΔP| < 1e-3` 替代置信度方案，S3 模型 90% 样本 5 步收敛
- **RL 奖励设计**: mine_continue 训练/评估分离, hit_mine=-5, pre_revealed 补分 (满分 160)
- **固定尺寸 RL**: `--no_mixed` flag, focused on 10×10/40
- **RL Board Pool**: 固定尺寸生成 + eval 池复用

### 变更
- **三阶段路线**: S_mixed → S3 (8×8/25), 密度对齐 10×10/40 (39%→40%)
- **max_refine_steps**: 12→16 (给 10×10/40 留余量)
- **hit_mine**: -10→-5 (降低惩罚，ret 拉正)
- **RL 环境**: 训练 `mine_continue=True`, 评估 `False` (分别优化信号和测量)

### Bug 修复
- `generate_data` 并行 worker 参数冲突 (partial 导致 seed→width)
- `load_pretrained` 误报 "migrated output head" 消息 (print 在 if 外)
- RL `_pre_revealed` 补分 (开局已翻格不计入导致满分 ≠ 160)
- eval `mine_continue` 错配 (修复后立刻回退，评估需测真实胜率)
- `first_done=False` OOD 回退 (模型未学过全覆态)

### 结果
- **S3 (8×8/25) 零样本 10×10/40**: 74% 胜率 (1000 局, 0 stuck)
- **RL v3 (pre_revealed 补分)**: ret=152, baseline=145+, cold eval 88%
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
- Refinement 步数：训练固定 4，推理上限 12

### 已知问题
- RL 微调从 S_mixed 起步曾退化至 73%（根因：checkpoint 迁移时 confidence 头被清零）
- 10K 数据时 train loss 0.377 vs val loss 0.729（过拟合）
