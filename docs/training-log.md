# 训练记录

当前主线: 三阶段监督预训练 — S1(规则) → S2(密度) → S3(高密度泛化) → RL 微调

## 日志格式

| 日期 | 阶段 | 设备 | 超参 | Val Loss | Val Acc | 胜率 | Checkpoint | 备注 |
|------|------|------|------|----------|---------|------|------------|------|

---

## 🆕 当前主线 (2026-05-30/31 重整)

### S1 — 8×8/10雷 规则学习

| 项目 | 值 |
|------|-----|
| 日期 | 2026-05-30 |
| 棋盘 | 8×8, 10 雷 (15.6%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 (并行 16 进程) |
| 超参 | lr=1e-3, wd=3e-4, batch=64, epochs=2, refine=4 |
| Val Acc | — (待跑) |
| 胜率 | — (待评估) |
| Checkpoint | `checkpoints/S1/` |
| 备注 | 三阶段首步. 并行生成已整合到 generate_data.py |

### S2 — 8×8/20雷 密度变化

| 项目 | 值 |
|------|-----|
| 日期 | 待跑 |
| 棋盘 | 8×8, 20 雷 (31.2%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=3e-4, wd=3e-4, batch=64, epochs=2, refine=4 |
| 继承 | `checkpoints/S1/best_model.pt` |
| Checkpoint | `checkpoints/S2/` |

### S3 — 8×8/25雷 高密度泛化

| 项目 | 值 |
|------|-----|
| 日期 | 2026-05-31 |
| 棋盘 | 8×8, 25 雷 (39.1%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=3e-4, wd=3e-4, batch=64, epochs=5, refine=4 |
| 继承 | `checkpoints/S2/best_model.pt` |
| 评估目标 | 10×10/40雷 零样本 |
| 已知结果 | 74% 胜率 (1000 局, 0 stuck) |
| Checkpoint | `checkpoints/S3/` |

### RL 微调 — 10×10/40 目标分布

| 项目 | 值 |
|------|-----|
| 日期 | 2026-05-31 |
| 入口 | `scripts/generate_rl_pool.py` → `scripts/train_rl.py` |
| 预训练 | `checkpoints/S3/best_model.pt` |
| 棋盘 | 10×10, 40 雷 |
| Reward | safe=+1, extra floodfill=+0.05/cell, mine=-20, no win bonus |
| 备注 | 纯 RL from scratch 20k 局可学到正 return，但样本效率低；主线仍为 S3 预训练 + 保守 RL |

---

## ⏳ 历史 (旧多阶段管线，已废弃)

### 2026-05-30: S1 (旧) — 4070S 首训 (refine=1, 5 epochs)

| 项目 | 值 |
|------|-----|
| **Val Loss** | **0.0023** (Epoch 5) |
| **Val Acc** | **97.5%** |
| 备注 | 环境测试用，非最新训练方案 |

### 旧管线结果（Mac M3 Pro MPS）

| 模型 | 任务 | Win Rate | Action Acc |
|------|------|----------|------------|
| S2.5 (refine 5) | 8×8/25 | 99.6% | 1.000 |
| S2.5 (refine 5) | 10×10/40 零样本 | 97.0% | 0.999 |
| S3 (无 refine) | 16×16/40 零样本 | 99.0% | 1.000 |
| 旧混合 checkpoint (refine 4) | 8×8/10 | 94% | - |
| RL (from 旧混合 checkpoint) | 8×8/10 | 73% ↓ | - |

> **RL 退化注记 (2026-05-30)**: 旧混合 checkpoint → RL 微调后从 94% 降至 73%。根因: confidence 头被清零。

---

*最后更新: 2026-05-31*
