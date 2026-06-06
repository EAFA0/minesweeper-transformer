# 训练记录

当前主线: Online BCE 三阶段训练 — S1(规则) → S2(密度) → S3(高密度泛化)
> **2026-06-06 更新**: V5 constraint channels 从 4 个扩为 8 个，当前模型输入为 19ch。此前 15ch checkpoint 与训练结果保留为历史记录，新架构需重新训练。
> **2026-06-03**: 全线切换 Online BCE。MSE 监督训练、data_dir、LEGACY_STAGES 退役。
> 评估用 EvalBoardPool，训练用 TrajectoryPool。全 BPTT refinement (无 detach)。
> **2026-06-04**: V4 架构落地 (grounding + residual + deep inference)，S3 重定义为 8×8/32。

全局策略: Online BCE 训练，refine 固定 4 步全 BPTT（无 detach），Deep Inference 评估/推理 16 步且收敛早停。

## 日志格式

| 日期 | 阶段 | 设备 | 超参 | Val Loss | Val Acc | 胜率 | Checkpoint | 备注 |
|------|------|------|------|----------|---------|------|------------|------|

---

## 2026-06-06: V5 S1 strict no-guess rerun

| 项目 | 值 |
|------|-----|
| Recipe | `v5_s1` |
| 架构 | V5 constraint residual, 15ch input, 1ch mine logit |
| 设备 | Mac MPS |
| 数据 | `data/`, 8×8/10, 5000 局 strict no-guess |
| 生成命令 | `PYTHONPATH=src uv run python3 scripts/generate_data.py --n_samples 5000 --output data --width 8 --height 8 --mines 10 --workers 8 --samples_per_file 2000 --force` |
| 数据校验 | 5000/5014 accepted, `total_ambiguous_cells=0`, `avg_ambig_per_game=0.0` |
| 训练命令 | `PYTHONPATH=src uv run python3 scripts/train_stage.py --recipe v5_s1 --arch V5 --device auto --eval_games 200` |
| 超参 | supervised `deep_mse`, lr=3e-4, epochs=5, batch=64, refine=4 |
| Epoch 5 loss | 0.0033 |
| 训练内评估 | 94/100 WR, action_acc=0.997 |
| 独立评估 | 187/200 WR = 93.50%, action_acc=0.9963, avg_steps=17.6, avg_refine=4.0 |
| Eval cache oracle | `data/eval_boards_8x8_10.npz`: 200/200 WR, `forced_guess_steps=0` |
| Checkpoint | `checkpoints/v5_s1_deep/best_model.pt` |
| 备注 | `generate_no_guess_board()` 现额外要求本项目 `ProbabilitySolver` 可无猜解完；外部 no-guess 但本 solver 会遇到非零最小概率的棋盘会被拒绝。 |

---

## 🆕 当前主线 (2026-05-30/31 重整)

### S1 — 8×8/10雷 规则学习

| 项目 | 值 |
|------|-----|
| 日期 | 2026-05-30 |
| 棋盘 | 8×8, 10 雷 (15.6%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 (并行 16 进程) |
| 超参 | lr=1e-3, wd=3e-4, batch=64, epochs=2, refine=global 1-16 |
| Val Acc | — (待跑) |
| 胜率 | — (待评估) |
| Checkpoint | `checkpoints/S1/` |
| 备注 | 三阶段首步. 并行生成已整合到 scripts/generate_data.py |

### S2 — 8×8/20雷 密度变化

| 项目 | 值 |
|------|-----|
| 日期 | 待跑 |
| 棋盘 | 8×8, 20 雷 (31.2%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=3e-4, wd=3e-4, batch=64, epochs=2, refine=global 1-16 |
| 继承 | `checkpoints/S1/best_model.pt` |
| Checkpoint | `checkpoints/S2/` |

### S3 — 8×8/32雷 高密度 (50%密度)

| 项目 | 值 |
|------|-----|
| 日期 | 2026-06-04 (重定义) |
| 棋盘 | 8×8, 32 雷 (50.0%) |
| 设备 | RTX 4070 SUPER |
| 数据 | 10000 局 |
| 超参 | lr=1e-4, wd=3e-4, batch=64, epochs=5, refine=4步BPTT |
| 继承 | `checkpoints/S2/best_model.pt` |
| 评估目标 | V4 架构冲击 50% 密度，验证 grounding + residual 有效性 |
| 已知结果 | 待训练 (V4 架构刚落地) |
| Checkpoint | `checkpoints/S3/` |

> **历史 S3**: 旧 S3 定义为 8×8/25雷 (39.1%)，零样本 10×10/40 达 74% 胜率。已被新定义取代。

---
> **RL 微调已放弃 (2026-06-03)**。代码已从 main 移除，历史追溯使用 git。优先用在线 BCE 复现 99%+ 胜率，再考虑 RL。

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

## 2026-06-01: Refinement 有效性交叉验证

> **实验目的**: 验证 refinement 训练是否对监督学习有效，分离训练与推理的贡献。
> **环境**: 开发机 CPU，S1 (8×8/10雷)，1K 样本，2 epoch，batch_size=64，lr=1e-3，weight_decay=3e-4。

| Exp | 训练配置 | 评估配置 | 胜率 | Act Acc | 备注 |
|-----|---------|---------|------|---------|------|
| A | refine=1 (单步) | max_steps=1 | 55.5% | 97.2% | 纯单步 baseline |
| C | refine=1 (单步) | max_steps=16 | 29.5% | 94.8% | 单步模型 + refinement 推理 = OOD 崩溃 |
| D | refine=8 | max_steps=1 | 48.0% | 96.2% | refinement 训练 + 单步推理 |
| B | refine=8 | max_steps=16 | **81.0%** | **98.9%** | 匹配组合，最优 |

**结论**:
- Refinement 训练 + refinement 推理是黄金组合，必须匹配使用
- 混用导致 OOD（单步模型只见过 prev=0.5，refinement 推理时 prev 为模型输出）
- 纯单步 55.5% → refinement 组合 81.0%，+25.5pp，refinement 明确有效
- S2.5 旧模型 99.6% 的"refine 5"实为 no-op（prev_probs 权重被 zero-pad），纯单步质量达到的
## 2026-06-01/02: 重构与 Self-Play 探索

### 架构清理
- **移除 confidence 头**: `num_classes` 2→1，砍掉 confidence_loss + ponder_penalty，纯 MSE loss
- **早停阈值**: `convergence_eps` 1e-3→0.05，refinement 在 3-5 步内收敛退出
- **温度移除**: `train.py` 和 `train_rl.py` 删除 `--temperature`
- **配置集中**: `src/config/training_policy.py` 统一管理所有 refinement 参数

### 监督预训练（新架构，从头训练）
| 阶段 | 棋盘 | 胜率 | Act Acc | 训练耗时 |
|------|------|------|---------|---------|
| S1 | 8×8/10 | 98.0% | 99.9% | 281s |
| S2 | 8×8/20 | 95.5% | 99.8% | 399s |
| S3 (旧) | 8×8/25 → 10×10/40 零样本 | **92-96%** | 99.8% | 890s |

### Self-Play 探索（以 REINFORCE→self-play MSE 替代 RL 微调）

| # | 起点 | 基线WR | LR | Buffer | 峰值 | 最终 | 结论 |
|---|------|--------|-----|--------|------|------|------|
| 1 | S1 | 0% | 1e-5 | ∞ | 16% | 9% | 无限 buffer → 过拟合崩塌 |
| 2 | S2 | 66% | 1e-5 | 环形1024 | **83%** | 52% | LR 太高 → 冲高后崩塌 |
| 3 | S2 | 66% | 1e-6 | 环形1024 | 76% | **72%** | 稳定 +6，LR 是关键 |
| 4 | S3 | 96% | 1e-6 | 环形1024 | 97% | 92% | 基线太高，无训练信号 |

**核心发现**:
- Self-play MSE 有天花板，S2 最佳甜点只到 76%，追不上纯监督 S3 的 96%
- 纯监督 S1→S2→S3 仍是当前最强方案
- 以 100% 为目标的"非纯监督"路径未找到，self-play + MSE 本质仍是监督学习的低效变体

### Refinement 信息传递验证
- prev_probs 权重 100% 非零，信道正常训练
- 因果性确认：不同 prev 输入 → 不同输出
- 链路一致性：步与步之间输入输出逐 bit 一致（Δ < 1e-7）

## 2026-06-02: V3 架构升级与高密度测试崩溃

### 架构演进 (V3 Dual-Track)
- **隐式记忆 (Hidden State)**: 引入 64 维 `mem_state`，与 CNN 提取的 1 通道局部概率解耦。
- **训练方式变更**: 放弃随机 $k$ 步截断，改为固定步长 (BPTT `train_max_steps=4`) 的展开训练，以极小化计算开销并强制模型利用内部循环。
- **参数变化**: 重新梳理输入通道，总参数量稳定在约 33 万。

### 训练与评估结果
| 阶段 | 棋盘 | 训练配置 | Val Loss | Val Acc | 胜率 (Evaluate) | Act Acc |
|------|------|---------|----------|---------|-----------------|---------|
| S1 | 8×8/10 | BPTT 4步, lr=1e-3 | 0.0027 | 97.6% | 98.5% (197/200) | 0.999 |
| S2 | 8×8/20 | 继承S1, BPTT 4步 | (待补充) | (待补充) | (待补充) | (待补充) |
| S3 (旧V3) | 8×8/32 | 继承S2, BPTT 4步 | 0.0012 | 98.6% | **36.7%** (22/60) | **0.959** |

**核心问题与反思**:
1. **S3 评估阶段发生严重劣化**：在验证集上近乎完美的模型 (Val Acc 98.6%)，在实战中胜率暴跌至 36.7%。
2. 此问题在讨论中经历了多次排查猜测（包括评估状态丢失、训练数据污染等），但均未获得最终确定的结论。
3. 当前模型已处于最新提交状态，有待其他 Agent 进一步审查评估逻辑或数据源是否存在隐藏断层。

---

## 2026-06-04 代码审计 & 架构迭代

### V4 架构修复（FAEX1 上午完成）

| 改动 | 说明 |
|------|------|
| Grounding 注入 | CNN 特征 `features_seq` 在每轮 Transformer 自循环中重新注入 |
| 外部残差 | `mem_seq + Transformer(mem_seq + PE + features)`，Transformer 建模 delta |
| 去末端 LayerNorm | 允许 confidence 信号在循环中增长，不被归一化压缩 |
| Deep Inference | `eval_max_steps`: 4 → 16，训练固定 4 步 BPTT，推理可深度思考 |
| dropout | 0.0 → 0.2 (Transformer Encoder) |
| 配置模块化 | ModelConfig / TrainingConfig / StageConfig / STAGES 拆分 |

### S3 碰撞测试重置

S3 重新定义为 `8×8/32`（50% 密度），直接在新 V4 上验证高密度能力。

### 代码审计结论

Nanobot 对完整训练/评估/模型管线进行了逐行审计：
- **无逻辑 bug** — 训练循环、梯度流、数据管线均正确
- 识别了三个非阻塞问题（记录为 pitfalls #11-13）
- BN batch_size=1 在 2D 卷积场景下被空间维度缓解，不构成阻塞

### 待验证

- [ ] 新 V4（grounding + residual + deep inference）在 S3 8×8/32 上的实际表现
- [ ] eval_max_steps=16 时训练/推理解耦是否导致分布偏移

---

*最后更新: 2026-06-04*
