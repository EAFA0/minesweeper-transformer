# 更新日志

本项目的所有重要变更都将记录在此文件中。

## [未发布] - 2026-05-30

### 新增
- **数据**: 新增多进程并行数据生成脚本 `scripts/generate_data_parallel.py`（16核 ~540 局/秒）
- **文档**: 建立完整文档体系（AGENTS.md、CHANGELOG.md、architecture.md、training-log.md、conventions.md、pitfalls.md）
- **环境**: RTX 4070 SUPER 训练环境就绪（ssh ubuntu@FAEX1.local）

### 变更
- **数据**: 并行生成替代单进程版本，S1 数据生成从 ~180s 降至 ~18s（10× 加速）
- **训练**: S1 预训练改用 CUDA，5 epoch 仅需 334s

### 结果
- **S1 (8×8/10雷)**: Val Acc 97.5%, Act Acc 100%, 297K 参数, 5 epochs

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
