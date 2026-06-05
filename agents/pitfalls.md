# Agent 避坑指南

记录 Agent 在本项目中踩过的坑，防止重复犯错。

## 坑 #1: 多个训练进程同时占用 GPU

- **症状**: GPU 显存占满，训练变慢或 OOM
- **原因**: 启动新训练前未检查已有进程；或 tmux session 里残留旧训练
- **正确做法**: 训练前先 `ps aux | grep train.py` 确认无残留；启动新 tmux session 前 `tmux kill-server`
- **记录日期**: 2026-05-30

## 坑 #2: 单进程数据生成浪费 16 核

- **症状**: 10K 样本生成耗时 ~180 秒
- **原因**: 忘记使用 `generate_data.py` 内置并行 worker
- **正确做法**: 用 `python scripts/generate_data.py --workers 0` 自动使用可用核心；需要固定核心数时使用 `--workers 16`
- **记录日期**: 2026-05-30

## 坑 #3: SSH 远程操作时忘记 venv 路径

- **症状**: `source venv/bin/activate: No such file or directory`
- **原因**: 训练机 venv 在 `.venv/` 而非 `venv/`
- **正确做法**: 训练机上统一用 `source .venv/bin/activate`
- **记录日期**: 2026-05-30

## 坑 #4: confidence 头被清零导致 RL 退化

- **症状**: 旧混合 checkpoint 监督训练 94% → RL 微调后暴跌至 73%
- **原因**: checkpoint 迁移时 `model.load_state_dict(checkpoint)` 与当前模型结构不匹配，confidence 头 (channel 1) 的权重被清零
- **正确做法**: 严格匹配模型结构和 checkpoint 的 key 列表；训练前打印 `model.state_dict().keys()` 验证
- **记录日期**: 2026-05-30

## 坑 #5: 数据目录混乱

- **症状**: `train_stage.py` 报 "Data already exists" 但数据不完整（曾被中途 kill）
- **原因**: 多进程/单进程生成到不同目录 (`data/S1/` vs `data/training/`)，`--force_data` 检查逻辑可能误判
- **正确做法**: 生成数据时指定正确的 `--output`，与 `train_stage.py` 期望的目录一致
- **记录日期**: 2026-05-30

## 预防原则

1. **操作前检查环境**: `nvidia-smi`, `ps aux`, `ls data/`
2. **统一数据目录**: 分阶段训练用 `data/{stage}/` 而非 `data/training/`
3. **验证 checkpoint**: 训练完成后立即评估，确认效果
4. **git 为唯一真相源**: 本地改代码 → push → 训练机 pull

## 坑 #6: RL 奖励设计需要全盘推演

- **症状**: ret 值忽高忽低（-220 ~ 152），baseline 不收敛，eval_wr 反复波动
- **原因**: 多次修补奖励但未端到端验证：
  - 踩雷惩罚太轻 → 模型可能为 floodfill 奖励赌博
  - 踩雷惩罚太重 → 梯度被负样本主导
  - `first_done=False` → 模型面对全覆棋盘 OOD，立刻崩溃
  - 通关奖励/预揭开补分 → 最后一步获得与动作无关的巨额奖励
  - eval `mine_continue` 与训练不一致 → eval 指标不可比
- **正确做法**: 保持即时奖励设计：安全 +1、额外 floodfill 小权重、踩雷负分、无通关奖励、无预揭开补分
- **记录日期**: 2026-05-31

## 坑 #7: eval 环境的 mine_continue 必须 False

- **症状**: eval_wr 恒 100%，失去评估意义
- **原因**: 训练用 `mine_continue=True`（踩雷不死），评估意外继承此设置
- **正确做法**: 训练 mine_continue=True（密集反馈），评估 mine_continue=False（真实规则）
- **记录日期**: 2026-05-31

## 坑 #8: 棋盘需从模型熟悉态出发

- **症状**: `first_done=False` → 模型见到全覆棋盘，P(mine) 输出不可靠，40% 首步踩雷
- **原因**: S3 训练数据从 after-first-click 态开始，模型从未见过全覆状态
- **正确做法**: RL pool 使用 `first_done=True`，让模型从 after-first-click 态接手；不要把预揭开格子补分归因给最后一步动作
- **记录日期**: 2026-05-31

## 坑 #9: refine 默认值不能分散定义

- **症状**: 监督训练、RL、评估胜率不一致，复现实验时需要到处查默认参数
- **原因**: `scripts/train.py`、`scripts/train_rl.py`、`model.predict()` 各自维护 refine 默认值
- **正确做法**: 统一从 `src/config/training_policy.py` 读取；CLI 不支持 `--refine`
- **记录日期**: 2026-06-01

## 坑 #10: BPTT detach 废掉 refinement 梯度 (CRITICAL)

- **症状**: 胜率卡在 ~70% 上不去，refinement 迭代输出几乎不变化
- **原因**: `train_epoch` 中 `if step < refinement_steps-1: detach()` 切断了中间步梯度。Step 3 接收 step 2 的 prev_probs 和 mem_state 都是 detach 纯数值，梯度无法反传。CNN channel 10 (prev_probs) 权重和 mem_state 残差路径梯度恒为 0，模型退化回纯单步前向。
- **正确做法**: refinement 循环中不加 detach，全 BPTT 展开。330K 参数 × 4 步 × 8×8 棋盘只需 ~500MB 显存，detach 完全没有必要。
- **根因溯源**: 此 detach 来自 RL 训练的内存爆炸修复（33GB → 500MB），被不留心从 RL 合入监督训练。
- **记录日期**: 2026-06-03

## 坑 #11: V4 架构缺少 grounding 和残差 — 状态漂移导致高密度失败

- **症状**: 6×6/18 (50%密度) Online BCE 胜率 6.5%，蒸馏 BCE 胜率 12%；格点准确率峰值 82.5%，远低于所需 99%+
- **根因排查**: 审计 V4 代码发现两个设计缺陷：
  1. CNN 特征仅在第一步使用，后续 refinement step 中 Transformer 只能看见自己的隐状态 → 状态逐渐漂移
  2. 无外部残差：`m_{t+1} = Transformer(m_t)`，Transformer 需重建完整表示 → BPTT 梯度容易消失
- **修复 (2026-06-04)**: 
  1. 每步重注 CNN 特征: `x = mem_seq + pe_seq + features_seq`
  2. 外部残差: `return mem_seq + self.transformer(x)`
  3. 去除末端 LayerNorm: 允许 confidence 在循环中增长
- **教训**: 循环架构必须保持 grounding（原始特征注入）+ residual（delta 建模），否则退化为无记忆的单步推理
- **记录日期**: 2026-06-04

## 坑 #12: BatchNorm + batch_size=1 的微妙权衡

- **症状**: 历史上 commit `0d35275` 改为 `model.eval()` 训练，随后 `c8ad0a2` 改回 `model.train()`
- **根因**: 在线 BCE 每次 forward 只处理 1 个棋盘。BatchNorm2d 统计量来自 H×W 个空间位置（8×8=64 样本/通道），虽非退化但噪声大。Running stats 被每步的噪声 EMA 更新，可能导致累积偏置。
- **当前选择**: `model.train()` — V4 CNN 只跑一次，单步噪声可接受；eval() 会完全冻结 running stats，若初始值不匹配更危险
- **未来改进方向**: 换 GroupNorm(num_groups=8) 彻底消除 batch 依赖，或训练初期用 eval 模式预热 BN 统计量
- **教训**: BN 的 batch_size 问题在 2D 场景下被空间维度缓解，不是根因但值得标记
- **记录日期**: 2026-06-04

## 坑 #13: BCE 数值稳定性 — sigmoid + BCE → 应换 _with_logits

- **症状**: 潜在 NaN（当 sigmoid 输出极端 0/1 时）
- **根因**: `torch.sigmoid() + F.binary_cross_entropy()` 组合不如 `F.binary_cross_entropy_with_logits()` 稳定。后者内部用 log-sum-exp trick 避免 log(0) 的 -inf
- **当前状态**: 未修复，实际未触发 NaN，但属最佳实践
- **记录日期**: 2026-06-04

---

*最后更新: 2026-06-04*
