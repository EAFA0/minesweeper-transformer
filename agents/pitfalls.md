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

## 坑 #4: 未监督 confidence 头污染 refinement

- **症状**: 胜率低、refinement 行为不可解释，日志显示的步数与真实 early-stop 行为可能不一致
- **原因**: confidence channel 没有监督信号，却被用于控制 early-stop；旧 checkpoint 迁移还可能让该通道随机或清零
- **正确做法**: V5 只保留 1ch mine logit；提前退出统一使用 `max|P_t - P_{t-1}| < convergence_eps`
- **记录日期**: 2026-06-06

## 坑 #5: 数据目录混乱

- **症状**: `train_stage.py` 报 "Data already exists" 但数据不完整（曾被中途 kill）
- **原因**: 多进程/单进程生成到不同目录 (`data/S1/` vs `data/`)，`--force_data` 检查逻辑可能误判；历史 `train_supervised.py` 还会在离线训练时自动启动 background generator，可能覆盖阶段数据文件。
- **正确做法**: 数据只能由 `scripts/generate_data.py` 生产；阶段数据统一写入 `data/S1` 到 `data/S5`，文件名统一为 `train_{stage}_{W}x{H}_{M}_{index}.npz`；supervised 训练必须只读取显式 `--data_dir`，读不到兼容轨迹时应直接失败。
- **记录日期**: 2026-05-30

## 预防原则

1. **操作前检查环境**: `nvidia-smi`, `ps aux`, `ls data/`
2. **统一数据目录**: 主线训练用 `data/`，并保证默认生成器只产 no-guess 数据
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

> **状态**: 已解决。V4 latent loop 路线已废弃，V5 使用显式反馈 + constraint channels 替代。

## 坑 #12: self_validated 不是 no-guess

- **症状**: S1 胜率卡在 80% 左右，`ProbabilitySolver` oracle 也到不了 99%
- **原因**: `generate_self_validated_board()` 的验证过程允许 safe hint，它生成的是 hint-solvable board，不是严格 no-guess board
- **正确做法**: 主训练/评估默认使用 `generate_no_guess_board()`；该函数必须通过本项目 `ProbabilitySolver` 无猜验证；eval cache 使用 `eval_boards_{W}x{H}_{M}.npz`；`self_validated` 只能用于明确标注的 hint-solvable 实验
- **记录日期**: 2026-06-06

- **症状**: 6×6/18 (50%密度) Online BCE 胜率 6.5%，蒸馏 BCE 胜率 12%；格点准确率峰值 82.5%，远低于所需 99%+
- **根因排查**: 审计 V4 代码发现两个设计缺陷：
  1. CNN 特征仅在第一步使用，后续 refinement step 中 Transformer 只能看见自己的隐状态 → 状态逐渐漂移
  2. 无外部残差：`m_{t+1} = Transformer(m_t)`，Transformer 需重建完整表示 → BPTT 梯度容易消失
- **教训**: 循环架构必须保持 grounding（原始特征注入）+ residual（delta 建模），否则退化为无记忆的单步推理。V5 的显式反馈（每步 CNN 重跑 + prev_probs + constraint channels）彻底解决了此问题。
- **记录日期**: 2026-06-04

## 坑 #12: BatchNorm + batch_size=1 的微妙权衡

- **症状**: 历史上 commit `0d35275` 改为 `model.eval()` 训练，随后 `c8ad0a2` 改回 `model.train()`
- **根因**: 在线 BCE 每次 forward 只处理 1 个棋盘。BatchNorm2d 统计量来自 H×W 个空间位置（8×8=64 样本/通道），虽非退化但噪声大。Running stats 被每步的噪声 EMA 更新，可能导致累积偏置。
- **当前选择**: `model.train()` — V5 CNN 每步重跑，单步噪声可接受；eval() 会完全冻结 running stats，若初始值不匹配更危险
- **未来改进方向**: 换 GroupNorm(num_groups=8) 彻底消除 batch 依赖，或训练初期用 eval 模式预热 BN 统计量
- **教训**: BN 的 batch_size 问题在 2D 场景下被空间维度缓解，不是根因但值得标记
- **记录日期**: 2026-06-04

## 坑 #13: BCE 数值稳定性 — sigmoid + BCE → 应换 _with_logits

- **症状**: 潜在 NaN（当 sigmoid 输出极端 0/1 时）
- **根因**: `torch.sigmoid() + F.binary_cross_entropy()` 组合不如 `F.binary_cross_entropy_with_logits()` 稳定。后者内部用 log-sum-exp trick 避免 log(0) 的 -inf
- **当前状态**: 已修复 (2026-06-06)。训练主路径改为 raw logits + `F.binary_cross_entropy_with_logits()`；概率仅用于 MSE、评估和动作选择。
- **记录日期**: 2026-06-04

---

*最后更新: 2026-06-06*
