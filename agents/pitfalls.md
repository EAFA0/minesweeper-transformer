# Agent 避坑指南

记录 Agent 在本项目中踩过的坑，防止重复犯错。

## 坑 #1: 多个训练进程同时占用 GPU

- **症状**: GPU 显存占满，训练变慢或 OOM
- **原因**: 启动新训练前未检查已有进程；或 tmux session 里残留旧训练
- **正确做法**: 训练前先 `ps aux | grep train.py` 确认无残留；启动新 tmux session 前 `tmux kill-server`
- **记录日期**: 2026-05-30

## 坑 #2: 单进程数据生成浪费 16 核

- **症状**: 10K 样本生成耗时 ~180 秒
- **原因**: `generate_data.py` 是单线程的，16 核机器利用率 6%
- **正确做法**: 用 `generate_data_parallel.py --workers 16`，同样数据 ~18 秒（10× 加速）
- **记录日期**: 2026-05-30

## 坑 #3: SSH 远程操作时忘记 venv 路径

- **症状**: `source venv/bin/activate: No such file or directory`
- **原因**: 训练机 venv 在 `.venv/` 而非 `venv/`
- **正确做法**: 训练机上统一用 `source .venv/bin/activate`
- **记录日期**: 2026-05-30

## 坑 #4: confidence 头被清零导致 RL 退化

- **症状**: S_mixed 监督训练 94% → RL 微调后暴跌至 73%
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

---

*最后更新: 2026-05-30*
