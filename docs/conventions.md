# 项目约定

每轮对话启动时容易遗忘的环境信息、命令行约定、命名规范。

## 环境

| 项目 | 值 |
|------|-----|
| 开发机 | 本 Linux 服务器, `/home/ubuntu/minesweeper-transformer/` |
| 训练机 | RTX 4070 SUPER, `ssh ubuntu@FAEX1.local` |
| venv 路径（训练机） | `~/minesweeper-transformer/.venv/` |
| Python 包管理 | `uv` (开发机), `.venv` 内 pip (训练机) |
| PyTorch 版本（训练机） | 2.12.0+cu130 |
| Git 主仓库 | `git@github.com:EAFA0/minesweeper-transformer.git` |

## tmux 约定

- **训练 session 名**: `train`（`ssh ubuntu@FAEX1.local -t tmux attach -t train`）
- 每次启动新训练时先 `tmux kill-server` 清理旧 session，再创建同名新 session
- 不要在训练 session 中运行其他命令（保持窗口输出干净）

## 命令行约定

### 全局策略

- 跨训练、RL、评估共享的默认策略统一在 `src/config/training_policy.py`
- refine 不再从 CLI 传参；监督训练随机 `k ∈ [1, 16]`，评估上限 16 且收敛早停，RL 固定 16 步保持 rollout/loss 一致
- 如需调整 refine/reward 默认策略，先修改全局策略文件，再同步本文档和 `CHANGELOG.md`

### 数据生成
```bash
# 推荐：固定尺寸并行生成 (auto workers)
python -m src.data.generator --n_samples 10000 --workers 0  # 0=auto

# 混合数据（单进程，因为每个 trajectory 尺寸不同）
python -m src.data.generator --mixed --min_size 4 --max_size 8 \
    --min_density 0.1 --max_density 0.5 --n_samples 12000

# 强制重新生成
python -m src.data.generator --n_samples 10000 --workers 0 --force
```

### 训练
```bash
# 分阶段训练（legacy stage 入口）
python scripts/train_stage.py --stage S1          # 从头训练 8x8/10
python scripts/train_stage.py --stage S2          # 继承 S1 → 8x8/15
python scripts/train_stage.py --stage S3          # 继承 S2 → 8x8/20
python scripts/train_stage.py --stage S4          # 继承 S3 → 8x8/25
python scripts/train_stage.py --stage S5          # 继承 S4 → 8x8/32

# 带强制数据重新生成
python scripts/train_stage.py --stage S1 --force_data

# 仅评估
python scripts/train_stage.py --stage S5 --eval_only
python scripts/train_stage.py --stage S5 --eval 10 10 40  # 零样本评估
python scripts/evaluate.py checkpoints/v5_replay_S5/best_model.pt \
    --width 8 --height 8 --mines 32 --n_games 200 --rule_guard
python scripts/collect_mistakes.py checkpoints/v5_replay_S5/best_model.pt \
    --width 8 --height 8 --mines 32 --n_games 500 --board_pool data \
    --output data/mistakes/S5_rule_guard_failures.npz

# Recipe 模式（当前主线）
python scripts/train_stage.py --recipe v5_curriculum_replay --arch V5

# 只跑 recipe 的 S5（从 checkpoints/v5_replay_S4/best_model.pt 继承）
python scripts/train_stage.py --recipe v5_curriculum_replay \
    --start_phase 5 --end_phase 5 --arch V5

# 直接调 train.py（调试用）
# Online BCE（自我探索，默认）
python scripts/train.py --board_width 8 --board_height 8 --board_mines 10 --n_games 5000 --device cuda \
    --save_dir checkpoints/online_run --lr 1e-4 --weight_decay 3e-4

# Supervised 模式（离线 npz 蒸馏）
python scripts/train.py --mode supervised --data_dir data/S1 --epochs 5 --device cuda \
    --save_dir checkpoints/S1 --lr 1e-3 --weight_decay 3e-4

# 当前主力 loss
python scripts/train.py --loss_type deep_mse_rank --n_games 5000
```

### 评估
```bash
python scripts/evaluate.py checkpoints/S1/best_model.pt \
    --width 8 --height 8 --mines 10 --n_games 100
```

## 命名规范

- **Checkpoint 目录**: `checkpoints/{stage}/` (如 `S1`, `S3`, `rl`)
- **数据目录**: `data/{stage}/` (如 `data/S1/`, `data/mixed/`)
- **RL Pool**: `rl_boards_{W}x{H}_{M}.npz` (如 `rl_boards_10x10_40.npz`)
- **并行生成数据**: `data/` (默认输出，严格 no-guess)
- **gitignore**: `data/` 和 `checkpoints/` 均不入库

## 安全规则

1. **禁止 `rm -rf`**: 用 `mv /tmp/` 代替
2. **禁止 `kill` 进程**: 训练进程用 `Ctrl+C` 或 tmux 关闭窗口
3. **GPU 独占**: 同一时间只跑一个训练进程（避免 OOM 和性能下降）
4. **代码先 push**: 修改代码后 commit + push，训练机 `git pull` 同步

## 常见错误

| 错误 | 正确做法 |
|------|----------|
| 直接在训练机上改代码 | 在开发机改，git push，训练机 pull |
| 忘记 `source .venv/bin/activate` | 训练机的 venv 必须手动激活 |
| 多个 train.py 同时跑 | tmux 中检查 `nvidia-smi` 确认无残留 |
| 跑旧版 S1 (非三阶段路线) | 用 `train_stage.py --stage S1`，refine 走全局策略 |
| 在 CLI 里加 `--refine` | 不支持；修改 `src/config/training_policy.py` |

---

*最后更新: 2026-06-06 (V5 单架构清理, archived 移除)*
