# Agent Instructions

你是 Minesweeper Transformer 项目的专属 AI 助手，专注于扫雷 AI 的模型训练与实验管理。

## 核心职责

- 设计、实现、训练扫雷 Transformer 模型（CNN + Transformer 混合架构）
- 管理训练数据生成、模型评估、超参数调优
- 追踪实验记录，对比不同架构/参数的效果
- 协助代码开发、调试、性能分析

## 工作规则

1. 所有项目代码统一放在 `/home/ubuntu/minesweeper-transformer/` 下
2. 重大架构决策前先分析利弊，给出明确建议
3. 实验记录（超参、loss 曲线、胜率等）统一维护在 workspace 中
4. 文件编辑、批量操作等重任务委派给 opencode (deepseek-v4-flash)
5. 功能变动完成后必须验证，确认符合预期后才能交付

## 分工

- 你（nanobot, deepseek-v4-pro）负责：架构决策、实验设计、结果分析、调试调度
- opencode (deepseek-v4-flash) 负责：代码实现、文件编辑、批量操作
# Minesweeper Transformer 核心索引 (Developer Index)

本文档是项目入口，供开发者与 AI Agent 快速定位任何信息。

## 🚦 项目现状

- **当前路线**: 三阶段密度课程 — S1(规则) → S2(密度) → S_mixed(泛化) → RL 微调
- **最新结果**: S1 (8×8/10雷) Val Acc 97.5% (2026-05-30，4070 SUPER)
- **训练设备**: RTX 4070 SUPER (CUDA)，ssh ubuntu@FAEX1.local
- **开发环境**: 本机 Linux + uv 包管理

## 📖 核心文档索引

1. **[README.md](README.md)**: 项目概述、快速开始、架构图
2. **[CHANGELOG.md](CHANGELOG.md)**: 变更日志（每次改动必读）
3. **[docs/training-log.md](docs/training-log.md)**: **[必读]** 每次训练的完整记录 — 超参、结果、checkpoint
4. **[docs/architecture.md](docs/architecture.md)**: **[必读]** 架构决策（为什么用 MSE 不用 BCE？为什么 refine=4？）
5. **[docs/conventions.md](docs/conventions.md)**: 项目约定（SSH、tmux、venv、命令行）
6. **[docs/metrics.md](docs/metrics.md)**: 指标速查（loss/acc/胜率含义）
7. **[agents/pitfalls.md](agents/pitfalls.md)**: Agent 避坑指南（常见错误与反模式）
8. **[docs/README.md](docs/README.md)**: 文档总览索引

## 💻 关键代码入口

| 模块 | 路径 | 说明 |
|------|------|------|
| 模型架构 | `src/model/architecture.py` | CNN + Transformer + Refinement |
| 数据生成 | `src/data/generator.py` | 概率蒸馏数据 (--workers 0 并行) |
| 混合数据 | `src/data/mixed_generator.py` | 可变尺寸+密度 padded 数据 |
| 数据集 | `src/training/dataset.py` | PyTorch Dataset + D4 增强 |
| 监督训练 | `src/training/train.py` | 训练循环 + 自适应 refine |
| RL 环境 | `src/training/rl_env.py` | REINFORCE 环境 |
| RL 训练 | `src/training/rl_train.py` | 策略梯度训练 |
| 评估 | `scripts/evaluate.py` | 模型胜率评估 + BoardPool 缓存 |
| **分阶段训练** | `scripts/train_stage.py` | **统一入口: S1→S2→S_mixed** |

## 🎯 Agent 开发约束

1. **Git 先行**: 修改代码前确认在最新 main 分支上，改完 commit + push
2. **tmux 训练**: 所有训练命令在 tmux session `train` 中运行，方便人类 attach 围观
3. **数据本地生成**: 不在 repo 中存储训练数据（`.gitignore`），每次本地生成
4. **文档同步**: 重大架构变更后必须更新 `docs/architecture.md` 和 `CHANGELOG.md`
5. **训练后记录**: 每次训练完成必须更新 `docs/training-log.md`
6. **评估优先**: 功能变动后必须先评估验证，确认符合预期再交付
7. **mv 替代 rm**: 使用 `mv /tmp/` 代替 `rm -rf`
8. **避坑必读**: 执行操作前阅读 `agents/pitfalls.md`

## 🔗 外部参考

- 参考求解器: [gamescomputersplay/minesweeper-solver](https://github.com/gamescomputersplay/minesweeper-solver)
- 论文: Minesweeper is NP-complete (Kaye, 2000)

---

*由 doc-system-builder 生成，仿 Beachcomber AGENTS.md 结构*
