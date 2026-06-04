# 文档总览

## 文档路由

### 设计决策
- **[architecture.md](architecture.md)**: 架构决策与演进 — 为什么这么设计？每个重大决策的背景、方案对比、最终选择

### 实验记录
- **[training-log.md](training-log.md)**: 训练记录表 — 每次训练的完整超参、结果、checkpoint 位置

### 工程规范
- **[conventions.md](conventions.md)**: 项目约定 — 环境变量、SSH 地址、tmux session 名、命令行简化
- **[metrics.md](metrics.md)**: 指标速查 — loss/acc/胜率/RL 指标的含义与参考值

### 项目入口
- **[../README.md](../README.md)**: 项目概述、快速开始、架构图
- **[../AGENTS.md](../AGENTS.md)**: 核心索引（人类 + AI 双读者）
- **[../CHANGELOG.md](../CHANGELOG.md)**: 变更日志

### 避坑
- **[../agents/pitfalls.md](../agents/pitfalls.md)**: Agent 常见错误与反模式

## 文档维护原则

1. **轻量优先**: 文档应短小精悍，避免"为了文档而文档"
2. **决策必记**: 任何重大架构变更必须在 architecture.md 留下记录
3. **训练必记**: 每次训练完成必须更新 training-log.md
4. **遗忘点即文档点**: 每次发现"之前讨论过但忘了"的事，就是该写进文档的信号
5. **主线一致**: 当前默认主线是 `S1 -> S2 -> S3`，RL 已归档至 `scripts/archived/`

---

*最后更新: 2026-05-31*
