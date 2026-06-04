#!/usr/bin/env bash
# 夜间文档同步审计 — 仅在有 git 变更时执行
# 定时任务: 0 2 * * * (北京时间)
set -euo pipefail

PROJECT_DIR="/home/ubuntu/minesweeper-transformer"
cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始夜间文档审计..."

# 1. 拉取最新代码
git pull origin main --ff-only 2>&1 || { echo "git pull 失败，跳过"; exit 0; }

# 2. 检查过去 24 小时有无新提交
CHANGES=$(git log --oneline --since="24 hours ago" | wc -l)
if [ "$CHANGES" -eq 0 ]; then
    echo "过去 24 小时无新提交，跳过审计。"
    exit 0
fi

echo "检测到 $CHANGES 个新提交，启动文档审计..."

# 3. 用 opencode (flash) 执行审计
ODE=~/.local/bin/ode
PROMPT="对比 docs/ 目录文档与 src/ 代码实现的一致性，修复过时的文档声明。

1. 扫描 docs/architecture.md、docs/training-log.md 中所有参数声明（steps、epochs、lr、board_size、channels、dropout 等）
2. 逐项对照 src/ 中的实际代码默认值和实现
3. 发现文档与代码不一致则更新文档使其匹配代码
4. 不要改代码，只修文档
5. 完成后 commit 并 push"

$ODE --model flash "$PROMPT" 2>&1

# 4. 如果有变更则 commit + push
if ! git diff --quiet docs/; then
    git add docs/
    git commit -m "docs: 夜间同步审计 ($(date '+%Y-%m-%d'))"
    git push origin main
    echo "文档已更新并推送。"
else
    echo "文档无变更，跳过 commit。"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 夜间文档审计完成。"
