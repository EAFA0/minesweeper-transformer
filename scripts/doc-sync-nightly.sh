#!/usr/bin/env bash
# 夜间文档同步审计 — 仅在有 git 变更时执行，只审查实际变动的文件
# 定时任务: 0 2 * * * (北京时间)
set -euo pipefail

PROJECT_DIR="/home/ubuntu/minesweeper-transformer"
cd "$PROJECT_DIR"
ODE=~/.local/bin/ode
STATE_FILE="$PROJECT_DIR/.doc-sync-last-audit"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始夜间文档审计..."

# 1. 拉取最新代码
git pull origin main --ff-only 2>&1 || { echo "git pull 失败，跳过"; exit 0; }

# 2. 获取上次审计点以来的新提交
LAST_AUDIT=$(cat "$STATE_FILE" 2>/dev/null || echo "HEAD~24hours")
# 如果 LAST_AUDIT 不是有效 commit，回退到 24 小时前
if ! git cat-file -e "$LAST_AUDIT" 2>/dev/null; then
    LAST_AUDIT=$(git log --since="24 hours ago" --format=%H | tail -1)
    [ -z "$LAST_AUDIT" ] && { echo "无有效基线，跳过。"; exit 0; }
fi

NEW_COMMITS=$(git log --oneline "$LAST_AUDIT..HEAD" | wc -l)
if [ "$NEW_COMMITS" -eq 0 ]; then
    echo "自 $LAST_AUDIT 以来无新提交，跳过审计。"
    exit 0
fi

echo "检测到 $NEW_COMMITS 个新提交 (基线: ${LAST_AUDIT:0:7})"

# 3. 提取实际变更的代码文件
CHANGED_FILES=$(git diff --name-only "$LAST_AUDIT" HEAD -- \
    src/ scripts/ docs/ \
    | grep -v __pycache__ \
    | grep -v '.bundle' \
    || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "无相关文件变更，跳过审计。"
    echo "$(git rev-parse HEAD)" > "$STATE_FILE"
    exit 0
fi

echo "变更文件:"
echo "$CHANGED_FILES" | sed 's/^/  /'

# 4. 构建精准 prompt — 只审查这些文件相关的文档
CHANGED_LIST=$(echo "$CHANGED_FILES" | tr '\n' ' ')
PROMPT="你是文档审计员。以下文件在过去 24 小时内有变更，请检查对应的文档是否过时：

变更的代码/脚本文件:
$CHANGED_LIST

需要检查的文档（仅检查与变更相关的部分，不必全量扫描）:
- docs/architecture.md — 如果架构代码变更，检查 ADR 描述
- docs/training-log.md — 如果训练参数变更，检查训练记录
- docs/conventions.md — 如果 CLI 入口变更，检查命令行示例
- docs/metrics.md — 如果 loss/指标逻辑变更，检查指标说明
- AGENTS.md — 如果模块路径变更，检查索引描述
- CHANGELOG.md — 如果有关键功能变更，补充条目

规则:
1. 寻找文档中引用的常量、路径、类名、参数默认值，对照变更文件中的实际值
2. 发现不一致则修复文档（不要改代码）
3. 只修确实被变更影响的文档，不必重写整个文档
4. 如果所有文档已准确，回复 'NO_CHANGES'"

$ODE --model flash "$PROMPT" 2>&1 || echo "[警告] opencode 执行异常，跳过本次审计"

# 5. commit + push (如果有变更)
if ! git diff --quiet docs/ AGENTS.md CHANGELOG.md 2>/dev/null; then
    git add docs/ AGENTS.md CHANGELOG.md
    git commit -m "docs: 夜间同步审计 ($(date '+%Y-%m-%d'))"
    git push origin main
    echo "文档已更新并推送。"
else
    echo "文档无变更，跳过 commit。"
fi

# 6. 更新审计基线
echo "$(git rev-parse HEAD)" > "$STATE_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 夜间文档审计完成。"
