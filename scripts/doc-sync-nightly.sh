#!/usr/bin/env bash
# 夜间文档同步审计 — 仅在有 git 变更时执行，只审查实际变动的文件
# 定时任务: 0 2 * * * (北京时间)
set -euo pipefail

PROJECT_DIR="/home/ubuntu/minesweeper-transformer"
cd "$PROJECT_DIR"
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

# 4. 提取变更摘要，供 AI 审查
CHANGED_LIST=$(echo "$CHANGED_FILES" | tr '\n' ' ')
CHANGED_SUMMARY=$(git diff --stat "$LAST_AUDIT" HEAD -- $CHANGED_LIST | tail -1 || echo "")

echo "变更摘要: $CHANGED_SUMMARY"

# 简单启发式检查：如果 CHANGELOG.md 不在变更列表中，提醒可能需要更新
if ! echo "$CHANGED_FILES" | grep -q 'CHANGELOG.md'; then
    echo "[提醒] CHANGELOG.md 未包含在本次变更中"
fi

# 检查文档是否引用了已删除/重命名的文件
for doc in docs/architecture.md docs/training-log.md docs/conventions.md docs/metrics.md AGENTS.md; do
    if [ -f "$doc" ]; then
        for f in $CHANGED_FILES; do
            fname=$(basename "$f")
            if grep -q "$fname" "$doc" 2>/dev/null; then
                echo "[提醒] $doc 引用了变更文件: $fname"
            fi
        done
    fi
done

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
