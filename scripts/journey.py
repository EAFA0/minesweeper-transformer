#!/usr/bin/env python3
"""训练旅程 — 简化 3 阶段 + RL。

Pipeline:
  Phase 1 — 概率蒸馏     S1 → S2 → S3 (混合数据可选)
  Phase 2 — 迭代 Refinement  自适应随机步数训练 + ponder penalty
  Phase 3 — RL 微调         REINFORCE warm-start 从 S3

用法:
    python scripts/journey.py --stage S1 S2 S3
    python scripts/journey.py --all
"""

import argparse
import subprocess
import sys

STAGES = {
    "S1":    "8×8/10雷 — 基础模式识别",
    "S2":    "8×8/20雷 — 高密度约束",
    "S3":    "10×10/40雷 — 大棋盘 + 高密度",
    "S1.5":  "8×8/15雷 — [可选] 中密度过渡",
    "S2.5":  "8×8/25雷 — [可选] 极高密度",
    "S2.75": "8×8/30雷 — [可选] 极限密度",
    "S3L":   "12×12/40雷 — [可选] 大棋盘",
    "S4L":   "16×16/80雷 — [可选] 最大规格",
}

STAGE_LIST = list(STAGES.keys())


def main():
    parser = argparse.ArgumentParser(description="Minesweeper Transformer 训练旅程")
    parser.add_argument("--stage", nargs="+", choices=STAGE_LIST)
    parser.add_argument("--all", action="store_true", help="运行全部阶段")
    parser.add_argument("--rl", action="store_true",
                        help="运行 RL 微调 (需先完成 S3)")
    args = parser.parse_args()

    if args.all:
        to_run = list(STAGES.keys())
    elif args.stage:
        to_run = args.stage
    else:
        print("核心路线: S1 → S2 → S3")
        print(f"\n{', '.join(STAGE_LIST)}")
        print("\n--all  运行全部  |  --stage S1 S2 S3  指定阶段  |  --rl  RL微调")
        sys.exit(1)

    for stage in to_run:
        subprocess.run(
            [sys.executable, "scripts/train_stage.py", "--stage", stage],
            check=False,
        )

    if args.rl:
        print("\n── RL 微调 ──")
        subprocess.run([
            sys.executable, "scripts/train_rl.py",
            "--pretrained", "checkpoints/S3/best_model.pt",
            "--width", "10", "--height", "10", "--mines", "40",
            "--mine_continue", "--total_games", "5000",
        ], check=False)


if __name__ == "__main__":
    main()
