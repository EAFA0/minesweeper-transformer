#!/usr/bin/env python3
"""训练旅程概述 — 各阶段和数据流。

完整 Pipeline（4 阶段）:
  Phase 1 — 概率蒸馏 (Supervised)    S1 → S1.5 → S2
  Phase 2 — 密度课程 (Curriculum)    S2.5 → S2.75 → S3d → S3.5d
  Phase 3 — 迭代 Refinement          训练时 unroll 5 步，推理时自适应停止
  Phase 4 — RL 微调 (REINFORCE)      从 S2/S2.5 权重起步，自验证棋盘

用法:
    python scripts/journey.py --all              # 全跑
    python scripts/journey.py --stage S2.5       # 单阶段
"""

import argparse
import subprocess
import sys

STAGES = {
    "S1":    "8×8/10雷 — 基础模式识别",
    "S1.5":  "8×8/15雷 — 中密度过渡",
    "S2":    "8×8/20雷 — 高密度约束",
    "S2.5":  "8×8/25雷 — 极高密度训练",
    "S2.75": "8×8/30雷 — 极限密度",
    "S3":    "12×12/40雷 — 大棋盘适应",
    "S3d":   "10×10/30雷 — 中等密度路径",
    "S3.5d": "10×10/40雷 — 高密度中等棋盘",
    "S4":    "16×16/80雷 — 最大规格",
}

STAGE_LIST = list(STAGES.keys())


def main():
    parser = argparse.ArgumentParser(description="Minesweeper Transformer 训练旅程")
    parser.add_argument("--stage", nargs="+", choices=STAGE_LIST,
                        help="指定阶段（可多个，如 --stage S1 S2）")
    parser.add_argument("--all", action="store_true",
                        help="运行全部阶段")
    args = parser.parse_args()

    if args.all:
        to_run = STAGE_LIST
    elif args.stage:
        to_run = args.stage
    else:
        print("Specify stages with --stage or use --all")
        print(f"\nAvailable stages: {', '.join(STAGE_LIST)}")
        print("\nStage descriptions:")
        for s, desc in STAGES.items():
            print(f"  {s:6s}  {desc}")
        sys.exit(1)

    for stage in to_run:
        subprocess.run(
            [sys.executable, "scripts/train_stage.py", "--stage", stage],
            check=False,
        )


if __name__ == "__main__":
    main()
