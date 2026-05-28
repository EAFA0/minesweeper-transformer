#!/usr/bin/env python3
"""Curriculum training journey — 从 8×8 初学到 16×16 专家。

完整流程：
  S1: 监督学习 (8×8/10雷)  → 学会基础模式识别
  S2: PPO RL    (8×8/20雷)  → 继承S1，高密度策略
  S3: PPO RL    (12×12/40雷) → 继承S2，大棋盘适应
  S4: PPO RL    (16×16/99雷) → 继承S3，终极规格

用法:
    python scripts/journey.py --stage S1     # 跑单个阶段
    python scripts/journey.py --stage S1 S2  # 跑多个阶段
    python scripts/journey.py --all           # 全跑

各阶段也可独立运行:
    python scripts/train_s1.py
    python scripts/train_s2.py
    python scripts/train_s3.py
    python scripts/train_s4.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    "S1": {
        "desc": "监督学习 8×8/10雷 — 基础模式识别",
        "script": "scripts/train_s1.py",
    },
    "S2": {
        "desc": "监督学习 8×8/20雷 — 高密度模式（继承S1）",
        "script": "scripts/train_s2.py",
    },
    "S3": {
        "desc": "监督学习 12×12/40雷 — 大棋盘适应（继承S2）",
        "script": "scripts/train_s3.py",
    },
    "S4": {
        "desc": "监督学习 16×16/80雷 — 终极规格（继承S3）",
        "script": "scripts/train_s4.py",
    },
}


def run(cmd, desc):
    print(f"\n{'='*50}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*50}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ FAILED: {desc}")
        return False
    print(f"✅ {desc}")
    return True


def main():
    p = argparse.ArgumentParser(description="Curriculum training journey")
    p.add_argument("--all", action="store_true", help="Run all stages")
    p.add_argument("--stage", nargs="+", choices=list(STAGES.keys()),
                   help="Run specific stage(s)")
    p.add_argument("--device", default="auto")

    args = p.parse_args()

    if not args.all and not args.stage:
        p.print_help()
        print("\n  Example: python scripts/journey.py --all")
        print("           python scripts/journey.py --stage S1 S2")
        print("           python scripts/train_s1.py     # 独立运行")
        return

    stages_to_run = list(STAGES.keys()) if args.all else args.stage

    for stage in stages_to_run:
        info = STAGES[stage]
        script = info["script"]
        desc = f"{stage}: {info['desc']}"
        cmd = [sys.executable, script, "--device", args.device]
        if not run(cmd, desc):
            print(f"\n⚠ {stage} failed — stopping")
            break

    print("\n✅ Journey complete!")


if __name__ == "__main__":
    main()
