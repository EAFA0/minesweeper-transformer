#!/usr/bin/env python3
"""Curriculum training journey — 从 8×8 初学到 16×16 专家。

完整流程（5 阶段递进）：
  S1:   监督学习 8×8/10雷  → 基础模式识别
  S1.5: 监督学习 8×8/15雷  → 中密度过渡（继承S1）
  S2:   监督学习 8×8/20雷  → 高密度模式（继承S1.5）
  S3:   监督学习 12×12/40雷 → 大棋盘适应（继承S2）
  S4:   监督学习 16×16/80雷 → 终极规格（继承S3）

用法:
    python scripts/journey.py --all              # 全跑
    python scripts/journey.py --stage S1_5       # 单阶段
    python scripts/journey.py --stage S1 S1_5 S2 # 多阶段
"""

import argparse
import subprocess
import sys

STAGES = {
    "S1":   {"desc": "8×8/10雷 — 基础模式识别",          "script": "scripts/train_s1.py"},
    "S1_5": {"desc": "8×8/15雷 — 中密度过渡（继承S1）",    "script": "scripts/train_s1_5.py"},
    "S2":   {"desc": "8×8/20雷 — 高密度模式（继承S1.5）",  "script": "scripts/train_s2.py"},
    "S3":   {"desc": "12×12/40雷 — 大棋盘适应（继承S2）",  "script": "scripts/train_s3.py"},
    "S4":   {"desc": "16×16/80雷 — 终极规格（继承S3）",    "script": "scripts/train_s4.py"},
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
    p.add_argument("--all", action="store_true")
    p.add_argument("--stage", nargs="+", choices=list(STAGES.keys()))
    p.add_argument("--device", default="auto")

    args = p.parse_args()

    if not args.all and not args.stage:
        p.print_help()
        print("\n  Example: python scripts/journey.py --all")
        print("           python scripts/journey.py --stage S1_5")
        return

    stages = list(STAGES.keys()) if args.all else args.stage

    for s in stages:
        info = STAGES[s]
        cmd = [sys.executable, info["script"], "--device", args.device]
        if not run(cmd, f"{s}: {info['desc']}"):
            print(f"\n⚠ {s} failed — stopping")
            break

    print("\n✅ Journey complete!")


if __name__ == "__main__":
    main()
