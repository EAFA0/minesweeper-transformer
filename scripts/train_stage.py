#!/usr/bin/env python3
"""Minesweeper Transformer — 多阶段密度课程训练（Online BCE）

核心路线 S1 → S2 → S3:
  S1 (规则):  8×8 / 10雷
  S2 (密度):  8×8 / 20雷
  S3 (高密度): 8×8 / 32雷

用法:
  uv run python3 scripts/train_stage.py --all
  uv run python3 scripts/train_stage.py --stage S1
  uv run python3 scripts/train_stage.py --stage S3 --eval 10 10 40
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import STAGES

PYTHON_CMD = ["uv", "run", "python3"]

def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")

def run_stage(stage_name, args):
    cfg = STAGES[stage_name]

    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name} — {cfg['desc']}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    pretrained = cfg.get("pretrained")
    if pretrained and not Path(pretrained).exists():
        print(f"❌ Pretrained checkpoint not found: {pretrained}")
        print("   Run the previous stage first.")
        return

    ckpt = Path(cfg["save_dir"]) / "best_model.pt"

    if args.eval_only:
        if ckpt.exists():
            eval_cmd = [
                *PYTHON_CMD, "scripts/evaluate.py",
                str(ckpt),
                "--stage", stage_name,
                "--arch", str(args.arch),
                "--n_games", str(args.eval_games),
                "--device", args.device,
            ]
            if args.eval:
                eval_cmd.extend(["--width", str(args.eval[0]), "--height", str(args.eval[1]), "--mines", str(args.eval[2])])
            run(eval_cmd, f"Evaluate {stage_name}")
        else:
            print(f"❌ No checkpoint: {ckpt}")
        return

    train_cmd = [
        *PYTHON_CMD, "scripts/train.py",
        "--stage", stage_name,
        "--device", args.device,
        "--mode", str(args.mode),
        "--arch", str(args.arch),
    ]
    
    if args.n_games is not None:
        train_cmd.extend(["--n_games", str(args.n_games)])
    if args.lr is not None:
        train_cmd.extend(["--lr", str(args.lr)])
    if args.data_dir:
        train_cmd.extend(["--data_dir", args.data_dir])
    if args.resume:
        resume_ckpt = Path(cfg["save_dir"]) / "final_model.pt"
        if resume_ckpt.exists():
            train_cmd.extend(["--resume_from", str(resume_ckpt)])
        else:
            print(f"⚠ No checkpoint to resume: {resume_ckpt}")

    run(train_cmd, f"{stage_name}: Train")

    if ckpt.exists():
        eval_cmd = [
            *PYTHON_CMD, "scripts/evaluate.py",
            str(ckpt),
            "--stage", stage_name,
            "--arch", str(args.arch),
            "--n_games", str(args.eval_games),
            "--device", args.device,
        ]
        if args.eval:
            eval_cmd.extend(["--width", str(args.eval[0]), "--height", str(args.eval[1]), "--mines", str(args.eval[2])])
        run(eval_cmd, f"Evaluate {stage_name}")
    else:
        print(f"⚠ No best checkpoint at {ckpt}")

def main():
    p = argparse.ArgumentParser(
        description="Minesweeper Transformer — Online BCE 分阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", choices=list(STAGES.keys()),
                   help="训练阶段: " + " | ".join(STAGES.keys()))
    p.add_argument("--all", action="store_true",
                   help="运行主线全部: " + " → ".join(STAGES.keys()))
    p.add_argument("--n_games", type=int, default=None, help="覆盖默认训练游戏数")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mode", type=str, default="online", choices=["online", "supervised"])
    p.add_argument("--arch", type=str, default="V4", choices=["V1", "V1_5", "V4", "V5"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--eval_games", type=int, default=200)
    p.add_argument("--eval", nargs=3, type=int, metavar=("W", "H", "M"), default=None)
    p.add_argument("--eval_only", action="store_true")

    args = p.parse_args()

    if args.all:
        stages_to_run = list(STAGES.keys())
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        print(f"\n核心路线: {' → '.join(STAGES.keys())}")
        print("\n--all  运行全部  |  --stage S1  指定阶段  |  --eval_only 仅评估")
        sys.exit(0)

    for stage in stages_to_run:
        run_stage(stage, args)

if __name__ == "__main__":
    main()
