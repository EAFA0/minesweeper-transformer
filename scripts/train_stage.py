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

    n_games = args.n_games if args.n_games is not None else cfg["n_games"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    pretrained = cfg["pretrained"]

    eval_w = cfg["width"]
    eval_h = cfg["height"]
    eval_m = cfg["mines"]
    if args.eval:
        eval_w, eval_h, eval_m = args.eval

    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name} — {cfg['desc']}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    # Validate pretrained checkpoint
    if pretrained and not Path(pretrained).exists():
        print(f"❌ Pretrained checkpoint not found: {pretrained}")
        print(f"   Run the previous stage first.")
        return

    # Eval-only mode
    if args.eval_only:
        ckpt = Path(cfg["save_dir"]) / "best_model.pt"
        if ckpt.exists():
            eval_cmd = [
                *PYTHON_CMD, "scripts/evaluate.py",
                str(ckpt),
                "--arch", str(args.arch),
                "--width", str(eval_w),
                "--height", str(eval_h),
                "--mines", str(eval_m),
                "--n_games", str(args.eval_games),
                "--device", args.device,
            ]
            run(eval_cmd, f"Evaluate {stage_name}")
        else:
            print(f"❌ No checkpoint: {ckpt}")
        return

    # Run Training
    train_cmd = [
        *PYTHON_CMD, "scripts/train.py",
        "--board_width", str(cfg["width"]),
        "--board_height", str(cfg["height"]),
        "--board_mines", str(cfg["mines"]),
        "--n_games", str(n_games),
        "--save_dir", cfg["save_dir"],
        "--device", args.device,
        "--lr", str(lr),
        "--weight_decay", str(cfg["weight_decay"]),
        "--mode", str(args.mode),
        "--arch", str(args.arch),
        "--pool_workers", str(args.pool_workers),
    ]
    if args.data_dir:
        train_cmd.extend(["--data_dir", args.data_dir])
    if pretrained and not args.resume:
        train_cmd.extend(["--pretrained", pretrained])
    if args.resume:
        resume_ckpt = Path(cfg["save_dir"]) / "final_model.pt"
        if resume_ckpt.exists():
            train_cmd.extend(["--resume", str(resume_ckpt)])
        else:
            print(f"⚠ No checkpoint to resume: {resume_ckpt}")
    run(train_cmd, f"{stage_name}: Train ({n_games} games)")

    # Evaluate
    ckpt = Path(cfg["save_dir"]) / "best_model.pt"
    if ckpt.exists():
        eval_cmd = [
            *PYTHON_CMD, "scripts/evaluate.py",
            str(ckpt),
            "--arch", str(args.arch),
            "--width", str(eval_w),
            "--height", str(eval_h),
            "--mines", str(eval_m),
            "--n_games", str(args.eval_games),
            "--device", args.device,
        ]
        run(eval_cmd, f"Evaluate {stage_name}")
    else:
        print(f"⚠ No best checkpoint at {ckpt}")


def main():
    p = argparse.ArgumentParser(
        description="Minesweeper Transformer — Online BCE 分阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/train_stage.py --all\n"
               "  python scripts/train_stage.py --stage S1\n"
               "  python scripts/train_stage.py --stage S3 --eval 10 10 40",
    )
    p.add_argument("--stage", choices=list(STAGES.keys()),
                   help="训练阶段: " + " | ".join(STAGES.keys()))
    p.add_argument("--all", action="store_true",
                   help="运行主线全部: " + " → ".join(STAGES.keys()))
    p.add_argument("--n_games", type=int, default=None,
                   help="覆盖默认训练游戏数")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default="auto",
                   choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--mode", type=str, default="online", choices=["online", "supervised"], help="Training mode")
    p.add_argument("--arch", type=str, default="V4", choices=["V1", "V1_5", "V4"], help="Model architecture version")
    p.add_argument("--data_dir", type=str, default="", help="Directory for offline npz data (supervised mode)")
    p.add_argument("--pool_workers", type=int, default=4, help="Number of background workers for data generation/loading")
    p.add_argument("--eval_games", type=int, default=200,
                   help="评估游戏数 (default: 200)")
    p.add_argument("--eval", nargs=3, type=int, metavar=("W", "H", "M"),
                   default=None, help="零样本评估: --eval 10 10 40")
    p.add_argument("--eval_only", action="store_true",
                   help="仅评估已有 checkpoint")

    args = p.parse_args()

    if args.all:
        stages_to_run = list(STAGES.keys())
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        print(f"\n核心路线: {' → '.join(STAGES.keys())}")
        print("\n--all  运行全部  |  --stage S1  指定阶段  |  --eval_only 仅评估")
        sys.exit(0)

    # Device detection
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    for stage in stages_to_run:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
