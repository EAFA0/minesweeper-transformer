#!/usr/bin/env python3
"""Minesweeper Transformer — 分阶段预训练入口

三阶段密度课程:
  S1 (规则):  8×8 / 10雷 → 学习基本扫雷规则
  S2 (密度):  8×8 / 20雷 → 学习雷密度可变
  S3 (泛化):  8×8 / 25雷 → 高密度泛化 (39%密度, 接近评估目标 10×10/40)

用法:
    python scripts/train_stage.py --stage S1        # 从头训练
    python scripts/train_stage.py --stage S2        # 继承 S1 → 密级提升
    python scripts/train_stage.py --stage S3        # 继承 S2 → 高密度泛化
    python scripts/train_stage.py --stage S3 --eval 10 10 40  # 零样本评估
    python scripts/train_stage.py --stage S1 --force_data       # 重新生成数据
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── 阶段预设 ───────────────────────────────────────────────────────────────

STAGES = {
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_samples": 10000, "epochs": 2,
        "data_dir": "data/S1", "save_dir": "checkpoints/S1",
        "lr": 1e-3, "weight_decay": 3e-4,
        "pretrained": None,
        "eval": {"width": 8, "height": 8, "mines": 10},
        "desc": "规则学习 — 8×8/10雷 (16%密度)",
    },
    "S2": {
        "width": 8, "height": 8, "mines": 20,
        "n_samples": 10000, "epochs": 2,
        "data_dir": "data/S2", "save_dir": "checkpoints/S2",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 20},
        "desc": "密度变化 — 8×8/20雷 (31%密度)",
    },
    "S3": {
        "width": 8, "height": 8, "mines": 25,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S3", "save_dir": "checkpoints/S3",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
        "eval": {"width": 10, "height": 10, "mines": 40},  # 零样本评估目标
        "desc": "高密度泛化 — 8×8/25雷 (39%密度, 接近目标 10×10/40)",
    },
}

PRETRAINED_CHAIN = {
    "S1": None,
    "S2": "S1",
    "S3": "S2",
}


def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")


def main():
    p = argparse.ArgumentParser(
        description="Minesweeper Transformer — 三阶段密度课程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/train_stage.py --stage S1\n"
               "  python scripts/train_stage.py --stage S2 --force_data\n"
               "  python scripts/train_stage.py --stage S3\n"
               "  python scripts/train_stage.py --stage S3 --eval 10 10 40",
    )
    p.add_argument("--stage", required=True, choices=list(STAGES.keys()),
                   help="训练阶段: S1 | S2 | S3")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖默认 epoch 数")
    p.add_argument("--lr", type=float, default=None,
                   help="覆盖默认学习率")
    p.add_argument("--force_data", action="store_true",
                   help="强制重新生成训练数据")
    p.add_argument("--resume", action="store_true",
                   help="从已有 checkpoint 续训")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=1000,
                   help="评估时玩的游戏数 (default: 1000)")
    p.add_argument("--n_samples", type=int, default=None,
                   help="覆盖默认训练游戏数")

    # Evaluation-only mode
    p.add_argument("--eval", nargs=3, type=int, metavar=("W", "H", "M"), default=None,
                   help="仅评估: 指定 width height mines (例如 --eval 10 10 40)")
    p.add_argument("--eval_only", action="store_true",
                   help="仅用预设评估参数进行评估，不训练")

    args = p.parse_args()
    cfg = STAGES[args.stage]

    # Merge overrides
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    n_samples = args.n_samples if args.n_samples is not None else cfg["n_samples"]
    pretrained = cfg["pretrained"]

    # Device detection
    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Stage: {args.stage} — {cfg['desc']}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    # Validate pretrained checkpoint
    if pretrained and not Path(pretrained).exists():
        prev_stage = PRETRAINED_CHAIN.get(args.stage)
        if prev_stage:
            print(f"❌ Pretrained checkpoint not found: {pretrained}")
            print(f"   Run --stage {prev_stage} first to generate it.")
            sys.exit(1)

    # ── Determine eval params ──────────────────────────────────────────
    eval_w = cfg["eval"]["width"]
    eval_h = cfg["eval"]["height"]
    eval_m = cfg["eval"]["mines"]
    if args.eval:
        eval_w, eval_h, eval_m = args.eval

    # ── Skip training if eval-only ─────────────────────────────────────
    if args.eval_only:
        ckpt = Path(cfg["save_dir"]) / "best_model.pt"
        if ckpt.exists():
            eval_cmd = [
                sys.executable, "scripts/evaluate.py",
                str(ckpt),
                "--width", str(eval_w),
                "--height", str(eval_h),
                "--mines", str(eval_m),
                "--n_games", str(args.n_games),
                "--device", args.device,
            ]
            run(eval_cmd, f"Evaluate {eval_w}×{eval_h}/{eval_m}")
        else:
            print(f"❌ No checkpoint at {ckpt}")
        return

    # ── 1) Generate data ──────────────────────────────────────────────
    cmd = [
        sys.executable, "scripts/generate_data.py",
        "--width", str(cfg["width"]),
        "--height", str(cfg["height"]),
        "--mines", str(cfg["mines"]),
        "--n_samples", str(n_samples),
        "--output", cfg["data_dir"],
        "--workers", "0",            # 0 = auto (cpu_count)
    ]
    if args.force_data:
        cmd.append("--force")
    run(cmd, f"{args.stage}: Generate data")

    # ── 2) Train ──────────────────────────────────────────────────────
    cmd = [
        sys.executable, "scripts/train.py",
        "--data_dir", cfg["data_dir"],
        "--epochs", str(epochs),
        "--save_dir", cfg["save_dir"],
        "--device", args.device,
        "--lr", str(lr),
        "--weight_decay", str(cfg["weight_decay"]),
        "--refine", "8",
    ]
    if pretrained and not args.resume:
        cmd.extend(["--pretrained", pretrained])
    if args.resume:
        resume_ckpt = Path(cfg["save_dir"]) / "final_model.pt"
        if resume_ckpt.exists():
            cmd.extend(["--resume", str(resume_ckpt)])
        else:
            print(f"⚠ No checkpoint to resume from: {resume_ckpt}")
    run(cmd, f"{args.stage}: Train ({epochs} epochs)")

    # ── 3) Evaluate ───────────────────────────────────────────────────
    ckpt = Path(cfg["save_dir"]) / "best_model.pt"
    if ckpt.exists():
        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            str(ckpt),
            "--width", str(eval_w),
            "--height", str(eval_h),
            "--mines", str(eval_m),
            "--n_games", str(args.n_games),
            "--device", args.device,
        ]
        run(eval_cmd, f"Evaluate {eval_w}×{eval_h}/{eval_m}")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
