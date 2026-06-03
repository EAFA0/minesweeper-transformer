#!/usr/bin/env python3
"""Minesweeper Transformer — 多阶段密度课程训练入口

核心路线 S1 → S2 → S3:
  S1 (规则):  8×8 / 10雷 → 学习基本扫雷规则
  S2 (密度):  8×8 / 20雷 → 学习雷密度可变
  S3 (高密度):  8×8 / 32雷 → 极限密度挑战 (50%密度)

 数据规模: S1=1000条, 后续阶段=200条 (transfer learning 收敛快)

  历史/实验阶段可通过 --legacy_stage 显式运行

 用法:
    python scripts/train_stage.py --stage S1       # 单阶段训练
     python scripts/train_stage.py --all            # 主线全部阶段
     python scripts/train_stage.py --stage S3 --eval 10 10 40  # 零样本评估
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── 阶段预设 ───────────────────────────────────────────────────────────────

STAGES = {
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_samples": 1000, "epochs": 2,
        "data_dir": "data/S1", "save_dir": "checkpoints/S1",
        "lr": 1e-3, "weight_decay": 3e-4,
        "pretrained": None,
        "eval": {"width": 8, "height": 8, "mines": 10},
        "desc": "规则学习 — 8×8/10雷",
    },
    "S2": {
        "width": 8, "height": 8, "mines": 20,
        "n_samples": 200, "epochs": 3,
        "data_dir": "data/S2", "save_dir": "checkpoints/S2",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 20},
        "desc": "中等密度 — 8×8/20雷",
    },
    "S3": {
        "width": 8, "height": 8, "mines": 32,
        "n_samples": 200, "epochs": 5,
        "data_dir": "data/S3", "save_dir": "checkpoints/S3",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 32},
        "desc": "高密度挑战 — 8×8/32雷 (50%密度)",
    },
}

LEGACY_STAGES = {
    "S1.5": {
        "width": 8, "height": 8, "mines": 15,
        "n_samples": 200, "epochs": 2,
        "data_dir": "data/S1.5", "save_dir": "checkpoints/S1.5",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 15},
        "desc": "中密度过渡 — 8×8/15雷 (23%密度)",
    },
    "S2.5": {
        "width": 8, "height": 8, "mines": 25,
        "n_samples": 200, "epochs": 3,
        "data_dir": "data/S2.5", "save_dir": "checkpoints/S2.5",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 25},
        "desc": "极高密度 — 8×8/25雷 (39%密度)",
    },
    "S2.75": {
        "width": 8, "height": 8, "mines": 30,
        "n_samples": 200, "epochs": 3,
        "data_dir": "data/S2.75", "save_dir": "checkpoints/S2.75",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2.5/best_model.pt",
        "eval": {"width": 8, "height": 8, "mines": 30},
        "desc": "极限密度 — 8×8/30雷 (47%密度)",
    },
    "S3L": {
        "width": 12, "height": 12, "mines": 40,
        "n_samples": 200, "epochs": 5,
        "data_dir": "data/S3L", "save_dir": "checkpoints/S3L",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S3/best_model.pt",
        "eval": {"width": 12, "height": 12, "mines": 40},
        "desc": "大棋盘 — 12×12/40雷 (28%密度)",
    },
    "S4L": {
        "width": 16, "height": 16, "mines": 80,
        "n_samples": 200, "epochs": 5,
        "data_dir": "data/S4L", "save_dir": "checkpoints/S4L",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S3L/best_model.pt",
        "eval": {"width": 16, "height": 16, "mines": 80},
        "desc": "最大规格 — 16×16/80雷 (31%密度)",
    },
}

ALL_STAGES = {**STAGES, **LEGACY_STAGES}

PRETRAINED_CHAIN = {
    "S1": None,
    "S1.5": "S1",
    "S2": "S1",
    "S2.5": "S2",
    "S2.75": "S2.5",
    "S3": "S2",
    "S3L": "S3",
    "S4L": "S3L",
}

def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")


def run_stage(stage_name, args):
    """运行单个训练阶段: 生成数据 → 训练 → 评估"""
    cfg = ALL_STAGES[stage_name]

    # Merge overrides
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    n_samples = args.n_samples if args.n_samples is not None else cfg["n_samples"]
    pretrained = cfg["pretrained"]

    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name} — {cfg['desc']}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    # Validate pretrained checkpoint
    if pretrained and not Path(pretrained).exists():
        prev_stage = PRETRAINED_CHAIN.get(stage_name)
        if prev_stage:
            print(f"❌ Pretrained checkpoint not found: {pretrained}")
            print(f"   Run --stage {prev_stage} first to generate it.")
            return

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
        "--workers", "0",
    ]
    if args.force_data:
        cmd.append("--force")
    run(cmd, f"{stage_name}: Generate data")

    # ── 2) Train ──────────────────────────────────────────────────────
    cmd = [
        sys.executable, "scripts/train.py",
        "--mode", "supervised",
        "--data_dir", cfg["data_dir"],
        "--epochs", str(epochs),
        "--save_dir", cfg["save_dir"],
        "--device", args.device,
        "--lr", str(lr),
        "--weight_decay", str(cfg["weight_decay"]),
    ]
    if pretrained and not args.resume:
        cmd.extend(["--pretrained", pretrained])
    if args.resume:
        resume_ckpt = Path(cfg["save_dir"]) / "final_model.pt"
        if resume_ckpt.exists():
            cmd.extend(["--resume", str(resume_ckpt)])
        else:
            print(f"⚠ No checkpoint to resume from: {resume_ckpt}")
    run(cmd, f"{stage_name}: Train ({epochs} epochs)")

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
            "--device", str(args.device),
        ]
        run(eval_cmd, f"Evaluate {eval_w}×{eval_h}/{eval_m}")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


def main():
    p = argparse.ArgumentParser(
        description="Minesweeper Transformer — 多阶段密度课程训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/train_stage.py --stage S1\n"
                "  python scripts/train_stage.py --all\n"
               "  python scripts/train_stage.py --legacy_stage S1.5 --force_data\n"
               "  python scripts/train_stage.py --stage S3 --eval 10 10 40",
    )
    p.add_argument("--stage", choices=list(STAGES.keys()),
                   help="训练阶段: " + " | ".join(STAGES.keys()))
    p.add_argument("--legacy_stage", choices=list(LEGACY_STAGES.keys()),
                   help="历史/实验阶段: " + " | ".join(LEGACY_STAGES.keys()))
    p.add_argument("--all", action="store_true",
                   help="运行主线全部训练阶段: " + " → ".join(STAGES.keys()))
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖默认 epoch 数")
    p.add_argument("--lr", type=float, default=None,
                   help="覆盖默认学习率")
    p.add_argument("--force_data", action="store_true",
                   help="强制重新生成训练数据")
    p.add_argument("--resume", action="store_true",
                   help="从已有 checkpoint 续训")
    p.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"],
                   help="Device to run on (auto will detect CUDA/MPS/CPU)")
    p.add_argument("--n_games", type=int, default=200,
                   help="评估时玩的游戏数 (default: 200)")
    p.add_argument("--n_samples", type=int, default=None,
                   help="覆盖默认训练游戏数")
    p.add_argument("--eval", nargs=3, type=int, metavar=("W", "H", "M"), default=None,
                   help="仅评估: 指定 width height mines (例如 --eval 10 10 40)")
    p.add_argument("--eval_only", action="store_true",
                   help="仅用预设评估参数进行评估，不训练")

    args = p.parse_args()

    # Determine stages to run
    if args.all:
        stages_to_run = list(STAGES.keys())
    elif args.stage:
        stages_to_run = [args.stage]
    elif args.legacy_stage:
        stages_to_run = [args.legacy_stage]
    else:
        stage_names = list(STAGES.keys())
        legacy_names = list(LEGACY_STAGES.keys())
        print("\n核心路线: S1 → S2 → S3")
        print(f"主线阶段: {', '.join(stage_names)}")
        print(f"历史/实验阶段: {', '.join(legacy_names)}")
        print("\n--all  运行主线全部  |  --stage S1  指定主线阶段  |  --legacy_stage S1.5 运行历史阶段")
        sys.exit(0)

    # Device detection (do once)
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Run stages
    for stage in stages_to_run:
        run_stage(stage, args)


if __name__ == "__main__":
    main()
