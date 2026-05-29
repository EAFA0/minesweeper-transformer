#!/usr/bin/env python3
"""Minesweeper Transformer — 分阶段训练入口 (概率蒸馏)

一个脚本替代所有 train_s*.py。
根据 --stage 参数自动选择棋盘尺寸、雷数、预训练权重等。

用法:
    python scripts/train_stage.py --stage S1            # 从头训练 8×8/10雷
    python scripts/train_stage.py --stage S2.5          # 继承 S2 → 8×8/25雷
    python scripts/train_stage.py --stage S3 --epochs 20  # 覆盖默认 epoch
    python scripts/train_stage.py --stage S1 --eval_only   # 仅评估
    python scripts/train_stage.py --stage S2 --force_data  # 强制重新生成数据
    python scripts/train_stage.py --stage S1 --resume      # 从 checkpoint 续训

密度课程 (3 阶段, 密度递增):
    S1 : 8×8 / 10雷   (15.6% 密度, 从头, --refine 4)
    S2 : 8×8 / 20雷   (31.3% 密度, 继承 S1, --refine 4)
    S3 : 10×10 / 40雷 (40.0% 密度, 继承 S2, --refine 4)

   → RL: python scripts/train_rl.py --pretrained checkpoints/S3/best_model.pt

可选阶段 (保留备用):
    S1.5  : 8×8 / 15雷    S2.5  : 8×8 / 25雷
    S2.75 : 8×8 / 30雷    S3L   : 12×12 / 40雷
    S4L   : 16×16 / 80雷

核心理念: 固定小棋盘，通过提升雷密度来增加约束复杂度。
ms-toollib 在 8×8/10×10 上表现良好，避免大棋盘的生成瓶颈。
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── 阶段预设 ───────────────────────────────────────────────────────────────

STAGES = {
    # ===== 核心路线: 密度递增 (3 阶段) =====
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S1", "save_dir": "checkpoints/S1",
        "lr": 1e-3, "weight_decay": 1e-4,
        "pretrained": None,
    },
    "S2": {
        "width": 8, "height": 8, "mines": 20,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S2", "save_dir": "checkpoints/S2",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
    },
    "S3": {
        "width": 10, "height": 10, "mines": 40,
        "n_samples": 10000, "epochs": 8,
        "data_dir": "data/S3", "save_dir": "checkpoints/S3",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
    },
    # ===== 可选: 更细粒度的密度阶梯 =====
    "S1.5": {
        "width": 8, "height": 8, "mines": 15,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S1_5", "save_dir": "checkpoints/S1_5",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
    },
    "S2.5": {
        "width": 8, "height": 8, "mines": 25,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S2_5", "save_dir": "checkpoints/S2_5",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
    },
    "S2.75": {
        "width": 8, "height": 8, "mines": 30,
        "n_samples": 10000, "epochs": 5,
        "data_dir": "data/S2_75", "save_dir": "checkpoints/S2_75",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S2_5/best_model.pt",
    },
    # ===== 可选: 大棋盘 =====
    "S3L": {
        "width": 12, "height": 12, "mines": 40,
        "n_samples": 10000, "epochs": 8,
        "data_dir": "data/S3L", "save_dir": "checkpoints/S3L",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
    },
    "S4L": {
        "width": 16, "height": 16, "mines": 80,
        "n_samples": 2000, "epochs": 8,
        "data_dir": "data/S4L", "save_dir": "checkpoints/S4L",
        "lr": 3e-4, "weight_decay": 1e-4,
        "pretrained": "checkpoints/S3L/best_model.pt",
    },
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
        description="Minesweeper Transformer — 分阶段训练 (概率蒸馏)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/train_stage.py --stage S1\n"
               "  python scripts/train_stage.py --stage S2 --epochs 20 --force_data\n"
               "  python scripts/train_stage.py --stage S1 --eval_only",
    )
    p.add_argument("--stage", required=True, choices=list(STAGES.keys()),
                   help="训练阶段")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖默认 epoch 数")
    p.add_argument("--lr", type=float, default=None,
                   help="覆盖默认学习率")
    p.add_argument("--force_data", action="store_true",
                   help="强制重新生成训练数据")
    p.add_argument("--resume", action="store_true",
                   help="从已有 checkpoint 续训")
    p.add_argument("--refine", type=int, default=4, dest="refinement_steps",
                   help="迭代 refinement 步数 (default: 4)")
    p.add_argument("--eval_only", action="store_true",
                   help="仅评估已有 checkpoint，不训练")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_games", type=int, default=500,
                   help="评估时玩的游戏数 (default: 500)")
    p.add_argument("--n_samples", type=int, default=None,
                   help="覆盖默认训练游戏数")

    args = p.parse_args()
    cfg = STAGES[args.stage]

    # Apply overrides
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    n_samples = args.n_samples if args.n_samples is not None else cfg["n_samples"]
    pretrained = cfg["pretrained"]

    # Device detection
    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Stage: {args.stage} | {cfg['width']}×{cfg['height']} / {cfg['mines']} mines | Device: {args.device}")

    # Validate pretrained
    if pretrained and not Path(pretrained).exists() and not args.eval_only:
        print(f"❌ Pretrained checkpoint not found: {pretrained}")
        prev_stage = {
            "S2": "S1", "S3": "S2",
            "S1.5": "S1", "S2.5": "S2", "S2.75": "S2.5",
            "S3L": "S2", "S4L": "S3L",
        }.get(args.stage)
        if prev_stage:
            print(f"   Run with --stage {prev_stage} first")
        sys.exit(1)

    # ── 1) Generate data ──────────────────────────────────────────────
    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/generate_data.py",
            "--width", str(cfg["width"]),
            "--height", str(cfg["height"]),
            "--mines", str(cfg["mines"]),
            "--n_samples", str(n_samples),
            "--output", cfg["data_dir"],
        ]
        if args.force_data:
            cmd.append("--force")
        run(cmd, f"{args.stage}: Generate data")

    # ── 2) Train ──────────────────────────────────────────────────────
    if not args.eval_only:
        cmd = [
            sys.executable, "scripts/train.py",
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
                print(f"   Starting fresh training")
        if args.refinement_steps > 1:
            cmd.extend(["--refine", str(args.refinement_steps)])
        run(cmd, f"{args.stage}: Train ({epochs} epochs{' +refine' if args.refinement_steps > 1 else ''})")

    # ── 3) Evaluate ───────────────────────────────────────────────────
    ckpt = Path(cfg["save_dir"]) / "best_model.pt"
    if ckpt.exists():
        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            str(ckpt),
            "--width", str(cfg["width"]),
            "--height", str(cfg["height"]),
            "--mines", str(cfg["mines"]),
            "--n_games", str(args.n_games),
            "--device", args.device,
        ]
        run(eval_cmd, f"{args.stage}: Evaluate")
    else:
        print(f"⚠ No checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
