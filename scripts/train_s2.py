#!/usr/bin/env python3
"""S2: 8×8 / 20 雷 — 监督 warmup + PPO 强化学习

两步走：
  ① 监督预热（5 epoch）：生成 20 雷部分轨迹数据，微调 S1 模型
     目的：重新校准 P(mine) 输出（S1 学的是 25% 雷密度，S2 是 39%）
  ② PPO RL：在预热模型基础上，用游戏反馈学习策略

用法:
    python scripts/train_s2.py                    # 完整流程
    python scripts/train_s2.py --skip_warmup       # 跳过预热，直接 PPO
    python scripts/train_s2.py --eval_only          # 仅评估
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.ppo_train import ActorCritic, PPOConfig, train_ppo, evaluate
from training.rl_env import MinesweeperEnv, Rewards
from model.architecture import ModelConfig

STAGE = "S2"
WARMUP_EPOCHS = 5
WARMUP_SAMPLES = 5000


def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")


def main():
    p = argparse.ArgumentParser(description="S2: Warmup + PPO RL")
    p.add_argument("--pretrained", default="checkpoints/S1/best_model.pt")
    p.add_argument("--total_games", type=int, default=20000)
    p.add_argument("--save_dir", default="checkpoints/S2")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_warmup", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_games", type=int, default=500)
    p.add_argument("--eval_ckpt", default="checkpoints/S2/best_model.pt")

    args = p.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    warmup_ckpt = save_dir / "warmup.pt"

    # ── Eval only ──
    if args.eval_only:
        ckpt = Path(args.eval_ckpt) if Path(args.eval_ckpt).exists() else warmup_ckpt
        if not ckpt.exists():
            print(f"❌ No checkpoint found at {ckpt}")
            sys.exit(1)

        model = ActorCritic(ModelConfig()).to(args.device)
        ckpt_data = torch.load(ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()

        env = MinesweeperEnv(
            width=8, height=8, total_mines=20,
            rewards=Rewards(), rng=np.random.default_rng(42),
        )
        wr = evaluate(model, env, args.eval_games, args.device)
        print(f"\nS2 Win Rate: {wr:.1%} ({int(wr * args.eval_games)}/{args.eval_games})")
        return

    # ── Supervised warmup ──
    pretrained_path = args.pretrained
    if not Path(pretrained_path).exists():
        print(f"⚠ Pretrained model not found: {pretrained_path}")
        print("  Run S1 first: python scripts/train_s1.py")
        pretrained_path = ""

    if not args.skip_warmup and not warmup_ckpt.exists():
        print("\n" + "=" * 50)
        print("  S2 Warmup: 监督微调 (重新校准 P(mine) 密度)")
        print("=" * 50)

        # Generate warmup data
        warmup_data = "data/S2_warmup"
        run([
            sys.executable, "scripts/generate_data.py",
            "--width", "8", "--height", "8", "--mines", "20",
            "--n_samples", str(WARMUP_SAMPLES),
            "--output", warmup_data,
        ], "S2 Warmup: Generate data (5K games, partial trajectories)")

        # Fine-tune
        run([
            sys.executable, "scripts/train.py",
            "--data_dir", warmup_data,
            "--epochs", str(WARMUP_EPOCHS),
            "--save_dir", str(save_dir),
            "--pretrained", pretrained_path,
            "--device", args.device,
            "--no_augment",  # 不需要增强，数据已经够多样
        ], f"S2 Warmup: Fine-tune ({WARMUP_EPOCHS} epochs)")

        # Rename best_model to warmup.pt
        best = save_dir / "best_model.pt"
        if best.exists():
            best.rename(warmup_ckpt)
            print(f"✅ S2 Warmup complete → {warmup_ckpt}")
        else:
            print("⚠ Warmup training produced no best_model.pt")

    # ── PPO RL ──
    ppo_pretrained = str(warmup_ckpt) if warmup_ckpt.exists() else pretrained_path

    print("\n" + "=" * 50)
    print(f"  S2 PPO: 强化学习微调")
    print(f"  Pretrained: {ppo_pretrained}")
    print("=" * 50)

    config = PPOConfig(
        width=8, height=8, total_mines=20,
        pretrained=ppo_pretrained,
        total_games=args.total_games,
        save_dir=str(save_dir),
        device=args.device,
    )
    train_ppo(config)


if __name__ == "__main__":
    main()
