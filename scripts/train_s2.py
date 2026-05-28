#!/usr/bin/env python3
"""S2: PPO 强化学习 — 8×8 / 20 雷

继承 S1 的监督学习权重，通过 PPO 在 20 雷环境下重新学习。
S1 教会模型"1-2-1 是什么意思"；S2 用 RL 教会"高密度下怎么决策"。

用法:
    python scripts/train_s2.py                         # 完整流程
    python scripts/train_s2.py --eval_only              # 仅评估已有 checkpoint
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.ppo_train import PPOConfig, train_ppo
from training.rl_env import MinesweeperEnv, Rewards
from training.ppo_train import ActorCritic, evaluate


def main():
    p = argparse.ArgumentParser(description="S2: PPO RL 8×8 / 20 mines")
    p.add_argument("--pretrained", default="checkpoints/S1/best_model.pt",
                   help="S1 checkpoint to load actor weights from")
    p.add_argument("--total_games", type=int, default=20000)
    p.add_argument("--games_per_update", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_dir", default="checkpoints/S2")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_games", type=int, default=500)
    p.add_argument("--eval_ckpt", default="checkpoints/S2/best_model.pt")

    args = p.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    if args.eval_only:
        # Evaluation mode
        ckpt_path = Path(args.eval_ckpt)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        from model.architecture import ModelConfig
        model = ActorCritic(ModelConfig()).to(args.device)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        env = MinesweeperEnv(width=8, height=8, total_mines=20, rewards=Rewards(),
                             rng=np.random.default_rng(42))
        wr = evaluate(model, env, args.eval_games, args.device)
        print(f"\nS2 Win Rate: {wr:.1%} ({int(wr * args.eval_games)}/{args.eval_games})")
        return

    # Training mode
    pretrained_path = args.pretrained
    if not Path(pretrained_path).exists():
        print(f"⚠ Pretrained model not found: {pretrained_path}")
        print("  Run S1 first: python scripts/train_s1.py")
        print("  Continuing without pretrained weights...")
        pretrained_path = ""

    config = PPOConfig(
        width=8, height=8, total_mines=20,
        pretrained=pretrained_path,
        total_games=args.total_games,
        games_per_update=args.games_per_update,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )

    train_ppo(config)


if __name__ == "__main__":
    main()
