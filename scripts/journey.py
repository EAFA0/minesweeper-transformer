#!/usr/bin/env python3
"""Curriculum training journey — from 8×8 beginner to 16×16 expert.

Runs multiple stages of training with increasing difficulty:
  S1: 8×8 / 10 mines  →  basic patterns (Phase 1)
  S2: 8×8 / 20 mines  →  high density
  S3: 12×12 / 40 mines →  larger board
  S4: 16×16 / 115 mines → full coverage

Each stage loads the previous stage's best model as pretrained weights.
Run all stages:
    python scripts/journey.py --all

Run a specific stage:
    python scripts/journey.py --stage S2

Resume from a stage:
    python scripts/journey.py --stage S3 --resume
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ─── Stage definitions ─────────────────────────────────────────────────────

STAGES = {
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_samples": 10000, "epochs": 50,
        "data_dir": "data/S1",
        "save_dir": "checkpoints/S1",
        "require_win": True,     # Phase 1: only fully solvable boards
        "pretrained": None,
    },
    "S2": {
        "width": 8, "height": 8, "mines": 20,
        "n_samples": 20000, "epochs": 20,
        "data_dir": "data/S2",
        "save_dir": "checkpoints/S2",
        "require_win": False,    # Partial trajectories at high density
        "pretrained": "checkpoints/S1/best_model.pt",
    },
    "S3": {
        "width": 12, "height": 12, "mines": 40,
        "n_samples": 20000, "epochs": 15,
        "data_dir": "data/S3",
        "save_dir": "checkpoints/S3",
        "require_win": False,
        "pretrained": "checkpoints/S2/best_model.pt",
    },
    "S4": {
        "width": 16, "height": 16, "mines": 99,
        "n_samples": 30000, "epochs": 10,
        "data_dir": "data/S4",
        "save_dir": "checkpoints/S4",
        "require_win": False,
        "pretrained": "checkpoints/S3/best_model.pt",
    },
}

STAGE_ORDER = ["S1", "S2", "S3", "S4"]


# ─── Commands ──────────────────────────────────────────────────────────────

def run(cmd: List[str], desc: str = "") -> bool:
    """Run a shell command, print status. Returns True on success."""
    print(f"\n{'─' * 50}")
    print(f"▶ {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 50}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ FAILED: {desc}")
        return False
    print(f"✅ {desc}")
    return True


def generate(stage: str) -> bool:
    s = STAGES[stage]
    cmd = [
        sys.executable, "scripts/generate_data.py",
        "--width", str(s["width"]),
        "--height", str(s["height"]),
        "--mines", str(s["mines"]),
        "--n_samples", str(s["n_samples"]),
        "--output", s["data_dir"],
    ]
    if s["require_win"]:
        cmd.append("--require_win")
    return run(cmd, f"{stage}: Generate data")


def train(stage: str, device: str) -> bool:
    s = STAGES[stage]
    cmd = [
        sys.executable, "scripts/train.py",
        "--data_dir", s["data_dir"],
        "--epochs", str(s["epochs"]),
        "--save_dir", s["save_dir"],
        "--device", device,
    ]
    if s["pretrained"] and Path(s["pretrained"]).exists():
        cmd += ["--pretrained", s["pretrained"]]
    return run(cmd, f"{stage}: Train")


def evaluate(stage: str, device: str, n_games: int = 500) -> bool:
    s = STAGES[stage]
    ckpt = Path(s["save_dir"]) / "best_model.pt"
    if not ckpt.exists():
        print(f"⚠ No checkpoint at {ckpt}, skipping eval")
        return True

    cmd = [
        sys.executable, "scripts/evaluate.py",
        str(ckpt),
        "--width", str(s["width"]),
        "--height", str(s["height"]),
        "--mines", str(s["mines"]),
        "--n_games", str(n_games),
        "--device", device,
    ]
    return run(cmd, f"{stage}: Evaluate")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training journey for Minesweeper Transformer"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all stages sequentially")
    parser.add_argument("--stage", choices=STAGE_ORDER,
                        help="Run a single stage")
    parser.add_argument("--resume", action="store_true",
                        help="Skip data generation if data already exists")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, auto")
    parser.add_argument("--eval_games", type=int, default=500,
                        help="Games to play during evaluation (default: 500)")

    args = parser.parse_args()

    if not args.all and not args.stage:
        parser.print_help()
        print("\nExample: python scripts/journey.py --all")
        print("         python scripts/journey.py --stage S2")
        return

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Resume mode: {args.resume}")

    if args.stage:
        stages_to_run = [args.stage]
    else:
        stages_to_run = STAGE_ORDER

    results = {}

    for stage in stages_to_run:
        print(f"\n{'═' * 60}")
        print(f"  STAGE {stage}")
        print(f"{'═' * 60}")

        # Generate data (skip if resuming and data exists)
        data_dir = Path(STAGES[stage]["data_dir"])
        if args.resume and data_dir.exists() and list(data_dir.glob("data_*.npz")):
            print(f"  Data already exists at {data_dir}, skipping generation")
        else:
            if not generate(stage):
                results[stage] = "FAILED (data)"
                break

        # Train
        if not train(stage, device):
            results[stage] = "FAILED (train)"
            break

        # Evaluate
        if not evaluate(stage, device, args.eval_games):
            results[stage] = "FAILED (eval)"
            break

        results[stage] = "OK"

    # Summary
    print(f"\n{'═' * 60}")
    print("  JOURNEY SUMMARY")
    print(f"{'═' * 60}")
    for stage, status in results.items():
        print(f"  {stage}: {status}")

    all_ok = all(v == "OK" for v in results.values())
    if all_ok:
        print("\n🎉 All stages complete!")
    else:
        print("\n⚠ Some stages failed — see above for details")


if __name__ == "__main__":
    main()
