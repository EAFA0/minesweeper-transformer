#!/usr/bin/env python3
"""Minesweeper Transformer — 多阶段密度课程训练（Online BCE）

核心路线 S1 → S2 → S3:
  S1 (规则):  8×8 / 10雷
  S2 (密度):  8×8 / 20雷
  S3 (高密度): 8×8 / 32雷

Recipe 模式:
  uv run python3 scripts/train_stage.py --recipe v5_s1 --arch V5

用法:
  uv run python3 scripts/train_stage.py --all
  uv run python3 scripts/train_stage.py --stage S1
  uv run python3 scripts/train_stage.py --stage S3 --eval 10 10 40
  uv run python3 scripts/train_stage.py --recipe v5_s1 --arch V5
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import STAGES, RECIPES

PYTHON_CMD = ["uv", "run", "python3"]

def run(cmd, desc=""):
    print(f"\n── {desc}")
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ {desc}")
        sys.exit(1)
    print(f"✅ {desc}")


def _build_train_cmd(phase, arch: str, device: str, extra: dict | None = None) -> list[str]:
    """Build a train.py command from a RecipePhase."""
    cmd = [
        *PYTHON_CMD, "scripts/train.py",
        "--mode", phase.mode,
        "--loss_type", phase.loss_type,
        "--arch", arch,
        "--device", device,
        "--n_games", str(phase.n_games),
        "--lr", str(phase.lr),
        "--save_dir", phase.save_dir,
        "--board_width", str(phase.board_width),
        "--board_height", str(phase.board_height),
        "--board_mines", str(phase.board_mines),
        "--refinement_steps", str(phase.refinement_steps),
    ]
    if phase.pretrained:
        cmd.extend(["--pretrained", phase.pretrained])
    if extra:
        if extra.get("data_dir"):
            cmd.extend(["--data_dir", extra["data_dir"]])
        if extra.get("resume_from"):
            cmd.extend(["--resume_from", extra["resume_from"]])
    return cmd


def _build_eval_cmd(ckpt_path: str, phase, arch: str, device: str,
                    eval_games: int, eval_extra: list | None = None) -> list[str]:
    """Build an evaluate.py command for a phase checkpoint."""
    cmd = [
        *PYTHON_CMD, "scripts/evaluate.py",
        str(ckpt_path),
        "--arch", arch,
        "--n_games", str(eval_games),
        "--device", device,
        "--width", str(phase.board_width),
        "--height", str(phase.board_height),
        "--mines", str(phase.board_mines),
    ]
    if eval_extra:
        cmd.extend(eval_extra)
    return cmd


def run_recipe(recipe_name: str, args):
    """Execute all phases of a training recipe sequentially."""
    recipe = RECIPES[recipe_name]

    print(f"\n{'='*60}")
    print(f"  Recipe: {recipe.name} ({len(recipe.phases)} phases)")
    print(f"  Arch: {args.arch}  |  Device: {args.device}")
    print(f"{'='*60}")

    for i, phase in enumerate(recipe.phases, 1):
        print(f"\n{'─'*60}")
        print(f"  Phase {i}/{len(recipe.phases)}: {phase.desc}")
        print(f"{'─'*60}")

        # Resolve pretrained from previous phase if not explicitly set
        pretrained = phase.pretrained
        if i > 1 and not pretrained:
            prev_phase = recipe.phases[i - 2]
            prev_best = Path(prev_phase.save_dir) / "best_model.pt"
            if prev_best.exists():
                pretrained = str(prev_best)
                print(f"  Auto pretrained: {pretrained}")
            else:
                print(f"  ⚠ Previous phase checkpoint not found: {prev_best}")

        # Check pretrained exists
        if pretrained and not Path(pretrained).exists():
            print(f"  ❌ Pretrained checkpoint not found: {pretrained}")
            print(f"     Run the previous phase first.")
            return

        # Build phase config with resolved pretrained
        phase_config = type(phase)(
            mode=phase.mode,
            loss_type=phase.loss_type,
            n_games=phase.n_games,
            lr=phase.lr,
            board_width=phase.board_width,
            board_height=phase.board_height,
            board_mines=phase.board_mines,
            refinement_steps=phase.refinement_steps,
            pretrained=pretrained,
            save_dir=phase.save_dir,
            desc=phase.desc,
        )

        extra = {}
        if args.data_dir:
            extra["data_dir"] = args.data_dir

        train_cmd = _build_train_cmd(phase_config, args.arch, args.device, extra)
        run(train_cmd, f"Phase {i}: {phase.desc}")

        # Evaluate after each phase
        ckpt = Path(phase.save_dir) / "best_model.pt"
        if ckpt.exists():
            eval_extra = None
            if args.eval:
                eval_extra = [
                    "--width", str(args.eval[0]),
                    "--height", str(args.eval[1]),
                    "--mines", str(args.eval[2]),
                ]
            eval_cmd = _build_eval_cmd(
                str(ckpt), phase, args.arch, args.device,
                args.eval_games, eval_extra,
            )
            run(eval_cmd, f"Evaluate Phase {i}")
        else:
            print(f"  ⚠ No best checkpoint at {ckpt}")


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
    p.add_argument("--recipe", type=str, default=None,
                   help="Recipe name (e.g. v5_s1). Runs all phases sequentially.")
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

    # ── Recipe mode ─────────────────────────────────────────────────────────
    if args.recipe:
        if args.recipe not in RECIPES:
            print(f"Unknown recipe: {args.recipe}")
            print(f"Available: {', '.join(RECIPES.keys())}")
            sys.exit(1)
        run_recipe(args.recipe, args)
        return

    # ── Legacy stage mode ───────────────────────────────────────────────────
    if args.all:
        stages_to_run = list(STAGES.keys())
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        print(f"\n核心路线: {' → '.join(STAGES.keys())}")
        print(f"Recipes: {', '.join(RECIPES.keys())}")
        print("\n--all  运行全部  |  --stage S1  指定阶段  |  --recipe v5_s1  Recipe模式  |  --eval_only 仅评估")
        sys.exit(0)

    for stage in stages_to_run:
        run_stage(stage, args)

if __name__ == "__main__":
    main()
