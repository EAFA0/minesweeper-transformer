#!/usr/bin/env python3
"""Run the full S1-S5 -> hard-example -> denoise reproduction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PYTHON = [sys.executable]


def run(cmd: list[str], log_file, desc: str) -> None:
    print(f"\n{'=' * 80}\n{desc}\n$ {' '.join(cmd)}\n{'=' * 80}", flush=True)
    log_file.write(f"\n{'=' * 80}\n{desc}\n$ {' '.join(cmd)}\n{'=' * 80}\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
    ret = proc.wait()
    log_file.flush()
    if ret != 0:
        raise SystemExit(f"Command failed ({ret}): {' '.join(cmd)}")


def train_cmd(
    save_dir: str,
    mines: int,
    data_dir: str,
    device: str,
    loss_type: str = "deep_mse_rank",
    pretrained: str = "",
    epochs: int = 5,
    lr: float = 3e-4,
    eval_games: int = 200,
) -> list[str]:
    cmd = [
        *PYTHON,
        "scripts/train.py",
        "--mode",
        "supervised",
        "--arch",
        "V5",
        "--loss_type",
        loss_type,
        "--data_dir",
        data_dir,
        "--save_dir",
        save_dir,
        "--board_width",
        "8",
        "--board_height",
        "8",
        "--board_mines",
        str(mines),
        "--n_games",
        "10000",
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--eval_games",
        str(eval_games),
        "--device",
        device,
    ]
    if pretrained:
        cmd.extend(["--pretrained", pretrained])
    return cmd


def eval_cmd(
    checkpoint: str,
    mines: int,
    device: str,
    n_games: int = 500,
    *extra: str,
) -> list[str]:
    return [
        *PYTHON,
        "scripts/evaluate.py",
        checkpoint,
        "--width",
        "8",
        "--height",
        "8",
        "--mines",
        str(mines),
        "--n_games",
        str(n_games),
        "--board_pool",
        "data",
        "--device",
        device,
        *extra,
    ]


def collect_cmd(
    checkpoint: str,
    output: str,
    device: str,
    n_games: int = 500,
) -> list[str]:
    return [
        *PYTHON,
        "scripts/collect_mistakes.py",
        checkpoint,
        "--width",
        "8",
        "--height",
        "8",
        "--mines",
        "32",
        "--n_games",
        str(n_games),
        "--board_pool",
        "data",
        "--device",
        device,
        "--output",
        output,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="repro_mps")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval_games", type=int, default=500)
    parser.add_argument("--final_games", type=int, default=1000)
    args = parser.parse_args()

    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{args.prefix}_full_repro_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"Full repro log: {log_path}")

    base = {
        "S1": f"checkpoints/{args.prefix}_v5_replay_S1",
        "S2": f"checkpoints/{args.prefix}_v5_replay_S2",
        "S3": f"checkpoints/{args.prefix}_v5_replay_S3",
        "S4": f"checkpoints/{args.prefix}_v5_replay_S4",
        "S5": f"checkpoints/{args.prefix}_v5_replay_S5",
    }

    with log_path.open("w") as log_file:
        stages = [
            ("S1", 10, "data/S1", "", 3e-4),
            ("S2", 15, "data/S2:0.7,data/S1:0.3", f"{base['S1']}/best_model.pt", 1e-4),
            (
                "S3",
                20,
                "data/S3:0.7,data/S1:0.15,data/S2:0.15",
                f"{base['S2']}/best_model.pt",
                1e-4,
            ),
            (
                "S4",
                25,
                "data/S4:0.7,data/S1:0.1,data/S2:0.1,data/S3:0.1",
                f"{base['S3']}/best_model.pt",
                1e-4,
            ),
            (
                "S5",
                32,
                "data/S5:0.6,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1",
                f"{base['S4']}/best_model.pt",
                1e-4,
            ),
        ]

        for stage, mines, data_dir, pretrained, lr in stages:
            run(
                train_cmd(
                    base[stage],
                    mines,
                    data_dir,
                    args.device,
                    pretrained=pretrained,
                    lr=lr,
                    eval_games=200,
                ),
                log_file,
                f"Train base {stage}",
            )
            run(
                eval_cmd(
                    f"{base[stage]}/best_model.pt",
                    mines,
                    args.device,
                    args.eval_games,
                ),
                log_file,
                f"Evaluate base {stage}",
            )

        s5_base = f"{base['S5']}/best_model.pt"
        m1 = f"data/mistakes/{args.prefix}_S5_rule_guard_failures.npz"
        run(collect_cmd(s5_base, m1, args.device), log_file, "Collect S5 base mistakes")

        ft1 = f"checkpoints/{args.prefix}_v5_replay_S5_mistake_ft"
        run(
            train_cmd(
                ft1,
                32,
                f"data/S5:0.55,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,{m1}:0.05",
                args.device,
                pretrained=s5_base,
                epochs=2,
                lr=1e-4,
                eval_games=200,
            ),
            log_file,
            "Train S5 mistake replay ft1",
        )

        m2 = f"data/mistakes/{args.prefix}_S5_after_mistake_ft.npz"
        run(
            collect_cmd(f"{ft1}/best_model.pt", m2, args.device),
            log_file,
            "Collect S5 ft1 mistakes",
        )

        ft2 = f"checkpoints/{args.prefix}_v5_replay_S5_mistake_ft2"
        run(
            train_cmd(
                ft2,
                32,
                f"data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,{m2}:0.08",
                args.device,
                pretrained=f"{ft1}/best_model.pt",
                epochs=1,
                lr=5e-5,
                eval_games=200,
            ),
            log_file,
            "Train S5 mistake replay ft2",
        )

        denoise1 = f"checkpoints/{args.prefix}_v5_replay_S5_denoise_rank"
        run(
            train_cmd(
                denoise1,
                32,
                f"data/S5:0.52,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,{m2}:0.08",
                args.device,
                loss_type="deep_mse_denoise_rank",
                pretrained=f"{ft2}/best_model.pt",
                epochs=2,
                lr=5e-5,
                eval_games=200,
            ),
            log_file,
            "Train S5 denoise ft1",
        )

        m3 = f"data/mistakes/{args.prefix}_S5_after_denoise_rank.npz"
        run(
            collect_cmd(f"{denoise1}/best_model.pt", m3, args.device),
            log_file,
            "Collect S5 denoise mistakes",
        )

        denoise2 = f"checkpoints/{args.prefix}_v5_replay_S5_denoise_rank_ft2"
        run(
            train_cmd(
                denoise2,
                32,
                f"data/S5:0.50,data/S1:0.1,data/S2:0.1,data/S3:0.1,data/S4:0.1,{m3}:0.10",
                args.device,
                loss_type="deep_mse_denoise_rank",
                pretrained=f"{denoise1}/best_model.pt",
                epochs=1,
                lr=3e-5,
                eval_games=200,
            ),
            log_file,
            "Train S5 denoise ft2",
        )

        final_ckpt = f"{denoise2}/best_model.pt"
        for mines in (10, 15, 20, 25, 32):
            run(
                eval_cmd(final_ckpt, mines, args.device, args.eval_games),
                log_file,
                f"Final naked eval mines={mines}",
            )

        run(
            eval_cmd(
                final_ckpt,
                32,
                args.device,
                args.final_games,
                "--preset",
                "s5_guarded_100",
            ),
            log_file,
            "Final S5 guarded 100% combo eval",
        )

    print(f"Done. Full log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
