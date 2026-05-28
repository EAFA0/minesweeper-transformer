"""Model evaluation — play full minesweeper games and measure win rate.

This is the real metric. Accuracy during training is just a proxy.
Win rate tells us whether the model can actually play minesweeper.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minesweeper.game import MinesweeperGame
from minesweeper.constants import CellState, MoveType, GameStatus
from model.architecture import MinesweeperTransformer, ModelConfig


def load_model(checkpoint_path: str, device: str) -> MinesweeperTransformer:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" in ckpt:
        config = ckpt["model_config"]
    else:
        config = ModelConfig()

    model = MinesweeperTransformer(config).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pick_action(
    model: MinesweeperTransformer,
    game: MinesweeperGame,
    device: str,
    threshold: float = 0.5,
) -> Optional[Tuple[MoveType, int, int]]:
    """Choose the next move based on model predictions.

    Strategy: reveal the covered cell with lowest P(mine).
    If all covered cells have P(mine) >= threshold (model thinks they're all mines),
    flag the highest-confidence one instead.

    Returns None if no covered cells remain.
    """
    # Get model predictions
    channels = game.board_to_channels()
    with torch.no_grad():
        x = torch.from_numpy(channels).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(x)).squeeze(0).squeeze(0).cpu().numpy()

    # Only consider covered cells
    covered = game.covered_cells  # (H, W) boolean
    if not covered.any():
        return None

    # Mask out non-covered cells with high probability
    masked_probs = np.where(covered, probs, 2.0)

    best_idx = np.argmin(masked_probs)
    best_r, best_c = divmod(int(best_idx), game.width)

    if probs[best_r, best_c] < threshold:
        # Safe enough to reveal
        return MoveType.REVEAL, best_r, best_c
    else:
        # Everything looks like a mine — flag the most confident one
        # Pick the covered cell with highest P(mine)
        covered_probs = np.where(covered, probs, -1.0)
        flag_idx = np.argmax(covered_probs)
        flag_r, flag_c = divmod(int(flag_idx), game.width)
        return MoveType.FLAG, flag_r, flag_c


def play_one_game(
    model: MinesweeperTransformer,
    device: str,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    max_steps: int = 100,
) -> Tuple[GameStatus, int]:
    """Play one game with the model. Returns (final_status, steps_taken)."""
    if rng is None:
        rng = np.random.default_rng()

    game = MinesweeperGame(width, height, total_mines)

    # Random first click
    r = rng.integers(0, height)
    c = rng.integers(0, width)
    game.make_move(r, c, MoveType.REVEAL)

    steps = 0
    while game.status == GameStatus.PLAYING and steps < max_steps:
        action = pick_action(model, game, device)
        if action is None:
            break

        move_type, mr, mc = action
        game.make_move(mr, mc, move_type)
        steps += 1

    return game.status, steps


def evaluate(
    checkpoint_path: str,
    n_games: int = 1000,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run evaluation. Returns statistics dict."""
    device = torch.device(device)
    model = load_model(checkpoint_path, device)
    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Device: {device}")
    print(f"Games to play: {n_games}")
    print()

    rng = np.random.default_rng(seed)
    results = {"won": 0, "lost": 0, "stuck": 0, "steps": []}
    t0 = time.time()

    for i in range(n_games):
        status, steps = play_one_game(
            model, device, width, height, total_mines, rng=rng
        )
        results["steps"].append(steps)

        if status == GameStatus.WON:
            results["won"] += 1
        elif status == GameStatus.LOST:
            results["lost"] += 1
        else:
            results["stuck"] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            wr = results["won"] / (i + 1)
            print(
                f"  [{i + 1:5d}/{n_games}] "
                f"win={results['won']:4d} ({wr:.1%})  "
                f"loss={results['lost']:4d}  "
                f"stuck={results['stuck']:3d}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - t0
    win_rate = results["won"] / n_games
    avg_steps = np.mean(results["steps"]) if results["steps"] else 0

    print()
    print("═" * 50)
    print(f"Total games:  {n_games}")
    print(f"Won:          {results['won']} ({win_rate:.1%})")
    print(f"Lost:         {results['lost']} ({results['lost']/n_games:.1%})")
    print(f"Stuck:        {results['stuck']} ({results['stuck']/n_games:.1%})")
    print(f"Avg steps:    {avg_steps:.1f}")
    print(f"Time:         {elapsed:.1f}s ({elapsed/n_games:.3f}s/game)")
    print("═" * 50)

    results["win_rate"] = win_rate
    results["avg_steps"] = avg_steps
    results["elapsed"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Minesweeper Transformer by playing full games"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("--n_games", type=int, default=1000,
                        help="Number of games to play (default: 1000)")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, or auto")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    evaluate(
        checkpoint_path=args.checkpoint,
        n_games=args.n_games,
        width=args.width,
        height=args.height,
        total_mines=args.mines,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
