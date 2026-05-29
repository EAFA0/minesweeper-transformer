"""Model evaluation — play full minesweeper games and measure win rate + action accuracy.

Supports two evaluation modes:
  --no_guess : No-guess boards (ms-toollib) — measures pure reasoning ability
  (default)  : Random boards — measures real-world performance including guessing

The model's probability estimates are used to pick moves (always lowest P(mine)).
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
    model = MinesweeperTransformer(ModelConfig()).to(device)
    model.load_pretrained(checkpoint_path, device)
    model.eval()
    return model


class BoardPool:
    """Save and reuse generated boards to avoid slow ms-toollib regeneration.

    Usage:
        pool = BoardPool("boards.npz", width=16, height=16, mines=80)
        for i in range(n_games):
            game = pool.get(i, rng=rng, use_no_guess=True)
            # ... play game ...
    """

    def __init__(self, path: Path, width: int, height: int, mines: int):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.mines = mines
        self._cache_mines: Optional[List[np.ndarray]] = None
        self._cache_visible: Optional[List[np.ndarray]] = None

    def _load(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._cache_mines is not None:
            return self._cache_mines, self._cache_visible or []
        if self.path.exists():
            data = np.load(self.path, allow_pickle=True)
            n = len(data.files) // 2  # mines_i + visible_i per board
            self._cache_mines = [data[f"mines_{i}"] for i in range(n)]
            self._cache_visible = [data[f"visible_{i}"] for i in range(n)]
            return self._cache_mines, self._cache_visible
        return [], []

    def _save(self, mines: List[np.ndarray], visibles: List[np.ndarray]) -> None:
        save_dict = {}
        for i, (m, v) in enumerate(zip(mines, visibles)):
            save_dict[f"mines_{i}"] = m
            save_dict[f"visible_{i}"] = v
        np.savez_compressed(self.path, **save_dict)
        self._cache_mines = mines
        self._cache_visible = visibles

    def get(self, idx: int, rng: np.random.Generator, use_no_guess: bool) -> Optional[MinesweeperGame]:
        """Get board #idx. Generates and caches if not in pool."""
        mines_list, visibles_list = self._load()
        if idx < len(mines_list):
            return MinesweeperGame.from_mine_mask(
                self.width, self.height, mines_list[idx],
                first_done=True, visible=visibles_list[idx],
            )

        # Generate new board
        if use_no_guess:
            from data.no_guess import generate_no_guess_board
            ng_rng = np.random.default_rng(rng.integers(0, 2**31))
            game = generate_no_guess_board(
                width=self.width, height=self.height, total_mines=self.mines,
                rng=ng_rng, max_attempts=200,
            )
        else:
            game = MinesweeperGame(self.width, self.height, self.mines)
            r = rng.integers(0, self.height)
            c = rng.integers(0, self.width)
            game.make_move(r, c, MoveType.REVEAL)

        if game is None or game.status != GameStatus.PLAYING:
            return None

        # Save to pool
        mine_mask = game.get_mine_mask()
        mines_list.append(mine_mask)
        visibles_list.append(game.visible.copy())
        self._save(mines_list, visibles_list)
        return game


def pick_action(
    model: MinesweeperTransformer,
    game: MinesweeperGame,
    device: str,
    refine_steps: int = 1,
) -> Optional[Tuple[MoveType, int, int]]:
    """Choose the next move: reveal the covered cell with lowest P(mine).

    When refine_steps > 1, uses iterative refinement with adaptive stopping
    (stops when probability distribution stabilizes).
    """
    channels = game.board_to_channels()
    with torch.no_grad():
        x = torch.from_numpy(channels).unsqueeze(0).to(device)
        if refine_steps > 1:
            probs = model.predict(x, refine_steps=refine_steps).squeeze(0).squeeze(0).cpu().numpy()
        else:
            probs = model.predict(x).squeeze(0).squeeze(0).cpu().numpy()

    covered = game.covered_cells
    if not covered.any():
        return None

    masked_probs = np.where(covered, probs, 2.0)
    best_idx = np.argmin(masked_probs)
    best_r, best_c = divmod(int(best_idx), game.width)
    return MoveType.REVEAL, best_r, best_c


def play_one_game(
    model: MinesweeperTransformer,
    device: str,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    rng: Optional[np.random.Generator] = None,
    use_no_guess: bool = False,
    no_guess_rng: Optional[np.random.Generator] = None,
    max_steps: int = 200,
    refine_steps: int = 1,
    prebuilt_game: Optional[MinesweeperGame] = None,
) -> dict:
    """Play one game with the model.

    If prebuilt_game is provided (from BoardPool), uses it directly.
    Otherwise generates a new board via ms-toollib or random placement.
    """
    if rng is None:
        rng = np.random.default_rng()

    if prebuilt_game is not None:
        game = prebuilt_game
    elif use_no_guess:
        from data.no_guess import generate_no_guess_board
        ng_rng = no_guess_rng or rng
        game = generate_no_guess_board(
            width=width, height=height, total_mines=total_mines,
            rng=ng_rng, max_attempts=200,
        )
        if game is None:
            return {
                "status": None, "steps": 0,
                "safe_reveals": 0, "mine_hits": 0,
                "action_accuracy": 0.0, "generation_failed": True,
            }
    else:
        game = MinesweeperGame(width, height, total_mines)
        r = rng.integers(0, height)
        c = rng.integers(0, width)
        game.make_move(r, c, MoveType.REVEAL)

    steps = 0
    safe_reveals = 0
    mine_hits = 0

    while game.status == GameStatus.PLAYING and steps < max_steps:
        action = pick_action(model, game, device, refine_steps=refine_steps)
        if action is None:
            break

        move_type, mr, mc = action
        is_safe = not game.get_mine_mask()[mr, mc]

        game.make_move(mr, mc, move_type)
        steps += 1

        if is_safe:
            safe_reveals += 1
        else:
            mine_hits += 1

    return {
        "status": game.status,
        "steps": steps,
        "safe_reveals": safe_reveals,
        "mine_hits": mine_hits,
        "action_accuracy": safe_reveals / max(1, safe_reveals + mine_hits),
        "generation_failed": False,
    }


def evaluate(
    checkpoint_path: str,
    n_games: int = 1000,
    width: int = 8,
    height: int = 8,
    total_mines: int = 10,
    seed: int = 42,
    device: str = "cpu",
    use_no_guess: bool = False,
    refine_steps: int = 1,
    board_pool: Optional[Path] = None,
) -> dict:
    """Run evaluation. Returns statistics dict."""
    device = torch.device(device)
    model = load_model(checkpoint_path, device)
    mode_str = "No-guess (reasoning test)" if use_no_guess else "Random (with guessing)"
    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Device: {device}")
    print(f"Mode: {mode_str}")
    print(f"Games to play: {n_games} ({width}×{height}, {total_mines} mines)")
    print()

    rng = np.random.default_rng(seed)
    # Separate RNG for no-guess generation to keep game sequence deterministic
    ng_rng = np.random.default_rng(seed + 1) if use_no_guess else None

    # Board pool (avoid regenerating slow ms-toollib boards)
    pool = None
    if board_pool:
        pool = BoardPool(board_pool, width, height, total_mines)
        mines, _ = pool._load()
        print(f"Board pool: {len(mines)} boards cached in {board_pool}")
        print()

    results = {
        "won": 0, "lost": 0, "stuck": 0, "gen_failed": 0,
        "steps": [], "action_accuracies": [],
        "total_safe_reveals": 0, "total_mine_hits": 0,
    }
    t0 = time.time()
    gen_failures = 0

    for i in range(n_games):
        prebuilt = pool.get(i, rng, use_no_guess) if pool else None
        game_stats = play_one_game(
            model, device, width, height, total_mines,
            rng=rng, use_no_guess=use_no_guess, no_guess_rng=ng_rng,
            refine_steps=refine_steps,
            prebuilt_game=prebuilt,
        )

        if game_stats.get("generation_failed"):
            gen_failures += 1
            results["gen_failed"] += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1:5d}/{n_games}] gen_failures={gen_failures}")
            continue

        results["steps"].append(game_stats["steps"])
        results["action_accuracies"].append(game_stats["action_accuracy"])
        results["total_safe_reveals"] += game_stats["safe_reveals"]
        results["total_mine_hits"] += game_stats["mine_hits"]

        status = game_stats["status"]
        if status == GameStatus.WON:
            results["won"] += 1
        elif status == GameStatus.LOST:
            results["lost"] += 1
        else:
            results["stuck"] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            played = i + 1 - gen_failures
            wr = results["won"] / max(1, played) if played > 0 else 0
            total_reveals = results["total_safe_reveals"] + results["total_mine_hits"]
            act_acc = results["total_safe_reveals"] / max(1, total_reveals)
            print(
                f"  [{i + 1:5d}/{n_games}] "
                f"win={results['won']:4d} ({wr:.1%})  "
                f"loss={results['lost']:4d}  "
                f"stuck={results['stuck']:3d}  "
                f"act_acc={act_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - t0
    played = n_games - gen_failures
    win_rate = results["won"] / max(1, played) if played > 0 else 0
    avg_steps = np.mean(results["steps"]) if results["steps"] else 0
    total_reveals = results["total_safe_reveals"] + results["total_mine_hits"]
    overall_action_acc = results["total_safe_reveals"] / max(1, total_reveals)

    print()
    print("═" * 60)
    print(f"Mode:              {mode_str}")
    print(f"Total games:       {n_games}")
    if gen_failures > 0:
        print(f"Gen failures:      {gen_failures} (skipped)")
    print(f"Won:               {results['won']} ({win_rate:.1%})")
    print(f"Lost:              {results['lost']} ({results['lost']/max(1,played):.1%})" if played > 0 else f"Lost:              0 (0.0%)")
    print(f"Stuck:             {results['stuck']} ({results['stuck']/max(1,played):.1%})" if played > 0 else f"Stuck:             0 (0.0%)")
    print(f"Action accuracy:   {overall_action_acc:.3f} ({results['total_safe_reveals']}/{total_reveals})")
    print(f"Avg steps:         {avg_steps:.1f}")
    print(f"Time:              {elapsed:.1f}s ({elapsed/max(1,played):.3f}s/game)" if played > 0 else f"Time:              {elapsed:.1f}s")
    print("═" * 60)

    results["win_rate"] = win_rate
    results["avg_steps"] = avg_steps
    results["overall_action_accuracy"] = overall_action_acc
    results["elapsed"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Minesweeper Transformer by playing full games"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("--n_games", type=int, default=500,
                        help="Number of games to play (default: 500)")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--no_guess", action="store_true",
                        help="Evaluate on no-guess boards (pure reasoning, no luck)")
    parser.add_argument("--refine", type=int, default=1,
                        help="Iterative refinement steps during inference (default: 1 = single-pass)")
    parser.add_argument("--board_pool", type=Path, default=None,
                        help="Save/load generated boards to .npz for reuse (speeds up repeated evals)")

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
        use_no_guess=args.no_guess,
        refine_steps=args.refine,
        board_pool=args.board_pool,
    )


if __name__ == "__main__":
    main()
