"""TrajectoryPool for unified offline/online data provision.

Provides a unified interface (`batch()` and `pop()`) over a pool of 
Minesweeper trajectories. In supervised mode, it efficiently loads `.npz` 
files from disk and supports runtime `refresh()` to seamlessly incorporate 
newly generated data produced by a background process.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from game.game import MinesweeperGame, board_state_to_channels


def _adjacent_mine_counts(mines: np.ndarray) -> np.ndarray:
    """Return (H, W) int array of 8-neighbor mine counts for each cell."""
    m = mines.astype(np.int64)
    counts = np.zeros_like(m)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(m, dr, axis=0), dc, axis=1)
            # zero out wrapped-around edges
            if dr == 1:
                shifted[0, :] = 0
            elif dr == -1:
                shifted[-1, :] = 0
            if dc == 1:
                shifted[:, 0] = 0
            elif dc == -1:
                shifted[:, -1] = 0
            counts += shifted
    return counts


class TrajectoryPool:
    eval_cache_prefix = "eval_boards"

    @staticmethod
    def _parse_data_sources(data_dir: str) -> List[Tuple[Path, float]]:
        """Parse data_dir as one or more weighted sources.

        Examples:
          data
          data/S2:0.7,data/S1:0.3
        """
        sources = []
        for raw in (data_dir or "data").split(","):
            spec = raw.strip()
            if not spec:
                continue
            path_text, weight = spec, 1.0
            if ":" in spec:
                path_text, weight_text = spec.rsplit(":", 1)
                weight = float(weight_text)
            if weight <= 0:
                raise ValueError(f"Data source weight must be positive: {spec}")
            sources.append((Path(path_text), weight))
        if not sources:
            sources.append((Path("data"), 1.0))
        return sources

    def _load_eval_file(self, p: Path):
        try:
            data = np.load(p, allow_pickle=True)
            n = len(data.files) // 2  # eval board cache has mines & visible
            for i in range(n):
                self._offline_buffer.append({
                    "mines": data[f"mines_{i}"],
                    "masks": [data[f"visible_{i}"]],
                    "actions": []
                })
            print(f"TrajectoryPool: Loaded {n} eval boards from {p}")
        except Exception as e:
            print(f"Error loading {p}: {e}")

    def _load_train_file(self, f: Path) -> int:
        try:
            data = np.load(f, allow_pickle=True)
            n = len([key for key in data.files if key.startswith("mines_")])
            if n == 0:
                # If there are no actions, it might just be empty, safely ignore
                return 0
            loaded = 0
            for i in range(n):
                mines = data[f"mines_{i}"]
                if mines.shape != (self.height, self.width):
                    continue
                if np.sum(mines) != self.mines:
                    continue
                masks = data[f"masks_{i}"]
                if len(masks) == 0:
                    continue
                traj = {
                    "mines": mines,
                    "actions": data[f"actions_{i}"],
                    "masks": masks,
                }
                if f"probs_{i}" in data:
                    traj["probs"] = data[f"probs_{i}"]
                if f"solver_safe_masks_{i}" in data:
                    traj["solver_safe_masks"] = data[f"solver_safe_masks_{i}"]
                self._offline_buffer.append(traj)
                loaded += 1
            return loaded
        except Exception as e:
            print(f"Error loading {f}: {e}")
            return 0

    def __init__(
        self,
        board_width: int,
        board_height: int,
        board_mines: int,
        pool_size: int = 100,
        pool_workers: int = 0,
        data_dir: str = "",
        compute_probs: bool = False,
        eval_mode: bool = False,
    ):
        self.width = board_width
        self.height = board_height
        self.mines = board_mines
        self.pool_size = pool_size
        self.compute_probs = compute_probs
        self.eval_mode = eval_mode
        self.data_dir = data_dir or "data"
        self._data_sources = self._parse_data_sources(self.data_dir)
        
        self._offline_buffer: List[Dict[str, Any]] = []
        self._source_buffers: List[List[Dict[str, Any]]] = [
            [] for _ in self._data_sources
        ]
        self._loaded_files = set()
        
        # Load offline data if data_dir is provided
        self.refresh()
        
    def refresh(self):
        """Scan data_dir for new .npz files and load them into the buffer."""
        if not self.data_dir:
            return
            
        for source_idx, (p, _weight) in enumerate(self._data_sources):
            if not p.exists():
                continue

            if p.is_dir():
                if self.eval_mode:
                    eval_file = p / f"{self.eval_cache_prefix}_{self.width}x{self.height}_{self.mines}.npz"
                    if eval_file.exists() and eval_file not in self._loaded_files:
                        before = len(self._offline_buffer)
                        self._load_eval_file(eval_file)
                        self._source_buffers[source_idx].extend(self._offline_buffer[before:])
                        self._loaded_files.add(eval_file)
                    continue

                new_files = [
                    f for f in p.glob("*.npz")
                    if f not in self._loaded_files and "eval_boards" not in f.name
                ]
                if new_files:
                    print(f"TrajectoryPool: Found {len(new_files)} new data files in {p}. Loading...")
                    loaded_total = 0
                    for f in sorted(new_files):
                        before = len(self._offline_buffer)
                        loaded = self._load_train_file(f)
                        loaded_total += loaded
                        self._source_buffers[source_idx].extend(self._offline_buffer[before:])
                        self._loaded_files.add(f)
                    print(
                        "TrajectoryPool: "
                        f"Loaded {loaded_total} matching trajectories; "
                        f"total={len(self._offline_buffer)}"
                    )
            elif p.is_file() and p not in self._loaded_files:
                before = len(self._offline_buffer)
                if self.eval_mode:
                    self._load_eval_file(p)
                else:
                    self._load_train_file(p)
                self._source_buffers[source_idx].extend(self._offline_buffer[before:])
                self._loaded_files.add(p)
                loaded = len(self._offline_buffer) - before
                print(f"TrajectoryPool: Loaded {loaded} trajectories from {p}")

    def _get_source_weights(self) -> np.ndarray:
        weights = []
        for source_buffer, (_path, weight) in zip(self._source_buffers, self._data_sources):
            weights.append(weight if source_buffer else 0.0)
        probs = np.asarray(weights, dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return probs
        return probs / total

    def _get_traj(self) -> Dict[str, Any]:
        """Get one trajectory randomly from the loaded offline buffer. Wait if empty."""
        while not self._offline_buffer:
            print("TrajectoryPool: Buffer empty, waiting for background data generation...")
            time.sleep(2)
            self.refresh()

        probs = self._get_source_weights()
        if probs.sum() > 0:
            source_idx = int(np.random.choice(len(self._source_buffers), p=probs))
            source_buffer = self._source_buffers[source_idx]
            return source_buffer[np.random.randint(len(source_buffer))]

        idx = np.random.randint(len(self._offline_buffer))
        return self._offline_buffer[idx]

    def pop(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mines, visible) for online training initialization."""
        traj = self._get_traj()
        # Initial visible state is masks[0]
        return traj["mines"], traj["masks"][0]

    def batch(
        self,
        batch_size: int,
        target_type: str = "probs",
        include_solver_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (channels, targets, mask) batch for supervised training."""
        b_channels, b_targets, b_masks = [], [], []
        b_solver_safe_masks = []

        for _ in range(batch_size):
            traj = self._get_traj()
            # Pick a random step from the trajectory
            t = np.random.randint(len(traj["masks"]))

            # Construct channels (C, H, W)
            mask = traj["masks"][t]
            mines = traj["mines"]

            channels = board_state_to_channels(
                covered=mask.astype(bool),
                revealed=~mask.astype(bool),
                numbers=_adjacent_mine_counts(mines),
            )

            b_channels.append(channels)
            b_masks.append(mask)
            if include_solver_safe:
                if "solver_safe_masks" in traj:
                    b_solver_safe_masks.append(traj["solver_safe_masks"][t])
                else:
                    b_solver_safe_masks.append(np.zeros_like(mask, dtype=bool))

            if target_type == "binary":
                # Ground truth: 1=mine, 0=safe (binary labels)
                b_targets.append(mines.astype(np.float32))
            else:
                # Solver probability distillation (default)
                if "probs" in traj:
                    b_targets.append(traj["probs"][t])
                else:
                    b_targets.append(np.zeros_like(mask, dtype=np.float32))

        result = (
            torch.from_numpy(np.stack(b_channels)),
            torch.from_numpy(np.stack(b_targets)),
            torch.from_numpy(np.stack(b_masks)),
        )
        if include_solver_safe:
            return (*result, torch.from_numpy(np.stack(b_solver_safe_masks)))
        return result

    # ── Evaluation Support ──────────────────────────────────────────────────
    
    def get_eval_game(self, idx: int, rng: np.random.Generator) -> Optional[MinesweeperGame]:
        """Get a specific game instance for deterministic evaluation.
        
        If we have offline data loaded, and idx is within range, use it.
        Otherwise, generate a new deterministic game and append it to our buffer.
        """
        # 1. Return from memory buffer if available
        if idx < len(self._offline_buffer):
            traj = self._offline_buffer[idx]
            return MinesweeperGame.from_mine_mask(
                self.width, self.height, traj["mines"],
                first_done=True, visible=traj["masks"][0]
            )
            
        # 2. Generate new deterministic game
        from data.no_guess import generate_no_guess_board
        game = generate_no_guess_board(
            width=self.width, height=self.height, total_mines=self.mines,
            rng=rng, max_attempts=200
        )
        
        from game.constants import GameStatus
        if game is None or game.status != GameStatus.PLAYING:
            return None
            
        # 3. Cache it in buffer (minimal trajectory info needed for eval)
        self._offline_buffer.append({
            "mines": game.get_mine_mask(),
            "masks": [game.visible.copy()],
            "actions": [], # Unused for eval setup
        })
        
        return game
        
    @property
    def total_states(self) -> int:
        """Total number of states (steps) loaded in the offline buffer."""
        return sum(len(t["masks"]) for t in self._offline_buffer)
        
    @property
    def total_games(self) -> int:
        """Total number of games (trajectories) loaded in the offline buffer."""
        return len(self._offline_buffer)
        
    def save_eval_cache(self) -> None:
        """Save any dynamically generated evaluation boards to disk."""
        if not self.data_dir or not self._offline_buffer:
            return
            
        p = Path(self.data_dir)
        # Avoid overwriting training data npz, use a specific name for eval cache
        if p.suffix == ".npz":
            out_path = p
        else:
            p.mkdir(parents=True, exist_ok=True)
            out_path = p / f"{self.eval_cache_prefix}_{self.width}x{self.height}_{self.mines}.npz"
            
        save_dict = {}
        for i, traj in enumerate(self._offline_buffer):
            save_dict[f"mines_{i}"] = traj["mines"]
            save_dict[f"visible_{i}"] = traj["masks"][0]
            
        np.savez_compressed(out_path, **save_dict)
        
    @property
    def eval_size(self) -> int:
        return len(self._offline_buffer)
