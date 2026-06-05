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

from game.game import MinesweeperGame

class TrajectoryPool:
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

    def _load_train_file(self, f: Path):
        try:
            data = np.load(f, allow_pickle=True)
            n = len(data.files) // 4  # assuming mines, actions, masks, probs
            if n == 0:
                # If there are no actions, it might just be empty, safely ignore
                return
            for i in range(n):
                traj = {
                    "mines": data[f"mines_{i}"],
                    "actions": data[f"actions_{i}"],
                    "masks": data[f"masks_{i}"],
                }
                if f"probs_{i}" in data:
                    traj["probs"] = data[f"probs_{i}"]
                self._offline_buffer.append(traj)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    def __init__(
        self,
        board_width: int,
        board_height: int,
        board_mines: int,
        pool_size: int = 100,
        pool_workers: int = 0,
        data_dir: str = "",
        mixed_mode: bool = False,
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
        
        self._offline_buffer: List[Dict[str, Any]] = []
        self._loaded_files = set()
        
        # Load offline data if data_dir is provided
        self.refresh()
        
    def refresh(self):
        """Scan data_dir for new .npz files and load them into the buffer."""
        if not self.data_dir:
            return
            
        p = Path(self.data_dir)
        if not p.exists():
            return
            
        if p.is_dir():
            if self.eval_mode:
                eval_file = p / f"eval_boards_{self.width}x{self.height}_{self.mines}.npz"
                if eval_file.exists() and eval_file not in self._loaded_files:
                    self._load_eval_file(eval_file)
                    self._loaded_files.add(eval_file)
            else:
                pattern = f"{self.width}x{self.height}_{self.mines}_*.npz"
                new_files = [f for f in p.glob(pattern) if f not in self._loaded_files and "eval_boards" not in f.name]
                if new_files:
                    print(f"TrajectoryPool: Found {len(new_files)} new data files. Loading...")
                    for f in sorted(new_files):
                        self._load_train_file(f)
                        self._loaded_files.add(f)
                    print(f"TrajectoryPool: Total trajectories loaded: {len(self._offline_buffer)}")
        elif p.is_file() and p not in self._loaded_files:
            if self.eval_mode:
                self._load_eval_file(p)
            else:
                self._load_train_file(p)
            self._loaded_files.add(p)

    def _get_traj(self) -> Dict[str, Any]:
        """Get one trajectory randomly from the loaded offline buffer. Wait if empty."""
        while not self._offline_buffer:
            print("TrajectoryPool: Buffer empty, waiting for background data generation...")
            time.sleep(2)
            self.refresh()
            
        idx = np.random.randint(len(self._offline_buffer))
        return self._offline_buffer[idx]

    def pop(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mines, visible) for online training initialization."""
        traj = self._get_traj()
        # Initial visible state is masks[0]
        return traj["mines"], traj["masks"][0]

    def batch(self, batch_size: int, target_type: str = "probs") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (channels, targets, mask) batch for supervised training."""
        b_channels, b_targets, b_masks = [], [], []

        for _ in range(batch_size):
            traj = self._get_traj()
            # Pick a random step from the trajectory
            t = np.random.randint(len(traj["masks"]))

            # Construct channels (C, H, W)
            mask = traj["masks"][t]
            mines = traj["mines"]

            # Simplified channel construction (similar to old dataset)
            channels = np.zeros((10, self.height, self.width), dtype=np.float32)
            channels[0] = mask  # covered
            # channels[1] is flagged, assuming 0 for now in dataset

            # Count adjacent mines for revealed cells
            for r in range(self.height):
                for c in range(self.width):
                    if not mask[r, c]:
                        rmin, rmax = max(0, r-1), min(self.height, r+2)
                        cmin, cmax = max(0, c-1), min(self.width, c+2)
                        adj = np.sum(mines[rmin:rmax, cmin:cmax])
                        if adj > 0:
                            channels[1 + int(adj), r, c] = 1.0

            b_channels.append(channels)
            b_masks.append(mask)

            if target_type == "binary":
                # Ground truth: 1=mine, 0=safe (binary labels)
                b_targets.append(mines.astype(np.float32))
            else:
                # Solver probability distillation (default)
                if "probs" in traj:
                    b_targets.append(traj["probs"][t])
                else:
                    b_targets.append(np.zeros_like(mask, dtype=np.float32))

        return (
            torch.from_numpy(np.stack(b_channels)),
            torch.from_numpy(np.stack(b_targets)),
            torch.from_numpy(np.stack(b_masks)),
        )

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
        from data.self_validated import generate_self_validated_board
        game = generate_self_validated_board(
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
        if p.is_dir():
            out_path = p / f"eval_boards_{self.width}x{self.height}_{self.mines}.npz"
        else:
            out_path = p
            
        save_dict = {}
        for i, traj in enumerate(self._offline_buffer):
            save_dict[f"mines_{i}"] = traj["mines"]
            save_dict[f"visible_{i}"] = traj["masks"][0]
            
        np.savez_compressed(out_path, **save_dict)
        
    @property
    def eval_size(self) -> int:
        return len(self._offline_buffer)
