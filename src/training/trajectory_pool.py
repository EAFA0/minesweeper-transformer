"""TrajectoryPool for unified offline/online data provision.

Provides a unified interface (`batch()` and `pop()`) over a pool of 
Minesweeper trajectories. Uses multiprocessing for asynchronous background
filling to avoid blocking the main training loop.
"""

import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# We will import generator functions later when Module 2 is done.
# For now, we stub the background worker logic.
from data.generator import generate_trajectory

class TrajectoryPool:
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
    ):
        self.width = board_width
        self.height = board_height
        self.mines = board_mines
        self.pool_size = pool_size
        self.compute_probs = compute_probs
        
        # Multiprocessing setup
        self.queue = mp.Queue(maxsize=pool_size)
        self.workers: List[mp.Process] = []
        
        # Load offline data if data_dir is provided
        self._offline_buffer: List[Dict[str, Any]] = []
        if data_dir:
            p = Path(data_dir)
            if p.exists() and p.is_dir():
                print(f"TrajectoryPool: Loading offline data from {data_dir}...")
                for f in p.glob("*.npz"):
                    try:
                        data = np.load(f, allow_pickle=True)
                        n = len(data.files) // 4  # assuming mines, actions, masks, probs
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
                print(f"TrajectoryPool: Loaded {len(self._offline_buffer)} trajectories offline.")
                
                # Pre-fill queue with offline data up to pool size
                for t in self._offline_buffer[:pool_size]:
                    try:
                        self.queue.put_nowait(t)
                    except queue.Full:
                        break

        # Start background workers if requested
        if pool_workers > 0:
            for _ in range(pool_workers):
                p = mp.Process(target=self._background_worker, daemon=True)
                p.start()
                self.workers.append(p)

    def _background_worker(self):
        """Worker loop generating data and pushing to queue."""
        # Setup local RNG seed per worker
        rng = np.random.default_rng()
        while True:
            try:
                # Blocks if queue is full
                traj = generate_trajectory(
                    width=self.width, 
                    height=self.height, 
                    total_mines=self.mines, 
                    compute_probs=self.compute_probs,
                    rng=rng
                )
                if traj:
                    self.queue.put(traj)
            except Exception as e:
                time.sleep(0.1)

    def _get_traj(self) -> Dict[str, Any]:
        """Get one trajectory, preferring queue, fallback to offline, then sync gen."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            if self._offline_buffer:
                idx = np.random.randint(len(self._offline_buffer))
                return self._offline_buffer[idx]
            else:
                # Sync fallback
                return generate_trajectory(
                    self.width, self.height, self.mines, 
                    compute_probs=self.compute_probs
                )

    def pop(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mines, visible) for online training initialization."""
        traj = self._get_traj()
        # Initial visible state is masks[0]
        return traj["mines"], traj["masks"][0]

    def batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (channels, probs, mask) batch for supervised training."""
        b_channels, b_probs, b_masks = [], [], []
        
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
            
            if "probs" in traj:
                b_probs.append(traj["probs"][t])
            else:
                b_probs.append(np.zeros_like(mask, dtype=np.float32))

        return (
            torch.from_numpy(np.stack(b_channels)),
            torch.from_numpy(np.stack(b_probs)),
            torch.from_numpy(np.stack(b_masks)),
        )
