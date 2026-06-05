import numpy as np
from pathlib import Path
from typing import List

class BaseDataWriter:
    """Base class for handling chunked data saving and atomic writes."""
    
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        samples_per_file: int = 2000,
        start_file_idx: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.samples_per_file = samples_per_file
        self.file_idx = start_file_idx
        self.buffer = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def append(self, sample: dict):
        """Append a single sample to the buffer and flush if full."""
        self.buffer.append(sample)
        if len(self.buffer) >= self.samples_per_file:
            self.flush()
            
    def extend(self, samples: List[dict]):
        """Append multiple samples."""
        for sample in samples:
            self.append(sample)
            
    def flush(self):
        """Write the current buffer to disk and increment the file index."""
        if not self.buffer:
            return
            
        out_path = self.output_dir / f"{self.prefix}_{self.file_idx:04d}.npz"
        tmp_path = out_path.with_suffix('.npz.tmp')
        
        data = self._format_buffer(self.buffer)
        
        # np.savez_compressed appends .npz automatically if the string doesn't end with .npz
        # Using a file handle bypasses this behavior
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **data)
            
        tmp_path.rename(out_path)
        
        self.buffer.clear()
        self.file_idx += 1
        
    def _format_buffer(self, _buffer: List[dict]) -> dict:
        """Format the buffer into a dict of arrays for np.savez_compressed."""
        raise NotImplementedError


class TrajectoryWriter(BaseDataWriter):
    """Writes full trajectories (mines, actions, masks, probs). Used by online BCE/MSE training."""
    
    def _format_buffer(self, buffer: List[dict]) -> dict:
        data = {}
        for i, traj in enumerate(buffer):
            data[f"mines_{i}"] = traj["mines"]
            data[f"actions_{i}"] = np.array(traj["actions"], dtype=np.int32)
            data[f"masks_{i}"] = np.array(traj["masks"], dtype=bool)
            if "probs" in traj:
                data[f"probs_{i}"] = np.array(traj["probs"], dtype=np.float32)
        return data
