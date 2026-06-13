"""Canonical data layout for generated datasets."""

from dataclasses import dataclass
from pathlib import Path


DATA_SCHEMA_VERSION = 1
DATA_ROOT = Path("data")


@dataclass(frozen=True)
class StageDataset:
    """Canonical stage dataset definition."""

    name: str
    width: int
    height: int
    mines: int
    n_samples: int = 10000
    samples_per_file: int = 2000

    @property
    def path(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def file_prefix(self) -> str:
        return f"train_{self.name}_{self.width}x{self.height}_{self.mines}"


STAGE_DATASETS: dict[str, StageDataset] = {
    "S1": StageDataset("S1", 8, 8, 10),
    "S2": StageDataset("S2", 8, 8, 15),
    "S3": StageDataset("S3", 8, 8, 20),
    "S4": StageDataset("S4", 8, 8, 25),
    "S5": StageDataset("S5", 8, 8, 32),
}


def get_stage_dataset(stage: str) -> StageDataset:
    """Return a canonical stage dataset or raise a clear error."""
    try:
        return STAGE_DATASETS[stage]
    except KeyError as exc:
        valid = ", ".join(STAGE_DATASETS)
        raise ValueError(f"Unknown stage dataset {stage!r}; expected one of: {valid}") from exc
