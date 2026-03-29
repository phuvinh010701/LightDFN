"""Dataset configuration — parses ``dataset.cfg`` JSON files.

Expected format::

    {
        "train": [["speech_clean.hdf5", 100], ["noise_music.hdf5", 10], ["rir.hdf5", 1]],
        "valid": [["speech_clean.hdf5", 100], ...],
        "test":  [["speech_clean.hdf5", 100], ...]
    }

Each entry is ``[filename, sampling_factor]``.  A sampling factor > 1
oversamples that file; < 1 undersamples it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.types import SplitType


@dataclass
class DatasetEntry:
    """One HDF5 file with its sampling weight."""

    path: Path
    sampling_factor: float = 1.0


@dataclass
class DatasetConfig:
    """Train / valid / test split configuration parsed from ``dataset.cfg``."""

    train: list[DatasetEntry]
    valid: list[DatasetEntry]
    test: list[DatasetEntry]

    def get_split(self, split: SplitType) -> list[DatasetEntry]:
        return getattr(self, split)

    @classmethod
    def from_path(
        cls,
        config_path: str | Path,
        ds_dir: str | Path | None = None,
    ) -> "DatasetConfig":
        """Load and parse a ``dataset.cfg`` file.

        Args:
            config_path: Path to the JSON config file.
            ds_dir: Base directory for resolving relative HDF5 paths.
                Defaults to the directory containing ``config_path``.
        """
        config_path = Path(config_path)
        base = Path(ds_dir) if ds_dir is not None else config_path.parent

        raw: dict[str, list[list]] = json.loads(config_path.read_text())

        def parse(key: str) -> list[DatasetEntry]:
            return [
                DatasetEntry(path=base / entry[0], sampling_factor=float(entry[1]))
                for entry in raw.get(key, [])
            ]

        return cls(train=parse("train"), valid=parse("valid"), test=parse("test"))
