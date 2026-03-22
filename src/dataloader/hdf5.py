import h5py
import numpy as np
from loguru import logger

from src.types import AudioDatasetType


class Hdf5Dataset:
    """Read audio samples from a single HDF5 file.

    Each file stores samples under ``<type>/<key>`` where ``<type>`` is
    ``"speech"``, ``"noise"`` or ``"rir"````.

    ``sampling_factor`` scales the effective dataset length: a factor of 100
    means each unique clip appears 100 times per epoch (oversampling). Access
    beyond the real key count wraps cyclically.

    Args:
        file_path: Path to the ``.hdf5`` file.
        sampling_factor: Scales the effective number of samples.
        max_len_s: If set, randomly crop samples to at most this many seconds.
    """

    def __init__(
        self,
        file_path: str,
        sampling_factor: float = 1.0,
        max_len_s: float | None = None,
    ) -> None:
        self.file_path = file_path
        self.max_len_s = max_len_s

        with h5py.File(file_path, "r") as f:
            self.sr: int = int(f.attrs["sr"].item())  # type: ignore[union-attr]
            self.max_freq: int = int(f.attrs["max_freq"].item())  # type: ignore[union-attr]
            self.dataset_type: AudioDatasetType = list(f.keys())[0]

            group = f[self.dataset_type]
            assert isinstance(group, h5py.Group)
            self.keys: list[str] = list(group.keys())

        self.effective_size = max(1, round(len(self.keys) * sampling_factor))
        self._h5file: h5py.File | None = None

    @property
    def _file(self) -> h5py.File:
        if self._h5file is None:
            self._h5file = h5py.File(self.file_path, "r")
        return self._h5file

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(
        self,
        idx: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """Return one sample ``(channels, samples)``."""
        key = self.keys[idx % len(self.keys)]
        ds = self._file[f"{self.dataset_type}/{key}"]
        assert isinstance(ds, h5py.Dataset)
        audio: np.ndarray = ds[:]  # type: ignore[assignment]

        if self.max_len_s is not None:
            max_samples = int(self.max_len_s * self.sr)
            if audio.shape[1] > max_samples:
                start = int(rng.integers(0, audio.shape[1] - max_samples + 1))
                audio = audio[:, start : start + max_samples]

        return audio

    def get_at_least(
        self,
        idx: int,
        min_samples: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """Load a sample, tiling it until it has at least ``min_samples`` frames."""
        audio = self.__getitem__(idx, rng=rng)
        while audio.shape[1] < min_samples:
            audio = np.concatenate([audio, audio], axis=1)
        return audio

    def close(self) -> None:
        if self._h5file is not None:
            self._h5file.close()
        self._h5file = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Hdf5Dataset('{self.file_path}', type='{self.dataset_type}', "
            f"keys={len(self.keys)}, effective={self.effective_size}, sr={self.sr})"
        )


if __name__ == "__main__":
    speech_hdf5_path = "datasets/hdf5/speech_clean.hdf5"

    speech_dataset = Hdf5Dataset(speech_hdf5_path)
    logger.info(speech_dataset)
    logger.info(speech_dataset[0])
    assert isinstance(speech_dataset[0], np.ndarray)
