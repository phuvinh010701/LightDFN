from typing import Optional

import h5py
import numpy as np

from src.constants import AUDIO_DATASET_TYPES


class Hdf5Dataset:
    def __init__(
        self,
        file_path: str,
        sampling_factor: float = 1.0,
        max_len_s: Optional[float] = None,
    ):
        super().__init__()

        self.file_path = file_path
        self.sampling_factor = sampling_factor
        self.max_len_s = max_len_s
        with h5py.File(file_path, "r") as f:
            self.sr = int(f.attrs["sr"])
            self.max_freq = int(f.attrs["max_freq"])
            self.dtype = f.attrs["dtype"]
            self.codec = f.attrs["codec"]
            self.dataset_type: AUDIO_DATASET_TYPES = list(f.keys())[0]
            self.keys = list(f[self.dataset_type].keys())
            self.effective_size = int(len(self.keys) * sampling_factor)

        self.h5file = h5py.File(file_path, "r")

    def __len__(self) -> int:
        """Return effective dataset size (with sampling factor)."""
        return self.effective_size

    def __getitem__(self, idx: int) -> np.ndarray:
        """Load audio sample from HDF5.

        Args:
            idx: Index (may be >= len(keys) due to oversampling)

        Returns:
            Audio array of shape (channels, samples)
        """
        key_idx = idx % len(self.keys)
        key = self.keys[key_idx]

        dataset = self.h5file[f"{self.dataset_type}/{key}"]
        audio = dataset[:]

        assert audio.ndim == 2, f"Expected 2D array, got {audio.ndim}"
        assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"

        # Trim to max length if specified
        if self.max_len_s is not None:
            max_samples = int(self.max_len_s * self.sr)
            if audio.shape[1] > max_samples:
                start = np.random.randint(0, audio.shape[1] - max_samples + 1)
                audio = audio[:, start : start + max_samples]

        return audio

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, "h5file"):
            self.h5file.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    ds = Hdf5Dataset("datasets/hdf5/speech_clean_minisize.hdf5")

    logging.info(ds[0])
    logging.info(ds[0].shape)
    logging.info(ds[0].dtype)
