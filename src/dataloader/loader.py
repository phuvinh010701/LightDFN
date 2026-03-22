"""Batch collation, DataLoader wrapper, and builder for TdDataset / FftDataset."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.configs.config import AugmentationConfig, DataLoaderConfig
from src.dataloader.dataset_config import DatasetEntry
from src.dataloader.fft import FftDataset
from src.dataloader.td import Sample, TdDataset
from src.types import SplitType


@dataclass
class DsBatch:
    """A collated batch of :class:`~src.dataloader.td.Sample` objects.

    Attributes:
        speech:    ``(batch, channels, samples)``
        noisy:     ``(batch, channels, samples)``
        noise:     ``(batch, channels, samples)``
        lengths:   Original sample lengths before padding, shape ``(batch,)``
        snr:       SNR values, shape ``(batch,)``
        gain:      Gain values, shape ``(batch,)``
        max_freq:  Bandwidth values, shape ``(batch,)``
        sample_id: Dataset indices, shape ``(batch,)``
        feat_erb:  ``(batch, channels, frames, nb_erb)``
        feat_spec: ``(batch, channels, frames, nb_spec)``
    """

    speech: Tensor
    noisy: Tensor
    noise: Tensor
    feat_erb: Tensor
    feat_spec: Tensor
    lengths: np.ndarray
    snr: np.ndarray
    gain: np.ndarray
    max_freq: np.ndarray
    sample_id: np.ndarray


def collate_fn(samples: list[Sample]) -> DsBatch:
    """Collate a list of :class:`~src.dataloader.td.Sample` into a :class:`DsBatch`.

    Time-domain signals are zero-padded to the longest sample in the batch.
    Feature tensors are stacked without padding (same frame count when
    ``max_len_s`` is fixed).
    """
    lengths = np.array([s.speech.shape[-1] for s in samples], dtype=np.int64)
    max_len = int(lengths.max())

    def pad_and_stack(tensors: list[Tensor]) -> Tensor:
        """Zero-pad each tensor to ``max_len`` along the last dimension and stack."""
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[-1]
            if pad_len > 0:
                t = torch.nn.functional.pad(t, (0, pad_len))
            padded.append(t)
        return torch.stack(padded)

    return DsBatch(
        speech=pad_and_stack([s.speech for s in samples]),
        noisy=pad_and_stack([s.noisy for s in samples]),
        noise=pad_and_stack([s.noise for s in samples]),
        feat_erb=torch.stack([s.feat_erb for s in samples]),
        feat_spec=torch.stack([s.feat_spec for s in samples]),
        lengths=lengths,
        snr=np.array([s.snr for s in samples], dtype=np.int32),
        gain=np.array([s.gain for s in samples], dtype=np.int32),
        max_freq=np.array([s.max_freq for s in samples], dtype=np.int32),
        sample_id=np.array([s.sample_id for s in samples], dtype=np.int64),
    )


class DeepFilterNetDataLoader:
    """Thin wrapper around :class:`~torch.utils.data.DataLoader` with epoch seeding.

    Call :meth:`start_epoch` before iterating to reproducibly shuffle each epoch.

    Args:
        dataset:     A :class:`TdDataset` or :class:`FftDataset`.
        batch_size:  Number of samples per batch.
        num_workers: DataLoader worker processes.
        seed:        Base seed; epoch ``e`` uses ``seed + e``.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._loader: DataLoader | None = None

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)  # type: ignore[arg-type]

    def start_epoch(self, epoch_idx: int) -> None:
        """Re-seed the DataLoader for the given epoch index."""
        g = torch.Generator()
        g.manual_seed(self.seed + epoch_idx)
        logger.debug(f"Seeding DataLoader with seed {self.seed + epoch_idx}")
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=g,
        )

    def __iter__(self) -> Iterator[DsBatch]:
        """Return an iterator over batches, initialising epoch 0 if needed."""
        if self._loader is None:
            self.start_epoch(0)
        return iter(self._loader)  # type: ignore[arg-type]


class DataLoaderBuilder:
    """Builds a :class:`DeepFilterNetDataLoader` from a :class:`DataLoaderConfig`.

    Args:
        config: Dataloader configuration.
    """

    def __init__(self, loader_config: DataLoaderConfig) -> None:
        self.loader_config = loader_config

    def build(
        self, split: SplitType, aug_config: AugmentationConfig, use_fft: bool = True
    ) -> DeepFilterNetDataLoader:
        """Build a dataloader for the requested split.

        Args:
            split (SplitType): Dataset split to load.
            aug_config (AugmentationConfig): Augmentation configuration.
            use_fft (bool): If ``True``, wraps the dataset with :class:`FftDataset`.

        Returns:
            DeepFilterNetDataLoader: Configured dataloader for the split.
        """

        def _entries(paths: list[str]) -> list[DatasetEntry]:
            """Wrap plain path strings into DatasetEntry objects with unit sampling."""
            return [DatasetEntry(path=p, sampling_factor=1.0) for p in paths]

        td_dataset = TdDataset(
            speech_files=_entries(self.loader_config.speech_hdf5),
            noise_files=_entries(self.loader_config.noise_hdf5),
            rir_files=_entries(self.loader_config.rir_hdf5),
            aug_config=aug_config,
            split=split,
            sr=self.loader_config.sr,
            max_len_s=self.loader_config.max_len_s,
            seed=self.loader_config.seed,
        )

        dataset: Dataset = td_dataset
        if use_fft:
            dataset = FftDataset(
                td_dataset=td_dataset,
                fft_size=self.loader_config.fft_size,
                hop_size=self.loader_config.hop_size,
                nb_erb=self.loader_config.nb_erb,
                nb_spec=self.loader_config.nb_spec,
                sr=self.loader_config.sr,
                min_nb_freqs=self.loader_config.min_nb_freqs,
                norm_tau=self.loader_config.norm_tau,
            )

        return DeepFilterNetDataLoader(
            dataset=dataset,
            batch_size=self.loader_config.batch_size,
            num_workers=self.loader_config.num_workers,
            seed=self.loader_config.seed,
        )


if __name__ == "__main__":
    from src.configs.config import load_config

    _, augmentation_config, data_loader_config, _ = load_config(
        "./src/configs/test.yaml"
    )
    builder = DataLoaderBuilder(data_loader_config)

    logger.info("FftDataset (training) smoke test")
    loader = builder.build(
        split="train", aug_config=augmentation_config
    )  # use_fft=True by default
    logger.info("Built dataloader OK")

    max_samples = int(data_loader_config.max_len_s * data_loader_config.sr)
    n_frames = (
        max_samples - data_loader_config.fft_size
    ) // data_loader_config.hop_size + 1
    for i, batch in enumerate(loader):
        assert batch.speech.shape[-1] == max_samples
        assert batch.feat_erb.shape[-2:] == (n_frames, data_loader_config.nb_erb)
        assert batch.feat_spec.shape[-2:] == (n_frames, data_loader_config.nb_spec)
        assert not torch.isnan(batch.feat_erb).any()
        assert not torch.isnan(batch.feat_spec.real).any()
        logger.info(
            f"  batch {i:03d} | speech {tuple(batch.speech.shape)} | snr {batch.snr.tolist()} | gain {batch.gain.tolist()}"
        )
    logger.info("all batches OK")

    logger.info("All smoke tests passed.")
