"""Time-domain dataset: loads speech/noise/RIR from HDF5 and mixes them."""

import bisect
import logging
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_audiomentations import Compose

from src.augmentations import (
    get_noise_augmentations,
    get_speech_augmentations,
    get_speech_distortions_td,
)
from src.configs.config import AugmentationConfig
from src.dataloader.dataset_config import DatasetConfig, DatasetEntry
from src.dataloader.hdf5 import Hdf5Dataset
from src.types import SplitType

logger = logging.getLogger(__name__)


_EPS = 1e-10


@dataclass
class TdSample:
    """Internal sample produced by :class:`TdDataset` before frequency-domain features.

    Attributes:
        speech:    Clean target signal, shape ``(channels, samples)``.
        noisy:     Degraded mixture (model input), shape ``(channels, samples)``.
        noise:     Scaled noise component in the mixture, shape ``(channels, samples)``.
        snr:       Target SNR in dB.
        gain:      Applied gain in dB.
        max_freq:  Signal bandwidth in Hz.
        sample_id: Dataset index.
    """

    speech: Tensor
    noisy: Tensor
    noise: Tensor
    snr: int
    gain: int
    max_freq: int
    sample_id: int


@dataclass
class Sample:
    """Full training sample produced by :class:`~src.dataloader.fft.FftDataset`.

    Extends :class:`TdSample` with frequency-domain features computed from the
    noisy signal. All fields are required — no Optional.

    Attributes:
        speech:    Clean target signal, shape ``(channels, samples)``.
        noisy:     Degraded mixture (model input), shape ``(channels, samples)``.
        noise:     Scaled noise component in the mixture, shape ``(channels, samples)``.
        snr:       Target SNR in dB.
        gain:      Applied gain in dB.
        max_freq:  Signal bandwidth in Hz.
        sample_id: Dataset index.
        feat_erb:  ERB filterbank features, shape ``(channels, frames, nb_erb)``.
        feat_spec: Complex spectrogram features, shape ``(channels, frames, nb_spec)``.
    """

    speech: Tensor
    noisy: Tensor
    noise: Tensor
    snr: int
    gain: int
    max_freq: int
    sample_id: int
    feat_erb: Tensor
    feat_spec: Tensor


def _apply_aug(pipeline: Compose, audio: Tensor, sr: int) -> Tensor:
    """Run a torch-audiomentations pipeline on a ``(C, T)`` Tensor.

    Returns a ``(C, T)`` Tensor.
    """
    out = pipeline(audio.unsqueeze(0), sample_rate=sr)  # (1, C, T)
    return out.squeeze(0)  # (C, T)


def _fft_convolve(signal: Tensor, kernel: Tensor) -> Tensor:
    """FFT convolution of a ``(C, T)`` signal with a 1-D kernel, trimmed to T."""
    out_len = signal.shape[1]
    n = 1 << (out_len + kernel.shape[0] - 2).bit_length()
    K = torch.fft.rfft(kernel, n=n)  # [n//2+1]
    S = torch.fft.rfft(signal, n=n)  # [C, n//2+1]
    return torch.fft.irfft(S * K, n=n)[:, :out_len]  # [C, out_len]


def _combine_noises(
    noises: list[Tensor],
    num_channels: int,
    target_length: int,
    rng: np.random.Generator,
) -> Tensor:
    """Combine a list of noise tensors into one (num_channels, target_length) tensor."""
    processed: list[Tensor] = []

    for noise in noises:
        c, t = noise.shape

        if t == 0:
            noise = torch.zeros(
                (c, target_length),
                dtype=noise.dtype,
                device=noise.device,
            )
        elif t < target_length:
            idx = torch.arange(target_length, device=noise.device) % t
            noise = noise[:, idx]
        elif t > target_length:
            start = int(rng.integers(0, t - target_length + 1))
            noise = noise[:, start : start + target_length]

        c = noise.shape[0]

        if c < num_channels:
            extra = num_channels - c
            extra_idx = rng.integers(0, c, size=extra)
            idx = np.concatenate([np.arange(c), extra_idx])
            noise = noise[torch.as_tensor(idx, device=noise.device)]
        elif c > num_channels:
            idx = np.sort(rng.choice(c, num_channels, replace=False))
            noise = noise[torch.as_tensor(idx, device=noise.device)]

        processed.append(noise)

    return torch.stack(processed, dim=0).mean(dim=0)


def _mix_audio_signal(
    clean: Tensor,
    noise: Tensor,
    snr_db: float,
    gain_db: float,
    clean_distorted: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mix clean speech and noise at a target SNR with anti-clipping protection.

    Returns ``(clean_out, noise_out, mixture)``.
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    clean_out: Tensor = clean * gain_linear
    clean_mix = (
        (clean_distorted * gain_linear)
        if clean_distorted is not None
        else clean_out.clone()
    )

    e_clean = clean_out.pow(2).sum().clamp(min=_EPS)
    e_noise = noise.pow(2).sum().clamp(min=_EPS)
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale: Tensor = (e_clean / (e_noise * snr_linear)).sqrt()
    noise_out = noise * scale

    mixture = clean_mix + noise_out

    max_all = max(
        clean_out.abs().max().item(),
        noise_out.abs().max().item(),
        mixture.abs().max().item(),
    )
    if max_all - 1.0 > _EPS:
        sf = 1.0 / (max_all + _EPS)
        clean_out = clean_out * sf
        noise_out = noise_out * sf
        mixture = mixture * sf

    return clean_out, noise_out, mixture


def _adjust_channels_jointly(
    num_channels: int,
    rng: np.random.Generator,
    *tensors: Tensor,
) -> tuple[Tensor, ...]:
    """Adjust a group of ``(C, T)`` tensors to exactly ``num_channels`` channels.

    All tensors receive the **same** channel indices so that linear relationships
    (e.g. ``noisy == speech + noise``) are preserved after the adjustment.

    Note:
        When ``c == num_channels`` the original tensors are returned as-is
        (no copy).  Callers must not mutate them in place.
    """
    c = tensors[0].shape[0]

    if not all(t.shape[0] == c for t in tensors[1:]):
        raise ValueError(
            f"all tensors must have the same channel count, "
            f"got {[t.shape[0] for t in tensors]}"
        )

    if c == num_channels:
        return tensors

    if c < num_channels:
        ch_idx = torch.as_tensor(
            rng.integers(0, c, size=num_channels - c), dtype=torch.long
        )
        return tuple(torch.cat([t, t[ch_idx]], dim=0) for t in tensors)

    ch_idx = torch.as_tensor(
        rng.choice(c, num_channels, replace=False), dtype=torch.long
    )
    return tuple(t[ch_idx] for t in tensors)


def _lookup(
    datasets: list[Hdf5Dataset], cum: list[int], idx: int
) -> tuple[Hdf5Dataset, int]:
    """Translate a flat index into ``(dataset, key_idx)`` using cumulative sizes."""
    ds_idx = bisect.bisect_right(cum, idx)
    key_idx = idx - (cum[ds_idx - 1] if ds_idx > 0 else 0)
    return datasets[ds_idx], key_idx


class TdDataset(Dataset):
    """PyTorch dataset that mixes speech, noise, and RIR into training pairs.

    Each sample is generated by:

    1. Loading and augmenting a speech clip.
    2. Loading and combining 2–5 noise segments.
    3. Optionally applying room reverberation via a real RIR.
    4. Optionally distorting the speech input (training only).
    5. Mixing at a randomly sampled SNR and gain.

    Args:
        speech_files: HDF5 speech dataset entries.
        noise_files:  HDF5 noise dataset entries.
        rir_files:    HDF5 RIR dataset entries (may be empty).
        aug_config:   Augmentation probabilities.
        split:        ``"train"``, ``"valid"``, or ``"test"``.
        sr:           Sample rate in Hz.
        max_len_s:    Crop clips to at most this many seconds (``None`` = no crop).
        num_channels: Force output to this many channels (duplicate/drop as needed).
        snrs:         Discrete SNR grid in dB.
        gains:        Discrete gain grid in dB.
        n_noises:     ``(min, max)`` number of noise tracks to combine.
        seed:         Base RNG seed — sample ``i`` uses ``seed + i``.
    """

    _DEFAULT_SNRS = [-5, 0, 5, 10, 20, 40]
    _DEFAULT_GAINS = [-6, 0, 6]

    def __init__(
        self,
        speech_files: list[DatasetEntry],
        noise_files: list[DatasetEntry],
        rir_files: list[DatasetEntry],
        aug_config: AugmentationConfig,
        split: SplitType = "train",
        sr: int = 48_000,
        max_len_s: float | None = None,
        num_channels: int = 1,
        snrs: list[int] | None = None,
        gains: list[int] | None = None,
        n_noises: tuple[int, int] = (2, 5),
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.split = split
        self.sr = sr
        self.max_len_s = max_len_s
        self.num_channels = num_channels
        self.snrs = snrs or self._DEFAULT_SNRS
        self.gains = gains or self._DEFAULT_GAINS
        self.n_noises = n_noises
        self.seed = seed
        self.is_train = split == "train"
        self.p_reverb = aug_config.p_reverb
        self.p_reverb_drr = aug_config.p_reverb_drr

        self._speech = self._open(speech_files, max_len_s=max_len_s)
        self._noise = self._open(noise_files)
        self._rir = self._open(rir_files)

        self._speech_cum = self._cumulative(self._speech)
        self._noise_cum = self._cumulative(self._noise)
        self._rir_cum = self._cumulative(self._rir)

        self._speech_aug = get_speech_augmentations(aug_config, sample_rate=sr)
        self._noise_aug = get_noise_augmentations(aug_config, sample_rate=sr)
        self._distortions = (
            get_speech_distortions_td(aug_config, sample_rate=sr)
            if self.is_train
            else None
        )
        for aug in filter(None, [self._speech_aug, self._noise_aug, self._distortions]):
            aug.train()

        logger.info(
            "%s | split=%s | speech=%d | noise=%d | rir=%d",
            self.__class__.__name__,
            split,
            self._speech_cum[-1] if self._speech_cum else 0,
            self._noise_cum[-1] if self._noise_cum else 0,
            self._rir_cum[-1] if self._rir_cum else 0,
        )

    def __len__(self) -> int:
        return self._speech_cum[-1] if self._speech_cum else 0

    def __getitem__(self, idx: int) -> TdSample:
        rng = np.random.default_rng(self.seed + idx)
        snr = int(rng.choice(self.snrs))
        gain = int(rng.choice(self.gains))

        speech = self._load_speech(idx, rng)
        noise = self._load_and_combine_noise(speech.shape, rng)

        if self._rir_cum and rng.random() < self.p_reverb:
            speech, noise, speech_rev = self._apply_rir(speech, noise, rng)
        else:
            speech_rev = None

        # Use fully reverberant speech as the noisy mixture base (matches Rust dataset.rs).
        # When no RIR was applied, speech_rev is None and we fall back to clean speech.
        speech_distorted_base = speech_rev if speech_rev is not None else speech
        speech_input = (
            _apply_aug(self._distortions, speech_distorted_base, self.sr)
            if self._distortions
            else speech_distorted_base
        )

        speech_out, noise_out, noisy = _mix_audio_signal(
            speech, noise, snr, gain, speech_input
        )

        speech_out, noise_out, noisy = _adjust_channels_jointly(
            self.num_channels, rng, speech_out, noise_out, noisy
        )

        ds, _ = _lookup(self._speech, self._speech_cum, idx % self._speech_cum[-1])
        return TdSample(
            speech=speech_out,
            noisy=noisy,
            noise=noise_out,
            snr=snr,
            gain=gain,
            max_freq=ds.max_freq,
            sample_id=idx,
        )

    def _load_speech(self, idx: int, rng: np.random.Generator) -> Tensor:
        ds, key_idx = _lookup(
            self._speech, self._speech_cum, idx % self._speech_cum[-1]
        )
        min_samples = int(self.max_len_s * self.sr * 1.1) if self.max_len_s else 0
        audio = torch.from_numpy(ds.get_at_least(key_idx, min_samples, rng))

        for _ in range(10):
            if self.max_len_s:
                max_s = int(self.max_len_s * self.sr)
                if audio.shape[1] > max_s:
                    start = int(rng.integers(0, audio.shape[1] - max_s + 1))
                    audio = audio[:, start : start + max_s]

            audio = _apply_aug(self._speech_aug, audio, self.sr)

            if audio.abs().max() > 1e-8:
                return audio

            logger.warning("Speech augmentation failed, retrying...")
            key_idx = int(rng.integers(0, ds.effective_size))
            audio = torch.from_numpy(ds.get_at_least(key_idx, min_samples, rng))

        return _apply_aug(self._speech_aug, audio, self.sr)

    def _load_noise(self, rng: np.random.Generator) -> Tensor:
        if not self._noise_cum:
            return torch.zeros(1, self.sr)
        flat_idx = int(rng.integers(0, self._noise_cum[-1]))
        ds, key_idx = _lookup(self._noise, self._noise_cum, flat_idx)
        audio = torch.from_numpy(ds.get(key_idx, rng))
        return _apply_aug(self._noise_aug, audio, self.sr)

    def _load_and_combine_noise(
        self, speech_shape: tuple[int, ...], rng: np.random.Generator
    ) -> Tensor:
        n = int(rng.integers(self.n_noises[0], self.n_noises[1] + 1))
        noises = [self._load_noise(rng) for _ in range(n)]
        return _combine_noises(noises, speech_shape[0], speech_shape[1], rng)

    def _apply_rir(
        self,
        speech: Tensor,
        noise: Tensor,
        rng: np.random.Generator,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Apply room impulse response reverberation.

        Returns ``(speech_target, noise_rev, speech_rev)`` where ``speech_rev``
        is the fully reverberant speech used as the noisy mixture input, and
        ``speech_target`` is the partially dereverberant clean training target.
        Returns ``None`` for ``speech_rev`` when no RIR data is available.
        """
        if not self._rir_cum:
            return speech, noise, None

        flat_idx = int(rng.integers(0, self._rir_cum[-1]))
        ds, key_idx = _lookup(self._rir, self._rir_cum, flat_idx)
        rir_np = ds.get(key_idx, rng)[0]  # first channel, (T,) numpy
        rir_np = (rir_np / (np.sqrt(np.sum(rir_np**2)) + 1e-10)).astype(np.float32)
        rir = torch.from_numpy(rir_np)

        peak = int(rir.abs().argmax().item())
        cutoff = peak + int(0.020 * self.sr)
        rir_target = rir.clone()
        if cutoff < len(rir_target):
            tail = torch.arange(len(rir_target) - cutoff, dtype=torch.float32)
            # Match Rust supress_late: decay[i] = 10^(-(i/sr) / tau),
            # where tau = -rt60 / log10(10^(-60/20)) = rt60 / 3  (rt60=0.5 → tau≈0.167s)
            _rt60_tau = 0.5 / 3.0
            rir_target[cutoff:] *= torch.pow(
                torch.tensor(10.0, dtype=tail.dtype), -(tail / self.sr) / _rt60_tau
            )
        # Re-normalise after decay so rir_target has unit energy
        rir_target_e = float(rir_target.pow(2).sum().sqrt().item())
        rir_target = rir_target / (rir_target_e + 1e-10)

        speech_rms = float(speech.pow(2).mean().sqrt().item())

        # speech_rev: fully reverberant — used as the noisy mixture input
        speech_rev = _fft_convolve(speech, rir)
        noise_rev = _fft_convolve(noise, rir)
        speech_little_rev = _fft_convolve(speech, rir_target)
        speech_target = self.p_reverb_drr * speech + (1.0 - self.p_reverb_drr) * speech_little_rev

        # Restore speech to original RMS level
        speech_rms_after = float(speech_target.pow(2).mean().sqrt().item())
        speech_target = speech_target * (speech_rms / (speech_rms_after + 1e-10))

        return speech_target, noise_rev, speech_rev

    @staticmethod
    def _open(
        entries: list[DatasetEntry], max_len_s: float | None = None
    ) -> list[Hdf5Dataset]:
        return [
            Hdf5Dataset(
                str(e.path), sampling_factor=e.sampling_factor, max_len_s=max_len_s
            )
            for e in entries
        ]

    @staticmethod
    def _cumulative(datasets: list[Hdf5Dataset]) -> list[int]:
        cum, total = [], 0
        for ds in datasets:
            total += ds.effective_size
            cum.append(total)
        return cum

    def close(self) -> None:
        for ds in self._speech + self._noise + self._rir:
            ds.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        n_speech = self._speech_cum[-1] if self._speech_cum else 0
        n_noise = self._noise_cum[-1] if self._noise_cum else 0
        n_rir = self._rir_cum[-1] if self._rir_cum else 0
        return (
            f"TdDataset(split='{self.split}', speech={n_speech}, "
            f"noise={n_noise}, rir={n_rir})"
        )


class Datasets:
    """Container for train / valid / test :class:`TdDataset` splits."""

    def __init__(self, train: TdDataset, valid: TdDataset, test: TdDataset) -> None:
        self.train = train
        self.valid = valid
        self.test = test

    def get(self, split: SplitType) -> TdDataset:
        return getattr(self, split)

    @classmethod
    def from_config(
        cls,
        dataset_config: DatasetConfig,
        aug_config: AugmentationConfig,
        sr: int = 48_000,
        max_len_s: float | None = None,
        snrs: list[int] | None = None,
        gains: list[int] | None = None,
        n_noises: tuple[int, int] = (2, 5),
        seed: int = 42,
    ) -> "Datasets":
        def build(split: SplitType) -> TdDataset:
            speech, noise, rir = _partition(dataset_config.get_split(split))
            return TdDataset(
                speech_files=speech,
                noise_files=noise,
                rir_files=rir,
                aug_config=aug_config,
                split=split,
                sr=sr,
                max_len_s=max_len_s,
                snrs=snrs,
                gains=gains,
                n_noises=n_noises,
                seed=seed,
            )

        return cls(train=build("train"), valid=build("valid"), test=build("test"))

    def close(self) -> None:
        for ds in (self.train, self.valid, self.test):
            ds.close()

    def __repr__(self) -> str:
        return f"Datasets(train={self.train}, valid={self.valid}, test={self.test})"


def _partition(
    entries: list[DatasetEntry],
) -> tuple[list[DatasetEntry], list[DatasetEntry], list[DatasetEntry]]:
    speech, noise, rir = [], [], []
    for entry in entries:
        try:
            with h5py.File(entry.path, "r") as f:
                ds_type = str(list(f.keys())[0])
        except OSError as exc:
            logger.warning("Skipping %s: %s", entry.path, exc)
            continue

        match ds_type:
            case "speech":
                speech.append(entry)
            case "noise":
                noise.append(entry)
            case "rir":
                rir.append(entry)
            case _:
                logger.warning(
                    "Unknown dataset type '%s' in %s — skipping", ds_type, entry.path
                )

    return speech, noise, rir
