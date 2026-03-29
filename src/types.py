from typing import Literal
from dataclasses import dataclass
from torch import Tensor
import numpy as np

AudioDatasetType = Literal["speech", "noise", "rir"]
SplitType = Literal["train", "valid", "test"]


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
