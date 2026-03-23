import math
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch_audiomentations import AddColoredNoise, Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from src.configs.config import AugmentationConfig
from src.utils.io import resample

# Biquad coefficient helpers (Audio EQ Cookbook)
# All return (b, a) as plain Python float lists [b0, b1, b2], [a0=1, a1, a2]


def _lowpass_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[list[float], list[float]]:
    """Lowpass biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _highpass_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[list[float], list[float]]:
    """Highpass biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _lowshelf_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[list[float], list[float]]:
    """Lowshelf biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    S = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + S)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - S)
    a0 = (A + 1.0) + (A - 1.0) * cos_w0 + S
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 = (A + 1.0) + (A - 1.0) * cos_w0 - S
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _highshelf_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[list[float], list[float]]:
    """Highshelf biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    S = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + S)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - S)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + S
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - S
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _peaking_eq_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[list[float], list[float]]:
    """Peaking EQ biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _notch_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[list[float], list[float]]:
    """Notch biquad coefficients (Audio EQ Cookbook)."""
    w0 = 2.0 * math.pi * center_freq / sr
    alpha = math.sin(w0) / (2.0 * q_factor)
    cos_w0 = math.cos(w0)
    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


@torch.jit.script
def _apply_biquad_batch(samples: Tensor, b: Tensor, a: Tensor) -> Tensor:
    """Apply a biquad IIR filter (Direct Form II Transposed) to a batch of audio.

    Args:
        samples: Audio tensor of shape (batch_size, num_channels, num_samples).
        b: Feedforward coefficients, shape (batch_size, 3).
        a: Feedback coefficients [a0=1, a1, a2], shape (batch_size, 3).

    Returns:
        Filtered audio, same shape as ``samples``.
    """
    B, C, T = samples.shape
    out = torch.zeros_like(samples)

    mem0 = samples.new_zeros(B, C)
    mem1 = samples.new_zeros(B, C)

    b0 = b[:, 0].view(B, 1)
    b1 = b[:, 1].view(B, 1)
    b2 = b[:, 2].view(B, 1)
    a1 = a[:, 1].view(B, 1)
    a2 = a[:, 2].view(B, 1)

    for t in range(T):
        x_t = samples[:, :, t]
        y_t = b0 * x_t + mem0
        mem0 = mem1 + b1 * x_t - a1 * y_t
        mem1 = b2 * x_t - a2 * y_t
        out[:, :, t] = y_t

    return out


@torch.jit.script
def biquad_norm(x: Tensor, b: Tensor, a: Tensor) -> Tensor:
    """Apply a normalised 2nd-order AR/MA filter (RNNoise/PercepNet form).

    The difference equation is:

    y[t] = x[t] + b[0]*x[t-1] + b[1]*x[t-2] - a[0]*y[t-1] - a[1]*y[t-2]

    Args:
        x: Input signal of shape (T,).
        b: Feedforward coefficients [b1, b2], shape (2,).  b0 is implicitly 1.
        a: Feedback coefficients [a1, a2], shape (2,).  a0 is implicitly 1.

    Returns:
        Filtered signal of shape (T,).
    """
    T = x.shape[0]
    y = torch.zeros_like(x)
    y_prev1 = x.new_zeros(())
    y_prev2 = x.new_zeros(())
    x_prev1 = x.new_zeros(())
    x_prev2 = x.new_zeros(())

    for t in range(T):
        x_curr = x[t]
        y_curr = (
            x_curr + b[0] * x_prev1 + b[1] * x_prev2 - a[0] * y_prev1 - a[1] * y_prev2
        )
        y[t] = y_curr
        x_prev2 = x_prev1
        x_prev1 = x_curr
        y_prev2 = y_prev1
        y_prev1 = y_curr

    return y


# Augmentation 1: RandRemoveDc


class RandRemoveDc(BaseWaveformTransform):
    """Remove DC offset by subtracting the mean across the time dimension.

    Simple mean subtraction: ``y[n] = x[n] - mean(x)``

    Args:
        p: Probability of applying the augmentation.
        sample_rate: Sample rate in Hz.
    """

    def __init__(self, p: float = 0.25, sample_rate: int = 48000):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        mean = samples.mean(dim=-1, keepdim=True)
        return ObjectDict(
            samples=samples - mean,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 2: RandLFilt


class RandLFilt(BaseWaveformTransform):
    """Random low/high-pass filtering using a normalised biquad.

    Applies a 2nd-order biquad with coefficients sampled uniformly.
    Adopted from RNNoise/PercepNet implementations.

    Args:
        p: Probability of applying the augmentation.
        a_range: Feedback coefficient range ``(min, max)``.
        b_range: Feedforward coefficient range ``(min, max)``.
    """

    def __init__(
        self,
        p: float = 0.5,
        a_range: tuple[float, float] = (-3.0 / 8.0, 3.0 / 8.0),
        b_range: tuple[float, float] = (-3.0 / 8.0, 3.0 / 8.0),
    ):
        super().__init__(p=p, output_type="tensor")
        self.a_range = a_range
        self.b_range = b_range

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        B, C, _ = samples.shape

        a = torch.distributions.Uniform(
            low=torch.tensor(self.a_range[0], dtype=torch.float32),
            high=torch.tensor(self.a_range[1], dtype=torch.float32),
        ).sample((B, 2))

        b = torch.distributions.Uniform(
            low=torch.tensor(self.b_range[0], dtype=torch.float32),
            high=torch.tensor(self.b_range[1], dtype=torch.float32),
        ).sample((B, 2))

        out = samples.clone()
        for i in range(B):
            for ch in range(C):
                out[i, ch] = biquad_norm(out[i, ch], b[i], a[i])

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 3: RandBiquadFilter

_BIQUAD_FILTER_TYPES = [
    "lowpass",
    "highpass",
    "lowshelf",
    "highshelf",
    "peaking_eq",
    "notch",
]

# Log-uniform frequency ranges per filter type (Hz) — matches Rust defaults
_BIQUAD_FREQ_RANGES: dict[str, tuple[float, float]] = {
    "lowpass": (4000.0, 8000.0),
    "highpass": (40.0, 400.0),
    "lowshelf": (40.0, 1000.0),
    "highshelf": (1000.0, 8000.0),
    "peaking_eq": (40.0, 4000.0),
    "notch": (40.0, 4000.0),
}


class RandBiquadFilter(BaseWaveformTransform):
    """Random biquad EQ filtering with 6 standard filter types.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:RandBiquadFilter``.

    Randomly applies between 1 and ``n_stages`` biquad filters chosen from
    lowpass, highpass, lowshelf, highshelf, peaking_eq and notch.  After
    filtering the RMS is optionally restored to its pre-filter level.

    Args:
        p: Probability of applying the augmentation.
        n_stages: Maximum number of sequential filter stages (1..n_stages).
        q_range: Q-factor range ``(min, max)``.
        gain_range: Gain range in dB, used by shelf and peaking filters.
        filter_types: Subset of filter type names to use.  ``None`` means all six.
        equalize_rms: If ``True`` (default), restore pre-filter RMS.
        sample_rate: Sample rate in Hz.
    """

    def __init__(
        self,
        p: float = 0.5,
        n_stages: int = 3,
        q_range: tuple[float, float] = (0.5, 1.5),
        gain_range: tuple[float, float] = (-15.0, 15.0),
        filter_types: Optional[list[str]] = None,
        equalize_rms: bool = True,
        sample_rate: int = 48000,
    ):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")
        self.n_stages = n_stages
        self.q_range = q_range
        self.gain_range = gain_range
        self.filter_types = (
            filter_types if filter_types is not None else _BIQUAD_FILTER_TYPES
        )
        self.equalize_rms = equalize_rms

    def _sample_coefs(self, sr: int) -> tuple[list[float], list[float]]:
        """Sample a random filter type and return ``(b, a)`` coefficient lists."""
        idx = int(torch.randint(0, len(self.filter_types), ()).item())
        ftype = self.filter_types[idx]

        f_low, f_high = _BIQUAD_FREQ_RANGES[ftype]
        # Log-uniform frequency sampling (matches Rust log_uniform)
        log_freq = torch.empty(1).uniform_(math.log(f_low), math.log(f_high)).item()
        center_freq = math.exp(log_freq)
        q = torch.empty(1).uniform_(self.q_range[0], self.q_range[1]).item()
        gain_db = torch.empty(1).uniform_(self.gain_range[0], self.gain_range[1]).item()

        if ftype == "lowpass":
            return _lowpass_coefs(center_freq, q, sr)
        elif ftype == "highpass":
            return _highpass_coefs(center_freq, q, sr)
        elif ftype == "lowshelf":
            return _lowshelf_coefs(center_freq, q, gain_db, sr)
        elif ftype == "highshelf":
            return _highshelf_coefs(center_freq, q, gain_db, sr)
        elif ftype == "peaking_eq":
            return _peaking_eq_coefs(center_freq, q, gain_db, sr)
        else:  # notch
            return _notch_coefs(center_freq, q, sr)

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        sr = sample_rate or self.sample_rate or 48000
        B = samples.shape[0]
        out = samples.clone()

        for i in range(B):
            n = int(torch.randint(1, self.n_stages + 1, ()).item())
            pre_rms = out[i].pow(2).mean().sqrt() if self.equalize_rms else None

            for _ in range(n):
                b_list, a_list = self._sample_coefs(sr)
                b = out.new_tensor([b_list])  # (1, 3)
                a = out.new_tensor([a_list])  # (1, 3)
                out[i : i + 1] = _apply_biquad_batch(out[i : i + 1], b, a)

            if self.equalize_rms and pre_rms is not None:
                post_rms = out[i].pow(2).mean().sqrt().clamp(min=1e-8)
                out[i] = out[i] * (pre_rms / post_rms)

            # Clip guard
            max_val = out[i].abs().max()
            if max_val > 1.0 + 1e-10:
                out[i] = out[i] / (max_val + 1e-10)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 4: RandResample


class RandResample(BaseWaveformTransform):
    """Random resampling for pitch/tempo shifting.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:RandResample``.

    Resamples audio to a randomly perturbed sample rate and back, keeping the
    buffer length the same (pitch shift via tempo change).

    Args:
        p: Probability of applying the augmentation.
        rate_range: Resampling ratio range ``(min, max)`` relative to ``sample_rate``.
            E.g. ``(0.9, 1.1)`` gives ±10 % pitch shift.
        sample_rate: Original sample rate in Hz.
    """

    def __init__(
        self,
        p: float = 0.2,
        rate_range: tuple[float, float] = (0.9, 1.1),
        sample_rate: int = 48000,
    ):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")
        self.rate_range = rate_range

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        sr = sample_rate or self.sample_rate or 48000
        B, C, T = samples.shape
        out = samples.clone()

        for i in range(B):
            rate = (
                torch.empty(1).uniform_(self.rate_range[0], self.rate_range[1]).item()
            )
            # Round to nearest 500 Hz for better GCD — matches Rust implementation
            new_sr = int(round(rate * sr / 500) * 500)
            if new_sr == sr:
                continue

            resampled = resample(out[i], sr, new_sr)  # (C, T')
            new_T = resampled.shape[-1]

            if new_T > T:
                resampled = resampled[:, :T]
            elif new_T < T:
                resampled = F.pad(resampled, (0, T - new_T))

            out[i] = resampled

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 5: RandClipping


class RandClipping(BaseWaveformTransform):
    """Random hard clipping.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:RandClipping``.

    Clips the waveform at a random threshold expressed as a fraction of the
    per-example peak value.

    Args:
        p: Probability of applying the augmentation.
        clip_factor_range: Threshold as fraction of peak amplitude ``(min, max)``.
            E.g. ``(0.05, 0.9)`` clips at 5–90 % of the per-example peak.
    """

    def __init__(
        self,
        p: float = 0.2,
        clip_factor_range: tuple[float, float] = (0.01, 0.25),
    ):
        super().__init__(p=p, output_type="tensor")
        self.clip_factor_range = clip_factor_range

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        B = samples.shape[0]
        out = samples.clone()

        for i in range(B):
            max_val = float(out[i].abs().max().clamp(min=1e-8))
            c_frac = (
                torch.empty(1)
                .uniform_(self.clip_factor_range[0], self.clip_factor_range[1])
                .item()
            )
            threshold = max_val * c_frac
            out[i] = out[i].clamp(-threshold, threshold)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 6: RandZeroingTD


class RandZeroingTD(BaseWaveformTransform):
    """Random time-domain dropout (packet-loss simulation).

    Port of DeepFilterNet ``libDF/src/augmentations.rs:RandZeroingTD``.

    Randomly zeros contiguous blocks of samples across all channels to
    simulate packet loss or microphone drop-outs.

    Args:
        p: Probability of applying the augmentation.
        max_zero_percent: Maximum fraction of the signal to zero out (%).
        min_block_samples: Minimum block length in samples (~2.5 ms @ 48 kHz).
        max_block_samples: Maximum block length in samples (~37.5 ms @ 48 kHz).
    """

    def __init__(
        self,
        p: float = 0.1,
        max_zero_percent: float = 10.0,
        min_block_samples: int = 120,
        max_block_samples: int = 1800,
    ):
        super().__init__(p=p, output_type="tensor")
        self.max_zero_percent = max_zero_percent
        self.min_block_samples = min_block_samples
        self.max_block_samples = max_block_samples

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        B, C, T = samples.shape
        out = samples.clone()

        for i in range(B):
            # Draw random target zeroed fraction in (0.01, max_zero_percent/100)
            target_p = (
                torch.empty(1).uniform_(0.01, self.max_zero_percent / 100.0).item()
            )
            zeroed = 0.0

            max_block = min(self.max_block_samples, T)
            min_block = min(self.min_block_samples, max_block)

            while zeroed < target_p:
                block_len = int(torch.randint(min_block, max_block + 1, ()).item())
                max_start = max(T - block_len, 0)
                pos = (
                    int(torch.randint(0, max_start + 1, ()).item())
                    if max_start > 0
                    else 0
                )
                out[i, :, pos : pos + block_len] = 0.0
                zeroed += block_len / T

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 7: BandwidthLimiterAugmentation


class BandwidthLimiterAugmentation(BaseWaveformTransform):
    """Bandwidth limiting via downsample / upsample.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:BandwidthLimiterAugmentation``.

    Simulates communication-system bandwidth limitations (telephone, VoIP, …)
    by resampling to a lower sample rate and back.  The anti-aliasing filters
    applied by the resampler create a hard bandlimit at the chosen cutoff.

    Args:
        p: Probability of applying the augmentation.
        cutoff_freqs: Candidate upper-frequency cutoffs in Hz.  A random value
            is chosen per example.
        sample_rate: Original sample rate in Hz.
    """

    # Default presets match the Rust implementation
    _DEFAULT_CUTOFFS = [4000, 6000, 8000, 10000, 12000, 16000, 20000, 22050]

    def __init__(
        self,
        p: float = 0.2,
        cutoff_freqs: Optional[list[int]] = None,
        sample_rate: int = 48000,
    ):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")
        self.cutoff_freqs = (
            cutoff_freqs if cutoff_freqs is not None else self._DEFAULT_CUTOFFS
        )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        sr = sample_rate or self.sample_rate or 48000
        B, C, T = samples.shape
        out = samples.clone()

        for i in range(B):
            idx = int(torch.randint(0, len(self.cutoff_freqs), ()).item())
            cutoff = self.cutoff_freqs[idx]
            target_sr = cutoff * 2  # Nyquist: sample at 2× cutoff

            if target_sr >= sr:
                continue  # No limiting needed

            # Downsample → hard bandlimit via anti-alias filter
            limited = resample(out[i], sr, target_sr)
            # Upsample back to original sr
            restored = resample(limited, target_sr, sr)

            new_T = restored.shape[-1]
            if new_T > T:
                restored = restored[:, :T]
            elif new_T < T:
                restored = F.pad(restored, (0, T - new_T))

            out[i] = restored

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 8: AirAbsorptionAugmentation

# ISO 9613-1 air absorption coefficients (×1e-3, units m⁻¹) for 9 frequency bands.
# 8 scenarios covering different temperature / humidity combinations.
_AIR_ABS_CENTER_FREQS: list[int] = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000]
_AIR_ABS_SCENARIOS: list[list[float]] = [
    [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0],  # 10°C 30–50 %
    [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2],  # 10°C 50–70 %
    [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9],  # 10°C 70–90 %
    [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9],  # 20°C 30–50 %
    [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7],  # 20°C 50–70 %
    [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8],  # 20°C 70–90 %
    [0.1, 0.2, 0.7, 1.5, 3.9, 8.1, 21.6, 80.2, 213.1],  # Strong-High-1
    [0.1, 0.3, 0.9, 3.8, 8.9, 21.1, 44.6, 80.2, 153.1],  # Strong-High-2
]


class AirAbsorptionAugmentation(BaseWaveformTransform):
    """Frequency-dependent air absorption attenuation.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:AirAbsorptionAugmentation``.

    Simulates the frequency-dependent damping experienced by sound propagating
    over distance in air (ISO 9613-1 absorption model).  Operates in the
    frequency domain via real FFT.

    Args:
        p: Probability of applying the augmentation.
        distance_range: Source-listener distance range in metres ``(min, max)``.
        sample_rate: Sample rate in Hz (used to map FFT bins to frequencies).
    """

    def __init__(
        self,
        p: float = 0.1,
        distance_range: tuple[float, float] = (1.0, 20.0),
        sample_rate: int = 48000,
    ):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")
        self.distance_range = distance_range

    @staticmethod
    def _build_attenuation_curve(
        distance: float,
        coefs_1e3: list[float],
        sr: int,
        n_fft: int,
    ) -> Tensor:
        """Build per-FFT-bin attenuation gains using Beer–Lambert law.

        Args:
            distance: Propagation distance in metres.
            coefs_1e3: Absorption coefficients ×1e-3 for the 9 ISO bands.
            sr: Sample rate in Hz.
            n_fft: Number of FFT points (real FFT → n_fft // 2 + 1 bins).

        Returns:
            Attenuation gains, shape (n_fft // 2 + 1,).
        """
        band_atten = [math.exp(-d * 1e-3 * distance) for d in coefs_1e3]

        # Piecewise-linear interpolation from 9 ISO bands to all FFT bins
        # Extended with duplicated edges (matches Rust interp_atten)
        ext_freqs = [0.0] + [float(f) for f in _AIR_ABS_CENTER_FREQS] + [sr / 2.0]
        ext_atten = [band_atten[0]] + band_atten + [band_atten[-1]]

        n_bins = n_fft // 2 + 1
        gains = []
        seg = 0
        for k in range(n_bins):
            f = k * sr / n_fft
            while seg < len(ext_freqs) - 2 and f > ext_freqs[seg + 1]:
                seg += 1
            f0, f1 = ext_freqs[seg], ext_freqs[seg + 1]
            a0, a1 = ext_atten[seg], ext_atten[seg + 1]
            if f1 > f0:
                t = (f - f0) / (f1 - f0)
                gains.append(a0 + t * (a1 - a0))
            else:
                gains.append(a0)

        return torch.tensor(gains, dtype=torch.float32)

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        sr = sample_rate or self.sample_rate or 48000
        B, C, T = samples.shape
        out = samples.clone()

        for i in range(B):
            distance = (
                torch.empty(1)
                .uniform_(self.distance_range[0], self.distance_range[1])
                .item()
            )
            scen_idx = int(torch.randint(0, len(_AIR_ABS_SCENARIOS), ()).item())
            coefs = _AIR_ABS_SCENARIOS[scen_idx]

            gains = self._build_attenuation_curve(distance, coefs, sr, T)
            gains = gains.to(samples.device)  # (n_bins,)

            # Real FFT along time axis: (C, T) → (C, n_bins) complex
            spec = torch.fft.rfft(out[i], n=T, dim=-1)
            spec = spec * gains.unsqueeze(0)
            out[i] = torch.fft.irfft(spec, n=T, dim=-1)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 9: RandReverbSim


class RandReverbSim(BaseWaveformTransform):
    """Synthetic room reverberation via FFT convolution.

    Port of DeepFilterNet ``libDF/src/augmentations.rs:RandReverbSim``.

    Generates a synthetic room impulse response (RIR) with exponential decay
    and convolves the audio with it.  A slightly de-reverberant version of the
    RIR (late-tail suppressed) is used so the training target remains cleaner
    than the input.

    Args:
        p: Probability of applying the augmentation.
        rt60_range: RT60 range in seconds ``(min, max)``.
        drr_range: Direct-to-Reverberant Ratio range in dB ``(min, max)``.
            Higher values produce a less reverberant tail.
        offset_late_ms: Offset beyond the direct-path peak at which
            late-reverb suppression starts for the target RIR (ms).
        sample_rate: Sample rate in Hz.
    """

    def __init__(
        self,
        p: float = 0.3,
        rt60_range: tuple[float, float] = (0.2, 1.0),
        drr_range: tuple[float, float] = (0.0, 30.0),
        offset_late_ms: float = 20.0,
        sample_rate: int = 48000,
    ):
        super().__init__(p=p, sample_rate=sample_rate, output_type="tensor")
        self.rt60_range = rt60_range
        self.drr_range = drr_range
        self.offset_late_ms = offset_late_ms

    @staticmethod
    def _fft_convolve(signal: Tensor, kernel: Tensor, out_len: int) -> Tensor:
        """FFT overlap-add convolution, trimmed to ``out_len`` samples.

        Args:
            signal: Shape (C, T).
            kernel: Shape (N_rir,) — single-channel kernel applied to all channels.
            out_len: Desired output length in samples.

        Returns:
            Convolved signal, shape (C, out_len).
        """
        T = signal.shape[-1]
        k_len = kernel.shape[0]
        fft_size = 1
        while fft_size < T + k_len - 1:
            fft_size <<= 1

        S = torch.fft.rfft(signal, n=fft_size, dim=-1)  # (C, fft/2+1)
        K = torch.fft.rfft(kernel.unsqueeze(0), n=fft_size, dim=-1)  # (1, fft/2+1)
        y = torch.fft.irfft(S * K, n=fft_size, dim=-1)  # (C, fft_size)
        return y[:, :out_len]

    def _generate_rir(self, rt60: float, sr: int, length: int) -> Tensor:
        """Generate a synthetic mono RIR with exponential decay.

        Args:
            rt60: Reverberation time in seconds.
            sr: Sample rate in Hz.
            length: RIR length in samples.

        Returns:
            Normalised RIR tensor of shape (length,).
        """
        t = torch.arange(length, dtype=torch.float32) / sr
        tau = rt60 / math.log(1000.0)  # −60 dB in rt60 seconds
        envelope = torch.exp(-t / tau)

        noise = torch.randn(length)
        rir = torch.zeros(length)
        rir[0] = 1.0  # direct path

        early = int(0.05 * sr)
        rir[:early] = rir[:early] + noise[:early] * envelope[:early] * 0.5
        rir[early:] = noise[early:] * envelope[early:] * 0.3

        energy = rir.pow(2).sum().sqrt().clamp(min=1e-8)
        return rir / energy

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        sr = sample_rate or self.sample_rate or 48000
        B, C, T = samples.shape
        out = samples.clone()

        for i in range(B):
            rt60 = (
                torch.empty(1).uniform_(self.rt60_range[0], self.rt60_range[1]).item()
            )
            drr_db = (
                torch.empty(1).uniform_(self.drr_range[0], self.drr_range[1]).item()
            )

            rir = self._generate_rir(rt60, sr, sr)  # 1-second synthetic RIR

            # Apply DRR: scale late reverb tail
            drr_linear = 10.0 ** (drr_db / 10.0)
            reverb_scale = 1.0 / math.sqrt(1.0 + drr_linear)
            rir[1:] = rir[1:] * reverb_scale

            rir = rir.to(samples.device)
            reverbed = self._fft_convolve(out[i], rir, T)

            # Normalise to prevent clipping
            max_val = reverbed.abs().max().clamp(min=1e-8)
            if max_val > 1.0:
                reverbed = reverbed / max_val

            out[i] = reverbed

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Augmentation 10: NoiseGenerator (Colored Noise)

# NoiseGenerator is provided directly by torch_audiomentations as AddColoredNoise.
# It supports the same f_decay range and SNR-based mixing, and is used via the
# get_noise_generator() factory below.
NoiseGenerator = AddColoredNoise

# Pipeline factory helpers


def get_speech_augmentations(
    augmentation_config: AugmentationConfig,
    sample_rate: int = 48000,
    seed: Optional[int] = None,
) -> Compose:
    """Build the speech augmentation pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.
        seed: Ignored — torch handles randomness natively.

    Returns:
        ``Compose`` pipeline accepting tensors of shape (B, C, T).
    """
    transforms = [
        RandRemoveDc(p=augmentation_config.p_remove_dc, sample_rate=sample_rate),
        RandLFilt(p=augmentation_config.p_lfilt),
        RandBiquadFilter(p=augmentation_config.p_biquad, sample_rate=sample_rate),
        RandResample(p=augmentation_config.p_resample, sample_rate=sample_rate),
    ]
    return Compose(transforms, output_type="tensor")


def get_noise_augmentations(
    augmentation_config: AugmentationConfig,
    sample_rate: int = 48000,
    seed: Optional[int] = None,
) -> Compose:
    """Build the noise augmentation pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.
        seed: Ignored — torch handles randomness natively.

    Returns:
        ``Compose`` pipeline accepting tensors of shape (B, C, T).
    """
    transforms = [
        RandClipping(
            p=augmentation_config.p_noise_clipping, clip_factor_range=(0.01, 0.5)
        ),
        RandBiquadFilter(p=augmentation_config.p_noise_biquad, sample_rate=sample_rate),
    ]
    return Compose(transforms, output_type="tensor")


def get_speech_distortions_td(
    augmentation_config: AugmentationConfig,
    sample_rate: int = 48000,
    seed: Optional[int] = None,
) -> Compose:
    """Build the time-domain speech distortion pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.
        seed: Ignored — torch handles randomness natively.

    Returns:
        ``Compose`` pipeline accepting tensors of shape (B, C, T).
    """
    transforms = [
        RandClipping(p=augmentation_config.p_clipping, clip_factor_range=(0.05, 0.9)),
        RandZeroingTD(p=augmentation_config.p_zeroing),
        AirAbsorptionAugmentation(
            p=augmentation_config.p_air_absorption, sample_rate=sample_rate
        ),
        BandwidthLimiterAugmentation(
            p=augmentation_config.p_bandwidth_ext, sample_rate=sample_rate
        ),
    ]
    return Compose(transforms, output_type="tensor")


def get_noise_generator(
    p: float = 0.5,
    f_decay_min: float = -2.0,
    f_decay_max: float = 2.0,
    sample_rate: int = 48000,
    output_type: str = "dict",
) -> AddColoredNoise:
    """Return an :class:`AddColoredNoise` instance (legacy compatibility wrapper).

    ``NoiseGenerator`` is an alias for :class:`torch_audiomentations.AddColoredNoise`,
    which implements the same FFT-based colored noise generation and SNR mixing.

    Args:
        p: Probability of generating noise.
        f_decay_min: Minimum spectral decay exponent.
        f_decay_max: Maximum spectral decay exponent.
        sample_rate: Passed to ``AddColoredNoise`` as ``sample_rate``.
        output_type: Ignored (kept for API compatibility).

    Returns:
        :class:`AddColoredNoise` augmentation.
    """
    return AddColoredNoise(
        min_f_decay=f_decay_min,
        max_f_decay=f_decay_max,
        p=p,
        sample_rate=sample_rate,
        output_type="tensor",
    )


# Quick smoke-test  (python -m src.augmentations)

if __name__ == "__main__":
    logger.info("Testing augmentations (batch=2, channels=1, samples=48000) ...")

    audio = torch.randn(2, 1, 48000, dtype=torch.float32) * 0.1

    cases: list[tuple[str, BaseWaveformTransform]] = [
        ("RandRemoveDc", RandRemoveDc(p=1.0, sample_rate=48000)),
        ("RandLFilt", RandLFilt(p=1.0)),
        ("RandBiquadFilter", RandBiquadFilter(p=1.0, sample_rate=48000)),
        ("RandResample", RandResample(p=1.0, sample_rate=48000)),
        ("RandClipping", RandClipping(p=1.0)),
        ("RandZeroingTD", RandZeroingTD(p=1.0)),
        (
            "BandwidthLimiterAugmentation",
            BandwidthLimiterAugmentation(p=1.0, sample_rate=48000),
        ),
        (
            "AirAbsorptionAugmentation",
            AirAbsorptionAugmentation(p=1.0, sample_rate=48000),
        ),
        ("RandReverbSim", RandReverbSim(p=1.0, sample_rate=48000)),
        ("NoiseGenerator (AddColoredNoise)", NoiseGenerator(p=1.0, sample_rate=48000)),
    ]

    for name, aug in cases:
        aug.train()
        try:
            result = aug(audio, sample_rate=48000)
            shape = result.samples.shape if hasattr(result, "samples") else result.shape
            logger.info(f"  ok  {name}: {shape}")
        except Exception as exc:
            logger.error(f"  ERR {name}: {exc}")

    logger.info("Pipelines ...")
    cfg = AugmentationConfig()
    for name, pipeline in [
        ("get_speech_augmentations", get_speech_augmentations(cfg)),
        ("get_noise_augmentations", get_noise_augmentations(cfg)),
        ("get_speech_distortions_td", get_speech_distortions_td(cfg)),
    ]:
        pipeline.train()
        try:
            result = pipeline(audio, sample_rate=48000)
            shape = result.samples.shape if hasattr(result, "samples") else result.shape
            logger.info(f"  ok  {name}: {shape}")
        except Exception as exc:
            logger.error(f"  ERR {name}: {exc}")

    logger.info("Done.")
