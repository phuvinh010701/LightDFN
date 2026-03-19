import os
from typing import Optional

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch import Tensor
from torch_audiomentations import Compose
from torch_audiomentations.core.composition import ObjectDict
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from src.configs.config import AugmentationConfig
from src.utils.io import resample


def biquad_filter_inplace(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> None:
    """Apply biquad filter in-place using Direct Form II.

    Port of DeepFilterNet biquad implementation.

    Transfer function:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)

    Args:
        audio: Audio array of shape (samples,) - modified in-place
        b: Feedforward coefficients [b0, b1, b2]
        a: Feedback coefficients [a0, a1, a2] (normalized, a0=1)
    """
    # Ensure normalized (a0 should be 1.0)
    if len(a) == 3 and a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    # State variables for Direct Form II
    w1, w2 = 0.0, 0.0

    for i in range(len(audio)):
        # Compute intermediate value
        w0 = audio[i] - a[1] * w1 - a[2] * w2

        # Compute output
        audio[i] = b[0] * w0 + b[1] * w1 + b[2] * w2

        # Update state
        w2 = w1
        w1 = w0


def biquad_norm(x: Tensor, b: Tensor, a: Tensor) -> Tensor:
    """
    x: (T,)
    b: (2,) -> [b0, b1]
    a: (2,) -> [a0, a1]
    """
    T = x.shape[0]
    y = torch.zeros_like(x)

    y_prev1 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    y_prev2 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    x_prev1 = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for t in range(T):
        x_curr = x[t]
        y_curr = b[0] * x_curr + b[1] * x_prev1 - a[0] * y_prev1 - a[1] * y_prev2

        y[t] = y_curr

        x_prev1 = x_curr
        y_prev2 = y_prev1
        y_prev1 = y_curr

    return y


def lowpass_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate lowpass biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Cutoff frequency in Hz
        q_factor: Q factor (0.5-1.5 typical, higher = sharper cutoff)
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients, each shape (3,)
    """
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = b0

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    # Normalize by a0
    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


def highpass_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate highpass biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Cutoff frequency in Hz
        q_factor: Q factor (0.5-1.5 typical)
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients
    """
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)

    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = b0

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


def lowshelf_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate lowshelf biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Center frequency in Hz
        q_factor: Q factor
        gain_db: Gain in dB (positive = boost, negative = cut)
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients
    """
    w0 = 2.0 * np.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    S = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + S)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - S)

    a0 = (A + 1.0) + (A - 1.0) * cos_w0 + S
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 = (A + 1.0) + (A - 1.0) * cos_w0 - S

    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


def highshelf_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate highshelf biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Center frequency in Hz
        q_factor: Q factor
        gain_db: Gain in dB
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients
    """
    w0 = 2.0 * np.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    S = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + S)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - S)

    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + S
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - S

    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


def peaking_eq_coefs(
    center_freq: float, q_factor: float, gain_db: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate peaking EQ biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Center frequency in Hz
        q_factor: Q factor (bandwidth)
        gain_db: Gain in dB
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients
    """
    w0 = 2.0 * np.pi * center_freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A

    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


def notch_coefs(
    center_freq: float, q_factor: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate notch biquad filter coefficients.

    Audio EQ Cookbook (W3C Standard)

    Args:
        center_freq: Center frequency in Hz to eliminate
        q_factor: Q factor (bandwidth)
        sr: Sample rate in Hz

    Returns:
        Tuple of (b, a) normalized coefficients
    """
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)

    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([1.0, a1, a2], dtype=np.float32) / a0

    return b, a


class RandRemoveDc(BaseWaveformTransform):
    """Remove DC offset by subtracting mean.

    Simple mean subtraction: y[n] = x[n] - mean(x)
    """

    def __init__(self, prob: float = 0.25, sample_rate: int = 48000):
        super().__init__(p=prob, sample_rate=sample_rate, output_type="tensor")

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """Remove DC offset from audio.

        Args:
            samples: Audio array of shape (batch_size, channels, samples)

        Returns:
            ObjectDict with DC offset removed
        """
        mean = samples.mean(dim=-1, keepdim=True)
        samples = samples - mean
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class BaseAugmentation(BaseWaveformTransform):
    """Base augmentation class."""

    def __init__(self, prob: float, seed: Optional[int] = None):
        super().__init__(p=prob, output_type="tensor")
        self.rng = np.random.default_rng(seed)

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """Apply augmentation."""
        if hasattr(self, "apply"):
            np_samples = samples.cpu().numpy()
            out_samples = np.zeros_like(np_samples)
            for b in range(np_samples.shape[0]):
                out_samples[b] = self.apply(np_samples[b], sample_rate)
            samples = torch.from_numpy(out_samples).to(samples.device)

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class RandLFilt(BaseWaveformTransform):
    """Random low/high-pass filtering using normalized biquad.

    Applies second-order biquad with coefficients sampled from ranges.
    Adopted from RNNoise/PercepNet implementations.
    """

    def __init__(
        self,
        p: float = 0.5,
        a_range: tuple[float, float] = (-3.0 / 8.0, 3.0 / 8.0),
        b_range: tuple[float, float] = (-3.0 / 8.0, 3.0 / 8.0),
    ):
        """Initialize RandLFilt.

        Args:
            prob: Probability of applying
            a_range: Feedback coefficient range (min, max)
            b_range: Feedforward coefficient range (min, max)
            seed: Random seed
        """
        super().__init__(p=p)
        self.a_range = a_range
        self.b_range = b_range

    def apply_transform(self, samples: Tensor) -> Tensor:
        """Apply random normalized biquad filter.

        Args:
            samples: Audio of shape (batch_size, channels, samples)

        Returns:
            Filtered audio
        """
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
            a_i = a[i]
            b_i = b[i]
            for ch in range(C):
                out[i, ch] = biquad_norm(out[i, ch], b_i, a_i)

        return out


# ============================================================================
# Augmentation 3: RandBiquadFilter
# ============================================================================


class RandBiquadFilter(BaseAugmentation):
    """Random biquad filtering with 6 filter types.

    Port of DeepFilterNet libDF/src/augmentations.rs:RandBiquadFilter

    Supports: lowpass, highpass, lowshelf, highshelf, peaking_eq, notch
    """

    FILTER_TYPES = [
        "lowpass",
        "highpass",
        "lowshelf",
        "highshelf",
        "peaking_eq",
        "notch",
    ]

    def __init__(
        self,
        prob: float = 0.5,
        freq_range: tuple[float, float] = (200.0, 8000.0),
        q_range: tuple[float, float] = (0.5, 1.5),
        gain_range: tuple[float, float] = (-10.0, 10.0),
        filter_types: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize RandBiquadFilter.

        Args:
            prob: Probability of applying
            freq_range: Center frequency range in Hz
            q_range: Q factor range
            gain_range: Gain range in dB (for shelf/peaking filters)
            filter_types: List of filter types to use (None = all)
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.freq_range = freq_range
        self.q_range = q_range
        self.gain_range = gain_range
        self.filter_types = filter_types if filter_types else self.FILTER_TYPES

    def apply(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Apply random biquad filter.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz

        Returns:
            Filtered audio
        """
        # Ensure 2D
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        # Sample parameters
        filter_type = self.rng.choice(self.filter_types)
        center_freq = self.rng.uniform(self.freq_range[0], self.freq_range[1])
        q_factor = self.rng.uniform(self.q_range[0], self.q_range[1])
        gain_db = self.rng.uniform(self.gain_range[0], self.gain_range[1])

        # Get coefficients
        if filter_type == "lowpass":
            b, a = lowpass_coefs(center_freq, q_factor, sample_rate)
        elif filter_type == "highpass":
            b, a = highpass_coefs(center_freq, q_factor, sample_rate)
        elif filter_type == "lowshelf":
            b, a = lowshelf_coefs(center_freq, q_factor, gain_db, sample_rate)
        elif filter_type == "highshelf":
            b, a = highshelf_coefs(center_freq, q_factor, gain_db, sample_rate)
        elif filter_type == "peaking_eq":
            b, a = peaking_eq_coefs(center_freq, q_factor, gain_db, sample_rate)
        elif filter_type == "notch":
            b, a = notch_coefs(center_freq, q_factor, sample_rate)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Apply per channel
        for ch_idx in range(audio.shape[0]):
            biquad_filter_inplace(audio[ch_idx], b, a)

        if squeeze:
            audio = audio.squeeze(0)

        return audio


# ============================================================================
# Augmentation 4: RandResample
# ============================================================================


class RandResample(BaseAugmentation):
    """Random resampling for pitch shifting.

    Port of DeepFilterNet libDF/src/augmentations.rs:RandResample

    Resamples audio to different sample rate and back to original.
    """

    def __init__(
        self,
        prob: float = 0.2,
        rate_range: tuple[float, float] = (0.9, 1.1),
        seed: Optional[int] = None,
    ):
        """Initialize RandResample.

        Args:
            prob: Probability of applying
            rate_range: Resampling rate range (0.5 to 2.0)
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.rate_range = rate_range

    def apply(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Apply random resampling.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)
            sample_rate: Original sample rate in Hz

        Returns:
            Resampled audio (same length as input)
        """
        # Sample resampling rate
        rate = self.rng.uniform(self.rate_range[0], self.rate_range[1])

        # Calculate target sample rate
        new_sr = int(sample_rate * rate)

        # Convert to tensor for resampling
        audio_torch = torch.from_numpy(audio).float()

        # Resample down and back up
        audio_resampled = resample(audio_torch, sample_rate, new_sr)
        audio_restored = resample(audio_resampled, new_sr, sample_rate)

        # Ensure same length (crop/pad if needed due to rounding)
        if audio.ndim == 1:
            target_len = audio.shape[0]
            if audio_restored.shape[0] > target_len:
                audio_restored = audio_restored[:target_len]
            elif audio_restored.shape[0] < target_len:
                pad = target_len - audio_restored.shape[0]
                audio_restored = torch.nn.functional.pad(audio_restored, (0, pad))
        else:
            target_len = audio.shape[1]
            if audio_restored.shape[1] > target_len:
                audio_restored = audio_restored[:, :target_len]
            elif audio_restored.shape[1] < target_len:
                pad = target_len - audio_restored.shape[1]
                audio_restored = torch.nn.functional.pad(audio_restored, (0, pad))

        return audio_restored.numpy()


# ============================================================================
# Augmentation 5: RandClipping
# ============================================================================


class RandClipping(BaseAugmentation):
    """Random soft clipping using tanh.

    Port of DeepFilterNet libDF/src/augmentations.rs:RandClipping

    Applies soft clipping: y = tanh(k * x) where k is the clipping factor.
    """

    def __init__(
        self,
        prob: float = 0.2,
        clip_factor_range: tuple[float, float] = (0.1, 3.0),
        seed: Optional[int] = None,
    ):
        """Initialize RandClipping.

        Args:
            prob: Probability of applying
            clip_factor_range: Clipping factor range (higher = more clipping)
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.clip_factor_range = clip_factor_range

    def apply(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Apply soft clipping.

        Args:
            audio: Audio of any shape

        Returns:
            Clipped audio
        """
        # Sample clipping factor
        k = self.rng.uniform(self.clip_factor_range[0], self.clip_factor_range[1])

        # Apply tanh clipping
        return np.tanh(k * audio).astype(np.float32)


# ============================================================================
# Augmentation 6: RandZeroingTD
# ============================================================================


class RandZeroingTD(BaseAugmentation):
    """Random time-domain dropout (spectral zeroing).

    Port of DeepFilterNet libDF/src/augmentations.rs:RandZeroingTD

    Randomly zeros out frequency bands to simulate packet loss.
    """

    def __init__(
        self,
        prob: float = 0.1,
        zero_prob: float = 0.2,
        max_consecutive: int = 5,
        seed: Optional[int] = None,
    ):
        """Initialize RandZeroingTD.

        Args:
            prob: Probability of applying augmentation
            zero_prob: Probability of zeroing each time frame
            max_consecutive: Maximum consecutive frames to zero
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.zero_prob = zero_prob
        self.max_consecutive = max_consecutive

    def apply(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Apply random dropout.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)

        Returns:
            Audio with random dropout
        """
        # Work on copy
        audio = audio.copy()

        # Determine frame size (e.g., 20ms)
        if sample_rate is None:
            sample_rate = 48000
        frame_size = int(sample_rate * 0.02)  # 20ms frames

        if audio.ndim == 1:
            n_samples = audio.shape[0]
        else:
            n_samples = audio.shape[1]

        n_frames = n_samples // frame_size

        # Randomly zero frames
        i = 0
        while i < n_frames:
            if self.rng.uniform() < self.zero_prob:
                # Zero this frame and possibly consecutive ones
                n_consecutive = self.rng.integers(1, self.max_consecutive + 1)
                n_consecutive = min(n_consecutive, n_frames - i)

                start = i * frame_size
                end = (i + n_consecutive) * frame_size

                if audio.ndim == 1:
                    audio[start:end] = 0.0
                else:
                    audio[:, start:end] = 0.0

                i += n_consecutive
            else:
                i += 1

        return audio


# ============================================================================
# Augmentation 7: BandwidthLimiterAugmentation
# ============================================================================


class BandwidthLimiterAugmentation(BaseAugmentation):
    """Bandwidth limiting via low-pass filtering and resampling.

    Port of DeepFilterNet libDF/src/augmentations.rs:BandwidthLimiterAugmentation

    Simulates communication systems (telephone, VoIP, etc.).
    8 presets for different bandwidths.
    """

    # Bandwidth presets (cutoff_freq, sample_rate)
    PRESETS = {
        0: (4000, 8000),  # Narrowband (telephone)
        1: (6000, 12000),  # Wideband
        2: (8000, 16000),  # Super-wideband
        3: (12000, 24000),  # Full-band
        4: (16000, 32000),  # High quality
        5: (18000, 36000),  # Higher quality
        6: (20000, 40000),  # Very high quality
        7: (22000, 44000),  # Near full bandwidth
    }

    def __init__(
        self,
        prob: float = 0.2,
        presets: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize BandwidthLimiter.

        Args:
            prob: Probability of applying
            presets: List of preset indices to use (None = all)
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.presets = presets if presets else list(self.PRESETS.keys())

    def apply(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Apply bandwidth limiting.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)
            sample_rate: Original sample rate

        Returns:
            Bandwidth-limited audio
        """
        # Choose random preset
        preset_idx = self.rng.choice(self.presets)
        cutoff_freq, target_sr = self.PRESETS[preset_idx]

        # Convert to tensor
        audio_torch = torch.from_numpy(audio).float()

        # Downsample to target SR (includes low-pass filtering)
        audio_limited = resample(audio_torch, sample_rate, target_sr)

        # Upsample back to original SR
        audio_restored = resample(audio_limited, target_sr, sample_rate)

        # Match length
        if audio.ndim == 1:
            target_len = audio.shape[0]
            if audio_restored.shape[0] > target_len:
                audio_restored = audio_restored[:target_len]
            elif audio_restored.shape[0] < target_len:
                pad = target_len - audio_restored.shape[0]
                audio_restored = torch.nn.functional.pad(audio_restored, (0, pad))
        else:
            target_len = audio.shape[1]
            if audio_restored.shape[1] > target_len:
                audio_restored = audio_restored[:, :target_len]
            elif audio_restored.shape[1] < target_len:
                pad = target_len - audio_restored.shape[1]
                audio_restored = torch.nn.functional.pad(audio_restored, (0, pad))

        return audio_restored.numpy()


# ============================================================================
# Augmentation 8: AirAbsorptionAugmentation
# ============================================================================


class AirAbsorptionAugmentation(BaseAugmentation):
    """Air absorption frequency attenuation.

    Port of DeepFilterNet libDF/src/augmentations.rs:AirAbsorptionAugmentation

    Simulates environmental frequency-dependent attenuation.
    """

    def __init__(
        self,
        prob: float = 0.1,
        distance_range: tuple[float, float] = (1.0, 100.0),
        seed: Optional[int] = None,
    ):
        """Initialize AirAbsorption.

        Args:
            prob: Probability of applying
            distance_range: Distance range in meters
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.distance_range = distance_range

    def apply(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Apply air absorption.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)
            sample_rate: Sample rate

        Returns:
            Audio with air absorption applied
        """
        # Sample distance
        distance = self.rng.uniform(self.distance_range[0], self.distance_range[1])

        # Air absorption coefficients (dB/m) - frequency dependent
        # Simplified model: higher frequencies attenuate more
        # alpha(f) ≈ k * f^2 where k depends on humidity/temperature

        # Apply frequency-dependent attenuation via high-shelf filter
        # Approximate: -0.01 to -0.1 dB per meter for high frequencies
        attenuation_db = -0.05 * distance  # Scale with distance

        # Use high-shelf filter to attenuate high frequencies
        center_freq = 4000.0  # Start attenuating above 4 kHz
        q_factor = 0.7

        b, a = highshelf_coefs(center_freq, q_factor, attenuation_db, sample_rate)

        # Apply filter
        if audio.ndim == 1:
            audio = audio.copy()
            biquad_filter_inplace(audio, b, a)
        else:
            audio = audio.copy()
            for ch_idx in range(audio.shape[0]):
                biquad_filter_inplace(audio[ch_idx], b, a)

        return audio


# ============================================================================
# Augmentation 9: RandReverbSim
# ============================================================================


class RandReverbSim(BaseAugmentation):
    """Room impulse response (RIR) convolution for reverb simulation.

    Port of DeepFilterNet libDF/src/augmentations.rs:RandReverbSim

    Supports:
    - 3 scenarios: noise-only, speech-only, both
    - RT60 decay control
    - DRR (Direct-to-Reverberant Ratio) control
    - Late reflection suppression
    """

    def __init__(
        self,
        prob: float = 0.3,
        rir_files: Optional[list[str]] = None,
        rt60_range: tuple[float, float] = (0.2, 1.0),
        drr_range: tuple[float, float] = (0.0, 30.0),
        seed: Optional[int] = None,
    ):
        """Initialize RandReverbSim.

        Args:
            prob: Probability of applying
            rir_files: List of RIR file paths (if None, generates synthetic)
            rt60_range: RT60 range in seconds
            drr_range: Direct-to-Reverberant Ratio range in dB
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.rir_files = rir_files
        self.rt60_range = rt60_range
        self.drr_range = drr_range

    def _generate_synthetic_rir(
        self, rt60: float, sample_rate: int, length_samples: int
    ) -> np.ndarray:
        """Generate synthetic RIR with exponential decay.

        Args:
            rt60: Reverberation time in seconds
            sample_rate: Sample rate
            length_samples: Length of RIR in samples

        Returns:
            Synthetic RIR
        """
        # Time axis
        t = np.arange(length_samples) / sample_rate

        # Exponential decay envelope
        decay_constant = -6.91 / rt60  # -60 dB in rt60 seconds
        envelope = np.exp(decay_constant * t)

        # Random noise for diffuse reflections
        noise = self.rng.normal(0, 1, size=length_samples)

        # Direct path (delta at start)
        rir = np.zeros(length_samples, dtype=np.float32)
        rir[0] = 1.0

        # Early reflections (first 50ms) - stronger
        early_samples = int(0.05 * sample_rate)
        rir[:early_samples] += noise[:early_samples] * envelope[:early_samples] * 0.5

        # Late reverberation
        rir[early_samples:] += noise[early_samples:] * envelope[early_samples:] * 0.3

        # Normalize
        rir = rir / np.sqrt(np.sum(rir**2))

        return rir

    def apply(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Apply reverb via RIR convolution.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)
            sample_rate: Sample rate

        Returns:
            Reverberated audio
        """
        # Sample RT60 and DRR
        rt60 = self.rng.uniform(self.rt60_range[0], self.rt60_range[1])
        drr_db = self.rng.uniform(self.drr_range[0], self.drr_range[1])

        # Generate RIR (1 second)
        rir_length = sample_rate
        rir = self._generate_synthetic_rir(rt60, sample_rate, rir_length)

        # Apply DRR: control ratio of direct to reverberant energy
        # DRR = 10 * log10(E_direct / E_reverb)
        drr_linear = 10.0 ** (drr_db / 10.0)

        # Scale reverb tail based on DRR
        # Higher DRR = less reverb
        reverb_scale = 1.0 / np.sqrt(1.0 + drr_linear)
        rir[1:] *= reverb_scale

        # Convolve audio with RIR
        if audio.ndim == 1:
            reverb_audio = scipy_signal.fftconvolve(audio, rir, mode="same")
        else:
            reverb_audio = np.zeros_like(audio)
            for ch_idx in range(audio.shape[0]):
                reverb_audio[ch_idx] = scipy_signal.fftconvolve(
                    audio[ch_idx], rir, mode="same"
                )

        # Normalize to prevent clipping
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 1.0:
            reverb_audio = reverb_audio / (max_val + 1e-8)

        return reverb_audio.astype(np.float32)


# ============================================================================
# Augmentation 10: NoiseGenerator (Colored Noise)
# ============================================================================


def generate_colored_noise(
    num_samples: int,
    f_decay: float,
    num_channels: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate colored noise with specified frequency decay.

    Args:
        num_samples: Number of samples to generate
        f_decay: Frequency decay exponent
                 -2.0 = Brown noise (1/f²)
                 -1.0 = Pink noise (1/f)
                  0.0 = White noise (flat)
                 +1.0 = Blue noise (f)
                 +2.0 = Purple/violet noise (f²)
        num_channels: Number of channels
        rng: Random number generator

    Returns:
        Colored noise array of shape (num_channels, num_samples)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate white noise in frequency domain
    n_fft = 2 ** int(np.ceil(np.log2(num_samples)))
    white_noise = rng.normal(0, 1, size=(num_channels, n_fft // 2 + 1))

    # Create frequency-dependent scaling
    freqs = np.arange(n_fft // 2 + 1)
    freqs[0] = 1  # Avoid division by zero at DC

    # Apply power law: S(f) = 1 / f^decay
    scale = freqs ** (-f_decay / 2.0)  # sqrt because we're scaling amplitude
    scale = scale / np.sqrt(np.mean(scale**2))  # Normalize

    # Apply scaling
    colored_spectrum = white_noise * scale[np.newaxis, :]

    # Convert to time domain
    colored_noise = np.fft.irfft(colored_spectrum, n=n_fft, axis=1)

    # Crop to desired length
    colored_noise = colored_noise[:, :num_samples]

    # Normalize RMS
    rms_val = np.sqrt(np.mean(colored_noise**2))
    if rms_val > 1e-8:
        colored_noise = colored_noise / rms_val

    return colored_noise.astype(np.float32)


class NoiseGenerator(BaseAugmentation):
    """Generate and add colored noise.

    Supports: white, pink, brown, blue, purple noise.
    """

    def __init__(
        self,
        prob: float = 0.5,
        f_decay_range: tuple[float, float] = (-2.0, 2.0),
        snr_range: tuple[float, float] = (0.0, 30.0),
        seed: Optional[int] = None,
    ):
        """Initialize NoiseGenerator.

        Args:
            prob: Probability of applying
            f_decay_range: Frequency decay exponent range
            snr_range: SNR range in dB
            seed: Random seed
        """
        super().__init__(prob, seed)
        self.f_decay_range = f_decay_range
        self.snr_range = snr_range

    def apply(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Add colored noise to audio.

        Args:
            audio: Audio of shape (channels, samples) or (samples,)

        Returns:
            Audio with noise added
        """
        # Determine shape
        if audio.ndim == 1:
            num_channels = 1
            num_samples = audio.shape[0]
            audio = audio[np.newaxis, :]
            squeeze = True
        else:
            num_channels, num_samples = audio.shape
            squeeze = False

        # Sample parameters
        f_decay = self.rng.uniform(self.f_decay_range[0], self.f_decay_range[1])
        snr_db = self.rng.uniform(self.snr_range[0], self.snr_range[1])

        # Generate noise
        noise = generate_colored_noise(num_samples, f_decay, num_channels, self.rng)

        # Mix at target SNR
        from src.utils.utils import mix_f

        scale = mix_f(audio, noise, snr_db)
        noisy = audio + noise * scale

        # Prevent clipping
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / (max_val + 1e-8)

        if squeeze:
            noisy = noisy.squeeze(0)

        return noisy


# ============================================================================
# Compose - Chain Multiple Augmentations
# ============================================================================


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable or use default."""
    return float(os.environ.get(key, default))


def get_speech_augmentations(
    augmentation_config: AugmentationConfig,
    sample_rate: int = 48000,
    seed: Optional[int] = None,
) -> Compose:
    """Get speech augmentation pipeline.

    Returns:
        Compose object with speech augmentations
    """

    transforms = [
        RandRemoveDc(prob=augmentation_config.p_remove_dc, sample_rate=sample_rate),
        RandLFilt(prob=augmentation_config.p_lfilt),
        # RandBiquadFilter(prob=augmentation_config.p_biquad, seed=seed),
        # RandResample(prob=augmentation_config.p_resample, seed=seed),
        # RandClipping(prob=augmentation_config.p_clipping, seed=seed),
    ]

    return Compose(transforms, output_type="tensor")


def get_noise_augmentations(seed: Optional[int] = None) -> Compose:
    """Get noise augmentation pipeline.

    Supports environment variables:
    - DF_P_CLIPPING_NOISE: Probability for noise clipping (default: 0.15)
    - DF_P_BIQUAD_NOISE: Probability for noise biquad (default: 0.4)

    Returns:
        Compose object with noise augmentations
    """
    p_clipping = get_env_float("DF_P_CLIPPING_NOISE", 0.15)
    p_biquad = get_env_float("DF_P_BIQUAD_NOISE", 0.4)

    transforms = [
        RandClipping(prob=p_clipping, seed=seed),
        RandBiquadFilter(prob=p_biquad, seed=seed),
    ]

    return Compose(transforms, output_type="tensor")


def get_speech_distortions_td(seed: Optional[int] = None) -> Compose:
    """Get time-domain speech distortion pipeline.

    Supports environment variables:
    - DF_P_ZEROING: Probability for time-domain dropout (default: 0.1)
    - DF_P_AIR_AUG: Probability for air absorption (default: 0.05)

    Returns:
        Compose object with distortion augmentations
    """
    p_zeroing = get_env_float("DF_P_ZEROING", 0.1)
    p_air = get_env_float("DF_P_AIR_AUG", 0.05)

    transforms = [
        RandZeroingTD(prob=p_zeroing, seed=seed),
        AirAbsorptionAugmentation(prob=p_air, seed=seed),
        BandwidthLimiterAugmentation(prob=0.2, seed=seed),
    ]

    return Compose(transforms, output_type="tensor")


def get_noise_generator(
    p: float = 0.5,
    f_decay_min: float = -2.0,
    f_decay_max: float = 2.0,
    sample_rate: int = 48000,
    output_type: str = "tensor",
) -> NoiseGenerator:
    """Get noise generator (legacy compatibility).

    Args:
        p: Probability of generating noise
        f_decay_min: Minimum frequency decay
        f_decay_max: Maximum frequency decay
        sample_rate: Sample rate
        output_type: Ignored (for compatibility)

    Returns:
        NoiseGenerator augmentation
    """
    return NoiseGenerator(prob=p, f_decay_range=(f_decay_min, f_decay_max))


if __name__ == "__main__":
    # Test augmentations
    import torch

    from src.configs.config import load_config

    print("Testing augmentations...")

    # Create test audio (batch_size, num_channels, num_samples)
    audio = torch.randn((2, 1, 48000), dtype=torch.float32) * 0.1

    # Test each augmentation
    augmentations = [
        ("RandRemoveDc", RandRemoveDc(prob=1.0)),
        ("RandLFilt", RandLFilt(prob=1.0)),
        # ("RandBiquadFilter", RandBiquadFilter(prob=1.0)),
        # ("RandResample", RandResample(prob=1.0)),
        # ("RandClipping", RandClipping(prob=1.0)),
        # ("RandZeroingTD", RandZeroingTD(prob=1.0)),
        # ("BandwidthLimiter", BandwidthLimiterAugmentation(prob=1.0)),
        # ("AirAbsorption", AirAbsorptionAugmentation(prob=1.0)),
        # ("RandReverbSim", RandReverbSim(prob=1.0)),
        # ("NoiseGenerator", NoiseGenerator(prob=1.0)),
    ]

    for name, aug in augmentations:
        try:
            output = aug(audio, sample_rate=48000)
            print(f"✓ {name}: {output.shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    # Test pipelines
    print("\nTesting pipelines...")
    model_config, augmentation_config = load_config()

    speech_augs = get_speech_augmentations(augmentation_config)
    # noise_augs = get_noise_augmentations()
    # distortion_augs = get_speech_distortions_td()

    output_speech = speech_augs(audio, sample_rate=48000)
    print(f"✓ Speech augmentations: {output_speech.shape}")

    # output_noise = noise_augs(audio, sample_rate=48000)
    # print(f"✓ Noise augmentations: {output_noise.shape}")

    # output_dist = distortion_augs(audio, sample_rate=48000)
    # print(f"✓ Distortion augmentations: {output_dist.shape}")

    print("\nAll tests passed!")
