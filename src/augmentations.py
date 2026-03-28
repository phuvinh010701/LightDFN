import math
import random
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from loguru import logger
from torch import Tensor
from torch_audiomentations import AddColoredNoise, Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from src.configs.config import AugmentationConfig
from src.utils.audio import fft_convolve
from src.utils.io import resample


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
            # b0 is implicitly 1; a0 is implicitly 1 — matches biquad_norm equation
            b_coeffs = samples.new_tensor([1.0, b[i, 0].item(), b[i, 1].item()])
            a_coeffs = samples.new_tensor([1.0, a[i, 0].item(), a[i, 1].item()])
            out[i] = AF.lfilter(out[i], a_coeffs, b_coeffs, clamp=False)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


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

    def _apply_random_filter(self, waveform: Tensor, sr: int) -> Tensor:
        """Apply one randomly sampled biquad filter stage."""
        ftype = random.choice(self.filter_types)
        f_low, f_high = _BIQUAD_FREQ_RANGES[ftype]
        center_freq = math.exp(random.uniform(math.log(f_low), math.log(f_high)))
        q = random.uniform(self.q_range[0], self.q_range[1])
        gain_db = random.uniform(self.gain_range[0], self.gain_range[1])

        if ftype == "lowpass":
            return AF.lowpass_biquad(waveform, sr, center_freq, q)
        elif ftype == "highpass":
            return AF.highpass_biquad(waveform, sr, center_freq, q)
        elif ftype == "lowshelf":
            return AF.bass_biquad(waveform, sr, gain_db, center_freq, q)
        elif ftype == "highshelf":
            return AF.treble_biquad(waveform, sr, gain_db, center_freq, q)
        elif ftype == "peaking_eq":
            return AF.equalizer_biquad(waveform, sr, center_freq, gain_db, q)
        else:  # notch
            return AF.bandreject_biquad(waveform, sr, center_freq, q)

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
            n = random.randint(1, self.n_stages)
            pre_rms = out[i].pow(2).mean().sqrt() if self.equalize_rms else None

            for _ in range(n):
                out[i] = self._apply_random_filter(out[i], sr)

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
            reverbed = fft_convolve(out[i], rir)

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


# NoiseGenerator is provided directly by torch_audiomentations as AddColoredNoise.
# It supports the same f_decay range and SNR-based mixing, and is used via the
# get_noise_generator() factory below.
NoiseGenerator = AddColoredNoise

# Pipeline factory helpers


def get_speech_augmentations(
    augmentation_config: AugmentationConfig,
    sample_rate: int = 48000,
) -> Compose:
    """Build the speech augmentation pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.

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
) -> Compose:
    """Build the noise augmentation pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.

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
) -> Compose:
    """Build the time-domain speech distortion pipeline.

    Args:
        augmentation_config: Probabilities loaded from YAML config.
        sample_rate: Sample rate in Hz.

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
    output_type: str = "tensor",
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
        output_type=output_type,
    )


def apply_augmentations(pipeline: Compose, audio: Tensor, sr: int) -> Tensor:
    """Run a torch-audiomentations pipeline on a ``(C, T)`` Tensor.

    Returns a ``(C, T)`` Tensor.
    """
    out = pipeline(audio.unsqueeze(0), sample_rate=sr)  # (1, C, T)
    return out.squeeze(0)  # (C, T)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torchaudio

    from src.utils.io import resample as io_resample

    TARGET_SR = 48000
    DEMO_DIR = Path("datasets/demo")
    OUT_DIR = Path("datasets/demo/augmented")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEMO_FILES = ["noise.wav", "noisy.mp3", "speech.wav"]

    def load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[Tensor, int]:
        """Load audio, convert to mono float32, resample to target_sr."""
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = io_resample(wav, sr, target_sr)
        return wav, target_sr

    def plot_comparison(
        orig: Tensor,
        aug: Tensor,
        sr: int,
        title: str,
        save_path: Path,
    ) -> None:
        """Plot waveform and spectrogram for original vs augmented."""
        orig_np = orig.squeeze(0).numpy()
        aug_np = aug.squeeze(0).numpy()
        t = torch.arange(orig_np.shape[0]) / sr

        fig, axes = plt.subplots(2, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        axes[0, 0].plot(t.numpy(), orig_np, linewidth=0.5)
        axes[0, 0].set_title("Original — waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")

        axes[0, 1].plot(t.numpy(), aug_np, linewidth=0.5, color="tab:orange")
        axes[0, 1].set_title(f"Augmented — waveform")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude")

        n_fft = 1024
        hop = 256
        for ax, signal, label, cmap in [
            (axes[1, 0], orig_np, "Original — spectrogram", "viridis"),
            (axes[1, 1], aug_np, "Augmented — spectrogram", "plasma"),
        ]:
            spec = torch.stft(
                torch.from_numpy(signal),
                n_fft=n_fft,
                hop_length=hop,
                return_complex=True,
            )
            log_power = (spec.abs() + 1e-8).log10().numpy()
            im = ax.imshow(
                log_power,
                origin="lower",
                aspect="auto",
                extent=[0, len(signal) / sr, 0, sr / 2],
                cmap=cmap,
            )
            ax.set_title(label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(im, ax=ax, label="log₁₀(|STFT|)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        logger.info(f"    plot → {save_path}")


    AUGMENTATION_CASES: list[tuple[str, BaseWaveformTransform]] = [
        ("RandRemoveDc", RandRemoveDc(p=1.0, sample_rate=TARGET_SR)),
        ("RandLFilt", RandLFilt(p=1.0)),
        ("RandBiquadFilter", RandBiquadFilter(p=1.0, sample_rate=TARGET_SR)),
        ("RandResample", RandResample(p=1.0, sample_rate=TARGET_SR)),
        ("RandClipping", RandClipping(p=1.0)),
        ("RandZeroingTD", RandZeroingTD(p=1.0)),
        ("BandwidthLimiterAugmentation", BandwidthLimiterAugmentation(p=1.0, sample_rate=TARGET_SR)),
        ("AirAbsorptionAugmentation", AirAbsorptionAugmentation(p=1.0, sample_rate=TARGET_SR)),
        ("RandReverbSim", RandReverbSim(p=1.0, sample_rate=TARGET_SR)),
        ("NoiseGenerator", NoiseGenerator(p=1.0, sample_rate=TARGET_SR)),
    ]

    cfg = AugmentationConfig()
    PIPELINE_CASES: list[tuple[str, Compose]] = [
        ("speech_pipeline", get_speech_augmentations(cfg, TARGET_SR)),
        ("noise_pipeline", get_noise_augmentations(cfg, TARGET_SR)),
        ("distortion_pipeline", get_speech_distortions_td(cfg, TARGET_SR)),
    ]

    errors = 0
    for fname in DEMO_FILES:
        fpath = DEMO_DIR / fname
        if not fpath.exists():
            logger.warning(f"File not found, skipping: {fpath}")
            continue

        logger.info(f"\n=== {fname} ===")
        wav, sr = load_mono(fpath)
        stem = fpath.stem

        for aug_name, aug in AUGMENTATION_CASES:
            aug.train()
            try:
                batch = wav.unsqueeze(0)  # (1, C, T)
                result = aug(batch, sample_rate=sr)
                aug_wav = result.samples.squeeze(0) if hasattr(result, "samples") else result.squeeze(0)

                out_wav = OUT_DIR / f"{stem}__{aug_name}.wav"
                torchaudio.save(str(out_wav), aug_wav, sr)

                out_plot = OUT_DIR / f"{stem}__{aug_name}.png"
                plot_comparison(wav, aug_wav, sr, f"{fname}  ·  {aug_name}", out_plot)

                logger.info(f"  ok  {aug_name}")
            except Exception as exc:
                logger.error(f"  ERR {aug_name}: {exc}")
                errors += 1

        for pipe_name, pipeline in PIPELINE_CASES:
            pipeline.train()
            try:
                batch = wav.unsqueeze(0)
                result = pipeline(batch, sample_rate=sr)
                aug_wav = result.samples.squeeze(0) if hasattr(result, "samples") else result.squeeze(0)

                out_wav = OUT_DIR / f"{stem}__{pipe_name}.wav"
                torchaudio.save(str(out_wav), aug_wav, sr)

                out_plot = OUT_DIR / f"{stem}__{pipe_name}.png"
                plot_comparison(wav, aug_wav, sr, f"{fname}  ·  {pipe_name}", out_plot)

                logger.info(f"  ok  {pipe_name}")
            except Exception as exc:
                logger.error(f"  ERR {pipe_name}: {exc}")
                errors += 1

    logger.info(f"\nDone. Output saved to {OUT_DIR.resolve()}  (errors: {errors})")
    sys.exit(1 if errors else 0)
