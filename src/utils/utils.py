"""Utility functions for audio processing, mixing, and analysis."""

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Audio Statistics and Normalization
# ============================================================================


def rms(values: Union[np.ndarray, Tensor]) -> float:
    """Calculate Root Mean Square (RMS) of audio values.

    Port of DeepFilterNet libDF/src/lib.rs:570-591

    Args:
        values: Audio samples (any shape)

    Returns:
        RMS value as float32

    Example:
        >>> audio = np.array([0.1, -0.2, 0.3, -0.4])
        >>> rms_val = rms(audio)
    """
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()

    vals_flat = values.flatten().astype(np.float32)
    n = len(vals_flat)
    if n == 0:
        return 0.0

    pow_sum = np.sum(vals_flat**2)
    return float(np.sqrt(pow_sum / n))


def normalize_audio(
    audio: Union[np.ndarray, Tensor], target_rms: Optional[float] = None
) -> Union[np.ndarray, Tensor]:
    """Normalize audio by RMS per channel.

    Port of DeepFilterNet libDF/src/transforms.rs:66-70 (rms_normalize)

    Args:
        audio: Audio array of shape (channels, samples) or (samples,)
        target_rms: Target RMS value. If None, normalizes to unit RMS.

    Returns:
        Normalized audio with same shape and type as input

    Example:
        >>> audio = np.random.randn(2, 48000)  # Stereo, 1 second
        >>> normalized = normalize_audio(audio)
    """
    is_tensor = isinstance(audio, Tensor)
    if is_tensor:
        device = audio.device
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio

    # Ensure 2D (channels, samples)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    # Calculate RMS per channel: shape (channels,)
    rms_values = np.sqrt(np.mean(audio_np**2, axis=1)) + 1e-8

    # Normalize
    normalized = audio_np / rms_values[:, np.newaxis]

    # Apply target RMS if specified
    if target_rms is not None:
        normalized = normalized * target_rms

    if squeeze:
        normalized = normalized.squeeze(0)

    # Convert back to original type
    if is_tensor:
        return torch.from_numpy(normalized.astype(np.float32)).to(device)
    return normalized.astype(np.float32)


# ============================================================================
# SNR-Based Mixing
# ============================================================================


def mix_f(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    """Calculate noise scaling factor for target SNR.

    Port of DeepFilterNet libDF/src/transforms.rs:58-63

    Implements the formula:
        scale = 1 / sqrt((E_noise / E_clean) * 10^(SNR/10) + eps)

    where:
        E_clean = sum(clean^2) + 1e-10
        E_noise = sum(noise^2) + 1e-10

    Args:
        clean: Clean speech signal, shape (channels, samples)
        noise: Noise signal, shape (channels, samples)
        snr_db: Target Signal-to-Noise Ratio in dB

    Returns:
        Scaling factor to apply to noise

    Example:
        >>> clean = np.random.randn(2, 48000)
        >>> noise = np.random.randn(2, 48000)
        >>> scale = mix_f(clean, noise, snr_db=10.0)
        >>> scaled_noise = noise * scale
    """
    e_clean = float(np.sum(clean**2) + 1e-10)
    e_noise = float(np.sum(noise**2) + 1e-10)
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Use float64 for stability, then cast back
    scale = 1.0 / np.sqrt((e_noise / e_clean) * snr_linear + 1e-10)
    return np.float32(scale)


def mix_audio_signal(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    gain_db: float = 0.0,
    clean_distorted: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix clean speech and noise at specified SNR with anti-clipping.

    Port of DeepFilterNet libDF/src/dataset.rs:2047-2074

    Pipeline:
        1. Apply gain to speech (linear gain = 10^(gain_db/20))
        2. Use clean_distorted if provided (e.g., reverberant speech)
        3. Scale noise to target SNR using clean speech energy
        4. Create mixture
        5. Guard against clipping (scale all if max > 1.0)

    Args:
        clean: Clean speech, shape (channels, samples)
        noise: Noise signal, shape (channels, samples)
        snr_db: Target SNR in dB
        gain_db: Gain to apply to speech in dB (default: 0)
        clean_distorted: Optional distorted speech (e.g., with reverb).
                        If provided, used for mixing instead of clean.

    Returns:
        Tuple of (clean_out, noise_out, mixture) all with anti-clipping applied

    Example:
        >>> clean = np.random.randn(2, 48000) * 0.5
        >>> noise = np.random.randn(2, 48000) * 0.3
        >>> clean_out, noise_out, mixture = mix_audio_signal(
        ...     clean, noise, snr_db=10.0, gain_db=3.0
        ... )
    """
    # Copy to avoid modifying inputs
    clean_out = clean.copy().astype(np.float32)
    noise_out = noise.copy().astype(np.float32)

    # Step 1: Apply gain to speech
    gain_linear = 10.0 ** (gain_db / 20.0)
    clean_out = clean_out * gain_linear

    # Step 2: Determine mixture source (may use distorted speech)
    if clean_distorted is not None:
        clean_mix = clean_distorted.copy().astype(np.float32) * gain_linear
    else:
        clean_mix = clean_out.copy()

    # Step 3: Scale noise to target SNR
    # Use clean_out for energy calculation (considers DRR if reverb applied)
    scale = mix_f(clean_out, noise_out, snr_db)
    noise_out = noise_out * scale

    # Step 4: Create mixture
    mixture = clean_mix + noise_out

    # Step 5: Anti-clipping guard
    max_clean = float(np.max(np.abs(clean_out)))
    max_noise = float(np.max(np.abs(noise_out)))
    max_mixture = float(np.max(np.abs(mixture)))
    max_all = max(max_clean, max_noise, max_mixture)

    # Check for NaN
    if np.isnan(max_all):
        raise ValueError("Found NaN in audio signals")

    # Scale down if clipping would occur
    if max_all - 1.0 > 1e-10:
        scale_factor = 1.0 / (max_all + 1e-10)
        clean_out = clean_out * scale_factor
        noise_out = noise_out * scale_factor
        mixture = mixture * scale_factor

    return clean_out, noise_out, mixture


# ============================================================================
# Multi-Noise Combining
# ============================================================================


def combine_noises(
    num_channels: int,
    target_length: int,
    noises: list[np.ndarray],
    noise_gains: Optional[list[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Combine multiple noise tracks with length/channel adjustment.

    Port of DeepFilterNet libDF/src/dataset.rs:1979-2023

    Pipeline for each noise:
        1. Adjust length (repeat if too short, random crop if too long)
        2. Adjust channels (random remove if too many, random duplicate if too few)
        3. Apply gain in dB (if provided)
        4. Average all noises

    Args:
        num_channels: Target number of channels
        target_length: Target length in samples
        noises: List of noise arrays, each shape (channels, samples)
        noise_gains: Optional list of gains in dB for each noise
        rng: Random number generator (if None, uses default)

    Returns:
        Combined noise array of shape (num_channels, target_length)

    Example:
        >>> noise1 = np.random.randn(1, 24000) * 0.1
        >>> noise2 = np.random.randn(2, 60000) * 0.2
        >>> combined = combine_noises(
        ...     num_channels=2,
        ...     target_length=48000,
        ...     noises=[noise1, noise2],
        ...     noise_gains=[-3.0, 0.0]
        ... )
        >>> combined.shape
        (2, 48000)
    """
    if rng is None:
        rng = np.random.default_rng()

    processed_noises = []

    for noise_idx, noise in enumerate(noises):
        noise = noise.copy().astype(np.float32)

        # Step 1: Adjust length
        # Repeat noise until it's at least target_length
        while noise.shape[1] < target_length:
            noise = np.concatenate([noise, noise], axis=1)

        # Random crop if too long
        if noise.shape[1] > target_length:
            excess = noise.shape[1] - target_length
            start = rng.integers(0, excess + 1)
            noise = noise[:, start : start + target_length]

        # Step 2: Adjust number of channels
        # Remove random channels if too many
        while noise.shape[0] > num_channels:
            ch_to_remove = rng.integers(0, noise.shape[0])
            noise = np.delete(noise, ch_to_remove, axis=0)

        # Duplicate random channels if too few
        while noise.shape[0] < num_channels:
            ch_to_dup = rng.integers(0, noise.shape[0])
            duplicate = noise[ch_to_dup : ch_to_dup + 1, :]
            noise = np.vstack([noise, duplicate])

        # Step 3: Apply gain if provided
        if noise_gains is not None and noise_idx < len(noise_gains):
            gain_db = noise_gains[noise_idx]
            gain_linear = 10.0 ** (gain_db / 20.0)
            noise = noise * gain_linear

        processed_noises.append(noise)

    # Step 4: Average all noises
    if len(processed_noises) == 0:
        return np.zeros((num_channels, target_length), dtype=np.float32)

    stacked = np.array(processed_noises)  # shape: (n_noises, n_channels, n_samples)
    combined = np.mean(stacked, axis=0)

    return combined.astype(np.float32)


# ============================================================================
# Gain Application
# ============================================================================


def apply_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in dB to signal.

    Port of DeepFilterNet libDF/src/dataset.rs:2055-2056

    Converts dB to linear gain: gain_linear = 10^(gain_db/20)

    Args:
        signal: Audio signal, any shape
        gain_db: Gain in decibels

    Returns:
        Signal with gain applied

    Example:
        >>> signal = np.random.randn(2, 48000) * 0.1
        >>> gained = apply_gain(signal, gain_db=6.0)  # +6 dB
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (signal * gain_linear).astype(np.float32)


def apply_gain_with_clipping_protection(
    signal: np.ndarray, gain_db: float
) -> np.ndarray:
    """Apply gain with anti-clipping protection.

    Port of DeepFilterNet libDF/src/dataset.rs:2067-2071

    If gained signal would clip (max > 1.0), scales down to prevent it.

    Args:
        signal: Audio signal, any shape
        gain_db: Gain in decibels

    Returns:
        Signal with gain applied and clipping prevented

    Example:
        >>> signal = np.random.randn(2, 48000) * 0.9
        >>> gained = apply_gain_with_clipping_protection(signal, gain_db=12.0)
    """
    gained = apply_gain(signal, gain_db)
    max_val = float(np.max(np.abs(gained)))

    # Scale down if would clip
    if max_val - 1.0 > 1e-10:
        scale = 1.0 / (max_val + 1e-10)
        gained = gained * scale

    return gained.astype(np.float32)


# ============================================================================
# Bandwidth Estimation
# ============================================================================


def rfftfreqs(n: int, sr: int) -> np.ndarray:
    """Calculate frequency values for real FFT output bins.

    Port of DeepFilterNet libDF/src/transforms.rs:510-514

    Args:
        n: Number of frequency bins (FFT_SIZE // 2 + 1)
        sr: Sample rate in Hz

    Returns:
        Array of frequency values in Hz

    Example:
        >>> freqs = rfftfreqs(481, 48000)  # 960-point FFT
        >>> freqs.shape
        (481,)
    """
    return np.arange(n, dtype=np.float32) * (sr / 2) / (n - 1)


def bw_filterbank(center_freqs: np.ndarray, cutoff_bins: np.ndarray) -> np.ndarray:
    """Create bandwidth estimation filterbank.

    Port of DeepFilterNet libDF/src/transforms.rs:480-507

    Creates 8 bands for frequency ranges (assumes 48 kHz):
        [0-8, 8-10, 10-12, 12-16, 16-18, 18-20, 20-22, 22-24] kHz

    Args:
        center_freqs: Center frequency for each FFT bin
        cutoff_bins: Array of 8 cutoff frequencies in Hz

    Returns:
        Filterbank matrix of shape (n_freqs, 8), normalized per band

    Example:
        >>> freqs = rfftfreqs(481, 48000)
        >>> cutoffs = np.array([8000., 10000., 12000., 16000.,
        ...                      18000., 20000., 22000., 24000.])
        >>> fb = bw_filterbank(freqs, cutoffs)
        >>> fb.shape
        (481, 8)
    """
    n_freqs = len(center_freqs)
    fb = np.zeros((n_freqs, 8), dtype=np.float32)

    for i, f in enumerate(center_freqs):
        if f <= cutoff_bins[0]:
            fb[i, 0] = 1.0
        elif f <= cutoff_bins[1]:
            fb[i, 1] = 1.0
        elif f <= cutoff_bins[2]:
            fb[i, 2] = 1.0
        elif f <= cutoff_bins[3]:
            fb[i, 3] = 1.0
        elif f <= cutoff_bins[4]:
            fb[i, 4] = 1.0
        elif f <= cutoff_bins[5]:
            fb[i, 5] = 1.0
        elif f <= cutoff_bins[6]:
            fb[i, 6] = 1.0
        else:
            fb[i, 7] = 1.0

    # Normalize per band
    fb = fb / (fb.sum(axis=0, keepdims=True) + 1e-10)
    return fb


def estimate_bandwidth(
    spectrum: np.ndarray,
    sr: int = 48000,
    db_cutoff: float = 120.0,
    window_size: int = 200,
) -> int:
    """Estimate audio bandwidth from spectrum.

    Port of DeepFilterNet libDF/src/transforms.rs:534-579

    Analyzes spectrum in frequency bands and finds cutoff frequency
    where energy drops below threshold.

    Args:
        spectrum: Complex spectrum of shape (batch, time, freq) or (time, freq)
        sr: Sample rate in Hz (must be 48000)
        db_cutoff: Energy threshold in dB (negative value)
        window_size: Number of frames to analyze together

    Returns:
        Estimated bandwidth as frequency bin index

    Example:
        >>> # Create spectrum from STFT
        >>> audio = np.random.randn(48000)
        >>> spec = np.fft.rfft(audio.reshape(-1, 960), axis=1)
        >>> spec = spec[np.newaxis, :, :]  # Add batch dim
        >>> bw = estimate_bandwidth(spec)
    """
    assert sr == 48000, "estimate_bandwidth assumes 48 kHz sampling rate"

    # Handle batch dimension
    if spectrum.ndim == 3:
        spectrum = spectrum[0]  # Take first batch item

    n_frames, n_freqs = spectrum.shape

    # Adjust window size if needed
    if n_frames < window_size:
        window_size = n_frames

    # Ensure db_cutoff is negative
    if db_cutoff > 0:
        db_cutoff = -db_cutoff

    # 1. Initialize bandwidth filterbank
    cutoff_freqs = np.array(
        [8000.0, 10000.0, 12000.0, 16000.0, 18000.0, 20000.0, 22000.0, 24000.0],
        dtype=np.float32,
    )
    center_freqs = rfftfreqs(n_freqs, sr)
    fb = bw_filterbank(center_freqs, cutoff_freqs)

    # 2. Convert spectrum to dB power
    magnitude = np.abs(spectrum)
    magnitude_db = 20.0 * np.log10(magnitude + 1e-16)

    # 3. Apply filterbank: shape (n_frames, 8)
    f_db = magnitude_db @ fb

    # 4. Create mapping from band index to frequency bin
    c_map = np.zeros(8, dtype=np.int32)
    for i in range(n_freqs):
        for j in range(8):
            if fb[i, j] > 0:
                c_map[j] = i

    # 5. Find cutoff indices for each window
    indices = []
    for t_start in range(0, n_frames, window_size):
        t_end = min(t_start + window_size, n_frames)
        window_data = f_db[t_start:t_end, :]
        max_per_band = np.max(window_data, axis=0)

        # Find first band (after band 0) below threshold
        cutoff_idx = 7  # Default to highest band
        for band_idx in range(1, 8):
            if max_per_band[band_idx] < db_cutoff:
                cutoff_idx = band_idx
                break

        indices.append(c_map[cutoff_idx])

    # 6. Return median of found indices
    indices_arr = np.array(indices)
    return int(np.median(indices_arr))
