"""ERB (Equivalent Rectangular Bandwidth) filterbank utilities."""

import numpy as np
import torch
from torch import Tensor


def freq2erb(freq_hz: float) -> float:
    """Convert frequency in Hz to ERB scale."""
    return 9.265 * np.log1p(freq_hz / (24.7 * 9.265))


def erb2freq(n_erb: float) -> float:
    """Convert ERB scale to frequency in Hz."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_fb_widths(
    sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int = 1
) -> np.ndarray:
    """Calculate ERB filterbank widths.

    Args:
        sr: Sample rate in Hz
        fft_size: FFT size
        nb_bands: Number of ERB bands
        min_nb_freqs: Minimum number of frequency bins per ERB band

    Returns:
        Array of widths (number of frequency bins) for each ERB band
    """
    # Init ERB filter bank
    nyq_freq = sr // 2
    freq_width = sr / fft_size
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(float(nyq_freq))

    erb = np.zeros(nb_bands, dtype=np.int32)
    step = (erb_high - erb_low) / nb_bands

    prev_freq = 0  # Last frequency band of the previous erb band
    freq_over = 0  # Number of frequency bands already stored in previous erb bands

    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = int(round(f / freq_width))
        nb_freqs = fb - prev_freq - freq_over

        if nb_freqs < min_nb_freqs:
            # Not enough freq bins in current erb bin
            freq_over = min_nb_freqs - nb_freqs  # Keep track of enforced bins
            nb_freqs = min_nb_freqs  # Enforce min_nb_freqs
        else:
            freq_over = 0

        erb[i - 1] = nb_freqs
        prev_freq = fb

    # Since we have fft_size/2+1 frequency bins
    erb[nb_bands - 1] += 1

    # Adjust if total is too large
    total = np.sum(erb)
    expected = fft_size // 2 + 1
    if total > expected:
        erb[nb_bands - 1] -= total - expected

    assert np.sum(erb) == expected, f"ERB widths sum {np.sum(erb)} != {expected}"

    return erb


def create_erb_fb(
    widths: np.ndarray, sr: int, normalized: bool = True, inverse: bool = False
) -> Tensor:
    """Create ERB filterbank matrix.

    Args:
        widths: Array of ERB band widths (from erb_fb_widths)
        sr: Sample rate in Hz
        normalized: Whether to normalize bands to unit energy
        inverse: If True, create inverse filterbank (ERB to freq)

    Returns:
        Filterbank matrix as PyTorch tensor
    """
    n_freqs = int(np.sum(widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

    # Starting points for each ERB band
    b_pts = np.cumsum(np.concatenate([[0], widths]))[:-1].astype(int)

    # Create filterbank matrix
    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1.0

    # Normalize to constant energy per band
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0, keepdim=True)

    return fb


def get_erb_filterbanks(sr: int, fft_size: int, nb_erb: int) -> tuple[Tensor, Tensor]:
    """Get forward and inverse ERB filterbanks.

    Args:
        sr: Sample rate
        fft_size: FFT size
        nb_erb: Number of ERB bands

    Returns:
        Tuple of (erb_fb, erb_inv_fb) tensors
    """
    widths = erb_fb_widths(sr, fft_size, nb_erb, min_nb_freqs=1)
    erb_fb = create_erb_fb(widths, sr, normalized=True, inverse=False)
    erb_inv_fb = create_erb_fb(widths, sr, normalized=True, inverse=True)
    return erb_fb, erb_inv_fb


if __name__ == "__main__":
    # Test ERB filterbank generation
    sr = 48000
    fft_size = 960
    nb_erb = 32

    print("Generating ERB filterbank:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  FFT size: {fft_size}")
    print(f"  ERB bands: {nb_erb}")

    widths = erb_fb_widths(sr, fft_size, nb_erb)
    print(f"\nERB widths: {widths}")
    print(f"Total frequency bins: {np.sum(widths)} (expected: {fft_size // 2 + 1})")

    erb_fb, erb_inv_fb = get_erb_filterbanks(sr, fft_size, nb_erb)
    print(f"\nERB filterbank shape: {erb_fb.shape}")
    print(f"ERB inverse filterbank shape: {erb_inv_fb.shape}")

    # Verify reconstruction
    test_input = torch.randn(481, 1)
    erb_features = (test_input.T @ erb_fb).T  # [32, 1]
    reconstructed = (erb_features.T @ erb_inv_fb).T  # [481, 1]

    print("\nTest reconstruction:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  ERB features shape: {erb_features.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstruction error: {(test_input - reconstructed).abs().mean():.6f}")
